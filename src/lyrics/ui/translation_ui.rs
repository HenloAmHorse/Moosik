//! Optional translation work, separate from original-lyrics lookup/readings.
//! One worker owns its slot until its completion is consumed, even on Cancel.

use super::*;
use crate::lyrics::ai;
use crate::lyrics::translation::{self, SourceLanguage};
use std::sync::mpsc::{self, TryRecvError};
use std::sync::{
    Arc,
    atomic::{AtomicBool, Ordering},
};

#[derive(Default)]
pub(super) struct TranslationState {
    open: bool,
    source: Option<SourceLanguage>,
    consent: bool,
    replace: bool,
    serial: u64,
    task: Option<Task>,
    preview: Option<Preview>,
    edit: Option<ManualEdit>,
    provider: Provider,
    ai: AiSettings,
}

/// Which online translation an explicit request uses.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub(super) enum Provider {
    #[default]
    Basic,
    Ai,
}

/// Context-aware AI settings, for this session only: nothing here is saved.
/// The key lives in memory and is never part of a result's identity.
#[derive(Default)]
struct AiSettings {
    model: String,
    fluent: bool,
    review: bool,
    key: Option<ai::ApiKey>,
    /// The masked field's text until "Use key" takes it; cleared then.
    key_input: String,
    /// What the user agreed to send: these settings and this sheet. Any
    /// settings change clears it (going back does not revive it), and a
    /// changed sheet no longer matches it. The key is not part of it.
    consent: Option<Consent>,
}

#[derive(Clone, Debug, PartialEq, Eq)]
struct Consent {
    config: ai::Config,
    track: Option<PathBuf>,
    rev: u64,
}

struct Snapshot {
    track: Option<PathBuf>,
    rev: u64,
    lyrics: Lyrics,
}

impl Snapshot {
    fn matches(&self, window: &LyricsWindow) -> bool {
        self.track == window.loaded_for
            && self.rev == window.sheet_rev
            && window.lyrics.as_ref().is_some_and(|l| {
                l.lines == self.lyrics.lines
                    && l.offset_ms == self.lyrics.offset_ms
                    && l.title == self.lyrics.title
                    && l.artist == self.lyrics.artist
                    && l.translation_credit == self.lyrics.translation_credit
            })
    }
}

struct Task {
    snapshot: Snapshot,
    serial: u64,
    cancel: Arc<AtomicBool>,
    rx: Receiver<Message>,
    progress: (usize, usize),
    /// The AI settings an AI request was made with; `None` otherwise.
    ai: Option<ai::Config>,
}

enum Message {
    Progress(usize, usize),
    Done(Result<Translated, String>),
}

struct Translated {
    rows: Vec<Option<String>>,
    credit: String,
    replaces_existing: bool,
    ai: Option<AiResult>,
}

struct AiResult {
    config: ai::Config,
    flags: Vec<Option<ai::RowFlags>>,
    notes: String,
    review: ai::Review,
}

struct Preview {
    snapshot: Snapshot,
    result: Translated,
    display: crate::lyrics::zh::Cache,
}

struct ManualEdit {
    snapshot: Snapshot,
    rows: Vec<String>,
    paste: String,
    paste_open: bool,
}

impl TranslationState {
    pub(super) fn is_busy(&self) -> bool {
        self.task.is_some()
    }
    pub(super) fn is_editing(&self) -> bool {
        self.edit.is_some()
    }
    pub(super) fn is_open(&self) -> bool {
        self.open
    }
    pub(super) fn clear_preview(&mut self) {
        self.preview = None;
        self.edit = None;
    }
    pub(super) fn reset_consent(&mut self) {
        self.consent = false;
        self.ai.consent = None;
        self.source = None;
    }
    pub(super) fn clear_sheet(&mut self) {
        self.clear_preview();
        self.open = false;
        self.consent = false;
        self.ai.consent = None;
        self.replace = false;
        self.source = None;
        self.serial = self.serial.wrapping_add(1);
    }
}

impl LyricsWindow {
    fn translation_snapshot(&self) -> Option<Snapshot> {
        Some(Snapshot {
            track: self.loaded_for.clone(),
            rev: self.sheet_rev,
            lyrics: self.lyrics.as_ref()?.clone(),
        })
    }

    /// The AI settings as they stand: `None` until a source language and a
    /// well-formed model ID are chosen.
    fn ai_config(&self) -> Option<ai::Config> {
        let t = &self.translation;
        let style = if t.ai.fluent { ai::Style::Fluent } else { ai::Style::Faithful };
        ai::Config::new(&t.ai.model, style, t.ai.review, t.source?).ok()
    }

    /// The only way the provider and AI settings change, from the panel and
    /// from tests alike: a different value clears the AI agreement.
    fn set_ai_setting<T: PartialEq>(&mut self, pick: impl FnOnce(&mut TranslationState) -> &mut T, value: T) {
        let field = pick(&mut self.translation);
        if *field != value {
            *field = value;
            self.translation.ai.consent = None;
        }
    }
    fn set_provider(&mut self, p: Provider) { self.set_ai_setting(|t| &mut t.provider, p); }
    fn set_source(&mut self, s: Option<SourceLanguage>) { self.set_ai_setting(|t| &mut t.source, s); }
    fn set_replace(&mut self, r: bool) { self.set_ai_setting(|t| &mut t.replace, r); }
    fn set_ai_model(&mut self, m: String) { self.set_ai_setting(|t| &mut t.ai.model, m); }
    fn set_ai_fluent(&mut self, f: bool) { self.set_ai_setting(|t| &mut t.ai.fluent, f); }
    fn set_ai_review(&mut self, r: bool) { self.set_ai_setting(|t| &mut t.ai.review, r); }

    /// Agree (or withdraw agreement) to send the current sheet with the
    /// current settings. Without complete settings there is nothing to agree to.
    fn agree_ai(&mut self, agreed: bool) {
        self.translation.ai.consent = match (agreed, self.ai_config()) {
            (true, Some(config)) => Some(Consent { config, track: self.loaded_for.clone(), rev: self.sheet_rev }),
            _ => None,
        };
    }

    /// The settings the user agreed to, if that agreement still matches the
    /// current settings and the current sheet.
    fn ai_agreed(&self) -> Option<ai::Config> {
        let c = self.translation.ai.consent.as_ref()?;
        let current = self.ai_config()?;
        (c.config == current && c.track == self.loaded_for && c.rev == self.sheet_rev).then_some(current)
    }

    /// Whether a result made with `captured` AI settings (`None`: not an AI
    /// result) may still be adopted: AI mode and exactly those settings.
    fn ai_matches(&self, captured: Option<&ai::Config>) -> bool {
        captured.is_none_or(|c| self.translation.provider == Provider::Ai && self.ai_config().as_ref() == Some(c))
    }

    pub(super) fn cancel_translation(&mut self) {
        if let Some(task) = &self.translation.task {
            task.cancel.store(true, Ordering::Release);
        }
    }

    /// Translation work whose result must never be adopted: cancel the
    /// request (no further requests once the worker sees it; a call already
    /// in flight finishes within its timeout), make its late completion stale,
    /// and drop an unapplied preview. The worker keeps its slot until its
    /// completion is consumed. A manual draft is the user's text and is kept.
    pub(super) fn abandon_translation(&mut self) {
        let t = &mut self.translation;
        let live = t.task.as_ref().is_some_and(|x| !x.cancel.load(Ordering::Acquire));
        if !live && t.preview.is_none() {
            return;
        }
        if let Some(task) = &t.task {
            task.cancel.store(true, Ordering::Release);
        }
        t.serial = t.serial.wrapping_add(1);
        t.preview = None;
        self.status = "Translation cancelled; nothing was applied.".into();
    }

    pub(super) fn open_translation(&mut self) {
        self.translation.open = !self.translation.open;
        if !self.translation.open {
            self.cancel_translation();
        }
        if self.translation.open && self.translation.source.is_none() {
            self.translation.source = self
                .lyrics
                .as_ref()
                .and_then(|l| translation::detect(&l.lines));
        }
        if self.translation.open
            && self.translation_display == crate::lyrics::zh::ChineseDisplay::Original
        {
            self.translation_display = crate::lyrics::zh::ChineseDisplay::Traditional;
        }
    }

    fn start_translation(&mut self, track: &TrackRef, machine: bool) {
        if self.busy() || self.editing || self.romaji_editing || self.translation.is_editing() {
            return;
        }
        let source = self.translation.source;
        let ai_mode = machine && self.translation.provider == Provider::Ai;
        if machine && !ai_mode && (!self.translation.consent || source.is_none()) {
            return;
        }
        // AI: agreement to exactly these settings and this sheet, a key, and
        // the sheet's physical bounds, all checked on borrowed state before
        // the snapshot below deep-copies the sheet.
        let ai_ready = if ai_mode {
            let Some(config) = self.ai_agreed() else { return };
            let Some(key) = self.translation.ai.key.clone() else { return };
            if let Some(l) = &self.lyrics
                && let Err(e) = ai::check_input(l.lines.iter().map(|x| x.text.as_str()))
            {
                self.status = e;
                return;
            }
            Some((config, key))
        } else {
            None
        };
        let Some(snapshot) = self.translation_snapshot() else {
            return;
        };
        if snapshot.track.as_deref() != Some(track.path) {
            return;
        }
        // The request is built and bounded here, before anything is sent.
        let ai_job = if let Some((config, key)) = ai_ready {
            match ai_job_for(&snapshot.lyrics, self.translation.replace, config, track) {
                Ok(job) => Some((job, key)),
                Err(e) => {
                    self.status = e;
                    return;
                }
            }
        } else {
            None
        };
        let ai_config = ai_job.as_ref().map(|(job, _)| job.config().clone());
        self.translation.clear_preview();
        self.translation.serial = self.translation.serial.wrapping_add(1);
        let serial = self.translation.serial;
        let cancel = Arc::new(AtomicBool::new(false));
        let stop = cancel.clone();
        let (tx, rx) = mpsc::channel();
        let replace = self.translation.replace;
        // Basic and supplied lookups work on a copy; the AI job already holds
        // what it sends and needs none.
        let mut input = if ai_job.is_none() { Some(snapshot.lyrics.clone()) } else { None };
        if let Some(input) = input.as_mut()
            && machine
            && !replace
        {
            // Keep already-translated rows entirely out of the request, rather
            // than spend quota and throw their translations away afterwards.
            for row in &mut input.lines {
                if row
                    .translation
                    .as_ref()
                    .is_some_and(|s| !s.trim().is_empty())
                {
                    row.text.clear();
                }
            }
        }
        let (artist, title, duration) = (
            track.artist.to_string(),
            track.title.to_string(),
            track.duration,
        );
        let spawned = std::thread::Builder::new().name("lyrics-translation".into()).spawn(move || {
            // A panic is converted to a completion in unwind builds. Release
            // builds use panic=abort; they make no recovery promise here.
            let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
                if stop.load(Ordering::Acquire) { return Err("Cancelled.".into()); }
                if let Some((job, key)) = ai_job {
                    let o = ai::translate(&job, &key, &stop, |done, total| {
                        let _ = tx.send(Message::Progress(done, total));
                    })?;
                    Ok(ai_translated(&job, o, replace))
                } else if machine {
                    let Some(input) = input else { return Err("Internal error: no input; nothing was sent.".into()) };
                    let source = source.ok_or_else(|| "Choose the source language.".to_string())?;
                    let rows = translation::translate(&input.lines, source, &stop, |done, total| {
                        let _ = tx.send(Message::Progress(done, total));
                    })?;
                    Ok(Translated { rows, credit: format!("{} ({} to Chinese)", translation::NAME, source.label()),
                                    replaces_existing: replace, ai: None })
                } else {
                    // Lookup is independent of LRCLIB's successful original
                    // lyric hit. Never install a different original sheet.
                    let Some(input) = input else { return Err("Internal error: no input; nothing was sent.".into()) };
                    let hit = super::super::netease::find_translation(&artist, &title, duration, &stop)?
                        .ok_or_else(|| "No matching translation found. Try online translation or paste/edit your own.".to_string())?;
                    if stop.load(Ordering::Acquire) { return Err("Cancelled.".into()); }
                    let rows = translation::supplement(&input, &hit)?;
                    Ok(Translated { rows, credit: format!("Supplement supplied by {}", hit.source),
                                    replaces_existing: false, ai: None })
                }
            })).unwrap_or_else(|_| Err("The translation worker stopped unexpectedly; nothing was applied.".into()));
            let _ = tx.send(Message::Done(result));
        });
        match spawned {
            Ok(_) => {
                self.translation.task = Some(Task {
                    snapshot,
                    serial,
                    cancel,
                    rx,
                    progress: (0, 0),
                    ai: ai_config,
                });
                self.status = if ai_mode {
                    "Translating with OpenAI…"
                } else if machine {
                    "Translating…"
                } else {
                    "Looking for a matching translation…"
                }
                .into();
            }
            Err(_) => {
                self.status = "Could not start translation; nothing was sent or applied.".into()
            }
        }
    }

    pub(super) fn poll_translation(&mut self) {
        loop {
            let Some(task) = self.translation.task.as_ref() else {
                return;
            };
            let message = match task.rx.try_recv() {
                Ok(m) => m,
                Err(TryRecvError::Empty) => return,
                Err(TryRecvError::Disconnected) => Message::Done(Err(
                    "The translation worker disconnected; nothing was applied. You can retry."
                        .into(),
                )),
            };
            match message {
                Message::Progress(done, total) => {
                    if let Some(task) = self.translation.task.as_mut() {
                        task.progress = (done, total);
                    }
                }
                Message::Done(result) => {
                    let task = self
                        .translation
                        .task
                        .take()
                        .unwrap_or_else(|| unreachable!());
                    let current = task.serial == self.translation.serial
                        && task.snapshot.matches(self)
                        && self.ai_matches(task.ai.as_ref())
                        && !self.editing
                        && !self.romaji_editing
                        && !self.translation.is_editing();
                    if task.cancel.load(Ordering::Acquire) {
                        if current {
                            self.status = "Translation cancelled; nothing was applied.".into();
                        }
                        return;
                    }
                    if !current {
                        // Do not replace the new track/editor's status with a
                        // stale request's result or error.
                        return;
                    }
                    match result {
                        Ok(result)
                            if result.rows.len() == task.snapshot.lyrics.lines.len()
                                && result
                                    .rows
                                    .iter()
                                    .any(|s| s.as_ref().is_some_and(|s| !s.trim().is_empty())) =>
                        {
                            self.status = match result.ai.as_ref().map(|a| (&a.review, a.flags.iter().flatten().filter(|f| f.any()).count())) {
                                None => "Translation ready to preview — Apply does not save it to disk.".into(),
                                Some((ai::Review::Failed(e), _)) => format!(
                                    "Unreviewed AI draft ready to preview: the review request failed. {e}"),
                                Some((review, flagged)) => format!(
                                    "{} ready to preview{} — Apply does not save it to disk.",
                                    if *review == ai::Review::Reviewed { "Reviewed AI translation" } else { "AI translation" },
                                    if flagged > 0 { format!("; {flagged} row(s) flagged for checking") } else { String::new() }),
                            };
                            self.translation.preview = Some(Preview {
                                snapshot: task.snapshot,
                                result,
                                display: crate::lyrics::zh::Cache::default(),
                            });
                        }
                        Ok(_) => {
                            self.status =
                                "No new translation was returned; the existing lyrics were kept."
                                    .into()
                        }
                        Err(e) => self.status = e,
                    }
                    return;
                }
            }
        }
    }

    fn apply_translation_preview(&mut self) {
        let Some(preview) = self.translation.preview.take() else {
            return;
        };
        if !preview.snapshot.matches(self) || self.editing || self.romaji_editing {
            self.status =
                "The lyrics changed. Find or translate again; nothing was applied.".into();
            return;
        }
        if !self.ai_matches(preview.result.ai.as_ref().map(|a| &a.config)) {
            self.status = "The AI settings changed since this preview was made; nothing was applied.".into();
            return;
        }
        let Some(lyrics) = &mut self.lyrics else {
            return;
        };
        let mut changed = false;
        // Supplied supplementation never replaces existing rows, even if the
        // machine-translation checkbox was ticked. Keep prior attribution
        // whenever any old translated row survives a replacement operation.
        let old_survives = !preview.result.replaces_existing
            || lyrics
                .lines
                .iter()
                .zip(&preview.result.rows)
                .any(|(line, new)| {
                    new.is_none()
                        && line
                            .translation
                            .as_ref()
                            .is_some_and(|s| !s.trim().is_empty())
                });
        for (line, translated) in lyrics.lines.iter_mut().zip(preview.result.rows) {
            if let Some(text) = translated
                && line.translation.as_ref() != Some(&text)
            {
                line.translation = Some(text);
                changed = true;
            }
        }
        if changed {
            let previous = lyrics.translation_credit.as_deref();
            let credit = preview.result.credit;
            lyrics.translation_credit =
                Some(merged_credit(&credit, previous.filter(|_| old_survives)));
            self.translation_changed();
        }
        self.status =
            "Translation applied — Save to keep it. Original lyrics and timing are unchanged."
                .into();
    }

    fn translation_changed(&mut self) {
        self.translation_dirty = true;
        self.show_translation = true;
        self.sheet_rev = self.sheet_rev.wrapping_add(1);
        self.zh.clear();
        self.zh_sheet = None;
        self.translation.clear_preview();
    }

    fn begin_translation_edit(&mut self) {
        if self.busy() || self.editing || self.romaji_editing || self.translation.is_editing() {
            return;
        }
        if let Some(snapshot) = self.translation_snapshot() {
            let rows = snapshot
                .lyrics
                .lines
                .iter()
                .map(|l| l.translation.clone().unwrap_or_default())
                .collect();
            self.translation.clear_preview();
            self.translation.edit = Some(ManualEdit {
                snapshot,
                rows,
                paste: String::new(),
                paste_open: false,
            });
        }
    }

    fn commit_translation_edit(&mut self) {
        if let Some(edit) = &self.translation.edit {
            if !edit.snapshot.matches(self) || self.editing || self.romaji_editing {
                self.status = "The lyrics changed; no translation edits were applied. Your editor buffer is kept; copy it before cancelling.".into();
                return;
            }
            for text in &edit.rows {
                if let Err(error) = translation::validate_translation_line(text) {
                    self.status = error;
                    return; // keep the unapplied buffer available for correction
                }
            }
        }
        let Some(edit) = self.translation.edit.take() else {
            return;
        };
        let Some(lyrics) = &mut self.lyrics else {
            return;
        };
        let mut changed = false;
        for (line, text) in lyrics.lines.iter_mut().zip(edit.rows) {
            let new = (!text.trim().is_empty()).then(|| text.trim().to_string());
            if new != line.translation {
                line.translation = new;
                changed = true;
            }
        }
        if changed {
            // An edited row may still be mostly machine text, so editing never
            // counts as proof of independent work: the earlier attribution,
            // with its complete unreviewed marker, stays while any translation
            // remains. Only clearing every row drops it (nothing is left to
            // attribute); a Replace-mode preview that overwrites every row is
            // the other action that does (see `apply_translation_preview`).
            let any_left = lyrics.lines.iter().any(|l| l.translation.is_some());
            let credit = any_left.then(|| merged_credit("Manually edited translation", lyrics.translation_credit.as_deref()));
            lyrics.translation_credit = credit;
            self.translation_changed();
            self.status =
                "Translation edited — Save to keep it. Originals and timing were not changed."
                    .into();
        }
    }

    pub(super) fn translation_panel(&mut self, ui: &mut egui::Ui, track: &TrackRef) {
        if self.translation.preview.as_ref().is_some_and(|p| !self.ai_matches(p.result.ai.as_ref().map(|a| &a.config))) {
            self.translation.preview = None;
            self.status = "The AI settings changed; the unapplied AI preview was discarded.".into();
        }
        let editing_sheet = self.editing || self.romaji_editing;
        let mut find = false;
        let mut machine = false;
        let mut cancel = false;
        let mut apply = false;
        let mut edit = false;
        let mut commit_edit = false;
        // Keep Apply/Cancel reachable in the normal 520x640 lyrics window,
        // even when a preview or whole-sheet paste editor is expanded.
        let panel_height = (ui.available_height() * 0.65).max(160.0);
        egui::ScrollArea::vertical().id_salt("translation_controls_scroll")
            .max_height(panel_height).show(ui, |ui| { ui.group(|ui| {
            ui.horizontal_wrapped(|ui| {
                ui.label("中文翻譯 / Chinese translation");
                if ui.button("✖").on_hover_text("Close this panel; an active request is cancelled.").clicked() {
                    self.translation.open = false;
                    cancel = true;
                }
            });
            if let Some(note) = self.lyrics.as_ref().and_then(|l| l.translation_credit.as_ref()) {
                ui.label(RichText::new(note).small());
            }
            let idle = !self.busy() && !editing_sheet && !self.translation.is_editing();
            ui.horizontal_wrapped(|ui| {
                find = ui.add_enabled(idle, egui::Button::new("Find supplied translation")).on_hover_text(
                    "Search NetEase independently of the original lookup. Attach only matching original lines; \
                     do not replace existing translations. Sends track metadata, not your lyric sheet.").clicked();
                edit = ui.add_enabled(idle, egui::Button::new("Edit / paste translation…")).clicked();
                use crate::lyrics::zh::ChineseDisplay as Z;
                egui::ComboBox::from_id_salt("translation_target")
                    .selected_text(match self.translation_display { Z::Simplified => "简体", _ => "繁體" })
                    .show_ui(ui, |ui| {
                        ui.selectable_value(&mut self.translation_display, Z::Traditional, "繁體（台灣字形）");
                        ui.selectable_value(&mut self.translation_display, Z::Simplified, "简体");
                    });
            });
            ui.separator();
            // Settings are edited as copies and go through the setters, so a
            // change clears the AI agreement before anything below reads it.
            let (mut source, mut replace, mut provider) =
                (self.translation.source, self.translation.replace, self.translation.provider);
            ui.horizontal_wrapped(|ui| {
                ui.label("Original language:");
                egui::ComboBox::from_id_salt("translation_source")
                    .selected_text(source.map(SourceLanguage::label).unwrap_or("Choose…"))
                    .show_ui(ui, |ui| {
                        for s in SourceLanguage::ALL {
                            ui.selectable_value(&mut source, Some(s), s.label());
                        }
                    });
                ui.label(RichText::new("One source language applies to the whole request.").small());
                ui.add_enabled(idle && self.translation.preview.is_none(),
                    egui::Checkbox::new(&mut replace, "Replace existing translation"));
            });
            ui.horizontal_wrapped(|ui| {
                ui.label("Online translation:");
                ui.add_enabled_ui(idle, |ui| {
                    ui.radio_value(&mut provider, Provider::Basic, "Basic (MyMemory)");
                    ui.radio_value(&mut provider, Provider::Ai, "Context-aware AI (OpenAI, your key)");
                });
            });
            self.set_source(source);
            self.set_replace(replace);
            self.set_provider(provider);
            if self.translation.provider == Provider::Ai {
                machine = self.ai_panel(ui, idle);
            } else {
                ui.label(RichText::new("Online translation sends lyric text to MyMemory, a third-party service. \
                    Its terms describe long-term segment storage and sharing with translation partners. \
                    Do not submit private lyrics. Output is machine-assisted \
                    and may mistranslate songs. No account/key is required; service quotas apply.").small());
                ui.horizontal_wrapped(|ui| {
                    ui.hyperlink_to("Service / data terms", translation::TERMS_URL);
                    ui.hyperlink_to("Quota limits", translation::LIMITS_URL);
                    ui.checkbox(&mut self.translation.consent, "I agree to send this sheet");
                    machine = ui.add_enabled(idle && self.translation.consent && self.translation.source.is_some(),
                        egui::Button::new("Translate to Chinese")).clicked();
                });
            }
            if let Some(task) = &self.translation.task {
                ui.horizontal_wrapped(|ui| {
                    let (done, total) = task.progress;
                    if total > 0 {
                        ui.label(if task.ai.is_some() { format!("{done}/{total} requests answered") }
                                 else { format!("{done}/{total} unique lines") });
                    }
                    cancel = ui.button("Cancel").clicked() || cancel;
                });
            }
            if self.translation.replace {
                ui.label(RichText::new("Existing source/user translations will be replaced only after Apply. \
                    Save keeps the previous sidecar as a backup.").small().color(pal::amber(ui.visuals().dark_mode)));
            }
            if let Some(preview) = &mut self.translation.preview {
                ui.separator();
                ui.label(RichText::new(&preview.result.credit).small());
                let amber = pal::amber(ui.visuals().dark_mode);
                if let Some(a) = &preview.result.ai {
                    match &a.review {
                        ai::Review::Failed(e) => { ui.label(RichText::new(format!(
                            "Unreviewed draft: the review request failed. {e}")).small().color(amber)); }
                        ai::Review::Reviewed => { ui.label(RichText::new(
                            "Reviewed once by the model — a second AI pass, not a human check.").small()); }
                        ai::Review::NotRequested => {}
                    }
                    if !a.notes.is_empty() {
                        egui::CollapsingHeader::new("Model notes (shown only, not saved)").id_salt("ai_notes")
                            .show(ui, |ui| { ui.label(RichText::new(&a.notes).small()); });
                    }
                }
                egui::ScrollArea::vertical().id_salt("translation_preview").max_height(155.0).show(ui, |ui| {
                    for (i, (original, translated)) in preview.snapshot.lyrics.lines.iter().zip(&preview.result.rows).enumerate() {
                        if let Some(text) = translated {
                            ui.label(RichText::new(&original.text).small());
                            let family = translation_family(ui, self.translation_display);
                            ui.label(RichText::new(preview.display.get(i, text, self.translation_display)).family(family));
                            if let Some(f) = preview.result.ai.as_ref().and_then(|a| a.flags.get(i).copied().flatten())
                                && f.any()
                            {
                                ui.label(RichText::new(format!("⚠ Check this row: {}", f.describe())).small().color(amber));
                            }
                        }
                    }
                });
                ui.horizontal(|ui| {
                    apply = ui.add_enabled(!editing_sheet, egui::Button::new("Apply translation")).clicked();
                    if ui.button("Discard preview").clicked() { self.translation.preview = None; }
                });
            }
            if let Some(editor) = &mut self.translation.edit {
                ui.separator();
                ui.label("One translation per original line. Empty removes it. Existing text is editable.");
                if ui.button("Paste a whole translated sheet…").clicked() { editor.paste_open = !editor.paste_open; }
                if editor.paste_open {
                    ui.label("Paste plain text, one line per original row, including blank rows. This changes the editor only.");
                    let family = translation_family(ui, self.translation_display);
                    ui.add(egui::TextEdit::multiline(&mut editor.paste).font(egui::FontId::new(14.0, family))
                        .desired_rows(4).desired_width(f32::INFINITY));
                    if ui.button("Use pasted rows in editor").clicked() {
                        match pasted_rows(&editor.paste, editor.rows.len()) {
                            Ok(rows) => { editor.rows = rows; editor.paste.clear(); editor.paste_open = false; }
                            Err(error) => self.status = error,
                        }
                    }
                }
                egui::ScrollArea::vertical().id_salt("translation_edit").max_height(230.0).show(ui, |ui| {
                    for (line, text) in editor.snapshot.lyrics.lines.iter().zip(&mut editor.rows) {
                        ui.label(RichText::new(&line.text).small());
                        let family = translation_family(ui, self.translation_display);
                        ui.add(egui::TextEdit::singleline(text).font(egui::FontId::new(14.0, family))
                            .desired_width(f32::INFINITY));
                    }
                });
                ui.horizontal(|ui| {
                    commit_edit = ui.add_enabled(!editing_sheet, egui::Button::new("Apply edits")).clicked();
                    if ui.button("Cancel edits").clicked() { self.translation.edit = None; }
                });
            }
        }); });
        ui.add_space(4.0);
        if cancel {
            self.cancel_translation();
            if self.translation.is_busy() {
                self.status = "Cancelling — waiting for the current bounded network call. Nothing will be applied.".into();
            }
        }
        if find {
            self.start_translation(track, false);
        }
        if machine {
            if self.translation_display == crate::lyrics::zh::ChineseDisplay::Original {
                self.translation_display = crate::lyrics::zh::ChineseDisplay::Traditional;
            }
            self.start_translation(track, true);
        }
        if edit {
            self.begin_translation_edit();
        }
        if apply {
            self.apply_translation_preview();
        }
        if commit_edit {
            self.commit_translation_edit();
        }
    }
}

impl LyricsWindow {
    /// The context-aware AI controls and what a request would send. Returns
    /// whether "Translate with AI" was pressed. Drawing never reads the
    /// environment, creates a client or sends anything.
    fn ai_panel(&mut self, ui: &mut egui::Ui, idle: bool) -> bool {
        // 1. Settings, edited as copies; the key is not a setting.
        let (mut model, mut fluent, mut review) =
            (self.translation.ai.model.clone(), self.translation.ai.fluent, self.translation.ai.review);
        {
            let status = &mut self.status;
            let s = &mut self.translation.ai;
            ui.add_enabled_ui(idle, |ui| {
                ui.horizontal_wrapped(|ui| {
                    ui.label("Model:");
                    egui::ComboBox::from_id_salt("ai_model")
                        .selected_text(if model.trim().is_empty() { "Choose…".to_string() } else { model.trim().to_string() })
                        .show_ui(ui, |ui| {
                            for m in ai::DOCUMENTED_MODELS {
                                ui.selectable_value(&mut model, m.to_string(), m);
                            }
                        });
                    ui.add(egui::TextEdit::singleline(&mut model).hint_text("or type a model ID").desired_width(130.0));
                    ui.hyperlink_to("Models", ai::MODELS_URL);
                    ui.hyperlink_to("Pricing", ai::PRICING_URL);
                });
                ui.label(RichText::new("No model is preselected; prices differ widely. A model OpenAI does not offer \
                    to your key is reported as an error and never replaced by another.").small());
                ui.horizontal_wrapped(|ui| {
                    ui.label("API key:");
                    if s.key.is_some() {
                        ui.label(RichText::new("set for this session only — never saved").small());
                        if ui.button("Forget key").clicked() {
                            s.key = None;
                        }
                    } else {
                        ui.add(egui::TextEdit::singleline(&mut s.key_input).password(true)
                            .hint_text("your own OpenAI API key").desired_width(170.0));
                        if ui.button("Use key").clicked() {
                            match ai::ApiKey::new(&s.key_input) {
                                Ok(k) => s.key = Some(k),
                                Err(e) => *status = e,
                            }
                            s.key_input.clear();
                        }
                        if ui.button(format!("Use {}", ai::ENV_KEY))
                            .on_hover_text("Reads this environment variable now. Nothing is sent until you press Translate with AI.")
                            .clicked()
                        {
                            match ai::ApiKey::from_env() {
                                Ok(k) => s.key = Some(k),
                                Err(e) => *status = e,
                            }
                        }
                    }
                });
                ui.horizontal_wrapped(|ui| {
                    ui.radio_value(&mut fluent, false, "Faithful");
                    ui.radio_value(&mut fluent, true, "More fluent (meaning kept)");
                    ui.checkbox(&mut review, "Review pass (one more paid request)");
                });
            });
        }
        // 2. Through the setters: a change clears the agreement now, before
        //    the disclosure, the checkbox and the button below read anything.
        self.set_ai_model(model);
        self.set_ai_fluent(fluent);
        self.set_ai_review(review);
        // 3. One configuration for the disclosure, the agreement and the job.
        let current = self.ai_config();
        let replace = self.translation.replace;
        let (sent, fill, bytes) = self.lyrics.as_ref().map_or((0, 0, 0), |l| {
            let rows: Vec<_> = l.lines.iter().filter(|x| !x.text.trim().is_empty()).collect();
            let fill = rows.iter().filter(|x| x.text.chars().any(char::is_alphanumeric)
                && (replace || x.translation.as_ref().is_none_or(|s| s.trim().is_empty()))).count();
            (rows.len(), fill, rows.iter().map(|x| x.text.trim().len()).sum::<usize>())
        });
        let requests = if self.translation.ai.review { "2 requests: translation, then one review" } else { "1 request" };
        let model = current.as_ref().map_or("the chosen model".to_string(), |c| c.model.clone());
        let filled = if replace {
            format!("all {fill} rows with words will be retranslated")
        } else {
            format!("only the {fill} rows without a translation will be filled")
        };
        ui.label(RichText::new(format!(
            "Sends to {} ({model}) the whole original sheet for context — {sent} rows, {bytes} bytes; {filled} — \
             plus the title and artist. Not sent: file paths, audio, settings or existing translations.\n\
             Paid with your key: {requests}, each limited to {} output tokens (reasoning tokens count and are billed) \
             and {} s; {} s in all; never retried. A request already sent may be billed even if you cancel.\n\
             Sent with store=false, so it is not kept as an OpenAI conversation, but OpenAI may keep abuse-monitoring \
             logs (up to 30 days): this is not zero retention or confidential processing. Do not send private lyrics. \
             Output is AI-assisted and may mistranslate; check it before applying.",
            ai::PROVIDER, ai::MAX_OUTPUT_TOKENS, ai::REQUEST_TIMEOUT.as_secs(), ai::OPERATION_TIMEOUT.as_secs())).small());
        let mut start = false;
        let was_agreed = self.ai_agreed().is_some();
        let mut agreed = was_agreed;
        let has_key = self.translation.ai.key.is_some();
        ui.horizontal_wrapped(|ui| {
            ui.hyperlink_to("OpenAI data controls", ai::DATA_CONTROLS_URL);
            ui.hyperlink_to("Pricing", ai::PRICING_URL);
            ui.add_enabled(idle && current.is_some(),
                egui::Checkbox::new(&mut agreed, "I agree to send this sheet to OpenAI with these settings"));
            start = ui.add_enabled(idle && was_agreed && agreed && has_key, egui::Button::new("Translate with AI")).clicked();
        });
        if agreed != was_agreed {
            self.agree_ai(agreed);
        }
        if current.is_none() {
            ui.label(RichText::new("Choose the original language and a model to continue.").small());
        } else if !has_key {
            ui.label(RichText::new("Enter your API key to continue.").small());
        }
        start
    }
}

/// The AI request for a sheet: every row's original text, and which rows
/// want a translation (every row with Replace, otherwise those without one).
fn ai_job_for(lyrics: &Lyrics, replace: bool, config: ai::Config, track: &TrackRef) -> Result<ai::Job, String> {
    let texts: Vec<&str> = lyrics.lines.iter().map(|l| l.text.as_str()).collect();
    let wanted: Vec<bool> = lyrics.lines.iter()
        .map(|l| replace || l.translation.as_ref().is_none_or(|s| s.trim().is_empty()))
        .collect();
    ai::Job::new(config, track.title, track.artist, &texts, &wanted)
}

/// A validated AI outcome as a translation result awaiting preview.
fn ai_translated(job: &ai::Job, o: ai::Outcome, replace: bool) -> Translated {
    let config = job.config().clone();
    Translated { rows: o.rows, credit: config.credit(&o.review), replaces_existing: replace,
                 ai: Some(AiResult { config, flags: o.flags, notes: o.notes, review: o.review }) }
}

fn pasted_rows(text: &str, expected: usize) -> Result<Vec<String>, String> {
    let mut rows: Vec<String> = text
        .split('\n')
        .map(|s| s.trim_end_matches('\r').to_string())
        .collect();
    // A conventional final newline is optional, but actual empty rows are
    // significant. Never silently zip/truncate a differently sized sheet.
    if rows.len() == expected + 1 && rows.last().is_some_and(String::is_empty) {
        rows.pop();
    }
    if rows.len() != expected {
        return Err(format!(
            "Expected {expected} translated rows, including blanks; got {}. Nothing was imported.",
            rows.len()
        ));
    }
    if rows
        .iter()
        .any(|s| s.len() > 4_000 || s.chars().any(|c| c.is_control() && c != '\t'))
    {
        return Err("A pasted row is too long or contains unsupported control characters. Nothing was imported.".into());
    }
    Ok(rows)
}

fn translation_family(ui: &egui::Ui, mode: crate::lyrics::zh::ChineseDisplay) -> egui::FontFamily {
    let name = match mode {
        crate::lyrics::zh::ChineseDisplay::Simplified => ZH_HANS,
        _ => ZH_HANT,
    };
    let family = egui::FontFamily::Name(name.into());
    // Headless harnesses can omit setup_fonts; never reference an absent
    // family. Product setup always installs both Chinese families.
    if ui.fonts(|f| f.families().contains(&family)) {
        family
    } else {
        egui::FontFamily::Proportional
    }
}

/// The sidecar's attribution is capped at this many characters.
const CREDIT_CHARS: usize = 160;
const UNREVIEWED: &str = "unreviewed draft";

/// `new` attribution, with `previous` when earlier rows survive. The cap
/// must never turn surviving generated rows into purportedly human or
/// reviewed ones, so the required warnings are reserved first and only the
/// optional history is cut: the machine-assisted note whenever any part is
/// machine-assisted, and the complete "unreviewed draft" marker whenever
/// unreviewed AI rows may survive. `new` itself always fits (an AI credit
/// with the longest model ID is under 150 characters).
fn merged_credit(new: &str, previous: Option<&str>) -> String {
    let machine = |s: &str| {
        let lower = s.to_lowercase();
        lower.contains("mymemory") || lower.contains("machine") || lower.contains("ai-assisted")
    };
    let unreviewed = |s: &str| s.to_lowercase().contains(UNREVIEWED);
    let prev = previous.filter(|p| *p != new);
    let required = if prev.is_some_and(unreviewed) && !unreviewed(new) {
        format!("Includes machine-assisted translation ({UNREVIEWED} AI rows); ")
    } else if prev.is_some_and(machine) && !machine(new) {
        "Includes machine-assisted translation; ".to_string()
    } else {
        String::new()
    };
    let optional = match prev {
        Some(old) => format!("{new}; previous: {old}"),
        None => new.to_string(),
    };
    let room = CREDIT_CHARS.saturating_sub(required.chars().count());
    format!("{required}{}", optional.chars().take(room).collect::<String>())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn window() -> LyricsWindow {
        LyricsWindow {
            loaded_for: Some(PathBuf::from("isolated-language-fixture.flac")),
            lyrics: Some(super::super::super::parse(
                "[ti:Example]\n[ar:Fixture]\n[offset:125]\n[00:01.00]<00:01.00>같이\n[00:02.00]Love",
                LyricSource::Embedded,
            )),
            ..Default::default()
        }
    }

    fn track() -> TrackRef<'static> {
        TrackRef {
            path: std::path::Path::new("isolated-language-fixture.flac"),
            artist: "Fixture",
            title: "Example",
            album: "",
            duration: None,
        }
    }

    fn result(replaces_existing: bool) -> Translated {
        Translated {
            rows: vec![Some("一起".into()), Some("愛".into())],
            credit: "Synthetic machine translation".into(),
            replaces_existing,
            ai: None,
        }
    }

    fn queue(w: &mut LyricsWindow) -> mpsc::Sender<Message> {
        let (tx, rx) = mpsc::channel();
        w.translation.task = Some(Task {
            snapshot: w.translation_snapshot().unwrap(),
            serial: w.translation.serial,
            cancel: Arc::new(AtomicBool::new(false)),
            rx,
            progress: (0, 0),
            ai: None,
        });
        tx
    }

    #[test]
    fn receipt_only_previews_and_apply_preserves_every_original_field() {
        let mut w = window();
        let before = w.lyrics.clone().unwrap();
        let tx = queue(&mut w);
        tx.send(Message::Progress(2, 2)).unwrap();
        tx.send(Message::Done(Ok(result(false)))).unwrap();
        w.poll_translation();
        assert!(!w.translation.is_busy());
        assert!(w.translation.preview.is_some());
        assert_eq!(w.lyrics.as_ref().unwrap().to_lrc(), before.to_lrc());
        assert!(!w.translation_dirty);
        assert!(
            w.lyrics
                .as_ref()
                .unwrap()
                .lines
                .iter()
                .all(|l| l.translation.is_none())
        );
        w.apply_translation_preview();
        let after = w.lyrics.as_ref().unwrap();
        assert_eq!(after.source, before.source);
        assert_eq!(after.to_lrc(), before.to_lrc());
        for (a, b) in after.lines.iter().zip(&before.lines) {
            assert_eq!(
                (&a.text, a.at, &a.marks, &a.romaji, &a.reading_override),
                (&b.text, b.at, &b.marks, &b.romaji, &b.reading_override)
            );
        }
        assert_eq!(after.lines[0].translation.as_deref(), Some("一起"));
        assert!(w.translation_dirty && w.show_translation);
        assert!(!w.dirty && !w.romaji_dirty);
        assert!(w.translation.preview.is_none());
    }

    #[test]
    fn superseded_track_revision_timing_and_text_cannot_receive_a_result() {
        for change in 0..5 {
            let mut w = window();
            let tx = queue(&mut w);
            match change {
                0 => w.loaded_for = Some(PathBuf::from("other.flac")),
                1 => w.sheet_rev += 1,
                2 => w.lyrics.as_mut().unwrap().offset_ms += 50,
                3 => w.lyrics.as_mut().unwrap().lines[0].text.push('!'),
                _ => w.translation.clear_sheet(),
            }
            w.status = "Current sheet status".into();
            tx.send(Message::Done(Ok(result(false)))).unwrap();
            w.poll_translation();
            assert!(!w.translation.is_busy());
            assert!(w.translation.preview.is_none());
            assert!(!w.translation_dirty);
            assert_eq!(w.status, "Current sheet status");
        }
    }

    #[test]
    fn cancelled_and_completed_but_unconsumed_jobs_keep_the_slot() {
        let mut w = window();
        let tx = queue(&mut w);
        w.translation.consent = true;
        w.translation.source = Some(SourceLanguage::Korean);
        let old_serial = w.translation.serial;
        w.poll_translation(); // empty, not retired
        w.cancel_translation();
        w.start_translation(&track(), true);
        assert!(w.translation.is_busy());
        assert_eq!(w.translation.serial, old_serial);
        tx.send(Message::Done(Ok(result(false)))).unwrap();
        w.start_translation(&track(), false); // queued completion still owns it
        assert_eq!(w.translation.serial, old_serial);
        w.poll_translation();
        assert!(!w.translation.is_busy());
        assert!(w.translation.preview.is_none());
        assert!(w.status.contains("cancelled"));
        assert!(!w.translation_dirty);
    }

    #[test]
    fn errors_and_disconnected_workers_retire_without_partial_application() {
        for disconnected in [false, true] {
            let mut w = window();
            let before = w.lyrics.as_ref().unwrap().to_lrc();
            let tx = queue(&mut w);
            if !disconnected {
                tx.send(Message::Done(Err("Synthetic quota error".into())))
                    .unwrap();
            }
            drop(tx);
            w.poll_translation();
            assert!(!w.translation.is_busy());
            assert!(w.translation.preview.is_none());
            assert_eq!(w.lyrics.as_ref().unwrap().to_lrc(), before);
            assert!(!w.translation_dirty);
            assert!(w.status.contains(if disconnected {
                "disconnected"
            } else {
                "quota"
            }));
        }
    }

    #[test]
    fn missing_consent_or_language_never_starts_a_worker() {
        for consent in [false, true] {
            let mut w = window();
            w.translation.consent = consent;
            if !consent {
                w.translation.source = Some(SourceLanguage::Korean);
            }
            w.start_translation(&track(), true);
            assert!(!w.translation.is_busy());
            assert_eq!(w.translation.serial, 0);
        }
    }

    #[test]
    fn an_editor_cancels_the_old_request_and_apply_rechecks_a_preview() {
        for reading in [false, true] {
            let mut w = window();
            let tx = queue(&mut w);
            if reading {
                w.begin_romaji_edit();
            } else {
                w.begin_edit();
            }
            assert!(
                w.translation
                    .task
                    .as_ref()
                    .unwrap()
                    .cancel
                    .load(Ordering::Acquire)
            );
            w.status = "Editing the sheet".into();
            tx.send(Message::Done(Ok(result(false)))).unwrap();
            w.poll_translation();
            assert!(w.translation.preview.is_none());
            assert_eq!(w.status, "Editing the sheet");
        }
        let mut w = window();
        let tx = queue(&mut w);
        tx.send(Message::Done(Ok(result(false)))).unwrap();
        w.poll_translation();
        w.lyrics.as_mut().unwrap().lines[0].romaji = Some("Changed".into());
        w.apply_translation_preview();
        assert!(!w.translation_dirty);
        assert!(w.lyrics.as_ref().unwrap().lines[0].translation.is_none());
    }

    #[test]
    fn supplied_supplement_keeps_machine_attribution_despite_replace_checkbox() {
        let mut w = window();
        let l = w.lyrics.as_mut().unwrap();
        l.lines[0].translation = Some("Prior machine row".into());
        l.translation_credit = Some("Prior machine credit".into());
        let tx = queue(&mut w);
        w.translation.replace = true;
        tx.send(Message::Done(Ok(Translated {
            rows: vec![None, Some("愛".into())],
            credit: "Supplement supplied by NetEase".into(),
            replaces_existing: false,
            ai: None,
        })))
        .unwrap();
        w.poll_translation();
        w.apply_translation_preview();
        let l = w.lyrics.as_ref().unwrap();
        assert_eq!(l.lines[0].translation.as_deref(), Some("Prior machine row"));
        let credit = l.translation_credit.as_ref().unwrap();
        assert!(credit.contains("Prior machine credit") && credit.contains("NetEase"));
    }

    #[test]
    fn replacement_policy_is_captured_not_read_from_mutable_ui_state() {
        let mut w = window();
        w.lyrics.as_mut().unwrap().translation_credit = Some("Old provider".into());
        let tx = queue(&mut w);
        tx.send(Message::Done(Ok(result(true)))).unwrap();
        w.poll_translation();
        w.translation.replace = false;
        w.apply_translation_preview();
        assert_eq!(
            w.lyrics.as_ref().unwrap().translation_credit.as_deref(),
            Some("Synthetic machine translation")
        );
    }

    #[test]
    fn unsaved_manual_rows_survive_competing_actions_and_can_be_applied() {
        let mut w = window();
        let before = w.lyrics.as_ref().unwrap().to_lrc();
        w.begin_translation_edit();
        w.translation.edit.as_mut().unwrap().rows[0] = "一起唱".into();
        w.translation.consent = true;
        w.translation.source = Some(SourceLanguage::Korean);
        w.start_translation(&track(), true);
        w.start_translation(&track(), false);
        w.start_fetch(&track());
        w.start_search();
        w.begin_edit();
        w.begin_romaji_edit();
        w.paste_buf = "Replacement sheet".into();
        w.take_paste();
        let hit = Hit {
            plain: Some("Different sheet".into()),
            ..Default::default()
        };
        w.choose(hit.clone());
        w.apply(&hit);
        assert!(!w.busy()); // manual edits do not trigger polling repaints
        assert!(!w.editing && !w.romaji_editing);
        assert_eq!(w.translation.edit.as_ref().unwrap().rows[0], "一起唱");
        w.commit_translation_edit();
        assert_eq!(w.lyrics.as_ref().unwrap().to_lrc(), before);
        assert_eq!(
            w.lyrics.as_ref().unwrap().lines[0].translation.as_deref(),
            Some("一起唱")
        );
        assert!(w.translation_dirty);
    }

    #[test]
    fn closing_the_panel_cancels_but_preserves_a_manual_buffer() {
        let mut w = window();
        w.open_translation();
        assert!(w.translation.open);
        let tx = queue(&mut w);
        w.open_translation();
        assert!(!w.translation.open);
        assert!(
            w.translation
                .task
                .as_ref()
                .unwrap()
                .cancel
                .load(Ordering::Acquire)
        );
        drop(tx);
        w.poll_translation();
        w.begin_translation_edit();
        w.translation.edit.as_mut().unwrap().rows[0] = "Unapplied".into();
        w.open_translation();
        w.open_translation();
        assert_eq!(w.translation.edit.as_ref().unwrap().rows[0], "Unapplied");
    }

    #[test]
    fn bulk_paste_is_row_exact_and_does_not_silently_truncate() {
        assert_eq!(
            pasted_rows("first\r\n\r\nlast\r\n", 3).unwrap(),
            ["first", "", "last"]
        );
        assert_eq!(pasted_rows("first\n\n", 2).unwrap(), ["first", ""]);
        assert!(pasted_rows("first", 2).is_err());
        assert!(pasted_rows("first\nsecond\nthird", 2).is_err());
        assert!(pasted_rows("first\0\nsecond", 2).is_err());
    }

    #[test]
    fn editing_originals_revokes_sheet_specific_consent() {
        let mut w = window();
        w.translation.consent = true;
        w.translation.source = Some(SourceLanguage::Japanese);
        w.begin_edit();
        assert!(!w.translation.consent);
        assert!(w.translation.source.is_none());
    }

    #[test]
    fn attribution_cap_never_erases_surviving_machine_provenance() {
        let mut credit = "MyMemory machine translation".to_string();
        for _ in 0..20 {
            credit = merged_credit("Supplement supplied by NetEase", Some(&credit));
            credit = merged_credit("Manually edited translation", Some(&credit));
            assert!(credit.chars().count() <= 160);
            assert!(credit.starts_with("Includes machine-assisted translation"));
        }
    }

    #[test]
    fn manual_lrc_syntax_is_refused_without_losing_the_editor_buffer() {
        for unsafe_text in [
            "[ti:天空]",
            "[00:01.00]天空",
            "<00:01.00>天空",
            "天空\n第二行",
        ] {
            let mut w = window();
            w.begin_translation_edit();
            w.translation.edit.as_mut().unwrap().rows[0] = unsafe_text.into();
            w.commit_translation_edit();
            assert!(w.translation.edit.is_some());
            assert_eq!(w.translation.edit.as_ref().unwrap().rows[0], unsafe_text);
            assert!(!w.translation_dirty);
            assert!(w.lyrics.as_ref().unwrap().lines[0].translation.is_none());
        }
    }

    #[test]
    fn pending_reading_confirmation_cannot_discard_manual_translation_rows() {
        let mut w = window();
        w.lyrics = Some(super::super::super::parse(
            "[00:01.00]剣(つるぎ)を",
            LyricSource::Online,
        ));
        w.begin_romaji_edit();
        w.romaji_edit[0].text = "ken o".into();
        w.commit_romaji_edit();
        assert!(!w.pending.is_empty());
        let count = w.pending.len();
        w.begin_translation_edit();
        w.translation.edit.as_mut().unwrap().rows[0] = "手中的劍".into();
        w.resolve_pending(true);
        assert_eq!(w.pending.len(), count);
        w.commit_translation_edit();
        assert_eq!(
            w.lyrics.as_ref().unwrap().lines[0].translation.as_deref(),
            Some("手中的劍")
        );

        w.begin_translation_edit();
        w.translation.edit.as_mut().unwrap().rows[0] = "Keep this draft".into();
        w.sheet_rev += 1; // even external changes cannot silently destroy it
        w.commit_translation_edit();
        assert_eq!(
            w.translation.edit.as_ref().unwrap().rows[0],
            "Keep this draft"
        );
        assert!(w.status.contains("buffer is kept"));
    }

    /// One frame of the real `LyricsWindow::ui`, headless (the viewport is
    /// embedded); `close` delivers the native close request.
    fn frame(w: &mut LyricsWindow, ctx: &egui::Context, present: bool, close: bool) {
        let mut input = egui::RawInput {
            screen_rect: Some(egui::Rect::from_min_size(
                egui::Pos2::ZERO,
                egui::vec2(520.0, 640.0),
            )),
            ..Default::default()
        };
        if close {
            input
                .viewports
                .entry(egui::ViewportId::ROOT)
                .or_default()
                .events
                .push(egui::ViewportEvent::Close);
        }
        let _ = ctx.run(input, |ctx| {
            let _ = w.ui(ctx, present.then(track), Duration::ZERO);
        });
    }

    #[test]
    fn toolbar_and_native_close_abandon_translation_until_the_late_completion_is_consumed() {
        let ctx = egui::Context::default();
        for native in [false, true] {
            let mut w = window();
            w.set_open(true);
            let tx = queue(&mut w);
            let serial = w.translation.serial;
            if native {
                frame(&mut w, &ctx, true, true);
            } else {
                w.toggle(); // what the main toolbar's Lyrics button calls
            }
            assert!(!w.is_open());
            let task = w.translation.task.as_ref().expect("slot held until consumed");
            assert!(task.cancel.load(Ordering::Acquire));
            assert_ne!(w.translation.serial, serial);
            // Hidden: never drawn, so nothing is consumed or adopted.
            tx.send(Message::Done(Ok(result(false)))).unwrap();
            assert_eq!(w.ui(&ctx, Some(track()), Duration::ZERO), LyricsAction::None);
            assert!(w.translation.is_busy());
            // Reopened on the same track and sheet: consumed, not adopted.
            w.toggle();
            frame(&mut w, &ctx, true, false);
            assert!(!w.translation.is_busy());
            assert!(w.translation.preview.is_none());
            assert!(!w.translation_dirty);
            assert!(w.lyrics.as_ref().unwrap().lines.iter().all(|l| l.translation.is_none()));
            assert!(!w.busy()); // the slot is free for a new request
        }
    }

    #[test]
    fn hiding_drops_an_unapplied_preview_but_keeps_a_manual_draft() {
        let mut w = window();
        w.set_open(true);
        let tx = queue(&mut w);
        tx.send(Message::Done(Ok(result(false)))).unwrap();
        w.poll_translation();
        assert!(w.translation.preview.is_some());
        w.toggle();
        w.toggle();
        assert!(w.translation.preview.is_none());
        w.apply_translation_preview();
        assert!(!w.translation_dirty);

        w.begin_translation_edit();
        w.translation.edit.as_mut().unwrap().rows[0] = "Unapplied".into();
        w.set_open(false);
        w.set_open(true);
        assert_eq!(w.translation.edit.as_ref().unwrap().rows[0], "Unapplied");
        w.commit_translation_edit();
        assert_eq!(
            w.lyrics.as_ref().unwrap().lines[0].translation.as_deref(),
            Some("Unapplied")
        );
        assert!(w.translation_dirty);
    }

    #[test]
    fn losing_the_current_track_abandons_translation_even_if_the_same_file_returns() {
        let ctx = egui::Context::default();
        // 0: completes while absent; 1: completes after the same file returns;
        // 2: a preview received before the loss.
        for case in 0..3 {
            let mut w = window();
            w.set_open(true);
            frame(&mut w, &ctx, true, false);
            w.dirty = true; // an unrelated unsaved sheet edit
            w.begin_translation_edit();
            w.translation.edit.as_mut().unwrap().rows[1] = "Draft".into();
            let draft = w.translation.edit.take();
            let tx = queue(&mut w);
            if case == 2 {
                tx.send(Message::Done(Ok(result(false)))).unwrap();
                w.poll_translation();
                assert!(w.translation.preview.is_some());
            }
            frame(&mut w, &ctx, false, false); // Some(track) -> None
            assert!(w.translation.preview.is_none());
            if case < 2 {
                assert!(w.translation.task.as_ref().unwrap().cancel.load(Ordering::Acquire));
            }
            if case == 0 {
                tx.send(Message::Done(Ok(result(false)))).unwrap();
                frame(&mut w, &ctx, false, false); // consumed while absent
                assert!(!w.translation.is_busy());
            }
            w.translation.edit = draft;
            frame(&mut w, &ctx, true, false); // the same pathname again
            if case == 1 {
                tx.send(Message::Done(Ok(result(false)))).unwrap();
                frame(&mut w, &ctx, true, false);
            }
            assert!(!w.translation.is_busy());
            assert!(w.translation.preview.is_none());
            w.apply_translation_preview();
            assert!(!w.translation_dirty);
            assert!(w.lyrics.as_ref().unwrap().lines.iter().all(|l| l.translation.is_none()));
            // Nothing unrelated was discarded.
            assert!(w.dirty);
            assert_eq!(w.translation.edit.as_ref().unwrap().rows[1], "Draft");
        }
    }

    #[test]
    fn a_manual_draft_survives_losing_and_regaining_the_track() {
        let ctx = egui::Context::default();
        let mut w = window();
        w.set_open(true);
        frame(&mut w, &ctx, true, false);
        w.begin_translation_edit();
        w.translation.edit.as_mut().unwrap().rows[0] = "Kept".into();
        frame(&mut w, &ctx, false, false);
        frame(&mut w, &ctx, true, false);
        w.commit_translation_edit();
        assert_eq!(
            w.lyrics.as_ref().unwrap().lines[0].translation.as_deref(),
            Some("Kept")
        );
    }

    #[test]
    fn translation_panels_build_in_a_normal_headless_lyrics_viewport() {
        let ctx = egui::Context::default();
        let mut w = window();
        w.translation.open = true;
        w.translation.source = Some(SourceLanguage::Korean);
        for stage in 0..3 {
            if stage == 1 {
                let tx = queue(&mut w);
                tx.send(Message::Done(Ok(result(false)))).unwrap();
                w.poll_translation();
            } else if stage == 2 {
                w.begin_translation_edit();
                w.translation.edit.as_mut().unwrap().paste_open = true;
            }
            let input = egui::RawInput {
                screen_rect: Some(egui::Rect::from_min_size(
                    egui::Pos2::ZERO,
                    egui::vec2(520.0, 640.0),
                )),
                ..Default::default()
            };
            let output = ctx.run(input, |ctx| {
                egui::CentralPanel::default().show(ctx, |ui| {
                    w.translation_panel(ui, &track());
                });
            });
            assert!(!output.shapes.is_empty());
            assert!(
                !w.translation_dirty,
                "drawing a panel cannot adopt a translation"
            );
        }
    }

    // ── Context-aware AI mode ───────────────────────────────────────────

    use crate::lyrics::ai::tests::{SEND_GATE, SENTINEL_KEY, answer, completed};
    use crate::lyrics::ai::{Reply, Transport};
    use std::collections::VecDeque;

    /// Configured and agreed, through the same setters the panel uses.
    fn ai_ready_with(w: &mut LyricsWindow, model: &str, source: SourceLanguage) {
        w.set_provider(Provider::Ai);
        w.set_source(Some(source));
        w.set_ai_model(model.into());
        w.translation.ai.key = Some(ai::ApiKey::new(SENTINEL_KEY).unwrap());
        w.agree_ai(true);
        assert!(w.ai_agreed().is_some());
    }

    fn ai_ready(w: &mut LyricsWindow) {
        ai_ready_with(w, "gpt-test-model", SourceLanguage::Korean);
    }

    fn ai_result(w: &LyricsWindow, review: ai::Review) -> Translated {
        let config = w.ai_config().unwrap();
        Translated {
            rows: vec![Some("一起".into()), Some("愛".into())],
            credit: config.credit(&review),
            replaces_existing: false,
            ai: Some(AiResult {
                config,
                flags: vec![None, Some(ai::RowFlags { model_doubtful: true, ..Default::default() })],
                notes: "The second row could mean either kind of love.".into(),
                review,
            }),
        }
    }

    fn ai_queue(w: &mut LyricsWindow) -> mpsc::Sender<Message> {
        let tx = queue(w);
        w.translation.task.as_mut().unwrap().ai = w.ai_config();
        tx
    }

    fn deliver(w: &mut LyricsWindow, t: Translated) {
        let tx = ai_queue(w);
        tx.send(Message::Done(Ok(t))).unwrap();
        w.poll_translation();
    }

    fn sends() -> usize {
        ai::REAL_SENDS.load(Ordering::SeqCst)
    }

    /// The real backend on the window's own sheet and settings against canned
    /// replies, turned into a result exactly as the worker does.
    fn canned(w: &LyricsWindow, replies: Vec<Result<Reply, Transport>>) -> Result<Translated, String> {
        let job = ai_job_for(w.lyrics.as_ref().unwrap(), w.translation.replace, w.ai_config().unwrap(), &track())?;
        let mut replies = VecDeque::from(replies);
        let o = ai::translate_with(&job, w.translation.ai.key.as_ref().unwrap(), &AtomicBool::new(false), |_, _| {},
                                   |_, _, _| replies.pop_front().expect("unexpected request"))?;
        Ok(ai_translated(&job, o, w.translation.replace))
    }

    fn ok(rows: &[(&str, &str)]) -> Result<Reply, Transport> {
        Ok(Reply { status: 200, body: completed(&answer(rows)) })
    }

    fn sheet(text: &str) -> LyricsWindow {
        LyricsWindow {
            loaded_for: Some(PathBuf::from("isolated-language-fixture.flac")),
            lyrics: Some(super::super::super::parse(text, LyricSource::Embedded)),
            ..Default::default()
        }
    }

    /// Save into a private fixture directory and load back.
    fn save_and_reload(l: &Lyrics) -> (Lyrics, Vec<u8>) {
        let root = std::env::temp_dir();
        let dir = (0u32..).map(|n| root.join(format!("moosik_ai_save_{}_{n}", std::process::id())))
            .find(|d| std::fs::create_dir(d).is_ok()).unwrap();
        let song = dir.join("song.flac");
        std::fs::write(&song, [0x4d, 0x6f, 0x6f]).unwrap();
        std::fs::write(dir.join("song.lrc"), l.to_lrc()).unwrap();
        let outcome = crate::lyrics::save_changes(&song, l, false, false, true);
        assert!(matches!(outcome, crate::lyrics::SaveOutcome::Saved { .. }), "{outcome:?}");
        let back = crate::lyrics::load(&song, None).unwrap();
        let bytes = std::fs::read(dir.join("song.translation.lrc")).unwrap();
        if dir.parent() == Some(root.as_path()) {
            let _ = std::fs::remove_dir_all(&dir);
        }
        (back, bytes)
    }

    #[test]
    fn basic_mode_and_unconfigured_or_unclicked_ai_never_send_anything() {
        let _gate = SEND_GATE.lock().unwrap_or_else(|e| e.into_inner());
        let before = sends();
        let ctx = egui::Context::default();
        for stage in 0..4 {
            let mut w = window();
            w.set_open(true);
            w.translation.open = true;
            match stage {
                0 => {}                                                   // Basic (the default)
                1 => w.set_provider(Provider::Ai),                        // AI, nothing configured
                2 => { ai_ready(&mut w); w.agree_ai(false); }             // configured, no agreement
                _ => ai_ready(&mut w),                                    // ready, never clicked
            }
            for present in [true, false, true, true] {
                frame(&mut w, &ctx, present, false); // playback/track changes
            }
            if stage == 1 || stage == 2 {
                w.start_translation(&track(), true); // not configured / not agreed
            }
            assert!(!w.translation.is_busy(), "stage {stage} started a worker");
            if stage < 2 {
                assert!(w.translation.ai.key.is_none(), "no key was looked for");
            }
        }
        assert_eq!(sends(), before, "nothing reached the transport");
    }

    #[test]
    fn an_explicit_ai_request_uses_the_one_worker_slot_and_sends_once() {
        let _gate = SEND_GATE.lock().unwrap_or_else(|e| e.into_inner());
        let before = sends();
        let mut w = window();
        ai_ready(&mut w);
        w.start_translation(&track(), true);
        assert!(w.translation.is_busy());
        assert_eq!(w.status, "Translating with OpenAI…");
        assert_eq!(w.translation.task.as_ref().unwrap().ai, w.ai_config());
        let serial = w.translation.serial;
        w.start_translation(&track(), true); // the slot is taken
        assert_eq!(w.translation.serial, serial);
        let deadline = std::time::Instant::now() + Duration::from_secs(20);
        while w.translation.is_busy() {
            assert!(std::time::Instant::now() < deadline, "worker did not complete");
            w.poll_translation();
            std::thread::yield_now();
        }
        // The test build's transport is inert: one attempt, refused, nothing
        // adopted and nothing retried.
        assert_eq!(sends(), before + 1);
        assert!(w.status.contains("secure connection"), "{}", w.status);
        assert!(w.translation.preview.is_none() && !w.translation_dirty);
    }

    /// One setting change: (name, change to B, change back to A), each
    /// through the panel's setters.
    type SettingChange = (&'static str, fn(&mut LyricsWindow), fn(&mut LyricsWindow));

    fn setting_changes() -> Vec<SettingChange> {
        vec![
            ("model", |w| w.set_ai_model("gpt-other".into()), |w| w.set_ai_model("gpt-test-model".into())),
            ("style", |w| w.set_ai_fluent(true), |w| w.set_ai_fluent(false)),
            ("review", |w| w.set_ai_review(true), |w| w.set_ai_review(false)),
            ("source", |w| w.set_source(Some(SourceLanguage::Japanese)), |w| w.set_source(Some(SourceLanguage::Korean))),
            ("replace", |w| w.set_replace(true), |w| w.set_replace(false)),
            ("provider", |w| w.set_provider(Provider::Basic), |w| w.set_provider(Provider::Ai)),
        ]
    }

    #[test]
    fn every_setting_change_expires_agreement_and_going_back_does_not_revive_it() {
        let _gate = SEND_GATE.lock().unwrap_or_else(|e| e.into_inner());
        let before = sends();
        for (name, to_b, to_a) in setting_changes() {
            let mut w = window();
            ai_ready(&mut w);
            to_b(&mut w);
            assert!(w.translation.ai.consent.is_none(), "{name}: A -> B kept the agreement");
            to_a(&mut w);
            assert!(w.ai_agreed().is_none(), "{name}: A -> B -> A revived the agreement");
            w.start_translation(&track(), true);
            assert!(!w.translation.is_busy(), "{name}: started without renewed agreement");
            w.agree_ai(true);
            assert!(w.ai_agreed().is_some(), "{name}: agreeing again works");
        }
        // Setting the same value is not a change.
        let mut w = window();
        ai_ready(&mut w);
        w.set_ai_model("gpt-test-model".into());
        w.set_provider(Provider::Ai);
        assert!(w.ai_agreed().is_some());
        assert_eq!(sends(), before);
    }

    #[test]
    fn a_changed_or_replaced_sheet_expires_agreement_but_a_key_change_does_not() {
        let _gate = SEND_GATE.lock().unwrap_or_else(|e| e.into_inner());
        let before = sends();
        // Any sheet revision no longer matches.
        let mut w = window();
        ai_ready(&mut w);
        w.sheet_rev += 1;
        assert!(w.ai_agreed().is_none());
        w.start_translation(&track(), true);
        assert!(!w.translation.is_busy());
        // Editing the originals clears it.
        let mut w = window();
        ai_ready(&mut w);
        w.begin_edit();
        assert!(w.translation.ai.consent.is_none());
        // Replacing the sheet clears it.
        let mut w = window();
        ai_ready(&mut w);
        w.paste_buf = "[00:01.00]다른 노래".into();
        w.take_paste();
        assert!(w.ai_agreed().is_none());
        // Another track's sheet does not match.
        let mut w = window();
        ai_ready(&mut w);
        w.loaded_for = Some(PathBuf::from("other.flac"));
        assert!(w.ai_agreed().is_none());
        // The key is not part of the agreement.
        let mut w = window();
        ai_ready(&mut w);
        w.translation.ai.key = None;
        w.translation.ai.key = Some(ai::ApiKey::new("sk-another-SENTINEL-key-0123456789").unwrap());
        assert!(w.ai_agreed().is_some());
        assert_eq!(sends(), before);
    }

    #[test]
    fn a_completion_made_under_other_settings_is_not_adopted() {
        for (name, to_b, _) in setting_changes() {
            if name == "replace" {
                continue; // the replace policy is captured in the result itself
            }
            let mut w = window();
            ai_ready(&mut w);
            let tx = ai_queue(&mut w);
            let result = ai_result(&w, ai::Review::NotRequested);
            to_b(&mut w);
            w.status = "Current status".into();
            tx.send(Message::Done(Ok(result))).unwrap();
            w.poll_translation();
            assert!(!w.translation.is_busy(), "{name}");
            assert!(w.translation.preview.is_none(), "{name}: adopted at completion");
            assert_eq!(w.status, "Current status");
        }
    }

    #[test]
    fn apply_rechecks_the_settings_even_without_drawing_the_panel() {
        for (name, to_b, _) in setting_changes() {
            if name == "replace" {
                continue;
            }
            let mut w = window();
            ai_ready(&mut w);
            { let r = ai_result(&w, ai::Review::NotRequested); deliver(&mut w, r); }
            assert!(w.translation.preview.is_some(), "{name}");
            to_b(&mut w);
            w.apply_translation_preview();
            assert!(!w.translation_dirty, "{name}: applied under other settings");
            assert!(w.lyrics.as_ref().unwrap().lines.iter().all(|l| l.translation.is_none()));
        }
        // Forgetting the key is not a settings change.
        let mut w = window();
        ai_ready(&mut w);
        { let r = ai_result(&w, ai::Review::Reviewed); deliver(&mut w, r); }
        w.translation.ai.key = None;
        w.apply_translation_preview();
        assert_eq!(w.lyrics.as_ref().unwrap().lines[1].translation.as_deref(), Some("愛"));
    }

    #[test]
    fn drawing_the_panel_drops_a_preview_made_under_other_settings() {
        let ctx = egui::Context::default();
        let mut w = window();
        ai_ready(&mut w);
        { let r = ai_result(&w, ai::Review::NotRequested); deliver(&mut w, r); }
        w.set_ai_fluent(true);
        let _ = ctx.run(egui::RawInput::default(), |ctx| {
            egui::CentralPanel::default().show(ctx, |ui| w.translation_panel(ui, &track()));
        });
        assert!(w.translation.preview.is_none());
    }

    #[test]
    fn agreement_needs_settings_and_a_key() {
        let mut w = window();
        w.set_provider(Provider::Ai);
        w.agree_ai(true);
        assert!(w.translation.ai.consent.is_none(), "nothing to agree to without a model and source");
        ai_ready(&mut w);
        w.translation.ai.key = None;
        w.start_translation(&track(), true);
        assert!(!w.translation.is_busy(), "no key");
        w.translation.clear_sheet();
        assert!(w.translation.ai.consent.is_none());
    }

    #[test]
    fn closing_or_losing_the_track_abandons_ai_work_like_any_other() {
        let ctx = egui::Context::default();
        for lose_track in [false, true] {
            let mut w = window();
            ai_ready(&mut w);
            w.set_open(true);
            frame(&mut w, &ctx, true, false);
            let tx = ai_queue(&mut w);
            let result = ai_result(&w, ai::Review::Reviewed);
            if lose_track {
                frame(&mut w, &ctx, false, false);
            } else {
                w.toggle();
            }
            assert!(w.translation.task.as_ref().unwrap().cancel.load(Ordering::Acquire));
            tx.send(Message::Done(Ok(result))).unwrap();
            if !lose_track {
                w.toggle();
            }
            frame(&mut w, &ctx, true, false);
            assert!(!w.translation.is_busy() && w.translation.preview.is_none());
            assert!(w.lyrics.as_ref().unwrap().lines.iter().all(|l| l.translation.is_none()));
        }
    }

    /// 2B. A failed-review draft's untouched rows keep the full warning
    /// through manual and supplied updates, the cap, a save and a reload.
    #[test]
    fn unreviewed_ai_rows_keep_their_complete_warning_through_edits_save_and_reload() {
        let longest = format!("gpt-{}", "x".repeat(60));
        assert_eq!(longest.len(), 64);
        for model in ["gpt-test-model", longest.as_str()] {
            let mut w = sheet("[00:01.00]같이\n[00:02.00]사랑");
            ai_ready_with(&mut w, model, SourceLanguage::Korean);
            w.set_ai_review(true);
            w.agree_ai(true);
            let t = canned(&w, vec![ok(&[("L001", "一起"), ("L002", "愛")]), Err(Transport::Timeout)]).unwrap();
            deliver(&mut w, t);
            assert!(w.status.contains("Unreviewed AI draft"), "{}", w.status);
            w.apply_translation_preview();
            let warned = |c: &str| c.contains("machine-assisted") || c.starts_with("AI-assisted");
            let check = |w: &LyricsWindow, when: &str| {
                let c = w.lyrics.as_ref().unwrap().translation_credit.clone().unwrap();
                assert!(c.chars().count() <= 160, "{when}: {c}");
                assert!(warned(&c) && c.contains("unreviewed draft"), "{model} {when}: {c}");
                assert_eq!(w.lyrics.as_ref().unwrap().lines[1].translation.as_deref(), Some("愛"), "{when}");
            };
            check(&w, "after apply");
            for i in 0..12 {
                // Manually edit only row 1, then a supplied row 1: row 2 stays AI.
                w.begin_translation_edit();
                w.translation.edit.as_mut().unwrap().rows[0] = format!("一同{i}");
                w.commit_translation_edit();
                check(&w, "after a manual edit");
                let tx = queue(&mut w);
                tx.send(Message::Done(Ok(Translated { rows: vec![Some(format!("供應{i}")), None],
                    credit: "Supplement supplied by NetEase".into(), replaces_existing: false, ai: None }))).unwrap();
                w.poll_translation();
                w.apply_translation_preview();
                check(&w, "after a supplied row");
            }
            let (back, _) = save_and_reload(w.lyrics.as_ref().unwrap());
            let c = back.translation_credit.unwrap();
            assert!(warned(&c) && c.contains("unreviewed draft"), "{model} after reload: {c}");
            assert_eq!(back.lines[1].translation.as_deref(), Some("愛"));
            // Editing every row is still ordinary editing, not proof that
            // the AI text is gone: the complete warning stays (correction 2).
            w.begin_translation_edit();
            w.translation.edit.as_mut().unwrap().rows = vec!["一同".into(), "愛情".into()];
            w.commit_translation_edit();
            let c = w.lyrics.as_ref().unwrap().translation_credit.clone().unwrap();
            assert!(warned(&c) && c.contains("unreviewed draft") && c.chars().count() <= 160, "{model}: {c}");
        }
    }

    /// Correction 2. Ordinary editing of AI rows never turns the attribution
    /// into a purely manual one: small corrections or punctuation and
    /// spacing changes to every row, one edited row, a supplement after
    /// edits, save and reload all keep the machine-assisted note and the
    /// complete unreviewed marker within the cap. Clearing every row is the
    /// editor action that drops the attribution.
    #[test]
    fn ordinary_edits_keep_machine_provenance_until_every_row_is_cleared() {
        let longest = format!("gpt-{}", "x".repeat(60));
        let edits: [&[&str]; 4] = [
            &["我們一起走吧", "我很愛你", ""],         // a small correction to every AI row
            &["我們一起走！", "我 愛你", ""],          // punctuation and spacing in every row
            &["我們一起走吧", "我愛你", ""],           // one row edited, one unchanged
            &["我們 一起走", "我愛你。", ""],
        ];
        for edit in edits {
            let mut w = sheet("[00:01.00]같이 가자
[00:02.00]사랑해
[00:03.00]오오");
            ai_ready_with(&mut w, &longest, SourceLanguage::Korean);
            w.set_ai_review(true);
            w.agree_ai(true);
            let t = canned(&w, vec![ok(&[("L001", "我們一起走"), ("L002", "我愛你"), ("L003", "哦哦")]), Err(Transport::Network)]).unwrap();
            deliver(&mut w, t);
            w.apply_translation_preview();
            let check = |w: &LyricsWindow, when: &str| {
                let c = w.lyrics.as_ref().unwrap().translation_credit.clone().unwrap();
                assert!(c.chars().count() <= 160, "{edit:?} {when}: {c}");
                assert!(c.contains("machine-assisted") || c.starts_with("AI-assisted"), "{edit:?} {when}: {c}");
                assert!(c.contains("unreviewed draft"), "{edit:?} {when}: {c}");
                assert_ne!(c, "Manually edited translation");
            };
            check(&w, "applied");
            w.begin_translation_edit();
            for (row, text) in w.translation.edit.as_mut().unwrap().rows.iter_mut().zip(edit) {
                if !text.is_empty() {
                    *row = text.to_string();
                }
            }
            w.commit_translation_edit();
            check(&w, "edited");
            assert_eq!(w.lyrics.as_ref().unwrap().lines[0].translation.as_deref(), Some(edit[0].trim()));
            // A supplied supplement after the edits.
            let tx = queue(&mut w);
            tx.send(Message::Done(Ok(Translated { rows: vec![None, None, Some("喔喔".into())],
                credit: "Supplement supplied by NetEase".into(), replaces_existing: false, ai: None }))).unwrap();
            w.poll_translation();
            w.apply_translation_preview();
            check(&w, "supplemented");
            let (back, _) = save_and_reload(w.lyrics.as_ref().unwrap());
            let c = back.translation_credit.clone().unwrap();
            assert_eq!(Some(&c), w.lyrics.as_ref().unwrap().translation_credit.as_ref(), "the whole attribution reloads");
            assert!(c.contains("machine-assisted") && c.contains("unreviewed draft"), "reloaded: {c}");
            // Clearing every row leaves nothing to attribute.
            w.begin_translation_edit();
            w.translation.edit.as_mut().unwrap().rows.iter_mut().for_each(|r| r.clear());
            w.commit_translation_edit();
            let l = w.lyrics.as_ref().unwrap();
            assert!(l.lines.iter().all(|l| l.translation.is_none()));
            assert_eq!(l.translation_credit, None, "cleared: no sidecar is written for an empty translation");
        }
    }

    /// 2C. Unchanged / non-Chinese gaps are a flagged preview, applied only
    /// on Apply, mapped by occurrence ID, existing rows untouched.
    #[test]
    fn unchanged_gaps_are_a_flagged_preview_that_only_apply_adopts() {
        let mut w = sheet("[00:01.00]Alice\n[00:02.00]oh oh\n[00:03.00]你好\n[00:04.00]空を見て");
        w.lyrics.as_mut().unwrap().lines[3].translation = Some("看天空".into());
        ai_ready_with(&mut w, "gpt-test-model", SourceLanguage::Japanese);
        let t = canned(&w, vec![ok(&[("L001", "Alice"), ("L002", "oh oh"), ("L003", "你好")])]).unwrap();
        deliver(&mut w, t);
        assert!(w.status.contains("3 row(s) flagged"), "{}", w.status);
        let p = w.translation.preview.as_ref().unwrap();
        assert_eq!(p.result.rows, [Some("Alice".into()), Some("oh oh".into()), Some("你好".into()), None]);
        let flags = &p.result.ai.as_ref().unwrap().flags;
        assert!(flags[..3].iter().all(|f| f.is_some_and(|f| f.unchanged)));
        assert!(flags[3].is_none());
        assert!(w.lyrics.as_ref().unwrap().lines[..3].iter().all(|l| l.translation.is_none()), "not applied before Apply");
        w.apply_translation_preview();
        let l = w.lyrics.as_ref().unwrap();
        assert_eq!(l.lines[0].translation.as_deref(), Some("Alice"));
        assert_eq!(l.lines[3].translation.as_deref(), Some("看天空"), "the existing row is untouched");
    }

    /// Correction 1. A Simplified original echoed back is stored as
    /// Traditional yet stays flagged unchanged in the preview, and Apply and
    /// save/reload keep the canonical form.
    #[test]
    fn an_echo_stays_flagged_after_canonical_conversion_through_apply_and_reload() {
        let mut w = sheet("[00:01.00]你的声音
[00:02.00]君の声");
        ai_ready_with(&mut w, "gpt-test-model", SourceLanguage::Japanese);
        let t = canned(&w, vec![ok(&[("L001", "你的声音"), ("L002", "你的声音")])]).unwrap();
        deliver(&mut w, t);
        assert!(w.status.contains("1 row(s) flagged"), "{}", w.status);
        let p = w.translation.preview.as_ref().unwrap();
        assert_eq!(p.result.rows, [Some("你的聲音".into()), Some("你的聲音".into())]);
        let flags = &p.result.ai.as_ref().unwrap().flags;
        assert!(flags[0].is_some_and(|f| f.unchanged && f.any()), "{flags:?}");
        assert!(flags[1].is_some_and(|f| !f.any()), "{flags:?}");
        w.apply_translation_preview();
        let (back, bytes) = save_and_reload(w.lyrics.as_ref().unwrap());
        let text = String::from_utf8(bytes).unwrap();
        assert_eq!(text.matches("你的聲音").count(), 2, "both AI rows stored canonical: {text}");
        assert_eq!(back.lines[0].translation.as_deref(), Some("你的聲音"));
    }

    /// 2D. New AI rows are stored as canonical Traditional through preview,
    /// Apply, Save and reload; Simplified is display only; other rows are
    /// never converted.
    #[test]
    fn ai_rows_are_stored_traditional_and_simplified_is_only_a_display() {
        for review_fails in [false, true] {
            let mut w = sheet("[00:01.00]君の声\n[00:02.00]遠くへ\n[00:03.00]愛してる");
            w.lyrics.as_mut().unwrap().lines[2].translation = Some("我爱你".into()); // existing, Simplified
            ai_ready_with(&mut w, "gpt-test-model", SourceLanguage::Japanese);
            w.set_ai_review(true);
            w.agree_ai(true);
            let simplified = [("L001", "你的声音"), ("L002", "向远方")];
            let second = if review_fails { Err(Transport::Network) } else { ok(&simplified) };
            let t = canned(&w, vec![ok(&simplified), second]).unwrap();
            assert_eq!(t.rows, [Some("你的聲音".into()), Some("向遠方".into()), None], "the result itself");
            deliver(&mut w, t);
            assert_eq!(w.translation.preview.as_ref().unwrap().result.rows[0].as_deref(), Some("你的聲音"), "preview rows");
            w.apply_translation_preview();
            w.translation_display = crate::lyrics::zh::ChineseDisplay::Simplified;
            let l = w.lyrics.clone().unwrap();
            assert_eq!(l.lines[0].translation.as_deref(), Some("你的聲音"), "stored, not the display");
            assert_eq!(l.lines[2].translation.as_deref(), Some("我爱你"), "an existing row is not converted");
            assert_eq!(crate::lyrics::zh::display("你的聲音", crate::lyrics::zh::ChineseDisplay::Simplified).as_deref(), Some("你的声音"));
            let (back, bytes) = save_and_reload(&l);
            let text = String::from_utf8(bytes).unwrap();
            assert!(text.contains("你的聲音") && text.contains("向遠方") && !text.contains("声音"), "{text}");
            assert_eq!(back.lines[1].translation.as_deref(), Some("向遠方"));
        }
        // A Basic result is not converted.
        let mut w = window();
        let tx = queue(&mut w);
        tx.send(Message::Done(Ok(Translated { rows: vec![Some("一起".into()), Some("爱".into())],
            credit: "MyMemory machine translation".into(), replaces_existing: false, ai: None }))).unwrap();
        w.poll_translation();
        w.apply_translation_preview();
        assert_eq!(w.lyrics.as_ref().unwrap().lines[1].translation.as_deref(), Some("爱"));
    }

    /// 2E. Oversized sheets are refused before the snapshot, with no worker,
    /// no transport call, and the sheet and preview kept.
    #[test]
    fn oversized_sheets_are_refused_before_copying_and_keep_everything() {
        let _gate = SEND_GATE.lock().unwrap_or_else(|e| e.into_inner());
        let before = sends();
        let line = |text: String| crate::lyrics::LyricLine { text, ..Default::default() };
        let mut too_many: Vec<_> = (0..=ai::MAX_ROWS).map(|_| line(String::new())).collect();
        too_many[0].text = "word".into();
        let wide = vec![line(format!("word{}", " ".repeat(ai::MAX_ROW_BYTES)))];
        let spaces: Vec<_> = (0..=ai::MAX_SOURCE_BYTES / ai::MAX_ROW_BYTES + 1)
            .map(|i| line(if i == 0 { "word".into() } else { " ".repeat(ai::MAX_ROW_BYTES) })).collect();
        for lines in [too_many, wide, spaces] {
            let mut w = window();
            let n = lines.len();
            w.lyrics = Some(Lyrics { lines, source: LyricSource::Embedded, offset_ms: 0, title: None, artist: None,
                                     translation_credit: None });
            ai_ready(&mut w);
            let tx = queue(&mut w);
            let mut rows = vec![None; n];
            rows[0] = Some("一個字".to_string());
            tx.send(Message::Done(Ok(Translated { rows, credit: "Synthetic".into(), replaces_existing: false, ai: None }))).unwrap();
            w.poll_translation();
            assert!(w.translation.preview.is_some());
            let before_sheet = w.lyrics.as_ref().unwrap().to_lrc();
            let serial = w.translation.serial;
            w.start_translation(&track(), true);
            assert!(!w.translation.is_busy(), "{n} rows: started");
            assert_eq!(w.translation.serial, serial);
            assert!(w.status.contains("limited to") || w.status.contains("exceeds"), "{}", w.status);
            assert!(w.translation.preview.is_some(), "the preview is kept");
            assert_eq!(w.lyrics.as_ref().unwrap().to_lrc(), before_sheet);
        }
        assert_eq!(sends(), before);
    }

    #[test]
    fn ai_rows_fill_gaps_keep_mixed_provenance_and_save_without_the_key() {
        let mut w = window();
        {
            let l = w.lyrics.as_mut().unwrap();
            l.lines[0].translation = Some("一同".into());
            l.translation_credit = Some("Manually edited translation".into());
        }
        ai_ready(&mut w);
        let mut result = ai_result(&w, ai::Review::Failed("synthetic review failure".into()));
        result.rows[0] = None; // fill-missing: the translated row was not asked for
        deliver(&mut w, result);
        assert!(w.status.contains("Unreviewed AI draft"), "{}", w.status);
        w.apply_translation_preview();
        let l = w.lyrics.as_ref().unwrap();
        assert_eq!(l.lines[0].translation.as_deref(), Some("一同"), "existing row kept");
        assert_eq!(l.lines[1].translation.as_deref(), Some("愛"));
        let credit = l.translation_credit.clone().unwrap();
        assert!(credit.starts_with("AI-assisted translation (OpenAI gpt-test-model") && credit.contains("unreviewed draft"), "{credit}");
        assert!(credit.contains("previous: Manually edited translation"), "{credit}");
        let (back, bytes) = save_and_reload(l);
        assert!(back.translation_credit.unwrap().contains("AI-assisted"));
        let text = String::from_utf8_lossy(&bytes);
        assert!(!text.contains(SENTINEL_KEY) && !text.contains("SENTINEL"), "the sidecar holds the key");
        assert!(!format!("{:?}", w.translation.ai.key).contains("SENTINEL"));
    }

    #[test]
    fn ai_panels_build_headless_without_adopting_anything() {
        let ctx = egui::Context::default();
        let mut w = window();
        w.translation.open = true;
        for stage in 0..4 {
            match stage {
                0 => w.set_provider(Provider::Ai),
                1 => ai_ready(&mut w),
                2 => {
                    { let r = ai_result(&w, ai::Review::Failed("synthetic".into())); deliver(&mut w, r); }
                    assert!(w.translation.preview.is_some());
                }
                _ => w.set_replace(true),
            }
            let output = ctx.run(egui::RawInput {
                screen_rect: Some(egui::Rect::from_min_size(egui::Pos2::ZERO, egui::vec2(520.0, 640.0))),
                ..Default::default()
            }, |ctx| {
                egui::CentralPanel::default().show(ctx, |ui| w.translation_panel(ui, &track()));
            });
            assert!(!output.shapes.is_empty());
            assert!(!w.translation_dirty);
        }
    }
}
