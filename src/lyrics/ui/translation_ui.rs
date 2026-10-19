//! Optional translation work, separate from original-lyrics lookup/readings.
//! One worker owns its slot until its completion is consumed, even on Cancel.

use super::*;
use crate::lyrics::ai;
use crate::lyrics::local_ai;
use crate::lyrics::translation::{self, SourceLanguage};
use std::sync::mpsc::{self, TryRecvError};
use std::sync::{
    Arc,
    atomic::{AtomicBool, Ordering},
};
#[path = "translation_workbench.rs"]
mod translation_workbench;

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
    local: LocalSettings,
    workbench: translation_workbench::Workbench,
}

/// Which online translation an explicit request uses.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub(super) enum Provider {
    #[default]
    Basic,
    Ai,
    /// PROTOTYPE: the downloadable local pack, on this computer.
    Local,
}

/// PROTOTYPE local-pack settings, for this session only.
struct LocalSettings {
    /// The model chosen; at start the first installed one, else the first listed.
    pack: &'static local_ai::Pack,
    source: Option<local_ai::Source>,
    target: local_ai::Target,
    fluent: bool,
    gpu: bool,
    low_ram: bool,
    preferences: local_ai::Preferences,
    root: std::path::PathBuf,
    /// Whether the pack is installed, as last checked (after install/remove).
    installed: Option<bool>,
    /// Bytes already downloaded and kept for resuming, as last checked.
    partial: u64,
    /// The download's terms are on screen, waiting for "Download now".
    consent_open: bool,
    install: Option<Install>,
    /// This computer and the advice for it, read when the advice is first
    /// opened (and again on request), never at start.
    advice: Option<(local_ai::advice::Machine, local_ai::advice::Advice)>,
}

/// The model the user last chose, by pack ID, in the local-model folder.
const CHOSEN: &str = "chosen-model.txt";

fn chosen_pack(root: &std::path::Path) -> Option<&'static local_ai::Pack> {
    let id = std::fs::read_to_string(root.join(CHOSEN)).ok()?;
    local_ai::PACKS.iter().copied().find(|p| p.id == id.trim())
}

fn remember_choice(root: &std::path::Path, pack: &local_ai::Pack) {
    if std::fs::create_dir_all(root).is_ok() {
        let _ = std::fs::write(root.join(CHOSEN), pack.id);
    }
}

impl Default for LocalSettings {
    fn default() -> Self {
        let root = local_ai::default_root();
        // The model chosen last time; else the first installed one; else the first listed.
        let pack = chosen_pack(&root)
            .or_else(|| local_ai::PACKS.iter().copied().find(|p| local_ai::installed(&root, p)))
            .unwrap_or(local_ai::PACKS[0]);
        #[cfg(not(test))]
        let preferences = local_ai::Preferences::load(&root);
        #[cfg(test)]
        let preferences = local_ai::Preferences::default();
        Self { pack, source: None, target: local_ai::Target::TraditionalChinese, fluent: false, gpu: true, low_ram: false, preferences,
               root, installed: None, partial: 0, consent_open: false, install: None, advice: None }
    }
}

impl super::LyricsWindow {
    /// "Which model for this computer?": the three kinds of advice, each with
    /// its route and costs, and a button that only selects it (no download,
    /// no run). Returns a pick the user chose, with whether it uses the card.
    fn advice_ui(&mut self, ui: &mut egui::Ui) -> Option<(&'static local_ai::Pack, local_ai::Device)> {
        use local_ai::advice;
        let mut chosen = None;
        egui::CollapsingHeader::new("Which model for this computer?").id_salt("local_advice").show(ui, |ui| {
            let l = &mut self.translation.local;
            if l.advice.is_none() {
                let m = advice::inventory();
                let a = advice::advise(local_ai::PACKS, &m);
                l.advice = Some((m, a));
            }
            let Some((m, a)) = l.advice.clone() else { return };
            let gb = local_ai::memory::gib;
            ui.label(format!(
                "This computer: {} RAM ({} free now){}. Planned within {} of graphics memory and {} of RAM, keeping \
                 {} of graphics memory and {} of RAM for Windows, the desktop and Moosik.",
                gb(m.ram_total), gb(m.ram_free),
                m.gpu.as_ref().map_or(", no graphics card's dedicated memory reported".to_string(),
                    |g| format!(", {} with {} of its own memory", g.name, gb(g.dedicated))),
                gb(a.budget.vram), gb(a.budget.ram), gb(advice::VRAM_RESERVE), gb(advice::RAM_RESERVE),
            ));
            for (what, tip, pick) in [
                ("Quality first", "The best-judged model this computer can run at all, on its fastest route that fits.", a.quality),
                ("Balanced", "The best-judged model that fits with room to spare and answers a 10-row sheet within \
                    about 15 s.", a.balanced),
                ("Largest that fits", "The biggest model this computer can hold. Size is not quality.", a.largest),
            ] {
                ui.add_space(4.0);
                ui.horizontal_wrapped(|ui| {
                    ui.label(egui::RichText::new(what).strong()).on_hover_text(tip);
                    if let Some(p) = pick
                        && ui.button("Use this").on_hover_text("Selects this model and route. Nothing is downloaded or run.").clicked()
                    {
                        chosen = Some((p.pack, p.device()));
                    }
                });
                match pick {
                    Some(p) => {
                        let c = p.cost;
                        ui.label(format!(
                            "{} — {}. About {:.0} s per 10-row sheet; {} RAM{}. {}.",
                            p.pack.label, p.path.label(), c.secs_per_sheet, gb(c.ram),
                            if c.vram > 0 { format!(", {} graphics memory", gb(c.vram)) } else { String::new() },
                            if c.measured { "Measured on one computer (RTX 2060 SUPER 8 GB)" } else { "Estimated, not measured" },
                        ));
                        let fits = local_ai::memory::room(p.need(), m.headroom());
                        let text = local_ai::memory::startup_text(p.need(), m.headroom());
                        let colour = if fits == local_ai::memory::Room::Fits { ui.visuals().text_color() } else { ui.visuals().warn_fg_color };
                        ui.label(egui::RichText::new(text).color(colour));
                    }
                    None => {
                        ui.label("Nothing known fits this computer.");
                    }
                }
            }
            ui.label(egui::RichText::new(local_ai::QUALITY_BASIS).small());
            if ui.small_button("Check this computer again").clicked() {
                self.translation.local.advice = None;
            }
        });
        chosen
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum InstallKind {
    Download,
    /// Using files already on this computer.
    Import,
}

struct Install {
    kind: InstallKind,
    cancel: Arc<AtomicBool>,
    rx: Receiver<InstallMessage>,
    progress: Option<local_ai::Progress>,
    /// (when, bytes received over the network) over the last half minute.
    samples: std::collections::VecDeque<(std::time::Instant, u64)>,
    /// When bytes last arrived.
    last_bytes: std::time::Instant,
}

enum InstallMessage {
    Progress(local_ai::Progress),
    /// What to tell the user.
    Done(Result<String, String>),
}

/// How far back the download speed is measured.
const SPEED_WINDOW: std::time::Duration = std::time::Duration::from_secs(30);

impl Install {
    fn new(kind: InstallKind, cancel: Arc<AtomicBool>, rx: Receiver<InstallMessage>) -> Self {
        let now = std::time::Instant::now();
        Self { kind, cancel, rx, progress: None, samples: Default::default(), last_bytes: now }
    }

    fn update(&mut self, p: local_ai::Progress) {
        let now = std::time::Instant::now();
        if self.progress.is_none_or(|o| o.fetched != p.fetched) {
            self.last_bytes = now;
        }
        self.samples.push_back((now, p.fetched));
        while self.samples.front().is_some_and(|(t, _)| now - *t > SPEED_WINDOW) {
            self.samples.pop_front();
        }
        self.progress = Some(p);
    }

    /// Network bytes per second over the last half minute, once at least
    /// five seconds of it are known. Bytes already on disk from an earlier
    /// attempt are not in it.
    fn speed(&self) -> Option<f64> {
        let ((t0, b0), (t1, b1)) = (self.samples.front()?, self.samples.back()?);
        let dt = (*t1 - *t0).as_secs_f64();
        (dt >= 5.0).then(|| b1.saturating_sub(*b0) as f64 / dt)
    }

    fn describe(&self, cancelling: bool) -> String {
        let gb = |b: u64| b as f64 / 1e9;
        let what = match self.progress {
            None => match self.kind {
                InstallKind::Download => "Starting the download…".to_string(),
                InstallKind::Import => "Reading the chosen files…".to_string(),
            },
            Some(p) => match p.phase {
                local_ai::Phase::Download => format!("Downloading {}: {:.2} of {:.2} GB", p.file, gb(p.done), gb(p.total)),
                local_ai::Phase::Check => format!("Checking {} against its pinned SHA-256…", p.file),
                local_ai::Phase::Copy => format!("Copying {}: {:.2} of {:.2} GB", p.file, gb(p.done), gb(p.total)),
                local_ai::Phase::Unpack => "Setting up the runtime…".to_string(),
            },
        };
        if cancelling { format!("Stopping. {what}") } else { what }
    }

    /// Speed, a cautious time left, and reconnects, while downloading.
    fn rate_line(&self) -> Option<String> {
        let p = self.progress?;
        if p.phase != local_ai::Phase::Download {
            return None;
        }
        let idle = self.last_bytes.elapsed().as_secs();
        let reconnects = match p.reconnects {
            0 => String::new(),
            1 => " · reconnected once".into(),
            n => format!(" · reconnected {n} times"),
        };
        Some(if idle >= 10 {
            format!("No data for {idle} s: waiting (it reconnects by itself), or cancel and resume later.{reconnects}")
        } else {
            match self.speed() {
                Some(s) if s > 0.0 => format!("{:.2} MB/s · about {} left{reconnects}", s / 1e6,
                    eta(p.total.saturating_sub(p.done) as f64 / s)),
                _ => format!("Measuring the speed…{reconnects}"),
            }
        })
    }
}

/// A time left, rounded up, never more precise than it can be.
fn eta(secs: f64) -> String {
    let m = (secs / 60.0).ceil() as u64;
    match m {
        0 | 1 => "a minute".into(),
        m if m < 60 => format!("{m} min"),
        m => format!("{} h {} min", m / 60, m % 60),
    }
}

struct LocalResult {
    identity: local_ai::Identity,
    flags: Vec<Option<local_ai::RowFlags>>,
    /// Corrective retries the request needed (each accepted, or it failed).
    retries: usize,
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

#[derive(Clone)]
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
    /// The identity a local request was made with; `None` otherwise.
    local: Option<local_ai::Identity>,
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
    local: Option<Box<LocalResult>>,
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
        self.workbench = Default::default();
        self.local.preferences.feedback.clear();
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

    /// The local identity as the settings stand: `None` until a source is chosen.
    fn local_identity(&self) -> Option<local_ai::Identity> {
        let l = &self.translation.local;
        let style = if l.fluent { ai::Style::Fluent } else { ai::Style::Faithful };
        let device = if !l.gpu { local_ai::Device::Cpu }
            else if l.low_ram && l.pack.id == local_ai::HYMT2_30B.id { local_ai::Device::GpuLowRam }
            else { local_ai::Device::Gpu };
        let mut identity = local_ai::Identity::new(l.pack, l.source?, l.target, style, device);
        identity.preferences = l.preferences.clone();
        Some(identity)
    }

    fn local_matches(&self, captured: Option<&local_ai::Identity>) -> bool {
        captured.is_none_or(|c| self.translation.provider == Provider::Local && self.local_identity().as_ref() == Some(c))
    }

    fn result_matches(&self, r: &Translated) -> bool {
        self.ai_matches(r.ai.as_ref().map(|a| &a.config)) && self.local_matches(r.local.as_ref().map(|l| &l.identity))
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
        self.translation.workbench = Default::default();
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
        self.keep_local_warning();
    }

    /// A local runtime process or download that could not be confirmed
    /// finished outranks every other status: no cancellation, track change or
    /// stale completion may hide it.
    fn keep_local_warning(&mut self) {
        if let Some(w) = local_ai::outstanding(&self.translation.local.root) {
            self.status = w;
        }
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
        let local_mode = machine && self.translation.provider == Provider::Local;
        if machine && !ai_mode && !local_mode && (!self.translation.consent || source.is_none()) {
            return;
        }
        let local_ready = if local_mode {
            let Some(identity) = self.local_identity() else { return };
            if identity.route == local_ai::Route::Translate && self.translation.local.installed != Some(true) {
                return;
            }
            Some(identity)
        } else {
            None
        };
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
        let local_job = if let Some(identity) = local_ready {
            let l = &snapshot.lyrics;
            let texts: Vec<&str> = l.lines.iter().map(|x| x.text.as_str()).collect();
            let revision = self.local_revision();
            let wanted: Vec<bool> = l.lines.iter()
                .map(|x| revision.is_some() || self.translation.replace || x.translation.as_ref().is_none_or(|s| s.trim().is_empty()))
                .collect();
            let job = local_ai::Job::new(identity, track.title, track.artist, &texts, &wanted)
                .and_then(|job| if let Some(r) = revision { job.revise(r) } else { Ok(job) });
            match job {
                Ok(job) => Some(job),
                Err(e) => {
                    self.status = e;
                    return;
                }
            }
        } else {
            None
        };
        let local_identity = local_job.as_ref().map(|j| j.identity.clone());
        let local_revision = local_job.as_ref().is_some_and(local_ai::Job::is_revision);
        let local_root = self.translation.local.root.clone();
        let local_pack = self.translation.local.pack;
        let ai_config = ai_job.as_ref().map(|(job, _)| job.config().clone());
        if local_mode {
            self.archive_local_preview();
        } else {
            self.translation.workbench = Default::default();
        }
        self.translation.clear_preview();
        self.translation.serial = self.translation.serial.wrapping_add(1);
        let serial = self.translation.serial;
        let cancel = Arc::new(AtomicBool::new(false));
        let stop = cancel.clone();
        let (tx, rx) = mpsc::channel();
        let replace = self.translation.replace || local_revision;
        // Basic and supplied lookups work on a copy; the AI job already holds
        // what it sends and needs none.
        let mut input = if ai_job.is_none() && local_job.is_none() { Some(snapshot.lyrics.clone()) } else { None };
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
                } else if let Some(job) = local_job {
                    let o = local_ai::run(&local_root, local_pack, &job, serial, &stop, |done, total| {
                        let _ = tx.send(Message::Progress(done, total));
                    })?;
                    let credit = job.identity.credit(local_pack);
                    Ok(Translated { rows: o.rows, credit, replaces_existing: replace, ai: None,
                                    local: Some(Box::new(LocalResult { identity: job.identity, flags: o.flags, retries: o.retries.len() })) })
                } else if machine {
                    let Some(input) = input else { return Err("Internal error: no input; nothing was sent.".into()) };
                    let source = source.ok_or_else(|| "Choose the source language.".to_string())?;
                    let rows = translation::translate(&input.lines, source, &stop, |done, total| {
                        let _ = tx.send(Message::Progress(done, total));
                    })?;
                    Ok(Translated { rows, credit: format!("{} ({} to Chinese)", translation::NAME, source.label()),
                                    replaces_existing: replace, ai: None, local: None })
                } else {
                    // Lookup is independent of LRCLIB's successful original
                    // lyric hit. Never install a different original sheet.
                    let Some(input) = input else { return Err("Internal error: no input; nothing was sent.".into()) };
                    let hit = super::super::netease::find_translation(&artist, &title, duration, &stop)?
                        .ok_or_else(|| "No matching translation found. Try online translation or paste/edit your own.".to_string())?;
                    if stop.load(Ordering::Acquire) { return Err("Cancelled.".into()); }
                    let rows = translation::supplement(&input, &hit)?;
                    Ok(Translated { rows, credit: format!("Supplement supplied by {}", hit.source),
                                    replaces_existing: false, ai: None, local: None })
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
                    local: local_identity,
                });
                self.status = if ai_mode {
                    "Translating with OpenAI…"
                } else if local_mode {
                    "Translating on this computer…"
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
                        && self.local_matches(task.local.as_ref())
                        && !self.editing
                        && !self.romaji_editing
                        && !self.translation.is_editing();
                    if task.cancel.load(Ordering::Acquire) {
                        if current {
                            self.status = "Translation cancelled; nothing was applied.".into();
                        }
                        self.keep_local_warning();
                        return;
                    }
                    if !current {
                        // Do not replace the new track/editor's status with a
                        // stale request's result or error (a process-safety
                        // warning is not a result: it stays).
                        self.keep_local_warning();
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
                            let local_flagged = result.local.as_ref().map(|l| l.flags.iter().flatten().filter(|f| f.any()).count());
                            let retried = result.local.as_ref().map_or(0, |l| l.retries);
                            let retried = if retried > 0 {
                                format!(" (the model's first answer for {retried} part(s) was refused and asked again once)")
                            } else {
                                String::new()
                            };
                            self.status = match result.ai.as_ref().map(|a| (&a.review, a.flags.iter().flatten().filter(|f| f.any()).count())) {
                                None if local_flagged.is_some_and(|n| n > 0) => format!(
                                    "Local translation ready to preview{retried}; {} row(s) flagged for checking — Apply does not save it to disk.",
                                    local_flagged.unwrap_or(0)),
                                None if result.local.is_some() => format!(
                                    "Local translation ready to preview{retried} — Apply does not save it to disk."),
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
                            self.prepare_workbench_rows();
                        }
                        Ok(_) => {
                            self.status =
                                "No new translation was returned; the existing lyrics were kept."
                                    .into()
                        }
                        Err(e) => self.status = e,
                    }
                    self.keep_local_warning();
                    return;
                }
            }
        }
    }

    fn apply_translation_preview(&mut self) {
        if self.translation.is_busy() { return; }
        let Some(preview) = self.translation.preview.take() else {
            return;
        };
        if !preview.snapshot.matches(self) || self.editing || self.romaji_editing {
            self.status =
                "The lyrics changed. Find or translate again; nothing was applied.".into();
            return;
        }
        if !self.result_matches(&preview.result) {
            self.status = "The AI settings changed since this preview was made; nothing was applied.".into();
            return;
        }
        let before = preview.snapshot.clone();
        let selected = self.translation.workbench.adopt.clone();
        let local = preview.result.local.is_some();
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
                .zip(&preview.result.rows).enumerate()
                .any(|(i, (line, new))| {
                    (new.is_none() || (local && selected.get(i) == Some(&false)))
                        && line
                            .translation
                            .as_ref()
                            .is_some_and(|s| !s.trim().is_empty())
                });
        for (i, (line, translated)) in lyrics.lines.iter_mut().zip(preview.result.rows).enumerate() {
            if local && selected.get(i) == Some(&false) { continue; }
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
            if local && let Some(after) = self.translation_snapshot() {
                self.translation.workbench.undo = Some((before, after));
            }
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
        self.poll_local_install();
        if self.translation.preview.as_ref().is_some_and(|p| !self.preview_compatible(p)) {
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
                ui.label("Translation");
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
                    ui.radio_value(&mut provider, Provider::Local, "On this computer (prototype, local pack)");
                });
            });
            self.set_source(source);
            self.set_replace(replace);
            self.set_provider(provider);
            if self.translation.provider == Provider::Ai {
                machine = self.ai_panel(ui, idle);
            } else if self.translation.provider == Provider::Local {
                machine = self.local_panel(ui, idle);
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
                                 // The local route reports the parts the sheet was sent in.
                                 else if task.local.is_some() { format!("part {} of {total} on this computer", (done + 1).min(total)) }
                                 else { format!("{done}/{total} unique lines") });
                    }
                    cancel = ui.button("Cancel").clicked() || cancel;
                });
            }
            if self.translation.replace {
                ui.label(RichText::new("Existing source/user translations will be replaced only after Apply. \
                    Save keeps the previous sidecar as a backup.").small().color(pal::amber(ui.visuals().dark_mode)));
            }
            self.local_history_ui(ui, idle);
            let preview_current = self.translation.preview.as_ref().is_some_and(|p| self.result_matches(&p.result));
            if let Some(preview) = &mut self.translation.preview {
                ui.separator();
                ui.label(RichText::new(&preview.result.credit).small());
                if let Some(l) = &preview.result.local {
                    ui.label(format!("Draft seed {} · {}", l.identity.preferences.seed, l.identity.preferences.variation.label()));
                    if !preview_current { ui.label("Instructions or seed changed. Generate a revision, or restore a version's settings before Apply."); }
                    if !preview_current && ui.add_enabled(idle, egui::Button::new("Restore this draft's settings")).clicked() {
                        self.translation.local.preferences = l.identity.preferences.clone();
                    }
                }
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
                    for (i, (original, translated)) in preview.snapshot.lyrics.lines.iter().zip(&mut preview.result.rows).enumerate() {
                        if let Some(text) = translated {
                            if preview.result.local.is_some() {
                                let w = &mut self.translation.workbench;
                                ui.add_enabled_ui(idle, |ui| {
                                    ui.horizontal_wrapped(|ui| {
                                        if let Some(v) = w.locked.get_mut(i) { ui.checkbox(v, "Lock"); }
                                        if let Some(v) = w.selected.get_mut(i) { ui.checkbox(v, "Revise"); }
                                        if let Some(v) = w.adopt.get_mut(i) { ui.checkbox(v, "Adopt"); }
                                    });
                                });
                                if let Some(previous) = w.history.last().filter(|p| p.snapshot.rev == preview.snapshot.rev)
                                    .and_then(|p| p.result.rows.get(i)).and_then(Option::as_ref)
                                {
                                    ui.columns(2, |columns| {
                                        columns[0].label(RichText::new(format!("Previous: {previous}")).small().weak());
                                        columns[1].label(RichText::new(format!("Current: {text}")).small());
                                    });
                                    if previous != text && ui.add_enabled(idle, egui::Button::new("Use previous for this row")).clicked() {
                                        text.clone_from(previous);
                                        if let Some(l) = preview.result.local.as_mut() && let Some(flag) = l.flags.get_mut(i) { *flag = None; }
                                        if !preview.result.credit.contains("selected draft rows") {
                                            preview.result.credit.push_str("; includes selected draft rows");
                                        }
                                    }
                                }
                            }
                            ui.label(RichText::new(&original.text).small());
                            let family = translation_family(ui, self.translation_display);
                            ui.label(RichText::new(preview.display.get(i, text, self.translation_display)).family(family));
                            if let Some(f) = preview.result.ai.as_ref().and_then(|a| a.flags.get(i).copied().flatten())
                                && f.any()
                            {
                                ui.label(RichText::new(format!("⚠ Check this row: {}", f.describe())).small().color(amber));
                            }
                            if let Some(f) = preview.result.local.as_ref().and_then(|l| l.flags.get(i).copied().flatten()) {
                                if f.any() {
                                    ui.label(RichText::new(format!("⚠ Check this row: {}", f.describe())).small().color(amber));
                                } else if f.foreign_script {
                                    // A note, not a warning: often a name kept as written.
                                    ui.label(RichText::new(format!("Note: {}", f.describe())).small().weak());
                                }
                            }
                        }
                    }
                });
                ui.horizontal(|ui| {
                    apply = ui.add_enabled(!editing_sheet && idle && preview_current, egui::Button::new("Apply translation")).clicked();
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

impl LyricsWindow {
    /// Install or import progress and completion; refreshes the installed state.
    fn poll_local_install(&mut self) {
        let l = &mut self.translation.local;
        if l.installed.is_none() {
            l.installed = Some(local_ai::installed(&l.root, l.pack));
            l.partial = local_ai::partial_bytes(&l.root, l.pack);
        }
        let Some(inst) = l.install.as_mut() else { return };
        loop {
            match inst.rx.try_recv() {
                Ok(InstallMessage::Progress(p)) => inst.update(p),
                Ok(InstallMessage::Done(r)) => {
                    // A cancelled download is a download, not a translation.
                    let r = match r {
                        Err(_) if inst.cancel.load(Ordering::Acquire) => Err(match inst.kind {
                            InstallKind::Download => "Download cancelled; what was downloaded is kept and resumes next time.",
                            InstallKind::Import => "Copying cancelled; nothing was installed.",
                        }.to_string()),
                        r => r,
                    };
                    self.status = match (r, local_ai::outstanding(&l.root)) {
                        (Ok(said), _) => said,
                        // The installing thread returned; its connection may not have.
                        (Err(e), Some(w)) => format!("{e} {w}"),
                        (Err(e), None) => e,
                    };
                    l.install = None;
                    l.installed = None;
                    return;
                }
                Err(TryRecvError::Empty) => return,
                Err(TryRecvError::Disconnected) => {
                    self.status = "The pack download stopped unexpectedly; install again to resume.".into();
                    l.install = None;
                    l.installed = None;
                    return;
                }
            }
        }
    }

    /// PROTOTYPE local-pack controls. Returns whether Translate was pressed.
    /// Drawing never downloads or starts anything.
    fn local_panel(&mut self, ui: &mut egui::Ui, idle: bool) -> bool {
        // In a column as wide as what can be seen, so its text wraps there
        // even when a wide row elsewhere in the panel made the panel wider.
        let w = visible_width(ui);
        let h = ui.available_height().max(1.0);
        ui.allocate_ui_with_layout(egui::vec2(w, h), egui::Layout::top_down(egui::Align::Min), |ui| self.local_panel_body(ui, idle)).inner
    }

    fn local_panel_body(&mut self, ui: &mut egui::Ui, idle: bool) -> bool {
        let gb = |b: u64| b as f64 / 1e9;
        let supported = local_ai::platform_supported();
        let mut start = false;
        let mut download = false;
        let mut remove = false;
        let mut import: Option<std::path::PathBuf> = None;
        let installing = self.translation.local.install.is_some();
        let (mut pack, mut source, mut target, mut fluent, mut gpu, mut low_ram) = {
            let l = &self.translation.local;
            (l.pack, l.source, l.target, l.fluent, l.gpu, l.low_ram)
        };
        ui.add_enabled_ui(idle && !installing, |ui| {
            ui.horizontal_wrapped(|ui| {
                ui.label("Model:");
                egui::ComboBox::from_id_salt("local_pack")
                    .selected_text(pack.label)
                    .width((ui.available_width() - 8.0).clamp(120.0, 420.0))
                    .truncate()
                    .show_ui(ui, |ui| {
                        for p in local_ai::PACKS {
                            let have = local_ai::installed(&self.translation.local.root, p);
                            let text = format!("{}{}", p.label, if have { " — installed" } else { "" });
                            if ui.selectable_label(std::ptr::eq(pack, *p), text).clicked() {
                                pack = p;
                            }
                        }
                    });
            });
            ui.label(pack.summary);
            if let Some((p, device)) = self.advice_ui(ui) {
                pack = p;
                gpu = device != local_ai::Device::Cpu;
                low_ram = device == local_ai::Device::GpuLowRam;
            }
            ui.horizontal_wrapped(|ui| {
                ui.label("From:");
                egui::ComboBox::from_id_salt("local_source")
                    .selected_text(source.map(local_ai::Source::label).unwrap_or("Choose…"))
                    .show_ui(ui, |ui| {
                        for s in local_ai::Source::ALL {
                            ui.selectable_value(&mut source, Some(s), s.label());
                        }
                    });
                ui.label("Into:");
                egui::ComboBox::from_id_salt("local_target").selected_text(target.label()).show_ui(ui, |ui| {
                    for t in local_ai::Target::ALL {
                        ui.selectable_value(&mut target, t, t.label());
                    }
                });
            });
            ui.horizontal_wrapped(|ui| {
                ui.radio_value(&mut fluent, false, "Faithful")
                    .on_hover_text("Asks for an accurate translation of each line. The default.");
                ui.radio_value(&mut fluent, true, "More fluent (meaning kept)")
                    .on_hover_text("Also asks for natural lyric phrasing, with the meaning kept whole: \
                                    nothing added or dropped, no rhyme at the meaning's cost. \
                                    It is not a singable adaptation.");
                ui.checkbox(&mut gpu, "Use the graphics card (Vulkan) when available");
            });
            if gpu && pack.id == local_ai::HYMT2_30B.id {
                ui.checkbox(&mut low_ram, "Use less RAM (experimental; needs a 7+ GiB graphics card)");
                if low_ram {
                    let need = local_ai::memory_need(pack, local_ai::Device::GpuLowRam).expect("30B route");
                    ui.label("Keeps the same model and context. Uses more graphics memory and commit headroom, and may run slower. No extra download.");
                    ui.label(local_ai::memory::startup_text(need, local_ai::memory::read()));
                }
            }
        });
        {
            let l = &mut self.translation.local;
            if !std::ptr::eq(l.pack, pack) {
                // Another model: its own install state, and no agreement carried over.
                l.installed = None;
                l.consent_open = false;
                remember_choice(&l.root, pack);
            }
            (l.pack, l.source, l.target, l.fluent, l.gpu, l.low_ram) = (pack, source, target, fluent, gpu, low_ram);
        }
        let installed = self.translation.local.installed == Some(true);
        let route = source.map(|s| local_ai::route(s, target));
        match route {
            Some(local_ai::Route::KeepOriginal) => {
                ui.label("Same language: the original rows are used as they are; no model runs.");
            }
            Some(local_ai::Route::ConvertCharacters) => {
                ui.label("Chinese to Chinese: the original rows are converted to Traditional characters (OpenCC s2tw); no model runs.");
            }
            _ => {}
        }
        let needs_pack = route.is_none_or(|r| r == local_ai::Route::Translate);
        // Work that could not be confirmed finished blocks every pack action
        // and model run here, and stays on screen until it has ended.
        let held = local_ai::outstanding(&self.translation.local.root);
        if let Some(w) = &held {
            ui.colored_label(egui::Color32::from_rgb(0xd0, 0x60, 0x20), w);
        }
        let idle = idle && held.is_none();
        let partial = self.translation.local.partial;
        let (download_b, peak_b, installed_b) = local_ai::disk_needs(pack);
        if needs_pack && !supported {
            ui.label("Translation on this computer needs 64-bit Windows: the local pack's runtime is a Windows x64 build, \
                so nothing is offered for download here. Same-language routes above still work.");
        } else if let Some(inst) = &self.translation.local.install {
            let cancelling = inst.cancel.load(Ordering::Acquire);
            ui.label(RichText::new(inst.describe(cancelling)).strong());
            let fraction = inst.progress.map_or(0.0, |p| if p.total > 0 { p.done as f32 / p.total as f32 } else { 0.0 });
            ui.add(egui::ProgressBar::new(fraction).show_percentage());
            if let Some(line) = inst.rate_line() {
                ui.label(line);
            }
            let label = match inst.kind {
                InstallKind::Download => "Cancel download",
                InstallKind::Import => "Cancel copying",
            };
            if ui.add_enabled(!cancelling, egui::Button::new(label)).clicked() {
                inst.cancel.store(true, Ordering::Release);
            }
        } else if needs_pack && installed {
            ui.label(format!("Installed: {:.2} GB in {}.", gb(installed_b), local_ai::pack_dir(&self.translation.local.root, pack).display()));
        } else if needs_pack && !self.translation.local.consent_open {
            let resume = if partial > 0 { format!(" {:.2} GB is already here and will be resumed.", gb(partial)) } else { String::new() };
            ui.label(format!("Not installed. It needs a one-time download of {:.2} GB, or files you already have.{resume}", gb(download_b)));
            ui.horizontal_wrapped(|ui| {
                let label = if partial > 0 { "Resume download…" } else { "Download…" };
                if ui.add_enabled(idle, egui::Button::new(label)).clicked() {
                    self.translation.local.consent_open = true;
                }
                ui.add_enabled_ui(idle, |ui| {
                    ui.menu_button("Use files I already have…", |ui| {
                        if ui.button("A pack folder (from another Moosik)…").clicked() {
                            ui.close_menu();
                            import = rfd::FileDialog::new().set_title("Choose an installed pack folder").pick_folder();
                        }
                        if ui.button("A model file (.gguf)…").clicked() {
                            ui.close_menu();
                            import = rfd::FileDialog::new().set_title("Choose the model file").add_filter("GGUF model", &["gguf"]).pick_file();
                        }
                    });
                });
            });
        } else if needs_pack {
            // The terms, before anything is fetched. Opening the panel or this
            // card downloads nothing; only "Download now" does.
            let free = local_ai::free_space(&self.translation.local.root);
            let room = free.is_none_or(|f| f + partial >= peak_b + 1_000_000_000);
            egui::Frame::group(ui.style()).show(ui, |ui| {
                ui.label(RichText::new(format!("Download {}?", pack.label)).heading());
                egui::ScrollArea::vertical().max_height(ui.ctx().screen_rect().height() * 0.45).show(ui, |ui| {
                    let label_w = 120.0;
                    egui::Grid::new("local_terms").num_columns(2).spacing([12.0, 8.0]).striped(true)
                        .min_col_width(60.0).max_col_width((ui.available_width() - label_w - 12.0).max(120.0)).show(ui, |ui| {
                        let row = |ui: &mut egui::Ui, what: &str, text: String| {
                            ui.label(RichText::new(what).strong());
                            ui.add(egui::Label::new(text).wrap());
                            ui.end_row();
                        };
                        row(ui, "What it is", pack.summary.to_string());
                        row(ui, "Translates", format!("{} into {}.", local_ai::Source::ALL.map(local_ai::Source::label).join(", "),
                            local_ai::Target::ALL.map(local_ai::Target::label).join(" or ")));
                        row(ui, "Download", format!("{:.2} GB, once: the model from Hugging Face and the llama.cpp runtime from \
                            GitHub (skipped when another installed pack already has it). Every file is checked against a pinned size \
                            and SHA-256 before it is used.", gb(download_b)));
                        row(ui, "Disk", format!("Up to {:.2} GB while installing, {:.2} GB once installed, in {}.{}", gb(peak_b),
                            gb(installed_b), self.translation.local.root.display(),
                            free.map(|f| format!(" {:.1} GB is free there now.", gb(f))).unwrap_or_default()));
                        row(ui, "Memory", pack.resources.to_string());
                        let gpu_need = if low_ram && pack.id == local_ai::HYMT2_30B.id { local_ai::memory_need(pack, local_ai::Device::GpuLowRam).expect("30B route") } else { pack.need_gpu };
                        row(ui, "Before each run", local_ai::memory::admission_text(gpu_need, pack.need_cpu));
                        row(ui, "Privacy", "Once installed it runs on this computer with no network access, no key and no \
                            account. Nothing is sent, and nothing falls back to an online service.".to_string());
                        row(ui, "Limitations", pack.limitations.to_string());
                    });
                    egui::CollapsingHeader::new("Licences and sources").show(ui, |ui| {
                        ui.label(RichText::new(pack.notices).small());
                    });
                });
                if !room {
                    ui.colored_label(egui::Color32::from_rgb(0xd0, 0x60, 0x20),
                        "There is not enough free space for the download and the install (plus 1 GB to spare).");
                }
                ui.horizontal_wrapped(|ui| {
                    let label = if partial > 0 {
                        format!("Resume download ({:.2} GB left)", gb(download_b.saturating_sub(partial)))
                    } else {
                        format!("Download now ({:.2} GB)", gb(download_b))
                    };
                    download = ui.add_enabled(idle && room, egui::Button::new(label)).clicked();
                    if ui.button("Not now").clicked() {
                        self.translation.local.consent_open = false;
                    }
                });
            });
        }
        let can = idle && !installing && source.is_some() && (installed || !needs_pack) && (supported || !needs_pack);
        start |= self.local_creative_settings(ui, can);
        ui.horizontal_wrapped(|ui| {
            let label = if self.translation.preview.as_ref().is_some_and(|p| p.result.local.is_some()) {
                "Revise selected unlocked rows"
            } else { "Translate on this computer" };
            start |= ui.add_enabled(can, egui::Button::new(label)).clicked();
            if installed {
                remove = ui.add_enabled(idle && !installing, egui::Button::new("Remove this model")).clicked();
            } else if partial > 0 && !installing {
                remove = ui.add_enabled(idle, egui::Button::new(format!("Discard partial download ({:.2} GB)", gb(partial)))).clicked();
            }
        });
        ui.label("Prototype. The result is machine translation by a local model and may mistranslate; check it before \
            applying. Each result names the model that made it.");
        if download {
            self.translation.local.consent_open = false;
            self.start_local_install(None);
        }
        if let Some(from) = import {
            self.start_local_install(Some(from));
        }
        if remove {
            let l = &mut self.translation.local;
            let what = if installed { "Local model removed." } else { "Partial download discarded." };
            self.status = match local_ai::remove(&l.root, pack) {
                Ok(()) => what.into(),
                Err(e) => e,
            };
            l.installed = None;
        }
        start
    }

    /// Download the selected pack, or (`from`) use files already on this
    /// computer for it; either on its own thread.
    fn start_local_install(&mut self, from: Option<std::path::PathBuf>) {
        let l = &mut self.translation.local;
        if l.install.is_some() {
            return;
        }
        let cancel = Arc::new(AtomicBool::new(false));
        let stop = cancel.clone();
        let root = l.root.clone();
        let pack = l.pack;
        let kind = if from.is_some() { InstallKind::Import } else { InstallKind::Download };
        let (tx, rx) = mpsc::channel();
        let spawned = std::thread::Builder::new().name("local-pack-install".into()).spawn(move || {
            let send = |p: local_ai::Progress| { let _ = tx.send(InstallMessage::Progress(p)); };
            let r = match from {
                None => local_ai::install_reporting(&root, pack, local_ai::PACKS, &mut local_ai::https_fetch, &stop, None, send)
                    .map(|()| format!("{} installed and verified.", pack.label)),
                Some(from) => local_ai::import(&root, pack, local_ai::PACKS, &from, &stop, send).map(|i| match i {
                    local_ai::Imported::Installed => format!("{} installed from {}, verified; nothing was downloaded.", pack.label, from.display()),
                    local_ai::Imported::Partial { copied, missing } => format!(
                        "Copied and verified {} from {}. {:.2} GB still has to be downloaded to finish: choose Download.",
                        copied.join(", "), from.display(), missing as f64 / 1e9),
                }),
            };
            let _ = tx.send(InstallMessage::Done(r));
        });
        match spawned {
            Ok(_) => l.install = Some(Install::new(kind, cancel, rx)),
            Err(_) => self.status = "Could not start the pack download.".into(),
        }
    }
}

/// The width of `ui` that can be seen from its cursor, so text can wrap there.
fn visible_width(ui: &egui::Ui) -> f32 {
    (ui.clip_rect().right() - ui.cursor().left() - 4.0).clamp(100.0, ui.available_width().max(100.0))
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
                 ai: Some(AiResult { config, flags: o.flags, notes: o.notes, review: o.review }), local: None }
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
            local: None,
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
            local: None,
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

    /// A process-safety warning (a local runtime process or download that
    /// could not be confirmed finished) survives cancellation, a stale
    /// completion and a track change, and goes once the work has ended.
    #[test]
    fn a_local_process_warning_survives_cancel_stale_results_and_track_changes() {
        let mut w = window();
        let root = std::env::temp_dir().join(format!("moosik_ui_outstanding_{}", std::process::id()));
        w.translation.local.root = root.clone();
        let held = local_ai::seam::outstanding_for(&root);
        // Cancelled, then its completion arrives.
        let tx = queue(&mut w);
        w.cancel_translation();
        tx.send(Message::Done(Err("Local translation cancelled; nothing was applied.".into()))).unwrap();
        w.poll_translation();
        assert!(w.status.contains("could not be confirmed stopped"), "{}", w.status);
        // A stale completion (the request was superseded).
        let tx = queue(&mut w);
        w.translation.serial = w.translation.serial.wrapping_add(1);
        tx.send(Message::Done(Ok(result(false)))).unwrap();
        w.poll_translation();
        assert!(w.status.contains("could not be confirmed stopped"), "{}", w.status);
        // A track change abandons the live request.
        let _tx = queue(&mut w);
        w.abandon_translation();
        assert!(w.status.contains("could not be confirmed stopped"), "{}", w.status);
        // Once the work has ended, cancellation reads as cancellation again.
        drop(held);
        w.translation.task = None;
        let tx = queue(&mut w);
        w.cancel_translation();
        tx.send(Message::Done(Err("x".into()))).unwrap();
        w.poll_translation();
        assert!(w.status.contains("cancelled") && !w.status.contains("confirmed"), "{}", w.status);
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
            local: None,
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

    /// The local setup card at narrow, ordinary and wide sizes and at three
    /// scale factors: every piece of text stays inside the window (it wraps
    /// rather than running off), and nothing on it is drawn smaller than the
    /// body text — the licence notes are folded away by default.
    #[test]
    fn the_local_setup_card_is_readable_at_any_size_and_scale() {
        // Tall enough that the whole panel is in view (the widths are the point).
        for (wd, ht) in [(420.0, 1600.0), (520.0, 1600.0), (900.0, 1600.0)] {
            for ppp in [1.0f32, 1.5, 2.0] {
                let ctx = egui::Context::default();
                ctx.set_pixels_per_point(ppp);
                let body = ctx.style().text_styles[&egui::TextStyle::Body].size;
                let mut w = window();
                w.translation.open = true;
                w.translation.provider = Provider::Local;
                w.translation.local.source = Some(local_ai::Source::Japanese);
                w.translation.local.installed = Some(false);
                w.translation.local.consent_open = true;
                let mut sizes = Vec::new();
                for frame in 0..3 {
                    let input = egui::RawInput {
                        screen_rect: Some(egui::Rect::from_min_size(egui::Pos2::ZERO, egui::vec2(wd, ht))),
                        ..Default::default()
                    };
                    let out = ctx.run(input, |ctx| {
                        egui::CentralPanel::default().show(ctx, |ui| {
                            egui::ScrollArea::vertical().show(ui, |ui| w.translation_panel(ui, &track()));
                        });
                    });
                    sizes.clear();
                    for cs in &out.shapes {
                        if let egui::Shape::Text(t) = &cs.shape {
                            let r = t.visual_bounding_rect();
                            let text = t.galley.text().to_string();
                            // The local panel's text wraps. (Rows of the shared panel above
                            // it — the language row, the provider radios — are older and do
                            // not wrap at the narrowest widths; not checked here.)
                            let card = ["GB", "Measured on", "no network access", "translation model", "Hy-MT2", "Qwen"].iter().any(|k| text.contains(k));
                            // The last frame, once a new scale has been applied.
                            assert!(frame < 2 || !card || r.right() <= wd + 1.0, "{wd}x{ht} @{ppp}: {text:?} runs to {}", r.right());
                            for sec in &t.galley.job.sections {
                                sizes.push((sec.format.font_id.size, text.clone()));
                            }
                        }
                    }
                }
                assert!(sizes.iter().any(|(_, t)| t.contains("Before each run")), "the card was not drawn");
                for (size, text) in &sizes {
                    if ["Download", "Disk", "Memory", "Before each run", "Privacy", "Limitations", "Translates", "What it is"].iter().any(|k| text == k) {
                        assert!(*size >= body, "{text:?} at {size} < body {body}");
                    }
                }
                let min_card = sizes.iter().filter(|(_, t)| t.len() > 40 && ["GB", "Measured on", "no network access", "translation model", "Into", "free"].iter().any(|k| t.contains(k))).map(|(s, _)| *s).fold(f32::MAX, f32::min);
                assert!(min_card >= body, "{wd}x{ht} @{ppp}: some card text is {min_card} < body {body}");
            }
        }
    }

    #[test]
    fn changing_the_memory_route_makes_an_old_translation_stale() {
        let mut w = window();
        w.translation.provider = Provider::Local;
        w.translation.local.pack = &local_ai::HYMT2_30B;
        w.translation.local.source = Some(local_ai::Source::Japanese);
        w.translation.local.gpu = true;
        let old = w.local_identity().unwrap();
        w.translation.local.low_ram = true;
        assert_eq!(w.local_identity().unwrap().device, local_ai::Device::GpuLowRam);
        assert!(!w.local_matches(Some(&old)));
        w.translation.local.pack = &local_ai::HYMT2_7B;
        assert_eq!(w.local_identity().unwrap().device, local_ai::Device::Gpu);
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
            local: None,
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
                    credit: "Supplement supplied by NetEase".into(), replaces_existing: false, ai: None, local: None }))).unwrap();
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
                credit: "Supplement supplied by NetEase".into(), replaces_existing: false, ai: None, local: None }))).unwrap();
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
            credit: "MyMemory machine translation".into(), replaces_existing: false, ai: None, local: None }))).unwrap();
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
            tx.send(Message::Done(Ok(Translated { rows, credit: "Synthetic".into(), replaces_existing: false, ai: None, local: None }))).unwrap();
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

    // ------------------------------------------------ PROTOTYPE local pack

    fn wait(w: &mut LyricsWindow) {
        let t0 = std::time::Instant::now();
        while w.translation.is_busy() {
            assert!(t0.elapsed() < std::time::Duration::from_secs(20));
            std::thread::sleep(std::time::Duration::from_millis(10));
            w.poll_translation();
        }
    }

    #[test]
    fn local_chinese_to_chinese_needs_no_pack_and_stores_traditional() {
        let mut w = sheet("[00:01.00]简体中文
[00:02.00]歌词");
        w.set_provider(Provider::Local);
        w.translation.local.source = Some(local_ai::Source::Chinese);
        w.translation.local.installed = Some(false);
        w.start_translation(&track(), true);
        wait(&mut w);
        let p = w.translation.preview.as_ref().expect("a preview");
        assert_eq!(p.result.rows, vec![Some("簡體中文".into()), Some("歌詞".into())]);
        assert!(p.result.local.is_some() && p.result.ai.is_none());
        w.apply_translation_preview();
        assert_eq!(w.lyrics.as_ref().unwrap().lines[1].translation.as_deref(), Some("歌詞"));
    }

    #[test]
    fn local_translation_does_not_start_without_an_installed_pack() {
        let mut w = window();
        w.set_provider(Provider::Local);
        w.translation.local.source = Some(local_ai::Source::Korean);
        w.translation.local.installed = Some(false);
        w.start_translation(&track(), true);
        assert!(!w.translation.is_busy());
        w.translation.local.source = None;
        w.translation.local.installed = Some(true);
        w.start_translation(&track(), true);
        assert!(!w.translation.is_busy(), "no source chosen");
    }

    fn workbench_draft(w: &mut LyricsWindow, seed: u32) {
        w.set_provider(Provider::Local);
        w.translation.local.source = Some(local_ai::Source::Korean);
        w.translation.local.preferences.seed = seed;
        let identity = w.local_identity().unwrap();
        w.translation.preview = Some(Preview {
            snapshot: w.translation_snapshot().unwrap(),
            result: Translated { rows:vec![Some(format!("第一句 {seed}")),Some(format!("第二句 {seed}"))],
                credit:"Local AI translation (synthetic fixture)".into(), replaces_existing:true, ai:None,
                local:Some(Box::new(LocalResult { identity,flags:vec![None;2],retries:0 })) },
            display:Default::default(),
        });
        w.prepare_workbench_rows();
    }

    #[test]
    fn workbench_history_restores_seed_and_draft_without_touching_the_sheet() {
        let mut w = window();
        let before = w.lyrics.clone().unwrap();
        workbench_draft(&mut w, 11);
        w.translation.local.preferences.instructions = "少用啊吧".into();
        assert!(!w.result_matches(&w.translation.preview.as_ref().unwrap().result));
        assert!(w.preview_compatible(w.translation.preview.as_ref().unwrap()));
        assert!(w.local_revision().is_some());
        w.archive_local_preview();
        workbench_draft(&mut w, 22);
        w.restore_local_draft(0);
        assert_eq!(w.translation.local.preferences.seed, 11);
        assert!(w.translation.local.preferences.instructions.is_empty());
        assert_eq!(w.translation.preview.as_ref().unwrap().result.rows[0].as_deref(),Some("第一句 11"));
        assert_eq!(w.translation.workbench.history.len(),1);
        assert_eq!(w.lyrics.as_ref().unwrap().lines,before.lines);
        // Bounded history, including repeated draws; failed requests retain
        // their archived predecessor instead of overwriting it.
        for seed in 30..45 { w.archive_local_preview(); workbench_draft(&mut w, seed); }
        assert_eq!(w.translation.workbench.history.len(),8);
        w.loaded_for = Some(PathBuf::from("different.flac"));
        assert!(w.local_revision().is_none());
        w.restore_local_draft(0);
        assert_eq!(w.translation.local.preferences.seed,44);
    }

    #[test]
    fn workbench_selected_adoption_and_undo_do_not_overwrite_newer_edits() {
        let mut w = window();
        let before = w.lyrics.clone().unwrap();
        workbench_draft(&mut w,11);
        w.translation.workbench.adopt[1] = false;
        w.apply_translation_preview();
        assert_eq!(w.lyrics.as_ref().unwrap().lines[0].translation.as_deref(),Some("第一句 11"));
        assert_eq!(w.lyrics.as_ref().unwrap().lines[1].translation,before.lines[1].translation);
        w.undo_local_adoption();
        assert_eq!(w.lyrics.as_ref().unwrap().lines,before.lines);
        workbench_draft(&mut w,22);
        w.apply_translation_preview();
        w.lyrics.as_mut().unwrap().lines[0].translation = Some("new manual edit".into());
        w.sheet_rev += 1;
        w.undo_local_adoption();
        assert_eq!(w.lyrics.as_ref().unwrap().lines[0].translation.as_deref(),Some("new manual edit"));
    }

    #[test]
    fn workbench_seed_change_invalidates_a_pending_completion_and_lock_selection_is_captured() {
        let mut w = window();
        workbench_draft(&mut w,11);
        w.translation.workbench.locked[0] = true;
        w.translation.workbench.selected[1] = false;
        let r = w.local_revision().unwrap();
        assert_eq!(r.locked,vec![true,false]);
        assert_eq!(r.selected,vec![true,false]);
        let identity = w.local_identity().unwrap();
        let mut result = w.translation.preview.take().unwrap().result;
        let tx = queue(&mut w);
        w.translation.task.as_mut().unwrap().local = Some(identity);
        w.translation.local.preferences.reroll();
        result.rows[0] = Some("stale result".into());
        tx.send(Message::Done(Ok(result))).unwrap();
        w.poll_translation();
        assert!(w.translation.preview.is_none());
        assert!(!w.translation_dirty);
    }

    #[test]
    fn a_local_preview_is_not_applied_after_its_identity_changed() {
        let mut w = sheet("[00:01.00]简体
[00:02.00]歌词");
        w.set_provider(Provider::Local);
        w.translation.local.source = Some(local_ai::Source::Chinese);
        w.start_translation(&track(), true);
        wait(&mut w);
        assert!(w.translation.preview.is_some());
        w.translation.local.target = local_ai::Target::English;
        w.apply_translation_preview();
        assert!(w.lyrics.as_ref().unwrap().lines.iter().all(|l| l.translation.is_none()), "nothing applied");
        // A queued completion made with other settings is never previewed.
        w.translation.local.target = local_ai::Target::TraditionalChinese;
        w.start_translation(&track(), true);
        w.translation.local.fluent = true;
        wait(&mut w);
        assert!(w.translation.preview.is_none());
    }
}

#[cfg(test)]
mod choice_tests {
    use super::*;

    /// The local model the user picks is remembered by ID in the model
    /// folder; an unknown or missing record falls back (to the first
    /// installed, then the first listed), and nothing else is written.
    #[test]
    fn the_chosen_local_model_is_remembered() {
        let root = std::env::temp_dir().join(format!("moosik_chosen_{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&root);
        assert!(chosen_pack(&root).is_none());
        remember_choice(&root, &local_ai::HYMT2_1_8B);
        assert_eq!(chosen_pack(&root).map(|p| p.id), Some(local_ai::HYMT2_1_8B.id));
        assert_eq!(std::fs::read_dir(&root).unwrap().count(), 1, "something besides the choice was written");
        std::fs::write(root.join(CHOSEN), "no-such-pack").unwrap();
        assert!(chosen_pack(&root).is_none());
        let _ = std::fs::remove_dir_all(&root);
    }
}
