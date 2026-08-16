//! The lyrics window: display, online lookup, manual search, and a tap-along
//! sync editor.
//!
//! Its own viewport rather than a panel, so it can sit on a second screen while
//! the player stays where it is — which is how a lyrics view actually gets used.

use super::{Hit, LyricSource, LyricLine, Lyrics};
use crate::pal;
use eframe::egui;
use egui::RichText;
use std::path::{Path, PathBuf};
use std::sync::mpsc::Receiver;
use std::time::Duration;

/// What the window wants the player to do. The window never touches playback
/// itself — it is handed a position and hands back an intent.
#[derive(Clone, Copy, PartialEq, Debug, Default)]
pub enum LyricsAction {
    #[default]
    None,
    Seek(Duration),
}

/// Everything the window needs to know about the playing track.
pub struct TrackRef<'a> {
    pub path: &'a Path,
    pub title: &'a str,
    pub artist: &'a str,
    pub album: &'a str,
    pub duration: Option<Duration>,
}

enum Fetched {
    /// The ladder found one.
    Hit(Box<Hit>),
    /// The ladder ran out — a real answer, not an error.
    Nothing,
    Failed(String),
}

pub struct LyricsWindow {
    pub open: bool,
    /// Track the loaded sheet belongs to, so a track change reloads.
    loaded_for: Option<PathBuf>,
    lyrics: Option<Lyrics>,
    status: String,

    fetch_rx: Option<Receiver<Fetched>>,
    /// Rows, plus a note naming any source that could not answer.
    search_rx: Option<Receiver<Result<(Vec<Hit>, Option<String>), String>>>,
    results: Vec<Hit>,
    search_query: String,
    search_open: bool,

    /// Paste box. The database does not have everything — most of a niche
    /// Vocaloid or UTAU library is simply not in it — so there has to be a way
    /// to bring lyrics in from anywhere and time them here.
    paste_open: bool,
    paste_buf: String,

    // ── Sync editor ────────────────────────────────────────────────────────
    editing: bool,
    edit: Vec<(Option<Duration>, String)>,
    /// Next line the Stamp key will time.
    cursor: usize,
    dirty: bool,

    // ── Display ────────────────────────────────────────────────────────────
    font_size: f32,
    auto_scroll: bool,
}

impl Default for LyricsWindow {
    fn default() -> Self {
        Self {
            open: false,
            loaded_for: None,
            lyrics: None,
            status: String::new(),
            fetch_rx: None,
            search_rx: None,
            results: Vec::new(),
            search_query: String::new(),
            search_open: false,
            paste_open: false,
            paste_buf: String::new(),
            editing: false,
            edit: Vec::new(),
            cursor: 0,
            dirty: false,
            font_size: 17.0,
            auto_scroll: true,
        }
    }
}

impl LyricsWindow {
    fn busy(&self) -> bool { self.fetch_rx.is_some() || self.search_rx.is_some() }

    /// Load whatever is already on disk or in the tags for this track.
    ///
    /// Never fetches. An automatic network call on every track change is a
    /// surprise the user did not ask for, so lookup is always a button.
    fn load_local(&mut self, track: &TrackRef) {
        self.lyrics = super::load(track.path, super::embedded_text(track.path).as_deref());
        self.loaded_for = Some(track.path.to_path_buf());
        self.results.clear();
        self.editing = false;
        self.dirty = false;
        self.cursor = 0;
        self.status = match &self.lyrics {
            Some(l) if l.is_synced() => String::new(),
            Some(_) => "Unsynced lyrics — “Sync…” can time them.".into(),
            None => "No lyrics found for this track.".into(),
        };
        if self.search_query.is_empty() || !self.search_open {
            self.search_query = format!("{} {}", track.artist, track.title);
        }
    }

    fn start_fetch(&mut self, track: &TrackRef) {
        let (tx, rx) = std::sync::mpsc::channel();
        let (a, t, al, d) = (track.artist.to_string(), track.title.to_string(),
                             track.album.to_string(), track.duration);
        std::thread::spawn(move || {
            let msg = match super::find_anywhere(&a, &t, &al, d) {
                Ok(Some(h)) => Fetched::Hit(Box::new(h)),
                Ok(None) => Fetched::Nothing,
                Err(e) => Fetched::Failed(e),
            };
            let _ = tx.send(msg);
        });
        self.fetch_rx = Some(rx);
        self.status = "Searching…".into();
    }

    fn start_search(&mut self) {
        let q = self.search_query.trim().to_string();
        if q.is_empty() { return; }
        let (tx, rx) = std::sync::mpsc::channel();
        std::thread::spawn(move || { let _ = tx.send(super::search_anywhere(&q)); });
        self.search_rx = Some(rx);
        self.results.clear();
        self.status = "Searching…".into();
    }

    /// Take pasted text as the sheet. Accepts LRC as readily as plain text —
    /// people paste both, and refusing timestamps would mean stripping the very
    /// thing that makes a paste worth having.
    fn take_paste(&mut self) {
        let l = super::parse(&self.paste_buf, LyricSource::Online);
        if l.is_empty() {
            self.status = "Nothing to use — the box is empty.".into();
            return;
        }
        let synced = l.is_synced();
        let n = l.lines.len();
        self.lyrics = Some(l);
        self.dirty = true;
        self.paste_open = false;
        self.paste_buf.clear();
        if synced {
            self.status = format!("{n} timed lines. Save to write the .lrc.");
        } else {
            // Straight into the editor: an untimed paste is not finished, and
            // making the user find the next button is a step with no decision
            // in it.
            self.begin_edit();
            self.status = format!(
                "{n} lines. Play the track and press Space (or ⏱) as each one starts.");
        }
    }

    /// Take a row the user picked from search.
    ///
    /// Some sources list results without their lyrics — fetching every row's
    /// text just to draw a list would be a request per row — so a row that
    /// arrived empty is filled in the background before it is applied.
    fn choose(&mut self, hit: Hit) {
        if hit.best_text().is_some() { self.apply(&hit); return; }
        let (tx, rx) = std::sync::mpsc::channel();
        std::thread::spawn(move || {
            let mut h = hit;
            let msg = match super::netease::fill_lyrics(&mut h) {
                Ok(()) if h.best_text().is_some() => Fetched::Hit(Box::new(h)),
                Ok(()) => Fetched::Nothing,
                Err(e) => Fetched::Failed(e),
            };
            let _ = tx.send(msg);
        });
        self.fetch_rx = Some(rx);
        self.status = "Fetching lyrics…".into();
    }

    /// Adopt a search result as the current sheet, without saving it.
    fn apply(&mut self, hit: &Hit) {
        match hit.best_text() {
            Some(text) => {
                let l = super::parse(text, LyricSource::Online);
                self.status = if l.is_synced() {
                    format!("{} — {} · via {}", hit.title, hit.artist, hit.source)
                } else {
                    format!("{} — {} · via {} (plain text; “Sync…” can time it)",
                            hit.title, hit.artist, hit.source)
                };
                self.lyrics = Some(l);
                self.dirty = true;
                self.editing = false;
            }
            None => self.status = "That result has no lyrics.".into(),
        }
    }

    fn save(&mut self, track: &TrackRef) {
        let Some(l) = &self.lyrics else { return };
        match super::save_sidecar(track.path, l) {
            Ok(p) => {
                self.dirty = false;
                if let Some(ly) = &mut self.lyrics { ly.source = LyricSource::Sidecar; }
                self.status = format!("Saved {}", p.file_name().unwrap_or_default().to_string_lossy());
            }
            Err(e) => self.status = format!("Could not save: {e}"),
        }
    }

    /// Move the current sheet into the editor. Lines keep whatever timing they
    /// already had, so this re-times an existing sheet as happily as it times a
    /// plain one.
    fn begin_edit(&mut self) {
        let lines = self.lyrics.as_ref().map(|l| l.lines.clone()).unwrap_or_default();
        self.edit = lines.into_iter().map(|l| (l.at, l.text)).collect();
        if self.edit.is_empty() { self.edit.push((None, String::new())); }
        // Resume at the first untimed line rather than the top, so a sync that
        // was interrupted picks up where it stopped.
        self.cursor = self.edit.iter().position(|(t, _)| t.is_none()).unwrap_or(0);
        self.editing = true;
        self.status = "Play the track and press Space (or ⏱) as each line starts.".into();
    }

    fn commit_edit(&mut self) {
        let mut lines: Vec<LyricLine> = self.edit.iter()
            .filter(|(t, s)| t.is_some() || !s.trim().is_empty())
            .map(|(at, text)| LyricLine { at: *at, text: text.clone() })
            .collect();
        lines.sort_by_key(|l| l.at);
        let old = self.lyrics.as_ref();
        self.lyrics = Some(Lyrics {
            lines,
            source: old.map(|l| l.source).unwrap_or(LyricSource::Online),
            offset_ms: old.map(|l| l.offset_ms).unwrap_or(0),
            title: old.and_then(|l| l.title.clone()),
            artist: old.and_then(|l| l.artist.clone()),
        });
        self.editing = false;
        self.dirty = true;
        self.status = "Edited — Save to write the .lrc.".into();
    }

    /// Draw the window. `pos` is the current playback position.
    pub fn ui(
        &mut self, ctx: &egui::Context, track: Option<TrackRef>, pos: Duration,
    ) -> LyricsAction {
        if !self.open { return LyricsAction::None; }
        let Some(track) = track else { return LyricsAction::None };

        if self.loaded_for.as_deref() != Some(track.path) {
            self.load_local(&track);
        }
        self.poll();

        let mut action = LyricsAction::None;
        let mut close = false;
        let vp_id = egui::ViewportId::from_hash_of("moosik_lyrics");
        let vp = egui::ViewportBuilder::default()
            .with_title(format!("Lyrics — {}", track.title))
            .with_inner_size([520.0, 640.0])
            .with_resizable(true);

        ctx.show_viewport_immediate(vp_id, vp, |vctx, _class| {
            if vctx.input(|i| i.viewport().close_requested()) { close = true; return; }
            // A network call in flight has no other reason to redraw.
            if self.busy() { vctx.request_repaint_after(Duration::from_millis(100)); }

            egui::TopBottomPanel::top("lyr_top").show(vctx, |ui| {
                ui.add_space(4.0);
                action = self.toolbar(ui, &track).max_or(action);
                ui.add_space(4.0);
            });
            egui::TopBottomPanel::bottom("lyr_bot").show(vctx, |ui| {
                ui.add_space(3.0);
                ui.horizontal_wrapped(|ui| {
                    let dark = ui.visuals().dark_mode;
                    if !self.status.is_empty() {
                        ui.label(RichText::new(&self.status).size(11.5)
                                 .color(pal::text_dim(dark)));
                    }
                    if self.dirty {
                        ui.label(RichText::new("• unsaved").size(11.5).color(pal::amber(dark)));
                    }
                });
                ui.add_space(3.0);
            });
            egui::CentralPanel::default().show(vctx, |ui| {
                if self.search_open { self.search_panel(ui); }
                if self.paste_open { self.paste_panel(ui); }
                if self.editing {
                    action = self.editor(ui, pos).max_or(action);
                } else {
                    self.viewer(ui, pos);
                }
            });
        });

        if close { self.open = false; }
        action
    }

    fn poll(&mut self) {
        if let Some(rx) = &self.fetch_rx
            && let Ok(msg) = rx.try_recv() {
            self.fetch_rx = None;
            match msg {
                Fetched::Hit(h) => self.apply(&h),
                Fetched::Nothing => self.status =
                    "No confident match. Try “Search…”, or “Paste…” if no \
                     database has this one.".into(),
                // Never phrased as "not found": a source that could not answer
                // has said nothing about whether the lyrics exist, and sending
                // the user off to transcribe by hand over a rate limit that
                // clears in a minute is the wrong outcome.
                Fetched::Failed(e) => self.status = format!("Could not finish the lookup — {e}"),
            }
        }
        if let Some(rx) = &self.search_rx
            && let Ok(res) = rx.try_recv() {
            self.search_rx = None;
            match res {
                Ok((hits, note)) => {
                    self.status = if hits.is_empty() {
                        "No results.".into()
                    } else {
                        format!("{} result{}", hits.len(), if hits.len() == 1 { "" } else { "s" })
                    };
                    // A source that could not answer has to be named, or a
                    // short list looks like a verdict on the song rather than
                    // on half the databases having been skipped.
                    if let Some(n) = note {
                        self.status.push_str(&format!(" · {n}"));
                    }
                    self.results = hits;
                }
                Err(e) => self.status = format!("Search could not finish — {e}"),
            }
        }
    }

    fn toolbar(&mut self, ui: &mut egui::Ui, track: &TrackRef) -> LyricsAction {
        let mut action = LyricsAction::None;
        ui.horizontal_wrapped(|ui| {
            let idle = !self.busy();
            if ui.add_enabled(idle && !self.editing, egui::Button::new("🔍 Find lyrics"))
                .on_hover_text(
                    "Look this track up on LRCLIB.\n\nTries the exact tags first, then \
                     progressively looser queries with the decoration stripped — brackets, \
                     feat. credits, full-width text — matching on duration throughout.")
                .clicked()
            {
                self.start_fetch(track);
            }
            if ui.add_enabled(idle, egui::Button::new("Search…"))
                .on_hover_text("Search LRCLIB by hand and pick the take you want.")
                .clicked()
            {
                self.search_open = !self.search_open;
            }
            if ui.add_enabled(!self.editing, egui::Button::new("📋 Paste…"))
                .on_hover_text(
                    "Paste lyrics from anywhere.\n\nThe database does not have \
                     everything — a niche Vocaloid or UTAU library is mostly \
                     missing from it — so this is the way in for the rest. \
                     Plain text drops straight into the sync editor; LRC with \
                     timestamps is kept as it is.")
                .clicked()
            {
                self.paste_open = !self.paste_open;
            }

            ui.separator();

            if self.editing {
                if ui.button("✔ Done").clicked() { self.commit_edit(); }
                if ui.button("✖ Cancel").clicked() {
                    self.editing = false;
                    self.status.clear();
                }
            } else {
                let has = self.lyrics.is_some();
                if ui.add_enabled(has, egui::Button::new("⏱ Sync…"))
                    .on_hover_text("Time the lines by tapping along with the song.")
                    .clicked()
                {
                    self.begin_edit();
                }
                if ui.add_enabled(has && self.dirty, egui::Button::new("💾 Save"))
                    .on_hover_text(format!(
                        "Write {}\n\nA sidecar file — the audio file is never modified.",
                        super::sidecar_path(track.path).file_name()
                            .unwrap_or_default().to_string_lossy()))
                    .clicked()
                {
                    self.save(track);
                }
            }

            ui.separator();
            ui.checkbox(&mut self.auto_scroll, "Follow")
                .on_hover_text("Keep the current line centred.");
            ui.add(egui::Slider::new(&mut self.font_size, 11.0..=34.0)
                   .show_value(false).trailing_fill(true))
                .on_hover_text("Text size");

            if let Some(l) = &self.lyrics
                && l.is_synced()
            {
                ui.separator();
                let mut off = l.offset_ms as f32 / 1000.0;
                if ui.add(egui::DragValue::new(&mut off).speed(0.05).suffix(" s")
                          .range(-30.0..=30.0))
                    .on_hover_text("Nudge the timing. Positive shows lines earlier.")
                    .changed()
                {
                    if let Some(l) = &mut self.lyrics {
                        l.offset_ms = (off * 1000.0).round() as i64;
                    }
                    self.dirty = true;
                }
            }
            let _ = &mut action;
        });
        action
    }

    fn search_panel(&mut self, ui: &mut egui::Ui) {
        ui.group(|ui| {
            ui.horizontal(|ui| {
                let te = ui.add(egui::TextEdit::singleline(&mut self.search_query)
                                .desired_width(f32::INFINITY - 90.0)
                                .hint_text("artist / title / anything"));
                let go = ui.add_enabled(!self.busy(), egui::Button::new("Go")).clicked();
                if go || (te.lost_focus() && ui.input(|i| i.key_pressed(egui::Key::Enter))) {
                    self.start_search();
                }
                if ui.button("✖").on_hover_text("Close search").clicked() {
                    self.search_open = false;
                }
            });
            if !self.results.is_empty() {
                let dark = ui.visuals().dark_mode;
                let mut chosen: Option<usize> = None;
                egui::ScrollArea::vertical().max_height(190.0)
                    .id_salt("lyr_results").show(ui, |ui| {
                    for (i, h) in self.results.iter().enumerate() {
                        let mins = (h.duration as u64) / 60;
                        let secs = (h.duration as u64) % 60;
                        // Sources that fetch lyrics separately return rows whose
                        // contents are not known until one is picked, so the
                        // badge says "not yet fetched" rather than lying with a
                        // plain-text icon.
                        let (icon, tint) = match (h.has_synced(), h.best_text().is_some()) {
                            (true, _) => ("⏱", pal::ok(dark)),
                            (false, true) => ("📄", pal::text_faint(dark)),
                            (false, false) => ("↓", pal::text_faint(dark)),
                        };
                        let resp = ui.horizontal(|ui| {
                            ui.label(RichText::new(icon).size(12.0).color(tint));
                            ui.vertical(|ui| {
                                ui.label(RichText::new(&h.title).size(13.0)
                                         .color(pal::text(dark)));
                                let mut sub = h.artist.clone();
                                if !h.album.is_empty() {
                                    sub.push_str(&format!(" · {}", h.album));
                                }
                                sub.push_str(&format!(" · {mins}:{secs:02} · {}", h.source));
                                ui.label(RichText::new(sub).size(11.0)
                                         .color(pal::text_faint(dark)));
                            });
                        }).response;
                        // The row is only a label stack, so give it its own
                        // click target across the full width.
                        let rect = resp.rect;
                        if ui.interact(rect, ui.id().with(("lyr_hit", i)),
                                       egui::Sense::click()).clicked() {
                            chosen = Some(i);
                        }
                        ui.separator();
                    }
                });
                if let Some(i) = chosen {
                    let h = self.results[i].clone();
                    self.choose(h);
                    self.search_open = false;
                }
            }
        });
        ui.add_space(4.0);
    }

    fn paste_panel(&mut self, ui: &mut egui::Ui) {
        let dark = ui.visuals().dark_mode;
        ui.group(|ui| {
            ui.label(RichText::new("Paste lyrics — plain text or LRC")
                     .size(12.0).color(pal::text_dim(dark)));
            egui::ScrollArea::vertical().max_height(200.0).id_salt("lyr_paste").show(ui, |ui| {
                ui.add(egui::TextEdit::multiline(&mut self.paste_buf)
                       .desired_width(f32::INFINITY)
                       .desired_rows(8)
                       .hint_text("One line per lyric line."));
            });
            ui.horizontal(|ui| {
                let n = self.paste_buf.lines().filter(|l| !l.trim().is_empty()).count();
                if ui.add_enabled(n > 0, egui::Button::new("Use these lines")).clicked() {
                    self.take_paste();
                }
                if ui.button("✖").on_hover_text("Close").clicked() {
                    self.paste_open = false;
                }
                ui.label(RichText::new(format!("{n} line{}", if n == 1 { "" } else { "s" }))
                         .size(11.0).color(pal::text_faint(dark)));
            });
        });
        ui.add_space(4.0);
    }

    fn viewer(&mut self, ui: &mut egui::Ui, pos: Duration) {
        let dark = ui.visuals().dark_mode;
        let Some(l) = &self.lyrics else {
            ui.centered_and_justified(|ui| {
                ui.label(RichText::new(
                    "No lyrics.\n\n\
                     “Find lyrics” looks this track up on LRCLIB.\n\
                     “Paste…” takes them from anywhere else and can time them here.")
                    .size(13.0).color(pal::text_faint(dark)));
            });
            return;
        };
        let active = l.active_index(pos);
        // `auto_shrink` defaults to true on both axes, which sizes the scroll
        // area to its content — so the scrollbar sits against the longest lyric
        // line, stranded in the middle of the window. Lyrics are short lines in
        // a wide panel, which makes it especially obvious.
        egui::ScrollArea::vertical().id_salt("lyr_view")
            .auto_shrink([false, false]).show(ui, |ui| {
            ui.add_space(6.0);
            for (i, line) in l.lines.iter().enumerate() {
                let is_active = active == Some(i);
                if line.text.trim().is_empty() {
                    ui.add_space(self.font_size * 0.6);
                    continue;
                }
                let (size, colour) = if is_active {
                    (self.font_size * 1.06, pal::accent(dark))
                } else if active.is_some_and(|a| i < a) {
                    (self.font_size, pal::text_faint(dark))
                } else {
                    (self.font_size, pal::text(dark))
                };
                let mut rt = RichText::new(&line.text).size(size).color(colour);
                if is_active { rt = rt.strong(); }
                let resp = ui.add(egui::Label::new(rt).sense(egui::Sense::click()));
                if is_active && self.auto_scroll {
                    resp.scroll_to_me(Some(egui::Align::Center));
                }
                ui.add_space(3.0);
            }
            ui.add_space(ui.available_height().max(80.0));
        });
    }

    fn editor(&mut self, ui: &mut egui::Ui, pos: Duration) -> LyricsAction {
        let mut action = LyricsAction::None;
        let dark = ui.visuals().dark_mode;

        // Space stamps the next line. This is a separate viewport, so the key
        // does not reach the player's play/pause binding — but it *would* reach
        // a lyric line being edited, so a focused text field takes it instead.
        let typing = ui.memory(|m| m.focused().is_some());
        let stamp = !typing && ui.input(|i| i.key_pressed(egui::Key::Space));

        ui.horizontal(|ui| {
            let tapped = ui.add(egui::Button::new(RichText::new("⏱ Stamp").size(14.0)))
                .on_hover_text("Time the highlighted line at the current position (Space)")
                .clicked() || stamp;
            if tapped && self.cursor < self.edit.len() {
                self.edit[self.cursor].0 = Some(pos);
                self.cursor += 1;
            }
            if ui.button("↩ Back").on_hover_text("Undo the last stamp").clicked()
                && self.cursor > 0
            {
                self.cursor -= 1;
                self.edit[self.cursor].0 = None;
            }
            if ui.button("Clear all times").clicked() {
                for e in &mut self.edit { e.0 = None; }
                self.cursor = 0;
            }
            if ui.button("+ Line").clicked() {
                self.edit.push((None, String::new()));
            }
            ui.label(RichText::new(format!("{}/{}", self.cursor.min(self.edit.len()),
                                           self.edit.len()))
                     .size(11.5).color(pal::text_faint(dark)));
        });
        ui.separator();

        let mut remove: Option<usize> = None;
        egui::ScrollArea::vertical().id_salt("lyr_edit")
            .auto_shrink([false, false]).show(ui, |ui| {
            for i in 0..self.edit.len() {
                let is_cursor = i == self.cursor;
                ui.horizontal(|ui| {
                    let stamp_txt = match self.edit[i].0 {
                        Some(t) => super::fmt_stamp(t),
                        None => "  --:--  ".into(),
                    };
                    let btn = egui::Button::new(
                        RichText::new(stamp_txt).monospace().size(12.0)
                            .color(if self.edit[i].0.is_some() { pal::ok(dark) }
                                   else { pal::text_faint(dark) }));
                    if ui.add(btn).on_hover_text("Jump the player here").clicked()
                        && let Some(t) = self.edit[i].0
                    {
                        action = LyricsAction::Seek(t);
                    }
                    if is_cursor {
                        ui.label(RichText::new("▶").size(12.0).color(pal::accent(dark)));
                    }
                    ui.add(egui::TextEdit::singleline(&mut self.edit[i].1)
                           .desired_width(ui.available_width() - 30.0)
                           .font(egui::FontId::proportional(self.font_size.min(16.0))));
                    if ui.small_button("✖").on_hover_text("Delete line").clicked() {
                        remove = Some(i);
                    }
                });
            }
            ui.add_space(60.0);
        });
        if let Some(i) = remove {
            self.edit.remove(i);
            self.cursor = self.cursor.min(self.edit.len());
        }
        action
    }
}

impl LyricsAction {
    /// Keep whichever of two actions is not `None`; later wins.
    fn max_or(self, other: LyricsAction) -> LyricsAction {
        match self { LyricsAction::None => other, a => a }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn win_with(text: &str) -> LyricsWindow {
        LyricsWindow {
            lyrics: Some(super::super::parse(text, LyricSource::Online)),
            ..Default::default()
        }
    }

    /// The point of the sync editor: plain text in, timed sheet out.
    #[test]
    fn syncing_plain_text_produces_a_synced_sheet() {
        let mut w = win_with("one\ntwo\nthree\n");
        assert!(!w.lyrics.as_ref().unwrap().is_synced());
        w.begin_edit();
        assert_eq!(w.cursor, 0, "an untimed sheet starts at the top");

        for (i, secs) in [1u64, 5, 9].iter().enumerate() {
            w.edit[i].0 = Some(Duration::from_secs(*secs));
            w.cursor += 1;
        }
        w.commit_edit();

        let l = w.lyrics.unwrap();
        assert!(l.is_synced());
        assert_eq!(l.lines.len(), 3);
        assert_eq!(l.lines[1].at, Some(Duration::from_secs(5)));
        assert!(w.dirty, "an edit must leave something to save");
    }

    /// An interrupted sync resumes where it stopped rather than at the top.
    #[test]
    fn editing_resumes_at_the_first_untimed_line() {
        let mut w = win_with("[00:01.00]one\n[00:05.00]two\nthree\nfour\n");
        w.begin_edit();
        assert_eq!(w.cursor, 2);
    }

    /// Stamping out of order must still yield a sheet in playback order,
    /// because `active_index` walks it forwards and stops at the first future
    /// timestamp.
    #[test]
    fn commit_sorts_by_time() {
        let mut w = win_with("a\nb\n");
        w.begin_edit();
        w.edit[0].0 = Some(Duration::from_secs(9));
        w.edit[1].0 = Some(Duration::from_secs(2));
        w.commit_edit();
        let l = w.lyrics.unwrap();
        assert_eq!(l.lines[0].text, "b");
        assert_eq!(l.active_index(Duration::from_secs(5)), Some(0));
    }

    /// Blank untimed rows are scaffolding, not lyrics.
    #[test]
    fn commit_drops_empty_untimed_rows() {
        let mut w = win_with("a\n");
        w.begin_edit();
        w.edit.push((None, "   ".into()));
        w.edit.push((None, String::new()));
        w.commit_edit();
        assert_eq!(w.lyrics.unwrap().lines.len(), 1);
    }

    /// An empty sheet still gives the editor a row to type into, rather than
    /// an editor with nothing in it and no way to add anything.
    #[test]
    fn editing_nothing_still_gives_a_row() {
        let mut w = LyricsWindow::default();
        w.begin_edit();
        assert_eq!(w.edit.len(), 1);
    }

    #[test]
    fn applying_a_hit_marks_it_unsaved_and_keeps_the_text() {
        let mut w = LyricsWindow::default();
        w.apply(&Hit {
            title: "T".into(), artist: "A".into(),
            synced: Some("[00:02.00]hi".into()), ..Default::default()
        });
        let l = w.lyrics.as_ref().unwrap();
        assert!(l.is_synced());
        assert_eq!(l.lines[0].text, "hi");
        assert!(w.dirty, "a fetched sheet is not on disk yet");
    }

    /// Plain text is worth showing — it is the whole point of the fallback —
    /// but it must not claim to be synced.
    #[test]
    fn a_plain_hit_is_shown_unsynced() {
        let mut w = LyricsWindow::default();
        w.apply(&Hit {
            title: "T".into(), artist: "A".into(),
            plain: Some("line one\nline two".into()), ..Default::default()
        });
        let l = w.lyrics.as_ref().unwrap();
        assert!(!l.is_synced());
        assert_eq!(l.lines.len(), 2);
        assert!(w.status.contains("Sync"), "should point at the fix: {}", w.status);
    }

    /// The path for every track the database does not have: paste plain text
    /// and land in the editor ready to tap, without another button in between.
    #[test]
    fn pasting_plain_text_opens_the_editor() {
        let mut w = LyricsWindow {
            paste_buf: "ぽいぽい\nもういいや\nさよなら".into(),
            paste_open: true,
            ..Default::default()
        };
        w.take_paste();
        assert!(w.editing, "an untimed paste is not finished");
        assert_eq!(w.edit.len(), 3);
        assert_eq!(w.cursor, 0);
        assert!(!w.paste_open);
        assert!(w.paste_buf.is_empty(), "the box must not keep a stale copy");
        assert!(w.dirty);
    }

    /// Pasting an LRC keeps its timing, and must not throw the user into an
    /// editor for a sheet that is already finished.
    #[test]
    fn pasting_lrc_keeps_the_timestamps() {
        let mut w = LyricsWindow {
            paste_buf: "[00:01.00]one\n[00:09.50]two".into(),
            paste_open: true,
            ..Default::default()
        };
        w.take_paste();
        assert!(!w.editing);
        let l = w.lyrics.as_ref().unwrap();
        assert!(l.is_synced());
        assert_eq!(l.lines[1].at, Some(Duration::from_millis(9_500)));
    }

    #[test]
    fn pasting_nothing_changes_nothing() {
        let mut w = LyricsWindow { paste_buf: "  \n\n ".into(), paste_open: true,
                                   ..Default::default() };
        w.take_paste();
        assert!(w.lyrics.is_none());
        assert!(w.paste_open, "the box stays open so the user can try again");
        assert!(!w.dirty);
    }

    #[test]
    fn an_action_survives_being_folded_with_none() {
        let seek = LyricsAction::Seek(Duration::from_secs(3));
        assert_eq!(LyricsAction::None.max_or(seek), seek);
        assert_eq!(seek.max_or(LyricsAction::None), seek);
    }
}
