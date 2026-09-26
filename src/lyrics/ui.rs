//! The lyrics window: display, online lookup, manual search, and a tap-along
//! sync editor.
//!
//! Its own viewport rather than a panel, so it can sit on a second screen while
//! the player stays where it is — which is how a lyrics view actually gets used.

use super::{Extra, Hit, LyricSource, LyricLine, Lyrics};
use super::romaji::Word;
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

/// Which script the viewer shows, for a sheet with Japanese in it.
#[derive(Clone, Copy, PartialEq, Eq, Debug, Default)]
enum Script {
    #[default]
    Original,
    /// Original with kana readings over the kanji.
    Furigana,
    Romaji,
    /// Original with the romaji under it.
    Both,
}

/// One line's generated reading, as words and as a whole line.
#[derive(Default)]
struct Auto {
    text: String,
    words: Vec<Word>,
    line: String,
    /// How the line reads, and the supplied romaji and override it was
    /// worked out for.
    reading: Option<(Option<String>, bool, super::romaji::Reading)>,
}

impl Auto {
    /// The line's reading for this supplied romaji and override, worked out
    /// once and kept until either changes.
    fn reading(&mut self, given: Option<&str>, overrides: bool) -> super::romaji::Reading {
        let fresh = self.reading.as_ref().is_some_and(|(g, o, _)| g.as_deref() == given && *o == overrides);
        if !fresh {
            let r = super::romaji::read_line(&self.words, given, overrides);
            self.reading = Some((given.map(str::to_string), overrides, r));
        }
        self.reading.as_ref().map(|r| r.2.clone()).unwrap_or_else(|| unreachable!())
    }
}

/// Generated reading for line `i`, recomputed only when that line's text
/// changes — so it follows edits without anything having to invalidate it.
fn auto_for<'a>(memo: &'a mut Vec<Auto>, i: usize, text: &str) -> &'a mut Auto {
    if memo.len() <= i { memo.resize_with(i + 1, Default::default); }
    let a = &mut memo[i];
    if a.text != text || a.words.is_empty() {
        let words = super::romaji::words(text);
        *a = Auto { text: text.to_string(), line: super::romaji::join(&words), words, reading: None };
    }
    a
}

/// One piece of a line as drawn: its text, a reading over or under it, and
/// whether the singing has reached it.
#[derive(Clone, Debug, PartialEq)]
struct Cell {
    text: String,
    above: Option<String>,
    below: Option<String>,
    lit: bool,
    /// A word gap before it, so the eye can tell where words divide.
    gap: bool,
}

struct Look {
    size: f32,
    colour: egui::Color32,
    /// Colour of what has been sung.
    lit: egui::Color32,
    strong: bool,
    /// Readings are generated, not supplied: shown in italics.
    guessed: bool,
}

/// Draw a line as cells, each with its reading centred over or under it.
fn draw_cells(ui: &mut egui::Ui, cells: &[Cell], look: &Look) -> egui::Response {
    let galley = |ui: &egui::Ui, rt: RichText| egui::WidgetText::from(rt)
        .into_galley(ui, Some(egui::TextWrapMode::Extend), f32::INFINITY, egui::TextStyle::Body);
    let (sa, sb) = (look.size * 0.5, look.size * 0.62);
    let row = |ui: &egui::Ui, on: bool, sz: f32| {
        if on { ui.fonts(|f| f.row_height(&egui::FontId::proportional(sz))) } else { 0.0 }
    };
    ui.horizontal_wrapped(|ui| {
        ui.spacing_mut().item_spacing = egui::vec2(0.0, look.size * 0.2);
        // Every cell reserves the same rows, so a line's text sits level.
        let ra = row(ui, cells.iter().any(|c| c.above.is_some()), sa);
        let rb = row(ui, cells.iter().any(|c| c.below.is_some()), sb);
        let fallback = ui.visuals().text_color();
        for c in cells {
            if c.gap { ui.add_space(look.size * 0.3); }
            let colour = if c.lit { look.lit } else { look.colour };
            let rt = |t: &str, sz: f32, reading: bool| {
                let mut r = RichText::new(t).size(sz).color(colour);
                if look.strong { r = r.strong(); }
                if reading && look.guessed { r = r.italics(); }
                r
            };
            let main = galley(ui, rt(&c.text, look.size, false));
            let above = c.above.as_deref().map(|t| galley(ui, rt(t, sa, true)));
            let below = c.below.as_deref().map(|t| galley(ui, rt(t, sb, true)));
            let w = [Some(&main), above.as_ref(), below.as_ref()].into_iter().flatten()
                .map(|g| g.size().x).fold(0.0, f32::max);
            let mh = main.size().y;
            let (rect, _) = ui.allocate_exact_size(egui::vec2(w, ra + mh + rb), egui::Sense::hover());
            let at = |g: &std::sync::Arc<egui::Galley>, y: f32| egui::pos2(rect.center().x - g.size().x / 2.0, y);
            if let Some(g) = above { ui.painter().galley(at(&g, rect.top()), g, fallback); }
            ui.painter().galley(at(&main, rect.top() + ra), main, fallback);
            if let Some(g) = below { ui.painter().galley(at(&g, rect.top() + ra + mh), g, fallback); }
        }
    }).response
}

/// What one line shows, worked out without drawing anything, so what the
/// viewer draws and what the tests check are the same thing.
#[derive(Clone, Debug, PartialEq)]
enum Shown {
    Cells(Vec<Cell>),
    /// Japanese text, how far into it the singing is, and a reading under it.
    Text { text: String, upto: Option<usize>, under: Option<String> },
    /// A line of romaji on its own.
    Romaji(String),
}

/// Work out what a line shows in `script`.
///
/// `upto` is how far into the line's text as stored the singing has got.
/// `reading` is required for every script but `Original`.
///
/// Where the user has confirmed a reading over the sheet's own furigana, the
/// Japanese is shown **without** the sheet's bracketed annotation. Showing
/// `剣(つるぎ)` next to the reading that replaced つるぎ would contradict it.
/// The stored text is unchanged; only what is drawn is.
fn present(
    script: Script, per_word: bool, line: &LyricLine, words: &[Word],
    reading: Option<&super::romaji::Reading>, upto: Option<usize>,
) -> Shown {
    let Some(r) = reading.filter(|_| script != Script::Original) else {
        return Shown::Text { text: line.text.clone(), upto, under: None };
    };
    let over = line.overrides_sheet();
    let shown_word = |w: &Word| if over && super::romaji::has_sheet_furigana(w) {
        super::romaji::without_brackets(&w.text)
    } else {
        w.text.clone()
    };
    let mut start = 0;
    let starts: Vec<usize> = words.iter().map(|w| { let s = start; start += w.text.len(); s }).collect();
    let lit = |k: usize| upto.is_some_and(|u| starts[k] < u);
    let cells: Option<Vec<Cell>> = match (script, &r.words) {
        (Script::Furigana, _) => Some(match &r.furigana {
            super::romaji::Furigana::Words(per) => per.iter().enumerate()
                .flat_map(|(k, pieces)| pieces.iter().map(move |(text, above)| Cell {
                    text: text.clone(), above: above.clone(), below: None, lit: lit(k), gap: false,
                }))
                .collect(),
            // The reading could not be placed word by word: it goes over the
            // whole line, said plainly.
            super::romaji::Furigana::WholeLine { text, kana } => vec![Cell {
                text: text.clone(), above: Some(kana.clone()), below: None,
                lit: upto.is_some_and(|u| u > 0), gap: false,
            }],
        }),
        (Script::Romaji, Some(u)) if upto.is_some() =>
            Some(words.iter().enumerate().map(|(k, w)| Cell {
                text: if w.jp { u[k].clone() } else { w.text.clone() },
                above: None, below: None, lit: lit(k),
                gap: k > 0 && w.jp && words[k - 1].jp,
            }).collect()),
        (Script::Both, Some(u)) if per_word && r.line != line.text =>
            Some(words.iter().enumerate().map(|(k, w)| Cell {
                text: shown_word(w), above: None,
                below: w.jp.then(|| u[k].clone()), lit: lit(k), gap: k > 0,
            }).collect()),
        _ => None,
    };
    if let Some(c) = cells { return Shown::Cells(c); }
    if script == Script::Romaji { return Shown::Romaji(r.line.clone()); }
    // The line as text, with the reading under it. Karaoke's position is in
    // the stored text; carried into the shown text word by word.
    let under = (r.line != line.text).then(|| r.line.clone());
    if !over {
        return Shown::Text { text: line.text.clone(), upto, under };
    }
    let mut text = String::new();
    let mut mapped = upto.map(|_| 0);
    for (w, &raw) in words.iter().zip(&starts) {
        let shown = shown_word(w);
        if let Some(u) = upto && u > raw {
            mapped = Some(text.len() + (u - raw).min(shown.len()));
        }
        text.push_str(&shown);
    }
    let mapped = mapped.map(|m| {
        let mut m = m.min(text.len());
        while !text.is_char_boundary(m) { m -= 1; }
        m
    });
    Shown::Text { text, upto: mapped, under }
}

/// A line with the sung part in one colour and the rest in another.
fn sung_job(text: &str, upto: usize, size: f32, lit: egui::Color32, rest: egui::Color32)
    -> egui::text::LayoutJob
{
    let mut job = egui::text::LayoutJob::default();
    let f = |color| egui::TextFormat { font_id: egui::FontId::proportional(size), color, ..Default::default() };
    job.append(&text[..upto], 0.0, f(lit));
    job.append(&text[upto..], 0.0, f(rest));
    job
}

/// A reading the user entered that would displace furigana written into the
/// sheet, waiting for them to say so. Made once, when the editor's Done is
/// pressed — never from drawing — and bound to the line's text, so it cannot
/// land on some other line if the sheet changes first.
#[derive(Clone, Debug)]
struct PendingOverride {
    /// The sheet revision it was asked under. Identical lines share their
    /// text, so index and text alone cannot tell the first occurrence from a
    /// second that an edit moved into its place. Any change to the lines makes
    /// the question stale.
    rev: u64,
    line: usize,
    text: String,
    romaji: String,
    conflicts: Vec<super::romaji::Conflict>,
}

/// One line in the romaji editor.
#[derive(Clone, Debug, Default)]
struct RomajiRow {
    text: String,
    /// What the row opened with, to tell an edit from a look.
    start: String,
    /// The reading the line had on opening, still in force until the row is
    /// edited or reset. A reading a source supplied stays supplied even when
    /// it happens to match the dictionary's.
    keep: Option<String>,
    /// Where the supplied reading disagrees with furigana the sheet wrote,
    /// the words it disagrees on. Worked out once, when the editor opens.
    conflicts: Vec<super::romaji::Conflict>,
    /// The user ticked "use this reading instead of the lyrics'". Their
    /// reading is otherwise not taken over the sheet's, however it got there.
    affirm: bool,
}

/// What to tell the user about a save, and whether it succeeded. Each of the
/// three outcomes is reported as what it is, with the backups named.
fn describe(outcome: super::SaveOutcome) -> (bool, String) {
    let name = |p: &PathBuf| p.file_name().unwrap_or_default().to_string_lossy().into_owned();
    let names = |ps: &[PathBuf]| ps.iter().map(name).collect::<Vec<_>>().join(", ");
    match outcome {
        super::SaveOutcome::Saved { written, removed, backups } => {
            let mut s = if written.is_empty() && removed.is_empty() {
                "Saved — the files already held this.".to_string()
            } else {
                format!("Saved {}", names(&written))
            };
            if !removed.is_empty() {
                s.push_str(&format!(" · removed {} (it no longer matched the sheet)", names(&removed)));
            }
            if !backups.is_empty() {
                s.push_str(&format!(" · previous versions kept as {}", names(&backups)));
            }
            (true, s)
        }
        super::SaveOutcome::RolledBack { error } => {
            (false, format!("Could not save — {error}. Nothing on disk was changed."))
        }
        super::SaveOutcome::NotUndone { error, stuck, backups } => {
            let stuck: Vec<PathBuf> = stuck.into_iter().map(|s| s.0).collect();
            (false, format!(
                "Could not save — {error} — and could not put back {}, which now hold the \
                 new version. Every previous version is in {}. Save again to finish.",
                names(&stuck), names(&backups)))
        }
    }
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
    /// Whole lines, so retiming a sheet keeps its romaji, translation and
    /// word timing rather than rebuilding lines from time and text alone.
    edit: Vec<LyricLine>,
    /// Next line the Stamp key will time.
    cursor: usize,
    dirty: bool,

    // ── Romaji ─────────────────────────────────────────────────────────────
    script: Script,
    /// In `Both`, romaji under each word rather than under the line.
    per_word: bool,
    /// Light the current line as it is sung.
    karaoke: bool,
    show_translation: bool,
    auto: Vec<Auto>,
    romaji_editing: bool,
    romaji_edit: Vec<RomajiRow>,
    /// Readings awaiting confirmation that they replace the sheet's furigana.
    pending: Vec<PendingOverride>,
    /// Bumped by every change to the sheet's lines — load, fetch, paste, a
    /// Sync commit — so a question asked about one version of the lines is
    /// never answered on another.
    sheet_rev: u64,
    romaji_dirty: bool,

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
            script: Script::default(),
            per_word: true,
            karaoke: true,
            show_translation: true,
            auto: Vec::new(),
            romaji_editing: false,
            romaji_edit: Vec::new(),
            pending: Vec::new(),
            sheet_rev: 0,
            romaji_dirty: false,
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
        self.romaji_editing = false;
        self.romaji_dirty = false;
        self.pending.clear();
        self.sheet_rev += 1;
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
        self.pending.clear();
        self.sheet_rev += 1;
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
                let mut l = super::parse(text, LyricSource::Online);
                if let Some(r) = &hit.romaji { super::attach(&mut l.lines, r, Extra::Romaji); }
                if let Some(t) = &hit.translation { super::attach(&mut l.lines, t, Extra::Translation); }
                self.status = if l.is_synced() {
                    format!("{} — {} · via {}", hit.title, hit.artist, hit.source)
                } else {
                    format!("{} — {} · via {} (plain text; “Sync…” can time it)",
                            hit.title, hit.artist, hit.source)
                };
                self.lyrics = Some(l);
                self.dirty = true;
                self.pending.clear();
                self.sheet_rev += 1;
                self.editing = false;
                self.romaji_editing = false;
            }
            None => self.status = "That result has no lyrics.".into(),
        }
    }

    /// Write whichever of the two sidecars has changed.
    ///
    /// The romaji file mirrors the sheet line for line, so it is rewritten
    /// whenever the sheet is and has any romaji in it — otherwise a retimed
    /// sheet would leave it pairing against lines that no longer exist.
    fn save(&mut self, track: &TrackRef) {
        let Some(l) = &self.lyrics else { return };
        let (saved, status) = describe(super::save_all(track.path, l, self.dirty, self.romaji_dirty));
        self.status = status;
        // The unsaved flags clear only on success, so a failed Save can simply
        // be pressed again.
        if saved {
            if self.dirty && let Some(ly) = &mut self.lyrics { ly.source = LyricSource::Sidecar; }
            self.dirty = false;
            self.romaji_dirty = false;
        }
    }

    /// Open the romaji editor, each line prefilled with what the viewer shows.
    fn begin_romaji_edit(&mut self) {
        let Some(l) = &self.lyrics else { return };
        self.romaji_edit = l.lines.iter().enumerate()
            .map(|(i, x)| {
                let a = auto_for(&mut self.auto, i, &x.text);
                let text = x.romaji.clone().unwrap_or_else(|| a.line.clone());
                let conflicts = match &x.romaji {
                    Some(r) if !x.overrides_sheet() => super::romaji::conflicts(&a.words, r),
                    _ => Vec::new(),
                };
                RomajiRow { start: text.clone(), text, keep: x.romaji.clone(), conflicts, affirm: false }
            })
            .collect();
        self.romaji_editing = true;
        self.status = "Fix any reading that is wrong. Lines you change are kept.".into();
    }

    /// Where the romaji came from survives the editor. A row left alone keeps
    /// what it had — a supplied reading stays supplied, a generated one stays
    /// generated. A row that was edited becomes a correction, unless it was
    /// blanked or put back to the generated text, which return it to
    /// generated; so does ↺, which also drops a supplied reading.
    fn commit_romaji_edit(&mut self) {
        self.romaji_editing = false;
        self.pending.clear();
        let Some(l) = &mut self.lyrics else { return };
        let mut changed = false;
        let mut pending = Vec::new();
        for (i, (line, row)) in l.lines.iter_mut().zip(&self.romaji_edit).enumerate() {
            let t = row.text.trim();
            let unchanged = row.text == row.start;
            let new = if unchanged {
                row.keep.clone()
            } else {
                let auto = &auto_for(&mut self.auto, i, &line.text).line;
                (!t.is_empty() && t != auto).then(|| t.to_string())
            };
            // A reading that would displace furigana the sheet wrote is not
            // taken until the user confirms it. A line already overridden was
            // confirmed once and is not asked about again.
            // Unchanged, a supplied reading that disagrees with the sheet is
            // still only offered, unless the user ticked to use it — then it
            // is asked about exactly as an edited one would be.
            let affirmed = row.affirm && !line.overrides_sheet();
            if (!unchanged || affirmed) && !line.overrides_sheet() && let Some(r) = &new {
                let c = super::romaji::conflicts(&auto_for(&mut self.auto, i, &line.text).words, r);
                if !c.is_empty() {
                    pending.push(PendingOverride {
                        rev: self.sheet_rev, line: i, text: line.text.clone(), romaji: r.clone(),
                        conflicts: c,
                    });
                    continue;
                }
            }
            let over = if new.is_some() && line.overrides_sheet() {
                line.reading_override.clone()
            } else {
                None
            };
            if new != line.romaji || over != line.reading_override {
                line.romaji = new;
                line.reading_override = over;
                changed = true;
            }
        }
        if changed {
            self.romaji_dirty = true;
        }
        self.status = if !pending.is_empty() {
            "Your reading differs from the furigana written in the lyrics — confirm or keep the sheet's.".into()
        } else if changed {
            "Romaji edited — Save to keep it.".into()
        } else {
            String::new()
        };
        self.pending = pending;
    }

    /// Answer the confirmation: use the user's readings over the sheet's
    /// furigana, or keep the sheet's. Keeping leaves those lines exactly as
    /// they were before the edit.
    fn resolve_pending(&mut self, use_mine: bool) {
        let pending = std::mem::take(&mut self.pending);
        if !use_mine {
            self.status = "Kept the reading written in the lyrics.".into();
            return;
        }
        let Some(l) = &mut self.lyrics else { return };
        let (mut n, mut stale) = (0, 0);
        for p in pending {
            // Only onto the line it was asked about, in the version of the
            // sheet it was asked about.
            if p.rev != self.sheet_rev {
                stale += 1;
                continue;
            }
            if let Some(line) = l.lines.get_mut(p.line) && line.text == p.text {
                line.romaji = Some(p.romaji);
                line.reading_override = Some(p.text);
                n += 1;
            }
        }
        if n > 0 { self.romaji_dirty = true; }
        self.status = if stale > 0 && n == 0 {
            "The lyrics changed since you were asked, so nothing was applied — open ✎ to choose again.".into()
        } else {
            format!("Your reading now replaces the lyrics' own on {n} line{} — Save to keep it.",
                    if n == 1 { "" } else { "s" })
        };
    }

    fn pending_panel(&mut self, ui: &mut egui::Ui) {
        let dark = ui.visuals().dark_mode;
        let mut answer = None;
        ui.group(|ui| {
            ui.label(RichText::new("Your reading differs from the furigana written in the lyrics")
                     .size(13.0).color(pal::text(dark)));
            for p in &self.pending {
                for c in &p.conflicts {
                    ui.label(RichText::new(format!("{} — lyrics: {} · yours: {}", c.word, c.sheet, c.proposed))
                             .size(12.5).color(pal::text_dim(dark)));
                }
            }
            ui.horizontal(|ui| {
                if ui.button("Use my reading").clicked() { answer = Some(true); }
                if ui.button("Keep the lyrics' reading").clicked() { answer = Some(false); }
            });
        });
        ui.add_space(4.0);
        if let Some(a) = answer { self.resolve_pending(a); }
    }

    /// Move the current sheet into the editor. Lines keep whatever timing they
    /// already had, so this re-times an existing sheet as happily as it times a
    /// plain one.
    fn begin_edit(&mut self) {
        let lines = self.lyrics.as_ref().map(|l| l.lines.clone()).unwrap_or_default();
        self.edit = lines;
        if self.edit.is_empty() { self.edit.push(LyricLine::default()); }
        // Resume at the first untimed line rather than the top, so a sync that
        // was interrupted picks up where it stopped.
        self.cursor = self.edit.iter().position(|l| l.at.is_none()).unwrap_or(0);
        // A pending confirmation is about lines this editor can delete,
        // reorder and retime; it is not carried across.
        self.pending.clear();
        self.editing = true;
        self.status = "Play the track and press Space (or ⏱) as each line starts.".into();
    }

    fn commit_edit(&mut self) {
        let mut lines: Vec<LyricLine> = self.edit.iter()
            .filter(|l| l.at.is_some() || !l.text.trim().is_empty())
            .cloned()
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
        self.sheet_rev += 1;
        self.pending.clear();
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
                    if self.dirty || self.romaji_dirty {
                        ui.label(RichText::new("• unsaved").size(11.5).color(pal::amber(dark)));
                    }
                });
                ui.add_space(3.0);
            });
            egui::CentralPanel::default().show(vctx, |ui| {
                if self.search_open { self.search_panel(ui); }
                if !self.pending.is_empty() { self.pending_panel(ui); }
                if self.paste_open { self.paste_panel(ui); }
                if self.editing {
                    action = self.editor(ui, pos).max_or(action);
                } else if self.romaji_editing {
                    self.romaji_editor(ui);
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
            if ui.add_enabled(idle && !self.editing && !self.romaji_editing, egui::Button::new("🔍 Find lyrics"))
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
            if ui.add_enabled(!self.editing && !self.romaji_editing, egui::Button::new("📋 Paste…"))
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

            if self.editing || self.romaji_editing {
                if ui.button("✔ Done").clicked() {
                    if self.editing { self.commit_edit(); } else { self.commit_romaji_edit(); }
                }
                if ui.button("✖ Cancel").clicked() {
                    self.editing = false;
                    self.romaji_editing = false;
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
                if ui.add_enabled(has && (self.dirty || self.romaji_dirty),
                                  egui::Button::new("💾 Save"))
                    .on_hover_text(format!(
                        "Write {}\n\nA sidecar file — the audio file is never modified. \
                         Corrected romaji goes to {} beside it.",
                        super::sidecar_path(track.path).file_name()
                            .unwrap_or_default().to_string_lossy(),
                        super::extra_path(track.path, Extra::Romaji).file_name()
                            .unwrap_or_default().to_string_lossy()))
                    .clicked()
                {
                    self.save(track);
                }
            }

            if self.is_japanese() && !self.editing && !self.romaji_editing {
                ui.separator();
                for (s, label) in [(Script::Original, "Original"), (Script::Furigana, "Furigana"),
                                   (Script::Romaji, "Romaji"), (Script::Both, "Both")] {
                    ui.selectable_value(&mut self.script, s, label);
                }
                if self.script == Script::Both {
                    ui.label(RichText::new("·").color(pal::text_faint(ui.visuals().dark_mode)));
                    ui.selectable_value(&mut self.per_word, false, "Lines")
                        .on_hover_text("Romaji under each line");
                    ui.selectable_value(&mut self.per_word, true, "Words")
                        .on_hover_text("Romaji under each word, so you can see which \
                                        sound belongs to which word");
                }
                if ui.button("✎").on_hover_text(
                    "Correct the romaji.\n\nRomaji the lyrics source supplied is used as it \
                     is; the rest is generated from a dictionary and shown in italics, \
                     because a song's reading of a kanji is often not the dictionary's.")
                    .clicked()
                {
                    self.begin_romaji_edit();
                }
            }

            if !self.editing && !self.romaji_editing {
                // Only offered where the sheet times its words.
                if self.lyrics.as_ref().is_some_and(|l| l.has_word_timing()) {
                    ui.separator();
                    ui.checkbox(&mut self.karaoke, "Karaoke")
                        .on_hover_text("Light each word as it is sung, from the word timing in the sheet.");
                }
                if self.lyrics.as_ref().is_some_and(|l| l.lines.iter().any(|x| x.translation.is_some())) {
                    ui.checkbox(&mut self.show_translation, "Translation")
                        .on_hover_text("The translation the lyrics source had, under each line.");
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
        let script = if l.lines.iter().any(|x| super::romaji::has_kana(&x.text)) {
            self.script
        } else {
            Script::Original
        };
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
                // How far the singing has got, on the current line only.
                let upto = (is_active && self.karaoke)
                    .then(|| l.sung(i, pos)).flatten()
                    .filter(|&u| line.text.is_char_boundary(u));
                let mut look = Look {
                    size, colour, lit: colour, strong: is_active, guessed: false,
                };
                if upto.is_some() { look.colour = pal::text(dark); }

                // Not generated unless shown: the dictionary loads on first use.
                let (words, reading) = if script == Script::Original {
                    (Vec::new(), None)
                } else {
                    let a = auto_for(&mut self.auto, i, &line.text);
                    let r = a.reading(line.romaji.as_deref(), line.overrides_sheet());
                    (a.words.clone(), Some(r))
                };
                look.guessed = reading.as_ref().is_some_and(|r| r.guessed);
                let text_label = |ui: &mut egui::Ui, look: &Look, text: &str, upto: Option<usize>| match upto {
                    Some(u) => ui.add(egui::Label::new(
                        sung_job(text, u, look.size, look.lit, look.colour))),
                    None => {
                        let mut rt = RichText::new(text).size(look.size).color(look.colour);
                        if look.strong { rt = rt.strong(); }
                        ui.add(egui::Label::new(rt).sense(egui::Sense::hover()))
                    }
                };
                let mut resp = match present(script, self.per_word, line, &words, reading.as_ref(), upto) {
                    Shown::Cells(c) => draw_cells(ui, &c, &look),
                    Shown::Romaji(romaji) => {
                        let mut rt = RichText::new(&romaji).size(size).color(colour);
                        if is_active { rt = rt.strong(); }
                        if look.guessed { rt = rt.italics(); }
                        ui.add(egui::Label::new(rt).sense(egui::Sense::hover()))
                    }
                    Shown::Text { text, upto, under } => {
                        let mut resp = text_label(ui, &look, &text, upto);
                        if let Some(romaji) = under {
                            ui.add_space(-2.0);
                            let mut rt = RichText::new(&romaji).size(size * 0.72).color(colour);
                            if look.guessed { rt = rt.italics(); }
                            resp = resp.union(ui.add(egui::Label::new(rt)));
                        }
                        resp
                    }
                };
                if script != Script::Original && look.guessed {
                    resp = resp.on_hover_text("Generated reading — may misread a kanji. ✎ corrects it.");
                }
                if self.show_translation && let Some(t) = &line.translation {
                    ui.add_space(-1.0);
                    ui.label(RichText::new(t).size(size * 0.8).color(pal::text_faint(dark)));
                }
                if is_active && self.auto_scroll {
                    resp.scroll_to_me(Some(egui::Align::Center));
                }
                ui.add_space(3.0);
            }
            ui.add_space(ui.available_height().max(80.0));
        });
    }

    fn is_japanese(&self) -> bool {
        self.lyrics.as_ref()
            .is_some_and(|l| l.lines.iter().any(|x| super::romaji::has_kana(&x.text)))
    }

    fn romaji_editor(&mut self, ui: &mut egui::Ui) {
        let dark = ui.visuals().dark_mode;
        let Some(l) = &self.lyrics else { return };
        egui::ScrollArea::vertical().id_salt("lyr_romaji")
            .auto_shrink([false, false]).show(ui, |ui| {
            for (i, line) in l.lines.iter().enumerate() {
                if line.text.trim().is_empty() { continue; }
                ui.label(RichText::new(&line.text).size(12.5).color(pal::text_faint(dark)));
                let auto = auto_for(&mut self.auto, i, &line.text).line.clone();
                ui.horizontal(|ui| {
                    let Some(row) = self.romaji_edit.get_mut(i) else { return };
                    ui.add(egui::TextEdit::singleline(&mut row.text)
                           .desired_width(ui.available_width() - 30.0)
                           .hint_text(&auto)
                           .font(egui::FontId::proportional(self.font_size.min(16.0))));
                    // Offered for a supplied reading too, even one that reads
                    // the same as the generated text: resetting it is how it
                    // stops being supplied.
                    if row.text.trim() != auto || row.keep.is_some() {
                        if ui.small_button("↺").on_hover_text(format!("Back to generated “{auto}”")).clicked() {
                            row.text = auto.clone();
                            row.keep = None;
                            row.affirm = false;
                        }
                    } else {
                        ui.label(RichText::new("auto").size(10.5).italics()
                                 .color(pal::text_faint(dark)));
                    }
                });
                // A supplied reading that disagrees with the lyrics' own
                // furigana: shown, and the lyrics' reading stays unless the
                // user ticks to use this one — and even then they are asked.
                if let Some(row) = self.romaji_edit.get_mut(i) && !row.conflicts.is_empty() {
                    for c in &row.conflicts {
                        ui.label(RichText::new(format!(
                            "{} — lyrics read {}, this reading says {}; the lyrics' reading is used",
                            c.word, c.sheet, c.proposed))
                            .size(11.0).color(pal::text_dim(dark)));
                    }
                    ui.checkbox(&mut row.affirm, "Use this reading instead of the lyrics'")
                        .on_hover_text("You will be asked to confirm when you press Done.");
                }
                ui.add_space(4.0);
            }
            ui.add_space(60.0);
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
                self.edit[self.cursor].at = Some(pos);
                self.cursor += 1;
            }
            if ui.button("↩ Back").on_hover_text("Undo the last stamp").clicked()
                && self.cursor > 0
            {
                self.cursor -= 1;
                self.edit[self.cursor].at = None;
            }
            if ui.button("Clear all times").clicked() {
                for e in &mut self.edit { e.at = None; }
                self.cursor = 0;
            }
            if ui.button("+ Line").clicked() {
                self.edit.push(LyricLine::default());
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
                    let stamp_txt = match self.edit[i].at {
                        Some(t) => super::fmt_stamp(t),
                        None => "  --:--  ".into(),
                    };
                    let btn = egui::Button::new(
                        RichText::new(stamp_txt).monospace().size(12.0)
                            .color(if self.edit[i].at.is_some() { pal::ok(dark) }
                                   else { pal::text_faint(dark) }));
                    if ui.add(btn).on_hover_text("Jump the player here").clicked()
                        && let Some(t) = self.edit[i].at
                    {
                        action = LyricsAction::Seek(t);
                    }
                    if is_cursor {
                        ui.label(RichText::new("▶").size(12.0).color(pal::accent(dark)));
                    }
                    let e = &mut self.edit[i];
                    // Word timing points into the text; once the text is
                    // retyped it points at the wrong letters.
                    if ui.add(egui::TextEdit::singleline(&mut e.text)
                           .desired_width(ui.available_width() - 30.0)
                           .font(egui::FontId::proportional(self.font_size.min(16.0)))).changed()
                    {
                        e.marks.clear();
                    }
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
            w.edit[i].at = Some(Duration::from_secs(*secs));
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
        w.edit[0].at = Some(Duration::from_secs(9));
        w.edit[1].at = Some(Duration::from_secs(2));
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
        w.edit.push(LyricLine { text: "   ".into(), ..Default::default() });
        w.edit.push(LyricLine::default());
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

    /// A correction is kept; a line left as generated stays generated, so
    /// saving never records a guess as though someone had checked it.
    #[test]
    fn only_changed_romaji_counts_as_a_correction() {
        let mut w = win_with("[00:01.00]本気\n[00:02.00]心\n");
        w.begin_romaji_edit();
        let auto1 = w.romaji_edit[1].text.clone();
        w.romaji_edit[0].text = "maji".into();
        w.commit_romaji_edit();
        let l = w.lyrics.as_ref().unwrap();
        assert_eq!(l.lines[0].romaji.as_deref(), Some("maji"));
        assert_eq!(l.lines[1].romaji, None, "untouched ({auto1}) stays generated");
        assert!(w.romaji_dirty);

        // Putting it back, or blanking it, returns it to generated.
        w.begin_romaji_edit();
        w.romaji_edit[0].text.clear();
        w.commit_romaji_edit();
        assert_eq!(w.lyrics.as_ref().unwrap().lines[0].romaji, None);
    }

    /// Provenance: a reading a source supplied is kept through the editor
    /// even when it matches the dictionary's, a generated one stays generated,
    /// and resetting or clearing drops the supplied one. Then through a save.
    #[test]
    fn the_editor_keeps_where_a_reading_came_from() {
        let mut w = win_with("[00:01.00]心\n[00:02.00]空\n[00:03.00]海\n[00:04.00]山\n");
        let generated: Vec<String> = (0..4)
            .map(|i| auto_for(&mut w.auto, i, &w.lyrics.as_ref().unwrap().lines[i].text).line.clone())
            .collect();
        {
            let l = w.lyrics.as_mut().unwrap();
            // Supplied, and identical to what the dictionary would say.
            l.lines[0].romaji = Some(generated[0].clone());
            l.lines[2].romaji = Some(generated[2].clone());
            l.lines[3].romaji = Some("yamaa".into());
        }

        // Open and Done, touching nothing: nothing changes.
        w.begin_romaji_edit();
        w.commit_romaji_edit();
        let r: Vec<_> = w.lyrics.as_ref().unwrap().lines.iter().map(|x| x.romaji.clone()).collect();
        assert_eq!(r, vec![Some(generated[0].clone()), None, Some(generated[2].clone()), Some("yamaa".into())]);
        assert!(!w.romaji_dirty);

        // Reset (↺) on a supplied reading equal to the generated one, and
        // clearing a supplied one, both drop it.
        w.begin_romaji_edit();
        w.romaji_edit[2].text = generated[2].clone();
        w.romaji_edit[2].keep = None;
        w.romaji_edit[3].text.clear();
        w.commit_romaji_edit();
        let r: Vec<_> = w.lyrics.as_ref().unwrap().lines.iter().map(|x| x.romaji.clone()).collect();
        assert_eq!(r, vec![Some(generated[0].clone()), None, None, None]);
        assert!(w.romaji_dirty);

        // And the supplied reading that survived is still supplied after a
        // save and a reload.
        let dir = std::env::temp_dir().join(format!("moosik_prov_{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let track = dir.join("song.flac");
        assert!(matches!(super::super::save_all(&track, w.lyrics.as_ref().unwrap(), true, true),
                         super::super::SaveOutcome::Saved { .. }));
        let back = super::super::load(&track, None).unwrap();
        let _ = std::fs::remove_dir_all(&dir);
        let r: Vec<_> = back.lines.iter().map(|x| x.romaji.clone()).collect();
        assert_eq!(r, vec![Some(generated[0].clone()), None, None, None]);
    }

    #[test]
    fn opening_and_closing_the_romaji_editor_changes_nothing() {
        let mut w = win_with("[00:01.00]心\n");
        w.begin_romaji_edit();
        w.commit_romaji_edit();
        assert!(!w.romaji_dirty);
        assert_eq!(w.lyrics.unwrap().lines[0].romaji, None);
    }

    /// Retiming a sheet must not drop the corrections made to its romaji.
    #[test]
    fn syncing_keeps_corrected_romaji() {
        let mut w = win_with("本気\n心\n");
        w.lyrics.as_mut().unwrap().lines[0].romaji = Some("maji".into());
        w.begin_edit();
        w.edit[0].at = Some(Duration::from_secs(4));
        w.edit[1].at = Some(Duration::from_secs(2));
        w.commit_edit();
        let l = w.lyrics.unwrap();
        assert_eq!(l.lines[1].text, "本気");
        assert_eq!(l.lines[1].romaji.as_deref(), Some("maji"));
    }

    /// Retiming carries everything else on the line with it; word timing is
    /// relative to the line, so it moves with the new time.
    #[test]
    fn syncing_keeps_word_timing_and_translation() {
        let mut w = win_with("[00:10.00]<00:10.00>ka<00:10.50>ra\n");
        w.lyrics.as_mut().unwrap().lines[0].translation = Some("x".into());
        w.begin_edit();
        w.edit[0].at = Some(Duration::from_secs(20));
        w.commit_edit();
        let l = w.lyrics.unwrap();
        assert_eq!(l.lines[0].translation.as_deref(), Some("x"));
        assert_eq!(l.sung(0, Duration::from_millis(20_100)), Some(2));
        assert!(l.to_lrc().contains("[00:20.00]<00:20.00>ka<00:20.50>ra"), "{}", l.to_lrc());
    }

    #[test]
    fn a_hit_brings_its_translation() {
        let mut w = LyricsWindow::default();
        w.apply(&Hit {
            synced: Some("[00:01.00]心\n[00:02.00]空".into()),
            translation: Some("[by:someone]\n[00:02.000]sky".into()),
            ..Default::default()
        });
        let l = w.lyrics.as_ref().unwrap();
        assert_eq!(l.lines[0].translation, None);
        assert_eq!(l.lines[1].translation.as_deref(), Some("sky"));
    }

    /// A scratch folder holding a track's sidecars, removed on drop. The track
    /// itself need not exist: nothing here reads audio.
    struct Scratch(std::path::PathBuf);
    impl Scratch {
        fn new(tag: &str) -> Self {
            let d = std::env::temp_dir().join(format!(
                "moosik_ui_{tag}_{}_{:?}", std::process::id(), std::thread::current().id()));
            let _ = std::fs::remove_dir_all(&d);
            std::fs::create_dir_all(&d).unwrap();
            Scratch(d)
        }
        fn track(&self) -> std::path::PathBuf { self.0.join("song.flac") }
        fn put(&self, name: &str, text: &str) { std::fs::write(self.0.join(name), text).unwrap(); }
        fn read(&self, name: &str) -> Option<String> { std::fs::read_to_string(self.0.join(name)).ok() }
    }
    impl Drop for Scratch {
        fn drop(&mut self) { let _ = std::fs::remove_dir_all(&self.0); }
    }
    fn track_ref(p: &std::path::Path) -> TrackRef<'_> {
        TrackRef { path: p, title: "t", artist: "a", album: "", duration: None }
    }

    /// Through the production path: load, Sync editor, Done, Save, reload.
    /// Line A carries romaji and a translation; A is deleted and B retimed to
    /// A's old time. The old extras pair by timestamp, so unless the save deals
    /// with them they attach to B on reload. They must not — and they must not
    /// be lost either.
    #[test]
    fn a_sync_edit_onto_an_annotated_time_does_not_inherit_its_extras() {
        let s = Scratch::new("sync");
        let track = s.track();
        let (romaji, trans) = ("[00:01.00]kokoro\n[00:05.00]\n", "[00:01.00]heart\n[00:05.00]\n");
        s.put("song.lrc", "[00:01.00]心\n[00:05.00]空\n");
        s.put("song.romaji.lrc", romaji);
        s.put("song.translation.lrc", trans);
        let tr = track_ref(&track);

        let mut w = LyricsWindow::default();
        w.load_local(&tr);
        let l = w.lyrics.as_ref().unwrap();
        assert_eq!(l.lines[0].romaji.as_deref(), Some("kokoro"), "precondition");
        assert_eq!(l.lines[0].translation.as_deref(), Some("heart"), "precondition");

        w.begin_edit();
        w.edit.remove(0);                                   // the editor's ✖ on line A
        w.edit[0].at = Some(Duration::from_secs(1));        // B stamped at A's old time
        w.commit_edit();
        w.save(&tr);
        assert!(!w.dirty, "the save reported: {}", w.status);

        let back = super::super::load(&track, None).unwrap();
        assert_eq!(back.lines.len(), 1);
        assert_eq!(back.lines[0].text, "空");
        assert_eq!(back.lines[0].romaji, None, "A's romaji attached to B");
        assert_eq!(back.lines[0].translation, None, "A's translation attached to B");
        // Recoverable, byte for byte, and the user is told where.
        assert_eq!(s.read("song.romaji.lrc.old").as_deref(), Some(romaji));
        assert_eq!(s.read("song.translation.lrc.old").as_deref(), Some(trans));
        assert!(w.status.contains("song.romaji.lrc.old"), "{}", w.status);
    }

    /// A Save that fails is reported as failed, leaves the sheet unsaved so it
    /// can be retried, and says that nothing on disk changed.
    #[test]
    fn a_failed_save_is_reported_and_can_be_retried() {
        let s = Scratch::new("fail");
        let track = s.track();
        // A folder where song.lrc should go makes the write fail.
        std::fs::create_dir(s.0.join("song.lrc")).unwrap();
        let tr = track_ref(&track);
        let mut w = win_with("[00:01.00]心\n");
        w.dirty = true;
        w.save(&tr);
        assert!(w.dirty, "a failed save must not clear the unsaved flag");
        assert!(w.status.starts_with("Could not save") && w.status.contains("Nothing on disk was changed"),
                "{}", w.status);
        std::fs::remove_dir(s.0.join("song.lrc")).unwrap();
        w.save(&tr);
        assert!(!w.dirty, "{}", w.status);
    }

    /// Every outcome's message names what matters: what was written or
    /// removed, where the previous versions are, and which files a failed
    /// rollback left holding the new version.
    #[test]
    fn every_save_outcome_is_described_as_it_is() {
        use super::super::SaveOutcome;
        let pb = |s: &str| std::path::PathBuf::from(s);
        let (ok, m) = describe(SaveOutcome::Saved {
            written: vec![pb("d/song.lrc")], removed: vec![pb("d/song.translation.lrc")],
            backups: vec![pb("d/song.translation.lrc.old")],
        });
        assert!(ok);
        assert!(m.contains("song.lrc") && m.contains("removed song.translation.lrc")
                && m.contains("song.translation.lrc.old"), "{m}");
        let (ok, m) = describe(SaveOutcome::RolledBack { error: std::io::Error::other("disk full") });
        assert!(!ok && m.contains("disk full") && m.contains("Nothing on disk was changed"), "{m}");
        let (ok, m) = describe(SaveOutcome::NotUndone {
            error: std::io::Error::other("disk full"),
            stuck: vec![(pb("d/song.lrc"), std::io::Error::other("locked"))],
            backups: vec![pb("d/song.lrc.old"), pb("d/song.romaji.lrc.old")],
        });
        assert!(!ok && m.contains("could not put back song.lrc")
                && m.contains("song.lrc.old") && m.contains("song.romaji.lrc.old"), "{m}");
    }

    // ── Readings over the sheet's own furigana ─────────────────────────────

    fn reading_of(w: &mut LyricsWindow, i: usize) -> super::super::romaji::Reading {
        let line = w.lyrics.as_ref().unwrap().lines[i].clone();
        auto_for(&mut w.auto, i, &line.text).reading(line.romaji.as_deref(), line.overrides_sheet())
    }
    fn furigana_first(r: &super::super::romaji::Reading) -> (String, Option<String>) {
        match &r.furigana {
            super::super::romaji::Furigana::Words(w) => w[0][0].clone(),
            super::super::romaji::Furigana::WholeLine { text, kana } => (text.clone(), Some(kana.clone())),
        }
    }
    const ANNOTATED: &str = "[00:01.00]剣(つるぎ)を\n";

    /// By default, and with a source's differing romaji, the sheet reads.
    #[test]
    fn the_sheets_furigana_reads_by_default_and_over_a_source() {
        let mut w = win_with(ANNOTATED);
        assert_eq!(furigana_first(&reading_of(&mut w, 0)), ("剣".into(), Some("つるぎ".into())));
        w.lyrics.as_mut().unwrap().lines[0].romaji = Some("ken o".into()); // from a source
        let r = reading_of(&mut w, 0);
        assert_eq!(r.line, "tsurugi o");
        assert_eq!(furigana_first(&r), ("剣".into(), Some("つるぎ".into())));
        // Opening and closing the editor on a source reading asks nothing.
        w.begin_romaji_edit();
        w.commit_romaji_edit();
        assert!(w.pending.is_empty());
    }

    /// Entering a conflicting reading asks first; confirming makes it win in
    /// every view. It is asked once, not again on the next Done.
    #[test]
    fn a_conflicting_reading_is_confirmed_before_it_wins() {
        let mut w = win_with(ANNOTATED);
        w.begin_romaji_edit();
        w.romaji_edit[0].text = "ken o".into();
        w.commit_romaji_edit();
        assert_eq!(w.pending.len(), 1);
        assert_eq!(w.pending[0].conflicts[0].sheet, "つるぎ");
        assert_eq!(w.pending[0].conflicts[0].proposed, "けん");
        assert_eq!(w.lyrics.as_ref().unwrap().lines[0].romaji, None, "not taken before confirming");
        w.resolve_pending(true);
        let l = &w.lyrics.as_ref().unwrap().lines[0];
        assert!(l.overrides_sheet());
        assert_eq!(l.text, "剣(つるぎ)を", "the sheet's own text is not touched");
        let r = reading_of(&mut w, 0);
        assert_eq!(r.line, "ken o");
        assert_eq!(furigana_first(&r), ("剣".into(), Some("けん".into())));
        // Done again without a change: nothing to confirm.
        w.begin_romaji_edit();
        w.commit_romaji_edit();
        assert!(w.pending.is_empty());
        assert!(w.lyrics.as_ref().unwrap().lines[0].overrides_sheet());
    }

    #[test]
    fn cancelling_keeps_the_sheets_reading() {
        let mut w = win_with(ANNOTATED);
        w.begin_romaji_edit();
        w.romaji_edit[0].text = "ken o".into();
        w.commit_romaji_edit();
        w.resolve_pending(false);
        let l = &w.lyrics.as_ref().unwrap().lines[0];
        assert_eq!(l.romaji, None);
        assert!(!l.overrides_sheet());
        assert_eq!(furigana_first(&reading_of(&mut w, 0)), ("剣".into(), Some("つるぎ".into())));
    }

    /// Reset (↺) removes the override and the sheet reads again.
    #[test]
    fn reset_removes_the_override() {
        let mut w = win_with(ANNOTATED);
        w.begin_romaji_edit();
        w.romaji_edit[0].text = "ken o".into();
        w.commit_romaji_edit();
        w.resolve_pending(true);
        w.begin_romaji_edit();
        let auto = auto_for(&mut w.auto, 0, "剣(つるぎ)を").line.clone();
        w.romaji_edit[0].text = auto;
        w.romaji_edit[0].keep = None;
        w.commit_romaji_edit();
        let l = &w.lyrics.as_ref().unwrap().lines[0];
        assert_eq!((l.romaji.clone(), l.reading_override.clone()), (None, None));
        assert_eq!(furigana_first(&reading_of(&mut w, 0)), ("剣".into(), Some("つるぎ".into())));
    }

    /// An override that cannot be placed word by word is shown over the whole
    /// line, not quietly as the old per-word furigana.
    #[test]
    fn an_unplaceable_override_is_shown_over_the_whole_line() {
        let mut w = win_with(ANNOTATED);
        w.begin_romaji_edit();
        w.romaji_edit[0].text = "zenzen chigau uta desu yo".into();
        w.commit_romaji_edit();
        assert_eq!(w.pending.len(), 1);
        w.resolve_pending(true);
        let r = reading_of(&mut w, 0);
        assert!(matches!(r.furigana, super::super::romaji::Furigana::WholeLine { .. }), "{r:?}");
        assert_eq!(r.line, "zenzen chigau uta desu yo");
    }

    /// Confirmed, saved, reloaded: still the user's reading.
    #[test]
    fn an_override_survives_save_and_reload() {
        let s = Scratch::new("override");
        let track = s.track();
        s.put("song.lrc", ANNOTATED);
        let tr = track_ref(&track);
        let mut w = LyricsWindow::default();
        w.load_local(&tr);
        w.begin_romaji_edit();
        w.romaji_edit[0].text = "ken o".into();
        w.commit_romaji_edit();
        w.resolve_pending(true);
        w.save(&tr);
        assert!(!w.romaji_dirty, "{}", w.status);
        assert!(s.read("song.reading-override.lrc").is_some());

        let mut again = LyricsWindow::default();
        again.load_local(&tr);
        assert!(again.lyrics.as_ref().unwrap().lines[0].overrides_sheet());
        assert_eq!(furigana_first(&reading_of(&mut again, 0)), ("剣".into(), Some("けん".into())));
    }

    /// Replacing the sheet, or a Sync edit that puts another line on the
    /// overridden line's time, does not carry the override over.
    #[test]
    fn an_override_stays_with_its_own_line() {
        let s = Scratch::new("override_line");
        let track = s.track();
        s.put("song.lrc", "[00:01.00]剣(つるぎ)を\n[00:05.00]剣(つるぎ)が\n");
        let tr = track_ref(&track);
        let mut w = LyricsWindow::default();
        w.load_local(&tr);
        w.begin_romaji_edit();
        w.romaji_edit[0].text = "ken o".into();
        w.commit_romaji_edit();
        w.resolve_pending(true);
        w.save(&tr);

        // Sync: delete the overridden line, stamp the other at its time.
        w.begin_edit();
        w.edit.remove(0);
        w.edit[0].at = Some(Duration::from_secs(1));
        w.commit_edit();
        w.save(&tr);
        let back = super::super::load(&track, None).unwrap();
        assert_eq!(back.lines[0].text, "剣(つるぎ)が");
        assert!(!back.lines[0].overrides_sheet());
        assert!(s.read("song.reading-override.lrc").is_none());
        assert!(s.read("song.reading-override.lrc.old").is_some(), "kept, not discarded");

        // And an override file left on disk for a different line at the same
        // time is not taken up by the line that now sits there.
        s.put("song.reading-override.lrc", "[00:01.00]剣(つるぎ)を\n");
        s.put("song.romaji.lrc", "[00:01.00]ken ga\n");
        let back = super::super::load(&track, None).unwrap();
        assert!(!back.lines[0].overrides_sheet(), "an override bound to other text attached");

        // Replacing the sheet entirely clears it too.
        let mut w = LyricsWindow::default();
        w.load_local(&tr);
        w.apply(&Hit { synced: Some("[00:01.00]剣(つるぎ)を".into()), ..Default::default() });
        assert!(w.pending.is_empty());
        assert!(!w.lyrics.as_ref().unwrap().lines[0].overrides_sheet());
    }

    /// What a line shows in `script`, through the viewer's own `present`.
    fn shown(w: &mut LyricsWindow, i: usize, script: Script, per_word: bool, upto: Option<usize>) -> Shown {
        let line = w.lyrics.as_ref().unwrap().lines[i].clone();
        let a = auto_for(&mut w.auto, i, &line.text);
        let r = a.reading(line.romaji.as_deref(), line.overrides_sheet());
        let words = a.words.clone();
        present(script, per_word, &line, &words, Some(&r), upto)
    }
    fn confirm(w: &mut LyricsWindow, reading: &str) {
        w.begin_romaji_edit();
        w.romaji_edit[0].text = reading.into();
        w.commit_romaji_edit();
        w.resolve_pending(true);
        assert!(w.lyrics.as_ref().unwrap().lines[0].overrides_sheet());
    }
    fn texts(sh: &Shown) -> Vec<String> {
        match sh {
            Shown::Cells(c) => c.iter().map(|c| c.text.clone()).collect(),
            Shown::Text { text, .. } => vec![text.clone()],
            Shown::Romaji(r) => vec![r.clone()],
        }
    }

    /// Unconfirmed, the sheet's annotation is shown as written and gives its
    /// reading.
    #[test]
    fn both_view_shows_the_annotation_and_its_reading_by_default() {
        let mut w = win_with(ANNOTATED);
        let Shown::Cells(c) = shown(&mut w, 0, Script::Both, true, None) else { panic!() };
        assert_eq!(c[0].text, "剣(つるぎ)");
        assert_eq!(c[0].below.as_deref(), Some("tsurugi"));
    }

    /// Confirmed over the sheet, the Japanese shown is the plain word, not
    /// the bracketed annotation the user's reading replaced — per word, and
    /// on the line when words are off.
    #[test]
    fn both_view_drops_the_replaced_annotation() {
        let mut w = win_with(ANNOTATED);
        confirm(&mut w, "ken o");
        let Shown::Cells(c) = shown(&mut w, 0, Script::Both, true, None) else { panic!() };
        assert_eq!(c[0].text, "剣");
        assert_eq!(c[0].below.as_deref(), Some("ken"));
        let Shown::Text { text, under, .. } = shown(&mut w, 0, Script::Both, false, None) else { panic!() };
        assert_eq!(text, "剣を");
        assert_eq!(under.as_deref(), Some("ken o"));
        // The stored line is untouched.
        assert_eq!(w.lyrics.as_ref().unwrap().lines[0].text, "剣(つるぎ)を");
    }

    /// The whole-line fallback, where the reading cannot be placed per word,
    /// draws the plain line too — in Both and in Furigana.
    #[test]
    fn the_whole_line_fallback_drops_the_replaced_annotation() {
        let mut w = win_with(ANNOTATED);
        confirm(&mut w, "zenzen chigau uta desu yo");
        for (script, per_word) in [(Script::Both, true), (Script::Both, false), (Script::Furigana, true)] {
            let sh = shown(&mut w, 0, script, per_word, None);
            assert!(texts(&sh).iter().all(|t| !t.contains('(')), "{script:?}: {sh:?}");
            assert!(texts(&sh).concat().contains("剣を"), "{script:?}: {sh:?}");
        }
    }

    /// Karaoke's position is in the stored text; where the annotation is not
    /// drawn it is carried into the shown text, and the end stays the end.
    #[test]
    fn karaoke_position_follows_the_shown_text() {
        let mut w = win_with(ANNOTATED);
        confirm(&mut w, "zenzen chigau uta desu yo");
        let all = "剣(つるぎ)を".len();
        let Shown::Text { text, upto, .. } = shown(&mut w, 0, Script::Both, false, Some(all)) else { panic!() };
        assert_eq!(upto, Some(text.len()));
        let Shown::Text { upto, .. } = shown(&mut w, 0, Script::Both, false, Some("剣".len())) else { panic!() };
        assert_eq!(upto, Some("剣".len()));
    }

    /// Two identical lines. A confirmation is pending for the first; a Sync
    /// edit deletes it, so the second now sits at its index with the same
    /// text. Confirming must not put the override on the second — in memory,
    /// or on disk after a save and reload.
    #[test]
    fn a_pending_confirmation_does_not_move_to_an_identical_line() {
        let s = Scratch::new("pending_dup");
        let track = s.track();
        s.put("song.lrc", "[00:01.00]剣(つるぎ)を\n[00:05.00]剣(つるぎ)を\n");
        let tr = track_ref(&track);
        let mut w = LyricsWindow::default();
        w.load_local(&tr);
        w.begin_romaji_edit();
        w.romaji_edit[0].text = "ken o".into();
        w.commit_romaji_edit();
        assert_eq!(w.pending.len(), 1);
        assert_eq!(w.pending[0].line, 0);

        w.begin_edit();
        w.edit.remove(0);
        w.commit_edit();
        assert_eq!(w.lyrics.as_ref().unwrap().lines[0].text, "剣(つるぎ)を", "the twin now sits at index 0");

        w.resolve_pending(true);
        let l = &w.lyrics.as_ref().unwrap().lines[0];
        assert!(!l.overrides_sheet() && l.romaji.is_none(), "{l:?}");
        w.save(&tr);
        assert!(s.read("song.reading-override.lrc").is_none());
        let back = super::super::load(&track, None).unwrap();
        assert!(!back.lines[0].overrides_sheet());
    }

    /// The revision check on its own, for a change that did not go through
    /// an editor that clears the question: the lines reorder under a pending
    /// confirmation, and it applies to nothing.
    #[test]
    fn a_confirmation_asked_about_an_older_sheet_applies_to_nothing() {
        let mut w = win_with("[00:01.00]剣(つるぎ)を\n[00:05.00]剣(つるぎ)を\n");
        w.begin_romaji_edit();
        w.romaji_edit[0].text = "ken o".into();
        w.commit_romaji_edit();
        w.lyrics.as_mut().unwrap().lines.swap(0, 1);
        w.sheet_rev += 1;
        w.resolve_pending(true);
        assert!(w.lyrics.as_ref().unwrap().lines.iter().all(|l| !l.overrides_sheet()));
        assert!(w.status.contains("nothing was applied"), "{}", w.status);
    }

    /// A reading a source supplied that disagrees with 剣(つるぎ) is shown in
    /// the editor as disagreeing. Ticking "use this reading instead", with no
    /// edit to its text, asks for confirmation; confirming makes it win, and
    /// it survives a save and reload.
    #[test]
    fn a_supplied_reading_can_be_chosen_over_the_lyrics_without_retyping_it() {
        let s = Scratch::new("affirm");
        let track = s.track();
        s.put("song.lrc", ANNOTATED);
        s.put("song.romaji.lrc", "[00:01.00]ken o\n"); // as a source supplied it
        let tr = track_ref(&track);
        let mut w = LyricsWindow::default();
        w.load_local(&tr);
        w.begin_romaji_edit();
        assert_eq!(w.romaji_edit[0].conflicts.len(), 1, "the disagreement is shown");
        assert_eq!(w.romaji_edit[0].text, "ken o");
        w.romaji_edit[0].affirm = true;
        w.commit_romaji_edit();
        assert_eq!(w.pending.len(), 1, "ticking asks; it does not apply on its own");
        assert!(!w.lyrics.as_ref().unwrap().lines[0].overrides_sheet());
        w.resolve_pending(true);
        assert!(w.lyrics.as_ref().unwrap().lines[0].overrides_sheet());
        w.save(&tr);
        let mut again = LyricsWindow::default();
        again.load_local(&tr);
        assert!(again.lyrics.as_ref().unwrap().lines[0].overrides_sheet());
        assert_eq!(furigana_first(&reading_of(&mut again, 0)), ("剣".into(), Some("けん".into())));
    }

    /// Ticked, then "keep the lyrics' reading": the supplied reading stays
    /// supplied, and the lyrics' annotation keeps deciding.
    #[test]
    fn keeping_the_lyrics_reading_leaves_a_supplied_one_as_it_was() {
        let mut w = win_with(ANNOTATED);
        w.lyrics.as_mut().unwrap().lines[0].romaji = Some("ken o".into());
        w.begin_romaji_edit();
        w.romaji_edit[0].affirm = true;
        w.commit_romaji_edit();
        w.resolve_pending(false);
        let l = &w.lyrics.as_ref().unwrap().lines[0];
        assert_eq!(l.romaji.as_deref(), Some("ken o"));
        assert!(!l.overrides_sheet());
        assert_eq!(furigana_first(&reading_of(&mut w, 0)), ("剣".into(), Some("つるぎ".into())));
    }

    #[test]
    fn a_hit_brings_its_romaji() {
        let mut w = LyricsWindow::default();
        w.apply(&Hit {
            synced: Some("[00:00.00]作词 : X\n[00:18.05]渇いた心".into()),
            romaji: Some("[00:18.050]ka wa i ta ko ko ro".into()),
            ..Default::default()
        });
        let l = w.lyrics.as_ref().unwrap();
        assert_eq!(l.lines[1].romaji.as_deref(), Some("ka wa i ta ko ko ro"));
    }

    #[test]
    fn an_action_survives_being_folded_with_none() {
        let seek = LyricsAction::Seek(Duration::from_secs(3));
        assert_eq!(LyricsAction::None.max_or(seek), seek);
        assert_eq!(seek.max_or(LyricsAction::None), seek);
    }
}
