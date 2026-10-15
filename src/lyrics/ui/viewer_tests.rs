//! The viewer under real egui passes: Follow settles, and what it costs.
//!
//! Headless: the viewer runs as a root viewport's content in a plain
//! `egui::Context` with the app's fonts, and each pass is `Context::run`. What
//! a test reads is egui's own output (the repaint it asks for, the scroll
//! offset it stores) and the viewer's test counters.

use super::*;

pub(super) const SR: [f32; 2] = [520.0, 640.0];

/// A timed sheet: `n` lines 4 s apart, every eighth blank, each with a
/// translation.
pub(super) fn sheet(n: usize) -> Lyrics {
    const JP: [&str; 5] = ["夜明けの空に君の声が響く", "遠い街の灯りを数えながら",
                           "忘れないで この胸の約束を", "風が吹いて 季節が変わっても",
                           "光の中で笑ってみせて"];
    let lines = (0..n).map(|i| {
        let at = Some(Duration::from_millis(2_000 + 4_000 * i as u64));
        if i % 8 == 7 { return LyricLine { at, ..Default::default() }; }
        LyricLine { at, text: JP[i % JP.len()].into(), translation: Some("在光芒中对我笑一笑".into()),
                    ..Default::default() }
    }).collect();
    Lyrics { lines, source: LyricSource::Sidecar, offset_ms: 0, title: None, artist: None, translation_credit: None }
}

/// When line `i` is being sung.
pub(super) fn at_line(i: usize) -> Duration { Duration::from_millis(2_000 + 4_000 * i as u64 + 500) }

pub(super) struct Rig {
    pub ctx: egui::Context,
    t: f64,
    pub w: LyricsWindow,
    /// The window's size.
    pub size: [f32; 2],
    /// Height taken above the viewer, as the toolbar and panels take it.
    pub top: f32,
}

/// One pass's observable result.
#[derive(Debug, Clone, Copy)]
pub(super) struct Pass {
    /// egui asked for another pass at once.
    pub repaint_now: bool,
    pub offset: f32,
    pub follow_requests: u64,
    pub stats: probe::Stats,
}

impl Rig {
    pub fn new(lines: usize) -> Self {
        let ctx = egui::Context::default();
        crate::setup_fonts(&ctx, None);
        let w = LyricsWindow {
            lyrics: Some(sheet(lines)),
            loaded_for: Some(PathBuf::from("song.flac")),
            ..Default::default()
        };
        Rig { ctx, t: 0.0, w, size: SR, top: 0.0 }
    }

    pub fn pass(&mut self, pos: Duration) -> Pass {
        let input = egui::RawInput {
            screen_rect: Some(egui::Rect::from_min_size(egui::Pos2::ZERO, egui::vec2(self.size[0], self.size[1]))),
            time: Some(self.t),
            predicted_dt: 1.0 / 60.0,
            ..Default::default()
        };
        self.t += 1.0 / 60.0;
        let _ = probe::take();
        let (w, top) = (&mut self.w, self.top);
        let out = self.ctx.run(input, |c| {
            egui::CentralPanel::default().show(c, |ui| {
                if top > 0.0 { ui.add_space(top); }
                w.viewer(ui, pos)
            });
        });
        let stats = probe::take();
        let repaint_now = out.viewport_output.get(&egui::ViewportId::ROOT)
            .is_some_and(|o| o.repaint_delay == Duration::ZERO);
        Pass { repaint_now, offset: self.offset(), follow_requests: stats.follow_requests, stats }
    }

    pub fn offset(&self) -> f32 {
        egui::containers::scroll_area::State::load(&self.ctx, self.scroll_id()).map_or(0.0, |s| s.offset.y)
    }

    /// The viewer's ScrollArea id, as it reported it on its last pass.
    fn scroll_id(&self) -> egui::Id {
        probe::scroll_id().expect("the viewer has run")
    }

    /// Move the scroll offset as a user's wheel or drag would leave it.
    pub fn scroll_by_hand(&mut self, to: f32) {
        let id = self.scroll_id();
        let mut s = egui::containers::scroll_area::State::load(&self.ctx, id).expect("a scroll state");
        s.offset.y = to;
        s.store(&self.ctx, id);
    }

    /// Passes until nothing asks for a repaint, at most `max`. How many it took,
    /// or `None` if it never settled.
    pub fn settle(&mut self, pos: Duration, max: usize) -> Option<usize> {
        (1..=max).find(|_| !self.pass(pos).repaint_now)
    }
}

/// Paused on a line whose centre cannot be reached — the first and the last,
/// where the offset is clamped — Follow must settle: no repaint asked for and
/// no further centring requests, pass after pass. Before the fix it asked for
/// an immediate repaint on every pass, which with nothing else pacing frames
/// is a redraw loop for as long as the player stays paused.
#[test]
fn follow_settles_when_paused_on_a_line_it_cannot_centre() {
    for (name, line) in [("first", 0usize), ("last", 58)] {
        let mut r = Rig::new(60);
        let pos = at_line(line);
        assert!(r.settle(pos, 120).is_some(), "{name}: never settled");
        for k in 0..30 {
            let p = r.pass(pos);
            assert!(!p.repaint_now, "{name}: pass {k} after settling asked for a repaint");
            assert_eq!(p.follow_requests, 0, "{name}: pass {k} after settling still centred");
        }
    }
}

/// A centrable line, paused: centred, then quiet.
#[test]
fn follow_centres_a_middle_line_and_goes_quiet() {
    let mut r = Rig::new(60);
    let pos = at_line(30);
    assert!(r.settle(pos, 120).is_some(), "never settled");
    let centred = r.offset();
    assert!(centred > 100.0, "setup: line 30 is not scrolled to ({centred})");
    for _ in 0..10 {
        let p = r.pass(pos);
        assert!(!p.repaint_now && p.follow_requests == 0);
        assert_eq!(p.offset, centred, "the settled offset moved");
    }
}

/// Moving to the next line centres it again, by itself.
#[test]
fn follow_centres_the_next_line_when_it_becomes_active() {
    let mut r = Rig::new(60);
    assert!(r.settle(at_line(30), 120).is_some());
    let first = r.offset();
    let p = r.pass(at_line(32));
    assert!(p.follow_requests > 0, "a new active line was not centred");
    assert!(r.settle(at_line(32), 120).is_some(), "never settled on the next line");
    assert!(r.offset() > first, "the view did not move down to the next line ({} → {})", first, r.offset());
}

/// Scrolling away by hand while Follow is on is undone, as it always was:
/// the view comes back to the active line.
#[test]
fn follow_brings_a_hand_scrolled_view_back() {
    let mut r = Rig::new(60);
    let pos = at_line(30);
    assert!(r.settle(pos, 120).is_some());
    let centred = r.offset();
    r.scroll_by_hand(centred - 300.0);
    let p = r.pass(pos);
    assert!(p.follow_requests > 0 || r.pass(pos).follow_requests > 0, "a hand scroll was not answered");
    assert!(r.settle(pos, 240).is_some(), "never settled after the hand scroll");
    assert!((r.offset() - centred).abs() < 0.5, "not back on the active line: {} vs {centred}", r.offset());
}

/// With Follow off, a hand scroll stays where it was put, and turning Follow
/// back on centres the active line.
#[test]
fn follow_off_leaves_the_view_and_on_again_recentres() {
    let mut r = Rig::new(60);
    let pos = at_line(30);
    assert!(r.settle(pos, 120).is_some());
    let centred = r.offset();
    r.w.auto_scroll = false;
    r.scroll_by_hand(centred - 300.0);
    for _ in 0..5 { assert_eq!(r.pass(pos).follow_requests, 0); }
    assert!((r.offset() - (centred - 300.0)).abs() < 0.5, "Follow off moved the view");
    r.w.auto_scroll = true;
    assert!(r.settle(pos, 240).is_some());
    assert!((r.offset() - centred).abs() < 0.5, "re-enabling Follow did not recentre");
}

/// A text-size change moves every line: the active one is centred again.
#[test]
fn follow_recentres_after_a_size_change() {
    let mut r = Rig::new(60);
    let pos = at_line(30);
    assert!(r.settle(pos, 120).is_some());
    let before = r.offset();
    r.w.font_size = 30.0;
    assert!(r.pass(pos).follow_requests > 0, "a size change was not followed");
    assert!(r.settle(pos, 240).is_some());
    assert!(r.offset() > before, "larger text, same offset: not recentred ({before} → {})", r.offset());
}

/// Every cached presentation equals one worked out from scratch now: fresh
/// words and a fresh reading for each line's current text, supplied romaji and
/// override. Returns how many cached presentations were compared.
fn cached_presentations_are_current(w: &LyricsWindow, script: Script, per_word: bool) -> usize {
    let l = w.lyrics.as_ref().unwrap();
    let mut n = 0;
    for (i, line) in l.lines.iter().enumerate() {
        let Some(a) = w.auto.get(i) else { continue };
        let Some((s, p, cached)) = &a.shown else { continue };
        if (*s, *p) != (script, per_word) { continue; }
        let words = super::super::romaji::words(&line.text);
        let reading = super::super::romaji::read_line(&words, line.romaji.as_deref(), line.overrides_sheet());
        let fresh = present(script, per_word, line, &words, Some(&reading), None);
        assert_eq!(cached, &fresh, "line {i} ({}) shows a stale presentation", line.text);
        n += 1;
    }
    n
}

/// The per-line presentation cache follows everything a presentation depends
/// on: script, per-word, supplied romaji, a confirmed override, and the text.
#[test]
fn cached_presentations_follow_their_inputs() {
    let mut r = Rig::new(24);
    let pos = at_line(10);
    for (script, per_word) in [(Script::Romaji, true), (Script::Furigana, true), (Script::Both, true),
                               (Script::Both, false), (Script::Romaji, true)] {
        r.w.script = script;
        r.w.per_word = per_word;
        r.pass(pos);
        assert!(cached_presentations_are_current(&r.w, script, per_word) > 10, "{script:?}: nothing cached");
    }
    // Supplied romaji on line 3, then a text edit on line 4, then an override.
    r.w.script = Script::Both;
    r.w.per_word = true;
    r.pass(pos);
    r.w.lyrics.as_mut().unwrap().lines[3].romaji = Some("kaze ga fuite".into());
    r.pass(pos);
    cached_presentations_are_current(&r.w, Script::Both, true);
    r.w.lyrics.as_mut().unwrap().lines[4].text = "光の中で笑ってみせて".into();
    r.pass(pos);
    cached_presentations_are_current(&r.w, Script::Both, true);
    let l = &mut r.w.lyrics.as_mut().unwrap().lines[5];
    l.romaji = Some("hoshi".into());
    l.reading_override = Some(l.text.clone());
    r.pass(pos);
    assert!(cached_presentations_are_current(&r.w, Script::Both, true) > 10);
}

/// Unchanged lines are not presented again pass after pass: their cached
/// presentation is the same allocation.
#[test]
fn an_unchanged_line_keeps_its_presentation() {
    let mut r = Rig::new(24);
    r.w.script = Script::Furigana;
    let pos = at_line(10);
    r.pass(pos);
    // The cells' heap buffer: a presentation worked out again is a new one.
    let ptr = |w: &LyricsWindow| match &w.auto[2].shown {
        Some((_, _, Shown::Cells(c))) => Some(c.as_ptr()),
        _ => None,
    };
    let before = ptr(&r.w);
    assert!(before.is_some(), "setup: line 2 was not presented");
    for _ in 0..5 { r.pass(pos); }
    assert_eq!(ptr(&r.w), before, "line 2 was presented again with nothing changed");
}

/// A line being sung is presented at its position every time, never from the
/// cache of the line without one.
#[test]
fn a_sung_position_is_never_served_from_the_cache() {
    let line = LyricLine { at: Some(Duration::ZERO), text: "夜明けの空に君の声が響く".into(), ..Default::default() };
    let mut a = Auto::default();
    let words = super::super::romaji::words(&line.text);
    a.text = line.text.clone();
    a.line = super::super::romaji::join(&words);
    a.words = words.clone();
    for script in [Script::Romaji, Script::Furigana, Script::Both] {
        let _ = a.shown(script, true, &line, None);
        for u in [3, 9, 18] {
            let reading = super::super::romaji::read_line(&words, None, false);
            let fresh = present(script, true, &line, &words, Some(&reading), Some(u));
            assert_eq!(&*a.shown(script, true, &line, Some(u)), &fresh, "{script:?} at {u}");
        }
    }
}


// ── Rows out of view ───────────────────────────────────────────────────────

/// A sheet whose line 6 sits at the wrap threshold: one line wide as plain
/// text, two lines when active (larger, bold).
fn sheet_with_threshold(n: usize) -> Lyrics {
    let mut l = sheet(n);
    l.lines[6].text = "あ".repeat(28);
    l
}

/// One pass's geometry: offset, content height, and the rows drawn.
type Frame = (f32, f32, Vec<(usize, f32, f32, egui::Id)>);

/// The same schedule with rows out of view skipped, and with every row drawn.
fn trace(skip: bool, setup: &dyn Fn(&mut LyricsWindow), lines: usize) -> Vec<Frame> {
    let mut r = Rig::new(lines);
    r.w.lyrics = Some(sheet_with_threshold(lines));
    r.w.draw_every_row = !skip;
    setup(&mut r.w);
    let mut out = Vec::new();
    let mut step = |r: &mut Rig, pos: Duration, n: usize| for _ in 0..n {
        let _ = probe::take_drawn();
        r.pass(pos);
        let (o, c) = probe::view();
        out.push((o, c, probe::take_drawn()));
    };
    for line in [4, 5, 6, 7 + 1, 6, 10] { step(&mut r, at_line(line), 40); }
    // The translation in Simplified (another face, other metrics), then back.
    r.w.translation_display = super::super::zh::ChineseDisplay::Simplified;
    step(&mut r, at_line(10), 40);
    r.w.translation_display = super::super::zh::ChineseDisplay::Original;
    step(&mut r, at_line(10), 40);
    let o = r.offset();
    r.scroll_by_hand(o - 400.0);
    step(&mut r, at_line(10), 60);
    for line in [lines / 2, lines - 2, 0, 3] { step(&mut r, at_line(line), 60); }
    r.w.auto_scroll = false;
    r.scroll_by_hand(0.0);
    step(&mut r, at_line(lines - 3), 20);
    r.w.auto_scroll = true;
    step(&mut r, at_line(lines - 3), 60);
    // The threshold line active while out of view, then not: it must be
    // skipped by its plain height, not the two-line one it had when active.
    r.w.auto_scroll = false;
    step(&mut r, at_line(6), 5);
    step(&mut r, at_line(8), 5);
    r.w.auto_scroll = true;
    step(&mut r, at_line(8), 60);
    out
}

fn assert_same_geometry(name: &str, a: &[Frame], b: &[Frame]) {
    assert_eq!(a.len(), b.len());
    for (k, ((oa, ca, da), (ob, cb, db))) in a.iter().zip(b).enumerate() {
        let mut ids: Vec<_> = da.iter().map(|d| d.3).collect();
        ids.sort_by_key(|id| id.value());
        ids.dedup();
        assert_eq!(ids.len(), da.len(), "{name}: pass {k}: two rows share an id");
        assert!((oa - ob).abs() <= 0.5, "{name}: pass {k}: offset {oa} vs {ob}");
        assert!((ca - cb).abs() <= 0.5, "{name}: pass {k}: content height {ca} vs {cb}");
        for (i, top, h, id) in da {
            if let Some((_, t2, h2, id2)) = db.iter().find(|(j, _, _, _)| j == i) {
                assert!((top - t2).abs() <= 0.5 && (h - h2).abs() <= 0.5,
                        "{name}: pass {k}: row {i} at {top}+{h} vs {t2}+{h2}");
                assert_eq!(id, id2, "{name}: pass {k}: row {i} has another id when rows are skipped");
            }
        }
    }
}

/// Skipping rows out of view changes nothing anyone can see: the scroll
/// offset, the scroll extent and every drawn row's place and height are
/// those of drawing every row, through line changes (one at the wrap
/// threshold), a hand scroll, seeks to the middle, last and first lines,
/// and Follow turned off and on with the active line out of view — in each
/// script, with and without translations, at two text sizes.
#[test]
fn skipping_rows_out_of_view_changes_no_geometry() {
    let configs: [(&str, Script, bool, f32); 6] = [
        ("original", Script::Original, true, 17.0), ("original.big", Script::Original, true, 26.0),
        ("romaji", Script::Romaji, true, 17.0), ("furigana", Script::Furigana, true, 17.0),
        ("both", Script::Both, false, 17.0), ("both.big", Script::Both, true, 24.0),
    ];
    for (name, script, translation, size) in configs {
        let setup = move |w: &mut LyricsWindow| {
            w.script = script;
            w.show_translation = translation;
            w.font_size = size;
        };
        let skipped = trace(true, &setup, 60);
        let full = trace(false, &setup, 60);
        assert_same_geometry(name, &skipped, &full);
        let drawn_skip: usize = skipped.iter().map(|f| f.2.len()).sum();
        let drawn_full: usize = full.iter().map(|f| f.2.len()).sum();
        assert!(drawn_skip * 2 < drawn_full, "{name}: skipping drew {drawn_skip} rows, everything {drawn_full}");
    }
}

/// Once measured, only the rows in view (and the active one) are laid out,
/// however long the sheet.
#[test]
fn a_long_sheet_lays_out_only_what_is_in_view() {
    let mut r = Rig::new(240);
    let pos = at_line(120);
    assert!(r.settle(pos, 240).is_some());
    for _ in 0..5 {
        let s = r.pass(pos).stats;
        assert_eq!(s.rows_visited, 240);
        assert!(s.rows_laid_out <= 25, "{} of 210 rows laid out", s.rows_laid_out);
        assert!(s.rows_laid_out >= 10, "only {} rows laid out", s.rows_laid_out);
    }
}

/// A change that can move every row lays every row out once, then only the
/// view again.
#[test]
fn a_size_change_measures_every_row_once() {
    let mut r = Rig::new(240);
    let pos = at_line(120);
    assert!(r.settle(pos, 240).is_some());
    r.w.font_size = 22.0;
    assert_eq!(r.pass(pos).stats.rows_laid_out, 210, "not every row was measured after the size change");
    assert!(r.settle(pos, 240).is_some());
    assert!(r.pass(pos).stats.rows_laid_out <= 25);
}

/// With Follow off and the view at the top, the active line far below is
/// still drawn — so Follow, turned on, can centre it at once.
#[test]
fn the_active_line_is_drawn_when_out_of_view() {
    let mut r = Rig::new(240);
    r.w.auto_scroll = false;
    let pos = at_line(200);
    r.pass(pos);
    r.pass(pos);
    let _ = probe::take_drawn();
    r.pass(pos);
    let drawn = probe::take_drawn();
    assert!(drawn.iter().any(|(i, _, _, _)| *i == 200), "the active line was skipped");
    assert!(drawn.iter().all(|(i, _, _, _)| *i < 30 || *i == 200), "rows far from view were drawn: {drawn:?}");
}

/// Editing a line out of view forgets that row's height, so it is measured
/// again rather than skipped by its old one.
#[test]
fn an_edited_line_out_of_view_is_measured_again() {
    let mut r = Rig::new(60);
    let pos = at_line(10);
    assert!(r.settle(pos, 240).is_some());
    let h = |w: &LyricsWindow| w.geometry.rows[50].h[0];
    let before = h(&r.w);
    assert!(before.is_some(), "setup: row 50 not measured");
    r.w.lyrics.as_mut().unwrap().lines[50].text = "あ".repeat(60);
    r.pass(pos);
    let after = h(&r.w);
    assert!(after.is_some() && after != before, "row 50 kept its old height ({before:?} → {after:?})");
}

// ── Chinese display of the translation ─────────────────────────────────────

fn shown_translations(r: &mut Rig, pos: Duration) -> Vec<(usize, String)> {
    let _ = probe::take_translations();
    r.pass(pos);
    probe::take_translations()
}

/// Switching the display while paused shows the new form on the next pass —
/// no playback needed — and back again; the stored text never changes.
#[test]
fn switching_the_chinese_display_shows_at_once_while_paused() {
    use super::super::zh::ChineseDisplay::*;
    let mut r = Rig::new(24);
    let pos = at_line(10);
    let stored: Vec<_> = r.w.lyrics.as_ref().unwrap().lines.iter().map(|l| l.translation.clone()).collect();
    let orig = shown_translations(&mut r, pos);
    assert!(!orig.is_empty() && orig.iter().all(|(_, t)| t == "在光芒中对我笑一笑"), "{orig:?}");
    r.w.translation_display = Traditional;
    let trad = shown_translations(&mut r, pos);
    assert!(!trad.is_empty() && trad.iter().all(|(_, t)| t == "在光芒中對我笑一笑"), "{trad:?}");
    r.w.translation_display = Simplified;
    assert!(shown_translations(&mut r, pos).iter().all(|(_, t)| t == "在光芒中对我笑一笑"));
    r.w.translation_display = Original;
    assert_eq!(shown_translations(&mut r, pos), orig);
    let after: Vec<_> = r.w.lyrics.as_ref().unwrap().lines.iter().map(|l| l.translation.clone()).collect();
    assert_eq!(after, stored, "the stored translation changed");
    assert!(!r.w.dirty && !r.w.romaji_dirty);
}

/// An edited translation, and a new sheet, are shown converted afresh — never
/// the conversion of what was there before.
#[test]
fn a_changed_translation_is_converted_again() {
    use super::super::zh::ChineseDisplay::*;
    let mut r = Rig::new(24);
    let pos = at_line(10);
    r.w.translation_display = Traditional;
    shown_translations(&mut r, pos);
    r.w.lyrics.as_mut().unwrap().lines[10].translation = Some("爱与梦".into());
    let got = shown_translations(&mut r, pos);
    assert!(got.contains(&(10, "愛與夢".to_string())), "{got:?}");
    // A different sheet at the same indices.
    let mut other = sheet(24);
    for l in other.lines.iter_mut() { if l.translation.is_some() { l.translation = Some("后来".into()); } }
    r.w.lyrics = Some(other);
    r.w.sheet_rev += 1;
    let got = shown_translations(&mut r, pos);
    assert!(!got.is_empty() && got.iter().all(|(_, t)| t == "後來"), "{got:?}");
}

/// Converting is done once per line and mode, not once per pass.
#[test]
fn a_translation_is_converted_once() {
    use super::super::zh::ChineseDisplay::*;
    let mut r = Rig::new(24);
    let pos = at_line(10);
    r.w.translation_display = Simplified;
    r.pass(pos);
    let n = r.w.zh.conversions;
    assert!(n > 0 && r.w.zh.worked_out() > 0);
    for _ in 0..5 { r.pass(pos); }
    assert_eq!(r.w.zh.conversions, n, "lines were converted again with nothing changed");
}

// ── The view's height ──────────────────────────────────────────────────────

/// Where Follow puts line `line` in a viewer of this size, worked out by a
/// viewer that has only ever been that size.
fn fresh_offset(size: [f32; 2], top: f32, line: usize) -> f32 {
    let mut r = Rig::new(60);
    r.size = size;
    r.top = top;
    assert!(r.settle(at_line(line), 240).is_some(), "the reference never settled");
    r.offset()
}

/// A settled view, then only its height changes — the window taller or
/// shorter, or the toolbar above taking more room — with the width, text and
/// line as they were. The active line is centred in the new view (where a
/// viewer that was always that size puts it), and then the view goes quiet.
/// The first and last lines, which cannot be centred, settle too.
#[test]
fn follow_recentres_when_only_the_view_height_changes() {
    let changes: [(&str, [f32; 2], f32); 3] = [
        ("taller", [SR[0], SR[1] + 260.0], 0.0),
        ("shorter", [SR[0], SR[1] - 240.0], 0.0),
        ("toolbar", SR, 140.0),
    ];
    for (name, size, top) in changes {
        for line in [30usize, 0, 58] {
            let mut r = Rig::new(60);
            let pos = at_line(line);
            assert!(r.settle(pos, 240).is_some(), "{name} line {line}: never settled");
            r.size = size;
            r.top = top;
            assert!(r.settle(pos, 240).is_some(), "{name} line {line}: never settled after the change");
            for k in 0..10 {
                let p = r.pass(pos);
                assert!(!p.repaint_now && p.follow_requests == 0, "{name} line {line}: pass {k} is not quiet");
            }
            let want = fresh_offset(size, top, line);
            assert!((r.offset() - want).abs() <= 0.5,
                    "{name} line {line}: offset {} where a viewer of that size centres at {want}", r.offset());
        }
    }
}

// ── The face replaced ──────────────────────────────────────────────────────

/// Lines that wrap, with Latin text whose width depends on the face.
fn wrapping_sheet(n: usize) -> Lyrics {
    let mut l = sheet(n);
    for (i, line) in l.lines.iter_mut().enumerate() {
        if i % 3 == 1 && !line.text.is_empty() {
            line.text = format!("{} — and the morning light keeps calling out your name again", line.text);
        }
    }
    l
}

/// One pass of [`face_trace`]: its geometry, whether it asked for a repaint,
/// rows laid out and Follow requests.
type FacePass = (Frame, bool, u64, u64);

/// Each pass's geometry and rows laid out, through face changes: the default
/// face, then two picked fonts in turn (both installed as "ui_choice", the
/// same name), then the default again — sizes, width and scale unchanged.
/// Then seeks below and above, so rows far from where the change happened
/// are placed. `None` when a picked font is not installed here.
fn face_trace(skip: bool) -> Option<Vec<Vec<FacePass>>> {
    const PICKS: [Option<&str>; 4] = [Some("Courier New"), Some("Impact"), None, Some("Courier New")];
    if PICKS.iter().flatten().any(|f| crate::fonts::find(f).is_none()) { return None; }
    let mut r = Rig::new(90);
    r.w.lyrics = Some(wrapping_sheet(90));
    r.w.draw_every_row = !skip;
    let mut phases = Vec::new();
    let step = |r: &mut Rig, pos: Duration, n: usize, out: &mut Vec<FacePass>| for _ in 0..n {
        let _ = probe::take_drawn();
        let p = r.pass(pos);
        let (o, c) = probe::view();
        out.push(((o, c, probe::take_drawn()), p.repaint_now, p.stats.rows_laid_out, p.follow_requests));
    };
    let mut first = Vec::new();
    step(&mut r, at_line(40), 60, &mut first);
    phases.push(first);
    for pick in PICKS {
        crate::setup_fonts(&r.ctx, pick);
        let mut out = Vec::new();
        step(&mut r, at_line(40), 60, &mut out);
        for line in [80, 2, 40] { step(&mut r, at_line(line), 40, &mut out); }
        phases.push(out);
    }
    Some(phases)
}

/// The defect: picking another font kept every off-screen row's height from
/// the old face, so the scroll extent, every row below and Follow's centring
/// were wrong. Skipping rows must give exactly what drawing every row gives,
/// across the change and back to the default; the change measures every row
/// once and then only the view again.
#[test]
fn a_new_face_measures_every_row_again() {
    let (Some(skipped), Some(full)) = (face_trace(true), face_trace(false)) else {
        println!("Courier New / Impact not installed: nothing to check here");
        return;
    };
    // Setup: each face lays the sheet out to another height.
    let extent = |p: &[FacePass]| p[59].0.1;
    for k in 1..full.len() {
        assert!((extent(&full[k]) - extent(&full[k - 1])).abs() > 1.0,
                "setup: phase {k}'s face gives the same content height as the one before ({})", extent(&full[k]));
    }
    for (k, (a, b)) in skipped.iter().zip(&full).enumerate() {
        let fa: Vec<Frame> = a.iter().map(|x| x.0.clone()).collect();
        let fb: Vec<Frame> = b.iter().map(|x| x.0.clone()).collect();
        assert_same_geometry(&format!("phase {k}"), &fa, &fb);
    }
    let rows = wrapping_sheet(90).lines.iter().filter(|l| !l.text.trim().is_empty()).count() as u64;
    for (k, phase) in skipped.iter().enumerate().skip(1) {
        assert_eq!(phase[0].2, rows, "phase {k}: the new face did not measure every row");
        // Follow answers in the pass the face changed, not a pass later when
        // the moved view gives it away.
        assert!(phase[0].3 > 0, "phase {k}: the first pass in the new face did not centre");
        // Follow went quiet on the line it was on, and later passes laid out
        // only the view.
        assert!(!phase[59].1, "phase {k}: Follow did not settle in the new face");
        assert!(phase[30..60].iter().all(|x| x.2 <= 30), "phase {k}: still laying out every row");
    }
    // Follow centred line 40 where a viewer that only ever had that face does.
    for (k, pick) in [None, Some("Courier New"), Some("Impact"), None, Some("Courier New")].into_iter().enumerate() {
        let mut fresh = Rig::new(90);
        fresh.w.lyrics = Some(wrapping_sheet(90));
        crate::setup_fonts(&fresh.ctx, pick);
        assert!(fresh.settle(at_line(40), 240).is_some(), "the reference never settled");
        let got = skipped[k][59].0.0;
        assert!((got - fresh.offset()).abs() <= 0.5, "phase {k}: Follow at {got}, a fresh viewer at {}", fresh.offset());
    }
}
