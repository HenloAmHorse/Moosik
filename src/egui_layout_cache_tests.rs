//! The text-layout cache across an immediate child viewport (vendored egui).
//!
//! egui keeps one `Fonts`, and one galley cache, per pixels-per-point, shared
//! by every viewport at that scale. The cache keeps what was used since its
//! last flush and is meant to be flushed once per frame. An immediate child's
//! pass runs inside its parent's; flushing at the child's start too evicted
//! everything the frame had not yet re-used — the parent's text included,
//! since the app draws its panels after its children — so both re-laid all
//! their text every frame.
//!
//! These run real passes: a root `Context::run`, and the child's pass run by
//! an immediate-viewport renderer as eframe's does (`Context::run` inside the
//! parent's). What they read is the shared cache itself.

use eframe::egui;
use std::cell::{Cell, RefCell};

const CHILD: &str = "layout_cache_child";

thread_local! {
    /// The child's scale, and the galleys created during its last pass.
    static CHILD_PPP: Cell<f32> = const { Cell::new(1.0) };
    static CHILD_MISSES: Cell<usize> = const { Cell::new(0) };
    static CHILD_CACHE: Cell<usize> = const { Cell::new(0) };
    /// The texture cap each viewport's renderer reports (`None`: egui's default).
    static ROOT_CAP: Cell<Option<usize>> = const { Cell::new(None) };
    static CHILD_CAP: Cell<Option<usize>> = const { Cell::new(None) };
    /// The font atlas the child's pass found in use at its start, and the cap
    /// its `Fonts` had then.
    static CHILD_ATLAS: RefCell<Option<Atlas>> = const { RefCell::new(None) };
    static CHILD_FONTS_CAP: Cell<usize> = const { Cell::new(0) };
}

type Atlas = std::sync::Arc<egui::mutex::Mutex<egui::epaint::TextureAtlas>>;

fn atlas(c: &egui::Context) -> Atlas { c.fonts(|f| f.texture_atlas()) }

fn child_id() -> egui::ViewportId { egui::ViewportId::from_hash_of(CHILD) }

fn input(id: egui::ViewportId, t: f64) -> egui::RawInput {
    let info = |ppp: f32, parent| egui::ViewportInfo {
        parent, native_pixels_per_point: Some(ppp),
        inner_rect: Some(egui::Rect::from_min_size(egui::Pos2::ZERO, egui::vec2(600.0, 500.0))),
        ..Default::default()
    };
    let mut viewports = egui::ViewportIdMap::default();
    viewports.insert(egui::ViewportId::ROOT, info(1.0, None));
    viewports.insert(child_id(), info(CHILD_PPP.with(Cell::get), Some(egui::ViewportId::ROOT)));
    let cap = if id == egui::ViewportId::ROOT { ROOT_CAP.with(Cell::get) } else { CHILD_CAP.with(Cell::get) };
    egui::RawInput {
        viewport_id: id,
        viewports,
        max_texture_side: cap,
        screen_rect: Some(egui::Rect::from_min_size(egui::Pos2::ZERO, egui::vec2(600.0, 500.0))),
        time: Some(t),
        ..Default::default()
    }
}

fn install_renderer(t: f64) {
    egui::Context::set_immediate_viewport_renderer(move |ctx, vp| {
        let egui::ImmediateViewport { ids, mut viewport_ui_cb, .. } = vp;
        let out = ctx.run(input(ids.this, t), |c| {
            CHILD_ATLAS.with(|a| *a.borrow_mut() = Some(atlas(c)));
            CHILD_FONTS_CAP.with(|m| m.set(c.fonts(|f| f.max_texture_side())));
            let before = c.fonts(|f| f.num_galleys_in_cache());
            viewport_ui_cb(c);
            let after = c.fonts(|f| f.num_galleys_in_cache());
            CHILD_MISSES.with(|m| m.set(after.saturating_sub(before)));
            CHILD_CACHE.with(|m| m.set(after));
        });
        let _ = ctx.tessellate(out.shapes, out.pixels_per_point);
    });
}

/// One frame: the child (if any) first, then the root's own labels, as the
/// app orders them. Returns the galleys the root's labels created.
fn frame(
    ctx: &egui::Context, t: f64, child: Option<&dyn Fn(&mut egui::Ui)>, root: &dyn Fn(&mut egui::Ui),
) -> usize {
    frame_with(ctx, t, &|_| {}, child, root)
}

/// [`frame`], with `pre` run in the root's pass before the child's.
fn frame_with(
    ctx: &egui::Context, t: f64, pre: &dyn Fn(&egui::Context), child: Option<&dyn Fn(&mut egui::Ui)>,
    root: &dyn Fn(&mut egui::Ui),
) -> usize {
    install_renderer(t);
    let mut misses = 0;
    let out = ctx.run(input(egui::ViewportId::ROOT, t), |c| {
        pre(c);
        if let Some(child) = child {
            c.show_viewport_immediate(child_id(), egui::ViewportBuilder::default(), |cc, _| {
                egui::CentralPanel::default().show(cc, |ui| child(ui));
            });
        }
        let before = c.fonts(|f| f.num_galleys_in_cache());
        egui::CentralPanel::default().show(c, |ui| root(ui));
        misses = c.fonts(|f| f.num_galleys_in_cache()).saturating_sub(before);
    });
    let _ = ctx.tessellate(out.shapes, out.pixels_per_point);
    misses
}

fn ctx() -> egui::Context {
    CHILD_PPP.with(|p| p.set(1.0));
    ROOT_CAP.with(|p| p.set(None));
    CHILD_CAP.with(|p| p.set(None));
    let c = egui::Context::default();
    c.set_embed_viewports(false);
    c
}

fn root_labels(ui: &mut egui::Ui) {
    for k in 0..40 { ui.label(format!("Root row {k} — a playlist entry")); }
}

fn child_labels(ui: &mut egui::Ui) {
    for k in 0..25 { ui.label(format!("Lyric line {k} 夜明けの空に")); }
}

/// The defect: with a child open, the root re-laid every label every frame,
/// and so did the child. Unchanged text must be laid out once.
#[test]
fn an_immediate_child_does_not_evict_unchanged_text() {
    CHILD_PPP.with(|p| p.set(1.0));
    let c = ctx();
    for k in 0..5 { frame(&c, k as f64 / 60.0, Some(&child_labels), &root_labels); }
    for k in 5..20 {
        let root = frame(&c, k as f64 / 60.0, Some(&child_labels), &root_labels);
        assert_eq!(root, 0, "frame {k}: the root re-laid {root} unchanged labels");
        let child = CHILD_MISSES.with(Cell::get);
        assert_eq!(child, 0, "frame {k}: the child re-laid {child} unchanged labels");
    }
}

/// Text that changes every frame is still let go: the shared cache stays
/// bounded with a child at the same scale drawing new text each frame.
#[test]
fn a_child_drawing_new_text_each_frame_keeps_the_cache_bounded() {
    CHILD_PPP.with(|p| p.set(1.0));
    let c = ctx();
    let n = Cell::new(0u64);
    let changing = |ui: &mut egui::Ui| {
        let base = n.get();
        for k in 0..25 { ui.label(format!("{} new each frame", base * 100 + k)); }
    };
    let mut peak = 0;
    for k in 0..200 {
        n.set(k);
        frame(&c, k as f64 / 60.0, Some(&changing), &root_labels);
        peak = peak.max(CHILD_CACHE.with(Cell::get));
    }
    // One frame's worth of each (40 + 25, plus the few panels add), and at
    // most the previous frame's changing text beside it.
    assert!(peak <= 2 * (40 + 25) + 20, "the cache grew to {peak} galleys");
}

/// A child on another scale has its own `Fonts`: it still flushes them at its
/// own pass, so its new text is let go too.
#[test]
fn a_child_on_another_scale_keeps_its_own_cache_bounded() {
    let c = ctx();
    CHILD_PPP.with(|p| p.set(2.0));
    let n = Cell::new(0u64);
    let changing = |ui: &mut egui::Ui| {
        let base = n.get();
        for k in 0..25 { ui.label(format!("{} new each frame", base * 100 + k)); }
    };
    let mut peak = 0;
    for k in 0..200 {
        n.set(k);
        frame(&c, k as f64 / 60.0, Some(&changing), &root_labels);
        peak = peak.max(CHILD_CACHE.with(Cell::get));
    }
    assert!(peak <= 2 * 25 + 20, "the other scale's cache grew to {peak} galleys");
    CHILD_PPP.with(|p| p.set(1.0));
}

/// Without a child nothing changes: unchanged text hits, new text is let go.
#[test]
fn the_root_alone_behaves_as_before() {
    let c = ctx();
    for k in 0..3 { frame(&c, k as f64 / 60.0, None, &root_labels); }
    assert_eq!(frame(&c, 0.1, None, &root_labels), 0);
    let n = Cell::new(0u64);
    let changing = |ui: &mut egui::Ui| {
        for k in 0..40 { ui.label(format!("{} root new", n.get() * 100 + k)); }
    };
    for k in 0..100 {
        n.set(k);
        frame(&c, k as f64 / 60.0, None, &changing);
    }
    let size = c.fonts(|f| f.num_galleys_in_cache());
    assert!(size <= 2 * 40 + 20, "the root's cache grew to {size}");
}

/// New font definitions are new `Fonts`: nothing laid out under the old ones
/// is reused, so no galley outlives the atlas it points into.
#[test]
fn new_fonts_lay_everything_out_again() {
    CHILD_PPP.with(|p| p.set(1.0));
    let c = ctx();
    for k in 0..5 { frame(&c, k as f64 / 60.0, Some(&child_labels), &root_labels); }
    let mut defs = egui::FontDefinitions::default();
    defs.families.get_mut(&egui::FontFamily::Proportional).unwrap().reverse();
    c.set_fonts(defs);
    let root = frame(&c, 0.2, Some(&child_labels), &root_labels);
    assert!(root >= 40, "the root reused {} galleys across new fonts", 40 - root.min(40));
    assert!(CHILD_MISSES.with(Cell::get) >= 25, "the child reused galleys across new fonts");
}

fn child_atlas() -> Atlas { CHILD_ATLAS.with(|a| a.borrow().clone()).expect("the child ran") }

fn same(a: &Atlas, b: &Atlas) -> bool { std::sync::Arc::ptr_eq(a, b) }

/// The patch's precondition, in the renderer Moosik uses: eframe's glow
/// integration reads one cap from its one painter and gives it to every
/// viewport's input, the immediate children's included — so a nested pass at
/// the parent's scale has the parent's cap. Read from the vendored eframe's
/// source, since no GL context can be made here. wgpu is not a renderer
/// feature of this build; nothing is claimed for it.
#[test]
fn glow_gives_every_viewport_its_one_painters_texture_cap() {
    let src = include_str!("../vendor/eframe-0.31.1/src/native/glow_integration.rs").replace("\r\n", "\n");
    assert!(src.contains("let max_texture_side = painter.max_texture_side();\n        glutin.max_texture_side = Some(max_texture_side);\n        for viewport in glutin.viewports.values_mut() {\n            if let Some(egui_winit) = viewport.egui_winit.as_mut() {\n                egui_winit.set_max_texture_side(max_texture_side);"),
            "glow no longer sets one cap for every existing viewport");
    // Viewports made later (the immediate children) get it at creation.
    assert!(src.contains("event_loop.system_theme(),\n                self.max_texture_side,\n            )"),
            "glow no longer gives a new viewport the painter's cap");
    let manifest = include_str!("../Cargo.toml");
    let line = manifest.lines().find(|l| l.starts_with("eframe = ")).expect("eframe dependency");
    assert!(!line.contains("wgpu"), "the wgpu renderer is enabled: the cap invariant is untested there");
}

/// Same scale, same cap: the child's pass uses the very `Fonts` (and atlas)
/// the root began, frame after frame.
#[test]
fn a_nested_pass_with_the_same_cap_uses_the_roots_fonts() {
    let c = ctx();
    ROOT_CAP.with(|p| p.set(Some(4096)));
    CHILD_CAP.with(|p| p.set(Some(4096)));
    for k in 0..5 {
        let root = RefCell::new(None);
        frame_with(&c, k as f64 / 60.0, &|c| *root.borrow_mut() = Some(atlas(c)), Some(&child_labels), &root_labels);
        assert!(same(&root.into_inner().unwrap(), &child_atlas()), "frame {k}: the child has other fonts");
        assert_eq!(CHILD_FONTS_CAP.with(Cell::get), 4096);
    }
}

/// A cap change (another renderer) is taken at the next frame's start: the
/// root recreates its fonts there, the child in that frame uses them, and
/// text is laid out again once, then reused.
#[test]
fn a_texture_cap_change_is_taken_at_the_next_frame_start() {
    let c = ctx();
    for k in 0..5 { frame(&c, k as f64 / 60.0, Some(&child_labels), &root_labels); }
    let old = atlas(&c);
    ROOT_CAP.with(|p| p.set(Some(1024)));
    CHILD_CAP.with(|p| p.set(Some(1024)));
    let root = RefCell::new(None);
    let misses = frame_with(&c, 0.2, &|c| *root.borrow_mut() = Some(atlas(c)), Some(&child_labels), &root_labels);
    let root = root.into_inner().unwrap();
    assert!(!same(&old, &root), "the root kept the old cap's atlas");
    assert!(same(&root, &child_atlas()), "the child recreated the fonts again");
    assert_eq!(CHILD_FONTS_CAP.with(Cell::get), 1024);
    assert!(misses >= 40, "the root reused galleys laid out against the old atlas");
    for k in 0..5 {
        assert_eq!(frame(&c, 0.3 + k as f64 / 60.0, Some(&child_labels), &root_labels), 0, "frame {k}: re-laid");
        assert_eq!(CHILD_MISSES.with(Cell::get), 0);
    }
}

/// Not the configured renderer's case: a nested child whose renderer reports
/// another cap begins its fonts at its own pass, exactly as upstream egui
/// does (recreating them for that cap). The patch claims nothing there.
#[test]
fn a_nested_pass_with_another_cap_is_left_to_upstream() {
    let c = ctx();
    ROOT_CAP.with(|p| p.set(Some(4096)));
    CHILD_CAP.with(|p| p.set(Some(1024)));
    let root = RefCell::new(None);
    frame_with(&c, 0.0, &|c| *root.borrow_mut() = Some(atlas(c)), Some(&child_labels), &root_labels);
    assert_eq!(CHILD_FONTS_CAP.with(Cell::get), 1024, "the child's pass did not begin its fonts");
    assert!(!same(&root.into_inner().unwrap(), &child_atlas()));
}

/// A frame that fills the atlas past egui's recreation threshold before the
/// child runs. Upstream's child pass recreated the atlas there, beneath the
/// root's galleys already laid out against the old one. Patched, the atlas in
/// use stays the one those galleys point into until the frame ends — without
/// overflowing, so nothing is overwritten — and is recreated at the next
/// frame's start, where everything is laid out against the new one.
#[test]
fn a_full_atlas_is_recreated_at_the_next_frame_start_not_under_the_root() {
    let c = ctx();
    // egui's smallest atlas.
    ROOT_CAP.with(|p| p.set(Some(1024)));
    CHILD_CAP.with(|p| p.set(Some(1024)));
    frame(&c, 0.0, Some(&child_labels), &root_labels);
    let held = RefCell::new((None, Vec::new(), 0.0f32));
    let fill = |c: &egui::Context| {
        let a = atlas(c);
        let mut galleys = Vec::new();
        let mut n = 0u32;
        while a.lock().fill_ratio() <= 0.8 {
            let s = char::from_u32(0x41 + n % 26).unwrap().to_string();
            let size = 60.0 + (n / 26) as f32 * 10.0;
            galleys.push(c.fonts(|f| f.layout_no_wrap(s, egui::FontId::proportional(size), egui::Color32::WHITE)));
            n += 1;
            assert!(n < 2_000, "setup: the atlas never filled");
        }
        let r = a.lock().fill_ratio();
        *held.borrow_mut() = (Some(a), galleys, r);
    };
    let after_child = RefCell::new(None);
    let check = |ui: &mut egui::Ui| {
        *after_child.borrow_mut() = Some(atlas(ui.ctx()));
        root_labels(ui);
    };
    frame_with(&c, 0.1, &fill, Some(&child_labels), &check);
    let (root, galleys, ratio) = held.into_inner();
    let root = root.unwrap();
    assert!(ratio > 0.8 && ratio < 1.0, "setup: fill {ratio}");
    assert!(!galleys.is_empty());
    assert!(same(&root, &child_atlas()), "the child recreated the atlas under the root's galleys");
    assert!(same(&root, after_child.borrow().as_ref().unwrap()), "the root's later text has another atlas");
    let end = root.lock().fill_ratio();
    assert!(end < 1.0, "the atlas overflowed in the frame (glyphs overwritten)");
    // The next frame's start recreates it.
    let misses = frame(&c, 0.2, Some(&child_labels), &root_labels);
    let now = atlas(&c);
    assert!(!same(&root, &now), "the full atlas was never recreated");
    assert!(now.lock().fill_ratio() < 0.5, "the new atlas is not fresh");
    assert!(same(&now, &child_atlas()));
    assert!(misses >= 40, "the root reused galleys from the old atlas");
    assert_eq!(frame(&c, 0.3, Some(&child_labels), &root_labels), 0, "then reused");
}

/// New definitions between frames: taken at the root's start; the child in
/// that frame uses the root's new fonts rather than recreating them again.
#[test]
fn new_fonts_between_frames_are_taken_once_at_the_frame_start() {
    let c = ctx();
    for k in 0..3 { frame(&c, k as f64 / 60.0, Some(&child_labels), &root_labels); }
    let old = atlas(&c);
    let mut defs = egui::FontDefinitions::default();
    defs.families.get_mut(&egui::FontFamily::Proportional).unwrap().reverse();
    c.set_fonts(defs);
    let root = RefCell::new(None);
    frame_with(&c, 0.1, &|c| *root.borrow_mut() = Some(atlas(c)), Some(&child_labels), &root_labels);
    let root = root.into_inner().unwrap();
    assert!(!same(&old, &root));
    assert!(same(&root, &child_atlas()));
}

/// New definitions set inside the root's pass, before a child's: upstream
/// egui takes them at the child's pass start (`update_fonts_mut`, ahead of the
/// patched lines and unchanged by them), so the root's galleys from earlier
/// in the frame were laid out against fonts replaced mid-frame. The patch
/// keeps exactly that — an inherited limitation, not a new one. Moosik never
/// does it: its font picker runs after every immediate child.
#[test]
fn new_fonts_inside_the_roots_pass_are_upstreams_behaviour() {
    let c = ctx();
    for k in 0..3 { frame(&c, k as f64 / 60.0, Some(&child_labels), &root_labels); }
    let root = RefCell::new(None);
    let set = |c: &egui::Context| {
        *root.borrow_mut() = Some(atlas(c));
        let mut defs = egui::FontDefinitions::default();
        defs.families.get_mut(&egui::FontFamily::Proportional).unwrap().reverse();
        c.set_fonts(defs);
    };
    frame_with(&c, 0.1, &set, Some(&child_labels), &root_labels);
    assert!(!same(&root.into_inner().unwrap(), &child_atlas()),
            "the child did not take the new definitions at its start, as upstream does");
}

// ── Mixed caps across several nested passes ───────────────────────────────

thread_local! {
    /// Each viewport's requested cap, and per pass: (viewport, requested cap,
    /// the shared `Fonts`' actual cap when its pass began).
    static CAPS: RefCell<std::collections::HashMap<egui::ViewportId, usize>> = RefCell::new(Default::default());
    static STARTS: RefCell<Vec<(&'static str, usize, usize)>> = const { RefCell::new(Vec::new()) };
}

const NAMES: [&str; 4] = ["root", "child", "grandchild", "sibling"];
fn vid(name: &str) -> egui::ViewportId {
    if name == "root" { egui::ViewportId::ROOT } else { egui::ViewportId::from_hash_of(("caps", name)) }
}
fn name_of(id: egui::ViewportId) -> &'static str { NAMES.into_iter().find(|n| vid(n) == id).unwrap() }

fn caps_input(id: egui::ViewportId) -> egui::RawInput {
    let info = |parent| egui::ViewportInfo {
        parent, native_pixels_per_point: Some(1.0),
        inner_rect: Some(egui::Rect::from_min_size(egui::Pos2::ZERO, egui::vec2(400.0, 300.0))),
        ..Default::default()
    };
    let mut viewports = egui::ViewportIdMap::default();
    viewports.insert(vid("root"), info(None));
    viewports.insert(vid("child"), info(Some(vid("root"))));
    viewports.insert(vid("grandchild"), info(Some(vid("child"))));
    viewports.insert(vid("sibling"), info(Some(vid("root"))));
    egui::RawInput {
        viewport_id: id, viewports,
        max_texture_side: CAPS.with(|c| c.borrow().get(&id).copied()),
        screen_rect: Some(egui::Rect::from_min_size(egui::Pos2::ZERO, egui::vec2(400.0, 300.0))),
        ..Default::default()
    }
}

fn record_start(c: &egui::Context, id: egui::ViewportId) {
    let requested = CAPS.with(|m| m.borrow()[&id]);
    STARTS.with(|s| s.borrow_mut().push((name_of(id), requested, c.fonts(|f| f.max_texture_side()))));
}

/// One frame at one scale: root, then a child whose pass shows a grandchild,
/// then a sibling of the child — each with the cap given.
fn caps_frame(c: &egui::Context, caps: [usize; 4]) -> Vec<(&'static str, usize, usize)> {
    CAPS.with(|m| { let mut m = m.borrow_mut(); for (n, cap) in NAMES.into_iter().zip(caps) { m.insert(vid(n), cap); } });
    STARTS.with(|s| s.borrow_mut().clear());
    egui::Context::set_immediate_viewport_renderer(|ctx, vp| {
        let egui::ImmediateViewport { ids, mut viewport_ui_cb, .. } = vp;
        let out = ctx.run(caps_input(ids.this), |c| {
            record_start(c, ids.this);
            viewport_ui_cb(c);
        });
        let _ = ctx.tessellate(out.shapes, out.pixels_per_point);
    });
    let show = |c: &egui::Context, name: &str, inner: &dyn Fn(&egui::Context)| {
        c.show_viewport_immediate(vid(name), egui::ViewportBuilder::default(), |cc, _| {
            egui::CentralPanel::default().show(cc, |ui| { ui.label(format!("{name} text")); });
            inner(cc);
        });
    };
    let out = c.run(caps_input(vid("root")), |c| {
        record_start(c, vid("root"));
        show(c, "child", &|cc| show(cc, "grandchild", &|_| {}));
        show(c, "sibling", &|_| {});
        egui::CentralPanel::default().show(c, |ui| { ui.label("root text"); });
    });
    let _ = c.tessellate(out.shapes, out.pixels_per_point);
    STARTS.with(|s| s.borrow().clone())
}

/// The fallback the patch claims for another cap — the pass begins its own
/// fonts, as upstream does — must hold against the **actual** shared fonts,
/// not against what an ancestor asked for. Root 4096, child 1024 (it
/// recreates the shared fonts at 1024), then a grandchild and a later
/// sibling asking 4096 again, as the root did: each must find fonts of the
/// cap it asked for when its pass begins. Comparing with the root's request
/// alone skipped them and left them on the child's 1024. The same-cap case
/// still shares one set of fonts throughout.
#[test]
fn a_nested_pass_checks_the_shared_fonts_actual_cap() {
    let c = ctx();
    for k in 0..3 {
        let starts = caps_frame(&c, [4096, 1024, 4096, 4096]);
        assert_eq!(starts.iter().map(|s| s.0).collect::<Vec<_>>(), NAMES, "setup: schedule");
        let wrong: Vec<_> = starts.iter().filter(|s| s.1 != s.2).collect();
        assert!(wrong.is_empty(), "frame {k}: passes began on fonts of another cap (name, asked, actual): {wrong:?}");
    }
    // Same cap everywhere: one set of fonts, no recreation between passes.
    let c = ctx();
    caps_frame(&c, [4096; 4]);
    let held = atlas(&c);
    let starts = caps_frame(&c, [4096; 4]);
    assert!(starts.iter().all(|s| s.2 == 4096));
    assert!(same(&held, &atlas(&c)), "same-cap passes recreated the fonts");
}
