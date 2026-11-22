//! The real app rendered offscreen (wgpu, no window), for checking screens
//! without showing anything. Run by hand:
//!   MOOSIK_SHOTS=<folder> cargo test --release screens -- --ignored
//! Uses the test profile, never the owner's.

use egui_kittest::kittest::Queryable;
use egui_kittest::Harness;
use std::path::PathBuf;

fn shot(h: &mut Harness<'_, crate::MoosikApp>, dir: &std::path::Path, name: &str) {
    match h.render() {
        Ok(img) => img.save(dir.join(format!("{name}.png"))).expect("save"),
        Err(e) => panic!("render {name}: {e}"),
    }
}

#[test]
#[ignore]
fn screens() {
    let dir = PathBuf::from(std::env::var("MOOSIK_SHOTS").expect("set MOOSIK_SHOTS"));
    std::fs::create_dir_all(&dir).unwrap();
    let mut h = Harness::builder()
        .with_size(egui::vec2(1300.0, 860.0))
        .wgpu()
        .build_eframe(|cc| crate::MoosikApp::new(cc));
    h.run_steps(20);
    shot(&mut h, &dir, "01-start");
    h.get_by_label("Library").click();
    h.run_steps(30);
    shot(&mut h, &dir, "02-library");
    h.get_by_label("YouTube").click();
    h.run_steps(30);
    shot(&mut h, &dir, "03-youtube");
    h.get_by_label("All controls").click();
    h.run_steps(40);
    shot(&mut h, &dir, "04-settings");
    // Tall enough to show the whole drawer.
    h.set_size(egui::vec2(1300.0, 1900.0));
    h.run_steps(20);
    shot(&mut h, &dir, "04b-settings-tall");
}

/// Library → YouTube with the components in place: a real search (network).
/// MOOSIK_YT_TOOLS = a folder holding yt-dlp.exe and deno.exe.
#[test]
#[ignore]
fn youtube_search_screen() {
    let dir = PathBuf::from(std::env::var("MOOSIK_SHOTS").expect("set MOOSIK_SHOTS"));
    let tools = PathBuf::from(std::env::var("MOOSIK_YT_TOOLS").expect("set MOOSIK_YT_TOOLS"));
    let into = crate::youtube::install::dir(&crate::moosik_dir());
    std::fs::create_dir_all(&into).unwrap();
    for f in ["yt-dlp.exe", "deno.exe"] {
        std::fs::copy(tools.join(f), into.join(f)).unwrap();
    }
    let mut h = Harness::builder()
        .with_size(egui::vec2(1300.0, 860.0))
        .wgpu()
        .build_eframe(|cc| crate::MoosikApp::new(cc));
    h.run_steps(10);
    h.get_by_label("Library").click();
    h.run_steps(20);
    h.get_by_label("YouTube").click();
    h.run_steps(20);
    let field = h.get_by_role(egui::accesskit::Role::TextInput);
    field.focus();
    field.type_text("me at the zoo");
    h.run_steps(2);
    h.press_key(egui::Key::Enter);
    // The search, then the thumbnails, arrive on their own threads.
    let t = std::time::Instant::now();
    while t.elapsed().as_secs() < 30 {
        h.step();
        std::thread::sleep(std::time::Duration::from_millis(200));
        if h.query_all_by_label("Play").count() > 0 && t.elapsed().as_secs() > 6 {
            break;
        }
    }
    h.run_steps(5);
    shot(&mut h, &dir, "05-youtube-results");
    // What sits at a point (for checking overlaps).
    if let Ok(at) = std::env::var("MOOSIK_PROBE") {
        let mut it = at.split(',').map(|v| v.parse::<f64>().unwrap());
        let (x, y) = (it.next().unwrap(), it.next().unwrap());
        for n in h.query_all(egui_kittest::kittest::By::new().predicate(move |n| {
            n.bounding_box().is_some_and(|b| b.x0 <= x && x <= b.x1 && b.y0 <= y && y <= b.y1)
        })) {
            eprintln!("PROBE {:?} {:?} {:?}", n.role(), n.label(), n.bounding_box());
        }
    }
    let _ = std::fs::remove_dir_all(&into);
}

/// Click a control by its label if it is there; say so if not.
fn try_click(h: &mut Harness<'_, crate::MoosikApp>, label: &str) {
    match h.query_by_label(label) {
        Some(n) => n.click(),
        None => eprintln!("SCREENS: no control labelled {label:?}"),
    }
}

/// Analyze: the panels, the controls drawer and the EQ.
#[test]
#[ignore]
fn analyze_screens() {
    let dir = PathBuf::from(std::env::var("MOOSIK_SHOTS").expect("set MOOSIK_SHOTS"));
    std::fs::create_dir_all(&dir).unwrap();
    let mut h = Harness::builder()
        .with_size(egui::vec2(1300.0, 860.0))
        .wgpu()
        .build_eframe(|cc| crate::MoosikApp::new(cc));
    h.run_steps(10);
    try_click(&mut h, "Analyze");
    h.run_steps(30);
    try_click(&mut h, "Harmony");
    h.run_steps(5);
    try_click(&mut h, "Stereo");
    h.run_steps(30);
    shot(&mut h, &dir, "06-analyze-panels");
    // The EQ switch's label carries a box glyph before "EQ".
    let eq = h.query_all(egui_kittest::kittest::By::new().label_contains("EQ")).map(|n| (n.label().unwrap_or_default(), n.role())).collect::<Vec<_>>();
    eprintln!("SCREENS: EQ candidates {eq:?}");
    if let Some(n) = h.query_all(egui_kittest::kittest::By::new().label_contains("EQ")).find(|n| n.label().is_some_and(|l| l.trim_start_matches(|c: char| !c.is_ascii_alphabetic()) == "EQ")) {
        n.click();
    }
    h.run_steps(30);
    shot(&mut h, &dir, "08-eq");
}

/// Info → Measured, Compare: the sections alone, with a measurement and a
/// comparison already done (a real fixture file when MOOSIK_FLAC is set).
#[test]
#[ignore]
fn info_measure_screen() {
    let dir = PathBuf::from(std::env::var("MOOSIK_SHOTS").expect("set MOOSIK_SHOTS"));
    let flac = PathBuf::from(std::env::var("MOOSIK_FLAC").expect("set MOOSIK_FLAC"));
    let (c, p) = (std::sync::atomic::AtomicBool::new(false), std::sync::atomic::AtomicU32::new(0));
    let mut report = crate::measure::measure(&flac, &c, &p).unwrap();
    // Some findings to show.
    report.declared_bits = Some(24);
    report.used_bits = Some(16);
    report.overs = 3;
    report.over_at = vec![12.0, 40.5, 71.0];
    let cmp = crate::measure::compare(&flac, &flac, &c, &p).unwrap();
    let track = crate::Track::load(flac.clone());
    let mut st = crate::measure::State::default();
    st.done = Some((flac.clone(), Ok(report)));
    st.compared = Some((flac.clone(), Ok(cmp)));
    for (dark, name) in [(true, "09-info-measure-dark"), (false, "10-info-measure-light")] {
        let mut h = Harness::builder().with_size(egui::vec2(460.0, 760.0)).wgpu().build_ui(|ui| {
            crate::apply_theme(ui.ctx(), dark);
            egui::ScrollArea::vertical().show(ui, |ui| {
                crate::measure_section(ui, true, &mut st, &track, Some(16_000.0));
            });
        });
        h.run_steps(3);
        shot_ui(&mut h, &dir, name);
    }
}

fn shot_ui(h: &mut Harness<'_>, dir: &std::path::Path, name: &str) {
    match h.render() {
        Ok(img) => img.save(dir.join(format!("{name}.png"))).expect("save"),
        Err(e) => panic!("render {name}: {e}"),
    }
}
