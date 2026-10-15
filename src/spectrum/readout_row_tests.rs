use super::*;

fn plot(w: f32) -> Rect {
    let outer = Rect::from_min_size(Pos2::new(12.0, 8.0), egui::vec2(w, 460.0));
    Rect::from_min_max(
        Pos2::new(outer.left() + DB_MARGIN, outer.top()),
        Pos2::new(outer.right(), outer.bottom() - FREQ_MARGIN),
    )
}

fn capture(f: impl FnOnce(&egui::Painter)) -> Vec<egui::epaint::ClippedShape> {
    let ctx = egui::Context::default();
    let input = egui::RawInput {
        screen_rect: Some(Rect::from_min_size(Pos2::ZERO, egui::vec2(1600.0, 900.0))),
        ..Default::default()
    };
    let mut f = Some(f);
    let out = ctx.run(input, |ctx| {
        let painter = ctx.layer_painter(egui::LayerId::new(
            egui::Order::Foreground,
            egui::Id::new("readout_row_tests"),
        ));
        if let Some(f) = f.take() {
            f(&painter);
        }
    });
    out.shapes
}

/// Every text box drawn, in order.
fn boxes(shapes: &[egui::epaint::ClippedShape]) -> Vec<(String, Rect)> {
    shapes
        .iter()
        .filter_map(|c| match &c.shape {
            egui::Shape::Text(t) => Some((
                t.galley.text().to_string(),
                Rect::from_min_size(t.pos, t.galley.size()),
            )),
            _ => None,
        })
        .collect()
}

/// Draw the whole row exactly as production does, and return what landed.
///
/// Including the analysis badge's background, which is the part that
/// actually collides — it is wider than its text and it is what was landing
/// on the frame rate.
fn draw_row(
    width: f32,
    lufs: Option<&str>,
    fps: &str,
    view: channels::ChannelView,
) -> Vec<(String, Rect)> {
    draw_row_with(width, lufs, fps, view, None)
}

fn draw_row_with(
    width: f32,
    lufs: Option<&str>,
    fps: &str,
    view: channels::ChannelView,
    analysis: Option<&str>,
) -> Vec<(String, Rect)> {
    let r = plot(width);
    let lufs = lufs.map(|s| s.to_string());
    let fps = fps.to_string();
    let analysis = analysis.map(|s| s.to_string());
    let shapes = capture(move |p| {
        let row = TopRow::plan(p, r, lufs.as_deref(), &fps, analysis.as_deref());
        let w = SpectrumWindow::new();
        // The legend, through the production function.
        match view {
            channels::ChannelView::Left => {
                w.label_channel(p, Pos2::new(row.legend_x, row.y), row.limit_x, "L", CHANNEL_L_COLOR);
            }
            channels::ChannelView::Overlay => {
                let used = w.label_channel(
                    p,
                    Pos2::new(row.legend_x, row.y),
                    row.limit_x,
                    "L",
                    CHANNEL_L_COLOR,
                );
                w.label_channel(
                    p,
                    Pos2::new(row.legend_x + used + 6.0, row.y),
                    row.limit_x,
                    "R",
                    CHANNEL_R_COLOR,
                );
            }
            _ => {}
        }
        // Then the two fixed readouts, as the caller draws them.
        p.text(
            Pos2::new(row.fps_x, row.y),
            egui::Align2::RIGHT_TOP,
            &fps,
            readout_font(),
            Color32::WHITE,
        );
        if let Some(l) = &lufs {
            p.text(
                Pos2::new(row.lufs_x, row.y),
                egui::Align2::LEFT_TOP,
                l,
                readout_font(),
                Color32::WHITE,
            );
        }
        // The badge, positioned and sized exactly as production does it.
        if let Some(t) = &analysis {
            let pos = Pos2::new(row.analysis_x, row.y + 2.0);
            let gal = p.layout_no_wrap(t.clone(), badge_font(), Color32::WHITE);
            let bg = Rect::from_min_size(
                Pos2::new(pos.x - gal.size().x - BADGE_PAD, pos.y - 3.0),
                gal.size() + egui::vec2(2.0 * BADGE_PAD, 6.0),
            );
            // Drawn as a filled rectangle in production; represented here by
            // a text shape carrying the same box, so one disjointness check
            // covers everything in the row.
            p.rect_filled(bg, 3.0, Color32::from_rgba_unmultiplied(0, 0, 0, 120));
            p.galley(
                Pos2::new(bg.left() + BADGE_PAD, bg.top() + 3.0),
                gal,
                Color32::WHITE,
            );
        }
    });
    let mut out = boxes(&shapes);
    // The badge is one visual object -- a background with its text inside
    // it -- so it becomes one item, sized by the background, which is the
    // part that actually collides. Comparing its own text against its own
    // background would report an overlap that is the design.
    let badge_bg: Option<Rect> = shapes.iter().find_map(|c| match &c.shape {
        egui::Shape::Rect(r) if r.rect.width() > 20.0 => Some(r.rect),
        _ => None,
    });
    if let Some(bg) = badge_bg {
        out.retain(|(_, b)| !bg.contains_rect(*b));
        out.push(("<badge>".to_string(), bg));
    }
    out
}

fn assert_disjoint(items: &[(String, Rect)], what: &str) {
    for i in 0..items.len() {
        for j in (i + 1)..items.len() {
            let (an, ar) = &items[i];
            let (bn, br) = &items[j];
            assert!(
                !ar.intersects(*br),
                "{what}: {an:?} at {ar:?} overlaps {bn:?} at {br:?}"
            );
        }
    }
}

#[test]
fn the_readouts_do_not_overlap_each_other() {
    // The default window is 700 px wide; the widest LUFS string and a
    // three-digit frame rate are the worst case.
    for width in [340.0f32, 500.0, 700.0, 900.0, 1400.0] {
        for lufs in [Some("-23.4 LUFS"), Some("— LUFS"), None] {
            for fps in ["9 fps", "60 fps", "240 fps"] {
                for view in [
                    channels::ChannelView::Left,
                    channels::ChannelView::Overlay,
                    channels::ChannelView::Mix,
                ] {
                    let items = draw_row(width, lufs, fps, view);
                    assert_disjoint(
                        &items,
                        &format!("width {width}, lufs {lufs:?}, fps {fps}, {view:?}"),
                    );
                }
            }
        }
    }
}

/// The specific collision the owner reported: the LUFS number and the
/// channel legend, at the size the window actually opens at.
#[test]
fn lufs_and_the_channel_legend_are_side_by_side() {
    let items = draw_row(700.0, Some("-23.4 LUFS"), "180 fps", channels::ChannelView::Left);
    let lufs = items
        .iter()
        .find(|(t, _)| t.contains("LUFS"))
        .expect("no LUFS readout");
    let legend = items.iter().find(|(t, _)| t == "L").expect("no legend");
    let fps = items
        .iter()
        .find(|(t, _)| t.contains("fps"))
        .expect("no fps readout");
    assert!(
        lufs.1.right() <= legend.1.left(),
        "LUFS {:?} still runs into the legend {:?}",
        lufs.1,
        legend.1
    );
    assert!(
        legend.1.right() <= fps.1.left(),
        "the legend {:?} still runs into the frame rate {:?}",
        legend.1,
        fps.1
    );
    // All three on the same line, which is the point of the row.
    assert!((lufs.1.top() - legend.1.top()).abs() < 4.0);
    assert!((lufs.1.top() - fps.1.top()).abs() < 4.0);
}


/// The owner's second report: the analysis badge landed on the frame rate.
///
/// It was anchored to the widget rectangle's top-right corner, which after
/// the row was introduced is where the frame rate sits. The badge is the
/// widest thing in the row and it comes and goes, so it takes the right end
/// and the frame rate moves left of it — rather than the frame rate jumping
/// sideways whenever an analysis starts.
#[test]
fn the_analysis_badge_does_not_land_on_the_frame_rate() {
    for width in [420.0f32, 700.0, 900.0, 1400.0] {
        for analysis in [
            "analysing 0%",
            "analysing 47%  ~1m 12s",
            "analysing 100%  ~12m 30s",
        ] {
            for lufs in [Some("-23.4 LUFS"), None] {
                for view in [channels::ChannelView::Left, channels::ChannelView::Mix] {
                    let items = draw_row_with(width, lufs, "240 fps", view, Some(analysis));
                    assert_disjoint(
                        &items,
                        &format!("width {width}, {analysis:?}, lufs {lufs:?}, {view:?}"),
                    );
                    // And the badge really was drawn, so this is not
                    // passing by drawing nothing.
                    assert!(
                        items.iter().any(|(t, _)| t == "<badge>"),
                        "no badge at width {width}"
                    );
                    assert!(
                        items.iter().any(|(t, _)| t.contains("fps")),
                        "no frame rate at width {width}"
                    );
                }
            }
        }
    }
}

/// The frame rate stays put when no analysis is running, and only moves
/// left when one starts. A readout that jumped about would be worse than
/// the overlap.
#[test]
fn the_frame_rate_only_moves_to_make_room_for_the_badge() {
    let idle = draw_row_with(900.0, Some("-23.4 LUFS"), "60 fps", channels::ChannelView::Left, None);
    let busy = draw_row_with(
        900.0,
        Some("-23.4 LUFS"),
        "60 fps",
        channels::ChannelView::Left,
        Some("analysing 47%  ~1m 12s"),
    );
    let fps_of = |v: &Vec<(String, Rect)>| {
        v.iter().find(|(t, _)| t.contains("fps")).expect("no fps").1
    };
    let a = fps_of(&idle);
    let b = fps_of(&busy);
    assert!(
        b.right() < a.right(),
        "the frame rate should shift left for the badge: {a:?} then {b:?}"
    );
    // The LUFS readout does not move; it is anchored to the other end.
    let lufs_of = |v: &Vec<(String, Rect)>| {
        v.iter().find(|(t, _)| t.contains("LUFS")).expect("no LUFS").1
    };
    assert_eq!(lufs_of(&idle).left(), lufs_of(&busy).left());
}
/// The row lives inside the plot, so the dB margin stays free for the axis
/// labels a Diff plot draws there.
#[test]
fn the_row_does_not_intrude_on_the_db_margin() {
    let r = plot(900.0);
    let items = draw_row(900.0, Some("-23.4 LUFS"), "180 fps", channels::ChannelView::Left);
    for (t, b) in &items {
        assert!(
            b.left() >= r.left(),
            "{t:?} at {b:?} reaches into the dB margin left of {}",
            r.left()
        );
        assert!(b.right() <= r.right() + 0.5, "{t:?} at {b:?} runs off the plot");
    }
}

/// When there is no room the legend is dropped rather than drawn over the
/// frame rate. A missing legend is recoverable by widening the window; two
/// numbers on top of each other are not readable at all.
#[test]
fn a_window_too_narrow_for_the_legend_drops_it() {
    let items = draw_row(150.0, Some("-23.4 LUFS"), "240 fps", channels::ChannelView::Left);
    assert_disjoint(&items, "narrow");
    assert!(
        !items.iter().any(|(t, _)| t == "L"),
        "the legend was drawn with no room for it: {items:?}"
    );
    // The two numbers still are.
    assert!(items.iter().any(|(t, _)| t.contains("LUFS")));
    assert!(items.iter().any(|(t, _)| t.contains("fps")));
}

/// Overlay draws two legends, and the second follows the measured width of
/// the first rather than a fixed 14 px guess.
#[test]
fn the_overlay_legend_pair_never_touches() {
    let items = draw_row(900.0, Some("-23.4 LUFS"), "60 fps", channels::ChannelView::Overlay);
    assert_disjoint(&items, "overlay");
    let l = items.iter().find(|(t, _)| t == "L").expect("no L");
    let r = items.iter().find(|(t, _)| t == "R").expect("no R");
    assert!(l.1.right() <= r.1.left(), "L {:?} runs into R {:?}", l.1, r.1);
}
