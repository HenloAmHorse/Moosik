use super::*;

/// Every row must be shown exactly once, and at the right size.
///
/// The ring is drawn as two quads and the split has to line up with where
/// the texture wraps — get it wrong by a row and the whole display is
/// stretched or a row is drawn twice, which is the sort of thing that looks
/// almost right and never gets noticed.
#[test]
fn slices_tile_the_texture_exactly_once() {
    let h = WATERFALL_ROWS_FALLBACK;
    for head in 0..h {
        let (split, top, bot) = waterfall_slices(head, h);
        assert!((0.0..=1.0).contains(&split), "head {head}: split {split}");
        // The two v-ranges must cover [0,1] with no gap and no overlap.
        assert!((bot.0 - 0.0).abs() < 1e-6);
        assert!((top.1 - 1.0).abs() < 1e-6);
        assert!((bot.1 - top.0).abs() < 1e-6, "head {head}: gap at the seam");
        // Each quad's share of the screen must equal its share of the
        // texture, or its rows are scaled differently from the other's.
        assert!((split - (top.1 - top.0)).abs() < 1e-6,
                "head {head}: top quad stretched");
        assert!(((1.0 - split) - (bot.1 - bot.0)).abs() < 1e-6,
                "head {head}: bottom quad stretched");
    }
}

/// Rows are written backwards, so the newest sits at `head + 1` — which is
/// exactly where the top of the screen starts reading.
#[test]
fn the_newest_row_lands_at_the_top() {
    let h = WATERFALL_ROWS_FALLBACK;
    // Just after writing row `t`, the head has moved back to `t - 1`.
    for t in [0usize, 1, 7, h - 1] {
        let head = (t + h - 1) % h;
        let (_, top, bot) = waterfall_slices(head, h);
        let newest_v = t as f32 / h as f32;
        // The newest row is the first row the top quad shows — unless it is
        // row 0, in which case the top quad is empty and the bottom starts there.
        let starts_at = if top.1 > top.0 { top.0 } else { bot.0 };
        assert!((starts_at - newest_v).abs() < 1e-6,
                "row {t}: screen starts at {starts_at}, newest is at {newest_v}");
    }
}

/// A full wrap must return to where it started, or the ring drifts.
#[test]
fn writing_a_full_ring_returns_the_head() {
    let h = WATERFALL_ROWS_FALLBACK;
    let mut head = 0usize;
    for _ in 0..h { head = (head + h - 1) % h; }
    assert_eq!(head, 0);
}
