use super::*;

const BARS: usize = 8;
const FFT: usize = 1024;
const RATE: f64 = 180.0;

fn window() -> SpectrumWindow {
    let mut w = SpectrumWindow::new();
    w.analyzer.sample_rate = 48_000;
    w.analyzer.live_rate = 48_000;
    w.analyzer.fft_size = FFT;
    w.analyzer.bar_count = BARS;
    w.bar_count = BARS;
    w.analyzer.rebuild_fft();
    w.analyzer.magnitudes = vec![0.0; BARS];
    w.analyzer.smoothed = vec![0.0; BARS];
    w.analyzer.peak_input = vec![0.0; BARS];
    w.peak_config.enabled = true;
    w.style = VizStyle::Waterfall;
    w
}

fn cache_rows(n: usize) -> Vec<Vec<f32>> {
    (0..n)
        .map(|f| {
            (0..BARS)
                .map(|b| (((f * 7 + b * 13) % 97) as f32) / 97.0)
                .collect()
        })
        .collect()
}

/// The same rows as the analyser would have produced, quantised
/// exactly as a fresh analysis quantises them — so a test sees what
/// the display would see, not a precision the cache never had.
fn cache(n: usize) -> cache::PreFrames {
    cache::PreFrames::from_analysis(&cache_rows(n))
}

fn tone(freq: f32, n: usize) -> Vec<f32> {
    (0..n)
        .map(|i| (std::f32::consts::TAU * freq * i as f32 / 48_000.0).sin() * 0.5)
        .collect()
}

/// Put the window into a state where the renderer has plenty to draw:
/// peaks the peak-hold pass would paint, and a waterfall ring with an
/// upload watermark that says it is already on the GPU.
fn fill_presentation(w: &mut SpectrumWindow) {
    w.analyzer.peak_input = vec![0.9; BARS];
    w.update_peaks(0.0);
    assert!(
        w.peak_vals.iter().any(|&v| v > 0.5),
        "setup: the renderer should have peaks to draw"
    );
    for _ in 0..40 {
        w.analyzer.waterfall_enabled = true;
        w.analyzer.push_waterfall_for_test(vec![0.5; BARS]);
    }
    // Stand in for a completed GPU upload without needing a context.
    w.waterfall_tex_w = BARS;
    w.waterfall_head = 7;
    w.waterfall_uploaded_seq = w.analyzer.waterfall_seq;
    assert!(!w.analyzer.waterfall.is_empty(), "setup: rows to draw");
}

fn assert_nothing_to_render(w: &SpectrumWindow, what: &str) {
    assert!(
        w.peak_vals.iter().all(|&v| v == 0.0),
        "{what}: peak markers from the previous producer are still drawable"
    );
    assert!(
        w.peak_alphas.iter().all(|&a| a == 1.0),
        "{what}: peak fade state survived"
    );
    assert!(
        w.analyzer.waterfall.is_empty(),
        "{what}: waterfall rows from the previous producer survived"
    );
    assert!(
        w.waterfall_texture.is_none(),
        "{what}: the uploaded waterfall texture is still drawable"
    );
    assert_eq!(w.waterfall_head, 0, "{what}: the ring head survived");
    assert_eq!(
        w.waterfall_uploaded_seq, w.analyzer.waterfall_seq,
        "{what}: the upload watermark still claims rows are on the GPU"
    );
    assert!(
        w.analyzer.magnitudes.iter().all(|&m| m == 0.0),
        "{what}: the spectrum survived"
    );
}

/// Stopping must leave the renderer with nothing to paint. `analyzer.reset`
/// alone left peak markers and the GPU ring alive.
#[test]
fn stopping_leaves_nothing_for_the_renderer() {
    let mut w = window();
    fill_presentation(&mut w);
    w.on_stop();
    assert_nothing_to_render(&w, "on_stop");
}

/// Track A's presentation must not survive into track B.
#[test]
fn a_new_track_keeps_none_of_the_previous_ones_presentation() {
    let mut w = window();
    w.on_play(Path::new("C:/music/a.flac"), 48_000);
    fill_presentation(&mut w);
    w.on_play(Path::new("C:/music/b.flac"), 48_000);
    assert_nothing_to_render(&w, "on_play");
}

/// And a seek is the same discontinuity.
#[test]
fn a_seek_keeps_none_of_the_previous_positions_presentation() {
    let mut w = window();
    w.mode = SpectrumMode::RealTime;
    fill_presentation(&mut w);
    w.on_seek(12.0);
    assert_nothing_to_render(&w, "on_seek");
}

/// A cache arriving while the live-FFT fallback has been running is a
/// producer change: the fallback's peaks and rows must not survive it, and
/// the cache itself must not be destroyed by the reset it triggers.
#[test]
fn a_cache_handoff_resets_the_fallback_presentation() {
    let mut w = window();
    w.mode = SpectrumMode::PreProcess;
    // No cache yet: tick_pre falls back to the live FFT.
    {
        let mut b = w.sample_buf.lock().unwrap();
        b.extend_from_slice(&tone(1000.0, FFT * 2));
    }
    w.last_fft_time = None;
    w.tick(0.0, true);
    fill_presentation(&mut w);

    // The worker finishes.
    let frames = cache(2048);
    w.analyzer.set_pre_frames(frames.clone(), RATE);
    w.last_fft_time = None;
    w.analyzer.smoothing = 0.0;
    w.tick(300.0 / RATE, true);

    assert!(
        !w.analyzer.pre_frames.is_empty(),
        "the reset destroyed the cache that caused it"
    );
    assert_eq!(
        w.analyzer.magnitudes, frames.row_vec(300),
        "the first cached tick must snap to the row the clock points at"
    );
    assert!(
        w.peak_vals
            .iter()
            .all(|&v| v <= w.analyzer.peak_input.iter().cloned().fold(0.0, f32::max)),
        "a fallback peak outlived the handoff"
    );
}

/// The same handoff while paused, where nothing is calling `tick_pre` and
/// the only signal is the revision counter.
#[test]
fn a_cache_arriving_while_paused_resets_the_presentation() {
    let mut w = window();
    w.mode = SpectrumMode::PreProcess;
    fill_presentation(&mut w);
    w.analyzer.set_pre_frames(cache(1024), RATE);
    w.tick(5.0, false);
    // Paused, so the new cache's row at the playhead is drawn after the
    // reset: the new producer's, not a survivor of the old one.
    let mut r = window();
    r.analyzer.set_pre_frames(cache(1024), RATE);
    r.analyzer.snap_pre_to(5.0);
    assert!(r.analyzer.magnitudes.iter().any(|&m| m != 0.0), "setup: the new row is not silent");
    assert_eq!(w.analyzer.magnitudes, r.analyzer.magnitudes, "the new row is not drawn while paused");
    assert_eq!(w.analyzer.smoothed, r.analyzer.smoothed, "the smoother is not the new row's");
    // Checked; set aside so everything else can be checked as cleared.
    w.analyzer.magnitudes.fill(0.0);
    w.analyzer.smoothed.fill(0.0);
    w.analyzer.peak_input.fill(0.0);
    assert_nothing_to_render(&w, "a paused cache handoff");
    assert!(!w.analyzer.pre_frames.is_empty(), "the cache was destroyed");
}

/// Dropping a cache is a producer change too.
#[test]
fn clearing_the_cache_resets_the_presentation() {
    let mut w = window();
    w.mode = SpectrumMode::PreProcess;
    w.analyzer.set_pre_frames(cache(1024), RATE);
    w.tick(0.0, false);
    fill_presentation(&mut w);
    w.analyzer.clear_pre_frames();
    w.tick(0.0, false);
    assert_nothing_to_render(&w, "clearing the cache");
}

/// An empty ring means nothing to draw. Returning early left the previous
/// texture in place and `draw_waterfall` went on painting it.
#[test]
fn an_empty_waterfall_discards_the_texture_rather_than_leaving_it_drawable() {
    let mut w = window();
    fill_presentation(&mut w);
    // Model an upload having happened, then the ring being emptied without
    // the texture being told.
    w.waterfall_uploaded_seq = 0;
    w.analyzer.waterfall.clear();
    let ctx = egui::Context::default();
    let pal = Palette::new(w.palette_kind, w.palette_accent);
    let _ = ctx.run(Default::default(), |_| {});
    w.update_waterfall_texture(&ctx, &pal);
    assert!(
        w.waterfall_texture.is_none(),
        "an empty ring left a drawable texture behind"
    );
    assert_eq!(w.waterfall_head, 0);
    assert_eq!(w.waterfall_uploaded_seq, w.analyzer.waterfall_seq);
}

/// Every production write to the cached matrix must go through the two
/// operations that keep the cursor and the revision in step with it.
///
/// Not the only proof — the behaviour above is — but the one that catches a
/// *new* direct write, which no behavioural test can be written for in
/// advance.
#[test]
fn nothing_writes_the_cached_matrix_directly() {
    let src = include_str!("../spectrum.rs");
    // Split so the needles do not match themselves: this test is inside the
    // file it scans, and a literal here would count as an occurrence.
    let assign = concat!("pre_frames", " = ");
    let clear = concat!("pre_frames", ".clear()");
    let assigns = src.matches(assign).count();
    let clears = src.matches(clear).count();
    assert_eq!(
        assigns, 2,
        "`pre_frames` is assigned {assigns} times; the two are \
             `set_pre_frames_with`, which installs a matrix, and \
             `clear_pre_frames`, which releases one"
    );
    assert_eq!(
        clears, 0,
        "`pre_frames` is cleared in place {clears} times; it is shared with \
             the sidecar reader, so it is *replaced* rather than emptied — \
             clearing through the `Arc` would deep-copy the matrix in order to \
             empty it"
    );

    // The channel state has more legitimate writers than the matrix: it is
    // a state machine and its transitions live in the analyser's own
    // methods. The count is pinned so a *new* writer, anywhere, has to be a
    // deliberate act rather than a quiet one.
    let ch_assign = concat!("self.pre_channels", " = ");
    let ch_assigns = src.matches(ch_assign).count();
    assert_eq!(
        ch_assigns, 10,
        "`pre_channels` is assigned {ch_assigns} times; the ten are              `set_pre_frames_with` (install), `clear_pre_frames` (release),              `request_pre_channels` (no path, waiting for the reader slot, and              start), `release_pre_channels` (from resident, and from a read in              flight), and `poll_pre_channels` (resident, wrong shape, refused).              An eleventh is a new way for the channels and the matrix they              belong to to disagree"
    );
}
