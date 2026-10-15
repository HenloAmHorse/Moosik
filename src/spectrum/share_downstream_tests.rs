//! Downstream state with forward-transform sharing (`aslt::run_shared_pass0`):
//! the whole pre-process run through a `SpectrumWindow` — cache, sidecar,
//! channel read, bars drawn at a grid of playheads for Mix and channels, the
//! Diff input and the waterfall the Mix rows build — compared between
//! production (sharing) and the member-by-member pass. The same steps as the
//! round-5 same-rate gate. State-exact, not a raster or framebuffer test.

use super::*;
use std::time::{Duration, Instant};

/// 16-bit stereo WAV of `len` frames: noise-like L, a tone and clicks in R.
fn write_fixture(path: &Path, sr: u32, len: usize) {
    let mut seed = 0x2545_F491_4F6C_DD1Du64;
    let mut data = Vec::with_capacity(len * 4);
    for i in 0..len {
        seed ^= seed << 13; seed ^= seed >> 7; seed ^= seed << 17;
        let l = ((seed >> 49) as i32 - 16_384) as i16;
        let t = i as f32 / sr as f32;
        let click = if i % (sr as usize / 4) < 3 { 20_000.0 } else { 0.0 };
        let r = ((std::f32::consts::TAU * 330.0 * t).sin() * 9_000.0 + click) as i16;
        data.extend(l.to_le_bytes());
        data.extend(r.to_le_bytes());
    }
    let mut f = Vec::new();
    f.extend(b"RIFF");
    f.extend(((36 + data.len()) as u32).to_le_bytes());
    f.extend(b"WAVEfmt ");
    f.extend(16u32.to_le_bytes());
    f.extend(1u16.to_le_bytes());
    f.extend(2u16.to_le_bytes());
    f.extend(sr.to_le_bytes());
    f.extend((sr * 4).to_le_bytes());
    f.extend(4u16.to_le_bytes());
    f.extend(16u16.to_le_bytes());
    f.extend(b"data");
    f.extend((data.len() as u32).to_le_bytes());
    f.extend(data);
    std::fs::write(path, f).unwrap();
}

/// Everything downstream of the analysis, as bytes.
#[derive(PartialEq, Eq, Debug)]
struct Downstream {
    mix: Vec<u16>,
    left: Vec<u16>,
    right: Vec<u16>,
    diff_input: Vec<i32>,
    mix_bars: Vec<u32>,
    channel_bars: Vec<u32>,
    waterfall: Vec<u32>,
    waterfall_rows: usize,
}

fn codes(f: &cache::PreFrames) -> Vec<u16> {
    (0..f.len()).flat_map(|i| f.row(i).unwrap_or(&[]).to_vec()).collect()
}

fn downstream(wav: &Path, cache_dir: &Path, bars: usize) -> Downstream {
    let _ = std::fs::remove_dir_all(cache_dir);
    let mut w = SpectrumWindow::new_isolated();
    w.mode = SpectrumMode::PreProcess;
    w.style = VizStyle::Bars;
    w.analyzer.cache_dir_override = Some(cache_dir.to_path_buf());
    w.analyzer.cache_budget_gb = 0.0;
    w.analyzer.pre_stereo_enabled = true;
    w.bar_count = bars;
    w.analyzer.bar_count = bars;
    w.bar_mapping = BarMappingMode::Superlet;
    w.analyzer.bar_mapping = BarMappingMode::Superlet;
    w.min_freq = 20.0;
    w.max_freq = 24_000.0;
    w.analyzer.min_freq = 20.0;
    w.analyzer.max_freq = 24_000.0;
    w.analyzer.aslt_preset = None;
    let mut c = aslt::AsltPreset::Standard.config();
    c.n_wavelets = 9;
    c.q_ratio = 2.0;
    c.scale = freq_scale::FreqScale::Blend(0.5);
    c.spread = 0.5;
    c.max_window_s = 0.25;
    w.analyzer.aslt_cfg = c;
    w.analyzer.pre_fps = 180.0;
    w.on_play(wav, 44_100);
    let deadline = Instant::now() + Duration::from_secs(600);
    let mut quiet = 0;
    while quiet < 3 {
        w.last_fft_time = Some(Instant::now() - Duration::from_millis(20));
        w.tick(0.0, false);
        quiet = if w.analyzer.analysis_idle() { quiet + 1 } else { 0 };
        assert!(Instant::now() < deadline, "never settled");
        std::thread::sleep(Duration::from_millis(5));
    }
    let mono = (*w.analyzer.pre_frames).clone();
    assert!(!mono.is_empty(), "nothing installed: {}", w.status_msg);
    w.channel_view = channels::ChannelView::Split;
    let until = Instant::now() + Duration::from_secs(120);
    while !w.analyzer.pre_channels_resident() && Instant::now() < until {
        w.last_fft_time = Some(Instant::now() - Duration::from_millis(20));
        w.tick(0.0, false);
        std::thread::sleep(Duration::from_millis(5));
    }
    let pair = w.analyzer.pre_stereo().expect("channels resident").clone();
    let (left, right) = (codes(pair.left()), codes(pair.right()));
    let diff_input = left.iter().zip(&right).map(|(a, b)| *a as i32 - *b as i32).collect();
    w.analyzer.waterfall_enabled = true;
    let secs = mono.len() as f64 / 180.0;
    let (mut mix_bars, mut channel_bars) = (Vec::new(), Vec::new());
    let mut t = 0.0;
    while t < secs {
        w.analyzer.snap_pre_to(t);
        mix_bars.extend(w.analyzer.magnitudes.iter().map(|m| m.to_bits()));
        channel_bars.extend(w.analyzer.bars_left.iter().chain(&w.analyzer.bars_right).map(|m| m.to_bits()));
        w.analyzer.push_waterfall_row();
        t += 0.0137;
    }
    let waterfall = w.analyzer.waterfall.iter().flatten().map(|v| v.to_bits()).collect();
    let waterfall_rows = w.analyzer.waterfall.len();
    let _ = std::fs::remove_dir_all(cache_dir);
    Downstream { mix: codes(&mono), left, right, diff_input, mix_bars, channel_bars, waterfall, waterfall_rows }
}

/// The fixture is forced onto the own-transform route (its block sets
/// refused, as a long track's are) by length; the reference run also turns
/// sharing off. Everything downstream must be the same.
#[test]
fn shared_pass_leaves_every_downstream_state_unchanged() {
    // An unusual length, so the override picks out this fixture only.
    const LEN: usize = 110_263;
    let dir = std::env::temp_dir().join(format!("moosik_share_downstream_{}", std::process::id()));
    std::fs::create_dir_all(&dir).unwrap();
    let wav = dir.join("fixture.wav");
    write_fixture(&wav, 44_100, LEN);
    let _calib = gpu_calib::isolated_test_env("share_downstream");
    let set = |share| aslt::route_override::set(aslt::route_override::Override { len: LEN, shared_budget: Some(0), share, full_inverse: false });
    let _ = aslt::route_override::take_work(LEN);
    set(false);
    let want = downstream(&wav, &dir.join("cache_ref"), 96);
    let ref_work = aslt::route_override::take_work(LEN);
    set(true);
    let got = downstream(&wav, &dir.join("cache_share"), 96);
    let share_work = aslt::route_override::take_work(LEN);
    aslt::route_override::clear(LEN);
    assert!(!ref_work.is_empty() && ref_work.iter().all(|w| w.shared_members == 0), "reference shared: {ref_work:?}");
    assert!(share_work.iter().any(|w| w.shared_members > 0), "the pre-process did not share: {share_work:?}");
    let _ = std::fs::remove_dir_all(&dir);
    assert!(want.waterfall_rows > 0 && !want.channel_bars.is_empty(), "setup: no display state");
    assert!(want.mix.iter().any(|&c| c > 0), "setup: silent analysis");
    assert_eq!(got, want, "downstream state differs");
}

/// The same whole pre-process and downstream replay through production (the
/// pruned inverse with the fused product, per-member scheduling of shared
/// windows) against the original full inverse, on the production sharing
/// route. State-exact: codes, Diff input, bars at a playhead grid for Mix and
/// channels, every waterfall row — not a rendered-pixel comparison. The work
/// counts witness which inverse each pre-process took.
#[test]
fn the_pruned_inverse_leaves_every_downstream_state_unchanged() {
    const LEN: usize = 110_291;
    let dir = std::env::temp_dir().join(format!("moosik_r10_downstream_{}", std::process::id()));
    std::fs::create_dir_all(&dir).unwrap();
    let wav = dir.join("fixture.wav");
    write_fixture(&wav, 44_100, LEN);
    let _calib = gpu_calib::isolated_test_env("r10_downstream");
    let set = |full_inverse| aslt::route_override::set(aslt::route_override::Override { len: LEN, shared_budget: Some(0), share: true, full_inverse });
    let _ = aslt::route_override::take_work(LEN);
    set(true);
    let want = downstream(&wav, &dir.join("cache_ref"), 96);
    let ref_work = aslt::route_override::take_work(LEN);
    set(false);
    let got = downstream(&wav, &dir.join("cache_r10"), 96);
    let work = aslt::route_override::take_work(LEN);
    aslt::route_override::clear(LEN);
    let _ = std::fs::remove_dir_all(&dir);
    assert!(!ref_work.is_empty() && ref_work.iter().all(|w| w.inverses_pruned == 0 && w.inverses_full > 0),
            "the reference did not take the whole inverse: {ref_work:?}");
    if is_x86_feature_detected!("avx") && is_x86_feature_detected!("fma") {
        assert!(!work.is_empty() && work.iter().all(|w| w.inverses_full == 0 && w.inverses_pruned > 0),
                "production did not take the pruned inverse: {work:?}");
    }
    assert!(work.iter().any(|w| w.shared_members > 0), "the pre-process did not share: {work:?}");
    assert!(want.waterfall_rows > 0 && !want.channel_bars.is_empty(), "setup: no display state");
    assert!(want.mix.iter().any(|&c| c > 0), "setup: silent analysis");
    assert_eq!(got, want, "downstream state differs");
}

/// Round 12-E: the same whole pre-process and downstream replay at the owner's
/// 1024 bars, production (the pruned inverse, now ending in the
/// requested-output Butterfly256/512 leaves wherever a base call asks for 1–4
/// outputs) against the original full inverse, which never calls a leaf's
/// pruned path. State-exact: codes, Diff input, bars at a playhead grid for
/// Mix and channels, every waterfall row — display *state*, neither a mesh
/// nor a pixel comparison. Which leaves a production run reaches is shown
/// separately, by the diagnostic-build tests.
#[test]
fn the_requested_output_leaves_leave_every_downstream_state_unchanged() {
    const LEN: usize = 110_357;
    let dir = std::env::temp_dir().join(format!("moosik_r12e_downstream_{}", std::process::id()));
    std::fs::create_dir_all(&dir).unwrap();
    let wav = dir.join("fixture.wav");
    write_fixture(&wav, 44_100, LEN);
    let _calib = gpu_calib::isolated_test_env("r12e_downstream");
    let set = |full_inverse| aslt::route_override::set(aslt::route_override::Override { len: LEN, shared_budget: Some(0), share: true, full_inverse });
    let _ = aslt::route_override::take_work(LEN);
    set(true);
    let want = downstream(&wav, &dir.join("cache_ref"), 1024);
    let ref_work = aslt::route_override::take_work(LEN);
    set(false);
    let got = downstream(&wav, &dir.join("cache_leaf"), 1024);
    let work = aslt::route_override::take_work(LEN);
    aslt::route_override::clear(LEN);
    let _ = std::fs::remove_dir_all(&dir);
    assert!(!ref_work.is_empty() && ref_work.iter().all(|w| w.inverses_pruned == 0 && w.inverses_full > 0),
            "the reference did not take the whole inverse: {ref_work:?}");
    if is_x86_feature_detected!("avx") && is_x86_feature_detected!("fma") {
        assert!(!work.is_empty() && work.iter().all(|w| w.inverses_full == 0 && w.inverses_pruned > 0),
                "production did not take the pruned inverse: {work:?}");
    }
    assert!(want.waterfall_rows > 0 && !want.channel_bars.is_empty(), "setup: no display state");
    assert!(want.mix.iter().any(|&c| c > 0), "setup: silent analysis");
    assert_eq!(got, want, "downstream state differs");
}
