//! Round 13 cache compatibility/readback harness (ignored; run alone under
//! the fail-closed runner, isolated profile). Two halves, built into both the
//! K0 reference and the candidate executables so each can read the other's
//! files:
//! - `r13_cache_produce`: fresh pre-process of `MOOSIK_R13_TRACK` at the
//!   owner's descriptor into the private cache directory `MOOSIK_R13_CACHE`
//!   (Mix cache and channel sidecar), then prints the installed code hashes;
//! - `r13_cache_consume`: plays the same track against a private *copy* of a
//!   cache directory written by either executable and prints what the shipped
//!   loader installed, whether any analysis job ran, and the code hashes.
//!
//! Byte identity of whole containers is checked outside, by SHA-256 of the
//! files themselves.
//!
//! R13 closeout: the consume half is a gate, not a report. It fails (writes
//! `"ok":false` with the reason, then panics) unless: no analysis ever starts
//! (checked every frame; the first sign stops the window), the installed
//! result is the requested track at the selected settings with resident
//! channels, the frame count and Mix/Left/Right codes equal the trusted
//! counterpart's (`MOOSIK_R13_EXPECT_FRAMES`/`_MIX`/`_LEFT`/`_RIGHT`, all
//! required), and the consumed directory is byte-identical afterwards. Each
//! result line carries `MOOSIK_TAG` so the runner can require exactly one.

use super::*;
use std::time::{Duration, Instant};

/// The rate the engine reports for the track (`MOOSIK_R13_RATE`, default 44100).
fn rate() -> u32 {
    std::env::var("MOOSIK_R13_RATE").ok().and_then(|v| v.parse().ok()).unwrap_or(44_100)
}

fn hash_frames(p: &cache::PreFrames) -> (usize, String) {
    let mut h = 0xcbf29ce484222325u64;
    for i in 0..p.len() {
        for v in p.row(i).unwrap_or(&[]) {
            h = (h ^ *v as u64).wrapping_mul(0x100000001b3);
        }
    }
    (p.len(), format!("{h:016x}"))
}

fn window(cache: &Path) -> SpectrumWindow {
    let mut w = SpectrumWindow::new_isolated();
    w.mode = SpectrumMode::PreProcess;
    w.style = VizStyle::Bars;
    w.analyzer.cache_dir_override = Some(cache.to_path_buf());
    w.analyzer.cache_budget_gb = 0.0;
    w.analyzer.pre_stereo_enabled = true;
    w.bar_count = 1024;
    w.analyzer.bar_count = 1024;
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
    w.analyzer.dsd_rate = 88_200;
    w.channel_view = channels::ChannelView::Split;
    w
}

/// Ticks until nothing is running or waiting and the channels are resident;
/// returns whether any analysis job was seen active.
fn settle(w: &mut SpectrumWindow) -> bool {
    let until = Instant::now() + Duration::from_secs(1800);
    let (mut quiet, mut saw_job) = (0, false);
    while quiet < 3 || !w.analyzer.pre_channels_resident() {
        w.last_fft_time = Some(Instant::now() - Duration::from_millis(20));
        w.tick(0.0, false);
        saw_job |= w.analyzer.analysis_busy();
        quiet = if w.analyzer.analysis_idle() { quiet + 1 } else { 0 };
        assert!(Instant::now() < until, "never settled");
        std::thread::sleep(Duration::from_millis(5));
    }
    saw_job
}

fn emit(line: &str) {
    println!("R13CACHE {line}");
    if let Ok(p) = std::env::var("MOOSIK_R13_CACHE_OUT") {
        use std::io::Write;
        writeln!(std::fs::OpenOptions::new().create(true).append(true).open(p).unwrap(), "{line}").unwrap();
    }
}

fn tag() -> String {
    std::env::var("MOOSIK_TAG").unwrap_or_default()
}

fn report(half: &str, w: &SpectrumWindow, saw_job: bool) {
    let mix = hash_frames(&w.analyzer.pre_frames);
    let pair = w.analyzer.pre_stereo().expect("channels resident").clone();
    let (l, r) = (hash_frames(pair.left()), hash_frames(pair.right()));
    emit(&format!("{{\"tag\":{},\"half\":\"{half}\",\"ok\":true,\"analysis_ran\":{saw_job},\"rate\":{},\"frames\":{},\"mix\":\"{}\",\"left\":\"{}\",\"right\":\"{}\",\"status\":{}}}",
                  serde_json::Value::String(tag()), rate(), mix.0, mix.1, l.1, r.1, serde_json::Value::String(w.status_msg.clone())));
}

/// Every file in `dir` with its bytes.
fn snapshot(dir: &Path) -> std::collections::BTreeMap<String, Vec<u8>> {
    std::fs::read_dir(dir).map(|rd| rd.flatten()
        .filter(|e| e.path().is_file())
        .map(|e| (e.file_name().to_string_lossy().into_owned(), std::fs::read(e.path()).unwrap()))
        .collect()).unwrap_or_default()
}

#[test]
#[ignore]
fn r13_cache_produce() {
    let track = PathBuf::from(std::env::var("MOOSIK_R13_TRACK").expect("MOOSIK_R13_TRACK"));
    let cache = PathBuf::from(std::env::var("MOOSIK_R13_CACHE").expect("MOOSIK_R13_CACHE"));
    std::fs::create_dir_all(&cache).unwrap();
    assert!(std::fs::read_dir(&cache).unwrap().next().is_none(), "cache directory not empty: would measure reuse");
    let _calib = gpu_calib::isolated_test_env("r13_cache");
    let mut w = window(&cache);
    w.on_play(&track, rate());
    let saw = settle(&mut w);
    assert!(saw, "a fresh produce ran no analysis");
    report("produce", &w, saw);
    w.on_stop();
}

/// The consume gate (see the module comment).
/// Returns the result line's fields (frames, mix, left, right, status).
fn consume_checked(w: &mut SpectrumWindow, track: &Path, cache: &Path) -> Result<(usize, String, String, String, String), String> {
    let need = |k: &str| std::env::var(k).map_err(|_| format!("{k} not set: a consume needs its trusted counterpart"));
    let want_frames: usize = need("MOOSIK_R13_EXPECT_FRAMES")?.parse().map_err(|e| format!("MOOSIK_R13_EXPECT_FRAMES: {e}"))?;
    let want = (need("MOOSIK_R13_EXPECT_MIX")?, need("MOOSIK_R13_EXPECT_LEFT")?, need("MOOSIK_R13_EXPECT_RIGHT")?);
    let before = snapshot(cache);
    if before.is_empty() {
        return Err("consume directory is empty".into());
    }
    w.on_play(track, rate());
    let until = Instant::now() + Duration::from_secs(300);
    let mut quiet = 0;
    while quiet < 3 || !w.analyzer.pre_channels_resident() {
        w.last_fft_time = Some(Instant::now() - Duration::from_millis(20));
        w.tick(0.0, false);
        if w.analyzer.analysis_busy() {
            w.on_stop();
            return Err("an analysis started during an intended consume".into());
        }
        quiet = if w.analyzer.analysis_idle() { quiet + 1 } else { 0 };
        if Instant::now() > until {
            return Err(format!("not installed with resident channels after 300 s (resident {})", w.analyzer.pre_channels_resident()));
        }
        std::thread::sleep(Duration::from_millis(5));
    }
    let want_path = w.current_cache_path();
    if w.analyzer.pre_cache_path().map(Path::to_path_buf) != want_path {
        return Err(format!("installed {:?} is not the requested track/settings' {want_path:?}", w.analyzer.pre_cache_path()));
    }
    let mix = hash_frames(&w.analyzer.pre_frames);
    let pair = w.analyzer.pre_stereo().ok_or("channels not resident")?.clone();
    let (l, r) = (hash_frames(pair.left()), hash_frames(pair.right()));
    if (mix.0, pair.left().len(), pair.right().len()) != (want_frames, want_frames, want_frames) {
        return Err(format!("frames mix {} left {} right {}, want {want_frames}", mix.0, pair.left().len(), pair.right().len()));
    }
    if (mix.1.as_str(), l.1.as_str(), r.1.as_str()) != (want.0.as_str(), want.1.as_str(), want.2.as_str()) {
        return Err(format!("codes {:?} differ from trusted {want:?}", (mix.1, l.1, r.1)));
    }
    let status = w.status_msg.clone();
    w.on_stop();
    if snapshot(cache) != before {
        return Err("the consumed directory changed".into());
    }
    Ok((mix.0, mix.1, l.1, r.1, status))
}

#[test]
#[ignore]
fn r13_cache_consume() {
    let track = PathBuf::from(std::env::var("MOOSIK_R13_TRACK").expect("MOOSIK_R13_TRACK"));
    let cache = PathBuf::from(std::env::var("MOOSIK_R13_CACHE").expect("MOOSIK_R13_CACHE"));
    let _calib = gpu_calib::isolated_test_env("r13_cache");
    let mut w = window(&cache);
    match consume_checked(&mut w, &track, &cache) {
        Ok((frames, mix, left, right, status)) => {
            emit(&format!("{{\"tag\":{},\"half\":\"consume\",\"ok\":true,\"analysis_ran\":false,\"rate\":{},\"frames\":{frames},\"mix\":\"{mix}\",\"left\":\"{left}\",\"right\":\"{right}\",\"status\":{}}}",
                          serde_json::Value::String(tag()), rate(), serde_json::Value::String(status)));
        }
        Err(e) => {
            emit(&format!("{{\"tag\":{},\"half\":\"consume\",\"ok\":false,\"rate\":{},\"what\":{}}}",
                          serde_json::Value::String(tag()), rate(), serde_json::Value::String(e.clone())));
            panic!("consume gate: {e}");
        }
    }
}
