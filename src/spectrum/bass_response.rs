//! Explicit, ignored research probe: temporal and frequency response of the
//! stored analysis, measured through the production pre-process entry.
//!
//! Required: MOOSIK_GPU=0, MOOSIK_BASS_ARM=aslt1|aslt9|cqt<BPO>|vqt<BPO>t<seconds> and
//! MOOSIK_BASS_FIXTURE=gate|pair (click: see below). Run only this test, in release, with
//! --ignored --nocapture --test-threads=1; capture the BASS_RESPONSE JSON line.
//!
//! Fixtures are synthetic 48 kHz mono-content (L = R) PCM16, 40 s, with at
//! least 15 s of silence around every event (longer than half of the longest
//! nominal support, 6.9 s at 20 Hz and 192 bins/octave):
//!
//! * gate:  tones at 40, 160, 640 and 2560 Hz, each 0.2 FS, on 15.0..25.0 s.
//! * click: one full-scale-ish sample (0.9) at 20.0 s. Invalid as designed:
//!   too little energy per narrow band, so bass bars sit at the code floor
//!   (see the report); kept for the record.
//! * pair: tone pairs one semitone apart (40/42.38, 160/169.5, 640/678 Hz),
//!   each 0.1 FS, steady 0..40 s.
//!
//! What is measured is the stored Mix code plane (12-bit levels, 80 dB over
//! 4095 codes, so code differences are dB differences). This is the analysis
//! definition plus quantization, BEFORE any display-time smoothing/history;
//! those are separate and not applied here. Levels are relative; the display
//! reference is not calibrated dBFS. Frame k's time is its centre sample / sr
//! (ASLT: k·hop; CQT: floor(k·sr/fps)).
//!
//! The arms are selectable settings, NOT equal-response settings: ASLT uses
//! Blend(0.5), q_ratio 2, max window 0.25 s, spread 0.5 (the owner's usual);
//! CQT uses a log grid and Blackman at 180 fps. The probe's purpose is to
//! state what each one does, so a later matched comparison can be designed.

use super::*;
use std::io::{BufWriter, Write};

const SR: u32 = 48_000;
const SECONDS: u32 = 40;
const DB_PER_CODE: f64 = 80.0 / 4095.0;

struct Fixture {
    root: PathBuf,
    temp: PathBuf,
}

impl Fixture {
    fn new() -> Self {
        let temp = std::fs::canonicalize(std::env::temp_dir()).expect("resolve OS temp directory");
        let nonce = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .expect("clock before epoch")
            .as_nanos();
        let requested = temp.join(format!("moosik_bass_{}_{nonce:x}", std::process::id()));
        std::fs::create_dir(&requested).expect("create a fresh private fixture");
        let root = std::fs::canonicalize(&requested).expect("resolve private fixture");
        assert_eq!(root.parent(), Some(temp.as_path()));
        Self { root, temp }
    }
}

impl Drop for Fixture {
    fn drop(&mut self) {
        if self.root.exists() && self.root.parent() == Some(self.temp.as_path()) {
            let _ = std::fs::remove_dir_all(&self.root);
        }
    }
}

fn sample(fixture: &str, n: u64) -> f64 {
    let t = n as f64 / f64::from(SR);
    let tone = |f: f64| (std::f64::consts::TAU * f * t).sin();
    match fixture {
        "gate" => {
            if (15.0..25.0).contains(&t) {
                0.2 * (tone(40.0) + tone(160.0) + tone(640.0) + tone(2560.0))
            } else {
                0.0
            }
        }
        "click" => {
            if n == 20 * u64::from(SR) {
                0.9
            } else {
                0.0
            }
        }
        "pair" => {
            let st = 2f64.powf(1.0 / 12.0);
            0.1 * [40.0, 160.0, 640.0].iter().map(|&f| tone(f) + tone(f * st)).sum::<f64>()
        }
        _ => panic!("MOOSIK_BASS_FIXTURE must be gate, click or pair"),
    }
}

fn write_wav(path: &Path, fixture: &str) {
    let frames = u64::from(SR) * u64::from(SECONDS);
    let data_bytes = u32::try_from(frames * 4).unwrap();
    let mut out = BufWriter::new(std::fs::File::create(path).unwrap());
    out.write_all(b"RIFF").unwrap();
    out.write_all(&(36 + data_bytes).to_le_bytes()).unwrap();
    out.write_all(b"WAVEfmt ").unwrap();
    out.write_all(&16u32.to_le_bytes()).unwrap();
    out.write_all(&1u16.to_le_bytes()).unwrap();
    out.write_all(&2u16.to_le_bytes()).unwrap();
    out.write_all(&SR.to_le_bytes()).unwrap();
    out.write_all(&(SR * 4).to_le_bytes()).unwrap();
    out.write_all(&4u16.to_le_bytes()).unwrap();
    out.write_all(&16u16.to_le_bytes()).unwrap();
    out.write_all(b"data").unwrap();
    out.write_all(&data_bytes.to_le_bytes()).unwrap();
    for n in 0..frames {
        let v = (sample(fixture, n).clamp(-1.0, 1.0) * f64::from(i16::MAX)).round() as i16;
        out.write_all(&v.to_le_bytes()).unwrap();
        out.write_all(&v.to_le_bytes()).unwrap();
    }
    out.flush().unwrap();
}

/// Bars' centre frequencies exactly as each production path computes them.
fn centres(mapping: &BarMappingMode, aslt_cfg: &aslt::AsltConfig, bars: usize) -> Vec<f64> {
    let (lo, hi) = (20.0f64, 24_000.0f64.min(f64::from(SR) / 2.0));
    match mapping {
        BarMappingMode::TrueCqt(_) => (0..bars).map(|i| lo * (hi / lo).powf((i as f64 + 0.5) / bars as f64)).collect(),
        _ => (0..bars).map(|i| f64::from(aslt_cfg.scale.bar_center(i, bars, 20.0, 24_000.0))).collect(),
    }
}

fn nearest(freqs: &[f64], f: f64) -> usize {
    (0..freqs.len()).min_by(|&a, &b| (freqs[a] / f).ln().abs().total_cmp(&(freqs[b] / f).ln().abs())).unwrap()
}

/// First and last frame time where `db` reaches `level`, if ever.
fn crossing(times: &[f64], db: &[f64], level: f64) -> Option<(f64, f64)> {
    let first = db.iter().position(|&d| d >= level)?;
    let last = db.iter().rposition(|&d| d >= level)?;
    Some((times[first], times[last]))
}

fn round(v: f64) -> f64 {
    (v * 1e4).round() / 1e4
}

#[test]
#[ignore = "manual research probe; one arm and fixture per exclusive release-test process"]
fn bass_response() {
    assert_eq!(std::env::var("MOOSIK_GPU").ok().as_deref(), Some("0"), "MOOSIK_GPU=0 is required");
    let arm = std::env::var("MOOSIK_BASS_ARM").expect("MOOSIK_BASS_ARM");
    let fixture_kind = std::env::var("MOOSIK_BASS_FIXTURE").expect("MOOSIK_BASS_FIXTURE");
    let bars = 1024usize;
    let mut aslt_cfg = aslt::AsltConfig {
        q_ratio: 2.0,
        max_window_s: 0.25,
        n_wavelets: 1,
        spread: 0.5,
        scale: freq_scale::FreqScale::Blend(0.5),
    };
    let mapping = match arm.as_str() {
        "aslt1" => BarMappingMode::Superlet,
        "aslt9" => {
            aslt_cfg.n_wavelets = 9;
            BarMappingMode::Superlet
        }
        "cqt12" | "cqt48" | "cqt192" => BarMappingMode::TrueCqt(cqt::Config {
            bins_per_octave: arm.strip_prefix("cqt").unwrap().parse().unwrap(),
            window: cqt::Window::Blackman,
            frame_rate: 180.0,
            bass_limit_s: None,
        }),
        // Responsive variable-Q: `vqt<BPO>t<seconds>`, e.g. vqt48t0.25.
        a if a.starts_with("vqt") => {
            let (bpo, t) = a["vqt".len()..].split_once('t').expect("vqt<BPO>t<seconds>");
            BarMappingMode::TrueCqt(cqt::Config {
                bins_per_octave: bpo.parse().unwrap(),
                window: cqt::Window::Blackman,
                frame_rate: 180.0,
                bass_limit_s: Some(t.parse().unwrap()),
            })
        }
        _ => panic!("unknown arm {arm}"),
    };
    let _calibration = gpu_calib::isolated_test_env("v155_bass");
    gpu_calib::set_mode(gpu_calib::GpuMode::Off);
    gpu_calib::set_workers(6);
    let fixture = Fixture::new();
    let wav = fixture.root.join("fixture.wav");
    write_wav(&wav, &fixture_kind);
    let unit = fixture.root.join("cache").join("track.spectrumcache");
    let cache_store = test_store(&unit).expect("private store must open");
    let progress = Arc::new(AtomicUsize::new(0));
    let abort = Arc::new(AtomicBool::new(false));
    let eta = Arc::new(AtomicUsize::new(usize::MAX));
    let message = preprocess_file(
        &wav,
        &unit,
        Some(cache_store),
        true,
        0,
        bars,
        &progress,
        DEFAULT_FFT_SIZE,
        16,
        0.875,
        &WindowFn::Hann,
        20.0,
        24_000.0,
        &InterpolationMode::None,
        &mapping,
        88_200,
        &aslt_cfg,
        180.0,
        &abort,
        &eta,
        StereoRequest::MixOnly,
    );
    let frames = match message {
        PreMessage::Done { frames, .. } => frames,
        PreMessage::Error(e) => panic!("refused: {e}"),
        PreMessage::Aborted => panic!("aborted"),
    };
    let times: Vec<f64> = (0..frames.len())
        .map(|k| match &mapping {
            BarMappingMode::TrueCqt(c) => cqt::frame_center(k, SR, c.frame_rate) as f64 / f64::from(SR),
            _ => (k * aslt::hop_for_fps(SR, 180.0)) as f64 / f64::from(SR),
        })
        .collect();
    let freqs = centres(&mapping, &aslt_cfg, bars);
    let series = |bar: usize| -> Vec<f64> {
        (0..frames.len()).map(|k| f64::from(frames.row(k).unwrap()[bar]) * DB_PER_CODE).collect()
    };
    let row_db = |t: f64| -> Vec<f64> {
        let k = times.iter().position(|&x| x >= t).unwrap();
        frames.row(k).unwrap().iter().map(|&c| f64::from(c) * DB_PER_CODE).collect()
    };
    // Width (cents) of the contiguous region around `peak` within `drop` dB.
    let width_cents = |row: &[f64], peak: usize, drop: f64| -> f64 {
        let floor = row[peak] - drop;
        let (mut a, mut b) = (peak, peak);
        while a > 0 && row[a - 1] >= floor {
            a -= 1;
        }
        while b + 1 < row.len() && row[b + 1] >= floor {
            b += 1;
        }
        1200.0 * (freqs[b] / freqs[a]).log2()
    };
    let mut results = Vec::new();
    match fixture_kind.as_str() {
        "gate" => {
            let mid = row_db(20.0);
            for f in [40.0, 160.0, 640.0, 2560.0] {
                let near = nearest(&freqs, f);
                // The strongest bar within a quarter-octave: the grid's own peak.
                let lo = nearest(&freqs, f / 2f64.powf(0.25));
                let hi = nearest(&freqs, f * 2f64.powf(0.25));
                let peak = (lo..=hi).max_by(|&a, &b| mid[a].total_cmp(&mid[b])).unwrap_or(near);
                let s = series(peak);
                let steady: f64 = {
                    let mut v: Vec<f64> = times.iter().zip(&s).filter(|(t, _)| (19.0..21.0).contains(*t)).map(|(_, d)| *d).collect();
                    v.sort_by(f64::total_cmp);
                    v[v.len() / 2]
                };
                let at = |drop: f64| crossing(&times, &s, steady - drop);
                let (on6, off6) = at(6.0).unwrap();
                let (on20, off20) = at(20.0).map_or((f64::NAN, f64::NAN), |x| x);
                let (on40, off40) = at(40.0).map_or((f64::NAN, f64::NAN), |x| x);
                results.push(serde_json::json!({
                    "tone_hz": f, "bar": peak, "bar_hz": round(freqs[peak]), "steady_db": round(steady),
                    "floor_db_before": round(s[times.iter().position(|&t| t >= 5.0).unwrap()]),
                    "onset_6db_s": round(on6 - 15.0), "onset_20db_s": round(on20 - 15.0), "onset_40db_s": round(on40 - 15.0),
                    "release_6db_s": round(off6 - 25.0), "release_20db_s": round(off20 - 25.0), "release_40db_s": round(off40 - 25.0),
                    "freq_width_3db_cents": round(width_cents(&mid, peak, 3.0)),
                    "freq_width_20db_cents": round(width_cents(&mid, peak, 20.0)),
                }));
            }
        }
        "click" => {
            for f in [30.0, 40.0, 60.0, 100.0, 160.0, 640.0, 2560.0] {
                let bar = nearest(&freqs, f);
                let s = series(bar);
                let k = (0..s.len()).max_by(|&a, &b| s[a].total_cmp(&s[b])).unwrap();
                let peak = s[k];
                let w = |drop: f64| crossing(&times, &s, peak - drop).map_or(f64::NAN, |(a, b)| b - a);
                let (a6, b6) = crossing(&times, &s, peak - 6.0).unwrap();
                results.push(serde_json::json!({
                    "bar_hz": round(freqs[bar]), "bar": bar, "peak_db": round(peak),
                    "peak_offset_s": round(times[k] - 20.0),
                    "first_6db_s": round(a6 - 20.0), "last_6db_s": round(b6 - 20.0),
                    "width_6db_s": round(w(6.0)), "width_20db_s": round(w(20.0)), "width_40db_s": round(w(40.0)),
                }));
            }
        }
        "pair" => {
            let row = row_db(20.0);
            let st = 2f64.powf(1.0 / 12.0);
            for f in [40.0, 160.0, 640.0] {
                let (a, b) = (nearest(&freqs, f), nearest(&freqs, f * st));
                let between = (a..=b).map(|i| row[i]).fold(f64::INFINITY, f64::min);
                let lower = row[a].min(row[b]);
                results.push(serde_json::json!({
                    "pair_hz": [f, round(f * st)], "bars": [a, b], "levels_db": [round(row[a]), round(row[b])],
                    "min_between_db": round(between), "dip_db": round(lower - between),
                    "bars_between": b.saturating_sub(a).saturating_sub(1),
                }));
            }
        }
        _ => unreachable!(),
    }
    println!(
        "BASS_RESPONSE {}",
        serde_json::json!({
            "arm": arm, "fixture": fixture_kind, "sr": SR, "seconds": SECONDS, "bars": bars,
            "frames": frames.len(), "max_code": frames.max_code(), "results": results,
        })
    );
}
