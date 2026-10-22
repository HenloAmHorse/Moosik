//! 1.5.5 compression screen (ignored, release only). Real production code
//! planes from synthetic audio; v4 against reversible LZ4 predictor variants.
//! Exact round trip is asserted for every variant before anything is timed.
use super::*;
use crate::spectrum::*;
use std::io::{BufWriter, Write};
use std::time::Instant;

const SR: u32 = 48_000;
const SECONDS: u64 = 30;

fn write_wav(path: &Path, kind: &str) {
    let frames = u64::from(SR) * SECONDS;
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
    let mut seed = 0x9E37_79B9_7F4A_7C15u64;
    let mut noise = move || {
        seed ^= seed << 13;
        seed ^= seed >> 7;
        seed ^= seed << 17;
        (seed >> 11) as f64 / (1u64 << 53) as f64 * 2.0 - 1.0
    };
    for n in 0..frames {
        let t = n as f64 / f64::from(SR);
        let tone = |f: f64| (std::f64::consts::TAU * f * t).sin();
        let v = match kind {
            "silence" => 0.0,
            // Harmonic notes with vibrato, a bass line and a beat of clicks.
            "music" => {
                let note = [220.0, 277.18, 329.63, 440.0][((t * 2.0) as usize) % 4];
                let vib = 1.0 + 0.003 * (std::f64::consts::TAU * 5.0 * t).sin();
                let env = (-(t * 2.0).fract() * 3.0).exp();
                let mut s = 0.0;
                for h in 1..8 {
                    s += env * 0.15 / h as f64 * tone(note * vib * h as f64);
                }
                s += 0.2 * tone([55.0, 41.2][((t / 2.0) as usize) % 2]);
                if (t * 4.0).fract() < 0.004 {
                    s += 0.5 * noise();
                }
                s + 0.003 * noise()
            }
            "noise" => 0.3 * noise(),
            _ => unreachable!(),
        };
        let v = (v.clamp(-1.0, 1.0) * f64::from(i16::MAX)).round() as i16;
        out.write_all(&v.to_le_bytes()).unwrap();
        out.write_all(&v.to_le_bytes()).unwrap();
    }
    out.flush().unwrap();
}

/// Predictor for cell (r, b). `left` resets per row like v4.
#[derive(Clone, Copy, Debug)]
enum Pred {
    Freq,
    Time,
    Grad,
    Med,
}

#[inline]
fn predict(p: Pred, left: i32, up: i32, ul: i32) -> i32 {
    match p {
        Pred::Freq => left,
        Pred::Time => up,
        Pred::Grad => (left + up - ul).clamp(0, V4_MAX_CODE as i32),
        Pred::Med => {
            let (mn, mx) = (left.min(up), left.max(up));
            if ul >= mx {
                mn
            } else if ul <= mn {
                mx
            } else {
                left + up - ul
            }
        }
    }
}

fn encode(p: Pred, codes: &[u16], bars: usize) -> (Vec<u8>, Vec<u8>) {
    let cells = codes.len();
    let mut lo = Vec::with_capacity(cells);
    let mut hi = Vec::with_capacity(cells.div_ceil(2));
    let mut pending = 0u8;
    let zero = vec![0u16; bars];
    let mut prev_row: &[u16] = &zero;
    for row in codes.chunks_exact(bars) {
        let mut left = 0i32;
        let mut ul = 0i32;
        for (b, &q) in row.iter().enumerate() {
            let q = q as i32;
            let up = prev_row[b] as i32;
            let d = wrap_signed(q - predict(p, left, up, ul), V4_BITS);
            left = q;
            ul = up;
            let z = (((d << 1) ^ (d >> 31)) as u32) & (V4_MAX_CODE as u32);
            lo.push((z & 0xFF) as u8);
            let nib = ((z >> 8) as u8) & 0x0F;
            if lo.len() % 2 == 1 {
                pending = nib
            } else {
                hi.push(pending | (nib << 4))
            }
        }
        prev_row = row;
    }
    if cells % 2 == 1 {
        hi.push(pending);
    }
    (lz4_flex::compress(&lo), lz4_flex::compress(&hi))
}

fn decode(p: Pred, blocks: &(Vec<u8>, Vec<u8>), frames: usize, bars: usize) -> Vec<u16> {
    let cells = frames * bars;
    let lo = lz4_flex::decompress(&blocks.0, cells).unwrap();
    let hi = lz4_flex::decompress(&blocks.1, cells.div_ceil(2)).unwrap();
    let mut codes: Vec<u16> = Vec::with_capacity(cells);
    for r in 0..frames {
        let base = r * bars;
        let mut left = 0i32;
        let mut ul = 0i32;
        for i in base..base + bars {
            let up = if r == 0 { 0 } else { codes[i - bars] as i32 };
            let packed = hi[i >> 1];
            let nib = if i % 2 == 0 { packed & 0x0F } else { packed >> 4 };
            let z = (lo[i] as u32) | ((nib as u32) << 8);
            let d = ((z >> 1) as i32) ^ -((z & 1) as i32);
            let q = (predict(p, left, up, ul) + d) & V4_MAX_CODE as i32;
            codes.push(q as u16);
            left = q;
            ul = up;
        }
    }
    codes
}

fn median(mut v: Vec<f64>) -> f64 {
    v.sort_by(f64::total_cmp);
    v[v.len() / 2]
}

#[test]
#[ignore = "exclusive release-only codec screen"]
fn release_codec_screen() {
    use serde_json::json;
    use std::hint::black_box;
    assert!(!black_box(cfg!(debug_assertions)), "run in release");
    let _calibration = gpu_calib::isolated_test_env("v155_codec");
    gpu_calib::set_mode(gpu_calib::GpuMode::Off);
    gpu_calib::set_workers(6);
    let root = std::env::temp_dir().join(format!("moosik_codec_{}", std::process::id()));
    std::fs::create_dir_all(&root).unwrap();
    let cqt48 = BarMappingMode::TrueCqt(cqt::Config {
        bins_per_octave: 48,
        window: cqt::Window::Blackman,
        frame_rate: 180.0,
        bass_limit_s: None,
    });
    let arms: Vec<(&str, BarMappingMode)> = vec![
        ("fft-cqtw", BarMappingMode::Cqt),
        ("fft-gauss", BarMappingMode::Gaussian),
        ("aslt", BarMappingMode::Superlet),
        ("truecqt48", cqt48),
    ];
    for kind in ["music", "noise", "silence"] {
        let wav = root.join(format!("{kind}.wav"));
        write_wav(&wav, kind);
        for (name, mapping) in &arms {
            for bars in [1024usize, 2048] {
                let unit = root.join(format!("c_{kind}_{name}_{bars}")).join("t.spectrumcache");
                let progress = Arc::new(AtomicUsize::new(0));
                let abort = Arc::new(AtomicBool::new(false));
                let eta = Arc::new(AtomicUsize::new(usize::MAX));
                let aslt_cfg = aslt::AsltConfig::default();
                let frames = match preprocess_file(
                    &wav, &unit, test_store(&unit), true, 0, bars, &progress, DEFAULT_FFT_SIZE, 16, 0.875,
                    &WindowFn::Hann, 20.0, 24_000.0, &InterpolationMode::None, mapping, 88_200, &aslt_cfg,
                    180.0, &abort, &eta, StereoRequest::MixOnly,
                ) {
                    PreMessage::Done { frames, .. } => frames,
                    PreMessage::Error(e) => panic!("{e}"),
                    PreMessage::Aborted => panic!("aborted"),
                };
                let (codes, nf) = (frames.codes.clone(), frames.frames);
                assert_eq!(frames.max_code, V4_MAX_CODE);
                // Production v4 bytes and decode (full checked path).
                let v4 = frames.to_v4();
                let back = decode_with(&v4, bars, MAX_BAR_COUNT, Limits::default()).unwrap();
                assert_eq!(back.codes, codes);
                let mut out = json!({"fixture":kind,"arm":name,"bars":bars,"frames":nf,
                    "raw_u16_bytes":codes.len()*2,"v4_bytes":v4.len()});
                let mut v4_enc = Vec::new();
                let mut v4_dec = Vec::new();
                for _ in 0..9 {
                    let t = Instant::now();
                    black_box(frames.to_v4());
                    v4_enc.push(t.elapsed().as_secs_f64() * 1e3);
                    let t = Instant::now();
                    black_box(decode_with(&v4, bars, MAX_BAR_COUNT, Limits::default()).unwrap());
                    v4_dec.push(t.elapsed().as_secs_f64() * 1e3);
                }
                out["v4_encode_ms"] = json!(median(v4_enc));
                out["v4_decode_ms"] = json!(median(v4_dec));
                for p in [Pred::Freq, Pred::Time, Pred::Grad, Pred::Med] {
                    let blocks = encode(p, &codes, bars);
                    assert_eq!(decode(p, &blocks, nf, bars), codes, "{p:?} round trip");
                    let (mut e, mut d) = (Vec::new(), Vec::new());
                    for _ in 0..9 {
                        let t = Instant::now();
                        black_box(encode(p, &codes, bars));
                        e.push(t.elapsed().as_secs_f64() * 1e3);
                        let t = Instant::now();
                        black_box(decode(p, &blocks, nf, bars));
                        d.push(t.elapsed().as_secs_f64() * 1e3);
                    }
                    let k = format!("{p:?}").to_lowercase();
                    out[format!("{k}_bytes")] = json!(blocks.0.len() + blocks.1.len() + V4_HEADER);
                    out[format!("{k}_encode_ms")] = json!(median(e));
                    out[format!("{k}_decode_ms")] = json!(median(d));
                }
                println!("codec_screen {out}");
            }
        }
    }
    let _ = std::fs::remove_dir_all(&root);
}
