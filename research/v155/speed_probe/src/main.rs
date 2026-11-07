//! Exclusive-machine core comparison, not decode/cache/UI or playback timing.
#![allow(dead_code)]
#[path = "../../../../src/spectrum/cqt.rs"]
mod cqt;
#[path = "../../frozen_cqt_f9a.rs"]
mod frozen;

fn samples(sr: u32, count: usize, seed: u64) -> Vec<f32> {
    let mut state = seed;
    (0..count)
        .map(|i| {
            state ^= state << 13;
            state ^= state >> 7;
            state ^= state << 17;
            let noise = ((state >> 11) as f64 / (1u64 << 53) as f64 - 0.5) * 0.12;
            let t = i as f64 / sr as f64;
            (0.55 * (std::f64::consts::TAU * (77.0 * t + 13.0 * t * t)).sin() + noise) as f32
        })
        .collect()
}

fn code(magnitude: f32) -> u16 {
    // Same operations as spectrum::aslt_code, stated explicitly.
    let db = (20.0 * (magnitude * 0.25).log10()).max(-80.0);
    let value = ((db + 80.0) / 80.0).clamp(0.0, 1.0);
    if value.is_nan() {
        0
    } else {
        (value.clamp(0.0, 1.0) * 4095.0).round() as u16
    }
}

fn digest(codes: &[u16]) -> String {
    // FNV is a compact identity aid, not a cryptographic or equivalence proof.
    format!(
        "{:016x}",
        codes
            .iter()
            .flat_map(|v| v.to_le_bytes())
            .fold(0xcbf29ce484222325u64, |h, b| (h ^ u64::from(b))
                .wrapping_mul(0x100000001b3))
    )
}

fn main() {
    let args: Vec<_> = std::env::args().skip(1).collect();
    assert!(
        (5..=7).contains(&args.len()),
        "arm sr seconds bars mono|joint [max_hz] [bins_per_octave]"
    );
    let arm = &args[0];
    let sr: u32 = args[1].parse().unwrap();
    let seconds: usize = args[2].parse().unwrap();
    let bars: usize = args[3].parse().unwrap();
    let joint = match args[4].as_str() {
        "mono" => false,
        "joint" => true,
        _ => panic!("mode"),
    };
    assert!(
        [44100, 48000, 96000].contains(&sr)
            && (1..=120).contains(&seconds)
            && (1..=2048).contains(&bars)
    );
    // The original five-argument matrix remains 20 Hz to 20 kHz. Explicitly
    // selected follow-up geometries can mirror the product's Nyquist clamp.
    let max_hz_requested: f64 = args.get(5).map_or(20_000.0, |v| v.parse().unwrap());
    let bins_per_octave: u16 = args.get(6).map_or(48, |v| v.parse().unwrap());
    assert!(max_hz_requested.is_finite() && max_hz_requested > 20.0);
    assert!((12..=192).contains(&bins_per_octave));
    let max_hz = max_hz_requested.min(f64::from(sr) / 2.0);
    let left = samples(sr, sr as usize * seconds, 0x155);
    let right = samples(sr, left.len(), 0xabcd);
    let frequencies: Vec<_> = (0..bars)
        .map(|i| 20.0 * (max_hz / 20.0).powf((i as f64 + 0.5) / bars as f64))
        .collect();
    // Pool creation and fixture generation are explicitly outside the timer.
    let workers = match arm.as_str() {
        "frozen" | "serial" => 0,
        "pool1" => 1,
        "pool2" => 2,
        "pool6" => 6,
        _ => panic!("arm"),
    };
    let pool = (workers > 0).then(|| {
        rayon::ThreadPoolBuilder::new()
            .num_threads(workers)
            .build()
            .unwrap()
    });
    let started = std::time::Instant::now();
    let (codes, frames, stats) = if arm == "frozen" {
        let out = frozen::analyze_input_codes(
            if joint {
                frozen::Input::Joint {
                    left: &left,
                    right: &right,
                }
            } else {
                frozen::Input::Mono(&left)
            },
            sr,
            &frequencies,
            &frozen::Config { bins_per_octave, ..Default::default() },
            frozen::Limits::default(),
            &|| true,
            &|_| {},
            &code,
        )
        .unwrap();
        (
            out.codes,
            out.frames,
            serde_json::json!({
                "kernel_ffts": out.work.kernel_ffts, "input_ffts": out.work.input_ffts,
                "inverse_ffts": out.work.inverse_ffts, "batches": out.work.batches,
                "workspace_bytes": out.work.peak_workspace_bytes, "output_bytes": out.work.output_bytes
            }),
        )
    } else {
        let run = || {
            cqt::analyze_input_codes(
                if joint {
                    cqt::Input::Joint {
                        left: &left,
                        right: &right,
                    }
                } else {
                    cqt::Input::Mono(&left)
                },
                sr,
                &frequencies,
                &cqt::Config { bins_per_octave, ..Default::default() },
                cqt::Limits::default(),
                &|| true,
                &|_| {},
                &code,
            )
            .unwrap()
        };
        let out = match pool {
            Some(ref pool) => pool.install(run),
            None => run(),
        };
        (
            out.codes,
            out.frames,
            serde_json::json!({
                "kernel_ffts": out.work.kernel_ffts, "input_ffts": out.work.input_ffts,
                "inverse_ffts": out.work.inverse_ffts, "pruned": out.work.inverses_pruned,
                "whole": out.work.inverses_full, "batches": out.work.batches,
                "parallel_groups": out.work.parallel_groups, "max_workers": out.work.max_inverse_workers,
                "workspace_bytes": out.work.peak_workspace_bytes, "output_bytes": out.work.output_bytes
            }),
        )
    };
    let wall = started.elapsed().as_secs_f64();
    println!(
        "{}",
        serde_json::json!({
            "arm": arm, "sr": sr, "seconds": seconds, "bars": bars, "joint": joint,
            "min_hz": 20.0, "max_hz": max_hz, "max_hz_requested": max_hz_requested,
            "bpo": bins_per_octave, "window": "Blackman", "fps": 180, "frames": frames,
            "wall_seconds": wall, "stats": stats, "codes_fnv64": codes.iter().map(|v| digest(v)).collect::<Vec<_>>()
        })
    );
}
