//! Tempo and beats (1.6.2): onset strength from spectral flux, the tempo from
//! its autocorrelation weighted toward common tempos, then the beats by
//! dynamic programming (D. Ellis, "Beat Tracking by Dynamic Programming",
//! 2007). Works on the same 22050 Hz mono the chord model hears.

use rustfft::{num_complex::Complex, FftPlanner};

const N: usize = 1024;
const HOP: usize = 256;

/// The beat grid of a track.
#[derive(Clone, Debug, PartialEq)]
pub struct Beats {
    pub bpm: f32,
    /// Beat times, seconds.
    pub at: Vec<f32>,
    /// The song's parts, found from the beats (see `sections`).
    pub sections: Vec<crate::sections::Section>,
}

/// A file's beats: from the cache beside the chords when it was done
/// before, otherwise decoded (22050 Hz mono, as the chord model hears it),
/// tracked and kept.
pub fn analyze(path: &std::path::Path, cache: &std::path::Path, cancel: &std::sync::atomic::AtomicBool) -> Result<Option<Beats>, String> {
    let file = crate::chords::beats_file(path, cache);
    if let Some(f) = &file
        && let Ok(t) = std::fs::read_to_string(f)
        && let Ok((bpm, at, parts)) = serde_json::from_str::<(f32, Vec<f32>, Vec<(f32, f32, char)>)>(&t)
    {
        let sections = parts.into_iter().map(|(start, end, label)| crate::sections::Section { start, end, label }).collect();
        return Ok((bpm > 0.0).then_some(Beats { bpm, at, sections }));
    }
    let mono = crate::chords::load_mono(path, cancel)?;
    let mut beats = track(&mono, crate::chords::SR);
    if let Some(b) = beats.as_mut() {
        b.sections = crate::sections::find(&mono, crate::chords::SR, &b.at);
    }
    if let Some(f) = file {
        let _ = std::fs::create_dir_all(cache);
        let json = beats.as_ref().map_or_else(|| "[0,[],[]]".to_string(), |b| {
            let parts: Vec<(f32, f32, char)> = b.sections.iter().map(|s| (s.start, s.end, s.label)).collect();
            serde_json::to_string(&(b.bpm, &b.at, parts)).unwrap_or_default()
        });
        let tmp = f.with_extension("part");
        if std::fs::write(&tmp, json).is_ok() {
            let _ = std::fs::rename(&tmp, &f);
        }
    }
    Ok(beats)
}

/// The beat at or before `t` (seconds), as (index, seconds since it).
pub fn last_beat(at: &[f32], t: f32) -> Option<(usize, f32)> {
    let i = at.partition_point(|&b| b <= t).checked_sub(1)?;
    Some((i, t - at[i]))
}

/// Onset strength per hop: half-wave rectified log-magnitude flux.
fn onsets(mono: &[f32], rate: u32) -> (Vec<f32>, f32) {
    let fft = FftPlanner::<f32>::new().plan_fft_forward(N);
    let win: Vec<f32> = (0..N).map(|i| 0.5 - 0.5 * (std::f32::consts::TAU * i as f32 / N as f32).cos()).collect();
    let frames = mono.len().saturating_sub(N) / HOP;
    let mut prev = vec![0f32; N / 2];
    let mut buf = vec![Complex::new(0f32, 0f32); N];
    let mut env = Vec::with_capacity(frames);
    for f in 0..frames {
        let s = &mono[f * HOP..f * HOP + N];
        for (b, (x, w)) in buf.iter_mut().zip(s.iter().zip(&win)) {
            *b = Complex::new(x * w, 0.0);
        }
        fft.process(&mut buf);
        let mut flux = 0.0;
        for (k, p) in prev.iter_mut().enumerate() {
            let m = (1.0 + 1000.0 * buf[k].norm()).ln();
            flux += (m - *p).max(0.0);
            *p = m;
        }
        env.push(flux);
    }
    // Remove the slow trend, keep what rises above it.
    let w = 16;
    let smooth: Vec<f32> = (0..env.len())
        .map(|i| {
            let (a, b) = (i.saturating_sub(w), (i + w + 1).min(env.len()));
            env[a..b].iter().sum::<f32>() / (b - a) as f32
        })
        .collect();
    let env: Vec<f32> = env.iter().zip(&smooth).map(|(e, s)| (e - s).max(0.0)).collect();
    let sd = (env.iter().map(|v| v * v).sum::<f32>() / env.len().max(1) as f32).sqrt().max(1e-9);
    (env.into_iter().map(|v| v / sd).collect(), rate as f32 / HOP as f32)
}

/// The beats of mono audio at `rate`, or None when it has no steady pulse.
pub fn track(mono: &[f32], rate: u32) -> Option<Beats> {
    let (env, fps) = onsets(mono, rate);
    if env.len() < (fps * 6.0) as usize {
        return None;
    }
    // Tempo: autocorrelation over 40–240 BPM, weighted by a log-Gaussian
    // around 120 BPM (one octave wide), then the strongest lag.
    let (lo, hi) = ((fps * 60.0 / 240.0) as usize, (fps * 60.0 / 40.0) as usize);
    let mut best = (0usize, 0f32);
    for lag in lo.max(1)..=hi.min(env.len() / 2) {
        let ac: f32 = env[lag..].iter().zip(&env).map(|(a, b)| a * b).sum::<f32>() / (env.len() - lag) as f32;
        let bpm = 60.0 * fps / lag as f32;
        let w = (-0.5 * ((bpm / 120.0).log2() / 1.0).powi(2)).exp();
        if ac * w > best.1 {
            best = (lag, ac * w);
        }
    }
    if best.1 <= 0.0 {
        return None;
    }
    let ac = |lag: usize| env[lag..].iter().zip(&env).map(|(a, b)| a * b).sum::<f32>() / (env.len() - lag) as f32;
    // Eighth-note hi-hats make the half beat the strongest pulse; above
    // 140 BPM, a clear pulse at half the tempo is the beat.
    if 60.0 * fps / best.0 as f32 > 140.0 && 2 * best.0 + 1 < env.len() / 2 && ac(2 * best.0) > 0.5 * ac(best.0) {
        best.0 *= 2;
    }
    // Refine the period to a fraction of a hop with a parabola.
    let (l, c, r) = (ac(best.0 - 1), ac(best.0), ac(best.0 + 1));
    let d = l - 2.0 * c + r;
    let period = best.0 as f32 + if d.abs() > 1e-9 { 0.5 * (l - r) / d } else { 0.0 };
    // Beats: dynamic programming, each beat rewarded by its onset strength
    // and penalised for straying from the period.
    let tight = 100.0;
    let n = env.len();
    let mut score = env.clone();
    let mut back = vec![usize::MAX; n];
    for i in 0..n {
        let (a, b) = ((i as f32 - 2.0 * period).max(0.0) as usize, (i as f32 - period / 2.0).max(0.0) as usize);
        let mut top = (f32::MIN, usize::MAX);
        for j in a..b.min(i) {
            let x = ((i - j) as f32 / period).ln();
            let s = score[j] - tight * x * x;
            if s > top.0 {
                top = (s, j);
            }
        }
        if top.1 != usize::MAX {
            score[i] = env[i] + top.0;
            back[i] = top.1;
        }
    }
    // Start from the best-scoring frame in the last period and walk back.
    let tail = (n as f32 - period).max(0.0) as usize;
    let mut i = (tail..n).max_by(|&a, &b| score[a].total_cmp(&score[b]))?;
    let mut at = Vec::new();
    loop {
        at.push((i * HOP + N / 2) as f32 / rate as f32);
        if back[i] == usize::MAX {
            break;
        }
        i = back[i];
    }
    at.reverse();
    Some(Beats { bpm: 60.0 * fps / period, at, sections: Vec::new() })
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Clicks (short decaying noise bursts) every `60/bpm` seconds, quiet hum between.
    fn clicks(bpm: f32, secs: f32, rate: u32, start: f32) -> Vec<f32> {
        let n = (secs * rate as f32) as usize;
        let mut seed = 7u32;
        let mut y: Vec<f32> = (0..n).map(|i| 0.02 * (i as f32 * 0.01).sin()).collect();
        let mut t = start;
        while t < secs {
            let s = (t * rate as f32) as usize;
            for k in 0..(rate as usize / 50) {
                if s + k < n {
                    seed = seed.wrapping_mul(1_103_515_245).wrapping_add(12345);
                    let noise = ((seed >> 9) as f32 / (1u32 << 23) as f32) - 1.0;
                    y[s + k] += 0.6 * noise * (-(k as f32) / 200.0).exp();
                }
            }
            t += 60.0 / bpm;
        }
        y
    }

    #[test]
    fn a_click_track_is_found_at_its_tempo_and_on_its_beats() {
        for bpm in [90.0, 120.0, 140.0] {
            let y = clicks(bpm, 30.0, 22_050, 0.25);
            let b = track(&y, 22_050).unwrap();
            assert!((b.bpm - bpm).abs() < 1.0, "{bpm}: {}", b.bpm);
            // Every found beat sits on a click, within 25 ms.
            let period = 60.0 / bpm;
            let off: Vec<f32> = b.at.iter().map(|&t| {
                let k = ((t - 0.25) / period).round();
                (t - (0.25 + k * period)).abs()
            }).collect();
            let bad = off.iter().filter(|&&o| o > 0.025).count();
            assert!(bad <= 1, "{bpm}: {bad} of {} beats off: {:?}", b.at.len(), &off[..8.min(off.len())]);
            assert!(b.at.len() as f32 > 0.9 * 30.0 / period, "{bpm}: only {} beats", b.at.len());
        }
    }

    #[test]
    fn silence_or_a_short_clip_has_no_beats() {
        assert!(track(&vec![0.0; 22_050 * 10], 22_050).is_none());
        assert!(track(&clicks(120.0, 3.0, 22_050, 0.0), 22_050).is_none());
    }

    /// A real file's beats (MOOSIK_FLAC), printed, and cached on the second run.
    #[test]
    #[ignore]
    fn a_file_gets_beats_and_keeps_them() {
        let f = std::path::PathBuf::from(std::env::var("MOOSIK_FLAC").expect("set MOOSIK_FLAC"));
        let cache = std::env::temp_dir().join(format!("moosik-beats-{}", std::process::id()));
        let c = std::sync::atomic::AtomicBool::new(false);
        let t = std::time::Instant::now();
        let b = analyze(&f, &cache, &c).unwrap().expect("a pulse");
        eprintln!("{:.1} BPM, {} beats, first {:?}, in {:?}; sections {:?}", b.bpm, b.at.len(), &b.at[..4.min(b.at.len())], t.elapsed(), b.sections.iter().map(|s| (s.label, s.start as u32)).collect::<Vec<_>>());
        let t = std::time::Instant::now();
        assert_eq!(analyze(&f, &cache, &c).unwrap(), Some(b));
        assert!(t.elapsed().as_millis() < 100, "from the cache");
        std::fs::remove_dir_all(&cache).unwrap();
    }
}
