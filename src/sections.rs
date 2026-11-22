//! Sections (1.6.2): where a song changes part, and which parts come back.
//! A chroma per beat, a self-similarity matrix of them, Foote's checkerboard
//! novelty for the boundaries, then parts with similar harmony share a
//! letter (A, B, A…). Rough by nature: it hears harmony, not lyrics or
//! arrangement, so a chorus over the verse's chords reads as the verse.

use rustfft::{num_complex::Complex, FftPlanner};

/// A part of the song: start and end (seconds) and its letter.
#[derive(Clone, Debug, PartialEq)]
pub struct Section {
    pub start: f32,
    pub end: f32,
    pub label: char,
}

/// Twelve pitch classes from a magnitude spectrum, 55 Hz to 2 kHz.
fn chroma(mag: &[f32], rate: u32, n: usize) -> [f32; 12] {
    let mut c = [0f32; 12];
    for (k, &m) in mag.iter().enumerate().skip(1) {
        let f = k as f32 * rate as f32 / n as f32;
        if !(55.0..2000.0).contains(&f) {
            continue;
        }
        let pc = ((12.0 * (f / 440.0).log2()).round() as i32 + 9).rem_euclid(12) as usize;
        c[pc] += m;
    }
    let norm = c.iter().map(|v| v * v).sum::<f32>().sqrt().max(1e-9);
    c.map(|v| v / norm)
}

/// The sections of mono audio at `rate`, given its beats (seconds).
pub fn find(mono: &[f32], rate: u32, beats: &[f32]) -> Vec<Section> {
    const N: usize = 4096;
    if beats.len() < 32 {
        return Vec::new();
    }
    let fft = FftPlanner::<f32>::new().plan_fft_forward(N);
    let win: Vec<f32> = (0..N).map(|i| 0.5 - 0.5 * (std::f32::consts::TAU * i as f32 / N as f32).cos()).collect();
    let mut buf = vec![Complex::new(0f32, 0f32); N];
    // One chroma per beat: from the window centred between it and the next.
    let feats: Vec<[f32; 12]> = beats
        .windows(2)
        .map(|w| {
            let c = (((w[0] + w[1]) / 2.0) * rate as f32) as isize - (N / 2) as isize;
            for (i, (b, wv)) in buf.iter_mut().zip(&win).enumerate() {
                let s = c + i as isize;
                let x = if s >= 0 && (s as usize) < mono.len() { mono[s as usize] } else { 0.0 };
                *b = Complex::new(x * wv, 0.0);
            }
            fft.process(&mut buf);
            let mag: Vec<f32> = buf[..N / 2].iter().map(|c| c.norm()).collect();
            chroma(&mag, rate, N)
        })
        .collect();
    let n = feats.len();
    let sim = |i: usize, j: usize| feats[i].iter().zip(&feats[j]).map(|(a, b)| a * b).sum::<f32>();
    // Novelty: a checkerboard kernel of `k` beats a side slid along the diagonal.
    let k = 16usize.min(n / 4);
    let mut nov = vec![0f32; n];
    for (i, v) in nov.iter_mut().enumerate().take(n - k).skip(k) {
        let mut s = 0.0;
        for a in i - k..i + k {
            for b in i - k..i + k {
                let same = (a < i) == (b < i);
                s += if same { sim(a, b) } else { -sim(a, b) };
            }
        }
        *v = s / (4 * k * k) as f32;
    }
    // Boundaries: novelty above the mean plus half a deviation that is the
    // highest within `k` beats either side (a chord change inside a part
    // makes a smaller bump nearby, not a boundary).
    let mean = nov.iter().sum::<f32>() / n as f32;
    let sd = (nov.iter().map(|v| (v - mean).powi(2)).sum::<f32>() / n as f32).sqrt();
    let mut cuts = vec![0usize];
    for i in 1..n - 1 {
        let around = &nov[i.saturating_sub(k)..(i + k + 1).min(n)];
        let top = around.iter().copied().fold(f32::MIN, f32::max);
        if nov[i] > mean + 0.5 * sd && nov[i] >= top && i - cuts.last().unwrap() >= k {
            cuts.push(i);
        }
    }
    cuts.push(n);
    // Each part's mean chroma; a part like an earlier one gets its letter.
    let mut parts: Vec<([f32; 12], char)> = Vec::new();
    let mut out: Vec<Section> = Vec::new();
    for w in cuts.windows(2) {
        let (a, b) = (w[0], w[1]);
        let mut m = [0f32; 12];
        for f in &feats[a..b] {
            for (x, y) in m.iter_mut().zip(f) {
                *x += y;
            }
        }
        let norm = m.iter().map(|v| v * v).sum::<f32>().sqrt().max(1e-9);
        let m = m.map(|v| v / norm);
        let label = parts
            .iter()
            .map(|(p, l)| (p.iter().zip(&m).map(|(x, y)| x * y).sum::<f32>(), *l))
            .filter(|(s, _)| *s > 0.97)
            .max_by(|x, y| x.0.total_cmp(&y.0))
            .map(|(_, l)| l)
            .unwrap_or_else(|| {
                let l = (b'A' + parts.len().min(25) as u8) as char;
                parts.push((m, l));
                l
            });
        let end = beats[b.min(beats.len() - 1)];
        // The same part again straight after is one part.
        match out.last_mut() {
            Some(last) if last.label == label => last.end = end,
            _ => out.push(Section { start: beats[a], end, label }),
        }
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Triads for `secs_each` each, at `rate`, as plain sines.
    fn chords(prog: &[[f32; 3]], secs_each: f32, rate: u32) -> Vec<f32> {
        let mut y = Vec::new();
        for c in prog {
            let n = (secs_each * rate as f32) as usize;
            for i in 0..n {
                let t = i as f32 / rate as f32;
                y.push(c.iter().map(|f| (std::f32::consts::TAU * f * t).sin()).sum::<f32>() * 0.2);
            }
        }
        y
    }

    #[test]
    fn verse_chorus_verse_chorus_is_heard_as_a_b_a_b() {
        let rate = 22_050;
        let (c, g, am, f) = ([261.6, 329.6, 392.0], [196.0, 246.9, 293.7], [220.0, 261.6, 329.6], [174.6, 220.0, 261.6]);
        // A chorus whose chords share little with the verse's, so the
        // boundary is a fact of the music, not a judgement.
        let (e, b) = ([329.6, 415.3, 493.9], [246.9, 311.1, 370.0]);
        let verse = [c, g, am, f, c, g, am, f];
        let chorus = [e, b, e, b, e, b, e, b];
        let mut prog = Vec::new();
        for part in [&verse, &chorus, &verse, &chorus] {
            prog.extend_from_slice(part);
        }
        // 2 s a chord at 120 BPM: four beats each, 32 beats a part.
        let y = chords(&prog, 2.0, rate);
        let beats: Vec<f32> = (0..(y.len() as f32 / rate as f32 / 0.5) as usize).map(|i| i as f32 * 0.5).collect();
        let s = find(&y, rate, &beats);
        let labels: String = s.iter().map(|x| x.label).collect();
        assert_eq!(labels, "ABAB", "{s:?}");
        for (x, want) in s.iter().zip([0.0, 16.0, 32.0, 48.0]) {
            assert!((x.start - want).abs() <= 1.0, "{s:?}");
        }
    }

    #[test]
    fn one_chord_throughout_is_one_part() {
        let rate = 22_050;
        let y = chords(&[[261.6, 329.6, 392.0]; 16], 2.0, rate);
        let beats: Vec<f32> = (0..64).map(|i| i as f32 * 0.5).collect();
        assert_eq!(find(&y, rate, &beats).len(), 1);
    }

    #[test]
    fn too_few_beats_make_no_sections() {
        assert!(find(&[0.0; 1000], 22_050, &[0.0, 0.5, 1.0]).is_empty());
    }
}
