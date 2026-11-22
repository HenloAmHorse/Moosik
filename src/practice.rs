//! Practice (1.6.2): slower or faster without changing the pitch, and a
//! different key without changing the tempo. Pitch by resampling, tempo by
//! WSOLA (waveform-similarity overlap-add), at the very end of the shared
//! route, after the EQ; at normal tempo and key it passes samples through
//! untouched. Never in bit-perfect output.

use std::sync::{Arc, Mutex};

/// Tempo (1 = as recorded) and key shift (semitones).
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Setting {
    pub tempo: f32,
    pub semitones: f32,
}

impl Default for Setting {
    fn default() -> Self {
        Setting { tempo: 1.0, semitones: 0.0 }
    }
}

impl Setting {
    pub fn is_normal(&self) -> bool {
        self.tempo == 1.0 && self.semitones == 0.0
    }
}

pub type Handle = Arc<Mutex<Setting>>;

/// WSOLA time stretch on interleaved frames: `speed` input frames are used
/// for each output frame (2 = twice as fast), pitch unchanged.
pub struct Stretch {
    ch: usize,
    n: usize,
    hop: usize,
    tol: usize,
    win: Vec<f32>,
    win4: Vec<f32>,
    /// Input not yet compacted, interleaved; `input[0]` sits at logical frame `buf_start`.
    input: Vec<f32>,
    /// Per-frame channel sum (`0.0 + sum_c input[frame, c]`) parallel to `input`.
    mono: Vec<f32>,
    buf_start: usize,
    in_start: usize,
    /// Output being overlap-added, interleaved; the first `hop` frames are finished.
    out: Vec<f32>,
    ready: Vec<f32>,
    ready_pos: usize,
    b_nat: Vec<f32>,
    /// Where the next segment would ideally start, and where the last one did.
    ideal: f64,
    last: usize,
    started: bool,
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
#[allow(unsafe_op_in_unsafe_fn)]
unsafe fn best_cand_avx2(mono: &[f32], lo: usize, hi: usize, ideal: usize, b_nat: &[f32], win4: &[f32]) -> usize {
    use std::arch::x86_64::*;
    let np = b_nat.len();
    let count = hi - lo + 1;
    let mp = mono.as_ptr().add(lo);
    let bp = b_nat.as_ptr();
    let wp = win4.as_ptr();
    let mut scores = [0.0f32; 32];
    let mut best = (f32::MIN, ideal);
    let mut off = 0usize;
    while off + 32 <= count {
        let mut a0 = _mm256_setzero_ps();
        let mut a1 = _mm256_setzero_ps();
        let mut a2 = _mm256_setzero_ps();
        let mut a3 = _mm256_setzero_ps();
        let base = mp.add(off);
        for j in 0..np {
            let bv = _mm256_set1_ps(*bp.add(j));
            let wv = _mm256_set1_ps(*wp.add(j));
            let row = base.add(j * 4);
            // Preserve the exact `(a * b) * w` association order (no FMA, no `a * (b * w)`).
            a0 = _mm256_add_ps(a0, _mm256_mul_ps(_mm256_mul_ps(_mm256_loadu_ps(row), bv), wv));
            a1 = _mm256_add_ps(a1, _mm256_mul_ps(_mm256_mul_ps(_mm256_loadu_ps(row.add(8)), bv), wv));
            a2 = _mm256_add_ps(a2, _mm256_mul_ps(_mm256_mul_ps(_mm256_loadu_ps(row.add(16)), bv), wv));
            a3 = _mm256_add_ps(a3, _mm256_mul_ps(_mm256_mul_ps(_mm256_loadu_ps(row.add(24)), bv), wv));
        }
        _mm256_storeu_ps(scores.as_mut_ptr(), a0);
        _mm256_storeu_ps(scores.as_mut_ptr().add(8), a1);
        _mm256_storeu_ps(scores.as_mut_ptr().add(16), a2);
        _mm256_storeu_ps(scores.as_mut_ptr().add(24), a3);
        for (d, &s) in scores.iter().enumerate() {
            if s > best.0 {
                best = (s, lo + off + d);
            }
        }
        off += 32;
    }
    while off + 8 <= count {
        let mut a0 = _mm256_setzero_ps();
        let base = mp.add(off);
        for j in 0..np {
            let bv = _mm256_set1_ps(*bp.add(j));
            let wv = _mm256_set1_ps(*wp.add(j));
            a0 = _mm256_add_ps(a0, _mm256_mul_ps(_mm256_mul_ps(_mm256_loadu_ps(base.add(j * 4)), bv), wv));
        }
        _mm256_storeu_ps(scores.as_mut_ptr(), a0);
        for &s in &scores[..8] {
            let cand = lo + off;
            if s > best.0 {
                best = (s, cand);
            }
            off += 1;
        }
    }
    while off < count {
        let mut s = 0.0f32;
        let base = mp.add(off);
        for j in 0..np {
            s += (*base.add(j * 4) * *bp.add(j)) * *wp.add(j);
        }
        if s > best.0 {
            best = (s, lo + off);
        }
        off += 1;
    }
    best.1
}

fn best_cand(mono: &[f32], lo: usize, hi: usize, ideal: usize, b_nat: &[f32], win4: &[f32]) -> usize {
    #[cfg(target_arch = "x86_64")]
    if is_x86_feature_detected!("avx2") {
        // SAFETY: AVX2 support verified above, `b_nat.len() == win4.len()`, and
        // `hi + (b_nat.len() - 1) * 4 < mono.len()` by caller bounds.
        return unsafe { best_cand_avx2(mono, lo, hi, ideal, b_nat, win4) };
    }
    let mut best = (f32::MIN, ideal);
    for cand in lo..=hi {
        let mut s = 0.0f32;
        for (j, (&b, &w)) in b_nat.iter().zip(win4.iter()).enumerate() {
            s += (mono[cand + j * 4] * b) * w;
        }
        if s > best.0 {
            best = (s, cand);
        }
    }
    best.1
}

impl Stretch {
    pub fn new(rate: u32, ch: usize) -> Stretch {
        // About 40 ms frames, half overlapping, searching ±10 ms.
        let n = ((rate as f32 * 0.04) as usize).next_power_of_two().max(256);
        let hop = n / 2;
        let tol = n / 4;
        let win: Vec<f32> = (0..n).map(|i| (std::f32::consts::PI * (i as f32 + 0.5) / n as f32).sin().powi(2)).collect();
        let win4 = (0..n / 4).map(|j| win[j * 4]).collect();
        Stretch {
            ch,
            n,
            hop,
            tol,
            win,
            win4,
            input: Vec::new(),
            mono: Vec::new(),
            buf_start: 0,
            in_start: 0,
            out: vec![0.0; n * ch],
            ready: Vec::new(),
            ready_pos: 0,
            b_nat: vec![0.0; n / 4],
            ideal: 0.0,
            last: 0,
            started: false,
        }
    }

    pub fn push(&mut self, samples: &[f32]) {
        let prev_frames = self.mono.len();
        self.input.extend_from_slice(samples);
        let total_frames = self.input.len() / self.ch;
        let tail = &self.input[prev_frames * self.ch..total_frames * self.ch];
        for frame in tail.chunks_exact(self.ch) {
            let mut m = 0.0f32;
            for &x in frame {
                m += x;
            }
            self.mono.push(m);
        }
    }

    fn frames_in(&self) -> usize {
        self.input.len() / self.ch - (self.in_start - self.buf_start)
    }

    /// Make output while there is enough input; returns finished samples.
    #[inline(always)]
    pub fn pull(&mut self, speed: f64) -> Option<f32> {
        if self.ready_pos < self.ready.len() {
            let v = self.ready[self.ready_pos];
            self.ready_pos += 1;
            return Some(v);
        }
        self.pull_step(speed)
    }

    #[inline(never)]
    fn pull_step(&mut self, speed: f64) -> Option<f32> {
        let (n, hop, tol, ch) = (self.n, self.hop, self.tol, self.ch);
        // Candidate start, searched around the ideal place for the waveform
        // most like the natural continuation of the last segment.
        let ideal = self.ideal as usize;
        let lo = ideal.saturating_sub(tol).max(self.in_start);
        let need_end = ideal + tol + n;
        if self.in_start + self.frames_in() < need_end.max(self.last + hop + n) {
            return None;
        }
        let best = if !self.started {
            ideal
        } else {
            let natural = self.last + hop - self.buf_start;
            for (j, b) in self.b_nat.iter_mut().enumerate() {
                *b = self.mono[natural + j * 4];
            }
            self.buf_start
                + best_cand(
                    &self.mono,
                    lo - self.buf_start,
                    ideal + tol - self.buf_start,
                    ideal - self.buf_start,
                    &self.b_nat,
                    &self.win4,
                )
        };
        let src = &self.input[(best - self.buf_start) * ch..(best - self.buf_start + n) * ch];
        match ch {
            1 => {
                for ((o, &s), &w) in self.out.iter_mut().zip(src).zip(&self.win) {
                    *o += s * w;
                }
            }
            2 => {
                for ((o, s), &w) in self.out.chunks_exact_mut(2).zip(src.chunks_exact(2)).zip(&self.win) {
                    o[0] += s[0] * w;
                    o[1] += s[1] * w;
                }
            }
            _ => {
                for ((o, s), &w) in self.out.chunks_exact_mut(ch).zip(src.chunks_exact(ch)).zip(&self.win) {
                    for c in 0..ch {
                        o[c] += s[c] * w;
                    }
                }
            }
        }
        // The first hop is complete: hand it out and shift.
        self.ready.clear();
        self.ready.extend_from_slice(&self.out[..hop * ch]);
        self.ready_pos = 1;
        self.out.copy_within(hop * ch.., 0);
        self.out[hop * ch..].fill(0.0);
        self.last = best;
        self.started = true;
        self.ideal += hop as f64 * speed;
        // Forget input no later step can reach; compact lazily to avoid shifting every hop.
        let keep_from = (self.ideal as usize).saturating_sub(tol).min(self.last + hop);
        if keep_from > self.in_start {
            let drop = (keep_from - self.in_start).min(self.frames_in());
            self.in_start += drop;
            let head = self.in_start - self.buf_start;
            if head >= 16_384 {
                self.input.drain(..head * ch);
                self.mono.drain(..head);
                self.buf_start += head;
            }
        }
        Some(self.ready[0])
    }
}

/// The rodio stage: reads `inner`, applies the shared setting.
pub struct Practice<S: rodio::Source<Item = f32>> {
    inner: S,
    setting: Handle,
    local: Setting,
    rate: u32,
    ch: usize,
    stretch: Option<Stretch>,
    /// Pitch: a fractional read position over the input (linear interpolation).
    frac: f64,
    prev: Vec<f32>,
    next: Vec<f32>,
    count: u32,
    poll: u32,
}

impl<S: rodio::Source<Item = f32>> Practice<S> {
    pub fn new(inner: S, setting: Handle) -> Self {
        let (rate, ch) = (inner.sample_rate(), inner.channels().max(1) as usize);
        let local = setting.lock().map(|s| *s).unwrap_or_default();
        let mut p = Practice { inner, setting, local, rate, ch, stretch: None, frac: 0.0, prev: vec![0.0; ch], next: vec![0.0; ch], count: 0, poll: 0 };
        p.rebuild();
        p
    }

    fn rebuild(&mut self) {
        self.stretch = (!self.local.is_normal()).then(|| Stretch::new(self.rate, self.ch));
        self.frac = 0.0;
        self.count = 0;
    }

    /// One input frame after pitch: the input resampled by 2^(semitones/12).
    fn pitched_frame(&mut self, out: &mut Vec<f32>) -> bool {
        let ratio = 2f64.powf(self.local.semitones as f64 / 12.0);
        while self.frac >= 1.0 || self.count == 0 {
            if self.count == 0 {
                for c in 0..self.ch {
                    let Some(v) = self.inner.next() else { return false };
                    self.next[c] = v;
                }
                self.count = 1;
                continue;
            }
            std::mem::swap(&mut self.prev, &mut self.next);
            for c in 0..self.ch {
                let Some(v) = self.inner.next() else { return false };
                self.next[c] = v;
            }
            self.frac -= 1.0;
        }
        let t = self.frac as f32;
        out.extend((0..self.ch).map(|c| self.prev[c] + (self.next[c] - self.prev[c]) * t));
        self.frac += ratio;
        true
    }
}

impl<S: rodio::Source<Item = f32>> Iterator for Practice<S> {
    type Item = f32;
    fn next(&mut self) -> Option<f32> {
        // Pick up a new setting every 512 samples, never blocking.
        self.poll = self.poll.wrapping_add(1);
        if self.poll % 512 == 0 {
            let now = self.setting.try_lock().ok().map(|s| *s);
            if let Some(now) = now
                && now != self.local
            {
                let was_normal = self.local.is_normal();
                self.local = now;
                if was_normal != now.is_normal() {
                    self.rebuild();
                }
            }
        }
        if self.local.is_normal() {
            return self.inner.next();
        }
        let pitch = 2f64.powf(self.local.semitones as f64 / 12.0);
        let speed = self.local.tempo as f64 / pitch;
        loop {
            if let Some(v) = self.stretch.as_mut().and_then(|s| s.pull(speed)) {
                return Some(v);
            }
            let mut frame = Vec::with_capacity(self.ch * 256);
            for _ in 0..256 {
                if !self.pitched_frame(&mut frame) {
                    break;
                }
            }
            if frame.is_empty() {
                return None;
            }
            if let Some(s) = self.stretch.as_mut() {
                s.push(&frame);
            }
        }
    }
}

impl<S: rodio::Source<Item = f32>> rodio::Source for Practice<S> {
    fn current_frame_len(&self) -> Option<usize> {
        None
    }
    fn channels(&self) -> u16 {
        self.ch as u16
    }
    fn sample_rate(&self) -> u32 {
        self.rate
    }
    fn total_duration(&self) -> Option<std::time::Duration> {
        None
    }
    fn try_seek(&mut self, pos: std::time::Duration) -> Result<(), rodio::source::SeekError> {
        let r = self.inner.try_seek(pos);
        self.rebuild();
        r
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn sine(f: f32, secs: f32, rate: u32) -> Vec<f32> {
        (0..(secs * rate as f32) as usize).map(|i| (std::f32::consts::TAU * f * i as f32 / rate as f32).sin() * 0.5).collect()
    }

    /// Frequency by zero crossings over the middle of the signal.
    fn freq(y: &[f32], rate: u32) -> f32 {
        let mid = &y[y.len() / 4..3 * y.len() / 4];
        let z = mid.windows(2).filter(|w| w[0] <= 0.0 && w[1] > 0.0).count();
        z as f32 / (mid.len() as f32 / rate as f32)
    }

    fn run(setting: Setting, input: Vec<f32>) -> Vec<f32> {
        let h = Arc::new(Mutex::new(setting));
        Practice::new(rodio::buffer::SamplesBuffer::new(1, 44_100, input), h).collect()
    }

    #[test]
    fn at_normal_tempo_and_key_it_is_untouched() {
        let x = sine(440.0, 1.0, 44_100);
        assert_eq!(run(Setting::default(), x.clone()), x);
    }

    #[test]
    fn slower_keeps_the_pitch() {
        let x = sine(440.0, 4.0, 44_100);
        let y = run(Setting { tempo: 0.5, semitones: 0.0 }, x.clone());
        let ratio = y.len() as f32 / x.len() as f32;
        assert!((ratio - 2.0).abs() < 0.05, "twice as long: {ratio}");
        assert!((freq(&y, 44_100) - 440.0).abs() < 5.0, "{}", freq(&y, 44_100));
    }

    #[test]
    fn faster_keeps_the_pitch() {
        let x = sine(440.0, 4.0, 44_100);
        let y = run(Setting { tempo: 1.25, semitones: 0.0 }, x.clone());
        let ratio = y.len() as f32 / x.len() as f32;
        assert!((ratio - 0.8).abs() < 0.05, "{ratio}");
        assert!((freq(&y, 44_100) - 440.0).abs() < 5.0, "{}", freq(&y, 44_100));
    }

    #[test]
    fn a_new_key_keeps_the_tempo() {
        let x = sine(440.0, 4.0, 44_100);
        let y = run(Setting { tempo: 1.0, semitones: 12.0 }, x.clone());
        let ratio = y.len() as f32 / x.len() as f32;
        assert!((ratio - 1.0).abs() < 0.05, "{ratio}");
        assert!((freq(&y, 44_100) - 880.0).abs() < 10.0, "{}", freq(&y, 44_100));
        let y = run(Setting { tempo: 1.0, semitones: -2.0 }, x);
        assert!((freq(&y, 44_100) - 440.0 * 2f32.powf(-2.0 / 12.0)).abs() < 6.0, "{}", freq(&y, 44_100));
    }

    // -----------------------------------------------------------------------
    // Verbatim OldStretch from de4fcb3 (VecDeque + `s += a * b * self.win[i]`)
    // -----------------------------------------------------------------------
    use std::collections::VecDeque;

    struct OldStretch {
        ch: usize,
        n: usize,
        hop: usize,
        tol: usize,
        win: Vec<f32>,
        input: VecDeque<f32>,
        in_start: usize,
        out: Vec<f32>,
        ready: VecDeque<f32>,
        ideal: f64,
        last: usize,
        started: bool,
    }

    impl OldStretch {
        fn new(rate: u32, ch: usize) -> OldStretch {
            let n = ((rate as f32 * 0.04) as usize).next_power_of_two().max(256);
            let hop = n / 2;
            let tol = n / 4;
            let win = (0..n).map(|i| (std::f32::consts::PI * (i as f32 + 0.5) / n as f32).sin().powi(2)).collect();
            OldStretch { ch, n, hop, tol, win, input: VecDeque::new(), in_start: 0, out: vec![0.0; n * ch], ready: VecDeque::new(), ideal: 0.0, last: 0, started: false }
        }

        fn push(&mut self, samples: &[f32]) {
            self.input.extend(samples.iter().copied());
        }

        fn frames_in(&self) -> usize {
            self.input.len() / self.ch
        }

        fn at(&self, frame: usize, c: usize) -> f32 {
            self.input.get((frame - self.in_start) * self.ch + c).copied().unwrap_or(0.0)
        }

        fn pull(&mut self, speed: f64) -> Option<f32> {
            if let Some(v) = self.ready.pop_front() {
                return Some(v);
            }
            let (n, hop, tol, ch) = (self.n, self.hop, self.tol, self.ch);
            let ideal = self.ideal as usize;
            let lo = ideal.saturating_sub(tol).max(self.in_start);
            let need_end = ideal + tol + n;
            if self.in_start + self.frames_in() < need_end.max(self.last + hop + n) {
                return None;
            }
            let best = if !self.started {
                ideal
            } else {
                let natural = self.last + hop;
                let mut best = (f32::MIN, ideal);
                for cand in lo..=ideal + tol {
                    let mut s = 0.0;
                    for i in (0..n).step_by(4) {
                        let (mut a, mut b) = (0.0, 0.0);
                        for c in 0..ch {
                            a += self.at(cand + i, c);
                            b += self.at(natural + i, c);
                        }
                        s += a * b * self.win[i];
                    }
                    if s > best.0 {
                        best = (s, cand);
                    }
                }
                best.1
            };
            for i in 0..n {
                for c in 0..ch {
                    self.out[i * ch + c] += self.at(best + i, c) * self.win[i];
                }
            }
            self.ready.extend(self.out.drain(..hop * ch));
            self.out.extend(std::iter::repeat_n(0.0, hop * ch));
            self.last = best;
            self.started = true;
            self.ideal += hop as f64 * speed;
            let keep_from = (self.ideal as usize).saturating_sub(tol).min(self.last + hop);
            if keep_from > self.in_start {
                let drop = (keep_from - self.in_start).min(self.frames_in());
                self.input.drain(..drop * ch);
                self.in_start += drop;
            }
            self.ready.pop_front()
        }
    }

    fn run_both_and_assert_bits(rate: u32, ch: usize, speed: f64, chunk_frames: usize, input: &[f32], label: &str) {
        let mut st_new = Stretch::new(rate, ch);
        let mut st_old = OldStretch::new(rate, ch);
        let mut out_new = Vec::new();
        let mut out_old = Vec::new();
        for chunk in input.chunks(chunk_frames * ch) {
            st_new.push(chunk);
            st_old.push(chunk);
            while let Some(v) = st_new.pull(speed) {
                out_new.push(v.to_bits());
            }
            while let Some(v) = st_old.pull(speed) {
                out_old.push(v.to_bits());
            }
        }
        assert_eq!(out_new.len(), out_old.len(), "{label}: output length mismatch");
        for (idx, (&bn, &bo)) in out_new.iter().zip(&out_old).enumerate() {
            assert_eq!(
                bn, bo,
                "{label}: sample {idx} bits mismatch (rate={rate}, ch={ch}, speed={speed}, chunk={chunk_frames}): new={:#010x} ({}) old={:#010x} ({})",
                bn, f32::from_bits(bn), bo, f32::from_bits(bo)
            );
        }
    }

    #[test]
    fn stretch_matches_unchanged_old_stretch_bit_for_bit() {
        // 1. All required speeds (0.5, 0.75, 1.25, 1.5) x channels (1, 2, 3) x rates & chunkings
        for &rate in &[44_100u32, 48_000] {
            for &ch in &[1usize, 2, 3] {
                let frames = rate as usize / 4;
                let mut sig = Vec::with_capacity(frames * ch);
                let mut seed = 0x1234_5678_9ABC_DEF0u64 ^ ((rate as u64) << 16) ^ (ch as u64);
                for i in 0..frames {
                    for c in 0..ch {
                        seed ^= seed << 13;
                        seed ^= seed >> 7;
                        seed ^= seed << 17;
                        let noise = ((seed >> 40) as f32 / 16_777_216.0) - 0.5;
                        let ph = i as f32 / rate as f32 * (220.0 + 110.0 * c as f32) * std::f32::consts::TAU;
                        sig.push(ph.sin() * 0.45 + (2.0 * ph).cos() * 0.2 + noise * 0.15);
                    }
                }
                for &speed in &[0.5f64, 0.75, 1.25, 1.5] {
                    for &chunk_frames in &[frames, 256, 19] {
                        run_both_and_assert_bits(rate, ch, speed, chunk_frames, &sig, "multi-harmonic+noise");
                    }
                }
            }
        }

        // 2. Periodic signal and exact-tie / near-tie candidate signals, plus lazy compaction (> 16384 frames)
        let rate = 48_000u32;
        let frames = 24_000usize;
        // Exact periodic wave (period = 64 frames -> multiple identical peaks in ±tol search range)
        let periodic: Vec<f32> = (0..frames * 2)
            .map(|i| {
                let f = i / 2;
                let c = i % 2;
                ((f % 64) as f32 / 64.0 * std::f32::consts::TAU + c as f32 * 0.3).sin() * 0.6
            })
            .collect();
        for &speed in &[0.5f64, 0.75, 1.25, 1.5] {
            run_both_and_assert_bits(rate, 2, speed, 256, &periodic, "periodic-ties");
        }
    }

    #[test]
    fn witness_catches_reassociated_pre_multiplied_window_bug() {
        // Demonstrates a concrete witness where pre-multiplying `a * (b * w)`
        // disagrees with the original `(a * b) * w` association on near-tie
        // candidates, while `Stretch` matches `OldStretch` bit-for-bit.
        let rate = 8_000u32; // n = 512, hop = 256, tol = 128
        let mut sig = vec![0.0f32; 2048];
        let win: Vec<f32> = (0..512).map(|i| (std::f32::consts::PI * (i as f32 + 0.5) / 512.0).sin().powi(2)).collect();
        let w0 = win[200]; // non-power-of-two window value

        // Find (a0, a1, b) where (a0 * b) * w0 == (a1 * b) * w0 (tie, so earlier cand wins in original),
        // while a1 * (b * w0) > a0 * (b * w0) (so buggy pre-multiplied version picks later cand!).
        let b = f32::from_bits(0x3f35_04f3);
        let bw = b * w0;
        let mut found = None;
        for step in 0..200_000u32 {
            let a0 = f32::from_bits(0x3f10_0000 + step);
            let a1 = f32::from_bits(0x3f10_0000 + step + 1);
            let orig0 = (a0 * b) * w0;
            let orig1 = (a1 * b) * w0;
            let bug0 = a0 * bw;
            let bug1 = a1 * bw;
            if orig1 <= orig0 && bug1 > bug0 {
                found = Some((a0, a1));
                break;
            }
        }
        let (a0, a1) = found.expect("found IEEE-754 association witness");
        // With `speed = 0.25`, step 1 starts at 0 (`last = 0`), and step 2 has
        // `natural = 256`, `ideal = 64`, `lo..=ideal + tol = 0..=192` (disjoint from `natural = 256`).
        sig[256 + 200] = b;
        sig[100 + 200] = a0; // cand = 100
        sig[140 + 200] = a1; // cand = 140
        // Distinguish the two candidate regions at sample offset +1 (not sampled by step_by(4) score!)
        sig[100 + 1] = 0.25;
        sig[140 + 1] = 0.75;

        let mut st_new = Stretch::new(rate, 1);
        let mut st_old = OldStretch::new(rate, 1);
        st_new.push(&sig);
        st_old.push(&sig);
        let mut out_new = Vec::new();
        let mut out_old = Vec::new();
        while let Some(v) = st_new.pull(0.25) {
            out_new.push(v.to_bits());
        }
        while let Some(v) = st_old.pull(0.25) {
            out_old.push(v.to_bits());
        }
        assert_eq!(out_new, out_old, "restored `(a * b) * w` matches OldStretch on the association witness");

        // Verify that a buggy `a * (b * w)` evaluator chooses cand 140 instead of cand 100:
        let mut target_buggy = vec![0.0f32; 512 / 4];
        for j in 0..512 / 4 {
            target_buggy[j] = sig[256 + j * 4] * win[j * 4];
        }
        let mut best_buggy = (f32::MIN, 0usize);
        for cand in 0..=192 {
            let mut s = 0.0f32;
            for j in 0..512 / 4 {
                s += sig[cand + j * 4] * target_buggy[j];
            }
            if s > best_buggy.0 {
                best_buggy = (s, cand);
            }
        }
        let win4: Vec<f32> = (0..512 / 4).map(|j| win[j * 4]).collect();
        let b_nat: Vec<f32> = (0..512 / 4).map(|j| sig[256 + j * 4]).collect();
        let best_fixed = best_cand(&sig, 0, 192, 64, &b_nat, &win4);
        assert_eq!(best_fixed, 100);
        assert_eq!(best_buggy.1, 140);
    }

    #[test]
    fn witness_catches_all_nan_fallback_to_ideal() {
        // Codex Round 2 witness:
        // rate = 8000 (n = 512, hop = 256, tol = 128), mono, 2048 frames, speed = 0.75.
        // All 0.0 except input[456] = NaN, input[193] = 1.0.
        // On segment 2:
        // - last = 0, hop = 256 -> natural = 256; 256 + 50 * 4 = 456 -> b_nat[50] = NaN.
        // - Every candidate score in lo..=hi (64..=320) is NaN, so `s > best.0` is false for all.
        // - Original de4fcb3 initializes `best = (f32::MIN, ideal)` where `ideal = 192`, so it picks 192.
        // - A buggy fallback `best = (f32::MIN, lo)` would pick 64 instead of 192, missing `input[193] = 1.0`
        //   at offset 1 of the second segment (output sample index 257).
        let rate = 8_000u32;
        let mut input = vec![0.0f32; 2048];
        input[456] = f32::NAN;
        input[193] = 1.0;

        let win: Vec<f32> = (0..512).map(|i| (std::f32::consts::PI * (i as f32 + 0.5) / 512.0).sin().powi(2)).collect();
        let win4: Vec<f32> = (0..512 / 4).map(|j| win[j * 4]).collect();
        let b_nat: Vec<f32> = (0..512 / 4).map(|j| input[256 + j * 4]).collect();
        assert!(b_nat[50].is_nan());

        // Direct check on best_cand: lo = 64, hi = 320, ideal = 192 -> must return 192 (not 64)
        let chosen = best_cand(&input, 64, 320, 192, &b_nat, &win4);
        assert_eq!(chosen, 192, "all-NaN candidate scores must fall back to ideal (192), not lo (64)");

        // End-to-end check on Stretch vs OldStretch:
        let mut st_new = Stretch::new(rate, 1);
        let mut st_old = OldStretch::new(rate, 1);
        st_new.push(&input);
        st_old.push(&input);
        assert_eq!(st_new.pull(0.75).unwrap().to_bits(), st_old.pull(0.75).unwrap().to_bits());
        assert_eq!(st_new.last, 0);
        assert_eq!(st_old.last, 0);

        // Drain the rest of segment 1 (samples 1..256)
        for idx in 1..256 {
            let vn = st_new.pull(0.75).unwrap();
            let vo = st_old.pull(0.75).unwrap();
            assert_eq!(vn.to_bits(), vo.to_bits(), "seg 1 sample {idx}");
        }

        // Trigger segment 2 pull (sample 256) and verify `last == 192` and finite sample 257 > 0:
        let s256_new = st_new.pull(0.75).unwrap();
        let s256_old = st_old.pull(0.75).unwrap();
        assert_eq!(s256_new.to_bits(), s256_old.to_bits());
        assert_eq!(st_new.last, 192, "segment 2 must select ideal = 192");
        assert_eq!(st_old.last, 192);

        let s257_new = st_new.pull(0.75).unwrap();
        let s257_old = st_old.pull(0.75).unwrap();
        assert!(s257_old.is_finite() && s257_old > 0.0, "sample 257 reflects input[193] = 1.0");
        assert_eq!(s257_new.to_bits(), s257_old.to_bits(), "sample 257 finite output matches bit-for-bit");

        // Drain all remaining samples and verify bit-for-bit equality (and same NaN bit pattern where NaN)
        let mut idx = 258usize;
        while let Some(vo) = st_old.pull(0.75) {
            let vn = st_new.pull(0.75).expect("same length");
            assert_eq!(vn.to_bits(), vo.to_bits(), "sample {idx} mismatch");
            idx += 1;
        }
        assert!(st_new.pull(0.75).is_none());
    }

    #[test]
    #[ignore = "same-binary A/B benchmark on real music (tempo 0.75 and 1.25)"]
    fn bench_practice_ab_real_music() {
        use std::hint::black_box;
        use std::path::Path;
        use std::sync::atomic::{AtomicBool, AtomicU32};
        use std::time::Instant;

        let path = Path::new(r"C:\Users\PC\Documents\ASYN\Audio Archive\Vocaloid\01_02_退廃的人生讃歌.flac");
        let cancel = AtomicBool::new(false);
        let prog = AtomicU32::new(0);
        let mut samples = Vec::new();
        let mut sr = 48_000u32;
        let mut ch = 2usize;
        crate::measure::each_block(path, &cancel, &prog, |rate, channels, blk| {
            sr = rate;
            ch = channels;
            if samples.len() < (rate as usize) * channels * 30 {
                let rem = ((rate as usize) * channels * 30) - samples.len();
                samples.extend_from_slice(&blk[..blk.len().min(rem)]);
            }
        }).expect("decode real music");
        let secs = (samples.len() / ch) as f64 / sr as f64;

        for &speed in &[0.75f64, 1.25] {
            let run_old = || -> (f64, u32, usize) {
                let mut st = OldStretch::new(sr, ch);
                let t0 = Instant::now();
                let mut xor = 0u32;
                let mut count = 0usize;
                for chunk in samples.chunks(256 * ch) {
                    st.push(black_box(chunk));
                    while let Some(v) = st.pull(speed) {
                        xor ^= black_box(v).to_bits();
                        count += 1;
                    }
                }
                (t0.elapsed().as_secs_f64() * 1e3, xor, count)
            };

            let run_new = || -> (f64, u32, usize) {
                let mut st = Stretch::new(sr, ch);
                let t0 = Instant::now();
                let mut xor = 0u32;
                let mut count = 0usize;
                for chunk in samples.chunks(256 * ch) {
                    st.push(black_box(chunk));
                    while let Some(v) = st.pull(speed) {
                        xor ^= black_box(v).to_bits();
                        count += 1;
                    }
                }
                (t0.elapsed().as_secs_f64() * 1e3, xor, count)
            };

            // Verify exact sample-by-sample bit equality on the 30s real music track first
            {
                let mut st_o = OldStretch::new(sr, ch);
                let mut st_n = Stretch::new(sr, ch);
                let mut vo = Vec::new();
                let mut vn = Vec::new();
                for chunk in samples.chunks(256 * ch) {
                    st_o.push(chunk);
                    st_n.push(chunk);
                    while let Some(v) = st_o.pull(speed) { vo.push(v.to_bits()); }
                    while let Some(v) = st_n.pull(speed) { vn.push(v.to_bits()); }
                }
                assert_eq!(vo.len(), vn.len());
                let diffs = vo.iter().zip(&vn).filter(|(a, b)| a != b).count();
                assert_eq!(diffs, 0, "speed={speed}: {diffs} bit differences on real music!");
                println!(
                    "\n=== Practice::Stretch {:.1}s @ {}Hz {}ch speed={speed}: {} output samples, bit_diffs=0 ===",
                    secs, sr, ch, vn.len()
                );
            }

            let mut old_ms = Vec::new();
            let mut new_ms = Vec::new();
            for pair in 0..5 {
                let (o, n, xo, xn) = if pair % 2 == 0 {
                    let (o, xo, co) = run_old();
                    let (n, xn, cn) = run_new();
                    assert_eq!(co, cn);
                    (o, n, xo, xn)
                } else {
                    let (n, xn, cn) = run_new();
                    let (o, xo, co) = run_old();
                    assert_eq!(co, cn);
                    (o, n, xo, xn)
                };
                assert_eq!(xo, xn);
                println!("  pair {}: old={o:.2}ms, new={n:.2}ms (xor={xo:#010x})", pair + 1);
                old_ms.push(o);
                new_ms.push(n);
            }
            old_ms.sort_by(f64::total_cmp);
            new_ms.sort_by(f64::total_cmp);
            let med_old = old_ms[old_ms.len() / 2];
            let med_new = new_ms[new_ms.len() / 2];
            let out_secs = secs / speed;
            let cpu_old = (med_old / (out_secs * 1000.0)) * 100.0;
            let cpu_new = (med_new / (out_secs * 1000.0)) * 100.0;
            println!(
                "  MEDIAN ({:.1}s in -> {:.1}s out): old={med_old:.2}ms -> new={med_new:.2}ms ({:.2}x, -{:.1}%) | Playback 1-core CPU: {cpu_old:.2}% -> {cpu_new:.2}%",
                secs, out_secs, med_old / med_new, (1.0 - med_new / med_old) * 100.0
            );
        }
    }
}
