//! DSD → PCM decimation — the spectrum analyzer's feed for DSD tracks
//! (phase 3 of DSD support). Playback never touches this: the DoP path sends
//! the raw bits to the DAC; this exists so analysis (spectrum, waveform,
//! LUFS, DR) can run on a faithful PCM rendering of the audible band.
//!
//! Two stages:
//!
//! 1. **Bit domain ÷8** — a 128-tap windowed-sinc FIR evaluated with byte
//!    lookup tables (each table folds 8 taps × 256 bit patterns), producing
//!    f32 PCM at `dsd_rate / 8` (DSD64 → 352.8 kHz) for ~2 table adds per
//!    input byte. Cutoff is low enough that everything folding into the final
//!    passband is already in the stopband.
//! 2. **PCM ÷2 cascade** — half-band FIRs (63 taps, even taps zero by
//!    construction ≈ 17 multiplies per output) halve the rate `k` times to
//!    land nearest the requested analysis rate: 176.4 kHz default, 352.8 kHz
//!    for a faster-reacting spectrum. 48k-family DSD lands on 192/384 kHz.
//!
//! DSD's rising ultrasonic noise shelf is exactly what these filters remove;
//! what remains is the audible band the analyzer should be showing.

use std::fs::File;
use std::io::{BufReader, Read, Seek};
use std::path::Path;

use super::dop::DSD_SILENCE;
use super::DsdReader;

/// Selectable analysis target rates (44.1k-family; a 48k-family DSD source
/// lands on 192/384 kHz via nearest-rate matching).
pub const ANALYSIS_RATES: [u32; 2] = [176_400, 352_800];
pub const DEFAULT_ANALYSIS_RATE: u32 = 176_400;

/// The output rate actually produced for a DSD source rate and requested
/// target: `dsd_rate / (8 × 2^k)` with `k` chosen nearest the target.
pub fn analysis_rate_for(dsd_rate: u32, target: u32) -> u32 {
    (0..=6u32)
        .map(|k| dsd_rate / (8 << k))
        .min_by_key(|&r| r.abs_diff(target))
        .unwrap_or(dsd_rate / 16)
}

fn halvings_for(dsd_rate: u32, target: u32) -> u32 {
    (0..=6u32)
        .min_by_key(|&k| (dsd_rate / (8 << k)).abs_diff(target))
        .unwrap_or(1)
}

// ---------------------------------------------------------------------------
// Filter design (windowed sinc)
// ---------------------------------------------------------------------------

/// Blackman window value at tap `j` of an `n`-tap filter.
fn blackman(j: usize, n: usize) -> f64 {
    let x = j as f64 / (n - 1) as f64;
    0.42 - 0.5 * (2.0 * std::f64::consts::PI * x).cos()
        + 0.08 * (4.0 * std::f64::consts::PI * x).cos()
}

/// Symmetric low-pass FIR: `n` taps, cutoff `fc` (cycles/sample of the input
/// rate), Blackman-windowed sinc, normalised to unity DC gain.
fn design_lowpass(n: usize, fc: f64) -> Vec<f32> {
    let c = (n - 1) as f64 / 2.0;
    let mut taps: Vec<f64> = (0..n)
        .map(|j| {
            let t = j as f64 - c;
            let sinc = if t.abs() < 1e-9 {
                2.0 * fc
            } else {
                (2.0 * std::f64::consts::PI * fc * t).sin() / (std::f64::consts::PI * t)
            };
            sinc * blackman(j, n)
        })
        .collect();
    let sum: f64 = taps.iter().sum();
    for t in &mut taps {
        *t /= sum;
    }
    taps.into_iter().map(|t| t as f32).collect()
}

// ---------------------------------------------------------------------------
// Stage 1: 1-bit ÷8 via byte LUTs
// ---------------------------------------------------------------------------

/// Stage-1 filter length: 16 history bytes = 128 taps.
const S1_BYTES: usize = 16;
const S1_TAPS: usize = S1_BYTES * 8;
/// Stage-1 cutoff (fraction of the DSD rate): 0.45 × the ÷8 output Nyquist.
/// With 128 Blackman taps the stopband starts around 0.077 — everything that
/// can fold into the final analysis band (≤ ¼ of the stage-1 output rate
/// whenever at least one ÷2 stage follows) is fully attenuated.
const S1_CUTOFF: f64 = 0.45 / 8.0;

/// Per-group byte tables: `lut[g][byte]` = Σ taps[8g+i] × (±1 per bit i),
/// where bit `i` of a history byte is the sample at lag `8g + i` (the
/// reader's bytes are MSB-first, so the LSB is the newest sample of a byte).
struct Stage1 {
    luts: Vec<[f32; 256]>,
}

impl Stage1 {
    fn new() -> Self {
        let taps = design_lowpass(S1_TAPS, S1_CUTOFF);
        let mut luts = vec![[0.0f32; 256]; S1_BYTES];
        for (g, lut) in luts.iter_mut().enumerate() {
            for (byte, slot) in lut.iter_mut().enumerate() {
                let mut acc = 0.0f32;
                for (i, &tap) in taps[g * 8..g * 8 + 8].iter().enumerate() {
                    let bit = (byte >> i) & 1;
                    acc += if bit == 1 { tap } else { -tap };
                }
                *slot = acc;
            }
        }
        Stage1 { luts }
    }

    /// One channel's output for its history ring (`hist[0]` = newest byte).
    #[inline]
    fn eval(&self, hist: &[u8; S1_BYTES]) -> f32 {
        self.luts
            .iter()
            .zip(hist.iter())
            .map(|(lut, &b)| lut[b as usize])
            .sum()
    }
}

// ---------------------------------------------------------------------------
// Stage 2: half-band ÷2
// ---------------------------------------------------------------------------

/// Half-band FIR length (odd). fc = 0.25 makes every even-offset tap zero
/// except the centre, so only ~N/4 multiplies survive per output.
const HB_TAPS: usize = 63;

/// One ÷2 half-band decimator: shared (index, coeff) pairs + per-instance
/// delay line. `push` consumes one input and yields an output on every
/// second sample.
struct Halfband {
    ring: Vec<f32>,
    pos: usize,
    phase: bool,
}

/// The nonzero (offset, coefficient) pairs of the half-band filter, shared
/// by every instance.
struct HalfbandCoeffs {
    pairs: Vec<(usize, f32)>,
}

impl HalfbandCoeffs {
    fn new() -> Self {
        let taps = design_lowpass(HB_TAPS, 0.25);
        let pairs = taps
            .iter()
            .enumerate()
            .filter(|&(_, &t)| t.abs() > 1e-9)
            .map(|(j, &t)| (j, t))
            .collect();
        HalfbandCoeffs { pairs }
    }
}

impl Halfband {
    fn new() -> Self {
        Halfband { ring: vec![0.0; HB_TAPS], pos: 0, phase: false }
    }

    fn reset(&mut self) {
        self.ring.fill(0.0);
        self.pos = 0;
        self.phase = false;
    }

    #[inline]
    fn push(&mut self, x: f32, co: &HalfbandCoeffs) -> Option<f32> {
        self.ring[self.pos] = x;
        self.pos = (self.pos + 1) % HB_TAPS;
        self.phase = !self.phase;
        if self.phase {
            return None;
        }
        // ring[pos] is the oldest sample (lag HB_TAPS-1); taps are symmetric
        // so ascending-lag vs descending-lag orientation is equivalent.
        let mut acc = 0.0f32;
        for &(j, t) in &co.pairs {
            acc += t * self.ring[(self.pos + j) % HB_TAPS];
        }
        Some(acc)
    }
}

// ---------------------------------------------------------------------------
// The full source
// ---------------------------------------------------------------------------

/// Streams a DSD file as interleaved f32 PCM at the analysis rate. Implements
/// `Iterator<Item = f32>` so it slots in wherever a rodio decoder's sample
/// stream would.
pub struct DsdPcmSource<R: Read + Seek> {
    reader: DsdReader<R>,
    stage1: Stage1,
    hb_coeffs: HalfbandCoeffs,
    /// Per-channel stage-1 byte history, newest first.
    hist: Vec<[u8; S1_BYTES]>,
    /// `cascade[k][ch]` — the k-th ÷2 stage's state for each channel.
    cascade: Vec<Vec<Halfband>>,
    channels: usize,
    out_rate: u32,
    /// Interleaved frames read from the container, consumed in chunks.
    inbuf: Vec<u8>,
    in_pos: usize,
    in_len: usize,
    /// Finished output samples (interleaved), drained by the iterator.
    outbuf: std::collections::VecDeque<f32>,
    eof: bool,
}

pub type DsdFilePcmSource = DsdPcmSource<BufReader<File>>;

/// Open a DSD file as a PCM sample stream near `target_rate`.
pub fn open_pcm_source(path: &Path, target_rate: u32) -> Result<DsdFilePcmSource, String> {
    DsdPcmSource::new(super::open_reader(path)?, target_rate)
}

impl<R: Read + Seek> DsdPcmSource<R> {
    pub fn new(reader: DsdReader<R>, target_rate: u32) -> Result<Self, String> {
        let info = reader.info().clone();
        let channels = info.channels as usize;
        let halvings = halvings_for(info.sample_rate, target_rate);
        let out_rate = info.sample_rate / (8 << halvings);
        if out_rate == 0 {
            return Err("DSD rate too low to decimate".into());
        }
        // Prime the bit history with DSD silence so the filter warm-up sits
        // at ≈0 instead of the −FS thump all-zero bits would produce.
        let hist = vec![[DSD_SILENCE; S1_BYTES]; channels];
        let cascade = (0..halvings)
            .map(|_| (0..channels).map(|_| Halfband::new()).collect())
            .collect();
        Ok(DsdPcmSource {
            reader,
            stage1: Stage1::new(),
            hb_coeffs: HalfbandCoeffs::new(),
            hist,
            cascade,
            channels,
            out_rate,
            inbuf: vec![0u8; 4096 * channels],
            in_pos: 0,
            in_len: 0,
            outbuf: std::collections::VecDeque::new(),
            eof: false,
        })
    }

    pub fn out_rate(&self) -> u32 { self.out_rate }
    pub fn channels(&self) -> usize { self.channels }

    /// Track length (from the DSD header's sample count).
    pub fn duration(&self) -> std::time::Duration {
        self.reader.info().duration()
    }

    /// Expected total output samples (all channels) — a progress hint.
    pub fn total_out_samples(&self) -> u64 {
        let per_ch = self.reader.total_frames() / (1 << self.cascade.len());
        per_ch * self.channels as u64
    }

    /// Jump to a time position. DSD is byte-addressable, so this is instant:
    /// seek the reader and restart the filters from their primed state (the
    /// few-ms warm-up transient sits at ≈0 thanks to the silence priming).
    pub fn seek_to_time(&mut self, pos: std::time::Duration) -> std::io::Result<()> {
        let info = self.reader.info();
        // Reader frames are bytes-per-channel = 8 DSD samples.
        let frame = (pos.as_secs_f64() * info.sample_rate as f64 / 8.0) as u64;
        self.reader.seek_to_frame(frame)?;
        for h in self.hist.iter_mut() {
            *h = [DSD_SILENCE; S1_BYTES];
        }
        for stage in self.cascade.iter_mut() {
            for hb in stage.iter_mut() {
                hb.reset();
            }
        }
        self.in_pos = 0;
        self.in_len = 0;
        self.outbuf.clear();
        self.eof = false;
        Ok(())
    }

    /// Process buffered input frames until the out queue has something or
    /// input is exhausted.
    fn pump(&mut self) {
        while self.outbuf.is_empty() {
            if self.in_pos >= self.in_len {
                if self.eof {
                    return;
                }
                match self.reader.read_frames(&mut self.inbuf) {
                    Ok(0) | Err(_) => { self.eof = true; return; }
                    Ok(n) => { self.in_len = n * self.channels; self.in_pos = 0; }
                }
            }
            let ch = self.channels;
            while self.in_pos + ch <= self.in_len {
                let frame = &self.inbuf[self.in_pos..self.in_pos + ch];
                self.in_pos += ch;
                for (c, &byte) in frame.iter().enumerate() {
                    let h = &mut self.hist[c];
                    h.copy_within(0..S1_BYTES - 1, 1);
                    h[0] = byte;
                    let mut v = self.stage1.eval(&self.hist[c]);
                    let mut alive = true;
                    for stage in self.cascade.iter_mut() {
                        match stage[c].push(v, &self.hb_coeffs) {
                            Some(next) => v = next,
                            None => { alive = false; break; }
                        }
                    }
                    if alive {
                        self.outbuf.push_back(v);
                    }
                }
                if !self.outbuf.is_empty() && self.in_pos + ch > self.in_len {
                    break;
                }
            }
            if !self.outbuf.is_empty() {
                return;
            }
        }
    }
}

impl<R: Read + Seek> Iterator for DsdPcmSource<R> {
    type Item = f32;

    fn next(&mut self) -> Option<f32> {
        if self.outbuf.is_empty() {
            self.pump();
        }
        self.outbuf.pop_front()
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::super::{parse_dsf, tests::make_dsf, DSD64_RATE};
    use super::*;
    use std::io::Cursor;

    fn source_from_bytes(
        ch_bytes: &[Vec<u8>], // per-channel byte streams (MSB-first temporal)
        rate: u32,
        target: u32,
    ) -> DsdPcmSource<Cursor<Vec<u8>>> {
        // Build via a DSF container (LSB-first storage, so reverse bits here;
        // the reader reverses them back).
        let block = ch_bytes[0].len();
        let blocks = vec![ch_bytes
            .iter()
            .map(|b| b.iter().map(|x| x.reverse_bits()).collect())
            .collect()];
        let file = make_dsf(
            ch_bytes.len() as u32, rate, 1,
            (block * 8) as u64, block as u32, &blocks, None,
        );
        let info = parse_dsf(&mut Cursor::new(&file)).unwrap();
        let reader = DsdReader::new(Cursor::new(file), info).unwrap();
        DsdPcmSource::new(reader, target).unwrap()
    }

    #[test]
    fn rate_mapping() {
        assert_eq!(analysis_rate_for(2_822_400, 176_400), 176_400); // DSD64 ÷16
        assert_eq!(analysis_rate_for(2_822_400, 352_800), 352_800); // DSD64 ÷8
        assert_eq!(analysis_rate_for(5_644_800, 176_400), 176_400); // DSD128 ÷32
        assert_eq!(analysis_rate_for(5_644_800, 352_800), 352_800);
        assert_eq!(analysis_rate_for(11_289_600, 176_400), 176_400); // DSD256 ÷64
        assert_eq!(analysis_rate_for(22_579_200, 176_400), 176_400); // DSD512 ÷128
        // 48k-family lands on its own integer sub-multiples.
        assert_eq!(analysis_rate_for(3_072_000, 176_400), 192_000);
        assert_eq!(analysis_rate_for(3_072_000, 352_800), 384_000);
    }

    #[test]
    fn dc_levels() {
        // 4096 bytes ≈ 2048 output samples at ÷16; skip the filter warm-up.
        let n = 4096usize;
        let full = vec![0xFFu8; n];
        let silence = vec![0x69u8; n];
        for (bytes, expect) in [(&full, 1.0f32), (&silence, 0.0f32)] {
            let src = source_from_bytes(&[bytes.clone()], DSD64_RATE, 176_400);
            let out: Vec<f32> = src.collect();
            assert!(out.len() > 200, "only {} samples", out.len());
            for &v in &out[out.len() / 2..] {
                assert!(
                    (v - expect).abs() < 0.02,
                    "expected ≈{expect}, got {v}"
                );
            }
        }
    }

    #[test]
    fn stereo_channels_stay_independent_and_interleaved() {
        let n = 4096usize;
        let src = source_from_bytes(
            &[vec![0xFFu8; n], vec![0x00u8; n]],
            DSD64_RATE,
            176_400,
        );
        assert_eq!(src.channels(), 2);
        let out: Vec<f32> = src.collect();
        let tail = &out[out.len() / 2..];
        for pair in tail.chunks(2) {
            assert!(pair[0] > 0.95, "ch0 should be ≈+1, got {}", pair[0]);
            assert!(pair[1] < -0.95, "ch1 should be ≈−1, got {}", pair[1]);
        }
    }

    /// Goertzel magnitude (normalised amplitude) of `signal` at `freq`.
    fn goertzel(signal: &[f32], freq: f32, rate: f32) -> f32 {
        let w = 2.0 * std::f32::consts::PI * freq / rate;
        let coeff = 2.0 * w.cos();
        let (mut s1, mut s2) = (0.0f32, 0.0f32);
        for &x in signal {
            let s0 = x + coeff * s1 - s2;
            s2 = s1;
            s1 = s0;
        }
        let power = s1 * s1 + s2 * s2 - coeff * s1 * s2;
        2.0 * power.max(0.0).sqrt() / signal.len() as f32
    }

    /// Encode a sine with a 1st-order sigma-delta modulator, decimate, and
    /// verify the tone comes back at the right frequency and level with a
    /// quiet floor elsewhere.
    #[test]
    fn sigma_delta_tone_round_trip() {
        for &target in &[176_400u32, 352_800] {
            let bits_n = DSD64_RATE as usize / 2; // 0.5 s
            let amp = 0.5f64;
            let f = 1000.0f64;
            let mut acc = 0.0f64;
            let mut bytes = vec![0u8; bits_n / 8];
            for (i, byte) in bytes.iter_mut().enumerate() {
                let mut b = 0u8;
                for k in 0..8 {
                    let n = i * 8 + k;
                    let x = amp * (2.0 * std::f64::consts::PI * f * n as f64
                        / DSD64_RATE as f64).sin();
                    let y = if acc + x >= 0.0 { 1.0 } else { -1.0 };
                    acc += x - y;
                    // MSB-first: earliest sample in the MSB.
                    b = (b << 1) | ((y > 0.0) as u8);
                }
                *byte = b;
            }
            let src = source_from_bytes(&[bytes], DSD64_RATE, target);
            let rate = src.out_rate() as f32;
            assert_eq!(rate as u32, target);
            let out: Vec<f32> = src.collect();
            let steady = &out[out.len() / 4..];

            let tone = goertzel(steady, 1000.0, rate);
            assert!(
                (tone - amp as f32).abs() < 0.05,
                "1 kHz tone at {target}: expected ≈0.5, got {tone}"
            );
            // Away from the tone the floor must be far down (the modulator's
            // shaped noise lives well above the audible band).
            for &probe in &[4_000.0f32, 10_000.0, 20_000.0] {
                let leak = goertzel(steady, probe, rate);
                assert!(
                    leak < 0.01,
                    "floor at {probe} Hz ({target}): {leak} too high vs tone {tone}"
                );
            }
        }
    }

    #[test]
    fn seek_lands_in_the_right_region() {
        use std::time::Duration;
        // First half DSD silence (≈0), second half all-ones (≈+1). Seeking to
        // 75% must yield ≈+1 immediately after the filter warm-up.
        let n = 8192usize; // bytes = 65536 samples ≈ 23 ms @ DSD64
        let mut bytes = vec![0x69u8; n];
        for b in &mut bytes[n / 2..] {
            *b = 0xFF;
        }
        let total = Duration::from_secs_f64(n as f64 * 8.0 / DSD64_RATE as f64);
        let mut src = source_from_bytes(&[bytes], DSD64_RATE, 176_400);
        src.seek_to_time(total.mul_f64(0.75)).unwrap();
        let out: Vec<f32> = src.collect();
        assert!(!out.is_empty());
        // Skip the filter's warm-up (≈ one filter length) and check the level.
        for &v in &out[out.len() / 2..] {
            assert!(v > 0.9, "expected ≈+1 after seek, got {v}");
        }
        // Roughly a quarter of the track should remain.
        let expect = n / 2 / 4; // ÷16 → bytes/2 outputs total, quarter remains
        assert!(
            (out.len() as i64 - expect as i64).unsigned_abs() <= 256,
            "expected ≈{expect} samples after 75% seek, got {}",
            out.len()
        );
    }

    #[test]
    fn output_length_tracks_decimation_ratio() {
        let n = 8192usize; // one channel, 65536 bits
        let src = source_from_bytes(&[vec![0x69u8; n]], DSD64_RATE, 176_400);
        let expect = n / 2; // ÷16 = bytes/2
        let got = src.count();
        assert!(
            (got as i64 - expect as i64).unsigned_abs() <= HB_TAPS as u64,
            "expected ≈{expect} samples, got {got}"
        );
    }
}
