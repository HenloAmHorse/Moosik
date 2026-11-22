//! Room correction (1.6.2): an impulse response from REW or any other tool,
//! convolved with what plays. Uniformly partitioned overlap-save, blocks of
//! `BLOCK` samples, so it adds `BLOCK` samples of latency; the response may be
//! up to `MAX_TAPS` long and is resampled to the stream's rate when they differ.

use rustfft::{num_complex::Complex, Fft, FftPlanner};
use std::path::Path;
use std::sync::Arc;

pub const BLOCK: usize = 1024;
pub const MAX_TAPS: usize = 1 << 17;

/// A loaded impulse response: one per channel (one for both), at its own rate.
#[derive(Debug, PartialEq)]
pub struct Fir {
    pub name: String,
    pub rate: u32,
    pub taps: Vec<Vec<f32>>,
}

/// Read an impulse response from a WAV (or any file Moosik decodes).
pub fn load(path: &Path) -> Result<Fir, String> {
    let (cancel, progress) = (std::sync::atomic::AtomicBool::new(false), std::sync::atomic::AtomicU32::new(0));
    let (mut rate, mut ch, mut data) = (0u32, 0usize, Vec::<f32>::new());
    crate::measure::each_block(path, &cancel, &progress, |r, c, s| {
        rate = r;
        ch = c;
        if data.len() < MAX_TAPS * c + c {
            data.extend_from_slice(s);
        }
    })?;
    if ch == 0 || data.is_empty() {
        return Err("no audio in it".into());
    }
    if ch > 2 {
        return Err(format!("{ch} channels; a room correction has one or two"));
    }
    let frames = data.len() / ch;
    if frames > MAX_TAPS {
        return Err(format!("{:.1} s long; at most {:.1} s", frames as f32 / rate as f32, MAX_TAPS as f32 / rate as f32));
    }
    let taps = (0..ch).map(|c| data.iter().skip(c).step_by(ch).copied().collect()).collect();
    let name = path.file_stem().and_then(|s| s.to_str()).unwrap_or("Room correction").to_string();
    Ok(Fir { name, rate, taps })
}

/// One channel's convolution state.
struct Lane {
    /// The response's partitions, in the frequency domain.
    parts: Vec<Vec<Complex<f32>>>,
    /// The spectra of the last inputs, newest first (a ring).
    history: Vec<Vec<Complex<f32>>>,
    head: usize,
    /// The last two input blocks (the overlap-save window).
    window: Vec<f32>,
    input: Vec<f32>,
    output: Vec<f32>,
}

/// The convolver for one stream: per channel, input gathered a block at a
/// time and output a block behind.
pub struct Convolver {
    fft: Arc<dyn Fft<f32>>,
    ifft: Arc<dyn Fft<f32>>,
    lanes: Vec<Lane>,
    pos: usize,
    scratch: Vec<Complex<f32>>,
}

impl Convolver {
    /// For `channels` channels at `rate` Hz. A one-channel response serves
    /// every channel; a two-channel one serves left and right.
    pub fn new(fir: &Fir, rate: u32, channels: usize) -> Convolver {
        let mut planner = FftPlanner::<f32>::new();
        let (fft, ifft) = (planner.plan_fft_forward(2 * BLOCK), planner.plan_fft_inverse(2 * BLOCK));
        let lanes = (0..channels)
            .map(|c| {
                let taps = &fir.taps[c.min(fir.taps.len() - 1)];
                let taps = if fir.rate == rate { taps.clone() } else { crate::chords::resample(taps, fir.rate, rate) };
                let parts = taps
                    .chunks(BLOCK)
                    .map(|chunk| {
                        let mut buf: Vec<Complex<f32>> = chunk.iter().map(|&t| Complex::new(t, 0.0)).collect();
                        buf.resize(2 * BLOCK, Complex::new(0.0, 0.0));
                        fft.process(&mut buf);
                        buf
                    })
                    .collect::<Vec<_>>();
                let n = parts.len().max(1);
                Lane {
                    parts,
                    history: vec![vec![Complex::new(0.0, 0.0); 2 * BLOCK]; n],
                    head: 0,
                    window: vec![0.0; 2 * BLOCK],
                    input: vec![0.0; BLOCK],
                    output: vec![0.0; BLOCK],
                }
            })
            .collect();
        Convolver { fft, ifft, lanes, pos: 0, scratch: vec![Complex::new(0.0, 0.0); 2 * BLOCK] }
    }

    /// One sample of channel `c`; the frame's last channel moves time on.
    /// What comes out is `BLOCK` samples behind what went in.
    pub fn process(&mut self, c: usize, x: f32) -> f32 {
        let lane = &mut self.lanes[c];
        lane.input[self.pos] = x;
        let y = lane.output[self.pos];
        if c + 1 == self.lanes.len() {
            self.pos += 1;
            if self.pos == BLOCK {
                self.pos = 0;
                for lane in &mut self.lanes {
                    Self::block(&*self.fft, &*self.ifft, lane, &mut self.scratch);
                }
            }
        }
        y
    }

    fn block(fft: &dyn Fft<f32>, ifft: &dyn Fft<f32>, lane: &mut Lane, acc: &mut [Complex<f32>]) {
        lane.window.copy_within(BLOCK.., 0);
        lane.window[BLOCK..].copy_from_slice(&lane.input);
        let n = lane.history.len();
        lane.head = (lane.head + n - 1) % n;
        let spec = &mut lane.history[lane.head];
        for (s, &w) in spec.iter_mut().zip(&lane.window) {
            *s = Complex::new(w, 0.0);
        }
        fft.process(spec);
        acc.iter_mut().for_each(|a| *a = Complex::new(0.0, 0.0));
        for (k, part) in lane.parts.iter().enumerate() {
            let h = &lane.history[(lane.head + k) % n];
            for ((a, x), p) in acc.iter_mut().zip(h).zip(part) {
                *a += x * p;
            }
        }
        ifft.process(acc);
        let scale = 1.0 / (2 * BLOCK) as f32;
        for (o, a) in lane.output.iter_mut().zip(&acc[BLOCK..]) {
            *o = a.re * scale;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn run(fir: &Fir, input: &[f32]) -> Vec<f32> {
        let mut cv = Convolver::new(fir, 48_000, 1);
        input.iter().map(|&x| cv.process(0, x)).collect()
    }

    #[test]
    fn a_unit_impulse_passes_audio_through_a_block_late() {
        let fir = Fir { name: "id".into(), rate: 48_000, taps: vec![vec![1.0]] };
        let x: Vec<f32> = (0..5000).map(|i| ((i * 7919) % 101) as f32 / 100.0 - 0.5).collect();
        let y = run(&fir, &x);
        for i in BLOCK..5000 {
            assert!((y[i] - x[i - BLOCK]).abs() < 1e-5, "{i}: {} vs {}", y[i], x[i - BLOCK]);
        }
    }

    #[test]
    fn a_long_response_convolves_exactly() {
        // Three partitions' worth of taps against direct convolution.
        let taps: Vec<f32> = (0..2500).map(|i| ((i * 31) % 17) as f32 / 17.0 - 0.5).map(|v| v * 0.01).collect();
        let fir = Fir { name: "room".into(), rate: 48_000, taps: vec![taps.clone()] };
        let x: Vec<f32> = (0..6000).map(|i| ((i * 13) % 29) as f32 / 29.0 - 0.5).collect();
        let y = run(&fir, &x);
        for n in (BLOCK..6000).step_by(97) {
            let m = n - BLOCK;
            let want: f32 = (0..taps.len()).filter(|&k| k <= m).map(|k| taps[k] * x[m - k]).sum();
            assert!((y[n] - want).abs() < 1e-4, "{n}: {} vs {want}", y[n]);
        }
    }

    #[test]
    fn a_stereo_response_keeps_its_channels_apart() {
        let fir = Fir { name: "lr".into(), rate: 48_000, taps: vec![vec![1.0], vec![0.5]] };
        let mut cv = Convolver::new(&fir, 48_000, 2);
        let mut last = (0.0, 0.0);
        for _ in 0..3 * BLOCK {
            last = (cv.process(0, 0.8), cv.process(1, 0.8));
        }
        assert!((last.0 - 0.8).abs() < 1e-5 && (last.1 - 0.4).abs() < 1e-5, "{last:?}");
    }

    #[test]
    fn a_wav_response_loads_and_runs_in_the_eq_path() {
        // A 16-bit stereo WAV: left a unit impulse, right half of one.
        let path = std::env::temp_dir().join(format!("moosik-ir-{}.wav", std::process::id()));
        let frames: Vec<[i16; 2]> = (0..64).map(|i| if i == 0 { [32767, 16384] } else { [0, 0] }).collect();
        let data: Vec<u8> = frames.iter().flat_map(|f| f.iter().flat_map(|v| v.to_le_bytes())).collect();
        let mut b = Vec::new();
        b.extend(b"RIFF");
        b.extend((36 + data.len() as u32).to_le_bytes());
        b.extend(b"WAVEfmt ");
        b.extend(16u32.to_le_bytes());
        b.extend(1u16.to_le_bytes());
        b.extend(2u16.to_le_bytes());
        b.extend(48_000u32.to_le_bytes());
        b.extend((48_000u32 * 4).to_le_bytes());
        b.extend(4u16.to_le_bytes());
        b.extend(16u16.to_le_bytes());
        b.extend(b"data");
        b.extend((data.len() as u32).to_le_bytes());
        b.extend(data);
        std::fs::write(&path, b).unwrap();
        let fir = load(&path).unwrap();
        std::fs::remove_file(&path).unwrap();
        assert_eq!((fir.rate, fir.taps.len(), fir.taps[0].len()), (48_000, 2, 64));
        let mut st = super::super::eq::EqState::new();
        st.sample_rate = 48_000;
        st.fir = Some(Arc::new(fir));
        let eq = Arc::new(std::sync::Mutex::new(st));
        let src = rodio::buffer::SamplesBuffer::new(2, 48_000, vec![0.5f32; 2 * 4 * BLOCK]);
        let out: Vec<f32> = super::super::eq::EqSource::new(src, eq).collect();
        let (l, r) = (out[2 * 3 * BLOCK], out[2 * 3 * BLOCK + 1]);
        assert!((l - 0.5).abs() < 1e-3 && (r - 0.25).abs() < 1e-3, "{l} {r}");
        assert_eq!(out[0], 0.0, "a block late");
    }
}
