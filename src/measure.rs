//! Whole-file measurements (1.6.2): loudness to ITU-R BS.1770-4 and EBU Tech
//! 3342, true peak, clipping and inter-sample overs, crest factor, DC offset,
//! the bit depth a file actually uses, and FLAC's own MD5 check. One decode
//! at the file's own rate and precision, on whatever thread calls it.

use std::path::Path;
use std::sync::atomic::{AtomicBool, AtomicU32, Ordering};
use symphonia::core::audio::{AudioBufferRef, SampleBuffer};
use symphonia::core::codecs::{DecoderOptions, CODEC_TYPE_FLAC};
use symphonia::core::formats::FormatOptions;
use symphonia::core::io::MediaSourceStream;
use symphonia::core::meta::MetadataOptions;
use symphonia::core::probe::Hint;

/// What a measurement found. Levels in dB; `None` where it does not apply.
#[derive(Clone, Debug, PartialEq)]
pub struct Report {
    pub rate: u32,
    pub channels: usize,
    pub seconds: f64,
    /// Integrated loudness, LUFS (BS.1770-4, gated).
    pub integrated: Option<f64>,
    /// Loudness range, LU (EBU Tech 3342).
    pub lra: Option<f64>,
    /// Loudest short-term (3 s) loudness, LUFS.
    pub max_short_term: Option<f64>,
    /// Loudest momentary (400 ms) loudness, LUFS.
    pub max_momentary: Option<f64>,
    pub sample_peak: f64,
    /// True peak, dBTP (4× oversampled up to 96 kHz, 2× to 192 kHz).
    pub true_peak: f64,
    /// Peak to loudness ratio: true peak − integrated.
    pub plr: Option<f64>,
    /// Peak to short-term loudness ratio at its smallest: true peak − max short-term.
    pub psr: Option<f64>,
    /// Sample peak − RMS of all channels.
    pub crest: f64,
    /// The DR meter's figure, dB, unrounded (shown as "DR 8"); see [`Dr`].
    pub dr: Option<f64>,
    /// Each channel's mean, as a fraction of full scale.
    pub dc: Vec<f64>,
    /// Runs of three or more samples at full scale, and where (seconds, at most 500).
    pub clips: u32,
    pub clip_at: Vec<f32>,
    /// Places the reconstructed wave goes above 0 dBFS between samples.
    pub overs: u32,
    pub over_at: Vec<f32>,
    /// Bits the container says it has, and bits the samples actually use
    /// (integer sources; a 24-bit file of 16-bit audio uses 16).
    pub declared_bits: Option<u32>,
    pub used_bits: Option<u32>,
    /// Left/right correlation over the whole file (stereo).
    pub correlation: Option<f64>,
    /// FLAC: whether the decoded audio matches the MD5 in its header
    /// (None: not FLAC, or no MD5 stored).
    pub md5_ok: Option<bool>,
}

/// Measure `path`. `progress` gets per-mille done; `cancel` stops it.
pub fn measure(path: &Path, cancel: &AtomicBool, progress: &AtomicU32) -> Result<Report, String> {
    let file = std::fs::File::open(path).map_err(|e| format!("{}: {e}", path.display()))?;
    let mut hint = Hint::new();
    if let Some(ext) = path.extension().and_then(|s| s.to_str()) {
        hint.with_extension(ext);
    }
    let mut probed = symphonia::default::get_probe()
        .format(&hint, MediaSourceStream::new(Box::new(file), Default::default()), &FormatOptions::default(), &MetadataOptions::default())
        .map_err(|e| format!("not a readable audio file: {e}"))?;
    let track = crate::audio_probe::audio_track(probed.format.as_ref()).ok_or("no audio track")?;
    let (track_id, params) = (track.id, track.codec_params.clone());
    let flac = params.codec == CODEC_TYPE_FLAC;
    let has_md5 = flac && params.verification_check.is_some();
    let mut decoder = symphonia::default::get_codecs()
        .make(&params, &DecoderOptions { verify: has_md5 })
        .map_err(|e| format!("cannot decode this file: {e}"))?;
    let total = params.n_frames.unwrap_or(0);
    let mut m: Option<Meter> = None;
    let mut bits_or: u32 = 0;
    let mut integer: Option<u32> = None;
    let mut buf: Option<SampleBuffer<f32>> = None;
    let mut frames = 0u64;
    loop {
        if cancel.load(Ordering::Relaxed) {
            return Err("cancelled".into());
        }
        let packet = match probed.format.next_packet() {
            Ok(p) => p,
            Err(symphonia::core::errors::Error::IoError(e)) if e.kind() == std::io::ErrorKind::UnexpectedEof => break,
            Err(e) => return Err(format!("reading failed: {e}")),
        };
        if packet.track_id() != track_id {
            continue;
        }
        let decoded = match decoder.decode(&packet) {
            Ok(d) => d,
            Err(symphonia::core::errors::Error::DecodeError(_)) => continue,
            Err(e) => return Err(format!("decoding failed: {e}")),
        };
        // The bits the samples use, from the integers as decoded.
        match &decoded {
            AudioBufferRef::S16(b) => { integer = Some(16); or_bits(b.planes().planes(), |v: &i16| *v as i32 as u32, &mut bits_or); }
            AudioBufferRef::S24(b) => { integer = Some(24); or_bits(b.planes().planes(), |v: &symphonia::core::sample::i24| v.0 as u32, &mut bits_or); }
            AudioBufferRef::S32(b) => { integer = Some(32); or_bits(b.planes().planes(), |v: &i32| *v as u32, &mut bits_or); }
            _ => {}
        }
        let spec = *decoded.spec();
        let n = decoded.frames();
        if buf.as_ref().is_none_or(|b| b.capacity() < n * spec.channels.count()) {
            buf = Some(SampleBuffer::new(n as u64, spec));
        }
        let b = buf.as_mut().expect("just made");
        b.copy_interleaved_ref(decoded);
        let meter = m.get_or_insert_with(|| Meter::new(spec.rate, spec.channels.count()));
        meter.push(b.samples());
        frames += n as u64;
        if total > 0 {
            progress.store((frames * 1000 / total).min(1000) as u32, Ordering::Relaxed);
        }
    }
    let md5_ok = if has_md5 { decoder.finalize().verify_ok } else { None };
    let meter = m.ok_or("no audio in the file")?;
    let declared = params.bits_per_sample.or(integer);
    let used = integer.map(|w| if bits_or == 0 { 0 } else { w.saturating_sub(bits_or.trailing_zeros().min(w)) });
    Ok(meter.finish(declared, used, md5_ok))
}

/// Every block of a file's audio as interleaved f32, with its rate and
/// channel count. `progress` gets per-mille done; `cancel` stops it.
pub(crate) fn each_block(path: &Path, cancel: &AtomicBool, progress: &AtomicU32, mut f: impl FnMut(u32, usize, &[f32])) -> Result<(), String> {
    let file = std::fs::File::open(path).map_err(|e| format!("{}: {e}", path.display()))?;
    let mut hint = Hint::new();
    if let Some(ext) = path.extension().and_then(|s| s.to_str()) {
        hint.with_extension(ext);
    }
    let mut probed = symphonia::default::get_probe()
        .format(&hint, MediaSourceStream::new(Box::new(file), Default::default()), &FormatOptions::default(), &MetadataOptions::default())
        .map_err(|e| format!("not a readable audio file: {e}"))?;
    let track = crate::audio_probe::audio_track(probed.format.as_ref()).ok_or("no audio track")?;
    let (track_id, params) = (track.id, track.codec_params.clone());
    let mut decoder = symphonia::default::get_codecs()
        .make(&params, &DecoderOptions::default())
        .map_err(|e| format!("cannot decode this file: {e}"))?;
    let total = params.n_frames.unwrap_or(0);
    let mut buf: Option<SampleBuffer<f32>> = None;
    let mut frames = 0u64;
    loop {
        if cancel.load(Ordering::Relaxed) {
            return Err("cancelled".into());
        }
        let packet = match probed.format.next_packet() {
            Ok(p) => p,
            Err(symphonia::core::errors::Error::IoError(e)) if e.kind() == std::io::ErrorKind::UnexpectedEof => break,
            Err(e) => return Err(format!("reading failed: {e}")),
        };
        if packet.track_id() != track_id {
            continue;
        }
        let decoded = match decoder.decode(&packet) {
            Ok(d) => d,
            Err(symphonia::core::errors::Error::DecodeError(_)) => continue,
            Err(e) => return Err(format!("decoding failed: {e}")),
        };
        let spec = *decoded.spec();
        let n = decoded.frames();
        if buf.as_ref().is_none_or(|b| b.capacity() < n * spec.channels.count()) {
            buf = Some(SampleBuffer::new(n as u64, spec));
        }
        let b = buf.as_mut().expect("just made");
        b.copy_interleaved_ref(decoded);
        f(spec.rate, spec.channels.count(), b.samples());
        frames += n as u64;
        if total > 0 {
            progress.store((frames * 1000 / total).min(1000) as u32, Ordering::Relaxed);
        }
    }
    Ok(())
}

/// The integrated loudness of interleaved samples already in memory.
pub fn loudness_of(samples: &[f32], rate: u32, channels: usize) -> Option<f64> {
    let mut m = IntegratedMeter::new(rate, channels.max(1));
    m.push(samples);
    m.finish()
}

/// Third-octave band centres, 20 Hz to 20 kHz.
pub fn third_octaves() -> Vec<f32> {
    (-17..=13).map(|k| 1000.0 * 2f32.powf(k as f32 / 3.0)).collect()
}

/// A file's long-term average spectrum (third-octave bands, dB, the mix of
/// all channels) and its integrated loudness.
#[derive(Clone, Debug, PartialEq)]
pub struct Ltas {
    pub bands: Vec<(f32, f32)>,
    pub integrated: Option<f64>,
}

pub fn ltas(path: &Path, cancel: &AtomicBool, progress: &AtomicU32) -> Result<Ltas, String> {
    use rustfft::{num_complex::Complex, FftPlanner};
    const N: usize = 8192;
    let win: Vec<f32> = (0..N).map(|i| 0.5 - 0.5 * (std::f32::consts::TAU * i as f32 / N as f32).cos()).collect();
    let fft = FftPlanner::<f32>::new().plan_fft_forward(N);
    let mut power = vec![0f64; N / 2 + 1];
    let (mut frames_done, mut rate) = (0usize, 0u32);
    let mut mono: Vec<f32> = Vec::new();
    let mut meter: Option<IntegratedMeter> = None;
    let mut buf = vec![Complex::new(0f32, 0f32); N];
    each_block(path, cancel, progress, |r, ch, samples| {
        rate = r;
        meter.get_or_insert_with(|| IntegratedMeter::new(r, ch)).push(samples);
        mono.extend(samples.chunks_exact(ch).map(|f| f.iter().sum::<f32>() / ch as f32));
        // Hann windows, half overlapping, as they fill.
        while mono.len() >= N {
            for (b, (x, w)) in buf.iter_mut().zip(mono.iter().zip(&win)) {
                *b = Complex::new(x * w, 0.0);
            }
            fft.process(&mut buf);
            for (p, c) in power.iter_mut().zip(&buf) {
                *p += c.norm_sqr() as f64;
            }
            frames_done += 1;
            mono.drain(..N / 2);
        }
    })?;
    if frames_done == 0 || rate == 0 {
        return Err("too short to compare".into());
    }
    let bin = rate as f32 / N as f32;
    let bands = third_octaves()
        .into_iter()
        .filter(|&c| c * 2f32.powf(1.0 / 6.0) < rate as f32 / 2.0)
        .map(|c| {
            let (lo, hi) = (c / 2f32.powf(1.0 / 6.0), c * 2f32.powf(1.0 / 6.0));
            let (a, b) = (((lo / bin).ceil() as usize).max(1), ((hi / bin).floor() as usize).min(N / 2));
            let p: f64 = if b >= a { power[a..=b].iter().sum::<f64>() } else { power[a.min(N / 2)] };
            (c, (10.0 * (p / frames_done as f64 + 1e-30).log10()) as f32)
        })
        .collect();
    Ok(Ltas { bands, integrated: meter.and_then(IntegratedMeter::finish) })
}

/// Two files side by side: how much louder B is, and B's spectrum against
/// A's once their loudness is matched (dB per third-octave band).
#[derive(Clone, Debug, PartialEq)]
pub struct Comparison {
    pub a: std::path::PathBuf,
    pub b: std::path::PathBuf,
    pub louder_db: Option<f64>,
    pub diff: Vec<(f32, f32)>,
}

pub fn compare(a: &Path, b: &Path, cancel: &AtomicBool, progress: &AtomicU32) -> Result<Comparison, String> {
    let half = AtomicU32::new(0);
    let la = ltas(a, cancel, &half)?;
    progress.store(500, Ordering::Relaxed);
    let lb = ltas(b, cancel, &half)?;
    progress.store(1000, Ordering::Relaxed);
    let louder_db = la.integrated.zip(lb.integrated).map(|(x, y)| y - x);
    let shift = louder_db.unwrap_or(0.0) as f32;
    let diff = la.bands.iter().zip(&lb.bands).map(|(&(f, x), &(_, y))| (f, y - x - shift)).collect();
    Ok(Comparison { a: a.to_path_buf(), b: b.to_path_buf(), louder_db, diff })
}

/// A measurement (or comparison) running on its own thread, for `path`.
pub struct Job<T = Report> {
    pub path: std::path::PathBuf,
    pub cancel: std::sync::Arc<AtomicBool>,
    pub progress: std::sync::Arc<AtomicU32>,
    rx: std::sync::mpsc::Receiver<Result<T, String>>,
}

impl Job<Report> {
    pub fn start(path: std::path::PathBuf) -> Job<Report> {
        let p = path.clone();
        Job::spawn(path, move |c, g| measure(&p, c, g))
    }
}

impl Job<Comparison> {
    /// Compare `a` (the track) with `b` (another master of it).
    pub fn compare(a: std::path::PathBuf, b: std::path::PathBuf) -> Job<Comparison> {
        let a2 = a.clone();
        Job::spawn(a, move |c, g| compare(&a2, &b, c, g))
    }
}

impl Job<(crate::abx::Clip, crate::abx::Clip)> {
    /// Both files into memory for a blind test.
    pub fn abx(a: std::path::PathBuf, b: std::path::PathBuf) -> Self {
        let a2 = a.clone();
        Job::spawn(a, move |c, g| {
            let ca = crate::abx::load(&a2, c, g)?;
            let cb = crate::abx::load(&b, c, g)?;
            Ok((ca, cb))
        })
    }
}

impl<T: Send + 'static> Job<T> {
    fn spawn(path: std::path::PathBuf, work: impl FnOnce(&AtomicBool, &AtomicU32) -> Result<T, String> + Send + 'static) -> Job<T> {
        let cancel = std::sync::Arc::new(AtomicBool::new(false));
        let progress = std::sync::Arc::new(AtomicU32::new(0));
        let (tx, rx) = std::sync::mpsc::channel();
        let (c, g) = (cancel.clone(), progress.clone());
        let _ = std::thread::Builder::new().name("measure".into()).spawn(move || {
            let _ = tx.send(work(&c, &g));
        });
        Job { path, cancel, progress, rx }
    }

    /// The outcome, once there is one.
    pub fn poll(&self) -> Option<Result<T, String>> {
        match self.rx.try_recv() {
            Ok(r) => Some(r),
            Err(std::sync::mpsc::TryRecvError::Empty) => None,
            Err(_) => Some(Err("the measurement stopped".into())),
        }
    }
}

/// The Info window's measurement: one running, the last one done.
#[derive(Default)]
pub struct State {
    pub job: Option<Job>,
    pub done: Option<(std::path::PathBuf, Result<Report, String>)>,
    pub compare_job: Option<Job<Comparison>>,
    pub compared: Option<(std::path::PathBuf, Result<Comparison, String>)>,
    /// A blind test being loaded, and one under way.
    pub abx_loading: Option<Job<(crate::abx::Clip, crate::abx::Clip)>>,
    pub abx: Option<crate::abx::Session>,
    pub abx_error: Option<String>,
}

/// How much a finding matters.
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum Level {
    Good,
    Note,
    Warn,
}

/// What a measurement says about the file itself, in words. `lossless`:
/// the codec is lossless; `cutoff`: the spectrum's sharp ceiling (Hz), if
/// the spectrum found a steep one.
pub fn verdicts(r: &Report, lossless: Option<bool>, cutoff: Option<f32>) -> Vec<(Level, String)> {
    let mut out = Vec::new();
    match r.md5_ok {
        Some(true) => out.push((Level::Good, "Matches the MD5 in its header: decoded exactly as encoded".into())),
        Some(false) => out.push((Level::Warn, "Does not match the MD5 in its header: the file is damaged".into())),
        None => {}
    }
    if let (Some(d), Some(u)) = (r.declared_bits, r.used_bits)
        && u < d
    {
        out.push(if u == 0 {
            (Level::Note, "Silent: every sample is zero".into())
        } else {
            (Level::Warn, format!("A {d}-bit file, but the audio uses only {u} bits: padded, not true {d}-bit"))
        });
    }
    if lossless == Some(true)
        && let Some(hz) = cutoff
        && hz < 20_500.0
        && hz < 0.45 * r.rate as f32
    {
        out.push((Level::Warn, format!("Lossless, but cut off sharply at {:.1} kHz as MP3 and AAC encoders do: probably made from a lossy file", hz / 1000.0)));
    }
    if r.clips > 0 {
        out.push((Level::Note, format!("{} clipped run{} (three or more samples at full scale)", r.clips, if r.clips == 1 { "" } else { "s" })));
    }
    if r.overs > 0 {
        out.push((Level::Note, format!("{} inter-sample over{}: the wave goes above full scale between samples (true peak {:+.1} dBTP)", r.overs, if r.overs == 1 { "" } else { "s" }, r.true_peak)));
    }
    if let Some(c) = r.correlation
        && c < -0.3
    {
        out.push((Level::Warn, format!("Left and right are mostly opposite (correlation {c:+.2}): one channel may be inverted")));
    }
    if r.dc.iter().any(|d| d.abs() > 0.01) {
        out.push((Level::Note, "A DC offset above 1 % of full scale".into()));
    }
    out
}

fn or_bits<T>(planes: &[&[T]], f: impl Fn(&T) -> u32, acc: &mut u32) {
    for p in planes {
        for v in *p {
            *acc |= f(v);
        }
    }
}

/// A biquad, direct form I, in f64.
#[derive(Clone, Copy)]
struct Biquad {
    b: [f64; 3],
    a: [f64; 3],
    x: [f64; 2],
    y: [f64; 2],
}

impl Biquad {
    fn run(&mut self, x: f64) -> f64 {
        let y = self.b[0] * x + self.b[1] * self.x[0] + self.b[2] * self.x[1] - self.a[1] * self.y[0] - self.a[2] * self.y[1];
        self.x = [x, self.x[0]];
        self.y = [y, self.y[0]];
        y
    }
}

/// BS.1770-4's K-weighting at any rate (the two stages as libebur128 designs them).
fn k_weighting(rate: u32) -> [Biquad; 2] {
    let fs = rate as f64;
    let (f0, g, q) = (1681.974450955533, 3.999843853973347, 0.7071752369554196);
    let k = (std::f64::consts::PI * f0 / fs).tan();
    let vh = 10f64.powf(g / 20.0);
    let vb = vh.powf(0.4996667741545416);
    let a0 = 1.0 + k / q + k * k;
    let shelf = Biquad {
        b: [(vh + vb * k / q + k * k) / a0, 2.0 * (k * k - vh) / a0, (vh - vb * k / q + k * k) / a0],
        a: [1.0, 2.0 * (k * k - 1.0) / a0, (1.0 - k / q + k * k) / a0],
        x: [0.0; 2],
        y: [0.0; 2],
    };
    let (f0, q) = (38.13547087602444, 0.5003270373238773);
    let k = (std::f64::consts::PI * f0 / fs).tan();
    let d = 1.0 + k / q + k * k;
    let hp = Biquad { b: [1.0, -2.0, 1.0], a: [1.0, 2.0 * (k * k - 1.0) / d, (1.0 - k / q + k * k) / d], x: [0.0; 2], y: [0.0; 2] };
    [shelf, hp]
}

/// Channel weights: 1 for front channels, 1.41 for surrounds (5.1 order:
/// L R C LFE Ls Rs), 0 for LFE.
fn weight(channel: usize, channels: usize) -> f64 {
    match (channels, channel) {
        (6, 3) => 0.0,
        (6, 4) | (6, 5) | (5, 3) | (5, 4) => 1.41,
        _ => 1.0,
    }
}

/// Oversampling factor for true peak, and its polyphase filter: a
/// Kaiser-windowed sinc, 12 taps a phase.
fn true_peak_filter(rate: u32) -> (usize, Vec<Vec<f64>>) {
    let os = if rate <= 96_000 { 4 } else if rate <= 192_000 { 2 } else { 1 };
    if os == 1 {
        return (1, vec![vec![1.0]]);
    }
    const TAPS: usize = 12;
    let n = TAPS * os;
    let beta = 7.0;
    let i0 = |x: f64| {
        let (mut s, mut t) = (1.0, 1.0);
        for k in 1..30 {
            t *= (x / 2.0) / k as f64;
            s += t * t;
        }
        s
    };
    let h: Vec<f64> = (0..n)
        .map(|i| {
            let t = i as f64 - (n - 1) as f64 / 2.0;
            let x = t / os as f64;
            let sinc = if x.abs() < 1e-12 { 1.0 } else { (std::f64::consts::PI * x).sin() / (std::f64::consts::PI * x) };
            let r = 2.0 * i as f64 / (n - 1) as f64 - 1.0;
            sinc * i0(beta * (1.0 - r * r).max(0.0).sqrt()) / i0(beta)
        })
        .collect();
    // Phase p takes every os-th tap from p; each phase sums to about 1.
    let phases = (0..os).map(|p| (0..TAPS).map(|j| h[p + j * os]).collect()).collect();
    (os, phases)
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
#[allow(unsafe_op_in_unsafe_fn)]
unsafe fn eval_tp_os4_avx2(buf: &[f64], count: usize, phases_t: &[[f64; 4]; 12], tp: &mut f64, over: &mut [bool]) {
    use std::arch::x86_64::*;
    let abs_mask = _mm256_castsi256_pd(_mm256_set1_epi64x(0x7FFF_FFFF_FFFF_FFFFu64 as i64));
    let mut vmax = _mm256_set1_pd(*tp);
    let vover = _mm256_set1_pd(1.0);
    let bp = buf.as_ptr();
    let pp = phases_t.as_ptr();
    for i in 0..count {
        let s = bp.add(i);
        let mut acc = _mm256_setzero_pd();
        for t in 0..12 {
            let pv = _mm256_loadu_pd((*pp.add(t)).as_ptr());
            let xv = _mm256_broadcast_sd(&*s.add(t));
            acc = _mm256_add_pd(acc, _mm256_mul_pd(pv, xv));
        }
        let va = _mm256_and_pd(acc, abs_mask);
        // `va` as first operand ensures a NaN lane in `va` selects `vmax` (second operand),
        // matching Rust's `f64::max(self.tp, v)`.
        vmax = _mm256_max_pd(va, vmax);
        if _mm256_movemask_pd(_mm256_cmp_pd::<_CMP_GT_OQ>(va, vover)) != 0 {
            *over.get_unchecked_mut(i) = true;
        }
    }
    let mut tmp = [0.0f64; 4];
    _mm256_storeu_pd(tmp.as_mut_ptr(), vmax);
    for v in tmp {
        *tp = (*tp).max(v);
    }
}

fn eval_tp_os4(buf: &[f64], count: usize, phases: &[Vec<f64>], phases_t: &[[f64; 4]; 12], tp: &mut f64, over: &mut [bool]) {
    #[cfg(target_arch = "x86_64")]
    if is_x86_feature_detected!("avx2") {
        // SAFETY: AVX2 support verified above; `buf.len() >= count + 11` and `over.len() >= count`.
        unsafe { eval_tp_os4_avx2(buf, count, phases_t, tp, over); }
        return;
    }
    for i in 0..count {
        for ph in phases {
            let v = ph.iter().zip(&buf[i..i + 12]).map(|(k, x)| k * x).sum::<f64>().abs();
            *tp = (*tp).max(v);
            over[i] |= v > 1.0;
        }
    }
}

/// Only the integrated loudness needed by comparison and ABX loading.
/// The filtering, segment boundaries, summation order and gates match
/// `Meter`; none of the unused peak, clipping or stereo measurements run.
struct IntegratedMeter {
    ch: usize,
    k: Vec<[Biquad; 2]>,
    seg: f64,
    seg_n: usize,
    seg_len: usize,
    segs: Vec<f64>,
}

impl IntegratedMeter {
    fn new(rate: u32, ch: usize) -> Self {
        Self { ch, k: vec![k_weighting(rate); ch], seg: 0.0, seg_n: 0,
               seg_len: (rate as usize / 10).max(1), segs: Vec::new() }
    }

    fn push(&mut self, samples: &[f32]) {
        for frame in samples.chunks_exact(self.ch) {
            let mut e = 0.0;
            for (c, &s) in frame.iter().enumerate() {
                let shelf = self.k[c][0].run(s as f64);
                let y = self.k[c][1].run(shelf);
                e += weight(c, self.ch) * y * y;
            }
            self.seg += e;
            self.seg_n += 1;
            if self.seg_n == self.seg_len {
                self.segs.push(self.seg / self.seg_len as f64);
                self.seg = 0.0;
                self.seg_n = 0;
            }
        }
    }

    fn finish(self) -> Option<f64> {
        // Keep the full meter's rolling 400 ms windows and their arithmetic,
        // including its treatment of a final incomplete 100 ms segment.
        if self.segs.len() < 4 { return None; }
        let mut blocks = Vec::with_capacity(self.segs.len() - 4 + 1);
        let mut acc: f64 = self.segs[..4].iter().sum();
        blocks.push(acc / 4.0);
        for i in 4..self.segs.len() {
            acc += self.segs[i] - self.segs[i - 4];
            blocks.push(acc / 4.0);
        }
        let abs_gate = 10f64.powf((-70.0 + 0.691) / 10.0);
        let gated: Vec<f64> = blocks.iter().copied().filter(|&b| b > abs_gate).collect();
        (!gated.is_empty()).then(|| {
            let mean = gated.iter().sum::<f64>() / gated.len() as f64;
            let rel = mean * 10f64.powf(-10.0 / 10.0);
            let g: Vec<f64> = gated.iter().copied().filter(|&b| b > rel).collect();
            -0.691 + 10.0 * (g.iter().sum::<f64>() / g.len().max(1) as f64).log10()
        })
    }
}

/// The figure of the Dynamic Range meter (the "DR 8" of the TT DR Meter,
/// foobar2000's DR Meter and the DR database). Each channel is cut into 3 s
/// blocks, the last one shorter. A block's RMS is taken ×√2, so that a sine
/// reads its own peak. The channel's figure is the second-highest block peak
/// over the RMS of its loudest fifth of blocks (at least one), in dB. The
/// file's figure is the mean of its channels'; a silent channel has none.
struct Dr {
    len: usize,
    n: usize,
    sq: Vec<f64>,
    pk: Vec<f64>,
    /// Each channel's finished blocks: (RMS ×√2, peak).
    blocks: Vec<Vec<(f64, f64)>>,
}

impl Dr {
    fn new(rate: u32, ch: usize) -> Self {
        Dr { len: 3 * rate as usize, n: 0, sq: vec![0.0; ch], pk: vec![0.0; ch], blocks: vec![Vec::new(); ch] }
    }

    fn push(&mut self, samples: &[f32]) {
        for frame in samples.chunks_exact(self.sq.len()) {
            for (c, &s) in frame.iter().enumerate() {
                let x = s as f64;
                self.sq[c] += x * x;
                self.pk[c] = self.pk[c].max(x.abs());
            }
            self.n += 1;
            if self.n == self.len {
                self.close();
            }
        }
    }

    fn close(&mut self) {
        for c in 0..self.sq.len() {
            self.blocks[c].push(((2.0 * self.sq[c] / self.n as f64).sqrt(), self.pk[c]));
            self.sq[c] = 0.0;
            self.pk[c] = 0.0;
        }
        self.n = 0;
    }

    fn finish(mut self) -> Option<f64> {
        if self.n > 0 {
            self.close();
        }
        let per: Vec<f64> = self.blocks.iter().filter(|b| !b.is_empty()).filter_map(|b| {
            let (mut rms, mut pk): (Vec<f64>, Vec<f64>) = b.iter().copied().unzip();
            rms.sort_by(|a, b| b.total_cmp(a));
            pk.sort_by(|a, b| b.total_cmp(a));
            let top = (b.len() / 5).max(1);
            let r = (rms[..top].iter().map(|r| r * r).sum::<f64>() / top as f64).sqrt();
            let p = pk[1.min(pk.len() - 1)];
            (r > 0.0 && p > 0.0).then(|| 20.0 * (p / r).log10())
        }).collect();
        (!per.is_empty()).then(|| per.iter().sum::<f64>() / per.len() as f64)
    }
}

/// Accumulates everything over interleaved f32 frames.
struct Meter {
    rate: u32,
    ch: usize,
    k: Vec<[Biquad; 2]>,
    /// K-weighted energy of the 100 ms segment in progress, and its length.
    seg: f64,
    seg_n: usize,
    seg_len: usize,
    /// Each finished 100 ms segment's channel-weighted mean square.
    segs: Vec<f64>,
    peak: f64,
    tp: f64,
    os: usize,
    phases: Vec<Vec<f64>>,
    phases_t: [[f64; 4]; 12],
    phase_bound: f64,
    hist: Vec<Vec<f64>>,
    h_max: Vec<f64>,
    scratch_l: Vec<f64>,
    scratch_r: Vec<f64>,
    over_buf: Vec<bool>,
    sum_sq: f64,
    sum: Vec<f64>,
    lr: (f64, f64, f64),
    n: u64,
    clip_run: Vec<u32>,
    clips: u32,
    clip_at: Vec<f32>,
    over_run: bool,
    overs: u32,
    over_at: Vec<f32>,
    dr: Dr,
}

impl Meter {
    fn new(rate: u32, ch: usize) -> Self {
        let (os, phases) = true_peak_filter(rate);
        let taps = phases[0].len();
        // Bounds any phase's output by its window's largest sample. The 1e-9
        // margin covers the rounding of the sums, so a skipped window can never
        // hold a larger true peak or an over.
        let phase_bound = phases.iter().map(|ph| ph.iter().map(|v| v.abs()).sum::<f64>()).fold(0.0f64, f64::max) * (1.0 + 1e-9);
        let mut phases_t = [[0.0f64; 4]; 12];
        if os == 4 && taps == 12 {
            for p in 0..4 {
                for t in 0..12 {
                    phases_t[t][p] = phases[p][t];
                }
            }
        }
        Meter {
            rate,
            ch,
            k: vec![k_weighting(rate); ch],
            seg: 0.0,
            seg_n: 0,
            seg_len: (rate as usize / 10).max(1),
            segs: Vec::new(),
            peak: 0.0,
            tp: 0.0,
            os,
            phases,
            phases_t,
            phase_bound,
            hist: vec![vec![0.0; taps]; ch],
            h_max: vec![0.0; ch],
            scratch_l: vec![0.0; 11 + 256],
            scratch_r: vec![0.0; 11 + 256],
            over_buf: vec![false; 256],
            sum_sq: 0.0,
            sum: vec![0.0; ch],
            lr: (0.0, 0.0, 0.0),
            n: 0,
            clip_run: vec![0; ch],
            clips: 0,
            clip_at: Vec::new(),
            over_run: false,
            overs: 0,
            over_at: Vec::new(),
            dr: Dr::new(rate, ch),
        }
    }

    fn push(&mut self, samples: &[f32]) {
        let ch = self.ch;
        let complete = (samples.len() / ch) * ch;
        if complete == 0 {
            return;
        }
        let samples = &samples[..complete];
        self.dr.push(samples);
        if ch == 2 && self.os == 4 && self.phases[0].len() == 12 {
            self.push_stereo_os4(samples);
            return;
        }
        self.push_generic(samples);
    }

    fn push_generic(&mut self, samples: &[f32]) {
        let ch = self.ch;
        let taps = self.phases[0].len();
        for frame in samples.chunks_exact(ch) {
            let t = self.n as f32 / self.rate as f32;
            let mut e = 0.0;
            let mut frame_over = false;
            for (c, &s) in frame.iter().enumerate() {
                let x = s as f64;
                let a = x.abs();
                self.peak = self.peak.max(a);
                self.sum_sq += x * x;
                self.sum[c] += x;
                // Full scale for three samples running: a clip.
                if a >= 0.9999 {
                    self.clip_run[c] += 1;
                    if self.clip_run[c] == 3 {
                        self.clips += 1;
                        if self.clip_at.len() < 500 {
                            self.clip_at.push(t);
                        }
                    }
                } else {
                    self.clip_run[c] = 0;
                }
                let shelf = self.k[c][0].run(x);
                let y = self.k[c][1].run(shelf);
                e += weight(c, ch) * y * y;
                // True peak: this channel's history through each phase.
                let h = &mut self.hist[c];
                h.copy_within(1..taps, 0);
                h[taps - 1] = x;
                if self.os == 1 {
                    self.tp = self.tp.max(a);
                    frame_over |= a > 1.0;
                } else {
                    for ph in &self.phases {
                        let v: f64 = ph.iter().zip(h.iter()).map(|(k, x)| k * x).sum::<f64>().abs();
                        self.tp = self.tp.max(v);
                        frame_over |= v > 1.0;
                    }
                }
            }
            if frame_over && !self.over_run {
                self.overs += 1;
                if self.over_at.len() < 500 {
                    self.over_at.push(t);
                }
            }
            self.over_run = frame_over;
            if ch >= 2 {
                let (l, r) = (frame[0] as f64, frame[1] as f64);
                self.lr.0 += l * r;
                self.lr.1 += l * l;
                self.lr.2 += r * r;
            }
            self.seg += e;
            self.seg_n += 1;
            if self.seg_n == self.seg_len {
                self.segs.push(self.seg / self.seg_len as f64);
                self.seg = 0.0;
                self.seg_n = 0;
            }
            self.n += 1;
        }
    }

    fn push_stereo_os4(&mut self, samples: &[f32]) {
        let seg_len = self.seg_len;
        for sub in samples.chunks(256 * 2) {
            let nf = sub.len() / 2;
            let mut kl = self.k[0];
            let mut kr = self.k[1];
            let mut sub_max_l = 0.0f64;
            let mut sub_max_r = 0.0f64;
            let mut seg = self.seg;
            let mut seg_n = self.seg_n;
            for (idx, frame) in sub.chunks_exact(2).enumerate() {
                let l = frame[0] as f64;
                let r = frame[1] as f64;
                let al = l.abs();
                let ar = r.abs();
                self.peak = self.peak.max(al).max(ar);
                sub_max_l = sub_max_l.max(al);
                sub_max_r = sub_max_r.max(ar);
                self.sum_sq += l * l;
                self.sum[0] += l;
                self.sum_sq += r * r;
                self.sum[1] += r;
                self.lr.0 += l * r;
                self.lr.1 += l * l;
                self.lr.2 += r * r;
                self.scratch_l[11 + idx] = l;
                self.scratch_r[11 + idx] = r;
                let shl = kl[0].run(l);
                let yl = kl[1].run(shl);
                let shr = kr[0].run(r);
                let yr = kr[1].run(shr);
                seg += yl * yl + yr * yr;
                seg_n += 1;
                if seg_n == seg_len {
                    self.segs.push(seg / seg_len as f64);
                    seg = 0.0;
                    seg_n = 0;
                }
            }
            self.k[0] = kl;
            self.k[1] = kr;
            self.seg = seg;
            self.seg_n = seg_n;

            if sub_max_l >= 0.9999 || sub_max_r >= 0.9999 {
                for idx in 0..nf {
                    let t = (self.n + idx as u64) as f32 / self.rate as f32;
                    let al = self.scratch_l[11 + idx].abs();
                    if al >= 0.9999 {
                        self.clip_run[0] += 1;
                        if self.clip_run[0] == 3 {
                            self.clips += 1;
                            if self.clip_at.len() < 500 {
                                self.clip_at.push(t);
                            }
                        }
                    } else {
                        self.clip_run[0] = 0;
                    }
                    let ar = self.scratch_r[11 + idx].abs();
                    if ar >= 0.9999 {
                        self.clip_run[1] += 1;
                        if self.clip_run[1] == 3 {
                            self.clips += 1;
                            if self.clip_at.len() < 500 {
                                self.clip_at.push(t);
                            }
                        }
                    } else {
                        self.clip_run[1] = 0;
                    }
                }
            } else {
                self.clip_run[0] = 0;
                self.clip_run[1] = 0;
            }

            let skip_bound = self.tp.min(1.0);
            let win_max_l = self.h_max[0].max(sub_max_l);
            let win_max_r = self.h_max[1].max(sub_max_r);
            let run_l = win_max_l * self.phase_bound > skip_bound;
            let run_r = win_max_r * self.phase_bound > skip_bound;
            if run_l || run_r {
                self.over_buf[..nf].fill(false);
                if run_l {
                    self.scratch_l[..11].copy_from_slice(&self.hist[0][1..12]);
                    eval_tp_os4(&self.scratch_l, nf, &self.phases, &self.phases_t, &mut self.tp, &mut self.over_buf[..nf]);
                }
                if run_r {
                    self.scratch_r[..11].copy_from_slice(&self.hist[1][1..12]);
                    eval_tp_os4(&self.scratch_r, nf, &self.phases, &self.phases_t, &mut self.tp, &mut self.over_buf[..nf]);
                }
                for (idx, &frame_over) in self.over_buf[..nf].iter().enumerate() {
                    if frame_over && !self.over_run {
                        self.overs += 1;
                        if self.over_at.len() < 500 {
                            self.over_at.push((self.n + idx as u64) as f32 / self.rate as f32);
                        }
                    }
                    self.over_run = frame_over;
                }
            } else {
                self.over_run = false;
            }

            if nf >= 12 {
                self.hist[0].copy_from_slice(&self.scratch_l[11 + nf - 12..11 + nf]);
                self.hist[1].copy_from_slice(&self.scratch_r[11 + nf - 12..11 + nf]);
            } else {
                if !run_l {
                    self.scratch_l[..11].copy_from_slice(&self.hist[0][1..12]);
                }
                if !run_r {
                    self.scratch_r[..11].copy_from_slice(&self.hist[1][1..12]);
                }
                self.hist[0].copy_from_slice(&self.scratch_l[nf - 1..nf + 11]);
                self.hist[1].copy_from_slice(&self.scratch_r[nf - 1..nf + 11]);
            }
            self.h_max[0] = self.hist[0].iter().fold(0.0f64, |m, v| m.max(v.abs()));
            self.h_max[1] = self.hist[1].iter().fold(0.0f64, |m, v| m.max(v.abs()));
            self.n += nf as u64;
        }
    }

    fn finish(self, declared_bits: Option<u32>, used_bits: Option<u32>, md5_ok: Option<bool>) -> Report {
        let lufs = |ms: f64| -0.691 + 10.0 * ms.log10();
        // Mean squares over windows of `w` segments, every segment.
        let windows = |w: usize| -> Vec<f64> {
            if self.segs.len() < w {
                return Vec::new();
            }
            let mut out = Vec::with_capacity(self.segs.len() - w + 1);
            let mut acc: f64 = self.segs[..w].iter().sum();
            out.push(acc / w as f64);
            for i in w..self.segs.len() {
                acc += self.segs[i] - self.segs[i - w];
                out.push(acc / w as f64);
            }
            out
        };
        let blocks = windows(4); // 400 ms, 75 % overlap
        let short = windows(30); // 3 s
        let abs_gate = 10f64.powf((-70.0 + 0.691) / 10.0);
        let gated: Vec<f64> = blocks.iter().copied().filter(|&b| b > abs_gate).collect();
        let integrated = (!gated.is_empty()).then(|| {
            let mean = gated.iter().sum::<f64>() / gated.len() as f64;
            let rel = mean * 10f64.powf(-10.0 / 10.0);
            let g: Vec<f64> = gated.iter().copied().filter(|&b| b > rel).collect();
            lufs(g.iter().sum::<f64>() / g.len().max(1) as f64)
        });
        let lra = {
            let st: Vec<f64> = short.iter().copied().filter(|&b| b > abs_gate).collect();
            (!st.is_empty()).then(|| {
                let mean = st.iter().sum::<f64>() / st.len() as f64;
                let rel = mean * 10f64.powf(-20.0 / 10.0);
                let mut l: Vec<f64> = st.iter().copied().filter(|&b| b > rel).map(lufs).collect();
                l.sort_by(|a, b| a.total_cmp(b));
                let at = |p: f64| l[((l.len() - 1) as f64 * p).round() as usize];
                at(0.95) - at(0.10)
            })
        };
        let max_of = |v: &[f64]| v.iter().copied().fold(None, |m: Option<f64>, x| Some(m.map_or(x, |m| m.max(x)))).filter(|&x| x > 0.0).map(lufs);
        let max_short_term = max_of(&short);
        let max_momentary = max_of(&blocks);
        let db = |x: f64| if x > 0.0 { 20.0 * x.log10() } else { -144.0 };
        // The phases fall between the samples, so the samples count as well:
        // a true peak is never below the sample peak.
        let true_peak = db(self.tp.max(self.peak));
        let n = self.n.max(1) as f64;
        let rms = (self.sum_sq / (n * self.ch as f64)).sqrt();
        let correlation = (self.ch >= 2 && self.lr.1 * self.lr.2 > 0.0).then(|| self.lr.0 / (self.lr.1 * self.lr.2).sqrt());
        Report {
            rate: self.rate,
            channels: self.ch,
            seconds: self.n as f64 / self.rate as f64,
            integrated,
            lra,
            max_short_term,
            max_momentary,
            sample_peak: db(self.peak),
            true_peak,
            plr: integrated.map(|i| true_peak - i),
            psr: max_short_term.map(|s| true_peak - s),
            crest: db(self.peak) - db(rms),
            dr: self.dr.finish(),
            dc: self.sum.iter().map(|s| s / n).collect(),
            clips: self.clips,
            clip_at: self.clip_at,
            overs: self.overs,
            over_at: self.over_at,
            declared_bits,
            used_bits,
            correlation,
            md5_ok,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn sine(rate: u32, f: f64, amp: f64, secs: f64, ch: usize) -> Vec<f32> {
        let n = (rate as f64 * secs) as usize;
        (0..n).flat_map(|i| {
            let v = (amp * (2.0 * std::f64::consts::PI * f * i as f64 / rate as f64).sin()) as f32;
            std::iter::repeat_n(v, ch)
        }).collect()
    }

    fn report(samples: &[f32], rate: u32, ch: usize) -> Report {
        let mut m = Meter::new(rate, ch);
        m.push(samples);
        m.finish(None, None, None)
    }

    fn lab_signal(rate: u32, ch: usize, frames: usize) -> Vec<f32> {
        let mut seed = 0x162c_771du32;
        let levels = [0.0f32, 3e-7, 0.02, 0.6, 0.001, 0.0, 0.15];
        let mut out = Vec::with_capacity(frames * ch);
        for i in 0..frames {
            let gain = levels[(i / (rate as usize / 10).max(1)) % levels.len()];
            for c in 0..ch {
                seed ^= seed << 13;
                seed ^= seed >> 17;
                seed ^= seed << 5;
                let noise = (seed >> 8) as f32 / (1u32 << 24) as f32 - 0.5;
                out.push(noise * gain * if c == 3 { 2.0 } else { 1.0 });
            }
        }
        out
    }

    /// Blocks of a known shape: a background of `bg` alternating in sign,
    /// with one sample of each spike at the start of its block.
    fn blocks_of(rate: u32, bg: f32, spikes: &[(usize, f32)], blocks: usize) -> Vec<f32> {
        let len = 3 * rate as usize;
        let mut s: Vec<f32> = (0..blocks * len).map(|i| if i % 2 == 0 { bg } else { -bg }).collect();
        for &(b, v) in spikes {
            s[b * len] = v;
        }
        s
    }

    #[test]
    fn dr_takes_the_second_peak_over_the_loudest_fifth() {
        let rate = 8_000;
        let len = (3 * rate) as f64;
        // Ten blocks of a quiet background, spikes in two of them. The second
        // peak is the smaller spike, and the loudest fifth is those two blocks.
        let (bg, hi, lo) = (0.1f32, 0.9f32, 0.6f32);
        let s = blocks_of(rate, bg, &[(3, hi), (7, lo)], 10);
        let ms = |spike: f32| 2.0 * ((len - 1.0) * (bg as f64).powi(2) + (spike as f64).powi(2)) / len;
        let rms = ((ms(hi) + ms(lo)) / 2.0).sqrt();
        let want = 20.0 * (lo as f64 / rms).log10();
        let got = report(&s, rate, 1).dr.unwrap();
        assert!((got - want).abs() < 1e-9, "{got} {want}");
        assert!((want - 12.54).abs() < 0.01, "{want}");
        // A full-scale sine reads its own peak: DR 0. In two channels, the same.
        let got = report(&sine(rate, 1_000.0, 1.0, 30.0, 2), rate, 2).dr.unwrap();
        assert!(got.abs() < 1e-6, "{got}");
    }

    #[test]
    fn dr_averages_the_channels_and_keeps_short_and_silent_ones_apart() {
        let rate = 8_000;
        // Left: the spiked blocks; right: a sine at half scale (DR 0).
        let left = blocks_of(rate, 0.1, &[(3, 0.9), (7, 0.6)], 10);
        let right = sine(rate, 1_000.0, 0.5, 30.0, 1);
        let both: Vec<f32> = left.iter().zip(&right).flat_map(|(&l, &r)| [l, r]).collect();
        let l = report(&left, rate, 1).dr.unwrap();
        let got = report(&both, rate, 2).dr.unwrap();
        assert!((got - l / 2.0).abs() < 1e-6, "{got} {l}");
        // A silent channel has no figure and is left out of the mean.
        let quiet: Vec<f32> = left.iter().flat_map(|&l| [l, 0.0]).collect();
        assert!((report(&quiet, rate, 2).dr.unwrap() - l).abs() < 1e-12);
        assert_eq!(report(&vec![0.0; 2 * rate as usize], rate, 2).dr, None);
        // Shorter than a block: one block, its own peak (DR of a sine, 0).
        let short = sine(rate, 1_000.0, 0.5, 1.0, 1);
        assert!(report(&short, rate, 1).dr.unwrap().abs() < 1e-6);
        // Chunked input gives the same figure, to the bit.
        let mut m = Meter::new(rate, 2);
        for c in both.chunks(2 * 999) {
            m.push(c);
        }
        assert_eq!(m.finish(None, None, None).dr.map(f64::to_bits), Some(got.to_bits()));
    }

    /// By hand, against a reference: MOOSIK_DR_LIST=<a file, one audio path a
    /// line> cargo test --release dr_probe -- --ignored --nocapture
    #[test]
    #[ignore]
    fn dr_probe() {
        let list = std::fs::read_to_string(std::env::var("MOOSIK_DR_LIST").expect("set MOOSIK_DR_LIST")).unwrap();
        for f in list.lines().map(str::trim).filter(|l| !l.is_empty()) {
            let r = measure(Path::new(f), &AtomicBool::new(false), &AtomicU32::new(0)).unwrap();
            println!("DR\t{:.6}\t{}\t{}\t{f}", r.dr.unwrap_or(f64::NAN), r.rate, r.channels);
        }
    }

    #[test]
    fn integrated_only_matches_the_full_meter_bit_for_bit() {
        for (rate, ch) in [(8_000, 1), (44_100, 2), (48_000, 6), (96_000, 2), (192_000, 1), (384_000, 2)] {
            let frames = rate as usize * 2 + 17;
            let s = lab_signal(rate, ch, frames);
            let segment = (rate as usize / 10).max(1);
            for n in [0, 1, 4 * segment - 1, 4 * segment, frames] {
                let s = &s[..n * ch];
                let expected = report(s, rate, ch).integrated.map(f64::to_bits);
                assert_eq!(loudness_of(s, rate, ch).map(f64::to_bits), expected, "{rate} Hz, {ch} channels, {n} frames");
                let (mut full, mut only) = (Meter::new(rate, ch), IntegratedMeter::new(rate, ch));
                for block in s.chunks(257 * ch) {
                    full.push(block);
                    only.push(block);
                }
                assert_eq!(only.finish().map(f64::to_bits), full.finish(None, None, None).integrated.map(f64::to_bits), "chunked: {rate} Hz, {ch} channels, {n} frames");
            }
        }
        // Match the public helper's channel fallback and ignored partial frame.
        assert_eq!(loudness_of(&[0.0, -0.0, 0.5], 48_000, 0), report(&[0.0, -0.0, 0.5], 48_000, 1).integrated);
        let s = lab_signal(8_000, 2, 4_123);
        assert_eq!(loudness_of(&s[..s.len() - 1], 8_000, 2).map(f64::to_bits), report(&s[..s.len() - 1], 8_000, 2).integrated.map(f64::to_bits));
    }

    #[test]
    fn integrated_only_preserves_nonfinite_and_silent_input_behavior() {
        for value in [0.0, -0.0, f32::from_bits(1), f32::from_bits(0x7fc0_1234), f32::INFINITY, f32::NEG_INFINITY] {
            let mut s = lab_signal(8_000, 2, 8_013);
            s[2_000] = value;
            assert_eq!(loudness_of(&s, 8_000, 2).map(f64::to_bits), report(&s, 8_000, 2).integrated.map(f64::to_bits), "sample bits {:08x}", value.to_bits());
        }
        let silence = vec![0.0; 8_000];
        assert_eq!(loudness_of(&silence, 8_000, 1), None);
    }

    /// Same-binary CPU timing of the old full-meter path and the public
    /// integrated-only path. Synthetic memory only; no decoder, device or I/O.
    #[test]
    #[ignore = "algorithm lab: in-memory CPU timing; run alone in release"]
    fn algorithm_lab_integrated_loudness_probe() {
        use std::hint::black_box;
        let (rate, ch) = (48_000, 2);
        let s = lab_signal(rate, ch, rate as usize * 30);
        let expected = report(&s, rate, ch).integrated.map(f64::to_bits);
        let mut timings = [Vec::new(), Vec::new()];
        for round in 0..5 {
            for which in if round % 2 == 0 { [0, 1] } else { [1, 0] } {
                let start = std::time::Instant::now();
                let result = if which == 0 { report(black_box(&s), rate, ch).integrated } else { loudness_of(black_box(&s), rate, ch) };
                timings[which].push(start.elapsed().as_secs_f64());
                assert_eq!(black_box(result).map(f64::to_bits), expected);
            }
        }
        eprintln!("algorithm_lab_integrated_loudness: 30 s synthetic, {rate} Hz, {ch} channels; full meter {:?} s; integrated only {:?} s; same result bits {:?}; excludes decode, FFT and I/O", timings[0], timings[1], expected);
    }

    #[test]
    fn a_full_scale_1k_sine_reads_as_bs1770_says() {
        // BS.1770: a 0 dBFS 997 Hz sine in one channel reads -3.01 LKFS; in
        // both channels of a stereo pair, +3 dB more.
        for rate in [44_100, 48_000, 96_000] {
            let r = report(&sine(rate, 997.0, 1.0, 10.0, 1), rate, 1);
            assert!((r.integrated.unwrap() + 3.01).abs() < 0.05, "{rate}: {:?}", r.integrated);
            let r = report(&sine(rate, 997.0, 1.0, 10.0, 2), rate, 2);
            assert!((r.integrated.unwrap() - 0.0).abs() < 0.05, "{rate}: {:?}", r.integrated);
        }
    }

    #[test]
    fn a_steady_tone_has_no_loudness_range_and_a_crest_of_3_db() {
        let r = report(&sine(48_000, 1000.0, 0.5, 20.0, 2), 48_000, 2);
        assert!(r.lra.unwrap() < 0.1, "{:?}", r.lra);
        assert!((r.crest - 3.01).abs() < 0.05, "{}", r.crest);
        assert!((r.correlation.unwrap() - 1.0).abs() < 1e-9);
        assert!(r.dc.iter().all(|d| d.abs() < 1e-3));
        assert_eq!((r.clips, r.overs), (0, 0));
    }

    #[test]
    fn steps_in_level_make_a_loudness_range() {
        // 20 s at -20 dB then 20 s at -30 dB: LRA close to 10 LU.
        let mut s = sine(48_000, 1000.0, 0.1, 20.0, 2);
        s.extend(sine(48_000, 1000.0, 0.0316, 20.0, 2));
        let r = report(&s, 48_000, 2);
        let lra = r.lra.unwrap();
        assert!((lra - 10.0).abs() < 1.0, "{lra}");
    }

    #[test]
    fn a_quarter_rate_sine_peaking_between_samples_shows_a_true_peak_over() {
        // fs/4 at 45° phase: every sample is at 0.707 of the wave's peak, so
        // samples at full scale hide a peak about 3 dB above it.
        let rate = 48_000;
        let n = rate as usize;
        let s: Vec<f32> = (0..n).map(|i| (1.0 / std::f64::consts::FRAC_1_SQRT_2 * 0.999 * (std::f64::consts::PI / 2.0 * i as f64 + std::f64::consts::PI / 4.0).sin()) as f32).collect();
        let r = report(&s, rate, 1);
        assert!(r.sample_peak <= 0.0 && r.sample_peak > -0.1, "{}", r.sample_peak);
        assert!(r.true_peak > 2.5 && r.true_peak < 3.5, "{}", r.true_peak);
        assert!(r.overs >= 1);
    }

    #[test]
    fn a_peak_that_falls_on_a_sample_is_the_true_peak_too() {
        // One full-scale sample in silence: between the samples the wave is lower.
        let mut s = vec![0f32; 48_000];
        s[24_000] = 1.0;
        let r = report(&s, 48_000, 1);
        assert!(r.sample_peak.abs() < 1e-9, "{}", r.sample_peak);
        assert!(r.true_peak >= r.sample_peak, "{} below {}", r.true_peak, r.sample_peak);
    }

    #[test]
    fn three_samples_at_full_scale_are_a_clip_and_dc_is_measured() {
        let mut s = vec![0.25f32; 48_000];
        for v in &mut s[1000..1004] {
            *v = 1.0;
        }
        let r = report(&s, 48_000, 1);
        assert_eq!(r.clips, 1);
        assert!((r.clip_at[0] - 1002.0 / 48_000.0).abs() < 1e-4);
        assert!((r.dc[0] - 0.25).abs() < 0.001);
    }

    /// A 24-bit PCM WAV of `samples` (already 24-bit integers).
    fn wav24(path: &Path, rate: u32, samples: &[i32]) {
        let data: Vec<u8> = samples.iter().flat_map(|v| v.to_le_bytes()[..3].to_vec()).collect();
        let mut b = Vec::new();
        b.extend(b"RIFF");
        b.extend((36 + data.len() as u32).to_le_bytes());
        b.extend(b"WAVEfmt ");
        b.extend(16u32.to_le_bytes());
        b.extend(1u16.to_le_bytes());
        b.extend(1u16.to_le_bytes());
        b.extend(rate.to_le_bytes());
        b.extend((rate * 3).to_le_bytes());
        b.extend(3u16.to_le_bytes());
        b.extend(24u16.to_le_bytes());
        b.extend(b"data");
        b.extend((data.len() as u32).to_le_bytes());
        b.extend(data);
        std::fs::write(path, b).unwrap();
    }

    #[test]
    fn a_24_bit_file_of_16_bit_audio_is_found_out() {
        let dir = std::env::temp_dir().join(format!("moosik-measure-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let tone: Vec<i32> = (0..48_000).map(|i| ((i as f64 * 0.05).sin() * 3_000_000.0) as i32).collect();
        let padded: Vec<i32> = tone.iter().map(|v| v & !0xff).collect();
        let (real, fake) = (dir.join("real.wav"), dir.join("fake.wav"));
        wav24(&real, 48_000, &tone);
        wav24(&fake, 48_000, &padded);
        let (c, p) = (AtomicBool::new(false), AtomicU32::new(0));
        let r = measure(&real, &c, &p).unwrap();
        assert_eq!((r.declared_bits, r.used_bits), (Some(24), Some(24)));
        assert_eq!(p.load(Ordering::Relaxed), 1000);
        let f = measure(&fake, &c, &p).unwrap();
        assert_eq!((f.declared_bits, f.used_bits), (Some(24), Some(16)));
        assert_eq!(f.md5_ok, None, "WAV has no MD5");
        std::fs::remove_dir_all(&dir).unwrap();
    }

    /// FLAC's MD5 against a real file and a damaged copy. MOOSIK_FLAC = a FLAC
    /// with an MD5 in its header.
    #[test]
    #[ignore]
    fn flac_md5_passes_and_a_flipped_byte_fails() {
        let src = std::path::PathBuf::from(std::env::var("MOOSIK_FLAC").expect("set MOOSIK_FLAC"));
        let (c, p) = (AtomicBool::new(false), AtomicU32::new(0));
        let t = std::time::Instant::now();
        let r = measure(&src, &c, &p).unwrap();
        eprintln!("{:.0} s of audio measured in {:?}: {r:?}", r.seconds, t.elapsed());
        assert_eq!(r.md5_ok, Some(true));
        let mut bytes = std::fs::read(&src).unwrap();
        let at = bytes.len() / 2;
        bytes[at] ^= 0x10;
        let bad = std::env::temp_dir().join(format!("moosik-bad-{}.flac", std::process::id()));
        std::fs::write(&bad, bytes).unwrap();
        let b = measure(&bad, &c, &p);
        eprintln!("damaged: {:?}", b.as_ref().map(|r| r.md5_ok));
        assert!(b.is_err() || b.unwrap().md5_ok == Some(false));
        std::fs::remove_file(&bad).unwrap();
    }

    #[test]
    fn the_findings_read_plainly() {
        let mut r = report(&sine(44_100, 1000.0, 0.5, 5.0, 2), 44_100, 2);
        assert!(verdicts(&r, Some(true), None).is_empty(), "nothing to say about a clean tone");
        r.md5_ok = Some(false);
        r.declared_bits = Some(24);
        r.used_bits = Some(16);
        let v = verdicts(&r, Some(true), Some(16_000.0));
        assert_eq!(v.len(), 3);
        assert!(v.iter().all(|(l, _)| *l == Level::Warn));
        assert!(v[2].1.contains("16.0 kHz"));
        // A cutoff near Nyquist is the format's, not an encoder's.
        r.md5_ok = None;
        r.used_bits = Some(24);
        assert!(verdicts(&r, Some(true), Some(21_000.0)).is_empty());
        assert!(verdicts(&r, Some(false), Some(16_000.0)).is_empty(), "a lossy file is expected to be cut off");
    }

    #[test]
    fn two_masters_compare_by_loudness_and_by_band() {
        let dir = std::env::temp_dir().join(format!("moosik-compare-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        // White noise, the same noise 6 dB up, and the noise low-passed.
        let mut seed = 12345u32;
        let noise: Vec<f64> = (0..48_000 * 6).map(|_| {
            seed = seed.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
            (seed >> 8) as f64 / (1u32 << 24) as f64 - 0.5
        }).collect();
        let to24 = |x: &[f64], g: f64| x.iter().map(|v| (v * g * 4_000_000.0) as i32).collect::<Vec<_>>();
        let mut lp = 0.0;
        let dull: Vec<f64> = noise.iter().map(|v| { lp += 0.2 * (v - lp); lp }).collect();
        let (a, b, c) = (dir.join("a.wav"), dir.join("b.wav"), dir.join("c.wav"));
        wav24(&a, 48_000, &to24(&noise, 1.0));
        wav24(&b, 48_000, &to24(&noise, 2.0));
        wav24(&c, 48_000, &to24(&dull, 1.0));
        let (k, p) = (AtomicBool::new(false), AtomicU32::new(0));
        let louder = compare(&a, &b, &k, &p).unwrap();
        // Real Compare entry point, with the unchanged full-file meter as
        // the oracle for the loudness that determines its spectral shift.
        let (old_a, old_b) = (measure(&a, &k, &p).unwrap().integrated.unwrap(), measure(&b, &k, &p).unwrap().integrated.unwrap());
        assert_eq!(louder.louder_db.unwrap().to_bits(), (old_b - old_a).to_bits());
        assert!((louder.louder_db.unwrap() - 6.02).abs() < 0.05, "{:?}", louder.louder_db);
        assert!(louder.diff.iter().all(|(_, d)| d.abs() < 0.05), "matched, the spectra are the same");
        assert_eq!(p.load(Ordering::Relaxed), 1000);
        let duller = compare(&a, &c, &k, &p).unwrap();
        let at = |f: f32| duller.diff.iter().find(|(c, _)| (*c - f).abs() / f < 0.05).unwrap().1;
        assert!(at(10_079.4) < at(100.8) - 10.0, "treble cut: {} vs {}", at(10_079.4), at(100.8));
        std::fs::remove_dir_all(&dir).unwrap();
    }

    #[test]
    fn opposite_channels_correlate_at_minus_one() {
        let mut s = sine(48_000, 440.0, 0.5, 2.0, 2);
        for f in s.chunks_exact_mut(2) {
            f[1] = -f[0];
        }
        let r = report(&s, 48_000, 2);
        assert!((r.correlation.unwrap() + 1.0).abs() < 1e-9);
    }

    // -----------------------------------------------------------------------
    // Verbatim OldMeter from de4fcb3 (scalar polyphase loop oracle)
    // -----------------------------------------------------------------------
    struct OldMeter {
        rate: u32,
        ch: usize,
        k: Vec<[Biquad; 2]>,
        seg: f64,
        seg_n: usize,
        seg_len: usize,
        segs: Vec<f64>,
        peak: f64,
        tp: f64,
        os: usize,
        phases: Vec<Vec<f64>>,
        hist: Vec<Vec<f64>>,
        sum_sq: f64,
        sum: Vec<f64>,
        lr: (f64, f64, f64),
        n: u64,
        clip_run: Vec<u32>,
        clips: u32,
        clip_at: Vec<f32>,
        over_run: bool,
        overs: u32,
        over_at: Vec<f32>,
    }

    impl OldMeter {
        fn new(rate: u32, ch: usize) -> Self {
            let (os, phases) = true_peak_filter(rate);
            let taps = phases[0].len();
            OldMeter {
                rate,
                ch,
                k: vec![k_weighting(rate); ch],
                seg: 0.0,
                seg_n: 0,
                seg_len: (rate as usize / 10).max(1),
                segs: Vec::new(),
                peak: 0.0,
                tp: 0.0,
                os,
                phases,
                hist: vec![vec![0.0; taps]; ch],
                sum_sq: 0.0,
                sum: vec![0.0; ch],
                lr: (0.0, 0.0, 0.0),
                n: 0,
                clip_run: vec![0; ch],
                clips: 0,
                clip_at: Vec::new(),
                over_run: false,
                overs: 0,
                over_at: Vec::new(),
            }
        }

        fn push(&mut self, samples: &[f32]) {
            let ch = self.ch;
            let taps = self.phases[0].len();
            for frame in samples.chunks_exact(ch) {
                let t = self.n as f32 / self.rate as f32;
                let mut e = 0.0;
                let mut frame_over = false;
                for (c, &s) in frame.iter().enumerate() {
                    let x = s as f64;
                    let a = x.abs();
                    self.peak = self.peak.max(a);
                    self.sum_sq += x * x;
                    self.sum[c] += x;
                    if a >= 0.9999 {
                        self.clip_run[c] += 1;
                        if self.clip_run[c] == 3 {
                            self.clips += 1;
                            if self.clip_at.len() < 500 {
                                self.clip_at.push(t);
                            }
                        }
                    } else {
                        self.clip_run[c] = 0;
                    }
                    let shelf = self.k[c][0].run(x);
                    let y = self.k[c][1].run(shelf);
                    e += weight(c, ch) * y * y;
                    let h = &mut self.hist[c];
                    h.copy_within(1..taps, 0);
                    h[taps - 1] = x;
                    if self.os == 1 {
                        self.tp = self.tp.max(a);
                        frame_over |= a > 1.0;
                    } else {
                        for ph in &self.phases {
                            let v: f64 = ph.iter().zip(h.iter()).map(|(k, x)| k * x).sum::<f64>().abs();
                            self.tp = self.tp.max(v);
                            frame_over |= v > 1.0;
                        }
                    }
                }
                if frame_over && !self.over_run {
                    self.overs += 1;
                    if self.over_at.len() < 500 {
                        self.over_at.push(t);
                    }
                }
                self.over_run = frame_over;
                if ch >= 2 {
                    let (l, r) = (frame[0] as f64, frame[1] as f64);
                    self.lr.0 += l * r;
                    self.lr.1 += l * l;
                    self.lr.2 += r * r;
                }
                self.seg += e;
                self.seg_n += 1;
                if self.seg_n == self.seg_len {
                    self.segs.push(self.seg / self.seg_len as f64);
                    self.seg = 0.0;
                    self.seg_n = 0;
                }
                self.n += 1;
            }
        }

        fn finish(self, declared_bits: Option<u32>, used_bits: Option<u32>, md5_ok: Option<bool>) -> Report {
            let lufs = |ms: f64| -0.691 + 10.0 * ms.log10();
            let windows = |w: usize| -> Vec<f64> {
                if self.segs.len() < w {
                    return Vec::new();
                }
                let mut out = Vec::with_capacity(self.segs.len() - w + 1);
                let mut acc: f64 = self.segs[..w].iter().sum();
                out.push(acc / w as f64);
                for i in w..self.segs.len() {
                    acc += self.segs[i] - self.segs[i - w];
                    out.push(acc / w as f64);
                }
                out
            };
            let blocks = windows(4);
            let short = windows(30);
            let abs_gate = 10f64.powf((-70.0 + 0.691) / 10.0);
            let gated: Vec<f64> = blocks.iter().copied().filter(|&b| b > abs_gate).collect();
            let integrated = (!gated.is_empty()).then(|| {
                let mean = gated.iter().sum::<f64>() / gated.len() as f64;
                let rel = mean * 10f64.powf(-10.0 / 10.0);
                let g: Vec<f64> = gated.iter().copied().filter(|&b| b > rel).collect();
                lufs(g.iter().sum::<f64>() / g.len().max(1) as f64)
            });
            let lra = {
                let st: Vec<f64> = short.iter().copied().filter(|&b| b > abs_gate).collect();
                (!st.is_empty()).then(|| {
                    let mean = st.iter().sum::<f64>() / st.len() as f64;
                    let rel = mean * 10f64.powf(-20.0 / 10.0);
                    let mut l: Vec<f64> = st.iter().copied().filter(|&b| b > rel).map(lufs).collect();
                    l.sort_by(|a, b| a.total_cmp(b));
                    let at = |p: f64| l[((l.len() - 1) as f64 * p).round() as usize];
                    at(0.95) - at(0.10)
                })
            };
            let max_of = |v: &[f64]| v.iter().copied().fold(None, |m: Option<f64>, x| Some(m.map_or(x, |m| m.max(x)))).filter(|&x| x > 0.0).map(lufs);
            let max_short_term = max_of(&short);
            let max_momentary = max_of(&blocks);
            let db = |x: f64| if x > 0.0 { 20.0 * x.log10() } else { -144.0 };
            let true_peak = db(self.tp.max(self.peak));
            let n = self.n.max(1) as f64;
            let rms = (self.sum_sq / (n * self.ch as f64)).sqrt();
            let correlation = (self.ch >= 2 && self.lr.1 * self.lr.2 > 0.0).then(|| self.lr.0 / (self.lr.1 * self.lr.2).sqrt());
            Report {
                rate: self.rate,
                channels: self.ch,
                seconds: self.n as f64 / self.rate as f64,
                integrated,
                lra,
                max_short_term,
                max_momentary,
                sample_peak: db(self.peak),
                true_peak,
                plr: integrated.map(|i| true_peak - i),
                psr: max_short_term.map(|s| true_peak - s),
                crest: db(self.peak) - db(rms),
                dr: None,
                dc: self.sum.iter().map(|s| s / n).collect(),
                clips: self.clips,
                clip_at: self.clip_at,
                overs: self.overs,
                over_at: self.over_at,
                declared_bits,
                used_bits,
                correlation,
                md5_ok,
            }
        }
    }

    /// Full 21-field bit-for-bit equality check on `Report`.
    /// Policy for floating-point fields (`f64` and `f32`):
    /// - Finite values, `±0.0`, and `±Inf`: must have identical IEEE-754 bits (`a.to_bits() == b.to_bits()`).
    /// - `NaN` values: both must be `NaN` (`a.is_nan() && b.is_nan()`) AND carry the exact same bit pattern (`a.to_bits() == b.to_bits()`).
    fn assert_f64_bits_eq(a: f64, b: f64, field: &str, ctx: &str) {
        if a.is_nan() || b.is_nan() {
            assert!(a.is_nan() && b.is_nan(), "{ctx}: {field} NaN mismatch: new={a} ({:#018x}) old={b} ({:#018x})", a.to_bits(), b.to_bits());
        }
        assert_eq!(a.to_bits(), b.to_bits(), "{ctx}: {field} bits mismatch: new={a} ({:#018x}) old={b} ({:#018x})", a.to_bits(), b.to_bits());
    }

    fn assert_opt_f64_bits_eq(a: Option<f64>, b: Option<f64>, field: &str, ctx: &str) {
        match (a, b) {
            (None, None) => {}
            (Some(va), Some(vb)) => assert_f64_bits_eq(va, vb, field, ctx),
            _ => panic!("{ctx}: {field} Option mismatch: new={a:?} old={b:?}"),
        }
    }

    fn assert_reports_bit_identical(r_new: &Report, r_old: &Report, ctx: &str) {
        assert_eq!(r_new.rate, r_old.rate, "{ctx}: rate");
        assert_eq!(r_new.channels, r_old.channels, "{ctx}: channels");
        assert_f64_bits_eq(r_new.seconds, r_old.seconds, "seconds", ctx);
        assert_opt_f64_bits_eq(r_new.integrated, r_old.integrated, "integrated", ctx);
        assert_opt_f64_bits_eq(r_new.lra, r_old.lra, "lra", ctx);
        assert_opt_f64_bits_eq(r_new.max_short_term, r_old.max_short_term, "max_short_term", ctx);
        assert_opt_f64_bits_eq(r_new.max_momentary, r_old.max_momentary, "max_momentary", ctx);
        assert_f64_bits_eq(r_new.sample_peak, r_old.sample_peak, "sample_peak", ctx);
        assert_f64_bits_eq(r_new.true_peak, r_old.true_peak, "true_peak", ctx);
        assert_opt_f64_bits_eq(r_new.plr, r_old.plr, "plr", ctx);
        assert_opt_f64_bits_eq(r_new.psr, r_old.psr, "psr", ctx);
        assert_f64_bits_eq(r_new.crest, r_old.crest, "crest", ctx);
        assert_eq!(r_new.dc.len(), r_old.dc.len(), "{ctx}: dc.len()");
        for (c, (&da, &db)) in r_new.dc.iter().zip(&r_old.dc).enumerate() {
            assert_f64_bits_eq(da, db, &format!("dc[{c}]"), ctx);
        }
        assert_eq!(r_new.clips, r_old.clips, "{ctx}: clips");
        assert_eq!(
            r_new.clip_at.iter().map(|v| v.to_bits()).collect::<Vec<_>>(),
            r_old.clip_at.iter().map(|v| v.to_bits()).collect::<Vec<_>>(),
            "{ctx}: clip_at"
        );
        assert_eq!(r_new.overs, r_old.overs, "{ctx}: overs");
        assert_eq!(
            r_new.over_at.iter().map(|v| v.to_bits()).collect::<Vec<_>>(),
            r_old.over_at.iter().map(|v| v.to_bits()).collect::<Vec<_>>(),
            "{ctx}: over_at"
        );
        assert_eq!(r_new.declared_bits, r_old.declared_bits, "{ctx}: declared_bits");
        assert_eq!(r_new.used_bits, r_old.used_bits, "{ctx}: used_bits");
        assert_opt_f64_bits_eq(r_new.correlation, r_old.correlation, "correlation", ctx);
        assert_eq!(r_new.md5_ok, r_old.md5_ok, "{ctx}: md5_ok");
    }

    #[test]
    fn avx2_true_peak_matches_unchanged_scalar_meter_bit_for_bit() {
        for &rate in &[44_100u32, 48_000, 88_200, 96_000, 192_000] {
            for &ch in &[1usize, 2, 6] {
                let frames = 12_345usize;
                let mut sig = lab_signal(rate, ch, frames);

                // Inject inter-sample true peak overs (fs/4 @ 45°), full-scale clips, 0.0, -0.0
                for i in 100..250 {
                    let v = (1.0 / std::f64::consts::FRAC_1_SQRT_2 * 0.999 * (std::f64::consts::PI / 2.0 * i as f64 + std::f64::consts::PI / 4.0).sin()) as f32;
                    sig[i * ch] = v;
                    if ch > 1 { sig[i * ch + 1] = -v; }
                }
                for i in 500..505 {
                    sig[i * ch] = 1.0;
                    if ch > 1 { sig[i * ch + 1] = -1.0; }
                }
                for i in 1000..1030 {
                    sig[i * ch] = if i % 2 == 0 { 0.0 } else { -0.0 };
                }

                for chunk_frames in [frames, 256, 63, 7] {
                    let mut m_new = Meter::new(rate, ch);
                    let mut m_old = OldMeter::new(rate, ch);
                    for blk in sig.chunks(chunk_frames * ch) {
                        m_new.push(blk);
                        m_old.push(blk);
                    }
                    assert_eq!(m_new.tp.to_bits(), m_old.tp.to_bits(), "tp bits mismatch rate={rate} ch={ch} chunk={chunk_frames}");
                    assert_eq!(m_new.peak.to_bits(), m_old.peak.to_bits(), "peak bits mismatch rate={rate} ch={ch}");
                    let r_new = m_new.finish(Some(24), Some(20), Some(true));
                    let r_old = m_old.finish(Some(24), Some(20), Some(true));
                    assert_reports_bit_identical(&r_new, &r_old, &format!("short rate={rate} ch={ch} chunk={chunk_frames}"));
                }

                // Also test non-finite specials: NaN, payload NaN, +Inf, -Inf, +0.0, -0.0, full scale,
                // verifying ALL 21 fields of `Report` under explicit NaN/bit-comparison policy.
                for special in [0.0f32, -0.0, 1.0, -1.0, f32::NAN, f32::from_bits(0x7fc0_1234), f32::INFINITY, f32::NEG_INFINITY] {
                    let mut s2 = sig[..2048 * ch].to_vec();
                    s2[300 * ch] = special;
                    if ch > 1 { s2[700 * ch + 1] = special; }
                    let mut m_new = Meter::new(rate, ch);
                    let mut m_old = OldMeter::new(rate, ch);
                    for blk in s2.chunks(199 * ch) {
                        m_new.push(blk);
                        m_old.push(blk);
                    }
                    assert_eq!(
                        m_new.tp.to_bits(),
                        m_old.tp.to_bits(),
                        "special {:08x} tp bits mismatch rate={rate} ch={ch}: new={} old={}",
                        special.to_bits(), m_new.tp, m_old.tp
                    );
                    assert_eq!(m_new.peak.to_bits(), m_old.peak.to_bits());
                    let r_new = m_new.finish(Some(24), Some(16), Some(false));
                    let r_old = m_old.finish(Some(24), Some(16), Some(false));
                    assert_reports_bit_identical(
                        &r_new,
                        &r_old,
                        &format!("special={:#010x} rate={rate} ch={ch}", special.to_bits()),
                    );
                }
            }
        }

        // Long finite signals (>= 4.2 seconds -> >= 42 segments of 100 ms) so that
        // `integrated`, `max_momentary`, `max_short_term`, `lra`, `plr`, and `psr` are all `Some(...)`,
        // and verify all 21 fields of `Report` bit-for-bit across rates and channel counts.
        for &(rate, ch) in &[(44_100u32, 2usize), (48_000, 1), (48_000, 2), (48_000, 6), (96_000, 2)] {
            let frames = rate as usize * 4 + rate as usize / 5 + 37; // ~4.2s
            let mut sig = lab_signal(rate, ch, frames);
            for i in 200..320 {
                let v = (1.0 / std::f64::consts::FRAC_1_SQRT_2 * 0.999 * (std::f64::consts::PI / 2.0 * i as f64 + std::f64::consts::PI / 4.0).sin()) as f32;
                sig[i * ch] = v;
                if ch > 1 { sig[i * ch + 1] = -v; }
            }
            for i in 1200..1206 {
                sig[i * ch] = 1.0;
            }
            let mut m_new = Meter::new(rate, ch);
            let mut m_old = OldMeter::new(rate, ch);
            for blk in sig.chunks(251 * ch) {
                m_new.push(blk);
                m_old.push(blk);
            }
            let r_new = m_new.finish(Some(24), Some(24), Some(true));
            let r_old = m_old.finish(Some(24), Some(24), Some(true));
            assert!(r_new.integrated.is_some(), "integrated must be populated on 4.2s signal");
            assert!(r_new.max_momentary.is_some(), "max_momentary must be populated on 4.2s signal");
            assert!(r_new.max_short_term.is_some(), "max_short_term must be populated on 4.2s signal");
            assert!(r_new.lra.is_some(), "lra must be populated on 4.2s signal");
            assert!(r_new.plr.is_some(), "plr must be populated on 4.2s signal");
            assert!(r_new.psr.is_some(), "psr must be populated on 4.2s signal");
            assert_reports_bit_identical(&r_new, &r_old, &format!("long-4.2s rate={rate} ch={ch}"));
        }
    }

    #[test]
    #[ignore = "same-binary A/B benchmark on real music tracks (48kHz and 44.1kHz)"]
    fn bench_true_peak_ab_real_music() {
        use std::hint::black_box;
        use std::time::Instant;

        for (label, path_str) in [
            ("song1_48k (216s Vocaloid FLAC)", r"C:\Users\PC\Documents\ASYN\Audio Archive\Vocaloid\01_02_退廃的人生讃歌.flac"),
            ("koi_no_wana_44k1 (255s FLAC)", r"C:\Users\PC\Documents\ASYN\Audio Archive\01 - Koi no Wana.flac"),
        ] {
            let path = Path::new(path_str);
            let cancel = AtomicBool::new(false);
            let prog = AtomicU32::new(0);
            let mut all_samples = Vec::new();
            let mut track_sr = 48_000u32;
            let mut track_ch = 2usize;
            each_block(path, &cancel, &prog, |sr, ch, blk| {
                track_sr = sr;
                track_ch = ch;
                all_samples.extend_from_slice(blk);
            }).expect("decode real music");
            let secs = (all_samples.len() / track_ch) as f64 / track_sr as f64;

            let run_old = || -> (f64, u64, u64, u32, usize, Report) {
                let mut m = OldMeter::new(track_sr, track_ch);
                let t0 = Instant::now();
                for blk in all_samples.chunks(4096 * track_ch) {
                    m.push(black_box(blk));
                }
                let dt = t0.elapsed().as_secs_f64() * 1e3;
                let tp_bits = m.tp.to_bits();
                let pk_bits = m.peak.to_bits();
                let overs = m.overs;
                let n_over_at = m.over_at.len();
                let rep = m.finish(None, None, None);
                (dt, tp_bits, pk_bits, overs, n_over_at, rep)
            };

            let run_new = || -> (f64, u64, u64, u32, usize, Report) {
                let mut m = Meter::new(track_sr, track_ch);
                let t0 = Instant::now();
                for blk in all_samples.chunks(4096 * track_ch) {
                    m.push(black_box(blk));
                }
                let dt = t0.elapsed().as_secs_f64() * 1e3;
                let tp_bits = m.tp.to_bits();
                let pk_bits = m.peak.to_bits();
                let overs = m.overs;
                let n_over_at = m.over_at.len();
                let rep = m.finish(None, None, None);
                (dt, tp_bits, pk_bits, overs, n_over_at, rep)
            };

            // Warm-up and exact bit verification
            let (_, tp_o, pk_o, ov_o, ova_o, rep_o) = run_old();
            let (_, tp_n, pk_n, ov_n, ova_n, rep_n) = run_new();
            assert_eq!(tp_o, tp_n, "{label}: tp.to_bits() mismatch!");
            assert_eq!(pk_o, pk_n, "{label}: peak.to_bits() mismatch!");
            assert_eq!(ov_o, ov_n, "{label}: overs mismatch!");
            assert_eq!(ova_o, ova_n, "{label}: over_at len mismatch!");
            assert_reports_bit_identical(&rep_n, &rep_o, label);
            println!(
                "\n=== {label}: {:.2}s @ {}Hz {}ch | tp={:.6} dBTP (bits={:#018x}) overs={} LUFS={:?} [BIT-EXACT] ===",
                secs, track_sr, track_ch, rep_n.true_peak, tp_n, ov_n, rep_n.integrated
            );

            let mut old_ms = Vec::new();
            let mut new_ms = Vec::new();
            for pair in 0..5 {
                let (o, n) = if pair % 2 == 0 {
                    let (o, to, _, _, _, _) = run_old();
                    let (n, tn, _, _, _, _) = run_new();
                    assert_eq!(to, tn);
                    (o, n)
                } else {
                    let (n, tn, _, _, _, _) = run_new();
                    let (o, to, _, _, _, _) = run_old();
                    assert_eq!(to, tn);
                    (o, n)
                };
                println!("  pair {}: old={o:.2}ms, new={n:.2}ms", pair + 1);
                old_ms.push(o);
                new_ms.push(n);
            }
            old_ms.sort_by(f64::total_cmp);
            new_ms.sort_by(f64::total_cmp);
            let med_old = old_ms[old_ms.len() / 2];
            let med_new = new_ms[new_ms.len() / 2];
            println!(
                "  MEDIAN ({:.1}s audio): old={med_old:.2}ms -> new={med_new:.2}ms ({:.2}x, saves {:.2}ms / -{:.1}%)",
                secs, med_old / med_new, med_old - med_new, (1.0 - med_new / med_old) * 100.0
            );
        }
    }
}
