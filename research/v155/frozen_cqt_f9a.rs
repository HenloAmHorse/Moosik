//! Frozen product CQT from f9a69999d8d97d032e2f3c07ecc204a65ed50369.
//! Independent baseline: do not optimize or share candidate helpers here.
//! Only the original test-module declaration was removed.

//! Genuine constant-Q analysis, independent of FFT-bin display weighting.
//!
//! Each supplied frequency has its own full, symmetric, inverse-frequency
//! window. `bins_per_octave` sets **support Q**, not the number of display
//! columns or a promise about the window's measured -3 dB bandwidth. The caller
//! supplies the logarithmic display sampling separately. No support cap,
//! decimation, thresholded kernel or recursive/sliding approximation is used.
//!
//! The dense overlap-save route shares each source FFT among kernels of the
//! same transform size. A stereo run does two convolutions; Mix is the magnitude
//! of their complex mean, never the mean of two magnitudes. Complex values live
//! for one FFT block only. Finished cells go through the caller's level/code
//! mapping immediately, into frame-major code planes.
//!
//! Limits bound explicit dense buffer payloads and returned code payloads, not
//! RSS: the input, Rust allocator overhead, small plan metadata and rustfft plan
//! allocations are excluded. Only one transform-size group is planned at a
//! time, and `max_fft_len` additionally bounds its possible plan size. Refusal
//! does not silently shorten a filter or fall back to FFT weighting.

use rustfft::{FftPlanner, num_complex::Complex};
use serde::{Deserialize, Serialize};
use std::collections::BTreeMap;
use std::f64::consts::{PI, TAU};
use std::fmt;
use std::mem::size_of;

/// Bump when the transform, clock, gain or numerical route changes.
pub const REVISION: u32 = 1;

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub enum Window {
    Hann,
    Hamming,
    Blackman,
    BlackmanHarris,
}

impl Window {
    pub fn label(self) -> &'static str {
        match self {
            Self::Hann => "Hann",
            Self::Hamming => "Hamming",
            Self::Blackman => "Blackman",
            Self::BlackmanHarris => "Blackman-Harris (4-term)",
        }
    }

    fn at(self, theta: f64) -> f64 {
        match self {
            Self::Hann => 0.5 + 0.5 * theta.cos(),
            Self::Hamming => 0.54 + 0.46 * theta.cos(),
            Self::Blackman => 0.42 + 0.5 * theta.cos() + 0.08 * (2.0 * theta).cos(),
            Self::BlackmanHarris => {
                0.35875
                    + 0.48829 * theta.cos()
                    + 0.14128 * (2.0 * theta).cos()
                    + 0.01168 * (3.0 * theta).cos()
            }
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Serialize, Deserialize)]
#[serde(default)]
pub struct Config {
    pub bins_per_octave: u16,
    pub window: Window,
    pub frame_rate: f32,
}

impl Default for Config {
    fn default() -> Self {
        Self {
            bins_per_octave: 48,
            window: Window::Blackman,
            frame_rate: 180.0,
        }
    }
}

impl Config {
    pub fn validate(&self) -> Result<(), Error> {
        if !(12..=192).contains(&self.bins_per_octave) {
            return Err(Error::Invalid(
                "CQT density must be 12 to 192 bins per octave",
            ));
        }
        if !self.frame_rate.is_finite() || !(1.0..=360.0).contains(&self.frame_rate) {
            return Err(Error::Invalid(
                "CQT frame rate must be finite and 1 to 360 fps",
            ));
        }
        Ok(())
    }

    /// Support Q; changing the window changes leakage/mainlobe width even at
    /// the same value. Display oversampling does not change this Q.
    pub fn support_q(&self) -> f64 {
        1.0 / (2.0_f64.powf(1.0 / f64::from(self.bins_per_octave)) - 1.0)
    }

    /// Full uncapped support in seconds, excluding the endpoint rounding.
    pub fn support_seconds(&self, frequency: f64) -> f64 {
        self.support_q() / frequency
    }
}

#[derive(Clone, Copy, Debug)]
pub struct Limits {
    /// Dense FFT/kernel/scratch vectors, excluding rustfft plan internals.
    pub workspace_bytes: usize,
    /// Sum of all returned `u16` planes (one mono, three joint).
    pub output_bytes: usize,
    /// Maximum permitted power-of-two transform length, not a support cap.
    pub max_fft_len: usize,
}

impl Default for Limits {
    fn default() -> Self {
        Self {
            workspace_bytes: 256 * 1024 * 1024,
            output_bytes: 1536 * 1024 * 1024,
            max_fft_len: 1 << 22,
        }
    }
}

#[derive(Clone, Copy)]
pub enum Input<'a> {
    Mono(&'a [f32]),
    Joint { left: &'a [f32], right: &'a [f32] },
}

#[derive(Debug, Default, Clone, Copy)]
pub struct WorkStats {
    /// Per-kernel, per-channel column orchestrations actually entered. This
    /// remains distinct from the FFT calls used to produce each column.
    pub column_convolutions: u64,
    /// Forward kernel transforms, shared by left and right.
    pub kernel_ffts: u64,
    /// Actual source forward transforms, counted per channel and attempt.
    pub input_ffts: u64,
    /// Actual dense inverse transforms, counted per channel and attempt.
    pub inverse_ffts: u64,
    pub batches: u64,
    /// Explicit simultaneously live dense vector payload, not RSS.
    pub peak_workspace_bytes: usize,
    pub output_bytes: usize,
}

#[derive(Debug)]
pub struct Output {
    /// Frame-major Mix, Left, Right. Mono has empty Left and Right planes.
    pub codes: [Vec<u16>; 3],
    pub frames: usize,
    pub bars: usize,
    pub work: WorkStats,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Error {
    Invalid(&'static str),
    Resource(&'static str),
    Allocation,
    Aborted,
}

impl fmt::Display for Error {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Invalid(message) => write!(f, "{message}"),
            Self::Resource(message) => {
                write!(f, "CQT resource limit: {message}; no filter was shortened")
            }
            Self::Allocation => write!(f, "CQT buffer allocation failed"),
            Self::Aborted => write!(f, "CQT analysis cancelled"),
        }
    }
}

impl std::error::Error for Error {}

/// The number of frame centres in `[0, samples)`. Exact frame locations use
/// [`frame_center`], not a rounded integer hop accumulated over the track.
pub fn frame_count(samples: usize, sample_rate: u32, frame_rate: f32) -> Result<usize, Error> {
    if sample_rate == 0 || !frame_rate.is_finite() || frame_rate <= 0.0 {
        return Err(Error::Invalid(
            "CQT clock must have positive sample and frame rates",
        ));
    }
    let frames = (samples as f64 * f64::from(frame_rate) / f64::from(sample_rate)).ceil();
    if !frames.is_finite() || frames >= usize::MAX as f64 {
        return Err(Error::Resource("frame count overflows"));
    }
    Ok(frames as usize)
}

/// `floor(frame * sample_rate / frame_rate)`; centred kernels need future
/// samples. Preprocess supplies them; this is not a zero-latency live transform.
pub fn frame_center(frame: usize, sample_rate: u32, frame_rate: f32) -> usize {
    (frame as f64 * f64::from(sample_rate) / f64::from(frame_rate)).floor() as usize
}

#[derive(Clone, Copy)]
struct Bin {
    index: usize,
    half: usize,
    frequency: f64,
}

fn checked_product(a: usize, b: usize) -> Result<usize, Error> {
    a.checked_mul(b)
        .ok_or(Error::Resource("buffer size overflows"))
}

fn zeroed<T: Default + Clone>(len: usize) -> Result<Vec<T>, Error> {
    let mut values = Vec::new();
    values
        .try_reserve_exact(len)
        .map_err(|_| Error::Allocation)?;
    values.resize(len, T::default());
    Ok(values)
}

fn poll(should_continue: &dyn Fn() -> bool) -> Result<(), Error> {
    if should_continue() {
        Ok(())
    } else {
        Err(Error::Aborted)
    }
}

fn layout(
    frequencies: &[f64],
    sample_rate: u32,
    config: &Config,
    limits: Limits,
) -> Result<BTreeMap<usize, Vec<Bin>>, Error> {
    config.validate()?;
    if sample_rate == 0 {
        return Err(Error::Invalid("CQT sample rate must be positive"));
    }
    if frequencies.is_empty() || frequencies.len() > 8192 {
        return Err(Error::Invalid("CQT needs 1 to 8192 frequency centres"));
    }
    let mut groups: BTreeMap<usize, Vec<Bin>> = BTreeMap::new();
    let mut previous = 0.0;
    for (index, &frequency) in frequencies.iter().enumerate() {
        if !frequency.is_finite()
            || frequency <= previous
            || frequency >= f64::from(sample_rate) * 0.5
        {
            return Err(Error::Invalid(
                "CQT centres must increase, be positive and be below Nyquist",
            ));
        }
        previous = frequency;
        let half = (config.support_q() * f64::from(sample_rate) / (2.0 * frequency)).ceil();
        if !half.is_finite() || half >= usize::MAX as f64 || half < 1.0 {
            return Err(Error::Resource("full support length is not representable"));
        }
        let half = half as usize;
        let taps = half
            .checked_mul(2)
            .and_then(|x| x.checked_add(1))
            .ok_or(Error::Resource("support length overflows"))?;
        let n = taps
            .checked_mul(2)
            .and_then(usize::checked_next_power_of_two)
            .ok_or(Error::Resource("FFT length overflows"))?;
        if n > limits.max_fft_len {
            return Err(Error::Resource(
                "full support requires an FFT larger than max_fft_len",
            ));
        }
        groups.entry(n).or_default().push(Bin {
            index,
            half,
            frequency,
        });
    }
    Ok(groups)
}

fn kernel(
    bin: Bin,
    n: usize,
    sr: u32,
    window: Window,
    should_continue: &dyn Fn() -> bool,
) -> Result<Vec<Complex<f64>>, Error> {
    let mut spectrum = zeroed(n)?;
    let mut sum = 0.0;
    for i in 0..=2 * bin.half {
        if i % 4096 == 0 {
            poll(should_continue)?;
        }
        let j = i as f64 - bin.half as f64;
        sum += window.at(PI * j / bin.half as f64);
    }
    let gain = 2.0 / sum;
    for i in 0..=2 * bin.half {
        if i % 4096 == 0 {
            poll(should_continue)?;
        }
        let j = i as f64 - bin.half as f64;
        let amplitude = gain * window.at(PI * j / bin.half as f64);
        let angle = -TAU * bin.frequency * j / f64::from(sr);
        // Reverse centred correlation kernel for circular convolution.
        let at = (n + bin.half - i) % n;
        spectrum[at] = Complex::new(amplitude * angle.cos(), amplitude * angle.sin());
    }
    Ok(spectrum)
}

fn fill_input(
    target: &mut [Complex<f64>],
    source: &[f32],
    start: usize,
    half: usize,
    should_continue: &dyn Fn() -> bool,
) -> Result<(), Error> {
    for (i, value) in target.iter_mut().enumerate() {
        if i % 4096 == 0 {
            poll(should_continue)?;
        }
        let source_at = if i >= half {
            start.checked_add(i - half)
        } else {
            start.checked_sub(half - i)
        };
        let sample = source_at
            .and_then(|at| source.get(at))
            .copied()
            .unwrap_or(0.0);
        if !sample.is_finite() {
            return Err(Error::Invalid("CQT input contains a non-finite sample"));
        }
        *value = Complex::new(f64::from(sample), 0.0);
    }
    Ok(())
}

/// Analyze an entire signal with full centred windows, retaining only code
/// planes. `map` receives unit-sinusoid-amplitude magnitudes, like ASLT's
/// existing level converter. It must be a pure mapping for budget-independent
/// outputs. `on_progress` is a monotonically increasing cell-count fraction,
/// not an elapsed-time/ETA prediction. Cancellation drops partial output.
#[allow(clippy::too_many_arguments)]
pub fn analyze_input_codes(
    input: Input<'_>,
    sample_rate: u32,
    frequencies: &[f64],
    config: &Config,
    limits: Limits,
    should_continue: &dyn Fn() -> bool,
    on_progress: &dyn Fn(f32),
    map: &dyn Fn(f32) -> u16,
) -> Result<Output, Error> {
    analyze_inner(
        input,
        sample_rate,
        frequencies,
        config,
        limits,
        should_continue,
        on_progress,
        map,
        #[cfg(test)]
        None,
    )
}

// The test-only observer sees the actual production FFT values before their
// magnitudes are taken. A conjugated/wrong-sign coefficient can pass every
// real-signal magnitude check; it must not pass the complex-reference gate.
#[cfg(test)]
type CoefficientObserver<'a> = &'a dyn Fn(usize, usize, Complex<f64>, Option<Complex<f64>>);

#[allow(clippy::too_many_arguments)]
fn analyze_inner(
    input: Input<'_>,
    sample_rate: u32,
    frequencies: &[f64],
    config: &Config,
    limits: Limits,
    should_continue: &dyn Fn() -> bool,
    on_progress: &dyn Fn(f32),
    map: &dyn Fn(f32) -> u16,
    #[cfg(test)] observer: Option<CoefficientObserver<'_>>,
) -> Result<Output, Error> {
    poll(should_continue)?;
    let groups = layout(frequencies, sample_rate, config, limits)?;
    let (left, right) = match input {
        Input::Mono(signal) => (signal, None),
        Input::Joint { left, right } => {
            if left.len() != right.len() {
                return Err(Error::Invalid(
                    "CQT stereo channels must have the same length",
                ));
            }
            (left, Some(right))
        }
    };
    let frames = frame_count(left.len(), sample_rate, config.frame_rate)?;
    if frames == 0 {
        return Err(Error::Invalid("CQT input is empty"));
    }
    let bars = frequencies.len();
    let cells = checked_product(frames, bars)?;
    let channels = if right.is_some() { 2 } else { 1 };
    let planes = if right.is_some() { 3 } else { 1 };
    let output_bytes = checked_product(checked_product(cells, planes)?, size_of::<u16>())?;
    if output_bytes > limits.output_bytes {
        return Err(Error::Resource("returned code planes exceed output_bytes"));
    }
    // Check every group's obvious dense minimum before committing potentially
    // large output planes. Exact rustfft scratch is checked with that group's
    // short-lived plan below; plans are not retained across the entire bank.
    for &n in groups.keys() {
        let dense_bytes = checked_product(n, size_of::<Complex<f64>>())?;
        let minimum = checked_product(dense_bytes, channels * 2 + 1)?;
        if minimum > limits.workspace_bytes {
            return Err(Error::Resource(
                "one full kernel and source/product buffers exceed workspace_bytes",
            ));
        }
    }
    let mut codes = [zeroed(cells)?, Vec::new(), Vec::new()];
    if right.is_some() {
        codes[1] = zeroed(cells)?;
        codes[2] = zeroed(cells)?;
    }
    let mut work = WorkStats {
        output_bytes,
        ..WorkStats::default()
    };
    let mut completed = 0usize;
    on_progress(0.0);
    for (n, bins) in groups {
        poll(should_continue)?;
        // Refuse the obvious minimum before asking rustfft to allocate a plan.
        let dense_bytes = checked_product(n, size_of::<Complex<f64>>())?;
        let base_vectors = channels * 2;
        let minimum = checked_product(dense_bytes, base_vectors + 1)?;
        if minimum > limits.workspace_bytes {
            return Err(Error::Resource(
                "one full kernel and source/product buffers exceed workspace_bytes",
            ));
        }
        // The planner and its cached plans die at the end of this one size.
        let mut planner = FftPlanner::<f64>::new();
        let forward = planner.plan_fft_forward(n);
        let inverse = planner.plan_fft_inverse(n);
        let scratch_len = forward
            .get_inplace_scratch_len()
            .max(inverse.get_inplace_scratch_len());
        let base_bytes = checked_product(dense_bytes, base_vectors)?
            .checked_add(checked_product(scratch_len, size_of::<Complex<f64>>())?)
            .ok_or(Error::Resource("workspace size overflows"))?;
        let available = limits
            .workspace_bytes
            .checked_sub(base_bytes)
            .ok_or(Error::Resource("FFT scratch exceeds workspace_bytes"))?;
        let batch_size = (available / dense_bytes).min(bins.len());
        if batch_size == 0 {
            return Err(Error::Resource(
                "FFT scratch leaves no room for a full kernel",
            ));
        }
        let mut source_left = zeroed::<Complex<f64>>(n)?;
        let mut source_right = zeroed::<Complex<f64>>(if right.is_some() { n } else { 0 })?;
        let mut product_left = zeroed::<Complex<f64>>(n)?;
        let mut product_right = zeroed::<Complex<f64>>(if right.is_some() { n } else { 0 })?;
        let mut scratch = zeroed::<Complex<f64>>(scratch_len)?;
        let half = bins
            .iter()
            .map(|bin| bin.half)
            .max()
            .expect("a nonempty group");
        let payload = n - 2 * half;
        for batch in bins.chunks(batch_size) {
            poll(should_continue)?;
            work.batches += 1;
            work.peak_workspace_bytes = work
                .peak_workspace_bytes
                .max(base_bytes + batch.len() * dense_bytes);
            let mut kernels = Vec::new();
            kernels
                .try_reserve_exact(batch.len())
                .map_err(|_| Error::Allocation)?;
            for &bin in batch {
                let mut spectrum = kernel(bin, n, sample_rate, config.window, should_continue)?;
                poll(should_continue)?;
                work.kernel_ffts += 1;
                forward.process_with_scratch(&mut spectrum, &mut scratch);
                kernels.push(spectrum);
            }
            let mut first = 0;
            while first < frames {
                poll(should_continue)?;
                let start = frame_center(first, sample_rate, config.frame_rate);
                let mut end = first + 1;
                while end < frames
                    && frame_center(end, sample_rate, config.frame_rate) - start < payload
                {
                    end += 1;
                }
                fill_input(&mut source_left, left, start, half, should_continue)?;
                work.input_ffts += 1;
                forward.process_with_scratch(&mut source_left, &mut scratch);
                if let Some(right) = right {
                    fill_input(&mut source_right, right, start, half, should_continue)?;
                    work.input_ffts += 1;
                    forward.process_with_scratch(&mut source_right, &mut scratch);
                }
                for (&bin, spectrum) in batch.iter().zip(&kernels) {
                    poll(should_continue)?;
                    if first == 0 {
                        work.column_convolutions += channels as u64;
                    }
                    for i in 0..n {
                        if i % 4096 == 0 {
                            poll(should_continue)?;
                        }
                        product_left[i] = source_left[i] * spectrum[i];
                        if right.is_some() {
                            product_right[i] = source_right[i] * spectrum[i];
                        }
                    }
                    work.inverse_ffts += 1;
                    inverse.process_with_scratch(&mut product_left, &mut scratch);
                    if right.is_some() {
                        poll(should_continue)?;
                        work.inverse_ffts += 1;
                        inverse.process_with_scratch(&mut product_right, &mut scratch);
                    }
                    let norm = 1.0 / n as f64;
                    for frame in first..end {
                        let at = half + frame_center(frame, sample_rate, config.frame_rate) - start;
                        let l = product_left[at] * norm;
                        let cell = frame * bars + bin.index;
                        #[cfg(test)]
                        if let Some(observer) = observer {
                            observer(frame, bin.index, l, right.map(|_| product_right[at] * norm));
                        }
                        if right.is_some() {
                            let r = product_right[at] * norm;
                            codes[0][cell] = map(((l + r) * 0.5).norm() as f32);
                            codes[1][cell] = map(l.norm() as f32);
                            codes[2][cell] = map(r.norm() as f32);
                        } else {
                            codes[0][cell] = map(l.norm() as f32);
                        }
                    }
                    completed += end - first;
                    on_progress(completed as f32 / cells as f32);
                }
                first = end;
            }
        }
    }
    poll(should_continue)?;
    Ok(Output {
        codes,
        frames,
        bars,
        work,
    })
}

// BEGIN TEST ACCESSORS
// No baseline body above this marker is changed. These only expose its existing
// private coefficient observer to an independent candidate comparison.
#[cfg(test)]
#[allow(clippy::too_many_arguments)]
pub(super) fn analyze_observed(
    input: Input<'_>, sample_rate: u32, frequencies: &[f64], config: &Config,
    limits: Limits, map: &dyn Fn(f32) -> u16, observer: CoefficientObserver<'_>,
) -> Result<Output, Error> {
    analyze_inner(input, sample_rate, frequencies, config, limits, &|| true,
        &|_| {}, map, Some(observer))
}
