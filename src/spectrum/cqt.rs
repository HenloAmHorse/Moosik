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

#[path = "cqt/parallel.rs"]
mod parallel;
#[path = "cqt/phases.rs"]
mod phases;
pub use phases::Timings as PhaseTimings;

/// Bump when the transform, clock, gain or numerical route changes.
pub const REVISION: u32 = 1;
/// The Responsive (variable-Q) transform's own revision: a separate method
/// with its own identity, never a revision of Full CQT.
pub const VQT_REVISION: u32 = 1;
/// Accepted bass support limits for the Responsive transform, in seconds.
pub const BASS_LIMIT_RANGE: std::ops::RangeInclusive<f32> = 0.05..=10.0;

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

/// Used when selecting Responsive, never as a migration of saved limits.
pub const DEFAULT_RESPONSIVE_LIMIT_S: f32 = 0.25;

#[derive(Clone, Copy, Debug, PartialEq, Serialize, Deserialize)]
#[serde(default)]
pub struct Config {
    pub bins_per_octave: u16,
    pub window: Window,
    pub frame_rate: f32,
    /// `None`: Full CQT, each frequency's complete constant-Q support.
    /// `Some(t)`: the separate Responsive variable-Q transform (VQT,
    /// Schörkhuber et al. 2014), bandwidth `f/Q + 1/t`: support
    /// `1/(f/Q + 1/t)` seconds, always shorter than both `Q/f` and `t`. In
    /// the treble, where `Q/f` is far below `t`, it approaches Full CQT; in
    /// the bass it trades frequency selectivity for a shorter response.
    pub bass_limit_s: Option<f32>,
}

impl Default for Config {
    fn default() -> Self {
        Self {
            bins_per_octave: 48,
            window: Window::Blackman,
            frame_rate: 180.0,
            bass_limit_s: None,
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
        if let Some(t) = self.bass_limit_s
            && !(t.is_finite() && BASS_LIMIT_RANGE.contains(&t))
        {
            return Err(Error::Invalid(
                "Responsive bass limit must be finite and 0.05 to 10 s",
            ));
        }
        Ok(())
    }

    /// Support Q; changing the window changes leakage/mainlobe width even at
    /// the same value. Display oversampling does not change this Q.
    pub fn support_q(&self) -> f64 {
        1.0 / (2.0_f64.powf(1.0 / f64::from(self.bins_per_octave)) - 1.0)
    }

    /// Support in seconds, excluding the endpoint rounding: Full CQT's
    /// uncapped `Q/f`, or the Responsive transform's `1/(f/Q + 1/t)`.
    pub fn support_seconds(&self, frequency: f64) -> f64 {
        match self.bass_limit_s {
            None => self.support_q() / frequency,
            Some(t) => 1.0 / (frequency / self.support_q() + 1.0 / f64::from(t)),
        }
    }

    /// Half the support in samples, before rounding up. Full CQT keeps its
    /// original expression exactly (its kernels, groups and caches are
    /// unchanged); the Responsive transform uses its own bandwidth.
    fn half_support(&self, frequency: f64, sample_rate: u32) -> f64 {
        match self.bass_limit_s {
            None => self.support_q() * f64::from(sample_rate) / (2.0 * frequency),
            Some(t) => {
                f64::from(sample_rate)
                    / (2.0 * (frequency / self.support_q() + 1.0 / f64::from(t)))
            }
        }
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

#[derive(Debug, Default, Clone, Copy, PartialEq, Eq)]
pub struct WorkStats {
    /// Per-kernel, per-channel column orchestrations actually entered. This
    /// remains distinct from the FFT calls used to produce each column.
    pub column_convolutions: u64,
    /// Forward kernel transforms, shared by left and right.
    pub kernel_ffts: u64,
    /// Actual source forward transforms, counted per channel and attempt.
    pub input_ffts: u64,
    /// Actual inverse transform calls (whole or requested-output), per channel
    /// and attempt. This is not a count of dense transforms avoided.
    pub inverse_ffts: u64,
    pub inverses_pruned: u64,
    pub inverses_full: u64,
    /// Values requested from calls that took the pruning-capable route.
    pub pruned_output_values: u64,
    /// Pruning-capable groups kept whole because optional workspace did not
    /// fit beside one full kernel, or its optional allocation was refused.
    pub pruning_workspace_fallbacks: u64,
    pub parallel_groups: u64,
    /// Maximum allocated parallel inverse slots; zero means serial only.
    pub max_inverse_workers: usize,
    pub parallel_allocation_fallbacks: u64,
    pub batches: u64,
    /// Explicit simultaneously live dense vector payload, not RSS.
    pub peak_workspace_bytes: usize,
    pub output_bytes: usize,
    /// Opt-in coarse wall phases. Disabled by default; no cache-key meaning.
    pub phases: PhaseTimings,
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

/// Optional requested-output bookkeeping. The dense transform buffers remain
/// present, so unsupported/dense/duplicate-index cases take the original full
/// route. The source FFT, kernel FFT and plan/block geometry never change.
struct Requested {
    indices: Vec<usize>,
    work_indices: Vec<usize>,
    work_values: Vec<Complex<f64>>,
    left: Vec<Complex<f64>>,
    right: Vec<Complex<f64>>,
}

impl Requested {
    fn bytes(inv: &dyn rustfft::Fft<f64>, cap: usize, channels: usize) -> Result<usize, Error> {
        let work = inv.pruned_work_len(cap);
        let indices = cap
            .checked_add(checked_product(2, work)?)
            .ok_or(Error::Resource("requested-output indices overflow"))?;
        let values = checked_product(channels, cap)?
            .checked_add(work)
            .ok_or(Error::Resource("requested-output values overflow"))?;
        checked_product(indices, size_of::<usize>())?
            .checked_add(checked_product(values, size_of::<Complex<f64>>())?)
            .ok_or(Error::Resource("requested-output workspace overflows"))
    }

    fn new(inv: &dyn rustfft::Fft<f64>, cap: usize, channels: usize) -> Result<Self, Error> {
        let work = inv.pruned_work_len(cap);
        Ok(Self {
            indices: zeroed(cap)?,
            work_indices: zeroed(checked_product(2, work)?)?,
            work_values: zeroed(work)?,
            left: zeroed(cap)?,
            right: zeroed(if channels == 2 { cap } else { 0 })?,
        })
    }

    /// `false` preserves the original route; the pruning API requires strictly
    /// increasing indices, and dense coverage is not worth its bookkeeping.
    fn prepare(
        &mut self,
        first: usize,
        end: usize,
        sr: u32,
        fps: f32,
        half: usize,
        start: usize,
    ) -> bool {
        let count = end - first;
        if count > self.indices.len() {
            return false;
        }
        for (slot, frame) in self.indices[..count].iter_mut().zip(first..end) {
            *slot = half + frame_center(frame, sr, fps) - start;
        }
        self.indices[..count]
            .windows(2)
            .all(|pair| pair[0] < pair[1])
    }

    #[allow(clippy::too_many_arguments)]
    fn outputs(
        &mut self,
        inv: &dyn rustfft::Fft<f64>,
        source: &[Complex<f64>],
        kernel: &[Complex<f64>],
        product: &mut [Complex<f64>],
        scratch: &mut [Complex<f64>],
        count: usize,
        right: bool,
    ) {
        let out = if right {
            &mut self.right[..count]
        } else {
            &mut self.left[..count]
        };
        inv.process_pruned_product_with_scratch(
            source,
            kernel,
            product,
            scratch,
            &self.indices[..count],
            out,
            rustfft::PrunedWork {
                idx: &mut self.work_indices,
                val: &mut self.work_values,
            },
        );
    }
}

/// The sparse request limit is also an allocation cap. A pathological frame
/// rate greater than the sample rate cannot request billions of work slots.
fn requested_cap(payload: usize, n: usize, frames: usize, sr: u32, fps: f32) -> usize {
    let clock_bound = (payload as f64 * f64::from(fps) / f64::from(sr)).ceil() as usize;
    clock_bound.saturating_add(1).min(frames).min(n / 4)
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
        let half = config.half_support(frequency, sample_rate).ceil();
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
    let mut spectrum = zeroed::<Complex<f64>>(n)?;
    let mut sum = 0.0;
    for i in 0..=2 * bin.half {
        if i % 4096 == 0 {
            poll(should_continue)?;
        }
        let j = i as f64 - bin.half as f64;
        let w = window.at(PI * j / bin.half as f64);
        sum += w;
        // Reuse the already charged spectrum storage; the second pass uses
        // the very same f64 window value, not another transcendental call.
        let at = (n + bin.half - i) % n;
        spectrum[at].re = w;
    }
    let gain = 2.0 / sum;
    for i in 0..=2 * bin.half {
        if i % 4096 == 0 {
            poll(should_continue)?;
        }
        let j = i as f64 - bin.half as f64;
        let at = (n + bin.half - i) % n;
        let amplitude = gain * spectrum[at].re;
        let angle = -TAU * bin.frequency * j / f64::from(sr);
        // Reverse centred correlation kernel for circular convolution.
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
    should_continue: &(dyn Fn() -> bool + Sync),
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
        #[cfg(test)]
        false,
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
    should_continue: &(dyn Fn() -> bool + Sync),
    on_progress: &dyn Fn(f32),
    map: &dyn Fn(f32) -> u16,
    #[cfg(test)] observer: Option<CoefficientObserver<'_>>,
    #[cfg(test)] force_full_inverse: bool,
) -> Result<Output, Error> {
    // Resolve once per run, not once per callback/task and not a static cache:
    // an explicit diagnostic run can be enabled without changing the analysis.
    let phase_timing = std::env::var_os("MOOSIK_CQT_PHASES").is_some_and(|v| v == "1");
    analyze_inner_profiled(
        input, sample_rate, frequencies, config, limits, should_continue, on_progress, map,
        #[cfg(test)] observer,
        #[cfg(test)] force_full_inverse,
        phase_timing,
    )
}

#[allow(clippy::too_many_arguments)]
fn analyze_inner_profiled(
    input: Input<'_>,
    sample_rate: u32,
    frequencies: &[f64],
    config: &Config,
    limits: Limits,
    should_continue: &(dyn Fn() -> bool + Sync),
    on_progress: &dyn Fn(f32),
    map: &dyn Fn(f32) -> u16,
    #[cfg(test)] observer: Option<CoefficientObserver<'_>>,
    #[cfg(test)] force_full_inverse: bool,
    phase_timing: bool,
) -> Result<Output, Error> {
    let mut profile = phases::Profile::new(phase_timing);
    let phase = profile.clock();
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
    profile.record(phases::Region::Setup, phase);
    let mut completed = 0usize;
    on_progress(0.0);
    for (n, bins) in groups {
        let group_phase = profile.begin_group(n, bins.len(), work);
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
        let phase = profile.clock();
        let mut planner = FftPlanner::<f64>::new();
        let forward = planner.plan_fft_forward(n);
        let inverse = planner.plan_fft_inverse(n);
        let scratch_len = forward
            .get_inplace_scratch_len()
            .max(inverse.get_inplace_scratch_len());
        let base_bytes = checked_product(dense_bytes, base_vectors)?
            .checked_add(checked_product(scratch_len, size_of::<Complex<f64>>())?)
            .ok_or(Error::Resource("workspace size overflows"))?;
        let half = bins
            .iter()
            .map(|bin| bin.half)
            .max()
            .expect("a nonempty group");
        let payload = n - 2 * half;
        profile.record(phases::Region::Planning, phase);
        #[cfg(test)]
        let allow_pruning = !force_full_inverse;
        #[cfg(not(test))]
        let allow_pruning = true;
        // Optional parallelism uses only an explicitly current analysis pool.
        // Allocate this *instead of* serial buffers, never beside them.
        let phase = profile.clock();
        let parallel_plan = if allow_pruning {
            parallel::Plan::for_group(
                n,
                channels,
                bins.len(),
                frames,
                half,
                sample_rate,
                config.frame_rate,
                scratch_len,
                &*inverse,
                limits,
                true,
            )
        } else { None };
        profile.record(phases::Region::Planning, phase);
        if let Some(plan) = parallel_plan {
            let phase = profile.clock();
            let workspace = parallel::Workspace::new(plan, &*inverse);
            profile.record(phases::Region::Allocation, phase);
            match workspace {
                Ok(mut workspace) => {
                    workspace.set_phase_timing(phase_timing);
                    work.parallel_groups += 1;
                    work.max_inverse_workers = work.max_inverse_workers.max(plan.slots);
                    work.peak_workspace_bytes = work.peak_workspace_bytes.max(plan.workspace_bytes);
                    for batch in bins.chunks(plan.batch_size) {
                        poll(should_continue)?;
                        work.batches += 1;
                        let phase = profile.clock();
                        let (kernels, entered) = workspace.prepare_kernels(
                            batch, &*forward, config.window, should_continue,
                        )?;
                        work.kernel_ffts += entered;
                        profile.record(phases::Region::Kernel, phase);
                        let mut first = 0;
                        while first < frames {
                            poll(should_continue)?;
                            let start = frame_center(first, sample_rate, config.frame_rate);
                            let mut end = first + 1;
                            while end < frames
                                && frame_center(end, sample_rate, config.frame_rate) - start
                                    < payload
                            {
                                end += 1;
                            }
                            let phase = profile.clock();
                            {
                                let (source_left, source_right, scratch) =
                                    workspace.source_buffers();
                                fill_input(source_left, left, start, half, should_continue)?;
                                work.input_ffts += 1;
                                forward.process_with_scratch(source_left, scratch);
                                if let Some(right) = right {
                                    fill_input(source_right, right, start, half, should_continue)?;
                                    work.input_ffts += 1;
                                    forward.process_with_scratch(source_right, scratch);
                                }
                            }
                            profile.record(phases::Region::Source, phase);
                            let phase = profile.clock();
                            let stats = workspace.process_block(
                                &kernels,
                                &*inverse,
                                parallel::Block {
                                    first,
                                    end,
                                    sample_rate,
                                    frame_rate: config.frame_rate,
                                    half,
                                    start,
                                },
                                should_continue,
                            )?;
                            profile.record(phases::Region::ParallelInverse, phase);
                            profile.workers(stats.worker_wall_ns, stats.worker_tasks);
                            work.inverse_ffts += stats.inverse_ffts;
                            work.inverses_pruned += stats.inverses_pruned;
                            work.inverses_full += stats.inverses_full;
                            work.pruned_output_values += stats.pruned_output_values;
                            let phase = profile.clock();
                            let norm = 1.0 / n as f64;
                            for (local, bin) in batch.iter().enumerate() {
                                poll(should_continue)?;
                                if first == 0 {
                                    work.column_convolutions += channels as u64;
                                }
                                let ls = workspace.coefficients(local, 0, end - first);
                                let rs =
                                    right.map(|_| workspace.coefficients(local, 1, end - first));
                                for frame in first..end {
                                    let l = ls[frame - first] * norm;
                                    let r = rs.map(|values| values[frame - first] * norm);
                                    let cell = frame * bars + bin.index;
                                    #[cfg(test)]
                                    if let Some(observer) = observer {
                                        observer(frame, bin.index, l, r);
                                    }
                                    if let Some(r) = r {
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
                            profile.record(phases::Region::ParallelFold, phase);
                            first = end;
                        }
                    }
                    profile.finish_group(group_phase, plan.slots, plan.batch_size, plan.workspace_bytes, work);
                    continue;
                }
                Err(Error::Allocation) => {
                    work.parallel_allocation_fallbacks += 1;
                }
                Err(error) => return Err(error),
            }
        }
        let phase = profile.clock();
        let mut requested = None;
        let mut requested_bytes = 0;
        if allow_pruning && inverse.prunes() {
            let cap = requested_cap(payload, n, frames, sample_rate, config.frame_rate);
            let bytes = Requested::bytes(&*inverse, cap, channels)?;
            // Optional optimization: do not reject a request the old dense
            // route admitted merely to make room for bookkeeping.
            let admitted = base_bytes
                .checked_add(dense_bytes)
                .and_then(|minimum| minimum.checked_add(bytes))
                .is_some_and(|minimum| minimum <= limits.workspace_bytes);
            if admitted {
                if let Ok(buffers) = Requested::new(&*inverse, cap, channels) {
                    requested = Some(buffers);
                    requested_bytes = bytes;
                } else {
                    work.pruning_workspace_fallbacks += 1;
                }
            } else {
                work.pruning_workspace_fallbacks += 1;
            }
        }
        let base_bytes = base_bytes
            .checked_add(requested_bytes)
            .ok_or(Error::Resource("requested-output workspace overflows"))?;
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
        profile.record(phases::Region::Allocation, phase);
        for batch in bins.chunks(batch_size) {
            poll(should_continue)?;
            work.batches += 1;
            work.peak_workspace_bytes = work
                .peak_workspace_bytes
                .max(base_bytes + batch.len() * dense_bytes);
            let phase = profile.clock();
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
            profile.record(phases::Region::Kernel, phase);
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
                let prune = requested.as_mut().is_some_and(|buffers| {
                    buffers.prepare(first, end, sample_rate, config.frame_rate, half, start)
                });
                let phase = profile.clock();
                fill_input(&mut source_left, left, start, half, should_continue)?;
                work.input_ffts += 1;
                forward.process_with_scratch(&mut source_left, &mut scratch);
                if let Some(right) = right {
                    fill_input(&mut source_right, right, start, half, should_continue)?;
                    work.input_ffts += 1;
                    forward.process_with_scratch(&mut source_right, &mut scratch);
                }
                profile.record(phases::Region::Source, phase);
                let phase = profile.clock();
                for (&bin, spectrum) in batch.iter().zip(&kernels) {
                    poll(should_continue)?;
                    if first == 0 {
                        work.column_convolutions += channels as u64;
                    }
                    if prune {
                        let buffers = requested.as_mut().expect("a prepared request");
                        work.inverse_ffts += 1;
                        work.inverses_pruned += 1;
                        work.pruned_output_values += (end - first) as u64;
                        buffers.outputs(
                            &*inverse,
                            &source_left,
                            spectrum,
                            &mut product_left,
                            &mut scratch,
                            end - first,
                            false,
                        );
                        if right.is_some() {
                            poll(should_continue)?;
                            work.inverse_ffts += 1;
                            work.inverses_pruned += 1;
                            work.pruned_output_values += (end - first) as u64;
                            buffers.outputs(
                                &*inverse,
                                &source_right,
                                spectrum,
                                &mut product_right,
                                &mut scratch,
                                end - first,
                                true,
                            );
                        }
                    } else {
                        // Frozen f9a6999 whole-product/whole-inverse semantics.
                        // Keep Mul and its operand order; this also covers
                        // unsupported plans, dense/duplicate requests and
                        // optional-workspace refusal without approximation.
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
                        work.inverses_full += 1;
                        inverse.process_with_scratch(&mut product_left, &mut scratch);
                        if right.is_some() {
                            poll(should_continue)?;
                            work.inverse_ffts += 1;
                            work.inverses_full += 1;
                            inverse.process_with_scratch(&mut product_right, &mut scratch);
                        }
                    }
                    let norm = 1.0 / n as f64;
                    for frame in first..end {
                        let at = half + frame_center(frame, sample_rate, config.frame_rate) - start;
                        let l = if prune {
                            requested.as_ref().expect("a prepared request").left[frame - first]
                        } else {
                            product_left[at]
                        } * norm;
                        let r = right.map(|_| if prune {
                            requested.as_ref().expect("a prepared request").right[frame - first]
                        } else { product_right[at] } * norm);
                        let cell = frame * bars + bin.index;
                        #[cfg(test)]
                        if let Some(observer) = observer {
                            observer(frame, bin.index, l, r);
                        }
                        if let Some(r) = r {
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
                profile.record(phases::Region::SerialCombined, phase);
                first = end;
            }
        }
        profile.finish_group(group_phase, 0, batch_size, base_bytes + batch_size * dense_bytes, work);
    }
    poll(should_continue)?;
    work.phases = profile.finish();
    profile.report(sample_rate, frames, bars, config.bins_per_octave, channels);
    Ok(Output {
        codes,
        frames,
        bars,
        work,
    })
}

#[cfg(test)]
#[path = "cqt/tests.rs"]
mod tests;

#[cfg(test)]
#[path = "cqt_baseline_tests.rs"]
mod baseline_tests;
