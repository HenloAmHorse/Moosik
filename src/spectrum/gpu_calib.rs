// Deciding, per machine, which block sizes are worth sending to the GPU.
//
// The threshold used to be a constant measured on one developer's GPU against
// one developer's CPU. That is not a property of the transform — it is the
// crossover point of a particular device against a particular core count, and
// on a laptop with an integrated GPU and sixteen cores it sits somewhere else,
// possibly nowhere at all. Worse, a device that is slow but *working* clears a
// hardcoded gate and never falls back, because nothing errors.
//
// So the threshold is measured here instead, in two stages:
//
//   * a synthetic probe on first run, to have a safe starting point rather than
//     a guess. It is deliberately distrusted — see `PROBE_MARGIN`;
//   * A/B sampling on real analyses afterwards, which is what actually decides.
//     Occasionally a group that would have gone to the device is run on the
//     cores instead and timed, so the two routes are compared on the same work
//     under the same conditions.
//
// The second stage exists because of a specific mistake. An isolated benchmark
// of this device measured 6.84x against eight cores; wiring the same code into
// a real analysis made it 35% *slower*. A synthetic probe alone would have
// measured that same flattering 6.84x and confidently picked a threshold that
// made every expensive preset worse. Only the real workload knows.

use std::collections::BTreeMap;
use std::path::PathBuf;
use std::sync::Mutex;

use serde::{Deserialize, Serialize};

/// Block sizes the probe considers, smallest first.
const PROBE_SIZES: [usize; 6] = [1 << 15, 1 << 16, 1 << 17, 1 << 18, 1 << 19, 1 << 20];

/// How much the probe must favour the device before it is believed.
///
/// A bare FFT benchmark measures the one phase the device is best at and none
/// of the CPU work that surrounds it in a real run. Measured here: 6.84x in
/// isolation against 1.09x end to end at the same size. Demanding a 2x win in
/// the probe keeps the starting point conservative; the A/B samples then move
/// it either way on evidence.
const PROBE_MARGIN: f64 = 2.0;

/// Paired samples needed at a size before the A/B verdict overrides the probe.
const MIN_SAMPLES: u32 = 3;

/// How much cheaper the device must be before work is routed to it. Hysteresis:
/// a route that is within a few percent is not worth the transfer or the risk.
const ADOPT_MARGIN: f64 = 0.9;

/// One block size's evidence. Times are wall-clock seconds for a whole group;
/// units are `bars × frames`, which normalises across tracks of different
/// lengths and bar counts. Kernel lengths within a size group are all close to
/// `n/2`, so bars inside one group cost roughly the same.
#[derive(Serialize, Deserialize, Default, Clone, Copy, Debug)]
struct SizeStat {
    gpu_secs: f64,
    gpu_units: f64,
    gpu_n: u32,
    cpu_secs: f64,
    cpu_units: f64,
    cpu_n: u32,
    /// Probe verdict for this size: the device beat the cores by `PROBE_MARGIN`.
    probe_ok: bool,
}

impl SizeStat {
    fn gpu_cost(&self) -> Option<f64> {
        (self.gpu_n >= MIN_SAMPLES && self.gpu_units > 0.0)
            .then(|| self.gpu_secs / self.gpu_units)
    }
    fn cpu_cost(&self) -> Option<f64> {
        (self.cpu_n >= MIN_SAMPLES && self.cpu_units > 0.0)
            .then(|| self.cpu_secs / self.cpu_units)
    }

    /// Whether this size should go to the device, and how sure we are.
    fn verdict(&self) -> (bool, bool) {
        match (self.gpu_cost(), self.cpu_cost()) {
            // Both routes measured on real work: that settles it.
            (Some(g), Some(c)) => (g < c * ADOPT_MARGIN, true),
            // Not enough evidence yet — go with the probe, keep sampling.
            _ => (self.probe_ok, false),
        }
    }
}

#[derive(Serialize, Deserialize, Default, Debug)]
pub struct Calibration {
    /// Adapter name and core count this was measured on. A machine change
    /// invalidates everything, because everything here is a ratio between them.
    adapter: String,
    cores: usize,
    probed: bool,
    sizes: BTreeMap<usize, SizeStat>,
}

fn path() -> PathBuf {
    super::home_dir().join(".moosik").join("gpu_calibration.json")
}

static STATE: Mutex<Option<Calibration>> = Mutex::new(None);

fn load_or_default(adapter: &str, cores: usize) -> Calibration {
    let fresh = || Calibration {
        adapter: adapter.to_string(), cores, probed: false, sizes: BTreeMap::new(),
    };
    let Ok(text) = std::fs::read_to_string(path()) else { return fresh() };
    match serde_json::from_str::<Calibration>(&text) {
        // Same machine: keep what was learned. Anything else starts over rather
        // than inheriting a ratio measured against different hardware.
        Ok(c) if c.adapter == adapter && c.cores == cores => c,
        _ => fresh(),
    }
}

fn save(c: &Calibration) {
    let p = path();
    if let Some(dir) = p.parent() { let _ = std::fs::create_dir_all(dir); }
    if let Ok(text) = serde_json::to_string_pretty(c) {
        let _ = std::fs::write(&p, text);
    }
}

/// Run the calibration once per process, loading from disk if a previous run
/// already did the probe on this machine.
fn with<R>(f: impl FnOnce(&mut Calibration) -> R) -> Option<R> {
    let adapter = super::gpu::GpuFft::describe()?;
    let cores = std::thread::available_parallelism().map(|n| n.get()).unwrap_or(1);
    let mut guard = STATE.lock().ok()?;
    if guard.is_none() {
        let mut c = load_or_default(&adapter, cores);
        if !c.probed {
            probe(&mut c);
            c.probed = true;
            save(&c);
        }
        *guard = Some(c);
    }
    Some(f(guard.as_mut()?))
}

/// Time the *convolution pipeline* on the device against the same work across
/// the cores. Only ever run once per machine.
///
/// It has to be the pipeline and not a bare FFT. A bare-FFT version of this was
/// written first and rejected every size on a card measured winning the real
/// work by 1.09–1.23x, because the device's advantage lives in the multiply and
/// the inverse it does *while the data is already there* — a forward transform
/// alone is the one part where the transfer dominates. Timing the easy proxy
/// and trusting it is the exact error this whole calibration exists to stop
/// making.
fn probe(c: &mut Calibration) {
    use super::aslt::{fft_block_len, frame_count, hop_for_fps, Morlet, SignalBlocks};
    use rayon::prelude::*;
    use std::time::Instant;

    let Some(g) = super::gpu::GpuFft::shared() else { return };

    // Short enough that a first run is not noticeably delayed, long enough that
    // the largest kernels are small relative to the signal, as on a real track.
    // This is charged once per machine, before the first analysis, so it is
    // kept to a couple of seconds rather than made maximally precise — the A/B
    // sampling that follows is the accurate part.
    let sr = 48_000u32;
    let sig: Vec<f32> = (0..(2.0 * sr as f32) as usize)
        .map(|i| (i as f32 * 0.0007).sin() * 0.5 + (i as f32 * 0.013).sin() * 0.3)
        .collect();
    let hop = hop_for_fps(sr, 180.0);
    let frames = frame_count(sig.len(), hop);

    for n in PROBE_SIZES {
        // Enough kernels at this size to look like a real chunk — a whole
        // preset puts dozens on some sizes, and a starved device measures as a
        // slow one.
        let mut mors: Vec<Morlet> = Vec::new();
        let mut cycles = 6.0f32;
        let base_f = (sr as f32 / (n as f32 / 64.0)).clamp(20.0, 8_000.0);
        while mors.len() < 12 && cycles < 6_000.0 {
            let m = Morlet::new(base_f, cycles, sr as f32);
            if fft_block_len(m.taps()) == n && m.taps() < sig.len() { mors.push(m); }
            cycles *= 1.06;
        }
        if mors.len() < 4 { continue }

        let blocks = SignalBlocks::build(&sig, n);
        let t = Instant::now();
        let cpu: Vec<Vec<f32>> = mors.par_iter()
            .map(|m| m.magnitudes_via_shared_for_test(&sig, &blocks, hop, frames)
                      .unwrap_or_default())
            .collect();
        let cpu_secs = t.elapsed().as_secs_f64();
        if cpu.iter().any(|v| v.is_empty()) { continue }

        let kernels: Vec<super::gpu::GpuKernel> = mors.iter()
            .map(|m| super::gpu::GpuKernel {
                re: m.re_taps(), im: m.im_taps(), half: m.half_width(),
            })
            .collect();
        let Ok(prep) = g.prepare_signal(&sig, n, (n / 2).max(1)) else { continue };
        // Warm: the first dispatch pays for pipeline and allocation setup that
        // a real run never repeats, and charging it here is how a fast device
        // measures as a slow one.
        let _ = g.convolve_with(&prep, sig.len(), hop, frames, &kernels[..1]);
        let t = Instant::now();
        let got = g.convolve_with(&prep, sig.len(), hop, frames, &kernels);
        let gpu_secs = t.elapsed().as_secs_f64();
        if got.is_err() { continue }

        let e = c.sizes.entry(n).or_default();
        e.probe_ok = gpu_secs > 0.0 && gpu_secs * PROBE_MARGIN < cpu_secs;
    }
}

// ---------------------------------------------------------------------------
// Worker threads for the pre-process
// ---------------------------------------------------------------------------

/// How many cores the analysis is allowed, and the pool that enforces it.
///
/// A dedicated pool rather than the global one, for two reasons: the global pool
/// can only be sized once per process (`build_global`), so a slider could not
/// take effect until a restart; and `install` makes this pool current for the
/// closure, so every nested `par_iter` inside the transform inherits the limit
/// without a single call site having to know about it.
static POOL: Mutex<Option<std::sync::Arc<rayon::ThreadPool>>> = Mutex::new(None);
static WORKERS: std::sync::atomic::AtomicUsize = std::sync::atomic::AtomicUsize::new(0);

/// Cores available, and the default the slider starts at.
pub fn core_count() -> usize {
    std::thread::available_parallelism().map(|n| n.get()).unwrap_or(4)
}

/// Two fewer than the machine has.
///
/// The output thread has a hard deadline and the decoder feeds it; leaving the
/// scheduler somewhere to put them that isn't "preempt a worker mid-FFT" is
/// worth more than the last few percent of analysis throughput. The analyser
/// scales sub-linearly at the top end anyway.
pub fn default_workers() -> usize { core_count().saturating_sub(2).max(1) }

/// Set the analysis thread budget. 0 means "the default".
pub fn set_workers(n: usize) {
    use std::sync::atomic::Ordering as O;
    let n = if n == 0 { default_workers() } else { n.clamp(1, core_count()) };
    if WORKERS.swap(n, O::Relaxed) != n
        && let Ok(mut g) = POOL.lock()
    {
        // Dropped here, but rayon keeps the old pool alive until its threads
        // finish, so a resize during an analysis is safe — it applies to the
        // next one.
        *g = None;
    }
}

pub fn workers() -> usize {
    use std::sync::atomic::Ordering as O;
    match WORKERS.load(O::Relaxed) { 0 => default_workers(), n => n }
}

/// Run `f` on the analysis pool, so it and everything it spawns are held to the
/// thread budget.
pub fn install<R: Send>(f: impl FnOnce() -> R + Send) -> R {
    let pool = {
        let Ok(mut g) = POOL.lock() else { return f() };
        if g.is_none() {
            *g = rayon::ThreadPoolBuilder::new()
                .num_threads(workers())
                .thread_name(|i| format!("moosik-analyse-{i}"))
                .build()
                .ok()
                .map(std::sync::Arc::new);
        }
        g.clone()
    };
    // A pool that would not build is no reason to refuse to analyse: the global
    // pool is capped too, it just cannot be resized without a restart.
    match pool {
        Some(p) => p.install(f),
        None => f(),
    }
}

/// What the user has asked for, overriding what the measurements say.
#[derive(Serialize, Deserialize, Clone, Copy, PartialEq, Eq, Debug, Default)]
pub enum GpuMode {
    /// Follow the calibration — the measured answer for this machine.
    #[default]
    Auto,
    /// Send every eligible block size to the device regardless of measurements.
    /// Useful for seeing the difference; not necessarily faster.
    Always,
    Off,
}

static MODE: std::sync::atomic::AtomicU8 = std::sync::atomic::AtomicU8::new(0);

pub fn set_mode(m: GpuMode) {
    use std::sync::atomic::Ordering as O;
    MODE.store(match m { GpuMode::Auto => 0, GpuMode::Always => 1, GpuMode::Off => 2 }, O::Relaxed);
}

pub fn mode() -> GpuMode {
    use std::sync::atomic::Ordering as O;
    match MODE.load(O::Relaxed) {
        1 => GpuMode::Always,
        2 => GpuMode::Off,
        _ => GpuMode::Auto,
    }
}

/// Throw away everything measured on this machine and probe again.
///
/// The probe takes a few seconds and holds the calibration lock, so callers run
/// it off the UI thread.
pub fn recalibrate() {
    let _ = std::fs::remove_file(path());
    if let Ok(mut g) = STATE.lock() { *g = None; }
    // Re-entering `with` performs the probe, since `probed` is now false.
    let _ = with(|_| ());
}

/// True while a device exists at all — the checkbox is meaningless without one.
pub fn device_present() -> bool {
    super::gpu::GpuFft::describe().is_some()
}

/// Whether this group's block size should go to the device.
///
/// Until a size is settled this alternates, so both routes get timed on real
/// work. That matters more than it sounds: an earlier version let the probe
/// *veto* the device, and since a vetoed size then never ran on the GPU, no
/// contradicting evidence could ever appear. The probe is a starting
/// preference, never a verdict — it only decides which route is sampled first,
/// so a weak device costs a couple of groups rather than a whole analysis.
// The test build routes every eligible size to the device for determinism, so
// nothing calls this there.
#[cfg_attr(test, allow(dead_code))]
pub fn use_device(n: usize) -> bool {
    match mode() {
        GpuMode::Off => return false,
        // Still bounded by the floor below: forcing a 4096-point block onto the
        // device is not a preference anyone can hold usefully, it is just slower.
        GpuMode::Always | GpuMode::Auto => {}
    }
    // Below the smallest size the probe considers, the device has never been
    // close on any hardware measured, and sampling it would cost real time to
    // re-learn that every machine.
    let Some(&floor) = PROBE_SIZES.first() else { return false };
    if n < floor { return false }
    if mode() == GpuMode::Always { return true }

    with(|c| {
        let s = c.sizes.entry(n).or_default();
        let (use_gpu, settled) = s.verdict();
        if settled { return use_gpu }
        // Unsettled: fill whichever side is short of samples, starting with the
        // one the probe favours.
        match (s.gpu_n < MIN_SAMPLES, s.cpu_n < MIN_SAMPLES) {
            (true, true) => s.probe_ok,
            (true, false) => true,
            (false, true) => false,
            (false, false) => use_gpu,
        }
    })
    .unwrap_or(false)
}

/// Groups smaller than this are not evidence of anything.
///
/// A real track is tens of millions of `bars × frames` — 1024 bars over a five
/// minute file is about 55M. Anything this far below that is dominated by fixed
/// costs (planning, staging, pool wake-up) rather than throughput, so timing it
/// measures overhead and calls it a verdict. It also keeps the test suite, which
/// analyses fractions of a second of synthetic tone and deliberately forces
/// thresholds it would never use in earnest, out of the persistent record.
const MIN_UNITS: f64 = 1.0e6;

/// Record what a group actually cost. `units` is `bars × frames`.
pub fn record(n: usize, units: f64, secs: f64, used_gpu: bool) {
    // The test build forces every qualifying group onto the device, so its
    // timings are all one-sided — recording them would teach this machine that
    // the route it never compared is the better one. The benchmark is large
    // enough to clear `MIN_UNITS`, so the size guard alone does not cover it.
    if cfg!(test) { return }
    if units < MIN_UNITS || secs <= 0.0 { return }
    let _ = with(|c| {
        let s = c.sizes.entry(n).or_default();
        if used_gpu {
            s.gpu_secs += secs; s.gpu_units += units; s.gpu_n += 1;
        } else {
            s.cpu_secs += secs; s.cpu_units += units; s.cpu_n += 1;
        }
        save(c);
    });
}

/// Human-readable state, for the debug panel.
pub fn describe() -> Vec<String> {
    with(|c| {
        let mut out = Vec::new();
        for (&n, s) in &c.sizes {
            let (use_gpu, settled) = s.verdict();
            let ratio = match (s.gpu_cost(), s.cpu_cost()) {
                (Some(g), Some(cc)) if g > 0.0 => format!("{:.2}× cores", cc / g),
                _ => "—".into(),
            };
            out.push(format!(
                "  2^{:<2} {:<8} {:>10}  gpu {}/cpu {}{}",
                n.trailing_zeros(),
                if use_gpu { "device" } else { "cores" },
                ratio, s.gpu_n, s.cpu_n,
                if settled { "" } else { " (probe)" },
            ));
        }
        if out.is_empty() { out.push("  (no calibration yet)".into()); }
        out
    })
    .unwrap_or_else(|| vec!["  (no device)".into()])
}

#[cfg(test)]
mod tests {
    use super::*;

    /// What the probe says on this machine. Diagnostic, not a pass/fail: the
    /// answer is hardware, and the point is to be able to see it.
    #[test]
    #[ignore = "hardware survey — run explicitly"]
    fn probe_survey() {
        let mut c = Calibration::default();
        probe(&mut c);
        if c.sizes.is_empty() {
            println!("no device, or every size failed to run");
            return;
        }
        println!("probe (device must win by {PROBE_MARGIN}x to be preferred):");
        for (&n, s) in &c.sizes {
            println!("  2^{:<2}  prefer_device={}", n.trailing_zeros(), s.probe_ok);
        }
        assert!(
            c.sizes.values().any(|s| s.probe_ok),
            "no size preferred the device — if this machine has a discrete GPU, \
             the probe is measuring the wrong thing again",
        );
    }

    /// The decision rule, without touching a device or the disk. Real work
    /// outranks the probe, and only once there is enough of it.
    #[test]
    fn real_samples_override_the_probe_once_there_are_enough() {
        // Probe said yes, no real evidence yet → follow the probe, unsettled.
        let mut s = SizeStat { probe_ok: true, ..Default::default() };
        assert_eq!(s.verdict(), (true, false));

        // Real work, but only on one side: still the probe's call.
        s.gpu_secs = 1.0; s.gpu_units = 1.0; s.gpu_n = MIN_SAMPLES;
        assert_eq!(s.verdict(), (true, false));

        // Both sides measured, device genuinely cheaper → settled, device.
        s.cpu_secs = 2.0; s.cpu_units = 1.0; s.cpu_n = MIN_SAMPLES;
        assert_eq!(s.verdict(), (true, true));

        // Same probe, but the device turns out slower on real work: the probe
        // is overruled. This is the case a hardcoded threshold cannot express,
        // and the one that makes a weak GPU safe.
        s.cpu_secs = 0.5;
        assert_eq!(s.verdict(), (false, true));

        // Too close to call → keep it on the cores (ADOPT_MARGIN).
        s.cpu_secs = 1.05;
        assert_eq!(s.verdict(), (false, true));
    }

    /// Sampling must stop once a size is settled, or every analysis would keep
    /// paying for A/B forever.
    #[test]
    fn sampling_stops_when_the_size_is_settled() {
        let settled = SizeStat {
            gpu_secs: 1.0, gpu_units: 1.0, gpu_n: MIN_SAMPLES,
            cpu_secs: 2.0, cpu_units: 1.0, cpu_n: MIN_SAMPLES,
            probe_ok: true,
        };
        let (_, is_settled) = settled.verdict();
        assert!(is_settled, "both sides measured should settle the size");

        let fresh = SizeStat { probe_ok: true, ..Default::default() };
        assert!(!fresh.verdict().1, "no evidence should stay unsettled");
    }

    /// The routing rule for an unsettled size, which is where a probe that got
    /// it wrong either self-corrects or deadlocks.
    ///
    /// An earlier version treated a negative probe as a veto: the size never
    /// ran on the device, so `gpu_n` stayed at zero, so the veto could never be
    /// contradicted. The probe on the machine this was written on happened to
    /// reject *every* size, which would have shipped the GPU permanently off on
    /// hardware measured winning by 1.09–1.23x.
    fn next_route(s: &SizeStat) -> bool {
        let (use_gpu, settled) = s.verdict();
        if settled { return use_gpu }
        match (s.gpu_n < MIN_SAMPLES, s.cpu_n < MIN_SAMPLES) {
            (true, true) => s.probe_ok,
            (true, false) => true,
            (false, true) => false,
            (false, false) => use_gpu,
        }
    }

    #[test]
    fn a_rejected_probe_still_gets_the_device_measured() {
        // Probe said no and nothing has run: start on the cores, as the probe
        // prefers — a weak device should not cost the first analysis.
        let mut s = SizeStat { probe_ok: false, ..Default::default() };
        assert!(!next_route(&s), "a negative probe should be tried on cores first");

        // Cores now have their samples. The device must get a turn regardless
        // of what the probe thought, or its verdict can never be overturned.
        s.cpu_n = MIN_SAMPLES; s.cpu_secs = 2.0; s.cpu_units = 1.0;
        assert!(next_route(&s), "a rejected size must still be measured on the device");

        // With both measured and the device actually faster, it wins — the
        // probe's rejection is now simply outvoted by evidence.
        s.gpu_n = MIN_SAMPLES; s.gpu_secs = 1.0; s.gpu_units = 1.0;
        assert!(next_route(&s), "real evidence should overturn the probe");
        assert!(s.verdict().1, "and the size should now be settled");
    }
}
