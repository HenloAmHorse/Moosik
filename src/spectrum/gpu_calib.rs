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
//
// # What a measurement has to be worth keeping
//
// Timing the two routes is the easy half. The hard half is refusing the
// timings that are not comparable, and there are more of those than there look
// to be: a group that asked for the device and fell back to the cores part-way
// is neither a device time nor a cores time; a run the user forced onto the
// device with Always is not evidence about what Auto should do; and a sample
// measured on a four-thread pool says nothing about a fourteen-thread one,
// because the thing being measured is a *ratio* between the device and the
// cores and one side of it just changed.
//
// So a route decision returns a handle. The handle carries the identity the
// decision was made under, is consumed exactly once when the group's cost is
// known, and releases what it reserved if the group never gets that far.
// Everything below is in service of that: an identity that includes what
// actually executed, a bounded number of attempts to settle a size, and a
// persistence layer that will not let a stale generation's delayed write land
// on top of a current one.

use std::collections::BTreeMap;
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicU64, Ordering as O};
use std::sync::Mutex;
use std::time::{Duration, Instant};

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

/// Groups smaller than this are not evidence of anything.
///
/// A real track is tens of millions of `bars × frames` — 1024 bars over a five
/// minute file is about 55M. Anything this far below that is dominated by fixed
/// costs (planning, staging, pool wake-up) rather than throughput, so timing it
/// measures overhead and calls it a verdict. It also keeps the test suite, which
/// analyses fractions of a second of synthetic tone and deliberately forces
/// thresholds it would never use in earnest, out of the persistent record.
const MIN_UNITS: f64 = 1.0e6;

/// Exploration attempts allowed at one size, under one identity.
///
/// `MIN_SAMPLES` is per side, so a size needs six clean samples to settle. This
/// is twice that: a size tolerates **six inconclusive outcomes** — a device
/// that declined, a group too small to measure, an analysis the user cancelled
/// — before it stops exploring and reports itself inconclusive.
///
/// It is deliberately *not* expressed in analyses, groups or seconds. One
/// analysis contains many groups at many sizes, and a group is not a GPU chunk;
/// a budget stated in any of those units would mean something different every
/// time it was quoted. This is a count of attempts at one size, and nothing
/// else.
///
/// It is also a separate quantity from `MIN_SAMPLES` on purpose. Successful
/// samples do not bound attempts: before this existed, a size whose groups were
/// all under `MIN_UNITS` incremented no counter and was re-explored on every
/// analysis for the life of the machine.
const MAX_ATTEMPTS: u32 = 12;

/// How long the state may sit unwritten. Writes are also flushed when a verdict
/// changes and when an analysis finishes, so this is the ceiling on how stale
/// the file can get, not the usual interval.
const SAVE_DEBOUNCE: Duration = Duration::from_secs(5);

/// The on-disk format. Bumping this discards nothing — it changes the filename
/// below, so an older binary keeps reading the file it wrote.
const SCHEMA: u32 = 1;

/// Performance-relevant producer revision.
///
/// Bumped when something changes that moves the device-against-cores ratio: the
/// transform, the routing split, the staging, the shader. **Not** bumped for a
/// release number, a UI change or a documentation commit — mirroring the
/// application version here would discard every measurement on every machine
/// for changes that cannot affect them.
///
/// History:
///   1 — first revision under this schema (Moosik 1.5.1 + batch 3).
///   2 — the CPU convolution routes got quicker (1.5.2). Measured in
///       thread-seconds on the owner's machine: the shared route by 4.8–8.8 %
///       and the own-transform route by 5.1–7.5 %. The shader is untouched.
///
///       What that does to a stored ratio is **not** uniform, and an earlier
///       draft of this note claimed it was. A cores sample is all CPU and does
///       get cheaper. A device sample is not all device: the group still builds
///       its kernels, fills its columns and runs whatever wavelets the device
///       declined on the cores, and those parts got cheaper too. So both sides
///       of a stored comparison move, by different and unmeasured amounts —
///       this campaign recorded **no** active-device case at all, so the effect
///       on the device side is inferred from which code changed, not observed.
///
///       That is a reason to invalidate the key, not a reason to trust a
///       correction. `ADOPT_MARGIN` is 0.9, so the device has to win by 10 % to
///       be adopted, and a stale figure of a few per cent sits inside that
///       margin. Worse, it would not correct itself: once a size is `Settled`,
///       only the winning route is ever run again, so the losing route's cost
///       is never re-measured.
///
///       The cost of bumping is that every machine re-explores from scratch,
///       which is the designed response to an identity change and not a defect.
///       The cost of not bumping is deciding on evidence that no longer
///       describes this program.
const PRODUCER: u32 = 2;

/// Everything that has to match for two measurements to be comparable.
///
/// One key, one set of numbers: a key that does not match is discarded, not
/// filed beside the current one. A history keyed by every identity anyone has
/// ever had would answer a question nobody asks — "what was this machine like
/// before?" — at the cost of a format that has to be migrated for ever.
#[derive(Serialize, Deserialize, Clone, PartialEq, Eq, Debug, Default)]
struct Key {
    schema: u32,
    producer: u32,
    /// Adapter name. Everything here is a ratio against this device.
    adapter: String,
    /// What the machine has.
    cores: usize,
    /// What actually ran — the size of the pool the analysis was installed on,
    /// which is not the same thing as either `cores` or the current setting.
    /// See [`exec_workers`].
    workers: usize,
}

impl Key {
    fn new(adapter: String, cores: usize, workers: usize) -> Self {
        Self { schema: SCHEMA, producer: PRODUCER, adapter, cores, workers }
    }
}

/// One block size's evidence. Times are wall-clock seconds for a whole group;
/// units are `bars × frames`, which normalises across tracks of different
/// lengths and bar counts. Kernel lengths within a size group are all close to
/// `n/2`, so bars inside one group cost roughly the same.
#[derive(Serialize, Deserialize, Default, Clone, Copy, Debug, PartialEq)]
struct SizeStat {
    gpu_secs: f64,
    gpu_units: f64,
    gpu_n: u32,
    cpu_secs: f64,
    cpu_units: f64,
    cpu_n: u32,
    /// Probe verdict for this size: the device beat the cores by `PROBE_MARGIN`.
    probe_ok: bool,
    /// Exploration attempts **started** here, successful or not. See
    /// [`MAX_ATTEMPTS`], and [`Decision`] for what counts as started.
    #[serde(default)]
    attempts: u32,
    /// Reservations taken and not yet accounted for.
    ///
    /// **Not persisted, and reset to zero on load**: an outstanding reservation
    /// belongs to work that is running, and no work survives a restart.
    ///
    /// Exhaustion is terminal only once this reaches zero. Without it the
    /// twelfth reservation would exhaust the size while the sample it is about
    /// to produce is still in flight — which is the boundary error this field
    /// exists to close.
    #[serde(skip)]
    outstanding: u32,
    /// The budget ran out before a verdict existed, and that is now permanent
    /// for this identity.
    ///
    /// Sticky on purpose. Without it, ordinary non-exploratory work keeps
    /// adding cores samples — every below-verdict group records one — and the
    /// moment `cpu_n` crosses `MIN_SAMPLES` beside a stale `gpu_n` from before
    /// exhaustion, `verdict` would announce a *measured* winner for a
    /// comparison that was abandoned. Exploration stopping has to mean it
    /// stopped. Cleared only by a reset or a new identity.
    #[serde(default)]
    exhausted: bool,
}

/// What a size's evidence adds up to.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
enum Verdict {
    /// Both routes measured on real work. This is the answer.
    Settled(bool),
    /// Not enough evidence yet. Follow the probe and keep sampling.
    Sampling(bool),
    /// The attempt budget is spent and the evidence never arrived. **This is
    /// not a measurement that the cores are faster** — it is the absence of one,
    /// and it routes to the cores because that is the route that always works,
    /// not because anything was compared.
    Inconclusive,
}

impl Verdict {
    fn to_device(self) -> bool {
        match self {
            Verdict::Settled(b) | Verdict::Sampling(b) => b,
            Verdict::Inconclusive => false,
        }
    }
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

    fn verdict(&self) -> Verdict {
        // Checked before the evidence, not after it. A size that gave up has
        // given up, and a cores sample recorded afterwards by ordinary
        // non-exploratory work must not complete a pair that exploration
        // abandoned.
        if self.exhausted {
            return Verdict::Inconclusive;
        }
        match (self.gpu_cost(), self.cpu_cost()) {
            // Both routes measured on real work: that settles it.
            (Some(g), Some(c)) => Verdict::Settled(g < c * ADOPT_MARGIN),
            _ if self.attempts >= MAX_ATTEMPTS => Verdict::Inconclusive,
            // Not enough evidence yet — go with the probe, keep sampling.
            _ => Verdict::Sampling(self.probe_ok),
        }
    }

    /// Latch exhaustion the moment the budget runs out without a verdict.
    ///
    /// Called after every change to `attempts` or to the samples, so the flag
    /// records the state at the moment exploration ended rather than being
    /// re-derived later from numbers that have since moved.
    /// Latch exhaustion only when the budget is spent **and every reservation
    /// it authorised has been accounted for**.
    ///
    /// Called from the outcome paths — `finish`, `consume_started`, `release` —
    /// and never from `decide`. Latching at reservation is wrong twice over:
    ///
    /// * it exhausts the size while the twelfth sample is still in flight, so a
    ///   run that would have settled is refused by the verdict it was about to
    ///   produce;
    /// * and it survives a refund, so an unstarted twelfth reservation that is
    ///   abandoned back to eleven leaves the size permanently exhausted.
    fn latch_exhaustion(&mut self) {
        if !self.exhausted
            && self.attempts >= MAX_ATTEMPTS
            && self.outstanding == 0
            && !(self.gpu_cost().is_some() && self.cpu_cost().is_some())
        {
            self.exhausted = true;
        }
    }

    /// Whether another exploration attempt may be reserved.
    ///
    /// The budget, not the verdict. `verdict` cannot answer this: while the
    /// last reservations are outstanding the size is deliberately not yet
    /// exhausted, and asking it would authorise a thirteenth attempt.
    fn may_reserve(&self) -> bool {
        self.attempts < MAX_ATTEMPTS
            && !matches!(self.verdict(), Verdict::Settled(_) | Verdict::Inconclusive)
    }

    /// Which route this size should be sampled on next.
    ///
    /// An earlier version treated a negative probe as a veto: the size never
    /// ran on the device, so `gpu_n` stayed at zero, so the veto could never be
    /// contradicted. The probe on the machine this was written on happened to
    /// reject *every* size, which would have shipped the GPU permanently off on
    /// hardware measured winning by 1.09–1.23x. The probe is a starting
    /// preference, never a verdict — it only decides which route is sampled
    /// first, so a weak device costs a couple of groups rather than an analysis.
    ///
    /// This is the production rule. Tests call it; they do not restate it.
    fn next_route(&self) -> bool {
        match self.verdict() {
            Verdict::Settled(b) => b,
            Verdict::Inconclusive => false,
            Verdict::Sampling(probe) => {
                // Fill whichever side is short of samples, starting with the
                // one the probe favours.
                match (self.gpu_n < MIN_SAMPLES, self.cpu_n < MIN_SAMPLES) {
                    (true, true) => probe,
                    (true, false) => true,
                    (false, true) => false,
                    (false, false) => self.verdict().to_device(),
                }
            }
        }
    }

    /// Reject a stat that cannot have come from this code.
    ///
    /// `finish` only ever adds a positive finite time to a positive finite unit
    /// count, and only ever alongside an increment. So a side with samples must
    /// have both totals strictly positive, and a side with none must have both
    /// at zero. Anything else is a file this code did not write, or one that
    /// was damaged after it did — and adopting it would divide by it.
    fn is_sane(&self) -> bool {
        let side = |n: u32, secs: f64, units: f64| {
            secs.is_finite()
                && units.is_finite()
                && if n == 0 { secs == 0.0 && units == 0.0 } else { secs > 0.0 && units > 0.0 }
        };
        side(self.gpu_n, self.gpu_secs, self.gpu_units)
            && side(self.cpu_n, self.cpu_secs, self.cpu_units)
    }
}

/// What is written to disk.
#[derive(Serialize, Deserialize, Default, Debug, Clone)]
struct Stored {
    key: Key,
    probed: bool,
    sizes: BTreeMap<usize, SizeStat>,
}

/// How a group actually ran, as opposed to how it was routed.
///
/// The distinction is the point. `gpu.is_some()` says a device was *selected*;
/// it does not say the device did the work, and the recovery sweep that runs
/// declined bars on the cores is invisible from there.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum Outcome {
    /// Selected a device, and every bar came back from it.
    Device,
    /// Never asked for a device. A clean cores time.
    Cores,
    /// Selected a device and at least one dispatch declined, so some bars were
    /// recomputed on the cores. **Neither side may have this.** Its wall time is
    /// a mixture, and charging it to either route is how a flaky device teaches
    /// a machine that the route it never cleanly ran is the better one.
    PartialFallback,
    /// Asked for a device and could not get one. Not a cores sample: the cores
    /// did the work, but only after paying for a staging attempt a genuine
    /// cores route never pays.
    Unavailable,
}

/// What became of a completed group. Returned so a test can assert on the
/// decision rather than on a side effect, and so the log can be specific.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum Landed {
    /// Trained the device side.
    Device,
    /// Trained the cores side.
    Cores,
    /// Dropped: the identity moved under it, the mode was not Auto, the group
    /// was too small, or the outcome was not a clean single-route run.
    Dropped(Dropped),
}

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum Dropped {
    /// A newer generation is current. An old pool's completion, or a sample
    /// from an analysis that started before the identity changed.
    StaleGeneration,
    /// The user forced the route. Always is not evidence about Auto.
    NotAuto,
    /// Under `MIN_UNITS`, or a time that is not a positive finite number.
    NotMeasurable,
    /// The route was not clean end to end.
    MixedRoute,
}

// ---------------------------------------------------------------------------
// The state machine
// ---------------------------------------------------------------------------

/// A routing decision, the group it belongs to, and the exploration attempt it
/// may have reserved.
///
/// # Ownership
///
/// **One decision belongs to exactly one group in exactly one execution.** It
/// is consumed once by [`Calib::finish`], or retired once when its
/// [`GroupScope`] ends — a cancelled group, an early return, an unwind, or a
/// group that never reached its recording site at all.
///
/// That last case is why the scope exists. A cancelled group skips `record`, so
/// before this the decision stayed parked on the thread; the *next* group on
/// that worker — possibly Always, Off, or a size below the routing floor, none
/// of which take a decision of their own — would then have its time recorded
/// against the abandoned group's block size, and the reservation would never
/// come back.
///
/// # Reserved, started, consumed
///
/// A *reservation* is the right to explore. It becomes a *started attempt* when
/// the group's work actually begins, and only then does it cost budget:
///
/// - dropped **before** the work started — refunded;
/// - dropped **after** the work started — consumed, because the exploration
///   really did happen and repeatedly starting and cancelling it must not buy
///   unlimited exploration;
/// - consumed by `finish` — consumed, whatever the outcome.
#[derive(Debug)]
pub struct Decision {
    n: usize,
    generation: u64,
    mode: GpuMode,
    to_device: bool,
    /// Whether this decision took an exploration attempt. A settled or
    /// exhausted size takes none: it is not exploring.
    reserved: bool,
    /// Whether the work this decision authorised actually began.
    started: bool,
    /// Set by `finish`, so retirement knows not to touch what was already
    /// accounted for.
    consumed: bool,
}

impl Drop for Decision {
    fn drop(&mut self) {
        if self.consumed || !self.reserved {
            return;
        }
        // A retirement is a lookup, never an initialization: it must not create
        // state, probe, or rekey anything. If the generation it belongs to is
        // no longer current there is nothing to give back — that set is gone.
        //
        // Never called with the state lock held: the owners are the group scope
        // and `record`, and both release the lock before dropping.
        if self.started {
            // The exploration happened. Charge it, and make sure the charge is
            // persisted rather than living only in memory.
            let _ = with_current(self.generation, |c| c.consume_started(self.n));
        } else {
            let _ = with_current(self.generation, |c| c.release(self.n));
        }
    }
}

/// The learned state for one identity, plus what it takes to persist it safely.
///
/// Held behind a mutex in production and constructed directly by tests, which
/// is the point: the transitions the tests drive are the ones production runs.
#[derive(Debug)]
struct Calib {
    stored: Stored,
    /// Bumped whenever the identity changes. Carried on every decision and
    /// checked when the sample lands, so an old pool's completion cannot train
    /// a set that has already been replaced.
    generation: u64,
    path: PathBuf,
    dirty: bool,
    last_save: Option<Instant>,
    /// The last save that failed. Kept so the state is not reset merely because
    /// a write did not land.
    save_failed: bool,
}

impl Calib {
    fn new(path: PathBuf, key: Key, generation: u64) -> Self {
        Self {
            stored: Stored { key, probed: false, sizes: BTreeMap::new() },
            generation,
            path,
            dirty: false,
            last_save: None,
            save_failed: false,
        }
    }

    /// Load the state for `key`, or start fresh.
    ///
    /// Anything that is not this exact identity starts over rather than
    /// inheriting a ratio measured against different hardware, a different
    /// producer, or a different execution pool. A truncated, corrupt or
    /// implausible file is the same case as a missing one.
    fn load(path: PathBuf, key: Key, generation: u64) -> Self {
        let fresh = |p: PathBuf| Self::new(p, key.clone(), generation);
        let Ok(text) = std::fs::read_to_string(&path) else { return fresh(path) };
        match serde_json::from_str::<Stored>(&text) {
            Ok(s) if s.key == key && s.sizes.values().all(|v| v.is_sane()) => Self {
                stored: s,
                generation,
                path,
                dirty: false,
                last_save: None,
                save_failed: false,
            },
            _ => fresh(path),
        }
    }

    /// Whether this size is eligible to be learned about at all.
    ///
    /// The floor is production's, not the probe's. `PROBE_SIZES` starts at
    /// 2^15 and `aslt`'s routing floor is 2^17, so using the probe's floor here
    /// accumulates entries at two sizes production never routes — numbers that
    /// can only mislead whoever reads the panel.
    fn eligible(n: usize) -> bool {
        n >= super::aslt::GPU_MIN_BLOCK
    }

    /// Route this group, and reserve an exploration attempt if this is one.
    fn decide(&mut self, n: usize, mode: GpuMode) -> Decision {
        let mut d = Decision {
            n,
            generation: self.generation,
            mode,
            to_device: false,
            reserved: false,
            started: false,
            consumed: false,
        };
        if mode == GpuMode::Off || !Self::eligible(n) {
            return d;
        }
        if mode == GpuMode::Always {
            // Forced, and not exploration: no attempt is taken and an exhausted
            // budget does not override it.
            d.to_device = true;
            return d;
        }
        let s = self.stored.sizes.entry(n).or_default();
        d.to_device = s.next_route();
        if s.may_reserve() {
            // Charged here, before the work, so overlapping reservations cannot
            // between them exceed the budget. Refundable until the work starts.
            //
            // **No exhaustion latch here.** The budget being spent is not the
            // same event as the exploration being over: the attempt this call
            // authorises has not produced its outcome yet, and the twelfth one
            // may be the sample that settles the size.
            s.attempts += 1;
            s.outstanding += 1;
            d.reserved = true;
            self.dirty = true;
        }
        d
    }

    /// Give back an attempt whose group never began.
    ///
    /// **Marks the state dirty.** A refund that lives only in memory is a
    /// budget that grows back across a restart, which is the opposite of a
    /// bound.
    fn release(&mut self, n: usize) {
        if let Some(s) = self.stored.sizes.get_mut(&n) {
            s.attempts = s.attempts.saturating_sub(1);
            s.outstanding = s.outstanding.saturating_sub(1);
            // The reservation is accounted for, so the budget may now be
            // judged — and it is judged against the *refunded* count, so
            // abandoning the twelfth back to eleven leaves the size explorable.
            s.latch_exhaustion();
            self.dirty = true;
        }
    }

    /// Keep an attempt whose work began and then did not finish.
    ///
    /// Nothing to change in the count — it was charged at the decision — but
    /// the exhaustion latch has to be evaluated, because this may be the
    /// attempt that spent the last of the budget.
    fn consume_started(&mut self, n: usize) {
        if let Some(s) = self.stored.sizes.get_mut(&n) {
            s.outstanding = s.outstanding.saturating_sub(1);
            s.latch_exhaustion();
            self.dirty = true;
        }
    }

    /// Account for a completed group.
    fn finish(&mut self, d: &mut Decision, units: f64, secs: f64, outcome: Outcome) -> Landed {
        d.consumed = true;
        if d.generation != self.generation {
            return Landed::Dropped(Dropped::StaleGeneration);
        }
        if d.mode != GpuMode::Auto {
            // The attempt was never taken for a forced route, so nothing to
            // give back.
            return Landed::Dropped(Dropped::NotAuto);
        }
        // Every remaining path below consumes the reservation, including the
        // ones that drop the sample: an outcome that teaches nothing is still
        // an outcome, and leaving it outstanding would hold exhaustion open
        // for ever.
        let account = |s: &mut SizeStat, reserved: bool| {
            if reserved {
                s.outstanding = s.outstanding.saturating_sub(1);
                s.latch_exhaustion();
            }
        };
        let used_gpu = match outcome {
            Outcome::Device => true,
            Outcome::Cores => false,
            // Spent its attempt and taught nothing. That is the whole reason
            // the budget exists.
            Outcome::PartialFallback | Outcome::Unavailable => {
                if let Some(s) = self.stored.sizes.get_mut(&d.n) {
                    account(s, d.reserved);
                }
                self.dirty = true;
                return Landed::Dropped(Dropped::MixedRoute);
            }
        };
        if !units.is_finite() || !secs.is_finite() || units < MIN_UNITS || secs <= 0.0 {
            if let Some(s) = self.stored.sizes.get_mut(&d.n) {
                account(s, d.reserved);
            }
            self.dirty = true;
            return Landed::Dropped(Dropped::NotMeasurable);
        }
        let Some(s) = self.stored.sizes.get_mut(&d.n) else {
            return Landed::Dropped(Dropped::NotMeasurable);
        };
        let before = s.verdict();
        if used_gpu {
            s.gpu_secs += secs;
            s.gpu_units += units;
            s.gpu_n += 1;
        } else {
            s.cpu_secs += secs;
            s.cpu_units += units;
            s.cpu_n += 1;
        }
        account(s, d.reserved);
        s.latch_exhaustion();
        let changed = s.verdict() != before;
        self.dirty = true;
        if changed {
            // A verdict change is one of the two moments a write is not
            // debounced: it is the state someone reading the panel is asking
            // about, and the state a restart would otherwise re-derive.
            self.save_now();
        }
        if used_gpu { Landed::Device } else { Landed::Cores }
    }

    /// Write if the debounce has elapsed. Called after every state change.
    fn maybe_save(&mut self) {
        if !self.dirty {
            return;
        }
        let due = self.last_save.is_none_or(|t| t.elapsed() >= SAVE_DEBOUNCE);
        if due {
            self.save_now();
        }
    }

    /// Write the state, atomically, keeping the old file if anything goes
    /// wrong.
    ///
    /// The temporary carries the process id and a nanosecond stamp because a
    /// fixed `.tmp` sibling is not enough when two instances write at once:
    /// they would interleave into one file and publish a blend of both.
    ///
    /// A rename is atomic but says nothing about *order*, so ordering is
    /// handled separately — every caller checks the generation before reaching
    /// here, and a save produced under a generation that is no longer current
    /// never gets this far.
    fn save_now(&mut self) {
        self.last_save = Some(Instant::now());
        match write_atomic_json(&self.path, &self.stored) {
            Ok(()) => {
                self.dirty = false;
                self.save_failed = false;
            }
            Err(e) => {
                // The old file is untouched, the temporary is gone, and the
                // state stays dirty so the next attempt retries it. A failed
                // write is not a reason to throw away valid measurements.
                self.save_failed = true;
                crate::mlog!("gpu     calibration could not be saved: {e}");
            }
        }
    }
}

/// Serialise, write to a unique sibling, sync, and rename over the
/// destination. Removes its own temporary on any failure.
fn write_atomic_json(path: &Path, s: &Stored) -> std::io::Result<()> {
    use std::io::Write as _;
    let text = serde_json::to_string_pretty(s).map_err(std::io::Error::other)?;
    if let Some(dir) = path.parent() {
        std::fs::create_dir_all(dir)?;
    }
    let stamp = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|d| d.as_nanos())
        .unwrap_or(0);
    let tmp = path.with_extension(format!("tmp-{}-{stamp:x}", std::process::id()));
    #[cfg(test)]
    inject::before_publish();
    #[cfg(test)]
    if inject::publish_fails() {
        return Err(std::io::Error::other("injected publication failure"));
    }
    let write = || -> std::io::Result<()> {
        let mut f = std::fs::File::create(&tmp)?;
        f.write_all(text.as_bytes())?;
        f.sync_all()
    };
    if let Err(e) = write() {
        let _ = std::fs::remove_file(&tmp);
        return Err(e);
    }
    if let Err(e) = std::fs::rename(&tmp, path) {
        let _ = std::fs::remove_file(&tmp);
        return Err(e);
    }
    Ok(())
}

// ---------------------------------------------------------------------------
// Global wiring
// ---------------------------------------------------------------------------

static STATE: Mutex<Option<Calib>> = Mutex::new(None);

/// Bumped on every identity change. Monotonic for the life of the process.
static GENERATION: AtomicU64 = AtomicU64::new(0);

/// Adapter presence, as decided once by [`warm`]: 0 unknown, 1 yes, 2 no.
///
/// Cached in an atomic rather than asked each time because the question is
/// posed from inside a repaint, and answering it the honest way means building
/// a wgpu instance and enumerating adapters — first-call-only, but that first
/// call is exactly the one a broken driver can sit on.
static DEVICE: std::sync::atomic::AtomicU8 = std::sync::atomic::AtomicU8::new(0);

/// A new file, beside the legacy one rather than over it.
///
/// The old `gpu_calibration.json` is never written and never removed. An older
/// binary still finds its own file exactly as it left it, which is the whole
/// downgrade story; putting a version field inside the old filename would force
/// every downgrade to discard and re-probe instead.
fn path() -> PathBuf {
    match test_dir() {
        Some(d) => d.join("gpu_calibration_v2.json"),
        None => super::home_dir().join(".moosik").join("gpu_calibration_v2.json"),
    }
}

#[cfg(test)]
fn test_dir() -> Option<PathBuf> {
    inject::TEST_DIR.lock().ok().and_then(|g| g.clone())
}

#[cfg(not(test))]
fn test_dir() -> Option<PathBuf> {
    None
}

/// The external dependencies a test replaces, so the schedules below drive the
/// **production** ownership, initialization, record, release, reset and
/// publication code rather than a copy of it.
///
/// What is injected and nothing more: the adapter name, the calibration
/// directory, what the probe does, whether pool construction succeeds, whether
/// recording is permitted in a test build, and two synchronization points. The
/// state machine itself is never substituted.
#[cfg(test)]
pub(super) mod inject {
    use std::path::PathBuf;
    use std::sync::atomic::{AtomicBool, Ordering as O};
    use std::sync::Mutex;

    /// Serialises the tests that touch these globals. Every schedule takes it
    /// for its whole body; without it two tests would rewrite one another's
    /// adapter, directory and live state.
    pub static ENV: Mutex<()> = Mutex::new(());

    pub static TEST_DIR: Mutex<Option<PathBuf>> = Mutex::new(None);
    /// `Some(None)` means "no adapter"; `None` means "ask the real device".
    pub static ADAPTER: Mutex<Option<Option<String>>> = Mutex::new(None);
    /// Replaces the body of `probe`. `None` leaves the real probe in place.
    pub static PROBE: Mutex<Option<Box<dyn Fn() + Send>>> = Mutex::new(None);
    /// Make `install` behave as though the pool would not build.
    pub static POOL_FAILS: AtomicBool = AtomicBool::new(false);
    /// Let `record` actually record. Off by default, because the test build
    /// forces every eligible size to the device and its timings are one-sided.
    pub static ALLOW_RECORD: AtomicBool = AtomicBool::new(false);
    /// Runs inside `save_now`, after the bytes are serialised and before they
    /// are published. The schedule for reset-against-save lives here.
    pub static BEFORE_PUBLISH: Mutex<Option<Box<dyn Fn() + Send>>> = Mutex::new(None);
    /// Make the next publication fail.
    pub static PUBLISH_FAILS: AtomicBool = AtomicBool::new(false);
    /// Runs inside `ensure_current`, after the state fast path has missed and
    /// before the initialization claim is taken. The schedule for a second
    /// initializer arriving in that window lives here.
    pub static BEFORE_CLAIM: Mutex<Option<Box<dyn Fn() + Send>>> = Mutex::new(None);
    /// Runs inside `ensure_current` in the gap between the **early** authority
    /// check releasing `STATE` and the initialization claim taking `INIT`.
    ///
    /// `BEFORE_CLAIM` cannot reach this window: it fires *before* the early
    /// check, so a newer execution that publishes there is seen by that check
    /// and the caller never reaches the claim at all. Only a hook placed here
    /// forces the schedule in which the early check passes and the **late**
    /// check is the one that has to reject.
    pub static AFTER_AUTHORITY_CHECK: Mutex<Option<Box<dyn Fn() + Send>>> =
        Mutex::new(None);

    pub fn adapter() -> Option<Option<String>> {
        ADAPTER.lock().ok().and_then(|g| g.clone())
    }
    pub fn pool_fails() -> bool {
        POOL_FAILS.load(O::Relaxed)
    }
    pub fn allow_record() -> bool {
        ALLOW_RECORD.load(O::Relaxed)
    }
    pub fn publish_fails() -> bool {
        PUBLISH_FAILS.load(O::Relaxed)
    }
    pub fn run_probe() -> bool {
        let f = PROBE.lock().ok().and_then(|g| g.as_ref().map(|_| ()));
        if f.is_none() {
            return false;
        }
        if let Ok(g) = PROBE.lock()
            && let Some(f) = g.as_ref()
        {
            f();
        }
        true
    }
    /// Fires once, then disarms: the callback re-enters `ensure_current`, and
    /// an armed hook would recurse for ever.
    pub fn before_claim() {
        let taken = BEFORE_CLAIM.lock().ok().and_then(|mut g| g.take());
        if let Some(f) = taken {
            f();
        }
    }

    /// Fires once, then disarms, for the same reason as `before_claim`.
    pub fn after_authority_check() {
        let taken = AFTER_AUTHORITY_CHECK.lock().ok().and_then(|mut g| g.take());
        if let Some(f) = taken {
            f();
        }
    }

    pub fn before_publish() {
        // The callback is cloned out from under the lock: it reaches back into
        // the calibration, and holding this while it runs would be a lock
        // ordering nobody could reason about.
        let taken = BEFORE_PUBLISH.lock().ok().and_then(|mut g| g.take());
        if let Some(f) = taken {
            f();
            if let Ok(mut g) = BEFORE_PUBLISH.lock() {
                *g = Some(f);
            }
        }
    }
}

/// The adapter this process should key on.
fn adapter_name() -> Option<String> {
    #[cfg(test)]
    if let Some(a) = inject::adapter() {
        return a;
    }
    super::gpu::GpuFft::describe()
}

/// Issues execution epochs, and remembers the newest one issued.
///
/// The worker count is not an execution identity. Two analyses on two-worker
/// pools are the same *key* and different *executions*, and an execution whose
/// state has since been replaced must not be able to put it back.
///
/// # The authority contract
///
/// **An execution may publish calibration state only while no execution newer
/// than itself has already published.** Authority is lost at the moment
/// something newer *replaces* the state, not at the moment something newer
/// merely *starts*.
///
/// Both halves of that matter:
///
/// * Losing it at replacement is what closes the defect this rule exists for.
///   `GENERATION` cannot: an execution superseded between its authority check
///   and its claim takes its generation *after* the newer one has finished, so
///   it holds the newest generation there is — it issued it — and the
///   generation test passes while its key, its measurements and its file are
///   the stale ones.
/// * *Not* losing it at mere supersession is what keeps concurrent first
///   access working. Four analyses starting together supersede one another by
///   construction; under a "only the newest may publish" rule whichever one
///   holds the claim is always stale by the time its probe returns, and
///   **nobody ever initializes**.
///
/// ## Linearization
///
/// [`LAST_PUBLISHED_EXEC`] is read and written **only while `STATE` is held**,
/// inside the same critical section that installs the state and writes the
/// file. Publication is therefore totally ordered by the `STATE` mutex, and
/// the test is against that order rather than against a separate atomic
/// protocol: for a superseded execution to win, its critical section would
/// have to come last in an order in which the newer execution's critical
/// section has already run — which is exactly what the test rejects.
///
/// The same predicate is applied under `INIT`, before the claim is taken, so a
/// stale initializer is turned away before it parks the claim across its own
/// probe. That earlier test is an optimisation; the one under `STATE` is what
/// the correctness argument rests on.
///
/// ## What `LATEST_EXEC` still decides
///
/// Whether an initialization may be *started* at all
/// ([`initializable_key`]): an execution that has already been superseded does
/// not begin creating state for its next group. It is raised with a SeqCst
/// `fetch_max` rather than a plain store, because two executions starting
/// together take their epochs from one `fetch_add` and their stores could
/// otherwise land in an order that leaves the *older* of the two recorded as
/// the newest.
static EXEC_SEQ: AtomicU64 = AtomicU64::new(0);
static LATEST_EXEC: AtomicU64 = AtomicU64::new(0);

/// The epoch of the execution that last published calibration state.
///
/// **Read and written only while `STATE` is held.** It is logically a field of
/// the live state; it lives out here because the state itself is `None`
/// between a reset and the next publication, and the ordering has to survive
/// that gap.
static LAST_PUBLISHED_EXEC: AtomicU64 = AtomicU64::new(0);

/// Whether an execution newer than `mine` has already published.
///
/// The caller must hold `STATE`. An epoch of 0 is not an execution at all and
/// may never publish.
fn superseded_by_a_publication(mine: u64) -> bool {
    mine == 0 || LAST_PUBLISHED_EXEC.load(O::SeqCst) > mine
}

/// Begin a new execution: allocate an epoch and make it the newest.
///
/// Every path that enters an execution goes through this, so supersession has
/// one implementation and one memory ordering.
fn supersede() -> u64 {
    let epoch = EXEC_SEQ.fetch_add(1, O::SeqCst) + 1;
    LATEST_EXEC.fetch_max(epoch, O::SeqCst);
    epoch
}

thread_local! {
    /// The execution this thread belongs to, or 0 outside one.
    ///
    /// Initialization authority is held by the **newest** execution only.
    /// Rejecting an old completion is not enough on its own: before this, an
    /// old analysis whose sample had just been refused would call `use_device`
    /// for its *next* group, miss the state fast path (its key no longer
    /// matches), claim initialization, and reload its own pool's file over the
    /// newer calibration. Being obsolete has to persist across the whole
    /// execution, not be re-decided per group.
    static EXEC_EPOCH: std::cell::Cell<u64> = const { std::cell::Cell::new(0) };

    /// The size of the pool this thread is executing an analysis on, or 0.
    ///
    /// Set by [`install`] around its closure. This is the execution identity:
    /// not `workers()`, which is the current *setting* and may already have
    /// moved on, and not `available_parallelism`, which is the machine. A
    /// thread reading 0 is not running an analysis and may read the calibration
    /// but must never create or train it.
    static EXEC: std::cell::Cell<usize> = const { std::cell::Cell::new(0) };
}

fn exec_workers() -> usize {
    EXEC.with(|e| e.get())
}

/// Whether this thread's execution is still the newest one started.
///
/// Only the newest may create or replace calibration state. An older execution
/// keeps running, keeps routing — to the cores, since it can read no state —
/// and records nothing. That is deliberate: losing an obsolete execution's
/// samples costs a few groups, and letting it reinstate its own key costs the
/// calibration.
fn exec_is_current() -> bool {
    let mine = EXEC_EPOCH.with(|e| e.get());
    mine != 0 && mine == LATEST_EXEC.load(O::SeqCst)
}

/// Run `f` with this thread marked as executing on a pool of `n`.
///
/// Restored through a guard rather than a trailing assignment, because a panic
/// inside `f` would otherwise leave the thread permanently claiming to be
/// executing an analysis on a pool that has gone. Rayon workers are reused, so
/// that claim would outlive the analysis and let a later unrelated caller
/// create or train calibration state under a stale identity.
fn with_exec<R>(n: usize, f: impl FnOnce() -> R) -> R {
    struct Restore(usize, u64);
    impl Drop for Restore {
        fn drop(&mut self) {
            EXEC.set(self.0);
            EXEC_EPOCH.set(self.1);
        }
    }
    let epoch = supersede();
    let _restore = Restore(EXEC.replace(n), EXEC_EPOCH.replace(epoch));
    f()
}

/// Enter an execution and return what was there, for a test that needs to hold
/// one execution open across another. Production always uses the guard.
#[cfg(test)]
fn enter_exec(n: usize) -> (usize, u64) {
    let epoch = supersede();
    (EXEC.replace(n), EXEC_EPOCH.replace(epoch))
}

#[cfg(test)]
fn leave_exec(prev: (usize, u64)) {
    EXEC.set(prev.0);
    EXEC_EPOCH.set(prev.1);
}

/// Run `f` as part of the execution already on this thread.
///
/// For work that belongs to an analysis already in progress — a nested call on
/// a stolen Rayon worker — rather than a new analysis. It does **not** issue a
/// new epoch, so it cannot hand initialization rights back to work that has
/// been superseded.
#[cfg(test)]
fn within_exec<R>(f: impl FnOnce() -> R) -> R {
    f()
}

/// Whether an initialization is in flight, and whose.
///
/// Separate from [`STATE`] so that the expensive half of initialization — the
/// probe — happens with **no calibration lock held at all**. See
/// [`ensure_current`] for why that is not a style preference.
enum Init {
    Idle,
    /// A probe is running under this generation. Nobody else starts one, and
    /// nobody waits for it either.
    Running(u64),
}

static INIT: Mutex<Init> = Mutex::new(Init::Idle);

// Lock order, everywhere, without exception: INIT then STATE. Nothing takes
// STATE and then INIT. `describe` uses `try_lock` and takes neither by force.

/// The identity this thread is executing under, or `None` if it is not
/// executing an analysis or the machine has no adapter.
fn current_key() -> Option<Key> {
    let workers = exec_workers();
    if workers == 0 {
        return None;
    }
    Some(Key::new(adapter_name()?, core_count(), workers))
}

/// The key this thread may **initialize** under, if any.
///
/// `None` off an execution thread, on a machine with no adapter, or when this
/// thread's execution has been superseded by a newer one.
fn initializable_key() -> Option<Key> {
    if !exec_is_current() {
        return None;
    }
    current_key()
}

/// Make the live state current for this thread's identity, and return the
/// generation it is current under.
///
/// # Why the probe runs outside every calibration lock
///
/// `probe` is a `par_iter` over the analysis pool, and since the probe moved
/// onto that pool it is Rayon work like any other. Rayon workers do not idle
/// while they join: a worker blocked on its own `par_iter` will take another
/// job from the queue and run it *on the same thread, inside the suspended
/// one*. If that stolen job is an analysis, it asks for the calibration — and
/// the lock it needs is held by the outer job suspended beneath it on that very
/// thread. That is a deadlock, and it is a property of the locked Rayon
/// implementation rather than an accident of timing.
///
/// This was derived from the source and the Rayon implementation. **It was not
/// observed as an application hang**, and the distinction is kept here and in
/// the report rather than upgraded into a sighting.
///
/// So: load and probe holding nothing, then publish under a generation check.
///
/// # The initialization policy, stated
///
/// Exactly one thread probes. Others do **not** wait — waiting is what would
/// reintroduce a blocking dependency into the pool — they get `None` for that
/// group and route it to the cores, which is always available and always
/// correct. So concurrent first access costs at most one probe and a handful of
/// cores-routed groups, and never a duplicate probe or a stall.
fn ensure_current() -> Option<u64> {
    let key = current_key()?;

    // Already current. The common path, and it holds STATE only for this
    // block — the braces are load-bearing. Everything below takes INIT, and
    // INIT-then-STATE is the one permitted order; holding STATE across the
    // INIT acquisition would close the cycle this function exists to open.
    {
        let g = STATE.lock().ok()?;
        if let Some(c) = g.as_ref()
            && c.stored.key == key
        {
            return Some(c.generation);
        }
    }

    // Past the fast path, so this call would have to *create* state. Only the
    // newest execution may. An obsolete one reads nothing and trains nothing.
    initializable_key()?;

    #[cfg(test)]
    inject::before_claim();

    let mine = EXEC_EPOCH.with(|e| e.get());

    // **The early authority check.** The test above ran before this thread
    // could be descheduled, and a whole newer execution can initialize, record
    // and publish in between — which is exactly the window the `BEFORE_CLAIM`
    // seam opens. Without this, a stale initializer parks the claim across its
    // own probe and then overwrites what replaced it, file included.
    //
    // A rejection here is **not** a refusal to look. If what was published in
    // the window carries this very key, it is served from the live state: a
    // lookup, not an initialization, and the same answer the fast path at the
    // top of this function would have given a moment earlier. That is how a
    // second initializer's unflushed samples survive.
    //
    // **This block releases `STATE` before the claim below takes `INIT`**, and
    // it has to: INIT-then-STATE is the only permitted order, so holding STATE
    // across the acquisition would close the very cycle this function exists to
    // open. A newer execution can therefore publish in the gap, and the check
    // at publication is what rejects that — see `AFTER_AUTHORITY_CHECK`, which
    // forces exactly that schedule.
    {
        let guard = STATE.lock().ok()?;
        if superseded_by_a_publication(mine) {
            return guard.as_ref().filter(|c| c.stored.key == key).map(|c| c.generation);
        }
    }

    #[cfg(test)]
    inject::after_authority_check();

    // Claim the right to initialize, or decline. Never wait: a competing
    // initialization sends this caller to the cores for this group rather than
    // blocking an analysis behind someone else's probe.
    let generation = {
        let mut init = INIT.lock().ok()?;
        match *init {
            Init::Running(_) => return None,
            Init::Idle => {
                let g = GENERATION.fetch_add(1, O::SeqCst) + 1;
                *init = Init::Running(g);
                g
            }
        }
    };

    // Recheck under the lock now that the claim is ours. Another initializer
    // may have published this very key while this thread was between the
    // fast-path miss and the claim, and its state may hold samples that have
    // not reached the disk - reloading the file would silently lose them.
    {
        let guard = STATE.lock().ok()?;
        if let Some(c) = guard.as_ref()
            && c.stored.key == key
        {
            let g = c.generation;
            drop(guard);
            release_claim(generation);
            return Some(g);
        }
    }

    // No lock of ours is held here. The probe may take seconds and may use the
    // whole pool.
    let mut c = Calib::load(path(), key, generation);
    if !c.stored.probed {
        probe(&mut c.stored);
        c.stored.probed = true;
        c.dirty = true;
    }

    // Publish, if this generation is still the newest one issued. Anything that
    // invalidates — a reset, a pool change, a newer initialization — bumps
    // `GENERATION` first, so the check below sees it, and it is made while
    // holding `STATE` so it is atomic with the write.
    let mut init = INIT.lock().ok()?;
    if matches!(*init, Init::Running(g) if g == generation) {
        *init = Init::Idle;
    }
    let mut guard = STATE.lock().ok()?;
    if GENERATION.load(O::SeqCst) != generation {
        // Revoked while probing. The measurements are discarded rather than
        // published over whatever replaced them.
        return None;
    }
    // **And nothing newer may have published while this one probed.**
    //
    // Tested inside the `STATE` critical section that performs both the write
    // and the save, so publication is ordered by this mutex and nothing can
    // land between the test and its effect. Before `save_now`, so a stale
    // initializer never touches the file and only then discovers it was stale.
    //
    // **This is reachable, and it is the guarantee.** The early check released
    // `STATE` before the claim took `INIT`, so a newer execution can publish in
    // that gap: it passes the early check and arrives here superseded.
    // `a_publication_superseded_after_the_early_check_is_rejected_late` forces
    // exactly that schedule through the `AFTER_AUTHORITY_CHECK` seam. The early
    // check is the optimisation — it saves a stale initializer from parking the
    // claim across a probe — and this one is what the correctness argument
    // rests on.
    if superseded_by_a_publication(mine) {
        return None;
    }
    if c.dirty {
        c.save_now();
    }
    LAST_PUBLISHED_EXEC.fetch_max(mine, O::SeqCst);
    *guard = Some(c);
    Some(generation)
}

/// Use the live state **only** if it is still the given generation.
///
/// A lookup, never an initialization. This is the accessor for completions and
/// retirements, and the distinction is the whole point: before this, `record`
/// and `Decision::drop` went through the initializing path, which formed a key
/// from the *caller's* pool and, on a mismatch, replaced the live calibration
/// with one loaded for that key — so an old two-worker analysis finishing after
/// the user moved to one worker did not merely get dropped, it **rekeyed the
/// live state back to two workers** and could re-probe while doing it.
///
/// Nothing here creates, probes, rekeys or resurrects anything.
fn with_current<R>(generation: u64, f: impl FnOnce(&mut Calib) -> R) -> Option<R> {
    let mut guard = STATE.lock().ok()?;
    let c = guard.as_mut()?;
    if c.generation != generation {
        return None;
    }
    let r = f(c);
    c.maybe_save();
    Some(r)
}

/// Give back an initialization claim without publishing anything.
///
/// The generation it was issued under is spent and is not reused; nothing is
/// invalidated, because nothing was replaced.
fn release_claim(generation: u64) {
    if let Ok(mut init) = INIT.lock()
        && matches!(*init, Init::Running(g) if g == generation)
    {
        *init = Init::Idle;
    }
}

/// Bump the generation and revoke any probe's right to publish.
///
/// `INIT` before `STATE`, and callers that also want `STATE` take it after this
/// returns — the lock order is not negotiable.
fn invalidate() {
    GENERATION.fetch_add(1, O::SeqCst);
    if let Ok(mut init) = INIT.lock() {
        *init = Init::Idle;
    }
}

/// Time the *convolution pipeline* on the device against the same work across
/// the cores. Only ever run once per identity.
///
/// It has to be the pipeline and not a bare FFT. A bare-FFT version of this was
/// written first and rejected every size on a card measured winning the real
/// work by 1.09–1.23x, because the device's advantage lives in the multiply and
/// the inverse it does *while the data is already there* — a forward transform
/// alone is the one part where the transfer dominates. Timing the easy proxy
/// and trusting it is the exact error this whole calibration exists to stop
/// making.
///
/// It runs on the analysis pool, because the identity it is seeding includes
/// that pool's size. Its CPU arm is a `par_iter` whose speed is decided by how
/// many threads it gets; probing on one pool and sampling on another compares
/// two different CPU arms and reports the difference as a device verdict.
fn probe(c: &mut Stored) {
    use super::aslt::{fft_block_len, frame_count, hop_for_fps, Morlet, SignalBlocks};
    use rayon::prelude::*;

    // A test supplies its own body so a schedule can control what the probe
    // does — how long it takes, and what it does while it is running — without
    // needing a device or a real two-second measurement.
    #[cfg(test)]
    if inject::run_probe() {
        return;
    }

    let Some(g) = super::gpu::GpuFft::shared() else { return };

    // Short enough that a first run is not noticeably delayed, long enough that
    // the largest kernels are small relative to the signal, as on a real track.
    // This is charged once per identity, before the first analysis, so it is
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

        let ok = gpu_secs > 0.0 && gpu_secs * PROBE_MARGIN < cpu_secs;
        crate::mlog!(
            "gpu     probe 2^{:<2} cores {cpu_secs:.3}s device {gpu_secs:.3}s → {}{}",
            n.trailing_zeros(),
            if ok { "device" } else { "cores" },
            if Calib::eligible(n) { "" } else { " (below the routing floor, not kept)" },
        );
        // Only sizes production actually routes are carried into the state.
        if Calib::eligible(n) {
            c.sizes.entry(n).or_default().probe_ok = ok;
        }
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
///
/// Deliberately does **not** touch the calibration state, and takes no lock
/// that a probe or a save can be holding. The new identity is not knowable
/// here: the pool is rebuilt lazily, and until it is built nothing knows how
/// many threads it will actually get. `install` does the invalidation, at the
/// first moment the answer exists.
pub fn set_workers(n: usize) {
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
    match WORKERS.load(O::Relaxed) { 0 => default_workers(), n => n }
}

/// Run `f` on the analysis pool, so it and everything it spawns are held to the
/// thread budget.
///
/// Also the one place that knows the *actual* execution-pool size — the pool
/// that was built, which is not always the pool that was asked for — so it
/// stamps that on the thread for the calibration to key on.
pub fn install<R: Send>(f: impl FnOnce() -> R + Send) -> R {
    #[cfg(test)]
    if inject::pool_fails() {
        // The construction-fallback path, reached through the real selection
        // below rather than simulated beside it: what executes is the ambient
        // pool, and that is what gets stamped.
        let n = rayon::current_num_threads().max(1);
        return with_exec(n, f);
    }
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
    // pool is capped too, it just cannot be resized without a restart. What
    // executed is then the global pool, and that is what is stamped — a
    // construction fallback must not be recorded as the size that was wanted.
    match pool {
        Some(p) => {
            let n = p.current_num_threads().max(1);
            p.install(move || with_exec(n, f))
        }
        None => {
            let n = rayon::current_num_threads().max(1);
            with_exec(n, f)
        }
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
    MODE.store(match m { GpuMode::Auto => 0, GpuMode::Always => 1, GpuMode::Off => 2 }, O::Relaxed);
}

pub fn mode() -> GpuMode {
    match MODE.load(O::Relaxed) {
        1 => GpuMode::Always,
        2 => GpuMode::Off,
        _ => GpuMode::Auto,
    }
}

/// Throw away everything measured under the current identity and probe again.
///
/// The probe takes a few seconds and holds the calibration lock, so callers run
/// it off the UI thread. The generation is bumped, so anything already in
/// flight under the old one lands as stale rather than into the new set.
///
/// Only the new file is removed. The legacy `gpu_calibration.json` is not this
/// code's to delete.
pub fn recalibrate() {
    // Order matters, and the old order was wrong. Removing the file first left
    // a window in which a save already in flight — every save happens while
    // holding `STATE` — could write the pre-reset measurements back out after
    // the deletion, and the reload that follows the reset would then find them.
    //
    // So: revoke first, so nothing in flight may publish; then take `STATE`,
    // which is the lock every save holds, and do the removal underneath it, so
    // no save can interleave with the deletion at all.
    invalidate();
    if let Ok(mut g) = STATE.lock() {
        *g = None;
        let _ = std::fs::remove_file(path());
    }
    // Re-entering under the pool performs the probe, since the state is gone
    // and the file with it.
    install(|| {
        let _ = ensure_current();
    });
}

/// True while a device exists at all — the checkbox is meaningless without one.
///
/// Reads the answer [`warm`] cached. Until that lands this reports `false`,
/// which shows the "no device" note for a moment on a cold start rather than
/// stalling the frame to find out.
pub fn device_present() -> bool {
    DEVICE.load(std::sync::atomic::Ordering::Acquire) == 1
}

/// Marks one group's ownership of one routing decision.
///
/// Held for the body of a group's iteration. On drop it retires whatever
/// decision that group parked, so the next group — on the same worker or any
/// other — starts with nothing of the previous one's.
///
/// This is what makes "one decision, one group" true rather than intended.
/// Without it a cancelled group, which skips `record` entirely, left its
/// decision on the thread; the next group to record would consume it and file
/// its own time under the abandoned group's block size, and the reservation
/// would never come back. Always, Off and below-floor groups make it worse,
/// because none of them parks a decision of its own — so they would inherit
/// whatever was left there and, through it, train Auto.
#[must_use = "the scope must outlive the group's work, or the decision retires immediately"]
pub struct GroupScope {
    /// The decision the **enclosing** group had parked, held aside for the
    /// duration of this one and put back when it ends.
    ///
    /// This is what makes the slot nesting-safe. A Rayon worker that blocks
    /// inside one analysis can pick up another and run it *inside* the first,
    /// on the same thread — the same work-stealing that produced the probe
    /// deadlock. Clearing the slot unconditionally would let the inner group
    /// retire the outer group's still-live decision, and the outer group would
    /// then record its time against nothing, or against the inner group's size.
    outer: Option<Decision>,
    // Not `Send`: a scope belongs to the thread that opened it, because the
    // decision it owns lives in that thread's slot.
    _not_send: std::marker::PhantomData<*const ()>,
}

/// Open a group. Any decision left over from a previous group is retired here,
/// not inherited.
pub fn begin_group() -> GroupScope {
    // Taken, not retired: whatever was parked belongs to an enclosing group
    // that has not finished, and it is given back when this scope ends.
    let outer = PENDING.with(|p| p.borrow_mut().take());
    GroupScope { outer, _not_send: std::marker::PhantomData }
}

impl GroupScope {
    /// The group's work is beginning. From here a reservation is a **started**
    /// attempt and is no longer refundable.
    ///
    /// Called once the group is actually going to compute something, so that a
    /// cancellation observed before any work started costs no budget, while
    /// repeatedly starting and cancelling cannot buy unlimited exploration.
    pub fn work_started(&self) {
        PENDING.with(|p| {
            if let Some(d) = p.borrow_mut().as_mut() {
                d.started = true;
            }
        });
    }
}

impl Drop for GroupScope {
    fn drop(&mut self) {
        // This group's own decision is retired; the enclosing group's is
        // restored. The two steps are separate because retiring reaches the
        // state lock, and the thread-local must not be borrowed across that.
        retire_pending();
        let outer = self.outer.take();
        if outer.is_some() {
            PENDING.with(|p| *p.borrow_mut() = outer);
        }
    }
}

/// Take whatever decision is parked on this thread and let it retire.
///
/// The take and the drop are deliberately separate statements: `Decision::drop`
/// reaches the state lock, and dropping it while the thread-local is still
/// borrowed would be a re-entrancy hazard the moment anything on that path
/// touched `PENDING` again.
fn retire_pending() {
    let d = PENDING.with(|p| p.borrow_mut().take());
    drop(d);
}

/// Whether this group's block size should go to the device, and the handle that
/// accounts for it.
///
/// The handle is parked on this thread and consumed by [`record`], or retired
/// by the enclosing [`GroupScope`]. **Call it once per group, inside a scope.**
// The test build routes every eligible size to the device for determinism, so
// nothing calls this there.
#[cfg_attr(test, allow(dead_code))]
pub fn use_device(n: usize) -> bool {
    if let Some(forced) = forced_route(mode(), n) {
        // A forced route parks nothing — and the scope has already cleared
        // whatever the previous group left, so there is nothing here to
        // inherit and nothing for a later `record` to consume.
        return forced;
    }
    let d = ensure_current().and_then(|g| with_current(g, |c| c.decide(n, GpuMode::Auto)));
    let to_device = d.as_ref().is_some_and(|d| d.to_device);
    // Assigned after the locks are released: replacing the slot drops any
    // previous decision, and retirement takes the state lock.
    let previous = PENDING.with(|p| std::mem::replace(&mut *p.borrow_mut(), d));
    drop(previous);
    to_device
}

/// The part of the routing answer that no state can change.
///
/// Off and Always are the user's answer, and must not become conditional on
/// whether the calibration state can be reached. `ensure_current` returns nothing
/// off an execution thread and nothing on a machine with no adapter, and an
/// explicit Always that quietly became "cores" in either case would be a
/// setting that does not do what it says.
///
/// `None` means "no forced answer — go and measure".
fn forced_route(mode: GpuMode, n: usize) -> Option<bool> {
    match mode {
        GpuMode::Off => Some(false),
        GpuMode::Always => Some(Calib::eligible(n)),
        GpuMode::Auto => None,
    }
}

thread_local! {
    /// The decision this thread is currently executing, awaiting its cost.
    static PENDING: std::cell::RefCell<Option<Decision>> =
        const { std::cell::RefCell::new(None) };
}

/// Record what a group actually cost. `units` is `bars × frames`.
///
/// Takes no block size: the size is the one the parked decision was made for,
/// and re-supplying it would let a caller record against a size it never routed.
pub fn record(units: f64, secs: f64, outcome: Outcome) -> Option<Landed> {
    // The test build forces every qualifying group onto the device, so its
    // timings are all one-sided — recording them would teach this machine that
    // the route it never compared is the better one. The benchmark is large
    // enough to clear `MIN_UNITS`, so the size guard alone does not cover it.
    #[cfg(test)]
    if !inject::allow_record() {
        retire_pending();
        return None;
    }
    let mut d = PENDING.with(|p| p.borrow_mut().take())?;
    // A lookup against the generation this decision was made under. Never the
    // initializing path: a completion from an execution that is no longer
    // current must not rekey, reload or re-probe anything on its way to being
    // dropped.
    let landed = with_current(d.generation, |c| c.finish(&mut d, units, secs, outcome));
    match landed {
        Some(l) => {
            crate::mlog!("gpu     sample 2^{:<2} {:?} → {l:?}", d.n.trailing_zeros(), outcome);
            Some(l)
        }
        // The generation moved on beneath it. Report it as what it is rather
        // than as nothing having happened.
        None => Some(Landed::Dropped(Dropped::StaleGeneration)),
    }
}

/// Whether this thread is holding a routing decision.
///
/// For the `aslt` tests, which check that the group loop's scope discipline
/// leaves nothing parked. A decision still here when an analysis has finished
/// is one the next group would consume.
#[cfg(test)]
pub fn pending_is_empty() -> bool {
    PENDING.with(|p| p.borrow().is_none())
}

/// An isolated calibration environment for a test in another module.
///
/// Two modules outside `gpu_calib` need one, for opposite reasons:
///
/// * `aslt`'s tests need a calibration that will actually hand out decisions —
///   an adapter, a scratch directory, a probe that does nothing — because
///   `route_to_device` short-circuits in a test build and never reaches
///   `use_device`, so a test routed through it can never park anything to leak.
/// * `spectrum`'s cancellation schedules need the opposite: the directory
///   redirected and **nothing else** injected, so the analysis they run behaves
///   as it would on a machine that has no calibration yet.
///
/// Both hold the shared environment lock for their lifetime and put everything
/// back on drop, exactly as the calibration's own fixture does.
///
/// `TEST_DIR` alone is not enough, and a fixture that locks only `TEST_DIR` for
/// its two assignments is unsafe. That mutex serialises one assignment; it does
/// not own the span between installing a directory and restoring it. A fixture
/// entering that span replaces the installed directory, and the first fixture's
/// drop then restores the `None` it had saved — which sends [`path`] back to
/// the running user's profile while the second fixture is still using it.
///
/// Lock order: `ENV` is taken **first** and released **last**; `TEST_DIR`,
/// `ADAPTER`, `PROBE` and `STATE` are only ever taken under it, and each is
/// released before the next is taken. Production never takes `ENV` — it does
/// not exist outside a test build — so holding it across an analysis cannot
/// deadlock against the code under test. `spectrum`'s `cancel_seam::ENV` is
/// always taken before this one and never after it, and no other test takes
/// both, so the two orders cannot cross.
#[cfg(test)]
pub struct AsltTestEnv {
    _guard: std::sync::MutexGuard<'static, ()>,
    dir: PathBuf,
}

#[cfg(test)]
impl AsltTestEnv {
    /// Whether any group actually took a decision. Without this a test that
    /// silently routed nothing would pass by vacuum.
    pub fn saw_a_decision(&self) -> bool {
        STATE
            .lock()
            .ok()
            .and_then(|g| g.as_ref().map(|c| !c.stored.sizes.is_empty()))
            .unwrap_or(false)
    }

    /// The directory this environment redirected the calibration to.
    pub fn dir(&self) -> &std::path::Path {
        &self.dir
    }
}

#[cfg(test)]
impl Drop for AsltTestEnv {
    fn drop(&mut self) {
        *inject::TEST_DIR.lock().unwrap_or_else(|e| e.into_inner()) = None;
        *inject::ADAPTER.lock().unwrap_or_else(|e| e.into_inner()) = None;
        *inject::PROBE.lock().unwrap_or_else(|e| e.into_inner()) = None;
        inject::ALLOW_RECORD.store(false, O::Relaxed);
        invalidate();
        *STATE.lock().unwrap_or_else(|e| e.into_inner()) = None;
        retire_pending();
        let _ = std::fs::remove_dir_all(&self.dir);
    }
}

#[cfg(test)]
fn test_env(tag: &str, live: bool) -> AsltTestEnv {
    let guard = inject::ENV.lock().unwrap_or_else(|e| e.into_inner());
    let dir = std::env::temp_dir().join(format!(
        "moosik_calib_{tag}_{}_{:x}",
        std::process::id(),
        std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .map(|x| x.as_nanos())
            .unwrap_or(0),
    ));
    std::fs::create_dir_all(&dir).unwrap();
    *inject::TEST_DIR.lock().unwrap_or_else(|e| e.into_inner()) = Some(dir.clone());
    if live {
        *inject::ADAPTER.lock().unwrap_or_else(|e| e.into_inner()) =
            Some(Some("Test Adapter".to_string()));
        *inject::PROBE.lock().unwrap_or_else(|e| e.into_inner()) = Some(Box::new(|| {}));
        inject::ALLOW_RECORD.store(true, O::Relaxed);
    }
    invalidate();
    *STATE.lock().unwrap_or_else(|e| e.into_inner()) = None;
    if live {
        set_mode(GpuMode::Auto);
    }
    AsltTestEnv { _guard: guard, dir }
}

/// A calibration that will hand out decisions. See [`AsltTestEnv`].
#[cfg(test)]
pub fn aslt_test_env(tag: &str) -> AsltTestEnv {
    test_env(tag, true)
}

/// Isolation and nothing else: the directory is redirected, no adapter, no
/// probe, no recording, and the mode is left alone. See [`AsltTestEnv`].
#[cfg(test)]
pub fn isolated_test_env(tag: &str) -> AsltTestEnv {
    test_env(tag, false)
}

/// Where the calibration would be read and written right now.
///
/// For the isolation regression in `spectrum`, which asserts on the resolved
/// path rather than performing any file operation on it.
#[cfg(test)]
pub fn path_for_test() -> PathBuf {
    path()
}

/// Write any pending state now. Called when an analysis finishes — one of the
/// two moments the debounce is skipped.
pub fn flush() {
    if let Ok(mut g) = STATE.lock()
        && let Some(c) = g.as_mut()
        && c.dirty
    {
        c.save_now();
    }
}

/// Human-readable state, for the debug panel.
///
/// Never blocks and never starts a probe. This runs inside a repaint, and the
/// probe behind the state lock holds it for as long as it takes to time six
/// block sizes on both routes — seconds on a fast machine, far longer on a slow
/// one. Calling into it from here froze the whole window the first time the
/// panel was opened on an un-probed machine, which looked like a hang rather
/// than a measurement. The warm-up thread does the work; the panel only ever
/// reports what is already known.
pub fn describe() -> Vec<String> {
    match STATE.try_lock() {
        Ok(g) => match g.as_ref() {
            Some(c) => describe_calibration(c),
            None => vec!["  (not measured yet)".into()],
        },
        Err(_) => vec!["  (measuring…)".into()],
    }
}

fn describe_calibration(c: &Calib) -> Vec<String> {
    let mut out = vec![format!(
        "  identity: {} / {} cores / {} workers (schema {}, producer {})",
        c.stored.key.adapter, c.stored.key.cores, c.stored.key.workers, SCHEMA, PRODUCER,
    )];
    for (&n, s) in &c.stored.sizes {
        let v = s.verdict();
        let ratio = match (s.gpu_cost(), s.cpu_cost()) {
            (Some(g), Some(cc)) if g > 0.0 => format!("{:.2}× cores", cc / g),
            _ => "—".into(),
        };
        out.push(format!(
            "  2^{:<2} {:<8} {:>10}  gpu {}/cpu {}  {}",
            n.trailing_zeros(),
            if v.to_device() { "device" } else { "cores" },
            ratio,
            s.gpu_n,
            s.cpu_n,
            match v {
                Verdict::Settled(_) => "settled".to_string(),
                Verdict::Sampling(_) => format!("sampling ({}/{MAX_ATTEMPTS})", s.attempts),
                // Said plainly, because it is not a verdict: nothing was
                // compared, and the panel must not read as though it was.
                Verdict::Inconclusive => "inconclusive — never measured".to_string(),
            },
        ));
    }
    if c.stored.sizes.is_empty() {
        out.push("  (no calibration yet)".into());
    }
    if c.save_failed {
        out.push("  (the last save failed; measurements are held in memory)".into());
    }
    out
}

/// Do the first-touch work — adapter enumeration and the probe — on a thread
/// that is allowed to take its time.
///
/// Both halves are first-call-only and both can be slow on hardware nobody
/// tested on: `GpuFft::describe` builds a wgpu instance and enumerates
/// adapters, which a sick driver can sit on for seconds, and the probe times
/// real convolutions at six block sizes. Neither belongs on the UI thread, and
/// before this existed both could land there.
///
/// The probe half runs through `install`, so it happens on the pool the
/// analyses will use and under the identity they will be keyed by.
pub fn warm() {
    std::thread::Builder::new()
        .name("moosik-gpu-warm".into())
        .spawn(|| {
            let t = Instant::now();
            let adapter = super::gpu::GpuFft::describe();
            crate::mlog!(
                "gpu     adapter {} ({:.2}s)",
                adapter.as_deref().unwrap_or("none — CPU only"),
                t.elapsed().as_secs_f64()
            );
            DEVICE.store(if adapter.is_some() { 1 } else { 2 }, std::sync::atomic::Ordering::Release);
            if adapter.is_none() { return }

            let t = Instant::now();
            // Through the same initialization path an analysis uses, so a
            // warm-up that is revoked while probing discards its result under
            // exactly the rule everything else follows.
            let n = install(|| {
                ensure_current()
                    .and_then(|g| with_current(g, |c| c.stored.sizes.len()))
                    .unwrap_or(0)
            });
            crate::mlog!("gpu     calibration ready, {n} size(s), {:.2}s", t.elapsed().as_secs_f64());
            for line in describe() { crate::mlog!("gpu    {line}"); }
        })
        .ok();
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::atomic::{AtomicBool, AtomicU32, Ordering as AO};
    use std::sync::mpsc;
    use std::sync::Arc;
    use std::time::Duration as Dur;

    /// What the probe says on this machine. Diagnostic, not a pass/fail: the
    /// answer is hardware, and the point is to be able to see it.
    #[test]
    #[ignore = "hardware survey — run explicitly"]
    fn probe_survey() {
        let mut c = Stored::default();
        probe(&mut c);
        if c.sizes.is_empty() {
            println!("no device, or every eligible size failed to run");
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

    // ── the environment every schedule runs in ──────────────────────────────

    /// Holds the injection globals for one test, and puts everything back.
    ///
    /// The point of this type is that the schedules below reach the **production**
    /// functions — `install`, `use_device`, `record`, `recalibrate`,
    /// `ensure_current`, `with_current`, `save_now` — with only their external
    /// dependencies replaced. Nothing here re-implements a decision.
    struct Env {
        _guard: std::sync::MutexGuard<'static, ()>,
        dir: PathBuf,
    }

    impl Env {
        fn new(tag: &str) -> Self {
            let guard = inject::ENV.lock().unwrap_or_else(|e| e.into_inner());
            let dir = std::env::temp_dir().join(format!(
                "moosik_calib_{tag}_{}_{:x}",
                std::process::id(),
                std::time::SystemTime::now()
                    .duration_since(std::time::UNIX_EPOCH)
                    .map(|x| x.as_nanos())
                    .unwrap_or(0),
            ));
            std::fs::create_dir_all(&dir).unwrap();

            *inject::TEST_DIR.lock().unwrap_or_else(|e| e.into_inner()) = Some(dir.clone());
            *inject::ADAPTER.lock().unwrap_or_else(|e| e.into_inner()) =
                Some(Some("Test Adapter".to_string()));
            // A probe that does nothing, unless a schedule replaces it.
            *inject::PROBE.lock().unwrap_or_else(|e| e.into_inner()) = Some(Box::new(|| {}));
            *inject::BEFORE_PUBLISH.lock().unwrap_or_else(|e| e.into_inner()) = None;
            inject::POOL_FAILS.store(false, AO::Relaxed);
            inject::PUBLISH_FAILS.store(false, AO::Relaxed);
            inject::ALLOW_RECORD.store(true, AO::Relaxed);

            // A fresh process-wide state for each schedule.
            invalidate();
            *STATE.lock().unwrap_or_else(|e| e.into_inner()) = None;
            set_mode(GpuMode::Auto);

            Self { _guard: guard, dir }
        }

        fn file(&self) -> PathBuf {
            self.dir.join("gpu_calibration_v2.json")
        }
        fn legacy(&self) -> PathBuf {
            self.dir.join("gpu_calibration.json")
        }
        fn temporaries(&self) -> Vec<String> {
            std::fs::read_dir(&self.dir)
                .unwrap()
                .filter_map(|e| e.ok())
                .map(|e| e.file_name().to_string_lossy().to_string())
                .filter(|n| n.contains("tmp-"))
                .collect()
        }
        /// The live state's view, without going through a decision.
        fn live<R>(&self, f: impl FnOnce(&Calib) -> R) -> Option<R> {
            let g = STATE.lock().unwrap_or_else(|e| e.into_inner());
            g.as_ref().map(f)
        }
        fn stat(&self, n: usize) -> Option<SizeStat> {
            self.live(|c| c.stored.sizes.get(&n).copied()).flatten()
        }
        /// What is actually on disk, parsed.
        fn on_disk(&self) -> Option<Stored> {
            let text = std::fs::read_to_string(self.file()).ok()?;
            serde_json::from_str(&text).ok()
        }
    }

    impl Drop for Env {
        fn drop(&mut self) {
            *inject::TEST_DIR.lock().unwrap_or_else(|e| e.into_inner()) = None;
            *inject::ADAPTER.lock().unwrap_or_else(|e| e.into_inner()) = None;
            *inject::PROBE.lock().unwrap_or_else(|e| e.into_inner()) = None;
            *inject::BEFORE_PUBLISH.lock().unwrap_or_else(|e| e.into_inner()) = None;
            inject::POOL_FAILS.store(false, AO::Relaxed);
            inject::PUBLISH_FAILS.store(false, AO::Relaxed);
            inject::ALLOW_RECORD.store(false, AO::Relaxed);
            invalidate();
            *STATE.lock().unwrap_or_else(|e| e.into_inner()) = None;
            set_mode(GpuMode::Auto);
            let _ = std::fs::remove_dir_all(&self.dir);
        }
    }

    /// A size production actually routes.
    const N: usize = 1 << 18;
    /// A second eligible size, for wrong-size attribution checks.
    const N2: usize = 1 << 19;
    /// A size below the routing floor.
    const SMALL: usize = 1 << 15;
    /// Comfortably above `MIN_UNITS`.
    const UNITS: f64 = 5.0e7;

    const _: () = assert!(SMALL < super::super::aslt::GPU_MIN_BLOCK);
    const _: () = assert!(N >= super::super::aslt::GPU_MIN_BLOCK);
    const _: () = assert!(N2 >= super::super::aslt::GPU_MIN_BLOCK);

    /// Run `f` as an analysis would: on the real analysis pool, with the real
    /// execution stamp.
    fn as_analysis<R: Send>(f: impl FnOnce() -> R + Send) -> R {
        install(f)
    }

    /// One complete group, through the production entry points.
    ///
    /// `use_device` → `record` is exactly the pair `analyze_routed` calls, and
    /// `begin_group` is the scope it opens. Nothing here constructs a `Calib`
    /// or calls `decide`/`finish` directly.
    fn group(n: usize, outcome: Outcome, secs: f64) -> Option<Landed> {
        let scope = begin_group();
        let _routed = use_device(n);
        scope.work_started();
        record(UNITS, secs, outcome)
    }

    /// A group that starts and is then cancelled: no `record`, scope dropped.
    fn cancelled_after_start(n: usize) {
        let scope = begin_group();
        let _routed = use_device(n);
        scope.work_started();
        drop(scope);
    }

    /// A group cancelled before its work began.
    fn cancelled_before_start(n: usize) {
        let scope = begin_group();
        let _routed = use_device(n);
        drop(scope);
    }

    /// Bounded watchdog. Runs `f` on its own thread and fails rather than
    /// wedging the suite if it does not return.
    ///
    /// This exists for the deadlock schedule: a mutant that reinstates the
    /// lock-across-probe would otherwise hang the whole test binary.
    fn within<T: Send + 'static>(limit: Dur, what: &str, f: impl FnOnce() -> T + Send + 'static) -> T {
        let (tx, rx) = mpsc::channel();
        let h = std::thread::spawn(move || {
            let r = f();
            let _ = tx.send(());
            r
        });
        match rx.recv_timeout(limit) {
            Ok(()) => h.join().expect("the watched thread panicked"),
            Err(_) => panic!(
                "{what}: did not finish within {limit:?} — treated as a deadlock. \
                 The thread is left running rather than joined, so the suite can \
                 still report."
            ),
        }
    }

    // ── A. parallel probe against analysis ──────────────────────────────────

    /// Record that this run reached a named stage, for the mutation runner.
    ///
    /// A deadlock is detected by a deadline expiring, and a deadline expiring
    /// says nothing on its own about *where* the run got to: a build that dies
    /// on startup, a filter that matches nothing and a genuinely wedged thread
    /// all look the same from outside. This writes a marker **immediately
    /// before the controlled operation** whose wedging is the thing being
    /// detected, so a timeout can be told apart from a run that never arrived.
    ///
    /// The marker carries the runner's unique id for *this* run, the test, and
    /// the stage. A file left over from a previous run, or a marker written by
    /// a different test, is therefore not mistaken for evidence — which a bare
    /// "the process started" file would be.
    ///
    /// Does nothing unless the runner asked for it. Written once per process:
    /// the call site below is inside a parallel body.
    fn mutant_handshake(test: &str, stage: &str) {
        use std::sync::atomic::{AtomicBool, Ordering};
        static WRITTEN: AtomicBool = AtomicBool::new(false);
        let (Ok(path), Ok(run)) = (
            std::env::var("MOOSIK_MUTANT_HANDSHAKE"),
            std::env::var("MOOSIK_MUTANT_RUN"),
        ) else {
            return;
        };
        if WRITTEN.swap(true, Ordering::SeqCst) {
            return;
        }
        // Written to a neighbouring temporary and renamed, so a reader can
        // never see half a marker.
        let tmp = format!("{path}.{}.tmp", std::process::id());
        if std::fs::write(&tmp, format!("{run}\n{test}\n{stage}\n")).is_ok() {
            let _ = std::fs::rename(&tmp, &path);
        }
    }

    /// The schedule the deadlock finding describes, executed.
    ///
    /// The probe runs parallel work on the analysis pool. While it is running,
    /// another job on that pool asks the calibration for a routing decision —
    /// which is what a Rayon worker does when it steals work while joining.
    ///
    /// **Source-derived schedule, now executed.** It is reproduced here by
    /// calling `ensure_current` from inside the probe's own parallel body,
    /// which is the same re-entrancy without depending on the scheduler
    /// choosing to steal. Under the old code the outer call held `STATE` for
    /// the whole probe and the inner one blocked on it for ever; under this
    /// code the outer call holds nothing while probing and the inner one
    /// returns immediately.
    #[test]
    fn an_analysis_can_run_while_the_probe_is_probing() {
        let _env = Env::new("probe_reentrancy");
        let inner_saw = Arc::new(AtomicU32::new(0));
        let seen = Arc::clone(&inner_saw);
        *inject::PROBE.lock().unwrap() = Some(Box::new(move || {
            use rayon::prelude::*;
            // Parallel work inside the probe, exactly as the real one does.
            let hits: u32 = (0..8u32)
                .into_par_iter()
                .map(|_| {
                    // A stolen analysis job asking for a decision. It must not
                    // block: nothing of ours is held.
                    //
                    // The handshake goes here and nowhere else: this call is
                    // the controlled operation. Under the mutant that holds the
                    // calibration lock across the probe it never returns, so a
                    // marker written immediately before it is what separates
                    // "wedged at the re-entrancy point" from "never got here".
                    mutant_handshake(
                        "an_analysis_can_run_while_the_probe_is_probing",
                        "probe-reentrancy-inner-ensure-current",
                    );
                    u32::from(ensure_current().is_some())
                })
                .sum();
            seen.store(hits + 1, AO::SeqCst);
        }));

        within(Dur::from_secs(20), "probe re-entrancy", || {
            as_analysis(|| {
                let g = ensure_current();
                assert!(g.is_some(), "initialization did not publish");
            })
        });

        assert!(
            inner_saw.load(AO::SeqCst) >= 1,
            "the probe's parallel body never ran, so nothing was re-entered"
        );
        // The inner callers declined rather than blocking or probing again.
        assert_eq!(
            inner_saw.load(AO::SeqCst) - 1,
            0,
            "a caller during initialization was served state that did not exist yet"
        );
    }

    /// A probe revoked while it is running must not publish.
    #[test]
    fn a_probe_invalidated_before_publication_is_discarded() {
        let _env = Env::new("probe_revoked");
        *inject::PROBE.lock().unwrap() = Some(Box::new(|| {
            // Something else resets, or the pool changes, while this runs.
            invalidate();
        }));

        let published = within(Dur::from_secs(20), "revoked probe", || {
            as_analysis(ensure_current)
        });
        assert!(published.is_none(), "a revoked probe published its result");
        assert!(
            STATE.lock().unwrap().is_none(),
            "a revoked probe installed state anyway"
        );
    }

    /// Concurrent first access probes once and never waits.
    #[test]
    fn concurrent_first_access_probes_once_and_nobody_waits() {
        const THREADS: usize = 4;
        let _env = Env::new("concurrent_init");
        let probes = Arc::new(AtomicU32::new(0));
        let count = Arc::clone(&probes);
        // The prober is held inside the probe until every other thread has
        // reached the claim, so the schedule is a barrier rather than a sleep
        // that has to be long enough.
        // Every thread announces itself before calling in; the prober then
        // holds the claim until all of them have, so the others meet a claim
        // that is genuinely taken. A bounded spin, not a sleep long enough to
        // hope.
        let entered = Arc::new(AtomicU32::new(0));
        let seen = Arc::clone(&entered);
        *inject::PROBE.lock().unwrap() = Some(Box::new(move || {
            count.fetch_add(1, AO::SeqCst);
            let deadline = std::time::Instant::now() + Dur::from_secs(5);
            while seen.load(AO::SeqCst) < THREADS as u32
                && std::time::Instant::now() < deadline
            {
                std::thread::yield_now();
            }
            std::thread::sleep(Dur::from_millis(60));
        }));

        // **One execution across all four threads.** `install` starts a new
        // execution per call, and only the newest may initialize, so spawning
        // four analyses tests supersession and not the claim. Every thread here
        // adopts the *same* execution stamp, which is what a Rayon pool's
        // workers do, so all four reach the claim and the guard is what
        // decides. Without this the mutant that removes the guard survives:
        // three of the four are turned away before they ever get there.
        let held = enter_exec(2);
        let stamp = (EXEC.with(|e| e.get()), EXEC_EPOCH.with(|e| e.get()));

        let outer = Arc::clone(&entered);
        within(Dur::from_secs(20), "concurrent init", move || {
            let hands: Vec<_> = (0..THREADS)
                .map(|_| {
                    let mine = Arc::clone(&outer);
                    std::thread::spawn(move || {
                        leave_exec(stamp);          // adopt, do not supersede
                        mine.fetch_add(1, AO::SeqCst);
                        ensure_current()
                    })
                })
                .collect();
            let got: Vec<_> = hands.into_iter().map(|h| h.join().unwrap()).collect();
            assert!(
                got.iter().any(|g| g.is_some()),
                "nobody managed to initialize"
            );
        });
        leave_exec(held);
        assert_eq!(
            probes.load(AO::SeqCst),
            1,
            "concurrent first access probed more than once"
        );
    }

    // ── B. a cancelled decision against worker reuse ────────────────────────

    /// The ownership contract. A cancelled Auto group must not leave its
    /// decision for whatever runs next on that worker — and Always, Off and
    /// below-floor groups are exactly what runs next, because none of them
    /// takes a decision of its own.
    #[test]
    fn a_cancelled_group_cannot_be_recorded_by_the_next_one() {
        let _env = Env::new("stale_decision");

        as_analysis(|| {
            // An eligible Auto group at N, started and then cancelled.
            cancelled_after_start(N);

            // Always, on a different size. Under the old code this consumed the
            // cancelled group's decision and filed its time under N.
            set_mode(GpuMode::Always);
            let landed = group(N2, Outcome::Device, 1.0);
            assert!(
                landed.is_none() || landed == Some(Landed::Dropped(Dropped::NotAuto)),
                "a forced group trained something: {landed:?}"
            );

            set_mode(GpuMode::Off);
            let landed = group(N2, Outcome::Cores, 1.0);
            assert!(
                landed.is_none() || landed == Some(Landed::Dropped(Dropped::NotAuto)),
                "an Off group trained something: {landed:?}"
            );

            // A below-floor group, which never reaches `use_device` in
            // production because `route_to_device` gates it first. Simulated
            // faithfully: open a scope, route nothing, then record.
            set_mode(GpuMode::Auto);
            let scope = begin_group();
            let landed = record(UNITS, 1.0, Outcome::Cores);
            drop(scope);
            assert!(landed.is_none(), "a below-floor group recorded: {landed:?}");
        });

        let s = _env.stat(N).expect("the cancelled group should have left its attempt");
        assert_eq!(
            (s.gpu_n, s.cpu_n),
            (0, 0),
            "a later group's time was filed under the cancelled group's size"
        );
        assert!(
            _env.stat(N2).is_none_or(|s| s.gpu_n == 0 && s.cpu_n == 0),
            "a forced route trained the automatic one"
        );
    }

    /// The scope retires the decision on an unwind, not only on a clean return.
    #[test]
    fn an_unwinding_group_retires_its_decision() {
        let _env = Env::new("unwind");
        as_analysis(|| {
            let hit = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
                let scope = begin_group();
                let _ = use_device(N);
                scope.work_started();
                panic!("simulated failure inside a group");
            }));
            assert!(hit.is_err(), "setup: the panic should have been caught");

            // Nothing left parked: the next group records for itself or not at
            // all.
            let landed = {
                let scope = begin_group();
                let l = record(UNITS, 1.0, Outcome::Cores);
                drop(scope);
                l
            };
            assert!(landed.is_none(), "the unwound group's decision survived: {landed:?}");
        });
    }

    /// The execution stamp is restored on an unwind too. A Rayon worker that
    /// kept claiming to be running an analysis would let later unrelated code
    /// create state under a stale identity.
    #[test]
    fn the_execution_stamp_is_restored_after_an_unwind() {
        let _env = Env::new("exec_unwind");
        assert_eq!(exec_workers(), 0, "setup: this thread is not an analysis pool");
        let hit = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            with_exec(4, || panic!("simulated failure inside an analysis"));
        }));
        assert!(hit.is_err(), "setup: the panic should have been caught");
        assert_eq!(exec_workers(), 0, "the execution stamp leaked past an unwind");
    }

    // ── C. old and new pool identities ──────────────────────────────────────

    /// Analysis A is held on one pool; the identity changes; B runs; A
    /// completes last. A must not train, rekey, or resurrect anything.
    #[test]
    fn a_held_analysis_cannot_train_or_rekey_the_set_that_replaced_it() {
        let _env = Env::new("held_pool");

        // A: decide under a 2-worker execution, but do not record yet.
        let held = with_exec(2, || {
            let scope = begin_group();
            let _ = use_device(N);
            scope.work_started();
            // Take the decision out so it survives the scope, which is exactly
            // the in-flight state a long group is in.
            PENDING.with(|p| p.borrow_mut().take())
        });
        let mut held = held.expect("A should have made a decision");
        let gen_a = held.generation;

        // B: a different execution identity entirely.
        let gen_b = with_exec(1, || {
            let g = ensure_current().expect("B should initialize");
            let landed = group(N, Outcome::Device, 1.0);
            assert_eq!(landed, Some(Landed::Device), "B's own sample was refused");
            g
        });
        assert_ne!(gen_a, gen_b, "the identity change did not move the generation");
        let b_before = _env.stat(N).expect("B should have a stat");
        let key_before = _env.live(|c| c.stored.key.clone()).unwrap();

        // A completes, from its own execution context.
        let landed = with_exec(2, || {
            with_current(held.generation, |c| c.finish(&mut held, UNITS, 9.0, Outcome::Device))
        });
        assert!(landed.is_none(), "A's completion reached the new set");

        assert_eq!(_env.stat(N), Some(b_before), "A changed B's numbers");
        assert_eq!(
            _env.live(|c| c.stored.key.clone()).unwrap(),
            key_before,
            "A's completion rekeyed the live state back to its own pool"
        );
        assert_eq!(
            _env.live(|c| c.generation).unwrap(),
            gen_b,
            "A's completion replaced the live generation"
        );

        // And the release path is equally harmless.
        drop(held);
        assert_eq!(_env.stat(N), Some(b_before), "A's retirement changed B's numbers");
    }

    /// Away and back. The worker count matches again but the execution does
    /// not, and the original token must not be revived.
    #[test]
    fn changing_workers_away_and_back_does_not_revive_the_old_execution() {
        let _env = Env::new("away_back");

        let held = with_exec(2, || {
            let scope = begin_group();
            let _ = use_device(N);
            scope.work_started();
            PENDING.with(|p| p.borrow_mut().take())
        });
        let mut held = held.expect("the first execution should have decided");

        with_exec(1, || ensure_current().expect("the 1-worker identity"));
        let gen_back = with_exec(2, || ensure_current().expect("back to 2 workers"));
        assert_ne!(held.generation, gen_back, "the original token was revived");

        let landed = with_exec(2, || {
            with_current(held.generation, |c| c.finish(&mut held, UNITS, 1.0, Outcome::Device))
        });
        assert!(landed.is_none(), "the held sample landed in the reloaded set");
        assert!(
            _env.stat(N).is_none_or(|s| s.gpu_n == 0),
            "the held sample trained the reloaded set"
        );
    }

    /// Pool-construction fallback, through the real selection path: what
    /// executes is the ambient pool, and that is what is stamped.
    #[test]
    fn a_pool_that_will_not_build_is_keyed_to_what_actually_ran() {
        let _env = Env::new("pool_fallback");
        inject::POOL_FAILS.store(true, AO::Relaxed);

        let stamped = as_analysis(|| {
            assert_ne!(exec_workers(), 0, "the fallback path did not stamp the thread");
            let _ = ensure_current();
            exec_workers()
        });
        let key = _env.live(|c| c.stored.key.clone()).expect("state should exist");
        assert_eq!(
            key.workers, stamped,
            "the key recorded a pool size that never ran"
        );
        assert_eq!(
            stamped,
            rayon::current_num_threads().max(1),
            "the fallback stamped something other than the pool that executed"
        );
    }

    // ── D. reset against publication ────────────────────────────────────────

    /// A save already in flight must not be able to put pre-reset measurements
    /// back, in memory or on disk.
    ///
    /// The old ordering removed the file *before* taking the state lock, so a
    /// save holding that lock could publish afterwards and the reload would
    /// find it.
    #[test]
    fn a_reset_cannot_be_undone_by_a_save_already_in_flight() {
        let _env = Env::new("reset_vs_save");

        // Build something worth losing.
        as_analysis(|| {
            for _ in 0..MIN_SAMPLES {
                group(N, Outcome::Device, 1.0);
                group(N, Outcome::Cores, 2.0);
            }
        });
        flush();
        assert!(_env.file().exists(), "setup: there should be a file to reset");
        let before = _env.on_disk().expect("setup: it should parse");
        assert!(!before.sizes.is_empty(), "setup: it should hold measurements");

        // The hook only signals and waits. It must not touch `STATE`: every
        // save in production runs *while holding* it, so a hook that took it
        // would deadlock against the very operation it is observing — which is
        // exactly what the first version of this test did.
        let (at_boundary, boundary_rx) = mpsc::channel::<()>();
        let (go_tx, go) = mpsc::channel::<()>();
        let go = std::sync::Mutex::new(go);
        let fired = Arc::new(AtomicU32::new(0));
        let count = Arc::clone(&fired);
        *inject::BEFORE_PUBLISH.lock().unwrap() = Some(Box::new(move || {
            if count.fetch_add(1, AO::SeqCst) == 0 {
                let _ = at_boundary.send(());
                let _ = go.lock().unwrap().recv_timeout(Dur::from_secs(10));
            }
        }));

        // A saver, holding STATE and parked at its publication boundary with
        // the pre-reset bytes already serialised.
        let saver = std::thread::spawn(|| {
            let mut g = STATE.lock().unwrap_or_else(|e| e.into_inner());
            if let Some(c) = g.as_mut() {
                c.dirty = true;
                c.save_now();
            }
        });
        boundary_rx
            .recv_timeout(Dur::from_secs(10))
            .expect("the saver never reached its publication boundary");

        // The reset, concurrently. Under the repaired ordering it revokes and
        // then blocks on STATE, so the removal cannot happen until the saver
        // has published and let go. Under the old ordering it deleted the file
        // first and the saver then wrote the old measurements back over the
        // deletion.
        let resetter = std::thread::spawn(recalibrate);
        std::thread::sleep(Dur::from_millis(150));
        let _ = go_tx.send(());

        saver.join().expect("the saver panicked");
        resetter.join().expect("the reset panicked");
        *inject::BEFORE_PUBLISH.lock().unwrap() = None;
        // At least once, for the parked saver. The reset publishes a fresh
        // empty state of its own afterwards, which legitimately fires it again.
        assert!(
            fired.load(AO::SeqCst) >= 1,
            "the saver never reached its publication boundary"
        );

        // Whatever the interleaving, the pre-reset measurements must not come
        // back — in memory or from disk.
        let live_now = _env.live(|c| c.stored.sizes.len()).unwrap_or(0);
        let reloaded = {
            invalidate();
            *STATE.lock().unwrap() = None;
            as_analysis(|| ensure_current().and_then(|g| with_current(g, |c| c.stored.sizes.len())))
        };
        assert_eq!(live_now, 0, "the reset left pre-reset measurements live");
        assert_eq!(
            reloaded,
            Some(0),
            "the reset reloaded pre-reset measurements from disk"
        );
    }

    /// The whole `recalibrate` entry point, not a reconstruction of it.
    #[test]
    fn recalibrate_clears_the_identity_and_leaves_the_legacy_file_alone() {
        let _env = Env::new("recalibrate");
        std::fs::write(_env.legacy(), b"legacy").unwrap();

        as_analysis(|| {
            for _ in 0..MIN_SAMPLES {
                group(N, Outcome::Device, 1.0);
            }
        });
        flush();
        assert!(_env.stat(N).is_some_and(|s| s.gpu_n > 0), "setup: something to reset");

        recalibrate();

        assert_eq!(
            as_analysis(|| ensure_current().and_then(|g| with_current(g, |c| c.stored.sizes.len()))),
            Some(0),
            "recalibration kept the old measurements"
        );
        assert_eq!(
            std::fs::read(_env.legacy()).unwrap(),
            b"legacy",
            "recalibration touched the legacy file"
        );
        assert!(_env.temporaries().is_empty(), "left {:?}", _env.temporaries());
    }

    // ── E. the attempt-budget lifecycle ─────────────────────────────────────

    /// A reservation abandoned before the work began is refunded; one abandoned
    /// after it began is not.
    #[test]
    fn a_refund_depends_on_whether_the_work_actually_started() {
        let _env = Env::new("refund");
        as_analysis(|| {
            cancelled_before_start(N);
            assert_eq!(
                _env.stat(N).map(|s| s.attempts),
                Some(0),
                "a reservation abandoned before the work started was not refunded"
            );

            cancelled_after_start(N);
            assert_eq!(
                _env.stat(N).map(|s| s.attempts),
                Some(1),
                "a started attempt was refunded"
            );
        });
    }

    /// Refunds and started attempts both reach the disk. A budget that grows
    /// back across a restart is not a bound.
    #[test]
    fn budget_changes_are_persisted() {
        let _env = Env::new("budget_persist");
        as_analysis(|| {
            cancelled_after_start(N);
            cancelled_after_start(N);
            cancelled_before_start(N);
        });
        flush();
        let on_disk = _env.on_disk().expect("a file should have been written");
        assert_eq!(
            on_disk.sizes.get(&N).map(|s| s.attempts),
            Some(2),
            "the persisted attempt count does not match the live one"
        );
        assert_eq!(
            _env.stat(N).map(|s| s.attempts),
            Some(2),
            "the live attempt count is not what was persisted"
        );
    }

    /// A refund that arrives **after** an intervening save must still reach the
    /// disk.
    ///
    /// The narrow case, and the one a mutation found uncovered. `decide` marks
    /// the state dirty when it reserves, so in the ordinary sequence the
    /// reservation's own flag carries the refund to disk and `release`'s flag is
    /// redundant. It stops being redundant the moment anything writes in
    /// between — a debounce elapsing, or a verdict change flushing — because
    /// the write clears `dirty` and the refund is then the only thing left to
    /// set it.
    #[test]
    fn a_refund_after_an_intervening_save_is_still_persisted() {
        let _env = Env::new("refund_after_save");
        as_analysis(|| {
            let scope = begin_group();
            let _ = use_device(N);
            assert_eq!(_env.stat(N).map(|s| s.attempts), Some(1), "setup: reserved");

            // Something writes, clearing the reservation's dirty flag.
            {
                let mut g = STATE.lock().unwrap();
                g.as_mut().unwrap().save_now();
            }
            assert!(!_env.live(|c| c.dirty).unwrap_or(true), "setup: the write cleared dirty");
            assert_eq!(
                _env.on_disk().and_then(|s| s.sizes.get(&N).map(|x| x.attempts)),
                Some(1),
                "setup: the reservation is on disk"
            );

            // The group is abandoned before its work began: a refund.
            drop(scope);
        });
        flush();
        assert_eq!(
            _env.stat(N).map(|s| s.attempts),
            Some(0),
            "the refund did not reach the live state"
        );
        assert_eq!(
            _env.on_disk().and_then(|s| s.sizes.get(&N).map(|x| x.attempts)),
            Some(0),
            "a refund after an intervening save never reached the disk"
        );
    }

    /// Repeatedly starting and cancelling cannot buy unlimited exploration.
    #[test]
    fn repeated_start_and_cancel_exhausts_the_budget() {
        let _env = Env::new("start_cancel");
        as_analysis(|| {
            for _ in 0..MAX_ATTEMPTS + 4 {
                cancelled_after_start(N);
            }
        });
        let s = _env.stat(N).expect("a stat should exist");
        assert_eq!(s.attempts, MAX_ATTEMPTS, "the budget was exceeded or refunded");
        assert!(s.exhausted, "exhaustion was not latched");
        assert_eq!(s.verdict(), Verdict::Inconclusive);
    }

    /// Every outcome a real group can have, driven through the production
    /// entry points, and what each costs.
    #[test]
    fn every_outcome_is_charged_and_classified_correctly() {
        let _env = Env::new("outcomes");
        as_analysis(|| {
            assert_eq!(group(N, Outcome::Device, 1.0), Some(Landed::Device));
            assert_eq!(group(N, Outcome::Cores, 2.0), Some(Landed::Cores));
            assert_eq!(
                group(N, Outcome::PartialFallback, 1.0),
                Some(Landed::Dropped(Dropped::MixedRoute))
            );
            assert_eq!(
                group(N, Outcome::Unavailable, 1.0),
                Some(Landed::Dropped(Dropped::MixedRoute))
            );
            // Undersized: below MIN_UNITS.
            let scope = begin_group();
            let _ = use_device(N);
            scope.work_started();
            assert_eq!(
                record(1.0, 1.0, Outcome::Device),
                Some(Landed::Dropped(Dropped::NotMeasurable))
            );
            drop(scope);
        });
        let s = _env.stat(N).expect("a stat should exist");
        assert_eq!((s.gpu_n, s.cpu_n), (1, 1), "an invalid group trained a side");
        assert_eq!(s.attempts, 5, "not every started attempt was charged");
    }

    /// Exhaustion is not undone by ordinary work. Once exploration has given
    /// up, cores samples from non-exploratory groups must not complete the pair
    /// it abandoned and announce a measured winner.
    #[test]
    fn ordinary_cores_work_cannot_revive_an_exhausted_size() {
        let _env = Env::new("no_revival");
        as_analysis(|| {
            // Two clean device samples, then the budget is spent on nothing.
            for _ in 0..MIN_SAMPLES {
                group(N, Outcome::Device, 1.0);
            }
            // Bounded. An unbounded `while attempts < MAX` loop here spins for
            // ever the moment a started attempt is refunded instead of charged
            // — which is one of the defects this suite exists to catch, so the
            // loop must not be the thing that hangs when it is reinstated.
            for _ in 0..MAX_ATTEMPTS * 2 {
                if _env.stat(N).map(|s| s.attempts).unwrap_or(0) >= MAX_ATTEMPTS {
                    break;
                }
                cancelled_after_start(N);
            }
            let s = _env.stat(N).unwrap();
            assert!(s.exhausted, "setup: the budget should be spent");
            assert!(s.gpu_cost().is_some(), "setup: the device side has samples");
            assert!(s.cpu_cost().is_none(), "setup: the cores side has none");

            // Now ordinary non-exploratory groups keep recording cores time.
            for _ in 0..MIN_SAMPLES + 2 {
                group(N, Outcome::Cores, 5.0);
            }
        });
        let s = _env.stat(N).unwrap();
        assert!(s.cpu_cost().is_some(), "setup: the cores side filled up");
        assert_eq!(
            s.verdict(),
            Verdict::Inconclusive,
            "ordinary work resurrected a verdict exploration had abandoned"
        );
        assert!(!s.next_route(), "an exhausted size started exploring again");
        assert_eq!(
            _env.stat(N).map(|s| s.attempts),
            Some(MAX_ATTEMPTS),
            "an exhausted size kept spending budget"
        );
    }

    /// Exhaustion survives a reload, and a reset clears it.
    #[test]
    fn exhaustion_reloads_and_a_reset_clears_it() {
        let _env = Env::new("exhaust_reload");
        as_analysis(|| {
            for _ in 0..MAX_ATTEMPTS {
                cancelled_after_start(N);
            }
        });
        flush();
        assert!(_env.on_disk().unwrap().sizes[&N].exhausted, "exhaustion was not written");

        // A reload under the same identity.
        invalidate();
        *STATE.lock().unwrap() = None;
        as_analysis(ensure_current);
        assert!(_env.stat(N).unwrap().exhausted, "exhaustion did not survive the reload");

        recalibrate();
        assert!(
            as_analysis(|| ensure_current().and_then(|g| with_current(g, |c| c.stored.sizes.get(&N).copied())))
                .flatten()
                .is_none_or(|s| !s.exhausted),
            "a reset did not clear exhaustion"
        );
    }

    /// A settled size explores no further and spends nothing.
    #[test]
    fn a_settled_size_stops_spending_attempts() {
        let _env = Env::new("settled");
        as_analysis(|| {
            for _ in 0..MIN_SAMPLES {
                group(N, Outcome::Device, 1.0);
                group(N, Outcome::Cores, 2.0);
            }
            let spent = _env.stat(N).unwrap().attempts;
            assert_eq!(_env.stat(N).unwrap().verdict(), Verdict::Settled(true));

            cancelled_after_start(N);
            assert_eq!(
                _env.stat(N).unwrap().attempts,
                spent,
                "a settled size spent an attempt"
            );
        });
    }

    // ── F. file publication ─────────────────────────────────────────────────

    /// A failed publication leaves the existing valid file **byte for byte**,
    /// keeps the measurements, keeps the retry, and removes its temporary.
    #[test]
    fn a_failed_publication_leaves_the_existing_file_byte_identical() {
        let _env = Env::new("publish_fails");
        as_analysis(|| {
            for _ in 0..MIN_SAMPLES {
                group(N, Outcome::Device, 1.0);
            }
        });
        flush();
        let good = std::fs::read(_env.file()).expect("setup: a file should exist");
        assert!(!good.is_empty());

        inject::PUBLISH_FAILS.store(true, AO::Relaxed);
        as_analysis(|| {
            group(N, Outcome::Cores, 2.0);
        });
        flush();
        inject::PUBLISH_FAILS.store(false, AO::Relaxed);

        assert_eq!(
            std::fs::read(_env.file()).unwrap(),
            good,
            "a failed publication changed the existing file"
        );
        assert!(
            _env.live(|c| c.save_failed).unwrap_or(false),
            "a failed publication was not noticed"
        );
        assert!(_env.live(|c| c.dirty).unwrap_or(false), "the retry was discarded");
        assert!(
            _env.stat(N).is_some_and(|s| s.cpu_n == 1),
            "a failed publication threw away valid state"
        );
        assert!(_env.temporaries().is_empty(), "left {:?}", _env.temporaries());

        // And the retry succeeds, so the failure was not terminal.
        flush();
        assert_ne!(
            std::fs::read(_env.file()).unwrap(),
            good,
            "the retry never landed"
        );
    }

    /// Genuinely overlapping writers, with a barrier rather than sequential
    /// calls. Each publishes a complete file; nobody observes a torn one.
    ///
    /// This is the only claim about concurrency the file format makes, and the
    /// earlier version of this test called `save_now` twice in a row, which
    /// proves nothing about it.
    #[test]
    fn genuinely_concurrent_writers_never_publish_a_torn_file() {
        let _env = Env::new("concurrent_writers");
        let dir = _env.dir.clone();
        let target = _env.file();

        let start = Arc::new(std::sync::Barrier::new(4));
        let stop = Arc::new(std::sync::atomic::AtomicBool::new(false));

        let mut hands = Vec::new();
        for w in 0..3u32 {
            let start = Arc::clone(&start);
            let stop = Arc::clone(&stop);
            let target = target.clone();
            hands.push(std::thread::spawn(move || {
                let mut s = Stored {
                    key: Key::new(format!("writer {w}"), 8, 6),
                    probed: true,
                    sizes: BTreeMap::new(),
                };
                for i in 0..64u32 {
                    s.sizes.insert(
                        N + i as usize,
                        SizeStat { gpu_n: w + 1, gpu_secs: 1.0, gpu_units: 1.0, ..Default::default() },
                    );
                }
                start.wait();
                let mut wrote = 0;
                while !stop.load(AO::Relaxed) {
                    if write_atomic_json(&target, &s).is_ok() {
                        wrote += 1;
                    }
                    if wrote > 40 {
                        break;
                    }
                }
                wrote
            }));
        }

        // A reader racing the writers.
        start.wait();
        let mut reads = 0;
        let mut parsed = 0;
        let until = std::time::Instant::now() + Dur::from_millis(400);
        while std::time::Instant::now() < until {
            if let Ok(text) = std::fs::read_to_string(&target) {
                reads += 1;
                if serde_json::from_str::<Stored>(&text).is_ok() {
                    parsed += 1;
                } else {
                    panic!("a reader observed a torn file of {} bytes", text.len());
                }
            }
        }
        stop.store(true, AO::Relaxed);
        let wrote: u32 = hands.into_iter().map(|h| h.join().unwrap()).sum();

        assert!(wrote >= 3, "the writers did not overlap: {wrote} publications");
        assert!(reads > 0, "the reader never saw the file");
        assert_eq!(reads, parsed, "a read did not parse");
        // Whatever won, it is exactly one writer's file and not a blend.
        let final_ = serde_json::from_str::<Stored>(&std::fs::read_to_string(&target).unwrap())
            .expect("the surviving file should parse");
        assert!(
            final_.key.adapter.starts_with("writer "),
            "the surviving file is not one of the writers': {:?}",
            final_.key.adapter
        );
        assert_eq!(final_.sizes.len(), 64, "the surviving file is incomplete");
        let strays: Vec<_> = std::fs::read_dir(&dir)
            .unwrap()
            .filter_map(|e| e.ok())
            .map(|e| e.file_name().to_string_lossy().to_string())
            .filter(|n| n.contains("tmp-"))
            .collect();
        assert!(strays.is_empty(), "left {strays:?}");
    }

    // ── closeout §2: the four remaining lifecycle defects ───────────────

    /// **§2.1.** An execution whose state has been replaced must not be able to
    /// put it back on its *next* group.
    ///
    /// Rejecting the old completion was never enough. After the rejection A
    /// keeps running: its next `use_device` misses the state fast path, because
    /// its key no longer matches, and the old code let it claim initialization
    /// and reload its own pool's file over B's newer calibration.
    #[test]
    fn an_old_execution_cannot_initialize_on_its_next_group() {
        let _env = Env::new("old_exec_next_group");

        // A starts on two workers and holds a decision inside its analysis.
        let a = enter_exec(2);
        let gen_a = ensure_current().expect("A should initialize");
        let scope_a = begin_group();
        let _ = use_device(N);
        scope_a.work_started();
        let mut held = PENDING.with(|p| p.borrow_mut().take()).expect("A decided");
        let prev_a = (EXEC.with(|e| e.get()), EXEC_EPOCH.with(|e| e.get()));

        // B starts on one worker, initializes, and trains.
        let b = enter_exec(1);
        let gen_b = ensure_current().expect("B should initialize");
        assert_ne!(gen_a, gen_b);
        for _ in 0..MIN_SAMPLES {
            group(N, Outcome::Device, 1.0);
            group(N, Outcome::Cores, 2.0);
        }
        flush();
        let b_key = _env.live(|c| c.stored.key.clone()).unwrap();
        let b_stat = _env.stat(N).expect("B trained");
        let b_disk = _env.on_disk().expect("B flushed");
        leave_exec(b);
        leave_exec(prev_a);

        // A resumes. Its held sample is refused, as before.
        assert_eq!(
            with_current(held.generation, |c| c.finish(&mut held, UNITS, 9.0, Outcome::Device)),
            None,
            "A's held completion reached the new set"
        );

        // And now the part that was broken: A begins ANOTHER group.
        let scope2 = begin_group();
        let routed = use_device(N2);
        scope2.work_started();
        let landed = record(UNITS, 5.0, Outcome::Device);
        drop(scope2);

        assert!(!routed, "an obsolete execution was given a device route");
        assert!(
            landed.is_none() || landed == Some(Landed::Dropped(Dropped::StaleGeneration)),
            "an obsolete execution trained something: {landed:?}"
        );
        assert_eq!(
            _env.live(|c| c.stored.key.clone()).unwrap(),
            b_key,
            "A's next group rekeyed the live state"
        );
        assert_eq!(
            _env.live(|c| c.generation).unwrap(),
            gen_b,
            "A's next group issued a new generation over B's"
        );
        assert_eq!(_env.stat(N), Some(b_stat), "A's next group changed B's numbers");
        assert_eq!(
            _env.on_disk().map(|s| s.sizes),
            Some(b_disk.sizes),
            "A's next group rewrote B's file"
        );
        assert!(
            _env.stat(N2).is_none(),
            "A's next group created an entry for its own size"
        );
        let _ = a;
    }

    /// The same shape after changing away and back — and here the boundary is
    /// narrower, deliberately.
    ///
    /// Once the live key matches A's again, A's *lookups* succeed: the state it
    /// finds is keyed to a two-worker pool and A really is running on one, so a
    /// fresh decision it takes now carries the **current** generation and its
    /// timing is comparable. That is a lookup, not initialization, and it is
    /// allowed.
    ///
    /// What must still not happen is A gaining **initialization** rights — it
    /// must not issue a generation, rekey, or reload its own file over the live
    /// state. And a decision A was already holding from before the round trip
    /// stays refused, which `changing_workers_away_and_back_does_not_revive_the_old_execution`
    /// covers.
    #[test]
    fn an_old_execution_stays_obsolete_after_a_round_trip() {
        let _env = Env::new("old_exec_round_trip");
        let a = enter_exec(2);
        ensure_current().expect("A initializes");
        let prev_a = (EXEC.with(|e| e.get()), EXEC_EPOCH.with(|e| e.get()));

        let b = enter_exec(1);
        ensure_current().expect("B initializes");
        leave_exec(b);
        let c2 = enter_exec(2);
        let gen_back = ensure_current().expect("back on two workers");
        for _ in 0..MIN_SAMPLES {
            group(N, Outcome::Device, 1.0);
        }
        let trained = _env.stat(N).expect("the newest execution trained");
        leave_exec(c2);
        leave_exec(prev_a);

        // A's key now matches the live one. Its lookup succeeds and its fresh
        // decision carries the current generation — that is intended.
        let scope = begin_group();
        let _ = use_device(N);
        scope.work_started();
        let landed = record(UNITS, 9.0, Outcome::Cores);
        drop(scope);
        assert_eq!(
            landed,
            Some(Landed::Cores),
            "a matching key should still be readable by an older execution"
        );

        // The boundary that matters: no new generation, no rekey, no reload.
        assert_eq!(
            _env.live(|c| c.generation).unwrap(),
            gen_back,
            "an older execution issued a generation over the newest one"
        );
        assert_eq!(
            _env.live(|c| c.stored.key.clone()).unwrap().workers,
            2,
            "the live key changed"
        );
        let now = _env.stat(N).unwrap();
        assert_eq!(
            now.gpu_n, trained.gpu_n,
            "the older execution's cores sample disturbed the device side"
        );
        assert_eq!(now.cpu_n, trained.cpu_n + 1, "the cores sample did not land once");
        let _ = a;
    }

    /// **§2.2.** The twelfth attempt is an attempt, not an epitaph: it may be
    /// the sample that settles the size.
    ///
    /// Eleven attempts are spent on outcomes that teach nothing or teach one
    /// side, and the twelfth supplies the third device sample. Under the old
    /// code `decide` latched exhaustion the moment it reserved that twelfth,
    /// so the verdict it was about to produce was refused by the size it was
    /// measuring.
    #[test]
    fn the_twelfth_attempt_may_still_settle_a_size() {
        let _env = Env::new("twelfth_settles");
        as_analysis(|| {
            // 6 undersized: charged, measure nothing.
            for _ in 0..6 {
                let scope = begin_group();
                let _ = use_device(N);
                scope.work_started();
                assert_eq!(
                    record(1.0, 1.0, Outcome::Cores),
                    Some(Landed::Dropped(Dropped::NotMeasurable))
                );
                drop(scope);
            }
            // 3 measurable cores.
            for _ in 0..MIN_SAMPLES {
                assert_eq!(group(N, Outcome::Cores, 2.0), Some(Landed::Cores));
            }
            // 2 measurable device.
            for _ in 0..2 {
                assert_eq!(group(N, Outcome::Device, 1.0), Some(Landed::Device));
            }
            let s = _env.stat(N).unwrap();
            assert_eq!(s.attempts, 11, "setup: eleven attempts should be spent");
            assert!(!s.exhausted, "setup: the budget is not spent yet");

            // Attempt 12: the third device sample.
            assert_eq!(
                group(N, Outcome::Device, 1.0),
                Some(Landed::Device),
                "the twelfth attempt was refused"
            );
        });
        let s = _env.stat(N).unwrap();
        assert_eq!(s.attempts, MAX_ATTEMPTS);
        assert_eq!((s.gpu_n, s.cpu_n), (3, 3));
        assert!(!s.exhausted, "a size that settled was also marked exhausted");
        assert_eq!(
            s.verdict(),
            Verdict::Settled(true),
            "the twelfth attempt settled the size and the verdict did not follow"
        );
    }

    /// An unstarted twelfth reservation, abandoned, leaves eleven — and eleven
    /// is not exhaustion, before or after a reload.
    #[test]
    fn an_abandoned_twelfth_reservation_does_not_exhaust_the_size() {
        let _env = Env::new("twelfth_refunded");
        as_analysis(|| {
            for _ in 0..11 {
                cancelled_after_start(N);
            }
            assert_eq!(_env.stat(N).map(|s| s.attempts), Some(11));
            assert!(!_env.stat(N).unwrap().exhausted, "eleven is not exhaustion");

            // Reserve the twelfth and abandon it before the work starts.
            cancelled_before_start(N);
        });
        let s = _env.stat(N).unwrap();
        assert_eq!(s.attempts, 11, "the refund did not land");
        assert_eq!(s.outstanding, 0, "a reservation was left outstanding");
        assert!(!s.exhausted, "a refunded reservation left the size exhausted");
        assert!(s.may_reserve(), "a refunded reservation ended exploration");

        flush();
        invalidate();
        *STATE.lock().unwrap() = None;
        as_analysis(ensure_current);
        let s = _env.stat(N).unwrap();
        assert_eq!(s.attempts, 11, "the reload changed the count");
        assert!(!s.exhausted, "the reload resurrected exhaustion");
        assert_eq!(s.outstanding, 0, "outstanding reservations were persisted");
    }

    /// Genuine terminal exhaustion still holds, and ordinary work still cannot
    /// revive it. The boundary fix must not weaken this.
    #[test]
    fn terminal_exhaustion_is_still_terminal() {
        let _env = Env::new("still_terminal");
        as_analysis(|| {
            for _ in 0..MAX_ATTEMPTS {
                cancelled_after_start(N);
            }
            let s = _env.stat(N).unwrap();
            assert_eq!(s.attempts, MAX_ATTEMPTS);
            assert_eq!(s.outstanding, 0);
            assert!(s.exhausted, "the budget was spent with no outcome and did not latch");
            assert!(!s.may_reserve());

            for _ in 0..MIN_SAMPLES + 2 {
                group(N, Outcome::Cores, 5.0);
            }
        });
        assert_eq!(
            _env.stat(N).unwrap().verdict(),
            Verdict::Inconclusive,
            "ordinary work revived a genuinely exhausted size"
        );
    }

    /// **§2.3.** A nested analysis on the same worker must not retire the
    /// enclosing group's live decision.
    ///
    /// Reached by a controlled re-entrant call rather than by hoping the
    /// scheduler interleaves: a Rayon worker that blocks inside one analysis
    /// can run another *inside* it, which is the same work-stealing behind the
    /// probe deadlock.
    #[test]
    fn a_nested_analysis_does_not_retire_the_outer_decision() {
        let _env = Env::new("nested_scope");
        as_analysis(|| {
            let outer = begin_group();
            let _ = use_device(N);
            outer.work_started();

            // Inner group, a different size, through the production scope.
            within_exec(|| {
                let inner = begin_group();
                let _ = use_device(N2);
                inner.work_started();
                assert_eq!(record(UNITS, 3.0, Outcome::Device), Some(Landed::Device));
                drop(inner);
            });

            // The outer decision must still be here, and must be the outer one.
            assert_eq!(
                record(UNITS, 7.0, Outcome::Cores),
                Some(Landed::Cores),
                "the nested group retired the outer decision"
            );
            drop(outer);
        });
        let a = _env.stat(N).expect("the outer size has a stat");
        let b = _env.stat(N2).expect("the inner size has a stat");
        assert_eq!(
            (a.gpu_n, a.cpu_n),
            (0, 1),
            "the outer timing did not land at its own size"
        );
        assert_eq!(a.cpu_secs, 7.0, "the outer timing landed with the wrong number");
        assert_eq!(
            (b.gpu_n, b.cpu_n),
            (1, 0),
            "the inner timing did not land at its own size"
        );
        assert_eq!(b.gpu_secs, 3.0);
    }

    /// The same, with the inner group cancelled, unwound, or forced — none of
    /// which may disturb the enclosing decision.
    #[test]
    fn a_nested_group_that_does_not_record_leaves_the_outer_decision_alone() {
        for case in ["cancelled", "unwound", "always", "off", "below_floor"] {
            let _env = Env::new("nested_quiet");
            as_analysis(|| {
                let outer = begin_group();
                let _ = use_device(N);
                outer.work_started();

                within_exec(|| match case {
                    "cancelled" => {
                        let inner = begin_group();
                        let _ = use_device(N2);
                        inner.work_started();
                        drop(inner);
                    }
                    "unwound" => {
                        let hit = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
                            let inner = begin_group();
                            let _ = use_device(N2);
                            inner.work_started();
                            panic!("simulated failure inside a nested group");
                        }));
                        assert!(hit.is_err(), "setup: the panic should have been caught");
                    }
                    "always" | "off" => {
                        let was = mode();
                        set_mode(if case == "always" { GpuMode::Always } else { GpuMode::Off });
                        let inner = begin_group();
                        let _ = use_device(N2);
                        inner.work_started();
                        let _ = record(UNITS, 1.0, Outcome::Device);
                        drop(inner);
                        set_mode(was);
                    }
                    _ => {
                        let inner = begin_group();
                        let _ = use_device(SMALL);
                        inner.work_started();
                        let _ = record(UNITS, 1.0, Outcome::Cores);
                        drop(inner);
                    }
                });

                assert_eq!(
                    record(UNITS, 7.0, Outcome::Cores),
                    Some(Landed::Cores),
                    "{case}: the nested group disturbed the outer decision"
                );
                drop(outer);
            });
            let a = _env.stat(N).expect("the outer size has a stat");
            assert_eq!(a.cpu_secs, 7.0, "{case}: the outer timing is wrong");
        }
    }

    /// **§2.4.** A second initializer arriving in the window between the state
    /// fast-path miss and the claim must not replace what the first published.
    ///
    /// Counting probe calls cannot see this: reloading an already-probed file
    /// runs no probe and still loses every sample that has not been flushed.
    #[test]
    fn a_second_initializer_does_not_replace_freshly_published_state() {
        let _env = Env::new("second_initializer");
        let fired = Arc::new(AtomicU32::new(0));
        let count = Arc::clone(&fired);

        // B runs inside A's window, on the same key, and records a sample it
        // does NOT flush.
        *inject::BEFORE_CLAIM.lock().unwrap() = Some(Box::new(move || {
            count.fetch_add(1, AO::SeqCst);
            ensure_current().expect("B should initialize");
            let scope = begin_group();
            let _ = use_device(N);
            scope.work_started();
            let _ = record(UNITS, 1.0, Outcome::Device);
            drop(scope);
        }));

        let returned = as_analysis(ensure_current);
        *inject::BEFORE_CLAIM.lock().unwrap() = None;

        assert_eq!(fired.load(AO::SeqCst), 1, "the window hook never fired");
        let s = _env.stat(N).expect("B's sample should be live");
        assert_eq!(s.gpu_n, 1, "B's unflushed sample was lost");
        assert_eq!(
            returned,
            Some(_env.live(|c| c.generation).unwrap()),
            "A returned a generation that is not the live one"
        );
        assert!(
            _env.on_disk().is_none_or(|d| d.sizes.get(&N).is_none_or(|x| x.gpu_n == 0)),
            "setup: B's sample should not have reached the disk"
        );
    }

    // ── stale execution authority ───────────────────────────────────────────

    /// **The closeout blocker.** An execution superseded between its authority
    /// check and its claim must not publish over what replaced it.
    ///
    /// `GENERATION` cannot reject this one. A claims *after* B has finished, so
    /// A holds the newest generation there is — it issued it — and a
    /// generation-only publication test passes while A's key, A's measurements
    /// and A's file are the stale ones.
    ///
    /// B runs on a **different** pool size, so the keys differ and A cannot be
    /// mistaken for a same-key re-entry; and B records an identifiable
    /// measurement and flushes it, so there are live samples and file contents
    /// worth protecting.
    #[test]
    fn a_superseded_execution_cannot_publish_over_the_one_that_replaced_it() {
        const A_WORKERS: usize = 2;
        const B_WORKERS: usize = 1;
        const B_SECS: f64 = 3.25;

        let _env = Env::new("stale_authority");
        let fired = Arc::new(AtomicU32::new(0));
        let taken = Arc::new(Mutex::new(None::<(u64, Vec<u8>, Key)>));
        let count = Arc::clone(&fired);
        let sample = Arc::clone(&taken);
        let file = _env.file();

        // B is newer, on a different pool, and completes entirely inside A's
        // window: it initializes, records, and flushes to disk.
        *inject::BEFORE_CLAIM.lock().unwrap() = Some(Box::new(move || {
            count.fetch_add(1, AO::SeqCst);
            with_exec(B_WORKERS, || {
                ensure_current().expect("B should initialize");
                let _ = group(N, Outcome::Device, B_SECS);
                flush();
            });
            let g = STATE.lock().unwrap_or_else(|e| e.into_inner());
            let c = g.as_ref().expect("B published state");
            *sample.lock().unwrap() = Some((
                c.generation,
                std::fs::read(&file).expect("B wrote a file"),
                c.stored.key.clone(),
            ));
        }));

        let returned = with_exec(A_WORKERS, ensure_current);
        *inject::BEFORE_CLAIM.lock().unwrap() = None;

        assert_eq!(fired.load(AO::SeqCst), 1, "the window hook never fired");
        let (b_gen, b_bytes, b_key) = taken.lock().unwrap().take().expect("B recorded nothing");
        assert_eq!(b_key.workers, B_WORKERS, "setup: B's key is not B's pool");

        assert!(
            returned.is_none(),
            "a superseded execution was handed a calibration generation"
        );
        assert_eq!(
            _env.live(|c| c.stored.key.clone()),
            Some(b_key),
            "a superseded execution replaced the live key"
        );
        assert_eq!(
            _env.live(|c| c.generation),
            Some(b_gen),
            "a superseded execution replaced the live generation"
        );
        let s = _env.stat(N).expect("B's measurement should still be live");
        assert_eq!(s.gpu_n, 1, "a superseded execution replaced B's measurements");
        assert_eq!(s.gpu_secs, B_SECS, "a superseded execution replaced B's timing");
        assert_eq!(
            std::fs::read(_env.file()).expect("the file should still be there"),
            b_bytes,
            "a superseded execution overwrote the calibration file"
        );
        assert!(_env.temporaries().is_empty(), "left {:?}", _env.temporaries());
    }

    /// The same window, on the **same** pool size: served, not replaced.
    ///
    /// The keys match here, so the right answer is not a refusal but a lookup —
    /// B's generation, with B's unflushed samples intact. Guards the rejection
    /// above from being a blanket one that costs the caller its state.
    #[test]
    fn a_superseded_execution_is_still_served_when_the_key_matches() {
        let _env = Env::new("same_pool_supersede");
        let fired = Arc::new(AtomicU32::new(0));
        let count = Arc::clone(&fired);

        *inject::BEFORE_CLAIM.lock().unwrap() = Some(Box::new(move || {
            count.fetch_add(1, AO::SeqCst);
            with_exec(2, || {
                ensure_current().expect("B should initialize");
                let _ = group(N, Outcome::Device, 9.5);
            });
        }));
        let returned = with_exec(2, ensure_current);
        *inject::BEFORE_CLAIM.lock().unwrap() = None;

        assert_eq!(fired.load(AO::SeqCst), 1, "the window hook never fired");
        assert_eq!(
            returned,
            _env.live(|c| c.generation),
            "A was not served the live generation for its own key"
        );
        assert_eq!(
            _env.stat(N).map(|s| s.gpu_secs),
            Some(9.5),
            "B's unflushed measurement was lost"
        );
        assert!(
            _env.on_disk().is_none_or(|d| d.sizes.get(&N).is_none_or(|x| x.gpu_n == 0)),
            "setup: B's sample should not have reached the disk"
        );
    }

    /// **The probe window.** A newer execution that starts while A probes finds
    /// the claim taken and **declines rather than waiting** — so at the moment
    /// A finishes there is nothing newer published, and the contract lets A
    /// publish. The newer execution then replaces it on its own next attempt,
    /// with no wedged claim and no reset in between.
    ///
    /// This is the boundary the authority contract names: a newer execution
    /// *exists* but has not *replaced* anything. Rejecting A here instead is
    /// what breaks `concurrent_first_access_probes_once_and_nobody_waits`,
    /// where every participant supersedes every other by construction.
    #[test]
    fn a_newer_execution_during_a_probe_declines_and_then_takes_over() {
        const A_WORKERS: usize = 2;
        const B_WORKERS: usize = 1;

        let _env = Env::new("probe_window");
        let fired = Arc::new(AtomicU32::new(0));
        let declined = Arc::new(AtomicU32::new(0));
        let count = Arc::clone(&fired);
        let no = Arc::clone(&declined);
        let once = Arc::new(AtomicBool::new(false));

        // Inside A's probe, which by design holds no calibration lock.
        *inject::PROBE.lock().unwrap() = Some(Box::new(move || {
            if once.swap(true, AO::SeqCst) {
                return;
            }
            count.fetch_add(1, AO::SeqCst);
            with_exec(B_WORKERS, || {
                if ensure_current().is_none() {
                    no.fetch_add(1, AO::SeqCst);
                }
            });
        }));

        let a = with_exec(A_WORKERS, ensure_current).expect("A should publish");
        assert_eq!(fired.load(AO::SeqCst), 1, "the probe hook never fired");
        assert_eq!(
            declined.load(AO::SeqCst),
            1,
            "the newer execution waited on the claim instead of declining"
        );
        assert_eq!(
            _env.live(|c| c.stored.key.workers),
            Some(A_WORKERS),
            "A did not publish its own key"
        );

        // No wedged claim: the newer execution initializes immediately
        // afterwards and replaces A's state, because their keys differ.
        let b = with_exec(B_WORKERS, ensure_current).expect("B cannot initialize after A");
        assert_ne!(a, b, "B was handed A's generation");
        assert_eq!(
            _env.live(|c| c.stored.key.workers),
            Some(B_WORKERS),
            "B did not replace A's state"
        );

        // A *new* 2-worker analysis is a different execution and may key back
        // to two workers; that is not the defect and is not asserted against
        // here. The stale case — an execution that passed its authority check
        // before B published — is
        // `a_superseded_execution_cannot_publish_over_the_one_that_replaced_it`,
        // and `with_exec` cannot express it because it always begins a new
        // execution.
    }

    /// The publication predicate itself, over its three bounded cases.
    ///
    /// Covered directly because the `STATE`-side test is unreachable through
    /// the public entry points today — the claim is held as a lock across the
    /// publication section, so nothing newer can publish in between. It is
    /// defence in depth for that section, and it is tested as such rather than
    /// left as an untested branch.
    #[test]
    fn the_publication_predicate_rejects_only_the_superseded() {
        let _env = Env::new("publish_predicate");
        let _guard = STATE.lock().unwrap_or_else(|e| e.into_inner());
        let was = LAST_PUBLISHED_EXEC.swap(7, AO::SeqCst);
        assert!(
            superseded_by_a_publication(0),
            "a thread outside any execution was allowed to publish"
        );
        assert!(
            superseded_by_a_publication(6),
            "an execution older than the last publication was allowed to publish"
        );
        assert!(
            !superseded_by_a_publication(7),
            "the execution that published last was refused its own slot"
        );
        assert!(
            !superseded_by_a_publication(8),
            "a newer execution was refused"
        );
        LAST_PUBLISHED_EXEC.store(was, AO::SeqCst);
    }

    /// **The late boundary, forced.** An execution superseded *after* the early
    /// authority check has already passed must be rejected at publication,
    /// before it saves or installs anything.
    ///
    /// This schedule is unreachable through `BEFORE_CLAIM`, which fires
    /// *before* the early check: a newer execution publishing there is seen by
    /// that check and the caller never reaches the claim. The window this test
    /// needs is the one after the early check releases `STATE` and before the
    /// claim takes `INIT` — a gap that exists because INIT-then-STATE is the
    /// only permitted lock order, so `STATE` cannot be held across the
    /// acquisition. `AFTER_AUTHORITY_CHECK` sits in exactly that gap.
    ///
    /// A therefore passes the early check, takes a claim, loads, probes, and is
    /// turned away only by the check at publication.
    ///
    /// **The generation allocator is deliberately not asserted on.** A may
    /// legitimately consume a generation while taking its claim; what has to
    /// survive is the published state, not the counter.
    #[test]
    fn a_publication_superseded_after_the_early_check_is_rejected_late() {
        const A_WORKERS: usize = 2;
        const B_WORKERS: usize = 1;
        const C_WORKERS: usize = 3;
        const B_SECS: f64 = 4.75;

        let _env = Env::new("late_authority");

        // Counted rather than assumed: reaching the probe is what shows A got
        // past the claim, and so that the late check is the only thing left
        // that can have stopped it.
        let probes = Arc::new(AtomicU32::new(0));
        let counter = Arc::clone(&probes);
        *inject::PROBE.lock().unwrap() = Some(Box::new(move || {
            counter.fetch_add(1, AO::SeqCst);
        }));

        let fired = Arc::new(AtomicU32::new(0));
        let taken = Arc::new(Mutex::new(None::<(u64, Vec<u8>, Key)>));
        let hits = Arc::clone(&fired);
        let sample = Arc::clone(&taken);
        let file = _env.file();

        // B is newer, on a different pool, and finishes entirely inside the
        // gap: it initializes, records an identifiable measurement, and flushes
        // it to disk.
        *inject::AFTER_AUTHORITY_CHECK.lock().unwrap() = Some(Box::new(move || {
            hits.fetch_add(1, AO::SeqCst);
            with_exec(B_WORKERS, || {
                ensure_current().expect("B should initialize");
                let _ = group(N, Outcome::Device, B_SECS);
                flush();
            });
            let g = STATE.lock().unwrap_or_else(|e| e.into_inner());
            let c = g.as_ref().expect("B published state");
            *sample.lock().unwrap() = Some((
                c.generation,
                std::fs::read(&file).expect("B wrote a file"),
                c.stored.key.clone(),
            ));
        }));

        let returned = with_exec(A_WORKERS, ensure_current);
        *inject::AFTER_AUTHORITY_CHECK.lock().unwrap() = None;

        assert_eq!(fired.load(AO::SeqCst), 1, "the late-boundary hook never fired");
        let (b_gen, b_bytes, b_key) = taken.lock().unwrap().take().expect("B recorded nothing");
        assert_eq!(b_key.workers, B_WORKERS, "setup: B's key is not B's pool");

        // One probe for B, one for A. A probing at all means it passed the
        // early check and took the claim, which is the schedule this test
        // exists to force.
        assert_eq!(
            probes.load(AO::SeqCst),
            2,
            "A never reached the late boundary: it was stopped before the claim"
        );

        assert!(
            returned.is_none(),
            "an execution superseded after the early check published anyway"
        );
        assert_eq!(
            _env.live(|c| c.stored.key.clone()),
            Some(b_key),
            "the late rejection did not protect B's live key"
        );
        assert_eq!(
            _env.live(|c| c.generation),
            Some(b_gen),
            "the late rejection did not protect B's live generation"
        );
        let s = _env.stat(N).expect("B's measurement should still be live");
        assert_eq!(s.gpu_n, 1, "B's measurements were replaced");
        assert_eq!(s.gpu_secs, B_SECS, "B's timing was replaced");
        assert_eq!(
            std::fs::read(_env.file()).expect("the file should still be there"),
            b_bytes,
            "a rejected initializer wrote the calibration file"
        );
        assert!(_env.temporaries().is_empty(), "left {:?}", _env.temporaries());

        // And the claim it took is not parked behind it: a third identity,
        // which has to initialize rather than hit the fast path, can.
        let c_gen = with_exec(C_WORKERS, ensure_current)
            .expect("the rejected initializer left its claim parked");
        assert_eq!(
            _env.live(|c| c.generation),
            Some(c_gen),
            "the generation handed back is not the live one"
        );
        assert_eq!(
            _env.live(|c| c.stored.key.workers),
            Some(C_WORKERS),
            "the third identity did not publish its own state"
        );
    }

    // ── identity and file handling, unchanged in intent ─────────────────────

    /// The state written under a key is the state read back under the same key,
    /// through the production load and save.
    #[test]
    fn a_matching_key_round_trips() {
        let _env = Env::new("round_trip");
        as_analysis(|| {
            group(N, Outcome::Device, 1.0);
            group(N, Outcome::Cores, 2.0);
        });
        flush();
        let live = _env.stat(N).expect("a stat should exist");
        let on_disk = _env.on_disk().expect("a file should exist");
        assert_eq!(on_disk.sizes.get(&N), Some(&live), "the numbers did not survive the file");
        assert!(_env.temporaries().is_empty(), "left {:?}", _env.temporaries());
    }

    /// Each component of the identity, changed on its own, discards the state.
    #[test]
    fn every_identity_component_invalidates_on_its_own() {
        let _env = Env::new("identity");
        as_analysis(|| {
            group(N, Outcome::Device, 1.0);
        });
        flush();
        let base = _env.live(|c| c.stored.key.clone()).expect("a key should exist");
        assert!(!_env.on_disk().unwrap().sizes.is_empty(), "setup: something was written");

        let variants = [
            ("schema", Key { schema: base.schema + 1, ..base.clone() }),
            ("producer", Key { producer: base.producer + 1, ..base.clone() }),
            ("adapter", Key { adapter: "Other Adapter".into(), ..base.clone() }),
            ("cores", Key { cores: base.cores + 1, ..base.clone() }),
            ("workers", Key { workers: base.workers + 1, ..base.clone() }),
        ];
        for (what, k) in variants {
            let got = Calib::load(_env.file(), k, 1);
            assert!(
                got.stored.sizes.is_empty(),
                "a change of {what} alone kept the previous machine's numbers"
            );
        }
    }

    /// A file that is missing, truncated, not JSON, or carrying impossible
    /// statistics is the same case as no file.
    #[test]
    fn unusable_files_are_refused_and_start_over() {
        let _env = Env::new("corrupt");
        as_analysis(|| {
            group(N, Outcome::Device, 1.0);
        });
        flush();
        let good = std::fs::read_to_string(_env.file()).unwrap();
        let key = _env.live(|c| c.stored.key.clone()).unwrap();

        let cases: Vec<(&str, String)> = vec![
            ("truncated", good[..good.len() / 2].to_string()),
            ("not json", "{{{".into()),
            ("empty", String::new()),
            ("wrong shape", r#"{"key":{},"sizes":[]}"#.into()),
            ("negative time", good.replace("\"gpu_secs\": 1.0", "\"gpu_secs\": -1.0")),
            ("count without time", good.replace("\"gpu_secs\": 1.0", "\"gpu_secs\": 0.0")),
            ("infinite units", good.replace(&format!("\"gpu_units\": {UNITS:?}"), "\"gpu_units\": 1e999")),
        ];
        for (what, text) in cases {
            assert_ne!(text, good, "{what}: the fixture is the untouched good file");
            std::fs::write(_env.file(), &text).unwrap();
            let got = Calib::load(_env.file(), key.clone(), 1);
            assert!(got.stored.sizes.is_empty(), "{what} was adopted");
            assert!(!got.stored.probed, "{what} was treated as already probed");
        }

        std::fs::write(_env.file(), &good).unwrap();
        assert!(!Calib::load(_env.file(), key, 1).stored.sizes.is_empty());
    }

    /// The legacy file is not this schema's to touch.
    #[test]
    fn the_legacy_file_is_left_byte_for_byte_alone() {
        let _env = Env::new("legacy");
        let old = br#"{"adapter":"Old","cores":8,"probed":true,"sizes":{}}"#;
        std::fs::write(_env.legacy(), old).unwrap();

        as_analysis(|| {
            group(N, Outcome::Device, 1.0);
        });
        flush();

        assert_eq!(std::fs::read(_env.legacy()).unwrap(), old, "the legacy file was modified");
        assert!(_env.file().exists(), "the new file was not written");
        assert_ne!(_env.file(), _env.legacy(), "the two schemas share a filename");
    }

    // ── the decision rule ───────────────────────────────────────────────────

    /// Real work outranks the probe, and only once there is enough of it.
    #[test]
    fn real_samples_override_the_probe_once_there_are_enough() {
        let mut s = SizeStat { probe_ok: true, ..Default::default() };
        assert_eq!(s.verdict(), Verdict::Sampling(true));

        s.gpu_secs = 1.0; s.gpu_units = 1.0; s.gpu_n = MIN_SAMPLES;
        assert_eq!(s.verdict(), Verdict::Sampling(true), "one side is not evidence");

        s.cpu_secs = 2.0; s.cpu_units = 1.0; s.cpu_n = MIN_SAMPLES;
        assert_eq!(s.verdict(), Verdict::Settled(true));

        s.cpu_secs = 0.5;
        assert_eq!(s.verdict(), Verdict::Settled(false));

        s.cpu_secs = 1.05;
        assert_eq!(s.verdict(), Verdict::Settled(false), "ADOPT_MARGIN moved");
    }

    /// A probe that said no must still get the device measured. This calls the
    /// production rule; it does not restate it.
    #[test]
    fn a_rejected_probe_still_gets_the_device_measured() {
        let mut s = SizeStat { probe_ok: false, ..Default::default() };
        assert!(!s.next_route(), "a negative probe should be tried on cores first");

        s.cpu_n = MIN_SAMPLES; s.cpu_secs = 2.0; s.cpu_units = 1.0;
        assert!(s.next_route(), "a rejected size must still be measured on the device");

        s.gpu_n = MIN_SAMPLES; s.gpu_secs = 1.0; s.gpu_units = 1.0;
        assert!(s.next_route(), "real evidence should overturn the probe");
        assert_eq!(s.verdict(), Verdict::Settled(true));
    }

    /// A clean exploration settles on the unchanged margin, through the
    /// production entry points.
    #[test]
    fn a_clean_exploration_settles_on_the_unchanged_margin() {
        let _env = Env::new("explore");
        as_analysis(|| {
            for _ in 0..MIN_SAMPLES {
                assert_eq!(group(N, Outcome::Device, 1.0), Some(Landed::Device));
                assert_eq!(group(N, Outcome::Cores, 2.0), Some(Landed::Cores));
            }
            assert_eq!(_env.stat(N).unwrap().verdict(), Verdict::Settled(true));

            // 5 % cheaper is inside ADOPT_MARGIN and must not adopt.
            for _ in 0..MIN_SAMPLES {
                group(N2, Outcome::Device, 0.95);
                group(N2, Outcome::Cores, 1.0);
            }
            assert_eq!(_env.stat(N2).unwrap().verdict(), Verdict::Settled(false));
        });
    }

    /// The mode at the decision is the one that counts, not the mode when the
    /// work finishes.
    #[test]
    fn the_mode_at_the_decision_is_the_one_that_counts() {
        let _env = Env::new("mode_change");
        as_analysis(|| {
            // Decided under Auto; the user switches to Always mid-group.
            let scope = begin_group();
            set_mode(GpuMode::Auto);
            let _ = use_device(N);
            scope.work_started();
            set_mode(GpuMode::Always);
            assert_eq!(record(UNITS, 1.0, Outcome::Device), Some(Landed::Device));
            drop(scope);

            // Decided under Always; the user switches back to Auto mid-group.
            let scope = begin_group();
            set_mode(GpuMode::Always);
            let _ = use_device(N);
            scope.work_started();
            set_mode(GpuMode::Auto);
            let landed = record(UNITS, 0.001, Outcome::Device);
            drop(scope);
            assert!(
                landed.is_none(),
                "a forced group recorded after a switch to Auto: {landed:?}"
            );
        });
        assert_eq!(_env.stat(N).unwrap().gpu_n, 1, "only the Auto group should have landed");
    }

    /// Off and Always are settled before any state is consulted.
    #[test]
    fn off_and_always_are_settled_without_consulting_the_state() {
        assert_eq!(forced_route(GpuMode::Off, N), Some(false), "Off routed to the device");
        assert_eq!(forced_route(GpuMode::Always, N), Some(true), "Always did not route");
        assert_eq!(
            forced_route(GpuMode::Always, SMALL),
            Some(false),
            "Always routed a size below the routing floor"
        );
        assert_eq!(forced_route(GpuMode::Auto, N), None, "Auto answered without measuring");

        assert_eq!(exec_workers(), 0, "setup: this thread is not an analysis pool");
        assert!(ensure_current().is_none(), "state was reachable off an execution thread");
    }

    /// Sizes below production's routing floor create neither entries nor
    /// exploration work.
    #[test]
    fn sizes_below_the_routing_floor_are_not_learned_about() {
        let _env = Env::new("floor");
        as_analysis(|| {
            let scope = begin_group();
            let routed = use_device(SMALL);
            assert!(!routed, "a below-floor size was routed to the device");
            scope.work_started();
            drop(scope);
            assert!(
                _env.live(|c| c.stored.sizes.is_empty()).unwrap_or(true),
                "a below-floor size created an entry"
            );

            group(N, Outcome::Device, 1.0);
            assert!(_env.stat(N).is_some(), "an eligible size created no entry");
        });
    }

    /// Off stays off, even with a settled device verdict on disk.
    #[test]
    fn a_disabled_device_is_not_enabled_by_the_new_schema() {
        let _env = Env::new("off");
        as_analysis(|| {
            for _ in 0..MIN_SAMPLES {
                group(N, Outcome::Device, 1.0);
                group(N, Outcome::Cores, 9.0);
            }
            assert_eq!(_env.stat(N).unwrap().verdict(), Verdict::Settled(true));
            set_mode(GpuMode::Off);
            let scope = begin_group();
            assert!(!use_device(N), "Off routed to the device");
            drop(scope);
        });
    }

    /// Writes are debounced; a verdict change and an explicit flush are not.
    #[test]
    fn writes_are_debounced_but_a_verdict_change_is_not() {
        let _env = Env::new("debounce");
        as_analysis(|| {
            group(N, Outcome::Device, 1.0);
        });
        flush();
        let before = std::fs::read_to_string(_env.file()).unwrap();

        // An ordinary sample: the debounce holds it.
        as_analysis(|| {
            group(N, Outcome::Device, 1.0);
        });
        assert_eq!(
            std::fs::read_to_string(_env.file()).unwrap(),
            before,
            "the debounce did not hold the write"
        );
        assert!(_env.live(|c| c.dirty).unwrap_or(false), "a held write left the state clean");

        // Enough to move the verdict: that write is not debounced.
        as_analysis(|| {
            for _ in 0..MIN_SAMPLES {
                group(N, Outcome::Device, 1.0);
                group(N, Outcome::Cores, 2.0);
            }
        });
        assert!(!_env.live(|c| c.dirty).unwrap_or(true), "a verdict change was not written through");
        assert_ne!(
            std::fs::read_to_string(_env.file()).unwrap(),
            before,
            "the verdict change never reached the file"
        );
        assert!(_env.temporaries().is_empty(), "left {:?}", _env.temporaries());
    }

    /// Reading the status must not need the lock a probe holds.
    #[test]
    fn reading_the_status_never_blocks_behind_the_state_lock() {
        let _env = Env::new("describe_nonblocking");
        let held = STATE.lock().unwrap();
        assert_eq!(describe(), vec!["  (measuring…)".to_string()]);
        let before = workers();
        // The worker setting touches neither STATE nor anything a probe holds.
        set_workers(before);
        assert_eq!(workers(), before, "the worker setting moved");
        drop(held);
    }

    /// The panel says "never measured" rather than naming a winner.
    #[test]
    fn an_inconclusive_size_says_so_rather_than_naming_a_winner() {
        let _env = Env::new("describe");
        as_analysis(|| {
            for _ in 0..MAX_ATTEMPTS {
                cancelled_after_start(N);
            }
        });
        let lines = _env.live(describe_calibration).expect("state should exist");
        let row = lines.iter().find(|l| l.contains("2^18")).expect("no row for the size");
        assert!(row.contains("inconclusive"), "an unmeasured size did not say so: {row}");
        assert!(lines[0].contains("workers"), "the identity is not shown: {}", lines[0]);
    }
}
