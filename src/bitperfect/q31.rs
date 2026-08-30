//! Float32 to signed Q1.31: the proof, the cache, and the runtime guard.
//!
//! # What this is for
//!
//! A DAC that advertises "32-bit" almost always means 32-bit *integer*. A
//! Float32 source has no exact route to it — IEEE-754 and Q1.31 are different
//! lattices, and the general conversion rounds. But a great many Float32 files
//! are not general: they are 16- or 24-bit masters that were exported as
//! floats, so every sample already sits exactly on an integer grid. For those
//! the conversion changes the representation and not one numerical value.
//!
//! That is worth having, and it is worth naming precisely. It is **not**
//! bit-perfect: signed zero and NaN payloads do not survive, and a listener
//! told "bit-perfect" would be told something false. The contract calls it
//! [`Fidelity::ValueExact`](super::state::Fidelity::ValueExact), and it never
//! renders the green diamond.
//!
//! # Why a whole-track scan
//!
//! One packet proves nothing about the next. A track can be exactly on-grid for
//! ten minutes and then contain one sample that is not — a fade, a dither
//! stage, a plugin that ran in floating point. Claiming value-exactness from a
//! priming packet would be the same class of error as claiming payload-exactness
//! from a preference.
//!
//! So the proof is a complete decode of the selected track, off the UI and
//! audio threads, and the result is cached against the file's identity. The
//! cache is advisory only: [`Q31Guard`] re-checks every packet before it is
//! published, because a cache can be stale and a wrong sample must never reach
//! the device.

use std::path::{Path, PathBuf};

// ---------------------------------------------------------------------------
// The predicate
// ---------------------------------------------------------------------------

/// Full-scale for Q1.31: `2^31`.
const Q31_SCALE: f64 = 2_147_483_648.0;

/// Why a sample cannot be represented exactly in Q1.31.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum Q31Reject {
    /// NaN, in any payload.
    NotFinite,
    /// At or above `+1.0`, which Q1.31 cannot hold.
    AboveRange,
    /// Below `-1.0`.
    BelowRange,
    /// In range and finite, but between two Q1.31 steps.
    OffGrid,
}

impl Q31Reject {
    pub fn describe(&self) -> &'static str {
        match self {
            Q31Reject::NotFinite => "not a number",
            Q31Reject::AboveRange => "at or above +1.0, which Q1.31 cannot represent",
            Q31Reject::BelowRange => "below -1.0",
            Q31Reject::OffGrid => "between two Q1.31 steps",
        }
    }
}

/// Whether `x` lands exactly on the Q1.31 lattice, and where.
///
/// The arithmetic is done in `f64`. Every `f32` converts to `f64` exactly, and
/// `f64` has 53 bits of mantissa against the 32 the result needs, so the
/// multiply is exact and the integrality test is a real test rather than a
/// rounding artefact. Doing it in `f32` would make the question unanswerable:
/// the product would already have been rounded.
///
/// `+1.0` is deliberately rejected. Q1.31 runs from `-1.0` to `+1.0 - 2^-31`,
/// so `+1.0` has no representation — clamping it to `i32::MAX` is a change of
/// value, which is exactly what this predicate exists to detect.
#[inline]
pub fn to_q31_exact(x: f32) -> Result<i32, Q31Reject> {
    if x.is_nan() {
        return Err(Q31Reject::NotFinite);
    }
    if x.is_infinite() {
        return Err(if x > 0.0 {
            Q31Reject::AboveRange
        } else {
            Q31Reject::BelowRange
        });
    }
    let d = x as f64;
    if d >= 1.0 {
        return Err(Q31Reject::AboveRange);
    }
    if d < -1.0 {
        return Err(Q31Reject::BelowRange);
    }
    let scaled = d * Q31_SCALE;
    if scaled.fract() != 0.0 {
        return Err(Q31Reject::OffGrid);
    }
    // In `-2^31 ..= 2^31 - 1` by the range checks above, so this is lossless.
    let i = scaled as i64;
    if i < i32::MIN as i64 || i > i32::MAX as i64 {
        return Err(Q31Reject::AboveRange);
    }
    let i = i as i32;
    // The round trip is part of the claim, not a belt-and-braces extra: the
    // promise is that the number the file holds and the number the device
    // receives are the same number.
    if (i as f64 / Q31_SCALE) as f32 != x {
        return Err(Q31Reject::OffGrid);
    }
    Ok(i)
}

/// The deterministic lossy conversion, for when the user permits processing.
///
/// Never mixed with the exact path and never called on the audio thread. The
/// order matters and is fixed:
///
/// 1. NaN becomes zero — there is no defensible sample to emit for it, and
///    passing the bit pattern through to an integer field is noise at full
///    scale.
/// 2. `-∞` and anything below `-1` clamp to `i32::MIN`.
/// 3. `+∞`, `+1.0` and anything above the Q1.31 maximum clamp to `i32::MAX`.
/// 4. Everything else is multiplied in `f64` and rounded ties-to-even.
///
/// No dither. Q1.31 is fine enough that the rounding error is far below any
/// plausible noise floor, and adding a random signal to something the user
/// asked to be as close as possible is not an improvement.
#[inline]
pub fn to_q31_processed(x: f32) -> i32 {
    if x.is_nan() {
        return 0;
    }
    let d = x as f64;
    if d <= -1.0 {
        return i32::MIN;
    }
    let scaled = d * Q31_SCALE;
    if scaled >= i32::MAX as f64 {
        return i32::MAX;
    }
    // `round_ties_even` rather than `round`: away-from-zero rounding biases a
    // signal that sits exactly between two steps, and half-step values are
    // common in material that was resampled.
    let r = scaled.round_ties_even();
    if r <= i32::MIN as f64 {
        i32::MIN
    } else if r >= i32::MAX as f64 {
        i32::MAX
    } else {
        r as i32
    }
}

// ---------------------------------------------------------------------------
// Whole-track verdict
// ---------------------------------------------------------------------------

/// What a complete scan of one track concluded.
#[derive(Clone, PartialEq, Debug)]
pub struct Q31Verdict {
    /// True when every sample in the track lands on the Q1.31 lattice.
    pub value_exact: bool,
    /// Samples examined, across every channel.
    pub samples: u64,
    /// The first sample that failed, if any: frame, channel, and why.
    pub first_failure: Option<Q31Failure>,
    /// The predicate/schema version this verdict was produced under.
    pub schema: u32,
}

#[derive(Clone, Copy, PartialEq, Debug)]
pub struct Q31Failure {
    pub frame: u64,
    pub channel: u16,
    pub reason: Q31Reject,
    /// The offending value, for the log.
    pub value: f32,
}

/// Bumped whenever the predicate or the record layout changes, so a verdict
/// produced by an older build is a cache miss rather than a wrong answer.
pub const SCHEMA: u32 = 1;

/// Examine one packet's worth of interleaved Float32 canonical payloads.
///
/// Accumulates into `verdict`, so a caller can drive it packet by packet across
/// a whole track without holding the track in memory. `frame_base` is the
/// frame index this packet starts at.
pub fn scan_packet(canon: &[u32], channels: u16, frame_base: u64, verdict: &mut Q31Verdict) {
    let ch = channels.max(1) as usize;
    for (i, &c) in canon.iter().enumerate() {
        let x = f32::from_bits(c);
        if let Err(reason) = to_q31_exact(x) {
            if verdict.first_failure.is_none() {
                verdict.first_failure = Some(Q31Failure {
                    frame: frame_base + (i / ch) as u64,
                    channel: (i % ch) as u16,
                    reason,
                    value: x,
                });
            }
            verdict.value_exact = false;
        }
    }
    verdict.samples += canon.len() as u64;
}

impl Q31Verdict {
    /// A verdict that has examined nothing yet. `value_exact` starts true and
    /// is only ever falsified, so an empty track is vacuously exact — and a
    /// track with no samples has nothing to misrepresent.
    pub fn new() -> Self {
        Q31Verdict {
            value_exact: true,
            samples: 0,
            first_failure: None,
            schema: SCHEMA,
        }
    }

    /// One line for the log and the status panel.
    pub fn describe(&self) -> String {
        match (&self.first_failure, self.value_exact) {
            (None, true) => format!(
                "every one of {} samples lands exactly on the Q1.31 lattice",
                self.samples
            ),
            (Some(f), _) => format!(
                "sample {} of channel {} is {} ({})",
                f.frame,
                f.channel,
                f.value,
                f.reason.describe()
            ),
            (None, false) => "some samples are not representable in Q1.31".into(),
        }
    }
}

impl Default for Q31Verdict {
    fn default() -> Self {
        Self::new()
    }
}

// ---------------------------------------------------------------------------
// Cache identity
// ---------------------------------------------------------------------------

/// What a cached verdict is keyed on.
///
/// Deliberately more than the path. A file that is replaced in place keeps its
/// name, and a verdict for the old contents applied to the new ones is exactly
/// the failure mode the runtime guard exists to catch — better not to hand it
/// that job in the first place.
#[derive(Clone, PartialEq, Eq, Debug, Hash)]
pub struct Q31Key {
    pub path: PathBuf,
    pub len: u64,
    /// Modification time in nanoseconds since the epoch, at full resolution.
    pub mtime_ns: i128,
    pub track_id: u32,
    pub schema: u32,
}

impl Q31Key {
    /// Build a key from the file as it is right now. `None` if the file cannot
    /// be interrogated, which is a cache miss rather than an error.
    pub fn of(path: &Path, track_id: u32) -> Option<Q31Key> {
        let md = std::fs::metadata(path).ok()?;
        let mtime = md.modified().ok()?;
        let ns = match mtime.duration_since(std::time::UNIX_EPOCH) {
            Ok(d) => d.as_nanos() as i128,
            // Files dated before the epoch are unusual but not impossible.
            Err(e) => -(e.duration().as_nanos() as i128),
        };
        Some(Q31Key {
            path: std::fs::canonicalize(path).unwrap_or_else(|_| path.to_path_buf()),
            len: md.len(),
            mtime_ns: ns,
            track_id,
            schema: SCHEMA,
        })
    }
}

// ---------------------------------------------------------------------------
// Durable cache
// ---------------------------------------------------------------------------

/// A verdict as it is written to disk.
///
/// Flat and self-describing, so a record written by a build with a different
/// `SCHEMA` can be *recognised* and dropped rather than misread. The key's
/// components are stored alongside the verdict for the same reason: a record
/// whose identity does not reconstruct is discarded, not trusted.
#[derive(serde::Serialize, serde::Deserialize, Clone, Debug)]
struct CacheRecord {
    path: PathBuf,
    len: u64,
    mtime_ns: String,
    track_id: u32,
    schema: u32,
    value_exact: bool,
    samples: u64,
    /// Monotonically increasing use counter, for pruning. Not a wall clock:
    /// a clock that moves backwards would evict the wrong entries.
    used: u64,
}

/// How many verdicts the file keeps.
///
/// A scan is minutes of decode for a large library, so the cache is worth
/// keeping; it is also a file in the user's profile, so it does not grow
/// without limit. At this size the file is a few hundred kilobytes.
const CACHE_CAPACITY: usize = 4096;

const CACHE_FILE: &str = "q31_verdicts.json";

/// Verdicts that survive a restart.
///
/// A whole-track scan reads and converts every sample in the file. Discarding
/// that at exit meant re-scanning the same track on every launch, which is
/// minutes of decode for a library of hi-res Float32 masters and the reason
/// the label so often never arrived before the track ended.
///
/// The cache is advisory in exactly the way the in-memory one was: `Q31Guard`
/// re-checks every packet regardless, so a stale or corrupted entry can cost
/// a label but can never put an unrepresentable sample on the wire.
#[derive(Debug, Default)]
pub struct Q31Cache {
    entries: std::collections::HashMap<Q31Key, (Q31Verdict, u64)>,
    clock: u64,
    dirty: bool,
}

impl Q31Cache {
    pub fn new() -> Self {
        Self::default()
    }

    /// Load from `dir`, tolerating every way the file can be unusable.
    ///
    /// A missing file, unreadable permissions, truncated JSON, a record from a
    /// different schema, or a path that no longer resolves are all a cold
    /// cache. None of them is an error worth showing a user, and none of them
    /// may stop playback.
    pub fn load(dir: &Path) -> Self {
        let mut cache = Q31Cache::new();
        let Ok(text) = std::fs::read_to_string(dir.join(CACHE_FILE)) else {
            return cache;
        };
        let records: Vec<CacheRecord> = match serde_json::from_str(&text) {
            Ok(r) => r,
            Err(e) => {
                crate::mlog!("q31     verdict cache is unreadable and will be rebuilt: {e}");
                return cache;
            }
        };
        let mut dropped = 0usize;
        for r in records {
            // A record from another schema was produced by a different
            // predicate. It is recognisable, and it is not evidence.
            if r.schema != SCHEMA {
                dropped += 1;
                continue;
            }
            let Ok(mtime_ns) = r.mtime_ns.parse::<i128>() else {
                dropped += 1;
                continue;
            };
            let key = Q31Key {
                path: r.path,
                len: r.len,
                mtime_ns,
                track_id: r.track_id,
                schema: r.schema,
            };
            cache.clock = cache.clock.max(r.used);
            cache.entries.insert(
                key,
                (
                    Q31Verdict {
                        value_exact: r.value_exact,
                        samples: r.samples,
                        // Not persisted: it is a diagnostic about one sample
                        // in one scan, and a verdict is useful without it.
                        first_failure: None,
                        schema: r.schema,
                    },
                    r.used,
                ),
            );
        }
        if dropped > 0 {
            crate::mlog!("q31     dropped {dropped} cached verdict(s) from an older schema");
        }
        cache
    }

    /// Look a verdict up, and mark it as used so pruning keeps it.
    pub fn get(&mut self, key: &Q31Key) -> Option<Q31Verdict> {
        self.clock += 1;
        let clock = self.clock;
        let (verdict, used) = self.entries.get_mut(key)?;
        *used = clock;
        self.dirty = true;
        Some(verdict.clone())
    }

    pub fn insert(&mut self, key: Q31Key, verdict: Q31Verdict) {
        self.clock += 1;
        self.entries.insert(key, (verdict, self.clock));
        self.dirty = true;
    }

    pub fn len(&self) -> usize {
        self.entries.len()
    }

    pub fn is_empty(&self) -> bool {
        self.entries.is_empty()
    }

    /// Drop the least recently used entries down to `CACHE_CAPACITY`.
    fn prune(&mut self) {
        if self.entries.len() <= CACHE_CAPACITY {
            return;
        }
        let mut used: Vec<u64> = self.entries.values().map(|(_, u)| *u).collect();
        // The `used` value of the newest entry that must go.
        let cut = self.entries.len() - CACHE_CAPACITY;
        used.sort_unstable();
        let threshold = used[cut - 1];
        self.entries.retain(|_, (_, u)| *u > threshold);
    }

    /// Write the cache to `dir`, atomically.
    ///
    /// Written to a temporary file in the same directory and renamed over the
    /// target, so a crash or a full disk during the write leaves the previous
    /// cache intact rather than a half-written file that the next launch would
    /// have to recognise as corrupt. A rename within one directory is atomic
    /// on every filesystem this runs on.
    ///
    /// Every failure is logged and swallowed. A cache that cannot be saved is
    /// a slower next launch, not a reason to interrupt anyone.
    pub fn save(&mut self, dir: &Path) {
        if !self.dirty {
            return;
        }
        self.prune();
        let records: Vec<CacheRecord> = self
            .entries
            .iter()
            .map(|(k, (v, used))| CacheRecord {
                path: k.path.clone(),
                len: k.len,
                // As a string: `i128` is outside what JSON numbers are
                // guaranteed to round-trip, and a truncated mtime is a key
                // that silently matches the wrong file.
                mtime_ns: k.mtime_ns.to_string(),
                track_id: k.track_id,
                schema: k.schema,
                value_exact: v.value_exact,
                samples: v.samples,
                used: *used,
            })
            .collect();
        let Ok(json) = serde_json::to_string(&records) else {
            return;
        };
        if let Err(e) = std::fs::create_dir_all(dir) {
            crate::mlog!("q31     could not create {}: {e}", dir.display());
            return;
        }
        let tmp = dir.join(format!("{CACHE_FILE}.tmp-{}", std::process::id()));
        if let Err(e) = std::fs::write(&tmp, json.as_bytes()) {
            crate::mlog!("q31     could not write the verdict cache: {e}");
            let _ = std::fs::remove_file(&tmp);
            return;
        }
        if let Err(e) = std::fs::rename(&tmp, dir.join(CACHE_FILE)) {
            crate::mlog!("q31     could not replace the verdict cache: {e}");
            let _ = std::fs::remove_file(&tmp);
            return;
        }
        self.dirty = false;
    }
}
// ---------------------------------------------------------------------------
// Runtime guard
// ---------------------------------------------------------------------------

/// Re-checks every packet before it is published, whatever the cache said.
///
/// The cache is advisory. A file can be replaced between the scan and playback,
/// a scan can have been produced by a build with a different predicate, and a
/// storage layer can hand back something other than what was written. None of
/// those may put an unrepresentable sample on the wire, so the conversion
/// itself carries the check: it converts and validates in the same pass, and
/// the first sample that fails stops the packet.
#[derive(Debug, Default)]
pub struct Q31Guard {
    frames_seen: u64,
    /// Samples that were *not* exactly representable in Q1.31.
    ///
    /// The guard used to refuse the packet and fault the session when it found
    /// one of these. That made an ordinary off-grid Float32 file — the common
    /// case, not the exotic one — stop dead on a route the user had explicitly
    /// permitted to process. The conversion is lossy by construction and was
    /// labelled Processed from the moment it opened; a sample that needs the
    /// lossy path is that route working, not that route failing.
    ///
    /// What the count is for is the *label*: a track may only be called
    /// value-exact if a complete scan proved it and this observed it, and one
    /// sample here is enough to withhold the claim for the rest of the
    /// session.
    off_grid: u64,
    /// The first sample that was not exactly representable, for diagnostics.
    first_off_grid: Option<Q31Failure>,
}

impl Q31Guard {
    pub fn new() -> Self {
        Q31Guard {
            frames_seen: 0,
            off_grid: 0,
            first_off_grid: None,
        }
    }

    /// Convert one packet of Float32 canonical payloads into Q1.31 canonical
    /// payloads, in place.
    ///
    /// Always succeeds: a sample on the Q1.31 lattice converts exactly, and
    /// one that is not takes the deterministic rounded path. Nothing here can
    /// stop playback, and nothing here decides a badge — it *records* what was
    /// needed so the badge can be decided honestly elsewhere.
    pub fn convert(&mut self, canon: &mut [u32], channels: u16) {
        let ch = channels.max(1) as usize;
        for (i, c) in canon.iter_mut().enumerate() {
            let x = f32::from_bits(*c);
            match to_q31_exact(x) {
                Ok(v) => *c = v as u32,
                Err(reason) => {
                    self.off_grid += 1;
                    if self.first_off_grid.is_none() {
                        self.first_off_grid = Some(Q31Failure {
                            frame: self.frames_seen + (i / ch) as u64,
                            channel: (i % ch) as u16,
                            reason,
                            value: x,
                        });
                    }
                    *c = to_q31_processed(x) as u32;
                }
            }
        }
        self.frames_seen += (canon.len() / ch) as u64;
    }

    /// Whether every sample this guard has seen was exactly representable.
    ///
    /// A precondition for the value-exact label, never on its own a proof of
    /// it: the guard has only seen what has played, and the claim is about the
    /// whole track. The complete scan supplies the rest.
    pub fn all_on_grid(&self) -> bool {
        self.off_grid == 0
    }

    pub fn off_grid(&self) -> u64 {
        self.off_grid
    }

    pub fn first_off_grid(&self) -> Option<&Q31Failure> {
        self.first_off_grid.as_ref()
    }

    pub fn frames_seen(&self) -> u64 {
        self.frames_seen
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    /// The values the contract names, in both directions.
    #[test]
    fn the_predicate_accepts_and_rejects_exactly_what_it_should() {
        // Must pass.
        assert_eq!(to_q31_exact(-1.0), Ok(i32::MIN));
        assert_eq!(to_q31_exact(0.0), Ok(0));
        assert_eq!(
            to_q31_exact(-0.0),
            Ok(0),
            "negative zero is numerically zero"
        );
        assert_eq!(to_q31_exact(f32::from_bits(0x3000_0000)), Ok(1), "+2^-31");
        assert_eq!(to_q31_exact(-f32::from_bits(0x3000_0000)), Ok(-1), "-2^-31");
        // The largest float below +1.0 is 1 - 2^-24, which is exactly on the
        // lattice. `i32::MAX / 2^31` is *not* reachable from f32 at all — it
        // rounds up to 1.0, which Q1.31 cannot hold — so the positive maximum
        // an f32 source can actually present is this one.
        let largest_below_one = f32::from_bits(0x3F7F_FFFF);
        assert!(largest_below_one < 1.0);
        assert_eq!(to_q31_exact(largest_below_one), Ok(i32::MAX - 127));
        assert_eq!(
            to_q31_exact((i32::MAX as f64 / Q31_SCALE) as f32),
            Err(Q31Reject::AboveRange),
            "the Q1.31 maximum is not an f32; it rounds to +1.0"
        );

        // Must fail.
        assert_eq!(to_q31_exact(1.0), Err(Q31Reject::AboveRange));
        assert_eq!(to_q31_exact(1.5), Err(Q31Reject::AboveRange));
        assert_eq!(to_q31_exact(-1.000_001), Err(Q31Reject::BelowRange));
        assert_eq!(to_q31_exact(f32::INFINITY), Err(Q31Reject::AboveRange));
        assert_eq!(to_q31_exact(f32::NEG_INFINITY), Err(Q31Reject::BelowRange));
        assert_eq!(
            to_q31_exact(f32::from_bits(0x2F80_0000)),
            Err(Q31Reject::OffGrid),
            "2^-32"
        );
        assert_eq!(
            to_q31_exact(-f32::from_bits(0x2F80_0000)),
            Err(Q31Reject::OffGrid)
        );
        // Subnormals are far below the Q1.31 step, so none of them are on grid.
        assert_eq!(
            to_q31_exact(f32::from_bits(0x0000_0001)),
            Err(Q31Reject::OffGrid)
        );
        assert_eq!(to_q31_exact(f32::MIN_POSITIVE), Err(Q31Reject::OffGrid));
        // Every NaN, not just the quiet one.
        for bits in [0x7FC0_0000u32, 0x7F80_0001, 0xFFC0_1234, 0xFF80_0001] {
            assert_eq!(
                to_q31_exact(f32::from_bits(bits)),
                Err(Q31Reject::NotFinite),
                "{bits:#010x}"
            );
        }
    }

    /// Material exported from 16- and 24-bit masters is the case this exists
    /// for: every lattice value must pass.
    #[test]
    fn the_pcm_lattices_are_all_value_exact() {
        for bits in [16u32, 24, 32] {
            let step = 1i64 << (32 - bits);
            let mut checked = 0u32;
            // Sweep the whole range at that depth, sampled.
            // Sample the range densely enough to be meaningful without
            // sweeping four billion values.
            let stride = step * (1 << 4);
            let mut v = i32::MIN as i64;
            while v <= i32::MAX as i64 - step {
                let x = (v as f64 / Q31_SCALE) as f32;
                // A 24-bit-or-narrower lattice value always survives the trip
                // through f32; a 32-bit one generally does not, and that is the
                // reason this conversion needs proving per track rather than
                // per format.
                match to_q31_exact(x).map(|q| q as i64) {
                    Ok(got) if got == v => checked += 1,
                    other => {
                        assert!(
                            bits > 24,
                            "{bits}-bit lattice value {v} was not recovered: {other:?}"
                        );
                    }
                }
                v += stride;
            }
            assert!(checked > 1000, "{bits}-bit: only {checked} values examined");
        }
    }

    /// A property, over a wide spread of real bit patterns: whatever the
    /// predicate accepts must round-trip to the same number.
    #[test]
    fn every_accepted_sample_round_trips_numerically() {
        let mut s: u64 = 0x243F_6A88_85A3_08D3;
        let mut accepted = 0u32;
        for _ in 0..200_000 {
            s = s
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            let x = f32::from_bits((s >> 32) as u32);
            match to_q31_exact(x) {
                Ok(q) => {
                    accepted += 1;
                    let back = (q as f64 / Q31_SCALE) as f32;
                    assert_eq!(
                        back.to_bits(),
                        x.to_bits(),
                        "accepted {x} ({:#010x}) did not round-trip",
                        x.to_bits()
                    );
                }
                Err(_) => {
                    // Rejection must be justified: an accepted-looking value
                    // that fails the round trip is what the check exists for.
                    assert!(
                        !x.is_finite()
                            || !(-1.0..1.0).contains(&x)
                            || ((x as f64) * Q31_SCALE).fract() != 0.0,
                        "{x} was rejected without cause"
                    );
                }
            }
        }
        assert!(accepted > 0, "the sweep must accept something");
    }

    /// Signed zero passes the numerical test and must still not be called
    /// payload-exact: the two zeros are different bit patterns and only one
    /// comes back.
    #[test]
    fn signed_zero_is_value_exact_but_not_payload_exact() {
        assert_eq!(to_q31_exact(0.0), Ok(0));
        assert_eq!(to_q31_exact(-0.0), Ok(0));
        // The distinction is lost, which is the whole reason this is a
        // different fidelity state.
        assert_ne!((0.0f32).to_bits(), (-0.0f32).to_bits());
        let recovered = (0i32 as f64 / Q31_SCALE) as f32;
        assert_eq!(recovered.to_bits(), (0.0f32).to_bits());
        assert_ne!(recovered.to_bits(), (-0.0f32).to_bits());
    }

    /// The processed conversion is deterministic, clamps at both ends, and
    /// rounds halves to even.
    #[test]
    fn the_processed_conversion_is_deterministic_and_clamped() {
        assert_eq!(to_q31_processed(f32::NAN), 0);
        assert_eq!(
            to_q31_processed(f32::from_bits(0xFFC0_1234)),
            0,
            "any NaN payload"
        );
        assert_eq!(to_q31_processed(f32::NEG_INFINITY), i32::MIN);
        assert_eq!(to_q31_processed(f32::INFINITY), i32::MAX);
        assert_eq!(to_q31_processed(-1.0), i32::MIN);
        assert_eq!(to_q31_processed(-2.0), i32::MIN);
        assert_eq!(to_q31_processed(1.0), i32::MAX, "+1.0 has no Q1.31 value");
        assert_eq!(to_q31_processed(2.0), i32::MAX);
        assert_eq!(to_q31_processed(0.0), 0);
        assert_eq!(to_q31_processed(-0.0), 0);

        // Exactly representable values are not disturbed by the lossy path.
        for v in [0i32, 1, -1, 1 << 16, -(1 << 16), i32::MIN] {
            let x = (v as f64 / Q31_SCALE) as f32;
            if to_q31_exact(x) == Ok(v) {
                assert_eq!(to_q31_processed(x), v, "{v} moved on the processed path");
            }
        }

        // Ties round to even, in both signs. A value scaling to exactly 1.5
        // goes to 2, and one scaling to exactly 2.5 also goes to 2 — that is
        // what ties-to-even means, and it is why `round` (away from zero) would
        // bias material that sits between steps.
        let step = 1.0f64 / Q31_SCALE;
        assert_eq!(to_q31_processed((1.5 * step) as f32), 2);
        assert_eq!(to_q31_processed((2.5 * step) as f32), 2);
        assert_eq!(to_q31_processed((-1.5 * step) as f32), -2);
        assert_eq!(to_q31_processed((-2.5 * step) as f32), -2);
        assert_eq!(to_q31_processed((0.5 * step) as f32), 0);

        // Deterministic: the same input twice is the same output.
        let mut s: u64 = 1;
        for _ in 0..10_000 {
            s = s.wrapping_mul(6364136223846793005).wrapping_add(1);
            let x = f32::from_bits((s >> 32) as u32);
            assert_eq!(to_q31_processed(x), to_q31_processed(x));
        }
    }

    /// A whole-track scan reports the first failure with enough detail to find
    /// it, and does not stop looking at the first packet.
    #[test]
    fn a_scan_reports_the_first_failing_sample() {
        let mut v = Q31Verdict::new();
        // Two clean packets, then one with a single off-grid sample.
        let clean: Vec<u32> = [0.0f32, 0.5, -0.5, 0.25]
            .iter()
            .map(|x| x.to_bits())
            .collect();
        scan_packet(&clean, 2, 0, &mut v);
        scan_packet(&clean, 2, 2, &mut v);
        assert!(v.value_exact, "{}", v.describe());
        assert_eq!(v.samples, 8);

        let off = f32::from_bits(0x2F80_0000); // 2^-32
        let dirty: Vec<u32> = [0.0f32, 0.5, off, 0.25]
            .iter()
            .map(|x| x.to_bits())
            .collect();
        scan_packet(&dirty, 2, 4, &mut v);
        assert!(!v.value_exact);
        let f = v.first_failure.expect("a failure must be recorded");
        assert_eq!(f.frame, 5, "frame 4 + one whole frame into the packet");
        assert_eq!(f.channel, 0);
        assert_eq!(f.reason, Q31Reject::OffGrid);
        assert!(v.describe().contains("channel 0"), "{}", v.describe());

        // A later clean packet does not undo the verdict.
        scan_packet(&clean, 2, 6, &mut v);
        assert!(!v.value_exact, "one failure is enough, permanently");
    }

    /// An on-grid packet converts exactly and leaves the claim intact.
    #[test]
    fn an_on_grid_packet_converts_exactly_and_keeps_the_claim() {
        let mut g = Q31Guard::new();
        let mut good: Vec<u32> = [0.0f32, 0.5, -0.5, 0.25]
            .iter()
            .map(|x| x.to_bits())
            .collect();
        let before = good.clone();
        g.convert(&mut good, 2);
        assert_ne!(good, before, "the conversion must actually convert");
        assert_eq!(good[1] as i32, 1 << 30, "0.5 is 2^30 in Q1.31");
        assert!(g.all_on_grid(), "nothing here needed rounding");
        assert_eq!(g.off_grid(), 0);
        assert_eq!(g.frames_seen(), 2);
    }

    /// An off-grid sample is *played*, rounded, and costs the claim.
    ///
    /// The symptom this replaces: an ordinary Float32 file with one off-grid
    /// sample used to stop dead on a route the user had explicitly permitted
    /// to process, because the guard treated "not exactly representable" as a
    /// fault rather than as the reason the rounded path exists.
    #[test]
    fn an_off_grid_sample_is_played_rounded_and_costs_the_claim() {
        let off = f32::from_bits(0x2F80_0000); // 2^-32, below the Q1.31 step
        let mut g = Q31Guard::new();
        let mut mixed: Vec<u32> = [0.0f32, 0.5, off, 0.25]
            .iter()
            .map(|x| x.to_bits())
            .collect();
        g.convert(&mut mixed, 2);

        // Every sample was published, including the one that had to round.
        assert_eq!(mixed[0] as i32, 0);
        assert_eq!(mixed[1] as i32, 1 << 30);
        assert_eq!(
            mixed[2] as i32, 0,
            "2^-32 rounds to zero, it does not fault"
        );
        assert_eq!(mixed[3] as i32, 1 << 29);

        // And the claim is gone for the rest of the session.
        assert!(!g.all_on_grid());
        assert_eq!(g.off_grid(), 1);
        let f = g.first_off_grid().expect("the sample must be located");
        assert_eq!(f.frame, 1);
        assert_eq!(f.channel, 0);
        assert_eq!(f.reason, Q31Reject::OffGrid);
    }

    /// A stale positive verdict cannot produce a value-exact claim on its own:
    /// the guard's own observation is a second, independent condition.
    #[test]
    fn a_verdict_alone_never_earns_the_claim() {
        let stale = Q31Verdict {
            value_exact: true,
            samples: 1 << 20,
            first_failure: None,
            schema: SCHEMA,
        };
        assert!(stale.value_exact);

        // The samples in front of the guard say otherwise, and the guard says
        // so — without stopping the audio.
        let mut g = Q31Guard::new();
        let mut canon = vec![f32::NAN.to_bits(); 4];
        g.convert(&mut canon, 2);
        assert!(
            !g.all_on_grid(),
            "the guard must contradict a verdict the samples do not support"
        );
        assert!(
            canon.iter().all(|&c| c as i32 == 0),
            "a NaN payload is published as silence, not as a fault"
        );
    }

    // -- durable cache -----------------------------------------------------

    fn cache_dir(tag: &str) -> PathBuf {
        let d = std::env::temp_dir().join(format!("moosik-q31-{tag}-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&d);
        std::fs::create_dir_all(&d).unwrap();
        d
    }

    fn a_key(path: &str, len: u64) -> Q31Key {
        Q31Key {
            path: PathBuf::from(path),
            len,
            mtime_ns: -1_234_567_890_123_456_789i128,
            track_id: 0,
            schema: SCHEMA,
        }
    }

    fn a_verdict(value_exact: bool) -> Q31Verdict {
        Q31Verdict {
            value_exact,
            samples: 4096,
            first_failure: None,
            schema: SCHEMA,
        }
    }

    /// A verdict survives a restart, and the identity it is filed under
    /// survives with it — including a negative nanosecond mtime, which does
    /// not round-trip through a JSON number.
    #[test]
    fn a_verdict_survives_a_restart() {
        let dir = cache_dir("roundtrip");
        let key = a_key("/music/track.wav", 123_456);

        let mut c = Q31Cache::new();
        assert!(c.is_empty());
        c.insert(key.clone(), a_verdict(true));
        c.save(&dir);

        let mut back = Q31Cache::load(&dir);
        assert_eq!(back.len(), 1);
        let v = back.get(&key).expect("the same key must find it again");
        assert!(v.value_exact);
        assert_eq!(v.samples, 4096);

        // A key that differs in any component is a miss, not a near-match.
        assert!(back.get(&a_key("/music/track.wav", 123_457)).is_none());
        assert!(back.get(&a_key("/music/other.wav", 123_456)).is_none());
        let mut wrong_time = key.clone();
        wrong_time.mtime_ns += 1;
        assert!(back.get(&wrong_time).is_none());

        let _ = std::fs::remove_dir_all(&dir);
    }

    /// A cache file that cannot be read is a cold cache, never an error and
    /// never a crash. Every way it can be unusable ends the same way.
    #[test]
    fn an_unusable_cache_file_is_simply_cold() {
        // Missing directory.
        let missing =
            std::env::temp_dir().join(format!("moosik-q31-absent-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&missing);
        assert!(Q31Cache::load(&missing).is_empty());

        // Truncated JSON.
        let dir = cache_dir("corrupt");
        std::fs::write(dir.join(CACHE_FILE), b"[{\"path\":\"/a\",\"len\":1,").unwrap();
        assert!(
            Q31Cache::load(&dir).is_empty(),
            "a torn file is a cold cache"
        );

        // Valid JSON of the wrong shape.
        std::fs::write(dir.join(CACHE_FILE), b"{\"not\":\"an array\"}").unwrap();
        assert!(Q31Cache::load(&dir).is_empty());

        // Empty file.
        std::fs::write(dir.join(CACHE_FILE), b"").unwrap();
        assert!(Q31Cache::load(&dir).is_empty());

        let _ = std::fs::remove_dir_all(&dir);
    }

    /// A record written under a different predicate is recognised and dropped,
    /// not read as though it meant the same thing.
    #[test]
    fn a_verdict_from_another_schema_is_discarded() {
        let dir = cache_dir("schema");
        let json = format!(
            "[{{\"path\":\"/music/a.wav\",\"len\":10,\"mtime_ns\":\"5\",\"track_id\":0,\
              \"schema\":{},\"value_exact\":true,\"samples\":8,\"used\":1}}]",
            SCHEMA + 1
        );
        std::fs::write(dir.join(CACHE_FILE), json.as_bytes()).unwrap();
        assert!(
            Q31Cache::load(&dir).is_empty(),
            "a verdict produced by a different predicate is not evidence"
        );
        let _ = std::fs::remove_dir_all(&dir);
    }

    /// The file does not grow without limit, and what survives pruning is what
    /// was used most recently.
    #[test]
    fn the_cache_prunes_to_its_capacity_keeping_what_was_used() {
        let dir = cache_dir("prune");
        let mut c = Q31Cache::new();
        for i in 0..(CACHE_CAPACITY + 500) {
            c.insert(a_key(&format!("/music/{i}.wav"), i as u64), a_verdict(true));
        }
        // Touch an early entry so it is the most recently used of all.
        let survivor = a_key("/music/0.wav", 0);
        assert!(c.get(&survivor).is_some());
        c.save(&dir);

        let mut back = Q31Cache::load(&dir);
        assert_eq!(back.len(), CACHE_CAPACITY, "the file is capped");
        assert!(
            back.get(&survivor).is_some(),
            "the entry that was used most recently must survive"
        );
        // ...and an untouched early entry must not.
        assert!(back.get(&a_key("/music/1.wav", 1)).is_none());
        let _ = std::fs::remove_dir_all(&dir);
    }

    /// The write is atomic: an existing cache is never replaced by a partial
    /// one, and no temporary file is left behind.
    #[test]
    fn the_cache_is_replaced_atomically() {
        let dir = cache_dir("atomic");
        let mut first = Q31Cache::new();
        first.insert(a_key("/music/one.wav", 1), a_verdict(true));
        first.save(&dir);
        let before = std::fs::read(dir.join(CACHE_FILE)).unwrap();

        let mut second = Q31Cache::new();
        second.insert(a_key("/music/two.wav", 2), a_verdict(false));
        second.save(&dir);
        let after = std::fs::read(dir.join(CACHE_FILE)).unwrap();
        assert_ne!(before, after, "the save must have replaced the file");

        // Whatever is on disk parses completely — never a half-written file.
        let mut back = Q31Cache::load(&dir);
        assert_eq!(back.len(), 1);
        assert!(back.get(&a_key("/music/two.wav", 2)).is_some());

        // And nothing is left lying around.
        let strays: Vec<_> = std::fs::read_dir(&dir)
            .unwrap()
            .filter_map(|e| e.ok())
            .map(|e| e.file_name().to_string_lossy().into_owned())
            .filter(|n| n != CACHE_FILE)
            .collect();
        assert!(strays.is_empty(), "temporary files left behind: {strays:?}");

        let _ = std::fs::remove_dir_all(&dir);
    }

    /// A save with nothing to say does not rewrite the file.
    #[test]
    fn an_unchanged_cache_is_not_rewritten() {
        let dir = cache_dir("clean");
        let mut c = Q31Cache::new();
        c.insert(a_key("/music/a.wav", 1), a_verdict(true));
        c.save(&dir);
        let first = std::fs::metadata(dir.join(CACHE_FILE)).unwrap().len();

        c.save(&dir); // nothing changed
        assert!(dir.join(CACHE_FILE).exists());
        assert_eq!(
            std::fs::metadata(dir.join(CACHE_FILE)).unwrap().len(),
            first
        );
        let _ = std::fs::remove_dir_all(&dir);
    }

    /// The cache key changes when the file does, and a schema bump invalidates
    /// every existing verdict.
    #[test]
    fn the_cache_key_tracks_the_file_and_the_schema() {
        let dir = std::env::temp_dir().join(format!("moosik_q31_{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let p = dir.join("a.wav");
        std::fs::write(&p, b"first").unwrap();

        let k1 = Q31Key::of(&p, 0).expect("a real file has a key");
        assert_eq!(k1.len, 5);
        assert_eq!(k1.schema, SCHEMA);

        // Same bytes, same key.
        assert_eq!(Q31Key::of(&p, 0), Some(k1.clone()));
        // A different track in the same container is a different key.
        assert_ne!(Q31Key::of(&p, 1), Some(k1.clone()));

        // Replaced in place: the length alone catches this one, which is the
        // point of keying on more than the path.
        std::fs::write(&p, b"second and longer").unwrap();
        let k2 = Q31Key::of(&p, 0).unwrap();
        assert_ne!(k1, k2);

        // A missing file has no key, which is a miss rather than an error.
        std::fs::remove_file(&p).unwrap();
        assert_eq!(Q31Key::of(&p, 0), None);
        let _ = std::fs::remove_dir(&dir);
    }
}
