//! Source-time temporal smoothing.
//!
//! The spectrum has two producers running at unrelated rates. Real-time FFT
//! frames arrive when the analyser tick is let through the `max_fps` throttle;
//! pre-processed frames were computed at `pre_fps` and sit in a cache the UI
//! samples by playback position. Both then feed one exponential moving average
//! whose only parameter is a retention value from a slider.
//!
//! Applying that EMA once per *UI tick* makes the visible decay a function of
//! the repaint schedule: the same slider setting decays twice as fast on a
//! 120 Hz monitor as on a 60 Hz one, and in pre-process mode it re-smooths
//! whichever cached row the clock happens to land on — smoothing the same row
//! repeatedly when the UI outruns the cache, and skipping rows entirely when it
//! does not. On a 60 Hz display against the 180 fps analysis default, two of
//! every three analysed rows were never read at all; the exact fraction varied
//! with the monitor, which is itself the problem.
//!
//! "Read" is the operative word. Every row now reaches the temporal reducer —
//! it contributes to the filter state, and to the interval the peak marker and
//! the waterfall are derived from. That is not the same as every row being
//! drawn: the bar display still shows one value per repaint, the most recently
//! crossed one. What changed is that the rows in between now influence it
//! instead of being discarded.
//!
//! This module holds the arithmetic that fixes it, kept free of analyser state
//! so it can be tested directly.
//!
//! # The contract
//!
//! The pre-process value `s` is **old-value retention at a 60 Hz reference
//! tick**. 60 Hz is the unit the number is quoted in, chosen for being a round
//! rate in the middle of the range — not a rate anything is claimed to run at.
//!
//! This applies to **pre-process only**. Real-time keeps the accepted-tick
//! recurrence it has always had, and the two modes carry separate settings.
//! Making one number mean the same thing in both was a change nobody asked for
//! and it cost real-time its accepted response; the pre-process control was the
//! one that was dead.
//!
//! For an arbitrary elapsed time the equivalent retention is
//!
//! ```text
//! alpha = s ^ (60 * dt)
//! ```
//!
//! so 0.5 at the 180 Hz pre-process rate is `0.5^(1/3) = 0.793700526`, not the
//! 0.83 a linear guess suggests.
//!
//! `s == 0` means **off** — the displayed value is the incoming value, with no
//! filter state at all. That was not previously true in pre-process mode, which
//! ignored the slider and hard-coded `0.5`.

/// Reference tick rate the persisted smoothing value is defined at.
///
/// A unit of measurement, and nothing more. The setting used to mean
/// "retention per accepted analyser tick", which had no fixed meaning at all —
/// it depended on `max_fps`, on the monitor, and in pre-process mode on the
/// cache rate as well. Anchoring it to a stated rate gives it one.
///
/// 60 is the anchor because it is a round number in the middle of the range,
/// not because anything runs at it. `SpectrumWindow::new` starts `max_fps` at
/// 60, but the application overwrites it from the primary monitor's refresh
/// rate at startup, so on a 144 or 240 Hz display the shipping value is 144 or
/// 240 and always has been.
///
/// That has a consequence worth stating plainly: a saved smoothing value does
/// **not** reproduce its previous appearance on a monitor faster than 60 Hz. It
/// could not — the old behaviour had no single appearance to reproduce, since
/// the same number decayed at a different rate on every machine. What it does
/// now is mean the same thing everywhere, which is the property that was
/// missing.
pub const SMOOTHING_REFERENCE_HZ: f64 = 60.0;

/// How many cached source rows a single UI tick may consume before it stops
/// replaying and snaps instead.
///
/// Two very different things cross rows: ordinary scheduling, and
/// discontinuities. A seek is unbounded — a jump of five minutes at 180 rows a
/// second crosses 54 000 of them, and replaying those would freeze the UI to
/// render a filter trajectory nobody asked for.
///
/// Ordinary scheduling is *not* reliably under this limit, and an earlier
/// version of this comment claimed it was. The claim rested on `pre_fps` being
/// clamped to 480, which is true of the superlet path and irrelevant to the FFT
/// one: there the cache rate is derived, `sample_rate / hop`, and a 192 kHz
/// track at FFT 1024 with 87.5% overlap produces 1500 rows a second. Against a
/// 1 fps display that is 1500 rows in a tick, and it will snap.
///
/// 512 is therefore deliberate bounded degradation rather than a limit nothing
/// reaches: it caps the per-tick work at roughly 2.8 s of source time at 180
/// rows a second, and an extreme rate against an extreme display setting snaps
/// instead of freezing. Snapping loses filter continuity for one tick; the
/// alternative loses the frame.
pub const MAX_CATCHUP_FRAMES: usize = 512;

/// The rolling waterfall advances **once per producer update**: one row per
/// cached row consumed in Pre-process, one per accepted analyser tick in
/// Real-time.
///
/// It briefly did something else. A previous phase capped it at 60 rows a
/// second so the time axis would not depend on `max_fps` — a property nobody
/// had complained about — and on a ~180 Hz display that made the waterfall
/// scroll three times slower and, in Pre-process, drop two of every three
/// analysed rows. The bars showed them; the history did not.
///
/// So the rate is the producer's, and the *depth* is the setting: the user
/// picks how many seconds of history to keep, and the row count follows from
/// the rate. See `WATERFALL_SECS_RANGE` and `waterfall_rows_for`.
///
/// One row per update also means nothing analysed is discarded, which the
/// per-tick version never managed either: it pushed one row per repaint
/// whatever the cursor had crossed, so a fast display duplicated rows and a
/// slow one skipped them.
///
/// How long the visible history is, given a producer rate and a wanted span.
///
/// Clamped at both ends. The floor keeps the ring meaningful when the producer
/// is very slow; the ceiling bounds memory, because every row costs
/// `bar_count` floats on the CPU and a texture row on the GPU — at 1024 bars
/// that is 4 KB and 4 KB, so the ceiling is a few megabytes each.
pub fn waterfall_rows_for(secs: f32, producer_hz: f64) -> usize {
    if !secs.is_finite() || !producer_hz.is_finite() || secs <= 0.0 || producer_hz <= 0.0 {
        return WATERFALL_MIN_ROWS;
    }
    ((secs as f64 * producer_hz).round() as usize)
        .clamp(WATERFALL_MIN_ROWS, WATERFALL_MAX_ROWS)
}

/// Floor and ceiling on the ring, in rows.
///
/// Both bind, and when they do the span on screen is **not** the span that was
/// asked for — the row count is what the ring actually holds, so the time it
/// covers follows from it. 0.2 s against a 60 Hz producer wants 12 rows and gets
/// 32, which is 0.53 s; 8 s against a 400 Hz one wants 3200 and gets 2048, which
/// is 5.1 s. [`waterfall_span_secs`] is the figure to show a user, not the
/// slider position.
pub const WATERFALL_MIN_ROWS: usize = 32;
pub const WATERFALL_MAX_ROWS: usize = 2048;

/// The span a ring of `rows` actually covers at `producer_hz` rows a second.
///
/// The inverse of [`waterfall_rows_for`], and the honest counterpart to the
/// slider: it differs from the requested span whenever a bound binds.
pub fn waterfall_span_secs(rows: usize, producer_hz: f64) -> f64 {
    if !producer_hz.is_finite() || producer_hz <= 0.0 {
        return 0.0;
    }
    rows as f64 / producer_hz
}

/// Range the history-depth control offers, in seconds.
pub const WATERFALL_SECS_RANGE: std::ops::RangeInclusive<f32> = 0.2..=8.0;

/// Default history depth, in seconds.
///
/// 0.67 s is what the accepted 1.4.5 build produced on a ~180 Hz display, where
/// it pushed one row per repaint into a 120-row ring — nominally, since that
/// figure assumes the display actually reached 180 Hz. It is a starting point
/// rather than a claim: the control exists precisely because the right answer
/// depends on the display and on what is being looked for.
pub const DEFAULT_WATERFALL_SECS: f32 = 0.67;

/// Retention to apply for an elapsed time of `dt` seconds, given the persisted
/// 60 Hz-reference setting `s`.
///
/// Returns a value in `0.0..=1.0`, where 0 replaces the old value outright and
/// 1 keeps it unchanged.
///
/// Edge cases are decided here rather than at the call sites:
///
/// * `s <= 0`, and anything not finite, is **off**, and stays off for every
///   `dt` including a degenerate one — otherwise "no smoothing" would silently
///   acquire some.
/// * `s >= 1` would never release a value; it is held at 1 so the arithmetic
///   below cannot produce a number outside the unit interval.
/// * a non-finite or non-positive `dt` means no time has passed that we can
///   account for, so nothing decays.
/// * a non-finite result — which `powf` can produce from an input that slipped
///   through — falls back to off, because showing the current frame is always
///   defensible and showing `NaN` never is.
pub fn alpha_for_dt(s: f32, dt: f64) -> f32 {
    if !s.is_finite() || s <= 0.0 {
        // NaN and the infinities land here too. Garbage must not be read as
        // "a little smoothing"; showing the incoming frame is always
        // defensible, so anything unusable resolves to off.
        return 0.0;
    }
    if s >= 1.0 {
        return 1.0;
    }
    if !dt.is_finite() || dt <= 0.0 {
        return 1.0;
    }
    let alpha = (s as f64).powf(SMOOTHING_REFERENCE_HZ * dt);
    if alpha.is_finite() {
        alpha.clamp(0.0, 1.0) as f32
    } else {
        0.0
    }
}

/// What a UI tick should do with the pre-processed cache, given where the
/// cursor was and which row the playback clock now points at.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum FramePlan {
    /// The clock is still inside the row already consumed. Doing nothing is
    /// what makes a repeated tick idempotent.
    Idle,
    /// Consume `first..=last` in source order, one filter step each.
    Consume { first: usize, last: usize },
    /// Discontinuity, or further than [`MAX_CATCHUP_FRAMES`]. Adopt `to`
    /// outright and reset the temporal filter.
    Snap { to: usize },
}

/// Decide the [`FramePlan`] for one tick.
///
/// `cursor` is the last row consumed, or `None` when there is no filter state
/// to continue from — a fresh track, a cache replacement, a bar-count change.
///
/// Backwards movement is always a snap. It happens on a seek and at the wrap of
/// a looping track, and in both cases the rows between the two positions were
/// not played, so running the filter over them would be inventing history.
pub fn plan(cursor: Option<usize>, target: usize, limit: usize) -> FramePlan {
    match cursor {
        None => FramePlan::Snap { to: target },
        Some(c) if target == c => FramePlan::Idle,
        Some(c) if target < c => FramePlan::Snap { to: target },
        Some(c) => {
            // `target > c`, so this cannot wrap.
            if target - c > limit.max(1) {
                FramePlan::Snap { to: target }
            } else {
                FramePlan::Consume { first: c + 1, last: target }
            }
        }
    }
}

/// Row the playback clock points at, clamped to the frames that exist.
///
/// Returns `None` for an empty cache, so callers cannot construct a cursor into
/// nothing. A non-finite or negative position lands on row 0 rather than
/// wrapping through the `as usize` cast.
pub fn target_frame(elapsed_secs: f64, frame_rate: f64, n_frames: usize) -> Option<usize> {
    if n_frames == 0 {
        return None;
    }
    let last = n_frames - 1;
    if !elapsed_secs.is_finite() || !frame_rate.is_finite() || frame_rate <= 0.0 {
        return Some(0);
    }
    let idx = elapsed_secs * frame_rate;
    if idx <= 0.0 {
        return Some(0);
    }
    if idx >= last as f64 {
        return Some(last);
    }
    Some(idx as usize)
}

/// How long a step takes to travel 95% of the way, in seconds, for a
/// pre-process retention of `s`.
///
/// The number the user is actually choosing. `s^(60·t) = 0.05` gives
/// `t = ln(0.05) / (60·ln s)`, independent of the source rate — which is the
/// property the whole normalisation exists for, and the one that makes this
/// figure quotable in a tooltip.
///
/// It is also how the shipped default was chosen, and how the acceptance
/// failure was diagnosed: 0.75 is 174 ms, which is seven times the ~24 ms the
/// previous release delivered, and is exactly the lag the owner saw.
pub fn step_response_95_secs(s: f32) -> f64 {
    if !s.is_finite() || s <= 0.0 {
        return 0.0;
    }
    if s >= 1.0 {
        return f64::INFINITY;
    }
    (0.05f64).ln() / (SMOOTHING_REFERENCE_HZ * (s as f64).ln())
}

/// Default pre-process retention.
///
/// Chosen to reproduce the character of the release the owner accepted rather
/// than to be a round number. That release hard-coded `alpha = 0.5` per
/// accepted UI tick and ran at roughly 170–180 Hz on the owner's machine, so a
/// step reached 95% in about 24 ms.
///
/// `0.125^(60/180) = 0.5`, so at a 180 Hz source rate this produces exactly
/// that alpha per row — and because the conversion is in time rather than in
/// ticks, it holds the same ~24 ms at every other source rate too, which the
/// hard-coded version never did.
///
/// It is deliberately **not** the real-time default of 0.75. That value was
/// never active in pre-process: the path ignored the setting entirely, so
/// nothing was ever displayed with it and there is no appearance to preserve.
/// Inheriting it on upgrade would hand every existing user a 174 ms response
/// they never chose.
pub const DEFAULT_PRE_SMOOTHING: f32 = 0.125;

/// Source-time spacing of one cached frame, in seconds.
///
/// Guarded because it is the `dt` every catch-up step is filtered with, and a
/// zero or non-finite frame rate would otherwise make the whole cascade a
/// silent no-op.
pub fn source_dt(frame_rate: f64) -> f64 {
    if frame_rate.is_finite() && frame_rate > 0.0 {
        1.0 / frame_rate
    } else {
        0.0
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The reference rate is the fixed point: at exactly 60 Hz the value is
    /// used unchanged.
    ///
    /// That is a statement about the arithmetic, not about anyone's settings.
    /// It does **not** mean a saved preset keeps its previous appearance: the
    /// pre-process path never read this value at all before, and the animation
    /// cap is taken from the primary monitor at startup, so on a 144 or 240 Hz
    /// display the old behaviour was something this conversion cannot and does
    /// not reproduce.
    #[test]
    fn sixty_hertz_is_the_identity() {
        for s in [0.05f32, 0.25, 0.5, 0.75, 0.9, 0.97] {
            let a = alpha_for_dt(s, 1.0 / 60.0);
            assert!((a - s).abs() < 1e-6, "s={s} gave {a}");
        }
    }

    /// The exact figure the contract names, kept as a literal so a change to
    /// the formula cannot pass quietly.
    #[test]
    fn half_at_one_eighty_is_the_cube_root() {
        let a = alpha_for_dt(0.5, 1.0 / 180.0);
        assert!(
            (a - 0.793_700_5).abs() < 1e-6,
            "0.5 at 180 Hz must be 0.5^(1/3) = 0.793700526, got {a}"
        );
    }

    /// Three steps at 180 Hz must retain exactly as much as one at 60 Hz.
    /// This is the property the whole module exists for.
    #[test]
    fn retention_composes_across_rates() {
        let one = alpha_for_dt(0.75, 1.0 / 60.0);
        let three = alpha_for_dt(0.75, 1.0 / 180.0).powi(3);
        assert!((one - three).abs() < 1e-6, "{one} vs {three}");

        let four = alpha_for_dt(0.75, 1.0 / 240.0).powi(4);
        assert!((one - four).abs() < 1e-6, "{one} vs {four}");
    }

    #[test]
    fn zero_is_off_at_every_rate() {
        for dt in [0.0, 1.0 / 240.0, 1.0 / 60.0, 1.0, f64::NAN, f64::INFINITY, -1.0] {
            assert_eq!(alpha_for_dt(0.0, dt), 0.0, "dt={dt}");
        }
    }

    #[test]
    fn degenerate_inputs_are_explicit() {
        // No time passed: nothing decays.
        assert_eq!(alpha_for_dt(0.75, 0.0), 1.0);
        assert_eq!(alpha_for_dt(0.75, -0.5), 1.0);
        assert_eq!(alpha_for_dt(0.75, f64::NAN), 1.0);
        assert_eq!(alpha_for_dt(0.75, f64::INFINITY), 1.0);
        // Out-of-range retention is clamped rather than trusted.
        assert_eq!(alpha_for_dt(1.0, 1.0 / 60.0), 1.0);
        assert_eq!(alpha_for_dt(2.0, 1.0 / 60.0), 1.0);
        assert_eq!(alpha_for_dt(-1.0, 1.0 / 60.0), 0.0);
        assert_eq!(alpha_for_dt(f32::NAN, 1.0 / 60.0), 0.0);
        assert_eq!(alpha_for_dt(f32::INFINITY, 1.0 / 60.0), 0.0);
        assert_eq!(alpha_for_dt(f32::NEG_INFINITY, 1.0 / 60.0), 0.0);
        // A very long gap decays to nothing rather than overflowing.
        let a = alpha_for_dt(0.75, 1e6);
        assert!(a.is_finite() && a == 0.0, "{a}");
    }

    #[test]
    fn a_repeated_tick_is_idle() {
        assert_eq!(plan(Some(7), 7, MAX_CATCHUP_FRAMES), FramePlan::Idle);
    }

    #[test]
    fn forward_movement_consumes_every_crossed_row() {
        assert_eq!(
            plan(Some(7), 10, MAX_CATCHUP_FRAMES),
            FramePlan::Consume { first: 8, last: 10 }
        );
        assert_eq!(
            plan(Some(0), 1, MAX_CATCHUP_FRAMES),
            FramePlan::Consume { first: 1, last: 1 }
        );
    }

    #[test]
    fn no_cursor_snaps() {
        assert_eq!(plan(None, 0, MAX_CATCHUP_FRAMES), FramePlan::Snap { to: 0 });
        assert_eq!(plan(None, 900, MAX_CATCHUP_FRAMES), FramePlan::Snap { to: 900 });
    }

    #[test]
    fn backward_movement_snaps() {
        assert_eq!(plan(Some(500), 499, MAX_CATCHUP_FRAMES), FramePlan::Snap { to: 499 });
        assert_eq!(plan(Some(500), 0, MAX_CATCHUP_FRAMES), FramePlan::Snap { to: 0 });
    }

    /// The bound is on the operation count, not on the distance jumped.
    #[test]
    fn catch_up_is_bounded() {
        let at_limit = plan(Some(0), MAX_CATCHUP_FRAMES, MAX_CATCHUP_FRAMES);
        assert_eq!(at_limit, FramePlan::Consume { first: 1, last: MAX_CATCHUP_FRAMES });
        let past = plan(Some(0), MAX_CATCHUP_FRAMES + 1, MAX_CATCHUP_FRAMES);
        assert_eq!(past, FramePlan::Snap { to: MAX_CATCHUP_FRAMES + 1 });
        // Five minutes at 180 fps.
        assert_eq!(plan(Some(10), 54_010, MAX_CATCHUP_FRAMES), FramePlan::Snap { to: 54_010 });
    }

    /// The ordinary schedules replay rather than snapping.
    ///
    /// Note what this does *not* claim. The rates here are the ones a user
    /// meets in practice, not every rate the code can produce: the FFT path
    /// derives its cache rate from `sample_rate / hop` and reaches 1500 rows a
    /// second at 192 kHz with a small window, which against a very low display
    /// rate exceeds the limit on purpose.
    #[test]
    fn ordinary_schedules_replay_rather_than_snapping() {
        for source_hz in [24.0f64, 46.875, 60.0, 120.0, 180.0, 240.0, 480.0] {
            for display_hz in [24.0f64, 30.0, 60.0, 144.0, 180.0, 240.0] {
                let crossed = (source_hz / display_hz).ceil() as usize;
                let p = plan(Some(1000), 1000 + crossed, MAX_CATCHUP_FRAMES);
                assert!(
                    matches!(p, FramePlan::Consume { .. }),
                    "source={source_hz} display={display_hz} crossed={crossed} snapped"
                );
            }
        }
    }

    /// And the rates that do exceed it snap, deliberately, rather than
    /// attempting 1500 rows of filter cascade inside one repaint.
    ///
    /// 192 kHz at FFT 1024 and 87.5% overlap is a hop of 128 samples, so
    /// 1500 cached rows a second. A display pinned to 1 fps crosses all of them
    /// in a tick.
    #[test]
    fn an_extreme_rate_against_a_slow_display_snaps_rather_than_freezing() {
        let source_hz = 192_000.0 / 128.0;
        assert_eq!(source_hz, 1500.0, "the derived FFT cache rate this is about");
        for display_hz in [1.0f64, 2.0] {
            let crossed = (source_hz / display_hz).ceil() as usize;
            assert!(crossed > MAX_CATCHUP_FRAMES);
            assert!(
                matches!(
                    plan(Some(1000), 1000 + crossed, MAX_CATCHUP_FRAMES),
                    FramePlan::Snap { .. }
                ),
                "display={display_hz}: {crossed} rows in one tick must snap"
            );
        }
        // A sane display rate at the same source rate still replays.
        let crossed = (source_hz / 60.0).ceil() as usize;
        assert!(
            matches!(
                plan(Some(1000), 1000 + crossed, MAX_CATCHUP_FRAMES),
                FramePlan::Consume { .. }
            ),
            "1500 rows/s against 60 fps is {crossed} rows and must replay"
        );
    }

    #[test]
    fn a_limit_of_zero_still_makes_progress() {
        // Guarded so a mis-set limit degrades to one row per tick rather than
        // snapping for ever and never running the filter at all.
        assert_eq!(plan(Some(3), 4, 0), FramePlan::Consume { first: 4, last: 4 });
        assert_eq!(plan(Some(3), 5, 0), FramePlan::Snap { to: 5 });
    }

    #[test]
    fn target_frame_clamps_to_what_exists() {
        assert_eq!(target_frame(0.0, 180.0, 0), None);
        assert_eq!(target_frame(0.0, 180.0, 10), Some(0));
        assert_eq!(target_frame(1.0, 180.0, 10), Some(9));
        assert_eq!(target_frame(1.0 / 180.0, 180.0, 10), Some(1));
        // Negative and non-finite positions land on the first row instead of
        // wrapping through the cast. Infinity is treated the same as NaN rather
        // than as "past the end": neither is a position a playback clock can
        // produce, and one rule for "not a number I can use" is easier to hold
        // than two.
        assert_eq!(target_frame(-5.0, 180.0, 10), Some(0));
        assert_eq!(target_frame(f64::NAN, 180.0, 10), Some(0));
        assert_eq!(target_frame(f64::INFINITY, 180.0, 10), Some(0));
        assert_eq!(target_frame(f64::NEG_INFINITY, 180.0, 10), Some(0));
        assert_eq!(target_frame(1.0, 0.0, 10), Some(0));
        assert_eq!(target_frame(1.0, f64::NAN, 10), Some(0));
    }

    #[test]
    fn source_dt_is_guarded() {
        assert!((source_dt(180.0) - 1.0 / 180.0).abs() < 1e-12);
        assert_eq!(source_dt(0.0), 0.0);
        assert_eq!(source_dt(-1.0), 0.0);
        assert_eq!(source_dt(f64::NAN), 0.0);
    }
}
