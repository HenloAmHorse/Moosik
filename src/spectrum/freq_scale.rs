#![allow(dead_code)] // a couple of helpers are kept for symmetry and tests

// How bar centre frequencies are distributed across the display.
//
// The bar *count* never changes here. This decides where they land, and every
// setting is a redistribution of the same 1024 bars.
//
//   * `Log` is the classic analyser axis: constant bars per octave, which is
//     why `grid_q` can be a single number for the whole spectrum. It is the
//     default and it delegates to the original functions rather than
//     reimplementing them, so selecting it is not merely equivalent to the
//     pre-scale behaviour, it *is* the pre-scale behaviour.
//   * `Erb` spaces bars per auditory filter (Glasberg & Moore 1990) — far
//     fewer in the bass, far more in the treble.
//   * `Blend` is anywhere between those two, and past either end.
//   * `Lens` adds a magnified region anywhere on the axis.
//
// Two things are worth knowing before reading the maths.
//
// The bass tilt is not a resolution control. A tone's blur in *bars* and a
// bassline's travel in *bars* both scale with bar density, so the ratio between
// them is the same at every setting — what changes is how much screen the bass
// gets to move across. Below a capped analysis window it is also free: the
// window fixes resolution in Hz, so widening the bass adds no work.
//
// The lens is not free. Above roughly a kilohertz nothing caps the window, so
// concentrating bars there raises the resolution actually demanded and
// lengthens those windows. The settings panel quotes the multiplier.

/// Equivalent rectangular bandwidth of the auditory filter at `f`, in Hz.
/// Glasberg & Moore (1990): `ERB = 24.7·(4.37·f/1000 + 1)`.
pub fn erb_hz(f: f32) -> f32 {
    24.7 * (4.37 * f / 1000.0 + 1.0)
}

/// Position of `f` on the ERB-number scale — how many auditory filters fit
/// below it. About 42.5 of them span 20 Hz to 24 kHz, which is the whole
/// resolving power the cochlea has.
pub fn erb_number(f: f32) -> f32 {
    21.4 * (4.37 * f.max(0.0) / 1000.0 + 1.0).log10()
}

/// Inverse of [`erb_number`].
pub fn erb_number_to_hz(e: f32) -> f32 {
    (10f32.powf(e / 21.4) - 1.0) * 1000.0 / 4.37
}

#[derive(Clone, Copy, PartialEq, Debug, Default,
         serde::Serialize, serde::Deserialize)]
pub enum FreqScale {
    /// Constant bars per octave. What the display has always done.
    #[default]
    Log,
    /// Constant bars per auditory filter.
    Erb,
    /// Anywhere between the two, and past either end.
    ///
    /// `0.0` is Log, `1.0` is ERB, and negative values give the bass *more*
    /// width than a log axis does. The blend is over screen position, so the
    /// axis stays monotonic and the endpoints stay pinned.
    ///
    /// This exists because the two named scales are not the choice they look
    /// like. Bar density decides how much screen an octave of bass gets, and a
    /// tone's blur in *bars* scales with the same density — so the ratio between
    /// how far a bassline travels and how smeared it looks is the same on both.
    /// What actually differs is how much of the display the bass occupies, and
    /// that is a matter of taste, not of correctness. Below a capped analysis
    /// window it is also free: the window fixes resolution in Hz, so widening
    /// the bass adds no work at all.
    Blend(f32),
    /// A blend with a region of the spectrum magnified.
    ///
    /// `tilt` is as [`FreqScale::Blend`]. The rest describe a lens: `hz` where
    /// it is centred, `oct` how many octaves wide it is, and `gain` how many
    /// extra bars per bar it packs in at the centre — 0 is off, 3 is roughly
    /// four times the local detail.
    ///
    /// Unlike the bass tilt, this is **not** free. Widening the bass costs
    /// nothing because the analysis window is already capped down there, so the
    /// resolution the grid asks for is more than the transform was going to
    /// deliver anyway. Above roughly a kilohertz nothing is capped, so packing
    /// bars in raises the resolution actually demanded and lengthens those
    /// windows. The settings panel quotes the cost for exactly this reason.
    Lens { tilt: f32, hz: f32, oct: f32, gain: f32 },
}

/// Error function, Abramowitz & Stegun 7.1.26 — good to about 1.5e-7, which is
/// far below anything a bar position can express.
fn erf(x: f32) -> f32 {
    let s = if x < 0.0 { -1.0 } else { 1.0 };
    let x = x.abs();
    let t = 1.0 / (1.0 + 0.327_591_1 * x);
    let y = 1.0 - (((((1.061_405_4 * t - 1.453_152) * t) + 1.421_413_7) * t
        - 0.284_496_74) * t + 0.254_829_6) * t * (-x * x).exp();
    s * y
}

/// Position of `f` under the log/ERB blend alone, before any lens.
fn tilted(tilt: f32, f: f32, min_freq: f32, max_freq: f32) -> f32 {
    let lg = {
        let (a, b) = (min_freq.log10(), max_freq.log10());
        if b <= a { 0.0 } else { (f.max(1e-6).log10() - a) / (b - a) }
    };
    if tilt == 0.0 { return lg }
    let er = {
        let (a, b) = (erb_number(min_freq), erb_number(max_freq));
        if b <= a { 0.0 } else { (erb_number(f) - a) / (b - a) }
    };
    lg + tilt * (er - lg)
}

/// Where the lens sits on the tilted axis, and how wide it is there.
fn lens_span(tilt: f32, hz: f32, oct: f32, min_freq: f32, max_freq: f32) -> (f32, f32) {
    let c = tilted(tilt, hz, min_freq, max_freq);
    let half = 2f32.powf(oct * 0.5);
    let lo = tilted(tilt, (hz / half).max(min_freq * 0.5), min_freq, max_freq);
    let hi = tilted(tilt, (hz * half).min(max_freq * 2.0), min_freq, max_freq);
    (c, ((hi - lo) * 0.5).max(1e-4))
}

/// Redistribute positions so bars bunch up around `c`.
///
/// Built from a density rather than a formula pulled out of the air: with
/// `d(x) = 1 + gain*exp(-((x-c)/w)^2)` the warped position is the normalised
/// integral of `d`. Because `d` is strictly positive the result is strictly
/// increasing, and because it is normalised the two ends stay pinned — so no
/// setting can fold the axis back on itself or push a bar off the display.
/// The integral of a Gaussian is an error function, which is why there is one
/// above.
fn lens_warp(t: f32, c: f32, w: f32, gain: f32) -> f32 {
    let k = gain * w * (std::f32::consts::PI.sqrt() * 0.5);
    let n = |x: f32| x + k * (erf((x - c) / w) + erf(c / w));
    let total = n(1.0);
    if total.abs() < 1e-6 { return t }
    (n(t) / total).clamp(0.0, 1.0)
}

impl FreqScale {
    /// Centre frequency of bar `i` of `n`.
    ///
    /// The `Log` arm is bit-identical to `aslt::bar_center_freq`, deliberately:
    /// switching scales must be the only thing that moves the bars.
    pub fn bar_center(self, i: usize, n: usize, min_freq: f32, max_freq: f32) -> f32 {
        // Delegated, not reimplemented. An identical-looking copy would be
        // unchanged only as far as the tolerance of whatever test compared them;
        // calling the original makes it unchanged by construction.
        if self == FreqScale::Log {
            return super::aslt::bar_center_freq(i, n, min_freq, max_freq);
        }
        let t = (i as f32 + 0.5) / n.max(1) as f32;
        self.freq_at(t, min_freq, max_freq)
    }

    /// How far the bass is stretched relative to a log axis: 0 for Log, 1 for
    /// ERB, negative for wider-than-log.
    pub fn tilt(self) -> f32 {
        match self {
            FreqScale::Log => 0.0,
            FreqScale::Erb => 1.0,
            // Past +1 the ERB term eventually outruns the log one in the treble
            // and the axis would fold back on itself; -1..1 keeps every blend
            // strictly increasing.
            FreqScale::Blend(a) => a.clamp(-1.0, 1.0),
            FreqScale::Lens { tilt, .. } => tilt.clamp(-1.0, 1.0),
        }
    }

    /// The lens, if one is active: centre in Hz, width in octaves, and gain.
    pub fn lens(self) -> Option<(f32, f32, f32)> {
        match self {
            FreqScale::Lens { hz, oct, gain, .. } if gain > 0.0 && oct > 0.0 => {
                Some((hz.max(1.0), oct.clamp(0.1, 10.0), gain.clamp(0.0, 8.0)))
            }
            _ => None,
        }
    }

    /// True when this is anything other than the plain log axis.
    pub fn is_warped(self) -> bool { self.tilt() != 0.0 || self.lens().is_some() }

    /// Snap a tilt back to a named scale when it lands exactly on one, so
    /// settings and cache filenames stay readable.
    pub fn from_tilt(a: f32) -> Self {
        if a == 0.0 { FreqScale::Log }
        else if a == 1.0 { FreqScale::Erb }
        else { FreqScale::Blend(a.clamp(-1.0, 1.0)) }
    }

    /// Same, but carrying a lens through. A gain of zero collapses back to the
    /// plain scales so the settings file and cache names stay tidy.
    pub fn from_parts(tilt: f32, hz: f32, oct: f32, gain: f32) -> Self {
        if gain <= 0.0 || oct <= 0.0 { return Self::from_tilt(tilt) }
        FreqScale::Lens {
            tilt: tilt.clamp(-1.0, 1.0),
            hz: hz.max(1.0),
            oct: oct.clamp(0.1, 10.0),
            gain: gain.clamp(0.0, 8.0),
        }
    }

    /// Q the bar grid can actually draw at bar `i` — the resolution the
    /// transform has to reach for a tone to land in one bar and no more.
    ///
    /// Derived from what the neighbouring bars are doing rather than from a
    /// closed form, which is what lets one definition serve both scales. Under
    /// `Log` this reduces to `aslt::grid_q`: with `r = 2^(1/per_octave)` the
    /// neighbour spacing is `f·(r − 1/r)/2`, and for `r` near 1 that is
    /// `f·(r − 1)`, giving `Q = 1/(r − 1)` — the existing formula.
    pub fn grid_q_at(self, i: usize, n: usize, min_freq: f32, max_freq: f32) -> f32 {
        // Log spacing has a closed form and always has. Using the derived value
        // here instead would shift every Q by up to a couple of percent — small,
        // but it would mean every existing cache and every measured figure in
        // this repository was taken against slightly different maths.
        if self == FreqScale::Log {
            return super::aslt::grid_q(n, min_freq, max_freq);
        }
        self.derived_grid_q_at(i, n, min_freq, max_freq)
    }

    /// The derived Q, used for `Erb` and exposed so the equivalence with the
    /// closed form can still be tested on the `Log` grid it no longer runs on.
    pub fn derived_grid_q_at(self, i: usize, n: usize, min_freq: f32, max_freq: f32) -> f32 {
        let n = n.max(2);
        let i = i.min(n - 1);
        let (lo, hi, span) = if i == 0 {
            (0usize, 1usize, 1.0)
        } else if i == n - 1 {
            (n - 2, n - 1, 1.0)
        } else {
            (i - 1, i + 1, 2.0)
        };
        let f = self.bar_center(i, n, min_freq, max_freq);
        let width = (self.bar_center(hi, n, min_freq, max_freq)
            - self.bar_center(lo, n, min_freq, max_freq)) / span;
        if width <= 0.0 { return 1.0 }
        (f / width).max(1.0)
    }

    /// Where `f` sits across the display, 0 at `min_freq` and 1 at `max_freq`.
    ///
    /// The inverse of [`FreqScale::bar_center`] up to the half-bar offset, and
    /// what anything drawn *over* the bars has to use — an EQ curve or a
    /// frequency label positioned on the log axis while the bars are on the ERB
    /// one would be wrong by an octave and a half at the bottom.
    pub fn position_of(self, f: f32, min_freq: f32, max_freq: f32) -> f32 {
        let t = tilted(self.tilt(), f, min_freq, max_freq);
        match self.lens() {
            None => t,
            Some((hz, oct, gain)) => {
                let (c, w) = lens_span(self.tilt(), hz, oct, min_freq, max_freq);
                lens_warp(t, c, w, gain)
            }
        }
    }

    /// Frequency at fractional position `t` across the display — the exact
    /// inverse of [`FreqScale::position_of`].
    pub fn freq_at(self, t: f32, min_freq: f32, max_freq: f32) -> f32 {
        match self {
            FreqScale::Log => {
                let (a, b) = (min_freq.log10(), max_freq.log10());
                10f32.powf(a + t * (b - a))
            }
            FreqScale::Erb => {
                let (a, b) = (erb_number(min_freq), erb_number(max_freq));
                erb_number_to_hz(a + t * (b - a))
            }
            // Neither a blend of two position functions nor a lens has a
            // closed-form inverse, so both are bisected. `position_of` is
            // strictly increasing for every permitted setting, which is what
            // makes that safe; 40 halvings take an f32 past its own precision.
            FreqScale::Blend(_) | FreqScale::Lens { .. } => {
                let (mut lo, mut hi) = (min_freq, max_freq);
                if hi <= lo { return lo }
                for _ in 0..40 {
                    let mid = 0.5 * (lo + hi);
                    if self.position_of(mid, min_freq, max_freq) < t { lo = mid } else { hi = mid }
                }
                0.5 * (lo + hi)
            }
        }
    }

    /// Bars falling within one octave centred on `f`. The headline difference
    /// between the two scales, and the reason the bass behaves differently.
    pub fn bars_per_octave_at(self, f: f32, n: usize, min_freq: f32, max_freq: f32) -> f32 {
        let (lo, hi) = (f / std::f32::consts::SQRT_2, f * std::f32::consts::SQRT_2);
        let pos = |x: f32| self.position_of(x, min_freq, max_freq);
        ((pos(hi) - pos(lo)) * n as f32).max(0.0)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::spectrum::aslt;

    /// The `Log` arm must be indistinguishable from what the transform already
    /// uses, or "switch scale" would silently mean "switch scale and also move
    /// every bar slightly", and any comparison would be measuring both.
    #[test]
    fn log_scale_reproduces_the_existing_bars_exactly() {
        for n in [64usize, 512, 1024] {
            for i in [0usize, 1, n / 3, n / 2, n - 1] {
                let mine = FreqScale::Log.bar_center(i, n, 20.0, 24_000.0);
                let theirs = aslt::bar_center_freq(i, n, 20.0, 24_000.0);
                assert!(
                    (mine - theirs).abs() < theirs * 1e-6,
                    "bar {i} of {n}: {mine} vs {theirs}",
                );
            }
        }
    }

    /// The `Log` grid must return the closed form *exactly*, not merely close
    /// to it — every cached analysis and every measurement in this repository
    /// was taken against that number.
    #[test]
    fn log_grid_q_is_the_closed_form_untouched() {
        let n = 1024;
        let closed = aslt::grid_q(n, 20.0, 24_000.0);
        for i in [0usize, 1, n / 4, n / 2, n - 1] {
            assert_eq!(FreqScale::Log.grid_q_at(i, n, 20.0, 24_000.0), closed);
        }
    }

    /// And the derived form the ERB grid uses has to agree with that closed
    /// form where both are defined, or it is measuring something else.
    #[test]
    fn the_derived_q_agrees_with_the_closed_form_on_a_log_grid() {
        let n = 1024;
        let closed = aslt::grid_q(n, 20.0, 24_000.0);
        // Away from the ends, where the centred difference applies.
        for i in [n / 4, n / 2, 3 * n / 4] {
            let derived = FreqScale::Log.derived_grid_q_at(i, n, 20.0, 24_000.0);
            assert!(
                (derived - closed).abs() < closed * 0.02,
                "bar {i}: derived {derived:.1} vs closed {closed:.1}",
            );
        }
    }

    /// The property the whole idea rests on: ERB moves bars out of the bass and
    /// into the treble, without changing how many there are.
    #[test]
    fn erb_redistributes_rather_than_removes() {
        let n = 1024;
        let (lo, hi) = (20.0f32, 24_000.0f32);

        let log_bass = FreqScale::Log.bars_per_octave_at(30.0, n, lo, hi);
        let erb_bass = FreqScale::Erb.bars_per_octave_at(30.0, n, lo, hi);
        let log_treb = FreqScale::Log.bars_per_octave_at(14_000.0, n, lo, hi);
        let erb_treb = FreqScale::Erb.bars_per_octave_at(14_000.0, n, lo, hi);

        assert!(erb_bass < log_bass * 0.4, "ERB should thin the bass: {erb_bass} vs {log_bass}");
        assert!(erb_treb > log_treb * 1.2, "ERB should thicken the treble: {erb_treb} vs {log_treb}");

        // Endpoints still cover the same range. Bar 0 sits half a bar above
        // `lo` by construction (`t = (i + 0.5)/n`), and half an ERB bar at the
        // bottom is a wider step in Hz than half a log bar, so the tolerance
        // here is the bar width, not an error budget.
        let first = FreqScale::Erb.bar_center(0, n, lo, hi);
        let last = FreqScale::Erb.bar_center(n - 1, n, lo, hi);
        assert!(first > lo && first < lo * 1.05, "first ERB bar at {first} Hz");
        assert!(last < hi && last > hi * 0.98, "last ERB bar at {last} Hz");

        // And the bass Q target falls, which is what would make it cheaper.
        let f = 60.0f32;
        let bar_of = |s: FreqScale| (0..n).min_by(|&a, &b| {
            let da = (s.bar_center(a, n, lo, hi) - f).abs();
            let db = (s.bar_center(b, n, lo, hi) - f).abs();
            da.partial_cmp(&db).unwrap()
        }).unwrap();
        let q_log = FreqScale::Log.grid_q_at(bar_of(FreqScale::Log), n, lo, hi);
        let q_erb = FreqScale::Erb.grid_q_at(bar_of(FreqScale::Erb), n, lo, hi);
        assert!(q_erb < q_log * 0.5, "60 Hz grid Q: log {q_log:.0} vs erb {q_erb:.0}");
    }

    /// Every setting on the slider has to produce a usable axis.
    ///
    /// The blend is a weighted sum of two position functions, and past the ends
    /// the ERB term eventually outruns the log one and the axis folds back on
    /// itself — bars would run backwards and a tone would land in two places.
    /// The clamp in `tilt` is what prevents it, so this is the test that the
    /// clamp is wide enough to be useful and tight enough to be safe.
    #[test]
    fn every_tilt_gives_a_strictly_increasing_axis() {
        let (lo, hi) = (20.0f32, 24_000.0f32);
        let n = 512;
        for step in -20..=20 {
            let scale = FreqScale::from_tilt(step as f32 / 20.0);
            let mut prev = 0.0f32;
            for i in 0..n {
                let f = scale.bar_center(i, n, lo, hi);
                assert!(
                    f > prev,
                    "tilt {:.2}: bar {i} at {f} Hz did not advance past {prev}",
                    scale.tilt(),
                );
                assert!(f >= lo * 0.99 && f <= hi * 1.01, "bar {i} escaped the range at {f} Hz");
                prev = f;
            }
        }
    }

    /// `position_of` and `freq_at` must invert each other, including through the
    /// bisection the blends rely on — everything drawn over the bars uses one
    /// and everything placing bars uses the other.
    #[test]
    fn position_and_frequency_round_trip() {
        let (lo, hi) = (20.0f32, 24_000.0f32);
        for tilt in [-1.0f32, -0.5, 0.0, 0.35, 1.0] {
            let s = FreqScale::from_tilt(tilt);
            for &hz in &[25.0f32, 60.0, 440.0, 3_000.0, 15_000.0] {
                let t = s.position_of(hz, lo, hi);
                let back = s.freq_at(t, lo, hi);
                assert!(
                    (back - hz).abs() < hz * 0.001,
                    "tilt {tilt}: {hz} Hz -> {t} -> {back} Hz",
                );
            }
        }
    }

    /// The slider has to do the thing it is named after, in both directions.
    #[test]
    fn negative_tilt_widens_the_bass_and_positive_narrows_it() {
        let (lo, hi) = (20.0f32, 24_000.0f32);
        let n = 1024;
        let bass = |t: f32| FreqScale::from_tilt(t).bars_per_octave_at(50.0, n, lo, hi);

        let wide = bass(-1.0);
        let log = bass(0.0);
        let erb = bass(1.0);
        assert!(wide > log, "negative tilt should widen the bass: {wide} vs {log}");
        assert!(log > erb, "positive tilt should narrow it: {log} vs {erb}");
        // And monotone in between, so dragging the slider never reverses.
        let mut prev = f32::INFINITY;
        for step in -10..=10 {
            let v = bass(step as f32 / 10.0);
            assert!(v < prev, "bass width should fall as tilt rises: {v} then {prev}");
            prev = v;
        }
    }

    /// The lens must concentrate bars where it is aimed, take them from
    /// elsewhere, and never change how many there are.
    #[test]
    fn the_lens_moves_bars_without_creating_them() {
        let (lo, hi) = (20.0f32, 24_000.0f32);
        let n = 1024;
        let plain = FreqScale::Log;
        let zoom = FreqScale::from_parts(0.0, 1000.0, 2.0, 3.0);

        let at = |s: FreqScale, f: f32| s.bars_per_octave_at(f, n, lo, hi);
        assert!(at(zoom, 1000.0) > at(plain, 1000.0) * 1.5,
                "the lens should magnify its centre: {} vs {}",
                at(zoom, 1000.0), at(plain, 1000.0));
        // Paid for out of the rest of the spectrum, not conjured.
        assert!(at(zoom, 60.0) < at(plain, 60.0),
                "bars for the lens have to come from somewhere");
        // Ends stay pinned, so no bar is pushed off the display.
        assert!(zoom.bar_center(0, n, lo, hi) > lo * 0.99);
        assert!(zoom.bar_center(n - 1, n, lo, hi) < hi * 1.01);
    }

    /// Every lens setting, like every tilt, has to leave the axis usable. The
    /// warp is the normalised integral of a strictly positive density, so this
    /// should hold by construction — which is exactly the kind of claim worth
    /// checking rather than trusting.
    #[test]
    fn every_lens_gives_a_strictly_increasing_axis() {
        let (lo, hi) = (20.0f32, 24_000.0f32);
        let n = 256;
        for &tilt in &[-1.0f32, 0.0, 1.0] {
            for &hz in &[25.0f32, 200.0, 1000.0, 12_000.0] {
                for &oct in &[0.25f32, 2.0, 6.0] {
                    for &gain in &[0.5f32, 3.0, 8.0] {
                        let s = FreqScale::from_parts(tilt, hz, oct, gain);
                        let mut prev = 0.0f32;
                        for i in 0..n {
                            let f = s.bar_center(i, n, lo, hi);
                            assert!(
                                f > prev && f.is_finite(),
                                "tilt {tilt} hz {hz} oct {oct} gain {gain}: \
                                 bar {i} at {f} did not advance past {prev}",
                            );
                            prev = f;
                        }
                        // And the inverse still agrees with the forward map.
                        let t = s.position_of(hz, lo, hi);
                        let back = s.freq_at(t, lo, hi);
                        assert!((back - hz).abs() < hz * 0.01,
                                "round trip at {hz} Hz gave {back}");
                    }
                }
            }
        }
    }

    /// A zero-gain lens has to be indistinguishable from no lens, or every
    /// "off" state would quietly be its own scale with its own cache.
    #[test]
    fn a_lens_with_no_gain_is_not_a_lens() {
        assert_eq!(FreqScale::from_parts(0.0, 1000.0, 2.0, 0.0), FreqScale::Log);
        assert_eq!(FreqScale::from_parts(1.0, 1000.0, 2.0, 0.0), FreqScale::Erb);
        assert_eq!(FreqScale::from_parts(0.5, 500.0, 1.0, 0.0), FreqScale::Blend(0.5));
        assert!(FreqScale::from_parts(0.0, 1000.0, 2.0, 0.0).lens().is_none());
    }

    /// Dump the comparison. Diagnostic, not pass/fail — this is the prototype.
    #[test]
    #[ignore = "survey — run explicitly"]
    fn erb_survey() {
        let n = 1024;
        let (lo, hi) = (20.0f32, 24_000.0f32);
        let cfg = aslt::AsltPreset::Extreme.config();

        println!("\n{} bars, {lo}–{hi} Hz", n);
        println!("total ERBs across the range: {:.1}", erb_number(hi) - erb_number(lo));
        println!("\n{:>8} {:>12} {:>12} {:>10} {:>10} {:>12} {:>12}",
                 "Hz", "log bars/oct", "erb bars/oct", "log Q", "erb Q",
                 "log window", "erb window");
        for f in [20.0f32, 30.0, 60.0, 100.0, 200.0, 500.0, 1000.0,
                  2000.0, 5000.0, 10_000.0, 20_000.0] {
            let bar_of = |s: FreqScale| (0..n).min_by(|&a, &b| {
                let da = (s.bar_center(a, n, lo, hi) - f).abs();
                let db = (s.bar_center(b, n, lo, hi) - f).abs();
                da.partial_cmp(&db).unwrap()
            }).unwrap();
            let ql = FreqScale::Log.grid_q_at(bar_of(FreqScale::Log), n, lo, hi);
            let qe = FreqScale::Erb.grid_q_at(bar_of(FreqScale::Erb), n, lo, hi);
            // Window each Q target needs at this frequency, via the same
            // relation the transform uses.
            let win = |q: f32| aslt::window_for_full_detail_above(f, q, &cfg);
            println!("{f:>8.0} {:>12.0} {:>12.0} {:>10.0} {:>10.0} {:>11.2}s {:>11.2}s",
                     FreqScale::Log.bars_per_octave_at(f, n, lo, hi),
                     FreqScale::Erb.bars_per_octave_at(f, n, lo, hi),
                     ql, qe, win(ql), win(qe));
        }
    }
}
