//! Which spectrum a channel view can honestly show.
//!
//! The live analyser has always averaged every channel of a frame down to one
//! mono stream, and the display has only ever drawn that. Separate left and
//! right spectra need a genuine two-channel tap on the stream being played —
//! and the interesting part is not producing them but refusing to produce them,
//! because there are several ways to end up with a plausible-looking pair of
//! plots that are not left and right at all:
//!
//! * mono material, where a copy of the same spectrum twice would be a lie
//!   dressed as a measurement;
//! * more than two channels, where any pairing is a downmix decision the
//!   player has no business making silently;
//! * DoP, which transports DSD marker and payload words inside a PCM carrier —
//!   an integer container the analyser could read but which is not audio;
//! * native DSD, which sends raw DSD to the driver over ASIO or ALSA and
//!   attaches no tap at all.
//!
//!   Neither route exposes live PCM to the spectrum tap, and they arrive at
//!   that by different means, so both are stated rather than one standing in
//!   for the other. Without saying so, the buffers hold whatever the previous
//!   track left in them;
//! * pre-process mode for a track whose cache has no stereo sidecar beside it.
//!   The mono cache stores one row per frame and cannot be made to yield
//!   channels however it is sliced; the sidecar holds the two that were
//!   analysed separately, and without it there is nothing to draw.
//!
//! Each of those has to say so on screen rather than fall back quietly, so the
//! decision lives here as a pure function over facts the caller has to supply.

use serde::{Deserialize, Serialize};

/// What the spectrum plot should draw.
#[derive(Clone, Copy, PartialEq, Eq, Debug, Serialize, Deserialize, Default)]
pub enum ChannelView {
    /// Every channel averaged to one stream. The historical behaviour, and
    /// still the default.
    #[default]
    Mix,
    /// Left only, full height.
    Left,
    /// Right only, full height.
    Right,
    /// Two stacked plots sharing frequency and dB scales.
    Split,
    /// Both in one plot, distinguished by colour.
    Overlay,
    /// Right minus left, per bar, about a centre line.
    ///
    /// Not a third way of drawing two spectra: it draws one derived series that
    /// neither channel view shows. Two curves an inch apart look identical
    /// whether they differ by half a decibel or six; the difference itself, on
    /// its own axis, is the thing worth looking at when the question is about
    /// balance, panning or a crossfeed network.
    Diff,
}

impl ChannelView {
    pub const ALL: [ChannelView; 6] = [
        ChannelView::Mix,
        ChannelView::Left,
        ChannelView::Right,
        ChannelView::Split,
        ChannelView::Overlay,
        ChannelView::Diff,
    ];

    pub fn label(self) -> &'static str {
        match self {
            ChannelView::Mix => "Mix",
            ChannelView::Left => "Left",
            ChannelView::Right => "Right",
            ChannelView::Split => "Split",
            ChannelView::Overlay => "Overlay",
            ChannelView::Diff => "Diff",
        }
    }

    /// Whether this view needs a second FFT. `Mix` does not, and must not pay
    /// for one.
    pub fn needs_channels(self) -> bool {
        !matches!(self, ChannelView::Mix)
    }

    /// Whether the left channel is drawn in this view.
    pub fn draws_left(self) -> bool {
        matches!(
            self,
            ChannelView::Left | ChannelView::Split | ChannelView::Overlay | ChannelView::Diff
        )
    }

    /// Whether the right channel is drawn in this view.
    pub fn draws_right(self) -> bool {
        matches!(
            self,
            ChannelView::Right | ChannelView::Split | ChannelView::Overlay | ChannelView::Diff
        )
    }
}

/// Why left and right are, or are not, available right now.
#[derive(Clone, PartialEq, Eq, Debug)]
pub enum ChannelAvailability {
    /// A live two-channel PCM tap is feeding the analyser.
    Available,
    /// The stream is one channel. There is no right to show.
    Mono,
    /// More than two channels. Which two are "left and right" is a layout
    /// question, and guessing it would be worse than declining.
    Multichannel(u16),
    /// Nothing is writing PCM into the tap: native DSD, or no stream at all.
    NoLiveTap,
    /// Pre-process mode, and this track's cache has no channels beside it —
    /// either stereo was off when it was analysed, or the track is not stereo.
    PreProcessMono,
    /// The chosen visualisation has no per-channel form in this phase.
    UnsupportedStyle,
}

impl ChannelAvailability {
    pub fn is_available(&self) -> bool {
        matches!(self, ChannelAvailability::Available)
    }

    /// One sentence for the UI, saying what is wrong and — where there is one —
    /// what would fix it. `None` when nothing is wrong.
    pub fn reason(&self) -> Option<String> {
        match self {
            ChannelAvailability::Available => None,
            ChannelAvailability::Mono => {
                Some("This track is mono — there is no second channel to show.".into())
            }
            ChannelAvailability::Multichannel(n) => Some(format!(
                "This track has {n} channels. Left/Right would have to guess a \
                 downmix, so only Mix is offered."
            )),
            ChannelAvailability::NoLiveTap => Some(
                "No live PCM to analyse. Native DSD reaches the device as DoP \
                 words, so there is no stereo pair to separate."
                    .into(),
            ),
            ChannelAvailability::PreProcessMono => Some(
                "This track was analysed without channels. Turn on stereo \
                 pre-analysis and analyse it again, or switch to Real-time."
                    .into(),
            ),
            ChannelAvailability::UnsupportedStyle => Some(
                "This visualisation has no per-channel form yet; it shows the \
                 Mix."
                    .into(),
            ),
        }
    }
}

/// Channel count the live tap is currently carrying.
///
/// `0` means nothing is writing — which is not the same as "the buffer is
/// empty", and is the distinction that stops a stereo buffer left over from
/// the previous track being read as this track's channels.
pub const NO_LIVE_TAP: u16 = 0;

/// Decide what the plot may show.
///
/// Every input is a fact the caller has to establish; nothing is inferred from
/// the contents of a buffer. In particular `tap_channels` comes from the
/// decoder or device format that built the tap, never from whether the stereo
/// buffer happens to be non-empty.
///
/// Ordering of the checks is deliberate — the caller is told the most
/// fundamental reason first, so switching visualisation does not reveal a
/// second, deeper obstacle it could not have guessed at.
pub fn availability(
    preprocess: bool,
    pre_channels: bool,
    style_supports_channels: bool,
    tap_channels: u16,
) -> ChannelAvailability {
    if preprocess {
        // The live tap says nothing here: what is on screen came from a file,
        // and whether that file has channels beside it is a property of the
        // file. This stays ahead of the style check, and the ordering rule
        // below is why: clearing this obstacle means analysing the track
        // again, which costs minutes, and being sent to do that only to find
        // the visualisation could never have shown it is the worst outcome
        // available.
        if !pre_channels {
            return ChannelAvailability::PreProcessMono;
        }
        if !style_supports_channels {
            return ChannelAvailability::UnsupportedStyle;
        }
        return ChannelAvailability::Available;
    }
    if !style_supports_channels {
        return ChannelAvailability::UnsupportedStyle;
    }
    match tap_channels {
        NO_LIVE_TAP => ChannelAvailability::NoLiveTap,
        1 => ChannelAvailability::Mono,
        2 => ChannelAvailability::Available,
        n => ChannelAvailability::Multichannel(n),
    }
}

/// The view that will actually be drawn, given what the user asked for.
///
/// Falling back to `Mix` is the only safe degradation: every other view would
/// have to invent a channel. The requested value is *not* rewritten in the
/// settings, so the view returns as soon as the obstacle does — putting on a
/// pair of headphones should not cost you your display preference.
pub fn effective_view(requested: ChannelView, avail: &ChannelAvailability) -> ChannelView {
    if avail.is_available() {
        requested
    } else {
        ChannelView::Mix
    }
}

/// Pull the last `want` frames of an interleaved stereo tap apart into two
/// contiguous channel buffers.
///
/// Extracted from the tick so it can be tested on its own. It is the one place
/// the pairing is undone, and undoing it wrongly is invisible downstream: an
/// average written into both buffers produces two perfectly plausible spectra
/// that are identical, which looks exactly like correlated material.
///
/// The buffers are cleared and refilled rather than reallocated, so a caller
/// that keeps them between frames allocates once.
pub fn deinterleave(
    frames: &[[f32; 2]],
    want: usize,
    left: &mut Vec<f32>,
    right: &mut Vec<f32>,
) -> bool {
    left.clear();
    right.clear();
    if want == 0 || frames.len() < want {
        return false;
    }
    let start = frames.len() - want;
    left.extend(frames[start..].iter().map(|f| f[0]));
    right.extend(frames[start..].iter().map(|f| f[1]));
    true
}

/// Which way the difference plot runs.
#[derive(Clone, Copy, PartialEq, Eq, Debug, Serialize, Deserialize, Default)]
pub enum DiffOrientation {
    /// Frequency across, difference up and down about a horizontal line.
    #[default]
    Horizontal,
    /// Frequency down, difference left and right about a vertical line.
    ///
    /// Worth having rather than a novelty: a tall narrow window has more room
    /// for frequency along its long edge, and a vertical centre line puts left
    /// on the left, which is the mapping most people expect from a stereo
    /// display and which the horizontal form cannot offer at all.
    Vertical,
}

impl DiffOrientation {
    pub fn label(self) -> &'static str {
        match self {
            DiffOrientation::Horizontal => "Horizontal",
            DiffOrientation::Vertical => "Vertical",
        }
    }
}

/// How the difference plot is arranged.
///
/// All three are preferences with no correct answer. Which channel belongs on
/// which side is convention, not fact; which end of the spectrum sits where is
/// the same; and the orientation depends on the shape of the window and on what
/// the viewer is used to. They are settings because arguing about the default
/// is less useful than letting people turn it round.
#[derive(Clone, Copy, PartialEq, Eq, Debug, Serialize, Deserialize, Default)]
pub struct DiffLayout {
    pub orientation: DiffOrientation,
    /// Put left where right would be. Off means positive — a louder right —
    /// is up in the horizontal form and to the right in the vertical one.
    pub flip_channels: bool,
    /// Run the frequency axis the other way: high to low instead of low to
    /// high across the plot, or bottom to top instead of top to bottom.
    pub flip_frequency: bool,
}

/// Which channel a difference favours.
///
/// Deliberately separate from where the difference is *drawn*. Position is a
/// preference — `flip_channels` moves it — while which channel is actually
/// louder is a fact about the audio. Reading the colour off the drawn position
/// makes Swap L/R recolour the plot, which is the one thing a colour key must
/// never do: the reader would see cyan move to the other side and conclude the
/// audio had changed.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum Louder {
    Left,
    Right,
    /// Neither, within the resolution of the difference — or not a number.
    Equal,
}

impl Louder {
    pub fn label(self) -> &'static str {
        match self {
            Louder::Left => "L",
            Louder::Right => "R",
            Louder::Equal => "=",
        }
    }
}

/// Which channel a right-minus-left difference favours.
///
/// Takes the *raw* difference, before any layout flip. `Equal` covers exact
/// zero and anything not finite, so a silent band draws in neither channel's
/// colour rather than defaulting to one of them.
pub fn louder(raw_diff: f32) -> Louder {
    if !raw_diff.is_finite() || raw_diff == 0.0 {
        Louder::Equal
    } else if raw_diff > 0.0 {
        Louder::Right
    } else {
        Louder::Left
    }
}

impl DiffLayout {
    /// Sign to apply to a raw difference before drawing it.
    pub fn sign(&self) -> f32 {
        if self.flip_channels { -1.0 } else { 1.0 }
    }

    /// Where a raw right-minus-left fraction sits on the difference axis, in
    /// `-1.0..=1.0`. Positive is up in the horizontal form and to the right in
    /// the vertical one.
    ///
    /// This and [`louder`] are the pair the renderer uses: this decides
    /// position, that decides colour, and they are separate on purpose.
    pub fn placed(&self, raw_diff: f32) -> f32 {
        raw_diff * self.sign()
    }

    /// Bar index to draw at display position `i`, for `n` bars.
    pub fn source_bar(&self, i: usize, n: usize) -> usize {
        if self.flip_frequency {
            n.saturating_sub(1).saturating_sub(i)
        } else {
            i
        }
    }

    /// Screen position, as a fraction of the frequency axis, for data at
    /// fraction `t` of the spectrum.
    ///
    /// The continuous counterpart of [`DiffLayout::source_bar`], for placing
    /// axis ticks — which sit at frequencies, not at bar indices, and so cannot
    /// use the index form. The two must agree or the labels drift off the bars
    /// they name; `flip_reverses_ticks_and_bars_together` holds them to it.
    pub fn along_fraction(&self, t: f32) -> f32 {
        if self.flip_frequency { 1.0 - t } else { t }
    }

    /// The channel each end of the difference axis belongs to, as
    /// (positive end, negative end).
    pub fn end_channels(&self) -> (Louder, Louder) {
        if self.flip_channels {
            (Louder::Left, Louder::Right)
        } else {
            (Louder::Right, Louder::Left)
        }
    }

    /// Labels for the two ends of the difference axis, in the order
    /// (positive end, negative end) — top and bottom, or right and left.
    ///
    /// Derived from [`DiffLayout::end_channels`] rather than written out again,
    /// so a label and the colour beside it cannot disagree.
    pub fn end_labels(&self) -> (&'static str, &'static str) {
        let (pos, neg) = self.end_channels();
        (pos.label(), neg.label())
    }
}

/// Full-scale of the difference axis, in decibels either side of the centre.
///
/// Bar magnitudes are a normalised 80 dB range, so a raw difference of 1.0
/// would be 80 dB — a scale on which every real recording is a flat line.
/// Channel differences that matter live in single figures, so the axis is
/// expanded to make them legible and clips beyond it rather than compressing,
/// because a difference of 30 dB and one of 60 dB are both simply "hard
/// panned" and there is nothing to tell apart up there.
pub const DIFF_FULL_SCALE_DB: f32 = 20.0;

/// Right minus left for one bar, as a fraction of the difference axis.
///
/// Positive means right is louder and is drawn above the centre line; negative
/// means left and is drawn below. Clamped to the axis, so a hard-panned bar
/// pins rather than running off the plot.
///
/// Both inputs are the normalised 0..1 magnitudes the bars are drawn from, so
/// their difference is already in units of 1/80 dB.
pub fn diff_fraction(left: f32, right: f32) -> f32 {
    if !left.is_finite() || !right.is_finite() {
        return 0.0;
    }
    let db = (right - left) * 80.0;
    (db / DIFF_FULL_SCALE_DB).clamp(-1.0, 1.0)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_centred_bar_is_zero_and_a_louder_right_is_positive() {
        assert_eq!(diff_fraction(0.5, 0.5), 0.0);
        assert!(
            diff_fraction(0.4, 0.6) > 0.0,
            "a louder right must read positive"
        );
        assert!(
            diff_fraction(0.6, 0.4) < 0.0,
            "a louder left must read negative"
        );
        // And the two directions are symmetric.
        assert_eq!(diff_fraction(0.4, 0.6), -diff_fraction(0.6, 0.4));
    }

    /// The axis is in decibels, and says which ones.
    #[test]
    fn the_axis_is_the_stated_number_of_decibels() {
        // A tenth of the normalised range is 8 dB.
        let f = diff_fraction(0.0, 0.1);
        assert!(
            (f - 8.0 / DIFF_FULL_SCALE_DB).abs() < 1e-6,
            "8 dB should be {} of the axis, got {f}",
            8.0 / DIFF_FULL_SCALE_DB
        );
        // Exactly full scale.
        let full = DIFF_FULL_SCALE_DB / 80.0;
        assert!((diff_fraction(0.0, full) - 1.0).abs() < 1e-6);
    }

    /// Beyond the axis it pins, rather than drawing off the plot.
    #[test]
    fn a_hard_panned_bar_pins_at_full_scale() {
        assert_eq!(diff_fraction(0.0, 1.0), 1.0);
        assert_eq!(diff_fraction(1.0, 0.0), -1.0);
    }

    #[test]
    fn nonsense_reads_as_no_difference() {
        assert_eq!(diff_fraction(f32::NAN, 0.5), 0.0);
        assert_eq!(diff_fraction(0.5, f32::INFINITY), 0.0);
    }

    #[test]
    fn flipping_channels_mirrors_the_reading_and_the_labels() {
        let plain = DiffLayout::default();
        let flipped = DiffLayout {
            flip_channels: true,
            ..Default::default()
        };
        assert_eq!(plain.sign(), 1.0);
        assert_eq!(flipped.sign(), -1.0);
        // The label has to move with the sign, or the plot lies about which
        // channel is which — which is worse than not offering the flip.
        assert_eq!(plain.end_labels(), ("R", "L"));
        assert_eq!(flipped.end_labels(), ("L", "R"));
    }

    #[test]
    fn colour_identity_survives_swapping_the_sides() {
        let plain = DiffLayout::default();
        let flipped = DiffLayout { flip_channels: true, ..Default::default() };
        // A band where right is louder. Its *position* moves when the sides are
        // swapped; the channel it belongs to does not.
        let raw = 0.4f32;
        assert_eq!(louder(raw), Louder::Right);
        assert!(plain.placed(raw) > 0.0);
        assert!(flipped.placed(raw) < 0.0, "swap must move it to the other side");
        // And the reading is the same magnitude either way round.
        assert_eq!(plain.placed(raw).abs(), flipped.placed(raw).abs());
        // Left-louder is the mirror of that.
        assert_eq!(louder(-raw), Louder::Left);
        assert!(plain.placed(-raw) < 0.0);
        assert!(flipped.placed(-raw) > 0.0);
    }

    #[test]
    fn a_band_with_no_difference_belongs_to_neither_channel() {
        assert_eq!(louder(0.0), Louder::Equal);
        assert_eq!(louder(-0.0), Louder::Equal);
        // Every non-finite value is `Equal`, infinities included. They cannot
        // arrive from `diff_fraction`, which clamps and sanitises first, so the
        // question is only what an unreachable input should do — and refusing
        // to name a channel is the safer answer than inventing one.
        assert_eq!(louder(f32::NAN), Louder::Equal);
        assert_eq!(louder(f32::INFINITY), Louder::Equal);
        assert_eq!(louder(f32::NEG_INFINITY), Louder::Equal);
    }

    #[test]
    fn the_end_labels_name_the_channel_at_that_end() {
        let plain = DiffLayout::default();
        let flipped = DiffLayout { flip_channels: true, ..Default::default() };
        assert_eq!(plain.end_channels(), (Louder::Right, Louder::Left));
        assert_eq!(flipped.end_channels(), (Louder::Left, Louder::Right));
        // The label is derived from the channel, so the two cannot drift apart.
        for l in [plain, flipped] {
            let (pc, nc) = l.end_channels();
            assert_eq!(l.end_labels(), (pc.label(), nc.label()));
        }
        // And the positive end really is where `placed` sends a positive value
        // for that channel.
        assert!(plain.placed(0.5) > 0.0 && plain.end_channels().0 == Louder::Right);
        assert!(flipped.placed(-0.5) > 0.0 && flipped.end_channels().0 == Louder::Left);
    }

    #[test]
    fn flip_reverses_ticks_and_bars_together() {
        // The tick placement and the bar placement are different functions —
        // one continuous, one by index — and a label that disagrees with the
        // bars it names is worse than no label.
        for l in [
            DiffLayout::default(),
            DiffLayout { flip_frequency: true, ..Default::default() },
        ] {
            const N: usize = 16;
            for i in 0..N {
                // Centre of display slot `i`, as a fraction.
                let t_screen = (i as f32 + 0.5) / N as f32;
                // The datum drawn there, by the index path.
                let src = l.source_bar(i, N);
                // Where the continuous path would put that datum's centre.
                let t_data = (src as f32 + 0.5) / N as f32;
                let placed = l.along_fraction(t_data);
                assert!(
                    (placed - t_screen).abs() < 1e-5,
                    "slot {i}: index path says bar {src}, tick path puts it at \
                     {placed} but the slot is at {t_screen}"
                );
            }
        }
    }

    #[test]
    fn flipping_frequency_reverses_the_axis_and_nothing_else() {
        let plain = DiffLayout::default();
        let flipped = DiffLayout {
            flip_frequency: true,
            ..Default::default()
        };
        assert_eq!(plain.source_bar(0, 8), 0);
        assert_eq!(plain.source_bar(7, 8), 7);
        assert_eq!(flipped.source_bar(0, 8), 7);
        assert_eq!(flipped.source_bar(7, 8), 0);
        // A reversal is a permutation: every bar appears exactly once.
        let mut seen: Vec<usize> = (0..8).map(|i| flipped.source_bar(i, 8)).collect();
        seen.sort_unstable();
        assert_eq!(seen, (0..8).collect::<Vec<_>>());
        // And it does not touch the channel sign.
        assert_eq!(flipped.sign(), 1.0);
    }

    #[test]
    fn the_axis_is_safe_at_the_degenerate_sizes() {
        let flipped = DiffLayout {
            flip_frequency: true,
            ..Default::default()
        };
        assert_eq!(flipped.source_bar(0, 0), 0);
        assert_eq!(flipped.source_bar(0, 1), 0);
        assert_eq!(flipped.source_bar(5, 1), 0);
    }

    #[test]
    fn diff_needs_both_channels() {
        assert!(ChannelView::Diff.needs_channels());
        assert!(ChannelView::Diff.draws_left() && ChannelView::Diff.draws_right());
    }

    /// Left is `[0]`, right is `[1]`, and neither is a blend of the two.
    #[test]
    fn deinterleave_keeps_the_channels_apart() {
        let frames: Vec<[f32; 2]> = (0..8).map(|i| [i as f32, -(i as f32)]).collect();
        let (mut l, mut r) = (Vec::new(), Vec::new());
        assert!(deinterleave(&frames, 8, &mut l, &mut r));
        assert_eq!(l, vec![0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0]);
        assert_eq!(r, vec![0.0, -1.0, -2.0, -3.0, -4.0, -5.0, -6.0, -7.0]);
        // The failure this is really guarding: an average in both buffers is
        // well-formed, plausible, and wrong.
        assert_ne!(l, r, "the channels must not be the same series");
    }

    /// Only the newest `want` frames are analysed.
    #[test]
    fn deinterleave_takes_the_tail() {
        let frames: Vec<[f32; 2]> = (0..10).map(|i| [i as f32, i as f32 + 100.0]).collect();
        let (mut l, mut r) = (Vec::new(), Vec::new());
        assert!(deinterleave(&frames, 3, &mut l, &mut r));
        assert_eq!(l, vec![7.0, 8.0, 9.0]);
        assert_eq!(r, vec![107.0, 108.0, 109.0]);
    }

    /// Too little history yields nothing rather than a short window, which
    /// would be analysed as if it were a whole one.
    #[test]
    fn deinterleave_refuses_a_short_buffer() {
        let frames: Vec<[f32; 2]> = (0..4).map(|i| [i as f32, i as f32]).collect();
        let (mut l, mut r) = (vec![9.0], vec![9.0]);
        assert!(!deinterleave(&frames, 8, &mut l, &mut r));
        assert!(l.is_empty() && r.is_empty(), "stale data must not survive");
        assert!(!deinterleave(&frames, 0, &mut l, &mut r));
    }

    /// Reused buffers must not grow every frame.
    #[test]
    fn deinterleave_reuses_its_buffers() {
        let frames: Vec<[f32; 2]> = (0..512).map(|i| [i as f32, -(i as f32)]).collect();
        let (mut l, mut r) = (Vec::new(), Vec::new());
        deinterleave(&frames, 256, &mut l, &mut r);
        let (cl, cr) = (l.capacity(), r.capacity());
        for _ in 0..32 {
            deinterleave(&frames, 256, &mut l, &mut r);
        }
        assert_eq!((l.capacity(), r.capacity()), (cl, cr));
    }

    #[test]
    fn only_a_two_channel_live_tap_is_available() {
        assert_eq!(availability(false, false, true, 2), ChannelAvailability::Available);
    }

    #[test]
    fn mono_offers_mix_alone() {
        let a = availability(false, false, true, 1);
        assert_eq!(a, ChannelAvailability::Mono);
        assert!(!a.is_available());
        assert_eq!(effective_view(ChannelView::Split, &a), ChannelView::Mix);
        assert!(a.reason().is_some_and(|r| r.contains("mono")));
    }

    /// The case that must never become a silent downmix.
    #[test]
    fn more_than_two_channels_declines_and_says_how_many() {
        for n in [3u16, 6, 8] {
            let a = availability(false, false, true, n);
            assert_eq!(a, ChannelAvailability::Multichannel(n));
            assert!(!a.is_available());
            assert!(a.reason().is_some_and(|r| r.contains(&n.to_string())));
            assert_eq!(effective_view(ChannelView::Overlay, &a), ChannelView::Mix);
        }
    }

    /// Native DSD has no PCM tap at all. Zero channels is what says so; an
    /// empty buffer would not, because a buffer is also empty a moment after a
    /// track starts.
    #[test]
    fn no_live_tap_is_distinct_from_mono() {
        let a = availability(false, false, true, NO_LIVE_TAP);
        assert_eq!(a, ChannelAvailability::NoLiveTap);
        assert_ne!(a, ChannelAvailability::Mono);
        assert!(a.reason().is_some_and(|r| r.contains("DoP")));
    }

    /// Pre-process without a sidecar declines, whatever the *file* is: what
    /// matters is whether channels were analysed, not whether they exist in the
    /// audio. A live tap reporting two channels must not make a mono cache look
    /// separable.
    #[test]
    fn preprocess_without_channels_declines_even_for_stereo_material() {
        let a = availability(true, false, true, 2);
        assert_eq!(a, ChannelAvailability::PreProcessMono);
        assert!(!a.is_available());
        // The reason has to name the remedy: this one costs an analysis, so
        // "switch to Real-time" alone would be telling half the story.
        let why = a.reason().expect("a refusal must say why");
        assert!(why.contains("analysed without channels"), "{why}");
        assert!(why.contains("stereo pre-analysis"), "{why}");
    }

    /// And with a sidecar it is available — from the cache, with no live tap at
    /// all, which is the whole point.
    #[test]
    fn preprocess_with_a_sidecar_is_available() {
        let a = availability(true, true, true, NO_LIVE_TAP);
        assert_eq!(a, ChannelAvailability::Available);
        assert!(a.is_available());
        assert!(a.reason().is_none());
        assert_eq!(effective_view(ChannelView::Diff, &a), ChannelView::Diff);
    }

    /// A cached pair the visualisation cannot draw reports the visualisation,
    /// not the cache — otherwise it would send the user off to analyse a track
    /// that has already been analysed.
    #[test]
    fn a_cached_pair_under_an_unsupported_style_blames_the_style() {
        let a = availability(true, true, false, NO_LIVE_TAP);
        assert_eq!(a, ChannelAvailability::UnsupportedStyle);
    }

    #[test]
    fn an_unsupported_visualisation_says_so_rather_than_going_blank() {
        let a = availability(false, false, false, 2);
        assert_eq!(a, ChannelAvailability::UnsupportedStyle);
        assert_eq!(effective_view(ChannelView::Left, &a), ChannelView::Mix);
    }

    /// The most fundamental obstacle is reported first, so a user does not
    /// clear one only to meet another they were never told about.
    #[test]
    fn the_reasons_are_ordered_from_most_fundamental() {
        // Pre-process without channels outranks both style and channel count,
        // because clearing it costs an analysis and clearing the style costs a
        // click. Being sent to spend minutes on the expensive one and then
        // meeting the cheap one is the outcome the ordering exists to prevent.
        assert_eq!(availability(true, false, false, 1), ChannelAvailability::PreProcessMono);
        // Style outranks channel count.
        assert_eq!(availability(false, false, false, 1), ChannelAvailability::UnsupportedStyle);
    }

    #[test]
    fn mix_never_asks_for_a_second_fft() {
        assert!(!ChannelView::Mix.needs_channels());
        assert!(!ChannelView::Mix.draws_left());
        assert!(!ChannelView::Mix.draws_right());
        for v in ChannelView::ALL.iter().filter(|v| **v != ChannelView::Mix) {
            assert!(v.needs_channels(), "{v:?}");
        }
    }

    #[test]
    fn each_view_draws_what_its_name_says() {
        assert!(ChannelView::Left.draws_left() && !ChannelView::Left.draws_right());
        assert!(ChannelView::Right.draws_right() && !ChannelView::Right.draws_left());
        for v in [ChannelView::Split, ChannelView::Overlay] {
            assert!(v.draws_left() && v.draws_right(), "{v:?}");
        }
    }

    /// An obstacle must not overwrite the preference: it comes back when the
    /// obstacle does.
    #[test]
    fn the_requested_view_survives_an_obstacle() {
        let want = ChannelView::Overlay;
        let blocked = availability(false, false, true, 1);
        assert_eq!(effective_view(want, &blocked), ChannelView::Mix);
        let cleared = availability(false, false, true, 2);
        assert_eq!(effective_view(want, &cleared), ChannelView::Overlay);
    }

    #[test]
    fn available_has_nothing_to_explain() {
        assert!(ChannelAvailability::Available.reason().is_none());
        for a in [
            ChannelAvailability::Mono,
            ChannelAvailability::Multichannel(6),
            ChannelAvailability::NoLiveTap,
            ChannelAvailability::PreProcessMono,
            ChannelAvailability::UnsupportedStyle,
        ] {
            assert!(a.reason().is_some(), "{a:?} must explain itself");
        }
    }
}
