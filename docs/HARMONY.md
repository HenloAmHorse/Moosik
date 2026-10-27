# Harmony and Stereo (1.6.0)

Analyze has two panels under the plot, switched on beside the **Analyzer**
heading: **Stereo** and **Harmony**. Both read audio the spectrum's tap
already keeps; neither changes how the spectrum is drawn.

## Stereo

What the bars cannot show: where each part of the sound sits between the
speakers.

- **Scope** — the last 1,024 sample pairs, mid up and side across. A mono
  signal is a vertical line; a wide one spreads sideways. The scope scales
  itself to the signal's level.
- **Width** — side energy over mid plus side, as a percentage: 0% mono, about
  50% two unrelated channels, 100% fully out of phase.
- **Balance** — right over left in dB, or *centre* within ±0.5 dB.
- **Correlation** — the bar under the scope, −1 to +1. It turns red below 0,
  where a mono fold-down would cancel part of the sound.
- **Stage** — ten octave bands, 31.5 Hz to 16 kHz, low at the bottom. Each is
  a capsule placed where the band sits (left to right), as wide as the band
  is, and red when the band is out of phase. Bands more than 48 dB below the
  loudest are hidden.

Everything comes from one 8192-point FFT of both channels, worked out every
frame while the panel is open and a two-channel track plays; values are
smoothed over about 0.15 s. A mono track says so instead of drawing.

## Harmony

The chord sounding now, the one coming next, the key, and a wheel of the
twelve notes.

### Two sources

**The live guess** works with nothing installed. About fifteen times a
second it compares the notes sounding (a chroma from an 8192-point FFT)
with chord templates; a chord has to win twice running before it shows. The
key comes from the last twenty seconds or so against major and minor key
profiles, marked *(guessing)* while the fit is weak. It is quick and often
wrong on dense mixes; the line under the panel says *Guessing*.

**The chord model** is a Rust port of lv-chordia 1.1.0 (Jiang, Chen, Li and
Xia, *Large-Vocabulary Chord Transcription via Chord Structure Decomposition*,
ISMIR 2019; MIT). It listens to the whole track once, then knows every chord
in it, including which comes next. Its vocabulary has 301 chords: on each of
twelve roots, major and minor triads with common inversions, sevenths
(dominant, major, minor), ninths, elevenths, thirteenths, diminished,
diminished seventh, half-diminished, augmented and suspended chords; plus no
chord.

- **Getting it:** *Get the chord model (27 MB, MIT)* under the panel
  downloads the lv-chordia wheel from PyPI, checks it against a pinned
  SHA-256, and keeps only its five weight files, each checked against its own
  pinned SHA-256. They are read by a reader that runs nothing. Nothing is
  downloaded unless you press the button.
- **What it does:** a constant-Q transform (36 bins an octave, after a tuning
  estimate, matching librosa's `hybrid_cqt` to within 1e-4), five networks
  averaged, and a Viterbi path over the 301 chords. The Rust port agrees with
  the Python original to within 3e-7 in probability and gives the same chords.
- **How long:** about 4 seconds for a three-minute song on an 8-thread desktop
  CPU with AVX2, plus decoding. While it works the panel keeps guessing live;
  the model's chords replace the guess when it finishes. A song worked out
  once is remembered (see *Files* below).
- **Up to 15 minutes a track.** The network holds the whole track at once,
  under 100 MB a minute at its peak (286 MB measured for three minutes);
  past 15 minutes (about 1.5 GB) the panel
  stays on the live guess and says why, rather than risk running out of
  memory on a long mix.

### How sure it is

Each chord carries the model's confidence: the average probability of its
triad over the chord's length. On 20 test tracks rendered with instruments
the model had not heard, the confidence matched how often it was right:

| Confidence | Share of the time | Major/minor right |
|---|---|---|
| 0.9 or more | 55% | 92.5% |
| 0.7–0.9 | 30% | 81.6% |
| 0.5–0.7 | 10% | 54.9% |
| below 0.5 | 5% | under 40% |

So the panel shows a chord below 0.7 dimmed, and below 0.5 faint with a
question mark. On the same tracks the model's root was right 79.8% of the
time, major/minor 82.7%, and sevenths 71.9% (mir_eval). For comparison,
trained human annotators disagree with each other on about one chord in ten.
These figures come from synthetic test tracks, not real recordings; your
music may score better or worse.

### Correcting a chord

Click the chord to open the picker: a root, then the chord (major, minor, 7,
maj7, m7, dim, m7b5, sus2, sus4, aug, or N.C.). Your choice replaces the
model's for that stretch, shows at full confidence, and is kept: corrections
live in their own file beside the model's, so a newer model does not lose
them.

### Files

In `~/.moosik/cache/chords/`, named by the track the way the analysis cache
knows it (path, size, modified time and a sample of the contents; see
[CACHE.md](CACHE.md)):

- `<key>.v2.json` — the model's chords, written aside then moved into place,
  and only if the file was unchanged while it was read;
- `<key>.fix.json` — your corrections.

Changing a file (its audio or its tags) gives it a new key: the model works it
out again, and earlier corrections no longer apply to it. Nothing removes old
chord files automatically yet.

The weights are in `~/.moosik/models/chords-lv-chordia-1.1.0/`.

## Silly extras

Both panels hide a few small animations (a chord that naps, a scope that
worries when the phase flips, and more). Each is found by doing what its hint
says and can be switched off in Settings → Silly extras.
