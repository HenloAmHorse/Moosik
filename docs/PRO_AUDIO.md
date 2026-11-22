# Pro audio tools (1.6.2)

Measuring a file, checking what it really is, and listening on headphones.
Nothing here changes a file, and with the headphone tools off the output is
exactly what it was.

## Measure this file

**Info → Measure this file** decodes the whole file once, at its own rate and
precision, on a background thread (about half a second for a 90-second CD-rate
track), with progress and **Cancel**. It reports:

| | What it is |
|---|---|
| Integrated loudness | ITU-R BS.1770-4: K-weighted, 400 ms blocks, gated at −70 LUFS and −10 LU |
| Loudness range | EBU Tech 3342: 3 s windows, gated at −70 LUFS and −20 LU, 10th to 95th percentile |
| Max short-term / momentary | the loudest 3 s and 400 ms windows |
| True peak | 4× oversampled up to 96 kHz, 2× up to 192 kHz (Kaiser-windowed sinc, 12 taps a phase); never below the sample peak |
| Sample peak | the largest sample |
| PLR / PSR | true peak minus integrated loudness / minus max short-term loudness |
| Crest factor | sample peak minus RMS of all channels |
| Dynamic range | the DR meter's figure ("DR8", as foobar2000's DR Meter and the DR database give it): each channel in 3 s blocks, the second-highest block peak over the RMS (×√2) of the loudest fifth of blocks, the channels averaged; shown rounded, with the exact value |
| DC offset | each channel's mean, as a percentage of full scale |
| L/R correlation | over the whole file |
| Bits used | for integer files: the bits the samples actually use |
| Clipped runs | three or more samples in a row at full scale, with their times |
| Inter-sample overs | where the reconstructed wave goes above 0 dBFS between samples |
| FLAC MD5 | whether the decoded audio matches the MD5 stored in the file's header |

Checked against: a full-scale 997 Hz sine reads −3.01 LUFS in one channel and
0.00 LUFS in two, at 44.1, 48 and 96 kHz, as BS.1770 defines; a test track
reads the same integrated loudness, loudness range and true peak as ffmpeg's
`ebur128` filter (−16.7 LUFS, 1.2 LU, −3.0 dBTP). The DR figure equals an
independent implementation of the DR meter's rules on five songs (44.1 to
192 kHz) to within a millionth of a dB. Rounded, it is the same as ffmpeg's
`drmeter`, which bins its levels and so differs by up to 0.1 dB.

Surround files weight the channels as BS.1770 does for 5.0/5.1 (surrounds
×1.41, LFE left out); other layouts count every channel equally.

## What the file really is

Above the numbers, the findings that matter, in words:

- **FLAC MD5** — *matches*: the audio decodes exactly as it was encoded;
  *does not match*: the file is damaged.
- **Padded bit depth** — a 24-bit file whose samples use only 16 bits is
  16-bit audio in a bigger container ("fake 24-bit").
- **A lossy source** — a lossless file whose spectrum stops sharply below
  20.5 kHz, as MP3 and AAC encoders cut it, was probably made from a lossy
  file ("fake lossless"). This uses the cut-off the spectrum analysis already
  finds, so the track must have been analysed; the existing **Hi-res check**
  row still flags upsampled ("fake hi-res") files.
- Clipping, inter-sample overs, a channel that looks inverted (correlation
  below −0.3) and a DC offset above 1 %.

These are evidence, not proof. A sharp 16 kHz cut-off can be a deliberate
choice, and a genuinely 24-bit recording can be quiet enough to leave its
lowest bits unused in places, though not across a whole file.

## Comparing two masters

**Info → Compare with another file…** takes another master of the same song
(a remaster, another release) and reports how much louder it is (integrated
loudness, LU), then draws its tone against the track's once their loudness
is matched: the difference per third octave, 20 Hz to 20 kHz, ±6 dB. Each
file's long-term average spectrum is the mix of its channels through 8192-point
Hann windows, half overlapping.

Checked: the same noise 6 dB louder reads 6.0 LU louder with a flat
difference; a low-passed copy reads more than 10 dB down in the treble
against the bass.

## Blind test (ABX)

After a comparison, **Blind test (ABX)…** loads both files into memory (up to
ten minutes of each) and plays them through a small player of its own, so A,
B and X switch instantly at the same moment of the music, looping. The louder
file is turned down by the loudness difference, so level cannot give the
answer away. X is A or B at random each trial; after each answer the score
shows the chance of doing that well by guessing (one-sided binomial), with a
plain verdict once there are eight trials (below 5 %: unlikely to be
guessing; below 1 %: very unlikely).

The two files must have the same sample rate and channel count. The test
plays through the system's shared output; if bit-perfect output holds the
device, it says so. The main player pauses when a test starts, and the test
ends with the Info window.

## On the timeline

After **Measure this file**, the seek bar marks the track's clipped runs
(full height) and its inter-sample overs (short ticks along the top).

## Headphone EQ from AutoEq

**Settings → Headphones → Find your headphones…** opens **Headphone EQ**.
The same window opens from **Settings → Pro tools** and from the EQ in
Analyze. In the window:

- **Search** the headphones and earphones that
  [AutoEq](https://github.com/jaakkopasanen/AutoEq) has measurements of
  (about 6,000). Spelling and spaces need not be exact: "hd650" and
  "sennhiser hd 650" both find the Sennheiser HD 650, words in any order.
- **Measured by:** pick one of the headphone's measurements.
- **Target:** autoeq.app's recommended curve comes first.
- **Bass boost:** it starts where autoeq.app starts it. Treble boost, tilt and
  max gain are under **Advanced**.
- **Use in Moosik:** puts the correction in the EQ as a global preset named
  "<headphones> · AutoEq", switched on. If bit-perfect output is on, the
  window says the EQ is bypassed there.
- **Save for other apps:** writes, in the headphones' own folder (**Open
  folder**):
  - `ParametricEQ.txt`, for Equalizer APO, Peace and other parametric
    equalizers;
  - `GraphicEQ.txt`, for Wavelet, JamesDSP and Equalizer APO;
  - fixed 10- and 31-band versions, for equalizers with sliders;
  - for USB Audio Player PRO, the values to type into its parametric
    equalizer, in its own layout;
  - `TBEQPresets\<headphones>.xml`, a ToneBoosters preset UAPP can load.
    This one is experimental: the format follows the community converter
    EqConverter and has not been confirmed on a phone;
  - a read-me that says which file goes with which app.

No sample rate is asked. Moosik's EQ is computed again at each song's own
rate, and the files carry frequencies, not coefficients.

The equalizers are made by autoeq.app, the AutoEq project's own web app. Its
server is open source but not a published API, so Moosik checks every answer
before using it. Only the headphones' name, the measurement and the settings
above are sent.

The list of headphones and targets (about 0.7 MB) is kept in
`~/.moosik/autoeq`. Moosik checks it for changes in the background when it
starts, at most once an hour; the window's checkbox turns that off.
Searching needs no network. A correction made once is kept, and **Use in
Moosik** falls back to it when autoeq.app does not answer.

**Analyze → Controls → EQ → Import AutoEq…** still reads a `ParametricEQ.txt`
from AutoEq or Equalizer APO: its preamp and up to 32 filters (peaking, low
and high shelves, high and low pass, notch; filters marked OFF stay off). It
becomes a global preset named after the file.

The EQ now has a **Preamp**: gain before the bands, so a profile's boosts do
not clip. It is saved in presets, drawn in the EQ curve, and included when the
bars show the EQ.

## Crossfeed

**Settings → Headphones → Crossfeed**: each ear also hears the other channel,
low-passed and quieter, and its own a little high-boosted, as from speakers in
front. After Boris Mikhaylov's bs2b, with its three strengths:

| | Cut-off | Feed |
|---|---|---|
| Light | 700 Hz | 4.5 dB |
| Chu Moy | 700 Hz | 6 dB |
| Jan Meier | 650 Hz | 9.5 dB |

Stereo only. It runs in the EQ stage, so it is never in bit-perfect output.
A centred low-frequency signal comes out up to about 1.8 dB louder, as with
bs2b; the EQ's preamp can take that back.

## Room correction

**Settings → Headphones → Room correction → Load impulse response…** takes a
WAV (or FLAC/AIFF) impulse response from REW or another room-measurement
tool: one channel for both, or left and right, up to about 2.7 s at 48 kHz.
It is convolved with what plays after the EQ (uniformly partitioned
overlap-save, 1024-sample blocks), so it adds about 21 ms of delay; a response
at another sample rate is resampled to the stream's. REW's parametric EQ
export uses the same text as AutoEq's and goes through **Import AutoEq…**.

Checked: a unit impulse passes audio unchanged one block late; a 2500-tap
response matches direct convolution to 1e-4.

## Which switch does what

The EQ's own on/off switch turns its bands and preamp on and off; crossfeed
and room correction have their own settings and stay on when the EQ is off.
Anything in this stage that changes samples (bands, preamp, crossfeed, room
correction) shows as **EQ active** in the signal-path report. None of it is
ever in bit-perfect output.

## Tempo and beats

With **Harmony** open, Moosik finds the playing track's beats once, in the
background (no model needed), and keeps them beside its chords: the panel
shows the tempo and a dot that lands on every beat. Onset strength is
spectral flux on the same 22050 Hz mono the chord model hears; the tempo is
the strongest pulse between 40 and 240 BPM, weighted toward 120; the beats
follow by dynamic programming (Ellis, 2007). When eighth-note hi-hats make
the half beat the strongest pulse, a clear pulse at half the tempo is taken
as the beat.

Checked: click tracks at 90, 120 and 140 BPM read within 1 BPM with the beats
within 25 ms of the clicks; five test tracks made at 105, 93, 84, 127 and
120 BPM read 105.2, 92.9, 84.0, 126.7 and 120.1. Music without a steady pulse
shows no tempo. Over 15 minutes, as for chords, nothing is analysed.

## Sections

With the beats found, Moosik also marks the song's parts: a strip along the
bottom of the seek bar, a colour and a letter a part (A, B, A…). One chroma per
beat, a self-similarity matrix of them, Foote's checkerboard novelty for the
boundaries (the highest point within 16 beats either side, above the mean
plus half a deviation), and parts with nearly the same harmony share a
letter; the same part twice in a row is one part.

It hears harmony only, not lyrics or arrangement: a chorus over the verse's
chords reads as the verse, and a song on one chord is one part.

Checked: verse–chorus–verse–chorus on distinct chords reads A B A B with the
boundaries within a beat; a key-change test track splits where the key
changes.

## Practice: tempo and key

**Settings → Practice**: tempo from 50 % to 150 % without changing the key,
and the key up or down six semitones without changing the tempo, for playing
along. The key moves by resampling, the tempo by WSOLA (waveform-similarity
overlap-add, 40 ms frames, ±10 ms search), as the last stage of the shared
route, after the EQ; at 100 % and as recorded, samples pass through
untouched. The position, seek bar and lyrics follow the new tempo, and the
seek bar shows a reminder ("80% +2") while either is changed. It lasts for the
session, counts as processing in the signal-path report, and is never in
bit-perfect output. The spectrum shows the recording as it is.

Checked: at 100 % output equals input exactly; at 50 % a 440 Hz tone lasts
twice as long and stays 440 Hz; at 125 % it lasts 0.8 as long at 440 Hz;
+12 semitones gives 880 Hz at the same length.
