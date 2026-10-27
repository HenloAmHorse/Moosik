//! Live stereo image and harmony for Analyze (1.6.0).
//!
//! Reads the PCM the analyser's tap already keeps and draws nothing; the
//! spectrum's own rendering is not touched. Two questions the bars cannot
//! answer: where each part of the spectrum sits between the speakers, and
//! which notes are sounding together.
//!
//! One 8192-point FFT per update serves both channels (left in the real part,
//! right in the imaginary, separated afterwards), and one more serves the
//! harmony. The shell asks for the stereo image every frame and for the
//! harmony about fifteen times a second, only while a panel is open and
//! something plays.

use crate::spectrum::{compute_chroma, cosine_sim, KS_MAJOR, KS_MINOR, NOTE_NAMES};
use rustfft::{num_complex::Complex, Fft, FftPlanner};
use std::sync::Arc;

/// Samples per analysis: 5.4 Hz a bin at 44.1 kHz, fine enough to tell
/// semitones apart from about 100 Hz up.
pub const N: usize = 8192;

/// Octave bands, by centre frequency.
pub const BANDS: [f32; 10] = [31.5, 63.0, 125.0, 250.0, 500.0, 1000.0, 2000.0, 4000.0, 8000.0, 16000.0];

/// Where one band sits.
#[derive(Clone, Copy, Debug, Default)]
pub struct Band {
    /// Power of left plus right, dB relative to full scale (about).
    pub level: f32,
    /// -1 all left, 0 centre, +1 all right.
    pub pan: f32,
    /// Side over mid plus side: 0 mono, 0.5 unrelated channels, 1 out of phase.
    pub width: f32,
    /// Phase correlation of the band, -1 to +1.
    pub corr: f32,
}

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum Kind {
    Maj,
    Min,
    Dim,
    Aug,
    Sus2,
    Sus4,
    Dom7,
    Maj7,
    Min7,
}

impl Kind {
    const ALL: [Kind; 9] = [Kind::Maj, Kind::Min, Kind::Dim, Kind::Aug, Kind::Sus2, Kind::Sus4, Kind::Dom7, Kind::Maj7, Kind::Min7];

    fn intervals(self) -> &'static [usize] {
        match self {
            Kind::Maj => &[0, 4, 7],
            Kind::Min => &[0, 3, 7],
            Kind::Dim => &[0, 3, 6],
            Kind::Aug => &[0, 4, 8],
            Kind::Sus2 => &[0, 2, 7],
            Kind::Sus4 => &[0, 5, 7],
            Kind::Dom7 => &[0, 4, 7, 10],
            Kind::Maj7 => &[0, 4, 7, 11],
            Kind::Min7 => &[0, 3, 7, 10],
        }
    }

    fn suffix(self) -> &'static str {
        match self {
            Kind::Maj => "",
            Kind::Min => "m",
            Kind::Dim => "dim",
            Kind::Aug => "aug",
            Kind::Sus2 => "sus2",
            Kind::Sus4 => "sus4",
            Kind::Dom7 => "7",
            Kind::Maj7 => "maj7",
            Kind::Min7 => "m7",
        }
    }

    /// A plain triad is the likelier reading of the same notes; the rest have
    /// to win by a margin.
    fn prior(self) -> f32 {
        match self {
            Kind::Maj | Kind::Min => 1.0,
            Kind::Dom7 | Kind::Maj7 | Kind::Min7 => 0.97,
            _ => 0.94,
        }
    }

    /// Major or minor, for reading a progression: sevenths count as their triad.
    fn minorish(self) -> Option<bool> {
        match self {
            Kind::Maj | Kind::Dom7 | Kind::Maj7 => Some(false),
            Kind::Min | Kind::Min7 => Some(true),
            _ => None,
        }
    }
}

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub struct Chord {
    pub root: usize,
    pub kind: Kind,
}

impl Chord {
    pub fn name(self) -> String {
        format!("{}{}", NOTE_NAMES[self.root], self.kind.suffix())
    }

    /// Its pitch classes.
    pub fn tones(self) -> Vec<usize> {
        self.kind.intervals().iter().map(|i| (self.root + i) % 12).collect()
    }
}

/// A key: root pitch class, minor or not, and how well the profile fits.
#[derive(Clone, Copy, PartialEq, Debug)]
pub struct Key {
    pub root: usize,
    pub minor: bool,
    pub fit: f32,
}

impl Key {
    pub fn name(self) -> String {
        format!("{} {}", NOTE_NAMES[self.root], if self.minor { "minor" } else { "major" })
    }
}

pub struct Live {
    fft: Arc<dyn Fft<f32>>,
    win: Vec<f32>,
    buf: Vec<Complex<f32>>,
    pub bands: [Band; BANDS.len()],
    pub corr: f32,
    /// Right over left, dB.
    pub balance: f32,
    pub width: f32,
    /// Pitch classes now, the loudest at 1.
    pub chroma: [f32; 12],
    /// Pitch classes over the last twenty seconds or so.
    key_acc: [f32; 12],
    pub key: Option<Key>,
    pub chord: Option<Chord>,
    pending: (Option<Chord>, u8),
    /// The chords as they changed, newest last.
    pub history: Vec<Chord>,
}

impl Default for Live {
    fn default() -> Self {
        let fft = FftPlanner::<f32>::new().plan_fft_forward(N);
        let win = (0..N).map(|i| 0.5 - 0.5 * (std::f32::consts::TAU * i as f32 / N as f32).cos()).collect();
        Live {
            fft,
            win,
            buf: vec![Complex::default(); N],
            bands: Default::default(),
            corr: 1.0,
            balance: 0.0,
            width: 0.0,
            chroma: [0.0; 12],
            key_acc: [0.0; 12],
            key: None,
            chord: None,
            pending: (None, 0),
            history: Vec::new(),
        }
    }
}

fn ema(a: f32, b: f32, k: f32) -> f32 {
    if a.is_finite() { a + (b - a) * k } else { b }
}

impl Live {
    /// A new track: forget the key and the chords.
    pub fn reset(&mut self) {
        let fft = Arc::clone(&self.fft);
        *self = Live { fft, ..Default::default() };
    }

    /// The stereo image of the last `N` frames. `dt`: seconds since the last call.
    pub fn stereo(&mut self, frames: &[[f32; 2]], rate: u32, dt: f32) {
        let Some(frames) = frames.len().checked_sub(N).map(|s| &frames[s..]) else { return };
        let k = 1.0 - (-dt / 0.15).exp();
        // Whole-signal correlation and balance, in the time domain.
        let (mut lr, mut ll, mut rr) = (0.0f64, 0.0f64, 0.0f64);
        for &[l, r] in frames {
            lr += (l * r) as f64;
            ll += (l * l) as f64;
            rr += (r * r) as f64;
        }
        if ll + rr < 1e-9 {
            return;
        }
        let corr = if ll * rr > 1e-18 { (lr / (ll * rr).sqrt()) as f32 } else { 0.0 };
        let (m, s) = (ll + rr + 2.0 * lr, ll + rr - 2.0 * lr);
        self.corr = ema(self.corr, corr, k);
        self.balance = ema(self.balance, (10.0 * ((rr + 1e-12) / (ll + 1e-12)).log10()).clamp(-24.0, 24.0) as f32, k);
        self.width = ema(self.width, (s / (m + s).max(1e-18)) as f32, k);

        // Both channels through one FFT: z = l + i·r, then
        // L_k = (Z_k + conj Z_{N-k}) / 2 and R_k = (Z_k − conj Z_{N-k}) / 2i.
        for (i, (&[l, r], w)) in frames.iter().zip(&self.win).enumerate() {
            self.buf[i] = Complex::new(l * w, r * w);
        }
        self.fft.process(&mut self.buf);
        let bin = rate as f32 / N as f32;
        let norm = (N as f32 * 0.5).powi(2);
        for (band, &c) in self.bands.iter_mut().zip(&BANDS) {
            let lo = ((c / std::f32::consts::SQRT_2) / bin).ceil().max(1.0) as usize;
            let hi = ((c * std::f32::consts::SQRT_2) / bin).floor() as usize;
            if hi >= N / 2 || hi < lo {
                band.level = f32::NEG_INFINITY;
                continue;
            }
            let (mut pl, mut pr, mut x, mut pm, mut ps) = (0.0f32, 0.0, 0.0, 0.0, 0.0);
            for j in lo..=hi {
                let (z, zc) = (self.buf[j], self.buf[N - j].conj());
                let l = (z + zc) * 0.5;
                let r = (z - zc) * Complex::new(0.0, -0.5);
                pl += l.norm_sqr();
                pr += r.norm_sqr();
                x += (l * r.conj()).re;
                pm += (l + r).norm_sqr();
                ps += (l - r).norm_sqr();
            }
            let level = 10.0 * ((pl + pr) / norm + 1e-12).log10();
            if pl + pr < 1e-9 {
                band.level = ema(band.level, -120.0, k);
                continue;
            }
            band.level = ema(band.level, level, k);
            band.pan = ema(band.pan, (pr - pl) / (pr + pl), k);
            band.width = ema(band.width, ps / (pm + ps), k);
            band.corr = ema(band.corr, x / (pl * pr).sqrt().max(1e-12), k);
        }
    }

    /// Notes, chord and key from the last `N` mono samples.
    pub fn harmony(&mut self, mono: &[f32], rate: u32, dt: f32) {
        let Some(mono) = mono.len().checked_sub(N).map(|s| &mono[s..]) else { return };
        let rms = (mono.iter().map(|x| x * x).sum::<f32>() / N as f32).sqrt();
        if rms < 1e-4 {
            // Silence: the notes fade, the chord goes, the key stays.
            self.chroma.iter_mut().for_each(|c| *c *= (-dt / 0.3).exp());
            self.settle(None);
            return;
        }
        for (i, (&s, w)) in mono.iter().zip(&self.win).enumerate() {
            self.buf[i] = Complex::new(s * w, 0.0);
        }
        self.fft.process(&mut self.buf);
        let norms: Vec<f32> = self.buf[..N / 2].iter().map(|c| c.norm() / N as f32).collect();
        let mut now = compute_chroma(&norms, rate, N);
        let top = now.iter().copied().fold(0.0, f32::max);
        if top <= 0.0 {
            return;
        }
        now.iter_mut().for_each(|c| *c /= top);
        let (k, kk) = (1.0 - (-dt / 0.25).exp(), 1.0 - (-dt / 20.0).exp());
        for i in 0..12 {
            self.chroma[i] = ema(self.chroma[i], now[i], k);
            self.key_acc[i] = ema(self.key_acc[i], now[i], kk);
        }
        self.key = best_key(&self.key_acc);
        self.settle(best_chord(&self.chroma));
    }

    /// A chord has to win twice running before it shows.
    fn settle(&mut self, c: Option<Chord>) {
        if c == self.chord {
            self.pending = (c, 0);
            return;
        }
        if self.pending.0 == c {
            self.pending.1 += 1;
        } else {
            self.pending = (c, 1);
        }
        if self.pending.1 >= 2 {
            self.chord = c;
            if let Some(c) = c
                && self.history.last() != Some(&c)
            {
                self.history.push(c);
                if self.history.len() > 16 {
                    self.history.remove(0);
                }
            }
        }
    }

    /// The last four chords go I – V – vi – IV in the key, in some rotation.
    pub fn four_chords(&self) -> bool {
        let (Some(key), Some(last)) = (self.key, self.history.len().checked_sub(4).map(|s| &self.history[s..])) else {
            return false;
        };
        if key.minor {
            return false;
        }
        const CYCLE: [(usize, bool); 4] = [(0, false), (7, false), (9, true), (5, false)];
        let got: Vec<Option<(usize, bool)>> =
            last.iter().map(|c| c.kind.minorish().map(|m| ((c.root + 12 - key.root) % 12, m))).collect();
        (0..4).any(|rot| (0..4).all(|i| got[i] == Some(CYCLE[(i + rot) % 4])))
    }
}

fn best_key(chroma: &[f32; 12]) -> Option<Key> {
    let mut best: Option<Key> = None;
    for root in 0..12 {
        for minor in [false, true] {
            let p = if minor { &KS_MINOR } else { &KS_MAJOR };
            let prof: [f32; 12] = std::array::from_fn(|i| p[(i + 12 - root) % 12]);
            let fit = cosine_sim(&centred(chroma), &centred(&prof));
            if best.is_none_or(|b| fit > b.fit) {
                best = Some(Key { root, minor, fit });
            }
        }
    }
    best.filter(|k| k.fit > 0.3)
}

fn centred(v: &[f32; 12]) -> [f32; 12] {
    let m = v.iter().sum::<f32>() / 12.0;
    v.map(|x| x - m)
}

/// How strong the 3rd and 5th harmonics show in the log-compressed chroma.
const HARM_3: f32 = 0.5;
const HARM_5: f32 = 0.25;

fn best_chord(chroma: &[f32; 12]) -> Option<Chord> {
    let c = centred(chroma);
    let mut best = (0.0f32, None);
    for root in 0..12 {
        for kind in Kind::ALL {
            // Each note with the pitch classes of its 3rd and 5th harmonics,
            // so a triad's own overtones do not read as a seventh.
            let mut t = [0.0f32; 12];
            for &i in kind.intervals() {
                for (h, w) in [(0, 1.0), (7, HARM_3), (4, HARM_5)] {
                    t[(root + i + h) % 12] += w;
                }
            }
            let s = cosine_sim(&c, &centred(&t)) * kind.prior();
            if s > best.0 {
                best = (s, Some(Chord { root, kind }));
            }
        }
    }
    if best.0 >= 0.6 { best.1 } else { None }
}

#[cfg(test)]
mod tests {
    use super::*;

    const SR: u32 = 44_100;

    fn tone(freqs: &[f32], n: usize) -> Vec<f32> {
        (0..n)
            .map(|i| {
                let t = i as f32 / SR as f32;
                freqs.iter().map(|f| (std::f32::consts::TAU * f * t).sin() * 0.2).sum()
            })
            .collect()
    }

    fn hz(midi: i32) -> f32 {
        440.0 * 2f32.powf((midi - 69) as f32 / 12.0)
    }

    fn stereo_of(l: &[f32], r: &[f32]) -> Live {
        let fr: Vec<[f32; 2]> = l.iter().zip(r).map(|(&a, &b)| [a, b]).collect();
        let mut a = Live::default();
        a.stereo(&fr, SR, 10.0);
        a
    }

    #[test]
    fn the_stereo_image_reads_mono_wide_and_one_sided() {
        let s = tone(&[1000.0], N);
        let mono = stereo_of(&s, &s);
        let b = mono.bands[5];
        assert!(b.width < 0.01 && b.pan.abs() < 0.01 && b.corr > 0.99, "{b:?}");
        assert!(mono.corr > 0.99 && mono.width < 0.01);

        let neg: Vec<f32> = s.iter().map(|x| -x).collect();
        let anti = stereo_of(&s, &neg);
        assert!(anti.bands[5].width > 0.99 && anti.bands[5].corr < -0.99, "{:?}", anti.bands[5]);
        assert!(anti.corr < -0.99);

        let quiet = vec![0.0; N];
        let right = stereo_of(&quiet, &s);
        assert!(right.bands[5].pan > 0.99, "{:?}", right.bands[5]);
        assert!(right.balance > 20.0);
        // Nothing at 63 Hz: the band is silent, not centred.
        assert!(right.bands[1].level < right.bands[5].level - 40.0);
    }

    fn harmony_of(midis: &[i32]) -> Live {
        // Each note with four overtones falling off as 1/k, as a string's do.
        let s: Vec<f32> = (0..N)
            .map(|i| {
                let t = i as f32 / SR as f32;
                midis.iter().flat_map(|&m| (1..=5).map(move |k| (k, hz(m))))
                    .map(|(k, f)| (std::f32::consts::TAU * k as f32 * f * t).sin() * 0.2 / k as f32)
                    .sum()
            })
            .collect();
        let mut a = Live::default();
        for _ in 0..3 {
            a.harmony(&s, SR, 0.07);
        }
        a
    }

    #[test]
    fn triads_are_named() {
        assert_eq!(harmony_of(&[60, 64, 67]).chord.map(Chord::name).as_deref(), Some("C"));
        assert_eq!(harmony_of(&[57, 60, 64]).chord.map(Chord::name).as_deref(), Some("Am"));
        assert_eq!(harmony_of(&[55, 59, 62, 65]).chord.map(Chord::name).as_deref(), Some("G7"));
    }

    #[test]
    fn silence_drops_the_chord() {
        let mut a = harmony_of(&[60, 64, 67]);
        for _ in 0..3 {
            a.harmony(&vec![0.0; N], SR, 0.07);
        }
        assert_eq!(a.chord, None);
    }

    #[test]
    fn a_scale_gives_its_key() {
        let mut a = Live::default();
        // C major, tonic and dominant weighted as a melody would.
        for m in [60, 62, 64, 65, 67, 69, 71, 72, 67, 64, 60, 67, 60] {
            let s = tone(&[hz(m)], N);
            for _ in 0..20 {
                a.harmony(&s, SR, 0.07);
            }
        }
        assert_eq!(a.key.map(|k| k.name()).as_deref(), Some("C major"));
    }

    #[test]
    fn four_chords_in_any_rotation() {
        let ch = |root, kind| Chord { root, kind };
        let mut a = Live { key: Some(Key { root: 0, minor: false, fit: 0.9 }), ..Default::default() };
        a.history = vec![ch(9, Kind::Min), ch(5, Kind::Maj), ch(0, Kind::Maj), ch(7, Kind::Dom7)];
        assert!(a.four_chords());
        a.history = vec![ch(0, Kind::Maj), ch(5, Kind::Maj), ch(9, Kind::Min), ch(7, Kind::Maj)];
        assert!(!a.four_chords());
    }
}
