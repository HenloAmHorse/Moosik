//! ABX blind listening tests (1.6.2). Both files are decoded into memory and
//! played by a small player of their own, so A, B and X switch instantly at
//! the same moment of the music; the louder file is turned down to the
//! quieter one's loudness, so a level difference cannot give the answer away.
//! X is A or B at random each trial; the score says how likely it is to have
//! been guessed.

use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicBool, AtomicU32, AtomicU8, AtomicUsize, Ordering};
use std::sync::Arc;

/// The most of each file taken: ten minutes is plenty to listen for a difference.
pub const MAX_SECONDS: u32 = 600;

/// One file in memory, interleaved.
pub struct Clip {
    pub rate: u32,
    pub channels: usize,
    pub samples: Vec<f32>,
    pub loudness: Option<f64>,
}

pub fn load(path: &Path, cancel: &AtomicBool, progress: &AtomicU32) -> Result<Clip, String> {
    let (mut rate, mut channels, mut samples) = (0u32, 0usize, Vec::<f32>::new());
    crate::measure::each_block(path, cancel, progress, |r, c, s| {
        rate = r;
        channels = c;
        if samples.len() < MAX_SECONDS as usize * r as usize * c {
            samples.extend_from_slice(s);
        }
    })?;
    if samples.is_empty() {
        return Err("no audio in it".into());
    }
    let loudness = crate::measure::loudness_of(&samples, rate, channels);
    Ok(Clip { rate, channels, samples, loudness })
}

/// Which one is sounding.
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum Side {
    A,
    B,
    X,
}

/// What the player and the screen share: the position (frames), the side,
/// whether it is playing, and what X is this trial.
#[derive(Default)]
pub struct Shared {
    pub pos: AtomicUsize,
    side: AtomicU8,
    pub playing: AtomicBool,
    x_is_a: AtomicBool,
}

impl Shared {
    pub fn side(&self) -> Side {
        match self.side.load(Ordering::Relaxed) {
            0 => Side::A,
            1 => Side::B,
            _ => Side::X,
        }
    }
    pub fn set_side(&self, s: Side) {
        self.side.store(match s { Side::A => 0, Side::B => 1, Side::X => 2 }, Ordering::Relaxed);
    }
}

/// The samples the player hands to the output: A, B or X at the shared
/// position, level-matched, silent while stopped. It loops.
pub struct Source {
    a: Arc<Clip>,
    b: Arc<Clip>,
    gains: (f32, f32),
    shared: Arc<Shared>,
    frames: usize,
    c: usize,
}

impl Source {
    pub fn new(a: Arc<Clip>, b: Arc<Clip>, shared: Arc<Shared>) -> Result<Source, String> {
        if a.rate != b.rate || a.channels != b.channels {
            return Err(format!(
                "the files differ in format ({} Hz × {} against {} Hz × {}); ABX needs the same rate and channels",
                a.rate, a.channels, b.rate, b.channels
            ));
        }
        let frames = (a.samples.len() / a.channels).min(b.samples.len() / b.channels);
        // Turn the louder one down by the difference.
        let gains = match (a.loudness, b.loudness) {
            (Some(la), Some(lb)) if la > lb => (10f32.powf(((lb - la) / 20.0) as f32), 1.0),
            (Some(la), Some(lb)) => (1.0, 10f32.powf(((la - lb) / 20.0) as f32)),
            _ => (1.0, 1.0),
        };
        Ok(Source { a, b, gains, shared, frames, c: 0 })
    }

    /// How many dB the louder file was turned down (A, B).
    pub fn trims_db(&self) -> (f32, f32) {
        (20.0 * self.gains.0.log10(), 20.0 * self.gains.1.log10())
    }
}

impl Iterator for Source {
    type Item = f32;
    fn next(&mut self) -> Option<f32> {
        let ch = self.a.channels;
        let pos = self.shared.pos.load(Ordering::Relaxed) % self.frames.max(1);
        let a = match self.shared.side() {
            Side::A => true,
            Side::B => false,
            Side::X => self.shared.x_is_a.load(Ordering::Relaxed),
        };
        let v = if !self.shared.playing.load(Ordering::Relaxed) {
            0.0
        } else if a {
            self.a.samples[pos * ch + self.c] * self.gains.0
        } else {
            self.b.samples[pos * ch + self.c] * self.gains.1
        };
        self.c += 1;
        if self.c == ch {
            self.c = 0;
            if self.shared.playing.load(Ordering::Relaxed) {
                self.shared.pos.store((pos + 1) % self.frames.max(1), Ordering::Relaxed);
            }
        }
        Some(v)
    }
}

impl rodio::Source for Source {
    fn current_frame_len(&self) -> Option<usize> {
        None
    }
    fn channels(&self) -> u16 {
        self.a.channels as u16
    }
    fn sample_rate(&self) -> u32 {
        self.a.rate
    }
    fn total_duration(&self) -> Option<std::time::Duration> {
        None
    }
}

/// The chance of getting at least `right` of `n` by guessing (one-sided
/// binomial, p = ½).
pub fn p_value(right: u32, n: u32) -> f64 {
    if n == 0 {
        return 1.0;
    }
    // Sum C(n, k) / 2^n for k >= right, in logs to stay finite.
    let ln_choose = |n: u32, k: u32| -> f64 { (1..=k).map(|i| ((n - k + i) as f64).ln() - (i as f64).ln()).sum() };
    (right..=n).map(|k| (ln_choose(n, k) - n as f64 * std::f64::consts::LN_2).exp()).sum::<f64>().min(1.0)
}

/// A test in progress: the two files, the player, and the answers so far.
pub struct Session {
    pub a: PathBuf,
    pub b: PathBuf,
    pub shared: Arc<Shared>,
    pub trims: (f32, f32),
    pub seconds: f32,
    /// Each trial: was the answer right.
    pub answers: Vec<bool>,
    _out: Option<(rodio::OutputStream, rodio::Sink)>,
    rng: u64,
}

impl Session {
    /// Start playing (stopped) on the default output.
    pub fn start(a: PathBuf, b: PathBuf, ca: Clip, cb: Clip) -> Result<Session, String> {
        let shared = Arc::new(Shared::default());
        let seconds = (ca.samples.len() / ca.channels).min(cb.samples.len() / cb.channels) as f32 / ca.rate as f32;
        let src = Source::new(Arc::new(ca), Arc::new(cb), shared.clone())?;
        let trims = src.trims_db();
        let (stream, handle) = rodio::OutputStream::try_default()
            .map_err(|e| format!("no audio output for the test ({e}); if bit-perfect output holds the device, switch it off first"))?;
        let sink = rodio::Sink::try_new(&handle).map_err(|e| format!("no audio output for the test: {e}"))?;
        sink.append(src);
        let seed = std::time::SystemTime::now().duration_since(std::time::UNIX_EPOCH).map_or(1, |d| d.as_nanos() as u64) | 1;
        let mut s = Session { a, b, shared, trims, seconds, answers: Vec::new(), _out: Some((stream, sink)), rng: seed };
        s.new_trial();
        Ok(s)
    }

    /// X becomes A or B at random.
    fn new_trial(&mut self) {
        self.rng ^= self.rng << 13;
        self.rng ^= self.rng >> 7;
        self.rng ^= self.rng << 17;
        self.shared.x_is_a.store(self.rng & 1 == 1, Ordering::Relaxed);
    }

    /// The listener says X is A (or not); the next trial starts.
    pub fn answer(&mut self, x_is_a: bool) {
        self.answers.push(self.shared.x_is_a.load(Ordering::Relaxed) == x_is_a);
        self.new_trial();
    }

    pub fn right(&self) -> u32 {
        self.answers.iter().filter(|&&r| r).count() as u32
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn clip(v: f32, frames: usize, loudness: Option<f64>) -> Arc<Clip> {
        Arc::new(Clip { rate: 48_000, channels: 2, samples: vec![v; frames * 2], loudness })
    }

    #[test]
    fn the_player_switches_sides_at_the_same_moment_and_matches_levels() {
        let shared = Arc::new(Shared::default());
        // B is 6 dB louder, so it is turned down to A's level.
        let mut s = Source::new(clip(0.25, 100, Some(-20.0)), clip(0.5, 100, Some(-14.0)), shared.clone()).unwrap();
        assert_eq!(s.next(), Some(0.0), "silent until played");
        shared.playing.store(true, Ordering::Relaxed);
        s.c = 0;
        let a: Vec<f32> = (0..20).map(|_| s.next().unwrap()).collect();
        assert!(a.iter().all(|&v| (v - 0.25).abs() < 1e-6));
        let at = shared.pos.load(Ordering::Relaxed);
        assert_eq!(at, 10, "ten frames played");
        shared.set_side(Side::B);
        let b = s.next().unwrap();
        assert!((b - 0.5 * 10f32.powf(-6.0 / 20.0)).abs() < 1e-4, "{b}");
        assert_eq!(shared.pos.load(Ordering::Relaxed), at, "same place");
        let (ta, tb) = s.trims_db();
        assert!(ta == 0.0 && (tb + 6.0).abs() < 1e-3);
    }

    #[test]
    fn different_formats_are_refused() {
        let a = Arc::new(Clip { rate: 44_100, channels: 2, samples: vec![0.0; 10], loudness: None });
        assert!(Source::new(a, clip(0.0, 5, None), Arc::new(Shared::default())).is_err());
    }

    #[test]
    fn guessing_chances_are_binomial() {
        assert!((p_value(10, 10) - 1.0 / 1024.0).abs() < 1e-12);
        assert!((p_value(0, 10) - 1.0).abs() < 1e-12);
        assert!((p_value(9, 10) - 11.0 / 1024.0).abs() < 1e-12);
        assert!((p_value(12, 16) - 0.0384).abs() < 1e-3);
        assert_eq!(p_value(0, 0), 1.0);
    }
}
