//! The exact identity of an analysis: what it was computed from, in full.
//!
//! # Why this exists
//!
//! The cache name used to be built from the settings rounded for readability —
//! hundredths, tenths, whole hertz — so two settings that change the result
//! could share a file, and a cache said nothing about which of them wrote it.
//! This is the replacement for every analysis written from now on. It is one
//! descriptor, [`AnalysisSettings`], used for four things that must agree:
//!
//! * the **store stem** — `store::stem(track, canonical)`; a 1.5.3 prereview
//!   build named its file `{source}_e{stamp}.spectrumcache` (FNV-1a-64), which
//!   is now only recognised as a legacy alternative;
//! * **completion matching** — a finished analysis is shown only if the
//!   descriptor it was launched with equals the one in force now;
//! * **verification on load** — the file carries its own record (in the v4
//!   header, see `cache`), and it is loaded as an exact hit only if that record
//!   is the one these settings produce at this track's sample rate;
//! * **alternate discovery** — a verified cache at other transform settings is
//!   compatible only if its record agrees on every other field, geometry
//!   included.
//!
//! # The record
//!
//! `key=value` fields in a fixed order, separated by `;`, every continuous
//! setting as the hex of its exact `f32` bits. Nothing is rounded and nothing
//! is hashed by Rust's `DefaultHasher`, whose algorithm is unspecified. A field
//! a method ignores is absent, so moving an FFT control does not change a
//! Superlet identity. The sample rate and hop are added by whoever knows them
//! (the analysis that decoded the file, or the display that is playing it); they
//! are not in the name, because a name must be computable for a track that has
//! not been opened.
//!
//! The record names the source by its *path*. The cache store adds the
//! content side (size, last-write time, sampled digest; `store::Descriptor`),
//! so an exact hit also needs the source to be unchanged.
//!
//! Anything that changes an analysis without changing a setting — a change to
//! the transform's code — must change its identity. The true CQT has its own
//! `cqt-rev` field; bumping it does not retire unrelated FFT or Superlet
//! records. [`SCHEMA`] remains the version of the common record format. The
//! compute route (CPU or GPU) is not part of the
//! identity: it is an implementation of the same transform, not a setting.
use std::path::{Path, PathBuf};

use super::aslt::AsltConfig;
use super::freq_scale::FreqScale;
use super::{AnalysisSettings, BarMappingMode, InterpolationMode, WindowFn};

/// The record's first field. Bump it to retire every record written before.
pub const SCHEMA: &str = "moosik-analysis/1";

/// FNV-1a, 64-bit. Specified, stable across Rust versions and platforms.
pub fn fnv64(bytes: &[u8]) -> u64 {
    let mut h: u64 = 0xcbf2_9ce4_8422_2325;
    for &b in bytes {
        h ^= b as u64;
        h = h.wrapping_mul(0x0000_0100_0000_01b3);
    }
    h
}

/// Which track, as the hash of its path's bytes.
pub fn source_of(track: &Path) -> u64 {
    fnv64(track.as_os_str().as_encoded_bytes())
}

fn bits(x: f32) -> String {
    format!("{:08x}", x.to_bits())
}

fn scale(s: FreqScale) -> String {
    match s {
        FreqScale::Log => "log".into(),
        FreqScale::Erb => "erb".into(),
        FreqScale::Blend(t) => format!("blend:{}", bits(t)),
        FreqScale::Lens { tilt, hz, oct, gain } => {
            format!("lens:{}:{}:{}:{}", bits(tilt), bits(hz), bits(oct), bits(gain))
        }
    }
}

/// The transform fields a verified alternate may differ in, and nothing else.
const TRANSFORM: [&str; 4] = ["q", "win", "n", "spread"];

impl AnalysisSettings {
    /// Every result-affecting setting, exactly. The name is a hash of this.
    pub fn canonical(&self) -> String {
        let map = match &self.mapping {
            BarMappingMode::FlatOverlap => "flat",
            BarMappingMode::Gaussian => "gauss",
            BarMappingMode::Cqt => "cqw",
            BarMappingMode::Superlet => "superlet",
            BarMappingMode::TrueCqt(c) if c.bass_limit_s.is_some() => "vqt",
            BarMappingMode::TrueCqt(_) => "true-cqt",
        };
        let mut out = format!(
            "{SCHEMA};src={:016x};bars={};lo={};hi={};map={map};dsd={}",
            source_of(&self.track),
            self.bars,
            bits(self.min_freq),
            bits(self.max_freq),
            self.dsd_rate.map_or("-".into(), |r| r.to_string()),
        );
        if let BarMappingMode::TrueCqt(c) = &self.mapping {
            let window = match c.window {
                super::cqt::Window::Hann => "hann",
                super::cqt::Window::Hamming => "hamming",
                super::cqt::Window::Blackman => "blackman",
                super::cqt::Window::BlackmanHarris => "blackmanharris",
            };
            // A separate method and revision, never the old FFT-based CQT
            // weighting. Its fixed mathematical policies are part of the
            // identity too: full symmetric windows, centered responses,
            // zero extension with the full-window gain, and independently
            // rounded sample centers (not a repeatedly added integer hop).
            // FFT and ASLT-only state cannot affect this transform.
            out += &format!(
                ";cqt-rev={};bpo={};window={window};scale=log;fps={};anchor=center;edge=zero-full;frame-clock=floor-k-sr-over-fps;grid-convention=log-cell-centres",
                super::cqt::REVISION,
                c.bins_per_octave,
                bits(c.frame_rate),
            );
            // The Responsive variable-Q transform is a different method: its
            // own revision and bass limit, appended only for it, so every
            // Full CQT record stays byte-identical.
            if let Some(t) = c.bass_limit_s {
                out += &format!(";vqt-rev={};bass-limit={}", super::cqt::VQT_REVISION, bits(t));
            }
            return out;
        }
        if let Some((fft, pad, ov, w, i, sc)) = &self.fft {
            let w = match w {
                WindowFn::Hann => "hann",
                WindowFn::Hamming => "hamming",
                WindowFn::Blackman => "blackman",
                WindowFn::FlatTop => "flattop",
            };
            let i = match i {
                InterpolationMode::None => "none",
                InterpolationMode::Linear => "linear",
                InterpolationMode::CatmullRom => "catmullrom",
                InterpolationMode::Pchip => "pchip",
                InterpolationMode::Akima => "akima",
                InterpolationMode::Lanczos => "lanczos",
            };
            out += &format!(
                ";fft={fft};pad={pad};ov={};window={w};interp={i};scale={}",
                bits(*ov),
                scale(*sc),
            );
        }
        if let Some((c, fps)) = &self.aslt {
            out += &format!(
                ";q={};win={};n={};spread={};scale={};fps={}",
                bits(c.q_ratio),
                bits(c.max_window_s),
                c.n_wavelets,
                bits(c.spread),
                scale(c.scale),
                bits(*fps),
            );
        }
        out
    }

    /// The hop these settings analyse at, at `sample_rate`.
    ///
    /// A true CQT has a variable integer spacing when `sr / fps` is not
    /// integral. This compatibility accessor is only its nominal floor;
    /// derive its rate from `Config::frame_rate`, never from this value.
    pub fn hop(&self, sample_rate: u32) -> usize {
        if let BarMappingMode::TrueCqt(c) = &self.mapping {
            return (sample_rate as f64 / c.frame_rate as f64).floor().max(1.0) as usize;
        }
        match (&self.fft, &self.aslt) {
            (_, Some((_, fps))) => super::aslt::hop_for_fps(sample_rate, *fps),
            (Some((fft, _, ov, ..)), _) => ((*fft as f32 * (1.0 - ov)).round() as usize).max(1),
            _ => 1,
        }
    }

    /// What a cache written at these settings, from audio at `sample_rate`,
    /// records about itself: the canonical settings and the frame geometry.
    pub fn record(&self, sample_rate: u32) -> String {
        if matches!(&self.mapping, BarMappingMode::TrueCqt(_)) {
            return format!("{};sr={sample_rate};hop=variable", self.canonical());
        }
        format!("{};sr={sample_rate};hop={}", self.canonical(), self.hop(sample_rate))
    }

    /// The sidecar identity, and the analysis *unit* in the cache store: the
    /// head file of this track's analysis at these settings,
    /// `<dir>/v2/<stem>.head` (see `store`). A unit names an analysis, not
    /// a file of rows: the head names whichever generation is committed, and
    /// an exact hit additionally needs the source snapshot to match.
    pub fn key(&self, dir: &Path) -> (super::stereo::Identity, PathBuf) {
        (self.identity(), unit_path(dir, &self.stem()))
    }

    /// The identity a channel payload is encoded with and checked against.
    pub fn identity(&self) -> super::stereo::Identity {
        super::stereo::Identity { source: source_of(&self.track), params: fnv64(self.canonical().as_bytes()) }
    }

    /// The store stem of this analysis: path and settings, never content.
    pub fn stem(&self) -> String {
        super::store::stem(&self.track, &self.canonical())
    }

    /// The name an exact-identity cache had before the store (`_e` files in
    /// the cache directory itself, written by 1.5.3 prereview builds). Under
    /// the owner's legacy policy (L1 + L1a) such a file is only ever offered
    /// as a labelled legacy alternative: its source identity is the path
    /// alone, so it is never an exact hit.
    pub fn legacy_record_key(&self, dir: &Path) -> PathBuf {
        let source = source_of(&self.track);
        let stamp = fnv64(self.canonical().as_bytes());
        dir.join(format!("{source:016x}_e{stamp:016x}.spectrumcache"))
    }
}

/// The store namespace inside a cache directory (decision D4).
pub fn store_dir(cache_dir: &Path) -> PathBuf {
    cache_dir.join("v2")
}

/// The unit path of a stem: its head file.
pub fn unit_path(cache_dir: &Path, stem: &str) -> PathBuf {
    store_dir(cache_dir).join(format!("{stem}.head"))
}

/// The stem a unit path names, if it is one.
pub fn stem_of_unit(unit: &Path) -> Option<String> {
    unit.file_name()?.to_str()?.strip_suffix(".head").map(str::to_string)
}

/// The name prefix every verified cache of `track` shares.
pub fn prefix_of(track: &Path) -> String {
    format!("{:016x}_e", source_of(track))
}

/// A record with the four transform fields removed, and those fields.
///
/// Two records whose remainders are equal describe the same track on the same
/// grid, scale, range, frame timing and sample rate — every field that decides
/// where a row or a bar sits — and differ at most in the transform's own
/// settings.
pub fn split_transform(record: &str) -> (String, Vec<(&str, &str)>) {
    let mut rest = Vec::new();
    let mut transform = Vec::new();
    for f in record.split(';') {
        match f.split_once('=') {
            Some((k, v)) if TRANSFORM.contains(&k) => transform.push((k, v)),
            _ => rest.push(f),
        }
    }
    (rest.join(";"), transform)
}

/// The exact transform a verified record states, given the scale it shares
/// with the display. `None` unless all four fields are present, parse, and are
/// settings this build could have written.
pub fn transform_of(fields: &[(&str, &str)], scale: FreqScale) -> Option<AsltConfig> {
    let get = |k: &str| fields.iter().find(|(f, _)| *f == k).map(|(_, v)| *v);
    let f = |k: &str| {
        u32::from_str_radix(get(k)?, 16).ok().map(f32::from_bits).filter(|x| x.is_finite())
    };
    let cfg = AsltConfig {
        q_ratio: f("q")?,
        max_window_s: f("win")?,
        n_wavelets: get("n")?.parse().ok()?,
        spread: f("spread")?,
        scale,
    };
    use super::aslt::{MAX_WINDOW_RANGE, N_WAVELETS_RANGE, Q_RATIO_RANGE, SPREAD_RANGE};
    (fields.len() == TRANSFORM.len()
        && Q_RATIO_RANGE.contains(&cfg.q_ratio)
        && MAX_WINDOW_RANGE.contains(&cfg.max_window_s)
        && N_WAVELETS_RANGE.contains(&cfg.n_wavelets)
        && SPREAD_RANGE.contains(&cfg.spread))
    .then_some(cfg)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn settings(mapping: BarMappingMode) -> AnalysisSettings {
        let (fft, aslt) = match &mapping {
            BarMappingMode::TrueCqt(_) => (None, None),
            BarMappingMode::Superlet => (None, Some((AsltConfig {
                q_ratio: 2.0,
                max_window_s: 0.25,
                n_wavelets: 9,
                spread: 0.5,
                scale: FreqScale::Blend(0.5),
            }, 180.0))),
            _ => (Some((8192, 2, 0.5, WindowFn::Hann, InterpolationMode::None, FreqScale::Log)), None),
        };
        AnalysisSettings {
            track: PathBuf::from("C:/music/x.flac"),
            bars: 1024,
            min_freq: 20.0,
            max_freq: 24_000.0,
            mapping,
            dsd_rate: None,
            fft,
            aslt,
        }
    }

    #[test]
    fn fnv_is_the_published_function() {
        // Reference values of FNV-1a-64.
        assert_eq!(fnv64(b""), 0xcbf2_9ce4_8422_2325);
        assert_eq!(fnv64(b"a"), 0xaf63_dc4c_8601_ec8c);
        assert_eq!(fnv64(b"foobar"), 0x8594_4171_f739_67e8);
    }

    #[test]
    fn a_record_splits_into_geometry_and_transform() {
        let r = "moosik-analysis/1;src=00;bars=8;q=40000000;win=3e800000;n=9;spread=3f000000;scale=log;fps=43340000;sr=44100;hop=245";
        let (rest, t) = split_transform(r);
        assert_eq!(rest, "moosik-analysis/1;src=00;bars=8;scale=log;fps=43340000;sr=44100;hop=245");
        let c = transform_of(&t, FreqScale::Log).unwrap();
        assert_eq!((c.q_ratio, c.max_window_s, c.n_wavelets, c.spread), (2.0, 0.25, 9, 0.5));
        // Missing, duplicated, non-finite or out-of-range fields are refused.
        assert!(transform_of(&t[..3], FreqScale::Log).is_none());
        let mut dup = t.clone();
        dup.push(("n", "9"));
        assert!(transform_of(&dup, FreqScale::Log).is_none());
        let mut nan = t.clone();
        nan[0] = ("q", "7fc00000");
        assert!(transform_of(&nan, FreqScale::Log).is_none());
        let mut big = t.clone();
        big[2] = ("n", "99");
        assert!(transform_of(&big, FreqScale::Log).is_none());
    }

    /// Adding one transform must not rename any existing analysis or turn
    /// the shipped FFT constant-Q weighting into a true-CQT cache.
    #[test]
    fn adding_true_cqt_leaves_all_old_records_byte_identical() {
        for (mapping, name) in [
            (BarMappingMode::FlatOverlap, "flat"),
            (BarMappingMode::Gaussian, "gauss"),
            (BarMappingMode::Cqt, "cqw"),
        ] {
            let s = settings(mapping);
            assert_eq!(s.record(48_000), format!(
                "moosik-analysis/1;src={:016x};bars=1024;lo=41a00000;hi=46bb8000;map={name};dsd=-;fft=8192;pad=2;ov=3f000000;window=hann;interp=none;scale=log;sr=48000;hop=4096",
                source_of(&s.track),
            ));
        }
        let s = settings(BarMappingMode::Superlet);
        assert_eq!(s.record(44_100), format!(
            "moosik-analysis/1;src={:016x};bars=1024;lo=41a00000;hi=46bb8000;map=superlet;dsd=-;q=40000000;win=3e800000;n=9;spread=3f000000;scale=blend:3f000000;fps=43340000;sr=44100;hop=245",
            source_of(&s.track),
        ));
    }

    #[test]
    fn true_cqt_has_its_own_pinned_record_and_variable_sample_clock() {
        let s = settings(BarMappingMode::TrueCqt(super::super::cqt::Config::default()));
        assert!(s.fft.is_none() && s.aslt.is_none());
        let expected = format!(
            "moosik-analysis/1;src={:016x};bars=1024;lo=41a00000;hi=46bb8000;map=true-cqt;dsd=-;cqt-rev=1;bpo=48;window=blackman;scale=log;fps=43340000;anchor=center;edge=zero-full;frame-clock=floor-k-sr-over-fps;grid-convention=log-cell-centres;sr=48000;hop=variable",
            source_of(&s.track),
        );
        assert_eq!(s.record(48_000), expected);
        // 48 kHz / 180 Hz is 266 2/3 samples, not a fixed hop of 266 or
        // 267. The record must never state either approximation as its clock.
        assert_eq!(s.hop(48_000), 266, "compatibility accessor is only the nominal floor");
        assert!(!s.record(48_000).contains(";hop=266"));
        assert!(!s.record(48_000).contains(";hop=267"));
        // The old ASLT-only alternate machinery must not strip out this
        // transform's own settings and offer it as a wavelet alternate.
        let (rest, transform) = split_transform(&expected);
        assert_eq!(rest, expected);
        assert!(transform.is_empty());
        assert!(transform_of(&transform, FreqScale::Log).is_none());
    }

    /// The Responsive variable-Q transform is its own method: `map=vqt` with
    /// its revision and bass limit. Full CQT's pinned record (above) has
    /// neither, so no Full CQT cache changes or is reused by it.
    #[test]
    fn responsive_cqt_has_its_own_method_and_limit_in_the_record() {
        use super::super::cqt::Config;
        let full = settings(BarMappingMode::TrueCqt(Config::default()));
        let r = |t: f32| settings(BarMappingMode::TrueCqt(Config { bass_limit_s: Some(t), ..Config::default() }));
        let expected = format!(
            "moosik-analysis/1;src={:016x};bars=1024;lo=41a00000;hi=46bb8000;map=vqt;dsd=-;cqt-rev=1;bpo=48;window=blackman;scale=log;fps=43340000;anchor=center;edge=zero-full;frame-clock=floor-k-sr-over-fps;grid-convention=log-cell-centres;vqt-rev=1;bass-limit=3f000000;sr=48000;hop=variable",
            source_of(&full.track),
        );
        assert_eq!(r(0.5).record(48_000), expected);
        let (id, unit) = full.key(Path::new("D"));
        for t in [0.25f32, 0.5, 1.0] {
            assert_ne!(r(t).identity(), id);
            assert_ne!(r(t).key(Path::new("D")).1, unit);
        }
        assert_ne!(r(0.25).identity(), r(0.5).identity());
        let (rest, transform) = split_transform(&expected);
        assert_eq!(rest, expected);
        assert!(transform.is_empty());
    }

    #[test]
    fn every_true_cqt_setting_separates_the_identity_from_other_methods() {
        use super::super::cqt::{Config, Window};
        let config = Config::default();
        let baseline = settings(BarMappingMode::TrueCqt(config));
        let (id, unit) = baseline.key(Path::new("D"));
        for c in [
            Config { bins_per_octave: 24, ..config },
            Config { window: Window::Hann, ..config },
            Config { window: Window::Hamming, ..config },
            Config { window: Window::BlackmanHarris, ..config },
            Config { frame_rate: f32::from_bits(config.frame_rate.to_bits() + 1), ..config },
        ] {
            let other = settings(BarMappingMode::TrueCqt(c));
            assert_ne!(other.identity(), id, "configuration shared an identity: {c:?}");
            assert_ne!(other.key(Path::new("D")).1, unit, "configuration shared a store unit: {c:?}");
        }
        for mapping in [BarMappingMode::Cqt, BarMappingMode::Superlet, BarMappingMode::FlatOverlap, BarMappingMode::Gaussian] {
            assert_ne!(settings(mapping).identity(), id, "another method shared a true-CQT identity");
        }
        assert_ne!(baseline.record(44_100), baseline.record(48_000), "sample rate was absent from the record");
        assert_eq!(baseline.identity(), baseline.clone().identity());
        assert_eq!(baseline.key(Path::new("D")).1, unit);

        // Display-grid and source settings remain part of the identity.
        let mut other = baseline.clone();
        other.bars += 1;
        assert_ne!(other.identity(), id);
        other = baseline.clone();
        other.min_freq = f32::from_bits(other.min_freq.to_bits() + 1);
        assert_ne!(other.identity(), id);
        other = baseline.clone();
        other.max_freq = f32::from_bits(other.max_freq.to_bits() + 1);
        assert_ne!(other.identity(), id);
        other = baseline.clone();
        other.dsd_rate = Some(88_200);
        assert_ne!(other.identity(), id);
        other = baseline;
        other.track = PathBuf::from("C:/music/other.flac");
        assert_ne!(other.identity(), id);
    }
}
