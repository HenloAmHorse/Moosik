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
//! the transform's code — must bump [`SCHEMA`], which retires every record
//! written before it. The compute route (CPU or GPU) is not part of the
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
        let map = match self.mapping {
            BarMappingMode::FlatOverlap => "flat",
            BarMappingMode::Gaussian => "gauss",
            BarMappingMode::Cqt => "cqw",
            BarMappingMode::Superlet => "superlet",
        };
        let mut out = format!(
            "{SCHEMA};src={:016x};bars={};lo={};hi={};map={map};dsd={}",
            source_of(&self.track),
            self.bars,
            bits(self.min_freq),
            bits(self.max_freq),
            self.dsd_rate.map_or("-".into(), |r| r.to_string()),
        );
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
    pub fn hop(&self, sample_rate: u32) -> usize {
        match (&self.fft, &self.aslt) {
            (_, Some((_, fps))) => super::aslt::hop_for_fps(sample_rate, *fps),
            (Some((fft, _, ov, ..)), _) => ((*fft as f32 * (1.0 - ov)).round() as usize).max(1),
            _ => 1,
        }
    }

    /// What a cache written at these settings, from audio at `sample_rate`,
    /// records about itself: the canonical settings and the frame geometry.
    pub fn record(&self, sample_rate: u32) -> String {
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
}
