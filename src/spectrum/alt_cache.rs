//! Other analyses of the track on screen, already on disk.
//!
//! # What counts as an alternate here
//!
//! Only a **Superlet** cache of the **same track** that shares every part of
//! the display's geometry with the user's current settings: bar count, the
//! frequency range, the frequency scale, the frame rate (so the hop, at the
//! same sample rate) and, for DSD, the decimation rate. What may differ is the
//! transform's own settings — sharpness, window cap, wavelet count, spread.
//! Those change the numbers in the rows, never where a row or a bar sits, so
//! such a cache can be installed exactly as an exact hit is.
//!
//! Anything else — an FFT cache, another bar count, frame rate, scale or range
//! — would need the display's axes and clock to follow the cache rather than
//! the settings. That is deliberately not offered: a button that installs rows
//! on the wrong grid is worse than no button.
//!
//! # Two kinds of candidate
//!
//! **Verified.** A cache written with exact identity carries a settings record
//! (see `identity`). It is a candidate only if its record agrees with the
//! record the display's settings would produce on every field but the four
//! transform settings — geometry, scale, frame timing and sample rate compared
//! at full precision, never through the readable rounding. Its settings are
//! then known exactly.
//!
//! **Legacy.** A cache written before records existed states its settings only
//! through its rounded file name, and nothing on disk says what it was really
//! analysed at. It is a candidate if re-keying the settings read back from its
//! name (the legacy key, for the track on screen) reproduces its name exactly —
//! which rules out another track's cache, short of a 64-bit hash collision, and
//! any name this parser misreads. Its settings are *nominal*: to the name's
//! rounding, unverified. A legacy cache is never loaded automatically, not even
//! one whose rounded name is today's settings; it is only offered, labelled.
//!
//! The track half of either key is a hash of the *path*, so a file replaced in
//! place by different audio still matches — the same limitation the exact
//! cache has.
//!
//! # Cost
//!
//! Discovery is one `read_dir`, a `stat` per sidecar, and for this track's
//! verified caches a read of the header and record (at most 64 KiB, never the
//! rows). It runs when the track or the settings change — never per frame, and
//! never a decode. Whether a candidate is actually readable is decided when it
//! is chosen, by a full read.
use std::path::{Path, PathBuf};

use super::aslt::AsltConfig;

/// One other analysis of this track, as discovered. Nothing here has been read.
#[derive(Clone, Debug, PartialEq)]
pub struct Candidate {
    /// A store analysis: its unit (head) path. A legacy analysis: its file.
    pub path: PathBuf,
    /// Its transform settings: exact from its record, or — for a legacy cache
    /// written before records — nominal, as its name states them to hundredths.
    pub cfg: AsltConfig,
    /// Written by an older version, outside the cache store (owner decision
    /// L1 + L1a): its source is identified by the path alone, never checked
    /// against the audio, so it is only ever offered, labelled, and never an
    /// exact hit; and it is never imported, rewritten or deleted.
    pub legacy: bool,
    /// Its settings are nominal (a rounded file name), not an exact record.
    pub nominal: bool,
    /// Size of the mono cache on disk when discovered.
    pub mono_bytes: u64,
    /// Size of its channel sidecar, if one was beside it.
    pub sidecar_bytes: Option<u64>,
}

/// The fixed parts every same-geometry alternate shares with `exact`:
/// everything before the four transform settings, and everything after.
///
/// `exact` is `…_s{q}-{window}-{n}-{spread}-{fps}{scale}.spectrumcache`; the
/// scale suffix can itself contain `-`, which is why the split stops at five.
fn frame(exact: &Path) -> Option<(String, String)> {
    let name = exact.file_name()?.to_str()?;
    let at = name.find("_s")? + 2;
    let mut parts = name[at..].splitn(5, '-');
    for _ in 0..4 {
        parts.next()?.parse::<u32>().ok()?;
    }
    let tail = parts.next()?;
    Some((name[..at].to_string(), format!("-{tail}")))
}

/// The transform settings a candidate's name states, if it is one.
fn settings(name: &str, prefix: &str, suffix: &str, scale: super::freq_scale::FreqScale)
    -> Option<AsltConfig>
{
    let mid = name.strip_prefix(prefix)?.strip_suffix(suffix)?;
    let v: Vec<u32> = mid.split('-').map(|s| s.parse().ok()).collect::<Option<_>>()?;
    let [q, w, n, s] = v[..] else { return None };
    let cfg = AsltConfig {
        q_ratio: q as f32 / 100.0,
        max_window_s: w as f32 / 100.0,
        n_wavelets: n as usize,
        spread: s as f32 / 100.0,
        scale,
    };
    // Bounded before anything is computed from it: the cost model is linear in
    // the wavelet count and runs on the UI thread.
    use super::aslt::{MAX_WINDOW_RANGE, N_WAVELETS_RANGE, Q_RATIO_RANGE, SPREAD_RANGE};
    (Q_RATIO_RANGE.contains(&cfg.q_ratio)
        && MAX_WINDOW_RANGE.contains(&cfg.max_window_s)
        && N_WAVELETS_RANGE.contains(&cfg.n_wavelets)
        && SPREAD_RANGE.contains(&cfg.spread))
    .then_some(cfg)
}

/// Every legacy cache of this track whose name states the display's geometry,
/// **including** one whose rounded name is the user's own settings: no legacy
/// cache is an exact hit any more, so that one is a candidate like the rest.
///
/// `legacy` is the legacy key at the user's settings, used as the template for
/// the name; `rekey` is the legacy key for the track on screen at the given
/// transform settings. A candidate whose name it does not reproduce is not this
/// track's, or not what its name appears to say.
pub fn discover_legacy(
    dir: &Path,
    legacy: &Path,
    scale: super::freq_scale::FreqScale,
    rekey: &dyn Fn(&AsltConfig) -> PathBuf,
) -> Vec<Candidate> {
    let Some((prefix, suffix)) = frame(legacy) else { return Vec::new() };
    let Ok(entries) = std::fs::read_dir(dir) else { return Vec::new() };
    let mut out: Vec<Candidate> = entries
        .filter_map(|e| e.ok())
        .filter_map(|e| {
            let name = e.file_name().into_string().ok()?;
            let cfg = settings(&name, &prefix, &suffix, scale)?;
            let path = e.path();
            if rekey(&cfg) != path { return None; }
            candidate(e, path, cfg, true, true)
        })
        .collect();
    out.sort_by(|a, b| a.path.cmp(&b.path));
    out
}

/// Every legacy cache of this track that carries an exact settings record (a
/// 1.5.3 prereview `_e` file in the cache folder itself), whose record agrees
/// with `mine` — the record the display's settings produce at this track's
/// rate — on everything but the transform, and whose name is the legacy
/// record key (`rekey`) for the transform its record states. **Including** the
/// one at the user's own settings: under L1a it is not an exact hit any more,
/// because its source is identified by the path alone.
pub fn discover_verified(
    dir: &Path,
    track: &Path,
    mine: &str,
    scale: super::freq_scale::FreqScale,
    rekey: &dyn Fn(&AsltConfig) -> PathBuf,
) -> Vec<Candidate> {
    let prefix = super::identity::prefix_of(track);
    let (want, _) = super::identity::split_transform(mine);
    let Ok(entries) = std::fs::read_dir(dir) else { return Vec::new() };
    let mut out: Vec<Candidate> = entries
        .filter_map(|e| e.ok())
        .filter_map(|e| {
            let name = e.file_name().into_string().ok()?;
            if !name.starts_with(&prefix) || !name.ends_with(".spectrumcache") { return None; }
            let path = e.path();
            let record = crate::spectrum::cache::read_record(&path)?;
            let (rest, fields) = super::identity::split_transform(&record);
            if rest != want { return None; }
            let cfg = super::identity::transform_of(&fields, scale)?;
            if rekey(&cfg) != path { return None; }
            candidate(e, path, cfg, true, false)
        })
        .collect();
    out.sort_by(|a, b| a.path.cmp(&b.path));
    out
}

/// Every analysis in the cache store of this track **as it is now** (its
/// stored descriptor names `source`, the snapshot taken for this discovery)
/// at other transform settings: same geometry, scale, frame timing and rate
/// as `mine`, and a stem that is the unit (`rekey`) for the transform its
/// record states. `exact_stem`, the user's own analysis, is not an alternate.
/// One directory listing and, for this track's heads, a read of each head and
/// of its Mix envelope header (never the rows).
#[allow(clippy::too_many_arguments)]
pub fn discover_store(
    store: &super::store::Store,
    cache_dir: &Path,
    track: &Path,
    source: &super::store::SourceSnapshot,
    mine: &str,
    exact_stem: Option<&str>,
    scale: super::freq_scale::FreqScale,
    rekey: &dyn Fn(&AsltConfig) -> PathBuf,
) -> Vec<Candidate> {
    let (want, _) = super::identity::split_transform(mine);
    store
        .heads_for(track)
        .into_iter()
        .filter(|f| Some(f.stem.as_str()) != exact_stem)
        .filter_map(|f| {
            let record = super::store::Descriptor::record_in(&f.text)?;
            let expected = super::store::Descriptor { record: record.to_string(), source: source.clone() };
            if expected.canonical() != f.text { return None; }
            let (rest, fields) = super::identity::split_transform(record);
            if rest != want { return None; }
            let cfg = super::identity::transform_of(&fields, scale)?;
            let unit = super::identity::unit_path(cache_dir, &f.stem);
            if rekey(&cfg) != unit { return None; }
            Some(Candidate { path: unit, cfg, legacy: false, nominal: false, mono_bytes: f.mix_bytes, sidecar_bytes: f.lr_bytes })
        })
        .collect()
}

fn candidate(e: std::fs::DirEntry, path: PathBuf, cfg: AsltConfig, legacy: bool, nominal: bool) -> Option<Candidate> {
    let mono_bytes = e.metadata().ok().filter(|m| m.is_file())?.len();
    let sidecar_bytes = std::fs::metadata(super::stereo::path_for(&path)).ok().map(|m| m.len());
    Some(Candidate { path, cfg, legacy, nominal, mono_bytes, sidecar_bytes })
}

/// How much a set of transform settings is known to.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Precision {
    /// The user's own settings, or a verified record: every value exactly.
    Exact,
    /// A legacy cache's file name: hundredths, and unverified.
    Nominal,
}

impl Candidate {
    /// How far this candidate's settings are known.
    pub fn precision(&self) -> Precision {
        if self.nominal { Precision::Nominal } else { Precision::Exact }
    }
}

/// A value as `p` knows it: in hundredths for a nominal legacy value, and
/// otherwise as the shortest text that reads back as the same `f32`, so two
/// exact settings that differ are never printed alike.
fn value(x: f32, p: Precision) -> String {
    match p {
        Precision::Exact => format!("{x}"),
        Precision::Nominal => format!("{x:.2}"),
    }
}

/// Whether two values differ, as far as `p` can tell: exactly, or — for a
/// nominal legacy value — in the hundredths its name kept.
fn differs(a: f32, b: f32, p: Precision) -> bool {
    match p {
        Precision::Exact => a != b,
        Precision::Nominal => (a * 100.0).round() as i64 != (b * 100.0).round() as i64,
    }
}

/// The preset a configuration is. Exact settings must equal the preset's
/// values exactly — a near-preset value is Custom. A nominal legacy name can
/// only say it is the preset to hundredths.
pub fn preset_of(cfg: &AsltConfig, p: Precision) -> Option<super::aslt::AsltPreset> {
    use super::aslt::AsltPreset as P;
    [P::Fast, P::Standard, P::High, P::Ultra, P::Extreme].into_iter().find(|preset| {
        let c = preset.config();
        c.n_wavelets == cfg.n_wavelets
            && !differs(c.q_ratio, cfg.q_ratio, p)
            && !differs(c.max_window_s, cfg.max_window_s, p)
            && !differs(c.spread, cfg.spread, p)
    })
}

/// The settings, stated in full: "High · wavelets 5 · max window 1.2 s · …",
/// to the precision they are known.
pub fn describe(cfg: &AsltConfig, p: Precision) -> String {
    format!(
        "{} · wavelets {} · max window {} s · sharpness {}× · spread {}",
        preset_of(cfg, p).map_or("Custom", |pr| pr.label()),
        cfg.n_wavelets,
        value(cfg.max_window_s, p),
        value(cfg.q_ratio, p),
        value(cfg.spread, p),
    )
}

/// Only the settings that differ, as "wavelets 9 vs 1; …" against `mine`.
/// `p` is the candidate's precision; the user's own values are exact.
pub fn differences(cand: &AsltConfig, mine: &AsltConfig, p: Precision) -> String {
    let e = Precision::Exact;
    let mut d = Vec::new();
    if cand.n_wavelets != mine.n_wavelets {
        d.push(format!("wavelets {} vs {}", cand.n_wavelets, mine.n_wavelets));
    }
    for (what, unit, a, b) in [
        ("max window", " s", cand.max_window_s, mine.max_window_s),
        ("sharpness", "×", cand.q_ratio, mine.q_ratio),
        ("spread", "", cand.spread, mine.spread),
    ] {
        if differs(a, b, p) {
            d.push(format!("{what} {}{unit} vs {}{unit}", value(a, p), value(b, e)));
        }
    }
    d.join("; ")
}

/// Bytes as the notice shows them.
pub fn fmt_bytes(b: u64) -> String {
    if b >= 1_000_000 { format!("{:.1} MB", b as f64 / 1e6) } else { format!("{:.0} kB", b as f64 / 1e3) }
}

#[cfg(test)]
mod tests {
    use super::*;
    use super::super::freq_scale::FreqScale;

    #[test]
    fn a_name_splits_around_the_transform_settings() {
        let (p, s) = frame(Path::new("/c/abc_b1024_f0_w0_p0_o0_n20_x24000_m3_i0_s100-60-4-50-180-w-35.spectrumcache")).unwrap();
        assert_eq!(p, "abc_b1024_f0_w0_p0_o0_n20_x24000_m3_i0_s");
        assert_eq!(s, "-180-w-35.spectrumcache");
        let c = settings("abc_b1024_f0_w0_p0_o0_n20_x24000_m3_i0_s100-120-9-50-180-w-35.spectrumcache",
                         &p, &s, FreqScale::Log).unwrap();
        assert_eq!((c.n_wavelets, c.max_window_s, c.q_ratio, c.spread), (9, 1.2, 1.0, 0.5));
        // Another frame rate, another scale, or an FFT name is not a candidate.
        for other in [
            "abc_b1024_f0_w0_p0_o0_n20_x24000_m3_i0_s100-120-9-50-60-w-35.spectrumcache",
            "abc_b1024_f0_w0_p0_o0_n20_x24000_m3_i0_s100-120-9-50-180.spectrumcache",
            "abc_b1024_f8192_w0_p16_o875_n20_x24000_m0_i0_g-log.spectrumcache",
            "abc_b512_f0_w0_p0_o0_n20_x24000_m3_i0_s100-120-9-50-180-w-35.spectrumcache",
            "abc_b1024_f0_w0_p0_o0_n20_x24000_m3_i0_s100-120-0-50-180-w-35.spectrumcache",
            // Out of what this build can write, or not a number it can hold.
            "abc_b1024_f0_w0_p0_o0_n20_x24000_m3_i0_s100-120-10-50-180-w-35.spectrumcache",
            "abc_b1024_f0_w0_p0_o0_n20_x24000_m3_i0_s100-120-4294967295-50-180-w-35.spectrumcache",
            "abc_b1024_f0_w0_p0_o0_n20_x24000_m3_i0_s100-120-99999999999-50-180-w-35.spectrumcache",
            "abc_b1024_f0_w0_p0_o0_n20_x24000_m3_i0_s201-120-9-50-180-w-35.spectrumcache",
            "abc_b1024_f0_w0_p0_o0_n20_x24000_m3_i0_s100-801-9-50-180-w-35.spectrumcache",
            "abc_b1024_f0_w0_p0_o0_n20_x24000_m3_i0_s100-9-9-50-180-w-35.spectrumcache",
            "abc_b1024_f0_w0_p0_o0_n20_x24000_m3_i0_s100-120-9-96-180-w-35.spectrumcache",
        ] {
            assert!(settings(other, &p, &s, FreqScale::Log).is_none(), "{other}");
        }
        // An FFT name has no transform part to frame.
        assert!(frame(Path::new("abc_b1024_f8192_w0_p16_o875_n20_x24000_m0_i0_g-log.spectrumcache")).is_none());
    }

    #[test]
    fn presets_are_recognised_and_differences_named() {
        let e = Precision::Exact;
        let high = super::super::aslt::AsltPreset::High.config();
        assert_eq!(preset_of(&high, e), Some(super::super::aslt::AsltPreset::High));
        let mut n1 = high.clone();
        n1.n_wavelets = 1;
        assert_eq!(preset_of(&n1, e), None);
        assert_eq!(differences(&high, &n1, e), "wavelets 5 vs 1");
        assert_eq!(differences(&high, &high, e), "");
    }

    /// Exact settings that the old hundredths made look identical are told
    /// apart in text and in the differences, and a value next to a preset's
    /// is not that preset. A legacy name keeps its nominal hundredths.
    #[test]
    fn exact_settings_are_described_to_their_own_precision() {
        use super::super::aslt::AsltPreset;
        let (e, n) = (Precision::Exact, Precision::Nominal);
        let base = AsltPreset::Standard.config();
        let with = |f: &dyn Fn(&mut AsltConfig)| { let mut c = base.clone(); f(&mut c); c };
        for (what, a, b, sa, sb) in [
            ("sharpness", with(&|c| c.q_ratio = 1.231), with(&|c| c.q_ratio = 1.234), "sharpness 1.231×", "sharpness 1.234×"),
            ("max window", with(&|c| c.max_window_s = 0.601), with(&|c| c.max_window_s = 0.604), "max window 0.601 s", "max window 0.604 s"),
            ("spread", with(&|c| c.spread = 0.501), with(&|c| c.spread = 0.504), "spread 0.501", "spread 0.504"),
        ] {
            let (da, db) = (describe(&a, e), describe(&b, e));
            assert_ne!(da, db, "{what}: described alike");
            assert!(da.contains(sa) && db.contains(sb), "{what}: {da} / {db}");
            let diff = differences(&a, &b, e);
            assert!(diff.starts_with(what) && diff.contains(sa.trim_start_matches(what).trim_start()), "{what}: {diff}");
            // A legacy name cannot tell them apart, and does not pretend to.
            assert_eq!(differences(&a, &b, n), "", "{what}: nominal hundredths claimed a difference");
            assert_eq!(describe(&a, n), describe(&b, n));
        }
        // Exactly a preset, and one step off it.
        assert_eq!(preset_of(&base, e), Some(AsltPreset::Standard));
        assert!(describe(&base, e).starts_with("Standard · "));
        let near = with(&|c| c.q_ratio = base.q_ratio + 0.001);
        assert_eq!(preset_of(&near, e), None, "a near-preset value was called the preset");
        assert!(describe(&near, e).starts_with("Custom · "));
        assert_eq!(preset_of(&near, n), Some(AsltPreset::Standard), "nominal: to hundredths only");
    }
}
