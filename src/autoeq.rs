//! Headphone EQ from AutoEq (1.6.2).
//!
//! The list of headphones and target curves is kept in `~/.moosik/autoeq` and
//! checked against autoeq.app in the background when Moosik starts (Settings
//! can turn that off), so searching it needs no network. Equalizing does:
//! autoeq.app, the AutoEq project's own web app (its server is open source,
//! `webapp/main.py` in jaakkopasanen/AutoEq, MIT), computes it from the
//! measurement, the target and the settings, and those are all that is sent.
//! It is not a published API, so every answer is checked before it is used.
//! What is made is kept beside the files written for other apps, so a
//! headphone done once also works without the network.

use std::collections::BTreeMap;
use std::path::{Path, PathBuf};
use std::time::Duration;

use serde::Deserialize;

use crate::spectrum::eq::{shelf_slope, BandKind, EqBand, MAX_BANDS};

pub const SITE: &str = "https://autoeq.app";

/// The rate equalizers are made for. Moosik computes its filters again at
/// each song's own rate, and the texts for other apps carry frequencies,
/// not coefficients, so no one has to choose it.
const FS: u32 = 48_000;

const ENTRIES: &str = "entries.json";
const TARGETS: &str = "targets.json";
const CHECKED: &str = "checked";

/// One measurement of a headphone: who measured it, on what, and the kind.
#[derive(Clone, Debug, PartialEq, Deserialize)]
pub struct Measurement {
    pub source: String,
    #[serde(default)]
    pub rig: Option<String>,
    pub form: String,
}

impl Measurement {
    pub fn label(&self) -> String {
        match &self.rig {
            Some(r) => format!("{} on {} ({})", self.source, r, self.form),
            None => format!("{} ({})", self.source, self.form),
        }
    }
}

/// The measurements a target suits.
#[derive(Clone, Debug, Deserialize)]
pub struct Fit {
    pub source: String,
    pub form: String,
    #[serde(default)]
    pub rig: Option<String>,
}

impl Fit {
    pub fn fits(&self, m: &Measurement) -> bool {
        self.source == m.source && self.form == m.form && (self.rig.is_none() || self.rig == m.rig)
    }
}

/// A shelf the target adds: autoeq.app's bass boost.
#[derive(Clone, Copy, Debug, PartialEq, Deserialize)]
pub struct Shelf {
    pub fc: f32,
    pub q: f32,
    pub gain: f32,
}

/// A target curve, the measurements it suits, and the bass boost
/// autoeq.app starts it with.
#[derive(Clone, Debug, Deserialize)]
pub struct Target {
    pub label: String,
    #[serde(default)]
    pub compatible: Vec<Fit>,
    #[serde(default)]
    pub recommended: Vec<Fit>,
    #[serde(rename = "bassBoost", default)]
    pub bass_boost: Option<Shelf>,
}

/// The headphones, by name, with their measurements, and the targets.
pub struct Catalog {
    pub names: Vec<String>,
    pub measurements: Vec<Vec<Measurement>>,
    pub targets: Vec<Target>,
    keys: Vec<Key>,
}

/// A name as searched: its letters and digits with the spaces taken out, and
/// where each of its words starts.
struct Key {
    chars: Vec<char>,
    starts: Vec<usize>,
}

impl Key {
    fn new(name: &str) -> Key {
        let (mut chars, mut starts, mut gap) = (Vec::new(), Vec::new(), true);
        for c in name.chars().flat_map(char::to_lowercase) {
            if c.is_alphanumeric() {
                if gap {
                    starts.push(chars.len());
                }
                chars.push(c);
                gap = false;
            } else {
                gap = true;
            }
        }
        Key { chars, starts }
    }
}

impl Catalog {
    pub fn parse(entries: &str, targets: &str) -> Result<Catalog, String> {
        let entries: BTreeMap<String, Vec<Measurement>> =
            serde_json::from_str(entries).map_err(|e| format!("the headphone list did not read: {e}"))?;
        let targets: Vec<Target> = serde_json::from_str(targets).map_err(|e| format!("the target list did not read: {e}"))?;
        let (names, measurements): (Vec<String>, Vec<Vec<Measurement>>) = entries.into_iter().filter(|(_, m)| !m.is_empty()).unzip();
        if names.is_empty() || targets.is_empty() {
            return Err("autoeq.app's lists were empty".into());
        }
        let keys = names.iter().map(|n| Key::new(n)).collect();
        Ok(Catalog { names, measurements, targets, keys })
    }

    /// The targets for a measurement: the recommended ones first, then the
    /// others it suits; every target when none says it suits it.
    pub fn targets_for(&self, m: &Measurement) -> Vec<usize> {
        let all = 0..self.targets.len();
        let mut out: Vec<usize> = all.clone().filter(|&i| self.targets[i].recommended.iter().any(|f| f.fits(m))).collect();
        let more: Vec<usize> = all.clone().filter(|&i| !out.contains(&i) && self.targets[i].compatible.iter().any(|f| f.fits(m))).collect();
        out.extend(more);
        if out.is_empty() {
            out = all.collect();
        }
        out
    }

    /// Headphones matching `query`, best first; see [`search`].
    pub fn search(&self, query: &str, limit: usize) -> Vec<usize> {
        search(&self.keys, query, limit)
    }
}

/// The fewest edits that turn `word` into some stretch of `text` (Sellers'
/// algorithm: the match may start and end anywhere in the text).
fn errors_within(word: &[char], text: &[char]) -> usize {
    let mut prev = vec![0usize; text.len() + 1];
    let mut cur = vec![0usize; text.len() + 1];
    for (i, &w) in word.iter().enumerate() {
        cur[0] = i + 1;
        for (j, &t) in text.iter().enumerate() {
            cur[j + 1] = (prev[j] + usize::from(w != t)).min(prev[j + 1] + 1).min(cur[j] + 1);
        }
        std::mem::swap(&mut prev, &mut cur);
    }
    prev.into_iter().min().unwrap_or(word.len())
}

/// Names matching `query`, best first, at most `limit`. Case, spaces and
/// punctuation do not count ("hd650" finds "HD 650"), nor does the order of
/// the words. Each word of the query must turn up in the name, with one
/// letter wrong, missing or extra allowed from four letters and two from
/// eight. Fewer mistakes rank first, then a match at the start of a word of
/// the name, then the shorter name.
fn search(keys: &[Key], query: &str, limit: usize) -> Vec<usize> {
    let words: Vec<Vec<char>> = query
        .split(|c: char| !c.is_alphanumeric())
        .filter(|w| !w.is_empty())
        .map(|w| w.chars().flat_map(char::to_lowercase).collect())
        .collect();
    if words.is_empty() {
        return Vec::new();
    }
    let mut hits: Vec<(usize, usize, usize)> = Vec::new();
    'names: for (i, key) in keys.iter().enumerate() {
        let mut score = 0;
        for w in &words {
            let allowed = match w.len() {
                0..=3 => 0,
                4..=7 => 1,
                _ => 2,
            };
            let errors = errors_within(w, &key.chars);
            if errors > allowed {
                continue 'names;
            }
            let at_start = key.starts.iter().any(|&s| key.chars[s..].starts_with(w));
            score += errors * 10 + if at_start { 0 } else { 3 };
        }
        hits.push((score, key.chars.len(), i));
    }
    hits.sort_unstable();
    hits.into_iter().take(limit).map(|h| h.2).collect()
}

/// Where the list, the made equalizers and the files for other apps live.
pub fn dir() -> PathBuf {
    crate::moosik_dir().join("autoeq")
}

/// The list as kept, when both files are there and read.
pub fn load(dir: &Path) -> Option<Catalog> {
    let e = std::fs::read_to_string(dir.join(ENTRIES)).ok()?;
    let t = std::fs::read_to_string(dir.join(TARGETS)).ok()?;
    Catalog::parse(&e, &t).ok()
}

fn write_atomic(path: &Path, text: &str) -> Result<(), String> {
    let tmp = path.with_extension("part");
    std::fs::write(&tmp, text).and_then(|_| std::fs::rename(&tmp, path)).map_err(|e| format!("{}: {e}", path.display()))
}

/// Fetch the list from autoeq.app and keep it if it reads. `Ok(true)` when it
/// changed. A list that does not read is never kept.
pub fn refresh(dir: &Path) -> Result<bool, String> {
    let e = get(&format!("{SITE}/entries"))?;
    let t = get(&format!("{SITE}/targets"))?;
    Catalog::parse(&e, &t)?;
    std::fs::create_dir_all(dir).map_err(|e| format!("{}: {e}", dir.display()))?;
    let kept = |name: &str| std::fs::read_to_string(dir.join(name)).ok();
    let changed = kept(ENTRIES).as_deref() != Some(e.as_str()) || kept(TARGETS).as_deref() != Some(t.as_str());
    if changed {
        write_atomic(&dir.join(ENTRIES), &e)?;
        write_atomic(&dir.join(TARGETS), &t)?;
    }
    let _ = std::fs::write(dir.join(CHECKED), b"");
    Ok(changed)
}

/// The start-up check: the list again from autoeq.app, on its own thread.
/// Skipped within an hour of the last check, so restarts do not repeat it.
pub fn refresh_in_background() {
    let dir = dir();
    let recent = std::fs::metadata(dir.join(CHECKED))
        .and_then(|m| m.modified())
        .ok()
        .and_then(|t| t.elapsed().ok())
        .is_some_and(|age| age < Duration::from_secs(3600));
    if recent {
        return;
    }
    let _ = std::thread::Builder::new().name("autoeq-list".into()).spawn(move || match refresh(&dir) {
        Ok(changed) => crate::mlog!("[autoeq] list checked{}", if changed { ", updated" } else { ", unchanged" }),
        Err(e) => crate::mlog!("[autoeq] list not checked: {e}"),
    });
}

/// What is asked of autoeq.app, beyond the headphone and its measurement:
/// the target and the website's adjustments.
#[derive(Clone, Debug, PartialEq)]
pub struct Settings {
    pub target: String,
    pub bass: Shelf,
    pub treble: f32,
    pub tilt: f32,
    pub max_gain: f32,
}

impl Settings {
    /// autoeq.app's own starting point for a target.
    pub fn for_target(t: &Target) -> Settings {
        Settings {
            target: t.label.clone(),
            bass: t.bass_boost.unwrap_or(Shelf { fc: 105.0, q: 0.7, gain: 0.0 }),
            treble: 0.0,
            tilt: 0.0,
            max_gain: 6.0,
        }
    }
}

/// The kinds of equalizer made, each its own request.
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum Kind {
    /// Two shelves and eight peaks: Moosik, Equalizer APO, Peace.
    Parametric,
    /// USB Audio Player PRO's layout, as autoeq.app makes it for that app.
    Uapp,
    /// Ten peaks, for the ToneBoosters preset UAPP can load.
    TenPeaks,
    /// Ten and 31 bands at fixed frequencies, for equalizers with sliders.
    Fixed10,
    Fixed31,
    /// Equalizer APO's GraphicEQ line: Wavelet, JamesDSP, Equalizer APO.
    Graphic,
}

fn request(name: &str, m: &Measurement, s: &Settings, kind: Kind) -> serde_json::Value {
    let mut r = serde_json::json!({
        "name": name, "source": m.source, "rig": m.rig, "target": s.target,
        "bass_boost_gain": s.bass.gain, "bass_boost_fc": s.bass.fc, "bass_boost_q": s.bass.q,
        "treble_boost_gain": s.treble, "tilt": s.tilt, "max_gain": s.max_gain,
        "fs": FS, "response": { "fr_fields": [] },
    });
    let o = r.as_object_mut().expect("an object");
    let peq = |config: &str| [("parametric_eq", serde_json::json!(true)), ("parametric_eq_config", serde_json::json!(config))];
    let fixed = |config: &str| [("fixed_band_eq", serde_json::json!(true)), ("fixed_band_eq_config", serde_json::json!(config))];
    let add: Vec<(&str, serde_json::Value)> = match kind {
        Kind::Parametric => peq("8_PEAKING_WITH_SHELVES").into(),
        Kind::Uapp => peq("USB_AUDIO_PLAYER_PRO").into(),
        Kind::TenPeaks => peq("10_PEAKING").into(),
        Kind::Fixed10 => fixed("10_BAND_GRAPHIC_EQ").into(),
        Kind::Fixed31 => fixed("31_BAND_GRAPHIC_EQ").into(),
        Kind::Graphic => vec![("graphic_eq", serde_json::json!(true))],
    };
    for (k, v) in add {
        o.insert(k.into(), v);
    }
    r
}

/// One filter of an answer.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Filter {
    pub kind: BandKind,
    pub fc: f64,
    pub q: f64,
    pub gain: f64,
}

/// An answer, checked.
#[derive(Clone, Debug, PartialEq)]
pub enum Made {
    Filters { preamp: f64, filters: Vec<Filter> },
    Graphic(String),
}

/// Ask autoeq.app for one kind of equalizer.
pub fn equalize(name: &str, m: &Measurement, s: &Settings, kind: Kind) -> Result<Made, String> {
    let body = post(&format!("{SITE}/equalize"), &request(name, m, s, kind).to_string())?;
    parse_answer(&body, kind)
}

fn parse_answer(body: &str, kind: Kind) -> Result<Made, String> {
    let v: serde_json::Value = serde_json::from_str(body).map_err(|e| format!("autoeq.app's answer did not read: {e}"))?;
    if kind == Kind::Graphic {
        let g = v["graphic_eq"].as_str().ok_or("autoeq.app's answer has no GraphicEQ")?;
        let points = g.strip_prefix("GraphicEQ:").ok_or("autoeq.app's GraphicEQ is not one")?;
        // Each point exactly a frequency and a gain, apart by spaces only, and
        // the line written again from those numbers: nothing else in the
        // answer (a line break, a second command) reaches the file.
        let point = |p: &str| -> Option<(f64, f64)> {
            let w: Vec<&str> = p.split(' ').filter(|x| !x.is_empty()).collect();
            let (f, gain): (f64, f64) = match w[..] { [f, gain] => (f.parse().ok()?, gain.parse().ok()?), _ => return None };
            (f > 0.0 && f <= 100_000.0 && gain.abs() <= 60.0).then_some((f, gain))
        };
        let points: Option<Vec<(f64, f64)>> = if g.len() < 64 * 1024 { points.split(';').map(point).collect() } else { None };
        let line = points.map(|p| p.iter().map(|(f, gain)| format!("{f} {gain}")).collect::<Vec<_>>().join("; "));
        return line.map(|l| Made::Graphic(format!("GraphicEQ: {l}"))).ok_or_else(|| "autoeq.app's GraphicEQ has a value out of range".into());
    }
    let p = &v[if matches!(kind, Kind::Fixed10 | Kind::Fixed31) { "fixed_band_eq" } else { "parametric_eq" }];
    let preamp = p["preamp"].as_f64().filter(|x| (-40.0..=10.0).contains(x)).ok_or("autoeq.app's answer has no usable preamp")?;
    let filters = p["filters"].as_array().ok_or("autoeq.app's answer has no filters")?;
    let filters = filters
        .iter()
        .map(|f| {
            let kind = match f["type"].as_str() {
                Some("PEAKING") => BandKind::Peaking,
                Some("LOW_SHELF") => BandKind::LowShelf,
                Some("HIGH_SHELF") => BandKind::HighShelf,
                other => return Err(format!("autoeq.app sent a filter Moosik does not have: {other:?}")),
            };
            let (fc, q, gain) = (f["fc"].as_f64(), f["q"].as_f64(), f["gain"].as_f64());
            match (fc, q, gain) {
                (Some(fc), Some(q), Some(gain)) if (5.0..=30_000.0).contains(&fc) && (0.01..=50.0).contains(&q) && gain.abs() <= 40.0 => {
                    Ok(Filter { kind, fc, q, gain })
                }
                _ => Err("autoeq.app sent a filter with a value out of range".into()),
            }
        })
        .collect::<Result<Vec<_>, String>>()?;
    if filters.is_empty() || filters.len() > MAX_BANDS {
        return Err(format!("autoeq.app sent {} filters", filters.len()));
    }
    Ok(Made::Filters { preamp, filters })
}

/// Filters as Moosik's EQ bands, at full precision; a shelf's Q as the shelf
/// slope Moosik's shelves take.
pub fn bands(filters: &[Filter]) -> Vec<EqBand> {
    filters
        .iter()
        .map(|f| {
            let gain_db = f.gain as f32;
            let q = if matches!(f.kind, BandKind::LowShelf | BandKind::HighShelf) { shelf_slope(f.q, gain_db) } else { f.q as f32 };
            EqBand { freq: f.fc as f32, gain_db, q, kind: f.kind, enabled: true }
        })
        .collect()
}

fn apo_type(k: BandKind) -> &'static str {
    match k {
        BandKind::LowShelf => "LSC",
        BandKind::HighShelf => "HSC",
        _ => "PK",
    }
}

/// An Equalizer APO parametric text ("ParametricEQ.txt"), as AutoEq writes it.
pub fn apo_text(preamp: f64, filters: &[Filter]) -> String {
    let mut s = format!("Preamp: {preamp:.1} dB\n");
    for (i, f) in filters.iter().enumerate() {
        s += &format!("Filter {}: ON {} Fc {:.0} Hz Gain {:.1} dB Q {:.2}\n", i + 1, apo_type(f.kind), f.fc, f.gain, f.q);
    }
    s
}

/// USB Audio Player PRO's equalizer is filled in by hand: its bands, in order.
fn uapp_sheet(title: &str, preamp: f64, filters: &[Filter]) -> String {
    let mut s = format!("USB Audio Player PRO, parametric EQ: {title}\nType these into UAPP's parametric equalizer, band by band.\n\nPreamp: {preamp:.1} dB\n\nBand  Type         Frequency    Gain       Q\n");
    for (i, f) in filters.iter().enumerate() {
        let kind = match f.kind {
            BandKind::LowShelf => "Low shelf",
            BandKind::HighShelf => "High shelf",
            _ => "Peak",
        };
        s += &format!("{:<5} {:<12} {:<12} {:<10} {:.2}\n", i + 1, kind, format!("{:.0} Hz", f.fc), format!("{:+.1} dB", f.gain), f.q);
    }
    s
}

/// A ToneBoosters equalizer preset, as USB Audio Player PRO loads them from
/// its TBEQPresets folder: ten bells. The layout and value scaling follow
/// the community converter EqConverter (Apache 2.0): frequency and Q are the
/// cube root of their place in 16 Hz–20 kHz and 0.1–10, gain and preamp are
/// linear in ±20 dB. None unless the filters are ten or fewer peaks.
fn toneboosters_xml(name: &str, preamp: f64, filters: &[Filter]) -> Option<String> {
    if filters.len() > 10 || filters.iter().any(|f| f.kind != BandKind::Peaking) {
        return None;
    }
    let cube = |x: f64, lo: f64, hi: f64| ((x.clamp(lo, hi) - lo) / (hi - lo)).cbrt();
    let gain = |g: f64| (g.clamp(-20.0, 20.0) + 20.0) / 40.0;
    let name: String = name.chars().map(|c| if c.is_ascii_alphanumeric() || " -_.()".contains(c) { c } else { '_' }).collect();
    let mut v = Vec::new();
    for i in 0..10 {
        match filters.get(i) {
            Some(f) => v.extend([cube(f.fc, 16.0, 20_000.0), gain(f.gain), 1.0, cube(f.q, 0.1, 10.0), 0.21428572, 0.0]),
            // An unused band, as the converter leaves it: off, at its defaults.
            None => v.extend([0.9282573, 0.5, 0.0, 0.39434525, 0.21428572, 0.0]),
        }
    }
    v.extend([0.0, gain(preamp), 1.0, 0.33333334, 0.05, 0.0]);
    let values: String = v.iter().map(|x| format!("<Value>{}</Value>\n", (x * 1e8).round() / 1e8)).collect();
    Some(format!("<?xml version=\"1.0\" encoding=\"iso-8859-1\"?>\n<Preset>\n<PresetInfo Name=\"{name}\" TenBand=\"1\">\n{values}</PresetInfo>\n</Preset>\n"))
}

/// A name safe for a Windows file.
pub fn file_name(s: &str) -> String {
    let t: String = s.chars().map(|c| if "<>:\"/\\|?*".contains(c) || c.is_control() { '_' } else { c }).collect();
    let t = t.trim().trim_end_matches('.').to_string();
    if t.is_empty() { "Headphones".into() } else { t.chars().take(100).collect() }
}

fn readme(name: &str, m: &Measurement, s: &Settings) -> String {
    format!(
        "{name}: equalizers made by autoeq.app (AutoEq) for Moosik\n\
         Measurement: {}\nTarget: {}\nBass boost: {:+.1} dB at {:.0} Hz; treble boost {:+.1} dB; tilt {:+.1} dB/oct; max gain {:.1} dB\n\n\
         Which file for which app:\n\
         - {name} ParametricEQ.txt: Moosik (Import AutoEq...), Equalizer APO, Peace, and other parametric equalizers.\n\
         - {name} GraphicEQ.txt: Wavelet, JamesDSP, Equalizer APO's GraphicEQ.\n\
         - {name} FixedBandEQ 10 bands.txt and 31 bands.txt: equalizers with 10 or 31 sliders.\n\
         - {name} USB Audio Player PRO.txt: the values to type into UAPP's parametric equalizer.\n\
         - TBEQPresets\\{name}.xml: a ToneBoosters equalizer preset for UAPP (experimental). Copy the TBEQPresets\n  \
           folder into UAPP's folder on the phone.\n\n\
         The preamp in each file keeps the boosts from clipping. The frequencies suit any sample rate.\n",
        m.label(), s.target, s.bass.gain, s.bass.fc, s.treble, s.tilt, s.max_gain
    )
}

/// The parametric equalizer kept for a headphone, if one was made before.
pub fn kept(dir: &Path, name: &str) -> Option<(f32, Vec<EqBand>)> {
    let path = dir.join(file_name(name)).join(format!("{} ParametricEQ.txt", file_name(name)));
    crate::spectrum::eq::parse_autoeq(&std::fs::read_to_string(path).ok()?).ok()
}

/// Write one made equalizer for a headphone (kept for Moosik too), in its
/// own folder under `dir`. Returns the folder.
pub fn write(dir: &Path, name: &str, kind: Kind, made: &Made) -> Result<PathBuf, String> {
    let folder = dir.join(file_name(name));
    let base = file_name(name);
    std::fs::create_dir_all(&folder).map_err(|e| format!("{}: {e}", folder.display()))?;
    let (file, text) = match (kind, made) {
        (Kind::Graphic, Made::Graphic(g)) => (format!("{base} GraphicEQ.txt"), format!("{g}\n")),
        (Kind::Parametric, Made::Filters { preamp, filters }) => (format!("{base} ParametricEQ.txt"), apo_text(*preamp, filters)),
        (Kind::Fixed10, Made::Filters { preamp, filters }) => (format!("{base} FixedBandEQ 10 bands.txt"), apo_text(*preamp, filters)),
        (Kind::Fixed31, Made::Filters { preamp, filters }) => (format!("{base} FixedBandEQ 31 bands.txt"), apo_text(*preamp, filters)),
        (Kind::Uapp, Made::Filters { preamp, filters }) => (format!("{base} USB Audio Player PRO.txt"), uapp_sheet(name, *preamp, filters)),
        (Kind::TenPeaks, Made::Filters { preamp, filters }) => {
            let xml = toneboosters_xml(name, *preamp, filters).ok_or("the ToneBoosters preset needs ten peaks or fewer")?;
            let tb = folder.join("TBEQPresets");
            std::fs::create_dir_all(&tb).map_err(|e| format!("{}: {e}", tb.display()))?;
            write_atomic(&tb.join(format!("{base}.xml")), &xml)?;
            return Ok(folder);
        }
        _ => return Err("autoeq.app answered with another kind of equalizer".into()),
    };
    write_atomic(&folder.join(file), &text)?;
    Ok(folder)
}

/// Write the read-me that says which file is for which app.
pub fn write_readme(dir: &Path, name: &str, m: &Measurement, s: &Settings) -> Result<(), String> {
    let folder = dir.join(file_name(name));
    std::fs::create_dir_all(&folder).map_err(|e| format!("{}: {e}", folder.display()))?;
    write_atomic(&folder.join("Read me.txt"), &readme(name, m, s))
}

const UA: &str = concat!("moosik/", env!("MOOSIK_VERSION"), " (https://github.com/HenloAmHorse/Moosik-Player)");

fn agent() -> ureq::Agent {
    ureq::Agent::config_builder()
        .timeout_global(Some(Duration::from_secs(60)))
        .timeout_connect(Some(Duration::from_secs(15)))
        .https_only(true)
        .user_agent(UA)
        .build()
        .into()
}

fn read(mut resp: ureq::http::Response<ureq::Body>) -> Result<String, String> {
    resp.body_mut().with_config().limit(16 << 20).read_to_string().map_err(|e| format!("autoeq.app: {e}"))
}

fn http_get(url: &str) -> Result<String, String> {
    read(agent().get(url).call().map_err(|e| format!("autoeq.app did not answer: {e}"))?)
}

fn http_post(url: &str, body: &str) -> Result<String, String> {
    read(agent().post(url).header("Content-Type", "application/json").send(body).map_err(|e| format!("autoeq.app did not answer: {e}"))?)
}

#[cfg(not(test))]
fn get(url: &str) -> Result<String, String> {
    http_get(url)
}

#[cfg(not(test))]
fn post(url: &str, body: &str) -> Result<String, String> {
    http_post(url, body)
}

/// Test builds reach the network only for the hand-run live check
/// (`MOOSIK_AUTOEQ_LIVE`); otherwise requests go to the answers a test sets
/// here (by URL, and for a POST the request's JSON), else fail.
#[cfg(test)]
pub(crate) type Fake = Box<dyn Fn(&str, Option<&serde_json::Value>) -> Result<String, String> + Send>;
#[cfg(test)]
pub(crate) static FAKE: std::sync::Mutex<Option<Fake>> = std::sync::Mutex::new(None);

#[cfg(test)]
fn live() -> bool {
    std::env::var_os("MOOSIK_AUTOEQ_LIVE").is_some()
}

#[cfg(test)]
fn get(url: &str) -> Result<String, String> {
    if let Some(f) = FAKE.lock().unwrap().as_ref() {
        return f(url, None);
    }
    if live() { http_get(url) } else { Err("no network in tests".into()) }
}

#[cfg(test)]
fn post(url: &str, body: &str) -> Result<String, String> {
    let v: serde_json::Value = serde_json::from_str(body).map_err(|e| e.to_string())?;
    if let Some(f) = FAKE.lock().unwrap().as_ref() {
        return f(url, Some(&v));
    }
    if live() { http_post(url, body) } else { Err("no network in tests".into()) }
}

#[cfg(test)]
pub(crate) mod tests {
    use super::*;

    /// Tests that set `FAKE` take turns.
    pub(crate) static TURN: std::sync::Mutex<()> = std::sync::Mutex::new(());

    pub(crate) const ENTRIES_JSON: &str = r#"{
        "Sennheiser HD 650": [{"form": "over-ear", "rig": "GRAS 45BC-10", "source": "oratory1990"},
                              {"form": "over-ear", "rig": null, "source": "Kuulokenurkka"}],
        "Sennheiser HD 600": [{"form": "over-ear", "rig": "GRAS 45BC-10", "source": "oratory1990"}],
        "Sennheiser HD 6XX": [{"form": "over-ear", "rig": "GRAS 43AG-7", "source": "crinacle"}],
        "Apple AirPods Pro 2": [{"form": "in-ear", "rig": "711", "source": "crinacle"}],
        "Audio-Technica ATH-M50x": [{"form": "over-ear", "rig": "HMS II.3", "source": "Rtings"}],
        "Moondrop Aria": [{"form": "in-ear", "rig": "711", "source": "crinacle"}],
        "Moondrop Aria Snow Edition": [{"form": "in-ear", "rig": "711", "source": "crinacle"}],
        "Nothing": []
    }"#;
    pub(crate) const TARGETS_JSON: &str = r#"[
        {"label": "Harman over-ear 2018", "compatible": [{"source": "oratory1990", "form": "over-ear"}],
         "recommended": [{"source": "oratory1990", "form": "over-ear"}], "bassBoost": {"fc": 105, "q": 0.7, "gain": 6}},
        {"label": "Harman in-ear 2019", "compatible": [{"source": "crinacle", "form": "in-ear", "rig": "711"}],
         "recommended": [], "bassBoost": {"fc": 105, "q": 0.7, "gain": 9.5}},
        {"label": "Flat", "compatible": [], "recommended": [], "bassBoost": {"fc": 105, "q": 0.7, "gain": 0}}
    ]"#;
    pub(crate) const PEQ_JSON: &str = r#"{"fr": {}, "parametric_eq": {"fs": 48000, "preamp": -4.8169, "filters": [
        {"type": "LOW_SHELF", "fc": 105.0, "q": 0.7, "gain": 2.3033},
        {"type": "PEAKING", "fc": 96.1385, "q": 1.8084, "gain": -0.9395},
        {"type": "HIGH_SHELF", "fc": 10000.0, "q": 0.7, "gain": -1.25}]}}"#;

    fn catalog() -> Catalog {
        Catalog::parse(ENTRIES_JSON, TARGETS_JSON).unwrap()
    }

    fn names<'a>(c: &'a Catalog, q: &str) -> Vec<&'a str> {
        c.search(q, 10).into_iter().map(|i| c.names[i].as_str()).collect()
    }

    #[test]
    fn the_search_forgives_spaces_case_order_and_a_typo() {
        let c = catalog();
        assert_eq!(c.names.len(), 7, "a headphone without measurements is left out");
        assert_eq!(names(&c, "hd650")[0], "Sennheiser HD 650");
        assert_eq!(names(&c, "HD 650")[0], "Sennheiser HD 650");
        assert_eq!(names(&c, "sennhiser hd650")[0], "Sennheiser HD 650", "one letter missing");
        assert_eq!(names(&c, "650 sennheiser")[0], "Sennheiser HD 650", "any order");
        assert!(!names(&c, "hd650").contains(&"Sennheiser HD 6XX"), "two letters off in five is not a match");
        assert_eq!(names(&c, "airpods pro2"), vec!["Apple AirPods Pro 2"]);
        assert_eq!(names(&c, "ath m50x"), vec!["Audio-Technica ATH-M50x"]);
        assert_eq!(names(&c, "atm50x"), vec!["Audio-Technica ATH-M50x"], "one letter missing");
        assert_eq!(names(&c, "moondrop aria"), vec!["Moondrop Aria", "Moondrop Aria Snow Edition"], "the shorter name first");
        assert!(names(&c, "  ").is_empty());
        assert!(names(&c, "zzzz").is_empty());
    }

    #[test]
    fn targets_follow_the_measurement() {
        let c = catalog();
        let hd650 = &c.measurements[c.names.iter().position(|n| n == "Sennheiser HD 650").unwrap()];
        let labels = |m: &Measurement| c.targets_for(m).into_iter().map(|i| c.targets[i].label.clone()).collect::<Vec<_>>();
        assert_eq!(labels(&hd650[0]), ["Harman over-ear 2018"]);
        assert_eq!(labels(&hd650[1]), ["Harman over-ear 2018", "Harman in-ear 2019", "Flat"], "none fits: every target");
        let s = Settings::for_target(&c.targets[0]);
        assert_eq!((s.bass.gain, s.max_gain, s.treble), (6.0, 6.0, 0.0), "autoeq.app's own starting point");
        assert_eq!(hd650[1].label(), "Kuulokenurkka (over-ear)");
    }

    #[test]
    fn the_request_names_the_measurement_and_the_kind() {
        let c = catalog();
        let m = &c.measurements[c.names.iter().position(|n| n == "Sennheiser HD 650").unwrap()][0];
        let s = Settings::for_target(&c.targets[0]);
        let r = request("Sennheiser HD 650", m, &s, Kind::Uapp);
        assert_eq!(r["parametric_eq_config"], "USB_AUDIO_PLAYER_PRO");
        assert_eq!((r["rig"].as_str(), r["bass_boost_gain"].as_f64(), r["fs"].as_u64()), (Some("GRAS 45BC-10"), Some(6.0), Some(48_000)));
        assert_eq!(request("x", m, &s, Kind::Fixed31)["fixed_band_eq_config"], "31_BAND_GRAPHIC_EQ");
        assert_eq!(request("x", m, &s, Kind::Graphic)["graphic_eq"], true);
        let other = &c.measurements[c.names.iter().position(|n| n == "Sennheiser HD 650").unwrap()][1];
        assert!(request("x", other, &s, Kind::Parametric)["rig"].is_null(), "a measurement without a rig");
    }

    #[test]
    fn an_answer_is_checked_and_written_for_each_app() {
        let made = parse_answer(PEQ_JSON, Kind::Parametric).unwrap();
        let Made::Filters { preamp, filters } = &made else { panic!("filters") };
        assert_eq!(filters.len(), 3);
        // The text for other apps reads back in Moosik as the same EQ.
        let text = apo_text(*preamp, filters);
        assert!(text.starts_with("Preamp: -4.8 dB\nFilter 1: ON LSC Fc 105 Hz Gain 2.3 dB Q 0.70\n"), "{text}");
        let (pre, back) = crate::spectrum::eq::parse_autoeq(&text).unwrap();
        assert_eq!((pre, back.len(), back[2].kind == BandKind::HighShelf), (-4.8, 3, true));
        // Moosik itself takes the full precision.
        assert_eq!(bands(filters)[1].freq, 96.1385);
        // A shelf's Q as the slope Moosik's shelves take; a peak keeps its Q.
        let b = bands(filters);
        assert_eq!((b[0].q, b[1].q, b[2].q), (shelf_slope(0.7, 2.3033f64 as f32), 1.8084f64 as f32, shelf_slope(0.7, -1.25)));
        // Out of range, unknown or missing: refused.
        assert!(parse_answer(&PEQ_JSON.replace("96.1385", "1e9"), Kind::Parametric).is_err());
        assert!(parse_answer(&PEQ_JSON.replace("PEAKING", "BAND_PASS"), Kind::Parametric).is_err());
        assert!(parse_answer(r#"{"parametric_eq": {"preamp": 0, "filters": []}}"#, Kind::Parametric).is_err());
        assert!(parse_answer("<html>", Kind::Parametric).is_err());
        let g = parse_answer(r#"{"graphic_eq": "GraphicEQ: 20 -0.8; 21 -1.1"}"#, Kind::Graphic).unwrap();
        assert_eq!(g, Made::Graphic("GraphicEQ: 20 -0.8; 21 -1.1".into()));
        assert!(parse_answer(r#"{"graphic_eq": "GraphicEQ: 20 -999"}"#, Kind::Graphic).is_err());
        // Nothing but a frequency and a gain per point, apart by spaces: a line
        // break and a second command, a word, infinity, NaN, an empty point,
        // a tab. And the line is written again from the numbers.
        for bad in ["GraphicEQ: 20 -0.8; 21 -1.1\nInclude: C:\\x.txt", "GraphicEQ: 20 -0.8 extra", "GraphicEQ: inf 1",
                    "GraphicEQ: 20 NaN", "GraphicEQ: 20 -0.8;; 21 1", "GraphicEQ: 0 1", "GraphicEQ: 20\t-0.8"] {
            assert!(parse_answer(&serde_json::json!({ "graphic_eq": bad }).to_string(), Kind::Graphic).is_err(), "{bad:?}");
        }
        let g = parse_answer(r#"{"graphic_eq": "GraphicEQ:  20.0  -0.80;21 -1.1 "}"#, Kind::Graphic).unwrap();
        assert_eq!(g, Made::Graphic("GraphicEQ: 20 -0.8; 21 -1.1".into()));
        // The UAPP sheet lists every band; the ToneBoosters preset takes peaks only.
        let sheet = uapp_sheet("HD 650", *preamp, filters);
        assert!(sheet.contains("Low shelf") && sheet.contains("High shelf") && sheet.contains("Preamp: -4.8 dB"), "{sheet}");
        assert!(toneboosters_xml("HD 650", *preamp, filters).is_none(), "shelves");
        let peaks: Vec<Filter> = filters.iter().map(|f| Filter { kind: BandKind::Peaking, ..*f }).collect();
        let xml = toneboosters_xml("HD 650 / x", *preamp, &peaks).unwrap();
        assert_eq!(xml.matches("<Value>").count(), 66);
        assert!(xml.contains("Name=\"HD 650 _ x\"") && xml.contains("TenBand=\"1\""), "{xml}");
        // 0 dB gain is the middle; 16 Hz and Q 0.1 are 0.
        let lone = [Filter { kind: BandKind::Peaking, fc: 16.0, q: 0.1, gain: 0.0 }];
        let xml = toneboosters_xml("a", 0.0, &lone).unwrap();
        assert!(xml.contains("<Value>0</Value>\n<Value>0.5</Value>\n<Value>1</Value>\n<Value>0</Value>\n<Value>0.21428572</Value>"), "{xml}");
    }

    #[test]
    fn files_go_in_the_headphones_own_folder() {
        let dir = std::env::temp_dir().join(format!("moosik-autoeq-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&dir);
        let made = parse_answer(PEQ_JSON, Kind::Parametric).unwrap();
        let folder = write(&dir, "AKG K371: \"Studio\"", Kind::Parametric, &made).unwrap();
        assert_eq!(folder, dir.join("AKG K371_ _Studio_"));
        assert!(kept(&dir, "AKG K371: \"Studio\"").is_some_and(|(p, b)| p == -4.8 && b.len() == 3), "kept for Moosik");
        assert!(write(&dir, "x", Kind::Graphic, &made).is_err(), "the wrong kind of answer");
        assert_eq!(file_name(" ..."), "Headphones");
        std::fs::remove_dir_all(&dir).unwrap();
    }

    #[test]
    fn the_list_is_kept_only_when_it_reads_and_tests_stay_offline() {
        let _turn = TURN.lock().unwrap_or_else(|e| e.into_inner());
        let dir = std::env::temp_dir().join(format!("moosik-autoeq-list-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&dir);
        *FAKE.lock().unwrap() = None;
        assert!(refresh(&dir).unwrap_err().contains("no network"), "nothing is fetched in tests");
        *FAKE.lock().unwrap() = Some(Box::new(|url: &str, _: Option<&serde_json::Value>| {
            Ok(if url.ends_with("/entries") { ENTRIES_JSON.into() } else { TARGETS_JSON.into() })
        }));
        assert_eq!(refresh(&dir), Ok(true));
        assert_eq!(refresh(&dir), Ok(false), "unchanged");
        assert_eq!(load(&dir).map(|c| c.names.len()), Some(7));
        *FAKE.lock().unwrap() = Some(Box::new(|_: &str, _: Option<&serde_json::Value>| Ok("<html>maintenance</html>".into())));
        assert!(refresh(&dir).is_err());
        assert_eq!(load(&dir).map(|c| c.names.len()), Some(7), "a list that does not read is never kept");
        *FAKE.lock().unwrap() = None;
        std::fs::remove_dir_all(&dir).unwrap();
    }

    /// By hand, against the real autoeq.app (network, temporary folders only):
    /// MOOSIK_AUTOEQ_LIVE=1 cargo test --release autoeq_live -- --ignored --nocapture
    #[test]
    #[ignore]
    fn autoeq_live() {
        assert!(live(), "set MOOSIK_AUTOEQ_LIVE=1");
        let _turn = TURN.lock().unwrap_or_else(|e| e.into_inner());
        *FAKE.lock().unwrap() = None;
        let dir = std::env::temp_dir().join(format!("moosik-autoeq-live-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&dir);
        assert!(refresh(&dir).unwrap(), "the list is new here");
        let c = load(&dir).expect("the list reads");
        println!("{} headphones, {} targets", c.names.len(), c.targets.len());
        let hit = c.search("sennhiser hd650", 5);
        println!("sennhiser hd650 -> {:?}", hit.iter().map(|&i| &c.names[i]).collect::<Vec<_>>());
        let i = hit[0];
        assert_eq!(c.names[i], "Sennheiser HD 650");
        let m = c.measurements[i].iter().find(|m| m.source == "oratory1990").expect("oratory1990's");
        let t = c.targets_for(m)[0];
        let s = Settings::for_target(&c.targets[t]);
        println!("{} with {} (bass {:+.1} dB)", m.label(), s.target, s.bass.gain);
        for kind in [Kind::Parametric, Kind::Graphic, Kind::Fixed10, Kind::Fixed31, Kind::Uapp, Kind::TenPeaks] {
            let made = equalize(&c.names[i], m, &s, kind).unwrap_or_else(|e| panic!("{kind:?}: {e}"));
            write(&dir, &c.names[i], kind, &made).unwrap_or_else(|e| panic!("{kind:?}: {e}"));
            match &made {
                Made::Filters { preamp, filters } => {
                    println!("{kind:?}: {} filters, preamp {preamp:.1} dB", filters.len());
                    if kind == Kind::Parametric {
                        print!("{}", apo_text(*preamp, filters));
                        assert_eq!((filters.len(), filters[0].kind, filters[9].kind), (10, BandKind::LowShelf, BandKind::HighShelf));
                    }
                }
                Made::Graphic(g) => println!("{kind:?}: {} points", g.matches(';').count() + 1),
            }
        }
        write_readme(&dir, &c.names[i], m, &s).unwrap();
        let files: Vec<String> = walk(&dir.join("Sennheiser HD 650"));
        println!("{files:?}");
        assert_eq!(files.len(), 7);
        std::fs::remove_dir_all(&dir).unwrap();
    }

    fn walk(d: &Path) -> Vec<String> {
        let mut out = Vec::new();
        for e in std::fs::read_dir(d).unwrap().flatten() {
            if e.path().is_dir() {
                out.extend(walk(&e.path()));
            } else {
                out.push(e.file_name().to_string_lossy().into_owned());
            }
        }
        out.sort();
        out
    }
}
