//! Chords for a whole track, from a trained model (lv-chordia: Jiang et al.,
//! "Large-Vocabulary Chord Transcription via Chord Structure Decomposition",
//! ISMIR 2019; MIT): CQT after a tuning estimate, five ChordNets averaged,
//! then a Viterbi path over 301 chords.
//!
//! The weights are the five `.sdict` files of the lv-chordia 1.1.0 wheel,
//! read with a pickle reader that runs nothing.
//!
//! The network, the CQT front end and the chord vocabulary here are a Rust
//! port of lv-chordia 1.1.0, Copyright (c) 2023 Music X Lab, MIT License
//! (text in `third_party/lv-chordia-1.1.0-LICENSE`). The weights are not
//! distributed with Moosik; the user downloads them on request.

mod audio;
mod cqt;
mod hmm;
pub mod install;
mod net;
mod pickle;

use crate::spectrum::store::SourceSnapshot;
use std::path::{Path, PathBuf};
use std::sync::atomic::AtomicBool;

pub use cqt::SR;

/// The weight files, in ensemble order.
pub const WEIGHT_FILES: [&str; 5] = [
    "joint_chord_net_ismir_naive_v1.0_reweight(0.0,10.0)_s0.best.sdict",
    "joint_chord_net_ismir_naive_v1.0_reweight(0.0,10.0)_s1.best.sdict",
    "joint_chord_net_ismir_naive_v1.0_reweight(0.0,10.0)_s2.best.sdict",
    "joint_chord_net_ismir_naive_v1.0_reweight(0.0,10.0)_s3.best.sdict",
    "joint_chord_net_ismir_naive_v1.0_reweight(0.0,10.0)_s4.best.sdict",
];

/// One chord and when it holds, in seconds. `name` is in Harte notation
/// ("C:maj", "A:min7", "G:maj/3", "N" for no chord).
#[derive(Clone, Debug, PartialEq)]
pub struct Segment {
    pub start: f32,
    pub end: f32,
    pub name: &'static str,
    /// How sure the model was, 0..1 (see `hmm::decode`).
    pub conf: f32,
}

#[cfg(test)]
impl Segment {
    fn with_conf(self, conf: f32) -> Segment {
        Segment { conf, ..self }
    }
}

pub struct Model {
    nets: Vec<net::Net>,
}

impl Model {
    /// The five networks from the folder holding `WEIGHT_FILES`.
    pub fn load(dir: &Path) -> Result<Model, String> {
        let nets = WEIGHT_FILES
            .iter()
            .map(|f| {
                let bytes = std::fs::read(dir.join(f)).map_err(|e| format!("{f}: {e}"))?;
                net::Net::from_state(pickle::read_state_dict(&bytes, "net")?).map_err(|e| format!("{f}: {e}"))
            })
            .collect::<Result<_, String>>()?;
        Ok(Model { nets })
    }

    /// Per frame (hop 512 at 22050 Hz), the heads averaged over the five networks.
    fn probs(&self, mono: &[f32]) -> Vec<[f32; 100]> {
        let c = cqt::cqt(mono, cqt::tuning(mono));
        let mut sum = vec![[0f32; 100]; c.len()];
        for n in &self.nets {
            for (s, p) in sum.iter_mut().zip(n.forward(&c)) {
                s.iter_mut().zip(p).for_each(|(a, b)| *a += b);
            }
        }
        let k = 1.0 / self.nets.len() as f32;
        sum.iter_mut().for_each(|s| s.iter_mut().for_each(|v| *v *= k));
        sum
    }

    /// The chords of `mono`: 22050 Hz, one channel, the whole track.
    pub fn chords(&self, mono: &[f32]) -> Vec<Segment> {
        let dt = cqt::HOP as f32 / SR as f32;
        hmm::decode(&self.probs(mono))
            .into_iter()
            .map(|(a, b, name, conf)| Segment { start: a as f32 * dt, end: b as f32 * dt, name, conf })
            .collect()
    }
}

/// A track's chords: from the cache when this exact file was done before,
/// otherwise decoded and recognised (then cached). `cancel` stops it. The
/// user's corrections (see `correct`) are laid over the model's chords.
pub fn analyze(model: &Model, path: &Path, cache: &Path, cancel: &AtomicBool) -> Result<Vec<Segment>, String> {
    let before = SourceSnapshot::of_path(path).ok();
    let key = before.as_ref().map(file_key);
    let file = key.map(|h| cache.join(format!("{h:016x}.v2.json")));
    let mut segs = match file.as_ref().and_then(|f| std::fs::read_to_string(f).ok()).and_then(|t| from_json(&t)) {
        Some(segs) => segs,
        None => {
            let mono = audio::load(path, cancel)?;
            let segs = model.chords(&mono);
            // Kept only if the file is still what was read, as the analysis
            // cache does (docs/CACHE.md, D2).
            if let Some(f) = &file
                && SourceSnapshot::of_path(path).ok() == before
            {
                save(cache, f, &to_json(&segs));
            }
            segs
        }
    };
    if let Some(h) = key {
        overlay(&mut segs, cache, h);
    }
    Ok(segs)
}

fn overlay(segs: &mut [Segment], cache: &Path, h: u64) {
    for f in std::fs::read_to_string(fix_file(cache, h)).ok().and_then(|t| from_json(&t)).unwrap_or_default() {
        if let Some(s) = segs.iter_mut().find(|s| (s.start - f.start).abs() < 0.01) {
            s.name = f.name;
            s.conf = 1.0;
        }
    }
}

/// The user says segment `i` of this track is really `name`: it changes in
/// `segs` and is kept beside the cache, apart from the model's chords so a
/// newer model doesn't lose it. Corrections are matched by start time.
pub fn correct(path: &Path, cache: &Path, segs: &mut [Segment], i: usize, name: &str) -> Result<(), String> {
    let name = hmm::CHORDS.iter().find(|c| c.0 == name).ok_or_else(|| format!("unknown chord {name}"))?.0;
    let seg = segs.get_mut(i).ok_or("no such chord")?;
    seg.name = name;
    seg.conf = 1.0;
    let seg = seg.clone();
    let snap = SourceSnapshot::of_path(path).map_err(|e| format!("can't read the file: {e}"))?;
    let f = fix_file(cache, file_key(&snap));
    let mut fixes = std::fs::read_to_string(&f).ok().and_then(|t| from_json(&t)).unwrap_or_default();
    fixes.retain(|x| (x.start - seg.start).abs() >= 0.01);
    fixes.push(seg);
    if save(cache, &f, &to_json(&fixes)) { Ok(()) } else { Err("can't save the correction".into()) }
}

/// Whole-file save: written aside, then moved over the old one.
fn save(dir: &Path, f: &Path, text: &str) -> bool {
    let _ = std::fs::create_dir_all(dir);
    let tmp = f.with_extension("part");
    std::fs::write(&tmp, text).is_ok() && std::fs::rename(&tmp, f).is_ok()
}

/// A track as the analysis cache knows it (docs/CACHE.md, D3): path, size,
/// modified time and a sample of its contents.
fn file_key(s: &SourceSnapshot) -> u64 {
    let mut b = s.path.to_string_lossy().into_owned().into_bytes();
    b.extend(s.size.to_le_bytes());
    b.extend(s.mtime_ns.to_le_bytes());
    b.extend(s.sample);
    crate::spectrum::identity::fnv64(&b)
}

fn fix_file(cache: &Path, h: u64) -> PathBuf {
    cache.join(format!("{h:016x}.fix.json"))
}

fn to_json(segs: &[Segment]) -> String {
    serde_json::to_string(&segs.iter().map(|s| (s.start, s.end, s.name, s.conf)).collect::<Vec<_>>()).unwrap_or_default()
}

fn from_json(t: &str) -> Option<Vec<Segment>> {
    let rows: Vec<(f32, f32, String, f32)> = serde_json::from_str(t).ok()?;
    rows.into_iter()
        .map(|(start, end, n, conf)| hmm::CHORDS.iter().find(|c| c.0 == n).map(|c| Segment { start, end, name: c.0, conf }))
        .collect()
}

/// The chord sounding at `t` seconds, and the one after it.
pub fn at(segs: &[Segment], t: f32) -> (Option<&Segment>, Option<&Segment>) {
    let i = segs.partition_point(|s| s.end <= t);
    (segs.get(i).filter(|s| s.start <= t), segs[i.min(segs.len())..].iter().find(|s| s.start > t))
}

/// Harte notation to the way a chord chart writes it: "C:maj" to "C",
/// "A:min7" to "Am7", "G:maj/3" to "G/B", "N" to "N.C.".
pub fn display(name: &str) -> String {
    const NOTES: [&str; 12] = ["C", "C#", "D", "Eb", "E", "F", "F#", "G", "Ab", "A", "Bb", "B"];
    if name == "N" || name == "X" {
        return "N.C.".into();
    }
    let (root, rest) = name.split_once(':').unwrap_or((name, "maj"));
    let (qual, bass) = rest.split_once('/').map_or((rest, None), |(q, b)| (q, Some(b)));
    let q = match qual {
        "maj" => "",
        "min" => "m",
        "7" => "7",
        "maj7" => "maj7",
        "min7" => "m7",
        "dim" => "dim",
        "aug" => "aug",
        "sus2" => "sus2",
        "sus4" => "sus4",
        "hdim7" => "m7b5",
        "dim7" => "dim7",
        "maj6" => "6",
        "min6" => "m6",
        "9" => "9",
        "maj9" => "maj9",
        "min9" => "m9",
        "11" => "11",
        "13" => "13",
        "minmaj7" => "m(maj7)",
        other => other,
    };
    let bass = bass.and_then(|b| {
        let semis = match b {
            "2" => 2, "b3" => 3, "3" => 4, "4" => 5, "b5" => 6, "5" => 7, "#5" => 8, "6" => 9, "b7" => 10, "7" => 11, "b2" => 1, "b6" => 8,
            _ => return None,
        };
        let r = NOTES.iter().position(|n| *n == root).or_else(|| ["C", "Db", "D", "D#", "E", "F", "Gb", "G", "G#", "A", "A#", "B"].iter().position(|n| *n == root))?;
        Some(NOTES[(r + semis) % 12])
    });
    match bass {
        Some(b) => format!("{root}{q}/{b}"),
        None => format!("{root}{q}"),
    }
}

/// The roots as the model's chord names spell them, from C.
pub const ROOTS: [&str; 12] = ["C", "C#", "D", "Eb", "E", "F", "F#", "G", "Ab", "A", "Bb", "B"];

/// The pitch classes (0 = C) a Harte chord name sounds, bass included.
pub fn tones(name: &str) -> Vec<usize> {
    let Some((root, rest)) = name.split_once(':') else { return Vec::new() };
    let Some(r) = ROOTS.iter().position(|n| *n == root) else { return Vec::new() };
    let (qual, bass) = rest.split_once('/').map_or((rest, None), |(q, b)| (q, Some(b)));
    let iv: &[usize] = match qual {
        "min" => &[0, 3, 7],
        "7" => &[0, 4, 7, 10],
        "maj7" => &[0, 4, 7, 11],
        "min7" => &[0, 3, 7, 10],
        "dim" => &[0, 3, 6],
        "aug" => &[0, 4, 8],
        "sus2" => &[0, 2, 7],
        "sus4" => &[0, 5, 7],
        "hdim7" => &[0, 3, 6, 10],
        "dim7" => &[0, 3, 6, 9],
        "maj6" => &[0, 4, 7, 9],
        "min6" => &[0, 3, 7, 9],
        "minmaj7" => &[0, 3, 7, 11],
        "9" => &[0, 4, 7, 10, 2],
        "maj9" => &[0, 4, 7, 11, 2],
        "min9" => &[0, 3, 7, 10, 2],
        "11" => &[0, 4, 7, 10, 2, 5],
        "13" => &[0, 4, 7, 10, 2, 9],
        _ => &[0, 4, 7],
    };
    let mut out: Vec<usize> = iv.iter().map(|i| (r + i) % 12).collect();
    let b = bass.and_then(|b| match b {
        "b2" => Some(1), "2" => Some(2), "b3" => Some(3), "3" => Some(4), "4" => Some(5), "b5" => Some(6),
        "5" => Some(7), "#5" | "b6" => Some(8), "6" => Some(9), "b7" => Some(10), "7" => Some(11), _ => None,
    });
    if let Some(b) = b
        && !out.contains(&((r + b) % 12))
    {
        out.push((r + b) % 12);
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The signal `lv_refvec.py` made the reference from: C, Am, F, G, two
    /// seconds each, three harmonics a note.
    fn signal() -> Vec<f32> {
        let sr = SR as f64;
        let mut y = Vec::new();
        for notes in [[60, 64, 67, 48], [57, 60, 64, 45], [53, 57, 60, 41], [55, 59, 62, 43]] {
            for i in 0..2 * SR as usize {
                let t = i as f64 / sr;
                let mut v = 0.0;
                for m in notes {
                    let f = 440.0 * 2f64.powf((m as f64 - 69.0) / 12.0);
                    for h in 1..=3 {
                        v += (std::f64::consts::TAU * f * h as f64 * t).sin() / h as f64;
                    }
                }
                y.push((v * (t / 0.01).min(1.0) * 0.05) as f32);
            }
        }
        y
    }

    fn reference() -> serde_json::Value {
        serde_json::from_str(include_str!("refvec.json")).unwrap()
    }

    #[test]
    fn chord_names_read_like_a_chart() {
        assert_eq!(display("C:maj"), "C");
        assert_eq!(display("A:min7"), "Am7");
        assert_eq!(display("G:maj/3"), "G/B");
        assert_eq!(display("C:min/b7"), "Cm/Bb");
        assert_eq!(display("N"), "N.C.");
        // every name in the vocabulary reads as something
        for (n, _) in hmm::CHORDS {
            assert!(!display(n).is_empty());
        }
    }

    #[test]
    fn a_chord_names_its_notes() {
        assert_eq!(tones("C:maj"), [0, 4, 7]);
        assert_eq!(tones("A:min7"), [9, 0, 4, 7]);
        assert_eq!(tones("D:maj/b7"), [2, 6, 9, 0]);
        assert!(tones("N").is_empty());
    }

    #[test]
    fn the_chord_at_a_time_and_the_next_one() {
        let s = |a, b, n| Segment { start: a, end: b, name: n, conf: 0.5 };
        let segs = [s(0.0, 1.0, "N"), s(1.0, 3.0, "C:maj"), s(3.0, 5.0, "G:maj")];
        let (now, next) = at(&segs, 2.0);
        assert_eq!((now.unwrap().name, next.unwrap().name), ("C:maj", "G:maj"));
        let (now, next) = at(&segs, 4.5);
        assert_eq!((now.unwrap().name, next.map(|n| n.name)), ("G:maj", None));
        assert_eq!(at(&segs, 9.0).0, None);
    }

    #[test]
    fn cached_chords_come_back_the_same() {
        let s = |a, b, n| Segment { start: a, end: b, name: n, conf: 0.5 };
        let segs = vec![s(0.0, 1.5, "N"), s(1.5, 3.25, "A:min7")];
        assert_eq!(from_json(&to_json(&segs)), Some(segs));
        assert_eq!(from_json("[[0,1,\"Q:weird\"]]"), None);
    }

    #[test]
    fn new_audio_under_the_same_name_size_and_time_is_a_new_track() {
        let dir = std::env::temp_dir().join(format!("moosik-chordkey-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(&dir).unwrap();
        let track = dir.join("t.flac");
        std::fs::write(&track, b"first take!").unwrap();
        let when = std::fs::metadata(&track).unwrap().modified().unwrap();
        let a = file_key(&SourceSnapshot::of_path(&track).unwrap());
        std::fs::write(&track, b"second take").unwrap();
        std::fs::File::options().write(true).open(&track).unwrap().set_modified(when).unwrap();
        let b = file_key(&SourceSnapshot::of_path(&track).unwrap());
        assert_ne!(a, b);
        std::fs::remove_dir_all(&dir).unwrap();
    }

    #[test]
    fn a_correction_outlives_the_model_cache() {
        let dir = std::env::temp_dir().join(format!("moosik-chordfix-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(&dir).unwrap();
        let track = dir.join("t.flac");
        std::fs::write(&track, b"not really audio").unwrap();
        let s = |a, b, n| Segment { start: a, end: b, name: n, conf: 0.4 };
        let model = vec![s(0.0, 2.0, "C:maj"), s(2.0, 4.0, "G:maj"), s(4.0, 6.0, "A:min")];
        let mut segs = model.clone();
        correct(&track, &dir, &mut segs, 1, "G:sus4").unwrap();
        correct(&track, &dir, &mut segs, 2, "A:min7").unwrap();
        correct(&track, &dir, &mut segs, 1, "B:dim").unwrap(); // a second thought replaces the first
        assert!(correct(&track, &dir, &mut segs, 0, "Q:weird").is_err());
        assert_eq!(segs[1], s(2.0, 4.0, "B:dim").with_conf(1.0));
        // Fresh from the model (e.g. a new model, a new cache): the user's word still holds.
        let mut again = model.clone();
        overlay(&mut again, &dir, file_key(&SourceSnapshot::of_path(&track).unwrap()));
        assert_eq!(again, segs);
        assert_eq!(again[0], model[0]);
        std::fs::remove_dir_all(&dir).unwrap();
    }

    #[test]
    fn the_signal_is_the_reference_one() {
        let s: f64 = signal().iter().map(|v| v.abs() as f64).sum();
        let want = reference()["signal_sum"].as_f64().unwrap();
        assert!((s - want).abs() / want < 1e-5, "{s} vs {want}");
    }

    #[test]
    fn tuning_and_cqt_match_librosa() {
        let y = signal();
        let r = reference();
        let tun = cqt::tuning(&y);
        assert!((tun - r["tuning"].as_f64().unwrap()).abs() < 1e-9, "tuning {tun}");
        let c = cqt::cqt(&y, tun);
        assert_eq!(c.len(), r["n_frames"].as_u64().unwrap() as usize);
        for (fi, want) in r["cqt_frames"].as_array().unwrap().iter().zip(r["cqt"].as_array().unwrap()) {
            let row = &c[fi.as_u64().unwrap() as usize];
            let want: Vec<f64> = want.as_array().unwrap().iter().map(|v| v.as_f64().unwrap()).collect();
            let top = want.iter().cloned().fold(0.0, f64::max);
            for (b, (g, w)) in row.iter().zip(&want).enumerate() {
                assert!((*g as f64 - w).abs() <= 1e-4 * top + 1e-6, "frame {fi} bin {b}: {g} vs {w}");
            }
        }
    }

    #[test]
    fn the_chord_vocabulary_is_well_formed() {
        assert_eq!(hmm::CHORDS.len(), 301);
        assert_eq!(hmm::CHORDS[0].0, "N");
        for (name, a) in hmm::CHORDS {
            for (i, &v) in a.iter().enumerate() {
                assert!((v as i32) < net::HEADS[i] as i32 && (i >= 2 || v >= 0), "{name}");
            }
        }
    }

    #[test]
    fn decoding_holds_a_chord_until_it_clearly_changes() {
        // Two chords, each clearly most likely for 50 frames, with a blip.
        let c = hmm::CHORDS.iter().position(|c| c.0 == "C:maj").unwrap();
        let g = hmm::CHORDS.iter().position(|c| c.0 == "G:maj").unwrap();
        let frame = |want: usize| {
            let mut p = [1e-3f32; 100];
            let a = hmm::CHORDS[want].1;
            let starts = [0usize, 73, 86, 90, 94, 97];
            for i in 0..6 {
                if a[i] >= 0 {
                    p[starts[i] + a[i] as usize] = 0.9;
                }
            }
            p
        };
        let mut probs: Vec<[f32; 100]> = (0..50).map(|_| frame(c)).chain((0..50).map(|_| frame(g))).collect();
        probs[20] = frame(g); // one stray frame doesn't pay the change penalty
        let segs = hmm::decode(&probs);
        let names: Vec<&str> = segs.iter().map(|s| s.2).collect();
        // The path always starts on "no chord" (as the original's does).
        assert_eq!(names, ["N", "C:maj", "G:maj"], "{segs:?}");
        assert_eq!((segs[1].0, segs[1].1), (1, 50));
    }

    /// The whole path against the Python original. Needs the weights:
    /// MOOSIK_LV_WEIGHTS = the folder with the five `.sdict` files.
    #[test]
    #[ignore]
    fn the_model_matches_the_python_original() {
        let dir = std::env::var("MOOSIK_LV_WEIGHTS").expect("set MOOSIK_LV_WEIGHTS to the weights folder");
        let m = Model::load(Path::new(&dir)).unwrap();
        let y = signal();
        let p = m.probs(&y);
        let r = reference();
        let mut worst = 0f32;
        for (fi, heads) in r["cqt_frames"].as_array().unwrap().iter().zip(r["probs"].as_array().unwrap()) {
            let row = &p[fi.as_u64().unwrap() as usize];
            let want: Vec<f32> = heads.as_array().unwrap().iter().flat_map(|h| h.as_array().unwrap().iter().map(|v| v.as_f64().unwrap() as f32)).collect();
            for (g, w) in row.iter().zip(&want) {
                worst = worst.max((g - w).abs());
            }
        }
        assert!(worst < 1e-3, "largest probability difference {worst}");
        let got: Vec<(f32, f32, &str)> = m.chords(&y).iter().map(|s| (s.start, s.end, s.name)).collect();
        let want: Vec<(f32, f32, String)> = r["labels"]
            .as_array()
            .unwrap()
            .iter()
            .map(|l| (l[0].as_f64().unwrap() as f32, l[1].as_f64().unwrap() as f32, l[2].as_str().unwrap().to_string()))
            .collect();
        assert_eq!(got.len(), want.len(), "{got:?}");
        for (g, w) in got.iter().zip(&want) {
            assert_eq!(g.2, w.2);
            assert!((g.0 - w.0).abs() < 1e-3 && (g.1 - w.1).abs() < 1e-3, "{g:?} vs {w:?}");
        }
        eprintln!("worst probability difference {worst}");
    }

    /// How long three minutes take (prints; needs the weights like the test above).
    #[test]
    #[ignore]
    fn three_minutes_take_seconds() {
        let dir = std::env::var("MOOSIK_LV_WEIGHTS").expect("set MOOSIK_LV_WEIGHTS to the weights folder");
        let t0 = std::time::Instant::now();
        let m = Model::load(Path::new(&dir)).unwrap();
        let load = t0.elapsed();
        let y: Vec<f32> = signal().iter().cycle().take(180 * SR as usize).copied().collect();
        for from in [44_100, 48_000] {
            let x: Vec<f32> = y.iter().cycle().take(180 * from as usize).copied().collect();
            let t = std::time::Instant::now();
            let _ = audio::resample(&x, from, SR);
            eprintln!("resampling 3 min from {from} Hz: {:?}", t.elapsed());
        }
        let t1 = std::time::Instant::now();
        let c = cqt::cqt(&y, cqt::tuning(&y));
        let front = t1.elapsed();
        let t2 = std::time::Instant::now();
        let _ = m.nets[0].forward(&c);
        let one = t2.elapsed();
        eprintln!("load {load:?}, tuning + CQT {front:?}, one network {one:?} (five: about {:?})", one * 5);
    }

    /// Chords for every `Track*/mix.flac` under MOOSIK_CHORD_EVAL, written to
    /// `<track>/chords_rust.json` for the Python scorer.
    #[test]
    #[ignore]
    fn dump_for_scoring() {
        let dir = std::env::var("MOOSIK_LV_WEIGHTS").expect("MOOSIK_LV_WEIGHTS");
        let set = std::env::var("MOOSIK_CHORD_EVAL").expect("MOOSIK_CHORD_EVAL");
        let m = Model::load(Path::new(&dir)).unwrap();
        let mut tracks: Vec<_> = std::fs::read_dir(&set).unwrap().flatten().map(|e| e.path()).filter(|p| p.join("mix.flac").exists()).collect();
        tracks.sort();
        for t in tracks.iter().take(20) {
            let t0 = std::time::Instant::now();
            let mono = audio::load(&t.join("mix.flac"), &AtomicBool::new(false)).unwrap();
            let segs = m.chords(&mono);
            std::fs::write(t.join("chords_rust.json"), to_json(&segs)).unwrap();
            eprintln!("{} {:.1}s for {:.0}s", t.display(), t0.elapsed().as_secs_f32(), mono.len() as f32 / SR as f32);
        }
    }
}
