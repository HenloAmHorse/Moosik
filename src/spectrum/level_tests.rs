//! "Normalized level": the loudness measure, the gain it gives, where the
//! offset lands on the bars, and the EQ drawn on them.

use super::*;

/// A mono 48 kHz WAV of a 1 kHz sine at `dbfs`.
fn sine_wav(path: &Path, dbfs: f32, secs: f32) {
    let sr = 48_000u32;
    let amp = 10f32.powf(dbfs / 20.0);
    let data: Vec<u8> = (0..(sr as f32 * secs) as usize).flat_map(|i| {
        let s = (std::f32::consts::TAU * 1000.0 * i as f32 / sr as f32).sin() * amp;
        ((s * 32_767.0) as i16).to_le_bytes()
    }).collect();
    let mut out = Vec::with_capacity(44 + data.len());
    out.extend(b"RIFF");
    out.extend(((36 + data.len()) as u32).to_le_bytes());
    out.extend(b"WAVEfmt ");
    out.extend(16u32.to_le_bytes());
    out.extend(1u16.to_le_bytes());
    out.extend(1u16.to_le_bytes());
    out.extend(sr.to_le_bytes());
    out.extend((sr * 2).to_le_bytes());
    out.extend(2u16.to_le_bytes());
    out.extend(16u16.to_le_bytes());
    out.extend(b"data");
    out.extend((data.len() as u32).to_le_bytes());
    out.extend(data);
    std::fs::write(path, out).unwrap();
}

struct Dir(PathBuf);
impl Drop for Dir {
    fn drop(&mut self) { let _ = std::fs::remove_dir_all(&self.0); }
}
fn dir(tag: &str) -> Dir {
    let d = std::env::temp_dir().join(format!("moosik-level-{tag}-{}", std::process::id()));
    std::fs::create_dir_all(&d).unwrap();
    Dir(d)
}

#[test]
fn a_quiet_song_measures_quieter_by_what_it_lacks() {
    let d = dir("measure");
    let (loud, quiet, silent) = (d.0.join("loud.wav"), d.0.join("quiet.wav"), d.0.join("silent.wav"));
    sine_wav(&loud, -10.0, 3.0);
    sine_wav(&quiet, -30.0, 3.0);
    sine_wav(&silent, -200.0, 3.0);
    let go = AtomicBool::new(false);
    let l = measure_lufs(&loud, 88_200, &go).unwrap();
    let q = measure_lufs(&quiet, 88_200, &go).unwrap();
    // Mean square 0.05, and K-weighting lifts 1 kHz by about 0.7 dB.
    assert!((l + 13.0).abs() < 0.3, "{l}");
    assert!((l - q - 20.0).abs() < 0.1, "{l} against {q}");
    assert_eq!(measure_lufs(&silent, 88_200, &go), None, "silence has no loudness");
    assert_eq!(measure_lufs(&loud, 88_200, &AtomicBool::new(true)), None, "a stopped measurement");
}

#[test]
fn the_gain_brings_a_song_to_the_target_within_limits() {
    assert_eq!(level_gain_db(LEVEL_TARGET_LUFS), 0.0);
    assert_eq!(level_gain_db(LEVEL_TARGET_LUFS - 8.0), 8.0);
    assert_eq!(level_gain_db(-60.0), 20.0, "near silence is not blown up");
    assert_eq!(level_gain_db(LEVEL_TARGET_LUFS + 3.0), -3.0);
    assert_eq!(level_gain_db(0.0), -6.0, "a very loud song comes down only a little");
}

#[test]
fn the_offset_moves_heard_bars_and_leaves_silence_silent() {
    let mut w = SpectrumWindow::new();
    w.analyzer.set_pre_frames(cache::PreFrames::from_analysis(&[vec![0.0, 0.25, 0.5, 1.0]]), 180.0);
    w.analyzer.smoothed = vec![0.0; 4];
    w.analyzer.magnitudes = vec![0.0; 4];
    w.analyzer.peak_input = vec![0.0; 4];
    w.analyzer.level_offset_db = 8.0;
    w.analyzer.apply_pre_frame(0, 0.0);
    let m = w.analyzer.magnitudes.clone();
    assert_eq!(m[0], 0.0, "the floor, and above the file's Nyquist, stays dark");
    assert!((m[1] - 0.35).abs() < 1e-3 && (m[2] - 0.6).abs() < 1e-3, "{m:?}");
    assert_eq!(m[3], 1.0, "the top holds");
    // Live bars take the same offset, and their floor stays down too.
    assert!((w.analyzer.bar_db(10f32.powf(-40.0 / 20.0), 0, 1.0) - 0.6).abs() < 1e-4);
    assert_eq!(w.analyzer.bar_db(0.0, 0, 1.0), 0.0);
}

#[test]
fn the_eq_moves_bars_in_db_and_reports_its_largest_boost() {
    let (bars, boost) = eq_on_bars(&[0.5, 0.5, 0.0, 0.99], |i| [8.0, -8.0, 8.0, 8.0][i]);
    for (got, want) in bars.iter().zip([0.6, 0.4, 0.0, 1.0]) {
        assert!((got - want).abs() < 1e-6, "{bars:?}");
    }
    assert_eq!(boost, 8.0);
    assert_eq!(eq_on_bars(&[0.5], |_| -3.0).1, 0.0, "a cut needs no room");
}

#[test]
fn the_window_offsets_by_the_songs_loudness_less_the_eq_room() {
    let d = dir("window");
    let song = d.0.join("song.wav");
    sine_wav(&song, -20.0, 3.0);
    let mut w = SpectrumWindow::new();
    w.normalize_level = true;
    w.current_path = Some(song);
    let t = Instant::now();
    while w.level.is_none() && t.elapsed() < Duration::from_secs(20) {
        w.tick_level();
        std::thread::sleep(Duration::from_millis(5));
    }
    let gain = level_gain_db(w.level.expect("measured"));
    assert!((gain - 13.0).abs() < 0.3, "−23 LUFS comes up about 13 dB: {gain}");
    assert_eq!(w.analyzer.level_offset_db, gain);
    w.eq_headroom_db = 6.0;
    w.tick_level();
    assert_eq!(w.analyzer.level_offset_db, gain - 6.0, "room for the EQ's boost");
    w.normalize_level = false;
    w.tick_level();
    assert_eq!(w.analyzer.level_offset_db, 0.0, "off: as recorded");
}

/// Where real songs measure, for setting `LEVEL_TARGET_LUFS`.
///
/// `MOOSIK_LEVEL_DIR=<folder> cargo test --release --bin moosik -- --ignored --nocapture level_survey`
#[test]
#[ignore = "reads a folder of songs"]
fn level_survey() {
    let dir = PathBuf::from(std::env::var("MOOSIK_LEVEL_DIR").expect("set MOOSIK_LEVEL_DIR"));
    let go = AtomicBool::new(false);
    for e in std::fs::read_dir(dir).unwrap().flatten() {
        let t = Instant::now();
        let l = measure_lufs(&e.path(), 88_200, &go);
        println!("{:>7} LUFS  gain {:>6}  {:>5.2}s  {}",
                 l.map_or("—".into(), |l| format!("{l:.1}")),
                 l.map_or("—".into(), |l| format!("{:+.1}", level_gain_db(l))),
                 t.elapsed().as_secs_f32(), e.file_name().to_string_lossy());
    }
}
