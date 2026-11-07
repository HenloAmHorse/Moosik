//! 1.6.1's screens, rendered offscreen (wgpu, no window) against the test
//! profile, never the owner's. Run by hand, e.g.:
//!   MOOSIK_SHOTS=<folder> MOOSIK_FLAC=<song> MOOSIK_TEST_VIDEO=<.webm>
//!   cargo test --release screens161 -- --ignored

use egui_kittest::kittest::Queryable;
use egui_kittest::Harness;
use std::path::PathBuf;

fn shot(h: &mut Harness<'_, crate::MoosikApp>, dir: &std::path::Path, name: &str) {
    match h.render() {
        Ok(img) => img.save(dir.join(format!("{name}.png"))).expect("save"),
        Err(e) => panic!("render {name}: {e}"),
    }
}

fn click(h: &mut Harness<'_, crate::MoosikApp>, label: &str) {
    h.get_by_label(label).click();
    h.run_steps(2);
}

/// Steps in real time, so the decoder and the engine get somewhere.
fn wait(h: &mut Harness<'_, crate::MoosikApp>, ms: u64) {
    for _ in 0..ms / 50 {
        h.step();
        std::thread::sleep(std::time::Duration::from_millis(50));
    }
}

/// A song with a music video and lyrics beside it: the video is Listen's
/// backdrop and the spectrum's, Watch shows it whole with the lyrics as
/// subtitles, and V / Esc switch back.
#[test]
#[ignore]
fn screens161_music_video() {
    let dir = PathBuf::from(std::env::var("MOOSIK_SHOTS").expect("set MOOSIK_SHOTS"));
    std::fs::create_dir_all(&dir).unwrap();
    let songs = std::env::temp_dir().join(format!("moosik-mv161-{}", std::process::id()));
    std::fs::create_dir_all(&songs).unwrap();
    let song = songs.join("song.flac");
    std::fs::copy(std::env::var("MOOSIK_FLAC").expect("set MOOSIK_FLAC"), &song).unwrap();
    std::fs::copy(std::env::var("MOOSIK_TEST_VIDEO").expect("set MOOSIK_TEST_VIDEO"), songs.join("song.video.webm")).unwrap();
    let lrc: String = (0..60).map(|s| format!("[00:{:02}.00]Line {s}, sung over the picture\n", s)).collect();
    std::fs::write(songs.join("song.lrc"), lrc).unwrap();
    let m = crate::moosik_dir();
    std::fs::create_dir_all(&m).unwrap();
    std::fs::write(m.join("last_playlist.json"), serde_json::to_string(&[&song]).unwrap()).unwrap();
    std::fs::write(m.join("player.json"), r#"{"volume":0.0,"loop_mode":"Sequential"}"#).unwrap();
    let mut h = Harness::builder().with_size(egui::vec2(1300.0, 860.0)).wgpu().build_eframe(|cc| crate::MoosikApp::new(cc));
    h.run_steps(5);
    click(&mut h, "Play");
    wait(&mut h, 2500);
    assert!(h.state().spectrum_window.video.is_some(), "a picture reached the spectrum");
    shot(&mut h, &dir, "mv1-listen");
    click(&mut h, "Watch");
    wait(&mut h, 600);
    assert!(h.state().shell.theater);
    assert!(!h.state().shell.cinema, "the bars stay a while on entering");
    shot(&mut h, &dir, "mv2-theater");
    // Still for three seconds: the bars step aside.
    h.input_mut().events.push(egui::Event::PointerMoved(egui::pos2(650.0, 400.0)));
    wait(&mut h, 3600);
    assert!(h.state().shell.cinema, "bars hidden");
    shot(&mut h, &dir, "mv2b-cinema");
    h.input_mut().events.push(egui::Event::PointerMoved(egui::pos2(660.0, 410.0)));
    h.run_steps(2);
    assert!(!h.state().shell.cinema, "a move brings them back");
    // A double click on the picture: full screen; Esc leaves it first.
    let at = egui::pos2(650.0, 300.0);
    for _ in 0..2 {
        h.input_mut().events.push(egui::Event::PointerButton { pos: at, button: egui::PointerButton::Primary, pressed: true, modifiers: Default::default() });
        h.input_mut().events.push(egui::Event::PointerButton { pos: at, button: egui::PointerButton::Primary, pressed: false, modifiers: Default::default() });
        h.run_steps(1);
    }
    h.run_steps(1);
    assert!(h.state().shell.full_screen, "double click");
    h.input_mut().events.push(egui::Event::Key { key: egui::Key::Escape, physical_key: None, pressed: true, repeat: false, modifiers: Default::default() });
    h.run_steps(2);
    assert!(!h.state().shell.full_screen && h.state().shell.theater, "Esc: out of full screen, still in theater");
    click(&mut h, "Back");
    assert!(!h.state().shell.theater);
    click(&mut h, "Analyze");
    wait(&mut h, 1000);
    shot(&mut h, &dir, "mv3-spectrum");
    h.state_mut().stop();
    h.run_steps(1);
    for n in ["last_playlist.json", "player.json"] {
        let _ = std::fs::remove_file(m.join(n));
    }
    let _ = std::fs::remove_dir_all(&songs);
}

/// Copy a tools folder (yt-dlp, its one-folder build or the exe, and
/// Deno) into the test profile, the way an install leaves them.
fn tools_into_profile(from: &std::path::Path) {
    fn copy(a: &std::path::Path, b: &std::path::Path) {
        if a.is_dir() {
            std::fs::create_dir_all(b).unwrap();
            for e in std::fs::read_dir(a).unwrap().flatten() {
                if e.file_name() != "cache" {
                    copy(&e.path(), &b.join(e.file_name()));
                }
            }
        } else if !b.is_file() {
            std::fs::copy(a, b).unwrap();
        }
    }
    copy(from, &crate::youtube::install::dir(&crate::moosik_dir()));
}

/// A song that came from YouTube, without a picture or video: its cover
/// arrives by itself, and Get video on the cover fetches the video, which
/// then plays behind it (network). MOOSIK_YT_TOOLS = a tools folder.
#[test]
#[ignore]
fn screens161_get_video() {
    let dir = PathBuf::from(std::env::var("MOOSIK_SHOTS").expect("set MOOSIK_SHOTS"));
    std::fs::create_dir_all(&dir).unwrap();
    tools_into_profile(&PathBuf::from(std::env::var("MOOSIK_YT_TOOLS").expect("set MOOSIK_YT_TOOLS")));
    let songs = std::env::temp_dir().join(format!("moosik-gv161-{}", std::process::id()));
    std::fs::create_dir_all(&songs).unwrap();
    let song = songs.join("zoo.flac");
    std::fs::copy(std::env::var("MOOSIK_FLAC").expect("set MOOSIK_FLAC"), &song).unwrap();
    let m = crate::moosik_dir();
    let info = crate::youtube::Info { id: "jNQXAC9IVRw".into(), title: "Me at the zoo".into(), artist: "jawed".into(), album: "YouTube".into(), duration: Some(19.0), url: crate::youtube::watch_url("jNQXAC9IVRw", false), video_asked: None };
    crate::youtube::store::record(&crate::youtube::store::file(&m), &song, &info).unwrap();
    std::fs::write(m.join("last_playlist.json"), serde_json::to_string(&[&song]).unwrap()).unwrap();
    std::fs::write(m.join("player.json"), r#"{"volume":0.0,"loop_mode":"Sequential"}"#).unwrap();
    let mut h = Harness::builder().with_size(egui::vec2(1300.0, 860.0)).wgpu().build_eframe(|cc| crate::MoosikApp::new(cc));
    h.run_steps(5);
    click(&mut h, "Play");
    wait(&mut h, 3000);
    assert!(crate::youtube::cover_path(&song).is_file(), "the cover arrived");
    shot(&mut h, &dir, "gv1-cover-and-get-video");
    click(&mut h, "Get video");
    let t = std::time::Instant::now();
    while h.state().spectrum_window.video.is_none() && t.elapsed().as_secs() < 60 {
        wait(&mut h, 250);
    }
    assert!(h.state().spectrum_window.video.is_some(), "the video came and shows");
    wait(&mut h, 500);
    shot(&mut h, &dir, "gv2-video-in");
    h.state_mut().stop();
    h.run_steps(1);
    for n in ["last_playlist.json", "player.json", "youtube.json"] {
        let _ = std::fs::remove_file(m.join(n));
    }
    let _ = std::fs::remove_dir_all(&songs);
}

/// What a music video costs while playing: Listen (shown) against the
/// Library (hidden, so not decoded), as process CPU over four seconds.
#[test]
#[ignore]
fn screens161_video_cost() {
    use windows_sys::Win32::System::Threading::{GetCurrentProcess, GetProcessTimes};
    let cpu = || {
        let (mut a, mut b, mut k, mut u) = (0u64, 0u64, 0u64, 0u64);
        // SAFETY: four valid out-pointers to FILETIME-sized integers.
        unsafe { GetProcessTimes(GetCurrentProcess(), &mut a as *mut _ as _, &mut b as *mut _ as _, &mut k as *mut _ as _, &mut u as *mut _ as _) };
        (k + u) as f64 / 1e4
    };
    let songs = std::env::temp_dir().join(format!("moosik-vc161-{}", std::process::id()));
    std::fs::create_dir_all(&songs).unwrap();
    let song = songs.join("song.flac");
    std::fs::copy(std::env::var("MOOSIK_FLAC").expect("set MOOSIK_FLAC"), &song).unwrap();
    std::fs::copy(std::env::var("MOOSIK_TEST_VIDEO").expect("set MOOSIK_TEST_VIDEO"), songs.join("song.video.webm")).unwrap();
    let m = crate::moosik_dir();
    std::fs::create_dir_all(&m).unwrap();
    std::fs::write(m.join("last_playlist.json"), serde_json::to_string(&[&song]).unwrap()).unwrap();
    std::fs::write(m.join("player.json"), r#"{"volume":0.0,"loop_mode":"Sequential"}"#).unwrap();
    let mut h = Harness::builder().with_size(egui::vec2(1300.0, 860.0)).build_eframe(|cc| crate::MoosikApp::new(cc));
    h.run_steps(5);
    click(&mut h, "Play");
    // Past the start, when the whole-track analysis is busiest.
    wait(&mut h, 8000);
    let measure = |h: &mut Harness<'_, crate::MoosikApp>, name: &str| {
        wait(h, 1000);
        let (c0, t0) = (cpu(), std::time::Instant::now());
        wait(h, 4000);
        let pct = 100.0 * (cpu() - c0) / (t0.elapsed().as_secs_f64() * 1000.0);
        eprintln!("VIDEO {name:<8} {pct:5.1} % of one core");
        pct
    };
    let listen = measure(&mut h, "listen");
    click(&mut h, "Library");
    let library = measure(&mut h, "library");
    h.state_mut().stop();
    h.run_steps(1);
    for n in ["last_playlist.json", "player.json"] {
        let _ = std::fs::remove_file(m.join(n));
    }
    let _ = std::fs::remove_dir_all(&songs);
    assert!(library < listen, "hidden video is not decoded");
}

/// What one spectrum frame costs (update, then the picture drawn), with
/// the music video behind it and without: 180 fps leaves 5.5 ms a frame.
#[test]
#[ignore]
fn screens161_spectrum_frame_cost() {
    let songs = std::env::temp_dir().join(format!("moosik-fc161-{}", std::process::id()));
    std::fs::create_dir_all(&songs).unwrap();
    let song = songs.join("song.flac");
    std::fs::copy(std::env::var("MOOSIK_FLAC").expect("set MOOSIK_FLAC"), &song).unwrap();
    std::fs::copy(std::env::var("MOOSIK_TEST_VIDEO").expect("set MOOSIK_TEST_VIDEO"), songs.join("song.video.webm")).unwrap();
    let m = crate::moosik_dir();
    std::fs::create_dir_all(&m).unwrap();
    std::fs::write(m.join("last_playlist.json"), serde_json::to_string(&[&song]).unwrap()).unwrap();
    std::fs::write(m.join("player.json"), r#"{"volume":0.0,"loop_mode":"Sequential"}"#).unwrap();
    let mut h = Harness::builder().with_size(egui::vec2(1920.0, 1080.0)).wgpu().build_eframe(|cc| crate::MoosikApp::new(cc));
    h.run_steps(5);
    click(&mut h, "Play");
    click(&mut h, "Analyze");
    wait(&mut h, 8000);
    let measure = |h: &mut Harness<'_, crate::MoosikApp>, name: &str| {
        wait(h, 1000);
        let (mut up, mut draw) = (Vec::new(), Vec::new());
        for _ in 0..120 {
            let t = std::time::Instant::now();
            h.step();
            up.push(t.elapsed().as_secs_f64() * 1e3);
            let t = std::time::Instant::now();
            let _ = h.render();
            draw.push(t.elapsed().as_secs_f64() * 1e3);
            std::thread::sleep(std::time::Duration::from_millis(5));
        }
        let stat = |v: &mut Vec<f64>| {
            v.sort_by(|a, b| a.partial_cmp(b).unwrap());
            (v.iter().sum::<f64>() / v.len() as f64, v[v.len() * 95 / 100])
        };
        let ((ua, u95), (da, d95)) = (stat(&mut up), stat(&mut draw));
        eprintln!("FRAME {name:<9} update {ua:5.2} ms (p95 {u95:5.2})   draw {da:5.2} ms (p95 {d95:5.2})");
    };
    measure(&mut h, "video");
    h.state_mut().appearance.video_spectrum = false;
    measure(&mut h, "no-video");
    h.state_mut().stop();
    h.run_steps(1);
    for n in ["last_playlist.json", "player.json"] {
        let _ = std::fs::remove_file(m.join(n));
    }
    let _ = std::fs::remove_dir_all(&songs);
}

/// Play on a result played before starts at once from the cache, without
/// running yt-dlp (no tools are installed in this test's profile).
#[test]
#[ignore]
fn screens161_cached_play_is_instant() {
    let m = crate::moosik_dir();
    let tools = crate::youtube::install::dir(&m);
    let _ = std::fs::remove_dir_all(&tools);
    let cache = crate::youtube::cache_dir(&m);
    std::fs::create_dir_all(&cache).unwrap();
    let id = "jNQXAC9IVRw";
    let song = cache.join(format!("{id}.flac"));
    std::fs::copy(std::env::var("MOOSIK_FLAC").expect("set MOOSIK_FLAC"), &song).unwrap();
    let info = crate::youtube::Info { id: id.into(), title: "Me at the zoo".into(), artist: "jawed".into(), album: "YouTube".into(), duration: Some(19.0), url: crate::youtube::watch_url(id, false), video_asked: None };
    crate::youtube::store::record(&crate::youtube::store::file(&m), &song, &info).unwrap();
    std::fs::write(m.join("player.json"), r#"{"volume":0.0,"loop_mode":"Sequential"}"#).unwrap();
    let mut h = Harness::builder().with_size(egui::vec2(1300.0, 860.0)).build_eframe(|cc| crate::MoosikApp::new(cc));
    h.run_steps(5);
    h.state_mut().appearance.youtube_dir = Some(std::env::temp_dir().join("moosik-never-the-music-folder"));
    let hit = crate::youtube::Hit { id: id.into(), title: "Me at the zoo".into(), channel: Some("jawed".into()), duration: Some(19.0), music: false };
    let t = std::time::Instant::now();
    h.state_mut().youtube_fetch(&hit, crate::shell::youtube_ui::Then::Play);
    h.run_steps(2);
    let elapsed = t.elapsed();
    let state = h.state();
    let i = state.current_index.expect("playing");
    assert_eq!(state.playlist[i].path, song);
    assert!(elapsed.as_millis() < 1000, "{elapsed:?}");
    // Save keeps it in the downloads folder (a temporary one here, never
    // the real Music folder), with its video.
    let keep = std::env::temp_dir().join(format!("moosik-save161-{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&keep);
    h.state_mut().appearance.youtube_dir = Some(keep.clone());
    std::fs::write(cache.join(format!("{id}.video.webm")), b"v").unwrap();
    h.run_steps(3);
    click(&mut h, "Save");
    let kept: Vec<String> = std::fs::read_dir(&keep).unwrap().flatten().map(|e| e.file_name().to_string_lossy().into_owned()).collect();
    assert!(kept.contains(&"jawed - Me at the zoo [jNQXAC9IVRw].flac".to_string()), "{kept:?}");
    assert!(kept.contains(&"jawed - Me at the zoo [jNQXAC9IVRw].video.webm".to_string()), "{kept:?}");
    assert!(h.query_by_label("Save").is_none(), "saved once");
    let _ = std::fs::remove_dir_all(&keep);
    let _ = std::fs::remove_file(cache.join(format!("{id}.video.webm")));
    h.state_mut().stop();
    h.run_steps(1);
    let _ = std::fs::remove_file(&song);
    let _ = std::fs::remove_file(crate::youtube::cover_path(&song));
    for n in ["player.json", "youtube.json", "last_playlist.json"] {
        let _ = std::fs::remove_file(m.join(n));
    }
}

/// An album link is its songs (network); Queue all adds them in order,
/// one at a time (here three cached ones, which come at once).
#[test]
#[ignore]
fn screens161_queue_all() {
    let dir = PathBuf::from(std::env::var("MOOSIK_SHOTS").expect("set MOOSIK_SHOTS"));
    std::fs::create_dir_all(&dir).unwrap();
    tools_into_profile(&PathBuf::from(std::env::var("MOOSIK_YT_TOOLS").expect("set MOOSIK_YT_TOOLS")));
    let m = crate::moosik_dir();
    std::fs::write(m.join("player.json"), r#"{"volume":0.0,"loop_mode":"Sequential"}"#).unwrap();
    let mut h = Harness::builder().with_size(egui::vec2(1300.0, 860.0)).wgpu().build_eframe(|cc| crate::MoosikApp::new(cc));
    h.run_steps(5);
    h.state_mut().appearance.youtube_dir = Some(std::env::temp_dir().join("moosik-never-the-music-folder"));
    h.state_mut().appearance.youtube_video = false;
    click(&mut h, "Library");
    h.run_steps(10);
    click(&mut h, "YouTube");
    h.run_steps(10);
    let album = "https://music.youtube.com/browse/MPREb_tQfaWH32ovE";
    h.get_by_role(egui::accesskit::Role::TextInput).type_text(album);
    h.run_steps(2);
    h.get_by_role(egui::accesskit::Role::TextInput).key_press(egui_kittest::kittest::Key::Enter);
    let t = std::time::Instant::now();
    while h.state().shell.yt_hits() == 0 && t.elapsed().as_secs() < 40 {
        wait(&mut h, 250);
    }
    wait(&mut h, 2500);
    shot(&mut h, &dir, "yt3-album");
    let n = h.state().shell.yt_hits();
    assert!(n > 10, "an album's songs, not a search for the link: {n}");
    assert_eq!(h.state().shell.yt_hit(0).unwrap().title, "Come Together (2019 Mix)");
    // Three of them, cached, queued in order.
    let cache = crate::youtube::cache_dir(&m);
    std::fs::create_dir_all(&cache).unwrap();
    let flac = std::env::var("MOOSIK_FLAC").expect("set MOOSIK_FLAC");
    let hits: Vec<_> = (0..3).map(|i| h.state().shell.yt_hit(i).unwrap()).collect();
    let mut songs = Vec::new();
    for hit in &hits {
        let song = cache.join(format!("{}.flac", hit.id));
        std::fs::copy(&flac, &song).unwrap();
        let info = crate::youtube::Info { id: hit.id.clone(), title: hit.title.clone(), artist: "The Beatles".into(), album: "Abbey Road".into(), duration: hit.duration, url: crate::youtube::watch_url(&hit.id, true), video_asked: None };
        crate::youtube::store::record(&crate::youtube::store::file(&m), &song, &info).unwrap();
        songs.push(song);
    }
    h.state_mut().shell.yt_show(hits, album);
    h.run_steps(3);
    let before = h.state().playlist.len();
    click(&mut h, "Queue all 3");
    h.run_steps(3);
    let got: Vec<PathBuf> = h.state().playlist[before..].iter().map(|t| t.path.clone()).collect();
    assert_eq!(got, songs);
    shot(&mut h, &dir, "yt4-queued");
    for s in &songs {
        let _ = std::fs::remove_file(s);
        let _ = std::fs::remove_file(crate::youtube::cover_path(s));
    }
    for n in ["player.json", "youtube.json", "last_playlist.json", "appearance.json"] {
        let _ = std::fs::remove_file(m.join(n));
    }
}

/// The YouTube view's Sign in: the button, the card when no browser is
/// signed in (made up here: the test never reads a real browser), and
/// signed in.
#[test]
#[ignore]
fn screens161_sign_in() {
    let dir = PathBuf::from(std::env::var("MOOSIK_SHOTS").expect("set MOOSIK_SHOTS"));
    std::fs::create_dir_all(&dir).unwrap();
    tools_into_profile(&PathBuf::from(std::env::var("MOOSIK_YT_TOOLS").expect("set MOOSIK_YT_TOOLS")));
    let mut h = Harness::builder().with_size(egui::vec2(1300.0, 860.0)).wgpu().build_eframe(|cc| crate::MoosikApp::new(cc));
    h.run_steps(5);
    click(&mut h, "Library");
    h.run_steps(10);
    click(&mut h, "YouTube");
    h.run_steps(10);
    assert!(h.query_by_label("Sign in").is_some());
    h.state_mut().shell.yt_sign_failed("The window was closed before signing in");
    h.run_steps(5);
    shot(&mut h, &dir, "yt5-sign-in-card");
    click(&mut h, "Close");
    h.state_mut().appearance.youtube_sign_in = Some("firefox".into());
    crate::youtube::set_sign_in(Some("firefox".into()));
    h.run_steps(5);
    shot(&mut h, &dir, "yt6-signed-in");
    assert!(h.query_by_label("Liked music").is_some());
    click(&mut h, "● Signed in");
    assert!(!crate::youtube::signed_in(), "a click signs out");
    let _ = std::fs::remove_file(crate::moosik_dir().join("appearance.json"));
}

/// Settings → YouTube, music videos included (a tall window shows the
/// whole drawer).
#[test]
#[ignore]
fn screens161_settings() {
    let dir = PathBuf::from(std::env::var("MOOSIK_SHOTS").expect("set MOOSIK_SHOTS"));
    std::fs::create_dir_all(&dir).unwrap();
    let mut h = Harness::builder().with_size(egui::vec2(1300.0, 4200.0)).wgpu().build_eframe(|cc| crate::MoosikApp::new(cc));
    h.run_steps(10);
    click(&mut h, "All controls");
    h.run_steps(40);
    shot(&mut h, &dir, "set1-settings");
    // Signed in through Firefox: the note says where the sign-in stays.
    click(&mut h, "Firefox");
    h.run_steps(5);
    assert_eq!(h.state().appearance.youtube_sign_in.as_deref(), Some("firefox"));
    assert!(crate::youtube::signed_in());
    shot(&mut h, &dir, "set2-signed-in");
    click(&mut h, "Signed out");
    assert!(!crate::youtube::signed_in());
    let _ = std::fs::remove_file(crate::moosik_dir().join("appearance.json"));
}

/// Library → YouTube with recent searches; one is searched again (network),
/// and a cached result says "ready". MOOSIK_YT_TOOLS = a tools folder.
#[test]
#[ignore]
fn screens161_youtube_view() {
    let dir = PathBuf::from(std::env::var("MOOSIK_SHOTS").expect("set MOOSIK_SHOTS"));
    std::fs::create_dir_all(&dir).unwrap();
    tools_into_profile(&PathBuf::from(std::env::var("MOOSIK_YT_TOOLS").expect("set MOOSIK_YT_TOOLS")));
    let m = crate::moosik_dir();
    let cache = crate::youtube::cache_dir(&m);
    std::fs::create_dir_all(&cache).unwrap();
    // "Me at the zoo", as if played before.
    std::fs::write(cache.join("jNQXAC9IVRw.webm"), b"").unwrap();
    let mut h = Harness::builder().with_size(egui::vec2(1300.0, 860.0)).wgpu().build_eframe(|cc| crate::MoosikApp::new(cc));
    h.state_mut().appearance.youtube_recent = vec!["me at the zoo".into(), "yoasobi".into(), "lofi hip hop radio".into()];
    h.run_steps(5);
    click(&mut h, "Library");
    h.run_steps(10);
    click(&mut h, "YouTube");
    h.run_steps(10);
    shot(&mut h, &dir, "yt1-recent");
    click(&mut h, "me at the zoo");
    let t = std::time::Instant::now();
    while h.state().shell.yt_hits() == 0 && t.elapsed().as_secs() < 30 {
        wait(&mut h, 250);
    }
    wait(&mut h, 2000);
    shot(&mut h, &dir, "yt2-results");
    assert!(h.state().shell.yt_hits() > 0);
    assert_eq!(h.state().appearance.youtube_recent.first().map(String::as_str), Some("me at the zoo"));
    let _ = std::fs::remove_file(cache.join("jNQXAC9IVRw.webm"));
    let _ = std::fs::remove_file(m.join("appearance.json"));
}

/// The whole way (network): search, Audio + video, Play; the song plays
/// first, its cover arrives, then the video, and Watch shows it.
#[test]
#[ignore]
fn screens161_full_flow() {
    let dir = PathBuf::from(std::env::var("MOOSIK_SHOTS").expect("set MOOSIK_SHOTS"));
    std::fs::create_dir_all(&dir).unwrap();
    tools_into_profile(&PathBuf::from(std::env::var("MOOSIK_YT_TOOLS").expect("set MOOSIK_YT_TOOLS")));
    let m = crate::moosik_dir();
    let _ = std::fs::remove_dir_all(crate::youtube::cache_dir(&m));
    std::fs::write(m.join("player.json"), r#"{"volume":0.0,"loop_mode":"Sequential"}"#).unwrap();
    let mut h = Harness::builder().with_size(egui::vec2(1300.0, 860.0)).wgpu().build_eframe(|cc| crate::MoosikApp::new(cc));
    h.state_mut().appearance.youtube_video = true;
    h.state_mut().appearance.video_height = 480;
    h.run_steps(5);
    click(&mut h, "Library");
    click(&mut h, "YouTube");
    h.run_steps(5);
    let field = h.get_by_role(egui::accesskit::Role::TextInput);
    field.focus();
    field.type_text("https://www.youtube.com/watch?v=dQw4w9WgXcQ");
    h.run_steps(2);
    h.press_key(egui::Key::Enter);
    let t = std::time::Instant::now();
    while h.state().shell.yt_hits() == 0 && t.elapsed().as_secs() < 30 {
        wait(&mut h, 250);
    }
    assert_eq!(h.state().shell.yt_hits(), 1, "the link is that one video");
    let t = std::time::Instant::now();
    // The row's Play (the player bar has one too).
    let hit = h.state().shell.yt_hit(0).unwrap();
    h.state_mut().youtube_fetch(&hit, crate::shell::youtube_ui::Then::Play);
    while h.state().current_index.is_none() && t.elapsed().as_secs() < 60 {
        wait(&mut h, 250);
    }
    let started = t.elapsed();
    assert!(h.state().current_index.is_some(), "it plays");
    let song = h.state().playlist[0].clone();
    eprintln!("FLOW audio playing after {started:?}: {} / {}", song.title, song.artist);
    while h.state().spectrum_window.video.is_none() && t.elapsed().as_secs() < 180 {
        wait(&mut h, 500);
    }
    eprintln!("FLOW video showing after {:?}", t.elapsed());
    assert!(h.state().spectrum_window.video.is_some(), "the video came");
    assert!(crate::youtube::cover_path(&song.path).is_file(), "the cover came");
    click(&mut h, "Listen");
    wait(&mut h, 1500);
    shot(&mut h, &dir, "flow1-listen");
    // Its lyrics were looked up unasked; at 35 s the song is singing.
    let lines = {
        let app = h.state_mut();
        let t = app.playlist[0].clone();
        let tref = crate::lyrics::ui::TrackRef { path: &t.path, title: &t.title, artist: &t.artist, album: &t.album, duration: t.duration };
        app.lyrics_window.nearby_lines(tref, std::time::Duration::from_secs(35), 0, 0)
    };
    eprintln!("FLOW lyrics at 35 s: {lines:?}");
    assert!(!lines.is_empty(), "lyrics found unasked");
    click(&mut h, "Watch");
    wait(&mut h, 1000);
    shot(&mut h, &dir, "flow2-theater");
    h.state_mut().stop();
    h.run_steps(1);
    for n in ["player.json", "youtube.json", "last_playlist.json", "appearance.json"] {
        let _ = std::fs::remove_file(m.join(n));
    }
}

/// Japanese lyrics on Listen: Original, Furigana, Romanization and Both
/// beside Follow, as in the lyrics window.
#[test]
#[ignore]
fn screens161_listen_readings() {
    let dir = PathBuf::from(std::env::var("MOOSIK_SHOTS").expect("set MOOSIK_SHOTS"));
    std::fs::create_dir_all(&dir).unwrap();
    let songs = std::env::temp_dir().join(format!("moosik-jp161-{}", std::process::id()));
    std::fs::create_dir_all(&songs).unwrap();
    let song = songs.join("song.flac");
    std::fs::copy(std::env::var("MOOSIK_FLAC").expect("set MOOSIK_FLAC"), &song).unwrap();
    let words = ["君の名前を呼んだ", "夜空に星が光る", "明日また会えるかな", "心の中で歌っている"];
    let lrc: String = (0..40).map(|s| format!("[00:{:02}.00]{}\n", s * 2, words[s % 4])).collect();
    std::fs::write(songs.join("song.lrc"), lrc).unwrap();
    let m = crate::moosik_dir();
    std::fs::create_dir_all(&m).unwrap();
    std::fs::write(m.join("last_playlist.json"), serde_json::to_string(&[&song]).unwrap()).unwrap();
    std::fs::write(m.join("player.json"), r#"{"volume":0.0,"loop_mode":"Sequential"}"#).unwrap();
    let mut h = Harness::builder().with_size(egui::vec2(1300.0, 860.0)).wgpu().build_eframe(|cc| crate::MoosikApp::new(cc));
    h.run_steps(5);
    click(&mut h, "Play");
    wait(&mut h, 1500);
    click(&mut h, "Both");
    wait(&mut h, 300);
    shot(&mut h, &dir, "jp1-listen-both");
    click(&mut h, "Furigana");
    wait(&mut h, 300);
    shot(&mut h, &dir, "jp2-listen-furigana");
    h.state_mut().stop();
    h.run_steps(1);
    for n in ["last_playlist.json", "player.json"] {
        let _ = std::fs::remove_file(m.join(n));
    }
    let _ = std::fs::remove_dir_all(&songs);
}

/// Listen without lyrics: Find lyrics on the page itself, and what it
/// found (network for the look). MOOSIK_FLAC = a tagged song.
#[test]
#[ignore]
fn screens161_listen_find_lyrics() {
    let dir = PathBuf::from(std::env::var("MOOSIK_SHOTS").expect("set MOOSIK_SHOTS"));
    std::fs::create_dir_all(&dir).unwrap();
    let songs = std::env::temp_dir().join(format!("moosik-lyr161-{}", std::process::id()));
    std::fs::create_dir_all(&songs).unwrap();
    let song = songs.join("song.flac");
    std::fs::copy(std::env::var("MOOSIK_FLAC").expect("set MOOSIK_FLAC"), &song).unwrap();
    let m = crate::moosik_dir();
    std::fs::create_dir_all(&m).unwrap();
    std::fs::write(m.join("last_playlist.json"), serde_json::to_string(&[&song]).unwrap()).unwrap();
    std::fs::write(m.join("player.json"), r#"{"volume":0.0,"loop_mode":"Sequential"}"#).unwrap();
    let mut h = Harness::builder().with_size(egui::vec2(1300.0, 860.0)).wgpu().build_eframe(|cc| crate::MoosikApp::new(cc));
    // Left to the button for now.
    h.state_mut().appearance.lyrics_auto = false;
    h.run_steps(5);
    click(&mut h, "Play");
    click(&mut h, "Listen");
    wait(&mut h, 1500);
    shot(&mut h, &dir, "lyr1-listen-no-lyrics");
    let label = ["Find lyrics", "🔍 Find lyrics"].into_iter().find(|l| h.query_by_label(l).is_some()).expect("Find lyrics on Listen");
    click(&mut h, label);
    let t = std::time::Instant::now();
    wait(&mut h, 300);
    shot(&mut h, &dir, "lyr2-looking");
    while h.query_by_label("Looking for lyrics…").is_some() && t.elapsed().as_secs() < 30 {
        wait(&mut h, 250);
    }
    eprintln!("LYRICS looked for {:?}", t.elapsed());
    wait(&mut h, 500);
    shot(&mut h, &dir, "lyr3-found");
    h.state_mut().stop();
    h.run_steps(1);
    for n in ["last_playlist.json", "player.json", "appearance.json"] {
        let _ = std::fs::remove_file(m.join(n));
    }
    let _ = std::fs::remove_dir_all(&songs);
}

/// Playing while it arrives (network): the size picked beside Audio +
/// video is the one fetched, the song plays before its audio is all in,
/// and the video shows while it still arrives.
#[test]
#[ignore]
fn screens161_stream_flow() {
    let dir = PathBuf::from(std::env::var("MOOSIK_SHOTS").expect("set MOOSIK_SHOTS"));
    std::fs::create_dir_all(&dir).unwrap();
    tools_into_profile(&PathBuf::from(std::env::var("MOOSIK_YT_TOOLS").expect("set MOOSIK_YT_TOOLS")));
    let m = crate::moosik_dir();
    let cache = crate::youtube::cache_dir(&m);
    let _ = std::fs::remove_dir_all(&cache);
    std::fs::write(m.join("player.json"), r#"{"volume":0.0,"loop_mode":"Sequential"}"#).unwrap();
    let mut h = Harness::builder().with_size(egui::vec2(1300.0, 860.0)).wgpu().build_eframe(|cc| crate::MoosikApp::new(cc));
    h.state_mut().appearance.youtube_video = true;
    h.state_mut().appearance.video_height = 720;
    h.run_steps(5);
    click(&mut h, "Library");
    click(&mut h, "YouTube");
    h.run_steps(5);
    // The size, beside Audio + video.
    h.get_by_role_and_label(egui::accesskit::Role::ComboBox, "Video size").click();
    h.run_steps(2);
    shot(&mut h, &dir, "st0-quality-menu");
    click(&mut h, "1080p");
    assert_eq!(h.state().appearance.video_height, 1080);
    let field = h.get_by_role(egui::accesskit::Role::TextInput);
    field.focus();
    field.type_text("https://www.youtube.com/watch?v=dQw4w9WgXcQ");
    h.run_steps(2);
    h.press_key(egui::Key::Enter);
    let t = std::time::Instant::now();
    while h.state().shell.yt_hits() == 0 && t.elapsed().as_secs() < 30 {
        wait(&mut h, 100);
    }
    let hit = h.state().shell.yt_hit(0).expect("the link is that one video");
    let t = std::time::Instant::now();
    h.state_mut().youtube_fetch(&hit, crate::shell::youtube_ui::Then::Play);
    while h.state().current_index.is_none() && t.elapsed().as_secs() < 90 {
        wait(&mut h, 50);
    }
    let playing = t.elapsed();
    assert!(h.state().current_index.is_some(), "it plays");
    let song = h.state().playlist[0].path.clone();
    let arriving = crate::youtube::stream::downloading(&song);
    eprintln!("STREAM playing after {playing:?}, audio still arriving: {arriving}");
    while h.state().spectrum_window.video.is_none() && t.elapsed().as_secs() < 180 {
        wait(&mut h, 50);
    }
    let video = song.with_extension("video.webm");
    eprintln!("STREAM video showing after {:?}, still arriving: {}", t.elapsed(), crate::youtube::stream::downloading(&video));
    assert!(h.state().spectrum_window.video.is_some(), "the video came");
    click(&mut h, "Listen");
    wait(&mut h, 1500);
    shot(&mut h, &dir, "st1-listen-while-arriving");
    while crate::youtube::stream::downloading(&video) && t.elapsed().as_secs() < 400 {
        wait(&mut h, 250);
    }
    eprintln!("STREAM video all in after {:?}: {} pixels tall", t.elapsed(), crate::video::webm_height(&video).unwrap_or(0));
    assert_eq!(crate::video::webm_height(&video), Some(1080), "the size picked");
    h.state_mut().stop();
    h.run_steps(1);
    for n in ["player.json", "youtube.json", "last_playlist.json", "appearance.json"] {
        let _ = std::fs::remove_file(m.join(n));
    }
}
