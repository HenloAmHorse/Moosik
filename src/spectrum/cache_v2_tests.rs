//! The analysis cache store through production entry points (owner decisions
//! D1–D8, `store.rs`). Every schedule here drives the real request → worker
//! (`preprocess_file`) → `Done` → installation → readback path on private
//! fixtures in the test's own temporary folder; interleavings are forced by
//! named seams (`cache::fault` hooks on the worker, `store::seam` points in the
//! store), never by sleeps. The store's own protocol, including a publisher
//! process killed at each boundary, is in `store_tests.rs`.
//!
//! Retained limitations are tested as such, not repaired: an edit confined to
//! unsampled bytes is undetected (`store_tests`), the check-to-commit window is
//! open (D2), and nothing here claims power-loss freshness (D7).

use super::*;

const DSD64: u32 = 2_822_400;

fn store_of(w: &mut SpectrumWindow) -> Arc<store::Store> {
    w.analyzer.store().expect("the cache store did not open")
}

fn shown_from_store(w: &SpectrumWindow) -> bool {
    w.analyzer.pre_generation_installed().is_some()
}

/// A Mix-and-channels WAV whose left and right are two different signals.
fn write_lr_wav(path: &Path, left_hz: f32, right_hz: f32) {
    let n = SR as usize;
    let mut data = Vec::with_capacity(n * 4);
    for i in 0..n {
        let t = i as f32 / SR as f32;
        let l = 0.3 * (std::f32::consts::TAU * left_hz * t).sin();
        let r = 0.3 * (std::f32::consts::TAU * right_hz * t).sin();
        for s in [l, r] {
            data.extend(((s * 32_767.0) as i16).to_le_bytes());
        }
    }
    let mut out = Vec::with_capacity(44 + data.len());
    out.extend(b"RIFF");
    out.extend(((36 + data.len()) as u32).to_le_bytes());
    out.extend(b"WAVEfmt ");
    out.extend(16u32.to_le_bytes());
    out.extend(1u16.to_le_bytes());
    out.extend(2u16.to_le_bytes());
    out.extend(SR.to_le_bytes());
    out.extend((SR * 4).to_le_bytes());
    out.extend(4u16.to_le_bytes());
    out.extend(16u16.to_le_bytes());
    out.extend(b"data");
    out.extend((data.len() as u32).to_le_bytes());
    out.extend(data);
    std::fs::write(path, out).unwrap();
}

/// A stereo DSDIFF of `secs` at DSD64: first-order sigma-delta of a tone per
/// channel, MSB-first, byte-interleaved, as DSDIFF stores it.
fn write_dff(path: &Path, secs: f64, f_l: f64, f_r: f64) {
    let bytes = (DSD64 as f64 * secs / 8.0) as usize;
    let modulate = |f: f64| -> Vec<u8> {
        let (mut acc, mut y) = (0.0f64, 0.0f64);
        (0..bytes)
            .map(|i| {
                let mut b = 0u8;
                for k in 0..8 {
                    let t = (i * 8 + k) as f64 / DSD64 as f64;
                    acc += 0.4 * (std::f64::consts::TAU * f * t).sin() - y;
                    y = if acc >= 0.0 { 1.0 } else { -1.0 };
                    b |= ((y > 0.0) as u8) << (7 - k);
                }
                b
            })
            .collect()
    };
    let (l, r) = (modulate(f_l), modulate(f_r));
    let audio: Vec<u8> = l.iter().zip(&r).flat_map(|(a, b)| [*a, *b]).collect();
    std::fs::write(path, crate::dsd::tests::make_dff(2, DSD64, b"DSD ", &audio, None)).unwrap();
}

/// Play `track` in a fresh window on `cache` and let it settle. DSD tracks are
/// analysed at 176.4 kHz.
fn play(cache: &Path, track: &Path, stereo_on: bool) -> SpectrumWindow {
    let mut w = window(cache, 1.0);
    w.analyzer.pre_stereo_enabled = stereo_on;
    start(&mut w, track);
    assert!(settle(&mut w), "setup: {} never settled", track.display());
    w
}

fn start(w: &mut SpectrumWindow, track: &Path) {
    if crate::dsd::is_dsd_path(track) {
        w.analyzer.dsd_rate = 176_400;
        w.dsd_fallback_rate = 176_400;
        w.on_play_live(track, 176_400, Some(176_400));
    } else {
        w.on_play(track, SR);
    }
}

/// The channels of what is installed, read the way a channel view reads
/// them: lazily, from the installed generation.
fn channels_of(w: &mut SpectrumWindow) -> stereo::Stereo {
    w.analyzer.request_pre_channels();
    let deadline = Instant::now() + Duration::from_secs(60);
    while !w.analyzer.pre_channels_resident() {
        assert!(Instant::now() < deadline, "the lazy channel read never landed: {:?}", w.analyzer.pre_channels().note());
        w.analyzer.poll_pre_channels();
        w.analyzer.request_pre_channels();
        std::thread::sleep(Duration::from_millis(2));
    }
    w.analyzer.pre_stereo().cloned().unwrap()
}

fn generations(w: &mut SpectrumWindow, track: &Path) -> Vec<u128> {
    let stem = identity::stem_of_unit(&key(w, track)).unwrap();
    store_of(w).generations(&stem)
}

// ── reuse, replacement, identity ────────────────────────────────────────────

/// PCM, DSF and DSDIFF: a fresh analysis is published and is an exact hit for
/// a new window; the same path replaced by other audio misses, is analysed
/// again, and the superseded generation is collected once nothing shows it.
#[test]
fn exact_reuse_follows_the_source_for_pcm_dsf_and_dff() {
    let d = Dir::new("v2_reuse");
    let wav = tracks(&d, 1).remove(0);
    let dsf = d.0.join("t.dsf");
    crate::spectrum::dsd_rate_tests::write_dsf(&dsf, DSD64, 0.3, 700.0, 2_500.0);
    let dff = d.0.join("t.dff");
    write_dff(&dff, 0.3, 700.0, 2_500.0);
    for (track, other) in [
        (&wav, Box::new(|p: &Path| write_wav(p, &[900.0, 3300.0], 11)) as Box<dyn Fn(&Path)>),
        (&dsf, Box::new(|p: &Path| crate::spectrum::dsd_rate_tests::write_dsf(p, DSD64, 0.3, 1_300.0, 4_100.0))),
        (&dff, Box::new(|p: &Path| write_dff(p, 0.3, 1_300.0, 4_100.0))),
    ] {
        let what = track.extension().unwrap().to_string_lossy().to_string();
        let cache = d.0.join(format!("cache_{what}"));
        let w1 = play(&cache, track, false);
        assert_eq!(spawned(&w1), 1, "{what}: setup");
        assert!(shown_from_store(&w1), "{what}: the fresh analysis was not published and installed");
        let first = (*w1.analyzer.pre_frames).clone();
        let w2 = play(&cache, track, false);
        assert_eq!(spawned(&w2), 0, "{what}: an exact hit was analysed again");
        assert!(*w2.analyzer.pre_frames == first, "{what}: the exact hit is not the published analysis");
        assert!(shown_from_store(&w2));
        drop((w1, w2));
        // The same path, other audio.
        other(track);
        let mut w3 = play(&cache, track, false);
        assert_eq!(spawned(&w3), 1, "{what}: a replaced file was taken as an exact hit");
        assert!(*w3.analyzer.pre_frames != first, "{what}: the replaced file shows the old analysis");
        assert_eq!(generations(&mut w3, track).len(), 1, "{what}: the superseded generation was not collected");
    }
}

/// D3, disclosed cost: a change of the last-write time alone — what a tag
/// edit does — is a different source and is analysed again. The audio did not
/// change, so the new analysis is the same rows.
#[test]
fn a_last_write_time_change_alone_is_analysed_again() {
    let d = Dir::new("v2_tag");
    let t = tracks(&d, 1).remove(0);
    let cache = d.0.join("cache");
    let w1 = play(&cache, &t, false);
    let first = (*w1.analyzer.pre_frames).clone();
    drop(w1);
    let later = std::fs::metadata(&t).unwrap().modified().unwrap() + Duration::from_secs(5);
    std::fs::File::options().write(true).open(&t).unwrap().set_modified(later).unwrap();
    let w2 = play(&cache, &t, false);
    assert_eq!(spawned(&w2), 1, "a changed source identity was taken as an exact hit");
    assert!(*w2.analyzer.pre_frames == first, "the same audio analysed differently");
}

/// Swapping left and right keeps the Mix exactly, but it is other audio: the
/// swapped file misses, and its channels are the swapped ones — never the
/// previous file's channels beside an identical Mix.
#[test]
fn a_left_right_swap_with_the_same_mix_is_another_source() {
    let d = Dir::new("v2_swap");
    let t = d.0.join("lr.wav");
    write_lr_wav(&t, 500.0, 3_000.0);
    let cache = d.0.join("cache");
    let mut w1 = play(&cache, &t, true);
    let before = channels_of(&mut w1);
    let mix = (*w1.analyzer.pre_frames).clone();
    drop(w1);
    write_lr_wav(&t, 3_000.0, 500.0);
    let mut w2 = play(&cache, &t, true);
    assert_eq!(spawned(&w2), 1, "the swapped file was taken as an exact hit");
    let after = channels_of(&mut w2);
    assert!(*w2.analyzer.pre_frames == mix, "setup: a swap should keep the Mix");
    assert!(after.left() == before.right() && after.right() == before.left(), "the channels are not the swapped file's");
    // And a fresh window reads the channels from the store, lazily.
    drop(w2);
    let mut w3 = play(&cache, &t, false);
    assert_eq!(spawned(&w3), 0);
    assert!(channels_of(&mut w3) == after, "the lazy read is not the published channels");
}

// ── legacy (L1 + L1a) ───────────────────────────────────────────────────────

/// A legacy analysis at the user's exact settings — a rounded-name file and a
/// 1.5.3 prereview record file — is never an exact hit: the track is
/// analysed, both are offered only as labelled legacy alternatives, choosing
/// one shows it, and neither Trim nor Clear All touches them.
#[test]
fn legacy_files_are_labelled_alternatives_only_and_never_deleted() {
    let d = Dir::new("v2_legacy");
    let t = tracks(&d, 1).remove(0);
    let cache = d.0.join("cache");
    let mut w = window(&cache, 1.0);
    let mine = w.analyzer.aslt_cfg.clone();
    let (_, rounded) = w.legacy_key_with(&t, &mine);
    let legacy_rows = cache::PreFrames::from_analysis(&vec![vec![0.5f32; BARS]; 40]);
    cache::write_atomic(&rounded, &legacy_rows.to_v4()).unwrap();
    let s = w.settings_with(&t, &mine);
    let record_file = s.legacy_record_key(&cache);
    let record_rows = cache::PreFrames::from_analysis(&vec![vec![0.25f32; BARS]; 40]);
    cache::write_atomic(&record_file, &record_rows.to_v4_with(s.record(SR).as_bytes())).unwrap();
    let before: Vec<(PathBuf, Vec<u8>)> = [&rounded, &record_file].iter().map(|p| (p.to_path_buf(), std::fs::read(p).unwrap())).collect();

    start(&mut w, &t);
    assert!(settle(&mut w));
    assert_eq!(spawned(&w), 1, "a legacy file was used as an exact hit");
    assert!(shown_from_store(&w));
    // While the exact analysis is shown, legacy is not offered.
    w.refresh_alternates();
    assert!(!w.legacy_offered());
    // Without it on screen, both are offered, labelled legacy.
    w.analyzer.clear_pre_frames();
    w.refresh_alternates();
    let n = w.alt_notice.as_ref().expect("no alternates offered");
    let kinds: Vec<(bool, bool)> = n.candidates.iter().map(|(c, _)| (c.legacy, c.nominal)).collect();
    assert!(kinds.contains(&(true, true)) && kinds.contains(&(true, false)), "{kinds:?}");
    w.accept_alternate(&record_file).expect("the record file could not be chosen");
    assert!(*w.analyzer.pre_frames == record_rows);
    assert!(w.alt_in_use.as_ref().is_some_and(|c| c.legacy && !c.nominal));
    // Trim to nothing and Clear All: legacy files are kept, byte for byte.
    w.analyzer.cache_budget_gb = 1e-9;
    let _ = w.trim_now();
    let _ = w.clear_all_caches();
    assert!(w.status_msg.contains("from older versions"), "status {:?}", w.status_msg);
    for (p, b) in &before {
        assert_eq!(std::fs::read(p).unwrap(), *b, "{} was changed or deleted", p.display());
    }
}

// ── source replaced while the job's handle is open ──────────────────────────

/// The job's file A is replaced at its path by B after it was decoded and
/// before publication. The checkpoint reads A's own handle, which is
/// unchanged, so A's analysis is committed — labelled A, which B can never
/// match. Installation compares a fresh snapshot of the path (now B) with A:
/// nothing of A is installed as B, not the rows, not the waveform or
/// statistics. B is then analysed in its own right.
#[test]
fn a_file_replaced_under_an_open_job_is_not_installed_as_the_new_file() {
    let d = Dir::new("v2_handle");
    let t = tracks(&d, 1).remove(0);
    let b = d.0.join("b.wav");
    write_wav(&b, &[1_100.0, 4_400.0], 13);
    let cache = d.0.join("cache");
    let mut w = window(&cache, 1.0);
    let (t2, b2) = (t.clone(), b.clone());
    cache::fault::on(cache::fault::Hook::Committing, move || {
        std::fs::rename(&b2, &t2).expect("setup: replace the path while the job holds A");
    });
    // The hook is per thread: run the worker on this thread's analyser via
    // the real worker function, as production's spawned thread would.
    let settings = w.analyzer.effective_settings(&t);
    let unit = settings.key(&cache).1;
    let st = store_of(&mut w);
    let msg = preprocess_file(
        &t, &unit, Some(st.clone()), false, SR, BARS, &Arc::new(AtomicUsize::new(0)),
        w.analyzer.pre_fft_size, w.analyzer.pad_factor, w.analyzer.overlap, &w.analyzer.window_fn,
        w.analyzer.min_freq, w.analyzer.max_freq, &w.analyzer.interp_mode, &w.analyzer.bar_mapping,
        w.analyzer.dsd_rate, &w.analyzer.aslt_cfg, w.analyzer.pre_fps,
        &Arc::new(AtomicBool::new(false)), &Arc::new(AtomicUsize::new(0)), StereoRequest::MixOnly,
    );
    let PreMessage::Done { ref meta, .. } = msg else { panic!("setup: the job did not finish") };
    assert!(meta.published.is_some(), "A's handle was unchanged, so A's analysis is committed");
    // Hand the completion to the window as its own job's.
    w.analyzer.sample_rate = SR;
    let (tx, rx) = std::sync::mpsc::channel();
    tx.send(msg).unwrap();
    w.analyzer.own_job_for_test(&t, rx, None);
    w.analyzer.try_receive_frames();
    assert!(w.analyzer.pre_frames.is_empty(), "A's analysis was installed for B");
    assert!(w.analyzer.pending_waveform.is_none() && w.analyzer.pending_analysis.is_none(), "A's waveform or statistics were installed for B");
    // B is not an exact hit of A's generation.
    let mut w2 = play(&cache, &t, false);
    assert_eq!(spawned(&w2), 1, "B was served A's analysis");
    assert!(shown_from_store(&w2));
    assert_eq!(generations(&mut w2, &t).len(), 1, "A's generation was not superseded by B's");
}

/// The checkpoint itself fails: nothing is committed or installed, and the
/// failure is reported. No retry is attempted (owner policy).
#[test]
fn a_checker_failure_commits_and_installs_nothing() {
    let d = Dir::new("v2_checker");
    let t = tracks(&d, 1).remove(0);
    let cache = d.0.join("cache");
    let mut w = window(&cache, 1.0);
    let _f = store::seam::fault(&identity::store_dir(&cache), |p, _| (p == "source.check").then_some(std::io::ErrorKind::Other));
    start(&mut w, &t);
    assert!(settle(&mut w));
    assert_eq!(spawned(&w), 1, "a checker failure was retried");
    assert!(w.analyzer.pre_frames.is_empty(), "an unconfirmed analysis was installed");
    assert!(!key(&w, &t).exists(), "an unconfirmed analysis was committed");
    assert!(generations(&mut w, &t).is_empty(), "the rejected generation was left");
}

/// A settings change while a job runs: the job's result is committed for the
/// settings it was made at (and found when they return) but not installed.
#[test]
fn a_superseded_settings_job_publishes_for_its_own_settings_only() {
    let d = Dir::new("v2_settings");
    let t = tracks(&d, 1).remove(0);
    let cache = d.0.join("cache");
    let mut w = window(&cache, 1.0);
    let hold = park(&mut w);
    start(&mut w, &t);
    let first = key(&w, &t);
    set_q(&mut w, 2.0);
    release(&mut w, &hold);
    assert!(settle(&mut w));
    assert!(w.analyzer.pre_frames.is_empty(), "a superseded result was installed");
    assert!(first.exists(), "the superseded job's analysis was not committed for its settings");
    assert!(!key(&w, &t).exists());
    set_q(&mut w, 1.0);
    assert!(!w.analyzer.pre_frames.is_empty() && shown_from_store(&w), "returning to its settings did not find it");
}

// ── readers, publishers, collection ─────────────────────────────────────────

/// A window looks the track up while another analysis of it is between its
/// payload writes and its head commit: it is served the old generation
/// whole, never a hybrid; afterwards the new one.
#[test]
fn a_lookup_during_publication_sees_the_old_analysis_whole() {
    let d = Dir::new("v2_midpub");
    let t = tracks(&d, 1).remove(0);
    let cache = d.0.join("cache");
    let w0 = play(&cache, &t, true);
    let old = ((*w0.analyzer.pre_frames).clone(), w0.analyzer.pre_stereo().cloned());
    drop(w0);
    let mut w1 = window(&cache, 1.0);
    w1.analyzer.pre_stereo_enabled = true;
    let (seen_tx, seen_rx) = std::sync::mpsc::channel::<()>();
    let (go_tx, go_rx) = std::sync::mpsc::channel::<()>();
    let (seen_tx, go_rx) = (Mutex::new(Some(seen_tx)), Mutex::new(go_rx));
    let _h = store::seam::hook(&identity::store_dir(&cache), move |p, _| {
        if p == "pub.mix_written" && let Some(tx) = seen_tx.lock().unwrap().take() {
            tx.send(()).unwrap();
            go_rx.lock().unwrap().recv().unwrap();
        }
    });
    w1.on_play(&t, SR);
    assert!(w1.reanalyze());
    seen_rx.recv_timeout(Duration::from_secs(120)).expect("setup: the publisher never wrote");
    let w2 = play(&cache, &t, true);
    assert_eq!(spawned(&w2), 0, "the old analysis was not served during publication");
    assert!(*w2.analyzer.pre_frames == old.0, "a hybrid or new Mix was served before the commit");
    assert!(w2.analyzer.pre_generation_installed().unwrap().has_lr, "the old pair was not served whole");
    go_tx.send(()).unwrap();
    assert!(settle(&mut w1));
    assert!(shown_from_store(&w1), "the new analysis did not commit");
}

/// Two windows analyse the same track at the same settings at once. One
/// publishes; the other finds the stem's publisher authority taken, shows
/// its result from memory, and commits nothing beside it.
#[test]
fn two_publishers_for_one_analysis_commit_once() {
    let d = Dir::new("v2_twopub");
    let t = tracks(&d, 1).remove(0);
    let cache = d.0.join("cache");
    let (mut a, mut b) = (window(&cache, 1.0), window(&cache, 1.0));
    let (in_tx, in_rx) = std::sync::mpsc::channel::<()>();
    let (go_tx, go_rx) = std::sync::mpsc::channel::<()>();
    let (in_tx, go_rx) = (Mutex::new(Some(in_tx)), Mutex::new(go_rx));
    let _h = store::seam::hook(&identity::store_dir(&cache), move |p, _| {
        if p == "pub.authority" && let Some(tx) = in_tx.lock().unwrap().take() {
            tx.send(()).unwrap();
            go_rx.lock().unwrap().recv().unwrap();
        }
    });
    a.on_play(&t, SR);
    in_rx.recv_timeout(Duration::from_secs(120)).expect("setup: the first publisher never started");
    b.on_play(&t, SR);
    assert!(settle(&mut b));
    assert!(!b.analyzer.pre_frames.is_empty(), "the second window showed nothing");
    assert!(!shown_from_store(&b), "the second publisher committed beside the first");
    go_tx.send(()).unwrap();
    assert!(settle(&mut a));
    assert!(shown_from_store(&a));
    assert_eq!(generations(&mut a, &t).len(), 1, "two generations were committed");
}

/// A completion queued in the channel, not yet taken, while Trim runs with a
/// budget of nothing: its generation is pinned by the publisher's pin the
/// message carries, so it survives, and installation takes the pin over.
#[test]
fn a_queued_completion_survives_a_trim_and_is_installed() {
    let d = Dir::new("v2_queued");
    let t = tracks(&d, 1).remove(0);
    let cache = d.0.join("cache");
    let mut w = window(&cache, 1.0);
    start(&mut w, &t);
    wait_exited_unconsumed(&w);
    w.analyzer.cache_budget_gb = 1e-9;
    let msg = w.trim_now();
    assert!(msg.contains("over"), "the queued generation was not reported as in use: {msg}");
    assert!(key(&w, &t).exists() && !generations(&mut w, &t).is_empty(), "the queued generation was trimmed");
    assert!(settle(&mut w));
    assert!(shown_from_store(&w), "the queued completion was not installed from the store");
    assert!(w.analyzer.pre_generation_installed().unwrap().mix_path().exists());
}

/// A lazy channel read is in flight (held) while the display lets go and a
/// Trim runs: the read holds the generation's pin, so the generation is kept
/// and the read completes from it.
#[test]
fn a_lazy_channel_read_keeps_its_generation_through_a_trim() {
    let d = Dir::new("v2_lazy");
    let t = tracks(&d, 1).remove(0);
    let cache = d.0.join("cache");
    drop(play(&cache, &t, true));
    let mut w = play(&cache, &t, false);
    let inst = w.analyzer.pre_generation_installed().cloned().expect("setup: no store analysis");
    let (lr_path, g) = (inst.lr_path(), inst.generation);
    drop(inst);
    let hold = Arc::new(AtomicBool::new(true));
    w.analyzer.sidecar_hold = Some(hold.clone());
    w.analyzer.request_pre_channels();
    // The display lets go; only the reader's clone pins the generation now.
    w.analyzer.clear_pre_frames();
    w.analyzer.cache_budget_gb = 1e-9;
    let r = w.trim_now();
    assert!(r.contains("in use"), "the read's generation was not reported in use: {r}");
    assert!(lr_path.exists() && generations(&mut w, &t).contains(&g), "a generation being read was deleted");
    hold.store(false, Ordering::Relaxed);
    let st = store_of(&mut w);
    let deadline = Instant::now() + Duration::from_secs(60);
    while w.analyzer.sidecar_outstanding.is_some() {
        assert!(Instant::now() < deadline, "the read never finished");
        w.analyzer.poll_pre_channels();
        std::thread::sleep(Duration::from_millis(2));
    }
    // Released, it goes on the next trim.
    let (_, left) = st.trim(0);
    assert_eq!(left, 0, "the released generation was not collectable");
}

// ── persistence boundaries ──────────────────────────────────────────────────

/// Cancellation at every boundary of the worker's persistence: after the
/// Mix encode, after the channels encode, at the commit point, and inside the
/// store after the pin, each payload, and just before the head. Nothing is
/// committed and nothing but the stable coordination files is left; the
/// previous analysis, when there is one, is still the head.
#[test]
fn cancellation_at_each_persistence_boundary_commits_nothing() {
    let worker_points = [cache::fault::Hook::CacheEncoded, cache::fault::Hook::SidecarEncoded, cache::fault::Hook::Committing];
    let store_points = ["pub.authority", "pub.created", "pub.lr_written", "pub.mix_written", "pub.before_head"];
    let cases: Vec<String> = worker_points.iter().map(|h| format!("{h:?}")).chain(store_points.iter().map(|s| s.to_string())).collect();
    for (i, case) in cases.iter().enumerate() {
        for prior in [false, true] {
            let d = Dir::new(&format!("v2_cancel_{i}_{prior}"));
            let t = tracks(&d, 1).remove(0);
            let cache = d.0.join("cache");
            let v2 = identity::store_dir(&cache);
            let mut w = window(&cache, 1.0);
            w.analyzer.pre_stereo_enabled = true;
            let old_head = if prior {
                start(&mut w, &t);
                assert!(settle(&mut w));
                Some(std::fs::read(key(&w, &t)).unwrap())
            } else {
                None
            };
            let abort = Arc::new(AtomicBool::new(false));
            let flip = abort.clone();
            let _h = if i < worker_points.len() {
                cache::fault::on(worker_points[i], move || flip.store(true, Ordering::Relaxed));
                None
            } else {
                let point = store_points[i - worker_points.len()];
                Some(store::seam::hook(&v2, move |p, _| {
                    if p == point {
                        flip.store(true, Ordering::Relaxed);
                    }
                }))
            };
            let settings = w.analyzer.effective_settings(&t);
            let msg = preprocess_file(
                &t, &settings.key(&cache).1, w.analyzer.store(), true, SR, BARS, &Arc::new(AtomicUsize::new(0)),
                w.analyzer.pre_fft_size, w.analyzer.pad_factor, w.analyzer.overlap, &w.analyzer.window_fn,
                w.analyzer.min_freq, w.analyzer.max_freq, &w.analyzer.interp_mode, &w.analyzer.bar_mapping,
                w.analyzer.dsd_rate, &w.analyzer.aslt_cfg, w.analyzer.pre_fps,
                &abort, &Arc::new(AtomicUsize::new(0)), StereoRequest::Joint,
            );
            let PreMessage::Done { meta, .. } = msg else { panic!("{case}: the analysis did not finish") };
            assert!(meta.published.is_none(), "{case} prior={prior}: a cancelled publication committed");
            drop(meta);
            assert_eq!(std::fs::read(key(&w, &t)).ok(), old_head, "{case} prior={prior}: the head changed");
            let gens = generations(&mut w, &t).len();
            assert_eq!(gens, prior as usize, "{case} prior={prior}: an uncommitted generation was left");
            let tmp: Vec<_> = std::fs::read_dir(&v2).unwrap().flatten().filter(|e| e.file_name().to_string_lossy().contains(".tmp")).collect();
            assert!(tmp.is_empty(), "{case} prior={prior}: a temporary was left");
        }
    }
}

/// A write that fails at each rename (the channels, the Mix, the head) is
/// reported, commits nothing, and leaves no uncommitted generation; the
/// analysis is still shown from memory.
#[test]
fn a_failed_write_at_each_rename_commits_nothing() {
    for nth in 1..=3usize {
        let d = Dir::new(&format!("v2_ioerr_{nth}"));
        let t = tracks(&d, 1).remove(0);
        let cache = d.0.join("cache");
        let mut w = window(&cache, 1.0);
        w.analyzer.pre_stereo_enabled = true;
        let count = Arc::new(AtomicUsize::new(0));
        let c2 = count.clone();
        let _f = store::seam::fault(&identity::store_dir(&cache), move |p, _| {
            (p == "rename" && c2.fetch_add(1, Ordering::SeqCst) + 1 == nth).then_some(std::io::ErrorKind::PermissionDenied)
        });
        start(&mut w, &t);
        assert!(settle(&mut w));
        assert!(!w.analyzer.pre_frames.is_empty(), "rename {nth}: the result was not shown from memory");
        assert!(!shown_from_store(&w), "rename {nth}: a failed write was installed from the store");
        assert!(!key(&w, &t).exists(), "rename {nth}: a failed publication committed a head");
        assert!(generations(&mut w, &t).is_empty(), "rename {nth}: the failed generation was left");
    }
}

/// Trim through the window when Windows refuses an unlink (a reader holds the
/// file without delete sharing): the failure is reported, and its bytes are
/// not counted as reclaimed.
#[cfg(windows)]
#[test]
fn trim_reports_an_os_refused_deletion_without_counting_it() {
    use std::os::windows::fs::OpenOptionsExt;
    let d = Dir::new("v2_osrefuse");
    let t = tracks(&d, 1).remove(0);
    let cache = d.0.join("cache");
    let mut w = play(&cache, &t, false);
    let mix = w.analyzer.pre_generation_installed().unwrap().mix_path();
    w.analyzer.clear_pre_frames();
    let hold = std::fs::OpenOptions::new().read(true).share_mode(0x1 | 0x2).open(&mix).unwrap();
    w.analyzer.cache_budget_gb = 1e-9;
    let r = w.trim_now();
    assert!(r.contains("could not be deleted") && r.contains("freed 0.0 MB"), "{r}");
    assert!(mix.exists());
    drop(hold);
    let r = w.trim_now();
    assert!(!mix.exists(), "{r}");
}

/// Where the filesystem scope is not validated, no control deletes anything
/// and each says so (D6).
#[test]
fn an_unvalidated_cache_folder_is_never_deleted_from() {
    let d = Dir::new("v2_scope");
    let t = tracks(&d, 1).remove(0);
    let cache = d.0.join("cache");
    store::test_scope::force_unvalidated(&identity::store_dir(&cache));
    let mut w = play(&cache, &t, false);
    assert!(shown_from_store(&w), "publication does not need deletion and still works");
    let unit = key(&w, &t);
    assert!(!w.clear_cache());
    assert!(w.status_msg.contains("deletion is disabled"), "{}", w.status_msg);
    assert!(!w.clear_all_caches());
    assert!(w.status_msg.contains("deletion is disabled"), "{}", w.status_msg);
    w.analyzer.cache_budget_gb = 1e-9;
    assert!(w.trim_now().contains("deletion is disabled"));
    assert!(unit.exists());
}
