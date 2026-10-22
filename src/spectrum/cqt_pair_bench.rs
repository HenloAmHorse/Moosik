//! Explicit, ignored end-to-end CQT/ASLT screen: one arm per process.
//!
//! Required: MOOSIK_GPU=0 and MOOSIK_CQT_PAIR_ARM=cqt12|cqt48|cqt50|cqt192|
//! aslt1|aslt9. Optional: MOOSIK_CQT_PAIR_SECONDS=1..120 (default 30),
//! MOOSIK_CQT_PAIR_SR=44100|48000|96000 (default 48000),
//! MOOSIK_CQT_PAIR_BARS=16..1024 (default 1024),
//! MOOSIK_CQT_PAIR_WORKERS=1..available CPUs (default 6),
//! MOOSIK_CQT_PAIR_MODE=mono|joint (default joint).
//!
//! Run only this exact test, in release, with --ignored --nocapture and
//! --test-threads=1. Capture the CQT_PAIR JSON line from stdout. All source
//! audio, cache files and calibration are private temporary fixtures. No
//! external audio file, owner profile, audio device or network is used.
//!
//! This compares selectable settings, NOT equal response/quality: CQT uses a
//! log grid/full cosine-sum windows; ASLT uses Blend(0.5), q_ratio=2 and a
//! 0.25 s maximum-window request. ASLT's 2x is Q ratio, not twice the columns.
//! The timer includes production decode/statistics/analysis/encode/publication,
//! but excludes fixture/pool setup and the separate disk readback gate. Process
//! peak counters are lifetime peaks, not resettable analysis-only allocations.

use super::*;
use std::io::{BufWriter, Write};

#[derive(serde::Serialize)]
struct Resources {
    cpu_seconds: Option<f64>,
    commit_bytes: Option<u64>,
    peak_commit_bytes: Option<u64>,
    working_set_bytes: Option<u64>,
    peak_working_set_bytes: Option<u64>,
}

#[cfg(windows)]
fn resources() -> Resources {
    #[repr(C)]
    #[derive(Default)]
    struct Pmc {
        cb: u32,
        page_faults: u32,
        peak_ws: usize,
        ws: usize,
        qppp: usize,
        qpp: usize,
        qpnpp: usize,
        qpn: usize,
        pagefile: usize,
        peak_pagefile: usize,
        private: usize,
    }
    #[link(name = "kernel32")]
    unsafe extern "system" {
        fn GetCurrentProcess() -> isize;
        fn K32GetProcessMemoryInfo(process: isize, counters: *mut Pmc, cb: u32) -> i32;
        fn GetProcessTimes(
            process: isize,
            c: *mut u64,
            e: *mut u64,
            k: *mut u64,
            u: *mut u64,
        ) -> i32;
    }
    let mut counters = Pmc {
        cb: std::mem::size_of::<Pmc>() as u32,
        ..Default::default()
    };
    let (mut created, mut exited, mut kernel, mut user) = (0u64, 0u64, 0u64, 0u64);
    let cb = counters.cb;
    // SAFETY: this process's pseudo-handle and correctly sized writable output
    // structures; the APIs do not retain any pointer.
    let (memory_ok, time_ok) = unsafe {
        let process = GetCurrentProcess();
        (
            K32GetProcessMemoryInfo(process, &mut counters, cb),
            GetProcessTimes(process, &mut created, &mut exited, &mut kernel, &mut user),
        )
    };
    assert_ne!(memory_ok, 0, "GetProcessMemoryInfo failed");
    assert_ne!(time_ok, 0, "GetProcessTimes failed");
    Resources {
        cpu_seconds: Some((kernel + user) as f64 * 100e-9),
        commit_bytes: Some(counters.pagefile as u64),
        peak_commit_bytes: Some(counters.peak_pagefile as u64),
        working_set_bytes: Some(counters.ws as u64),
        peak_working_set_bytes: Some(counters.peak_ws as u64),
    }
}

#[cfg(not(windows))]
fn resources() -> Resources {
    Resources {
        cpu_seconds: None,
        commit_bytes: None,
        peak_commit_bytes: None,
        working_set_bytes: None,
        peak_working_set_bytes: None,
    }
}

/// This guard owns one newly created, resolved direct child of the OS temp
/// directory. It never accepts an environment-provided deletion target.
struct Fixture {
    root: PathBuf,
    temp: PathBuf,
}

impl Fixture {
    fn new() -> Self {
        let temp = std::fs::canonicalize(std::env::temp_dir()).expect("resolve OS temp directory");
        let nonce = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .expect("clock before epoch")
            .as_nanos();
        let requested = temp.join(format!("moosik_cqt_pair_{}_{nonce:x}", std::process::id()));
        std::fs::create_dir(&requested).expect("create a fresh, never reused private fixture");
        let root = std::fs::canonicalize(&requested).expect("resolve private fixture");
        assert_eq!(root.parent(), Some(temp.as_path()));
        Self { root, temp }
    }

    fn retire(&mut self) {
        if self.root.exists() {
            assert_eq!(self.root.parent(), Some(self.temp.as_path()));
            std::fs::remove_dir_all(&self.root).expect("retire exact owned fixture directory");
        }
    }
}

impl Drop for Fixture {
    fn drop(&mut self) {
        if self.root.exists()
            && self.root.parent() == Some(self.temp.as_path())
            && let Err(error) = std::fs::remove_dir_all(&self.root)
        {
            eprintln!(
                "CQT_PAIR private fixture remains at {}: {error}",
                self.root.display()
            );
        }
    }
}

fn env_number(name: &str, default: u32) -> u32 {
    match std::env::var(name) {
        Ok(value) => value
            .parse()
            .unwrap_or_else(|_| panic!("{name} must be an integer")),
        Err(std::env::VarError::NotPresent) => default,
        Err(error) => panic!("{name}: {error}"),
    }
}

/// Stereo 16-bit PCM RIFF with deterministic tones, gated tones, chirp and
/// independent noise. Generation/serialization are outside the analysis timer.
fn write_fixture(path: &Path, sr: u32, seconds: u32) -> (u64, String) {
    let frames = u64::from(sr) * u64::from(seconds);
    let data_bytes = u32::try_from(frames * 4).expect("bounded WAV size");
    let file = std::fs::OpenOptions::new()
        .write(true)
        .create_new(true)
        .open(path)
        .unwrap();
    let mut output = BufWriter::new(file);
    output.write_all(b"RIFF").unwrap();
    output.write_all(&(36 + data_bytes).to_le_bytes()).unwrap();
    output.write_all(b"WAVEfmt ").unwrap();
    output.write_all(&16u32.to_le_bytes()).unwrap();
    output.write_all(&1u16.to_le_bytes()).unwrap(); // PCM
    output.write_all(&2u16.to_le_bytes()).unwrap();
    output.write_all(&sr.to_le_bytes()).unwrap();
    output.write_all(&(sr * 4).to_le_bytes()).unwrap();
    output.write_all(&4u16.to_le_bytes()).unwrap();
    output.write_all(&16u16.to_le_bytes()).unwrap();
    output.write_all(b"data").unwrap();
    output.write_all(&data_bytes.to_le_bytes()).unwrap();
    let mut seeds = [0x155_cafe_9123_4567u64, 0x9b17_558d_3141_5926u64];
    for sample in 0..frames {
        let time = sample as f64 / f64::from(sr);
        for (channel, seed) in seeds.iter_mut().enumerate() {
            *seed ^= *seed << 13;
            *seed ^= *seed >> 7;
            *seed ^= *seed << 17;
            let noise = ((*seed >> 11) as f64 / (1u64 << 53) as f64 * 2.0 - 1.0) * 0.08;
            let phase = channel as f64 * 0.91;
            let gate = if (time % 1.0) < 0.35 { 1.0 } else { 0.0 };
            let chirp = 90.0 * time + 0.5 * (2700.0 / f64::from(seconds)) * time * time;
            let value = 0.26 * (std::f64::consts::TAU * 55.0 * time + phase).sin()
                + 0.19 * (std::f64::consts::TAU * (440.0 + channel as f64 * 7.0) * time).cos()
                + 0.15 * gate * (std::f64::consts::TAU * 5300.0 * time + phase).sin()
                + 0.13 * (std::f64::consts::TAU * chirp + phase).sin()
                + noise;
            let value = (value.clamp(-1.0, 1.0) * f64::from(i16::MAX)).round() as i16;
            output.write_all(&value.to_le_bytes()).unwrap();
        }
    }
    output.flush().unwrap();
    drop(output);
    let raw = std::fs::read(path).unwrap();
    assert_eq!(raw.len() as u64, 44 + u64::from(data_bytes));
    (frames, store::hex(&store::sha256(&[&raw])))
}

fn plane_digest(plane: &cache::PreFrames) -> String {
    let mut digest = ring::digest::Context::new(&ring::digest::SHA256);
    let mut bytes = Vec::with_capacity(plane.bars() * 2);
    for frame in 0..plane.len() {
        bytes.clear();
        for &value in plane.row(frame).expect("complete memory row") {
            bytes.extend_from_slice(&value.to_le_bytes());
        }
        digest.update(&bytes);
    }
    store::hex(digest.finish().as_ref())
}

fn same_plane(memory: &cache::PreFrames, disk: &cache::PreFrames) {
    assert_eq!(
        (memory.len(), memory.bars(), memory.max_code()),
        (disk.len(), disk.bars(), disk.max_code())
    );
    for frame in 0..memory.len() {
        assert_eq!(
            memory.row(frame),
            disk.row(frame),
            "disk row differs at {frame}"
        );
    }
}

#[test]
#[ignore = "manual isolated production-entry benchmark; one arm per exclusive release-test process"]
fn production_pair() {
    assert_eq!(
        std::env::var("MOOSIK_GPU").ok().as_deref(),
        Some("0"),
        "MOOSIK_GPU=0 is required"
    );
    let arm = std::env::var("MOOSIK_CQT_PAIR_ARM").expect("select exactly one benchmark arm");
    let seconds = env_number("MOOSIK_CQT_PAIR_SECONDS", 30);
    let sr = env_number("MOOSIK_CQT_PAIR_SR", 48_000);
    let bars = env_number("MOOSIK_CQT_PAIR_BARS", 1024) as usize;
    let requested_workers = env_number("MOOSIK_CQT_PAIR_WORKERS", 6) as usize;
    let mode = std::env::var("MOOSIK_CQT_PAIR_MODE").unwrap_or_else(|_| "joint".into());
    assert!((1..=120).contains(&seconds));
    assert!([44_100, 48_000, 96_000].contains(&sr));
    assert!((16..=MAX_BAR_COUNT).contains(&bars));
    assert!((1..=std::thread::available_parallelism().unwrap().get()).contains(&requested_workers));
    let joint = match mode.as_str() {
        "joint" => true,
        "mono" => false,
        _ => panic!("MOOSIK_CQT_PAIR_MODE must be mono or joint"),
    };
    let mut aslt_cfg = aslt::AsltConfig {
        q_ratio: 2.0,
        max_window_s: 0.25,
        n_wavelets: 1,
        spread: 0.5,
        scale: freq_scale::FreqScale::Blend(0.5),
    };
    let mapping = match arm.as_str() {
        "aslt1" => BarMappingMode::Superlet,
        "aslt9" => {
            aslt_cfg.n_wavelets = 9;
            BarMappingMode::Superlet
        }
        "cqt12" | "cqt48" | "cqt50" | "cqt192" => BarMappingMode::TrueCqt(cqt::Config {
            bins_per_octave: arm.strip_prefix("cqt").unwrap().parse().unwrap(),
            window: cqt::Window::Blackman,
            frame_rate: 180.0,
            bass_limit_s: None,
        }),
        _ => panic!("unknown benchmark arm {arm}"),
    };
    let _calibration = gpu_calib::isolated_test_env("v155_pair");
    let calibration_path = gpu_calib::path_for_test();
    assert!(
        calibration_path.starts_with(std::env::temp_dir()),
        "private calibration required"
    );
    gpu_calib::set_mode(gpu_calib::GpuMode::Off);
    gpu_calib::set_workers(requested_workers);
    // Prewarm the actual owner-selected pool, not Rayon's implicit global one.
    // Pool creation is outside the timer and is explicitly part of this scope.
    let (actual_workers, in_pool) = gpu_calib::install(|| {
        (
            rayon::current_num_threads(),
            rayon::current_thread_index().is_some(),
        )
    });
    assert!(in_pool, "selected analysis pool did not start");
    assert_eq!(
        actual_workers, requested_workers,
        "worker selection silently changed"
    );
    let mut fixture = Fixture::new();
    let wav = fixture.root.join("synthetic.wav");
    let (input_frames, input_sha256) = write_fixture(&wav, sr, seconds);
    let unit = fixture.root.join("cache").join("track.spectrumcache");
    let cache_store = test_store(&unit).expect("private store must open");
    assert!(
        published_heads(&unit).is_empty(),
        "fresh fixture unexpectedly has a cache"
    );
    let progress = Arc::new(AtomicUsize::new(0));
    let abort = Arc::new(AtomicBool::new(false));
    let eta = Arc::new(AtomicUsize::new(usize::MAX));
    let before = resources();
    let start = Instant::now();
    let message = preprocess_file(
        &wav,
        &unit,
        Some(cache_store.clone()),
        true,
        0,
        bars,
        &progress,
        DEFAULT_FFT_SIZE,
        16,
        0.875,
        &WindowFn::Hann,
        20.0,
        24_000.0,
        &InterpolationMode::None,
        &mapping,
        88_200,
        &aslt_cfg,
        180.0,
        &abort,
        &eta,
        if joint {
            StereoRequest::Joint
        } else {
            StereoRequest::MixOnly
        },
    );
    let wall_seconds = start.elapsed().as_secs_f64();
    let after = resources();
    assert!(wall_seconds.is_finite() && wall_seconds > 0.0);
    let (frames, stereo, frame_rate, meta) = match message {
        PreMessage::Done {
            frames,
            stereo,
            frame_rate,
            meta,
            ..
        } => (frames, stereo, frame_rate, meta),
        PreMessage::Error(error) => {
            panic!("production pre-process refused this arm; not a valid timing: {error}")
        }
        PreMessage::Aborted => panic!("production pre-process aborted; not a valid timing"),
    };
    assert_eq!(progress.load(Ordering::Relaxed), 100);
    assert!(!frames.is_empty());
    assert_eq!(frames.bars(), bars);
    assert!(frame_rate.is_finite() && frame_rate > 0.0);
    let (expected_frames, expected_rate) = if let BarMappingMode::TrueCqt(cfg) = &mapping {
        (
            cqt::frame_count(input_frames as usize, sr, cfg.frame_rate).unwrap(),
            f64::from(cfg.frame_rate),
        )
    } else {
        let hop = aslt::hop_for_fps(sr, 180.0);
        (
            aslt::frame_count(input_frames as usize, hop),
            f64::from(sr) / hop as f64,
        )
    };
    assert_eq!(
        frames.len(),
        expected_frames,
        "production frame count changed"
    );
    assert_eq!(frame_rate, expected_rate, "production frame clock changed");
    let published = meta
        .published
        .as_ref()
        .expect("analysis without successful publication is not a valid cache benchmark");
    assert_eq!(published.has_lr, joint);
    assert!(published.pin.is_some());
    let heads = published_heads(&unit);
    assert_eq!(
        heads.len(),
        1,
        "exactly one private generation must be published"
    );
    let (head, (mix_path, lr_path)) = &heads[0];
    assert_eq!(
        head.file_stem().and_then(|v| v.to_str()),
        Some(published.stem.as_str())
    );
    let head_bytes = std::fs::metadata(head).unwrap().len();
    let mix_raw = std::fs::read(mix_path).unwrap();
    let (mix_envelope, descriptor, mix_payload) =
        store::decode_envelope(&mix_raw).expect("valid published Mix envelope");
    assert_eq!(mix_envelope.generation, published.generation);
    assert_eq!(mix_envelope.descriptor, published.envelope.descriptor);
    assert_eq!(mix_envelope.decoded, published.envelope.decoded);
    let decoded_mix =
        cache::decode_with(mix_payload, bars, MAX_BAR_COUNT, cache::Limits::default())
            .expect("decode actual published Mix");
    same_plane(&frames, &decoded_mix);
    let mut hashes = vec![plane_digest(&frames)];
    let (lr_file_bytes, lr_payload_bytes) = if joint {
        let memory = stereo.as_ref().expect("joint run must return channels");
        let raw = std::fs::read(
            lr_path
                .as_ref()
                .expect("joint cache must publish an L/R file"),
        )
        .unwrap();
        let (envelope, lr_descriptor, payload) =
            store::decode_envelope(&raw).expect("valid published L/R envelope");
        assert_eq!(envelope.generation, published.generation);
        assert_eq!(envelope.descriptor, mix_envelope.descriptor);
        assert_eq!(envelope.decoded, mix_envelope.decoded);
        assert_eq!(envelope.mix, mix_envelope.mix);
        assert_eq!(lr_descriptor, descriptor);
        let decoded = stereo::decode_with(
            payload,
            meta.identity,
            &decoded_mix,
            MAX_BAR_COUNT,
            cache::Limits::default(),
        )
        .expect("decode actual bound L/R payload");
        same_plane(memory.left(), decoded.left());
        same_plane(memory.right(), decoded.right());
        hashes.push(plane_digest(memory.left()));
        hashes.push(plane_digest(memory.right()));
        (raw.len() as u64, payload.len() as u64)
    } else {
        assert!(stereo.is_none());
        assert!(lr_path.is_none());
        (0, 0)
    };
    assert!(head_bytes > 0 && !mix_payload.is_empty());
    assert!(!joint || lr_payload_bytes > 0);
    let namespace = store_dir_for(&unit);
    let namespace_bytes: u64 = std::fs::read_dir(&namespace)
        .unwrap()
        .map(|entry| {
            let metadata = entry.unwrap().metadata().unwrap();
            assert!(
                metadata.is_file(),
                "unexpected directory in this private namespace"
            );
            metadata.len()
        })
        .sum();
    let payload_file_bytes = mix_raw.len() as u64 + lr_file_bytes;
    assert!(namespace_bytes >= payload_file_bytes + head_bytes);
    let cpu_seconds = before
        .cpu_seconds
        .zip(after.cpu_seconds)
        .map(|(a, b)| b - a);
    assert!(cpu_seconds.is_none_or(|v| v.is_finite() && v >= 0.0));
    let record = serde_json::json!({
        "schema": "moosik-v155-pair-v1",
        "build_label": env!("MOOSIK_BUILD_LABEL"),
        "runner_source_revision": std::env::var("MOOSIK_CQT_PAIR_SOURCE_REV").ok(),
        "arm": arm,
        "mode": mode,
        "comparison": "selectable-settings-not-matched-response",
        "input_sha256": input_sha256,
        "input_sample_rate": sr,
        "input_frames": input_frames,
        "audio_seconds": input_frames as f64 / f64::from(sr),
        "bars": bars,
        "nominal_frame_rate": 180.0,
        "actual_frame_rate": frame_rate,
        "output_frames": frames.len(),
        "selected_workers": requested_workers,
        "observed_pool_workers": actual_workers,
        "gpu_disabled": true,
        "mapping": format!("{mapping:?}"),
        "aslt_config": format!("{aslt_cfg:?}"),
        "descriptor": descriptor,
        "wall_seconds": wall_seconds,
        "cpu_seconds": cpu_seconds,
        "rtf": wall_seconds / (input_frames as f64 / f64::from(sr)),
        "resources_before": before,
        "resources_after": after,
        "process_peak_scope": "lifetime-including-setup-not-analysis-only",
        "timer_scope": "production-decode-statistics-analysis-encode-publication; excludes-fixture-pool-readback",
        "mix_file_bytes": mix_raw.len(),
        "mix_payload_bytes": mix_payload.len(),
        "lr_file_bytes": lr_file_bytes,
        "lr_payload_bytes": lr_payload_bytes,
        "head_bytes": head_bytes,
        "namespace_bytes": namespace_bytes,
        "coordination_bytes": namespace_bytes - payload_file_bytes - head_bytes,
        "code_plane_sha256": hashes,
        "disk_readback_all_codes_equal": true,
        "work": format!("{:?}", meta.work),
    });
    // Release every generation pin/Store before deleting the exact fixture.
    drop(meta);
    drop(stereo);
    drop(frames);
    drop(cache_store);
    fixture.retire();
    println!("CQT_PAIR {record}");
}
