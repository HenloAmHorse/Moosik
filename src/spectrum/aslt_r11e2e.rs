//! Round 11 measurement harness (ignored; one test per process). The same file
//! in the baseline tree (`4e85cf7` plus this file only) and the candidate tree,
//! so both arms run identical harness code: each binary is its own arm.

/// Process memory and CPU, by the Win32 calls the earlier rounds used.
mod probe {
    #[repr(C)]
    #[derive(Default)]
    struct Pmc {
        cb: u32, page_faults: u32, peak_ws: usize, ws: usize, qppp: usize, qpp: usize,
        qpnpp: usize, qnpp: usize, pagefile: usize, peak_pagefile: usize, private: usize,
    }
    #[link(name = "kernel32")]
    unsafe extern "system" {
        fn GetCurrentProcess() -> isize;
        fn K32GetProcessMemoryInfo(process: isize, counters: *mut Pmc, cb: u32) -> i32;
        fn GetProcessTimes(process: isize, c: *mut u64, e: *mut u64, k: *mut u64, u: *mut u64) -> i32;
    }
    /// Process user + kernel CPU seconds (checked call).
    pub fn cpu_s() -> f64 {
        let (mut c, mut e, mut k, mut u) = (0u64, 0u64, 0u64, 0u64);
        // SAFETY: a pseudo-handle for this process and four valid out-pointers.
        let ok = unsafe { GetProcessTimes(GetCurrentProcess(), &mut c, &mut e, &mut k, &mut u) };
        assert!(ok != 0, "GetProcessTimes failed");
        (k + u) as f64 * 100e-9
    }
    /// (commit, peak commit, working set, peak working set) bytes (checked call).
    pub fn mem() -> (u64, u64, u64, u64) {
        let mut c = Pmc { cb: std::mem::size_of::<Pmc>() as u32, ..Default::default() };
        // SAFETY: a pseudo-handle for this process and a correctly sized struct.
        let ok = unsafe { K32GetProcessMemoryInfo(GetCurrentProcess(), &mut c, c.cb) };
        assert!(ok != 0, "K32GetProcessMemoryInfo failed");
        (c.pagefile as u64, c.peak_pagefile as u64, c.ws as u64, c.peak_ws as u64)
    }
}

fn fnv(bytes: impl IntoIterator<Item = u8>) -> u64 {
    bytes.into_iter().fold(0xcbf29ce484222325u64, |h, b| (h ^ b as u64).wrapping_mul(0x100000001b3))
}

/// Fresh decode of `MOOSIK_R11_TRACK` → joint superlet analysis at the owner's
/// descriptor (n 9, q 2, 1024 bars, 180 fps, Blend 0.5, spread 0.5, window
/// 0.25 s, 20 Hz–24 kHz) → cache and sidecar publication, through
/// `preprocess_file` (the function the analysis worker runs), on a
/// `MOOSIK_R11_WORKERS` (default 6) pool, cores only, with an isolated
/// calibration and a fresh temporary cache path. Every code of Mix, Left and
/// Right compared against `MOOSIK_R11_REF` when given. Route witness: the
/// pre-process's own `WorkCount`; pool witness: the calibration key's
/// `workers`, which records the pool the analysis actually ran on. One JSON
/// line to `<MOOSIK_R11_OUT>/runs.jsonl`.
#[test]
#[ignore]
fn r11_e2e() {
    use std::io::{Read, Write};
    use std::sync::Arc;
    use std::sync::atomic::{AtomicBool, AtomicUsize};
    let var = |k: &str| std::env::var(k).unwrap_or_else(|_| panic!("{k} not set"));
    assert_eq!(std::env::var("MOOSIK_GPU").ok().as_deref(), Some("0"), "cores route only");
    let _calib = super::super::gpu_calib::isolated_test_env("r11e2e");
    let calib_path = super::super::gpu_calib::path_for_test();
    assert!(calib_path.starts_with(std::env::temp_dir()), "calibration not isolated");
    let label = var("MOOSIK_R11_LABEL");
    let out = std::path::PathBuf::from(var("MOOSIK_R11_OUT"));
    std::fs::create_dir_all(&out).unwrap();
    let track = std::path::PathBuf::from(var("MOOSIK_R11_TRACK"));
    let dsd_rate: u32 = std::env::var("MOOSIK_R11_DSD_RATE").ok().and_then(|v| v.parse().ok()).unwrap_or(88_200);
    let workers: usize = std::env::var("MOOSIK_R11_WORKERS").ok().and_then(|v| v.parse().ok()).unwrap_or(6);
    // The fixture above captured the worker selection and puts it back
    // exactly, after the workers its tickets went to have retired.
    super::super::gpu_calib::set_workers(workers);
    let dir = std::env::temp_dir().join(format!("moosik_r11e2e_{}_{label}", std::process::id()));
    std::fs::create_dir_all(&dir).unwrap();
    let cache = dir.join("track.spectrumcache");
    let sidecar = super::super::stereo::path_for(&cache);
    assert!(!cache.exists() && !sidecar.exists(), "a cache already exists: this would measure reuse");
    let cfg = super::AsltConfig { q_ratio: 2.0, max_window_s: 0.25, n_wavelets: 9, spread: 0.5,
                                  scale: super::super::freq_scale::FreqScale::Blend(0.5) };
    let progress = Arc::new(AtomicUsize::new(0));
    let abort = Arc::new(AtomicBool::new(false));
    let eta = Arc::new(AtomicUsize::new(usize::MAX));
    let mem0 = probe::mem();
    let (cpu0, t0) = (probe::cpu_s(), std::time::Instant::now());
    let msg = super::super::preprocess_file(
        &track, &cache, crate::spectrum::test_store(&cache), false, 0, 1024, &progress,
        super::super::DEFAULT_FFT_SIZE, 16, 0.875, &super::super::WindowFn::Hann, 20.0, 24_000.0,
        &super::super::InterpolationMode::None, &super::super::BarMappingMode::Superlet,
        dsd_rate, &cfg, 180.0, &abort, &eta, super::super::StereoRequest::Joint,
    );
    let wall = t0.elapsed().as_secs_f64();
    let cpu = probe::cpu_s() - cpu0;
    let mem1 = probe::mem();
    let super::super::PreMessage::Done { frames, stereo, frame_rate, meta, .. } = msg else { panic!("pre-process did not finish") };
    let stereo = stereo.expect("joint run returned no channels");
    let n = frames.len();
    let planes: [Vec<u16>; 3] = [&frames, stereo.left(), stereo.right()]
        .map(|p| (0..n).flat_map(|f| p.row(f).expect("row").to_vec()).collect());
    let cache_bytes = std::fs::metadata(&cache).map(|m| m.len()).unwrap_or(0);
    let sidecar_bytes = std::fs::metadata(&sidecar).map(|m| m.len()).unwrap_or(0);
    let hashes: Vec<String> = planes.iter().map(|p| format!("{:016x}", fnv(p.iter().flat_map(|v| v.to_le_bytes())))).collect();
    super::super::gpu_calib::flush();
    let calib: serde_json::Value = std::fs::read_to_string(&calib_path).ok()
        .and_then(|t| serde_json::from_str(&t).ok()).unwrap_or(serde_json::Value::Null);
    let compare = match std::env::var("MOOSIK_R11_REF") {
        Ok(rp) => {
            let mut rf = std::io::BufReader::new(std::fs::File::open(&rp).expect("reference codes"));
            let mut u8b = [0u8; 8];
            rf.read_exact(&mut u8b).unwrap();
            let rframes = u64::from_le_bytes(u8b) as usize;
            let mut diffs = [0u64; 3];
            let mut compared = 0u64;
            let mut ok = rframes == n;
            for (pi, p) in planes.iter().enumerate() {
                rf.read_exact(&mut u8b).unwrap();
                let rl = u64::from_le_bytes(u8b) as usize;
                ok &= rl == p.len();
                let mut buf = vec![0u8; rl * 2];
                rf.read_exact(&mut buf).unwrap();
                for (v, r) in p.iter().zip(buf.chunks_exact(2)) {
                    compared += 1;
                    if *v != u16::from_le_bytes([r[0], r[1]]) { diffs[pi] += 1; }
                }
            }
            format!(",\"ref\":\"{}\",\"ref_shape_ok\":{ok},\"compared\":{compared},\"diff_mix\":{},\"diff_l\":{},\"diff_r\":{}",
                    rp.replace('\\', "/"), diffs[0], diffs[1], diffs[2])
        }
        Err(_) => String::new(),
    };
    let secs = n as f64 / frame_rate;
    let line = format!(
        "{{\"label\":\"{label}\",\"track\":\"{}\",\"dsd_rate\":{dsd_rate},\"workers\":{workers},\
         \"calib_key\":{},\"frames\":{n},\"frame_rate\":{frame_rate},\"secs\":{secs},\"wall_s\":{wall},\"cpu_s\":{cpu},\
         \"rtf\":{},\"commit_before\":{},\"peak_commit\":{},\"peak_ws\":{},\"cache_bytes\":{cache_bytes},\
         \"sidecar_bytes\":{sidecar_bytes},\"hash_mix\":\"{}\",\"hash_l\":\"{}\",\"hash_r\":\"{}\",\"work\":{}{compare}}}",
        track.display().to_string().replace('\\', "/"), calib["key"], wall / secs, mem0.0, mem1.1, mem1.3,
        hashes[0], hashes[1], hashes[2], serde_json::Value::String(format!("{:?}", meta.work)),
    );
    println!("R11E2E {line}");
    let mut j = std::fs::OpenOptions::new().create(true).append(true).open(out.join("runs.jsonl")).unwrap();
    writeln!(j, "{line}").unwrap();
    let _ = std::fs::remove_file(&cache);
    let _ = std::fs::remove_file(&sidecar);
    let _ = std::fs::remove_dir(&dir);
}
