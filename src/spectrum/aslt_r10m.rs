//! Round 10 measurement only (test builds, `#[ignore]`): the round-9 bench plus
//! a stage census (`r10t`), arm selection and a streaming code comparison
//! against a reference `.codes` file. Arms: `ref` = no-blocks pass member by
//! member; `a` = `Seams::production()` (the accepted baseline); research arms
//! are added beside them. Same binary, same input, same pool.

use super::*;

pub(crate) mod probe {
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

fn load_pcm(path: &str) -> (u32, Vec<f32>, Vec<f32>) {
    let b = std::fs::read(path).expect("read pcm");
    let sr = u32::from_le_bytes(b[0..4].try_into().unwrap());
    let len = u64::from_le_bytes(b[4..12].try_into().unwrap()) as usize;
    let f = |i: usize| f32::from_le_bytes(b[12 + 4 * i..16 + 4 * i].try_into().unwrap());
    ((sr), (0..len).map(f).collect(), (len..2 * len).map(f).collect())
}

fn fnv(bytes: impl IntoIterator<Item = u8>) -> u64 {
    bytes.into_iter().fold(0xcbf29ce484222325u64, |h, b| (h ^ b as u64).wrapping_mul(0x100000001b3))
}

/// One timed joint analysis (n 9, q 2, 1024 bars, 180 fps, Blend 0.5, spread
/// 0.5, window 0.25 s, 20 Hz–24 kHz) of a dumped track on a `MOOSIK_R10_WORKERS`
/// (default 6) pool; codes written to `<out>/<label>.codes`; one JSON line.
#[test]
#[ignore]
fn r10_bench() {
    use std::io::Write;
    let var = |k: &str| std::env::var(k).unwrap_or_else(|_| panic!("{k} not set"));
    assert_eq!(std::env::var("MOOSIK_GPU").ok().as_deref(), Some("0"), "cores route only");
    let _calib = super::super::gpu_calib::isolated_test_env("r10bench");
    let calib = super::super::gpu_calib::path_for_test();
    assert!(calib.starts_with(std::env::temp_dir()), "calibration not isolated: {}", calib.display());
    let arm = var("MOOSIK_R10_ARM");
    let label = var("MOOSIK_R10_LABEL");
    let out = std::path::PathBuf::from(var("MOOSIK_R10_OUT"));
    std::fs::create_dir_all(&out).unwrap();
    let workers: usize = std::env::var("MOOSIK_R10_WORKERS").ok().and_then(|v| v.parse().ok()).unwrap_or(6);
    let (sr, mut l, mut r) = load_pcm(&var("MOOSIK_R10_PCM"));
    if let Some(secs) = std::env::var("MOOSIK_R10_SECS").ok().and_then(|v| v.parse::<f64>().ok()) {
        let n = ((secs * sr as f64) as usize).min(l.len());
        l.truncate(n);
        r.truncate(n);
    }
    let cfg = AsltConfig { q_ratio: 2.0, max_window_s: 0.25, n_wavelets: 9, spread: 0.5,
                           scale: super::super::freq_scale::FreqScale::Blend(0.5) };
    let hop = hop_for_fps(sr, 180.0);
    let a_max = (sr as f32 / 2.0).min(24_000.0);
    let map = |m: f32| crate::spectrum::aslt_code(m);
    let mut seams = Seams { codes: Some(&map), ..Seams::production() };
    match arm.as_str() {
        "ref" => seams.share_geometry = None,
        // The original full inverse (the pre-round-11 production inverse).
        "a" => {
            assert_eq!(seams.share_geometry, Some(SHARE_GEOMETRY));
            seams.full_inverse = true;
        }
        // Production: pruned inverse, fused product, per-member scheduling.
        "cand" => assert_eq!(seams.share_geometry, Some(SHARE_GEOMETRY)),
        o => panic!("unknown arm {o}"),
    }
    if let Some(mb) = std::env::var("MOOSIK_R10_BUDGET_MB").ok().and_then(|v| v.parse::<usize>().ok()) {
        seams.shared_budget = mb << 20;
    }
    // Progress as the pre-process sees it: when each bar is reported done.
    let t0 = std::time::Instant::now();
    let stamps = std::sync::Mutex::new(Vec::<f64>::new());
    let done = |_: usize| stamps.lock().unwrap().push(t0.elapsed().as_secs_f64());
    // Partial progress (the shared pass's member fractions), as the
    // pre-process would see it.
    let part_stamps = std::sync::Mutex::new(Vec::<f64>::new());
    let part = |_: usize, _: f32| part_stamps.lock().unwrap().push(t0.elapsed().as_secs_f64());
    seams.bar_progress = Some(&part);
    r10t::reset();
    r10t::CENSUS.store(true, std::sync::atomic::Ordering::SeqCst);
    let pool = rayon::ThreadPoolBuilder::new().num_threads(workers).build().unwrap();
    let mem0 = probe::mem();
    let (cpu0, t1) = (probe::cpu_s(), std::time::Instant::now());
    let res = pool.install(|| analyze_routed(
        Input::Joint { left: &l, right: &r }, sr, 1024, 20.0, a_max, hop, &cfg,
        &|| true, &done, route_to_device, seams,
    ));
    let wall = t1.elapsed().as_secs_f64();
    let cpu = probe::cpu_s() - cpu0;
    let mem1 = probe::mem();
    let done_count = stamps.lock().unwrap().len();
    let mut st = stamps.into_inner().unwrap();
    st.extend(part_stamps.into_inner().unwrap());
    st.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let mut prev = 0.0f64;
    let max_gap = st.iter().map(|&t| { let g = t - prev; prev = t; g }).fold(0.0f64, f64::max).max(wall - prev);
    let (planes, frames) = res.codes.expect("codes");
    let mut hashes = Vec::new();
    for p in &planes { hashes.push(format!("{:016x}", fnv(p.iter().flat_map(|v| v.to_le_bytes())))); }
    if std::env::var("MOOSIK_R10_WRITE").ok().as_deref() == Some("1") {
        let mut f = std::io::BufWriter::new(std::fs::File::create(out.join(format!("{label}.codes"))).unwrap());
        f.write_all(&(frames as u64).to_le_bytes()).unwrap();
        for p in &planes {
            f.write_all(&(p.len() as u64).to_le_bytes()).unwrap();
            for v in p { f.write_all(&v.to_le_bytes()).unwrap(); }
        }
    }
    // Every code of every plane against the reference, streamed.
    let compare = match std::env::var("MOOSIK_R10_REF") {
        Ok(rp) => {
            use std::io::Read;
            let mut rf = std::io::BufReader::new(std::fs::File::open(&rp).expect("reference codes"));
            let mut u8b = [0u8; 8];
            rf.read_exact(&mut u8b).unwrap();
            let rframes = u64::from_le_bytes(u8b) as usize;
            let mut diffs = [0u64; 3];
            let mut first: Option<(usize, usize)> = None;
            let mut lens_ok = rframes == frames;
            for (pi, p) in planes.iter().enumerate() {
                rf.read_exact(&mut u8b).unwrap();
                let rl = u64::from_le_bytes(u8b) as usize;
                lens_ok &= rl == p.len();
                let mut buf = vec![0u8; rl * 2];
                rf.read_exact(&mut buf).unwrap();
                for (i, (v, r)) in p.iter().zip(buf.chunks_exact(2)).enumerate() {
                    if *v != u16::from_le_bytes([r[0], r[1]]) {
                        diffs[pi] += 1;
                        if first.is_none() { first = Some((pi, i)); }
                    }
                }
            }
            let cells: usize = planes.iter().map(|p| p.len()).sum();
            format!(",\"ref\":\"{}\",\"ref_shape_ok\":{lens_ok},\"cells\":{cells},\"diff_mix\":{},\"diff_l\":{},\"diff_r\":{},\"first_diff\":\"{:?}\"",
                    rp.replace('\\', "/"), diffs[0], diffs[1], diffs[2], first)
        }
        Err(_) => String::new(),
    };
    let secs = l.len() as f64 / sr as f64;
    let w = res.work;
    let line = format!(
        "{{\"label\":\"{label}\",\"arm\":\"{arm}\",\"sr\":{sr},\"secs\":{secs},\"workers\":{workers},\"frames\":{frames},\
         \"wall_s\":{wall},\"cpu_s\":{cpu},\"rtf\":{},\"commit_before\":{},\"peak_commit\":{},\"peak_ws\":{},\
         \"share_peak_bytes\":{},\"shared_members\":{},\"share_refused\":{},\"share_subbatches\":{},\"share_batches\":{},\
         \"callbacks\":{},\"progress_events\":{},\"max_progress_gap_s\":{max_gap},\
         \"hash_mix\":\"{}\",\"hash_l\":\"{}\",\"hash_r\":\"{}\"}}",
        wall / secs, mem0.0, mem1.1, mem1.3, w.share_peak_bytes, w.shared_members, w.share_refused_members,
        w.share_subbatches, w.share_batches, done_count, st.len(), hashes[0], hashes[1], hashes[2],
    );
    // Stage census.
    let mut census = format!(",\"hop\":{hop},\"f_lo\":20,\"f_hi\":{a_max},\"process_cpu_s\":{cpu}");
    for (i, name) in r10t::NAMES.iter().enumerate() {
        let ns = r10t::NS[i].load(std::sync::atomic::Ordering::Relaxed);
        let c = r10t::CNT[i].load(std::sync::atomic::Ordering::Relaxed);
        census.push_str(&format!(",\"t_{name}_s\":{},\"n_{name}\":{c}", ns as f64 * 1e-9));
    }
    for (lg, a) in r10t::INV.iter().enumerate() {
        let c = a.load(std::sync::atomic::Ordering::Relaxed);
        if c > 0 { census.push_str(&format!(",\"inv_2^{lg}\":{c}")); }
    }
    let line = format!("{}{census}{}}}", &line[..line.len() - 1], compare);
    println!("R10RUN {line}");
    let mut j = std::fs::OpenOptions::new().create(true).append(true).open(out.join("runs.jsonl")).unwrap();
    writeln!(j, "{line}").unwrap();
}

/// End to end, through the production pre-process (`preprocess_file`, the
/// function the analysis worker runs): fresh decode of `MOOSIK_R10_TRACK` →
/// joint superlet analysis at the owner's descriptor → cache publication to a
/// fresh temporary path (none exists beforehand: a reuse benchmark is refused).
/// Arms as `r10_bench`. Codes compared against `MOOSIK_R10_REF` (an
/// analysis-core `.codes` file of the same track) when given. One JSON line.
#[test]
#[ignore]
fn r10_e2e() {
    use std::io::Write;
    use std::sync::Arc;
    use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};
    let var = |k: &str| std::env::var(k).unwrap_or_else(|_| panic!("{k} not set"));
    assert_eq!(std::env::var("MOOSIK_GPU").ok().as_deref(), Some("0"), "cores route only");
    let _calib = super::super::gpu_calib::isolated_test_env("r10e2e");
    assert!(super::super::gpu_calib::path_for_test().starts_with(std::env::temp_dir()), "calibration not isolated");
    let arm = var("MOOSIK_R10_ARM");
    let label = var("MOOSIK_R10_LABEL");
    let out = std::path::PathBuf::from(var("MOOSIK_R10_OUT"));
    std::fs::create_dir_all(&out).unwrap();
    let track = std::path::PathBuf::from(var("MOOSIK_R10_TRACK"));
    let dsd_rate: u32 = std::env::var("MOOSIK_R10_DSD_RATE").ok().and_then(|v| v.parse().ok()).unwrap_or(88_200);
    let workers: usize = std::env::var("MOOSIK_R10_WORKERS").ok().and_then(|v| v.parse().ok()).unwrap_or(6);
    // This harness runs alone in its own process: the default seeds the
    // `Seams::production()` the pre-process builds internally.
    match arm.as_str() {
        "a" => r10t::FULL_DEFAULT.store(true, Ordering::SeqCst),
        "cand" => {}
        o => panic!("unknown arm {o}"),
    }
    // The fixture above captured the worker selection and puts it back
    // exactly, after the workers its tickets went to have retired.
    super::super::gpu_calib::set_workers(workers);
    let dir = std::env::temp_dir().join(format!("moosik_r10e2e_{}_{label}", std::process::id()));
    std::fs::create_dir_all(&dir).unwrap();
    let cache = dir.join("track.spectrumcache");
    let sidecar = super::super::stereo::path_for(&cache);
    assert!(!cache.exists() && !sidecar.exists(), "a cache already exists: this would measure reuse");
    let cfg = AsltConfig { q_ratio: 2.0, max_window_s: 0.25, n_wavelets: 9, spread: 0.5,
                           scale: super::super::freq_scale::FreqScale::Blend(0.5) };
    r10t::reset();
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
    let super::super::PreMessage::Done { frames, stereo, frame_rate, .. } = msg else { panic!("pre-process did not finish") };
    let stereo = stereo.expect("joint run returned no channels");
    let n = frames.len();
    let planes: [Vec<u16>; 3] = [&frames, stereo.left(), stereo.right()]
        .map(|p| (0..n).flat_map(|f| p.row(f).expect("row").to_vec()).collect());
    let cache_bytes = std::fs::metadata(&cache).map(|m| m.len()).unwrap_or(0);
    let sidecar_bytes = std::fs::metadata(&sidecar).map(|m| m.len()).unwrap_or(0);
    let hashes: Vec<String> = planes.iter().map(|p| format!("{:016x}", fnv(p.iter().flat_map(|v| v.to_le_bytes())))).collect();
    let compare = match std::env::var("MOOSIK_R10_REF") {
        Ok(rp) => {
            use std::io::Read;
            let mut rf = std::io::BufReader::new(std::fs::File::open(&rp).expect("reference codes"));
            let mut u8b = [0u8; 8];
            rf.read_exact(&mut u8b).unwrap();
            let rframes = u64::from_le_bytes(u8b) as usize;
            let mut diffs = [0u64; 3];
            let mut ok = rframes == n;
            for (pi, p) in planes.iter().enumerate() {
                rf.read_exact(&mut u8b).unwrap();
                let rl = u64::from_le_bytes(u8b) as usize;
                ok &= rl == p.len();
                let mut buf = vec![0u8; rl * 2];
                rf.read_exact(&mut buf).unwrap();
                for (v, r) in p.iter().zip(buf.chunks_exact(2)) {
                    if *v != u16::from_le_bytes([r[0], r[1]]) { diffs[pi] += 1; }
                }
            }
            format!(",\"ref\":\"{}\",\"ref_shape_ok\":{ok},\"diff_mix\":{},\"diff_l\":{},\"diff_r\":{}",
                    rp.replace('\\', "/"), diffs[0], diffs[1], diffs[2])
        }
        Err(_) => String::new(),
    };
    let secs = n as f64 / frame_rate;
    let line = format!(
        "{{\"label\":\"{label}\",\"arm\":\"{arm}\",\"e2e\":true,\"track\":\"{}\",\"dsd_rate\":{dsd_rate},\"workers\":{workers},\
         \"frames\":{n},\"frame_rate\":{frame_rate},\"secs\":{secs},\"wall_s\":{wall},\"cpu_s\":{cpu},\"rtf\":{},\
         \"commit_before\":{},\"peak_commit\":{},\"peak_ws\":{},\"cache_bytes\":{cache_bytes},\"sidecar_bytes\":{sidecar_bytes},\
         \"hash_mix\":\"{}\",\"hash_l\":\"{}\",\"hash_r\":\"{}\"{compare}}}",
        track.display().to_string().replace('\\', "/"), wall / secs, mem0.0, mem1.1, mem1.3, hashes[0], hashes[1], hashes[2],
    );
    println!("R10E2E {line}");
    let mut j = std::fs::OpenOptions::new().create(true).append(true).open(out.join("runs.jsonl")).unwrap();
    writeln!(j, "{line}").unwrap();
    r10t::FULL_DEFAULT.store(false, Ordering::SeqCst);
    let _ = std::fs::remove_file(&cache);
    let _ = std::fs::remove_file(&sidecar);
    let _ = std::fs::remove_dir(&dir);
}
