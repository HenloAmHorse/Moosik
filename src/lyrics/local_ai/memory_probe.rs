//! Opt-in research using the pinned runtime, production prompt/chunk/parser
//! and existing child supervision. Never runs in the normal test suite.
use super::*;

#[test]
#[ignore]
fn local_memory_route_probe() {
    let root = PathBuf::from(std::env::var("MOOSIK_MEMORY_PROBE_ROOT").expect("private evidence folder required"));
    let installed = PathBuf::from(std::env::var("MOOSIK_MEMORY_PROBE_PACK").expect("read-only installed 30B folder required"));
    let tag = std::env::var("MOOSIK_MEMORY_PROBE_ROUTE").expect("route required");
    let args: &'static [&'static str] = match tag.as_str() {
        "baseline" => &["-ngl", "99", "--cpu-moe"],
        "production-lowram" => HY30_LOW_RAM_ARGS,
        "experts24" => &["-ngl", "99", "--n-cpu-moe", "24"],
        "experts36" => &["-ngl", "99", "--n-cpu-moe", "36"],
        "experts40" => &["-ngl", "99", "--n-cpu-moe", "40"],
        "experts44" => &["-ngl", "99", "--n-cpu-moe", "44"],
        "experts40-smallbatch" => &["-ngl", "99", "--n-cpu-moe", "40", "-b", "512", "-ub", "128"],
        "experts40-buffered" => &["-ngl", "99", "--n-cpu-moe", "40", "--load-mode", "none"],
        "experts36-buffered" => &["-ngl", "99", "--n-cpu-moe", "36", "--load-mode", "none"],
        "baseline-buffered" => &["-ngl", "99", "--cpu-moe", "--load-mode", "none"],
        "experts20" => &["-ngl", "99", "--n-cpu-moe", "20"],
        "experts16" => &["-ngl", "99", "--n-cpu-moe", "16"],
        "experts20-smallbatch" => &["-ngl", "99", "--n-cpu-moe", "20", "-b", "512", "-ub", "128"],
        "lazy" => &["-ngl", "99", "--cpu-moe", "--lazy-mode", "on"],
        _ => panic!("unknown probe route"),
    };
    let count = std::env::var("MOOSIK_MEMORY_PROBE_ROWS").unwrap_or("16".into()).parse::<usize>().unwrap();
    assert!((1..=150).contains(&count));
    let cancellation_ms = std::env::var("MOOSIK_MEMORY_PROBE_CANCEL_MS").ok().map(|s| s.parse::<u64>().unwrap());
    let suffix = cancellation_ms.map_or(String::new(), |ms| format!("-cancel{ms}"));
    let scratch = root.join(format!("{tag}-{count}{suffix}"));
    std::fs::create_dir_all(&scratch).unwrap();
    let cancel = AtomicBool::new(false);
    let stop = Stop::new(&cancel);
    let held = verify_runtime(&installed.join("runtime"), &HYMT2_30B, &cancel).unwrap();
    let keep = Keep { root: root.clone(), held: std::sync::Arc::new(held), scratch: None, handed_over: Default::default() };
    let mut llama = Llama { variation: Variation::default(), runtime: installed.join("runtime"), model: installed.join(HYMT2_30B.files[0].name),
        scratch: scratch.clone(), device: Device::Gpu, n_ctx: HYMT2_30B.n_ctx, keep,
        // Research only: experimental placements have no admitted product
        // requirement yet. Supervision still enforces the system memory floor.
        need: if tag == "production-lowram" { HY30_LOW_RAM_NEED } else { memory::Need { phys: 0, commit: 0 } },
        adapter: HYMT2_30B.adapter, gpu_args: args };
    let fixture = ["駅のホームで君を待っていた", "帰らないと知っていても", "雨に消えた約束を", "今も胸に抱いている",
        "遠い灯りが揺れるたび", "言えなかった言葉がある", "昨日の風は戻らない", "それでも朝はやって来る",
        "君の名前を呼ばずに", "空いた椅子を見つめていた", "さよならだけが残っても", "この歌を忘れないで"];
    let lines: Vec<String> = (0..count).map(|i| fixture[i % fixture.len()].to_owned()).collect();
    let refs: Vec<&str> = lines.iter().map(String::as_str).collect();
    let job = Job::new(Identity::new(&HYMT2_30B, Source::Japanese, Target::TraditionalChinese, Style::Faithful, Device::Gpu),
        "Memory route fixture", "Synthetic", &refs, &vec![true; count]).unwrap();
    peak::RUNS.with(|r| r.borrow_mut().clear());
    let before = memory::read();
    let start = Instant::now();
    let finished = AtomicBool::new(false);
    let requested_at = std::sync::atomic::AtomicU64::new(0);
    let result = std::thread::scope(|scope| {
        if let Some(ms) = cancellation_ms {
            let (finished, requested_at, cancel) = (&finished, &requested_at, &cancel);
            scope.spawn(move || {
                while start.elapsed() < Duration::from_millis(ms) && !finished.load(Ordering::Acquire) {
                    std::thread::sleep(Duration::from_millis(5));
                }
                if !finished.load(Ordering::Acquire) {
                    requested_at.store(start.elapsed().as_micros() as u64 + 1, Ordering::Release);
                    cancel.store(true, Ordering::Release);
                }
            });
        }
        let result = translate(&job, &HYMT2_30B, &mut llama, &stop, |_, _| {});
        finished.store(true, Ordering::Release);
        result
    });
    let elapsed = start.elapsed().as_secs_f64();
    let requested_us = requested_at.load(Ordering::Acquire);
    let peaks = peak::RUNS.with(|r| r.borrow().clone());
    peak::LAST_LOG.with(|l| std::fs::write(scratch.join("last-runtime.log"), &*l.borrow()).unwrap());
    let record = serde_json::json!({ "route":tag, "args":args, "rows":count, "wall_s":elapsed,
        "cancel_requested_after_ms":cancellation_ms,
        "cancel_to_return_ms":(requested_us > 0).then(|| elapsed * 1000.0 - (requested_us - 1) as f64 / 1000.0),
        "before_phys":before.phys, "before_commit":before.commit, "child_peaks_ws_commit_cpu":peaks,
        "ok":result.is_ok(), "output":result.as_ref().ok().map(|o| &o.rows), "error":result.as_ref().err(),
        "context":HYMT2_30B.n_ctx, "sampling":HYMT2_30B.adapter.sampling() });
    std::fs::write(scratch.join("result.json"), serde_json::to_vec_pretty(&record).unwrap()).unwrap();
    println!("{record}");
    // An unsuccessful route is a measured rejection, not a reason to lose
    // the record or abort the rest of an explicit research campaign.
}
