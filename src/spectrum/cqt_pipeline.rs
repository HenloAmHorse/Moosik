//! Adapter from the genuine CQT core to the existing cache/publication path.

use super::*;

#[allow(clippy::too_many_arguments)]
pub(super) fn run(
    input: aslt::Input<'_>,
    sample_rate: u32,
    bars: usize,
    min_freq: f32,
    max_freq: f32,
    cfg: &cqt::Config,
    abort: &AtomicBool,
    progress: &Arc<AtomicUsize>,
    eta_secs: &AtomicUsize,
    base: usize,
    span: usize,
) -> Result<Passes, PassFailed> {
    let lo = f64::from(min_freq);
    let hi = f64::from(max_freq).min(f64::from(sample_rate) / 2.0);
    if bars == 0 || !lo.is_finite() || !hi.is_finite() || lo <= 0.0 || hi <= lo {
        return Err(PassFailed::Failed(
            "CQT needs a positive frequency range below Nyquist".into(),
        ));
    }
    let frequencies: Vec<f64> = (0..bars)
        .map(|i| lo * (hi / lo).powf((i as f64 + 0.5) / bars as f64))
        .collect();
    let input = match input {
        aslt::Input::Mono(signal) => cqt::Input::Mono(signal),
        aslt::Input::Joint { left, right } => cqt::Input::Joint { left, right },
    };
    let samples = match input {
        cqt::Input::Mono(signal) => signal.len(),
        cqt::Input::Joint { left, .. } => left.len(),
    };
    let frames = cqt::frame_count(samples, sample_rate, cfg.frame_rate)
        .map_err(|e| PassFailed::Failed(e.to_string()))?;
    if frames > cache::MAX_FRAMES {
        return Err(PassFailed::Failed(format!(
            "CQT needs {frames} rows, above the cache format's {}-row limit; lower the explicit analysis frame rate",
            cache::MAX_FRAMES
        )));
    }
    let started = Instant::now();
    // Use the same bounded, owner-selected analysis pool as the other routes;
    // the core never silently creates work on Rayon's global pool.
    let output = gpu_calib::install(|| {
        cqt::analyze_input_codes(
            input,
            sample_rate,
            &frequencies,
            cfg,
            cqt::Limits::default(),
            &|| !abort.load(Ordering::Relaxed),
            &|p| {
                store_progress_max(progress, base + (p * span as f32) as usize);
                if p > 0.0 {
                    let remaining =
                        started.elapsed().as_secs_f64() * (1.0 - f64::from(p)) / f64::from(p);
                    eta_secs.store(remaining.max(0.0) as usize, Ordering::Relaxed);
                }
            },
            &aslt_code,
        )
    })
    .map_err(|error| match error {
        cqt::Error::Aborted => PassFailed::Aborted,
        other => PassFailed::Failed(other.to_string()),
    })?;
    if abort.load(Ordering::Relaxed) {
        return Err(PassFailed::Aborted);
    }
    let bars = output.bars;
    crate::mlog!(
        "[cqt] {} bars, {} frames @ {} fps; {} kernel / {} input / {} inverse FFTs ({} requested-output calls, {} whole, {} requested values), {} batches; {} parallel groups / max {} parallel slots, {} pruning / {} parallel allocation fallbacks; peak explicit workspace {:.1} MiB (not RSS), code planes {:.1} MiB, {:.2}s",
        bars,
        output.frames,
        cfg.frame_rate,
        output.work.kernel_ffts,
        output.work.input_ffts,
        output.work.inverse_ffts,
        output.work.inverses_pruned,
        output.work.inverses_full,
        output.work.pruned_output_values,
        output.work.batches,
        output.work.parallel_groups,
        output.work.max_inverse_workers,
        output.work.pruning_workspace_fallbacks,
        output.work.parallel_allocation_fallbacks,
        output.work.peak_workspace_bytes as f64 / (1024.0 * 1024.0),
        output.work.output_bytes as f64 / (1024.0 * 1024.0),
        started.elapsed().as_secs_f64()
    );
    let [mix, left, right] = output.codes;
    let plane = |codes: Vec<u16>| {
        (!codes.is_empty()).then(|| cache::PreFrames::from_v4_codes(codes, output.frames, bars))
    };
    Ok(Passes {
        mix: Vec::new(),
        left: Vec::new(),
        right: Vec::new(),
        codes: Some((
            cache::PreFrames::from_v4_codes(mix, output.frames, bars),
            plane(left),
            plane(right),
        )),
        work: aslt::WorkCount {
            column_convolutions: output.work.column_convolutions,
            ..Default::default()
        },
    })
}
