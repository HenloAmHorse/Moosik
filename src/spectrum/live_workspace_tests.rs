//! Workspace changes against the allocating FFT stage shipped in 1.5.4.
//! The older downstream tests freeze the mapping, not the FFT stage; these
//! checks deliberately keep the old snapshot/window/FFT/norm expression too.
use super::*;

fn allocating_fft(a: &SpectrumAnalyzer, pcm: &[f32]) -> (Vec<Complex<f32>>, Vec<f32>) {
    // Frozen expression from 330b8ac::process_realtime, including fft.process's
    // own scratch allocation. Never call the new workspace helpers here.
    let fft_size = a.fft_size;
    let rt_padded = fft_size * 2;
    let samples = pcm[pcm.len() - fft_size..].to_vec();
    let mut buf: Vec<Complex<f32>> = samples
        .iter()
        .zip(a.window_coeffs.iter())
        .map(|(s, w)| Complex { re: s * w, im: 0.0 })
        .chain(std::iter::repeat_n(
            Complex { re: 0.0, im: 0.0 },
            rt_padded - fft_size,
        ))
        .collect();
    a.rt_fft_plan.process(&mut buf);
    let norms = buf[..rt_padded / 2].iter().map(|c| c.norm()).collect();
    (buf, norms)
}

fn pcm(n: usize, phase: f32) -> Vec<f32> {
    (0..n)
        .map(|i| {
            let x = i as f32;
            0.4 * (x * 0.017 + phase).sin()
                + 0.13 * (x * 0.419 - phase).cos()
                + ((i.wrapping_mul(7919) % 65521) as f32 / 65521.0 - 0.5) * 0.04
        })
        .collect()
}
fn fft_bits(v: &[Complex<f32>]) -> Vec<(u32, u32)> {
    v.iter().map(|c| (c.re.to_bits(), c.im.to_bits())).collect()
}
fn bits(v: &[f32]) -> Vec<u32> {
    v.iter().map(|x| x.to_bits()).collect()
}

#[test]
fn actual_live_fft_workspaces_match_the_allocating_stage_and_stop_allocating() {
    let mut a = SpectrumAnalyzer::new(new_sample_buf());
    // This gate measures actual FFT executions, not identical-input reuse.
    a.rt_disable_input_reuse = true;
    // Up/down sizes exercise resizing and stale zero-pad data. Every supported
    // window participates, not just the Hann used by the isolated prototype.
    for size in [2048, 8192, 16384, 4096] {
        for window in [
            WindowFn::Hann,
            WindowFn::Hamming,
            WindowFn::Blackman,
            WindowFn::FlatTop,
        ] {
            a.fft_size = size;
            a.window_fn = window;
            a.rebuild_fft();
            for tick in 0..5 {
                let x = if tick == 4 {
                    vec![0.0; size + 53]
                } else {
                    pcm(size + 53, tick as f32)
                };
                {
                    let mut g = a.sample_buf.lock().unwrap();
                    g.clear();
                    g.extend_from_slice(&x);
                }
                let (want_fft, want_norms) = allocating_fft(&a, &x);
                assert!(a.transform_realtime());
                assert_eq!(
                    fft_bits(&a.rt_work),
                    fft_bits(&want_fft),
                    "Mix size={size} tick={tick}"
                );
                assert_eq!(bits(&a.last_fft_norms), bits(&want_norms));
                let mix = bits(&a.last_fft_norms);
                for phase in [-0.3, 0.7] {
                    let ch = pcm(size + 31, phase);
                    let (want_fft, want_norms) = allocating_fft(&a, &ch);
                    a.transform_channel(1, &ch);
                    assert_eq!(
                        fft_bits(&a.ch_scratch),
                        fft_bits(&want_fft),
                        "channel size={size}"
                    );
                    assert_eq!(bits(&a.ch_norms[1]), bits(&want_norms));
                    assert_eq!(
                        bits(&a.last_fft_norms),
                        mix,
                        "a channel overwrote Mix norms"
                    );
                }
                let arm = crate::alloc_probe::arm();
                let accepted = a.transform_realtime();
                let mix_allocs = crate::alloc_probe::disarm(arm);
                assert!(accepted);
                assert_eq!(mix_allocs, 0, "Mix FFT size={size}");
                let arm = crate::alloc_probe::arm();
                a.transform_channel(1, &x);
                let channel_allocs = crate::alloc_probe::disarm(arm);
                assert_eq!(channel_allocs, 0, "channel FFT size={size}");
            }
        }
    }
}

#[test]
fn short_input_does_not_spend_snap_or_replace_previous_mix_measurement() {
    let mut a = SpectrumAnalyzer::new(new_sample_buf());
    let x = pcm(a.fft_size, 0.0);
    a.sample_buf.lock().unwrap().extend_from_slice(&x);
    a.process_realtime(1.0 / 180.0);
    let before = realtime_route_tests::state(&a);
    a.sample_buf.lock().unwrap().clear();
    a.snap_next_realtime = true;
    a.process_realtime(1.0 / 180.0);
    assert_eq!(before, realtime_route_tests::state(&a));
    assert!(a.snap_next_realtime);
    a.process_channels(&[], &[], 1.0 / 180.0);
    assert!(a.bars_left.is_empty() && a.bars_right.is_empty());
}

#[test]
fn a_channel_tick_without_a_mix_tick_keeps_mix_norms_and_rebuilds_workspaces() {
    let mut a = SpectrumAnalyzer::new(new_sample_buf());
    for size in [2048, 8192, 4096] {
        a.fft_size = size;
        a.bar_count = 128;
        a.rebuild_fft();
        a.last_fft_norms = vec![0.25, 0.75];
        let left = pcm(size + 10, 0.3);
        let right = pcm(size + 20, 1.7);
        a.process_channels(&left, &right, 1.0 / 180.0);
        assert_eq!(a.last_fft_norms, [0.25, 0.75]);
        assert_eq!(a.bars_left.len(), 128);
        assert_eq!(a.bars_right.len(), 128);
        let (_, norms) = allocating_fft(&a, &right);
        assert_eq!(bits(&a.ch_norms[1]), bits(&norms));
    }
}
