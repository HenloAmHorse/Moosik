//! Isolated reproduction of the live FFT stage's allocation pattern.
//! No mapping, mutex, GUI, playback or frame pacing; NOT an app benchmark.
use rustfft::{Fft, FftPlanner, num_complex::Complex};
use std::alloc::{GlobalAlloc, Layout, System};
use std::hint::black_box;
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::time::Instant;

struct CountAlloc;
static COUNT: AtomicBool = AtomicBool::new(false);
static CALLS: AtomicU64 = AtomicU64::new(0);
static BYTES: AtomicU64 = AtomicU64::new(0);
#[global_allocator]
static ALLOC: CountAlloc = CountAlloc;
// Instrumentation only. Delegates all allocation semantics to System; never
// changes layouts, returned pointers or lifetimes. Counting is off for timing.
unsafe impl GlobalAlloc for CountAlloc {
    unsafe fn alloc(&self, l: Layout) -> *mut u8 {
        if COUNT.load(Ordering::Relaxed) {
            CALLS.fetch_add(1, Ordering::Relaxed);
            BYTES.fetch_add(l.size() as u64, Ordering::Relaxed);
        }
        unsafe { System.alloc(l) }
    }
    unsafe fn dealloc(&self, p: *mut u8, l: Layout) {
        unsafe { System.dealloc(p, l) }
    }
    unsafe fn realloc(&self, p: *mut u8, l: Layout, n: usize) -> *mut u8 {
        if COUNT.load(Ordering::Relaxed) {
            CALLS.fetch_add(1, Ordering::Relaxed);
            BYTES.fetch_add(n as u64, Ordering::Relaxed);
        }
        unsafe { System.realloc(p, l, n) }
    }
}
fn baseline(x: &[f32], w: &[f32], fft: &dyn Fft<f32>) -> (Vec<Complex<f32>>, Vec<f32>) {
    let samples = x.to_vec();
    let n = samples.len() * 2;
    let mut buf: Vec<_> = samples
        .iter()
        .zip(w)
        .map(|(s, w)| Complex::new(s * w, 0.0))
        .chain(std::iter::repeat_n(Complex::default(), n - samples.len()))
        .collect();
    fft.process(&mut buf);
    let norms = buf[..n / 2].iter().map(|v| v.norm()).collect();
    (buf, norms)
}
struct Reused {
    buf: Vec<Complex<f32>>,
    scratch: Vec<Complex<f32>>,
    norms: Vec<f32>,
}
impl Reused {
    fn new(size: usize, fft: &dyn Fft<f32>) -> Self {
        Self {
            buf: vec![Complex::default(); size * 2],
            scratch: vec![Complex::default(); fft.get_inplace_scratch_len()],
            norms: vec![0.0; size],
        }
    }
    fn run(&mut self, x: &[f32], w: &[f32], fft: &dyn Fft<f32>) {
        assert_eq!(x.len(), self.norms.len());
        assert_eq!(x.len(), w.len());
        for (dst, (s, w)) in self.buf[..x.len()].iter_mut().zip(x.iter().zip(w)) {
            *dst = Complex::new(s * w, 0.0);
        }
        self.buf[x.len()..].fill(Complex::default());
        fft.process_with_scratch(&mut self.buf, &mut self.scratch);
        for (dst, v) in self.norms.iter_mut().zip(&self.buf) {
            *dst = v.norm();
        }
    }
}
fn measured_allocs(f: impl FnOnce()) -> (u64, u64) {
    CALLS.store(0, Ordering::Relaxed);
    BYTES.store(0, Ordering::Relaxed);
    COUNT.store(true, Ordering::Relaxed);
    f();
    COUNT.store(false, Ordering::Relaxed);
    (CALLS.load(Ordering::Relaxed), BYTES.load(Ordering::Relaxed))
}
fn main() {
    println!(
        "size,rep,baseline_us,reused_us,baseline_alloc_calls,reused_alloc_calls,baseline_alloc_bytes,reused_alloc_bytes,equal_complex_and_norm_bits"
    );
    for size in [4096, 8192, 16384] {
        let mut p = FftPlanner::new();
        let fft = p.plan_fft_forward(size * 2);
        let x: Vec<_> = (0..size).map(|i| ((i as f32) * 0.037).sin()).collect();
        let w: Vec<_> = (0..size)
            .map(|i| 0.5 - 0.5 * (std::f32::consts::TAU * i as f32 / (size - 1) as f32).cos())
            .collect();
        let mut reused = Reused::new(size, fft.as_ref());
        reused.run(&x, &w, fft.as_ref());
        let (buf, norms) = baseline(&x, &w, fft.as_ref());
        let equal = buf
            .iter()
            .zip(&reused.buf)
            .all(|(a, b)| a.re.to_bits() == b.re.to_bits() && a.im.to_bits() == b.im.to_bits())
            && norms
                .iter()
                .zip(&reused.norms)
                .all(|(a, b)| a.to_bits() == b.to_bits());
        assert!(equal);
        let (ba, bb) = measured_allocs(|| {
            black_box(baseline(&x, &w, fft.as_ref()));
        });
        let (ra, rb) = measured_allocs(|| {
            reused.run(&x, &w, fft.as_ref());
            black_box(&reused.norms);
        });
        assert_eq!(ra, 0);
        // The first screen showed a large first-case warmup trend. Retain that
        // screen, but do not turn its best/worst ratio into a speed claim.
        for _ in 0..2000 {
            black_box(baseline(&x, &w, fft.as_ref()));
            reused.run(&x, &w, fft.as_ref());
            black_box(&reused.norms);
        }
        for rep in 0..5 {
            let base = || {
                let start = Instant::now();
                for _ in 0..1000 {
                    black_box(baseline(black_box(&x), &w, fft.as_ref()));
                }
                start.elapsed().as_secs_f64() * 1000.0
            };
            let mut opt = || {
                let start = Instant::now();
                for _ in 0..1000 {
                    reused.run(black_box(&x), &w, fft.as_ref());
                    black_box(&reused.norms);
                }
                start.elapsed().as_secs_f64() * 1000.0
            };
            // milliseconds/1000 frames == microseconds/frame
            let (b, r) = if rep % 2 == 0 {
                (base(), opt())
            } else {
                let r = opt();
                let b = base();
                (b, r)
            };
            println!("{size},{rep},{b:.6},{r:.6},{ba},{ra},{bb},{rb},{equal}");
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn reusing_workspaces_preserves_every_fft_and_norm_bit() {
        for size in [256, 1024, 4096, 8192, 16384] {
            let mut planner = FftPlanner::new();
            let fft = planner.plan_fft_forward(size * 2);
            let w: Vec<_> = (0..size)
                .map(|i| 0.5 - 0.5 * (std::f32::consts::TAU * i as f32 / (size - 1) as f32).cos())
                .collect();
            let mut reused = Reused::new(size, fft.as_ref());
            for tick in 0..8 {
                let x: Vec<_> = (0..size)
                    .map(|i| {
                        if tick == 7 {
                            0.0
                        } else {
                            ((i as f32) * 0.017 + tick as f32).sin()
                        }
                    })
                    .collect();
                reused.run(&x, &w, fft.as_ref());
                let (buf, norms) = baseline(&x, &w, fft.as_ref());
                for (a, b) in buf.iter().zip(&reused.buf) {
                    assert_eq!(a.re.to_bits(), b.re.to_bits());
                    assert_eq!(a.im.to_bits(), b.im.to_bits());
                }
                for (a, b) in norms.iter().zip(&reused.norms) {
                    assert_eq!(a.to_bits(), b.to_bits());
                }
                assert_eq!(measured_allocs(|| reused.run(&x, &w, fft.as_ref())), (0, 0));
            }
        }
    }
}
