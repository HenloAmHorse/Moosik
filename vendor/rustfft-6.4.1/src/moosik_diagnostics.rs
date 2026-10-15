//! **Moosik (1.5.3 round 12-E), feature `moosik-diagnostics` only.** Global
//! counters of requested-output leaf calls, for diagnostic test builds that
//! must show which leaf path a run reached. Never compiled into an ordinary
//! or timed build: the application enables it only through its own
//! `leaf-diagnostics` feature, which is off by default.
//!
//! The counters are process-wide, so a test that asserts on them must be the
//! only analysis running in its process (run it on its own, `--exact`).

use std::sync::atomic::{AtomicU64, Ordering};

static COUNTS: [AtomicU64; 4] = [const { AtomicU64::new(0) }; 4];

/// Counts one call: 0/1 = Butterfly256 requested-output leaf / full leaf,
/// 2/3 = Butterfly512.
pub fn count(which: usize) {
    COUNTS[which].fetch_add(1, Ordering::Relaxed);
}

/// `[256 leaf, 256 full, 512 leaf, 512 full]` since the last `reset`.
pub fn counts() -> [u64; 4] {
    [0, 1, 2, 3].map(|i| COUNTS[i].load(Ordering::SeqCst))
}

pub fn reset() {
    for c in &COUNTS {
        c.store(0, Ordering::SeqCst);
    }
}

/// Pruned mixed-radix column passes taken live-row / dense (round 13 lane A).
static PASSES: [AtomicU64; 2] = [const { AtomicU64::new(0) }; 2];

pub fn pass(live: bool) {
    PASSES[if live { 0 } else { 1 }].fetch_add(1, Ordering::Relaxed);
}

/// `(live, dense)` since the last call; clears.
pub fn passes_take() -> (u64, u64) {
    (PASSES[0].swap(0, Ordering::SeqCst), PASSES[1].swap(0, Ordering::SeqCst))
}

/// One pruned mixed-radix level as it ran (R13 closeout): transform length,
/// rows (= the radix), the live-row mask, whether the live-row column pass
/// was taken, and the element size (4 = f32, 8 = f64).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Level {
    pub len: usize,
    pub rows: usize,
    pub live: u32,
    pub taken: bool,
    pub elem: usize,
}

std::thread_local! {
    /// Per thread, so a test reads only the levels its own calls ran — not
    /// another test's, whatever runs beside it. Recorded only on a thread
    /// that asked (`levels_record`): an analysis's pool threads never drain
    /// a record, so recording everywhere grew without bound.
    static LEVELS: std::cell::RefCell<Option<Vec<Level>>> = const { std::cell::RefCell::new(None) };
}

/// Counts a level's pass (process-wide), and records the level when this
/// thread is recording.
pub fn level(len: usize, rows: usize, live: u32, taken: bool, elem: usize) {
    pass(taken);
    LEVELS.with(|l| {
        if let Some(v) = l.borrow_mut().as_mut() {
            v.push(Level { len, rows, live, taken, elem });
        }
    });
}

/// Starts (cleared) or stops recording on this thread.
pub fn levels_record(on: bool) {
    LEVELS.with(|l| *l.borrow_mut() = if on { Some(Vec::new()) } else { None });
}

/// This thread's levels since the last call, outermost first per transform;
/// clears. Empty when this thread is not recording.
pub fn levels_take() -> Vec<Level> {
    LEVELS.with(|l| l.borrow_mut().as_mut().map(std::mem::take).unwrap_or_default())
}

/// A radix-4 AVX mixed-radix plan over `inner` (length `4 * inner.len()`),
/// built with the same constructor the planner uses — for tests that need a
/// geometry the planner does not choose (f64 power-of-two plans are radix-8
/// chains). `None` without AVX+FMA or for an element type other than f32/f64.
pub fn mixed_radix4<T: crate::FftNum>(inner: std::sync::Arc<dyn crate::Fft<T>>) -> Option<std::sync::Arc<dyn crate::Fft<T>>> {
    #[cfg(all(target_arch = "x86_64", feature = "avx"))]
    {
        use std::any::TypeId;
        if TypeId::of::<T>() == TypeId::of::<f64>() {
            return crate::avx::MixedRadix4xnAvx::<f64, T>::new(inner).ok().map(|f| std::sync::Arc::new(f) as std::sync::Arc<dyn crate::Fft<T>>);
        }
        if TypeId::of::<T>() == TypeId::of::<f32>() {
            return crate::avx::MixedRadix4xnAvx::<f32, T>::new(inner).ok().map(|f| std::sync::Arc::new(f) as std::sync::Arc<dyn crate::Fft<T>>);
        }
    }
    #[cfg(not(all(target_arch = "x86_64", feature = "avx")))]
    let _ = inner;
    None
}
