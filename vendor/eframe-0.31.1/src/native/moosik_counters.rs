//! Moosik local patch: counts of window events, passes and presents, by
//! viewport kind (the root window, or any other). Read by Moosik's
//! measurement log; nothing here changes behaviour. See `PATCH.md`.

use std::sync::atomic::{AtomicU64, Ordering};

/// What is counted.
#[derive(Clone, Copy, Debug)]
pub enum Counter {
    /// `WindowEvent::Moved`.
    Moved,
    /// `WindowEvent::Resized`.
    Resized,
    /// `WindowEvent::ScaleFactorChanged`.
    ScaleFactor,
    /// `run_ui_and_paint` entered for this viewport (a redraw).
    Redraw,
    /// A redraw of an immediate child, handed to its parent instead.
    RedirectedToParent,
    /// A buffer swap: a frame actually presented in this viewport's window.
    Present,
}

/// How many [`Counter`]s there are.
pub const COUNTERS: usize = 6;

static COUNTS: [[AtomicU64; 2]; COUNTERS] = [const { [const { AtomicU64::new(0) }; 2] }; COUNTERS];

pub(crate) fn bump(c: Counter, root: bool) {
    COUNTS[c as usize][usize::from(!root)].fetch_add(1, Ordering::Relaxed);
}

/// Every counter so far: `[counter][0 = root, 1 = other viewports]`, in the
/// order of [`Counter`].
pub fn snapshot() -> [[u64; 2]; COUNTERS] {
    std::array::from_fn(|c| std::array::from_fn(|k| COUNTS[c][k].load(Ordering::Relaxed)))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn counts_by_kind() {
        let a = snapshot();
        bump(Counter::Present, true);
        bump(Counter::Present, false);
        bump(Counter::Present, false);
        let b = snapshot();
        assert!(b[Counter::Present as usize][0] >= a[Counter::Present as usize][0] + 1);
        assert!(b[Counter::Present as usize][1] >= a[Counter::Present as usize][1] + 2);
    }
}
