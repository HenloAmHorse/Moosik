//! Memory admission for a model run: whether this computer has room to load
//! the model now, and whether it still has room while the model runs.
//!
//! Two quantities, kept apart, both read from `GlobalMemoryStatusEx`:
//!
//! * **available physical memory** (`ullAvailPhys`): RAM that can be handed
//!   out without paging. A model run's working set comes out of this.
//! * **commit headroom** (`ullAvailPageFile`: the commit limit minus the
//!   commit charge): what can still be promised, RAM and page file together.
//!   A model run's commit comes out of this, and running out of it fails
//!   allocations system-wide, not just Moosik's.
//!
//! Graphics memory is not read here: the GPU route's dedicated and shared
//! video memory are the driver's to manage, and the runtime reports its own
//! failure to allocate them.
//!
//! The figures a run needs (`Need`) come from measurements on one computer
//! and are stated in the panel; the reserve keeps room for the desktop and
//! for playback. A reading Windows will not give is treated as "no room":
//! Moosik does not start a multi-gigabyte load it cannot account for.
//!
//! The contract, exactly: admission is one sampled reading before each model
//! load, not a reservation — memory can be taken by anything right after it.
//! During a generation the reading is repeated every `INTERVAL` and the run
//! is stopped below `FLOOR`. A reading Windows will not give refuses a load,
//! but does not stop a run already admitted (the request deadline still
//! bounds it). Tokenizer children are not watched. [`admission_text`] says
//! this to the user from the same constants.

/// What a run is admitted on, in words, from the same constants `admit` uses.
pub fn admission_text(gpu: Need, cpu: Need) -> String {
    let g = |b: u64| format!("{:.0} GiB", b as f64 / (1u64 << 30) as f64);
    format!(
        "Before each model load Moosik reads this computer's available memory once and starts only if there is at \
         least {} of available RAM and {} of commit headroom on the graphics-card route ({} and {} on the CPU-only \
         route, set conservatively above one measured sheet) — the model's need plus {} kept free. This is a check, not a reservation. \
         While the model generates, the reading is repeated every {} ms and the run is stopped if either falls below \
         {}; if Windows gives no reading, the load is refused; a missing reading does not stop a run already admitted. The \
         tokenizer steps are not watched this way.",
        g(gpu.phys + RESERVE), g(gpu.commit + RESERVE), g(cpu.phys + RESERVE), g(cpu.commit + RESERVE),
        g(RESERVE), INTERVAL.as_millis(), g(FLOOR),
    )
}

/// What one model run needs, in bytes, beyond what is already in use.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Need {
    pub phys: u64,
    pub commit: u64,
}

/// Left free for the desktop and playback, on both measures.
pub const RESERVE: u64 = 2 << 30;
/// While a run is in progress it is stopped if either measure falls below
/// this: the computer is then under pressure from something, possibly the
/// run itself, and the run is what Moosik can give back.
pub const FLOOR: u64 = 1 << 30;
/// How often a running model's headroom is read.
pub const INTERVAL: std::time::Duration = std::time::Duration::from_millis(500);

/// One reading. `None` where Windows did not answer.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Default)]
pub struct Headroom {
    pub phys: Option<u64>,
    pub commit: Option<u64>,
}

#[cfg(windows)]
fn system() -> Headroom {
    #[repr(C)]
    struct MemoryStatusEx {
        length: u32,
        load: u32,
        total_phys: u64,
        avail_phys: u64,
        total_page_file: u64,
        avail_page_file: u64,
        total_virtual: u64,
        avail_virtual: u64,
        avail_extended_virtual: u64,
    }
    #[link(name = "kernel32")]
    unsafe extern "system" {
        fn GlobalMemoryStatusEx(m: *mut MemoryStatusEx) -> i32;
    }
    let mut m = MemoryStatusEx {
        length: std::mem::size_of::<MemoryStatusEx>() as u32,
        load: 0, total_phys: 0, avail_phys: 0, total_page_file: 0, avail_page_file: 0,
        total_virtual: 0, avail_virtual: 0, avail_extended_virtual: 0,
    };
    if unsafe { GlobalMemoryStatusEx(&mut m) } == 0 {
        return Headroom::default();
    }
    Headroom { phys: Some(m.avail_phys), commit: Some(m.avail_page_file) }
}

#[cfg(not(windows))]
fn system() -> Headroom {
    Headroom::default()
}

/// The current reading (or, in a test, the injected one).
pub fn read() -> Headroom {
    #[cfg(test)]
    if let Some(h) = seam::next() {
        return h;
    }
    system()
}

pub fn gib(b: u64) -> String {
    format!("{:.1} GiB", b as f64 / (1u64 << 30) as f64)
}

/// Advice and the actual load decision share this result. Missing either
/// reading cannot establish that the model fits.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Room {
    Fits,
    Short { phys: u64, commit: u64 },
    Unknown,
}

pub fn room(need: Need, h: Headroom) -> Room {
    let (Some(p), Some(c)) = (h.phys, h.commit) else { return Room::Unknown };
    let phys = need.phys.saturating_add(RESERVE).saturating_sub(p);
    let commit = need.commit.saturating_add(RESERVE).saturating_sub(c);
    if phys == 0 && commit == 0 { Room::Fits } else { Room::Short { phys, commit } }
}

/// Includes the reserve, so displayed requirements match admission.
pub fn startup_text(need: Need, h: Headroom) -> String {
    let required = format!("Startup requires {} of available RAM and {} of commit headroom, including {} kept free",
        gib(need.phys.saturating_add(RESERVE)), gib(need.commit.saturating_add(RESERVE)), gib(RESERVE));
    match room(need, h) {
        Room::Fits => format!("{required}. Both are available now."),
        Room::Short { phys, commit } => format!("{required}. Short by {} of RAM and {} of commit headroom now.", gib(phys), gib(commit)),
        Room::Unknown => format!("{required}. Available memory could not be read; the model cannot start until a reading is available."),
    }
}

/// Whether a run needing `need` may start now. The error is the sentence for
/// the panel.
pub fn admit(need: Need) -> Result<(), String> {
    let h = read();
    match room(need, h) {
        Room::Unknown => Err("Moosik could not read how much memory this computer has free, so it did not start the local \
                    model. Nothing was applied."
            .into()),
        Room::Short { .. } => Err(format!("Not enough free memory to run the local model now. {} Nothing was started or applied. Close something large and try again.", startup_text(need, h))),
        Room::Fits => Ok(()),
    }
}

/// A reading taken during a run: `Err` with the sentence when it is too low
/// to continue. An unreadable reading mid-run does not stop the run (it was
/// admitted on a real one); the deadline still bounds it.
pub fn still_room() -> Result<(), String> {
    let h = read();
    match (h.phys, h.commit) {
        (Some(p), _) if p < FLOOR => Err(format!(
            "The computer ran low on memory ({} available) while the local model was running, so Moosik stopped it. \
             Nothing was applied.", gib(p))),
        (_, Some(c)) if c < FLOOR => Err(format!(
            "The computer ran low on commit headroom ({} left) while the local model was running, so Moosik stopped \
             it. Nothing was applied.", gib(c))),
        _ => Ok(()),
    }
}

/// Test seam: readings to return in order, per thread; the last repeats.
#[cfg(test)]
pub(crate) mod seam {
    use super::Headroom;
    use std::cell::RefCell;
    thread_local!(static READINGS: RefCell<Option<(Vec<Headroom>, usize)>> = const { RefCell::new(None) });
    pub fn set(r: Vec<Headroom>) {
        READINGS.with(|x| *x.borrow_mut() = Some((r, 0)));
    }
    pub fn clear() {
        READINGS.with(|x| *x.borrow_mut() = None);
    }
    pub(super) fn next() -> Option<Headroom> {
        READINGS.with(|x| {
            let mut x = x.borrow_mut();
            let (v, i) = x.as_mut()?;
            let h = *v.get(*i).or(v.last())?;
            *i += 1;
            Some(h)
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn startup_rule_checks_both_measures_and_unknowns() {
        let need = Need { phys: 17 << 30, commit: 4 << 30 };
        let at = Headroom { phys: Some(19 << 30), commit: Some(6 << 30) };
        assert_eq!(room(need, at), Room::Fits);
        assert_eq!(room(need, Headroom { phys: Some(18 << 30), ..at }), Room::Short { phys: 1 << 30, commit: 0 });
        assert_eq!(room(need, Headroom { commit: Some((6 << 30) - 1), ..at }), Room::Short { phys: 0, commit: 1 });
        assert_eq!(room(need, Headroom { phys: None, ..at }), Room::Unknown);
        assert_eq!(room(need, Headroom { commit: None, ..at }), Room::Unknown);
        for h in [at, Headroom { phys: Some(18 << 30), ..at }, Headroom::default()] {
            seam::set(vec![h]);
            assert_eq!(admit(need).is_ok(), room(need, h) == Room::Fits);
            if let Err(e) = admit(need)
                && room(need, h) != Room::Unknown {
                assert!(e.contains(&startup_text(need, h)), "{e}");
            }
        }
        seam::clear();
    }
}
