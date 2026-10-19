//! Opt-in measurement log. With `MOOSIK_FRAME_LOG` naming a file, one JSON
//! line a second: the window events, redraws and presents counted by the
//! patched eframe (by root window and other windows), how many passes the
//! app's own UI and the spectrum window ran, and what the spectrum window
//! last was (inner size, scale, view, the whole-track view's work so far).
//! Off, and costing one check per pass, without the variable.
//!
//! A "pass" is a run of the UI code; a "present" is a frame actually handed
//! to the window. They are counted separately because they differ.

use std::io::Write;
use std::path::PathBuf;
use std::sync::{Mutex, OnceLock};
use std::time::{Instant, SystemTime, UNIX_EPOCH};

struct State {
    path: PathBuf,
    since: Instant,
    root_passes: u64,
    spectrum_passes: u64,
    last: [[u64; 2]; eframe::moosik_counters::COUNTERS],
    spectrum: String,
}

fn state() -> Option<&'static Mutex<State>> {
    static S: OnceLock<Option<Mutex<State>>> = OnceLock::new();
    S.get_or_init(|| {
        let path = PathBuf::from(std::env::var_os("MOOSIK_FRAME_LOG")?);
        Some(Mutex::new(State {
            path,
            since: Instant::now(),
            root_passes: 0,
            spectrum_passes: 0,
            last: eframe::moosik_counters::snapshot(),
            spectrum: "null".into(),
        }))
    })
    .as_ref()
}

/// One pass of the root window's UI. Writes a line once a second.
pub fn root_pass() {
    let Some(m) = state() else { return };
    let Ok(mut s) = m.lock() else { return };
    s.root_passes += 1;
    let dt = s.since.elapsed().as_secs_f64();
    if dt < 1.0 {
        return;
    }
    let now = eframe::moosik_counters::snapshot();
    let d = |c: usize| [now[c][0] - s.last[c][0], now[c][1] - s.last[c][1]];
    let t = SystemTime::now().duration_since(UNIX_EPOCH).map_or(0, |d| d.as_millis());
    let line = format!(
        "{{\"t\":{t},\"dt\":{dt:.3},\"root_passes\":{},\"spectrum_passes\":{},\"moved\":{:?},\"resized\":{:?},\
         \"scale\":{:?},\"redraw\":{:?},\"redirected\":{:?},\"present\":{:?},\"spectrum\":{}}}\n",
        s.root_passes, s.spectrum_passes, d(0), d(1), d(2), d(3), d(4), d(5), s.spectrum
    );
    if let Ok(mut f) = std::fs::OpenOptions::new().create(true).append(true).open(&s.path) {
        let _ = f.write_all(line.as_bytes());
    }
    s.last = now;
    s.since = Instant::now();
    s.root_passes = 0;
    s.spectrum_passes = 0;
}

/// One pass of the spectrum window's UI; `note` (JSON) is what it is now.
/// `note` is only built when the log is on.
pub fn spectrum_pass(note: impl FnOnce() -> String) {
    let Some(m) = state() else { return };
    let Ok(mut s) = m.lock() else { return };
    s.spectrum_passes += 1;
    s.spectrum = note();
}
