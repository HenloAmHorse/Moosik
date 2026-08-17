//! Session log.
//!
//! Every run writes a file under `~/.moosik/logs/`. The point is a bug report
//! from a machine nobody can reach: "it's a mess, I have zero information" is
//! not a report anyone can act on, and a console window is no help when the
//! app is a GUI, when it froze rather than crashed, or when the user already
//! closed it.
//!
//! Three properties matter, and each one is a decision:
//!
//! * **It survives.** Every line is written and flushed immediately — no
//!   `BufWriter`. A log that loses its last twenty lines loses exactly the
//!   lines describing the failure. The cost is one write syscall per event,
//!   and events are things like "device opened", not per-frame chatter.
//! * **It is timestamped from process start**, not the wall clock. When
//!   reading a session you care that the freeze began 4 s after the analysis
//!   did, not that it was 14:32 UTC. The wall clock appears once, in the
//!   banner, and in the file name.
//! * **It records the environment**, because half of what goes wrong on a
//!   strange machine is the machine. The banner is written before anything
//!   can fail, so even a log with one line in it still says what it ran on.

use std::io::Write as _;
use std::path::{Path, PathBuf};
use std::sync::{Mutex, OnceLock};
use std::time::{Instant, SystemTime, UNIX_EPOCH};

/// How many past sessions to keep. Enough to cover "it did it again yesterday",
/// small enough that nobody's home directory grows a log problem.
const KEEP: usize = 10;

struct Sink {
    file: Option<std::fs::File>,
    start: Instant,
}

static SINK: OnceLock<Mutex<Sink>> = OnceLock::new();
static PATH: OnceLock<PathBuf> = OnceLock::new();

/// Directory holding this and past session logs.
pub fn dir() -> PathBuf {
    crate::moosik_dir().join("logs")
}

/// This session's log file, once [`init`] has run.
pub fn path() -> Option<&'static Path> {
    PATH.get().map(|p| p.as_path())
}

/// Civil date from a Unix timestamp, UTC.
///
/// UTC rather than local time on purpose: a log that travels to a bug report
/// should not need the reporter's time zone to be interpreted. Algorithm is
/// Howard Hinnant's `civil_from_days`.
fn civil(secs: i64) -> (i64, u32, u32, u32, u32, u32) {
    let days = secs.div_euclid(86_400);
    let rem = secs.rem_euclid(86_400);
    let (hh, mm, ss) = ((rem / 3600) as u32, ((rem % 3600) / 60) as u32, (rem % 60) as u32);
    let z = days + 719_468;
    let era = z.div_euclid(146_097);
    let doe = z.rem_euclid(146_097);
    let yoe = (doe - doe / 1460 + doe / 36_524 - doe / 146_096) / 365;
    let doy = doe - (365 * yoe + yoe / 4 - yoe / 100);
    let mp = (5 * doy + 2) / 153;
    let d = (doy - (153 * mp + 2) / 5 + 1) as u32;
    let m = (if mp < 10 { mp + 3 } else { mp - 9 }) as u32;
    let y = yoe + era * 400 + if m <= 2 { 1 } else { 0 };
    (y, m, d, hh, mm, ss)
}

fn now_unix() -> i64 {
    SystemTime::now().duration_since(UNIX_EPOCH).map(|d| d.as_secs() as i64).unwrap_or(0)
}

/// Delete all but the newest [`KEEP`] logs. Best-effort: a log directory we
/// cannot tidy is not a reason to start the session without a log.
fn prune(d: &Path) {
    let Ok(rd) = std::fs::read_dir(d) else { return };
    let mut files: Vec<PathBuf> = rd
        .flatten()
        .map(|e| e.path())
        .filter(|p| p.extension().is_some_and(|e| e == "log"))
        .collect();
    if files.len() <= KEEP { return }
    // The names sort chronologically by construction, so no metadata call.
    files.sort();
    for old in &files[..files.len() - KEEP] {
        let _ = std::fs::remove_file(old);
    }
}

/// Open this session's log and start recording. Safe to call once; later calls
/// do nothing.
///
/// Called before anything else in `main` so that a failure during start-up —
/// the hardest kind to report, because the user has nothing on screen yet —
/// still lands in a file.
pub fn init() {
    if SINK.get().is_some() { return }

    let d = dir();
    let _ = std::fs::create_dir_all(&d);
    prune(&d);

    let (y, mo, dy, h, mi, s) = civil(now_unix());
    let name = format!("moosik-{y:04}{mo:02}{dy:02}-{h:02}{mi:02}{s:02}.log");
    let file = std::fs::File::create(d.join(&name)).ok();
    let _ = PATH.set(d.join(&name));
    let _ = SINK.set(Mutex::new(Sink { file, start: Instant::now() }));

    banner(y, mo, dy, h, mi, s);
}

fn banner(y: i64, mo: u32, dy: u32, h: u32, mi: u32, s: u32) {
    write_line(&format!("Moosik {} — session log", env!("CARGO_PKG_VERSION")));
    write_line(&format!("started   {y:04}-{mo:02}-{dy:02} {h:02}:{mi:02}:{s:02} UTC"));
    write_line(&format!("os        {} {}", std::env::consts::OS, std::env::consts::ARCH));
    write_line(&format!(
        "cores     {}",
        std::thread::available_parallelism().map(|n| n.get()).unwrap_or(0)
    ));
    write_line(&format!(
        "exe       {}",
        std::env::current_exe().map(|p| p.display().to_string()).unwrap_or_else(|_| "?".into())
    ));
    write_line(&format!(
        "profile   {}",
        if cfg!(debug_assertions) { "debug" } else { "release" }
    ));
    if let Some(p) = path() {
        // Also on stderr, so anyone running from a console knows where to look.
        eprintln!("[moosik] logging to {}", p.display());
    }
    write_line("---");
}

/// Append one line. Used by the [`mlog!`] macro; call that instead.
pub fn write_line(msg: &str) {
    let Some(sink) = SINK.get() else {
        // Before init, or after a poisoning — still better than silence.
        eprintln!("{msg}");
        return;
    };
    let Ok(mut g) = sink.lock() else { return };
    let t = g.start.elapsed().as_secs_f64();
    let line = format!("[{t:9.3}] {msg}");
    eprintln!("{line}");
    if let Some(f) = g.file.as_mut() {
        // Written and flushed per line on purpose — see the module docs.
        let _ = writeln!(f, "{line}");
        let _ = f.flush();
    }
}

/// Record an event in the session log and on stderr.
///
/// For events, not frames: device negotiation, track loads, analysis
/// boundaries, anything that failed. Something that happens sixty times a
/// second belongs behind a counter, not in here.
#[macro_export]
macro_rules! mlog {
    ($($arg:tt)*) => { $crate::log::write_line(&format!($($arg)*)) };
}

/// Route panics into the session log, with a backtrace, before the normal
/// stderr handler runs.
///
/// The previous version of this wrote a bare one-line record to `crash.log`
/// with no backtrace, which identifies *that* something panicked but not the
/// path that reached it. A panic on a machine the developer cannot touch is
/// exactly the case where the backtrace is the whole report.
pub fn install_panic_hook() {
    let default = std::panic::take_hook();
    std::panic::set_hook(Box::new(move |info| {
        let loc = info
            .location()
            .map(|l| format!("{}:{}:{}", l.file(), l.line(), l.column()))
            .unwrap_or_else(|| "?".into());
        let msg = info
            .payload()
            .downcast_ref::<&str>()
            .map(|s| (*s).to_string())
            .or_else(|| info.payload().downcast_ref::<String>().cloned())
            .unwrap_or_else(|| "<non-string panic payload>".into());
        let thread = std::thread::current().name().unwrap_or("unnamed").to_string();

        write_line(&format!("PANIC on '{thread}' at {loc}: {msg}"));
        // force_capture ignores RUST_BACKTRACE, which a user double-clicking an
        // exe has certainly not set — and their run is the one that matters.
        let bt = std::backtrace::Backtrace::force_capture();
        for line in bt.to_string().lines() {
            write_line(&format!("    {line}"));
        }
        default(info);
    }));
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The date arithmetic is the one part here that can be silently wrong for
    /// years at a time, so it gets pinned against known values.
    #[test]
    fn civil_dates_match_known_timestamps() {
        assert_eq!(civil(0), (1970, 1, 1, 0, 0, 0));
        assert_eq!(civil(946_684_799), (1999, 12, 31, 23, 59, 59));
        assert_eq!(civil(946_684_800), (2000, 1, 1, 0, 0, 0));
        // 29 Feb 2024 — the leap-year case the naive version gets wrong.
        assert_eq!(civil(1_709_164_800), (2024, 2, 29, 0, 0, 0));
        assert_eq!(civil(1_755_331_200), (2025, 8, 16, 8, 0, 0));
    }

    /// Names must sort chronologically, because `prune` sorts by name rather
    /// than asking the filesystem for modification times.
    #[test]
    fn log_names_sort_in_time_order() {
        let name = |s: i64| {
            let (y, mo, d, h, mi, sec) = civil(s);
            format!("moosik-{y:04}{mo:02}{d:02}-{h:02}{mi:02}{sec:02}.log")
        };
        let mut v = vec![name(1_755_331_200), name(0), name(946_684_800)];
        v.sort();
        assert_eq!(v, vec![name(0), name(946_684_800), name(1_755_331_200)]);
    }
}
