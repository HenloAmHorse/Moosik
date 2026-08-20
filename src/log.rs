//! Session log.
//!
//! Every run tries to create a file under `~/.moosik/logs/`; when that fails
//! the reason is recorded and the UI says so rather than offering a path. The point is a bug report
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
//!
//! Retention is **single-process**. Two copies of Moosik running at once each
//! prune this directory and can remove a log the other is still writing;
//! doing that safely needs a lock the operating system releases on crash.

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
static ERROR: OnceLock<String> = OnceLock::new();

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
    let (hh, mm, ss) = (
        (rem / 3600) as u32,
        ((rem % 3600) / 60) as u32,
        (rem % 60) as u32,
    );
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
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|d| d.as_secs() as i64)
        .unwrap_or(0)
}

/// What retention actually managed to do.
///
/// Returned rather than assumed, and deliberately not reduced to a single
/// number. A directory that refuses a delete, or a scan that failed halfway,
/// leaves more files than the policy nominally allows, and reporting "exactly
/// ten" in either case is a claim the code cannot support.
#[derive(Debug, Default, PartialEq, Eq)]
pub struct Retention {
    /// Session logs seen by the final rescan, or `None` if that rescan failed.
    ///
    /// `None` rather than a fabricated number: the previous version reported
    /// `1` when it could not read the directory, which is a measurement nobody
    /// took. Even when present this is a snapshot — another process may add or
    /// remove a log the instant after. See [`Retention::exact`].
    pub kept: Option<usize>,
    pub removed: usize,
    /// Deletes that failed for a reason other than the file already being gone.
    pub failed: usize,
    /// Deletes that found nothing. Another process reached the same end state
    /// first, which is success by someone else's hand, not a failure.
    pub already_gone: usize,
    /// Why the directory could not be read, if it could not.
    pub scan_error: Option<String>,
    /// True when this run resolved everything it attempted and the current log
    /// still existed at the final rescan.
    ///
    /// Still only a best-effort snapshot of a directory other processes may be
    /// writing to — it says this run left nothing unresolved, not that the
    /// directory now holds exactly [`KEEP`] files and will continue to.
    pub exact: bool,
}

/// True for Moosik's current and accepted legacy reserved session-log schemas.
///
/// It does **not** prove Moosik produced the file — another program could write
/// the same name. What it establishes is that the name belongs to a schema this
/// module reserves, validated rather than pattern-matched: the date and time
/// must be a real calendar instant, the process id and any collision suffix
/// must be canonical decimal, and the suffix must lie in the range the creator
/// can emit. A shape-only check would have accepted `moosik-99999999-999999-…`
/// and made it a deletion candidate; a `parse::<u32>()`-only check would have
/// accepted `+1` and `0001`, which this module never writes.
///
/// `the_stamp_creator_and_validator_agree_on_supported_stamps` holds this and
/// the creator together for representative supported stamps, and exhaustively
/// across the collision suffix range.
///
/// The legacy `moosik-<stamp>.log` form is accepted too, so upgrading does not
/// strand the previous version's logs as files nothing will ever clean up.
fn is_session_log(name: &str) -> bool {
    let Some(rest) = name.strip_prefix("moosik-") else {
        return false;
    };
    let Some(rest) = rest.strip_suffix(".log") else {
        return false;
    };
    let mut parts = rest.split('-');

    let (Some(date), Some(time)) = (parts.next(), parts.next()) else {
        return false;
    };
    if date.len() != 8 || time.len() != 6 {
        return false;
    }
    if !date.bytes().chain(time.bytes()).all(|b| b.is_ascii_digit()) {
        return false;
    }
    // Canonical decimal only. `parse::<u32>` also accepts "+1", "0001" and
    // "01" — spellings `create_session_file` cannot emit — so a round trip
    // through `to_string` is what keeps the check to the reserved schema.
    let num = |s: &str| {
        (!s.is_empty() && s.bytes().all(|b| b.is_ascii_digit()))
            .then(|| s.parse::<u32>().ok())
            .flatten()
            .filter(|v| v.to_string() == s)
    };
    // Zero-padded by construction, so these are parsed as plain fields.
    let field = |s: &str| s.parse::<u32>().ok();
    let (Some(y), Some(mo), Some(d)) = (field(&date[0..4]), field(&date[4..6]), field(&date[6..8]))
    else {
        return false;
    };
    let (Some(h), Some(mi), Some(sec)) =
        (field(&time[0..2]), field(&time[2..4]), field(&time[4..6]))
    else {
        return false;
    };
    if !(1..=12).contains(&mo) || d < 1 || d > days_in_month(y, mo) {
        return false;
    }
    if h > 23 || mi > 59 || sec > 59 {
        return false;
    }

    // Then nothing (legacy), a pid, or a pid and a collision counter.
    match (parts.next(), parts.next(), parts.next()) {
        (None, _, _) => true,
        (Some(pid), None, _) => num(pid).is_some(),
        (Some(pid), Some(n), None) => {
            num(pid).is_some() && num(n).is_some_and(|n| (1..MAX_COLLISION_SUFFIX).contains(&n))
        }
        _ => false,
    }
}

/// One past the highest collision suffix [`create_session_file`] will try.
const MAX_COLLISION_SUFFIX: u32 = 64;

fn days_in_month(year: u32, month: u32) -> u32 {
    match month {
        1 | 3 | 5 | 7 | 8 | 10 | 12 => 31,
        4 | 6 | 9 | 11 => 30,
        2 if year.is_multiple_of(4) && (!year.is_multiple_of(100) || year.is_multiple_of(400)) => {
            29
        }
        2 => 28,
        _ => 0,
    }
}

/// Every session log in `d`, sorted, or why the directory could not be read.
///
/// Entry and `file_type` failures are returned rather than skipped. Pruning
/// decisions are made from this list, so an entry that could not be classified
/// is missing evidence, not an absent file.
fn session_logs(d: &Path) -> Result<Vec<PathBuf>, String> {
    let rd = std::fs::read_dir(d).map_err(|e| format!("{}: {e}", d.display()))?;
    let mut files = Vec::new();
    for entry in rd {
        let entry = entry.map_err(|e| format!("{}: {e}", d.display()))?;
        let ft = entry
            .file_type()
            .map_err(|e| format!("{}: {e}", entry.path().display()))?;
        if !ft.is_file() {
            continue;
        }
        let path = entry.path();
        if path
            .file_name()
            .and_then(|n| n.to_str())
            .is_some_and(is_session_log)
        {
            files.push(path);
        }
    }
    files.sort();
    Ok(files)
}

/// Keep the newest [`KEEP`] session logs, counting `current`, and never
/// consider `current` itself for deletion.
///
/// Two mistakes are being avoided. Pruning *before* creating the new file made
/// the steady state `KEEP + 1` and deleted a real log before finding out
/// whether its replacement could be opened. Pruning by name alone made the new
/// file a deletion candidate: names sort chronologically only down to the
/// second, and the process id disambiguating a same-second collision is not
/// ordered, so `...-120000-100.log` sorts before an older `...-120000-999.log`.
///
/// **This is single-process retention only.** Every running copy of Moosik
/// prunes the same directory and computes the same "oldest" set, so a second
/// instance can still delete a log a live peer is appending to. Fixing that
/// needs an OS-owned lock the kernel releases on crash — a directory marker
/// with a timeout is not sufficient, and a `Drop` on a `static` never runs at
/// all. Tracked as open work rather than approximated here.
fn prune(d: &Path, current: &Path) -> Retention {
    prune_inner(d, current, || {})
}

/// `prune`, with a hook that runs between enumeration and the first unlink.
///
/// The hook exists so the window another process can act in is a real,
/// deterministic point in the code rather than something a test approximates by
/// doing its work beforehand — which tests nothing, because the scan never sees
/// the files.
fn prune_inner(d: &Path, current: &Path, after_scan: impl FnOnce()) -> Retention {
    let mut out = Retention::default();
    let files = match session_logs(d) {
        Ok(f) => f,
        Err(e) => {
            out.scan_error = Some(e);
            return out;
        }
    };

    let eligible: Vec<&PathBuf> = files.iter().filter(|p| p.as_path() != current).collect();
    let budget = KEEP.saturating_sub(1); // `current` occupies one slot

    after_scan();

    if eligible.len() > budget {
        for old in &eligible[..eligible.len() - budget] {
            match std::fs::remove_file(old) {
                Ok(()) => out.removed += 1,
                // Someone else already removed it. The end state is the one we
                // wanted; counting it as a failure would misreport a success.
                Err(e) if e.kind() == std::io::ErrorKind::NotFound => out.already_gone += 1,
                // One refusal must not abandon the rest: a single locked file
                // would otherwise let the directory grow without bound.
                Err(_) => out.failed += 1,
            }
        }
    }

    match session_logs(d) {
        Ok(after) => {
            out.kept = Some(after.len());
            out.exact = out.failed == 0 && out.scan_error.is_none() && current.exists();
        }
        Err(e) => out.scan_error = Some(e),
    }
    out
}

/// The session stamp, in one place.
///
/// Production and the seam test call this rather than each spelling the format
/// out, so the shared formatter and
/// `the_stamp_creator_and_validator_agree_on_supported_stamps` detect
/// supported-schema drift for representative stamps between what `init` emits
/// and what `is_session_log` will accept.
fn session_stamp(y: i64, mo: u32, d: u32, h: u32, mi: u32, s: u32) -> String {
    format!("{y:04}{mo:02}{d:02}-{h:02}{mi:02}{s:02}")
}

/// Create this session's log file, or say why it could not be created.
///
/// `create_new` rather than `create`: the name carries only whole seconds, so
/// two instances launched together would otherwise open the same path and
/// truncate each other's records. On a collision this walks a suffix rather
/// than overwriting, and the process id keeps separate instances apart even
/// when the clock does not.
///
/// The timestamp leads, so names sort chronologically *across* seconds. Within
/// one second the pid and suffix decide the order and neither is chronological
/// — which is why retention excludes the current log by path rather than
/// trusting its position in a sorted list.
fn create_session_file(d: &Path, stamp: &str) -> std::io::Result<(std::fs::File, PathBuf)> {
    let pid = std::process::id();
    let mut last_err = None;
    for n in 0..MAX_COLLISION_SUFFIX {
        let name = if n == 0 {
            format!("moosik-{stamp}-{pid}.log")
        } else {
            format!("moosik-{stamp}-{pid}-{n}.log")
        };
        let path = d.join(name);
        match std::fs::OpenOptions::new()
            .write(true)
            .create_new(true)
            .open(&path)
        {
            Ok(f) => return Ok((f, path)),
            Err(e) if e.kind() == std::io::ErrorKind::AlreadyExists => {
                last_err = Some(e);
            }
            Err(e) => return Err(e),
        }
    }
    Err(last_err
        .unwrap_or_else(|| std::io::Error::other("could not find an unused session log name")))
}

/// Everything [`init`] decided, in a form that can be asserted on.
#[derive(Debug)]
pub struct LogInitOutcome {
    pub path: Option<PathBuf>,
    pub error: Option<String>,
    pub retention: Retention,
}

/// Create the session log and apply retention. Pure with respect to globals so
/// it can be driven against a scratch directory in tests.
fn open_session(d: &Path, stamp: &str) -> (Option<std::fs::File>, LogInitOutcome) {
    match std::fs::create_dir_all(d).and_then(|()| create_session_file(d, stamp)) {
        Ok((f, path)) => {
            // Only now, with a real file in hand, is it safe to remove any.
            let retention = prune(d, &path);
            (
                Some(f),
                LogInitOutcome {
                    path: Some(path),
                    error: None,
                    retention,
                },
            )
        }
        Err(e) => (
            None,
            LogInitOutcome {
                path: None,
                error: Some(format!("{}: {e}", d.display())),
                retention: Retention::default(),
            },
        ),
    }
}

/// Open this session's log and start recording. Safe to call once; later calls
/// do nothing.
///
/// Called before anything else in `main` so that a failure during start-up —
/// the hardest kind to report, because the user has nothing on screen yet —
/// still lands in a file.
pub fn init() {
    if SINK.get().is_some() {
        return;
    }

    let d = dir();
    let (y, mo, dy, h, mi, sec) = civil(now_unix());
    let stamp = session_stamp(y, mo, dy, h, mi, sec);
    let (file, outcome) = open_session(&d, &stamp);

    // No path is published unless a file exists. Publishing one regardless is
    // how the UI came to offer a "Log" button pointing at nothing.
    if let Some(ref p) = outcome.path {
        let _ = PATH.set(p.clone());
    }
    if let Some(ref e) = outcome.error {
        let _ = ERROR.set(e.clone());
    }

    let _ = SINK.set(Mutex::new(Sink {
        file,
        start: Instant::now(),
    }));
    banner(y, mo, dy, h, mi, sec);
    if let Some(e) = outcome.error {
        write_line(&format!("WARNING no log file — {e}"));
    }
    let r = &outcome.retention;
    if let Some(ref e) = r.scan_error {
        write_line(&format!("WARNING log directory scan failed — {e}"));
    }
    if r.failed > 0 {
        write_line(&format!(
            "WARNING {} old log(s) could not be removed",
            r.failed
        ));
    }
    if let Some(kept) = r.kept
        && !r.exact
    {
        write_line(&format!(
            "note    {kept} log(s) present; retention left work unfinished"
        ));
    }
}

/// Why this session has no log file, if it has none.
///
/// The UI asks before offering to open one, so a failure is visible rather
/// than a button that does nothing.
pub fn error() -> Option<&'static str> {
    ERROR.get().map(|s| s.as_str())
}

fn banner(y: i64, mo: u32, dy: u32, h: u32, mi: u32, s: u32) {
    write_line(&format!(
        "Moosik {} — session log",
        env!("CARGO_PKG_VERSION")
    ));
    write_line(&format!(
        "started   {y:04}-{mo:02}-{dy:02} {h:02}:{mi:02}:{s:02} UTC"
    ));
    write_line(&format!(
        "os        {} {}",
        std::env::consts::OS,
        std::env::consts::ARCH
    ));
    write_line(&format!(
        "cores     {}",
        std::thread::available_parallelism()
            .map(|n| n.get())
            .unwrap_or(0)
    ));
    write_line(&format!(
        "exe       {}",
        std::env::current_exe()
            .map(|p| p.display().to_string())
            .unwrap_or_else(|_| "?".into())
    ));
    write_line(&format!(
        "profile   {}",
        if cfg!(debug_assertions) {
            "debug"
        } else {
            "release"
        }
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
        let thread = std::thread::current()
            .name()
            .unwrap_or("unnamed")
            .to_string();

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
    /// than asking the filesystem for modification times. The timestamp leads,
    /// so ordering holds *across* seconds; within one second the pid decides
    /// and is not chronological, which is why retention excludes the current
    /// log by path rather than trusting its position in the sorted list.
    #[test]
    fn log_names_sort_in_time_order() {
        let name = |s: i64, pid: u32| {
            let (y, mo, d, h, mi, sec) = civil(s);
            format!("moosik-{y:04}{mo:02}{d:02}-{h:02}{mi:02}{sec:02}-{pid}.log")
        };
        let mut v = vec![
            name(1_755_331_200, 2),
            name(0, 99_999),
            name(946_684_800, 1),
        ];
        v.sort();
        assert_eq!(
            v,
            vec![
                name(0, 99_999),
                name(946_684_800, 1),
                name(1_755_331_200, 2)
            ]
        );
    }

    fn scratch(tag: &str) -> PathBuf {
        let d = std::env::temp_dir().join(format!("moosik-log-test-{}-{tag}", std::process::id()));
        let _ = std::fs::remove_dir_all(&d);
        std::fs::create_dir_all(&d).unwrap();
        d
    }

    fn names_in(d: &Path) -> Vec<String> {
        let mut v: Vec<String> = std::fs::read_dir(d)
            .unwrap()
            .flatten()
            .map(|e| e.file_name().to_string_lossy().into_owned())
            .collect();
        v.sort();
        v
    }

    /// Session log *files*. Directories are excluded here for the same reason
    /// `prune` excludes them — a directory named like a log is not one.
    fn sessions_in(d: &Path) -> Vec<String> {
        let mut v: Vec<String> = std::fs::read_dir(d)
            .unwrap()
            .flatten()
            .filter(|e| e.file_type().map(|t| t.is_file()).unwrap_or(false))
            .map(|e| e.file_name().to_string_lossy().into_owned())
            .filter(|n| is_session_log(n))
            .collect();
        v.sort();
        v
    }

    /// Only files this module could have written are deletion candidates.
    #[test]
    fn the_name_filter_matches_our_logs_and_nothing_else() {
        for ok in [
            "moosik-20260818-120000.log",      // legacy
            "moosik-20260818-120000-1234.log", // current
            "moosik-20260818-120000-1234-7.log",
        ] {
            // collision suffix
            assert!(is_session_log(ok), "{ok} should be recognised");
        }
        for no in [
            "notes.txt",
            "other.log",
            "moosik.log",
            "moosik-.log",
            "moosik-2026081-120000-1.log", // short date
            "moosik-20260818-12000-1.log", // short time
            "moosik-20260818-120000-abc.log",
            "moosik-20260818-120000-1-2-3.log",
            "moosik-20260818-120000-1234.log.bak",
            // Shape alone is not enough: this predicate authorises
            // deletion, so the instant has to be a real one.
            "moosik-99999999-999999-1.log",
            "moosik-20261318-120000-1.log",           // month 13
            "moosik-20260800-120000-1.log",           // day 0
            "moosik-20260231-120000-1.log",           // 31 February
            "moosik-20260229-120000-1.log",           // 2026 is not a leap year
            "moosik-20260818-240000-1.log",           // hour 24
            "moosik-20260818-126000-1.log",           // minute 60
            "moosik-20260818-120099-1.log",           // second 99
            "moosik-20260818-120000-99999999999.log", // pid past u32
            "moosik-20260818-120000-1234-0.log",      // suffix 0 is never emitted
            "moosik-20260818-120000-1234-64.log",     // nor is 64
            // Spellings `parse::<u32>` accepts but the creator never
            // writes. Without the canonical round trip these were
            // deletion candidates.
            "moosik-20260818-120000-+1.log",
            "moosik-20260818-120000-0001.log",
            "moosik-20260818-120000-1234-01.log",
            "moosik-20260818-120000-1234-+2.log",
        ] {
            assert!(
                !is_session_log(no),
                "{no} must not be treated as a session log"
            );
        }
        // The leap day itself is valid.
        assert!(is_session_log("moosik-20240229-120000-1.log"));
        // Pid 0 is canonical decimal, so it is accepted; only non-canonical
        // spellings are rejected.
        assert!(is_session_log("moosik-20260818-120000-0.log"));
        // Every suffix the creator can emit must survive the round trip.
        for n in 1..MAX_COLLISION_SUFFIX {
            assert!(
                is_session_log(&format!("moosik-20260818-120000-7-{n}.log")),
                "suffix {n}"
            );
        }
    }

    /// The production stamp formatter feeds the creator, and the resulting
    /// names are accepted by the validator that authorises deletion.
    ///
    /// If the formatter changes shape, or the creator's suffix range moves, or
    /// the validator tightens, a log Moosik wrote would stop being eligible for
    /// retention and the directory would grow without bound — and nothing else
    /// would notice.
    ///
    /// Coverage is uneven by design: representative supported four-digit stamps
    /// (not the full output domain of `civil(i64)`, and not the arbitrary
    /// strings `create_session_file` will accept), but the collision suffix
    /// range is covered exhaustively.
    #[test]
    fn the_stamp_creator_and_validator_agree_on_supported_stamps() {
        let d = scratch("seam");

        for (y, mo, dy, h, mi, sec) in [
            (1970, 1, 1, 0, 0, 0),
            (2024, 2, 29, 23, 59, 59), // leap day, last second
            (2026, 8, 18, 12, 0, 0),
            (9999, 12, 31, 23, 59, 59),
        ] {
            let stamp = session_stamp(y, mo, dy, h, mi, sec);
            let (_f, path) = create_session_file(&d, &stamp).unwrap();
            let name = path.file_name().unwrap().to_str().unwrap();
            assert!(
                is_session_log(name),
                "formatter produced {stamp}, creator wrote {name}, validator rejected it"
            );
        }

        // Exhaust one stamp: the base name plus every collision suffix the
        // creator can emit. `create_new` means each created file already
        // reserves its name, so no handle needs holding. Each gets a distinct
        // sentinel so "still present" can be checked as "still the file we
        // wrote".
        let stamp = session_stamp(2026, 1, 1, 0, 0, 0);
        let pid = std::process::id();
        let mut created = Vec::new();
        let mut produced = std::collections::BTreeSet::new();
        for i in 0..MAX_COLLISION_SUFFIX {
            let (mut f, path) = create_session_file(&d, &stamp)
                .unwrap_or_else(|e| panic!("creation {i} of {MAX_COLLISION_SUFFIX} failed: {e}"));
            let name = path.file_name().unwrap().to_str().unwrap().to_string();
            assert!(
                is_session_log(&name),
                "creator wrote {name}, validator rejected it"
            );
            writeln!(f, "sentinel {i}").unwrap();
            f.flush().unwrap();
            produced.insert(name);
            created.push((path, i));
        }

        // The exact set, not merely the right count.
        let expected: std::collections::BTreeSet<String> =
            std::iter::once(format!("moosik-{stamp}-{pid}.log"))
                .chain((1..MAX_COLLISION_SUFFIX).map(|n| format!("moosik-{stamp}-{pid}-{n}.log")))
                .collect();
        assert_eq!(produced, expected, "creator did not emit exactly base plus 1..=63");

        // One more must fail, and specifically because every name is taken.
        let err = create_session_file(&d, &stamp)
            .expect_err("the range is exhausted; creation must fail, not reuse a name");
        assert_eq!(
            err.kind(),
            std::io::ErrorKind::AlreadyExists,
            "exhaustion must surface as AlreadyExists, got {err}"
        );

        // Nothing was clobbered: each file still holds its own sentinel.
        for (path, i) in &created {
            let got = std::fs::read_to_string(path)
                .unwrap_or_else(|e| panic!("{} disappeared during exhaustion: {e}", path.display()));
            assert_eq!(
                got,
                format!("sentinel {i}
"),
                "{} was overwritten",
                path.display()
            );
        }

        let _ = std::fs::remove_dir_all(&d);
    }

    /// Two instances starting inside the same second must not share a file.
    ///
    /// The name carries whole seconds only, and the old code used
    /// `File::create`, which truncates. The second process would silently wipe
    /// the first one's log and the two would then interleave into it.
    #[test]
    fn a_second_session_in_the_same_second_gets_its_own_file() {
        let d = scratch("collision");
        let (mut a, pa) = create_session_file(&d, "20260818-120000").unwrap();
        writeln!(a, "first").unwrap();
        let (mut b, pb) = create_session_file(&d, "20260818-120000").unwrap();
        writeln!(b, "second").unwrap();

        assert_ne!(pa, pb, "same-second sessions must not share a path");
        assert_eq!(
            std::fs::read_to_string(&pa).unwrap().trim(),
            "first",
            "the first session's log must survive intact"
        );
        assert_eq!(std::fs::read_to_string(&pb).unwrap().trim(), "second");
        let _ = std::fs::remove_dir_all(&d);
    }

    /// Retention counts the session being written and never deletes it.
    #[test]
    fn retention_keeps_the_current_session_and_lands_on_the_documented_count() {
        for existing in [0usize, 9, 10, 11, 40] {
            let d = scratch(&format!("prune{existing}"));
            for i in 0..existing {
                std::fs::write(
                    d.join(format!(
                        "moosik-202608{:02}-1200{:02}-7.log",
                        i % 28 + 1,
                        i % 60
                    )),
                    "x",
                )
                .unwrap();
            }
            // Things retention must not touch.
            std::fs::write(d.join("notes.txt"), "keep").unwrap();
            std::fs::write(d.join("other.log"), "keep").unwrap();
            std::fs::create_dir(d.join("moosik-20260101-000000-1.log")).unwrap();

            let (_f, out) = open_session(&d, "20260819-235959");
            let path = out.path.clone().expect("create must succeed");

            assert!(
                path.exists(),
                "the current session was deleted ({existing} existing)"
            );
            assert_eq!(out.retention.failed, 0);
            assert!(
                out.retention.exact,
                "nothing was unresolved, so the count is exact"
            );
            assert_eq!(
                out.retention.kept,
                Some((existing + 1).min(KEEP)),
                "{existing} existing -> want min(existing+1, {KEEP}) kept",
            );
            assert_eq!(
                Some(sessions_in(&d).len()),
                out.retention.kept,
                "report must match disk"
            );
            let names = names_in(&d);
            assert!(names.iter().any(|n| n == "notes.txt"));
            assert!(names.iter().any(|n| n == "other.log"));
            assert!(
                d.join("moosik-20260101-000000-1.log").is_dir(),
                "directory deleted"
            );
            let _ = std::fs::remove_dir_all(&d);
        }
    }

    /// The case name-sorting alone gets wrong: within one second the process id
    /// decides the order, and it is not chronological. A current log whose pid
    /// sorts first was previously the one chosen for deletion.
    #[test]
    fn a_current_log_that_sorts_first_is_still_not_deleted() {
        let d = scratch("sortfirst");
        for pid in 900..(900 + KEEP + 5) {
            std::fs::write(d.join(format!("moosik-20260818-120000-{pid}.log")), "x").unwrap();
        }
        let current = d.join("moosik-20260818-120000-100.log");
        std::fs::write(&current, "current").unwrap();
        assert_eq!(
            sessions_in(&d).first().unwrap(),
            "moosik-20260818-120000-100.log",
            "test setup: the current log must sort first"
        );

        let out = prune(&d, &current);
        assert!(
            current.exists(),
            "the current session must survive retention"
        );
        assert_eq!(std::fs::read_to_string(&current).unwrap(), "current");
        assert_eq!(out.kept, Some(KEEP));
        let _ = std::fs::remove_dir_all(&d);
    }

    /// A peer that removes a file *after* this pruner has listed it, but
    /// before the unlink, must be counted as already-done rather than failed.
    ///
    /// The previous version of this test deleted the files before calling
    /// `prune`, so the scan never saw them and the `NotFound` arm never ran. It
    /// asserted its own setup. The hook puts the removal in the actual window.
    #[test]
    fn a_removal_racing_the_unlink_is_counted_as_done_not_failed() {
        let d = scratch("race");
        for i in 0..KEEP + 6 {
            std::fs::write(
                d.join(format!("moosik-202608{:02}-1200{:02}-7.log", i % 28 + 1, i)),
                "x",
            )
            .unwrap();
        }
        let current = d.join("moosik-20260901-000000-1.log");
        std::fs::write(&current, "current").unwrap();

        let doomed: Vec<PathBuf> = session_logs(&d)
            .unwrap()
            .into_iter()
            .filter(|p| p != &current)
            .take(6)
            .collect();

        // Runs after enumeration, before the first unlink: exactly the window a
        // second pruner would act in.
        let out = prune_inner(&d, &current, || {
            for p in &doomed {
                std::fs::remove_file(p).unwrap();
            }
        });

        assert!(
            out.already_gone >= 1,
            "the NotFound arm must actually be exercised"
        );
        assert_eq!(out.failed, 0, "a file already gone is not a failed delete");
        assert!(current.exists());
        assert_eq!(out.kept, Some(sessions_in(&d).len()));
        let _ = std::fs::remove_dir_all(&d);
    }

    /// A delete the filesystem refuses must not be reported as success, must
    /// not abort the remaining candidates, and must make the count inexact.
    ///
    /// The refusal is produced by holding the file open without sharing delete
    /// rights, which is a real lock. Setting the read-only attribute does not
    /// work: `remove_file` clears it on Windows and deletes anyway.
    #[cfg(windows)]
    #[test]
    fn a_refused_delete_is_reported_and_does_not_stop_the_rest() {
        use std::os::windows::fs::OpenOptionsExt;
        const FILE_SHARE_READ: u32 = 0x0000_0001;

        let d = scratch("refused");
        for i in 0..KEEP + 6 {
            std::fs::write(
                d.join(format!("moosik-202608{:02}-1200{:02}-7.log", i % 28 + 1, i)),
                "x",
            )
            .unwrap();
        }
        // Oldest, so first in line for deletion.
        let locked = session_logs(&d).unwrap().into_iter().next().unwrap();
        let _held = std::fs::OpenOptions::new()
            .read(true)
            .share_mode(FILE_SHARE_READ) // no FILE_SHARE_DELETE
            .open(&locked)
            .unwrap();

        let current = d.join("moosik-20260901-000000-1.log");
        std::fs::write(&current, "current").unwrap();
        let out = prune(&d, &current);

        assert_eq!(out.failed, 1, "the refused delete must be reported");
        assert!(
            !out.exact,
            "a failed delete means the count cannot be called exact"
        );
        assert!(locked.exists(), "the locked file must survive");
        assert!(
            out.removed >= 1,
            "the other candidates must still be attempted"
        );
        assert!(current.exists());
        assert_eq!(
            out.kept,
            Some(sessions_in(&d).len()),
            "reported count must match a real rescan"
        );

        drop(_held);
        let _ = std::fs::remove_dir_all(&d);
    }

    /// A directory that cannot be scanned is surfaced, not swallowed.
    #[test]
    fn a_scan_failure_is_reported_rather_than_treated_as_an_empty_directory() {
        let d = scratch("scanfail");
        let not_a_dir = d.join("regular-file");
        std::fs::write(&not_a_dir, "x").unwrap();
        let out = prune(&not_a_dir, &not_a_dir.join("moosik-20260101-000000-1.log"));
        assert!(
            out.scan_error.is_some(),
            "the scan failure must be reported"
        );
        assert!(
            !out.exact,
            "nothing can be claimed exactly after a failed scan"
        );
        assert_eq!(
            out.kept, None,
            "a count nobody measured must not be invented"
        );
        assert_eq!(out.removed, 0);
        let _ = std::fs::remove_dir_all(&d);
    }

    /// Legacy and current names sharing a timestamp both count, and the older
    /// shape is eligible for removal rather than accumulating forever.
    #[test]
    fn legacy_and_current_names_are_retained_together() {
        let d = scratch("legacy");
        for i in 0..8 {
            std::fs::write(d.join(format!("moosik-202608{:02}-120000.log", i + 1)), "x").unwrap();
        }
        for i in 0..8 {
            std::fs::write(
                d.join(format!("moosik-202608{:02}-120000-5-{}.log", i + 10, i + 1)),
                "x",
            )
            .unwrap();
        }
        let (_f, out) = open_session(&d, "20260901-000000");
        assert_eq!(out.retention.kept, Some(KEEP));
        assert!(out.retention.exact);
        assert_eq!(sessions_in(&d).len(), KEEP);
        assert!(out.path.unwrap().exists());
        let _ = std::fs::remove_dir_all(&d);
    }

    /// A directory that cannot be written yields an error and no path. The UI
    /// offers the log button on the strength of that path.
    #[test]
    fn a_failed_create_reports_an_error_rather_than_a_path() {
        let d = scratch("nodir");
        let missing = d.join("f.txt");
        std::fs::write(&missing, "not a directory").unwrap();
        // create_dir_all over a regular file fails, so open_session must too.
        let (file, out) = open_session(&missing, "20260818-120000");
        assert!(file.is_none());
        assert!(
            out.path.is_none(),
            "no path may be published without a file"
        );
        assert!(out.error.is_some(), "the failure must be reported");
        assert_eq!(out.retention, Retention::default());
        let _ = std::fs::remove_dir_all(&d);
    }
}
