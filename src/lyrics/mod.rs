//! Lyrics: LRC parsing, sidecar/embedded loading, and online lookup.
//!
//! Lyrics live in a `.lrc` file beside the track rather than inside its tags.
//! That is deliberate. A sidecar needs no rewrite of the audio file, which
//! matters most for DSD — DSF carries its ID3 blob behind a header pointer, so
//! writing tags there means rebuilding the header and a bug eats the audio
//! rather than the metadata. A sidecar is also hand-editable and is what every
//! other player already reads, so fixing a track here fixes it everywhere.
//!
//! Embedded lyrics are still *read* (lofty normalises `USLT`, Vorbis `LYRICS`
//! and the iTunes atom to one key), they are simply never written.

pub mod http;
pub mod lrclib;
pub mod matching;
pub mod netease;
pub mod romaji;
pub mod ui;

pub use matching::Hit;

/// A source's automatic lookup: artist, title, album, duration.
type SourceFind =
    fn(&str, &str, &str, Option<std::time::Duration>) -> Result<Option<Hit>, String>;

/// Look a track up across every source, in order, stopping at the first
/// confident match.
///
/// LRCLIB first: it is purpose-built for this, needs no key, and its records
/// are contributed by people syncing lyrics deliberately. NetEase second, where
/// it earns its place — for a Japanese or Vocaloid library LRCLIB is mostly
/// empty and NetEase mostly is not.
///
/// A source that errors does not stop the ladder. One service being down or
/// unreachable is not a reason to report "no lyrics" when another might have
/// them; the error is only surfaced if *every* source failed and none matched.
pub fn find_anywhere(
    artist: &str, title: &str, album: &str, duration: Option<std::time::Duration>,
) -> Result<Option<Hit>, String> {
    let sources: [(&str, SourceFind); 2] =
        [(lrclib::NAME, lrclib::find), (netease::NAME, netease::find)];

    let mut errors: Vec<String> = Vec::new();
    for (name, f) in sources {
        match f(artist, title, album, duration) {
            Ok(Some(h)) => return Ok(Some(h)),
            Ok(None) => {}
            Err(e) => errors.push(format!("{name}: {e}")),
        }
    }
    // "Not found" is only honest when every source actually answered. If one
    // was rate-limited or unreachable, saying the track has no lyrics sends the
    // user off to transcribe something a service would have handed over a
    // minute later — so report what went wrong instead.
    if !errors.is_empty() { return Err(errors.join("; ")); }
    Ok(None)
}

/// Every source's free-text search, pooled, for the manual picker.
///
/// Unlike [`find_anywhere`] this applies no acceptance gate: the user is doing
/// the choosing, and hiding plausible rows would only get in the way. Rows are
/// labelled with where they came from.
///
/// Returns the rows plus a note naming any source that could not answer. The
/// note matters: if one source is rate-limited and the other simply has
/// nothing, a bare short list reads as "no database has this song" when half
/// the databases were never asked.
pub fn search_anywhere(query: &str) -> Result<(Vec<Hit>, Option<String>), String> {
    let mut out = Vec::new();
    let mut errors = Vec::new();
    match lrclib::search(query) {
        Ok(h) => out.extend(h),
        Err(e) => errors.push(format!("{}: {e}", lrclib::NAME)),
    }
    match netease::search(query) {
        Ok(h) => out.extend(h),
        Err(e) => errors.push(format!("{}: {e}", netease::NAME)),
    }
    if out.is_empty() && !errors.is_empty() { return Err(errors.join("; ")); }
    let note = (!errors.is_empty()).then(|| errors.join("; "));
    Ok((out, note))
}

use std::path::{Path, PathBuf};
use std::time::Duration;

/// One line of a lyric sheet. `at` is `None` for unsynced lyrics.
#[derive(Clone, Debug, PartialEq, Default)]
pub struct LyricLine {
    pub at: Option<Duration>,
    pub text: String,
    /// Romaji someone vouched for — a source supplied it, or the user wrote
    /// it. `None` means "generate it", which is what lets the UI tell a
    /// guess from a correction.
    pub romaji: Option<String>,
    /// A translation, when the source has one.
    pub translation: Option<String>,
    /// Word timing from enhanced LRC (`<mm:ss.xx>` inside the line): a byte
    /// offset into `text` and when that part is sung, measured from `at`. From
    /// the line rather than the song, so retiming the line keeps its words in
    /// step with it.
    pub marks: Vec<(usize, Duration)>,
    /// The user confirmed that their romaji replaces furigana written into
    /// the sheet on this line. Holds the line's text as it was when they
    /// confirmed, and counts only while the line still reads exactly that
    /// ([`LyricLine::overrides_sheet`]): an override belongs to one lyric
    /// line, not to whatever line lands on its timestamp after an edit.
    ///
    /// Separate from `romaji` on purpose. Romaji a source supplied and romaji
    /// the user confirmed over the sheet are different things; only the
    /// second may displace what the sheet says.
    pub reading_override: Option<String>,
}

impl LyricLine {
    /// The user's confirmed reading replaces the sheet's own furigana here.
    pub fn overrides_sheet(&self) -> bool {
        self.romaji.is_some() && self.reading_override.as_deref() == Some(self.text.as_str())
    }
}

/// Something kept beside a sheet in its own file, line for line with it.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum Extra {
    Romaji,
    Translation,
    /// Confirmed readings that replace the sheet's furigana.
    Override,
}

impl Extra {
    pub const ALL: [Extra; 3] = [Extra::Romaji, Extra::Translation, Extra::Override];

    fn field(self, l: &mut LyricLine) -> &mut Option<String> {
        match self {
            Extra::Romaji => &mut l.romaji,
            Extra::Translation => &mut l.translation,
            Extra::Override => &mut l.reading_override,
        }
    }
    /// What is saved for this line. An override is saved only while it is in
    /// force: one left behind by retyping the line is dropped, not kept on
    /// disk to be matched against later.
    fn get(self, l: &LyricLine) -> Option<&String> {
        match self {
            Extra::Romaji => l.romaji.as_ref(),
            Extra::Translation => l.translation.as_ref(),
            Extra::Override => l.reading_override.as_ref().filter(|_| l.overrides_sheet()),
        }
    }
}

/// Where a sheet came from, for the UI to label and for deciding whether
/// "Save" would overwrite something the user put there by hand.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum LyricSource {
    /// A `.lrc` beside the track.
    Sidecar,
    /// A tag inside the audio file.
    Embedded,
    /// Fetched online, not yet saved.
    Online,
}

#[derive(Clone, Debug)]
pub struct Lyrics {
    pub lines: Vec<LyricLine>,
    pub source: LyricSource,
    /// Milliseconds to shift playback position by when highlighting, from the
    /// sheet's own `[offset:]` tag plus anything the user dialled in. Positive
    /// makes lines appear earlier, which is the LRC convention.
    pub offset_ms: i64,
    /// Title/artist from the sheet's own metadata tags, when it carried any.
    pub title: Option<String>,
    pub artist: Option<String>,
}

impl Lyrics {
    /// True when at least one line carries a timestamp. A sheet where only
    /// some lines are timed still counts — the untimed ones simply never
    /// highlight, which is better than refusing to show the sheet at all.
    pub fn is_synced(&self) -> bool {
        self.lines.iter().any(|l| l.at.is_some())
    }

    pub fn is_empty(&self) -> bool {
        self.lines.iter().all(|l| l.text.trim().is_empty())
    }

    /// Index of the line that should be highlighted at `pos`, i.e. the last one
    /// whose timestamp has passed.
    ///
    /// Untimed lines inherit the highlight state of the timed line above them,
    /// so a sheet that mixes the two does not flicker back to "nothing active"
    /// on every untimed line.
    pub fn active_index(&self, pos: Duration) -> Option<usize> {
        if !self.is_synced() { return None; }
        let pos_ms = pos.as_millis() as i64 + self.offset_ms;
        let mut best = None;
        for (i, line) in self.lines.iter().enumerate() {
            match line.at {
                Some(t) if (t.as_millis() as i64) <= pos_ms => best = Some(i),
                Some(_) => break,
                // Untimed line between timed ones: keep it selectable so the
                // highlight walks through it rather than jumping over.
                None if best.is_some() => best = Some(i),
                None => {}
            }
        }
        best
    }

    /// How far into line `i` the singing has got at `pos`, as a byte offset
    /// into its text — from the sheet's own word timing only. A line without
    /// any has no karaoke: a guessed sweep never matches the singing, and
    /// looks worse than none.
    pub fn sung(&self, i: usize, pos: Duration) -> Option<usize> {
        let line = self.lines.get(i)?;
        if line.marks.is_empty() { return None; }
        let now = pos.as_millis() as i64 + self.offset_ms - line.at?.as_millis() as i64;
        Some(line.marks.iter()
            .find(|m| m.1.as_millis() as i64 > now)
            .map_or(line.text.len(), |m| m.0))
    }

    /// True when some line carries word timing, i.e. karaoke has anything to show.
    pub fn has_word_timing(&self) -> bool {
        self.lines.iter().any(|l| !l.marks.is_empty())
    }

    /// Serialise back to LRC. Round-trips [`parse`] for synced sheets.
    pub fn to_lrc(&self) -> String {
        let mut out = String::new();
        if let Some(t) = &self.title  { out.push_str(&format!("[ti:{t}]\n")); }
        if let Some(a) = &self.artist { out.push_str(&format!("[ar:{a}]\n")); }
        if self.offset_ms != 0 { out.push_str(&format!("[offset:{}]\n", self.offset_ms)); }
        if !out.is_empty() { out.push('\n'); }
        for line in &self.lines {
            match line.at {
                Some(t) => out.push_str(&format!("[{}]{}\n", fmt_stamp(t), with_marks(line))),
                None => { out.push_str(&line.text); out.push('\n'); }
            }
        }
        out
    }
}

/// `mm:ss.cc`, the form every LRC reader accepts — or `mm:ss.mmm` when the
/// time is not a whole centisecond, which [`parse_stamp`] reads back exactly.
/// Rounding to centiseconds would move a word mark by up to 9 ms, more than a
/// frame at high refresh rates. Minutes are not wrapped at 60 — a 70-minute
/// track is `[70:xx.xx]`, which is what other players write.
pub fn fmt_stamp(t: Duration) -> String {
    let ms = t.as_millis() as u64;
    let (m, s) = (ms / 60_000, (ms / 1000) % 60);
    if ms.is_multiple_of(10) {
        format!("{m:02}:{s:02}.{:02}", (ms % 1000) / 10)
    } else {
        format!("{m:02}:{s:02}.{:03}", ms % 1000)
    }
}

/// Parse `[mm:ss]`, `[mm:ss.xx]` or `[mm:ss.xxx]`. Returns `None` for anything
/// else in brackets, which is then treated as a metadata tag.
fn parse_stamp(body: &str) -> Option<Duration> {
    let (mins, rest) = body.split_once(':')?;
    let mins: u64 = mins.trim().parse().ok()?;
    let (secs, frac) = match rest.split_once(['.', ':']) {
        Some((s, f)) => (s, Some(f)),
        None => (rest, None),
    };
    let secs: u64 = secs.trim().parse().ok()?;
    if secs >= 60 { return None; }
    let ms = match frac {
        None => 0,
        Some(f) => {
            let f = f.trim();
            if f.is_empty() || !f.bytes().all(|b| b.is_ascii_digit()) { return None; }
            // Two digits are centiseconds, three are milliseconds. Anything
            // longer is truncated rather than rejected.
            let v: u64 = f[..f.len().min(3)].parse().ok()?;
            match f.len() {
                1 => v * 100,
                2 => v * 10,
                _ => v,
            }
        }
    };
    Some(Duration::from_millis(mins * 60_000 + secs * 1000 + ms))
}

/// Parse an LRC sheet, or plain text if it carries no timestamps.
///
/// Handles the things real sheets do: several timestamps on one line for a
/// repeated refrain, metadata tags mixed in, an `[offset:]`, and word-level
/// `<mm:ss.xx>` marks from enhanced LRC (stripped — the line-level timing is
/// what gets drawn).
pub fn parse(text: &str, source: LyricSource) -> Lyrics {
    let mut lines: Vec<LyricLine> = Vec::new();
    let mut offset_ms = 0i64;
    let (mut title, mut artist) = (None, None);

    for raw in text.lines() {
        let mut rest = raw.trim_start();
        let mut stamps: Vec<Duration> = Vec::new();
        let mut meta_only = false;

        // Peel leading bracket groups. Timestamps accumulate; anything else is
        // a metadata tag, and a line that is *only* metadata produces no lyric.
        while rest.starts_with('[') {
            let Some(end) = rest.find(']') else { break };
            let body = &rest[1..end];
            if let Some(d) = parse_stamp(body) {
                stamps.push(d);
            } else if let Some((k, v)) = body.split_once(':') {
                let v = v.trim().to_string();
                match k.trim().to_ascii_lowercase().as_str() {
                    "offset" => offset_ms = v.replace('+', "").trim().parse().unwrap_or(0),
                    "ti" => title = Some(v).filter(|s| !s.is_empty()),
                    "ar" => artist = Some(v).filter(|s| !s.is_empty()),
                    _ => {}
                }
                if stamps.is_empty() { meta_only = true; }
            } else {
                break;
            }
            rest = &rest[end + 1..];
        }

        let (stripped, marks) = strip_word_marks(rest);
        let lead = stripped.len() - stripped.trim_start().len();
        let text = stripped.trim().to_string();
        if stamps.is_empty() {
            // A metadata-only line contributes nothing; a blank line in an
            // unsynced sheet is a real paragraph break and is kept.
            if !meta_only { lines.push(LyricLine { text, ..Default::default() }); }
        } else {
            // Word marks are song times; kept relative to the line's own, so
            // a refrain written once with several times carries them to each.
            let marks: Vec<(usize, Duration)> = marks.into_iter()
                .map(|(o, t)| (o.saturating_sub(lead).min(text.len()), t.saturating_sub(stamps[0])))
                .collect();
            for at in stamps {
                lines.push(LyricLine {
                    at: Some(at), text: text.clone(), marks: marks.clone(), ..Default::default()
                });
            }
        }
    }

    // Several timestamps on one source line put the refrain out of order.
    // Untimed lines keep their position relative to the timed line before them.
    if lines.iter().any(|l| l.at.is_some()) {
        // Blank untimed lines around a synced sheet are layout, not content —
        // the gap under a metadata header, or the file's final newline. Both
        // sort to the front (`None` < `Some`) and would otherwise show as an
        // empty first line that can never highlight.
        lines.retain(|l| l.at.is_some() || !l.text.is_empty());
        // A refrain written once with all of its times comes out of order, so
        // the sheet has to be sorted. An *untimed* line in a partly-synced
        // sheet must sort with the timed line above it, not to the very front:
        // it belongs where the author put it, and `None < Some` would otherwise
        // pile every not-yet-timed line at the top — which is exactly what a
        // half-finished sync looks like. The sort is stable, so equal keys keep
        // their written order.
        let mut carry: Option<Duration> = None;
        let keys: Vec<Option<Duration>> = lines.iter()
            .map(|l| { if l.at.is_some() { carry = l.at; } carry })
            .collect();
        let mut idx: Vec<usize> = (0..lines.len()).collect();
        idx.sort_by_key(|&i| keys[i]);
        lines = idx.into_iter().map(|i| lines[i].clone()).collect();
    } else {
        while lines.last().is_some_and(|l| l.text.is_empty()) { lines.pop(); }
    }

    Lyrics { lines, source, offset_ms, title, artist }
}

/// Take enhanced-LRC word timings (`<00:12.34>`) out of a line's text,
/// returning where each fell in what is left. Anything else in angle brackets
/// is text — `<3` is a lyric, not a timestamp.
fn strip_word_marks(s: &str) -> (String, Vec<(usize, Duration)>) {
    let mut out = String::with_capacity(s.len());
    let mut marks = Vec::new();
    let mut rest = s;
    while let Some(open) = rest.find('<') {
        out.push_str(&rest[..open]);
        let after = &rest[open + 1..];
        match after.find('>').and_then(|close| parse_stamp(&after[..close]).map(|t| (close, t))) {
            Some((close, t)) => { marks.push((out.len(), t)); rest = &after[close + 1..]; }
            None => { out.push('<'); rest = after; }
        }
    }
    out.push_str(rest);
    (out, marks)
}

/// A line's text with its word marks written back in.
fn with_marks(line: &LyricLine) -> String {
    let Some(at) = line.at else { return line.text.clone() };
    let mut out = String::new();
    let mut last = 0;
    for &(o, t) in &line.marks {
        if o < last || !line.text.is_char_boundary(o) { continue; }
        out.push_str(&line.text[last..o]);
        out.push_str(&format!("<{}>", fmt_stamp(at + t)));
        last = o;
    }
    out.push_str(&line.text[last..]);
    out
}

/// Path of the `.lrc` that belongs to `track` — the same name with the
/// extension swapped. Returned whether or not it exists, so callers can use it
/// as the save target too.
pub fn sidecar_path(track: &Path) -> PathBuf {
    track.with_extension("lrc")
}

/// Load lyrics for a track: the sidecar first, then whatever the tags carry,
/// with any saved romaji and translation attached.
///
/// The sidecar wins because it is the one the user can edit, and anything this
/// app saves lands there — so a saved fix must not be shadowed by a stale tag.
pub fn load(track: &Path, embedded: Option<&str>) -> Option<Lyrics> {
    let mut l = load_sheet(track, embedded)?;
    for e in Extra::ALL {
        if let Ok(text) = std::fs::read_to_string(extra_path(track, e)) {
            attach(&mut l.lines, &text, e);
        }
    }
    Some(l)
}

fn load_sheet(track: &Path, embedded: Option<&str>) -> Option<Lyrics> {
    let sidecar = sidecar_path(track);
    if let Ok(text) = std::fs::read_to_string(&sidecar) {
        let l = parse(&text, LyricSource::Sidecar);
        if !l.is_empty() { return Some(l); }
    }
    let l = parse(embedded?, LyricSource::Embedded);
    (!l.is_empty()).then_some(l)
}

/// Where romaji or a translation for `track` is kept: `song.romaji.lrc` and
/// `song.translation.lrc` beside `song.lrc`.
///
/// Their own files, so `song.lrc` stays the plain sheet every other player
/// expects. Each mirrors that sheet line for line with the same timestamps,
/// which also makes it a usable LRC in its own right.
pub fn extra_path(track: &Path, e: Extra) -> PathBuf {
    track.with_extension(match e {
        Extra::Romaji => "romaji.lrc",
        Extra::Translation => "translation.lrc",
        Extra::Override => "reading-override.lrc",
    })
}

/// Pair a romaji or translation sheet onto `lines`. A blank line means "not
/// supplied"; for romaji, that leaves the line to be generated.
///
/// What [`save_extra`] wrote pairs line for line: same count, same stamps.
/// That is checked on the raw file, not the parsed one, because parsing drops
/// blank untimed lines — exactly the blanks a partly-synced sheet's romaji has.
/// Anything else (a source whose sheet skips the credits, a sheet retimed
/// since) pairs timed lines by timestamp; an unsynced one only has order.
pub fn attach(lines: &mut [LyricLine], text: &str, e: Extra) {
    let cs = |d: Option<Duration>| d.map(|d| d.as_millis() / 10);
    let set = |l: &mut LyricLine, t: &str| {
        if !t.trim().is_empty() { *e.field(l) = Some(t.trim().to_string()); }
    };
    let raw: Vec<LyricLine> = text.lines()
        .map(|r| parse(r, LyricSource::Sidecar).lines.into_iter().next().unwrap_or_default())
        .collect();
    if raw.len() == lines.len() && raw.iter().zip(lines.iter()).all(|(r, l)| cs(r.at) == cs(l.at)) {
        for (l, r) in lines.iter_mut().zip(&raw) { set(l, &r.text); }
        return;
    }
    let sheet = parse(text, LyricSource::Sidecar);
    if !sheet.is_synced() {
        for (l, r) in lines.iter_mut().zip(&sheet.lines) { set(l, &r.text); }
        return;
    }
    let mut used = vec![false; sheet.lines.len()];
    for l in lines.iter_mut().filter(|l| l.at.is_some()) {
        if let Some(j) = (0..sheet.lines.len())
            .find(|&j| !used[j] && cs(sheet.lines[j].at) == cs(l.at))
        {
            used[j] = true;
            set(l, &sheet.lines[j].text);
        }
    }
}

/// Lyrics stored inside the audio file's tags, if any.
///
/// lofty normalises ID3's `USLT`, the Vorbis `LYRICS`/`UNSYNCEDLYRICS` comments
/// and the iTunes `©lyr` atom onto one key, so this covers every container the
/// player opens — including DSD, whose ID3 blob `probe_tagged` already knows how
/// to reach.
pub fn embedded_text(path: &Path) -> Option<String> {
    use lofty::prelude::{ItemKey, TaggedFileExt as _};
    let tagged = crate::probe_tagged(path)?;
    let tag = tagged.primary_tag().or_else(|| tagged.first_tag())?;
    tag.get_string(&ItemKey::Lyrics)
        .filter(|s| !s.trim().is_empty())
        .map(str::to_string)
}

/// A romaji or translation sidecar's contents: every line of the sheet, with
/// its timestamp, blank where there is none — for romaji, where it is still
/// generated, so a guess is never saved as though someone had checked it.
pub fn render_extra(lyrics: &Lyrics, e: Extra) -> String {
    Lyrics {
        lines: lyrics.lines.iter().map(|l| LyricLine {
            at: l.at, text: e.get(l).cloned().unwrap_or_default(), ..Default::default()
        }).collect(),
        source: lyrics.source,
        offset_ms: 0,
        title: None,
        artist: None,
    }.to_lrc()
}

/// Write one extra on its own. Tests use it to lay files down; the app saves
/// through [`save_all`].
#[cfg(test)]
pub fn save_extra(track: &Path, lyrics: &Lyrics, e: Extra) -> std::io::Result<PathBuf> {
    let path = extra_path(track, e);
    std::fs::write(&path, render_extra(lyrics, e))?;
    Ok(path)
}

/// How a save ended. There are exactly three outcomes, and the UI reports
/// each as it is.
#[derive(Debug)]
pub enum SaveOutcome {
    /// Every file now holds what was saved. `backups` are copies of extras
    /// that were overwritten or removed, kept so nothing hand-written is lost.
    Saved { written: Vec<PathBuf>, removed: Vec<PathBuf>, backups: Vec<PathBuf> },
    /// Something failed and every file that had changed was put back byte for
    /// byte. Disk is as it was before Save; retrying is safe.
    RolledBack { error: std::io::Error },
    /// Something failed and putting it back failed too. `stuck` names each
    /// live file that could not be restored; `backups` hold every previous
    /// version, so nothing is lost, but the files on disk do not match each
    /// other until a Save succeeds.
    NotUndone {
        error: std::io::Error,
        stuck: Vec<(PathBuf, std::io::Error)>,
        backups: Vec<PathBuf>,
    },
}

/// A point in a save where a test may make it fail.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum Stage {
    /// About to claim this name for a backup or temporary file.
    Reserve,
    /// About to copy this file's current contents to its backup.
    Backup,
    /// This file has just been replaced or removed.
    Applied,
    /// About to put this file back from its backup.
    Restore,
}

/// Save the sheet and whichever extras it affects, all or nothing.
///
/// `sheet`: the sheet changed — by fetch, paste, or the Sync editor, it makes
/// no difference. Then both extras are in scope: they pair with the sheet by
/// timestamp, so whatever is on disk must be rewritten to match it, or removed
/// if the sheet now has none, or it will reattach to whichever new line lands
/// on an old time. `romaji`: the romaji alone changed.
///
/// **Nothing hand-written is discarded.** An extra whose bytes would be
/// removed or changed is first copied to a new backup (`song.romaji.lrc.old`,
/// then `.old.2`, …, never over an existing file), and the backup is kept.
///
/// **All or nothing, file by file.** Every file about to change is backed up
/// before any is touched. Each is replaced by writing a temporary file and
/// renaming it over the old one, so no file is ever half-written. If any step
/// fails, the files already changed are restored from their backups, newest
/// first, and checked byte for byte. This is not crash atomicity across files:
/// a power cut between two renames leaves them mismatched, with every previous
/// version still in its backup.
pub fn save_all(track: &Path, lyrics: &Lyrics, sheet: bool, romaji: bool) -> SaveOutcome {
    save_all_with(track, lyrics, sheet, romaji, &mut |_, _| Ok(()))
}

pub(crate) fn save_all_with(
    track: &Path, lyrics: &Lyrics, sheet: bool, romaji: bool,
    hook: &mut dyn FnMut(Stage, &Path) -> std::io::Result<()>,
) -> SaveOutcome {
    struct Change { path: PathBuf, new: Option<String>, existed: bool, backup: Option<PathBuf>, extra: bool }

    // One save of a track at a time. A second one would back up files the
    // first is halfway through replacing, and each could roll back over the
    // other. Refused before anything is touched.
    let Some(_guard) = SaveGuard::acquire(track) else {
        return SaveOutcome::RolledBack {
            error: std::io::Error::new(std::io::ErrorKind::WouldBlock,
                                       "another save of this track is still in progress"),
        };
    };

    // What each file should hold afterwards. `None` means it should not exist.
    let mut wanted: Vec<(PathBuf, Option<String>, bool)> = Vec::new();
    if sheet {
        wanted.push((sidecar_path(track), Some(lyrics.to_lrc()), false));
    }
    for e in Extra::ALL {
        // A romaji edit can confirm or reset an override, so the override
        // file travels with the romaji.
        if sheet || (romaji && e != Extra::Translation) {
            let any = lyrics.lines.iter().any(|l| e.get(l).is_some());
            wanted.push((extra_path(track, e), any.then(|| render_extra(lyrics, e)), true));
        }
    }
    let mut changes: Vec<Change> = Vec::new();
    for (path, new, extra) in wanted {
        // Only "not found" means absent. Any other failure to read is a file
        // that exists and cannot be vouched for: treating it as missing would
        // leave it in place, reported as saved, to reattach on the next load.
        let old = match std::fs::read(&path) {
            Ok(b) => Some(b),
            Err(e) if e.kind() == std::io::ErrorKind::NotFound => None,
            Err(e) => {
                let name = path.file_name().unwrap_or_default().to_string_lossy().into_owned();
                return SaveOutcome::RolledBack {
                    error: std::io::Error::new(e.kind(), format!("could not read {name}: {e}")),
                };
            }
        };
        if old.as_deref() == new.as_deref().map(str::as_bytes) {
            continue; // already exactly this
        }
        changes.push(Change { path, new, existed: old.is_some(), backup: None, extra });
    }

    // Back up everything that will change before touching anything. A failure
    // here has changed nothing live, so the copies made so far go too.
    for i in 0..changes.len() {
        if !changes[i].existed { continue; }
        let r = backup(&changes[i].path, hook);
        match r {
            Ok(to) => changes[i].backup = Some(to),
            Err(error) => {
                for c in &changes[..i] {
                    if let Some(b) = &c.backup { let _ = std::fs::remove_file(b); }
                }
                return SaveOutcome::RolledBack { error };
            }
        }
    }

    // Apply, in order: sheet, romaji, translation.
    let mut applied = 0;
    let mut failure = None;
    for c in &changes {
        let r = match &c.new {
            Some(text) => replace_file(&c.path, text.as_bytes(), hook),
            None => std::fs::remove_file(&c.path),
        };
        // A step counts as applied once the live file has changed, even if
        // what comes after it fails.
        if let Err(e) = r { failure = Some(e); break; }
        applied += 1;
        if let Err(e) = hook(Stage::Applied, &c.path) { failure = Some(e); break; }
    }

    let backups_of = |changes: &[Change], all: bool| -> Vec<PathBuf> {
        changes.iter().filter(|c| all || c.extra).filter_map(|c| c.backup.clone()).collect()
    };
    let Some(error) = failure else {
        // The sheet's own backup was only there to roll back with.
        for c in changes.iter().filter(|c| !c.extra) {
            if let Some(b) = &c.backup { let _ = std::fs::remove_file(b); }
        }
        let written = changes.iter().filter(|c| c.new.is_some()).map(|c| c.path.clone()).collect();
        let removed = changes.iter().filter(|c| c.new.is_none()).map(|c| c.path.clone()).collect();
        return SaveOutcome::Saved { written, removed, backups: backups_of(&changes, false) };
    };

    // Roll back what changed, newest first, and check it.
    let mut stuck = Vec::new();
    for c in changes[..applied].iter().rev() {
        let r = hook(Stage::Restore, &c.path).and_then(|_| match &c.backup {
            Some(b) => {
                let bytes = std::fs::read(b)?;
                replace_file(&c.path, &bytes, hook)?;
                if std::fs::read(&c.path)? != bytes {
                    return Err(std::io::Error::other("restored file does not match its backup"));
                }
                Ok(())
            }
            None => std::fs::remove_file(&c.path),
        });
        if let Err(e) = r { stuck.push((c.path.clone(), e)); }
    }
    if stuck.is_empty() {
        // Disk is exactly as it was; the copies are redundant.
        for c in &changes {
            if let Some(b) = &c.backup { let _ = std::fs::remove_file(b); }
        }
        SaveOutcome::RolledBack { error }
    } else {
        SaveOutcome::NotUndone { error, stuck, backups: backups_of(&changes, true) }
    }
}

/// Replace `path` with `bytes` by writing a temporary file beside it and
/// renaming it over, so the file is never seen half-written.
fn replace_file(
    path: &Path, bytes: &[u8], hook: &mut dyn FnMut(Stage, &Path) -> std::io::Result<()>,
) -> std::io::Result<()> {
    use std::io::Write as _;
    let (tmp, mut f) = reserve(&format!("{}.saving", path.display()), hook)?;
    let r = f.write_all(bytes).and_then(|_| f.sync_all())
        .and_then(|_| { drop(f); std::fs::rename(&tmp, path) });
    if r.is_err() { let _ = std::fs::remove_file(&tmp); }
    r
}

/// Copy `path` to a new backup beside it — `p.old`, then `p.old.2`, … — and
/// return where it went.
fn backup(
    path: &Path, hook: &mut dyn FnMut(Stage, &Path) -> std::io::Result<()>,
) -> std::io::Result<PathBuf> {
    hook(Stage::Backup, path)?;
    let (to, mut f) = reserve(&format!("{}.old", path.display()), hook)?;
    let r = std::fs::File::open(path)
        .and_then(|mut src| std::io::copy(&mut src, &mut f))
        .and_then(|_| f.sync_all());
    if let Err(e) = r {
        drop(f);
        let _ = std::fs::remove_file(&to);
        return Err(e);
    }
    Ok(to)
}

/// Claim a new file named `base`, then `base.2`, … and return it open.
///
/// `create_new` makes the claim and the check one step: a name that appears
/// after we looked is simply skipped, never written over.
fn reserve(
    base: &str, hook: &mut dyn FnMut(Stage, &Path) -> std::io::Result<()>,
) -> std::io::Result<(PathBuf, std::fs::File)> {
    for n in 1..=10_000u32 {
        let p = PathBuf::from(if n == 1 { base.to_string() } else { format!("{base}.{n}") });
        hook(Stage::Reserve, &p)?;
        match std::fs::OpenOptions::new().write(true).create_new(true).open(&p) {
            Ok(f) => return Ok((p, f)),
            Err(e) if e.kind() == std::io::ErrorKind::AlreadyExists => continue,
            Err(e) => return Err(e),
        }
    }
    Err(std::io::Error::other(format!("no free name beside {base}")))
}

/// This process's claim to be saving a track, released when dropped — on
/// every return path, success or failure.
///
/// Per process only. Two copies of the app saving one track at the same
/// moment are not excluded; every file is still replaced whole and every
/// replaced file still backed up first, but the two could interleave.
struct SaveGuard(PathBuf);

impl SaveGuard {
    fn active() -> &'static std::sync::Mutex<std::collections::HashSet<PathBuf>> {
        static A: std::sync::OnceLock<std::sync::Mutex<std::collections::HashSet<PathBuf>>> =
            std::sync::OnceLock::new();
        A.get_or_init(Default::default)
    }

    fn acquire(track: &Path) -> Option<SaveGuard> {
        let key = std::path::absolute(track).unwrap_or_else(|_| track.to_path_buf());
        let mut set = Self::active().lock().unwrap_or_else(|p| p.into_inner());
        set.insert(key.clone()).then(|| SaveGuard(key))
    }
}

impl Drop for SaveGuard {
    fn drop(&mut self) {
        Self::active().lock().unwrap_or_else(|p| p.into_inner()).remove(&self.0);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn ms(m: u64) -> Option<Duration> { Some(Duration::from_millis(m)) }

    #[test]
    fn parses_a_plain_synced_sheet() {
        let l = parse("[00:12.34]hello\n[01:05.00]world\n", LyricSource::Sidecar);
        assert!(l.is_synced());
        assert_eq!(l.lines.len(), 2);
        assert_eq!(l.lines[0].at, ms(12_340));
        assert_eq!(l.lines[0].text, "hello");
        assert_eq!(l.lines[1].at, ms(65_000));
    }

    #[test]
    fn accepts_every_timestamp_precision() {
        // Two digits are centiseconds, three are milliseconds, none is whole
        // seconds — all three appear in sheets found in the wild.
        let l = parse("[00:01]a\n[00:02.5]b\n[00:03.25]c\n[00:04.125]d\n", LyricSource::Sidecar);
        let at: Vec<_> = l.lines.iter().map(|x| x.at).collect();
        assert_eq!(at, vec![ms(1000), ms(2500), ms(3250), ms(4125)]);
    }

    #[test]
    fn one_line_may_carry_several_timestamps() {
        // A refrain is written once with every time it recurs.
        let l = parse("[00:10.00][01:10.00][02:10.00]chorus\n[00:20.00]verse\n",
                      LyricSource::Sidecar);
        assert_eq!(l.lines.len(), 4);
        // And the result must come out in playback order, not source order.
        assert_eq!(l.lines[0].at, ms(10_000));
        assert_eq!(l.lines[1].at, ms(20_000));
        assert_eq!(l.lines[1].text, "verse");
        assert_eq!(l.lines[3].at, ms(130_000));
    }

    #[test]
    fn reads_metadata_and_offset() {
        let l = parse("[ti:Song]\n[ar:Someone]\n[offset:+500]\n[00:01.00]x\n",
                      LyricSource::Sidecar);
        assert_eq!(l.title.as_deref(), Some("Song"));
        assert_eq!(l.artist.as_deref(), Some("Someone"));
        assert_eq!(l.offset_ms, 500);
        // Metadata lines are not lyrics.
        assert_eq!(l.lines.len(), 1);
    }

    #[test]
    fn unsynced_text_survives_intact() {
        let l = parse("first line\n\nsecond line\n", LyricSource::Sidecar);
        assert!(!l.is_synced());
        assert_eq!(l.lines.len(), 3);
        assert_eq!(l.lines[1].text, "");
        assert!(l.active_index(Duration::from_secs(30)).is_none());
    }

    /// A half-finished sync: the untimed tail must stay at the bottom, where
    /// the author left it, not jump to the top because `None < Some`.
    #[test]
    fn untimed_lines_stay_below_the_timed_line_above_them() {
        let l = parse("[00:01.00]one\n[00:05.00]two\nthree\nfour\n", LyricSource::Sidecar);
        let texts: Vec<&str> = l.lines.iter().map(|x| x.text.as_str()).collect();
        assert_eq!(texts, vec!["one", "two", "three", "four"]);
        // An untimed line before the first timestamp does belong at the front.
        let l = parse("intro\n[00:05.00]two\n", LyricSource::Sidecar);
        assert_eq!(l.lines[0].text, "intro");
    }

    #[test]
    fn strips_enhanced_word_timings() {
        let l = parse("[00:01.00]<00:01.00>ka<00:01.50>ra<00:02.00>ke\n", LyricSource::Sidecar);
        assert_eq!(l.lines[0].text, "karake");
    }

    #[test]
    fn active_index_walks_the_sheet() {
        let l = parse("[00:00.00]a\n[00:10.00]b\n[00:20.00]c\n", LyricSource::Sidecar);
        assert_eq!(l.active_index(Duration::from_secs(0)), Some(0));
        assert_eq!(l.active_index(Duration::from_millis(9_999)), Some(0));
        assert_eq!(l.active_index(Duration::from_secs(10)), Some(1));
        assert_eq!(l.active_index(Duration::from_secs(999)), Some(2));
    }

    #[test]
    fn nothing_is_active_before_the_first_line() {
        let l = parse("[00:05.00]a\n", LyricSource::Sidecar);
        assert_eq!(l.active_index(Duration::from_secs(0)), None);
        assert_eq!(l.active_index(Duration::from_secs(5)), Some(0));
    }

    #[test]
    fn offset_shifts_the_highlight_earlier() {
        // The LRC convention: a positive offset makes lines appear sooner.
        let mut l = parse("[00:10.00]a\n", LyricSource::Sidecar);
        assert_eq!(l.active_index(Duration::from_secs(9)), None);
        l.offset_ms = 2000;
        assert_eq!(l.active_index(Duration::from_secs(9)), Some(0));
    }

    #[test]
    fn round_trips_through_lrc() {
        let src = "[ti:Song]\n[00:01.50]one\n[01:02.25]two\n";
        let a = parse(src, LyricSource::Sidecar);
        let b = parse(&a.to_lrc(), LyricSource::Sidecar);
        assert_eq!(a.lines, b.lines);
        assert_eq!(a.title, b.title);
    }

    #[test]
    fn stamps_past_an_hour_do_not_wrap() {
        // A 70-minute track needs [70:xx], not [10:xx].
        let l = parse("[70:30.00]late\n", LyricSource::Sidecar);
        assert_eq!(l.lines[0].at, ms(70 * 60_000 + 30_000));
        assert!(l.to_lrc().contains("[70:30.00]"), "{}", l.to_lrc());
    }

    #[test]
    fn a_bracketed_lyric_is_not_mistaken_for_a_tag() {
        // Vocaloid titles are full of these, and so are backing-vocal lines.
        let l = parse("[00:01.00][Miku] sings\n", LyricSource::Sidecar);
        assert_eq!(l.lines.len(), 1);
        assert_eq!(l.lines[0].text, "[Miku] sings");
    }

    #[test]
    fn rejects_impossible_timestamps() {
        // 90 seconds is not a valid ss field; treat it as text, not a stamp.
        let l = parse("[00:90.00]x\n", LyricSource::Sidecar);
        assert!(!l.is_synced());
    }

    fn attach_r(lines: &mut [LyricLine], text: &str) { attach(lines, text, Extra::Romaji) }

    #[test]
    fn word_marks_are_kept_and_written_back() {
        let src = "[00:10.00]<00:10.00>ka<00:10.50>ra<00:11.00>oke\n";
        let l = parse(src, LyricSource::Sidecar);
        assert_eq!(l.lines[0].text, "karaoke");
        assert_eq!(l.lines[0].marks, vec![
            (0, Duration::ZERO), (2, Duration::from_millis(500)), (4, Duration::from_secs(1)),
        ]);
        assert_eq!(l.to_lrc(), src);
    }

    /// A refrain written once with several times gets its word timing at each.
    #[test]
    fn a_repeated_line_keeps_its_words_in_step() {
        let l = parse("[00:10.00][01:10.00]<00:10.00>a <00:10.50>b\n", LyricSource::Sidecar);
        assert_eq!(l.lines[1].at, Some(Duration::from_secs(70)));
        assert_eq!(l.sung(1, Duration::from_millis(70_100)), Some(2), "\"a \" is being sung");
        assert_eq!(l.sung(1, Duration::from_millis(70_600)), Some(3));
    }

    /// Millisecond word timing survives a save. At 180 FPS a frame is 5.6 ms;
    /// these two marks are 5 ms apart and fall in different frames, and
    /// centisecond output would have put both at 10.00 — the same frame.
    #[test]
    fn millisecond_timing_survives_a_save() {
        let src = "[00:10.00]<00:10.004>a<00:10.009>b\n";
        let l = parse(src, LyricSource::Sidecar);
        assert_eq!(l.to_lrc(), src, "exact times keep their form");
        let back = parse(&l.to_lrc(), LyricSource::Sidecar);
        assert_eq!(back.lines, l.lines);
        let frame = 1000.0 / 180.0;
        let (a, b) = (10_004.0_f64, 10_009.0_f64);
        assert_ne!((a / frame).floor(), (b / frame).floor(), "the marks are a frame apart");
        // Between them only "a" has begun.
        assert_eq!(back.sung(0, Duration::from_micros(10_006_000)), Some(1));
    }

    /// Retiming a line to a millisecond start moves its words with it, and
    /// all of it is written and read back exactly.
    #[test]
    fn a_retimed_line_keeps_millisecond_word_timing() {
        let mut l = parse("[00:10.00]<00:10.004>a<00:10.009>b\n", LyricSource::Sidecar);
        l.lines[0].at = Some(Duration::from_millis(20_003));
        let out = l.to_lrc();
        assert_eq!(out, "[00:20.003]<00:20.007>a<00:20.012>b\n");
        assert_eq!(parse(&out, LyricSource::Sidecar).lines, l.lines);
    }

    /// Whole centiseconds are still written the way every reader expects.
    #[test]
    fn centisecond_times_keep_two_digits() {
        assert_eq!(fmt_stamp(Duration::from_millis(61_500)), "01:01.50");
        assert_eq!(fmt_stamp(Duration::from_millis(4_120)), "00:04.12");
        assert_eq!(fmt_stamp(Duration::from_millis(4_125)), "00:04.125");
        assert_eq!(fmt_stamp(Duration::from_millis(70 * 60_000 + 7)), "70:00.007");
    }

    #[test]
    fn angle_brackets_that_are_not_times_are_text() {
        let l = parse("[00:01.00]I <3 you\n", LyricSource::Sidecar);
        assert_eq!(l.lines[0].text, "I <3 you");
        assert!(l.lines[0].marks.is_empty());
    }

    #[test]
    fn karaoke_follows_the_marks() {
        let l = parse("[00:10.00]<00:10.00>ka<00:10.50>ra<00:11.00>oke\n[00:20.00]x\n",
                      LyricSource::Sidecar);
        let at = |ms| l.sung(0, Duration::from_millis(ms));
        assert_eq!(at(10_100), Some(2), "\"ka\" is being sung");
        assert_eq!(at(10_600), Some(4));
        assert_eq!(at(11_200), Some(7));
    }

    /// No word timing, no karaoke: a guessed sweep never matches the singing.
    #[test]
    fn a_line_without_word_timing_has_no_karaoke() {
        let l = parse("[00:10.00]さくら\n[01:00.00]x\n", LyricSource::Sidecar);
        assert_eq!(l.sung(0, Duration::from_secs(11)), None);
        assert!(!l.has_word_timing());
    }

    #[test]
    fn translation_round_trips_beside_the_sheet() {
        let dir = std::env::temp_dir().join(format!("moosik_trans_{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let track = dir.join("song.flac");
        let mut l = parse("[00:01.00]心\n[00:02.00]空\n", LyricSource::Sidecar);
        l.lines[1].translation = Some("sky".into());
        let p = save_extra(&track, &l, Extra::Translation).unwrap();
        assert_eq!(p.file_name().unwrap(), "song.translation.lrc");
        let mut back = parse("[00:01.00]心\n[00:02.00]空\n", LyricSource::Sidecar);
        attach(&mut back.lines, &std::fs::read_to_string(&p).unwrap(), Extra::Translation);
        let _ = std::fs::remove_dir_all(&dir);
        assert_eq!(back.lines[0].translation, None);
        assert_eq!(back.lines[1].translation.as_deref(), Some("sky"));
        assert_eq!(back.lines[1].romaji, None, "a translation is not romaji");
    }

    /// A scratch folder with a track in it, removed on drop.
    struct Scratch(PathBuf);
    impl Scratch {
        fn new(tag: &str) -> Self {
            let d = std::env::temp_dir()
                .join(format!("moosik_{tag}_{}_{:?}", std::process::id(), std::thread::current().id()));
            let _ = std::fs::remove_dir_all(&d);
            std::fs::create_dir_all(&d).unwrap();
            Scratch(d)
        }
        fn track(&self) -> PathBuf { self.0.join("song.flac") }
        fn read(&self, name: &str) -> Option<String> { std::fs::read_to_string(self.0.join(name)).ok() }
    }
    impl Drop for Scratch {
        fn drop(&mut self) { let _ = std::fs::remove_dir_all(&self.0); }
    }

    /// The old sheet's romaji and translation must not reappear on a new
    /// sheet whose timestamps happen to match — and must not be lost either.
    #[test]
    fn a_replaced_sheet_does_not_pick_up_the_old_extras() {
        let s = Scratch::new("replace");
        let t = s.track();
        std::fs::write(sidecar_path(&t), "[00:01.00]古い\n[00:02.00]歌\n").unwrap();
        let old_r = "[00:01.00]furui\n[00:02.00]uta\n";
        let old_t = "[00:01.00]old\n[00:02.00]song\n";
        std::fs::write(extra_path(&t, Extra::Romaji), old_r).unwrap();
        std::fs::write(extra_path(&t, Extra::Translation), old_t).unwrap();
        assert!(load(&t, None).unwrap().lines[0].romaji.is_some(), "precondition");

        // Same timestamps, different song, no extras of its own.
        let new = parse("[00:01.00]新しい\n[00:02.00]夢\n", LyricSource::Online);
        let SaveOutcome::Saved { backups, removed, .. } = save_all(&t, &new, true, false) else {
            panic!("save failed")
        };
        assert_eq!(backups.len(), 2);
        assert_eq!(removed.len(), 2, "the new sheet has no extras, so neither file may stay");

        let back = load(&t, None).unwrap();
        assert_eq!(back.lines[0].text, "新しい");
        assert!(back.lines.iter().all(|l| l.romaji.is_none() && l.translation.is_none()),
                "{:?}", back.lines);
        // Recoverable, byte for byte.
        assert_eq!(s.read("song.romaji.lrc.old").as_deref(), Some(old_r));
        assert_eq!(s.read("song.translation.lrc.old").as_deref(), Some(old_t));
        assert!(s.read("song.romaji.lrc").is_none());
    }

    /// A replacement that brings its own extras writes them fresh, with the
    /// old ones kept aside — and a second replacement never overwrites the
    /// first set-aside copy.
    #[test]
    fn a_replacement_with_its_own_extras_keeps_every_old_copy() {
        let s = Scratch::new("replace2");
        let t = s.track();
        std::fs::write(sidecar_path(&t), "[00:01.00]古い\n").unwrap();
        std::fs::write(extra_path(&t, Extra::Romaji), "[00:01.00]first\n").unwrap();
        let mut new = parse("[00:01.00]新しい\n", LyricSource::Online);
        new.lines[0].romaji = Some("atarashii".into());
        assert!(matches!(save_all(&t, &new, true, false), SaveOutcome::Saved { .. }));
        std::fs::write(extra_path(&t, Extra::Romaji), "[00:01.00]second\n").unwrap();
        assert!(matches!(save_all(&t, &new, true, false), SaveOutcome::Saved { .. }));
        assert_eq!(s.read("song.romaji.lrc.old").as_deref(), Some("[00:01.00]first\n"));
        assert_eq!(s.read("song.romaji.lrc.old.2").as_deref(), Some("[00:01.00]second\n"));
        assert_eq!(load(&t, None).unwrap().lines[0].romaji.as_deref(), Some("atarashii"));
    }

    /// If the new sheet cannot be written, nothing is left half-done: the
    /// extras go back where they were, still paired with the old sheet.
    #[test]
    fn a_failed_save_puts_the_old_extras_back() {
        let s = Scratch::new("fail");
        let t = s.track();
        let old_r = "[00:01.00]furui\n";
        std::fs::write(extra_path(&t, Extra::Romaji), old_r).unwrap();
        std::fs::write(extra_path(&t, Extra::Translation), "[00:01.00]old\n").unwrap();
        // A folder where song.lrc should go makes writing it fail.
        std::fs::create_dir(sidecar_path(&t)).unwrap();
        let new = parse("[00:01.00]新しい\n", LyricSource::Online);
        assert!(matches!(save_all(&t, &new, true, false), SaveOutcome::RolledBack { .. }));
        assert_eq!(s.read("song.romaji.lrc").as_deref(), Some(old_r));
        assert!(s.read("song.translation.lrc").is_some());
        assert!(s.read("song.romaji.lrc.old").is_none() && s.read("song.translation.lrc.old").is_none());
    }

    // ── A save that fails part-way ─────────────────────────────────────────

    const OLD_SHEET: &str = "[00:01.00]古い\n[00:02.00]歌\n";
    const OLD_R: &str = "[00:01.00]furui\n[00:02.00]uta\n";
    const OLD_T: &str = "[00:01.00]old\n[00:02.00]song\n";

    /// Old sheet and both extras on disk; a new sheet with extras of its own,
    /// so every one of the three files changes.
    fn failing_setup(tag: &str) -> (Scratch, PathBuf, Lyrics) {
        let s = Scratch::new(tag);
        let t = s.track();
        std::fs::write(sidecar_path(&t), OLD_SHEET).unwrap();
        std::fs::write(extra_path(&t, Extra::Romaji), OLD_R).unwrap();
        std::fs::write(extra_path(&t, Extra::Translation), OLD_T).unwrap();
        let mut new = parse("[00:01.00]新しい\n[00:03.00]夢\n", LyricSource::Online);
        new.lines[0].romaji = Some("atarashii".into());
        new.lines[1].translation = Some("dream".into());
        (s, t, new)
    }

    /// Every file in the folder, by name, with its bytes.
    fn snapshot(s: &Scratch) -> std::collections::BTreeMap<String, Vec<u8>> {
        std::fs::read_dir(&s.0).unwrap()
            .map(|e| e.unwrap())
            .map(|e| (e.file_name().to_string_lossy().into_owned(), std::fs::read(e.path()).unwrap()))
            .collect()
    }

    fn fail_at(stage: Stage, file: &'static str)
        -> impl FnMut(Stage, &Path) -> std::io::Result<()>
    {
        move |st, p| {
            if st == stage && p.file_name().is_some_and(|n| n == file) {
                Err(std::io::Error::other(format!("injected at {stage:?} {file}")))
            } else {
                Ok(())
            }
        }
    }

    /// A failure right after the main sheet is written, and one right after
    /// the first extra is written: in both, every live file goes back to its
    /// exact previous bytes, and nothing else is left in the folder — no
    /// backup, no temporary file. A retry then succeeds.
    #[test]
    fn a_save_that_fails_after_a_write_is_rolled_back_exactly() {
        for (tag, file) in [("after_sheet", "song.lrc"), ("after_romaji", "song.romaji.lrc")] {
            let (s, t, new) = failing_setup(tag);
            let before = snapshot(&s);
            let out = save_all_with(&t, &new, true, false, &mut fail_at(Stage::Applied, file));
            assert!(matches!(out, SaveOutcome::RolledBack { .. }), "{tag}: {out:?}");
            assert_eq!(snapshot(&s), before, "{tag}: disk is not as it was");

            // Safe to retry, and the retry is a normal save.
            let SaveOutcome::Saved { backups, .. } = save_all(&t, &new, true, false) else {
                panic!("{tag}: the retry failed")
            };
            assert_eq!(backups.len(), 2, "{tag}: both old extras kept");
            let back = load(&t, None).unwrap();
            assert_eq!(back.lines[0].romaji.as_deref(), Some("atarashii"));
            assert_eq!(back.lines[1].translation.as_deref(), Some("dream"));
            assert_eq!(s.read("song.romaji.lrc.old").as_deref(), Some(OLD_R));
            assert_eq!(s.read("song.translation.lrc.old").as_deref(), Some(OLD_T));
            assert!(s.read("song.lrc.old").is_none(), "{tag}: the sheet's own backup outlived success");
        }
    }

    /// A failure while backing up, before anything live has changed: nothing
    /// changes, and the copies already made are cleared away.
    #[test]
    fn a_save_that_fails_while_backing_up_changes_nothing() {
        let (s, t, new) = failing_setup("backup");
        let before = snapshot(&s);
        let out = save_all_with(&t, &new, true, false,
                                &mut fail_at(Stage::Backup, "song.translation.lrc"));
        assert!(matches!(out, SaveOutcome::RolledBack { .. }), "{out:?}");
        assert_eq!(snapshot(&s), before);
    }

    /// The rollback itself fails. The outcome says so and names the file left
    /// holding the new version; every previous version is in a backup, byte
    /// for byte; the files it could restore are restored. And a retry brings
    /// everything back into agreement without losing those backups.
    #[test]
    fn a_rollback_that_fails_is_reported_and_recoverable() {
        let (s, t, new) = failing_setup("undo");
        let mut hook = {
            let mut after = fail_at(Stage::Applied, "song.romaji.lrc");
            let mut restore = fail_at(Stage::Restore, "song.lrc");
            move |st: Stage, p: &Path| { after(st, p)?; restore(st, p) }
        };
        let out = save_all_with(&t, &new, true, false, &mut hook);
        let SaveOutcome::NotUndone { stuck, backups, .. } = out else { panic!("{out:?}") };
        assert_eq!(stuck.len(), 1);
        assert_eq!(stuck[0].0.file_name().unwrap(), "song.lrc");
        // song.lrc holds the new sheet; its previous version is in a backup.
        assert_eq!(std::fs::read_to_string(sidecar_path(&t)).unwrap(), new.to_lrc());
        assert_eq!(s.read("song.lrc.old").as_deref(), Some(OLD_SHEET));
        // The romaji was restored; the translation was never touched.
        assert_eq!(s.read("song.romaji.lrc").as_deref(), Some(OLD_R));
        assert_eq!(s.read("song.translation.lrc").as_deref(), Some(OLD_T));
        let names: Vec<_> = backups.iter().map(|b| b.file_name().unwrap().to_string_lossy().into_owned()).collect();
        assert!(names.contains(&"song.lrc.old".to_string()), "{names:?}");
        assert!(names.contains(&"song.romaji.lrc.old".to_string()), "{names:?}");

        // Retry: everything agrees again, and no earlier backup is lost.
        assert!(matches!(save_all(&t, &new, true, false), SaveOutcome::Saved { .. }));
        let back = load(&t, None).unwrap();
        assert_eq!(back.to_lrc(), new.to_lrc());
        assert_eq!(back.lines[0].romaji.as_deref(), Some("atarashii"));
        assert_eq!(s.read("song.lrc.old").as_deref(), Some(OLD_SHEET));
        assert_eq!(s.read("song.romaji.lrc.old").as_deref(), Some(OLD_R));
        assert!(s.read("song.romaji.lrc.old.2").is_some(), "the retry kept its own copy too");
    }

    /// An extra that exists but cannot be read is not "missing". Treating it
    /// so would report success and leave it to reattach on the next load. The
    /// save fails, and nothing on disk changes.
    #[test]
    fn an_unreadable_extra_fails_the_save_safely() {
        let s = Scratch::new("unreadable");
        let t = s.track();
        std::fs::write(sidecar_path(&t), OLD_SHEET).unwrap();
        // A folder where the romaji file is expected: it exists, and reading
        // it fails with something other than "not found".
        std::fs::create_dir(extra_path(&t, Extra::Romaji)).unwrap();
        let before = snapshot_names(&s);
        let new = parse("[00:01.00]新しい\n", LyricSource::Online);
        let out = save_all(&t, &new, true, false);
        let SaveOutcome::RolledBack { error } = out else { panic!("{out:?}") };
        assert!(error.to_string().contains("song.romaji.lrc"), "{error}");
        assert_eq!(s.read("song.lrc").as_deref(), Some(OLD_SHEET));
        assert_eq!(snapshot_names(&s), before);
    }

    /// The same with a real file another program holds open with no sharing,
    /// which is how an unreadable file usually happens on Windows.
    #[cfg(windows)]
    #[test]
    fn a_locked_extra_fails_the_save_safely() {
        use std::os::windows::fs::OpenOptionsExt as _;
        let (s, t, _) = failing_setup("locked");
        let before = snapshot_names(&s);
        let lock = std::fs::OpenOptions::new().read(true).share_mode(0)
            .open(extra_path(&t, Extra::Translation)).unwrap();
        let new = parse("[00:01.00]新しい\n", LyricSource::Online);
        let out = save_all(&t, &new, true, false);
        assert!(matches!(out, SaveOutcome::RolledBack { .. }), "{out:?}");
        drop(lock);
        assert_eq!(snapshot_names(&s), before);
        assert_eq!(s.read("song.lrc").as_deref(), Some(OLD_SHEET));
        assert_eq!(s.read("song.translation.lrc").as_deref(), Some(OLD_T));
    }

    /// A backup or temporary name that some other writer takes between our
    /// choosing it and creating it is skipped, and the other file is left
    /// exactly as it was.
    #[test]
    fn a_name_taken_at_the_last_moment_is_never_written_over() {
        let (s, t, new) = failing_setup("race");
        let intruders = ["song.romaji.lrc.old", "song.lrc.saving"];
        let mut hook = |st: Stage, p: &Path| {
            if st == Stage::Reserve && let Some(n) = p.file_name()
                && intruders.iter().any(|i| n == *i) && !p.exists()
            {
                std::fs::write(p, "someone else's").unwrap();
            }
            Ok(())
        };
        let out = save_all_with(&t, &new, true, false, &mut hook);
        let SaveOutcome::Saved { backups, .. } = out else { panic!("{out:?}") };
        for i in intruders {
            assert_eq!(s.read(i).as_deref(), Some("someone else's"), "{i} was written over");
        }
        assert!(backups.iter().any(|b| b.file_name().unwrap() == "song.romaji.lrc.old.2"), "{backups:?}");
        assert_eq!(s.read("song.romaji.lrc.old.2").as_deref(), Some(OLD_R));
        assert_eq!(load(&t, None).unwrap().to_lrc(), new.to_lrc());
    }

    /// A second save of the same track while one is in progress is refused
    /// without touching anything; the first finishes normally, and once it
    /// has, the track can be saved again.
    #[test]
    fn a_second_save_of_the_same_track_waits_its_turn() {
        let (s, t, new) = failing_setup("concurrent");
        let other = parse("[00:07.00]別の\n", LyricSource::Online);
        let mut inner = None;
        let mut hook = |st: Stage, p: &Path| {
            if st == Stage::Applied && p.file_name().is_some_and(|n| n == "song.lrc") && inner.is_none() {
                let mid = snapshot_names(&s);
                let out = save_all(&t, &other, true, false);
                assert_eq!(snapshot_names(&s), mid, "the refused save touched something");
                inner = Some(out);
            }
            Ok(())
        };
        let out = save_all_with(&t, &new, true, false, &mut hook);
        assert!(matches!(out, SaveOutcome::Saved { .. }), "{out:?}");
        let Some(SaveOutcome::RolledBack { error }) = inner else { panic!("{inner:?}") };
        assert_eq!(error.kind(), std::io::ErrorKind::WouldBlock);
        assert_eq!(load(&t, None).unwrap().to_lrc(), new.to_lrc());
        // Released: a later save goes ahead.
        assert!(matches!(save_all(&t, &other, true, false), SaveOutcome::Saved { .. }));
    }

    fn snapshot_names(s: &Scratch) -> std::collections::BTreeMap<String, Option<Vec<u8>>> {
        std::fs::read_dir(&s.0).unwrap()
            .map(|e| e.unwrap())
            .map(|e| (e.file_name().to_string_lossy().into_owned(), std::fs::read(e.path()).ok()))
            .collect()
    }

    /// Hand-written lines that did not pair with the sheet are not lost when
    /// the sheet is saved: the file is rewritten from what paired, and the
    /// original, unpaired lines and all, is kept.
    #[test]
    fn unmatched_hand_written_lines_are_kept_in_a_backup() {
        let s = Scratch::new("unmatched");
        let t = s.track();
        std::fs::write(sidecar_path(&t), "[00:01.00]心\n").unwrap();
        let hand = "[00:01.00]kokoro\n[00:09.00]a line with no partner\n";
        std::fs::write(extra_path(&t, Extra::Romaji), hand).unwrap();
        let mut l = load(&t, None).unwrap();
        l.offset_ms = 100;
        let SaveOutcome::Saved { backups, .. } = save_all(&t, &l, true, false) else { panic!() };
        assert_eq!(backups.len(), 1);
        assert_eq!(s.read("song.romaji.lrc.old").as_deref(), Some(hand));
    }

    /// Saving edits to the sheet already on disk is not a replacement: its
    /// extras are its own and nothing is set aside.
    #[test]
    fn saving_the_same_sheet_sets_nothing_aside() {
        let s = Scratch::new("same");
        let t = s.track();
        std::fs::write(sidecar_path(&t), "[00:01.00]心\n").unwrap();
        std::fs::write(extra_path(&t, Extra::Romaji), "[00:01.00]kokoro\n").unwrap();
        let mut l = load(&t, None).unwrap();
        l.offset_ms = 200;
        let SaveOutcome::Saved { backups, .. } = save_all(&t, &l, true, false) else {
            panic!("save failed")
        };
        assert!(backups.is_empty(), "an extra that already matches is not rewritten or copied");
        assert_eq!(load(&t, None).unwrap().lines[0].romaji.as_deref(), Some("kokoro"));
    }

    #[test]
    fn romaji_pairs_by_position_when_the_shapes_match() {
        let mut l = parse("[00:01.00]心\n[00:02.00]空\n", LyricSource::Sidecar);
        attach_r(&mut l.lines, "[00:01.00]kokoro\n[00:02.00]\n");
        assert_eq!(l.lines[0].romaji.as_deref(), Some("kokoro"));
        assert_eq!(l.lines[1].romaji, None, "blank means generate it");
    }

    /// NetEase's romaji sheet leaves out the credit lines its lyric sheet
    /// starts with, so the counts differ and only the timestamps line up.
    #[test]
    fn romaji_pairs_by_timestamp_when_the_shapes_differ() {
        let mut l = parse("[00:00.00]作词 : X\n[00:18.05]渇いた心\n", LyricSource::Online);
        attach_r(&mut l.lines, "[00:18.050]ka wa i ta\n");
        assert_eq!(l.lines[0].romaji, None);
        assert_eq!(l.lines[1].romaji.as_deref(), Some("ka wa i ta"));
    }

    #[test]
    fn unsynced_romaji_pairs_by_order() {
        let mut l = parse("心\n空\n海\n", LyricSource::Sidecar);
        attach_r(&mut l.lines, "kokoro\n\numi\n");
        let r: Vec<_> = l.lines.iter().map(|x| x.romaji.as_deref()).collect();
        assert_eq!(r, vec![Some("kokoro"), None, Some("umi")]);
    }

    /// What `save_extra` writes must come back onto the same lines, blanks
    /// included — for a partly-synced sheet too, where a dropped blank could
    /// shift the untimed lines.
    #[test]
    fn saved_romaji_round_trips() {
        let dir = std::env::temp_dir().join(format!("moosik_romaji_{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let track = dir.join("song.flac");
        let src = "[00:01.00]心\n[00:02.00]空\n海\n山\n";
        let mut l = parse(src, LyricSource::Sidecar);
        l.lines[0].romaji = Some("kokoro".into());
        l.lines[3].romaji = Some("yama".into());
        let p = save_extra(&track, &l, Extra::Romaji).unwrap();
        assert_eq!(p.file_name().unwrap(), "song.romaji.lrc");
        let mut back = parse(src, LyricSource::Sidecar);
        attach(&mut back.lines, &std::fs::read_to_string(&p).unwrap(), Extra::Romaji);
        let _ = std::fs::remove_dir_all(&dir);
        let got: Vec<_> = back.lines.iter().map(|x| x.romaji.clone()).collect();
        let want: Vec<_> = l.lines.iter().map(|x| x.romaji.clone()).collect();
        assert_eq!(got, want);
    }

    #[test]
    fn empty_input_is_empty_not_a_panic() {
        assert!(parse("", LyricSource::Sidecar).is_empty());
        assert!(parse("\n\n\n", LyricSource::Sidecar).is_empty());
    }
}
