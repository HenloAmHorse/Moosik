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
#[derive(Clone, Debug, PartialEq)]
pub struct LyricLine {
    pub at: Option<Duration>,
    pub text: String,
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

    /// Serialise back to LRC. Round-trips [`parse`] for synced sheets.
    pub fn to_lrc(&self) -> String {
        let mut out = String::new();
        if let Some(t) = &self.title  { out.push_str(&format!("[ti:{t}]\n")); }
        if let Some(a) = &self.artist { out.push_str(&format!("[ar:{a}]\n")); }
        if self.offset_ms != 0 { out.push_str(&format!("[offset:{}]\n", self.offset_ms)); }
        if !out.is_empty() { out.push('\n'); }
        for line in &self.lines {
            match line.at {
                Some(t) => out.push_str(&format!("[{}]{}\n", fmt_stamp(t), line.text)),
                None => { out.push_str(&line.text); out.push('\n'); }
            }
        }
        out
    }
}

/// `mm:ss.cc`, the form every LRC reader accepts. Minutes are not wrapped at
/// 60 — a 70-minute track is `[70:xx.xx]`, which is what other players write.
pub fn fmt_stamp(t: Duration) -> String {
    let cs = (t.as_millis() / 10) as u64;
    format!("{:02}:{:02}.{:02}", cs / 6000, (cs / 100) % 60, cs % 100)
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

        let text = strip_word_marks(rest).trim().to_string();
        if stamps.is_empty() {
            // A metadata-only line contributes nothing; a blank line in an
            // unsynced sheet is a real paragraph break and is kept.
            if !meta_only { lines.push(LyricLine { at: None, text }); }
        } else {
            for at in stamps {
                lines.push(LyricLine { at: Some(at), text: text.clone() });
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

/// Remove enhanced-LRC word timings (`<00:12.34>`) from a line's text.
fn strip_word_marks(s: &str) -> String {
    if !s.contains('<') { return s.to_string(); }
    let mut out = String::with_capacity(s.len());
    let mut depth = 0u32;
    for c in s.chars() {
        match c {
            '<' => depth += 1,
            '>' if depth > 0 => depth -= 1,
            _ if depth == 0 => out.push(c),
            _ => {}
        }
    }
    out
}

/// Path of the `.lrc` that belongs to `track` — the same name with the
/// extension swapped. Returned whether or not it exists, so callers can use it
/// as the save target too.
pub fn sidecar_path(track: &Path) -> PathBuf {
    track.with_extension("lrc")
}

/// Load lyrics for a track: the sidecar first, then whatever the tags carry.
///
/// The sidecar wins because it is the one the user can edit, and anything this
/// app saves lands there — so a saved fix must not be shadowed by a stale tag.
pub fn load(track: &Path, embedded: Option<&str>) -> Option<Lyrics> {
    let sidecar = sidecar_path(track);
    if let Ok(text) = std::fs::read_to_string(&sidecar) {
        let l = parse(&text, LyricSource::Sidecar);
        if !l.is_empty() { return Some(l); }
    }
    let l = parse(embedded?, LyricSource::Embedded);
    (!l.is_empty()).then_some(l)
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

/// Write a sheet to the track's sidecar path.
pub fn save_sidecar(track: &Path, lyrics: &Lyrics) -> std::io::Result<PathBuf> {
    let path = sidecar_path(track);
    std::fs::write(&path, lyrics.to_lrc())?;
    Ok(path)
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

    #[test]
    fn empty_input_is_empty_not_a_panic() {
        assert!(parse("", LyricSource::Sidecar).is_empty());
        assert!(parse("\n\n\n", LyricSource::Sidecar).is_empty());
    }
}
