//! LRCLIB — <https://lrclib.net>.
//!
//! The first source tried. It needs no key and no account, serves synced
//! lyrics, and is built for exactly this use, so it is the one to ask first.
//! Its coverage of Western music is good; for a Japanese or Vocaloid library it
//! thins out badly, which is what [`super::netease`] is for.
//!
//! Ranking and acceptance live in [`super::matching`] — this module only knows
//! how to ask LRCLIB questions.

use super::http::{esc, get_json};
use super::matching::{best_of, normalise_artist, normalise_title, Hit};
use std::time::Duration;

pub const NAME: &str = "LRCLIB";
const BASE: &str = "https://lrclib.net/api";

fn hit_from(v: &serde_json::Value) -> Hit {
    let s = |k: &str| v.get(k).and_then(|x| x.as_str()).unwrap_or("").to_string();
    let opt = |k: &str| v.get(k).and_then(|x| x.as_str())
        .filter(|x| !x.trim().is_empty()).map(str::to_string);
    Hit {
        source: NAME,
        title: s("trackName"),
        artist: s("artistName"),
        album: s("albumName"),
        duration: v.get("duration").and_then(|x| x.as_f64()).unwrap_or(0.0),
        instrumental: v.get("instrumental").and_then(|x| x.as_bool()).unwrap_or(false),
        plain: opt("plainLyrics"),
        synced: opt("syncedLyrics"),
        // LRCLIB returns lyrics with the record, so nothing to fetch later.
        id: None,
    }
}

/// The signed lookup: exact fields, database decides.
pub fn get_exact(
    artist: &str, title: &str, album: &str, duration: Option<Duration>,
) -> Result<Option<Hit>, String> {
    let mut url = format!(
        "{BASE}/get?artist_name={}&track_name={}&album_name={}",
        esc(artist), esc(title), esc(album)
    );
    if let Some(d) = duration {
        url.push_str(&format!("&duration={}", d.as_secs()));
    }
    Ok(get_json(&url, None)?.as_ref().map(hit_from))
}

fn as_hits(v: Option<serde_json::Value>) -> Vec<Hit> {
    v.and_then(|v| v.as_array().cloned())
        .map(|a| a.iter().map(hit_from).collect())
        .unwrap_or_default()
}

/// Free-text search.
pub fn search(query: &str) -> Result<Vec<Hit>, String> {
    Ok(as_hits(get_json(&format!("{BASE}/search?q={}", esc(query)), None)?))
}

/// Structured search — lets the database match the fields separately, which
/// beats one blob of text when the artist is right but the title is decorated.
pub fn search_fields(artist: &str, title: &str) -> Result<Vec<Hit>, String> {
    let url = format!("{BASE}/search?track_name={}&artist_name={}", esc(title), esc(artist));
    Ok(as_hits(get_json(&url, None)?))
}

/// The ladder, strictest first, stopping at the first rung that yields a hit
/// with lyrics.
///
/// Each rung gives up a constraint the tags may have got wrong:
/// 1. exact, all four fields — right when the tags are clean
/// 2. exact without the album — albums are the field most often wrong or absent
/// 3. exact without the duration — the database's own length comes from
///    whatever rip its uploader had, and its `/get` matches within a couple of
///    seconds, so a file that disagrees by more is refused outright by rungs 1
///    and 2 even when the title and artist are identical
/// 4. structured search on *normalised* artist and title — beats decoration
/// 5. normalised title alone — rescues a track whose artist field holds the
///    producer where the database has the vocalist
///
/// The search rungs are gated by [`super::matching::best_of`]'s `strict` mode:
/// the exact lookups can be trusted to have matched on what they were given,
/// but a fuzzy search returns whatever is vaguely close and something has to
/// refuse the merely-close-ish.
pub fn find(
    artist: &str, title: &str, album: &str, duration: Option<Duration>,
) -> Result<Option<Hit>, String> {
    let (na, nt) = (normalise_artist(artist), normalise_title(title));
    // Even an exact lookup gets checked: `/get` matches on the strings given,
    // but "given the wrong tags" is the case this whole module exists for.
    let ok = |h: &Hit| h.best_text().is_some()
        && super::matching::plausible(h, &na, &nt, duration);

    // Rungs 1 and 2 short-circuit: LRCLIB's `/get` matches the duration itself,
    // within a second or two, so an answer here is as good as it gets.
    if let Some(h) = get_exact(artist, title, album, duration)?
        && ok(&h) { return Ok(Some(h)); }
    if !album.is_empty()
        && let Some(h) = get_exact(artist, title, "", duration)?
        && ok(&h) { return Ok(Some(h)); }

    if nt.is_empty() { return Ok(None); }

    // The looser rungs are *pooled* rather than tried in turn, and the best of
    // everything wins. Returning at the first plausible one cost accuracy: the
    // duration-free lookup answers with whichever record the database considers
    // canonical, which for `KING` was a 136 s take when the file is 148 s and a
    // 152 s record existed. It cleared the widened gate, short-circuited, and
    // the better match was never looked at. Scoring the whole pool at once
    // keeps the fallback's reach without letting it pre-empt a closer hit.
    let mut pool: Vec<Hit> = Vec::new();
    if duration.is_some()
        && let Some(h) = get_exact(artist, title, "", None)? { pool.push(h); }
    if !na.is_empty() { pool.extend(search_fields(&na, &nt)?); }
    pool.extend(search(&nt)?);
    Ok(best_of(pool, &na, &nt, duration, true))
}

#[cfg(test)]
mod tests {
    use super::*;
    use super::super::matching::similarity;

    /// Hits the real service. Not part of the normal run — it needs network and
    /// would fail for reasons that have nothing to do with this code.
    ///
    /// `cargo test --bin moosik -- --ignored --nocapture live_lrclib`
    #[test]
    #[ignore = "network — run explicitly"]
    fn live_lrclib() {
        let cases: &[(&str, &str, &str, u64)] = &[
            ("Radiohead", "Creep", "Pablo Honey", 238),
            // Decorated Vocaloid tags: neither the exact lookup nor a plain
            // title query can win on their own.
            ("DECO*27", "【初音ミク】ヴァンパイア【オリジナル】", "", 180),
            ("Kanaria", "KING", "", 148),
            // The duration-gate regression: LRCLIB records this at 133 s and
            // the file is 124 s, which the old flat ±8 s gate refused.
            ("TAK, Hatsune Miku", "MTMTM (feat. Hatsune Miku)", "", 124),
        ];
        for (artist, title, album, secs) in cases {
            let d = Some(Duration::from_secs(*secs));
            match find(artist, title, album, d) {
                Ok(Some(h)) => println!(
                    "{title} -> {} / {} ({:.0}s) synced={}",
                    h.title, h.artist, h.duration, h.has_synced()),
                Ok(None) => println!("{title} -> no match"),
                Err(e) => println!("{title} -> ERROR {e}"),
            }
        }
    }

    /// Prints similarity for real cases so the thresholds stay set from data.
    ///
    /// `cargo test --bin moosik -- --ignored --nocapture threshold_survey`
    #[test]
    #[ignore = "network — run explicitly"]
    fn threshold_survey() {
        let cases: &[(&str, &str)] = &[
            ("えいぷ", "はんぶんこ"), ("TAK", "MTMTM"), ("", "SWIPE×SWIPE"),
            ("DECO*27", "ヴァンパイア"), ("Kanaria", "KING"),
        ];
        println!("\n{:<14} {:<22} {:>6} {:>7}", "query", "candidate", "title", "artist");
        for (artist, title) in cases {
            let (na, nt) = (normalise_artist(artist), normalise_title(title));
            for h in search(&nt).unwrap_or_default().iter().take(3) {
                println!("{:<14} {:<22} {:>6.2} {:>7.2}",
                         title.chars().take(13).collect::<String>(),
                         format!("{} / {}", h.title, h.artist).chars().take(21)
                             .collect::<String>(),
                         similarity(&nt, &normalise_title(&h.title)),
                         similarity(&na, &normalise_artist(&h.artist)));
            }
        }
    }
}
