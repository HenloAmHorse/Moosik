//! QQ Music (QQ音乐) — the third source, for the Taiwanese and Chinese
//! releases the other two lack.
//!
//! Measured on twenty Taiwanese indie and Vocaloid tracks (2026-11-07):
//! 草東沒有派對, 告五人, 落日飛車 and 康士坦的變化球 had synced lyrics here and
//! in neither LRCLIB nor NetEase, whose catalogues miss many Taiwanese labels.
//! Kugou and lyrics.ovh were tried on the same tracks and found nothing the
//! others had not, so they are not asked.
//!
//! Like NetEase this is not a documented public API: it may change without
//! warning, so a failure is reported and the ladder moves on. Its catalogue is
//! in Simplified Chinese; matching folds the two scripts together (see
//! [`super::matching::similarity`]).

use super::http::{esc, get_json};
use super::matching::{instrumental_note, Hit};
use std::time::Duration;

pub const NAME: &str = "QQ Music";
const SEARCH: &str = "https://c.y.qq.com/soso/fcgi-bin/client_search_cp";
const LYRIC: &str = "https://c.y.qq.com/lyric/fcgi-bin/fcg_query_lyric_new.fcg";
const REFERER: &str = "https://y.qq.com/";
const LIMIT: usize = 10;

/// Like NetEase, the real status is in the body, not the HTTP status.
fn check_code(v: &serde_json::Value) -> Result<(), String> {
    match v.get("code").and_then(|c| c.as_i64()) {
        None | Some(0) => Ok(()),
        Some(c) => Err(format!("refused (code {c})")),
    }
}

/// Search rows, without lyrics: those are fetched only for the row chosen.
fn hits_from(v: &serde_json::Value) -> Vec<Hit> {
    let songs = v.pointer("/data/song/list").and_then(|s| s.as_array()).cloned().unwrap_or_default();
    songs.iter().map(|s| Hit {
        source: NAME,
        title: s.get("name").and_then(|x| x.as_str()).unwrap_or("").to_string(),
        artist: s.get("singer").and_then(|x| x.as_array())
            .map(|a| a.iter().filter_map(|p| p.get("name").and_then(|n| n.as_str()))
                .collect::<Vec<_>>().join(", "))
            .unwrap_or_default(),
        album: s.pointer("/album/name").and_then(|x| x.as_str()).unwrap_or("").to_string(),
        // Seconds here, like LRCLIB.
        duration: s.get("interval").and_then(|x| x.as_f64()).unwrap_or(0.0),
        id: s.get("mid").and_then(|x| x.as_str()).filter(|m| !m.is_empty()).map(str::to_string),
        ..Default::default()
    }).collect()
}

pub fn search(query: &str) -> Result<Vec<Hit>, String> {
    let url = format!("{SEARCH}?w={}&format=json&p=1&n={LIMIT}&cr=1&new_json=1", esc(query));
    let Some(v) = get_json(&url, Some(REFERER))? else { return Ok(Vec::new()) };
    check_code(&v)?;
    Ok(hits_from(&v))
}

/// Fetch the lyrics for one search row.
pub fn fill_lyrics(h: &mut Hit) -> Result<(), String> {
    let Some(id) = h.id.clone() else { return Ok(()) };
    let url = format!("{LYRIC}?songmid={}&format=json&nobase64=1&g_tk=5381", esc(&id));
    let Some(v) = get_json(&url, Some(REFERER))? else { return Ok(()) };
    // −1901 is its answer for a song with no words; the next candidate may have them.
    if v.get("code").and_then(|c| c.as_i64()) == Some(-1901) { return Ok(()); }
    check_code(&v)?;
    let sheet = |key: &str| v.get(key).and_then(|x| x.as_str())
        .filter(|t| !t.trim().is_empty()).map(str::to_string);
    let Some(text) = sheet("lyric") else { return Ok(()) };
    if instrumental_note(&text) {
        h.instrumental = true;
        return Ok(());
    }
    // Mostly Chinese, like NetEase's.
    h.translation = sheet("trans");
    if text.contains('[') && text.contains(']') {
        h.synced = Some(text);
    } else {
        h.plain = Some(text);
    }
    Ok(())
}

/// Best match for a track, lyrics included; ranked and fetched the way
/// NetEase's are.
pub fn find(
    artist: &str, title: &str, _album: &str, duration: Option<Duration>,
) -> Result<Option<Hit>, String> {
    super::netease::find_with(artist, title, duration, false, || false, search, fill_lyrics)
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A row as the search answers it: several singers joined, seconds, the
    /// song's `mid` kept as the handle and no words yet.
    #[test]
    fn search_rows_carry_singers_seconds_and_a_handle_but_no_text() {
        let v = serde_json::json!({"code": 0, "data": {"song": {"list": [
            {"mid": "002P0HWP3lJJFk", "name": "披星戴月的想你", "interval": 349,
             "singer": [{"name": "告五人"}, {"name": "Guest"}], "album": {"name": "X"}}
        ]}}});
        check_code(&v).unwrap();
        let h = &hits_from(&v)[0];
        assert_eq!((h.title.as_str(), h.artist.as_str(), h.duration), ("披星戴月的想你", "告五人, Guest", 349.0));
        assert_eq!(h.id.as_deref(), Some("002P0HWP3lJJFk"));
        assert!(h.best_text().is_none());
        assert!(check_code(&serde_json::json!({"code": 2001})).unwrap_err().contains("2001"));
    }

    /// `cargo test --bin moosik -- --ignored --nocapture live_qq`
    #[test]
    #[ignore = "network — run explicitly"]
    fn live_qq() {
        // Neither LRCLIB nor NetEase had these; tags as a Taiwanese library has them.
        let cases: &[(&str, &str, u64)] = &[
            ("草東沒有派對", "大風吹", 236),
            ("告五人", "披星戴月的想你", 349),
            ("落日飛車", "Bluebird", 217),
            ("康士坦的變化球", "美好的事可不可以發生在我身上", 362),
            // Instrumental: the service's "pure music" note is not lyrics.
            ("久石譲", "Summer", 153),
        ];
        for (artist, title, secs) in cases {
            match find(artist, title, "", Some(Duration::from_secs(*secs))) {
                // Metadata and line counts only, never the words.
                Ok(Some(h)) => println!("{title} -> {} / {} ({:.0}s) synced={} lines={}",
                    h.title, h.artist, h.duration, h.has_synced(),
                    h.best_text().map(|t| t.lines().count()).unwrap_or(0)),
                Ok(None) => println!("{title} -> no match"),
                Err(e) => println!("{title} -> ERROR {e}"),
            }
        }
    }
}
