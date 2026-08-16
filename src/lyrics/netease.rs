//! NetEase Cloud Music (网易云音乐) — the second source, and the one that
//! actually covers a Vocaloid library.
//!
//! Measured against the tracks LRCLIB could not find, this had synced lyrics
//! for nearly all of them: 聖人君子でありたい, ダダダダダル, アダチ・レイ,
//! カンケーガール, ちっちゃな私, アクマバライ, いつまでたってもテイカー and the
//! rest. LRCLIB returned zero hits for those. That gap is the whole reason this
//! module exists — Japanese and Vocaloid material is simply not in the
//! Western-leaning databases, and no amount of better matching invents it.
//!
//! ## Why here and not Genius or UtaTen
//!
//! Both were considered and both are a poor fit, for the same two reasons:
//! neither publishes lyrics through an API (Genius's API deliberately excludes
//! them), so the only route is scraping their HTML — which breaks on any
//! redesign — and neither carries timestamps, so even a successful scrape gives
//! an unsynced sheet that still has to be timed by hand. The paste box already
//! covers that case, and does it without a scraper to maintain.
//!
//! This endpoint returns LRC directly as JSON. It is not a documented public
//! API, so it may change without warning; failures here are reported and the
//! ladder simply moves on rather than treating them as fatal.

use super::http::{esc, get_json};
use super::matching::{acceptable, normalise_artist, normalise_title, score, Hit};
use std::time::Duration;

pub const NAME: &str = "NetEase";
const SEARCH: &str = "https://music.163.com/api/search/get/web";
const LYRIC: &str = "https://music.163.com/api/song/lyric";
const REFERER: &str = "https://music.163.com";
/// Enough to reach past covers and instrumentals to the real take; more only
/// costs latency, since only the chosen one's lyrics are fetched.
const LIMIT: usize = 10;

/// Turn the envelope's own status code into an error.
///
/// This service answers *everything* with HTTP 200 and puts the real status in
/// the JSON body, so a refusal looks exactly like an empty result to anything
/// checking the transport. Code 405 is its rate limit — "操作频繁，请稍候再试",
/// roughly "too many requests, try again shortly" — and reading that as "this
/// song has no lyrics" is the worst possible misreading: the user is told their
/// track does not exist when the truth is to wait a minute.
fn check_code(v: &serde_json::Value) -> Result<(), String> {
    match v.get("code").and_then(|c| c.as_i64()) {
        None | Some(200) => Ok(()),
        Some(405) => Err("rate limited — wait a moment and try again".into()),
        Some(c) => {
            let msg = v.get("message").or_else(|| v.get("msg"))
                .and_then(|m| m.as_str()).unwrap_or("no detail");
            Err(format!("refused (code {c}: {msg})"))
        }
    }
}

/// Search. The returned hits carry no lyrics yet — those cost a request each,
/// so they are fetched only for the candidate actually chosen.
pub fn search(query: &str) -> Result<Vec<Hit>, String> {
    let url = format!("{SEARCH}?s={}&type=1&offset=0&limit={LIMIT}", esc(query));
    let Some(v) = get_json(&url, Some(REFERER))? else { return Ok(Vec::new()) };
    check_code(&v)?;
    let songs = v.pointer("/result/songs").and_then(|s| s.as_array()).cloned()
        .unwrap_or_default();
    Ok(songs.iter().map(|s| Hit {
        source: NAME,
        title: s.get("name").and_then(|x| x.as_str()).unwrap_or("").to_string(),
        artist: s.pointer("/artists/0/name").and_then(|x| x.as_str())
            .unwrap_or("").to_string(),
        album: s.pointer("/album/name").and_then(|x| x.as_str()).unwrap_or("").to_string(),
        // Milliseconds here, unlike LRCLIB's seconds.
        duration: s.get("duration").and_then(|x| x.as_i64()).unwrap_or(0) as f64 / 1000.0,
        instrumental: false,
        plain: None,
        synced: None,
        id: s.get("id").and_then(|x| x.as_i64()).map(|id| id.to_string()),
    }).collect())
}

/// Fetch the lyrics for one search hit.
pub fn fill_lyrics(h: &mut Hit) -> Result<(), String> {
    let Some(id) = h.id.clone() else { return Ok(()) };
    let url = format!("{LYRIC}?id={id}&lv=1&kv=1&tv=-1");
    let Some(v) = get_json(&url, Some(REFERER))? else { return Ok(()) };
    check_code(&v)?;
    let text = v.pointer("/lrc/lyric").and_then(|x| x.as_str()).unwrap_or("");
    if text.trim().is_empty() { return Ok(()); }
    // A sheet with no timestamps at all is still worth having — the sync editor
    // can time it — but it must not claim to be synced.
    if text.contains('[') && text.contains(']') {
        h.synced = Some(text.to_string());
    } else {
        h.plain = Some(text.to_string());
    }
    Ok(())
}

/// Best match for a track, lyrics included.
///
/// Ranks *before* fetching, because each lyric costs a request: the candidates
/// are scored on title, artist and duration, then the best few are fetched in
/// order until one actually has words. Searching by artist and title together
/// first, then by title alone, mirrors the LRCLIB ladder.
pub fn find(
    artist: &str, title: &str, _album: &str, duration: Option<Duration>,
) -> Result<Option<Hit>, String> {
    let (na, nt) = (normalise_artist(artist), normalise_title(title));
    if nt.is_empty() { return Ok(None); }

    // Request budget matters here: this service rate-limits, and every wasted
    // call brings the next real lookup closer to a refusal. So the second
    // search only runs when the first found nothing worth having, and only the
    // best couple of candidates have their lyrics fetched.
    let rank = |pool: Vec<Hit>| -> Vec<Hit> {
        let mut v: Vec<Hit> = pool.into_iter()
            .filter(|h| acceptable(h, &na, &nt, duration))
            .collect();
        v.sort_by(|a, b| {
            score(b, &na, &nt, duration)
                .partial_cmp(&score(a, &na, &nt, duration))
                .unwrap_or(std::cmp::Ordering::Equal)
        });
        v
    };

    let mut ranked = if na.is_empty() {
        Vec::new()
    } else {
        rank(search(&format!("{na} {nt}"))?)
    };
    if ranked.is_empty() { ranked = rank(search(&nt)?); }

    // Two, not one: an instrumental or karaoke upload can rank well and carry
    // nothing, and giving up after the first would miss the real take. Two, not
    // four: past that the candidates are poor enough that the requests are
    // better saved.
    for mut h in ranked.into_iter().take(2) {
        fill_lyrics(&mut h)?;
        if h.best_text().is_some() { return Ok(Some(h)); }
    }
    Ok(None)
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A listed row has no lyrics yet, and must not pretend otherwise.
    ///
    /// The id used to live in `plain`, which is the very field `best_text`
    /// reads — so the UI displayed the raw handle where the song should be. The
    /// earlier test for this was worthless: it blanked `plain` by hand before
    /// checking, so it exercised a state the real code never produced. This one
    /// asserts on a hit exactly as `search` builds it.
    #[test]
    fn a_listed_row_carries_no_text_until_it_is_fetched() {
        let listed = Hit {
            source: NAME,
            title: "恋の罠".into(),
            id: Some("26136440".into()),
            ..Default::default()
        };
        assert!(listed.best_text().is_none(),
                "an unfetched row must not surface its id as lyrics");
        assert!(!listed.has_synced());
        // And a row with no id at all is simply nothing to fetch, not an error.
        let mut bare = Hit::default();
        assert!(fill_lyrics(&mut bare).is_ok());
        assert!(bare.best_text().is_none());
    }

    /// The service answers everything with HTTP 200 and hides the real status in
    /// the body, so a refusal is indistinguishable from an empty result unless
    /// the code is read. Reading 405 as "no lyrics" tells the user their track
    /// does not exist when the truth is to wait a minute.
    #[test]
    fn a_throttled_response_is_an_error_not_an_empty_result() {
        let limited = serde_json::json!({
            "code": 405, "message": "操作频繁，请稍候再试", "msg": "操作频繁，请稍候再试"
        });
        let e = check_code(&limited).unwrap_err();
        assert!(e.contains("rate limited"), "{e}");

        // Other refusals carry their own detail through.
        let other = serde_json::json!({"code": 400, "message": "bad"});
        assert!(check_code(&other).unwrap_err().contains("400"));

        // A real answer passes, and so does one with no code at all.
        assert!(check_code(&serde_json::json!({"code": 200, "result": {}})).is_ok());
        assert!(check_code(&serde_json::json!({"result": {}})).is_ok());
    }

    /// Guards the same mistake at the source: whatever `search` returns must be
    /// text-free, whichever field a future edit decides to put the handle in.
    #[test]
    fn search_rows_are_built_without_text() {
        let json = serde_json::json!({"result": {"songs": [
            {"id": 26136440, "name": "恋の罠", "duration": 241000,
             "artists": [{"name": "KOTOKO"}], "album": {"name": "X"}}
        ]}});
        let songs = json.pointer("/result/songs").unwrap().as_array().unwrap();
        let h = Hit {
            source: NAME,
            title: songs[0]["name"].as_str().unwrap().into(),
            duration: songs[0]["duration"].as_i64().unwrap() as f64 / 1000.0,
            id: songs[0]["id"].as_i64().map(|i| i.to_string()),
            ..Default::default()
        };
        assert_eq!(h.duration, 241.0, "durations arrive in milliseconds");
        assert!(h.best_text().is_none());
        assert_eq!(h.id.as_deref(), Some("26136440"));
    }

    /// How many requests in a row the service will actually answer.
    ///
    /// `cargo test --bin moosik -- --ignored --nocapture netease_budget`
    #[test]
    #[ignore = "network — run explicitly"]
    fn netease_budget() {
        use super::super::http::get_json;
        for i in 0..24 {
            let url = format!("{SEARCH}?s={}&type=1&offset=0&limit=5", esc("ダダダダダル"));
            let n = match get_json(&url, Some(REFERER)) {
                Ok(Some(v)) => v.pointer("/result/songs").and_then(|s| s.as_array())
                    .map(|a| a.len()).unwrap_or(0),
                Ok(None) => 0,
                Err(e) => { println!("{i}: ERROR {e}"); continue; }
            };
            println!("{i}: {n} songs");
            if n == 0 { println!("   ^ stopped answering after {i} requests"); break; }
        }
    }

    /// Raw endpoint check, for telling "the service refused us" apart from
    /// "our matching rejected everything".
    ///
    /// `cargo test --bin moosik -- --ignored --nocapture netease_raw`
    #[test]
    #[ignore = "network — run explicitly"]
    fn netease_raw() {
        use super::super::http::get_json;
        for q in ["恋の罠", "ダダダダダル"] {
            let url = format!("{SEARCH}?s={}&type=1&offset=0&limit=5", esc(q));
            match get_json(&url, Some(REFERER)) {
                Err(e) => println!("{q}: HTTP ERROR {e}"),
                Ok(None) => println!("{q}: empty body"),
                Ok(Some(v)) => {
                    let code = v.get("code").and_then(|c| c.as_i64());
                    let n = v.pointer("/result/songs").and_then(|s| s.as_array())
                        .map(|a| a.len()).unwrap_or(0);
                    println!("{q}: code={code:?} songs={n}");
                    if n == 0 {
                        // Print the shape, not the payload, to see what it is
                        // objecting to.
                        println!("   keys: {:?}", v.as_object()
                                 .map(|o| o.keys().cloned().collect::<Vec<_>>()));
                        println!("   {:.200}", v.to_string());
                    }
                }
            }
        }
    }

    /// `cargo test --bin moosik -- --ignored --nocapture live_netease`
    #[test]
    #[ignore = "network — run explicitly"]
    fn live_netease() {
        // The tracks LRCLIB had nothing for.
        let cases: &[(&str, &str, u64)] = &[
            ("しゃいと", "聖人君子でありたい (feat. 重音テト)", 129),
            ("雨良 Amala", "ダダダダダル", 160),
            ("マサラダ", "カンケーガール", 243),
            ("重音テト", "ちっちゃな私", 231),
            ("まちーしゃ", "いつまでたってもテイカー (feat. 重音テト)", 161),
            ("TAK, Hatsune Miku", "MTMTM (feat. Hatsune Miku)", 124),
            // Romanised tag, Japanese record: only the manual search can bridge
            // this, so it also checks that a picked row fetches its text.
            ("", "恋の罠", 241),
        ];
        for (artist, title, secs) in cases {
            // Paced deliberately. Firing every case back to back sends ~20
            // requests in a few seconds and trips the rate limit, which then
            // fails the rest of the run for reasons that have nothing to do
            // with matching. Real use is one lookup per button press.
            std::thread::sleep(Duration::from_secs(6));
            match find(artist, title, "", Some(Duration::from_secs(*secs))) {
                // Report only metadata and line counts, never the words.
                Ok(Some(h)) => println!(
                    "{title}\n   -> {} / {} ({:.0}s) synced={} lines={}",
                    h.title, h.artist, h.duration, h.has_synced(),
                    h.best_text().map(|t| t.lines().count()).unwrap_or(0)),
                Ok(None) => println!("{title}\n   -> no match"),
                Err(e) => println!("{title}\n   -> ERROR {e}"),
            }
        }
    }
}
