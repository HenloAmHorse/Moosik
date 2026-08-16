//! Ranking and acceptance, shared by every lyric source.
//!
//! Fetching is the easy half. The hard half is deciding whether what came back
//! is the same song — and refusing it when it is not, because wrong lyrics
//! applied confidently are worse than none: nothing tells the user to look.
//!
//! Tags in the wild carry decoration the databases do not, Vocaloid tags
//! especially: `【初音ミク】タイトル【オリジナルMV】`, an artist field holding the
//! producer where the database has the vocalist, full-width punctuation,
//! `feat.` chains. Normalisation strips all of that before anything is compared.

use std::time::Duration;

/// One candidate record, from whichever source produced it.
#[derive(Clone, Debug, Default)]
pub struct Hit {
    /// Which source this came from, for the UI to name.
    pub source: &'static str,
    pub title: String,
    pub artist: String,
    pub album: String,
    /// Track length in seconds, as the database has it. 0 when unknown.
    pub duration: f64,
    pub instrumental: bool,
    pub plain: Option<String>,
    pub synced: Option<String>,
    /// Opaque handle the producing source needs to fetch this record's lyrics
    /// later, for sources that list results without them.
    ///
    /// Its own field on purpose. It was first parked inside `plain`, on the
    /// reasoning that only one back end would ever look — and `best_text`
    /// promptly handed the raw handle to the UI as though it were the song,
    /// because `plain` is precisely the field it reads. A value that must never
    /// be mistaken for lyrics does not belong in a field that means lyrics.
    pub id: Option<String>,
}

impl Hit {
    /// Prefer synced; fall back to plain so an obscure track still shows
    /// *something* the user can then time by hand.
    pub fn best_text(&self) -> Option<&str> {
        self.synced.as_deref().or(self.plain.as_deref())
    }
    pub fn has_synced(&self) -> bool {
        self.synced.as_deref().is_some_and(|s| !s.trim().is_empty())
    }
}

/// How far a candidate's duration may sit from the track's before it is
/// rejected, when nothing else corroborates it.
///
/// Deliberately looser than the +/-2 s LRCLIB's own guidance suggests. A false
/// rejection is silent — the user sees "no lyrics" for a song a database has —
/// and several seconds of honest disagreement between a file and a database is
/// normal. Ranking still prefers the closest take, so the gate only has to
/// exclude the obviously wrong, like a 98 s edit standing in for a 180 s song.
const DURATION_TOLERANCE_S: f64 = 8.0;

/// Tolerance once the title and artist both agree strongly.
///
/// The flat ±8 s gate was rejecting correct matches. `MTMTM` is the worked
/// example: LRCLIB records it as 133 s, the file is 124 s, and 9 s put it just
/// outside — so a record with an identical title *and* an identical artist was
/// thrown away. Databases disagree with files about length routinely, because
/// they are populated from whatever rip the uploader had.
///
/// The duration gate exists to disambiguate between candidates. When the title
/// and artist already identify the song, there is nothing left to disambiguate
/// and the gate is only doing damage. It stays narrow when they do not.
const DURATION_TOLERANCE_CORROBORATED_S: f64 = 30.0;

/// Seconds of disagreement allowed, given how well everything else matches.
fn tolerance(h: &Hit, norm_artist: &str, norm_title: &str) -> f64 {
    let strong_title = similarity(norm_title, &normalise_title(&h.title)) >= 0.9;
    let strong_artist = !norm_artist.is_empty()
        && artists_agree(norm_artist, &normalise_artist(&h.artist));
    if strong_title && strong_artist {
        DURATION_TOLERANCE_CORROBORATED_S
    } else {
        DURATION_TOLERANCE_S
    }
}

fn duration_ok(
    h: &Hit, want: Option<Duration>, norm_artist: &str, norm_title: &str,
) -> bool {
    match want {
        // A record with no duration is not evidence against itself.
        Some(d) if h.duration > 0.0 =>
            (h.duration - d.as_secs_f64()).abs() <= tolerance(h, norm_artist, norm_title),
        _ => true,
    }
}

/// Least title agreement an automatic match may have.
///
/// Without this the scorer would accept a hit whose title agreed *not at all*,
/// because a synced sheet with a close duration outscored the zero it got for
/// the title. That is how `SWIPE×SWIPE` acquired the lyrics of an unrelated
/// song called `Swipe` — and wrong lyrics, confidently applied, are worse than
/// none, because nothing tells the user to go looking.
const MIN_TITLE_SIM: f64 = 0.62;
/// A matching title is not enough on its own: covers, and different songs that
/// happen to share a name, both clear it. Something else has to agree too —
/// either the artist, or a duration close enough to be the same recording.
const MIN_ARTIST_SIM: f64 = 0.4;
const CORROBORATING_SECS: f64 = 3.0;

/// Whether two normalised artist strings name the same act.
///
/// Straight similarity is too strict here, because the two sides routinely
/// list different numbers of credits: a tag of `TAK` against a database row of
/// `TAK/Hatsune Miku` scores 0.27 and would be refused, though it is plainly
/// the same artist. So one being contained in the other counts as agreement —
/// which is asymmetric on purpose, since a credit list is a superset, not a
/// different name. `えいぷ` is still nowhere inside `ステレオポニー`.
fn artists_agree(a: &str, b: &str) -> bool {
    if a.is_empty() || b.is_empty() { return false; }
    if similarity(a, b) >= MIN_ARTIST_SIM { return true; }
    let (short, long) = if a.chars().count() <= b.chars().count() { (a, b) } else { (b, a) };
    // Guard against a one- or two-character "artist" matching everything.
    short.chars().count() >= 3 && long.contains(short)
}

/// Duration sanity alone, for a hit a source matched by exact field lookup.
///
/// There the database has already matched on the strings it was given, so the
/// title does not need re-checking; all that is left is whether it is the same
/// *recording*.
pub fn plausible(
    h: &Hit, norm_artist: &str, norm_title: &str, duration: Option<Duration>,
) -> bool {
    duration_ok(h, duration, norm_artist, norm_title)
}

/// Whether a candidate is close enough to accept without the user looking at
/// it. Manual search deliberately does not use this — there, the user *is* the
/// check, and hiding plausible rows would only get in the way.
pub fn acceptable(
    h: &Hit, norm_artist: &str, norm_title: &str, duration: Option<Duration>,
) -> bool {
    if !duration_ok(h, duration, norm_artist, norm_title) { return false; }
    if similarity(norm_title, &normalise_title(&h.title)) < MIN_TITLE_SIM { return false; }
    if norm_artist.is_empty() { return true; }
    if artists_agree(norm_artist, &normalise_artist(&h.artist)) { return true; }
    match duration {
        Some(d) if h.duration > 0.0 =>
            (h.duration - d.as_secs_f64()).abs() <= CORROBORATING_SECS,
        // Nothing to corroborate with, so the title alone has to carry it.
        _ => true,
    }
}

/// Pick the best candidate: duration first, then how well the text agrees.
///
/// `strict` applies [`acceptable`] — on for the automatic ladder, off when a
/// human is choosing.
pub fn best_of(
    hits: Vec<Hit>, norm_artist: &str, norm_title: &str, duration: Option<Duration>,
    strict: bool,
) -> Option<Hit> {
    hits.into_iter()
        .filter(|h| h.best_text().is_some()
                    && duration_ok(h, duration, norm_artist, norm_title))
        .filter(|h| !strict || acceptable(h, norm_artist, norm_title, duration))
        .max_by(|a, b| {
            score(a, norm_artist, norm_title, duration)
                .partial_cmp(&score(b, norm_artist, norm_title, duration))
                .unwrap_or(std::cmp::Ordering::Equal)
        })
}

/// Higher is better. Synced lyrics outrank plain ones outright, because a plain
/// sheet is a consolation prize; below that it is duration agreement, then how
/// much of the title and artist survive in common.
pub fn score(h: &Hit, norm_artist: &str, norm_title: &str, duration: Option<Duration>) -> f64 {
    let mut s = 0.0;
    if h.has_synced() { s += 10.0; }
    if let (Some(d), true) = (duration, h.duration > 0.0) {
        let off = (h.duration - d.as_secs_f64()).abs();
        s += 6.0 * (1.0 - (off / DURATION_TOLERANCE_S).min(1.0));
    }
    s += 5.0 * similarity(norm_title, &normalise_title(&h.title));
    s += 3.0 * similarity(norm_artist, &normalise_artist(&h.artist));
    if h.instrumental { s -= 8.0; }
    s
}

/// Character-bigram Dice coefficient, 0–1.
///
/// Word-token overlap was the first attempt and it is useless here: Japanese
/// does not put spaces between words, so every CJK title is a single token and
/// the score is 1 or 0 with nothing in between. `ぽい` against `ぽいぽい` scored
/// zero, and `SWIPE×SWIPE` against `Swipe` scored zero while still being
/// *accepted*, because nothing required the title to agree at all.
///
/// Bigrams work on both scripts without a word segmenter: they measure how much
/// of one string's character sequence appears in the other's.
pub fn similarity(a: &str, b: &str) -> f64 {
    let bigrams = |s: &str| -> Vec<(char, char)> {
        let cs: Vec<char> = s.chars().filter(|c| !c.is_whitespace()).collect();
        if cs.len() < 2 {
            // A one-character title has no bigram; pair it with itself so it
            // can still match another one-character title exactly.
            return cs.first().map(|&c| vec![(c, c)]).unwrap_or_default();
        }
        cs.windows(2).map(|w| (w[0], w[1])).collect()
    };
    let (mut x, mut y) = (bigrams(a), bigrams(b));
    if x.is_empty() || y.is_empty() { return 0.0; }
    x.sort_unstable(); y.sort_unstable();
    // Multiset intersection, so a repeated bigram only counts as often as both
    // sides actually contain it.
    let (mut i, mut j, mut shared) = (0, 0, 0usize);
    while i < x.len() && j < y.len() {
        match x[i].cmp(&y[j]) {
            std::cmp::Ordering::Less => i += 1,
            std::cmp::Ordering::Greater => j += 1,
            std::cmp::Ordering::Equal => { shared += 1; i += 1; j += 1; }
        }
    }
    2.0 * shared as f64 / (x.len() + y.len()) as f64
}

/// Fold full-width Latin and the ideographic space to ASCII.
///
/// Japanese tags are routinely typed in a full-width IME, so `ＭＩＫＵ` and
/// `MIKU` are different strings to every byte comparison but the same word to a
/// human — and to the database, which stores whichever the uploader typed.
fn fold_width(s: &str) -> String {
    s.chars()
        .map(|c| match c as u32 {
            0xFF01..=0xFF5E => char::from_u32(c as u32 - 0xFEE0).unwrap_or(c),
            0x3000 => ' ',
            _ => c,
        })
        .collect()
}

/// Strip bracketed groups: `【】`, `（）`, `〔〕`, `()`, `[]`, `{}`.
///
/// Only when something survives — a title that is *entirely* bracketed is a
/// title, not decoration, and emptying it would be worse than leaving it.
fn strip_brackets(s: &str) -> String {
    let mut out = String::with_capacity(s.len());
    let mut depth = 0u32;
    for c in s.chars() {
        match c {
            '【' | '（' | '〔' | '［' | '(' | '[' | '{' => depth += 1,
            '】' | '）' | '〕' | '］' | ')' | ']' | '}' if depth > 0 => depth -= 1,
            _ if depth == 0 => out.push(c),
            _ => {}
        }
    }
    if out.trim().is_empty() { s.to_string() } else { out }
}

/// Decorations that are never part of a song's name.
const NOISE: &[&str] = &[
    "official music video", "official video", "official audio", "music video",
    "lyric video", "off vocal", "instrumental", "karaoke", "full version",
    "short version", "full ver", "short ver", "tv size", "hq", "hd",
];

fn collapse(s: &str) -> String {
    s.split_whitespace().collect::<Vec<_>>().join(" ")
}

/// Normalise a title for searching: fold width, drop bracketed decoration, cut
/// at `feat.`, remove known noise words, lowercase, collapse whitespace.
pub fn normalise_title(s: &str) -> String {
    let s = fold_width(s);
    let s = strip_brackets(&s);
    let mut s = s.to_lowercase();
    for sep in [" feat.", " feat ", " ft.", " ft ", " featuring "] {
        if let Some(i) = s.find(sep) { s.truncate(i); }
    }
    for n in NOISE {
        s = s.replace(n, " ");
    }
    // Separators that carry no meaning once the words are split out.
    let s: String = s.chars()
        .map(|c| if "-_/\\|~・､、,;:!?\"'’”“".contains(c) { ' ' } else { c })
        .collect();
    collapse(&s)
}

/// Normalise an artist for searching.
///
/// Same treatment as a title, plus: only the first credit is kept. A Vocaloid
/// track is routinely tagged `ProducerP feat. 初音ミク`, or with the whole chorus
/// listed, while the database has one name — matching on the first is far more
/// likely to land than matching on the whole string.
pub fn normalise_artist(s: &str) -> String {
    let base = normalise_title(s);
    for sep in [" & ", " x ", " vs ", " with "] {
        if let Some(i) = base.find(sep) { return collapse(&base[..i]); }
    }
    base
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn folds_full_width_latin() {
        assert_eq!(normalise_title("ＭＩＫＵ　Ｓｏｎｇ"), "miku song");
    }

    #[test]
    fn strips_vocaloid_decoration() {
        // The exact shape that defeats a plain title query.
        assert_eq!(normalise_title("【初音ミク】ロキ【オリジナルMV】"), "ロキ");
        assert_eq!(normalise_title("Senbonzakura (Official Music Video)"), "senbonzakura");
    }

    #[test]
    fn a_fully_bracketed_title_is_left_alone() {
        // Stripping here would leave nothing to search for.
        assert_eq!(normalise_title("【ロキ】"), "【ロキ】".to_lowercase());
    }

    #[test]
    fn cuts_at_a_feature_credit() {
        assert_eq!(normalise_title("Song Name feat. Hatsune Miku"), "song name");
        // No space after the marker either — " ft." is enough to recognise it.
        assert_eq!(normalise_title("Song Name ft.初音ミク"), "song name");
        // But a word merely starting with those letters is not a credit.
        assert_eq!(normalise_title("Soft Machine"), "soft machine");
    }

    #[test]
    fn artist_keeps_only_the_first_credit() {
        assert_eq!(normalise_artist("DECO*27 feat. 初音ミク"), "deco*27");
        assert_eq!(normalise_artist("Kanaria & 可不"), "kanaria");
    }

    #[test]
    fn noise_words_go_away() {
        assert_eq!(normalise_title("Track - TV Size"), "track");
        assert_eq!(normalise_title("Track [Off Vocal]"), "track");
    }

    #[test]
    fn duration_gate_separates_a_song_from_its_extended_mix() {
        let short = Hit { duration: 210.0, ..Default::default() };
        let long  = Hit { duration: 400.0, ..Default::default() };
        let want = Some(Duration::from_secs(212));
        assert!(duration_ok(&short, want, "", ""));
        assert!(!duration_ok(&long, want, "", ""));
        // No duration on either side means no opinion, not a rejection.
        assert!(duration_ok(&long, None, "", ""));
        assert!(duration_ok(&Hit { duration: 0.0, ..Default::default() }, want, "", ""));
    }

    /// The regression that lost `MTMTM`: a database recorded it at 133 s, the
    /// file was 124 s, and a flat ±8 s gate threw away a record whose title and
    /// artist were both identical.
    #[test]
    fn a_corroborated_match_survives_a_length_disagreement() {
        let h = Hit { title: "MTMTM".into(), artist: "TAK".into(), duration: 133.0,
                      synced: Some("x".into()), ..Default::default() };
        let want = Some(Duration::from_secs(124));
        assert!(duration_ok(&h, want, "tak", "mtmtm"),
                "identical title and artist should widen the gate");
        assert!(acceptable(&h, "tak", "mtmtm", want));
        // With nothing corroborating it, the same 9 s gap is still refused.
        assert!(!duration_ok(&h, want, "", "something else"));
        // And the widened gate is not unlimited — an extended mix still loses.
        let long = Hit { duration: 300.0, ..h.clone() };
        assert!(!duration_ok(&long, want, "tak", "mtmtm"));
    }

    #[test]
    fn synced_outranks_plain_and_duration_breaks_ties() {
        let plain = Hit { title: "a".into(), duration: 200.0,
                          plain: Some("x".into()), ..Default::default() };
        let synced = Hit { title: "a".into(), duration: 200.0,
                           synced: Some("[00:01.00]x".into()), ..Default::default() };
        let want = Some(Duration::from_secs(200));
        assert!(score(&synced, "", "a", want) > score(&plain, "", "a", want));

        let near = Hit { duration: 201.0, ..synced.clone() };
        let far  = Hit { duration: 203.5, ..synced.clone() };
        assert!(score(&near, "", "a", want) > score(&far, "", "a", want));
    }

    #[test]
    fn best_of_rejects_everything_out_of_range() {
        let hits = vec![Hit { title: "t".into(), duration: 400.0,
                              synced: Some("x".into()), ..Default::default() }];
        assert!(best_of(hits, "", "t", Some(Duration::from_secs(200)), true).is_none());
    }

    /// A credit list is a superset of the artist, not a different artist.
    #[test]
    fn a_longer_credit_list_still_names_the_same_act() {
        assert!(artists_agree("tak", "tak hatsune miku"));
        assert!(artists_agree("tak hatsune miku", "tak"));
        assert!(artists_agree("deco*27", "deco*27 hatsune miku"));
        // But an unrelated name is not rescued by being short.
        assert!(!artists_agree("えいぷ", "ステレオポニー"));
        // Nor may a two-character fragment match everything it appears in.
        assert!(!artists_agree("ab", "abcdefgh"));
    }

    #[test]
    fn no_artist_to_compare_lets_the_title_decide_alone() {
        // Some files have no usable artist tag at all; that must not become an
        // automatic refusal of every candidate.
        let h = Hit { title: "KING".into(), artist: "Kanaria".into(), duration: 152.0,
                      synced: Some("x".into()), ..Default::default() };
        assert!(acceptable(&h, "", "king", None));
    }

    /// The bug this gate exists for: a synced sheet with a plausible duration
    /// used to win on score alone, even with a title that agreed not at all.
    #[test]
    fn a_hit_whose_title_does_not_agree_is_refused() {
        let hits = vec![Hit {
            title: "Swipe".into(), artist: "Natoo".into(), duration: 203.0,
            synced: Some("[00:01.00]x".into()), ..Default::default()
        }];
        let want = Some(Duration::from_secs(203));
        assert!(best_of(hits.clone(), "someone", "swipe×swipe", want, true).is_none(),
                "wrong lyrics are worse than none");
        // The same row is still offered when a human is doing the choosing.
        assert!(best_of(hits, "someone", "swipe×swipe", want, false).is_some());
    }

    /// A title can match while the song does not: covers, and unrelated songs
    /// that share a name. Something else has to corroborate.
    #[test]
    fn a_matching_title_alone_is_not_enough() {
        let cover = Hit {
            title: "はんぶんこ".into(), artist: "ステレオポニー".into(), duration: 243.0,
            synced: Some("x".into()), ..Default::default()
        };
        // Different artist, duration nowhere near: refused.
        assert!(!acceptable(&cover, "えいぷ", "はんぶんこ", Some(Duration::from_secs(200))));
        // Same title and a duration that says it is the same recording: taken.
        assert!(acceptable(&cover, "えいぷ", "はんぶんこ", Some(Duration::from_secs(243))));
        // Right artist, and the duration no longer has to carry the decision —
        // a few seconds of disagreement between a file and a database is normal.
        assert!(acceptable(&cover, "ステレオポニー", "はんぶんこ", Some(Duration::from_secs(235))));
        // But agreement on name is not a licence to ignore length entirely: 43 s
        // apart is a different recording, not a different rip of the same one.
        assert!(!acceptable(&cover, "ステレオポニー", "はんぶんこ", Some(Duration::from_secs(200))));
    }

    /// Bracketed credits are stripped before comparison, so the database's
    /// `MTMTM (feat. Hatsune Miku)` and a tag of `MTMTM` are the same title.
    #[test]
    fn decorated_database_titles_still_agree() {
        let h = Hit {
            title: "MTMTM (feat. Hatsune Miku)".into(), artist: "TAK/Hatsune Miku".into(),
            duration: 133.0, synced: Some("x".into()), ..Default::default()
        };
        assert!(acceptable(&h, "tak", "mtmtm", Some(Duration::from_secs(133))));
    }

    /// Word tokens cannot measure Japanese, which has no spaces — this is the
    /// whole reason the similarity is character-based.
    #[test]
    fn similarity_works_without_word_boundaries() {
        assert!((similarity("ぽい", "ぽい") - 1.0).abs() < 1e-9);
        assert!(similarity("ヴァンパイア", "ヴァンパイア") > 0.99);
        // Related but not the same, and it must land below the accept gate.
        assert!(similarity("swipe×swipe", "swipe") < MIN_TITLE_SIM,
                "got {}", similarity("swipe×swipe", "swipe"));
        // Genuinely unrelated.
        assert!(similarity("カンケーガール", "senbonzakura") < 0.2);
        assert_eq!(similarity("", "abc"), 0.0);
    }

    #[test]
    fn an_instrumental_loses_to_a_real_sheet() {
        let want = Some(Duration::from_secs(200));
        let inst = Hit { duration: 200.0, instrumental: true,
                         synced: Some("x".into()), title: "a".into(), ..Default::default() };
        let real = Hit { duration: 200.0, synced: Some("x".into()),
                         title: "a".into(), ..Default::default() };
        assert!(score(&real, "", "a", want) > score(&inst, "", "a", want));
    }

    #[test]
    fn similarity_is_bounded_and_symmetric() {
        for (a, b) in [("abc", "abc"), ("abc", "xyz"), ("", ""), ("a", "a"), ("ab", "abcd")] {
            let s = similarity(a, b);
            assert!((0.0..=1.0).contains(&s), "{a}/{b} gave {s}");
            assert!((s - similarity(b, a)).abs() < 1e-12, "{a}/{b} is not symmetric");
        }
        assert_eq!(similarity("abc", "abc"), 1.0);
        // Single characters have no bigram of their own but must still compare.
        assert_eq!(similarity("あ", "あ"), 1.0);
        assert_eq!(similarity("あ", "い"), 0.0);
    }
}
