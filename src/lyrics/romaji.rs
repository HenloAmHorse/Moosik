//! Japanese → romaji, for lyric lines no source supplied romaji for.
//!
//! Readings come from a morphological analyser (lindera with IPADIC, embedded),
//! because kana is fixed but kanji is not: 今日 is *kyou* or *konnichi* by
//! context, and only splitting the sentence into words decides which. Even so
//! this is a guess wherever the singer's reading departs from the dictionary's
//! — 本気 sung as *maji*, 運命 as *sadame* — and lyrics do that on purpose. So
//! whatever this produces is shown as "auto" and can be corrected by hand; a
//! correction is kept, and this is never asked about that line again.

use lindera::dictionary::load_dictionary;
use lindera::mode::Mode;
use lindera::segmenter::Segmenter;
use std::borrow::Cow;
use std::sync::OnceLock;

/// Loaded on first use: the dictionary is sizeable to deserialise, and most
/// sessions never show a Japanese sheet.
fn segmenter() -> Option<&'static Segmenter> {
    static S: OnceLock<Option<Segmenter>> = OnceLock::new();
    S.get_or_init(|| {
        let d = load_dictionary("embedded://ipadic").ok()?;
        Some(Segmenter::new(Mode::Normal, d, None))
    }).as_ref()
}

fn is_kana(c: char) -> bool {
    matches!(c, '\u{3041}'..='\u{3096}' | '\u{30A1}'..='\u{30FA}' | 'ー')
}

fn is_kanji(c: char) -> bool {
    matches!(c, '\u{4E00}'..='\u{9FFF}' | '\u{3400}'..='\u{4DBF}' | '々')
}

/// True when the text has kana. Kanji alone does not count: a Chinese sheet is
/// all kanji, and reading it through a Japanese dictionary produces nonsense.
pub fn has_kana(s: &str) -> bool {
    s.chars().any(|c| is_kana(c) && c != 'ー')
}

fn is_japanese(s: &str) -> bool {
    s.chars().any(|c| is_kana(c) || is_kanji(c))
}

/// One word of a lyric line: the text as written, and how it is read.
#[derive(Clone, Debug, PartialEq, Default)]
pub struct Word {
    pub text: String,
    pub romaji: String,
    /// The reading in hiragana — what furigana shows. Empty when not Japanese.
    pub kana: String,
    /// False for Latin, spaces and punctuation, which read as themselves and
    /// get nothing written under them.
    pub jp: bool,
}

/// Romaji for one lyric line. Text with nothing Japanese in it comes back
/// unchanged, so an English line in a Japanese song stays as written.
#[cfg(test)]
pub fn line(text: &str) -> String {
    join(&words(text))
}

/// The words of a line read out as one line of romaji.
pub fn join(words: &[Word]) -> String {
    let per: Vec<String> = words.iter().map(|w| w.romaji.clone()).collect();
    join_as(words, &per)
}

/// The words read out as one line, with `per[k]` as word `k`'s romaji.
fn join_as(words: &[Word], per: &[String]) -> String {
    if words.iter().all(|w| !w.jp) { return words.iter().map(|w| w.text.as_str()).collect(); }
    let mut out = String::new();
    let mut prev: Option<(&Word, &str)> = None;
    for (w, r) in words.iter().zip(per) {
        if let Some((p, pr)) = prev {
            // Latin written straight against Japanese still needs a space
            // once both are romaji: が Bigger, not "gaBigger".
            let meets_latin = p.jp != w.jp
                && pr.ends_with(|c: char| c.is_ascii_alphanumeric())
                && r.starts_with(|c: char| c.is_ascii_alphanumeric());
            if (p.jp && w.jp) || meets_latin { out.push(' '); }
        }
        out.push_str(r);
        prev = Some((w, r));
    }
    tidy(&out)
}

/// True when the word carries furigana written into the sheet, `剣(つるぎ)`.
pub fn has_sheet_furigana(w: &Word) -> bool {
    w.jp && w.text.contains(['(', '（']) && chunks(&w.text).iter().any(|c| c.1.is_some())
}

/// The word as written with any bracketed furigana taken out: `剣(つるぎ)` → `剣`.
pub fn without_brackets(text: &str) -> String {
    chunks(text).into_iter()
        .map(|(written, kana)| match kana {
            Some(_) => written[..written.find(['(', '（']).unwrap_or(written.len())].to_string(),
            None => written,
        })
        .collect()
}

/// Furigana for one line, as drawn.
#[derive(Clone, Debug, PartialEq)]
pub enum Furigana {
    /// Per word, the pieces of [`ruby`].
    Words(Vec<Vec<(String, Option<String>)>>),
    /// The reading could not be placed word by word, so it goes over the
    /// whole line — honest about what is known, rather than showing a
    /// per-word reading that is not the one the user chose.
    WholeLine { text: String, kana: String },
}

/// How one line reads, once the sheet's own furigana, any supplied romaji and
/// any confirmed override are taken together. Every view draws from this.
#[derive(Clone, Debug, PartialEq)]
pub struct Reading {
    /// Romaji per word, or `None` where it cannot be placed word by word.
    pub words: Option<Vec<String>>,
    /// The line's romaji as shown.
    pub line: String,
    /// Generated rather than supplied: shown in italics.
    pub guessed: bool,
    pub furigana: Furigana,
}

/// Work out how a line reads.
///
/// `given` is romaji someone supplied: a lyrics source, or the user.
/// `overrides` is true only when the user has **confirmed** that their reading
/// replaces the sheet's own furigana on this line.
///
/// The policy:
/// - A word the sheet annotates, `剣(つるぎ)`, reads as annotated. That holds
///   even against supplied romaji, a source's included; it is what the lyricist
///   wrote.
/// - Only a confirmed override changes that, and then it changes it
///   everywhere: Romaji, Both and Furigana.
/// - If supplied romaji cannot be placed word by word, the fallback is
///   explicit. With an override, the user's reading goes over the whole line.
///   Without one, on a line the sheet annotates, the sheet's reading is shown
///   as generated (italic); a source reading that cannot be placed is never
///   shown in a way that contradicts the sheet's annotation.
pub fn read_line(words: &[Word], given: Option<&str>, overrides: bool) -> Reading {
    let generated: Vec<String> = words.iter().map(|w| w.romaji.clone()).collect();
    let as_written: String = words.iter().map(|w| w.text.as_str()).collect();
    let sheet_ruby = || Furigana::Words(words.iter().map(|w| ruby(w, None)).collect());
    let Some(g) = given else {
        let line = join_as(words, &generated);
        return Reading { guessed: line != as_written, line, words: Some(generated), furigana: sheet_ruby() };
    };
    match align_with(words, g, &has_sheet_furigana) {
        Some(a) => {
            let per: Vec<String> = words.iter().zip(&a).map(|(w, r)| {
                if has_sheet_furigana(w) && !overrides { w.romaji.clone() } else { r.clone() }
            }).collect();
            let furigana = Furigana::Words(words.iter().zip(&a).map(|(w, r)| {
                if !w.jp {
                    vec![(w.text.clone(), None)]
                } else if has_sheet_furigana(w) && !overrides {
                    ruby(w, None)
                } else {
                    let base = Word { text: without_brackets(&w.text), ..w.clone() };
                    ruby(&base, Some(&romaji_to_kana(r)))
                }
            }).collect());
            Reading { line: join_as(words, &per), words: Some(per), guessed: false, furigana }
        }
        None if overrides => Reading {
            words: None,
            line: g.to_string(),
            guessed: false,
            furigana: Furigana::WholeLine { text: without_brackets(&as_written), kana: romaji_to_kana(g) },
        },
        None if words.iter().any(has_sheet_furigana) => {
            let line = join_as(words, &generated);
            Reading { line, words: Some(generated), guessed: true, furigana: sheet_ruby() }
        }
        None => Reading { words: None, line: g.to_string(), guessed: false, furigana: sheet_ruby() },
    }
}

/// Where a proposed reading disagrees with furigana the sheet wrote.
#[derive(Clone, Debug, PartialEq)]
pub struct Conflict {
    /// The word as written, without its brackets; the whole line when the
    /// reading could not be placed word by word.
    pub word: String,
    /// The sheet's reading, in kana.
    pub sheet: String,
    /// The proposed reading, in kana.
    pub proposed: String,
}

/// Each word the sheet annotates whose reading `proposed` would change.
/// Empty when it changes none — including on a line with no annotations.
pub fn conflicts(words: &[Word], proposed: &str) -> Vec<Conflict> {
    if !words.iter().any(has_sheet_furigana) { return Vec::new(); }
    let norm = |s: &str| -> String {
        s.chars().filter(|c| c.is_alphanumeric()).flat_map(char::to_lowercase).collect()
    };
    match align_with(words, proposed, &has_sheet_furigana) {
        Some(a) => words.iter().zip(&a)
            .filter(|(w, r)| has_sheet_furigana(w) && norm(r) != norm(&w.romaji))
            .map(|(w, r)| Conflict {
                word: without_brackets(&w.text),
                sheet: w.kana.clone(),
                proposed: romaji_to_kana(r),
            })
            .collect(),
        None if norm(proposed) != norm(&join(words)) => vec![Conflict {
            word: without_brackets(&words.iter().map(|w| w.text.as_str()).collect::<String>()),
            sheet: words.iter().map(|w| w.kana.as_str()).collect(),
            proposed: romaji_to_kana(proposed),
        }],
        None => Vec::new(),
    }
}

/// A lyric line split into words, each with its reading — what the viewer
/// draws when it puts romaji under each word rather than under the line.
///
/// Putting the words back together gives the line exactly as written.
pub fn words(text: &str) -> Vec<Word> {
    if !is_japanese(text) {
        return vec![Word { text: text.into(), romaji: text.into(), kana: String::new(), jp: false }];
    }
    let mut w = Words::default();
    for (written, furigana) in chunks(text) {
        match furigana {
            // Furigana is the sung reading and is taken whole. The analyser
            // never sees it: it would happily split つるぎ|を as つる|ぎを.
            Some(kana) => {
                let kana = std::mem::take(&mut w.held) + &kana;
                w.push(Word { text: written, romaji: kana_to_romaji(&kana), kana: hiragana(&kana), jp: true }, false);
                w.prev_pos = "名詞".into();
                w.seen_jp = true;
                w.after_furigana = true;
            }
            None => w.analyse(&written),
        }
    }
    let mut out = w.out;
    for w in &mut out {
        if w.jp { w.romaji = tidy(&w.romaji); }
    }
    out
}

#[derive(Default)]
struct Words {
    out: Vec<Word>,
    seen_jp: bool,
    prev_pos: String,
    /// A word-final っ doubles the next word's consonant: 気取っ+た → "kidotta".
    held: String,
    /// The last word came with furigana, so kana straight after it is its
    /// ending — 誘(いざな)う is one word — unless it is a particle.
    after_furigana: bool,
}

impl Words {
    /// Add a word, or extend the last one: endings stay on their word, and a
    /// run of Latin, spaces and punctuation is one piece.
    fn push(&mut self, w: Word, attach: bool) {
        match self.out.last_mut() {
            Some(last) if (attach && last.jp) || (!w.jp && !last.jp) => {
                last.text.push_str(&w.text);
                last.romaji.push_str(&w.romaji);
                last.kana.push_str(&w.kana);
            }
            _ => self.out.push(w),
        }
    }

    fn plain(&mut self, s: &str) {
        if !s.is_empty() {
            self.push(Word { text: s.into(), romaji: kana_to_romaji(s), kana: String::new(), jp: false }, false);
        }
    }

    fn analyse(&mut self, text: &str) {
        let Some(seg) = segmenter() else {
            let jp = is_japanese(text);
            let kana = if jp { hiragana(text) } else { String::new() };
            return self.push(Word { text: text.into(), romaji: kana_to_romaji(text), kana, jp }, false);
        };
        let Ok(mut tokens) = seg.segment(Cow::Borrowed(text)) else { return self.plain(text) };
        // The analyser drops whitespace, so the gaps between tokens are put
        // back from the text itself.
        let mut cursor = 0;
        for t in tokens.iter_mut() {
            let okurigana = std::mem::take(&mut self.after_furigana) && t.byte_start == 0;
            self.plain(&text[cursor..t.byte_start]);
            cursor = t.byte_end;
            let surface = t.surface.to_string();
            let d = t.details();
            // IPADIC: pos, pos1, pos2, pos3, conj type, conj form, base,
            // reading, pronunciation. Unknown words carry fewer fields.
            let known = d.len() >= 9 && d[7] != "*";
            let kana = if !known {
                surface.as_str()
            } else if d[0] == "助詞" && self.seen_jp {
                // The pronunciation field is what turns the particles は/へ/を
                // into wa/e/o. It is not used elsewhere: it also flattens
                // おう→オー, which would spell 東京 "tookyoo". And not at the
                // start of a line, where は is the interjection はぁ.
                d[8]
            } else {
                d[7]
            };
            // Endings join the word they inflect: 渇い+た → "kawaita", not
            // "kawai ta". After a noun they are words of their own: 学生です
            // → "gakusei desu", の為 → "no tame".
            let inflects = matches!(self.prev_pos.as_str(), "動詞" | "形容詞" | "助動詞");
            let attach = (known
                && ((inflects && (d[0] == "助動詞" || d[1] == "接続助詞" || d[1] == "非自立"))
                    || (d[1] == "接尾" && self.prev_pos != "助詞")))
                // A stray small kana or long mark is the tail of the word
                // before: はぁ, not "ha a".
                || surface.starts_with(['ぁ', 'ぃ', 'ぅ', 'ぇ', 'ぉ', 'ァ', 'ィ', 'ゥ', 'ェ', 'ォ', 'ー'])
                || (okurigana && !(known && d[0] == "助詞")
                    && surface.chars().all(|c| matches!(c, '\u{3041}'..='\u{3096}')));
            let pos = if known { d[0].to_string() } else { String::new() };
            let jp = is_japanese(&surface);

            let mut kana = std::mem::take(&mut self.held) + kana;
            if kana.ends_with('っ') || kana.ends_with('ッ') {
                kana.pop();
                self.held.push('っ');
            }
            let hira = if jp { hiragana(&kana) } else { String::new() };
            self.push(Word { text: surface, romaji: kana_to_romaji(&kana), kana: hira, jp }, attach);
            self.seen_jp |= jp;
            self.prev_pos = pos;
        }
        self.plain(&text[cursor..]);
    }
}

/// Spread a whole line's romaji — a source's, or the user's correction — over
/// its words, by lining its letters up against the generated reading's.
///
/// Edit distance, so a word read differently (聖 as *hijiri*, not *sei*) still
/// lands under the right word as long as the rest of the line agrees. `None`
/// when the two disagree too much to trust the placement; the caller then
/// shows the line's romaji whole instead.
#[cfg(test)]
pub fn align(words: &[Word], romaji: &str) -> Option<Vec<String>> {
    align_with(words, romaji, &|_| false)
}

/// [`align`], with the words `open` picks left open: they bring no letters of
/// their own, and whatever the given romaji has in their place is theirs at
/// no cost. Used for words the sheet annotates, whose generated reading is the
/// sheet's — so a reading that differs from it there must not count against
/// placing the rest of the line.
fn align_with(words: &[Word], romaji: &str, open: &dyn Fn(&Word) -> bool) -> Option<Vec<String>> {
    // Latin-script letters only: an unread kanji left in the generated
    // reading is not a letter of anything the source wrote.
    let letter = |c: &char| (c.is_alphanumeric() && (*c as u32) < 0x2000) || *c == '\'';
    let c: Vec<char> = romaji.chars().filter(letter).collect();
    let mut g: Vec<char> = Vec::new();
    let mut bounds = vec![0];
    let mut free_at = Vec::new();
    for w in words {
        if open(w) { free_at.push(g.len()); } else { g.extend(w.romaji.chars().filter(letter)); }
        bounds.push(g.len());
    }
    let same = |a: char, b: char| a.to_lowercase().eq(b.to_lowercase());
    let (n, m) = (g.len(), c.len());
    // Inserting a given letter costs nothing where an open word sits.
    let ins = |i: usize| usize::from(!free_at.contains(&i));
    let mut dp = vec![vec![0usize; m + 1]; n + 1];
    for (i, row) in dp.iter_mut().enumerate() { row[0] = i; }
    for j in 1..=m { dp[0][j] = dp[0][j - 1] + ins(0); }
    for i in 1..=n {
        for j in 1..=m {
            let sub = dp[i - 1][j - 1] + usize::from(!same(g[i - 1], c[j - 1]));
            dp[i][j] = sub.min(dp[i - 1][j] + 1).min(dp[i][j - 1] + ins(i));
        }
    }
    if dp[n][m] * 2 > n.max(m) { return None; }

    // Walk back along one cheapest path. Generated position i lines up with
    // given positions lo[i]..=hi[i]; more than one where the given romaji has
    // letters the generated one lacks.
    let (mut lo, mut hi) = (vec![usize::MAX; n + 1], vec![0; n + 1]);
    let (mut i, mut j) = (n, m);
    let mut mark = |i: usize, j: usize| { lo[i] = lo[i].min(j); hi[i] = hi[i].max(j); };
    mark(n, m);
    while i > 0 || j > 0 {
        if i > 0 && j > 0
            && dp[i][j] == dp[i - 1][j - 1] + usize::from(!same(g[i - 1], c[j - 1]))
        {
            i -= 1; j -= 1;
        } else if i > 0 && dp[i][j] == dp[i - 1][j] + 1 {
            i -= 1;
        } else {
            j -= 1;
        }
        mark(i, j);
    }
    // Extra letters at a boundary go to the word after it — unless the word
    // before has no letters of its own (a kanji left unread), which is
    // exactly the word they are the reading of. The last word runs to the end.
    let mut out = Vec::with_capacity(words.len());
    let mut from = 0;
    for (k, w) in words.iter().enumerate() {
        let end = bounds[k + 1];
        let to = if k + 1 == words.len() { m }
                 else if end == bounds[k] { hi[end] }
                 else { lo[end] };
        let to = to.max(from);
        let r: String = c[from..to].iter().collect();
        from = to;
        if !w.jp { out.push(w.romaji.clone()); continue; }
        if r.is_empty() { return None; }
        // An open word takes its letters for free, so without a limit it
        // would swallow a reading of some other line entirely. Its share may
        // be at most about twice the sheet's own reading.
        if open(w) && r.chars().count() > 2 * w.romaji.chars().filter(letter).count() + 2 {
            return None;
        }
        out.push(r);
    }
    Some(out)
}

fn tidy(s: &str) -> String {
    s.split_whitespace().collect::<Vec<_>>().join(" ")
}

/// Split a line at kanji carrying furigana in brackets — 剣(つるぎ), a habit
/// of lyric sheets — returning each piece as written, with the furigana for
/// those that have it. That is the reading the song actually uses, so it beats
/// anything the dictionary would pick.
fn chunks(s: &str) -> Vec<(String, Option<String>)> {
    let cs: Vec<char> = s.chars().collect();
    let mut out: Vec<(String, Option<String>)> = Vec::new();
    let mut plain = String::new();
    let mut i = 0;
    while i < cs.len() {
        if !is_kanji(cs[i]) { plain.push(cs[i]); i += 1; continue; }
        let start = i;
        while i < cs.len() && is_kanji(cs[i]) { i += 1; }
        if i < cs.len() && matches!(cs[i], '(' | '（') {
            let open = i + 1;
            let mut j = open;
            while j < cs.len() && is_kana(cs[j]) { j += 1; }
            if j > open && j < cs.len() && matches!(cs[j], ')' | '）') {
                if !plain.is_empty() { out.push((std::mem::take(&mut plain), None)); }
                out.push((cs[start..=j].iter().collect(), Some(cs[open..j].iter().collect())));
                i = j + 1;
                continue;
            }
        }
        plain.extend(&cs[start..i]);
    }
    if !plain.is_empty() { out.push((plain, None)); }
    out
}

/// Katakana folded to hiragana; everything else untouched.
pub fn hiragana(s: &str) -> String {
    s.chars().map(|c| match c {
        '\u{30A1}'..='\u{30F6}' => char::from_u32(c as u32 - 0x60).unwrap_or(c),
        _ => c,
    }).collect()
}

/// A word split for furigana: each piece as written, with the reading to put
/// over it where it has kanji. `reading` overrides the word's own, in
/// hiragana.
///
/// Kana in the word is matched against the reading so the furigana sits only
/// over the kanji — 渇(かわ)いた, 持(も)て余(あま)して — as it would in print.
/// When the two cannot be matched up, the whole word carries the whole reading
/// rather than a wrong split.
pub fn ruby(w: &Word, reading: Option<&str>) -> Vec<(String, Option<String>)> {
    if !w.jp || !w.text.chars().any(is_kanji) {
        return vec![(w.text.clone(), None)];
    }
    // Furigana the sheet wrote itself: 剣(つるぎ) shows as 剣 under つるぎ,
    // whatever other reading is offered. It is what the lyricist marked, and
    // matching a reading against the bracketed text would only fail and put
    // a second reading over the literal brackets. A reading the user has
    // confirmed over the sheet's is applied by `read_line`, which takes the
    // brackets out before it asks for the split.
    if w.text.contains(['(', '（']) && chunks(&w.text).iter().any(|c| c.1.is_some()) {
        return chunks(&w.text).into_iter().map(|(written, kana)| match kana {
            Some(k) => (written[..written.find(['(', '（']).unwrap_or(written.len())].to_string(), Some(k)),
            None => (written, None),
        }).collect();
    }
    let reading: Vec<char> = hiragana(reading.unwrap_or(&w.kana)).chars().collect();
    let whole = || vec![(w.text.clone(), Some(reading.iter().collect()))];
    // Runs of kanji and of everything else, in order.
    let mut runs: Vec<(bool, String)> = Vec::new();
    for c in w.text.chars() {
        match runs.last_mut() {
            Some((k, r)) if *k == is_kanji(c) => r.push(c),
            _ => runs.push((is_kanji(c), c.to_string())),
        }
    }
    let mut out = Vec::new();
    let mut at = 0;
    for (i, (kanji, run)) in runs.iter().enumerate() {
        let run_h: Vec<char> = hiragana(run).chars().collect();
        if !kanji {
            if reading.get(at..at + run_h.len()) != Some(&run_h[..]) { return whole(); }
            at += run_h.len();
            out.push((run.clone(), None));
            continue;
        }
        // The kanji read up to where the next kana run turns up.
        let end = match runs.get(i + 1) {
            None => reading.len(),
            Some((_, next)) => {
                let next: Vec<char> = hiragana(next).chars().collect();
                match (at + 1..reading.len()).find(|&j| reading[j..].starts_with(&next)) {
                    Some(j) => j,
                    None => return whole(),
                }
            }
        };
        if end <= at { return whole(); }
        out.push((run.clone(), Some(reading[at..end].iter().collect())));
        at = end;
    }
    if at != reading.len() { return whole(); }
    out
}

/// Romaji back to hiragana, for turning a corrected reading into furigana.
/// Hepburn as written by this module or by people; anything it does not
/// recognise is kept as it is.
pub fn romaji_to_kana(s: &str) -> String {
    static TABLE: OnceLock<std::collections::HashMap<String, String>> = OnceLock::new();
    let table = TABLE.get_or_init(|| {
        let mut t = std::collections::HashMap::new();
        let small = |c: char| matches!(c, 'ぁ' | 'ぃ' | 'ぅ' | 'ぇ' | 'ぉ' | 'ゃ' | 'ゅ' | 'ょ' | 'ゎ' | 'っ');
        for c in ('\u{3041}'..='\u{3094}').filter(|c| !small(*c)) {
            let s: String = c.to_string();
            t.entry(kana_to_romaji(&s)).or_insert(s);
        }
        for base in "きぎしじちにひびぴみり".chars() {
            for y in "ゃゅょ".chars() {
                let s: String = [base, y].iter().collect();
                t.entry(kana_to_romaji(&s)).or_insert(s);
            }
        }
        t
    });
    let cs: Vec<char> = s.to_lowercase().chars().collect();
    let mut out = String::new();
    let mut i = 0;
    while i < cs.len() {
        // A doubled consonant is っ: kitte, matcha.
        if i + 1 < cs.len() && cs[i].is_ascii_alphabetic() && !"aeioun".contains(cs[i])
            && (cs[i + 1] == cs[i] || (cs[i] == 't' && cs[i + 1] == 'c'))
        {
            out.push('っ');
            i += 1;
            continue;
        }
        let hit = (1..=3).rev().find_map(|n| {
            let key: String = cs.get(i..i + n)?.iter().collect();
            table.get(&key).map(|k| (n, k))
        });
        match hit {
            Some((n, k)) => { out.push_str(k); i += n; }
            None => { if cs[i] != '\'' && !cs[i].is_whitespace() { out.push(cs[i]); } i += 1; }
        }
    }
    out
}

/// Hepburn for one kana, `None` for anything else.
fn kana(c: char) -> Option<&'static str> {
    Some(match c {
        'あ' => "a", 'い' => "i", 'う' => "u", 'え' => "e", 'お' => "o",
        'か' => "ka", 'き' => "ki", 'く' => "ku", 'け' => "ke", 'こ' => "ko",
        'が' => "ga", 'ぎ' => "gi", 'ぐ' => "gu", 'げ' => "ge", 'ご' => "go",
        'さ' => "sa", 'し' => "shi", 'す' => "su", 'せ' => "se", 'そ' => "so",
        'ざ' => "za", 'じ' => "ji", 'ず' => "zu", 'ぜ' => "ze", 'ぞ' => "zo",
        'た' => "ta", 'ち' => "chi", 'つ' => "tsu", 'て' => "te", 'と' => "to",
        'だ' => "da", 'ぢ' => "ji", 'づ' => "zu", 'で' => "de", 'ど' => "do",
        'な' => "na", 'に' => "ni", 'ぬ' => "nu", 'ね' => "ne", 'の' => "no",
        'は' => "ha", 'ひ' => "hi", 'ふ' => "fu", 'へ' => "he", 'ほ' => "ho",
        'ば' => "ba", 'び' => "bi", 'ぶ' => "bu", 'べ' => "be", 'ぼ' => "bo",
        'ぱ' => "pa", 'ぴ' => "pi", 'ぷ' => "pu", 'ぺ' => "pe", 'ぽ' => "po",
        'ま' => "ma", 'み' => "mi", 'む' => "mu", 'め' => "me", 'も' => "mo",
        'や' => "ya", 'ゆ' => "yu", 'よ' => "yo",
        'ら' => "ra", 'り' => "ri", 'る' => "ru", 'れ' => "re", 'ろ' => "ro",
        'わ' => "wa", 'ゐ' => "i", 'ゑ' => "e", 'を' => "o", 'ん' => "n",
        'ゔ' => "vu",
        'ぁ' => "a", 'ぃ' => "i", 'ぅ' => "u", 'ぇ' => "e", 'ぉ' => "o",
        'ゃ' => "ya", 'ゅ' => "yu", 'ょ' => "yo", 'ゎ' => "wa",
        '、' => ", ", '。' => ". ", '「' | '『' | '｢' => "\"", '」' | '』' | '｣' => "\"",
        '　' => " ", '・' => " ", '〜' => "~",
        _ => return None,
    })
}

/// Kana to Hepburn romaji. Katakana is folded to hiragana first; anything that
/// is not kana passes through untouched.
pub fn kana_to_romaji(s: &str) -> String {
    let cs: Vec<char> = s.chars().map(|c| match c {
        '\u{30A1}'..='\u{30F6}' => char::from_u32(c as u32 - 0x60).unwrap_or(c),
        _ => c,
    }).collect();
    let mut out = String::new();
    let mut geminate = false;
    let mut i = 0;
    while i < cs.len() {
        let c = cs[i];
        i += 1;
        if c == 'っ' { geminate = true; continue; }
        if c == 'ー' {
            if let Some(v) = out.chars().last().filter(|v| "aeiou".contains(*v)) {
                out.push(v);
            }
            continue;
        }
        let Some(base) = kana(c) else {
            geminate = false;
            // Full-width ASCII (！？（）：) to the ordinary forms.
            out.push(match c {
                '\u{FF01}'..='\u{FF5E}' => char::from_u32(c as u32 - 0xFEE0).unwrap_or(c),
                _ => c,
            });
            continue;
        };
        let mut syl = base.to_string();
        // Small ya/yu/yo: きゃ → kya, しゃ → sha.
        if let Some(&n) = cs.get(i)
            && let Some(y) = match n { 'ゃ' => Some('a'), 'ゅ' => Some('u'), 'ょ' => Some('o'), _ => None }
            && base.ends_with('i') && base.len() > 1
        {
            let stem = &base[..base.len() - 1];
            syl = if matches!(base, "shi" | "chi" | "ji") {
                format!("{stem}{y}")
            } else {
                format!("{stem}y{y}")
            };
            i += 1;
        // Small vowels in loanwords: ファ → fa, ティ → ti, ウィ → wi.
        } else if let Some(&n) = cs.get(i)
            && let Some(v) = match n { 'ぁ' => Some("a"), 'ぃ' => Some("i"), 'ぅ' => Some("u"),
                                       'ぇ' => Some("e"), 'ぉ' => Some("o"), _ => None }
        {
            // Same vowel is a drawn-out syllable, not a loanword sound:
            // はぁ → "haa".
            syl = if base.ends_with(v) {
                format!("{base}{v}")
            } else {
                let stem = base.trim_end_matches(['a', 'i', 'u', 'e', 'o']);
                let stem = match (stem, c) { ("", 'う') => "w", ("", 'い') => "y", (s, _) => s };
                format!("{stem}{v}")
            };
            i += 1;
        }
        if geminate {
            geminate = false;
            if syl.starts_with("ch") {
                out.push('t');
            } else if let Some(f) = syl.chars().next().filter(|f| !"aeioun".contains(*f)) {
                out.push(f);
            }
        }
        out.push_str(&syl);
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn kana_follows_hepburn() {
        assert_eq!(kana_to_romaji("さくら"), "sakura");
        assert_eq!(kana_to_romaji("しゃしん"), "shashin");
        assert_eq!(kana_to_romaji("きょう"), "kyou");
        assert_eq!(kana_to_romaji("ちょっと"), "chotto");
        assert_eq!(kana_to_romaji("まっちゃ"), "matcha");
        assert_eq!(kana_to_romaji("がっこう"), "gakkou");
        assert_eq!(kana_to_romaji("じゃあね"), "jaane");
    }

    #[test]
    fn katakana_loanwords() {
        assert_eq!(kana_to_romaji("パーティー"), "paatii");
        assert_eq!(kana_to_romaji("ファン"), "fan");
        assert_eq!(kana_to_romaji("ウィンドウ"), "windou");
        assert_eq!(kana_to_romaji("ヴァイオリン"), "vaiorin");
        assert_eq!(kana_to_romaji("ミク"), "miku");
    }

    #[test]
    fn non_kana_passes_through() {
        assert_eq!(kana_to_romaji("abc 123"), "abc 123");
        assert_eq!(line("Won't you show me all your secrets now?"),
                   "Won't you show me all your secrets now?");
    }

    /// The reason for the analyser: kanji need a reading, and particles need
    /// their spoken form.
    #[test]
    fn a_sentence_gets_readings_and_particles() {
        assert_eq!(line("私は学生です"), "watashi wa gakusei desu");
        assert_eq!(line("東京へ行く"), "toukyou e iku");
        assert_eq!(line("本を読んだ"), "hon o yonda");
    }

    #[test]
    fn endings_stay_on_their_word() {
        let r = line("渇いた心");
        assert!(r.starts_with("kawaita"), "{r}");
    }

    /// Found comparing against a human-written romaji sheet: each of these
    /// was wrong on a real song.
    #[test]
    fn lessons_from_real_lyrics() {
        // っ at the end of one word belongs to the next word's consonant.
        assert!(line("気取った").contains("kidotta"), "{}", line("気取った"));
        // Furigana in the sheet is the sung reading.
        assert_eq!(line("剣(つるぎ)").replace(' ', ""), "tsurugi");
        assert!(line("誘（いざな）う").starts_with("izanau"), "{}", line("誘（いざな）う"));
        // A line opening with は is an interjection, not a particle.
        assert!(line("はぁ〜").starts_with("haa"), "{}", line("はぁ〜"));
        assert!(line("口がBigger").ends_with("ga Bigger"), "{}", line("口がBigger"));
        // Full-width punctuation comes out as the ordinary kind.
        assert!(line("見て（感じて）").contains('('), "{}", line("見て（感じて）"));
        assert_eq!(line("いや、これ"), "iya, kore");
        // A bracket that is not furigana is left alone.
        assert_eq!(chunks("恋(感じて"), vec![("恋(感じて".to_string(), None)]);
    }

    fn texts(ws: &[Word]) -> Vec<(&str, &str)> {
        ws.iter().map(|w| (w.text.as_str(), w.romaji.as_str())).collect()
    }

    #[test]
    fn a_line_splits_into_words_with_their_readings() {
        assert_eq!(texts(&words("渇いた心")), vec![("渇いた", "kawaita"), ("心", "kokoro")]);
        // Nothing is lost or reordered: the words put back together are the line.
        let l = "“汝、今すぐ Bigger な夢を";
        assert_eq!(words(l).iter().map(|w| w.text.as_str()).collect::<String>(), l);
    }

    /// The word keeps its kanji on screen, with the furigana's reading.
    #[test]
    fn furigana_stays_on_its_word() {
        let w = words("剣(つるぎ)を");
        assert_eq!(w[0].text, "剣(つるぎ)");
        assert_eq!(w[0].romaji.replace(' ', ""), "tsurugi");
        assert_eq!(w.iter().map(|w| w.text.as_str()).collect::<String>(), "剣(つるぎ)を");
    }

    #[test]
    fn a_sources_romaji_lands_under_the_right_words() {
        let w = words("渇いた心");
        assert_eq!(align(&w, "ka wa i ta ko ko ro").unwrap(), vec!["kawaita", "kokoro"]);
        // A reading the dictionary did not guess still finds its word.
        let w = words("聖なる魂");
        let a = align(&w, "hijirinaru tamashii").unwrap();
        assert_eq!(a.last().unwrap(), "tamashii", "{a:?}");
        assert!(a[0].starts_with("hijiri"), "{a:?}");
    }

    /// A kanji the dictionary could not read stays in the generated text; it
    /// must not count as letters to line up against.
    #[test]
    fn an_unread_kanji_is_not_a_letter() {
        let w = vec![
            Word { text: "悔".into(), romaji: "悔".into(), jp: true, ..Default::default() },
            Word { text: "い".into(), romaji: "i".into(), jp: true, ..Default::default() },
        ];
        assert_eq!(align(&w, "ku i"), None, "nothing to place 悔's reading by");
        let w = vec![
            Word { text: "心".into(), romaji: "kokoro".into(), jp: true, ..Default::default() },
            Word { text: "悔".into(), romaji: "悔".into(), jp: true, ..Default::default() },
            Word { text: "愛".into(), romaji: "ai".into(), jp: true, ..Default::default() },
        ];
        assert_eq!(align(&w, "kokoro ku ai").unwrap(), vec!["kokoro", "ku", "ai"]);
    }

    #[test]
    fn romaji_for_some_other_line_is_not_forced_onto_the_words() {
        assert_eq!(align(&words("渇いた心"), "zenzen chigau uta desu yo"), None);
    }

    fn s(x: &str) -> String { x.to_string() }

    #[test]
    fn furigana_sits_over_the_kanji_only() {
        let w = &words("渇いた")[0];
        assert_eq!(ruby(w, None), vec![(s("渇"), Some(s("かわ"))), (s("いた"), None)]);
        let w = &words("持て余して")[0];
        assert_eq!(ruby(w, None), vec![
            (s("持"), Some(s("も"))), (s("て"), None), (s("余"), Some(s("あま"))), (s("して"), None),
        ]);
        let w = &words("心")[0];
        assert_eq!(ruby(w, None), vec![(s("心"), Some(s("こころ")))]);
        // Kana-only words get none.
        assert_eq!(ruby(&words("さよなら")[0], None), vec![(s("さよなら"), None)]);
    }

    #[test]
    fn furigana_from_the_sheet_drops_its_brackets() {
        let w = &words("剣(つるぎ)")[0];
        assert_eq!(ruby(w, None), vec![(s("剣"), Some(s("つるぎ")))]);
    }

    /// With source or corrected romaji present the viewer passes a reading;
    /// the sheet's own furigana still wins, and the brackets never show.
    #[test]
    fn furigana_from_the_sheet_wins_over_a_supplied_reading() {
        let w = &words("剣(つるぎ)")[0];
        assert_eq!(ruby(w, Some("けん")), vec![(s("剣"), Some(s("つるぎ")))]);
        let w = &words("誘(いざな)う")[0];
        assert_eq!(ruby(w, Some("さそう")), vec![(s("誘"), Some(s("いざな"))), (s("う"), None)]);

        // The viewer's own path: a line with romaji, spread over its words,
        // turned back into kana, then split for furigana.
        let ws = words("剣(つるぎ)を");
        let under = align(&ws, "tsu ru gi wo").unwrap();
        let shown: Vec<_> = ws.iter().zip(&under)
            .flat_map(|(w, u)| ruby(w, Some(&romaji_to_kana(u))))
            .collect();
        assert_eq!(shown, vec![(s("剣"), Some(s("つるぎ"))), (s("を"), None)]);
        assert!(shown.iter().all(|(t, _)| !t.contains(['(', '（'])), "{shown:?}");
    }

    /// A reading that does not fit the word's kana is not split wrongly.
    #[test]
    fn a_reading_that_does_not_fit_goes_over_the_whole_word() {
        let w = &words("渇いた")[0];
        assert_eq!(ruby(w, Some("かついた")), vec![(s("渇"), Some(s("かつ"))), (s("いた"), None)]);
        assert_eq!(ruby(w, Some("なにか")), vec![(s("渇いた"), Some(s("なにか")))]);
    }

    #[test]
    fn romaji_turns_back_into_kana() {
        assert_eq!(romaji_to_kana("kawaita"), "かわいた");
        assert_eq!(romaji_to_kana("hijiri"), "ひじり");
        assert_eq!(romaji_to_kana("kitte"), "きって");
        assert_eq!(romaji_to_kana("matcha"), "まっちゃ");
        assert_eq!(romaji_to_kana("shashin"), "しゃしん");
        assert_eq!(romaji_to_kana("tsurugi"), "つるぎ");
        assert_eq!(romaji_to_kana("ka wa i ta"), "かわいた");
        // Every generated reading of a content word survives the round trip.
        for w in words("渇いた心 持て余して 展開 聖なる魂") {
            if w.jp { assert_eq!(romaji_to_kana(&w.romaji), w.kana, "{}", w.text); }
        }
    }

    fn furigana_of(r: &Reading) -> Vec<(String, Option<String>)> {
        match &r.furigana {
            Furigana::Words(w) => w.iter().flatten().cloned().collect(),
            Furigana::WholeLine { text, kana } => vec![(text.clone(), Some(kana.clone()))],
        }
    }

    /// By default the sheet's furigana decides, in every view.
    #[test]
    fn the_sheets_furigana_reads_by_default() {
        let w = words("剣(つるぎ)を");
        let r = read_line(&w, None, false);
        assert_eq!(r.line, "tsurugi o");
        assert_eq!(furigana_of(&r)[0], (s("剣"), Some(s("つるぎ"))));
    }

    /// Romaji from a lyrics source that reads the annotated word differently
    /// does not displace the sheet's reading, in romaji or furigana; the rest
    /// of the line still takes the source's reading.
    #[test]
    fn a_source_reading_does_not_displace_the_sheets_furigana() {
        let w = words("剣(つるぎ)を");
        // Letters at a word boundary cannot be told apart phonetically, so the
        // source reads "o" here: the question is only which word it lands on.
        let r = read_line(&w, Some("ken o"), false);
        assert_eq!(r.words.as_ref().unwrap()[0], "tsurugi");
        assert_eq!(r.line, "tsurugi o");
        assert_eq!(furigana_of(&r)[0], (s("剣"), Some(s("つるぎ"))));
        assert!(!r.guessed);
    }

    /// A confirmed override wins everywhere, and the brackets never show.
    #[test]
    fn a_confirmed_override_wins_in_every_view() {
        let w = words("剣(つるぎ)を");
        let r = read_line(&w, Some("ken o"), true);
        assert_eq!(r.line, "ken o");
        assert_eq!(furigana_of(&r)[0], (s("剣"), Some(s("けん"))));
        assert!(furigana_of(&r).iter().all(|(t, _)| !t.contains('(')));
    }

    /// An override that cannot be placed word by word goes over the whole
    /// line, rather than falling back to the sheet's per-word reading.
    #[test]
    fn an_override_that_cannot_be_placed_covers_the_whole_line() {
        let w = words("剣(つるぎ)を");
        let r = read_line(&w, Some("zenzen chigau uta desu yo"), true);
        assert_eq!(r.line, "zenzen chigau uta desu yo");
        assert_eq!(r.furigana, Furigana::WholeLine {
            text: s("剣を"), kana: romaji_to_kana("zenzen chigau uta desu yo"),
        });
        // Without the override the sheet decides, marked as generated.
        let r = read_line(&w, Some("zenzen chigau uta desu yo"), false);
        assert_eq!(r.line, "tsurugi o");
        assert!(r.guessed);
    }

    #[test]
    fn conflicts_are_found_only_where_the_sheet_annotates() {
        let w = words("剣(つるぎ)を振る");
        let c = conflicts(&w, "ken o furu");
        assert_eq!(c, vec![Conflict { word: s("剣"), sheet: s("つるぎ"), proposed: s("けん") }]);
        assert!(conflicts(&w, "tsurugi o furu").is_empty(), "agreeing is not a conflict");
        assert!(conflicts(&words("心を振る"), "shin o furu").is_empty(), "no annotation, nothing to conflict with");
    }

    #[test]
    fn a_chinese_sheet_is_not_japanese() {
        assert!(!has_kana("编曲 作词"));
        assert!(has_kana("渇いた心"));
        assert!(!has_kana("ーー"));
    }
}
