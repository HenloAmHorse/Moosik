//! Offline, pronunciation-oriented Revised Romanization for Korean lyrics.
//!
//! The vowel/consonant spellings and sound-change policy follow the National
//! Institute of Korean Language's rules:
//! <https://www.korean.go.kr/front_eng/roman/roman_01.do>.
//! This module implements regular liaison, final-consonant reduction, nasal and
//! liquid assimilation, palatalization, and aspiration adjacent to `ㅎ`. Derived
//! tense consonants are deliberately not written; explicitly spelled tense
//! consonants are. `ㅢ` remains `ui` as required by Revised Romanization.
//!
//! This is not a morphological analyzer or pronunciation dictionary. Sound
//! changes are applied within uninterrupted runs of modern Hangul syllables,
//! never across spaces, punctuation, Latin text, or isolated Jamo. In particular,
//! compound boundaries, inserted `ㄴ`, irregular stems, noun/verb distinctions
//! before `ㅎ`, and exceptional `ㄴㄹ` pronunciations cannot generally be inferred
//! from spelling. A small table covers five examples given explicitly by NIKL;
//! the `밟` exception to `ㄼ` reduction and six common `ㄺ` predicate stems are
//! also handled. Other stop + `ㅎ` sequences
//! use the regular aspiration rule, which can differ from the noun exception in
//! official Romanization. Proper-name capitalization, name-specific spelling,
//! optional separating hyphens, and administrative-unit formatting are left to
//! the caller. Hanja has no unique Korean reading and is preserved verbatim.
//!
//! Modern precomposed syllables and canonical `L + V (+ T)` / `LV + T` Jamo
//! sequences produce the same output. Standalone modern canonical/compatibility
//! Jamo are transliterated as letters (including `ng` for isolated `ㅇ`), since
//! they have no syllabic pronunciation context. Compatibility Jamo sequences are
//! not composed: their initial/final roles are ambiguous. Archaic, extended, and
//! halfwidth Jamo and Hangul tone marks are detected but preserved unchanged.
//! All non-Hangul characters, including whitespace and mixed English, are kept
//! exactly as supplied. No network, tokenizer, runtime, or extra crate is used.

const ONSETS: [&str; 19] = [
    "g", "kk", "n", "d", "tt", "r", "m", "b", "pp", "s", "ss", "", "j", "jj", "ch", "k", "t", "p",
    "h",
];
const VOWELS: [&str; 21] = [
    "a", "ae", "ya", "yae", "eo", "e", "yeo", "ye", "o", "wa", "wae", "oe", "yo", "u", "wo", "we",
    "wi", "yu", "eu", "ui", "i",
];
const ISOLATED_CODAS: [&str; 27] = [
    "g", "kk", "gs", "n", "nj", "nh", "d", "l", "lg", "lm", "lb", "ls", "lt", "lp", "lh", "m", "b",
    "bs", "s", "ss", "ng", "j", "ch", "k", "t", "p", "h",
];
const COMPATIBILITY_CONSONANTS: [&str; 30] = [
    "g", "kk", "gs", "n", "nj", "nh", "d", "tt", "r", "lg", "lm", "lb", "ls", "lt", "lp", "lh",
    "m", "b", "pp", "bs", "s", "ss", "ng", "j", "jj", "ch", "k", "t", "p", "h",
];

// A deliberately finite list of whole-word pronunciations from NIKL's examples.
// The first two require compound-boundary information; the next two are the
// noun exception to stop + ㅎ aspiration; the last has exceptional ㄴㄹ -> ㄴㄴ.
// Exact whole-run matching avoids inventing morphological rules for substrings.
const WORD_EXCEPTIONS: [(&str, &str); 5] = [
    ("학여울", "hangnyeoul"),
    ("알약", "allyak"),
    ("묵호", "mukho"),
    ("집현전", "jiphyeonjeon"),
    ("신문로", "sinmunno"),
];

const G: u8 = 0;
const N: u8 = 2;
const D: u8 = 3;
const R: u8 = 5;
const M: u8 = 6;
const B: u8 = 7;
const S: u8 = 9;
const SILENT: u8 = 11;
const J: u8 = 12;
const CH: u8 = 14;
const K: u8 = 15;
const T: u8 = 16;
const P: u8 = 17;
const H: u8 = 18;
const I: u8 = 20;

#[derive(Clone, Copy, Debug)]
struct Syllable {
    onset: u8,
    vowel: u8,
    coda: u8,
}

impl Syllable {
    fn character(self) -> char {
        // All fields originate in the modern Hangul ranges parsed below.
        char::from_u32(
            0xac00
                + u32::from(self.onset) * 588
                + u32::from(self.vowel) * 28
                + u32::from(self.coda),
        )
        .expect("modern Hangul syllable")
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Coda {
    None,
    K,
    N,
    T,
    L,
    M,
    P,
    Ng,
}

impl Coda {
    fn spelling(self) -> &'static str {
        match self {
            Self::None => "",
            Self::K => "k",
            Self::N => "n",
            Self::T => "t",
            Self::L => "l",
            Self::M => "m",
            Self::P => "p",
            Self::Ng => "ng",
        }
    }
}

/// Whether text contains Hangul syllables, Jamo, or Hangul tone marks.
///
/// Detection includes older/extended and halfwidth Jamo even when `romanize`
/// preserves them, so callers can correctly identify Korean-containing text.
pub fn has_hangul(text: &str) -> bool {
    text.chars().any(|ch| {
        matches!(
            ch as u32,
            0x1100..=0x11ff
                | 0x302e..=0x302f
                | 0x3131..=0x318e
                | 0xa960..=0xa97c
                | 0xac00..=0xd7a3
                | 0xd7b0..=0xd7c6
                | 0xd7cb..=0xd7fb
                | 0xffa0..=0xffbe
                | 0xffc2..=0xffc7
                | 0xffca..=0xffcf
                | 0xffd2..=0xffd7
                | 0xffda..=0xffdc
        )
    })
}

/// Romanize modern Hangul while preserving every non-Hangul character.
///
/// See the module documentation for the pronunciation and normalization limits.
pub fn romanize(text: &str) -> String {
    let characters: Vec<char> = text.chars().collect();
    let mut output = String::with_capacity(text.len());
    let mut position = 0;
    while position < characters.len() {
        if let Some((syllable, consumed)) = parse_syllable(&characters, position) {
            let mut syllables = vec![syllable];
            position += consumed;
            while let Some((syllable, consumed)) = parse_syllable(&characters, position) {
                syllables.push(syllable);
                position += consumed;
            }
            romanize_run(&syllables, &mut output);
        } else {
            let character = characters[position];
            if let Some(spelling) = isolated_jamo(character) {
                output.push_str(spelling);
            } else {
                output.push(character);
            }
            position += 1;
        }
    }
    output
}

fn parse_syllable(characters: &[char], position: usize) -> Option<(Syllable, usize)> {
    let character = *characters.get(position)? as u32;
    if (0xac00..=0xd7a3).contains(&character) {
        let offset = character - 0xac00;
        let mut syllable = Syllable {
            onset: (offset / 588) as u8,
            vowel: ((offset % 588) / 28) as u8,
            coda: (offset % 28) as u8,
        };
        // Unicode canonical composition also permits a precomposed LV + T.
        if syllable.coda == 0
            && let Some(coda) = characters.get(position + 1).and_then(|ch| modern_coda(*ch))
        {
            syllable.coda = coda;
            return Some((syllable, 2));
        }
        return Some((syllable, 1));
    }
    if !(0x1100..=0x1112).contains(&character) {
        return None;
    }
    let vowel = *characters.get(position + 1)? as u32;
    if !(0x1161..=0x1175).contains(&vowel) {
        return None;
    }
    let coda = characters.get(position + 2).and_then(|ch| modern_coda(*ch));
    Some((
        Syllable {
            onset: (character - 0x1100) as u8,
            vowel: (vowel - 0x1161) as u8,
            coda: coda.unwrap_or(0),
        },
        if coda.is_some() { 3 } else { 2 },
    ))
}

fn modern_coda(character: char) -> Option<u8> {
    let codepoint = character as u32;
    (0x11a8..=0x11c2)
        .contains(&codepoint)
        .then(|| (codepoint - 0x11a7) as u8)
}

fn isolated_jamo(character: char) -> Option<&'static str> {
    match character as u32 {
        0x1100..=0x1112 => {
            let index = character as usize - 0x1100;
            Some(if index == SILENT as usize {
                "ng"
            } else {
                ONSETS[index]
            })
        }
        0x1161..=0x1175 => Some(VOWELS[character as usize - 0x1161]),
        0x11a8..=0x11c2 => Some(ISOLATED_CODAS[character as usize - 0x11a8]),
        0x3131..=0x314e => Some(COMPATIBILITY_CONSONANTS[character as usize - 0x3131]),
        0x314f..=0x3163 => Some(VOWELS[character as usize - 0x314f]),
        _ => None,
    }
}

fn romanize_run(syllables: &[Syllable], output: &mut String) {
    // Check the small explicit list after canonical composition as well.
    let word: String = syllables
        .iter()
        .map(|syllable| syllable.character())
        .collect();
    if let Some((_, spelling)) = WORD_EXCEPTIONS
        .iter()
        .find(|(exception, _)| *exception == word)
    {
        output.push_str(spelling);
        return;
    }

    let mut onsets: Vec<u8> = syllables.iter().map(|syllable| syllable.onset).collect();
    let mut codas: Vec<Coda> = syllables
        .iter()
        .enumerate()
        .map(|(index, syllable)| reduce_coda(*syllable, syllables.get(index + 1)))
        .collect();
    for index in 0..syllables.len().saturating_sub(1) {
        let current = syllables[index];
        let next = syllables[index + 1];
        if current.coda == 0 {
            continue;
        }
        if next.onset == SILENT {
            let (coda, onset) = liaison(current.coda, next.vowel);
            codas[index] = coda;
            onsets[index + 1] = onset;
            continue;
        }
        if let Some((coda, onset)) = aspirate(current.coda, next) {
            codas[index] = coda;
            onsets[index + 1] = onset;
            continue;
        }
        let (coda, onset) = assimilate(codas[index], next.onset);
        codas[index] = coda;
        onsets[index + 1] = onset;
    }
    for (index, syllable) in syllables.iter().enumerate() {
        // ㄹ is r before a vowel, but an adjacent pair of ㄹ is written ll.
        if onsets[index] == R && index > 0 && codas[index - 1] == Coda::L {
            output.push('l');
        } else {
            output.push_str(ONSETS[onsets[index] as usize]);
        }
        output.push_str(VOWELS[syllable.vowel as usize]);
        output.push_str(codas[index].spelling());
    }
}

fn reduce_coda(syllable: Syllable, next: Option<&Syllable>) -> Coda {
    match syllable.coda {
        0 => Coda::None,
        1..=3 | 24 => Coda::K,
        4..=6 => Coda::N,
        7 | 19 | 20 | 22 | 23 | 25 | 27 => Coda::T,
        8 | 12 | 13 | 15 => Coda::L,
        // Only predicate stems keep ㄹ before ㄱ; nouns retain ㄱ (닭과).
        // This finite stem list covers NIKL's examples without assuming that
        // every ㄺ syllable is a verb/adjective. Other stems use the default.
        // https://www.korean.go.kr/front/onlineQna/onlineQnaView.do?qna_seq=329140
        9 if next.is_some_and(|next| next.onset == G)
            && matches!(
                syllable.character(),
                '읽' | '맑' | '묽' | '늙' | '밝' | '얽'
            ) =>
        {
            Coda::L
        }
        9 => Coda::K,
        10 | 16 => Coda::M,
        // 밟 has exceptional ㄼ -> ㅂ; liaison before vowels is handled later.
        11 if syllable.character() == '밟' => Coda::P,
        11 => Coda::L,
        14 | 17 | 18 | 26 => Coda::P,
        21 => Coda::Ng,
        _ => unreachable!("modern Hangul coda"),
    }
}

fn liaison(coda: u8, following_vowel: u8) -> (Coda, u8) {
    let palatal = following_vowel == I;
    match coda {
        1 => (Coda::None, G),
        2 => (Coda::None, 1), // Explicit ㄲ is still kk.
        3 => (Coda::K, S),
        4 | 6 => (Coda::None, N), // ㄶ loses ㅎ, then ㄴ links to the vowel.
        5 => (Coda::N, J),
        7 => (Coda::None, if palatal { J } else { D }),
        8 | 15 => (Coda::None, R), // ㄹㅎ loses ㅎ before a vowel.
        9 => (Coda::L, G),
        10 => (Coda::L, M),
        11 => (Coda::L, B),
        12 => (Coda::L, S),
        13 => (Coda::L, if palatal { CH } else { T }),
        14 => (Coda::L, P),
        16 => (Coda::None, M),
        17 => (Coda::None, B),
        18 => (Coda::P, S),
        19 => (Coda::None, S),
        20 => (Coda::None, 10), // Explicit ㅆ is still ss.
        21 => (Coda::Ng, SILENT),
        22 => (Coda::None, J),
        23 => (Coda::None, CH),
        24 => (Coda::None, K),
        25 => (Coda::None, if palatal { CH } else { T }),
        26 => (Coda::None, P),
        27 => (Coda::None, SILENT),
        _ => unreachable!("nonempty modern Hangul coda"),
    }
}

fn aspirate(coda: u8, next: Syllable) -> Option<(Coda, u8)> {
    // A final ㅎ (also in ㄶ/ㅀ) aspirates a following plain stop/affricate.
    if matches!(coda, 6 | 15 | 27) {
        let remainder = match coda {
            6 => Coda::N,
            15 => Coda::L,
            _ => Coda::None,
        };
        let onset = match next.onset {
            G => K,
            D => T,
            B => P,
            J => CH,
            // ㅎ + ㅅ is tense in speech, but derived tension is not written.
            S => S,
            _ => return None,
        };
        return Some((remainder, onset));
    }
    if next.onset != H {
        return None;
    }
    // Preserve the surviving first member where a cluster's second consonant
    // combines with ㅎ. Without morphology this is the regular (verbal) rule.
    let (remainder, onset) = match coda {
        1 | 2 | 3 | 24 => (Coda::None, K),
        5 => (Coda::N, CH),
        7 | 19 | 20 | 25 => (Coda::None, T),
        9 => (Coda::L, K),
        11 | 14 => (Coda::L, P),
        12 | 13 => (Coda::L, T),
        17 | 18 | 26 => (Coda::None, P),
        22 | 23 => (Coda::None, CH),
        _ => return None,
    };
    Some((
        remainder,
        if onset == T && next.vowel == I {
            CH
        } else {
            onset
        },
    ))
}

fn assimilate(coda: Coda, onset: u8) -> (Coda, u8) {
    match (coda, onset) {
        (Coda::N | Coda::L, R) | (Coda::L, N) => (Coda::L, R),
        (Coda::M | Coda::Ng, R) => (coda, N),
        (Coda::K, R) => (Coda::Ng, N),
        (Coda::T, R) => (Coda::N, N),
        (Coda::P, R) => (Coda::M, N),
        (Coda::K, N | M) => (Coda::Ng, onset),
        (Coda::T, N | M) => (Coda::N, onset),
        (Coda::P, N | M) => (Coda::M, onset),
        _ => (coda, onset),
    }
}

#[cfg(test)]
mod tests {
    use super::{has_hangul, romanize};

    fn check_examples(examples: &[(&str, &str)]) {
        for (hangul, expected) in examples {
            assert_eq!(romanize(hangul), *expected, "{hangul}");
        }
    }

    #[test]
    fn official_basic_examples() {
        check_examples(&[
            ("구미", "gumi"),
            ("영동", "yeongdong"),
            ("백암", "baegam"),
            ("옥천", "okcheon"),
            ("합덕", "hapdeok"),
            ("호법", "hobeop"),
            ("월곶", "wolgot"),
            ("벚꽃", "beotkkot"),
            ("한밭", "hanbat"),
            ("구리", "guri"),
            ("설악", "seorak"),
            ("칠곡", "chilgok"),
            ("임실", "imsil"),
            ("울릉", "ulleung"),
            ("대관령", "daegwallyeong"),
            ("의의", "uiui"),
        ]);
    }

    #[test]
    fn official_assimilation_and_compound_examples() {
        check_examples(&[
            ("백마", "baengma"),
            ("신문로", "sinmunno"),
            ("종로", "jongno"),
            ("왕십리", "wangsimni"),
            ("별내", "byeollae"),
            ("신라", "silla"),
            ("학여울", "hangnyeoul"),
            ("알약", "allyak"),
            ("속리산", "songnisan"),
            ("극락전", "geungnakjeon"),
            ("독립문", "dongnimmun"),
            ("촉석루", "chokseongnu"),
        ]);
    }

    #[test]
    fn official_palatalization_and_aspiration_examples() {
        check_examples(&[
            ("해돋이", "haedoji"),
            ("같이", "gachi"),
            ("굳히다", "guchida"),
            ("좋고", "joko"),
            ("놓다", "nota"),
            ("잡혀", "japyeo"),
            ("낳지", "nachi"),
            ("묵호", "mukho"),
            ("집현전", "jiphyeonjeon"),
            ("좋아", "joa"),
            ("많아", "mana"),
            ("싫어", "sireo"),
            ("많고", "manko"),
            ("싫다", "silta"),
            ("맞히다", "machida"),
        ]);
    }

    #[test]
    fn derived_tense_consonants_are_not_transcribed() {
        check_examples(&[
            ("압구정", "apgujeong"),
            ("낙동강", "nakdonggang"),
            ("죽변", "jukbyeon"),
            ("낙성대", "nakseongdae"),
            ("합정", "hapjeong"),
            ("팔당", "paldang"),
            ("샛별", "saetbyeol"),
            ("울산", "ulsan"),
            ("먹다", "meokda"),
            ("꽃과", "kkotgwa"),
        ]);
    }

    #[test]
    fn batchim_clusters_reduce_or_link_in_context() {
        check_examples(&[
            ("값", "gap"),
            ("밖", "bak"),
            ("넋", "neok"),
            ("앉다", "anda"),
            ("앉아", "anja"),
            ("읽다", "ikda"),
            ("읽고", "ilgo"),
            ("읽어", "ilgeo"),
            ("맑게", "malge"),
            ("묽고", "mulgo"),
            ("밝고", "balgo"),
            ("늙고", "neulgo"),
            ("얽거나", "eolgeona"),
            ("닭고기", "dakgogi"),
            ("흙과", "heukgwa"),
            ("삶", "sam"),
            ("젊다", "jeomda"),
            ("젊어", "jeolmeo"),
            ("넓다", "neolda"),
            ("밟다", "bapda"),
            ("밟는", "bamneun"),
            ("밟아", "balba"),
            ("읊다", "eupda"),
            ("읊어", "eulpeo"),
            ("없어", "eopseo"),
            ("꽃이", "kkochi"),
            ("옷이", "osi"),
            ("훑어", "hulteo"),
            ("핥이", "halchi"),
            ("놓는", "nonneun"),
        ]);
    }

    #[test]
    fn preserves_mixed_lyrics_whitespace_and_punctuation() {
        check_examples(&[
            ("I love 너를 100% 💜", "I love neoreul 100% 💜"),
            ("I'm so 빛나 ✨", "I'm so binna ✨"),
            ("  사랑해!\r\n\t안녕?  ", "  saranghae!\r\n\tannyeong?  "),
            ("먹는 love (한글) 24/7", "meongneun love (hangeul) 24/7"),
            ("국 물 국,물 국A물", "guk mul guk,mul gukAmul"),
            ("café 愛 + 日本語 😀", "café 愛 + 日本語 😀"),
            ("", ""),
        ]);
    }

    #[test]
    fn canonical_jamo_normalization_matches_precomposed() {
        check_examples(&[
            ("사랑해", "saranghae"),
            ("같이", "gachi"),
            ("각이", "gagi"),
            ("학여울", "hangnyeoul"),
            ("김치", "gimchi"),
        ]);
        // Every one of the 11,172 modern syllables has equivalent canonical Jamo.
        for codepoint in 0xac00..=0xd7a3 {
            let character = char::from_u32(codepoint).unwrap();
            let offset = codepoint - 0xac00;
            let mut decomposed = String::new();
            decomposed.push(char::from_u32(0x1100 + offset / 588).unwrap());
            decomposed.push(char::from_u32(0x1161 + (offset % 588) / 28).unwrap());
            if offset % 28 != 0 {
                decomposed.push(char::from_u32(0x11a7 + offset % 28).unwrap());
            }
            assert_eq!(romanize(&character.to_string()), romanize(&decomposed));
        }
    }

    #[test]
    fn isolated_jamo_and_unsupported_hangul_are_explicit() {
        check_examples(&[
            ("ㄱ ㄲ ㄳ ㄴ ㄹ ㅇ ㅎ ㅏ ㅢ", "g kk gs n r ng h a ui"),
            ("ᄀ ᄋ ᅡ ᆨ ᆼ", "g ng a g ng"),
            ("ㅋㅋㅋ", "kkk"),
            ("ㅎㅏ", "ha"),
            ("ᄓ ᅠ ㅤ ㆁ ꥠ ힰ ﾡ 〮", "ᄓ ᅠ ㅤ ㆁ ꥠ ힰ ﾡ 〮"),
        ]);
    }

    #[test]
    fn detects_supported_and_preserved_hangul_ranges() {
        for text in ["한글", "ㄱ", "가", "ᆨ", "ᄓ", "ㆁ", "ꥠ", "ힰ", "ﾡ", "〮"] {
            assert!(has_hangul(text), "{text}");
        }
        for text in ["", "Love 100% 😀", "愛", "日本語", "café", "\n\t"] {
            assert!(!has_hangul(text), "{text}");
        }
    }
}
