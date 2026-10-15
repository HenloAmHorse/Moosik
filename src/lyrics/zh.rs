//! Showing a lyric translation in Traditional or Simplified Chinese.
//!
//! A display choice, nothing more: the translation as stored, saved and
//! edited is never converted. The conversion is OpenCC's, phrase by phrase,
//! offline and deterministic (`ferrous-opencc`, dictionaries compiled in):
//!
//! - **Traditional (Taiwan)** is OpenCC `s2tw`: Simplified to Traditional,
//!   then Taiwan's standard character forms. Character forms only — Taiwan
//!   *vocabulary* (`s2twp`, e.g. 软件 → 軟體) is a different policy, not
//!   applied. Traditional text goes through it unchanged apart from those
//!   character forms.
//! - **Simplified** is OpenCC `tw2s`: Taiwan forms back to standard
//!   Traditional, then to Simplified. Simplified text goes through unchanged.
//!
//! Nothing here decides what language a translation is in. A translation
//! carries no language tag, and a line of Han characters alone is not proof
//! of Chinese, so conversion is the user's explicit choice and applies to
//! every translation line, with one exception: a line that contains a kana
//! syllable or a hangul letter is certainly not Chinese, and is shown as it is
//! rather than have its kanji or hanja rewritten. Letters only, not whole
//! Unicode blocks: the katakana middle dot ・ (U+30FB), which Chinese text
//! uses between the parts of a name, and the blocks' other punctuation, sound
//! marks and long-vowel bar ー say nothing about the language.

use std::sync::OnceLock;

/// How a translation is shown.
#[derive(Clone, Copy, PartialEq, Eq, Debug, Default, serde::Serialize, serde::Deserialize)]
pub enum ChineseDisplay {
    /// As stored. The default, and what every sheet showed before.
    #[default]
    Original,
    /// Traditional Chinese, Taiwan character forms.
    Traditional,
    /// Simplified Chinese.
    Simplified,
}

fn s2tw() -> &'static ferrous_opencc::OpenCC {
    static C: OnceLock<ferrous_opencc::OpenCC> = OnceLock::new();
    C.get_or_init(|| ferrous_opencc::OpenCC::from_config(ferrous_opencc::config::BuiltinConfig::S2tw)
        .expect("the built-in s2tw configuration"))
}

fn tw2s() -> &'static ferrous_opencc::OpenCC {
    static C: OnceLock<ferrous_opencc::OpenCC> = OnceLock::new();
    C.get_or_init(|| ferrous_opencc::OpenCC::from_config(ferrous_opencc::config::BuiltinConfig::Tw2s)
        .expect("the built-in tw2s configuration"))
}

/// A kana syllable or a hangul letter: a line that has one is not Chinese.
fn certainly_not_chinese(s: &str) -> bool {
    s.chars().any(|c| matches!(c,
        // Hiragana ぁ–ゖ, katakana ァ–ヺ, the digraph letters ゟ and ヿ, the
        // small-katakana extension, and halfwidth ｦ–ｯ, ｱ–ﾝ.
        '\u{3041}'..='\u{3096}' | '\u{30A1}'..='\u{30FA}' | '\u{309F}' | '\u{30FF}' | '\u{31F0}'..='\u{31FF}'
        | '\u{FF66}'..='\u{FF6F}' | '\u{FF71}'..='\u{FF9D}'
        // Hangul syllables, jamo and their extensions, compatibility and
        // halfwidth jamo.
        | '\u{AC00}'..='\u{D7A3}' | '\u{1100}'..='\u{11FF}' | '\u{A960}'..='\u{A97C}'
        | '\u{D7B0}'..='\u{D7FB}' | '\u{3131}'..='\u{318E}' | '\u{FFA0}'..='\u{FFDC}'))
}

/// `text` as `mode` shows it. `None` when it is shown as it is.
pub fn display(text: &str, mode: ChineseDisplay) -> Option<String> {
    if mode == ChineseDisplay::Original || certainly_not_chinese(text) {
        return None;
    }
    let out = match mode {
        ChineseDisplay::Traditional => s2tw().convert(text),
        ChineseDisplay::Simplified => tw2s().convert(text),
        ChineseDisplay::Original => unreachable!(),
    };
    (out != text).then_some(out)
}

/// A converted translation per line, worked out once per line text and mode,
/// for the lines that have been shown. Only the current mode is kept.
#[derive(Default)]
pub struct Cache {
    mode: Option<ChineseDisplay>,
    /// Per line: the translation it was worked out for, and the result.
    lines: Vec<Option<(String, Option<String>)>>,
    /// Conversions run, for tests.
    #[cfg(test)]
    pub conversions: u64,
}

impl Cache {
    /// What line `i`'s `translation` shows in `mode`, worked out when the
    /// line, its translation or the mode has changed since the last time.
    pub fn get<'a>(&'a mut self, i: usize, translation: &'a str, mode: ChineseDisplay) -> &'a str {
        if self.mode != Some(mode) {
            self.mode = Some(mode);
            self.lines.clear();
        }
        if self.lines.len() <= i { self.lines.resize_with(i + 1, || None); }
        let slot = &mut self.lines[i];
        if slot.as_ref().is_none_or(|(t, _)| t != translation) {
            *slot = Some((translation.to_string(), display(translation, mode)));
            #[cfg(test)]
            { self.conversions += 1; }
        }
        match slot {
            Some((_, Some(converted))) => converted,
            _ => translation,
        }
    }

    /// Forget everything: a new sheet.
    pub fn clear(&mut self) {
        self.mode = None;
        self.lines.clear();
    }

    #[cfg(test)]
    pub fn worked_out(&self) -> usize { self.lines.iter().flatten().count() }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ChineseDisplay::*;

    fn show(s: &str, m: ChineseDisplay) -> String { display(s, m).unwrap_or_else(|| s.to_string()) }

    #[test]
    fn original_is_never_converted() {
        for s in ["我们说过这些话", "我們說過這些話", "Love you", ""] {
            assert_eq!(display(s, Original), None);
        }
    }

    #[test]
    fn simplified_becomes_traditional_with_taiwan_character_forms() {
        assert_eq!(show("我们说过这些话，长长的门边开着车。", Traditional), "我們說過這些話，長長的門邊開著車。");
        // Phrase-aware: 头发 → 頭髮, 发现 → 發現, 后来 → 後來, 皇后 stays 皇后.
        assert_eq!(show("后来我发现她的头发", Traditional), "後來我發現她的頭髮");
        assert_eq!(show("皇后", Traditional), "皇后");
        // Taiwan character forms, from standard Traditional too: 裏 → 裡.
        assert_eq!(show("心裏", Traditional), "心裡");
    }

    #[test]
    fn taiwan_vocabulary_is_not_substituted() {
        // s2twp would give 軟體 / 網路; character forms only keep the words.
        assert_eq!(show("软件", Traditional), "軟件");
        assert_eq!(show("网络", Traditional), "網絡");
    }

    #[test]
    fn traditional_becomes_simplified() {
        assert_eq!(show("我們說過這些話，長長的門邊開著車。", Simplified), "我们说过这些话，长长的门边开着车。");
        assert_eq!(show("心裡", Simplified), "心里");
    }

    #[test]
    fn text_already_in_the_target_script_is_unchanged() {
        assert_eq!(display("我們說過這些話。", Traditional), None);
        assert_eq!(display("我们说过这些话。", Simplified), None);
    }

    #[test]
    fn latin_punctuation_digits_and_emoji_pass_through() {
        assert_eq!(show("Love 你们 100 次！OK，再见 2024。🎵 (live)", Traditional),
                   "Love 你們 100 次！OK，再見 2024。🎵 (live)");
        assert_eq!(show("「歌詞」『夢』…—", Simplified), "「歌词」『梦』…—");
    }

    #[test]
    fn a_line_with_kana_or_hangul_is_left_alone() {
        for s in ["夜明けの空に君の声が響く", "サクラ咲く", "사랑해 愛"] {
            assert_eq!(display(s, Traditional), None, "{s}");
            assert_eq!(display(s, Simplified), None, "{s}");
        }
    }

    /// The katakana middle dot is punctuation that Chinese text uses too (in
    /// transliterated names): a Chinese line with it converts. So do lines
    /// with the other non-letters of the kana blocks.
    #[test]
    fn punctuation_from_the_kana_blocks_is_not_kana() {
        assert_eq!(show("爱・梦", Traditional), "愛・夢");
        assert_eq!(show("愛・夢", Simplified), "爱・梦");
        assert_eq!(show("约翰・列侬说过这些话", Traditional), "約翰・列儂說過這些話");
        for s in ["爱ー梦", "爱゠梦", "爱｡梦", "爱･梦", "爱゛梦"] {
            assert!(display(s, Traditional).is_some(), "{s} was left alone");
        }
        // With the rest intact around it.
        assert_eq!(show("Love・你们 100 次！🎵", Traditional), "Love・你們 100 次！🎵");
    }

    /// Every kana syllable and hangul letter still marks a line as not
    /// Chinese, alone or beside the dot.
    #[test]
    fn every_kana_syllable_and_hangul_letter_still_counts() {
        for c in ['ぁ', 'あ', 'ゖ', 'ァ', 'ア', 'ヺ', 'ㇰ', 'ｦ', 'ｱ', 'ﾝ', '가', '힣', 'ᄀ', 'ㄱ', 'ㆎ', 'ﾡ'] {
            let s = format!("爱{c}梦");
            assert_eq!(display(&s, Traditional), None, "{s}");
            assert_eq!(display(&s, Simplified), None, "{s}");
        }
        // The two kana digraph letters: ゟ (U+309F, より) and ヿ (U+30FF, コト).
        for s in ["爱ゟ梦", "愛ゟ夢", "爱ヿ梦", "愛ヿ夢"] {
            assert_eq!(display(s, Traditional), None, "{s}");
            assert_eq!(display(s, Simplified), None, "{s}");
        }
        for s in ["ナ・ルー", "愛・ナ", "사랑・愛"] {
            assert_eq!(display(s, Traditional), None, "{s}");
        }
    }

    #[test]
    fn the_cache_follows_the_text_and_the_mode() {
        let mut c = Cache::default();
        assert_eq!(c.get(0, "爱与梦", Traditional), "愛與夢");
        assert_eq!(c.get(0, "爱与梦", Traditional), "愛與夢");
        assert_eq!(c.worked_out(), 1);
        // The translation edited.
        assert_eq!(c.get(0, "梦与光", Traditional), "夢與光");
        // The mode changed.
        assert_eq!(c.get(0, "梦与光", Simplified), "梦与光");
        assert_eq!(c.get(0, "夢與光", Simplified), "梦与光");
        assert_eq!(c.get(0, "夢與光", Original), "夢與光");
        c.clear();
        assert_eq!(c.worked_out(), 0);
    }
}
