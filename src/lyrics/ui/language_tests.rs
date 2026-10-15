//! Language support through the production reading/presentation entry points.
use super::*;

#[test]
fn korean_generated_readings_are_cached_and_do_not_use_japanese_words() {
    let mut cache = Vec::new();
    let a = auto_for(&mut cache, 0, "같이 Love 漢字");
    assert_eq!(a.line, "gachi Love 漢字");
    assert!(a.words.iter().all(|w| !w.jp));
    let r = a.reading(None, false).clone();
    assert!(r.guessed);
    assert_eq!(r.line, "gachi Love 漢字");
    let pointer = a.line.as_ptr();
    for _ in 0..200 {
        let a = auto_for(&mut cache, 0, "같이 Love 漢字");
        assert_eq!(
            a.line.as_ptr(),
            pointer,
            "unchanged lyrics must reuse the reading"
        );
        assert_eq!(a.reading(None, false), &r);
    }
}

#[test]
fn supplied_and_corrected_korean_readings_take_priority() {
    let mut cache = Vec::new();
    let a = auto_for(&mut cache, 0, "같이");
    assert_eq!(
        a.reading(Some("my sung reading"), false).line,
        "my sung reading"
    );
    assert!(!a.reading(Some("my sung reading"), false).guessed);
    assert_eq!(a.reading(None, false).line, "gachi");
    assert!(a.reading(None, false).guessed);
    assert_eq!(auto_for(&mut cache, 0, "안녕하세요").line, "annyeonghaseyo");
}

#[test]
fn korean_script_modes_show_whole_lines_and_never_invent_furigana() {
    let mut cache = Vec::new();
    let line = LyricLine {
        text: "같이 Love".into(),
        ..Default::default()
    };
    let a = auto_for(&mut cache, 0, &line.text);
    let shown = a.shown(Script::Romaji, true, &line, None);
    assert!(matches!(shown.as_ref(), Shown::Romaji(s) if s == "gachi Love"));
    let shown = a.shown(Script::Both, true, &line, None);
    assert!(
        matches!(shown.as_ref(), Shown::Text { text, under: Some(under), .. }
        if text == "같이 Love" && under == "gachi Love")
    );
    let shown = a.shown(Script::Furigana, true, &line, None);
    assert!(matches!(shown.as_ref(), Shown::Cells(c)
        if c.len() == 1 && c[0].text == line.text && c[0].above.is_none()));
}

#[test]
fn korean_controls_and_supplied_latin_controls_are_not_japanese_gated() {
    let mut w = LyricsWindow {
        script: Script::Both,
        lyrics: Some(super::super::parse("[00:01.00]같이", LyricSource::Online)),
        ..Default::default()
    };
    assert!(!w.is_japanese());
    assert!(w.has_readings());
    assert_eq!(w.display_script(), Script::Both);
    w.script = Script::Furigana;
    assert_eq!(w.display_script(), Script::Original);
    w.lyrics = Some(super::super::parse("Love you", LyricSource::Online));
    assert!(!w.has_readings());
    w.lyrics.as_mut().unwrap().lines[0].romaji = Some("supplied alternate reading".into());
    assert!(w.has_readings());
}

#[test]
fn supplied_latin_readings_are_not_overruled_by_japanese_alignment() {
    let mut cache = Vec::new();
    for supplied in ["Love ya", "LOVE YOU"] {
        let line = LyricLine {
            text: "Love you".into(),
            romaji: Some(supplied.into()),
            ..Default::default()
        };
        let a = auto_for(&mut cache, 0, &line.text);
        assert_eq!(a.reading(Some(supplied), false).line, supplied);
        assert!(!a.reading(Some(supplied), false).guessed);
        assert!(
            matches!(a.shown(Script::Romaji, true, &line, None).as_ref(),
            Shown::Romaji(s) if s == supplied)
        );
        assert!(matches!(a.shown(Script::Both, true, &line, None).as_ref(),
            Shown::Text { text, under: Some(s), .. } if text == "Love you" && s == supplied));
    }
}

#[test]
fn korean_editor_corrections_round_trip_without_sheet_annotation_override() {
    let mut w = LyricsWindow {
        lyrics: Some(super::super::parse("[00:01.00]같이", LyricSource::Online)),
        ..Default::default()
    };
    w.begin_romaji_edit();
    assert_eq!(w.romaji_edit[0].text, "gachi");
    w.romaji_edit[0].text = "gachi (sung)".into();
    w.commit_romaji_edit();
    assert_eq!(
        w.lyrics.as_ref().unwrap().lines[0].romaji.as_deref(),
        Some("gachi (sung)")
    );
    assert!(w.pending.is_empty());
    assert!(w.romaji_dirty);
    w.begin_romaji_edit();
    w.romaji_edit[0].text = "gachi".into();
    w.commit_romaji_edit();
    assert!(w.lyrics.as_ref().unwrap().lines[0].romaji.is_none());
}

#[test]
fn korean_and_english_chinese_translations_draw_in_both_character_sets() {
    for original in ["같이", "We keep the promise"] {
        let mut rig = viewer_tests::Rig::new(1);
        rig.w.lyrics = Some(super::super::parse(original, LyricSource::Online));
        rig.w.lyrics.as_mut().unwrap().lines[0].translation = Some("我们记得约定".into());
        for (mode, expected) in [
            (
                super::super::zh::ChineseDisplay::Traditional,
                "我們記得約定",
            ),
            (super::super::zh::ChineseDisplay::Simplified, "我们记得约定"),
        ] {
            rig.w.translation_display = mode;
            probe::take_translations();
            rig.pass(Duration::ZERO);
            assert!(
                probe::take_translations()
                    .iter()
                    .any(|(_, t)| t == expected)
            );
        }
    }
}
