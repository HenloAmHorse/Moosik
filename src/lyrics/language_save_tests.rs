//! Translation persistence against isolated track fixtures, with no network.

use super::*;
use std::collections::BTreeMap;
use std::sync::atomic::{AtomicU64, Ordering};

struct Scratch {
    root: PathBuf,
    dir: PathBuf,
}

impl Scratch {
    fn new() -> Self {
        static NEXT: AtomicU64 = AtomicU64::new(0);
        let root = std::fs::canonicalize(std::env::temp_dir()).unwrap();
        loop {
            let seq = NEXT.fetch_add(1, Ordering::Relaxed);
            let dir = root.join(format!(
                "moosik_language_save_{}_{}",
                std::process::id(),
                seq
            ));
            match std::fs::create_dir(&dir) {
                Ok(()) => return Self { root, dir },
                Err(e) if e.kind() == std::io::ErrorKind::AlreadyExists => continue,
                Err(e) => panic!("could not create isolated fixture: {e}"),
            }
        }
    }

    fn track(&self) -> PathBuf {
        self.dir.join("song.flac")
    }

    fn put(&self, name: &str, bytes: impl AsRef<[u8]>) {
        std::fs::write(self.dir.join(name), bytes).unwrap();
    }

    fn read(&self, name: &str) -> Vec<u8> {
        std::fs::read(self.dir.join(name)).unwrap()
    }

    fn snapshot(&self) -> BTreeMap<String, Vec<u8>> {
        std::fs::read_dir(&self.dir)
            .unwrap()
            .map(|entry| {
                let entry = entry.unwrap();
                (
                    entry.file_name().to_string_lossy().into_owned(),
                    std::fs::read(entry.path()).unwrap(),
                )
            })
            .collect()
    }

    fn seeded() -> Self {
        let scratch = Self::new();
        scratch.put("song.flac", [0x4d, 0x6f, 0x6f, 0xff, 0x00, 0x7f]);
        scratch.put("song.lrc", "[ti:Original]\r\n[ar:Artist]\r\n[offset:+17]\r\n[00:01.000]剣(つるぎ)\r\n[00:02.000]空\r\n");
        scratch.put(
            "song.romaji.lrc",
            "[00:01.000]tsurugi\r\n[00:02.000]sora\r\n",
        );
        scratch.put(
            "song.reading-override.lrc",
            "[00:01.000]剣(つるぎ)\r\n[00:02.000]\r\n",
        );
        scratch.put(
            "song.translation.lrc",
            "[00:01.000]hand translated sword\r\n[00:02.000]hand translated sky\r\n",
        );
        scratch
    }
}

impl Drop for Scratch {
    fn drop(&mut self) {
        // Remove only the exact temporary child this fixture created.
        if self.dir.parent() == Some(self.root.as_path())
            && self
                .dir
                .file_name()
                .is_some_and(|n| n.to_string_lossy().starts_with("moosik_language_save_"))
        {
            let _ = std::fs::remove_dir_all(&self.dir);
        }
    }
}

fn generated(track: &Path) -> Lyrics {
    let mut lyrics = load(track, None).unwrap();
    lyrics.lines[0].translation = Some("劍".into());
    lyrics.lines[1].translation = Some("天空".into());
    lyrics.translation_credit = Some("Machine-assisted Chinese translation · test provider".into());
    lyrics
}

fn assert_saved(outcome: SaveOutcome) {
    assert!(matches!(outcome, SaveOutcome::Saved { .. }), "{outcome:?}");
}

#[test]
fn translation_only_save_preserves_original_readings_and_audio_bytes() {
    let scratch = Scratch::seeded();
    let track = scratch.track();
    let before = scratch.snapshot();
    let lyrics = generated(&track);
    let outcome = save_changes(&track, &lyrics, false, false, true);
    let SaveOutcome::Saved {
        written,
        removed,
        backups,
    } = outcome
    else {
        panic!("{outcome:?}")
    };
    assert_eq!(written, vec![extra_path(&track, Extra::Translation)]);
    assert!(removed.is_empty());
    assert_eq!(backups, vec![scratch.dir.join("song.translation.lrc.old")]);
    for name in [
        "song.flac",
        "song.lrc",
        "song.romaji.lrc",
        "song.reading-override.lrc",
    ] {
        assert_eq!(
            scratch.read(name),
            before[name],
            "translation save changed {name}"
        );
    }
    assert_eq!(
        scratch.read("song.translation.lrc.old"),
        before["song.translation.lrc"]
    );
    let back = load(&track, None).unwrap();
    assert_eq!(back.lines, lyrics.lines);
    assert_eq!(back.translation_credit, lyrics.translation_credit);
}

#[test]
fn failure_after_translation_write_restores_every_file_byte_for_byte() {
    for sheet_changed in [false, true] {
        let scratch = Scratch::seeded();
        let track = scratch.track();
        let before = scratch.snapshot();
        let mut lyrics = generated(&track);
        if sheet_changed {
            lyrics.lines[0].at = Some(Duration::from_secs(4));
            lyrics.lines[1].at = Some(Duration::from_secs(7));
        }
        let mut reached = false;
        let mut hook = |stage: Stage, path: &Path| {
            if stage == Stage::Applied && path == extra_path(&track, Extra::Translation) {
                reached = true;
                return Err(std::io::Error::other("injected after translation write"));
            }
            Ok(())
        };
        let outcome = save_changes_with(&track, &lyrics, sheet_changed, false, true, &mut hook);
        assert!(reached, "failure did not reach the translation transaction");
        assert!(
            matches!(outcome, SaveOutcome::RolledBack { .. }),
            "{outcome:?}"
        );
        assert_eq!(
            scratch.snapshot(),
            before,
            "rollback left changed bytes or temporary files"
        );
        assert_saved(save_changes(&track, &lyrics, sheet_changed, false, true));
        assert_eq!(
            scratch.read("song.translation.lrc.old"),
            before["song.translation.lrc"]
        );
    }
}

#[test]
fn partly_synced_translation_placeholders_keep_their_rows_with_attribution() {
    let scratch = Scratch::new();
    let track = scratch.track();
    let source = "untimed intro\n[00:01.00]timed line\nuntimed middle\nuntimed translated line\n[00:02.00]another timed line\nuntimed ending\n";
    scratch.put("song.lrc", source);
    let mut lyrics = parse(source, LyricSource::Sidecar);
    // The leading, middle, and final untimed rows have real blank translation
    // placeholders; only their neighbours have translations.
    lyrics.lines[1].translation = Some("第一行".into());
    lyrics.lines[3].translation = Some("未同步的翻譯".into());
    lyrics.lines[4].translation = Some("第二行".into());
    lyrics.translation_credit = Some("Machine-assisted Chinese translation".into());
    let extra = render_extra(&lyrics, Extra::Translation);
    assert!(
        extra.starts_with('\n'),
        "leading placeholder disappeared: {extra:?}"
    );
    assert!(
        extra.contains("[00:02.00]第二行\n\n[moosik_translation:"),
        "trailing placeholder disappeared: {extra:?}"
    );
    assert_saved(save_changes(&track, &lyrics, false, false, true));
    let back = load(&track, None).unwrap();
    assert_eq!(
        back.lines, lyrics.lines,
        "blank placeholders shifted an untimed translation"
    );
    assert_eq!(back.translation_credit, lyrics.translation_credit);
}

#[test]
fn untimed_paragraph_blanks_round_trip_with_the_translation_marker() {
    let scratch = Scratch::new();
    let track = scratch.track();
    let source = "intro\n\nverse\nending\n";
    scratch.put("song.lrc", source);
    let mut lyrics = parse(source, LyricSource::Sidecar);
    lyrics.lines[0].translation = Some("序曲".into());
    lyrics.lines[2].translation = Some("歌詞".into());
    lyrics.translation_credit = Some("Machine-assisted Chinese translation".into());
    assert_saved(save_changes(&track, &lyrics, false, false, true));
    let back = load(&track, None).unwrap();
    assert_eq!(
        back.lines, lyrics.lines,
        "paragraph or trailing translation blank shifted the rows"
    );
    assert_eq!(back.translation_credit, lyrics.translation_credit);
}

#[test]
fn generated_attribution_survives_reload_and_sheet_retiming() {
    let scratch = Scratch::seeded();
    let track = scratch.track();
    let lyrics = generated(&track);
    assert_saved(save_changes(&track, &lyrics, false, false, true));
    let translation_before = scratch.read("song.translation.lrc");
    let mut retimed = load(&track, None).unwrap();
    retimed.lines[0].at = Some(Duration::from_millis(4_003));
    retimed.lines[1].at = Some(Duration::from_millis(7_007));
    assert_saved(save_all(&track, &retimed, true, false));
    let back = load(&track, None).unwrap();
    assert_eq!(back.lines, retimed.lines);
    assert_eq!(back.translation_credit, lyrics.translation_credit);
    assert_eq!(
        scratch.read("song.translation.lrc.old.2"),
        translation_before
    );
    let sheet = String::from_utf8(scratch.read("song.lrc")).unwrap();
    assert!(
        !sheet.contains("moosik_translation"),
        "translation attribution leaked into the original sheet"
    );
    for extra in [Extra::Romaji, Extra::Override] {
        let bytes = std::fs::read(extra_path(&track, extra)).unwrap();
        assert!(
            !String::from_utf8(bytes)
                .unwrap()
                .contains("moosik_translation")
        );
    }
}

#[test]
fn unmarked_translation_sidecars_stay_compatible_and_are_not_rewritten() {
    let scratch = Scratch::new();
    let track = scratch.track();
    scratch.put("song.lrc", "[00:01.00]one\n[00:02.00]two\n");
    let old = "[00:01.00]第一行\n[00:02.00]第二行\n";
    scratch.put("song.translation.lrc", old);
    let lyrics = load(&track, None).unwrap();
    assert_eq!(lyrics.translation_credit, None);
    assert_eq!(lyrics.lines[0].translation.as_deref(), Some("第一行"));
    assert_eq!(lyrics.lines[1].translation.as_deref(), Some("第二行"));
    let before = scratch.snapshot();
    let outcome = save_changes(&track, &lyrics, false, false, true);
    let SaveOutcome::Saved {
        written,
        removed,
        backups,
    } = outcome
    else {
        panic!("{outcome:?}")
    };
    assert!(written.is_empty() && removed.is_empty() && backups.is_empty());
    assert_eq!(scratch.snapshot(), before);
}

#[test]
fn attribution_cannot_inject_lrc_rows_or_metadata() {
    let scratch = Scratch::new();
    let track = scratch.track();
    scratch.put("song.lrc", "[00:01.00]one\n");
    let mut lyrics = load(&track, None).unwrap();
    lyrics.lines[0].translation = Some("一行".into());
    lyrics.translation_credit = Some("Provider\r\n[00:99.00]bad[by:other]".into());
    assert_saved(save_changes(&track, &lyrics, false, false, true));
    let saved = String::from_utf8(scratch.read("song.translation.lrc")).unwrap();
    assert_eq!(
        saved,
        "[00:01.00]一行\n[moosik_translation:Provider00:99.00badby:other]\n"
    );
    let back = load(&track, None).unwrap();
    assert_eq!(back.lines.len(), 1);
    assert_eq!(back.lines[0].translation.as_deref(), Some("一行"));
    assert_eq!(
        back.translation_credit.as_deref(),
        Some("Provider00:99.00badby:other")
    );
}

#[test]
fn attribution_length_limit_preserves_complete_unicode_characters() {
    let mut lyrics = parse("[00:01.00]one\n", LyricSource::Sidecar);
    lyrics.lines[0].translation = Some("一行".into());
    lyrics.translation_credit = Some("譯".repeat(200));
    let saved = render_extra(&lyrics, Extra::Translation);
    let credit = translation_credit(&saved).unwrap();
    assert_eq!(credit.chars().count(), 160);
    assert_eq!(credit, "譯".repeat(160));
    assert!(!lyrics.to_lrc().contains("moosik_translation"));
}
