//! Tag reading and editing.
//!
//! Reading is unrestricted — every item in the file's primary tag, whatever it
//! is. Writing is deliberately narrow, because this is the only part of the
//! player that modifies a user's audio files:
//!
//! * **DSD is never written.** DSF stores its ID3 blob behind a pointer in the
//!   header, and `probe_tagged` reaches it by extracting the blob and handing it
//!   to lofty's MPEG reader — a read-only trick. Writing through that path would
//!   produce a file whose header points at a tag of the wrong length, which
//!   costs the audio, not the metadata.
//! * **Writes go to a copy**, which then replaces the original by rename. A
//!   tag write resizes a region in the middle of the file; interrupt one in
//!   place and the file is rubble. The original is untouched until a complete
//!   new file exists.
//! * **Every write is read back** and compared before it is accepted.

pub mod ui;

use lofty::prelude::{ItemKey, TagExt as _, TaggedFileExt as _};
use std::path::{Path, PathBuf};

/// The fields the editor offers, in display order.
///
/// Deliberately not "every key lofty knows". These are the ones that are worth
/// correcting by hand and that every container can store; anything more
/// exotic is visible in the viewer and best left to a dedicated tagger.
pub const EDITABLE: &[(ItemKey, &str)] = &[
    (ItemKey::TrackTitle,  "Title"),
    (ItemKey::TrackArtist, "Artist"),
    (ItemKey::AlbumTitle,  "Album"),
    (ItemKey::AlbumArtist, "Album artist"),
    (ItemKey::TrackNumber, "Track #"),
    (ItemKey::TrackTotal,  "of"),
    (ItemKey::DiscNumber,  "Disc #"),
    (ItemKey::DiscTotal,   "of"),
    (ItemKey::Year,        "Year"),
    (ItemKey::Genre,       "Genre"),
    (ItemKey::Composer,    "Composer"),
    (ItemKey::Comment,     "Comment"),
];

/// Current values for [`EDITABLE`], in the same order. An empty string means
/// the field is absent, and writing one back removes the key rather than
/// storing an empty value.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct Edits {
    pub values: Vec<String>,
}

impl Edits {
    pub fn empty() -> Self {
        Self { values: vec![String::new(); EDITABLE.len()] }
    }
    /// Indices whose value differs from `other` — what a write would touch.
    pub fn changed_from(&self, other: &Edits) -> Vec<usize> {
        (0..EDITABLE.len())
            .filter(|&i| self.values.get(i) != other.values.get(i))
            .collect()
    }
}

/// Why a file cannot be written, or `Ok(())` if it can.
pub fn writable(path: &Path) -> Result<(), String> {
    if crate::dsd::is_dsd_path(path) {
        return Err("DSD files are read-only here — writing a DSF tag means \
                    rebuilding the header pointer, and getting it wrong costs \
                    the audio. Edit these in a dedicated tagger.".into());
    }
    if !path.is_file() {
        return Err("File not found.".into());
    }
    match std::fs::metadata(path) {
        Ok(m) if m.permissions().readonly() => Err("File is read-only.".into()),
        Ok(_) => Ok(()),
        Err(e) => Err(e.to_string()),
    }
}

/// Every item in the file's primary tag, as `(key, value)` for display.
///
/// Keys lofty does not have a name for come back under their raw identifier,
/// so a `TXXX` frame or an odd Vorbis comment is still visible rather than
/// silently dropped.
pub fn read_all(path: &Path) -> Vec<(String, String)> {
    let Some(tagged) = crate::probe_tagged(path) else { return Vec::new() };
    let Some(tag) = tagged.primary_tag().or_else(|| tagged.first_tag()) else {
        return Vec::new();
    };
    let mut out: Vec<(String, String)> = tag.items()
        .filter_map(|item| {
            let key = match item.key() {
                ItemKey::Unknown(s) => s.clone(),
                k => format!("{k:?}"),
            };
            item.value().text().map(|v| (key, v.to_string()))
        })
        .collect();
    out.sort_by(|a, b| a.0.cmp(&b.0));
    out
}

/// Which tag format the file actually carries, for the UI to state plainly.
pub fn tag_kind(path: &Path) -> Option<String> {
    let tagged = crate::probe_tagged(path)?;
    let tag = tagged.primary_tag().or_else(|| tagged.first_tag())?;
    Some(format!("{:?}", tag.tag_type()))
}

/// Current values of the editable fields.
pub fn read_edits(path: &Path) -> Edits {
    let mut e = Edits::empty();
    let Some(tagged) = crate::probe_tagged(path) else { return e };
    let Some(tag) = tagged.primary_tag().or_else(|| tagged.first_tag()) else { return e };
    for (i, (key, _)) in EDITABLE.iter().enumerate() {
        e.values[i] = tag.get_string(key).unwrap_or_default().to_string();
    }
    e
}

/// Write `edits` into `path`'s tag, preserving everything else it carries.
///
/// Writes a modified copy and renames it over the original, so a failure
/// part-way leaves the original file exactly as it was. The result is read back
/// and compared before the write is reported as successful — a tag that silently
/// did not take is worse than an error, because the user believes it worked.
pub fn write(path: &Path, edits: &Edits) -> Result<(), String> {
    writable(path)?;

    let mut tagged = crate::probe_tagged(path).ok_or("Could not read the file's tags.")?;
    // A file with no tag at all still needs somewhere to put one.
    if tagged.primary_tag().is_none() && tagged.first_tag().is_none() {
        let ty = tagged.primary_tag_type();
        tagged.insert_tag(lofty::tag::Tag::new(ty));
    }
    let tag = match tagged.primary_tag_mut() {
        Some(_) => tagged.primary_tag_mut().expect("just checked"),
        None => tagged.first_tag_mut().ok_or("This format cannot store tags.")?,
    };

    for (i, (key, _)) in EDITABLE.iter().enumerate() {
        match edits.values[i].trim() {
            // Blank means "remove", not "store an empty string" — an empty
            // frame is not the same as an absent one to other readers.
            "" => tag.remove_key(key),
            v => { tag.insert_text(key.clone(), v.to_string()); }
        }
    }
    let tag = tag.clone();

    let tmp = temp_beside(path);
    std::fs::copy(path, &tmp).map_err(|e| format!("Could not stage a copy: {e}"))?;
    if let Err(e) = tag.save_to_path(&tmp, lofty::config::WriteOptions::default()) {
        let _ = std::fs::remove_file(&tmp);
        return Err(format!("Write failed (original untouched): {e}"));
    }

    // Verify against the staged file before it replaces anything.
    let written = read_edits(&tmp);
    let want: Vec<String> = edits.values.iter().map(|v| v.trim().to_string()).collect();
    if written.values != want {
        let _ = std::fs::remove_file(&tmp);
        let bad = (0..EDITABLE.len())
            .find(|&i| written.values[i] != want[i])
            .map(|i| EDITABLE[i].1).unwrap_or("a field");
        return Err(format!(
            "The file did not keep the change to “{bad}” — this format may not \
             support it. Original untouched."));
    }

    std::fs::rename(&tmp, path).map_err(|e| {
        let _ = std::fs::remove_file(&tmp);
        format!("Could not replace the original: {e}")
    })
}

/// A staging path in the same directory, so the final rename stays on one
/// volume — a cross-device rename is a copy, which is exactly the
/// half-written state this is avoiding.
fn temp_beside(path: &Path) -> PathBuf {
    let stamp = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|d| d.as_nanos()).unwrap_or(0);
    let name = path.file_name().map(|s| s.to_string_lossy().to_string())
        .unwrap_or_else(|| "track".into());
    path.with_file_name(format!(".moosik-{stamp:x}-{name}"))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn dsd_is_refused_before_anything_is_touched() {
        // The whole reason this check exists: the DSD read path is a
        // read-only trick and must never become a write path.
        let e = writable(Path::new("x.dsf")).unwrap_err();
        assert!(e.contains("DSD"), "{e}");
        assert!(writable(Path::new("x.dff")).is_err());
    }

    #[test]
    fn a_missing_file_is_refused() {
        assert!(writable(Path::new("definitely-not-here.flac")).is_err());
    }

    #[test]
    fn edits_report_only_what_actually_changed() {
        let a = Edits::empty();
        let mut b = a.clone();
        assert!(a.changed_from(&b).is_empty());
        b.values[0] = "New title".into();
        assert_eq!(b.changed_from(&a), vec![0]);
        b.values[3] = "VA".into();
        assert_eq!(b.changed_from(&a), vec![0, 3]);
    }

    #[test]
    fn the_staging_file_is_a_sibling() {
        // A rename across volumes is a copy, which defeats the point.
        let p = Path::new("C:/music/album/track.flac");
        let t = temp_beside(p);
        assert_eq!(t.parent(), p.parent());
        assert_ne!(t.file_name(), p.file_name());
    }

    #[test]
    fn every_editable_field_has_a_slot() {
        assert_eq!(Edits::empty().values.len(), EDITABLE.len());
    }

    /// Round-trips a real file. Ignored by default because it writes to disk.
    ///
    /// `cargo test --bin moosik -- --ignored --nocapture tag_round_trip`
    #[test]
    #[ignore = "writes a temp file — run explicitly"]
    fn tag_round_trip() {
        // A real, minimal WAV. Building the container by hand rather than
        // shipping a fixture keeps the test self-contained, and WAV is the
        // one format whose header is short enough to write out honestly.
        let dir = std::env::temp_dir().join("moosik-tag-test");
        let _ = std::fs::create_dir_all(&dir);
        let path = dir.join("t.wav");
        let audio = vec![0u8; 800]; // 200 frames of 16-bit stereo silence
        let mut b = Vec::new();
        b.extend_from_slice(b"RIFF");
        b.extend_from_slice(&(36u32 + audio.len() as u32).to_le_bytes());
        b.extend_from_slice(b"WAVEfmt ");
        b.extend_from_slice(&16u32.to_le_bytes());   // PCM chunk size
        b.extend_from_slice(&1u16.to_le_bytes());    // PCM
        b.extend_from_slice(&2u16.to_le_bytes());    // stereo
        b.extend_from_slice(&44100u32.to_le_bytes());
        b.extend_from_slice(&176400u32.to_le_bytes()); // byte rate
        b.extend_from_slice(&4u16.to_le_bytes());    // block align
        b.extend_from_slice(&16u16.to_le_bytes());   // bits
        b.extend_from_slice(b"data");
        b.extend_from_slice(&(audio.len() as u32).to_le_bytes());
        b.extend_from_slice(&audio);
        std::fs::write(&path, &b).unwrap();

        let mut e = read_edits(&path);
        e.values[0] = "ヴァンパイア".into();
        e.values[1] = "DECO*27".into();
        e.values[4] = "3".into();
        match write(&path, &e) {
            Ok(()) => {
                let back = read_edits(&path);
                assert_eq!(back.values[0], "ヴァンパイア");
                assert_eq!(back.values[1], "DECO*27");
                assert_eq!(back.values[4], "3");
                // Blanking removes the key rather than storing "".
                let mut clear = back.clone();
                clear.values[4] = String::new();
                write(&path, &clear).unwrap();
                assert_eq!(read_edits(&path).values[4], "");
                println!("round trip ok; staged files left: {}", staged_count(&dir));
                assert_eq!(staged_count(&dir), 0, "staging files must not leak");
            }
            Err(e) => panic!("write failed: {e}"),
        }
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[cfg(test)]
    fn staged_count(dir: &Path) -> usize {
        std::fs::read_dir(dir).map(|rd| rd.filter_map(|e| e.ok())
            .filter(|e| e.file_name().to_string_lossy().starts_with(".moosik-"))
            .count()).unwrap_or(0)
    }
}
