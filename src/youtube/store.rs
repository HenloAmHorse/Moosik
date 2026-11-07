//! What Moosik knows about audio it got from YouTube, by file: title,
//! artist, album and where it came from. yt-dlp cannot write tags into the
//! file without ffmpeg, so this stands in for them; it is consulted only for
//! a file that has no tags of its own. One JSON file, saved whole.

use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::path::{Path, PathBuf};
use std::sync::Mutex;

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct Info {
    pub id: String,
    pub title: String,
    pub artist: String,
    pub album: String,
    pub duration: Option<f32>,
    pub url: String,
    /// The size its music video was last asked for at, so one YouTube has
    /// no taller version of is not fetched again for nothing.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub video_asked: Option<u32>,
}

/// The records, loaded once and kept in step with the file.
static RECORDS: Mutex<Option<(PathBuf, HashMap<String, Info>)>> = Mutex::new(None);

fn key(path: &Path) -> String {
    path.to_string_lossy().to_lowercase()
}

fn with<T>(file: &Path, f: impl FnOnce(&mut HashMap<String, Info>) -> T) -> T {
    let mut g = RECORDS.lock().unwrap_or_else(|p| p.into_inner());
    if g.as_ref().is_none_or(|(p, _)| p != file) {
        let map = std::fs::read_to_string(file).ok().and_then(|t| serde_json::from_str(&t).ok()).unwrap_or_default();
        *g = Some((file.to_path_buf(), map));
    }
    f(&mut g.as_mut().expect("just loaded").1)
}

/// The record file in a Moosik folder.
pub fn file(moosik: &Path) -> PathBuf {
    moosik.join("youtube.json")
}

/// What is known about `path`, if it came from YouTube.
pub fn lookup(records: &Path, path: &Path) -> Option<Info> {
    with(records, |m| m.get(&key(path)).cloned())
}

/// Remember where `path` came from; written aside, then moved into place.
pub fn record(records: &Path, path: &Path, info: &Info) -> Result<(), String> {
    with(records, |m| {
        m.insert(key(path), info.clone());
        let text = serde_json::to_string_pretty(m).map_err(|e| e.to_string())?;
        if let Some(d) = records.parent() {
            let _ = std::fs::create_dir_all(d);
        }
        let tmp = records.with_extension("json.part");
        std::fs::write(&tmp, text).and_then(|_| std::fs::rename(&tmp, records)).map_err(|e| format!("{}: {e}", records.display()))
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_record_survives_a_reload_and_ignores_case() {
        let dir = std::env::temp_dir().join(format!("moosik-ytstore-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&dir);
        let f = file(&dir);
        let info = Info { id: "jNQXAC9IVRw".into(), title: "Me at the zoo".into(), artist: "jawed".into(), album: "YouTube".into(), duration: Some(19.0), url: "u".into(), video_asked: None };
        record(&f, Path::new(r"C:\Cache\jNQXAC9IVRw.webm"), &info).unwrap();
        *RECORDS.lock().unwrap() = None; // as if Moosik had restarted
        assert_eq!(lookup(&f, Path::new(r"c:\cache\JNQXAC9IVRW.WEBM")), Some(info));
        assert_eq!(lookup(&f, Path::new(r"C:\Cache\other.webm")), None);
        std::fs::remove_dir_all(&dir).unwrap();
    }
}
