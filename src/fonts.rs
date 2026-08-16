//! System font discovery, so the UI font can be chosen rather than assumed.
//!
//! Family names are read out of each file's `name` table rather than guessed
//! from the filename. A picker listing `segoeui`, `seguibl`, `seguili` is not a
//! picker anyone can use; `Segoe UI` is. The parsing is a few dozen lines and
//! buys the whole feature its usability.

use std::path::{Path, PathBuf};

/// One selectable face.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct FontEntry {
    /// Family name from the font's own `name` table, e.g. "Segoe UI".
    pub family: String,
    pub path: PathBuf,
    /// Face index within a collection (`.ttc`); 0 for a plain file.
    pub index: u32,
}

/// Directories that hold fonts on each platform, user-installed first so a
/// font the user added wins over a system one of the same name.
fn font_dirs() -> Vec<PathBuf> {
    let mut dirs: Vec<PathBuf> = Vec::new();
    #[cfg(windows)]
    {
        if let Ok(local) = std::env::var("LOCALAPPDATA") {
            dirs.push(PathBuf::from(local).join(r"Microsoft\Windows\Fonts"));
        }
        let root = std::env::var("SystemRoot").unwrap_or_else(|_| r"C:\Windows".into());
        dirs.push(PathBuf::from(root).join("Fonts"));
    }
    #[cfg(target_os = "macos")]
    {
        if let Some(h) = home() { dirs.push(h.join("Library/Fonts")); }
        dirs.push("/Library/Fonts".into());
        dirs.push("/System/Library/Fonts".into());
    }
    #[cfg(all(unix, not(target_os = "macos")))]
    {
        if let Some(h) = home() {
            dirs.push(h.join(".local/share/fonts"));
            dirs.push(h.join(".fonts"));
        }
        dirs.push("/usr/share/fonts".into());
        dirs.push("/usr/local/share/fonts".into());
    }
    dirs
}

#[allow(dead_code)]
fn home() -> Option<PathBuf> {
    std::env::var_os("HOME").map(PathBuf::from)
}

/// Every usable face on the system, one row per family, sorted by name.
///
/// Deduplicated by family because a family spans several files (regular, bold,
/// italic) and egui applies its own weight; offering "Segoe UI" eleven times
/// would be noise. The first file seen for a family wins, and directories are
/// walked user-first so a user-installed font takes precedence.
pub fn list() -> Vec<FontEntry> {
    let mut found: Vec<FontEntry> = Vec::new();
    let mut seen: std::collections::HashSet<String> = std::collections::HashSet::new();
    for dir in font_dirs() {
        collect_dir(&dir, 0, &mut found, &mut seen);
    }
    found.sort_by(|a, b| a.family.to_lowercase().cmp(&b.family.to_lowercase()));
    found
}

fn collect_dir(
    dir: &Path, depth: u32,
    out: &mut Vec<FontEntry>, seen: &mut std::collections::HashSet<String>,
) {
    // Linux nests fonts a few levels deep; the cap stops a symlink loop from
    // turning font discovery into a filesystem crawl.
    if depth > 3 { return; }
    let Ok(rd) = std::fs::read_dir(dir) else { return };
    for e in rd.flatten() {
        let path = e.path();
        if path.is_dir() { collect_dir(&path, depth + 1, out, seen); continue; }
        let ext = path.extension().and_then(|s| s.to_str())
            .map(|s| s.to_ascii_lowercase()).unwrap_or_default();
        if !matches!(ext.as_str(), "ttf" | "otf" | "ttc" | "otc") { continue; }
        for (family, index) in families_in(&path) {
            if seen.insert(family.to_lowercase()) {
                out.push(FontEntry { family, path: path.clone(), index });
            }
        }
    }
}

/// Family names in a font file, with the face index each belongs to.
fn families_in(path: &Path) -> Vec<(String, u32)> {
    let Ok(data) = std::fs::read(path) else { return Vec::new() };
    // A collection begins with 'ttcf' and a table of offsets to real sfnt
    // headers; a plain font is its own sfnt header at offset 0.
    if data.len() >= 12 && &data[0..4] == b"ttcf" {
        let n = be32(&data, 8).unwrap_or(0).min(64);
        (0..n)
            .filter_map(|i| {
                let off = be32(&data, 12 + 4 * i as usize)? as usize;
                family_at(&data, off).map(|f| (f, i))
            })
            .collect()
    } else {
        family_at(&data, 0).map(|f| vec![(f, 0)]).unwrap_or_default()
    }
}

fn be16(d: &[u8], at: usize) -> Option<u16> {
    Some(u16::from_be_bytes([*d.get(at)?, *d.get(at + 1)?]))
}
fn be32(d: &[u8], at: usize) -> Option<u32> {
    Some(u32::from_be_bytes([*d.get(at)?, *d.get(at + 1)?, *d.get(at + 2)?, *d.get(at + 3)?]))
}

/// Read name ID 1 (font family) from the `name` table of the sfnt at `base`.
fn family_at(d: &[u8], base: usize) -> Option<String> {
    let num_tables = be16(d, base + 4)? as usize;
    let mut name_off = None;
    for i in 0..num_tables.min(512) {
        let rec = base + 12 + 16 * i;
        if d.get(rec..rec + 4)? == b"name" {
            name_off = Some(be32(d, rec + 8)? as usize);
            break;
        }
    }
    let n = name_off?;
    let count = be16(d, n + 2)? as usize;
    let storage = n + be16(d, n + 4)? as usize;

    // Prefer the Windows/Unicode record; fall back to Mac Roman, which is all
    // some older fonts carry.
    let mut mac: Option<String> = None;
    for i in 0..count.min(1024) {
        let r = n + 6 + 12 * i;
        if be16(d, r + 6)? != 1 { continue; } // nameID 1 = family
        let (platform, len, off) = (be16(d, r)?, be16(d, r + 8)? as usize,
                                    be16(d, r + 10)? as usize);
        let bytes = d.get(storage + off..storage + off + len)?;
        match platform {
            3 | 0 => {
                // UTF-16BE.
                let u: Vec<u16> = bytes.chunks_exact(2)
                    .map(|c| u16::from_be_bytes([c[0], c[1]])).collect();
                let s = String::from_utf16_lossy(&u);
                if !s.trim().is_empty() { return Some(s.trim().to_string()); }
            }
            1 if mac.is_none() => {
                let s: String = bytes.iter().map(|&b| b as char).collect();
                if !s.trim().is_empty() { mac = Some(s.trim().to_string()); }
            }
            _ => {}
        }
    }
    mac
}

/// Find a listed entry by family name.
pub fn find(family: &str) -> Option<FontEntry> {
    list().into_iter().find(|f| f.family.eq_ignore_ascii_case(family))
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The parser has to survive whatever is on disk — a truncated file, a
    /// bitmap font with the wrong extension, a directory of junk — without
    /// panicking, because it runs over every font the system has.
    #[test]
    fn malformed_input_yields_nothing_rather_than_panicking() {
        assert!(family_at(&[], 0).is_none());
        assert!(family_at(&[0u8; 8], 0).is_none());
        assert!(family_at(&[0xFF; 64], 0).is_none());
        // Claims 65535 tables in 20 bytes.
        let mut d = vec![0u8; 20];
        d[4] = 0xFF; d[5] = 0xFF;
        assert!(family_at(&d, 0).is_none());
    }

    #[test]
    fn a_truncated_collection_header_is_survivable() {
        let dir = std::env::temp_dir().join("moosik-font-test");
        let _ = std::fs::create_dir_all(&dir);
        let p = dir.join("bad.ttc");
        std::fs::write(&p, b"ttcf\x00\x01\x00\x00\xff\xff\xff\xff").unwrap();
        assert!(families_in(&p).is_empty());
        let _ = std::fs::remove_dir_all(&dir);
    }

    /// Reads the real system, so it asserts only what must hold anywhere.
    #[test]
    fn the_system_has_fonts_and_they_are_unique_and_sorted() {
        let all = list();
        if all.is_empty() { return; } // a headless box may genuinely have none
        let mut names: Vec<String> = all.iter().map(|f| f.family.to_lowercase()).collect();
        let before = names.len();
        names.dedup();
        assert_eq!(names.len(), before, "families must be listed once each");
        assert!(all.windows(2).all(|w| w[0].family.to_lowercase() <= w[1].family.to_lowercase()),
                "list must be sorted for a usable picker");
        assert!(all.iter().all(|f| !f.family.trim().is_empty()));
        // A name read from a real table should be legible, not mojibake.
        assert!(all.iter().any(|f| f.family.chars().any(|c| c.is_alphabetic())));
    }

    /// `cargo test --bin moosik -- --ignored --nocapture list_installed`
    #[test]
    #[ignore = "inspection — run explicitly"]
    fn list_installed() {
        let all = list();
        println!("{} families", all.len());
        for f in &all {
            if f.family.to_lowercase().contains("fira")
                || f.family.to_lowercase().contains("consol")
                || f.family.to_lowercase().contains("segoe")
            {
                println!("  {:?} idx={} {}", f.family, f.index, f.path.display());
                // What the picker would actually do with a click on this row.
                match find(&f.family) {
                    Some(e) => println!("     find -> {} ({} bytes)", e.path.display(),
                                        std::fs::metadata(&e.path).map(|m| m.len()).unwrap_or(0)),
                    None => println!("     find -> NOTHING (picker would silently no-op)"),
                }
            }
        }
    }

    #[test]
    fn find_is_case_insensitive() {
        let Some(first) = list().into_iter().next() else { return };
        assert_eq!(find(&first.family.to_uppercase()).map(|f| f.path), Some(first.path));
    }
}
