//! Zip archives held in memory: just enough for the pinned or checksummed
//! downloads Moosik unpacks (stored or deflated members).

fn u16_at(b: &[u8], at: usize) -> Result<usize, String> {
    b.get(at..at + 2).map(|s| u16::from_le_bytes([s[0], s[1]]) as usize).ok_or_else(|| "damaged archive".to_string())
}
fn u32_at(b: &[u8], at: usize) -> Result<usize, String> {
    b.get(at..at + 4).map(|s| u32::from_le_bytes([s[0], s[1], s[2], s[3]]) as usize).ok_or_else(|| "damaged archive".to_string())
}

/// One member of a zip archive held in memory, inflated.
pub fn member(zip: &[u8], name: &str) -> Result<Vec<u8>, String> {
    let mut found = None;
    walk(zip, &mut |this, data| {
        if this == name.as_bytes() {
            found = Some(data()?);
            return Ok(false);
        }
        Ok(true)
    })?;
    found.ok_or_else(|| format!("{name} is not in the download"))
}

/// Every file in a zip archive held in memory, by its path inside it
/// (folders are left out; a path that would leave the folder is refused).
pub fn all(zip: &[u8]) -> Result<Vec<(String, Vec<u8>)>, String> {
    let mut out = Vec::new();
    walk(zip, &mut |this, data| {
        let name = String::from_utf8_lossy(this).replace('\\', "/");
        if name.ends_with('/') {
            return Ok(true);
        }
        if name.starts_with('/') || name.contains(':') || name.split('/').any(|p| p == "..") {
            return Err(format!("unsafe path in archive: {name}"));
        }
        out.push((name, data()?));
        Ok(true)
    })?;
    Ok(out)
}

/// Each central-directory entry's name, with a way to inflate it; `f`
/// returns whether to go on.
fn walk(zip: &[u8], f: &mut dyn FnMut(&[u8], &dyn Fn() -> Result<Vec<u8>, String>) -> Result<bool, String>) -> Result<(), String> {
    // End of central directory: the last 22+ bytes, signature PK\x05\x06.
    let tail = zip.len().saturating_sub(22 + 65_535);
    let eocd = (tail..zip.len().saturating_sub(21)).rev().find(|&i| zip[i..i + 4] == [0x50, 0x4b, 0x05, 0x06]).ok_or("not a zip archive")?;
    let (entries, mut at) = (u16_at(zip, eocd + 10)?, u32_at(zip, eocd + 16)?);
    for _ in 0..entries {
        if u32_at(zip, at)? != 0x0201_4b50 {
            return Err("damaged archive".into());
        }
        let (method, csize, usize_) = (u16_at(zip, at + 10)?, u32_at(zip, at + 20)?, u32_at(zip, at + 24)?);
        let (nlen, xlen, clen, local) = (u16_at(zip, at + 28)?, u16_at(zip, at + 30)?, u16_at(zip, at + 32)?, u32_at(zip, at + 42)?);
        let this = zip.get(at + 46..at + 46 + nlen).ok_or("damaged archive")?;
        at += 46 + nlen + xlen + clen;
        let data = || -> Result<Vec<u8>, String> {
            if u32_at(zip, local)? != 0x0403_4b50 {
                return Err("damaged archive".into());
            }
            let start = local + 30 + u16_at(zip, local + 26)? + u16_at(zip, local + 28)?;
            let data = zip.get(start..start + csize).ok_or("damaged archive")?;
            match method {
                0 => Ok(data.to_vec()),
                8 => miniz_oxide::inflate::decompress_to_vec_with_limit(data, usize_).map_err(|_| "damaged archive".to_string()),
                _ => Err("unsupported compression".into()),
            }
        };
        if !f(this, &data)? {
            break;
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A zip of stored members, as any archiver would write it.
    fn zip(files: &[(&str, &[u8])]) -> Vec<u8> {
        let (mut out, mut dir) = (Vec::new(), Vec::new());
        for (name, data) in files {
            let at = out.len() as u32;
            let head = |sig: u32, central: bool| {
                let mut h = sig.to_le_bytes().to_vec();
                if central {
                    h.extend([20, 0]);
                }
                h.extend([20, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]);
                h.extend((data.len() as u32).to_le_bytes());
                h.extend((data.len() as u32).to_le_bytes());
                h.extend((name.len() as u16).to_le_bytes());
                h.extend([0, 0]);
                if central {
                    h.extend([0; 10]);
                    h.extend(at.to_le_bytes());
                }
                h.extend(name.as_bytes());
                h
            };
            out.extend(head(0x0403_4b50, false));
            out.extend(*data);
            dir.extend(head(0x0201_4b50, true));
        }
        let (at, len) = (out.len() as u32, dir.len() as u32);
        out.extend(dir);
        out.extend([0x50, 0x4b, 5, 6, 0, 0, 0, 0]);
        out.extend((files.len() as u16).to_le_bytes());
        out.extend((files.len() as u16).to_le_bytes());
        out.extend(len.to_le_bytes());
        out.extend(at.to_le_bytes());
        out.extend([0, 0]);
        out
    }

    #[test]
    fn all_members_come_out_and_escapes_are_refused() {
        let z = zip(&[("yt-dlp.exe", b"exe"), ("_internal/", b""), ("_internal/python.dll", b"dll")]);
        assert_eq!(member(&z, "_internal/python.dll").unwrap(), b"dll");
        let all = all(&z).unwrap();
        assert_eq!(all, [("yt-dlp.exe".to_string(), b"exe".to_vec()), ("_internal/python.dll".to_string(), b"dll".to_vec())]);
        for bad in ["../evil.exe", "a/../../evil.exe", "C:/evil.exe", "/evil.exe", r"a\..\..\evil.exe"] {
            assert!(super::all(&zip(&[(bad, b"x")])).is_err(), "{bad}");
        }
    }
}
