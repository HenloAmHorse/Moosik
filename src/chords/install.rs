//! Getting the weights: the lv-chordia 1.1.0 wheel from PyPI (pinned by
//! SHA-256), and its five `.sdict` members out of it, each checked against
//! its own pinned SHA-256. Only those members are read; the files are
//! written under fixed names, never under a name taken from the archive.

use std::io::{Read, Write};
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicBool, Ordering};

const WHEEL_URL: &str = "https://files.pythonhosted.org/packages/80/76/02402f0182378f6157497ee7cafce62e72e4b05993e090ca29670b973fa3/lv_chordia-1.1.0-py3-none-any.whl";
const WHEEL_SHA256: &str = "292241f03e1029e6739ee7a2a91fbe348c25c97b8b3641db76cbef9f9dee0cb0";
pub const WHEEL_BYTES: u64 = 27_182_788;
const MEMBER_DIR: &str = "lv_chordia-1.1.0.data/data/share/lv-chordia/cache_data/";
/// SHA-256 of each weight file, in `super::WEIGHT_FILES` order.
const MEMBER_SHA256: [&str; 5] = [
    "921b42d5d1cf9ce1c0c0e45a74d409b8066e0acec46058ef74e24ee0fb540761",
    "bcb75859e0efa256696cf5da396b320093317b9b1d9560c304f46c25fe1f8b17",
    "acddf85c3fff29954c4877021177d72e2cba9f729ce80c1010f054c477bf3f61",
    "65d81a3ab73435aaaade586981b4cabdf57b8953d76052703e6968c32ef8421c",
    "5ff6b0ec85640e17a09a9b3de68c93fdd45adc24488e8fa9be5715c28d561122",
];
const LICENSE: &str = "MIT License\n\nCopyright (c) 2023 Music X Lab\n\nPermission is hereby granted, free of charge, to any person obtaining a copy\nof this software and associated documentation files (the \"Software\"), to deal\nin the Software without restriction, including without limitation the rights\nto use, copy, modify, merge, publish, distribute, sublicense, and/or sell\ncopies of the Software, and to permit persons to whom the Software is\nfurnished to do so, subject to the following conditions:\n\nThe above copyright notice and this permission notice shall be included in all\ncopies or substantial portions of the Software.\n\nTHE SOFTWARE IS PROVIDED \"AS IS\", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR\nIMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,\nFITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE\nAUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER\nLIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,\nOUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE\nSOFTWARE.\n";
pub const NOTICE: &str = "Chord model: lv-chordia 1.1.0 (Junyan Jiang, Ke Chen, Wei Li, Gus Xia: \"Large-Vocabulary Chord Transcription via Chord Structure Decomposition\", ISMIR 2019), MIT License, Music X Lab. Downloaded from PyPI; only its five weight files are kept.";

/// Where the weights live.
pub fn dir(moosik: &Path) -> PathBuf {
    moosik.join("models").join("chords-lv-chordia-1.1.0")
}

fn sha256(bytes: &[u8]) -> String {
    ring::digest::digest(&ring::digest::SHA256, bytes).as_ref().iter().map(|b| format!("{b:02x}")).collect()
}

/// All five weight files are there and are the pinned ones.
pub fn installed(dir: &Path) -> bool {
    super::WEIGHT_FILES.iter().zip(MEMBER_SHA256).all(|(f, want)| std::fs::read(dir.join(f)).is_ok_and(|b| sha256(&b) == want))
}

/// Download, check and unpack. `progress(done, total)` in bytes; `cancel`
/// stops it, keeping a partial download for next time.
pub fn install(dir: &Path, cancel: &AtomicBool, mut progress: impl FnMut(u64, u64)) -> Result<(), String> {
    std::fs::create_dir_all(dir).map_err(|e| format!("{}: {e}", dir.display()))?;
    let part = dir.join("download.whl.part");
    let have = std::fs::metadata(&part).map(|m| m.len()).unwrap_or(0);
    if have != WHEEL_BYTES {
        fetch(&part, have, cancel, &mut progress)?;
    }
    let wheel = std::fs::read(&part).map_err(|e| format!("{}: {e}", part.display()))?;
    if sha256(&wheel) != WHEEL_SHA256 {
        let _ = std::fs::remove_file(&part);
        return Err("The download is not the pinned lv-chordia 1.1.0 file; it was deleted and nothing was installed.".into());
    }
    for (name, want) in super::WEIGHT_FILES.iter().zip(MEMBER_SHA256) {
        let bytes = member(&wheel, &format!("{MEMBER_DIR}{name}"))?;
        if sha256(&bytes) != want {
            return Err(format!("{name} in the download is not the pinned file; nothing was installed."));
        }
        let tmp = dir.join(format!("{name}.part"));
        std::fs::write(&tmp, &bytes).map_err(|e| format!("{}: {e}", tmp.display()))?;
        std::fs::rename(&tmp, dir.join(name)).map_err(|e| format!("{}: {e}", dir.display()))?;
    }
    std::fs::write(dir.join("LICENSE-lv-chordia.txt"), LICENSE).map_err(|e| format!("{}: {e}", dir.display()))?;
    std::fs::write(dir.join("NOTICE.txt"), NOTICE).map_err(|e| format!("{}: {e}", dir.display()))?;
    let _ = std::fs::remove_file(&part);
    Ok(())
}

/// Download the wheel into `part`, resuming after `have` bytes when the server allows.
fn fetch(part: &Path, have: u64, cancel: &AtomicBool, progress: &mut impl FnMut(u64, u64)) -> Result<(), String> {
    // No overall limit: a slow line may need minutes for 27 MB.
    let agent: ureq::Agent = ureq::Agent::config_builder()
        .timeout_connect(Some(std::time::Duration::from_secs(30)))
        .timeout_recv_response(Some(std::time::Duration::from_secs(60)))
        .build()
        .into();
    let mut req = agent.get(WHEEL_URL);
    if have > 0 && have < WHEEL_BYTES {
        req = req.header("Range", format!("bytes={have}-"));
    }
    let resp = req.call().map_err(|e| format!("Downloading the chord model failed: {e}"))?;
    let resumed = resp.status().as_u16() == 206;
    let mut out = std::fs::OpenOptions::new()
        .create(true)
        .write(true)
        .append(resumed)
        .truncate(!resumed)
        .open(&part)
        .map_err(|e| format!("{}: {e}", part.display()))?;
    let mut done = if resumed { have } else { 0 };
    let mut body = resp.into_body().into_reader();
    let mut buf = vec![0u8; 1 << 16];
    loop {
        if cancel.load(Ordering::Relaxed) {
            return Err("Cancelled; the part downloaded so far is kept for next time.".into());
        }
        let n = body.read(&mut buf).map_err(|e| format!("Downloading the chord model failed: {e}"))?;
        if n == 0 {
            break;
        }
        if done + n as u64 > WHEEL_BYTES {
            drop(out);
            let _ = std::fs::remove_file(&part);
            return Err("The download is larger than the pinned file; nothing was installed.".into());
        }
        out.write_all(&buf[..n]).map_err(|e| format!("{}: {e}", part.display()))?;
        done += n as u64;
        progress(done, WHEEL_BYTES);
    }
    drop(out);
    // A connection closed early is an interruption, not a bad file: keep
    // what came so the next try resumes instead of failing the checksum.
    if done < WHEEL_BYTES {
        return Err(format!("The download stopped at {done} of {WHEEL_BYTES} bytes; it will resume from there next time."));
    }
    Ok(())
}

fn u16_at(b: &[u8], at: usize) -> Result<usize, String> {
    b.get(at..at + 2).map(|s| u16::from_le_bytes([s[0], s[1]]) as usize).ok_or_else(|| "damaged archive".to_string())
}
fn u32_at(b: &[u8], at: usize) -> Result<usize, String> {
    b.get(at..at + 4).map(|s| u32::from_le_bytes([s[0], s[1], s[2], s[3]]) as usize).ok_or_else(|| "damaged archive".to_string())
}

/// One member of a zip archive held in memory, inflated.
fn member(zip: &[u8], name: &str) -> Result<Vec<u8>, String> {
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
        if this != name.as_bytes() {
            continue;
        }
        if u32_at(zip, local)? != 0x0403_4b50 {
            return Err("damaged archive".into());
        }
        let start = local + 30 + u16_at(zip, local + 26)? + u16_at(zip, local + 28)?;
        let data = zip.get(start..start + csize).ok_or("damaged archive")?;
        return match method {
            0 => Ok(data.to_vec()),
            8 => miniz_oxide::inflate::decompress_to_vec_with_limit(data, usize_).map_err(|_| "damaged archive".to_string()),
            _ => Err("unsupported compression".into()),
        };
    }
    Err(format!("{name} is not in the download"))
}
