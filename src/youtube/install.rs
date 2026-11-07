//! The two optional components YouTube needs: yt-dlp (Unlicense) and Deno
//! (MIT), the JavaScript runtime yt-dlp uses for YouTube. Downloaded only
//! when the user asks, always the latest release, each checked against the
//! SHA-256 its own release publishes. That catches a damaged or swapped
//! file on the way; it is not a signature, so it trusts GitHub and the two
//! projects. Files are written aside and moved into place, so a failed
//! update leaves the working copy as it was.

use std::io::{Read, Write};
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicBool, Ordering};

// The one-folder build: the one-file yt-dlp.exe unpacks itself on every
// run, which costs about 0.6 s per search and per download.
const YTDLP_URL: &str = "https://github.com/yt-dlp/yt-dlp/releases/latest/download/yt-dlp_win.zip";
const YTDLP_SUMS: &str = "https://github.com/yt-dlp/yt-dlp/releases/latest/download/SHA2-256SUMS";
const DENO_URL: &str = "https://github.com/denoland/deno/releases/latest/download/deno-x86_64-pc-windows-msvc.zip";
const DENO_SUM: &str = "https://github.com/denoland/deno/releases/latest/download/deno-x86_64-pc-windows-msvc.zip.sha256sum";
/// Download and disk sizes as of October 2026, for the setup card.
pub const ABOUT_DOWNLOAD_MB: u32 = 67;
pub const ABOUT_DISK_MB: u32 = 140;

pub fn dir(moosik: &Path) -> PathBuf {
    moosik.join("tools").join("youtube")
}
/// yt-dlp in its folder; or, installed before 1.6.1's one-folder build,
/// the one-file yt-dlp.exe until the next update.
pub fn ytdlp_path(dir: &Path) -> PathBuf {
    let folder = dir.join("yt-dlp").join("yt-dlp.exe");
    if folder.is_file() { folder } else { dir.join("yt-dlp.exe") }
}
pub fn deno_path(dir: &Path) -> PathBuf {
    dir.join("deno.exe")
}

/// Both components are there.
pub fn installed(dir: &Path) -> bool {
    ytdlp_path(dir).is_file() && deno_path(dir).is_file()
}

/// The installed versions ("yt-dlp 2026.10.26, Deno 2.9.8"), asked of the
/// programs themselves.
pub fn versions(dir: &Path) -> Option<String> {
    let ask = |exe: PathBuf| -> Option<String> {
        let mut c = std::process::Command::new(exe);
        super::hide(&mut c);
        let out = c.arg("--version").stdin(std::process::Stdio::null()).output().ok()?;
        String::from_utf8_lossy(&out.stdout).lines().next().map(|l| l.trim().to_string())
    };
    let y = ask(ytdlp_path(dir))?;
    let d = ask(deno_path(dir))?;
    let d = d.split_whitespace().nth(1).unwrap_or(&d).to_string();
    Some(format!("yt-dlp {y}, Deno {d}"))
}

/// Which part is being fetched, and how far along.
pub trait Report {
    fn step(&mut self, what: &str, done: u64, total: u64);
}
impl<F: FnMut(&str, u64, u64)> Report for F {
    fn step(&mut self, what: &str, done: u64, total: u64) {
        self(what, done, total)
    }
}

/// Install or update both, to the latest releases.
pub fn install(dir: &Path, cancel: &AtomicBool, report: &mut impl Report) -> Result<(), String> {
    std::fs::create_dir_all(dir).map_err(|e| format!("{}: {e}", dir.display()))?;
    let sums = String::from_utf8_lossy(&get(YTDLP_SUMS, cancel, &mut |_, _| {}, 1 << 20)?).into_owned();
    let want = sum_for(&sums, "yt-dlp_win.zip").ok_or("yt-dlp's release lists no checksum for yt-dlp_win.zip")?;
    let ytzip = get(YTDLP_URL, cancel, &mut |d, t| report.step("yt-dlp", d, t), 200 << 20)?;
    check(&ytzip, &want, "the yt-dlp download")?;
    let files = crate::unzip::all(&ytzip)?;
    if !files.iter().any(|(n, _)| n == "yt-dlp.exe") {
        return Err("yt-dlp.exe is not in the download".into());
    }
    let sum = String::from_utf8_lossy(&get(DENO_SUM, cancel, &mut |_, _| {}, 1 << 20)?).into_owned();
    let want = first_sum(&sum).ok_or("Deno's release lists no checksum")?;
    let zip = get(DENO_URL, cancel, &mut |d, t| report.step("Deno", d, t), 400 << 20)?;
    check(&zip, &want, "the Deno download")?;
    let deno = crate::unzip::member(&zip, "deno.exe")?;
    // The folder is written aside and swapped in whole.
    let (new, folder) = (dir.join("yt-dlp.part"), dir.join("yt-dlp"));
    let _ = std::fs::remove_dir_all(&new);
    for (name, bytes) in &files {
        let p = new.join(name);
        if let Some(d) = p.parent() {
            std::fs::create_dir_all(d).map_err(|e| format!("{}: {e}", d.display()))?;
        }
        std::fs::write(&p, bytes).map_err(|e| format!("{}: {e}", p.display()))?;
    }
    let _ = std::fs::remove_dir_all(&folder);
    std::fs::rename(&new, &folder).map_err(|e| format!("{}: {e}", folder.display()))?;
    let _ = std::fs::remove_file(dir.join("yt-dlp.exe"));
    put(&deno_path(dir), &deno)?;
    let _ = std::fs::write(dir.join("NOTICE.txt"), NOTICE);
    Ok(())
}

pub const NOTICE: &str = "yt-dlp (https://github.com/yt-dlp/yt-dlp), The Unlicense; its Windows build bundles Python and other components under their own licenses, listed in the yt-dlp repository. Deno (https://github.com/denoland/deno), MIT License, Copyright the Deno authors. Downloaded by the user from the projects' GitHub releases; not part of Moosik.";

fn sha256(b: &[u8]) -> String {
    ring::digest::digest(&ring::digest::SHA256, b).as_ref().iter().map(|x| format!("{x:02x}")).collect()
}

fn check(bytes: &[u8], want: &str, what: &str) -> Result<(), String> {
    if sha256(bytes) == want.to_ascii_lowercase() { Ok(()) } else { Err(format!("{what} does not match its published checksum; nothing was installed.")) }
}

/// The hash on the `<hash>  <name>` line of a SHA256SUMS file.
fn sum_for(sums: &str, name: &str) -> Option<String> {
    sums.lines().find_map(|l| {
        let mut it = l.split_whitespace();
        let (h, n) = (it.next()?, it.next()?);
        (n.trim_start_matches('*') == name && is_hex64(h)).then(|| h.to_ascii_lowercase())
    })
}

/// The first 64-hex-digit word in a checksum file, whatever its layout
/// (Deno publishes PowerShell's `Get-FileHash` output).
fn first_sum(text: &str) -> Option<String> {
    text.split(|c: char| !c.is_ascii_hexdigit()).find(|w| is_hex64(w)).map(str::to_ascii_lowercase)
}

fn is_hex64(s: &str) -> bool {
    s.len() == 64 && s.bytes().all(|b| b.is_ascii_hexdigit())
}

fn put(path: &Path, bytes: &[u8]) -> Result<(), String> {
    let tmp = path.with_extension("part");
    std::fs::File::create(&tmp)
        .and_then(|mut f| f.write_all(bytes))
        .and_then(|_| std::fs::rename(&tmp, path))
        .map_err(|e| format!("{}: {e}", path.display()))
}

/// The agent for these downloads. What is limited is getting connected,
/// not how long the whole takes: a slow line needs minutes for Deno's
/// 45 MB. ureq's response limit would run on through the body ("timeout:
/// receive response"), so only half an hour bounds it all.
fn agent() -> ureq::Agent {
    ureq::Agent::config_builder()
        .timeout_connect(Some(std::time::Duration::from_secs(30)))
        .timeout_global(Some(std::time::Duration::from_secs(30 * 60)))
        .build()
        .into()
}

/// GET a URL into memory, at most `limit` bytes.
fn get(url: &str, cancel: &AtomicBool, progress: &mut dyn FnMut(u64, u64), limit: u64) -> Result<Vec<u8>, String> {
    let resp = agent().get(url).call().map_err(|e| format!("Downloading failed: {e}"))?;
    let total = resp.headers().get("content-length").and_then(|v| v.to_str().ok()).and_then(|v| v.parse::<u64>().ok()).unwrap_or(0);
    if total > limit {
        return Err("The download is larger than expected; nothing was installed.".into());
    }
    let mut body = resp.into_body().into_reader();
    let mut out = Vec::with_capacity(total as usize);
    let mut buf = vec![0u8; 1 << 16];
    loop {
        if cancel.load(Ordering::Relaxed) {
            return Err("Cancelled; nothing was installed.".into());
        }
        let n = body.read(&mut buf).map_err(|e| format!("Downloading failed: {e}"))?;
        if n == 0 {
            break;
        }
        out.extend_from_slice(&buf[..n]);
        if out.len() as u64 > limit {
            return Err("The download is larger than expected; nothing was installed.".into());
        }
        progress(out.len() as u64, total);
    }
    Ok(out)
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The real download (about 66 MB, network) into a temporary folder.
    #[test]
    #[ignore]
    fn install_for_real() {
        let dir = std::env::temp_dir().join(format!("moosik-ytinstall-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&dir);
        let mut steps = Vec::new();
        install(&dir, &AtomicBool::new(false), &mut |w: &str, d, t| {
            if steps.last() != Some(&w.to_string()) {
                steps.push(w.to_string());
            }
            let _ = (d, t);
        }).unwrap();
        assert!(installed(&dir));
        assert!(ytdlp_path(&dir).ends_with("yt-dlp/yt-dlp.exe"), "the one-folder build");
        assert_eq!(steps, ["yt-dlp", "Deno"]);
        let v = versions(&dir).unwrap();
        eprintln!("installed: {v}");
        assert!(v.starts_with("yt-dlp 20") && v.contains("Deno 2."));
        std::fs::remove_dir_all(&dir).unwrap();
    }

    /// A body that trickles in slower than ureq's response limit is cut off
    /// by it ("timeout: receive response", what a slow line saw updating);
    /// the downloads here set none and read it whole.
    #[test]
    fn a_slow_download_is_read_whole() {
        use std::io::Write;
        let server = std::net::TcpListener::bind("127.0.0.1:0").unwrap();
        let url = format!("http://{}/deno.zip", server.local_addr().unwrap());
        let serving = std::thread::spawn(move || {
            for _ in 0..2 {
                let (mut s, _) = server.accept().unwrap();
                let mut req = [0u8; 4096];
                let _ = s.read(&mut req);
                s.write_all(b"HTTP/1.1 200 OK\r\nContent-Length: 10\r\nConnection: close\r\n\r\n").unwrap();
                for b in 0..10u8 {
                    std::thread::sleep(std::time::Duration::from_millis(120));
                    if s.write_all(&[b]).is_err() {
                        break;
                    }
                }
            }
        });
        let tight: ureq::Agent = ureq::Agent::config_builder()
            .timeout_recv_response(Some(std::time::Duration::from_millis(500)))
            .build()
            .into();
        let cut = tight.get(&url).call().map_err(|e| e.to_string()).and_then(|r| r.into_body().read_to_vec().map_err(|e| e.to_string()));
        assert!(cut.as_ref().is_err_and(|e| e.contains("receive response")), "{cut:?}");
        let t = agent().config().timeouts();
        assert!(t.recv_response.is_none() && t.recv_body.is_none(), "nothing caps the body but the whole");
        let body = get(&url, &AtomicBool::new(false), &mut |_, _| {}, 100).unwrap();
        assert_eq!(body, (0..10).collect::<Vec<u8>>());
        serving.join().unwrap();
    }

    #[test]
    fn checksums_are_read_from_both_projects_layouts() {
        let sums = "aa  yt-dlp_linux\nf5389d3698ecde4ef06f4e6b1ce0294fe9ac1396228c8990b8d3416d6d177a7d  yt-dlp.exe\n";
        assert_eq!(sum_for(sums, "yt-dlp.exe").as_deref(), Some("f5389d3698ecde4ef06f4e6b1ce0294fe9ac1396228c8990b8d3416d6d177a7d"));
        assert_eq!(sum_for(sums, "yt-dlp_x86.exe"), None);
        let deno = "\r\nAlgorithm : SHA256\r\nHash      : CBA84B26E0DF5AE7F74FC66B86A95B866872A86292EBD84FA1E959927EDD93BF\r\nPath      : C:\\a\\deno.zip\r\n";
        assert_eq!(first_sum(deno).as_deref(), Some("cba84b26e0df5ae7f74fc66b86a95b866872a86292ebd84fa1e959927edd93bf"));
        assert!(check(b"abc", "BA7816BF8F01CFEA414140DE5DAE2223B00361A396177A9CB410FF61F20015AD", "x").is_ok());
        assert!(check(b"abd", "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad", "x").is_err());
    }
}
