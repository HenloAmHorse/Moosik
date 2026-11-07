//! Sign in to YouTube in a browser window of Moosik's own: a Chromium-family
//! browser (Chrome, Brave, Edge, Vivaldi, Chromium) with a profile kept in
//! the tools folder, apart from the one you
//! browse with. Once the account's cookies appear there, they are asked of
//! the browser itself (its DevTools port, on this computer only) and
//! written as a cookies.txt for yt-dlp. Asking the browser is what gets
//! past Chrome's and Edge's newer cookie encryption, which yt-dlp cannot
//! read from the files.

use std::io::{Read, Write};
use std::net::TcpStream;
use std::path::{Path, PathBuf};
use std::process::{Child, Command};
use std::sync::atomic::{AtomicBool, Ordering};
use std::time::{Duration, Instant};

const SIGN_IN_URL: &str = "https://accounts.google.com/ServiceLogin?service=youtube&continue=https%3A%2F%2Fwww.youtube.com%2F";

/// A Chromium-family browser here (they all speak DevTools): Chrome,
/// Brave, Edge, Vivaldi or Chromium, wherever this system keeps them.
pub fn browser() -> Option<PathBuf> {
    let env = |k: &str| std::env::var(k).map(PathBuf::from).ok();
    let mut places: Vec<Option<PathBuf>> = Vec::new();
    if cfg!(windows) {
        for (root, rel) in [
            ("ProgramFiles", r"Google\Chrome\Application\chrome.exe"),
            ("LOCALAPPDATA", r"Google\Chrome\Application\chrome.exe"),
            ("ProgramFiles", r"BraveSoftware\Brave-Browser\Application\brave.exe"),
            ("LOCALAPPDATA", r"BraveSoftware\Brave-Browser\Application\brave.exe"),
            ("ProgramFiles(x86)", r"Microsoft\Edge\Application\msedge.exe"),
            ("ProgramFiles", r"Microsoft\Edge\Application\msedge.exe"),
            ("LOCALAPPDATA", r"Vivaldi\Application\vivaldi.exe"),
            ("LOCALAPPDATA", r"Chromium\Application\chrome.exe"),
        ] {
            places.push(env(root).map(|p| p.join(rel)));
        }
    } else if cfg!(target_os = "macos") {
        for app in ["Google Chrome", "Brave Browser", "Microsoft Edge", "Vivaldi", "Chromium"] {
            places.push(Some(PathBuf::from(format!("/Applications/{app}.app/Contents/MacOS/{app}"))));
        }
    } else {
        let path = std::env::var_os("PATH").unwrap_or_default();
        for name in ["google-chrome", "google-chrome-stable", "chromium", "chromium-browser", "brave-browser", "microsoft-edge", "vivaldi"] {
            places.extend(std::env::split_paths(&path).map(|d| Some(d.join(name))));
        }
    }
    places.into_iter().flatten().find(|p| p.is_file())
}

/// Moosik's own browser profile, and the cookies.txt made from it.
pub fn profile(tools: &Path) -> PathBuf {
    tools.join("browser")
}
pub fn cookies_file(tools: &Path) -> PathBuf {
    tools.join("cookies.txt")
}

/// Start the browser on `url` with Moosik's profile. Headless, it has a
/// DevTools port of its choosing; the window has none, since Google refuses
/// to sign in a browser that can be driven ("may not be secure"), and
/// Chrome opens no port for a window anyway.
fn launch(exe: &Path, profile: &Path, url: &str, headless: bool) -> std::io::Result<Child> {
    let _ = std::fs::remove_file(profile.join("DevToolsActivePort"));
    let mut c = Command::new(exe);
    c.arg(format!("--user-data-dir={}", profile.display()))
        .args(["--no-first-run", "--no-default-browser-check", "--disable-sync"]);
    if headless {
        c.args(["--headless=new", "--remote-debugging-port=0"]);
        super::hide(&mut c);
        c.arg(url);
    } else {
        c.arg(format!("--app={url}")).arg("--window-size=520,720");
    }
    c.spawn()
}

/// The browser's DevTools address, once it has written it down.
fn devtools(profile: &Path, wait: Duration) -> Option<(u16, String)> {
    let until = Instant::now() + wait;
    while Instant::now() < until {
        if let Ok(s) = std::fs::read_to_string(profile.join("DevToolsActivePort"))
            && let Some((port, path)) = s.split_once('\n')
            && let Ok(port) = port.trim().parse()
        {
            return Some((port, path.trim().to_string()));
        }
        std::thread::sleep(Duration::from_millis(200));
    }
    None
}

/// One DevTools command over a WebSocket of the smallest kind: one masked
/// text frame out, frames read until the answer to it.
fn call(port: u16, path: &str, method: &str) -> Result<serde_json::Value, String> {
    let e = |e: std::io::Error| e.to_string();
    let mut s = TcpStream::connect(("127.0.0.1", port)).map_err(e)?;
    s.set_read_timeout(Some(Duration::from_secs(10))).map_err(e)?;
    write!(s, "GET {path} HTTP/1.1\r\nHost: 127.0.0.1:{port}\r\nUpgrade: websocket\r\nConnection: Upgrade\r\nSec-WebSocket-Key: dGhlIHNhbXBsZSBub25jZQ==\r\nSec-WebSocket-Version: 13\r\n\r\n").map_err(e)?;
    let mut head = Vec::new();
    let mut b = [0u8; 1];
    while !head.ends_with(b"\r\n\r\n") {
        s.read_exact(&mut b).map_err(e)?;
        head.push(b[0]);
    }
    if !head.starts_with(b"HTTP/1.1 101") {
        return Err("the browser would not talk".into());
    }
    s.write_all(&frame(&serde_json::json!({ "id": 1, "method": method }).to_string())).map_err(e)?;
    loop {
        let text = read_message(&mut s).map_err(e)?;
        let v: serde_json::Value = serde_json::from_str(&text).map_err(|e| e.to_string())?;
        if v["id"] == 1 {
            return if v["error"].is_null() { Ok(v["result"].clone()) } else { Err(v["error"]["message"].as_str().unwrap_or("refused").to_string()) };
        }
    }
}

/// A client's text frame: final, masked (with a zero mask, which the
/// standard allows, so the payload goes as it is).
fn frame(text: &str) -> Vec<u8> {
    let n = text.len();
    let mut f = vec![0x81];
    match n {
        0..=125 => f.push(0x80 | n as u8),
        126..=65535 => {
            f.push(0x80 | 126);
            f.extend((n as u16).to_be_bytes());
        }
        _ => {
            f.push(0x80 | 127);
            f.extend((n as u64).to_be_bytes());
        }
    }
    f.extend([0, 0, 0, 0]);
    f.extend(text.as_bytes());
    f
}

/// The next whole text message, its frames joined; other frames skipped.
fn read_message(s: &mut impl Read) -> std::io::Result<String> {
    let mut msg = Vec::new();
    loop {
        let mut h = [0u8; 2];
        s.read_exact(&mut h)?;
        let (fin, op) = (h[0] & 0x80 != 0, h[0] & 0x0f);
        let mut n = (h[1] & 0x7f) as u64;
        if n == 126 {
            let mut x = [0u8; 2];
            s.read_exact(&mut x)?;
            n = u16::from_be_bytes(x) as u64;
        } else if n == 127 {
            let mut x = [0u8; 8];
            s.read_exact(&mut x)?;
            n = u64::from_be_bytes(x);
        }
        let mask = if h[1] & 0x80 != 0 {
            let mut m = [0u8; 4];
            s.read_exact(&mut m)?;
            Some(m)
        } else {
            None
        };
        let mut p = vec![0u8; n as usize];
        s.read_exact(&mut p)?;
        if let Some(m) = mask {
            p.iter_mut().enumerate().for_each(|(i, b)| *b ^= m[i % 4]);
        }
        if matches!(op, 0 | 1) {
            msg.extend(p);
            if fin {
                return Ok(String::from_utf8_lossy(&msg).into_owned());
            }
        }
    }
}

/// The browser's YouTube and Google cookies.
fn cookies(port: u16, path: &str) -> Result<Vec<serde_json::Value>, String> {
    let r = call(port, path, "Storage.getCookies")?;
    Ok(r["cookies"].as_array().cloned().unwrap_or_default()
        .into_iter()
        .filter(|c| c["domain"].as_str().is_some_and(|d| d.ends_with("youtube.com") || d.ends_with("google.com")))
        .collect())
}

/// Signed in: YouTube has set the account's cookies.
fn signed_in(cookies: &[serde_json::Value]) -> bool {
    cookies.iter().any(|c| c["domain"].as_str().is_some_and(|d| d.ends_with("youtube.com")) && matches!(c["name"].as_str(), Some("LOGIN_INFO" | "SAPISID")))
}

/// Cookies in the Netscape cookies.txt form yt-dlp reads.
fn netscape(cookies: &[serde_json::Value]) -> String {
    let mut out = String::from("# Netscape HTTP Cookie File\n# Made by Moosik from its own browser window; holds your YouTube sign-in.\n\n");
    for c in cookies {
        let s = |k: &str| c[k].as_str().unwrap_or("");
        let domain = s("domain");
        let tf = |b: bool| if b { "TRUE" } else { "FALSE" };
        let expires = c["expires"].as_f64().filter(|e| *e > 0.0).map_or(0, |e| e as i64);
        out += &format!("{}{domain}\t{}\t{}\t{}\t{expires}\t{}\t{}\n",
            if c["httpOnly"].as_bool() == Some(true) { "#HttpOnly_" } else { "" },
            tf(domain.starts_with('.')), s("path"), tf(c["secure"].as_bool() == Some(true)), s("name"), s("value"));
    }
    out
}

/// Open the sign-in window and wait for the account (the browser notes it
/// in the profile's Preferences), then close the window; or wait for it to
/// be closed by hand. Then read the account's cookies out of the profile
/// with the browser headless and write them to `cookies_file`. `cancel`
/// closes the window and gives up.
pub fn sign_in(tools: &Path, cancel: &AtomicBool) -> Result<PathBuf, String> {
    let exe = browser().ok_or("No Chrome, Brave, Edge, Vivaldi or Chromium found")?;
    let profile = profile(tools);
    std::fs::create_dir_all(&profile).map_err(|e| e.to_string())?;
    let prefs = profile.join("Default").join("Preferences");
    let mut child = launch(&exe, &profile, SIGN_IN_URL, false).map_err(|e| format!("could not open the browser: {e}"))?;
    let mut seen: Option<Instant> = None;
    // ponytail: no deadline, Cancel ends it
    while child.try_wait().ok().flatten().is_none() {
        if cancel.load(Ordering::Relaxed) {
            let _ = child.kill();
            let _ = child.wait();
            return Err("cancelled".into());
        }
        if seen.is_none() && google_account(&prefs) {
            seen = Some(Instant::now());
        }
        // A few seconds more for Google to send the window on to YouTube.
        if seen.is_some_and(|t| t.elapsed() > Duration::from_secs(3)) {
            close(&mut child);
            break;
        }
        std::thread::sleep(Duration::from_millis(500));
    }
    let c = read_cookies(&exe, &profile)?;
    if !signed_in(&c) {
        return Err("The window was closed before signing in".into());
    }
    let f = cookies_file(tools);
    std::fs::write(&f, netscape(&c)).map_err(|e| e.to_string())?;
    Ok(f)
}

/// A Google account signed in to on the web in this profile.
fn google_account(prefs: &Path) -> bool {
    std::fs::read(prefs)
        .ok()
        .and_then(|b| serde_json::from_slice::<serde_json::Value>(&b).ok())
        .is_some_and(|v| v["account_info"].as_array().is_some_and(|a| !a.is_empty()))
}

/// Close the window as its close button would, so the browser keeps the
/// cookies it was given; forced after ten seconds.
fn close(child: &mut Child) {
    let id = child.id().to_string();
    #[cfg(windows)]
    let mut c = Command::new("taskkill");
    #[cfg(windows)]
    c.args(["/PID", &id]);
    #[cfg(not(windows))]
    let mut c = Command::new("kill");
    #[cfg(not(windows))]
    c.args(["-TERM", &id]);
    super::hide(&mut c);
    let _ = c.status();
    let until = Instant::now() + Duration::from_secs(10);
    while child.try_wait().ok().flatten().is_none() && Instant::now() < until {
        std::thread::sleep(Duration::from_millis(200));
    }
    let _ = child.kill();
    let _ = child.wait();
}

/// The profile's YouTube and Google cookies, from the browser run headless
/// on the sign-in page: signed in to Google, it goes on to YouTube, which
/// fills in YouTube's cookies if the window was closed before they came.
fn read_cookies(exe: &Path, profile: &Path) -> Result<Vec<serde_json::Value>, String> {
    let mut child = launch(exe, profile, SIGN_IN_URL, true).map_err(|e| format!("could not start the browser: {e}"))?;
    let got = devtools(profile, Duration::from_secs(20))
        .ok_or_else(|| "the browser did not start".to_string())
        .and_then(|(port, path)| {
            let until = Instant::now() + Duration::from_secs(15);
            let mut c = cookies(port, &path);
            while c.as_ref().is_ok_and(|c| !signed_in(c)) && Instant::now() < until {
                std::thread::sleep(Duration::from_millis(500));
                c = cookies(port, &path);
            }
            let _ = call(port, &path, "Browser.close");
            c
        });
    if got.is_err() {
        let _ = child.kill();
    }
    let _ = child.wait();
    got
}

/// Sign out: the cookies.txt and the window's profile go.
pub fn sign_out(tools: &Path) {
    let _ = std::fs::remove_file(cookies_file(tools));
    let _ = std::fs::remove_dir_all(profile(tools));
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn frames_go_out_masked_and_come_back_whole() {
        let f = frame("hi");
        assert_eq!(f, [0x81, 0x82, 0, 0, 0, 0, b'h', b'i']);
        let long = "x".repeat(300);
        assert_eq!(&frame(&long)[..4], &[0x81, 0x80 | 126, 1, 44]);
        // Two frames of one message (text, then continuation), a ping between.
        let mut wire = vec![0x01, 3, b'a', b'b', b'c', 0x89, 0, 0x80, 2, b'd', b'e'];
        let mut r = &wire[..];
        assert_eq!(read_message(&mut r).unwrap(), "abcde");
        wire.clear();
    }

    #[test]
    fn an_account_in_preferences_is_a_google_sign_in() {
        let d = std::env::temp_dir().join(format!("moosik-prefs-{}", std::process::id()));
        std::fs::create_dir_all(&d).unwrap();
        let p = d.join("Preferences");
        assert!(!google_account(&p), "no file yet");
        std::fs::write(&p, r#"{"account_info":[],"signin":{}}"#).unwrap();
        assert!(!google_account(&p));
        std::fs::write(&p, r#"{"account_info":[{"account_id":"1","email":"a@b"}]}"#).unwrap();
        assert!(google_account(&p));
        let _ = std::fs::remove_dir_all(&d);
    }

    #[test]
    fn cookies_become_a_cookies_txt_and_say_when_signed_in() {
        let c: Vec<serde_json::Value> = serde_json::from_str(r#"[
            {"name":"YSC","value":"a","domain":".youtube.com","path":"/","expires":-1,"httpOnly":true,"secure":true},
            {"name":"SAPISID","value":"b","domain":".youtube.com","path":"/","expires":1893456000.5,"httpOnly":false,"secure":true}]"#).unwrap();
        let t = netscape(&c);
        assert!(t.starts_with("# Netscape HTTP Cookie File"));
        assert!(t.contains("#HttpOnly_.youtube.com\tTRUE\t/\tTRUE\t0\tYSC\ta\n"), "{t}");
        assert!(t.contains(".youtube.com\tTRUE\t/\tTRUE\t1893456000\tSAPISID\tb\n"), "{t}");
        assert!(signed_in(&c));
        assert!(!signed_in(&c[..1]));
    }

    /// Against a real browser, headless (no window): a fresh profile visits
    /// YouTube and its cookies are read back over DevTools.
    #[test]
    #[ignore]
    fn a_real_browser_hands_over_its_cookies() {
        let exe = browser().expect("a Chromium-family browser");
        let profile = std::env::temp_dir().join(format!("moosik-cdp-{}", std::process::id()));
        let mut child = launch(&exe, &profile, "https://www.youtube.com/", true).unwrap();
        let (port, path) = devtools(&profile, Duration::from_secs(20)).expect("DevTools port");
        let mut got = Vec::new();
        for _ in 0..20 {
            got = cookies(port, &path).unwrap();
            if got.iter().any(|c| c["domain"].as_str().is_some_and(|d| d.ends_with("youtube.com"))) {
                break;
            }
            std::thread::sleep(Duration::from_millis(500));
        }
        let _ = call(port, &path, "Browser.close");
        let _ = child.wait();
        // Started again on the same profile, as after the sign-in window.
        let again = read_cookies(&exe, &profile).unwrap();
        let _ = std::fs::remove_dir_all(&profile);
        let kept = |c: &serde_json::Value| c["expires"].as_f64().is_some_and(|e| e > 0.0) && c["domain"].as_str().is_some_and(|d| d.ends_with("youtube.com"));
        for c in got.iter().filter(|c| kept(c)) {
            assert!(again.iter().any(|a| a["name"] == c["name"]), "the profile kept {}", c["name"]);
        }
        eprintln!("{} cookies: {:?}", got.len(), got.iter().map(|c| c["name"].as_str().unwrap_or("")).collect::<Vec<_>>());
        assert!(!got.is_empty(), "YouTube set cookies and the browser gave them");
        assert!(!signed_in(&got), "a fresh profile is not signed in");
        assert!(netscape(&got).lines().count() > 3);
        if let Ok(out) = std::env::var("MOOSIK_COOKIES_OUT") {
            std::fs::write(out, netscape(&got)).unwrap();
        }
    }
}
