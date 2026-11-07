//! YouTube and YouTube Music as a source (1.6.1), through yt-dlp.
//!
//! Nothing here talks to YouTube itself: yt-dlp does, with Deno as the
//! JavaScript runtime it needs for YouTube. Both are optional components the
//! user downloads (see `install`). Audio is kept exactly as delivered (Opus
//! or AAC, lossy) and played from a file: a played result is fetched into the
//! cache first, a downloaded one is copied to the music folder.
//!
//! Every yt-dlp run is a hidden child process (no console window), reads no
//! user config, and keeps its cache inside Moosik's folder.

pub mod install;
pub mod store;
pub mod stream;
pub mod window;

use std::io::{BufRead, BufReader, Read};
use std::path::{Path, PathBuf};
use std::process::{Command, Stdio};
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};

pub use store::Info;

/// Results asked for per search.
pub const RESULTS: usize = 25;

/// Songs taken from a playlist or album link.
pub const LIST_MAX: usize = 200;

/// One search result.
#[derive(Clone, Debug, PartialEq)]
pub struct Hit {
    pub id: String,
    pub title: String,
    /// The channel, where the search says (YouTube Music's song search does not).
    pub channel: Option<String>,
    pub duration: Option<f32>,
    /// From YouTube Music.
    pub music: bool,
}

pub fn watch_url(id: &str, music: bool) -> String {
    if music { format!("https://music.youtube.com/watch?v={id}") } else { format!("https://www.youtube.com/watch?v={id}") }
}

/// A yt-dlp command: hidden, no user config, cache in our folder, UTF-8 out,
/// Deno from beside it.
/// The browsers yt-dlp can borrow a YouTube sign-in from.
pub const BROWSERS: &[&str] = &["firefox", "edge", "chrome", "brave"];

/// Where yt-dlp finds the YouTube sign-in: one of `BROWSERS`, or a
/// cookies.txt file. None: signed out.
static SIGN_IN: std::sync::RwLock<Option<String>> = std::sync::RwLock::new(None);

pub fn set_sign_in(s: Option<String>) {
    if let Ok(mut g) = SIGN_IN.write() {
        *g = s;
    }
}

pub fn signed_in() -> bool {
    SIGN_IN.read().is_ok_and(|g| g.is_some())
}

/// yt-dlp's arguments for a sign-in setting.
fn sign_in_args(s: &str) -> [&str; 2] {
    if BROWSERS.contains(&s) { ["--cookies-from-browser", s] } else { ["--cookies", s] }
}

fn ytdlp(tools: &Path) -> Command {
    let mut c = Command::new(install::ytdlp_path(tools));
    hide(&mut c);
    if let Ok(g) = SIGN_IN.read()
        && let Some(s) = g.as_deref()
    {
        c.args(sign_in_args(s));
    }
    c.arg("--ignore-config")
        .arg("--encoding").arg("utf-8")
        .arg("--cache-dir").arg(tools.join("cache"))
        .arg("--js-runtimes").arg(format!("deno:{}", install::deno_path(tools).display()))
        .env("PYTHONUTF8", "1")
        .stdin(Stdio::null());
    c
}

/// Put a started yt-dlp in a job that ends with Moosik: its Windows build
/// runs a second process, which would otherwise outlive a closed Moosik.
fn bind_to_moosik(child: &std::process::Child) {
    #[cfg(windows)]
    {
        use std::os::windows::io::AsRawHandle;
        use std::sync::OnceLock;
        use windows_sys::Win32::System::JobObjects::{
            AssignProcessToJobObject, CreateJobObjectW, JobObjectExtendedLimitInformation, SetInformationJobObject,
            JOBOBJECT_EXTENDED_LIMIT_INFORMATION, JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE,
        };
        static JOB: OnceLock<usize> = OnceLock::new();
        // SAFETY: plain Win32 calls; the job handle is kept for the life of
        // the process and closed by Windows when Moosik exits.
        let job = *JOB.get_or_init(|| unsafe {
            let job = CreateJobObjectW(std::ptr::null(), std::ptr::null());
            if job.is_null() {
                return 0;
            }
            let mut info: JOBOBJECT_EXTENDED_LIMIT_INFORMATION = std::mem::zeroed();
            info.BasicLimitInformation.LimitFlags = JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE;
            SetInformationJobObject(job, JobObjectExtendedLimitInformation, &info as *const _ as *const _,
                std::mem::size_of::<JOBOBJECT_EXTENDED_LIMIT_INFORMATION>() as u32);
            job as usize
        });
        if job != 0 {
            // SAFETY: both handles are valid for this call.
            unsafe { AssignProcessToJobObject(job as _, child.as_raw_handle() as _) };
        }
    }
    #[cfg(not(windows))]
    let _ = child;
}

/// No console window for a child process (Moosik has none to lend it).
pub fn hide(c: &mut Command) {
    #[cfg(windows)]
    {
        use std::os::windows::process::CommandExt;
        const CREATE_NO_WINDOW: u32 = 0x0800_0000;
        c.creation_flags(CREATE_NO_WINDOW);
    }
    #[cfg(not(windows))]
    let _ = c;
}

/// The last line yt-dlp marked as an error, else the last line it wrote.
fn reason(stderr: &str) -> String {
    let lines: Vec<&str> = stderr.lines().map(str::trim).filter(|l| !l.is_empty()).collect();
    let line = lines.iter().rev().find(|l| l.starts_with("ERROR:")).or(lines.last()).copied().unwrap_or("yt-dlp stopped without saying why");
    let why: String = line.trim_start_matches("ERROR:").trim().chars().take(300).collect();
    // What an older yt-dlp says after YouTube has changed.
    const STALE: &[&str] = &["nsig", "Unable to extract", "HTTP Error 403", "Requested format is not available", "Signature extraction failed", "player response"];
    if STALE.iter().any(|s| stderr.contains(s)) {
        format!("{why} (YouTube may have changed: try Update)")
    } else if ["Sign in", "sign in", "members-only", "Join this channel", "Private video", "confirm your age", "playlist does not exist"].iter().any(|s| why.contains(s)) && !signed_in() {
        format!("{why} (sign in: the Sign in button on the YouTube page)")
    } else if why.contains("cookie") || why.contains("DPAPI") {
        format!("{why} (close the browser and try again, or use a cookies.txt file)")
    } else {
        why
    }
}

/// The ids of the results whose audio is in the cache.
pub fn cached_ids(dir: &Path) -> std::collections::HashSet<String> {
    let Ok(rd) = std::fs::read_dir(dir) else { return Default::default() };
    rd.flatten()
        .map(|e| e.path())
        .filter(|p| crate::audio_formats::is_importable(p))
        .filter_map(|p| p.file_stem().and_then(|s| s.to_str()).filter(|s| is_id(s)).map(str::to_string))
        .collect()
}

/// Search YouTube (or YouTube Music's songs). Live streams are left out:
/// they have no end to fetch.
pub fn search(tools: &Path, query: &str, music: bool) -> Result<Vec<Hit>, String> {
    let query = query.trim();
    if query.is_empty() {
        return Ok(Vec::new());
    }
    // A pasted link is that one video; yt-dlp names it.
    if let Some((id, music)) = parse_link(query) {
        let out = ytdlp(tools)
            .args(["--skip-download", "--no-warnings", "--no-playlist", "--print", "%(.{id,title,channel,duration})j"])
            .arg(watch_url(&id, music))
            .output()
            .map_err(|e| format!("could not run yt-dlp: {e}"))?;
        if !out.status.success() {
            return Err(reason(&String::from_utf8_lossy(&out.stderr)));
        }
        let v: serde_json::Value = serde_json::from_slice(&out.stdout).map_err(|e| format!("yt-dlp gave an unreadable answer: {e}"))?;
        let title = v["title"].as_str().unwrap_or(&id).to_string();
        return Ok(vec![Hit { id, title, channel: v["channel"].as_str().map(str::to_string), duration: v["duration"].as_f64().map(|d| d as f32), music }]);
    }
    // A playlist or album link is its songs, in order.
    let (target, music, max) = if let Some((url, from_music)) = parse_list(query) {
        (url, from_music, LIST_MAX)
    } else if music {
        (format!("https://music.youtube.com/search?q={}#songs", encode(query)), true, RESULTS)
    } else {
        (format!("ytsearch{RESULTS}:{query}"), false, RESULTS)
    };
    let out = ytdlp(tools)
        .args(["--flat-playlist", "-J", "--no-warnings", "--playlist-end", &max.to_string()])
        .arg(&target)
        .output()
        .map_err(|e| format!("could not run yt-dlp: {e}"))?;
    if !out.status.success() {
        return Err(reason(&String::from_utf8_lossy(&out.stderr)));
    }
    parse_search(&String::from_utf8_lossy(&out.stdout), music, max)
}

fn parse_search(json: &str, music: bool, max: usize) -> Result<Vec<Hit>, String> {
    let v: serde_json::Value = serde_json::from_str(json).map_err(|e| format!("yt-dlp gave an unreadable answer: {e}"))?;
    let entries = v["entries"].as_array().cloned().unwrap_or_default();
    Ok(entries
        .iter()
        .filter(|e| e["ie_key"].as_str() == Some("Youtube"))
        .filter(|e| !matches!(e["live_status"].as_str(), Some("is_live" | "is_upcoming")))
        .filter_map(|e| {
            let id = e["id"].as_str()?.to_string();
            let title = e["title"].as_str()?.to_string();
            let channel = e["channel"].as_str().or(e["uploader"].as_str()).map(str::to_string);
            Some(Hit { id, title, channel, duration: e["duration"].as_f64().map(|d| d as f32), music })
        })
        .take(max)
        .collect())
}

/// A YouTube playlist or a YouTube Music album link, as (what to give
/// yt-dlp, from Music). A watch link inside a playlist stays one video.
pub fn parse_list(s: &str) -> Option<(String, bool)> {
    let s = s.trim();
    let rest = s.strip_prefix("https://").or_else(|| s.strip_prefix("http://")).unwrap_or(s);
    let rest = rest.strip_prefix("www.").or_else(|| rest.strip_prefix("m.")).unwrap_or(rest);
    let music = rest.starts_with("music.youtube.com/");
    let path = rest.strip_prefix("music.").unwrap_or(rest).strip_prefix("youtube.com/")?;
    // LL, LM and WL (liked videos, liked music, watch later) are two letters.
    let safe = |id: &str| (2..=64).contains(&id.len()) && id.bytes().all(|b| b.is_ascii_alphanumeric() || b == b'-' || b == b'_');
    if let Some(q) = path.strip_prefix("playlist?") {
        let id = q.split(['&', '#']).find_map(|kv| kv.strip_prefix("list="))?;
        return safe(id).then(|| (format!("https://www.youtube.com/playlist?list={id}"), music));
    }
    let id = path.strip_prefix("browse/MPREb_")?.split(['?', '&', '#', '/']).next()?;
    safe(id).then(|| (format!("https://music.youtube.com/browse/MPREb_{id}"), true))
}

/// A YouTube or YouTube Music link to one video, as (id, from Music).
pub fn parse_link(s: &str) -> Option<(String, bool)> {
    let s = s.trim();
    let rest = s.strip_prefix("https://").or_else(|| s.strip_prefix("http://")).unwrap_or(s);
    let rest = rest.strip_prefix("www.").or_else(|| rest.strip_prefix("m.")).unwrap_or(rest);
    let music = rest.starts_with("music.youtube.com/");
    let id = if let Some(r) = rest.strip_prefix("youtu.be/") {
        r.split(['?', '&', '#', '/']).next()?
    } else if let Some(r) = rest.strip_prefix("youtube.com/shorts/").or_else(|| rest.strip_prefix("youtube.com/live/")) {
        r.split(['?', '&', '#', '/']).next()?
    } else if rest.starts_with("youtube.com/watch") || rest.starts_with("music.youtube.com/watch") {
        let q = rest.split_once('?')?.1;
        q.split(['&', '#']).find_map(|kv| kv.strip_prefix("v="))?
    } else {
        return None;
    };
    is_id(id).then(|| (id.to_string(), music))
}

/// Percent-encode a search query for a URL.
fn encode(s: &str) -> String {
    s.bytes()
        .map(|b| match b {
            b'A'..=b'Z' | b'a'..=b'z' | b'0'..=b'9' | b'-' | b'_' | b'.' | b'~' => (b as char).to_string(),
            b' ' => "+".to_string(),
            _ => format!("%{b:02X}"),
        })
        .collect()
}

/// Bytes fetched so far and the expected total (0 while unknown).
#[derive(Default)]
pub struct Progress {
    pub done: AtomicU64,
    pub total: AtomicU64,
}

/// The picture alone as VP9 WebM, which Windows decodes (see video.rs), at
/// most `height` and 30 frames a second where offered (half the decoding
/// for a backdrop), else at most `height`, else whatever VP9 there is.
fn video_format(height: u32) -> String {
    format!("bv[ext=webm][vcodec^=vp9][height<={height}][fps<=30]/bv[ext=webm][vcodec^=vp9][height<={height}]/bv[ext=webm][vcodec^=vp9]")
}

/// A file a fetch has begun or finished bringing: where it will be, and
/// whether it is the picture.
#[derive(Clone, Debug, PartialEq)]
pub enum Arrival {
    Started { file: PathBuf, video: bool, info: Info },
    Finished { file: PathBuf, video: bool },
}

/// What a fetch brought.
#[derive(Clone, Debug, PartialEq)]
pub struct Fetched {
    pub audio: Option<PathBuf>,
    pub video: Option<PathBuf>,
    pub info: Info,
}

/// Fetch a result into `dir`: its audio as delivered (`<id>.<ext>`), and
/// with `video` (a height) its picture (`<id>.video.webm`), one after the
/// other from one look at YouTube; with `audio` false, the picture alone.
/// Each file can be read while it arrives (see `stream`), and `arrived`
/// hears as each begins and ends; `progress` is the audio's and the
/// picture's. A picture Windows cannot play is left out. `cancel` stops it
/// (the child is killed).
pub fn fetch(tools: &Path, id: &str, music: bool, audio: bool, video: Option<u32>, dir: &Path, cancel: &AtomicBool,
             progress: [&Progress; 2], arrived: &mut dyn FnMut(Arrival)) -> Result<Fetched, String> {
    if !is_id(id) {
        return Err("not a YouTube video id".into());
    }
    std::fs::create_dir_all(dir).map_err(|e| format!("{}: {e}", dir.display()))?;
    match fetch_once(tools, id, music, audio, video, true, dir, cancel, progress, arrived) {
        // No picture Windows can play, or (less likely) nothing without
        // the HLS lists left out: the audio alone, from a full look.
        Err(e) if audio && e.contains("Requested format is not available") => {
            fetch_once(tools, id, music, true, None, false, dir, cancel, progress, arrived)
        }
        Err(e) if e.contains("Requested format is not available") => Err("no video Windows can play".into()),
        r => r,
    }
}

#[allow(clippy::too_many_arguments)]
fn fetch_once(tools: &Path, id: &str, music: bool, audio: bool, video: Option<u32>, skip_hls: bool, dir: &Path, cancel: &AtomicBool,
              progress: [&Progress; 2], arrived: &mut dyn FnMut(Arrival)) -> Result<Fetched, String> {
    let format = match (audio, video) {
        (true, Some(h)) => format!("bestaudio,{}", video_format(h)),
        (false, Some(h)) => video_format(h),
        _ => "bestaudio".to_string(),
    };
    let mut c = ytdlp(tools);
    // The HLS lists are another slow request, and nothing played here
    // comes from them.
    if skip_hls {
        c.args(["--extractor-args", "youtube:skip=hls"]);
    }
    let mut child = c
        .args(["-f", &format, "--no-playlist", "--newline", "--progress", "--no-warnings"])
        .arg("--progress-template")
        .arg("download:MOOSIK-PROGRESS %(progress.downloaded_bytes)s %(progress.total_bytes,progress.total_bytes_estimate)s")
        .arg("-o")
        .arg(dir.join("%(id)s%(height&.video|)s.%(ext)s"))
        .arg("--print")
        .arg("before_dl:MOOSIK-START %(ext)s %(filesize)s %(height)s %(.{id,title,track,artist,channel,album,duration})j")
        .arg("--print")
        .arg("after_move:MOOSIK-FILE %(filepath)s")
        .arg(watch_url(id, music))
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .spawn()
        .map_err(|e| format!("could not run yt-dlp: {e}"))?;
    bind_to_moosik(&child);
    let stderr = child.stderr.take();
    let err_thread = std::thread::spawn(move || {
        let mut s = String::new();
        if let Some(mut e) = stderr {
            let _ = e.read_to_string(&mut s);
        }
        s
    });
    let mut info = None;
    let mut got: [Option<PathBuf>; 2] = [None, None];
    // The file arriving now, and whether it is the picture.
    let mut now: Option<(PathBuf, bool)> = None;
    if let Some(out) = child.stdout.take() {
        for line in BufReader::new(out).lines() {
            if cancel.load(Ordering::Relaxed) {
                kill_tree(&mut child);
                if let Some((f, _)) = now {
                    stream::end(&f, false);
                }
                return Err("cancelled".into());
            }
            let Ok(line) = line else { break };
            if let Some(rest) = line.strip_prefix("MOOSIK-PROGRESS ") {
                let p = progress[now.as_ref().is_some_and(|n| n.1) as usize];
                let mut it = rest.split_whitespace().map(|n| n.parse::<f64>().unwrap_or(0.0) as u64);
                p.done.store(it.next().unwrap_or(0), Ordering::Relaxed);
                p.total.store(it.next().unwrap_or(0), Ordering::Relaxed);
            } else if let Some((ext, size, video, i)) = line.strip_prefix("MOOSIK-START ").and_then(|r| parse_start(r, music)) {
                let file = dir.join(format!("{id}{}.{ext}", if video { ".video" } else { "" }));
                // A size YouTube gives is the size the file will be; without
                // one, the file is played once it is all in.
                if let Some(total) = size {
                    stream::begin(&file, part_of(&file), total);
                }
                arrived(Arrival::Started { file: file.clone(), video, info: i.clone() });
                info = Some(i);
                now = Some((file, video));
            } else if let Some(rest) = line.strip_prefix("MOOSIK-FILE ") {
                let f = PathBuf::from(rest.trim());
                if let Some((file, video)) = now.take() {
                    stream::end(&file, true);
                    arrived(Arrival::Finished { file, video });
                    got[video as usize] = Some(f);
                }
            }
        }
    }
    let status = child.wait().map_err(|e| e.to_string())?;
    if let Some((f, _)) = now {
        stream::end(&f, false);
    }
    let stderr = err_thread.join().unwrap_or_default();
    let [a, v] = got;
    match (status.success(), info) {
        (true, Some(info)) if a.iter().chain(&v).all(|f| f.is_file() && f.starts_with(dir)) && (a.is_some() || v.is_some()) => {
            Ok(Fetched { audio: a, video: v, info })
        }
        (true, _) => Err("yt-dlp finished but did not say where the file went".into()),
        _ if stderr.contains("Requested format is not available") => Err("Requested format is not available".into()),
        _ => Err(reason(&stderr)),
    }
}

/// Where yt-dlp keeps a file while it arrives.
fn part_of(file: &Path) -> PathBuf {
    let mut p = file.as_os_str().to_owned();
    p.push(".part");
    PathBuf::from(p)
}

/// A MOOSIK-START line: the extension, the size where YouTube gives it,
/// whether it is the picture (it has a height), and what the video is.
fn parse_start(rest: &str, music: bool) -> Option<(String, Option<u64>, bool, Info)> {
    let mut it = rest.splitn(4, ' ');
    let ext = it.next()?.to_string();
    let size = it.next()?.parse().ok();
    let video = it.next()?.parse::<u32>().is_ok();
    let info = parse_info(it.next()?, music)?;
    (!ext.is_empty() && ext.bytes().all(|b| b.is_ascii_alphanumeric())).then_some((ext, size, video, info))
}

/// Stop a yt-dlp run and everything it started. The Windows yt-dlp.exe is
/// a one-file bundle that unpacks and runs a second process; killing only
/// the first leaves the second downloading.
fn kill_tree(child: &mut std::process::Child) {
    #[cfg(windows)]
    {
        let mut c = Command::new("taskkill");
        hide(&mut c);
        let _ = c.args(["/T", "/F", "/PID", &child.id().to_string()]).stdout(Stdio::null()).stderr(Stdio::null()).status();
    }
    let _ = child.kill();
    let _ = child.wait();
}

/// A result's thumbnail (320×180 JPEG) from YouTube's image server.
pub fn thumbnail(id: &str) -> Result<Vec<u8>, String> {
    ytimg(id, "mqdefault")
}

/// One of a video's pictures from YouTube's image server, by name.
fn ytimg(id: &str, name: &str) -> Result<Vec<u8>, String> {
    if !is_id(id) {
        return Err("not a YouTube video id".into());
    }
    // One agent for them all: the pictures after the first reuse its
    // connection, which on a slow line is most of the wait.
    static AGENT: std::sync::OnceLock<ureq::Agent> = std::sync::OnceLock::new();
    let agent = AGENT.get_or_init(|| ureq::Agent::config_builder().timeout_global(Some(std::time::Duration::from_secs(45))).build().into());
    let resp = agent.get(&format!("https://i.ytimg.com/vi/{id}/{name}.jpg")).call().map_err(|e| e.to_string())?;
    let mut out = Vec::new();
    resp.into_body().into_reader().take(2 << 20).read_to_end(&mut out).map_err(|e| e.to_string())?;
    Ok(out)
}

/// Where a song's cover is kept, beside it (the art cache reads it for a
/// file with no picture of its own).
pub fn cover_path(song: &Path) -> PathBuf {
    song.with_extension("cover.jpg")
}

/// A video's largest picture as a square cover, JPEG: the 1280×720 one
/// where there is one, else the 480×360 one (a 16:9 picture between black
/// bands), its middle square either way.
pub fn cover(id: &str) -> Result<Vec<u8>, String> {
    let (bytes, banded) = match ytimg(id, "maxresdefault") {
        Ok(b) => (b, false),
        Err(_) => (ytimg(id, "hqdefault")?, true),
    };
    let img = image::load_from_memory(&bytes).map_err(|e| e.to_string())?.to_rgb8();
    let mut out = Vec::new();
    square(&img, banded).write_to(&mut std::io::Cursor::new(&mut out), image::ImageFormat::Jpeg).map_err(|e| e.to_string())?;
    Ok(out)
}

/// The middle square of a picture; `banded` first drops the black bands
/// above and below a 16:9 picture in a 4:3 frame.
fn square(img: &image::RgbImage, banded: bool) -> image::RgbImage {
    let (w, mut h) = img.dimensions();
    let mut top = 0;
    if banded {
        let inner = w * 9 / 16;
        top = h.saturating_sub(inner) / 2;
        h = inner.min(h);
    }
    let side = w.min(h);
    image::imageops::crop_imm(img, (w - side) / 2, top + (h - side) / 2, side, side).to_image()
}

/// A YouTube video id: eleven characters of `[A-Za-z0-9_-]`.
pub fn is_id(id: &str) -> bool {
    id.len() == 11 && id.bytes().all(|b| b.is_ascii_alphanumeric() || b == b'-' || b == b'_')
}

fn parse_info(json: &str, music: bool) -> Option<Info> {
    let v: serde_json::Value = serde_json::from_str(json).ok()?;
    let s = |k: &str| v[k].as_str().filter(|s| !s.is_empty()).map(str::to_string);
    let id = s("id")?;
    // A video's own title says who and what, wrapped in decoration, unless
    // YouTube knows the track itself.
    let (title, artist) = match (s("track"), s("artist")) {
        (Some(t), a) => (t, a.or(s("channel"))),
        (None, a) => {
            let (t, by) = tidy(&s("title")?, s("channel").as_deref());
            (t, a.or(by))
        }
    };
    Some(Info {
        title,
        artist: artist.unwrap_or_else(|| "Unknown Artist".into()),
        album: s("album").unwrap_or_else(|| if music { "YouTube Music".into() } else { "YouTube".into() }),
        duration: v["duration"].as_f64().map(|d| d as f32),
        url: watch_url(&id, music),
        id,
        video_asked: None,
    })
}

/// A video title as song and artist: "Artist - Song (Official Video)" or
/// "Artist「Song」MV" without the decoration, else the title as it is with
/// the channel (less " - Topic" and "VEVO") as the artist.
pub fn tidy(title: &str, channel: Option<&str>) -> (String, Option<String>) {
    const NOISE: &[&str] = &["official", "video", "mv", "m/v", "audio", "lyric", "visualizer", "visualiser", "hd", "4k", "hq",
        "remaster", "music", "公式", "歌詞", "full", "ver."];
    let channel = channel.map(|c| c.trim_end_matches(" - Topic").trim_end_matches("VEVO").trim().to_string()).filter(|c| !c.is_empty());
    // Drop bracketed parts that are only decoration.
    let mut t = title.to_string();
    for (open, close) in [('(', ')'), ('[', ']'), ('【', '】'), ('（', '）')] {
        while let Some(a) = t.find(open) {
            let Some(b) = t[a..].find(close).map(|b| a + b) else { break };
            let inner = t[a + open.len_utf8()..b].to_lowercase();
            if NOISE.iter().any(|w| inner.contains(w)) {
                t.replace_range(a..b + close.len_utf8(), " ");
            } else {
                break;
            }
        }
    }
    // Japanese quotes hold the song: 「Song」 or 『Song』, the artist before.
    for (open, close) in [('「', '」'), ('『', '』')] {
        if let (Some(a), Some(b)) = (t.find(open), t.rfind(close))
            && a < b
        {
            let song = t[a + open.len_utf8()..b].trim().to_string();
            let before = t[..a].trim().trim_end_matches(['-', '–', '—', '/']).trim().to_string();
            if !song.is_empty() {
                return (song, Some(before).filter(|s| !s.is_empty()).or(channel));
            }
        }
    }
    // Trailing decoration outside brackets: "… Official Music Video".
    let lower = t.to_lowercase();
    for tail in [" official music video", " official video", " music video", " official audio", " lyric video", " mv"] {
        if let Some(i) = lower.rfind(tail).filter(|&i| i + tail.len() == lower.trim_end().len()) {
            t.truncate(i);
            break;
        }
    }
    let t = t.split_whitespace().collect::<Vec<_>>().join(" ");
    for sep in [" - ", " – ", " — ", " / "] {
        if let Some((a, b)) = t.split_once(sep) {
            let (a, b) = (a.trim(), b.trim());
            if !a.is_empty() && !b.is_empty() {
                return (b.to_string(), Some(a.to_string()));
            }
        }
    }
    (if t.is_empty() { title.trim().to_string() } else { t }, channel)
}

/// Where played results are kept, and how much of it may be used.
pub fn cache_dir(moosik: &Path) -> PathBuf {
    moosik.join("cache").join("youtube")
}
pub const CACHE_BYTES: u64 = 1 << 30;

/// A result fetched before and still in the cache: its audio file (not
/// its video, cover, lyrics or a download in progress).
pub fn cached(dir: &Path, id: &str) -> Option<PathBuf> {
    if !is_id(id) {
        return None;
    }
    std::fs::read_dir(dir).ok()?.flatten().map(|e| e.path()).find(|p| {
        p.file_stem().and_then(|s| s.to_str()) == Some(id) && crate::audio_formats::is_importable(p)
    })
}

/// Delete the oldest cached audio until the folder fits `CACHE_BYTES`,
/// never a file in `keep` (on the playlist).
pub fn prune(dir: &Path, keep: &[PathBuf]) {
    let Ok(rd) = std::fs::read_dir(dir) else { return };
    let mut files: Vec<(std::time::SystemTime, u64, PathBuf)> = rd
        .flatten()
        .filter_map(|e| {
            let m = e.metadata().ok()?;
            m.is_file().then(|| (m.modified().unwrap_or(std::time::UNIX_EPOCH), m.len(), e.path()))
        })
        .collect();
    let mut used: u64 = files.iter().map(|f| f.1).sum();
    files.sort();
    for (_, len, path) in files {
        if used <= CACHE_BYTES {
            break;
        }
        if keep.contains(&path) {
            continue;
        }
        if std::fs::remove_file(&path).is_ok() {
            used -= len;
        }
    }
}

/// Copy a fetched file into `folder` as "Artist - Title [id].ext".
pub fn keep(file: &Path, info: &Info, folder: &Path) -> Result<PathBuf, String> {
    std::fs::create_dir_all(folder).map_err(|e| format!("{}: {e}", folder.display()))?;
    let ext = file.extension().and_then(|e| e.to_str()).unwrap_or("audio");
    let name = format!("{} - {} [{}].{ext}", clean(&info.artist), clean(&info.title), info.id);
    let dest = folder.join(name);
    let tmp = dest.with_extension("part");
    std::fs::copy(file, &tmp).and_then(|_| std::fs::rename(&tmp, &dest)).map_err(|e| format!("{}: {e}", dest.display()))?;
    // The cover goes along, if it is in already.
    let _ = std::fs::copy(cover_path(file), cover_path(&dest));
    Ok(dest)
}

/// A title made safe for a Windows file name.
fn clean(s: &str) -> String {
    let t: String = s.chars().map(|c| if "<>:\"/\\|?*".contains(c) || c.is_control() { '_' } else { c }).collect();
    let t = t.trim().trim_end_matches('.').to_string();
    t.chars().take(80).collect()
}

/// Where downloads go: the user's Music folder, in a Moosik folder.
pub fn music_dir() -> PathBuf {
    std::env::var("USERPROFILE")
        .or_else(|_| std::env::var("HOME"))
        .map(PathBuf::from)
        .unwrap_or_else(|_| PathBuf::from("."))
        .join("Music")
        .join("Moosik")
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn search_results_keep_videos_and_drop_live_streams_and_channels() {
        let json = r#"{"entries":[
            {"ie_key":"Youtube","id":"aaaaaaaaaaa","title":"Song","channel":"Band","duration":213.0},
            {"ie_key":"Youtube","id":"bbbbbbbbbbb","title":"Radio","live_status":"is_live"},
            {"ie_key":"YoutubeTab","id":"UCxyz","title":"A channel"},
            {"ie_key":"Youtube","id":"ccccccccccc","title":"Bare"}]}"#;
        let hits = parse_search(json, false, RESULTS).unwrap();
        assert_eq!(hits.len(), 2);
        assert_eq!(hits[0], Hit { id: "aaaaaaaaaaa".into(), title: "Song".into(), channel: Some("Band".into()), duration: Some(213.0), music: false });
        assert_eq!(hits[1].channel, None);
        assert!(parse_search("not json", false, RESULTS).is_err());
    }

    #[test]
    fn a_music_track_is_named_by_its_artist_and_track() {
        let i = parse_info(r#"{"id":"m9SMT5ipbxk","title":"MV title","track":"Idol","artist":"YOASOBI","channel":"YOASOBI","album":"Idol","duration":213}"#, true).unwrap();
        assert_eq!((i.title.as_str(), i.artist.as_str(), i.album.as_str()), ("Idol", "YOASOBI", "Idol"));
        assert_eq!(i.url, "https://music.youtube.com/watch?v=m9SMT5ipbxk");
        let v = parse_info(r#"{"id":"jNQXAC9IVRw","title":"Me at the zoo","channel":"jawed"}"#, false).unwrap();
        assert_eq!((v.title.as_str(), v.artist.as_str(), v.album.as_str()), ("Me at the zoo", "jawed", "YouTube"));
    }

    #[test]
    fn links_to_one_video_are_recognised() {
        let id = "dQw4w9WgXcQ".to_string();
        for (l, music) in [
            ("https://www.youtube.com/watch?v=dQw4w9WgXcQ", false),
            ("youtube.com/watch?list=PL1&v=dQw4w9WgXcQ&t=42", false),
            ("https://youtu.be/dQw4w9WgXcQ?si=abc", false),
            ("https://m.youtube.com/watch?v=dQw4w9WgXcQ", false),
            ("https://www.youtube.com/shorts/dQw4w9WgXcQ", false),
            ("https://music.youtube.com/watch?v=dQw4w9WgXcQ&feature=share", true),
        ] {
            assert_eq!(parse_link(l), Some((id.clone(), music)), "{l}");
        }
        for l in ["never gonna give you up", "https://example.com/watch?v=dQw4w9WgXcQ", "https://youtu.be/short", "https://www.youtube.com/playlist?list=PL1"] {
            assert_eq!(parse_link(l), None, "{l}");
        }
    }

    #[test]
    fn playlist_and_album_links_are_their_songs() {
        let list = "PLFgquLnL59alCl_2TQvOiD5Vgm1hCaGSI";
        assert_eq!(parse_list(&format!("https://www.youtube.com/playlist?list={list}&si=x")), Some((format!("https://www.youtube.com/playlist?list={list}"), false)));
        assert_eq!(parse_list(&format!("music.youtube.com/playlist?list={list}")), Some((format!("https://www.youtube.com/playlist?list={list}"), true)));
        assert_eq!(parse_list("https://music.youtube.com/playlist?list=LM"), Some(("https://www.youtube.com/playlist?list=LM".into(), true)));
        assert_eq!(parse_list("https://music.youtube.com/browse/MPREb_tQfaWH32ovE?si=1"), Some(("https://music.youtube.com/browse/MPREb_tQfaWH32ovE".into(), true)));
        for l in ["https://www.youtube.com/watch?v=dQw4w9WgXcQ&list=PL1234567890", "https://www.youtube.com/playlist?list=--exec=calc;x", "https://example.com/playlist?list=PL1234567890", "abbey road"] {
            assert_eq!(parse_list(l), None, "{l}");
        }
    }

    #[test]
    fn ids_names_and_queries_are_made_safe() {
        assert!(is_id("jNQXAC9IVRw"));
        assert!(!is_id("jNQXAC9IVR"));
        assert!(!is_id("--exec=calc"));
        assert_eq!(clean("AC/DC: Back?"), "AC_DC_ Back_");
        assert_eq!(encode("a b&c"), "a+b%26c");
        assert_eq!(encode("夜"), "%E5%A4%9C");
        assert_eq!(reason("[youtube] x\nERROR: [youtube] abc: Video unavailable\n"), "[youtube] abc: Video unavailable");
        assert_eq!(sign_in_args("firefox"), ["--cookies-from-browser", "firefox"]);
        assert_eq!(sign_in_args(r"C:\x\cookies.txt"), ["--cookies", r"C:\x\cookies.txt"]);
        assert!(reason("ERROR: [youtube] abc: Sign in to confirm your age").ends_with("(sign in: the Sign in button on the YouTube page)"));
    }

    /// Against the real thing (network). MOOSIK_YT_TOOLS = a folder holding
    /// yt-dlp.exe and deno.exe.
    #[test]
    #[ignore]
    fn search_and_fetch_for_real() {
        let tools = PathBuf::from(std::env::var("MOOSIK_YT_TOOLS").expect("set MOOSIK_YT_TOOLS"));
        let t = std::time::Instant::now();
        let hits = search(&tools, "me at the zoo", false).unwrap();
        eprintln!("search: {} hits in {:?}", hits.len(), t.elapsed());
        assert!(!hits.is_empty());
        let music = search(&tools, "yoasobi", true).unwrap();
        eprintln!("music search: {} hits, first {:?}", music.len(), music.first());
        assert!(music.len() <= RESULTS && !music.is_empty());
        let one = search(&tools, "https://youtu.be/jNQXAC9IVRw", false).unwrap();
        assert_eq!(one.len(), 1);
        assert_eq!(one[0].title, "Me at the zoo");
        let jpeg = thumbnail("jNQXAC9IVRw").unwrap();
        assert_eq!(&jpeg[..2], &[0xff, 0xd8], "a JPEG");
        let c = image::load_from_memory(&cover("dQw4w9WgXcQ").unwrap()).unwrap();
        assert!(c.width() == c.height() && c.width() >= 360, "{}×{}", c.width(), c.height());
        let dir = std::env::temp_dir().join(format!("moosik-ytfetch-{}", std::process::id()));
        let p = [Progress::default(), Progress::default()];
        let t = std::time::Instant::now();
        let mut news = Vec::new();
        let got = fetch(&tools, "jNQXAC9IVRw", false, true, Some(720), &dir, &AtomicBool::new(false), [&p[0], &p[1]], &mut |a| news.push(a)).unwrap();
        let (file, video, info) = (got.audio.unwrap(), got.video.unwrap(), got.info);
        eprintln!("fetch: {} and {} in {:?}, {:?}, {} of {} bytes", file.display(), video.display(), t.elapsed(), info, p[0].done.load(Ordering::Relaxed), p[0].total.load(Ordering::Relaxed));
        assert!(file.is_file() && info.title == "Me at the zoo");
        assert!(p.iter().all(|p| p.done.load(Ordering::Relaxed) > 0), "progress was reported for both");
        // Each file was told of as it began and as it ended, the audio first.
        assert!(matches!(&news[..], [Arrival::Started { video: false, .. }, Arrival::Finished { video: false, .. },
            Arrival::Started { video: true, .. }, Arrival::Finished { video: true, .. }]), "{news:?}");
        assert!(!stream::downloading(&file) && !stream::downloading(&video));
        // The music video, beside the audio, and Windows can show it.
        assert_eq!(crate::video::beside(&file), Some(video.clone()));
        let v = crate::video::Video::open(video);
        v.follow(std::time::Duration::from_secs(5));
        let shown = (0..300).find_map(|_| {
            std::thread::sleep(std::time::Duration::from_millis(10));
            v.take()
        });
        assert!(shown.is_some_and(|f| f.image.size[0] > 0), "a picture, failure: {:?}", v.failed());
        drop(v);
        let cancel = AtomicBool::new(true);
        let none = Progress::default();
        assert_eq!(fetch(&tools, "jNQXAC9IVRw", false, true, None, &dir.join("c"), &cancel, [&none, &none], &mut |_| {}).err().as_deref(), Some("cancelled"));
        std::fs::remove_dir_all(&dir).unwrap();
    }

    /// A music video shows while it is still arriving (network; as above).
    #[test]
    #[ignore]
    fn a_video_shows_while_it_arrives() {
        let tools = PathBuf::from(std::env::var("MOOSIK_YT_TOOLS").expect("set MOOSIK_YT_TOOLS"));
        let dir = std::env::temp_dir().join(format!("moosik-ytstream-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&dir);
        let (tx, rx) = std::sync::mpsc::channel();
        let (t, d) = (tools.clone(), dir.clone());
        let p = std::sync::Arc::new([Progress::default(), Progress::default()]);
        let pp = p.clone();
        let start = std::time::Instant::now();
        let fetching = std::thread::spawn(move || {
            fetch(&t, "dQw4w9WgXcQ", false, false, Some(720), &d, &AtomicBool::new(false), [&pp[0], &pp[1]], &mut |a| { let _ = tx.send(a); })
        });
        let Ok(Arrival::Started { file, video: true, .. }) = rx.recv() else { panic!("no video began") };
        let began = start.elapsed();
        assert!(stream::downloading(&file) && crate::video::beside(&dir.join("dQw4w9WgXcQ.webm")) == Some(file.clone()));
        let v = crate::video::Video::open(file.clone());
        v.follow(std::time::Duration::from_secs(3));
        let shown = (0..3000).find_map(|_| {
            std::thread::sleep(std::time::Duration::from_millis(10));
            v.take()
        });
        let (done, total) = (p[1].done.load(Ordering::Relaxed), p[1].total.load(Ordering::Relaxed));
        eprintln!("began at {began:?}; first picture at {:?} with {done} of {total} bytes in; gpu {}", start.elapsed(), v.on_gpu());
        assert!(shown.is_some_and(|f| f.image.size[0] > 0), "a picture, failure: {:?}", v.failed());
        assert!(stream::downloading(&file), "shown before it was all in");
        drop(v);
        let got = fetching.join().unwrap().unwrap();
        eprintln!("all in at {:?}", start.elapsed());
        assert_eq!(got.video.as_deref(), Some(file.as_path()));
        std::fs::remove_dir_all(&dir).unwrap();
    }

    #[test]
    fn video_titles_become_song_and_artist() {
        let t = |title: &str, ch: Option<&str>| {
            let (s, a) = tidy(title, ch);
            (s, a.unwrap_or_default())
        };
        let w = |s: &str, a: &str| (s.to_string(), a.to_string());
        assert_eq!(t("Rick Astley - Never Gonna Give You Up (Official Music Video)", Some("Rick Astley")), w("Never Gonna Give You Up", "Rick Astley"));
        assert_eq!(t("YOASOBI「アイドル」 Official Music Video", Some("Ayase / YOASOBI")), w("アイドル", "YOASOBI"));
        assert_eq!(t("【MV】米津玄師 - Lemon", None), w("Lemon", "米津玄師"));
        assert_eq!(t("Lemon", Some("Kenshi Yonezu - Topic")), w("Lemon", "Kenshi Yonezu"));
        assert_eq!(t("Never Gonna Give You Up [4K Remaster]", Some("RickAstleyVEVO")), w("Never Gonna Give You Up", "RickAstley"));
        assert_eq!(t("Me at the zoo", Some("jawed")), w("Me at the zoo", "jawed"));
        // A bracket that is part of the name stays.
        assert_eq!(t("Song (feat. Someone)", Some("Artist")), w("Song (feat. Someone)", "Artist"));
        assert_eq!(t("Artist — Song (Live at Budokan)", None), w("Song (Live at Budokan)", "Artist"));
        assert_eq!(t("ずっと真夜中でいいのに。『秒針を噛む』MV", None), w("秒針を噛む", "ずっと真夜中でいいのに。"));
    }

    #[test]
    fn covers_are_the_middle_square_without_the_bands() {
        // 32×24: a 32×18 picture between bands of black three rows tall,
        // red on the left half and blue on the right.
        let img = image::RgbImage::from_fn(32, 24, |x, y| {
            if !(3..21).contains(&y) { image::Rgb([0, 0, 0]) } else if x < 16 { image::Rgb([255, 0, 0]) } else { image::Rgb([0, 0, 255]) }
        });
        let sq = square(&img, true);
        assert_eq!(sq.dimensions(), (18, 18));
        assert!(sq.pixels().all(|p| p.0 != [0, 0, 0]), "no band left");
        assert_eq!(sq.get_pixel(0, 0).0, [255, 0, 0]);
        assert_eq!(sq.get_pixel(17, 17).0, [0, 0, 255]);
        assert_eq!(square(&image::RgbImage::new(1280, 720), false).dimensions(), (720, 720));
    }

    #[test]
    fn a_cached_result_is_its_audio_only() {
        let dir = std::env::temp_dir().join(format!("moosik-ytcached-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(&dir).unwrap();
        let id = "jNQXAC9IVRw";
        for f in ["jNQXAC9IVRw.video.webm", "jNQXAC9IVRw.cover.jpg", "jNQXAC9IVRw.webm.part", "jNQXAC9IVRw.lrc", "dQw4w9WgXcQ.webm"] {
            std::fs::write(dir.join(f), b"x").unwrap();
        }
        assert_eq!(cached(&dir, id), None, "only its video, cover, lyrics and a part");
        std::fs::write(dir.join("jNQXAC9IVRw.webm"), b"x").unwrap();
        assert_eq!(cached(&dir, id), Some(dir.join("jNQXAC9IVRw.webm")));
        assert_eq!(cached(&dir, "../../etc"), None);
        assert_eq!(cached_ids(&dir), ["jNQXAC9IVRw".to_string(), "dQw4w9WgXcQ".to_string()].into_iter().collect::<std::collections::HashSet<_>>());
        std::fs::remove_dir_all(&dir).unwrap();
    }

    #[test]
    fn the_cache_drops_oldest_first_and_spares_the_playlist() {
        let dir = std::env::temp_dir().join(format!("moosik-ytcache-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(&dir).unwrap();
        // Three files of 0.4 GiB each would be slow; use sparse sizes via set_len.
        let mk = |n: &str, secs: u64| {
            let p = dir.join(n);
            let f = std::fs::File::create(&p).unwrap();
            f.set_len(CACHE_BYTES / 2).unwrap();
            f.set_modified(std::time::UNIX_EPOCH + std::time::Duration::from_secs(secs)).unwrap();
            p
        };
        let (a, b, c) = (mk("a.webm", 100), mk("b.webm", 200), mk("c.webm", 300));
        prune(&dir, &[a.clone()]);
        assert!(a.exists(), "on the playlist");
        assert!(!b.exists(), "oldest not kept");
        assert!(c.exists());
        std::fs::remove_dir_all(&dir).unwrap();
    }
}
