//! Library → Sources → YouTube (1.6.1): the setup card for the optional
//! components, search, and each result's Play / Queue / Download. All the
//! work runs on threads; this only starts it, shows it and collects it.

use super::*;
use crate::youtube::{self, Hit, Info, Progress};
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Mutex};

/// What happens when a fetched result arrives.
#[derive(Clone, Copy, PartialEq, Debug)]
pub(crate) enum Then {
    Play,
    Queue,
    Download,
}

/// What a fetch's thread has to say.
enum Msg {
    Arrival(youtube::Arrival),
    Done(Result<youtube::Fetched, String>),
}

pub(super) struct Fetch {
    hit: Hit,
    then: Then,
    /// The video comes too, after the audio.
    with_video: bool,
    /// This is the video alone, for the song at this path.
    video_for: Option<PathBuf>,
    /// The audio's and the video's.
    progress: [Arc<Progress>; 2],
    /// The video's size asked for.
    height: u32,
    cancel: Arc<AtomicBool>,
    rx: Chan<Msg>,
    /// The audio, once it has begun to arrive, and what it is.
    audio: Option<(PathBuf, Info)>,
    /// It is on the playlist; it is all in.
    listed: bool,
    audio_in: bool,
}

/// How much of a song must be in before it plays while the rest arrives:
/// about 45 s of YouTube's Opus, which a slow line still outruns.
const START_BYTES: u64 = 768 << 10;

#[derive(Default)]
pub(super) struct Yt {
    /// Library shows YouTube in place of the playlist.
    pub(super) open: bool,
    music: bool,
    query: String,
    searching: Option<Chan<Result<Vec<Hit>, String>>>,
    /// Sign in: the window open (closed by setting the flag), and why it
    /// last failed.
    signing: Option<(Arc<AtomicBool>, Chan<Result<PathBuf, String>>)>,
    sign_failed: Option<String>,
    hits: Vec<Hit>,
    searched: Option<String>,
    error: Option<String>,
    /// Which part, bytes so far, total; cancel; outcome.
    install: Option<(Arc<Mutex<(String, u64, u64)>>, Arc<AtomicBool>, Chan<Result<(), String>>)>,
    installed: Option<bool>,
    versions: Option<Chan<Option<String>>>,
    version_line: Option<String>,
    fetches: Vec<Fetch>,
    /// Queue all: the songs still to add, fetched one at a time so they
    /// arrive in order.
    waiting: std::collections::VecDeque<Hit>,
    /// Result thumbnails by id (None: could not be had), and the thread
    /// bringing the current search's in.
    thumbs: HashMap<String, Option<egui::TextureHandle>>,
    thumbs_rx: Option<Chan<(String, Option<egui::ColorImage>)>>,
    /// The cache's size, and when it was measured.
    cache_size: Option<(u64, Instant)>,
    /// When a lo-fi search started the rain.
    rain_at: Option<Instant>,
    /// The current song's video and its latest picture, and the song it
    /// was looked for (None: look again).
    pub(super) video: Option<(crate::video::Video, Option<egui::TextureHandle>)>,
    /// When its first picture came, for the fade in.
    pub(super) video_since: Option<Instant>,
    /// Its picture inside any black bars, and what is shown this frame.
    video_content: [f32; 4],
    pub(super) shown: Option<crate::video::Shown>,
    video_checked: Option<Option<PathBuf>>,
    /// A YouTube song's cover being fetched, and the songs already tried.
    cover_job: Option<(PathBuf, Chan<Result<(), String>>)>,
    cover_tried: std::collections::HashSet<PathBuf>,
    /// Where the current song came from, if YouTube.
    pub(super) current: Option<Info>,
    /// Cached songs kept with Save this session.
    saved: std::collections::HashSet<PathBuf>,
    /// Results whose audio is cached (they play at once), as of when.
    cached: Option<(std::collections::HashSet<String>, Instant)>,
}

/// A lo-fi search, however it is spelt.
fn is_lofi(q: &str) -> bool {
    let q = q.to_lowercase();
    ["lofi", "lo-fi", "lo fi", "ローファイ"].iter().any(|w| q.contains(w))
}

/// How many recent searches are kept.
const RECENT: usize = 8;

/// The music video sizes to choose from: the tallest picture fetched, its
/// name, and what it costs (Rick Astley's 3:33 video: 17, 29, 144 and 342
/// MB from 720p up).
const QUALITIES: [(u32, &str, &str); 5] = [
    (480, "480p", "Smallest download, lightest to show"),
    (720, "720p", "The default: about 5 MB a minute"),
    (1080, "1080p", "Sharp in a window; about twice the download of 720p"),
    (1440, "1440p (2K)", "Sharp full screen on a 1440p screen; about 5 times the download of 1080p. Plays while it downloads on about 6 Mbit/s or more"),
    (2160, "2160p (4K)", "The sharpest; about 12 times the download of 1080p, shown at your screen's size. Plays while it downloads on about 15 Mbit/s or more; slower, it catches up as it arrives"),
];

/// A quality's name, or the height as it is.
fn quality_name(h: u32) -> String {
    QUALITIES.iter().find(|q| q.0 == h).map_or_else(|| format!("{h}p"), |q| q.1.to_string())
}

/// The video every internet user is legally obliged to know.
const RICK: &str = "dQw4w9WgXcQ";

fn fmt_secs(s: f32) -> String {
    let s = s.max(0.0) as u64;
    if s >= 3600 { format!("{}:{:02}:{:02}", s / 3600, s / 60 % 60, s % 60) } else { format!("{}:{:02}", s / 60, s % 60) }
}

impl MoosikApp {
    fn yt_tools(&self) -> PathBuf {
        youtube::install::dir(&moosik_dir())
    }

    pub(super) fn yt_downloads(&self) -> PathBuf {
        self.appearance.youtube_dir.clone().unwrap_or_else(youtube::music_dir)
    }

    /// A song or video is still arriving into the cache.
    pub(super) fn youtube_fetching(&self) -> bool {
        !self.shell.yt.fetches.is_empty()
    }

    /// The YouTube part of the settings drawer.
    pub(super) fn youtube_settings_ui(&mut self, ui: &mut egui::Ui, tok: &Tokens) {
        let small = |t: String| RichText::new(t).size(11.5).color(tok.dim);
        ui.label(small(format!("Downloads go to {}", self.yt_downloads().display())));
        ui.horizontal_wrapped(|ui| {
            if ui.small_button("Change…").clicked()
                && let Some(d) = rfd::FileDialog::new().set_title("Folder for YouTube downloads").pick_folder()
            {
                self.appearance.youtube_dir = Some(d);
                save_appearance(&moosik_dir(), &self.appearance);
            }
            if self.appearance.youtube_dir.is_some() && ui.small_button(r"Back to Music\Moosik").clicked() {
                self.appearance.youtube_dir = None;
                save_appearance(&moosik_dir(), &self.appearance);
            }
        });
        ui.add_space(6.0);
        let cache = youtube::cache_dir(&moosik_dir());
        let size = match self.shell.yt.cache_size {
            Some((n, at)) if at.elapsed().as_secs_f32() < 5.0 => n,
            _ => {
                let n = std::fs::read_dir(&cache).map(|rd| rd.flatten().filter_map(|e| e.metadata().ok()).filter(|m| m.is_file()).map(|m| m.len()).sum()).unwrap_or(0);
                self.shell.yt.cache_size = Some((n, Instant::now()));
                n
            }
        };
        ui.horizontal_wrapped(|ui| {
            ui.label(small(format!("Played from YouTube: {:.0} MB kept (up to {} MB)", size as f64 / 1e6, youtube::CACHE_BYTES / 1_000_000)));
            if size > 0 && !self.youtube_fetching()
                && ui.small_button("Clear").on_hover_text("Delete played audio not on the playlist").clicked()
            {
                youtube::prune(&cache, &self.youtube_keep(), 0);
                self.shell.yt.cache_size = None;
            }
        });
        ui.add_space(6.0);
        // Signed in, what needs an account plays too: age-restricted and
        // members-only videos, private playlists, your likes.
        ui.label(small("Sign in (for age-restricted, members-only and private videos, and your own playlists)".into()));
        ui.horizontal_wrapped(|ui| {
            let mut pick: Option<Option<String>> = None;
            if ui.selectable_label(self.appearance.youtube_sign_in.is_none(), "Signed out").clicked() {
                pick = Some(None);
            }
            for (b, name) in youtube::BROWSERS.iter().zip(["Firefox", "Edge", "Chrome", "Brave"]) {
                let on = self.appearance.youtube_sign_in.as_deref() == Some(*b);
                if ui.selectable_label(on, name).on_hover_text(format!("Use the YouTube account you are signed in to in {name}")).clicked() {
                    pick = Some(Some(b.to_string()));
                }
            }
            let file = self.appearance.youtube_sign_in.as_deref().filter(|s| !youtube::BROWSERS.contains(s));
            if ui.selectable_label(file.is_some(), "cookies.txt…")
                .on_hover_text(file.map_or("A cookies.txt exported from your browser; works whichever browser you use".to_string(), str::to_string))
                .clicked()
                && let Some(f) = rfd::FileDialog::new().set_title("cookies.txt with your YouTube sign-in").add_filter("cookies.txt", &["txt"]).pick_file()
            {
                pick = Some(Some(f.display().to_string()));
            }
            if let Some(p) = pick {
                self.appearance.youtube_sign_in = p.clone();
                youtube::set_sign_in(p);
                save_appearance(&moosik_dir(), &self.appearance);
            }
        });
        if self.appearance.youtube_sign_in.is_some() {
            ui.label(small("Your sign-in stays on this computer: yt-dlp reads it from there for each request and sends it only to YouTube. Chrome and Edge may need closing first; if they still fail, a cookies.txt works. YouTube can limit accounts used this way, so a second account is safer.".into()));
        }
        ui.add_space(6.0);
        ui.label(small("Music videos (also chosen beside Audio + video on the YouTube page)".into()));
        ui.horizontal_wrapped(|ui| {
            let mut changed = false;
            for (h, name, tip) in QUALITIES {
                if ui.selectable_label(self.appearance.video_height == h, name).on_hover_text(tip).clicked() {
                    self.appearance.video_height = h;
                    changed = true;
                }
            }
            ui.end_row();
            changed |= ui.checkbox(&mut self.appearance.video_listen, "Behind Listen").changed();
            changed |= ui.checkbox(&mut self.appearance.video_spectrum, "Behind the spectrum").changed();
            ui.end_row();
            let b = ui.add(egui::Slider::new(&mut self.appearance.video_brightness, 0.1..=1.0).text("Brightness").show_value(false));
            // Saved when let go, not on every step of a drag.
            changed |= b.drag_stopped() || (b.changed() && !b.dragged());
            ui.end_row();
            if ui.checkbox(&mut self.appearance.video_software, "Decode on the processor")
                .on_hover_text("Leave the graphics card out of it, if videos look wrong; it costs more of the processor")
                .changed()
            {
                changed = true;
                // Opened again the next frame, the other way.
                self.shell.yt.video = None;
                self.shell.yt.video_checked = None;
            }
            if changed {
                save_appearance(&moosik_dir(), &self.appearance);
            }
        });
        ui.add_space(6.0);
        let tools = self.yt_tools();
        if youtube::install::installed(&tools) {
            ui.horizontal_wrapped(|ui| {
                ui.label(small(self.shell.yt.version_line.clone().unwrap_or_else(|| "yt-dlp and Deno installed".into())));
                if self.shell.yt.install.is_none() && ui.small_button("Remove").on_hover_text("Delete yt-dlp and Deno from your Moosik folder").clicked() {
                    let _ = std::fs::remove_dir_all(&tools);
                    self.shell.yt.installed = None;
                    self.shell.yt.version_line = None;
                }
            });
        } else {
            ui.label(small("yt-dlp and Deno are not installed; Library → YouTube offers them.".into()));
        }
    }

    /// Moves YouTube work along; called every frame from the shell.
    pub(super) fn tick_youtube(&mut self, ctx: &egui::Context) {
        let tools = self.yt_tools();
        let yt = &mut self.shell.yt;
        // An idle window does not redraw, so work in flight asks to be looked
        // at again: a fetch can finish while another mode is on screen.
        if yt.searching.is_some() || yt.install.is_some() || yt.versions.is_some() || !yt.fetches.is_empty() {
            ctx.request_repaint_after(std::time::Duration::from_millis(150));
        }
        if yt.installed.is_none() {
            yt.installed = Some(youtube::install::installed(&tools));
        }
        if yt.installed == Some(true) && yt.version_line.is_none() && yt.versions.is_none() {
            let (tx, rx) = std::sync::mpsc::channel();
            let t = tools.clone();
            let _ = std::thread::Builder::new().name("yt-versions".into()).spawn(move || {
                let _ = tx.send(youtube::install::versions(&t));
            });
            yt.versions = Some(rx);
        }
        if let Some(rx) = &yt.versions
            && let Ok(v) = rx.try_recv()
        {
            yt.versions = None;
            yt.version_line = Some(v.unwrap_or_else(|| "versions unknown".into()));
        }
        if let Some((_, _, rx)) = &yt.install
            && let Ok(r) = rx.try_recv()
        {
            yt.install = None;
            yt.installed = None;
            yt.version_line = None;
            if let Err(e) = r {
                yt.error = Some(e);
            }
        }
        if let Some((_, rx)) = &yt.signing
            && let Ok(r) = rx.try_recv()
        {
            yt.signing = None;
            match r {
                Ok(f) => {
                    yt.sign_failed = None;
                    let f = f.display().to_string();
                    youtube::set_sign_in(Some(f.clone()));
                    self.appearance.youtube_sign_in = Some(f);
                    save_appearance(&moosik_dir(), &self.appearance);
                }
                Err(e) if e == "cancelled" => {}
                Err(e) => yt.sign_failed = Some(e),
            }
        }
        if self.shell.yt.signing.is_some() {
            ctx.request_repaint_after(std::time::Duration::from_millis(500));
        }
        let yt = &mut self.shell.yt;
        if let Some(rx) = &yt.searching
            && let Ok(r) = rx.try_recv()
        {
            yt.searching = None;
            match r {
                Ok(h) => {
                    // Thumbnails, one after another, for the ones not seen yet
                    // (about 230 KB each in memory, so a long session starts over).
                    if yt.thumbs.len() > 200 {
                        yt.thumbs.clear();
                    }
                    let ids: Vec<String> = h.iter().map(|h| h.id.clone()).filter(|id| !yt.thumbs.contains_key(id)).collect();
                    let (tx, rx) = std::sync::mpsc::channel();
                    let _ = std::thread::Builder::new().name("yt-thumbs".into()).spawn(move || {
                        for id in ids {
                            let img = youtube::thumbnail(&id).ok().and_then(|b| image::load_from_memory(&b).ok()).map(|i| {
                                let i = i.to_rgba8();
                                egui::ColorImage::from_rgba_unmultiplied([i.width() as usize, i.height() as usize], i.as_raw())
                            });
                            if tx.send((id, img)).is_err() {
                                break;
                            }
                        }
                    });
                    yt.thumbs_rx = Some(rx);
                    yt.hits = h;
                    yt.error = None;
                }
                Err(e) => yt.error = Some(e),
            }
        }
        // Each fetch's news, in the order it came.
        let mut i = 0;
        while i < self.shell.yt.fetches.len() {
            match self.shell.yt.fetches[i].rx.try_recv() {
                Ok(Msg::Arrival(a)) => self.youtube_arrival(i, a),
                Ok(Msg::Done(r)) => {
                    let f = self.shell.yt.fetches.remove(i);
                    self.youtube_done(ctx, f, r);
                }
                Err(std::sync::mpsc::TryRecvError::Empty) => {
                    self.youtube_list_when_ready(ctx, i);
                    i += 1;
                }
                Err(std::sync::mpsc::TryRecvError::Disconnected) => {
                    self.shell.yt.fetches.remove(i);
                }
            }
        }
        // Queue all: the next song once the last one's audio is in (cached
        // ones come at once, so several may go in one frame).
        while !self.shell.yt.fetches.iter().any(|f| f.then == Then::Queue && f.video_for.is_none() && !f.audio_in) {
            let Some(hit) = self.shell.yt.waiting.pop_front() else { break };
            self.youtube_fetch(&hit, Then::Queue);
        }
    }

    /// A video is in: put it beside its song, where playback looks for it.
    fn youtube_video_arrived(&mut self, file: &std::path::Path, song: &std::path::Path) {
        let dest = song.with_extension("video.webm");
        if file != dest {
            let moved = std::fs::rename(file, &dest).or_else(|_| std::fs::copy(file, &dest).map(|_| ()));
            if let Err(e) = moved {
                self.set_transient(format!("Could not keep the video: {e}"));
                return;
            }
        }
        self.shell.yt.video_checked = None;
    }

    /// A file of fetch `i` has begun or finished arriving.
    fn youtube_arrival(&mut self, i: usize, a: youtube::Arrival) {
        match a {
            youtube::Arrival::Started { file, video: false, info } => {
                // Named from the record while it arrives: the file has no tags.
                if let Err(e) = youtube::store::record(&youtube::store::file(&moosik_dir()), &file, &info) {
                    self.set_transient(format!("Could not remember where {} came from: {e}", info.title));
                }
                self.shell.yt.fetches[i].audio = Some((file, info));
            }
            youtube::Arrival::Started { file, video: true, .. } => {
                // The video for a song kept elsewhere reads as if beside it.
                if let Some(song) = &self.shell.yt.fetches[i].video_for {
                    let beside = song.with_extension("video.webm");
                    if beside != file {
                        youtube::stream::alias(&file, &beside);
                    }
                }
                // Looked for again: it shows while it arrives.
                self.shell.yt.video_checked = None;
            }
            youtube::Arrival::Finished { file, video: false } => {
                let f = &mut self.shell.yt.fetches[i];
                f.audio_in = true;
                if f.listed {
                    self.youtube_audio_in(&file);
                }
            }
            youtube::Arrival::Finished { file, video: true } => {
                let f = &self.shell.yt.fetches[i];
                let (video_for, asked) = (f.video_for.clone(), f.height);
                let song = video_for.clone().or_else(|| f.audio.as_ref().map(|(a, _)| a.clone()));
                if let Some(song) = video_for {
                    self.youtube_video_arrived(&file, &song);
                }
                if let Some(song) = song {
                    self.youtube_note_video(&song, asked);
                }
                self.shell.yt.video_checked = None;
            }
        }
    }

    /// A song played while it arrived is all in: its details from the file,
    /// and its analysis can start.
    fn youtube_audio_in(&mut self, file: &std::path::Path) {
        for t in self.playlist.iter_mut().filter(|t| t.path == file) {
            *t = crate::Track::load(file.to_path_buf());
        }
        self.spectrum_window.track_ready(file);
    }

    /// Fetch `i`'s song goes on the playlist (and plays, for Play) once
    /// enough of it is in to play on while the rest arrives.
    fn youtube_list_when_ready(&mut self, ctx: &egui::Context, i: usize) {
        let f = &self.shell.yt.fetches[i];
        if f.listed || f.then == Then::Download {
            return;
        }
        let Some((file, info)) = f.audio.clone() else { return };
        let (d, t) = (f.progress[0].done.load(Ordering::Relaxed), f.progress[0].total.load(Ordering::Relaxed));
        if !f.audio_in && !(youtube::stream::downloading(&file) && t > 0 && d >= t.min(START_BYTES)) {
            return;
        }
        let then = f.then;
        self.shell.yt.fetches[i].listed = true;
        self.youtube_list(ctx, file, info, then);
    }

    /// A fetched song onto the playlist, or played.
    fn youtube_list(&mut self, ctx: &egui::Context, file: PathBuf, info: Info, then: Then) {
        if info.id == RICK && then == Then::Play && self.toy("rickroll") {
            self.found("rickroll");
            self.celebrate(ctx, self.track_accent, false);
            self.set_transient("Never gonna give you up ♪");
        }
        self.youtube_arrived(file, info, then);
    }

    /// A fetch is over.
    fn youtube_done(&mut self, ctx: &egui::Context, f: Fetch, r: Result<youtube::Fetched, String>) {
        let got = match r {
            Err(e) if e == "cancelled" => return,
            Err(e) => {
                let what = if f.video_for.is_some() { " (video)" } else { "" };
                self.set_transient(format!("{}{what}: {e}", f.hit.title));
                return;
            }
            Ok(got) => got,
        };
        if let Some(audio) = got.audio.clone() {
            if f.then == Then::Download {
                if let Some(dest) = self.youtube_arrived(audio, got.info.clone(), Then::Download)
                    && let Some(v) = &got.video
                {
                    match std::fs::copy(v, dest.with_extension("video.webm")) {
                        Ok(_) => self.youtube_note_video(&dest, f.height),
                        Err(e) => self.set_transient(format!("Could not keep the video: {e}")),
                    }
                }
            } else if !f.listed {
                // All in before it was listed (or it could not be played
                // while it arrived).
                self.youtube_list(ctx, audio, got.info.clone(), f.then);
            }
        }
        if (f.with_video || f.video_for.is_some()) && got.video.is_none() {
            self.set_transient(format!("{}: no video Windows can play", got.info.title));
        }
    }

    /// Play the current song's video along with it, if it has one, and
    /// hand over its latest picture.
    pub(crate) fn video_frame(&mut self, ctx: &egui::Context) -> Option<crate::video::Shown> {
        self.shell.yt.shown = self.video_picture(ctx);
        self.shell.yt.shown
    }

    fn video_picture(&mut self, ctx: &egui::Context) -> Option<crate::video::Shown> {
        let song = self.current_index.and_then(|i| self.playlist.get(i)).map(|t| t.path.clone());
        if self.shell.yt.video_checked.as_ref() != Some(&song) {
            let file = song.as_deref().and_then(crate::video::beside);
            if self.shell.yt.video.as_ref().map(|(v, _)| &v.path) != file.as_ref() {
                let software = self.appearance.video_software;
                // No picture larger than the screen is worth handing over.
                let max = ctx.input(|i| i.viewport().monitor_size).map_or([u32::MAX; 2], |m| {
                    let px = m * ctx.pixels_per_point();
                    [px.x.round() as u32, px.y.round() as u32]
                });
                self.shell.yt.video = file.map(|f| (crate::video::Video::open_with(f, software, max), None));
                self.shell.yt.video_content = [0.0, 0.0, 1.0, 1.0];
            }
            if let Some(p) = &song {
                self.youtube_cover_for(p);
            }
            self.shell.yt.current = song.as_deref().and_then(|p| youtube::store::lookup(&youtube::store::file(&moosik_dir()), p));
            // Lyrics are looked for unasked (a setting, on at first): Listen
            // shows them and theater has them as subtitles. A song with no
            // artist to go by is left to the button.
            if self.appearance.lyrics_auto
                && let Some(t) = self.current_index.and_then(|i| self.playlist.get(i))
                && t.artist != "Unknown Artist"
            {
                let tref = lyrics::ui::TrackRef { path: &t.path, title: &t.title, artist: &t.artist, album: &t.album, duration: t.duration };
                self.lyrics_window.find_if_missing(tref);
            }
            self.shell.yt.video_checked = Some(song);
        }
        if let Some((p, rx)) = &self.shell.yt.cover_job
            && let Ok(r) = rx.try_recv()
        {
            let p = p.clone();
            self.shell.yt.cover_job = None;
            if r.is_ok() {
                self.art_cache.forget(&p);
            }
        }
        let pos = self.elapsed();
        let playing = self.play_state == PlayState::Playing;
        if let Some(e) = self.shell.yt.video.as_ref().and_then(|(v, _)| v.failed()) {
            // The VP9 decoder comes with Windows 11 ("VP9 Video Extensions").
            self.set_transient(format!("The music video cannot be shown: {e}"));
            self.shell.yt.video = None;
        }
        // Decoding only while the picture shows: the Library, say, hides it,
        // and a fifth of a core for nothing adds up.
        let v = &self.appearance.visual;
        let listen = v.modern && v.mode == UiMode::Listen && (self.shell.theater || self.appearance.video_listen);
        let seen = listen || (self.spectrum_window.open && self.appearance.video_spectrum);
        let (v, tex) = self.shell.yt.video.as_mut()?;
        if seen {
            v.follow(pos);
        }
        if let Some(f) = v.take() {
            self.shell.yt.video_content = f.content;
            let img = f.image;
            match tex {
                Some(t) => t.set(img, egui::TextureOptions::LINEAR),
                None => {
                    *tex = Some(ctx.load_texture("music-video", img, egui::TextureOptions::LINEAR));
                    self.shell.yt.video_since = Some(Instant::now());
                }
            }
        }
        if playing && seen {
            ctx.request_repaint_after(std::time::Duration::from_millis(33));
        }
        let [l, t, r, b] = self.shell.yt.video_content;
        let uv = Rect::from_min_max(Pos2::new(l, t), Pos2::new(r, b));
        tex.as_ref().map(|tx| crate::video::Shown { tex: tx.id(), size: tx.size_vec2() * uv.size(), uv })
    }

    /// A song from YouTube without a picture gets the video's, once.
    fn youtube_cover_for(&mut self, song: &std::path::Path) {
        let yt = &mut self.shell.yt;
        if yt.cover_job.is_some() || yt.cover_tried.contains(song) || youtube::cover_path(song).is_file() {
            return;
        }
        let Some(info) = youtube::store::lookup(&youtube::store::file(&moosik_dir()), song) else { return };
        yt.cover_tried.insert(song.to_path_buf());
        let (tx, rx) = std::sync::mpsc::channel();
        let dest = youtube::cover_path(song);
        let _ = std::thread::Builder::new().name("yt-cover".into()).spawn(move || {
            let r = youtube::cover(&info.id).and_then(|jpg| {
                let tmp = dest.with_extension("part");
                std::fs::write(&tmp, jpg).and_then(|_| std::fs::rename(&tmp, &dest)).map_err(|e| e.to_string())
            });
            let _ = tx.send(r);
        });
        yt.cover_job = Some((song.to_path_buf(), rx));
    }

    /// The current YouTube song's video: how far its fetch has got (0–1),
    /// or None when there is none in flight.
    pub(super) fn current_video_fetch(&self) -> Option<f32> {
        let song = self.current_index.and_then(|i| self.playlist.get(i)).map(|t| &t.path)?;
        let f = self.shell.yt.fetches.iter().find(|f| {
            f.video_for.as_ref() == Some(song) || (f.with_video && f.audio.as_ref().is_some_and(|(a, _)| a == song))
        })?;
        let (d, t) = (f.progress[1].done.load(Ordering::Relaxed), f.progress[1].total.load(Ordering::Relaxed));
        Some(if t > 0 { d as f32 / t as f32 } else { 0.0 })
    }

    /// The current song came from YouTube and is only in the cache.
    pub(super) fn can_save_current(&self) -> bool {
        let Some(song) = self.current_index.and_then(|i| self.playlist.get(i)).map(|t| &t.path) else { return false };
        self.shell.yt.current.is_some() && !self.shell.yt.saved.contains(song) && song.starts_with(youtube::cache_dir(&moosik_dir()))
    }

    /// Keep the current cached YouTube song in the downloads folder, with
    /// its video and cover, as Download would have.
    pub(super) fn save_current(&mut self) {
        let Some(song) = self.current_index.and_then(|i| self.playlist.get(i)).map(|t| t.path.clone()) else { return };
        let Some(info) = self.shell.yt.current.clone() else { return };
        let folder = self.yt_downloads();
        match youtube::keep(&song, &info, &folder) {
            Ok(dest) => {
                let _ = youtube::store::record(&youtube::store::file(&moosik_dir()), &dest, &info);
                if let Some(v) = crate::video::beside(&song)
                    && let Some(ext) = v.extension().and_then(|e| e.to_str())
                {
                    let _ = std::fs::copy(&v, dest.with_extension(format!("video.{ext}")));
                }
                self.shell.yt.saved.insert(song);
                self.set_transient(format!("Saved {} to {}", info.title, folder.display()));
            }
            Err(e) => self.set_transient(format!("Could not save {}: {e}", info.title)),
        }
    }

    /// Fetch the music video for the current song, which came from YouTube.
    pub(super) fn get_current_video(&mut self) {
        let Some(song) = self.current_index.and_then(|i| self.playlist.get(i)).map(|t| t.path.clone()) else { return };
        let Some(info) = self.shell.yt.current.clone() else { return };
        let hit = Hit { music: info.url.contains("music.youtube.com"), id: info.id, title: info.title, channel: Some(info.artist), duration: info.duration };
        if self.shell.yt.fetches.iter().any(|f| f.hit.id == hit.id) {
            return;
        }
        self.youtube_start(&hit, Then::Queue, false, Some(song));
    }

    /// The cached files to keep: the playlist's songs, their videos, covers
    /// and saved lyrics.
    pub(super) fn youtube_keep(&self) -> Vec<PathBuf> {
        self.playlist.iter().flat_map(|t| [t.path.clone(), t.path.with_extension("video.webm"), youtube::cover_path(&t.path), t.path.with_extension("lrc")]).collect()
    }

    /// Where the audio went (to play or keep), or None if it went nowhere.
    fn youtube_arrived(&mut self, file: PathBuf, info: Info, then: Then) -> Option<PathBuf> {
        let records = youtube::store::file(&moosik_dir());
        let (path, note) = match then {
            Then::Download => match youtube::keep(&file, &info, &self.yt_downloads()) {
                Ok(p) => (p, "Downloaded"),
                Err(e) => {
                    self.set_transient(format!("Could not save {}: {e}", info.title));
                    return None;
                }
            },
            _ => (file, ""),
        };
        if let Err(e) = youtube::store::record(&records, &path, &info) {
            self.set_transient(format!("Could not remember where {} came from: {e}", info.title));
        }
        self.youtube_cover_for(&path);
        match then {
            Then::Download => self.set_transient(format!("{note} {} to {}", info.title, self.yt_downloads().display())),
            Then::Queue | Then::Play => {
                let mut t = crate::Track::load(path.clone());
                // Still arriving, it has no length of its own yet.
                if t.duration.is_none() {
                    t.duration = info.duration.map(std::time::Duration::from_secs_f32);
                }
                self.playlist.push(t);
                if then == Then::Play {
                    self.play_index(self.playlist.len() - 1);
                }
            }
        }
        youtube::prune(&youtube::cache_dir(&moosik_dir()), &self.youtube_keep(), youtube::CACHE_BYTES);
        Some(path)
    }

    fn youtube_search(&mut self) {
        let yt = &mut self.shell.yt;
        let q = yt.query.trim().to_string();
        if q.is_empty() || yt.searching.is_some() {
            return;
        }
        // Remembered, newest first, links aside.
        if !q.contains("youtu") {
            let recent = &mut self.appearance.youtube_recent;
            recent.retain(|r| !r.eq_ignore_ascii_case(&q));
            recent.insert(0, q.clone());
            recent.truncate(RECENT);
            save_appearance(&moosik_dir(), &self.appearance);
        }
        let (tx, rx) = std::sync::mpsc::channel();
        let (tools, music) = (self.yt_tools(), self.shell.yt.music);
        let qq = q.clone();
        let _ = std::thread::Builder::new().name("yt-search".into()).spawn(move || {
            let _ = tx.send(youtube::search(&tools, &qq, music));
        });
        if is_lofi(&q) && self.toy("lofi_rain") {
            self.found("lofi_rain");
            self.shell.yt.rain_at = Some(Instant::now());
        }
        let yt = &mut self.shell.yt;
        yt.searching = Some(rx);
        yt.searched = Some(q);
        yt.error = None;
    }

    pub(crate) fn youtube_fetch(&mut self, hit: &Hit, then: Then) {
        if self.shell.yt.fetches.iter().any(|f| f.hit.id == hit.id) {
            return;
        }
        let with_video = self.appearance.youtube_video;
        // Played before and still cached: no need to ask YouTube again.
        let m = moosik_dir();
        if let Some(file) = youtube::cached(&youtube::cache_dir(&m), &hit.id)
            && let Some(info) = youtube::store::lookup(&youtube::store::file(&m), &file)
        {
            // A kept video smaller than now asked for makes way for a new
            // one (before the song plays and opens it).
            if with_video
                && !self.youtube_video_enough(&file)
                && let Some(v) = crate::video::beside(&file)
            {
                let _ = std::fs::remove_file(v);
            }
            if let Some(song) = self.youtube_arrived(file, info, then)
                && with_video
            {
                self.youtube_then_video(hit, then, song);
            }
            return;
        }
        self.youtube_start(hit, then, with_video, None);
    }

    /// The video for a song just in: from the cache if it is there, else
    /// fetched; nothing if the song has it already.
    fn youtube_then_video(&mut self, hit: &Hit, then: Then, song: PathBuf) {
        if self.youtube_video_enough(&song) {
            return;
        }
        // Smaller than asked for: a new one in its place.
        if let Some(v) = crate::video::beside(&song) {
            let _ = std::fs::remove_file(v);
        }
        let cached = youtube::cache_dir(&moosik_dir()).join(format!("{}.video.webm", hit.id));
        let want = self.appearance.video_height;
        if cached.is_file() && crate::video::webm_height(&cached).is_some_and(|h| h >= want) {
            // Copied: the cached song it belongs to may still want it.
            match std::fs::copy(&cached, song.with_extension("video.webm")) {
                Ok(_) => self.shell.yt.video_checked = None,
                Err(e) => self.set_transient(format!("Could not keep the video: {e}")),
            }
        } else {
            self.youtube_start(hit, then, false, Some(song));
        }
    }

    /// The song has its video, as tall as now asked for (or as YouTube had
    /// when it was asked for at least that), or one of the user's own.
    fn youtube_video_enough(&self, song: &std::path::Path) -> bool {
        let Some(v) = crate::video::beside(song) else { return false };
        let asked = youtube::store::lookup(&youtube::store::file(&moosik_dir()), song).and_then(|i| i.video_asked);
        asked.or_else(|| crate::video::webm_height(&v)).is_none_or(|h| h >= self.appearance.video_height)
    }

    /// Remember the size `song`'s video was asked for.
    fn youtube_note_video(&self, song: &std::path::Path, asked: u32) {
        let records = youtube::store::file(&moosik_dir());
        if let Some(mut i) = youtube::store::lookup(&records, song) {
            i.video_asked = Some(asked);
            let _ = youtube::store::record(&records, song, &i);
        }
    }

    /// Start fetching a result's audio (and with `with_video` its video,
    /// in the same run), or with `video_for` the video alone, for the song
    /// already at that path.
    fn youtube_start(&mut self, hit: &Hit, then: Then, with_video: bool, video_for: Option<PathBuf>) {
        let audio = video_for.is_none();
        let height = self.appearance.video_height;
        let video = (with_video || !audio).then_some(height);
        let (tx, rx) = std::sync::mpsc::channel();
        let progress = [Arc::new(Progress::default()), Arc::new(Progress::default())];
        let cancel = Arc::new(AtomicBool::new(false));
        let (tools, dir) = (self.yt_tools(), youtube::cache_dir(&moosik_dir()));
        let (p, c, h) = (progress.clone(), cancel.clone(), hit.clone());
        let _ = std::thread::Builder::new().name("yt-fetch".into()).spawn(move || {
            let news = tx.clone();
            let r = youtube::fetch(&tools, &h.id, h.music, audio, video, &dir, &c, [&p[0], &p[1]], &mut |a| {
                let _ = news.send(Msg::Arrival(a));
            });
            let _ = tx.send(Msg::Done(r));
        });
        self.shell.yt.fetches.push(Fetch { hit: hit.clone(), then, with_video, video_for, progress, height, cancel, rx, audio: None, listed: false, audio_in: false });
    }

    fn youtube_install(&mut self) {
        let (tx, rx) = std::sync::mpsc::channel();
        let state = Arc::new(Mutex::new((String::new(), 0u64, 0u64)));
        let cancel = Arc::new(AtomicBool::new(false));
        let (s, c, dir) = (state.clone(), cancel.clone(), self.yt_tools());
        let _ = std::thread::Builder::new().name("yt-install".into()).spawn(move || {
            let mut report = |what: &str, done: u64, total: u64| {
                if let Ok(mut g) = s.lock() {
                    *g = (what.to_string(), done, total);
                }
            };
            let _ = tx.send(youtube::install::install(&dir, &c, &mut report));
        });
        self.shell.yt.install = Some((state, cancel, rx));
        self.shell.yt.error = None;
    }

    /// The YouTube view, in the Library's list area.
    pub(super) fn youtube_ui(&mut self, ui: &mut egui::Ui, tok: &Tokens, accent: Color32) {
        if let Some(rx) = self.shell.yt.thumbs_rx.take() {
            loop {
                match rx.try_recv() {
                    Ok((id, img)) => {
                        let tex = img.map(|i| ui.ctx().load_texture(format!("yt:{id}"), i, egui::TextureOptions::LINEAR));
                        self.shell.yt.thumbs.insert(id, tex);
                    }
                    Err(std::sync::mpsc::TryRecvError::Empty) => {
                        ui.ctx().request_repaint_after(std::time::Duration::from_millis(150));
                        self.shell.yt.thumbs_rx = Some(rx);
                        break;
                    }
                    // All in: nothing more to wait for.
                    Err(std::sync::mpsc::TryRecvError::Disconnected) => break,
                }
            }
        }
        // Lo-fi: a little rain for six seconds.
        if let Some(at) = self.shell.yt.rain_at {
            let t = at.elapsed().as_secs_f32();
            if t > 6.0 {
                self.shell.yt.rain_at = None;
            } else {
                ui.ctx().request_repaint();
                let area = ui.max_rect();
                let p = ui.ctx().layer_painter(LayerId::new(egui::Order::Foreground, Id::new("yt_rain")));
                let fade = (t.min(1.0) * (6.0 - t).min(1.0)).clamp(0.0, 1.0);
                for k in 0..48 {
                    let seed = (k as f32 * 0.6180339).fract();
                    let x = area.left() + area.width() * seed;
                    let y = area.top() + ((t * 0.9 + seed * 3.7).fract()) * area.height();
                    let c = with_alpha(accent, 0.35 * fade);
                    p.line_segment([Pos2::new(x, y), Pos2::new(x - 3.0, y + 12.0)], Stroke::new(1.2, c));
                }
            }
        }
        ui.horizontal(|ui| {
            ui.label(RichText::new("YouTube").size(16.0).strong().color(tok.ink));
            ui.add_space(10.0);
            for (music, label) in [(false, "Videos"), (true, "Music")] {
                if ui.selectable_label(self.shell.yt.music == music, label)
                    .on_hover_text(if music { "YouTube Music: songs" } else { "YouTube: everything" }).clicked()
                    && self.shell.yt.music != music
                {
                    self.shell.yt.music = music;
                    self.shell.yt.hits.clear();
                    self.shell.yt.searched = None;
                }
            }
            ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
                self.youtube_sign_in_button(ui, tok, accent);
                ui.add_space(8.0);
                // The video's size, where the video is chosen.
                if self.appearance.youtube_video {
                    let mut h = self.appearance.video_height;
                    egui::ComboBox::new("yt_quality", "Video size")
                        .selected_text(quality_name(h))
                        .show_ui(ui, |ui| {
                            for (q, name, tip) in QUALITIES {
                                ui.selectable_value(&mut h, q, name).on_hover_text(tip);
                            }
                        })
                        .response
                        .on_hover_text("The music video's size, for what you fetch next");
                    if h != self.appearance.video_height {
                        self.appearance.video_height = h;
                        save_appearance(&moosik_dir(), &self.appearance);
                    }
                }
                for (video, label, tip) in [(true, "Audio + video", "Also bring the music video, to play behind Listen and the spectrum"),
                    (false, "Audio", "Only the audio")] {
                    if ui.selectable_label(self.appearance.youtube_video == video, label).on_hover_text(tip).clicked()
                        && self.appearance.youtube_video != video
                    {
                        self.appearance.youtube_video = video;
                        save_appearance(&moosik_dir(), &self.appearance);
                    }
                }
            });
        });
        ui.add_space(8.0);
        if let Some(e) = self.shell.yt.sign_failed.clone() {
            let mut close = false;
            egui::Frame::NONE.fill(tok.chip).corner_radius(10).inner_margin(Margin::same(12)).show(ui, |ui| {
                ui.set_width(ui.available_width());
                ui.label(RichText::new("Not signed in").size(13.0).strong().color(tok.ink));
                ui.label(RichText::new(e).size(11.5).color(tok.dim));
                ui.label(RichText::new("Sign in opens a window of Chrome, Brave, Edge, Vivaldi or Chromium, whichever is here; without one, a cookies.txt exported from any browser works.").size(11.0).color(tok.faint));
                ui.horizontal_wrapped(|ui| {
                    if ui.small_button("Try again").clicked() {
                        self.youtube_sign_in_window();
                    }
                    if ui.small_button("Use a cookies.txt…").clicked()
                        && let Some(f) = rfd::FileDialog::new().set_title("cookies.txt with your YouTube sign-in").add_filter("cookies.txt", &["txt"]).pick_file()
                    {
                        let f = f.display().to_string();
                        youtube::set_sign_in(Some(f.clone()));
                        self.appearance.youtube_sign_in = Some(f);
                        save_appearance(&moosik_dir(), &self.appearance);
                        close = true;
                    }
                    close |= ui.small_button("Close").clicked();
                });
            });
            if close {
                self.shell.yt.sign_failed = None;
            }
            ui.add_space(8.0);
        }
        if self.shell.yt.installed != Some(true) {
            self.youtube_setup(ui, tok, accent);
            return;
        }
        let edit = ui.add(egui::TextEdit::singleline(&mut self.shell.yt.query)
            .hint_text(if self.shell.yt.music { "Search YouTube Music songs…" } else { "Search YouTube…" })
            .desired_width(f32::INFINITY));
        if edit.lost_focus() && ui.input(|i| i.key_pressed(egui::Key::Enter)) {
            self.youtube_search();
        }
        // An empty box offers the recent searches.
        if self.shell.yt.query.trim().is_empty() && !self.appearance.youtube_recent.is_empty() {
            ui.add_space(4.0);
            let mut pick = None;
            let mut clear = false;
            ui.horizontal_wrapped(|ui| {
                ui.label(RichText::new("Recent").size(11.0).color(tok.dim));
                for q in &self.appearance.youtube_recent {
                    let short: String = if q.chars().count() > 28 { q.chars().take(27).chain(['…']).collect() } else { q.clone() };
                    if ui.small_button(short).on_hover_text(q.as_str()).clicked() {
                        pick = Some(q.clone());
                    }
                }
                clear = ui.small_button("Clear").on_hover_text("Forget the recent searches").clicked();
            });
            if let Some(q) = pick {
                self.shell.yt.query = q;
                self.youtube_search();
            } else if clear {
                self.appearance.youtube_recent.clear();
                save_appearance(&moosik_dir(), &self.appearance);
            }
        }
        ui.add_space(6.0);
        let small = 11.5;
        if self.shell.yt.searching.is_some() {
            ui.label(RichText::new("Searching…").size(small).color(tok.dim));
        } else if let Some(e) = &self.shell.yt.error {
            ui.label(RichText::new(e.as_str()).size(small).color(Color32::from_rgb(220, 110, 110)));
        } else if self.shell.yt.searched.is_some() && self.shell.yt.hits.is_empty() {
            ui.label(RichText::new("Nothing found.").size(small).color(tok.dim));
        }
        // Signed in: your own lists, a click away.
        if youtube::signed_in() && self.shell.yt.query.trim().is_empty() {
            let mut pick = None;
            ui.horizontal_wrapped(|ui| {
                ui.label(RichText::new("Yours").size(11.0).color(tok.dim));
                for (label, url) in [("Liked videos", "https://www.youtube.com/playlist?list=LL"),
                    ("Liked music", "https://music.youtube.com/playlist?list=LM"),
                    ("Watch later", "https://www.youtube.com/playlist?list=WL")] {
                    if ui.small_button(label).clicked() {
                        pick = Some(url);
                    }
                }
            });
            if let Some(url) = pick {
                self.shell.yt.query = url.into();
                self.youtube_search();
            }
            ui.add_space(4.0);
        }
        // A playlist or album: all of it at once.
        let list = self.shell.yt.searched.as_deref().is_some_and(|q| youtube::parse_list(q).is_some());
        if list && self.shell.yt.hits.len() > 1 {
            ui.horizontal(|ui| {
                let left = self.shell.yt.waiting.len();
                if left > 0 {
                    ui.label(RichText::new(format!("Adding the rest: {left} to go")).size(small).color(tok.dim));
                    if ui.small_button("Stop").on_hover_text("Add no more; what is in stays").clicked() {
                        self.shell.yt.waiting.clear();
                    }
                } else if ui.small_button(format!("Queue all {}", self.shell.yt.hits.len())).on_hover_text("Add every song to the end of the queue, in order").clicked() {
                    self.shell.yt.waiting = self.shell.yt.hits.iter().cloned().collect();
                }
            });
            ui.add_space(4.0);
        }
        let footer = 26.0;
        let hits = self.shell.yt.hits.clone();
        // What is cached, looked at again every few seconds (a fetch adds
        // to it, the cache trim takes away).
        if self.shell.yt.cached.as_ref().is_none_or(|(_, at)| at.elapsed().as_secs_f32() > 3.0) {
            self.shell.yt.cached = Some((youtube::cached_ids(&youtube::cache_dir(&moosik_dir())), Instant::now()));
        }
        let ready = self.shell.yt.cached.as_ref().map(|(c, _)| c.clone()).unwrap_or_default();
        let mut act: Option<(Hit, Option<Then>)> = None;
        egui::ScrollArea::vertical().id_salt("yt_hits").auto_shrink([false, false])
            .max_height((ui.available_height() - footer).max(40.0)).show(ui, |ui| {
                for hit in &hits {
                    let fetch = self.shell.yt.fetches.iter().find(|f| f.hit.id == hit.id);
                    let (r, _) = ui.allocate_exact_size(Vec2::new(ui.available_width(), 44.0), Sense::hover());
                    let hovered = ui.rect_contains_pointer(r);
                    if hovered {
                        ui.painter().rect_filled(r, 8, tok.chip);
                    }
                    let buttons = 3.0 * 64.0;
                    let thumb = Rect::from_min_size(r.left_top() + Vec2::new(6.0, 4.0), Vec2::new(64.0, 36.0));
                    match self.shell.yt.thumbs.get(&hit.id) {
                        Some(Some(t)) => {
                            egui::Image::new(t).corner_radius(4).paint_at(ui, thumb);
                        }
                        _ => {
                            ui.painter().rect_filled(thumb, 4, tok.chip);
                        }
                    }
                    let tx = 78.0;
                    let text_w = (r.width() - buttons - tx - 10.0).max(60.0);
                    let mut job = egui::text::LayoutJob::simple_singleline(hit.title.clone(), crate::name_font(&hit.title, 13.5), tok.ink);
                    job.wrap = egui::text::TextWrapping::truncate_at_width(text_w);
                    let g = ui.fonts(|f| f.layout_job(job));
                    ui.painter().galley(r.left_top() + Vec2::new(tx, 6.0), g, tok.ink);
                    let sub = [hit.channel.clone(), hit.duration.map(fmt_secs)].into_iter().flatten().collect::<Vec<_>>().join(" · ");
                    let sr = ui.painter().text(r.left_top() + Vec2::new(tx, 26.0), egui::Align2::LEFT_TOP, sub, egui::FontId::proportional(11.0), tok.dim);
                    if ready.contains(&hit.id) {
                        ui.painter().text(sr.right_top() + Vec2::new(8.0, 0.0), egui::Align2::LEFT_TOP, "ready", egui::FontId::proportional(11.0), accent);
                    }
                    let mut bui = ui.new_child(egui::UiBuilder::new().max_rect(Rect::from_min_max(Pos2::new(r.right() - buttons - 8.0, r.top()), r.max))
                        .layout(egui::Layout::right_to_left(egui::Align::Center)));
                    match fetch {
                        Some(f) => {
                            if bui.small_button("Cancel").clicked() {
                                act = Some((hit.clone(), None));
                            }
                            // The audio first; then, while it plays, the video.
                            let video = f.video_for.is_some() || f.audio_in;
                            let p = &f.progress[video as usize];
                            let (d, t) = (p.done.load(Ordering::Relaxed), p.total.load(Ordering::Relaxed));
                            let q = if t > 0 { d as f32 / t as f32 } else { 0.0 };
                            let what = match f.then {
                                _ if video => "Video",
                                Then::Play if f.listed => "Playing",
                                Then::Play => "Loading",
                                Then::Queue => "Adding",
                                Then::Download => "Downloading",
                            };
                            bui.add(egui::ProgressBar::new(q).desired_width(110.0).text(RichText::new(what).size(10.5)));
                        }
                        None => {
                            for (label, tip, then) in [("Download", "Save to your Music folder", Then::Download),
                                ("Queue", "Add to the end of the queue", Then::Queue), ("Play", "Play now", Then::Play)] {
                                if bui.small_button(label).on_hover_text(tip).clicked() {
                                    act = Some((hit.clone(), Some(then)));
                                }
                            }
                        }
                    }
                }
            });
        match act {
            Some((hit, Some(then))) => self.youtube_fetch(&hit, then),
            Some((hit, None)) => {
                if let Some(f) = self.shell.yt.fetches.iter().find(|f| f.hit.id == hit.id) {
                    f.cancel.store(true, Ordering::Relaxed);
                }
            }
            None => {}
        }
        // Footer: versions, update, where downloads go.
        ui.horizontal(|ui| {
            let line = self.shell.yt.version_line.clone().unwrap_or_else(|| "…".into());
            ui.label(RichText::new(line).size(10.5).color(tok.faint));
            if self.shell.yt.install.is_some() {
                ui.label(RichText::new("Updating…").size(10.5).color(tok.dim));
            } else if ui.small_button("Update").on_hover_text("Get the latest yt-dlp and Deno (YouTube changes often break older yt-dlp)").clicked() {
                self.youtube_install();
            }
            if ui.small_button("Downloads folder").on_hover_text(self.yt_downloads().display().to_string()).clicked() {
                let d = self.yt_downloads();
                let _ = std::fs::create_dir_all(&d);
                #[cfg(windows)]
                let _ = std::process::Command::new("explorer").arg(&d).spawn();
            }
        });
        let _ = accent;
    }

    /// Sign in / Signing in… / signed in (with Sign out on its menu), in the
    /// YouTube view's header.
    fn youtube_sign_in_button(&mut self, ui: &mut egui::Ui, tok: &Tokens, accent: Color32) {
        if self.shell.yt.installed != Some(true) {
            return;
        }
        if let Some((cancel, _)) = &self.shell.yt.signing {
            if ui.small_button("Cancel").on_hover_text("Close the sign-in window").clicked() {
                cancel.store(true, Ordering::Relaxed);
            }
            ui.label(RichText::new("Sign in in the window that opened…").size(11.5).color(tok.dim));
            return;
        }
        match self.appearance.youtube_sign_in.clone() {
            Some(s) => {
                let own = youtube::window::cookies_file(&self.yt_tools());
                let via = match s.as_str() {
                    "firefox" => "Firefox", "edge" => "Edge", "chrome" => "Chrome", "brave" => "Brave",
                    _ if std::path::Path::new(&s) == own => "Moosik's sign-in window",
                    _ => "cookies.txt",
                };
                let r = ui.add(egui::Button::new(RichText::new("● Signed in").size(11.5).color(accent)).frame(false))
                    .on_hover_text(format!("Through {via}. Click to sign out"));
                if r.clicked() {
                    if std::path::Path::new(&s) == own {
                        youtube::window::sign_out(&self.yt_tools());
                    }
                    self.appearance.youtube_sign_in = None;
                    youtube::set_sign_in(None);
                    save_appearance(&moosik_dir(), &self.appearance);
                }
            }
            None => {
                if ui.button("Sign in").on_hover_text("Sign in to YouTube in a window of its own: age-restricted, members-only and private videos, your playlists and likes").clicked() {
                    self.youtube_sign_in_window();
                }
            }
        }
    }

    /// Open the sign-in window; its thread waits for the account.
    fn youtube_sign_in_window(&mut self) {
        let (tx, rx) = std::sync::mpsc::channel();
        let cancel = Arc::new(AtomicBool::new(false));
        let (tools, c) = (self.yt_tools(), cancel.clone());
        let _ = std::thread::Builder::new().name("yt-sign-in".into()).spawn(move || {
            let _ = tx.send(youtube::window::sign_in(&tools, &c));
        });
        self.shell.yt.signing = Some((cancel, rx));
        self.shell.yt.sign_failed = None;
    }

    /// Before the components are there: what they are, what they cost, and
    /// what using them means.
    fn youtube_setup(&mut self, ui: &mut egui::Ui, tok: &Tokens, accent: Color32) {
        let body = |ui: &mut egui::Ui, t: &str| {
            ui.label(RichText::new(t).size(12.5).color(tok.dim));
        };
        egui::Frame::NONE.fill(tok.chip).corner_radius(12).inner_margin(Margin::same(16)).show(ui, |ui| {
            ui.set_width(ui.available_width());
            ui.label(RichText::new("Play and download from YouTube").size(15.0).strong().color(tok.ink));
            ui.add_space(6.0);
            body(ui, "Moosik uses two programs for this, which it downloads only if you ask: yt-dlp, which talks to YouTube, and Deno, the JavaScript runtime yt-dlp needs for it. Both come from their projects' GitHub releases and are checked against the checksums those releases publish.");
            ui.add_space(4.0);
            body(ui, &format!("About {} MB to download, {} MB on disk, kept in your Moosik folder. yt-dlp is in the public domain (Unlicense); Deno is MIT-licensed. Neither is part of Moosik.",
                youtube::install::ABOUT_DOWNLOAD_MB, youtube::install::ABOUT_DISK_MB));
            ui.add_space(4.0);
            body(ui, "Audio stays as YouTube delivers it (Opus or AAC) and is marked lossy. YouTube's terms limit downloading: use this for what you have the right to play or keep. Nothing behind a paywall or DRM is available.");
            ui.add_space(10.0);
            match &self.shell.yt.install {
                Some((state, cancel, _)) => {
                    let (what, d, t) = state.lock().map(|g| g.clone()).unwrap_or_default();
                    let q = if t > 0 { d as f32 / t as f32 } else { 0.0 };
                    let cancel = cancel.clone();
                    ui.horizontal(|ui| {
                        ui.add(egui::ProgressBar::new(q).desired_width(260.0)
                            .text(RichText::new(format!("{} {:.1} / {:.1} MB", if what.is_empty() { "Starting" } else { &what }, d as f32 / 1e6, t as f32 / 1e6)).size(11.0)));
                        if ui.small_button("Cancel").clicked() {
                            cancel.store(true, Ordering::Relaxed);
                        }
                    });
                }
                None => {
                    if ui.add(egui::Button::new(RichText::new(format!("Get yt-dlp and Deno ({} MB)", youtube::install::ABOUT_DOWNLOAD_MB)).color(accent))).clicked() {
                        self.youtube_install();
                    }
                }
            }
            if let Some(e) = &self.shell.yt.error {
                ui.add_space(6.0);
                ui.label(RichText::new(e.as_str()).size(11.5).color(Color32::from_rgb(220, 110, 110)));
            }
        });
    }

    /// The Sources part of the Library sidebar.
    pub(super) fn sources_sidebar(&mut self, ui: &mut egui::Ui, tok: &Tokens, accent: Color32) {
        ui.add_space(10.0);
        ui.label(RichText::new("SOURCES").size(11.0).strong().color(tok.dim).extra_letter_spacing(0.8));
        ui.add_space(6.0);
        let active = self.shell.yt.open;
        let (r, resp) = ui.allocate_exact_size(Vec2::new(ui.available_width(), 34.0), Sense::click());
        resp.widget_info(|| egui::WidgetInfo::selected(egui::WidgetType::SelectableLabel, true, active, "YouTube"));
        let fill = if active { with_alpha(accent, 0.2) } else if resp.hovered() { tok.chip } else { Color32::TRANSPARENT };
        let p = ui.painter_at(r);
        p.rect_filled(r, 8, fill);
        // A little play-button badge.
        let b = Rect::from_center_size(Pos2::new(r.left() + 18.0, r.center().y), Vec2::new(18.0, 13.0));
        p.rect_filled(b, 4, if active { accent } else { tok.dim });
        let c = b.center();
        p.add(egui::Shape::convex_polygon(vec![c + Vec2::new(-2.5, -3.5), c + Vec2::new(4.0, 0.0), c + Vec2::new(-2.5, 3.5)], tok.bg, Stroke::NONE));
        p.text(Pos2::new(r.left() + 36.0, r.center().y), egui::Align2::LEFT_CENTER, "YouTube", egui::FontId::proportional(13.0), tok.ink);
        let n = self.shell.yt.fetches.len();
        if n > 0 {
            p.text(Pos2::new(r.right() - 10.0, r.center().y), egui::Align2::RIGHT_CENTER, format!("{n}↓"), egui::FontId::proportional(11.0), accent);
        }
        if resp.clicked() {
            self.shell.yt.open = !active;
        }
    }
}

#[cfg(test)]
mod tests {
    #[test]
    fn lo_fi_is_heard_however_it_is_spelt() {
        for q in ["lofi hip hop", "Lo-Fi beats", "chill LO FI", "ローファイ"] {
            assert!(super::is_lofi(q), "{q}");
        }
        assert!(!super::is_lofi("metal"));
    }

    #[test]
    fn durations_read_like_a_clock() {
        assert_eq!(super::fmt_secs(19.0), "0:19");
        assert_eq!(super::fmt_secs(213.4), "3:33");
        assert_eq!(super::fmt_secs(3725.0), "1:02:05");
    }
}

#[cfg(test)]
impl super::ShellState {
    /// How many search results are showing.
    pub(crate) fn yt_hits(&self) -> usize {
        self.yt.hits.len()
    }

    /// A search result, by place.
    pub(crate) fn yt_hit(&self, i: usize) -> Option<Hit> {
        self.yt.hits.get(i).cloned()
    }

    /// As if Sign in had found no browser signed in.
    pub(crate) fn yt_sign_failed(&mut self, why: &str) {
        self.yt.sign_failed = Some(why.into());
    }

    /// Show these results, as if `searched` had found them.
    pub(crate) fn yt_show(&mut self, hits: Vec<Hit>, searched: &str) {
        self.yt.hits = hits;
        self.yt.searched = Some(searched.into());
    }
}
