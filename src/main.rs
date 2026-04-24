mod spectrum;

use eframe::egui;
use egui::{Color32, RichText, Slider, Vec2};
use lofty::prelude::*;
use lofty::probe::Probe;
use rodio::{Decoder, OutputStream, OutputStreamHandle, Sink, Source};
use serde::{Deserialize, Serialize};
use spectrum::{SampleBuf, SpectralCeiling, StereoBuf, TrackAnalysis, SpectrumSource, SpectrumWindow};
use spectrum::eq::{EqSource, EqStateHandle};
use std::collections::HashMap;
use std::collections::HashSet;
use std::fs::File;
use std::io::BufReader;
use std::path::{Path, PathBuf};
use std::time::{Duration, Instant};

fn setup_fonts(ctx: &egui::Context) {
    // Each entry: (font key, candidate paths in priority order).
    // We try every path for each script; first hit wins.
    // All loaded fonts are pushed as fallbacks so egui uses them for missing glyphs.
    let scripts: &[(&str, &[&str])] = &[
        ("jp", &[
            r"C:\Windows\Fonts\meiryo.ttc",
            r"C:\Windows\Fonts\yugothic.ttf",
            r"C:\Windows\Fonts\msgothic.ttc",
            "/System/Library/Fonts/ヒラギノ角ゴシック W3.ttc",
            "/usr/share/fonts/truetype/takao-gothic/TakaoGothic.ttf",
        ]),
        ("zh_tw", &[
            r"C:\Windows\Fonts\msjh.ttc",
            "/System/Library/Fonts/PingFang.ttc",
        ]),
        ("zh_cn", &[
            r"C:\Windows\Fonts\msyh.ttc",
            r"C:\Windows\Fonts\simsun.ttc",
        ]),
        ("ko", &[
            r"C:\Windows\Fonts\malgun.ttf",
            "/System/Library/Fonts/AppleSDGothicNeo.ttc",
            "/usr/share/fonts/truetype/nanum/NanumGothic.ttf",
        ]),
        // Single pan-CJK font covers everything on Linux/fallback
        ("cjk", &[
            "/usr/share/fonts/noto-cjk/NotoSansCJK-Regular.ttc",
            "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc",
            "/usr/share/fonts/truetype/noto/NotoSansCJK-Regular.ttc",
        ]),
    ];

    let mut fonts = egui::FontDefinitions::default();
    let mut any = false;

    for (name, paths) in scripts {
        for path in *paths {
            if let Ok(data) = std::fs::read(path) {
                fonts.font_data.insert(
                    name.to_string(),
                    egui::FontData::from_owned(data).into(),
                );
                for fam in [egui::FontFamily::Proportional, egui::FontFamily::Monospace] {
                    fonts.families.entry(fam).or_default().push(name.to_string());
                }
                any = true;
                break; // found one for this script, skip remaining paths
            }
        }
    }

    if any {
        ctx.set_fonts(fonts);
    }
}

fn main() -> eframe::Result {
    let icon = eframe::icon_data::from_png_bytes(
        include_bytes!("../assets/icon.png")
    ).expect("invalid icon PNG");

    let options = eframe::NativeOptions {
        viewport: egui::ViewportBuilder::default()
            .with_title("Moosik")
            .with_inner_size([800.0, 600.0])
            .with_min_inner_size([600.0, 400.0])
            .with_icon(icon),
        ..Default::default()
    };
    eframe::run_native(
        "Moosik",
        options,
        Box::new(|cc| Ok(Box::new(MoosikApp::new(cc)))),
    )
}

// ---------------------------------------------------------------------------
// Data types
// ---------------------------------------------------------------------------

#[derive(Clone, Debug)]
struct Track {
    path: PathBuf,
    title: String,
    artist: String,
    album: String,
    year: Option<u32>,
    genre: Option<String>,
    track_number: Option<u32>,
    duration: Option<Duration>,
    // audio stream properties (from lofty AudioProperties)
    sample_rate: Option<u32>,
    channels: Option<u8>,
    bit_depth: Option<u8>,
    bitrate: Option<u32>,   // avg kbps
    file_size: u64,
}

impl Track {
    fn load(path: PathBuf) -> Self {
        let m = read_metadata(&path);
        Track {
            path,
            title: m.title, artist: m.artist, album: m.album,
            year: m.year, genre: m.genre, track_number: m.track_number,
            duration: m.duration,
            sample_rate: m.sample_rate, channels: m.channels,
            bit_depth: m.bit_depth, bitrate: m.bitrate,
            file_size: m.file_size,
        }
    }

    fn display_title(&self) -> &str { &self.title }
}

struct TrackMeta {
    title: String, artist: String, album: String,
    year: Option<u32>, genre: Option<String>, track_number: Option<u32>,
    duration: Option<Duration>,
    sample_rate: Option<u32>, channels: Option<u8>,
    bit_depth: Option<u8>, bitrate: Option<u32>,
    file_size: u64,
}

fn read_metadata(path: &PathBuf) -> TrackMeta {
    let fallback_title = path.file_stem()
        .and_then(|s| s.to_str()).unwrap_or("Unknown").to_string();
    let file_size = std::fs::metadata(path).map(|m| m.len()).unwrap_or(0);

    let tagged = Probe::open(path).ok().and_then(|p| p.read().ok());

    let (title, artist, album, year, genre, track_number) = if let Some(ref t) = tagged {
        let tag = t.primary_tag().or_else(|| t.first_tag());
        if let Some(tag) = tag {
            (
                tag.title().map(|s| s.to_string()).unwrap_or_else(|| fallback_title.clone()),
                tag.artist().map(|s| s.to_string()).unwrap_or_else(|| "Unknown Artist".to_string()),
                tag.album().map(|s| s.to_string()).unwrap_or_else(|| "Unknown Album".to_string()),
                tag.year(),
                tag.genre().map(|s| s.to_string()),
                tag.track(),
            )
        } else {
            (fallback_title.clone(), "Unknown Artist".to_string(), "Unknown Album".to_string(), None, None, None)
        }
    } else {
        (fallback_title, "Unknown Artist".to_string(), "Unknown Album".to_string(), None, None, None)
    };

    let (duration, sample_rate, channels, bit_depth, bitrate) = if let Some(ref t) = tagged {
        let p = t.properties();
        let secs = p.duration().as_secs();
        (
            if secs > 0 { Some(Duration::from_secs(secs)) } else { None },
            p.sample_rate(), p.channels(), p.bit_depth(), p.audio_bitrate(),
        )
    } else {
        (None, None, None, None, None)
    };

    TrackMeta { title, artist, album, year, genre, track_number,
                duration, sample_rate, channels, bit_depth, bitrate, file_size }
}

// ---------------------------------------------------------------------------
// Audio info helpers
// ---------------------------------------------------------------------------

fn codec_name(path: &Path) -> &'static str {
    match path.extension().and_then(|e| e.to_str()) {
        Some("flac") => "FLAC",
        Some("mp3")  => "MP3",
        Some("ogg")  => "Ogg Vorbis",
        Some("wav")  => "WAV / PCM",
        _            => "Unknown",
    }
}

fn is_lossless(path: &Path) -> bool {
    matches!(path.extension().and_then(|e| e.to_str()), Some("flac") | Some("wav"))
}

fn channel_layout(n: u8) -> &'static str {
    match n { 1 => "Mono", 2 => "Stereo", 4 => "Quad",
              6 => "5.1 Surround", 8 => "7.1 Surround", _ => "Multi-channel" }
}

fn fmt_hz(sr: u32) -> String {
    if sr.is_multiple_of(1000) { format!("{}kHz", sr / 1000) }
    else { format!("{:.1}kHz", sr as f32 / 1000.0) }
}

fn fmt_size(bytes: u64) -> String {
    if bytes >= 1_073_741_824 {
        format!("{:.2} GB  ({} bytes)", bytes as f64 / 1_073_741_824.0, bytes)
    } else if bytes >= 1_048_576 {
        format!("{:.1} MB  ({} bytes)", bytes as f64 / 1_048_576.0, bytes)
    } else {
        format!("{:.1} KB  ({} bytes)", bytes as f64 / 1024.0, bytes)
    }
}

// ---------------------------------------------------------------------------
// Playback engine (wraps rodio Sink)
// ---------------------------------------------------------------------------

struct Engine {
    _stream: OutputStream,
    stream_handle: OutputStreamHandle,
    sink: Option<Sink>,
    // We track position manually because rodio Sink doesn't expose elapsed time
    started_at: Option<Instant>,
    paused_elapsed: Duration,
    current_duration: Option<Duration>,
    volume: f32,
    sample_buf: SampleBuf,
    stereo_buf: StereoBuf,
    pub last_sample_rate: u32,
    eq: Option<EqStateHandle>,
}

impl Engine {
    fn new(sample_buf: SampleBuf, stereo_buf: StereoBuf) -> Option<Self> {
        let (stream, stream_handle) = OutputStream::try_default().ok()?;
        Some(Engine {
            _stream: stream,
            stream_handle,
            sink: None,
            started_at: None,
            paused_elapsed: Duration::ZERO,
            current_duration: None,
            volume: 1.0,
            sample_buf,
            stereo_buf,
            last_sample_rate: 44_100,
            eq: None,
        })
    }

    fn play_file(&mut self, path: &PathBuf, duration: Option<Duration>) -> Result<(), String> {
        self.stop();
        let file = File::open(path).map_err(|e| format!("Open failed: {e}"))?;
        let decoder = Decoder::new(BufReader::new(file))
            .map_err(|e| format!("Decode failed: {e}"))?;
        let sink = Sink::try_new(&self.stream_handle)
            .map_err(|e| format!("Sink failed: {e}"))?;
        sink.set_volume(self.volume);
        self.last_sample_rate = decoder.sample_rate();
        let tapped = SpectrumSource::new(decoder, self.sample_buf.clone(), self.stereo_buf.clone());
        if let Some(ref eq) = self.eq {
            sink.append(EqSource::new(tapped.convert_samples::<f32>(), eq.clone()));
        } else {
            sink.append(tapped);
        }
        self.sink = Some(sink);
        self.started_at = Some(Instant::now());
        self.paused_elapsed = Duration::ZERO;
        self.current_duration = duration;
        Ok(())
    }

    fn pause(&mut self) {
        if let Some(ref sink) = self.sink
            && !sink.is_paused() {
            sink.pause();
            if let Some(started) = self.started_at.take() {
                self.paused_elapsed += started.elapsed();
            }
        }
    }

    fn resume(&mut self) {
        if let Some(ref sink) = self.sink
            && sink.is_paused() {
            sink.play();
            self.started_at = Some(Instant::now());
        }
    }

    fn stop(&mut self) {
        if let Some(sink) = self.sink.take() {
            sink.stop();
        }
        self.started_at = None;
        self.paused_elapsed = Duration::ZERO;
        self.current_duration = None;
    }

    fn is_finished(&self) -> bool {
        self.sink.as_ref().is_some_and(|s| s.empty())
    }

    fn elapsed(&self) -> Duration {
        let running = self.started_at.map(|t| t.elapsed()).unwrap_or(Duration::ZERO);
        self.paused_elapsed + running
    }

    fn set_volume(&mut self, vol: f32) {
        self.volume = vol;
        if let Some(ref sink) = self.sink {
            sink.set_volume(vol);
        }
    }

    /// Seek to `target`.
    ///
    /// Preferred path: `Sink::try_seek` — works for WAV and any format whose symphonia
    /// reader can seek without the byte-length hint (e.g. MP3).
    ///
    /// Fallback path: stop sink, reopen file, consume N samples to reach `target`,
    /// start a fresh sink.  Handles FLAC, where symphonia 0.5.5 returns
    /// `Unseekable` because rodio's `ReadSeekSource::byte_len()` always returns `None`.
    fn seek_to(&mut self, path: &PathBuf, target: Duration) {
        let was_paused = self.sink.as_ref().map(|s| s.is_paused()).unwrap_or(false);

        // --- Fast path: in-place seek via Sink::try_seek ---
        let fast_ok = self.sink.as_ref().is_some_and(|sink| {
            sink.try_seek(target).is_ok()
        });

        if fast_ok {
            if let Some(ref _sink) = self.sink {
                self.paused_elapsed = target;
                if was_paused { self.started_at = None; }
                else { self.started_at = Some(Instant::now()); }
            }
            return;
        }

        // --- Slow path: reopen + skip samples ---
        if let Some(sink) = self.sink.take() { sink.stop(); }

        let file = match File::open(path) { Ok(f) => f, Err(e) => { eprintln!("seek reopen: {e}"); return; } };
        let decoder = match Decoder::new(BufReader::new(file)) { Ok(d) => d, Err(e) => { eprintln!("seek decode: {e}"); return; } };

        let sr = decoder.sample_rate();
        let ch = decoder.channels() as u64;
        let samples_to_skip = (target.as_secs_f64() * sr as f64 * ch as f64) as u64;

        // Consume samples up to the target. Each `next()` call decodes one sample;
        // FLAC decodes frames lazily so this runs well above real-time even in debug.
        let mut decoder = decoder;
        let mut consumed = 0u64;
        while consumed < samples_to_skip {
            if decoder.next().is_none() { break; }
            consumed += 1;
        }

        let sink = match Sink::try_new(&self.stream_handle) { Ok(s) => s, Err(_) => return };
        sink.set_volume(self.volume);
        let tapped = SpectrumSource::new(decoder, self.sample_buf.clone(), self.stereo_buf.clone());
        if let Some(ref eq) = self.eq {
            sink.append(EqSource::new(tapped.convert_samples::<f32>(), eq.clone()));
        } else {
            sink.append(tapped);
        }

        if was_paused { sink.pause(); self.started_at = None; }
        else { self.started_at = Some(Instant::now()); }
        self.paused_elapsed = target;
        self.sink = Some(sink);
        // Grace period: suppress is_finished() for 500 ms so the audio thread has
        // time to pick up the source before we auto-advance.
    }
}

// ---------------------------------------------------------------------------
// App state
// ---------------------------------------------------------------------------

#[derive(PartialEq)]
enum PlayState { Stopped, Playing, Paused }

#[derive(PartialEq, Clone, Copy)]
enum LoopMode { Sequential, RepeatAll, RepeatOne }

// ---------------------------------------------------------------------------
// Album art helpers
// ---------------------------------------------------------------------------

/// Fit `(art_w × art_h)` into `container` while preserving aspect ratio
/// (letterboxed / pillarboxed, centred).
fn fit_rect_preserve(container: egui::Rect, art_w: u32, art_h: u32) -> egui::Rect {
    if art_w == 0 || art_h == 0 { return container; }
    let art_aspect   = art_w as f32 / art_h as f32;
    let cont_aspect  = container.width() / container.height().max(1.0);
    if art_aspect > cont_aspect {
        let new_h = container.width() / art_aspect;
        let pad   = (container.height() - new_h) / 2.0;
        egui::Rect::from_min_max(
            egui::Pos2::new(container.left(),  container.top()    + pad),
            egui::Pos2::new(container.right(), container.bottom() - pad),
        )
    } else {
        let new_w = container.height() * art_aspect;
        let pad   = (container.width() - new_w) / 2.0;
        egui::Rect::from_min_max(
            egui::Pos2::new(container.left()  + pad, container.top()),
            egui::Pos2::new(container.right() - pad, container.bottom()),
        )
    }
}

// ---------------------------------------------------------------------------
// Album art loading and caching
// ---------------------------------------------------------------------------

enum ArtEntry {
    /// No embedded art found in this file.
    NoArt,
    /// Art decoded and uploaded to the GPU.
    Loaded { texture: egui::TextureHandle, width: u32, height: u32 },
}

struct ArtCache {
    entries: HashMap<PathBuf, ArtEntry>,
}

impl ArtCache {
    fn new() -> Self { Self { entries: HashMap::new() } }

    /// Returns `(TextureId, orig_width, orig_height)` if art is available,
    /// loading and caching it on first call.  Returns `None` if the file has
    /// no embedded art or decoding failed.
    fn get_or_load(
        &mut self,
        path: &Path,
        ctx: &egui::Context,
    ) -> Option<(egui::TextureId, u32, u32)> {
        if !self.entries.contains_key(path) {
            let entry = match Self::load_from_file(path, ctx) {
                Some((tex, w, h)) => ArtEntry::Loaded { texture: tex, width: w, height: h },
                None              => ArtEntry::NoArt,
            };
            self.entries.insert(path.to_path_buf(), entry);
        }
        match self.entries.get(path)? {
            ArtEntry::Loaded { texture, width, height } =>
                Some((texture.id(), *width, *height)),
            ArtEntry::NoArt => None,
        }
    }

    /// True when we already know the answer (loaded or confirmed no-art).
    #[allow(dead_code)]
    fn is_known(&self, path: &Path) -> bool { self.entries.contains_key(path) }

    /// True when the file has art that has been successfully loaded.
    #[allow(dead_code)]
    fn has_art(&self, path: &Path) -> bool {
        matches!(self.entries.get(path), Some(ArtEntry::Loaded { .. }))
    }

    fn load_from_file(path: &Path, ctx: &egui::Context)
        -> Option<(egui::TextureHandle, u32, u32)>
    {
        let bytes = Self::extract_bytes(path)?;
        let img = image::load_from_memory(&bytes).ok()?.to_rgba8();
        let (w, h) = img.dimensions();
        let pixels: Vec<egui::Color32> = img.pixels()
            .map(|p| egui::Color32::from_rgba_unmultiplied(p.0[0], p.0[1], p.0[2], p.0[3]))
            .collect();
        let ci = egui::ColorImage { size: [w as usize, h as usize], pixels };
        let tex = ctx.load_texture(
            format!("art:{}", path.to_string_lossy()),
            ci,
            egui::TextureOptions::LINEAR,
        );
        Some((tex, w, h))
    }

    fn extract_bytes(path: &Path) -> Option<Vec<u8>> {
        let tagged = Probe::open(path).ok()?.read().ok()?;
        let tag = tagged.primary_tag().or_else(|| tagged.first_tag())?;
        let pic = tag.pictures().iter()
            .find(|p| p.pic_type() == lofty::picture::PictureType::CoverFront)
            .or_else(|| tag.pictures().first())?;
        Some(pic.data().to_vec())
    }
}

// ---------------------------------------------------------------------------

struct MoosikApp {
    playlist: Vec<Track>,
    current_index: Option<usize>,
    play_state: PlayState,
    engine: Option<Engine>,
    volume: f32,
    seek_pos: f32,
    seeking: bool,
    status_msg: String,
    spectrum_window: SpectrumWindow,
    info_open: bool,
    loop_mode: LoopMode,
    // multi-select
    selected: HashSet<usize>,
    last_clicked: Option<usize>,
    // drag-to-reorder
    drag_src: Option<usize>,
    drag_over_row: Option<usize>,
    // named playlist store
    playlist_store: Vec<SavedPlaylist>,
    active_saved_playlist: Option<usize>,
    show_save_playlist_input: bool,
    playlist_name_buf: String,
    // album art
    art_cache: ArtCache,
    art_hover: Option<(usize, Instant)>, // (track index, time hover started)
}

impl MoosikApp {
    fn new(cc: &eframe::CreationContext) -> Self {
        setup_fonts(&cc.egui_ctx);
        let spectrum_window = SpectrumWindow::new();
        let mut engine = Engine::new(spectrum_window.sample_buf.clone(), spectrum_window.stereo_buf.clone());
        if let Some(ref mut e) = engine {
            e.eq = Some(spectrum_window.eq_state.clone());
        }
        let saved_paths = load_last_playlist();
        let playlist: Vec<Track> = saved_paths.into_iter().map(Track::load).collect();
        let playlist_store = load_playlist_store();
        MoosikApp {
            playlist,
            current_index: None,
            play_state: PlayState::Stopped,
            engine,
            volume: 0.8,
            seek_pos: 0.0,
            seeking: false,
            status_msg: String::new(),
            spectrum_window,
            info_open: false,
            loop_mode: LoopMode::RepeatAll,
            selected: HashSet::new(),
            last_clicked: None,
            drag_src: None,
            drag_over_row: None,
            playlist_store,
            active_saved_playlist: None,
            show_save_playlist_input: false,
            playlist_name_buf: String::new(),
            art_cache: ArtCache::new(),
            art_hover: None,
        }
    }

    fn add_files(&mut self) {
        let paths = rfd::FileDialog::new()
            .add_filter("Audio", &["flac", "wav", "mp3", "ogg"])
            .set_title("Add audio files")
            .pick_files();
        if let Some(paths) = paths {
            for p in paths { self.playlist.push(Track::load(p)); }
        }
    }

    fn add_folder(&mut self) {
        let Some(root) = rfd::FileDialog::new().set_title("Load folder").pick_folder() else { return };
        let mut stack = vec![root];
        let mut found = Vec::new();
        while let Some(dir) = stack.pop() {
            let Ok(rd) = std::fs::read_dir(&dir) else { continue };
            for entry in rd.flatten() {
                let path = entry.path();
                if path.is_dir() {
                    stack.push(path);
                } else if let Some(ext) = path.extension().and_then(|e| e.to_str())
                    && matches!(ext.to_lowercase().as_str(), "flac" | "wav" | "mp3" | "ogg") {
                    found.push(path);
                }
            }
        }
        found.sort();
        for p in found { self.playlist.push(Track::load(p)); }
    }

    fn play_index(&mut self, idx: usize) {
        if idx >= self.playlist.len() {
            return;
        }
        self.current_index = Some(idx);
        let track = &self.playlist[idx];
        let path = track.path.clone();
        let duration = track.duration;
        let result = if let Some(ref mut engine) = self.engine {
            engine.play_file(&path, duration)
        } else {
            Err("No audio engine".to_string())
        };
        match result {
            Ok(()) => {
                self.play_state = PlayState::Playing;
                self.status_msg = format!("Playing: {}", self.playlist[idx].title);
            }
            Err(e) => {
                self.play_state = PlayState::Stopped;
                self.status_msg = format!("Error: {e}");
                return;
            }
        }
        self.seek_pos = 0.0;
        self.seeking = false;
        let sr = self.engine.as_ref().map(|e| e.last_sample_rate).unwrap_or(44_100);
        self.spectrum_window.on_play(&self.playlist[idx].path, sr);
    }

    fn toggle_play_pause(&mut self) {
        match self.play_state {
            PlayState::Playing => {
                if let Some(ref mut engine) = self.engine {
                    engine.pause();
                }
                self.play_state = PlayState::Paused;
            }
            PlayState::Paused => {
                if let Some(ref mut engine) = self.engine {
                    engine.resume();
                }
                self.play_state = PlayState::Playing;
            }
            PlayState::Stopped => {
                // Play current selection or first track
                let idx = self.current_index.unwrap_or(0);
                if !self.playlist.is_empty() {
                    self.play_index(idx);
                }
            }
        }
    }

    fn stop(&mut self) {
        if let Some(ref mut engine) = self.engine {
            engine.stop();
        }
        self.play_state = PlayState::Stopped;
        self.seek_pos = 0.0;
        self.status_msg = String::new();
        self.spectrum_window.on_stop();
    }

    fn next_track(&mut self) {
        if self.playlist.is_empty() { return; }
        match self.loop_mode {
            LoopMode::RepeatOne => {
                if let Some(i) = self.current_index { self.play_index(i); }
            }
            LoopMode::RepeatAll => {
                let idx = self.current_index.map(|i| (i + 1) % self.playlist.len()).unwrap_or(0);
                self.play_index(idx);
            }
            LoopMode::Sequential => {
                if let Some(i) = self.current_index {
                    if i + 1 < self.playlist.len() { self.play_index(i + 1); }
                    else { self.stop(); }
                }
            }
        }
    }

    fn prev_track(&mut self) {
        if self.playlist.is_empty() {
            return;
        }
        let idx = match self.current_index {
            Some(i) if i > 0 => i - 1,
            _ => self.playlist.len() - 1,
        };
        self.play_index(idx);
    }

    fn elapsed(&self) -> Duration {
        self.engine.as_ref().map(|e| e.elapsed()).unwrap_or(Duration::ZERO)
    }

    fn current_duration(&self) -> Option<Duration> {
        self.current_index.and_then(|i| self.playlist[i].duration)
    }

    fn format_duration(d: Duration) -> String {
        let total = d.as_secs();
        format!("{:02}:{:02}", total / 60, total % 60)
    }
}

// ---------------------------------------------------------------------------
// Playlist persistence
// ---------------------------------------------------------------------------

fn moosik_dir() -> PathBuf {
    std::env::var("USERPROFILE")
        .or_else(|_| std::env::var("HOME"))
        .map(PathBuf::from)
        .unwrap_or_else(|_| PathBuf::from("."))
        .join(".moosik")
}

#[derive(Serialize, Deserialize, Clone)]
struct SavedPlaylist {
    name: String,
    paths: Vec<PathBuf>,
}

fn save_last_playlist(tracks: &[Track]) {
    let dir = moosik_dir();
    let _ = std::fs::create_dir_all(&dir);
    let paths: Vec<&PathBuf> = tracks.iter().map(|t| &t.path).collect();
    if let Ok(json) = serde_json::to_string(&paths) {
        let _ = std::fs::write(dir.join("last_playlist.json"), json);
    }
}

fn load_last_playlist() -> Vec<PathBuf> {
    let path = moosik_dir().join("last_playlist.json");
    std::fs::read_to_string(path)
        .ok()
        .and_then(|s| serde_json::from_str(&s).ok())
        .unwrap_or_default()
}

fn save_playlist_store(playlists: &[SavedPlaylist]) {
    let dir = moosik_dir();
    let _ = std::fs::create_dir_all(&dir);
    if let Ok(json) = serde_json::to_string(playlists) {
        let _ = std::fs::write(dir.join("playlists.json"), json);
    }
}

fn load_playlist_store() -> Vec<SavedPlaylist> {
    let path = moosik_dir().join("playlists.json");
    std::fs::read_to_string(path)
        .ok()
        .and_then(|s| serde_json::from_str(&s).ok())
        .unwrap_or_default()
}

fn row(ui: &mut egui::Ui, label: &str, value: &str) {
    ui.label(egui::RichText::new(label).color(Color32::from_gray(150)).size(12.0));
    ui.label(egui::RichText::new(value).size(12.0));
    ui.end_row();
}

fn section(ui: &mut egui::Ui, heading: &str) {
    ui.separator();
    ui.label(egui::RichText::new(heading).strong().size(13.0).color(Color32::from_rgb(120, 180, 255)));
    ui.end_row();
}

fn show_track_info(ui: &mut egui::Ui, t: &Track, spectral_ceiling: Option<SpectralCeiling>, analysis: Option<&TrackAnalysis>) {
    use egui::Grid;

    let codec  = codec_name(&t.path);
    let lossy  = !is_lossless(&t.path);
    let fname  = t.path.file_name().and_then(|s| s.to_str()).unwrap_or("?");

    Grid::new("info_grid").num_columns(2).spacing([12.0, 4.0]).striped(true).show(ui, |ui| {
        // ── File ────────────────────────────────────────────────────────
        section(ui, "File");
        row(ui, "Filename", fname);
        row(ui, "Size",     &fmt_size(t.file_size));
        row(ui, "Format",   codec);
        row(ui, "Lossless", if lossy { "No (lossy)" } else { "Yes" });

        // ── Stream ──────────────────────────────────────────────────────
        section(ui, "Stream");
        row(ui, "Sample Rate", &t.sample_rate.map(|sr| format!("{} Hz  ({})", sr, fmt_hz(sr)))
                                              .unwrap_or_else(|| "Unknown".into()));
        row(ui, "Bit Depth",   &t.bit_depth.map(|b| format!("{b} bit"))
                                            .unwrap_or_else(|| "Unknown".into()));
        row(ui, "Channels",    &t.channels.map(|c| format!("{c}  ({})", channel_layout(c)))
                                           .unwrap_or_else(|| "Unknown".into()));
        row(ui, "Duration",    &t.duration.map(|d| {
                                    let s = d.as_secs();
                                    format!("{:02}:{:02}", s / 60, s % 60)
                                }).unwrap_or_else(|| "--:--".into()));
        row(ui, "Avg Bitrate", &t.bitrate.map(|b| format!("{b} kbps"))
                                          .unwrap_or_else(|| "Unknown".into()));

        // ── Tags ────────────────────────────────────────────────────────
        section(ui, "Tags");
        row(ui, "Title",    &t.title);
        row(ui, "Artist",   &t.artist);
        row(ui, "Album",    &t.album);
        row(ui, "Year",     &t.year.map(|y| y.to_string()).unwrap_or_else(|| "—".into()));
        row(ui, "Genre",    t.genre.as_deref().unwrap_or("—"));
        row(ui, "Track #",  &t.track_number.map(|n| n.to_string()).unwrap_or_else(|| "—".into()));

        // ── Inferred ────────────────────────────────────────────────────
        section(ui, "Inferred");

        // PCM throughput
        if let (Some(sr), Some(ch), Some(bd)) = (t.sample_rate, t.channels, t.bit_depth) {
            let kbps = sr as u64 * ch as u64 * bd as u64 / 1000;
            row(ui, "PCM throughput", &format!("{kbps} kbps  ({sr} × {ch} ch × {bd} bit)"));
        }

        // Nyquist
        if let Some(sr) = t.sample_rate {
            row(ui, "Nyquist limit", &format!("{} Hz", sr / 2));
        }

        // Lossless-only: uncompressed size + compression ratio
        if !lossy
            && let (Some(sr), Some(ch), Some(bd), Some(dur)) =
                (t.sample_rate, t.channels, t.bit_depth, t.duration)
        {
            let uncompressed = sr as u64 * ch as u64 * (bd as u64 / 8) * dur.as_secs();
            row(ui, "Uncompressed", &fmt_size(uncompressed));
            if uncompressed > 0 && t.file_size > 0 {
                let ratio = t.file_size as f64 / uncompressed as f64 * 100.0;
                row(ui, "Compression ratio", &format!("{ratio:.1}%  of uncompressed"));
            }
        }

        // Spectral ceiling + rolloff analysis
        match spectral_ceiling {
            None => { row(ui, "Spectral ceiling", "Analyzing…"); }
            Some(sc) => {
                let hz_str = if sc.hz >= 1000.0 { format!("{:.1} kHz", sc.hz / 1000.0) }
                             else { format!("{:.0} Hz", sc.hz) };
                row(ui, "Spectral ceiling", &hz_str);

                let rolloff_str = if sc.rolloff_octaves.is_finite() {
                    if sc.rolloff_octaves < 0.35 {
                        format!("{:.2} oct  —  steep (brick-wall)", sc.rolloff_octaves)
                    } else if sc.rolloff_octaves < 1.0 {
                        format!("{:.2} oct  —  moderate", sc.rolloff_octaves)
                    } else {
                        format!("{:.1} oct  —  gradual (natural)", sc.rolloff_octaves)
                    }
                } else {
                    "No clear cutoff  —  gradual".to_string()
                };
                row(ui, "Rolloff shape", &rolloff_str);

                // Upsampling verdict — only flag when rolloff is steep AND ceiling
                // matches a standard Nyquist AND the file's SR is higher than that.
                let file_sr = t.sample_rate.unwrap_or(0);
                let verdict = if sc.rolloff_octaves < 0.35 {
                    match sc.matched_standard_sr {
                        Some(orig_sr) if file_sr > orig_sr + orig_sr / 4 =>
                            format!("⚠ Likely upsampled from {} Hz", orig_sr),
                        Some(_) =>
                            "⚠ Steep rolloff — check source".to_string(),
                        None =>
                            "⚠ Steep rolloff at ceiling".to_string(),
                    }
                } else {
                    "OK — natural rolloff".to_string()
                };
                row(ui, "Hi-res check", &verdict);
            }
        }

        // ── Loudness & Dynamics ─────────────────────────────────────────
        if let Some(a) = analysis {
            section(ui, "Loudness & Dynamics");
            let lufs_str = if a.integrated_lufs.is_finite() {
                format!("{:.1} LUFS", a.integrated_lufs)
            } else { "—".to_string() };
            row(ui, "Integrated loudness", &lufs_str);
            row(ui, "DR score", &format!("DR {:02}", a.dr_score));
            row(ui, "Peak level", &format!("{:.1} dBFS", a.peak_dbfs));
            let clip_str = if a.clip_count == 0 {
                "None".to_string()
            } else {
                format!("{} sample{}", a.clip_count, if a.clip_count == 1 { "" } else { "s" })
            };
            row(ui, "Clipping", &clip_str);

            // ── Musical ─────────────────────────────────────────────────
            section(ui, "Musical");
            let bpm_str = if a.bpm > 0.0 { format!("{:.0} BPM", a.bpm) } else { "—".to_string() };
            row(ui, "Estimated BPM", &bpm_str);
            let key_str = if a.key_name.is_empty() { "—".to_string() } else { a.key_name.clone() };
            row(ui, "Detected key", &key_str);
        }
    });

    // ── Loudness history graph ───────────────────────────────────────────
    if let Some(a) = analysis
        && !a.loudness_history.is_empty() {
        ui.add_space(8.0);
            ui.label(egui::RichText::new("Loudness History (per second)")
                .size(12.0).color(Color32::from_rgb(120, 180, 255)));
            ui.add_space(4.0);

            let plot_h = 60.0_f32;
            let (rect, _) = ui.allocate_exact_size(
                egui::Vec2::new(ui.available_width(), plot_h),
                egui::Sense::hover(),
            );
            if ui.is_rect_visible(rect) {
                let painter = ui.painter_at(rect);
                painter.rect_filled(rect, 2.0, Color32::from_rgb(10, 12, 18));

                let history = &a.loudness_history;
                let n = history.len();
                // Display range: −70 to −5 LUFS
                let lo = -70.0f32;
                let hi = -5.0f32;
                let span = hi - lo;

                // Reference lines: −14 LUFS (streaming), −23 LUFS (broadcast)
                for &ref_lufs in &[-14.0f32, -23.0f32] {
                    let t = ((ref_lufs - lo) / span).clamp(0.0, 1.0);
                    let y = rect.bottom() - t * rect.height();
                    painter.line_segment(
                        [egui::Pos2::new(rect.left(), y), egui::Pos2::new(rect.right(), y)],
                        egui::Stroke::new(0.5, Color32::from_rgba_unmultiplied(100, 100, 255, 80)),
                    );
                    let label = format!("{ref_lufs:.0}");
                    painter.text(
                        egui::Pos2::new(rect.left() + 2.0, y - 2.0),
                        egui::Align2::LEFT_BOTTOM,
                        label,
                        egui::FontId::monospace(8.0),
                        Color32::from_rgba_unmultiplied(100, 100, 255, 140),
                    );
                }

                // Draw the LUFS line
                if n >= 2 {
                    let points: Vec<egui::Pos2> = history.iter().enumerate().map(|(i, &v)| {
                        let x = rect.left() + (i as f32 / (n - 1) as f32) * rect.width();
                        let t = ((v - lo) / span).clamp(0.0, 1.0);
                        let y = rect.bottom() - t * rect.height();
                        egui::Pos2::new(x, y)
                    }).collect();
                    painter.add(egui::Shape::line(
                        points,
                        egui::Stroke::new(1.5, Color32::from_rgb(80, 200, 120)),
                    ));
                }
        }
    }
}

impl eframe::App for MoosikApp {
    fn update(&mut self, ctx: &egui::Context, _frame: &mut eframe::Frame) {
        // --- Keyboard shortcuts (only when no text field is focused) ---
        let no_text_focus = ctx.memory(|m| m.focused().is_none());
        if no_text_focus {
            let (space, ctrl_left, ctrl_right, arrow_left, arrow_right, arrow_up, arrow_down) =
                ctx.input(|i| (
                    i.key_pressed(egui::Key::Space),
                    i.modifiers.ctrl && i.key_pressed(egui::Key::ArrowLeft),
                    i.modifiers.ctrl && i.key_pressed(egui::Key::ArrowRight),
                    !i.modifiers.ctrl && i.key_pressed(egui::Key::ArrowLeft),
                    !i.modifiers.ctrl && i.key_pressed(egui::Key::ArrowRight),
                    i.key_pressed(egui::Key::ArrowUp),
                    i.key_pressed(egui::Key::ArrowDown),
                ));

            if space       { self.toggle_play_pause(); }
            if ctrl_left   { self.prev_track(); }
            if ctrl_right  { self.next_track(); }
            if arrow_left  {
                let pos = self.elapsed().saturating_sub(Duration::from_secs(5));
                if let Some(idx) = self.current_index {
                    let path = self.playlist[idx].path.clone();
                    if let Some(ref mut engine) = self.engine { engine.seek_to(&path, pos); }
                    self.spectrum_window.on_seek(pos.as_secs_f64());
                }
            }
            if arrow_right {
                let pos = self.elapsed() + Duration::from_secs(5);
                let capped = self.current_duration().map(|d| pos.min(d)).unwrap_or(pos);
                if let Some(idx) = self.current_index {
                    let path = self.playlist[idx].path.clone();
                    if let Some(ref mut engine) = self.engine { engine.seek_to(&path, capped); }
                    self.spectrum_window.on_seek(capped.as_secs_f64());
                }
            }
            if arrow_up {
                self.volume = (self.volume + 0.05).min(1.0);
                if let Some(ref mut engine) = self.engine { engine.set_volume(self.volume); }
            }
            if arrow_down {
                self.volume = (self.volume - 0.05).max(0.0);
                if let Some(ref mut engine) = self.engine { engine.set_volume(self.volume); }
            }
        }

        // --- auto-advance when track finishes ---
        if self.play_state == PlayState::Playing {
            let finished = self.engine.as_ref().map(|e| e.is_finished()).unwrap_or(false);
            if finished {
                self.next_track();
            }
        }

        // sync volume to engine on startup
        if let Some(ref mut engine) = self.engine
            && (engine.volume - self.volume).abs() > 0.001 {
            engine.set_volume(self.volume);
        }

        // Request repaint while playing or while background analysis is running.
        // The analysis check ensures the waveform seekbar appears as soon as
        // analysis finishes even if playback is paused.
        let is_analyzing = self.spectrum_window.analyzer.is_analyzing.load(std::sync::atomic::Ordering::Relaxed);
        if self.play_state == PlayState::Playing || is_analyzing {
            let interval_ms = if self.spectrum_window.open {
                (1000.0 / self.spectrum_window.max_fps.max(1.0)) as u64
            } else {
                50 // ~20 fps is plenty for the seek bar counter alone
            };
            ctx.request_repaint_after(Duration::from_millis(interval_ms));
        }

        // Advance spectrum analyzer and render window
        let elapsed_secs = self.elapsed().as_secs_f64();
        let is_playing = self.play_state == PlayState::Playing;
        self.spectrum_window.tick(elapsed_secs, is_playing);
        // Update current art for the spectrum window
        self.spectrum_window.current_art = self.current_index
            .and_then(|i| self.playlist.get(i))
            .and_then(|t| self.art_cache.get_or_load(&t.path, ctx));
        self.spectrum_window.show(ctx);

        // --- Info window (separate OS viewport) ---
        if self.info_open
            && let Some(idx) = self.current_index {
            let track = self.playlist[idx].clone();
            let mut close = false;
            let vp_id = egui::ViewportId::from_hash_of("moosik_info");
            let vp_builder = egui::ViewportBuilder::default()
                .with_title(format!("Info — {}", track.title))
                .with_inner_size([460.0, 540.0])
                .with_resizable(true);
            ctx.show_viewport_immediate(vp_id, vp_builder, |vp_ctx, _class| {
                if vp_ctx.input(|i| i.viewport().close_requested()) {
                    close = true;
                    return;
                }
                egui::CentralPanel::default().show(vp_ctx, |ui| {
                    egui::ScrollArea::vertical().show(ui, |ui| {
                        show_track_info(ui, &track, self.spectrum_window.spectral_ceiling.clone(), self.spectrum_window.track_analysis.as_ref());
                    });
                });
            });
            if close { self.info_open = false; }
        }

        // ---------------------------------------------------------------
        // Top panel – now playing metadata
        // ---------------------------------------------------------------
        egui::TopBottomPanel::top("now_playing").min_height(70.0).show(ctx, |ui| {
            ui.add_space(8.0);
            if let Some(idx) = self.current_index {
                let track = &self.playlist[idx];
                ui.horizontal(|ui| {
                    ui.add_space(12.0);
                    ui.vertical(|ui| {
                        ui.label(RichText::new(&track.title).size(18.0).strong().color(Color32::WHITE));
                        ui.label(
                            RichText::new(format!("{} — {}", track.artist, track.album))
                                .size(13.0)
                                .color(Color32::from_gray(180)),
                        );
                    });
                    ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
                        ui.add_space(12.0);
                        let state_icon = match self.play_state {
                            PlayState::Playing => "▶ Playing",
                            PlayState::Paused => "⏸ Paused",
                            PlayState::Stopped => "⏹ Stopped",
                        };
                        ui.label(RichText::new(state_icon).size(13.0).color(Color32::from_rgb(120, 180, 255)));
                    });
                });
            } else {
                ui.add_space(4.0);
                ui.horizontal(|ui| {
                    ui.add_space(12.0);
                    ui.label(RichText::new("No track loaded").size(16.0).color(Color32::from_gray(120)));
                });
            }
            ui.add_space(8.0);
        });

        // ---------------------------------------------------------------
        // Bottom panel – controls + seek + volume
        // ---------------------------------------------------------------
        let panel_min_h = if self.spectrum_window.waveform.is_some() { 138.0 } else { 110.0 };
        egui::TopBottomPanel::bottom("controls").min_height(panel_min_h).show(ctx, |ui| {
            ui.add_space(10.0);

            // --- Seek bar (custom painter — full panel width, reliable click/drag) ---
            let elapsed = self.elapsed();
            let total = self.current_duration();
            let elapsed_str = Self::format_duration(elapsed);
            let total_str = total
                .map(Self::format_duration)
                .unwrap_or_else(|| "--:--".to_string());

            // Allocate the full available width for the seek row.
            let seek_bar_h = if self.spectrum_window.waveform.is_some() { 48.0_f32 } else { 22.0_f32 };
            let (row_rect, seek_resp) = ui.allocate_exact_size(
                Vec2::new(ui.available_width(), seek_bar_h),
                egui::Sense::click_and_drag(),
            );

            // Track geometry: leave room for time labels on each side.
            let h_pad    = 10.0_f32;
            let lbl_w    = 42.0_f32; // enough for "MM:SS" in monospace 11
            let track_x0 = row_rect.left()  + h_pad + lbl_w + 6.0;
            let track_x1 = row_rect.right() - h_pad - lbl_w - 6.0;
            let track_y  = row_rect.center().y;

            // 1. Sync seek_pos from engine when the user is not interacting.
            if !self.seeking
                && let Some(dur) = total
                && dur.as_secs_f32() > 0.0 {
                self.seek_pos =
                    (elapsed.as_secs_f32() / dur.as_secs_f32()).clamp(0.0, 1.0);
            }

            // 2. Override with pointer position while the user is dragging or clicking.
            if (seek_resp.dragged() || seek_resp.clicked())
                && let Some(ptr) = seek_resp.interact_pointer_pos() {
                let span = (track_x1 - track_x0).max(1.0);
                self.seek_pos = ((ptr.x - track_x0) / span).clamp(0.0, 1.0);
            }
            if seek_resp.dragged() {
                self.seeking = true;
            }

            // 3. Draw the track.
            let painter = ui.painter_at(row_rect);
            let fill_x = (track_x0 + self.seek_pos * (track_x1 - track_x0))
                .clamp(track_x0, track_x1);

            if let Some(wf) = self.spectrum_window.waveform.as_deref() {
                // Waveform seek bar
                let n = wf.len();
                let track_w = track_x1 - track_x0;
                let cy = track_y;
                let max_half_h = (row_rect.height() / 2.0 - 4.0).max(1.0);
                for (i, &rms) in wf.iter().enumerate() {
                    let x0 = track_x0 + (i as f32 / n as f32) * track_w;
                    let x1 = (track_x0 + ((i + 1) as f32 / n as f32) * track_w).max(x0 + 1.0);
                    let half_h = (rms * max_half_h).max(1.0);
                    let col_frac = (i as f32 + 0.5) / n as f32;
                    let color = if col_frac <= self.seek_pos {
                        Color32::from_rgb(60, 130, 220)
                    } else {
                        Color32::from_gray(50)
                    };
                    painter.rect_filled(
                        egui::Rect::from_min_max(
                            egui::Pos2::new(x0, cy - half_h),
                            egui::Pos2::new(x1, cy + half_h),
                        ),
                        0.0, color,
                    );
                }
                // Clip markers
                if let Some(ref analysis) = self.spectrum_window.track_analysis {
                    let track_w = track_x1 - track_x0;
                    for &pos in &analysis.clip_positions {
                        let x = track_x0 + pos * track_w;
                        painter.line_segment(
                            [egui::Pos2::new(x, row_rect.top() + 1.0),
                             egui::Pos2::new(x, row_rect.bottom() - 1.0)],
                            egui::Stroke::new(1.5, Color32::from_rgb(220, 60, 60)),
                        );
                    }
                }
                // Playhead line
                painter.line_segment(
                    [egui::Pos2::new(fill_x, row_rect.top() + 2.0),
                     egui::Pos2::new(fill_x, row_rect.bottom() - 2.0)],
                    egui::Stroke::new(2.0, Color32::WHITE),
                );
            } else {
                // Fallback: thin bar
                painter.rect_filled(
                    egui::Rect::from_min_max(
                        egui::Pos2::new(track_x0, track_y - 2.0),
                        egui::Pos2::new(track_x1, track_y + 2.0),
                    ),
                    2.0,
                    Color32::from_gray(65),
                );
                if fill_x > track_x0 {
                    painter.rect_filled(
                        egui::Rect::from_min_max(
                            egui::Pos2::new(track_x0, track_y - 2.0),
                            egui::Pos2::new(fill_x,   track_y + 2.0),
                        ),
                        2.0,
                        Color32::from_rgb(80, 160, 255),
                    );
                }
                let handle_r = if seek_resp.hovered() || seek_resp.dragged() { 8.0_f32 } else { 6.0_f32 };
                painter.circle_filled(egui::Pos2::new(fill_x, track_y), handle_r, Color32::WHITE);
            }
            // Time labels
            painter.text(
                egui::Pos2::new(row_rect.left() + h_pad, track_y),
                egui::Align2::LEFT_CENTER,
                &elapsed_str,
                egui::FontId::monospace(11.0),
                Color32::from_gray(190),
            );
            painter.text(
                egui::Pos2::new(row_rect.right() - h_pad, track_y),
                egui::Align2::RIGHT_CENTER,
                &total_str,
                egui::FontId::monospace(11.0),
                Color32::from_gray(190),
            );

            // 4. Commit seek on drag-release or click.
            if seek_resp.drag_stopped() || seek_resp.clicked() {
                if let (Some(dur), Some(idx)) = (total, self.current_index) {
                    let target_secs = self.seek_pos * dur.as_secs_f32();
                    let target = Duration::from_secs_f32(target_secs);
                    let path = self.playlist[idx].path.clone();
                    if let Some(ref mut engine) = self.engine {
                        engine.seek_to(&path, target);
                    }
                    self.spectrum_window.on_seek(target_secs as f64);
                }
                self.seeking = false;
            }

            ui.add_space(6.0);

            // --- Playback buttons + volume ---
            ui.horizontal(|ui| {
                ui.add_space(10.0);

                let btn = |label: &str| egui::Button::new(RichText::new(label).size(18.0)).min_size(Vec2::new(40.0, 36.0));

                if ui.add(btn("⏮")).clicked() {
                    self.prev_track();
                }

                let play_label = if self.play_state == PlayState::Playing { "⏸" } else { "▶" };
                if ui.add(btn(play_label)).clicked() {
                    self.toggle_play_pause();
                }

                if ui.add(btn("⏹")).clicked() {
                    self.stop();
                }

                if ui.add(btn("⏭")).clicked() {
                    self.next_track();
                }

                ui.add_space(8.0);
                let loop_icon = match self.loop_mode {
                    LoopMode::Sequential => "➡",
                    LoopMode::RepeatAll  => "🔁",
                    LoopMode::RepeatOne  => "🔂",
                };
                let loop_tip = match self.loop_mode {
                    LoopMode::Sequential => "Sequential (click for Repeat All)",
                    LoopMode::RepeatAll  => "Repeat All (click for Repeat One)",
                    LoopMode::RepeatOne  => "Repeat One (click for Sequential)",
                };
                if ui.add(btn(loop_icon)).on_hover_text(loop_tip).clicked() {
                    self.loop_mode = match self.loop_mode {
                        LoopMode::Sequential => LoopMode::RepeatAll,
                        LoopMode::RepeatAll  => LoopMode::RepeatOne,
                        LoopMode::RepeatOne  => LoopMode::Sequential,
                    };
                }

                ui.add_space(20.0);

                // Volume
                ui.label(RichText::new("🔊").size(16.0));
                let vol_slider = ui.add(
                    Slider::new(&mut self.volume, 0.0..=1.0)
                        .show_value(false)
                        .trailing_fill(true),
                );
                if vol_slider.changed()
                    && let Some(ref mut engine) = self.engine {
                    engine.set_volume(self.volume);
                }
                ui.label(
                    RichText::new(format!("{}%", (self.volume * 100.0) as u32))
                        .size(12.0)
                        .color(Color32::from_gray(180)),
                );

                // Sample rate + PCM bitrate indicator
                if self.play_state != PlayState::Stopped {
                    let sr = self.engine.as_ref().map(|e| e.last_sample_rate).unwrap_or(0);
                    if sr > 0 {
                        let pcm = self.current_index.and_then(|i| {
                            let t = &self.playlist[i];
                            Some(sr * t.channels? as u32 * t.bit_depth? as u32 / 1000)
                        });
                        let label = match pcm {
                            Some(kbps) => format!("{}  ·  {}kbps", fmt_hz(sr), kbps),
                            None       => fmt_hz(sr),
                        };
                        ui.add_space(8.0);
                        ui.label(RichText::new(label).size(11.0).color(Color32::from_gray(140)));
                    }
                }

                if self.play_state != PlayState::Stopped {
                    let lufs = self.spectrum_window.momentary_lufs;
                    if lufs.is_finite() {
                        ui.add_space(6.0);
                        ui.label(RichText::new(format!("{:.1} LUFS", lufs))
                            .size(11.0).color(Color32::from_rgb(120, 210, 140)));
                    }
                }

                // Open files button pushed right
                ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
                    ui.add_space(10.0);
                    if ui.button(RichText::new("+ Add Files").size(13.0)).clicked() {
                        self.add_files();
                    }
                    ui.add_space(4.0);
                    if ui.button(RichText::new("📁 Load Folder").size(13.0)).clicked() {
                        self.add_folder();
                    }
                    ui.add_space(8.0);
                    let spectrum_label = if self.spectrum_window.open {
                        RichText::new("📊 Spectrum").size(13.0).color(Color32::from_rgb(100, 200, 255))
                    } else {
                        RichText::new("📊 Spectrum").size(13.0)
                    };
                    if ui.button(spectrum_label).clicked() {
                        self.spectrum_window.open = !self.spectrum_window.open;
                    }
                    ui.add_space(8.0);
                    let info_enabled = self.current_index.is_some();
                    let info_label = if self.info_open {
                        RichText::new("ℹ Info").size(13.0).color(Color32::from_rgb(100, 200, 255))
                    } else {
                        RichText::new("ℹ Info").size(13.0)
                    };
                    if ui.add_enabled(info_enabled, egui::Button::new(info_label)).clicked() {
                        self.info_open = !self.info_open;
                    }
                });
            });

            if !self.status_msg.is_empty() {
                ui.add_space(4.0);
                ui.horizontal(|ui| {
                    ui.add_space(10.0);
                    ui.label(RichText::new(&self.status_msg).size(11.0).color(Color32::from_gray(140)));
                });
            }

            ui.add_space(6.0);
        });

        // ---------------------------------------------------------------
        // Central panel – playlist store + playlist
        // ---------------------------------------------------------------
        egui::CentralPanel::default().show(ctx, |ui| {
            // ── Playlist store bar ──────────────────────────────────────
            ui.add_space(4.0);
            ui.horizontal(|ui| {
                ui.add_space(8.0);
                let selected_text = self.active_saved_playlist
                    .and_then(|i| self.playlist_store.get(i))
                    .map(|p| p.name.as_str())
                    .unwrap_or("— unsaved —")
                    .to_string();
                egui::ComboBox::from_id_salt("pl_combo")
                    .selected_text(&selected_text)
                    .width(150.0)
                    .show_ui(ui, |ui| {
                        if ui.selectable_label(self.active_saved_playlist.is_none(), "— unsaved —").clicked() {
                            self.active_saved_playlist = None;
                        }
                        for i in 0..self.playlist_store.len() {
                            let name = self.playlist_store[i].name.clone();
                            let is_sel = self.active_saved_playlist == Some(i);
                            if ui.selectable_label(is_sel, &name).clicked() {
                                let paths = self.playlist_store[i].paths.clone();
                                self.stop();
                                self.playlist = paths.into_iter().map(Track::load).collect();
                                self.current_index = None;
                                self.selected.clear();
                                self.last_clicked = None;
                                self.active_saved_playlist = Some(i);
                                self.show_save_playlist_input = false;
                            }
                        }
                    });
                ui.add_space(4.0);
                if let Some(idx) = self.active_saved_playlist {
                    if !self.playlist.is_empty()
                        && ui.small_button("🔄 Update").on_hover_text("Overwrite saved playlist").clicked()
                    {
                        let paths: Vec<PathBuf> = self.playlist.iter().map(|t| t.path.clone()).collect();
                        self.playlist_store[idx].paths = paths;
                        save_playlist_store(&self.playlist_store);
                    }
                    ui.add_space(2.0);
                    if ui.small_button(RichText::new("🗑").color(Color32::from_rgb(200, 80, 80)))
                        .on_hover_text("Delete saved playlist").clicked()
                    {
                        self.playlist_store.remove(idx);
                        self.active_saved_playlist = None;
                        save_playlist_store(&self.playlist_store);
                    }
                    ui.add_space(4.0);
                }
                if !self.playlist.is_empty() {
                    let btn_label = if self.show_save_playlist_input { "✕" } else { "💾 Save As" };
                    if ui.small_button(btn_label).clicked() {
                        self.show_save_playlist_input = !self.show_save_playlist_input;
                        if self.show_save_playlist_input && self.playlist_name_buf.is_empty() {
                            self.playlist_name_buf = "My Playlist".to_string();
                        }
                    }
                }
            });

            if self.show_save_playlist_input {
                let mut do_save = false;
                let mut do_cancel = false;
                ui.horizontal(|ui| {
                    ui.add_space(8.0);
                    ui.label("Name:");
                    let te = ui.add(
                        egui::TextEdit::singleline(&mut self.playlist_name_buf).desired_width(140.0)
                    );
                    if te.lost_focus() && ui.input(|i| i.key_pressed(egui::Key::Enter)) {
                        do_save = true;
                    }
                    if ui.small_button("Save").clicked() { do_save = true; }
                    if ui.small_button("Cancel").clicked() { do_cancel = true; }
                });
                if do_save {
                    let name = self.playlist_name_buf.trim().to_string();
                    if !name.is_empty() {
                        let paths = self.playlist.iter().map(|t| t.path.clone()).collect();
                        self.playlist_store.push(SavedPlaylist { name, paths });
                        self.active_saved_playlist = Some(self.playlist_store.len() - 1);
                        save_playlist_store(&self.playlist_store);
                        self.show_save_playlist_input = false;
                        self.playlist_name_buf.clear();
                    }
                }
                if do_cancel {
                    self.show_save_playlist_input = false;
                }
            }

            ui.add_space(2.0);

            // ── Playlist header ─────────────────────────────────────────
            let n_sel = self.selected.len();
            ui.horizontal(|ui| {
                ui.add_space(8.0);
                ui.label(RichText::new(format!("Playlist  ({} tracks)", self.playlist.len()))
                    .size(13.0).color(Color32::from_gray(160)));
                ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
                    ui.add_space(8.0);
                    if !self.playlist.is_empty()
                        && ui.small_button(RichText::new("Clear All").color(Color32::from_rgb(200, 80, 80))).clicked()
                    {
                        self.stop();
                        self.playlist.clear();
                        self.current_index = None;
                        self.selected.clear();
                        self.active_saved_playlist = None;
                    }
                    if n_sel > 0 {
                        ui.add_space(4.0);
                        let del_label = format!("Delete {n_sel} selected");
                        if ui.small_button(RichText::new(del_label).color(Color32::from_rgb(220, 100, 60))).clicked() {
                            let current_removed = self.current_index
                                .map(|ci| self.selected.contains(&ci)).unwrap_or(false);
                            let mut to_remove: Vec<usize> = self.selected.drain().collect();
                            to_remove.sort_unstable_by(|a, b| b.cmp(a)); // descending: safe in-place removal
                            for &idx in &to_remove {
                                self.playlist.remove(idx);
                            }
                            if current_removed {
                                self.stop();
                                self.current_index = None;
                            } else if let Some(ci) = self.current_index {
                                let removed_before = to_remove.iter().filter(|&&r| r < ci).count();
                                self.current_index = Some(ci - removed_before);
                            }
                            self.last_clicked = None;
                        }
                    }
                });
            });
            ui.separator();

            // ── Drag / selection tracking ────────────────────────────────
            let dragging = self.drag_src.is_some();
            let pointer_released = ctx.input(|i| i.pointer.primary_released());
            let pointer_pos = ctx.input(|i| i.pointer.hover_pos());
            let ctrl_held = ctx.input(|i| i.modifiers.ctrl);
            let shift_held = ctx.input(|i| i.modifiers.shift);

            let mut play_requested: Option<usize> = None;
            let mut drag_started_at: Option<usize> = None;
            let mut click_action: Option<(usize, bool, bool)> = None;

            let n = self.playlist.len();
            let mut new_drop_row = n;
            let mut last_row_bottom: Option<(f32, f32, f32)> = None; // (left, right, y)

            egui::ScrollArea::vertical().auto_shrink([false, false]).show(ui, |ui| {
                if self.playlist.is_empty() {
                    ui.add_space(40.0);
                    ui.vertical_centered(|ui| {
                        ui.label(RichText::new("No tracks loaded").size(15.0).color(Color32::from_gray(100)));
                        ui.add_space(8.0);
                        ui.label(RichText::new("Click \"+ Add Files\" to get started").size(12.0).color(Color32::from_gray(80)));
                    });
                    return;
                }

                let available_w = ui.available_width();

                for i in 0..n {
                    let track_title  = self.playlist[i].display_title().to_string();
                    let track_artist = self.playlist[i].artist.clone();
                    let track_dur    = self.playlist[i].duration;
                    let is_current        = self.current_index == Some(i);
                    let is_selected       = self.selected.contains(&i);
                    let is_being_dragged  = self.drag_src == Some(i);

                    let base_color = if is_selected {
                        Color32::from_rgb(40, 70, 130)
                    } else if is_current {
                        Color32::from_rgb(30, 60, 100)
                    } else if i % 2 == 0 {
                        Color32::from_gray(28)
                    } else {
                        Color32::from_gray(22)
                    };

                    let (rect, response) = ui.allocate_exact_size(
                        Vec2::new(available_w, 32.0),
                        egui::Sense::click_and_drag(),
                    );

                    // Determine drop position: first row whose center is below pointer
                    if dragging
                        && let Some(pp) = pointer_pos
                        && pp.y <= rect.center().y && new_drop_row == n {
                        new_drop_row = i;
                    }

                    // Insertion line before this row
                    if dragging {
                        let src = self.drag_src;
                        let no_op = src == Some(i) || src.map(|s| s + 1) == Some(i);
                        if self.drag_over_row == Some(i) && !no_op {
                            ui.painter().line_segment(
                                [egui::Pos2::new(rect.left(), rect.top()),
                                 egui::Pos2::new(rect.right(), rect.top())],
                                egui::Stroke::new(2.0, Color32::from_rgb(100, 180, 255)),
                            );
                        }
                    }

                    if ui.is_rect_visible(rect) {
                        let alpha = if is_being_dragged { 80u8 } else { 255u8 };
                        let c = base_color;
                        ui.painter().rect_filled(
                            rect, 0.0,
                            Color32::from_rgba_unmultiplied(c.r(), c.g(), c.b(), alpha),
                        );

                        let inner = rect.shrink2(Vec2::new(10.0, 0.0));
                        let cy = rect.center().y;

                        // Drag handle — 2×3 dot grid, no font dependency
                        for col in [0.0f32, 4.0] {
                            for row in [-4.0f32, 0.0, 4.0] {
                                ui.painter().circle_filled(
                                    egui::Pos2::new(inner.min.x + 4.0 + col, cy + row),
                                    1.5,
                                    Color32::from_gray(75),
                                );
                            }
                        }

                        // ── Album art thumbnail ───────────────────────────
                        let show_art = self.spectrum_window.art_settings.playlist_show;
                        let art_offset = if show_art { 34.0f32 } else { 0.0 };
                        if show_art {
                            let thumb_size = 28.0;
                            let thumb_x = inner.min.x + 14.0;
                            let thumb_y = cy - thumb_size / 2.0;
                            let thumb_rect = egui::Rect::from_min_size(
                                egui::Pos2::new(thumb_x, thumb_y),
                                Vec2::splat(thumb_size),
                            );
                            let track_path = &self.playlist[i].path;
                            let art = self.art_cache.get_or_load(track_path, ctx);
                            if let Some((tex_id, art_w, art_h)) = art {
                                let art_rect = fit_rect_preserve(thumb_rect, art_w, art_h);
                                ui.painter().image(
                                    tex_id, art_rect,
                                    egui::Rect::from_min_max(egui::Pos2::ZERO, egui::Pos2::new(1.0, 1.0)),
                                    Color32::WHITE,
                                );

                                // Hover detection — show full art after 1s
                                let over = ctx.pointer_hover_pos()
                                    .map_or(false, |p| thumb_rect.contains(p));
                                if over {
                                    match self.art_hover {
                                        Some((idx, _)) if idx == i => {}
                                        _ => self.art_hover = Some((i, Instant::now())),
                                    }
                                    // Keep repainting so the timer fires without needing mouse movement
                                    ctx.request_repaint();
                                } else if self.art_hover.map_or(false, |(idx, _)| idx == i) {
                                    self.art_hover = None;
                                }

                                if let Some((hi, since)) = self.art_hover {
                                    if hi == i && since.elapsed().as_secs_f32() >= 1.0 {
                                        let max_dim = 512.0f32;
                                        let aspect = art_w as f32 / art_h.max(1) as f32;
                                        let (pw, ph) = if aspect >= 1.0 {
                                            (max_dim, max_dim / aspect)
                                        } else {
                                            (max_dim * aspect, max_dim)
                                        };
                                        let screen = ctx.screen_rect();
                                        let px = (thumb_rect.right() + 8.0)
                                            .min(screen.right() - pw - 4.0);
                                        let py = (thumb_rect.center().y - ph / 2.0)
                                            .clamp(screen.top() + 4.0, screen.bottom() - ph - 4.0);
                                        egui::Area::new(egui::Id::new("art_hover_popup"))
                                            .fixed_pos(egui::Pos2::new(px, py))
                                            .order(egui::Order::Tooltip)
                                            .interactable(false)
                                            .show(ctx, |ui| {
                                                let (_, r) = ui.allocate_space(Vec2::new(pw, ph));
                                                ui.painter().rect_filled(r.expand(2.0), 3.0, Color32::from_black_alpha(100));
                                                ui.painter().image(
                                                    tex_id, r,
                                                    egui::Rect::from_min_max(egui::Pos2::ZERO, egui::Pos2::new(1.0, 1.0)),
                                                    Color32::WHITE,
                                                );
                                            });
                                    }
                                }
                            } else if self.spectrum_window.art_settings.playlist_placeholder {
                                ui.painter().rect_filled(thumb_rect, 3.0, Color32::from_gray(35));
                                ui.painter().text(
                                    thumb_rect.center(),
                                    egui::Align2::CENTER_CENTER,
                                    "♪",
                                    egui::FontId::proportional(14.0),
                                    Color32::from_gray(70),
                                );
                            }
                        }

                        // Track number
                        ui.painter().text(
                            egui::Pos2::new(inner.min.x + 24.0 + art_offset, cy),
                            egui::Align2::CENTER_CENTER,
                            format!("{}", i + 1),
                            egui::FontId::proportional(12.0),
                            if is_current { Color32::from_rgb(120, 180, 255) } else { Color32::from_gray(100) },
                        );

                        // Title
                        let title_x = inner.min.x + 42.0 + art_offset;
                        let title_w = (inner.width() * 0.45 - art_offset).max(0.0);
                        ui.painter().text(
                            egui::Pos2::new(title_x, cy),
                            egui::Align2::LEFT_CENTER,
                            &track_title,
                            egui::FontId::proportional(13.0),
                            if is_current { Color32::WHITE } else { Color32::from_gray(210) },
                        );

                        // Artist
                        ui.painter().text(
                            egui::Pos2::new(title_x + title_w + 8.0, cy),
                            egui::Align2::LEFT_CENTER,
                            &track_artist,
                            egui::FontId::proportional(12.0),
                            Color32::from_gray(150),
                        );

                        // Duration
                        if let Some(dur) = track_dur {
                            ui.painter().text(
                                egui::Pos2::new(inner.max.x - 8.0, cy),
                                egui::Align2::RIGHT_CENTER,
                                Self::format_duration(dur),
                                egui::FontId::monospace(12.0),
                                Color32::from_gray(120),
                            );
                        }

                        if response.hovered() && !is_being_dragged {
                            ui.painter().rect_filled(rect, 0.0, Color32::from_rgba_unmultiplied(255, 255, 255, 8));
                        }
                    }

                    // Drag start
                    if response.drag_started() && !dragging {
                        drag_started_at = Some(i);
                    }

                    // Click (only when not starting a drag)
                    if !dragging && !response.drag_started() {
                        if response.double_clicked() {
                            play_requested = Some(i);
                        } else if response.clicked() {
                            click_action = Some((i, ctrl_held, shift_held));
                        }
                    }

                    last_row_bottom = Some((rect.left(), rect.right(), rect.bottom()));
                }

                // Insertion line at end of list
                if dragging
                    && let (Some((rx0, rx1, ry)), Some(drop)) = (last_row_bottom, self.drag_over_row) {
                    let src = self.drag_src;
                    let no_op = src.map(|s| s + 1) == Some(n);
                    if drop == n && !no_op {
                        ui.painter().line_segment(
                            [egui::Pos2::new(rx0, ry), egui::Pos2::new(rx1, ry)],
                            egui::Stroke::new(2.0, Color32::from_rgb(100, 180, 255)),
                        );
                    }
                }
            });

            // ── Apply drag start ─────────────────────────────────────────
            if let Some(i) = drag_started_at {
                self.drag_src = Some(i);
                self.selected.clear();
            }
            if dragging {
                self.drag_over_row = Some(new_drop_row);
            }

            // ── Finalize drag on release ──────────────────────────────────
            if dragging && pointer_released {
                let src = self.drag_src.take().unwrap_or(0);
                let dst = self.drag_over_row.take().unwrap_or(n);
                if src != dst && src + 1 != dst && dst <= n {
                    let track = self.playlist.remove(src);
                    let insert_at = if dst > src { dst - 1 } else { dst };
                    self.playlist.insert(insert_at, track);
                    self.current_index = self.current_index.map(|ci| {
                        if ci == src              { insert_at }
                        else if src < ci && ci < dst { ci - 1 }
                        else if dst <= ci && ci < src { ci + 1 }
                        else                      { ci }
                    });
                }
            }

            // ── Apply click selection ─────────────────────────────────────
            if let Some(idx) = play_requested {
                self.play_index(idx);
            }
            if let Some((i, ctrl, shift)) = click_action {
                if ctrl {
                    if self.selected.contains(&i) { self.selected.remove(&i); }
                    else { self.selected.insert(i); }
                    self.last_clicked = Some(i);
                } else if shift {
                    let anchor = self.last_clicked.unwrap_or(i);
                    let (lo, hi) = if anchor <= i { (anchor, i) } else { (i, anchor) };
                    for j in lo..=hi { self.selected.insert(j); }
                    self.last_clicked = Some(i);
                } else {
                    self.selected.clear();
                    self.selected.insert(i);
                    self.last_clicked = Some(i);
                }
            }
        });
    }

    fn on_exit(&mut self, _gl: Option<&eframe::glow::Context>) {
        save_last_playlist(&self.playlist);
        self.spectrum_window.art_settings.save();
    }
}
