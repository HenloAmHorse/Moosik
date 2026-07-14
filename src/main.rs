mod bitperfect;
mod media_controls;
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

/// The single source of colour truth, all derived from the app's identity:
/// the icon's Eigengrau base (#16161d) and Kugelblitz accent (#94b1ff). Every
/// widget-level surface comes from `apply_theme`; these are the handful of
/// custom-painted spots (seek bar, playlist rows, status text) that paint
/// outside egui's widget system and so need explicit colours.
mod pal {
    use egui::Color32;

    /// Kugelblitz — the brand accent (a perfectly flat spectrum's blue). This is
    /// the fixed identity colour used as the album-art blend target and the
    /// dark-theme accent; use `accent(dark)` for on-screen text/strokes so the
    /// light theme gets a deeper, higher-contrast blue.
    pub const ACCENT: Color32 = Color32::from_rgb(0x94, 0xb1, 0xff);

    /// Accent for UI text / strokes / selection, tuned per theme for contrast.
    pub fn accent(dark: bool) -> Color32 {
        if dark { ACCENT } else { Color32::from_rgb(0x3a, 0x55, 0xc2) }
    }

    // Text tiers for custom-painted labels (playlist rows, now-playing, seek
    // times). egui's own widgets pick up the right colour from the theme, but
    // these are painted by hand and so must resolve per theme — otherwise the
    // dark-theme light greys / white are near-invisible on the light background.
    /// Strongest text (titles, now-playing) — white on dark, near-black on light.
    pub fn text_strong(dark: bool) -> Color32 {
        if dark { Color32::WHITE } else { Color32::from_rgb(0x12, 0x13, 0x18) }
    }
    /// Primary body text.
    pub fn text(dark: bool) -> Color32 {
        if dark { Color32::from_gray(210) } else { Color32::from_rgb(0x25, 0x27, 0x30) }
    }
    /// Secondary / dimmer text (artist, subtitle).
    pub fn text_dim(dark: bool) -> Color32 {
        if dark { Color32::from_gray(150) } else { Color32::from_rgb(0x55, 0x58, 0x63) }
    }
    /// Faint text (track number, duration, hints).
    pub fn text_faint(dark: bool) -> Color32 {
        if dark { Color32::from_gray(120) } else { Color32::from_rgb(0x74, 0x77, 0x82) }
    }
    /// Positive / active green text (bit-perfect, ReplayGain on, LUFS) — deepened
    /// on light so it isn't a pale mint on white.
    pub fn ok(dark: bool) -> Color32 {
        if dark { Color32::from_rgb(120, 230, 170) } else { Color32::from_rgb(0x12, 0x7a, 0x4c) }
    }

    // Playlist rows — tinted neutrals in each theme's panel ramp, plus
    // accent-tinted states so the current/selected track reads as "lit".
    pub fn row_even(dark: bool) -> Color32 {
        if dark { Color32::from_rgb(0x1b, 0x1c, 0x25) } else { Color32::from_rgb(0xf2, 0xf3, 0xf8) }
    }
    pub fn row_odd(dark: bool) -> Color32 {
        if dark { Color32::from_rgb(0x16, 0x17, 0x1f) } else { Color32::from_rgb(0xe8, 0xea, 0xf1) }
    }
    pub fn row_current(dark: bool) -> Color32 {
        if dark { Color32::from_rgb(0x27, 0x31, 0x4e) } else { Color32::from_rgb(0xcd, 0xd9, 0xf5) }
    }
    pub fn row_selected(dark: bool) -> Color32 {
        if dark { Color32::from_rgb(0x30, 0x40, 0x66) } else { Color32::from_rgb(0xb6, 0xc7, 0xef) }
    }

    // Seek bar.
    pub fn track_bg(dark: bool) -> Color32 {
        if dark { Color32::from_rgb(0x2a, 0x2d, 0x3a) } else { Color32::from_rgb(0xd4, 0xd7, 0xe2) }
    }
    pub fn wave_unplayed(dark: bool) -> Color32 {
        if dark { Color32::from_rgb(0x39, 0x3d, 0x4d) } else { Color32::from_rgb(0xbc, 0xc0, 0xce) }
    }

    // Semantic (kept distinct from the accent on purpose).
    pub fn warn(dark: bool) -> Color32 {
        if dark { Color32::from_rgb(0xe0, 0x6c, 0x5c) } else { Color32::from_rgb(0xbf, 0x39, 0x2b) }
    }
    pub fn amber(dark: bool) -> Color32 {
        if dark { Color32::from_rgb(0xd9, 0xa8, 0x5c) } else { Color32::from_rgb(0x9c, 0x6f, 0x1e) }
    }
    pub fn muted(dark: bool) -> Color32 {
        if dark { Color32::from_gray(0x78) } else { Color32::from_gray(0x80) }
    }
}

/// Darken a (typically bright, cover-derived) accent so it reads against the
/// light theme's pale chrome, preserving hue.
fn dim_for_light(c: egui::Color32) -> egui::Color32 {
    let m = |x: u8| (x as f32 * 0.60) as u8;
    egui::Color32::from_rgb(m(c.r()), m(c.g()), m(c.b()))
}

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

/// A cohesive theme built from the app's own identity colors: the icon's
/// **Eigengrau** (#16161d) background and **Kugelblitz** (#94b1ff) accent. Softer
/// brand-tinted surfaces, gently rounded corners, and accent-coloured selection
/// and hover — a calmer look than egui's flat default grey. `dark` selects the
/// dark (default) or light variant; both share the accent and layout.
fn apply_theme(ctx: &egui::Context, dark: bool) {
    use egui::{Color32, CornerRadius, Stroke};

    let accent = pal::accent(dark);

    // A short ramp of tinted neutrals: dark climbs from the Eigengrau base,
    // light descends from near-white.
    let (bg0, bg1, bg2, bg3, bg4, bg5, line, faint) = if dark {
        (
            Color32::from_rgb(0x14, 0x15, 0x1c), // deepest (text fields, wells)
            Color32::from_rgb(0x18, 0x19, 0x21), // panels
            Color32::from_rgb(0x1e, 0x20, 0x2a), // windows / surfaces
            Color32::from_rgb(0x25, 0x27, 0x33), // resting widgets
            Color32::from_rgb(0x30, 0x33, 0x43), // hovered widgets
            Color32::from_rgb(0x3a, 0x3e, 0x52), // pressed widgets
            Color32::from_rgb(0x2a, 0x2c, 0x39), // hairline separators
            Color32::from_rgb(0x1c, 0x1d, 0x26), // faint
        )
    } else {
        (
            Color32::from_rgb(0xfb, 0xfb, 0xfd),
            Color32::from_rgb(0xec, 0xed, 0xf2),
            Color32::from_rgb(0xf4, 0xf5, 0xf9),
            Color32::from_rgb(0xe1, 0xe3, 0xea),
            Color32::from_rgb(0xd3, 0xd6, 0xe1),
            Color32::from_rgb(0xc2, 0xc6, 0xd5),
            Color32::from_rgb(0xd0, 0xd3, 0xdd),
            Color32::from_rgb(0xe7, 0xe8, 0xef),
        )
    };

    let mut style = (*ctx.style()).clone();
    // Start from egui's matching base so text colours / fg strokes are sensible,
    // then override the surfaces and accent.
    style.visuals = if dark { egui::Visuals::dark() } else { egui::Visuals::light() };
    let v = &mut style.visuals;
    v.dark_mode = dark;

    v.panel_fill = bg1;
    v.window_fill = bg2;
    v.extreme_bg_color = bg0;
    v.faint_bg_color = faint;
    v.window_stroke = Stroke::new(1.0, line);
    v.window_corner_radius = CornerRadius::same(9);
    v.menu_corner_radius = CornerRadius::same(7);

    // Selection + hyperlinks carry the accent.
    v.selection.bg_fill = if dark {
        accent.gamma_multiply(0.30)
    } else {
        Color32::from_rgb(0xc4, 0xd3, 0xf3)
    };
    v.selection.stroke = Stroke::new(1.0, accent);
    v.hyperlink_color = accent;

    let r = CornerRadius::same(6);
    let w = &mut v.widgets;
    w.noninteractive.corner_radius = r;
    w.noninteractive.bg_fill = bg2;
    w.noninteractive.bg_stroke = Stroke::new(1.0, line);

    w.inactive.corner_radius = r;
    w.inactive.bg_fill = bg3;
    w.inactive.weak_bg_fill = bg3;
    w.inactive.bg_stroke = Stroke::new(1.0, if dark {
        Color32::from_rgb(0x2f, 0x31, 0x3f)
    } else {
        Color32::from_rgb(0xcb, 0xce, 0xd8)
    });

    w.hovered.corner_radius = r;
    w.hovered.bg_fill = bg4;
    w.hovered.weak_bg_fill = bg4;
    w.hovered.bg_stroke = Stroke::new(1.0, accent.gamma_multiply(0.55));

    w.active.corner_radius = r;
    w.active.bg_fill = bg5;
    w.active.weak_bg_fill = bg5;
    w.active.bg_stroke = Stroke::new(1.0, accent);

    w.open.corner_radius = r;
    w.open.bg_fill = bg3;

    // A touch more breathing room, without disturbing the compact layout.
    style.spacing.item_spacing = egui::vec2(8.0, 5.0);
    style.spacing.button_padding = egui::vec2(7.0, 3.0);

    ctx.set_style(style);
}

/// Raise the Windows system timer resolution to 1 ms. The default (~15.6 ms)
/// would clamp the frame limiter's `thread::sleep` to ~64 fps; 1 ms keeps it
/// accurate up to high-refresh rates. Windows restores the default on process
/// exit, so no matching `timeEndPeriod` is needed.
#[cfg(windows)]
fn raise_timer_resolution() {
    unsafe { windows_sys::Win32::Media::timeBeginPeriod(1); }
}
#[cfg(not(windows))]
fn raise_timer_resolution() {}

/// The primary monitor's current refresh rate in Hz, if it can be determined.
/// Used to default the spectrum window's Max FPS so the animation runs as
/// smoothly as the display allows without the user having to bump it manually.
#[cfg(windows)]
fn monitor_refresh_hz() -> Option<f32> {
    use windows_sys::Win32::Graphics::Gdi::{
        EnumDisplaySettingsW, DEVMODEW, ENUM_CURRENT_SETTINGS,
    };
    unsafe {
        let mut dm: DEVMODEW = std::mem::zeroed();
        dm.dmSize = std::mem::size_of::<DEVMODEW>() as u16;
        if EnumDisplaySettingsW(std::ptr::null(), ENUM_CURRENT_SETTINGS, &mut dm) != 0 {
            let hz = dm.dmDisplayFrequency;
            // 0 or 1 mean "hardware default / unspecified" per the Win32 docs.
            if hz > 1 {
                return Some(hz as f32);
            }
        }
    }
    None
}
#[cfg(not(windows))]
fn monitor_refresh_hz() -> Option<f32> { None }

/// Record panics to `~/.moosik/crash.log` (and still print to stderr). The
/// release profile uses `panic = "abort"`, so a panic on *any* thread (decode,
/// output, seek worker) takes the whole process down with no visible stack —
/// this leaves a durable breadcrumb of the message, location, and thread so a
/// "crash" can be diagnosed from evidence instead of guessed at.
fn install_crash_logger() {
    let default = std::panic::take_hook();
    std::panic::set_hook(Box::new(move |info| {
        let loc = info.location()
            .map(|l| format!("{}:{}:{}", l.file(), l.line(), l.column()))
            .unwrap_or_else(|| "?".into());
        let msg = info.payload().downcast_ref::<&str>().map(|s| (*s).to_string())
            .or_else(|| info.payload().downcast_ref::<String>().cloned())
            .unwrap_or_else(|| "<non-string panic payload>".into());
        let thread = std::thread::current().name().unwrap_or("unnamed").to_string();
        let secs = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .map(|d| d.as_secs())
            .unwrap_or(0);
        let dir = moosik_dir();
        let _ = std::fs::create_dir_all(&dir);
        if let Ok(mut f) = std::fs::OpenOptions::new()
            .create(true).append(true).open(dir.join("crash.log"))
        {
            use std::io::Write;
            let _ = writeln!(f, "[{secs}] panic on '{thread}' at {loc}: {msg}");
        }
        default(info); // keep the normal stderr output
    }));
}

fn main() -> eframe::Result {
    raise_timer_resolution();
    install_crash_logger();
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
    // ReplayGain tags (dB gain, linear peak), if present in the file.
    rg_track_gain: Option<f32>,
    rg_album_gain: Option<f32>,
    rg_track_peak: Option<f32>,
    rg_album_peak: Option<f32>,
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
            rg_track_gain: m.rg_track_gain, rg_album_gain: m.rg_album_gain,
            rg_track_peak: m.rg_track_peak, rg_album_peak: m.rg_album_peak,
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
    rg_track_gain: Option<f32>, rg_album_gain: Option<f32>,
    rg_track_peak: Option<f32>, rg_album_peak: Option<f32>,
}

/// Parse a ReplayGain gain string ("-7.30 dB", "+2.4", …) to dB.
fn parse_rg_gain(s: &str) -> Option<f32> {
    s.trim().trim_end_matches("dB").trim_end_matches("DB").trim().parse::<f32>().ok()
}

/// Parse a ReplayGain peak string (linear sample peak, e.g. "0.988553").
fn parse_rg_peak(s: &str) -> Option<f32> {
    s.trim().parse::<f32>().ok().filter(|p| *p > 0.0)
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

    // ReplayGain tags (Vorbis comment / ID3 TXXX / iTunes atoms — lofty
    // normalises them all to these ItemKeys).
    let (rg_track_gain, rg_album_gain, rg_track_peak, rg_album_peak) = tagged
        .as_ref()
        .and_then(|t| t.primary_tag().or_else(|| t.first_tag()))
        .map(|tag| {
            use lofty::tag::ItemKey;
            (
                tag.get_string(&ItemKey::ReplayGainTrackGain).and_then(parse_rg_gain),
                tag.get_string(&ItemKey::ReplayGainAlbumGain).and_then(parse_rg_gain),
                tag.get_string(&ItemKey::ReplayGainTrackPeak).and_then(parse_rg_peak),
                tag.get_string(&ItemKey::ReplayGainAlbumPeak).and_then(parse_rg_peak),
            )
        })
        .unwrap_or((None, None, None, None));

    TrackMeta { title, artist, album, year, genre, track_number,
                duration, sample_rate, channels, bit_depth, bitrate, file_size,
                rg_track_gain, rg_album_gain, rg_track_peak, rg_album_peak }
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

/// A rodio slow-path seek (FLAC decode-and-discard) running on a background
/// thread, so the UI never blocks while millions of samples are skipped.
struct PendingSeek {
    rx: std::sync::mpsc::Receiver<Option<Decoder<BufReader<File>>>>,
    target: Duration,
    was_paused: bool,
}

struct Engine {
    // ── rodio path (normal mode) ───────────────────────────────────────────
    _stream: OutputStream,
    stream_handle: OutputStreamHandle,
    sink: Option<Sink>,
    // ── bit-perfect path (direct cpal stream fed by a symphonia thread) ────
    bp: Option<bitperfect::BpStream>,
    pub bit_perfect: bool,
    /// Selected bit-perfect output device name; None = system default.
    pub bp_device: Option<String>,
    /// Position within the current track corresponding to `bp_played_at_track_start`
    /// frames played — i.e. the seek offset the current track started at (0 after
    /// a gapless roll-over, or the seek target on the first track of a session).
    bp_base: Duration,
    /// Value of `bp.played()` at which the current track began, so per-track
    /// position can be derived from the session-cumulative frame counter across
    /// gapless hand-offs.
    bp_played_at_track_start: Duration,
    // ── shared ─────────────────────────────────────────────────────────────
    // We track position manually because rodio Sink doesn't expose elapsed time
    started_at: Option<Instant>,
    paused_elapsed: Duration,
    current_duration: Option<Duration>,
    volume: f32,
    sample_buf: SampleBuf,
    stereo_buf: StereoBuf,
    pub last_sample_rate: u32,
    eq: Option<EqStateHandle>,
    /// ReplayGain factor (linear) applied on the rodio path only; 1.0 = off.
    /// Never applied on the bit-perfect path (a gain multiply breaks bit-perfect).
    replay_gain: f32,
    /// In-flight background seek (rodio slow path); None when idle.
    pending_seek: Option<PendingSeek>,
    /// Set once a bit-perfect stream has grabbed the output device. On Windows,
    /// WASAPI exclusive mode suspends the long-lived rodio (shared-mode)
    /// OutputStream and cpal doesn't recover it when exclusive mode is released
    /// — so the stream is recreated before the next normal-mode sink, else the
    /// rodio path is silent forever after bit-perfect has run once.
    rodio_stream_dirty: bool,
}

impl Engine {
    fn new(sample_buf: SampleBuf, stereo_buf: StereoBuf) -> Option<Self> {
        let (stream, stream_handle) = OutputStream::try_default().ok()?;
        Some(Engine {
            _stream: stream,
            stream_handle,
            sink: None,
            bp: None,
            bit_perfect: false,
            bp_device: None,
            bp_base: Duration::ZERO,
            bp_played_at_track_start: Duration::ZERO,
            started_at: None,
            paused_elapsed: Duration::ZERO,
            current_duration: None,
            volume: 1.0,
            sample_buf,
            stereo_buf,
            last_sample_rate: 44_100,
            eq: None,
            replay_gain: 1.0,
            pending_seek: None,
            rodio_stream_dirty: false,
        })
    }

    /// Recreate the rodio output stream if a bit-perfect (exclusive) stream may
    /// have invalidated it. No-op unless flagged, so normal playback pays
    /// nothing. Must be called before building any normal-mode sink.
    fn ensure_rodio_stream(&mut self) {
        if !self.rodio_stream_dirty { return; }
        if let Ok((stream, handle)) = OutputStream::try_default() {
            self._stream = stream;
            self.stream_handle = handle;
        }
        self.rodio_stream_dirty = false;
    }

    /// Effective rodio sink volume: user volume × ReplayGain.
    fn rodio_volume(&self) -> f32 { self.volume * self.replay_gain }

    fn play_file(&mut self, path: &PathBuf, duration: Option<Duration>) -> Result<(), String> {
        self.stop();
        if self.bit_perfect {
            self.start_bp(path, Duration::ZERO)?;
        } else {
            self.ensure_rodio_stream();
            let file = File::open(path).map_err(|e| format!("Open failed: {e}"))?;
            let decoder = Decoder::new(BufReader::new(file))
                .map_err(|e| format!("Decode failed: {e}"))?;
            let sink = Sink::try_new(&self.stream_handle)
                .map_err(|e| format!("Sink failed: {e}"))?;
            sink.set_volume(self.rodio_volume());
            self.last_sample_rate = decoder.sample_rate();
            let tapped = SpectrumSource::new(decoder, self.sample_buf.clone(), self.stereo_buf.clone());
            if let Some(ref eq) = self.eq {
                sink.append(EqSource::new(tapped.convert_samples::<f32>(), eq.clone()));
            } else {
                sink.append(tapped);
            }
            self.sink = Some(sink);
        }
        self.started_at = Some(Instant::now());
        self.paused_elapsed = Duration::ZERO;
        self.current_duration = duration;
        Ok(())
    }

    /// Decode `path` from `start` and play it on the bit-perfect stream,
    /// (re)opening the cpal stream if the device or stream format changed.
    fn start_bp(&mut self, path: &Path, start: Duration) -> Result<(), String> {
        let prep = bitperfect::prepare(path, start)
            .map_err(|e| format!("Bit-perfect: {e}"))?;
        self.last_sample_rate = prep.sample_rate;

        let reuse = self.bp.as_ref().is_some_and(|s| {
            s.sample_rate == prep.sample_rate
                && s.channels == prep.channels
                && s.requested_device == self.bp_device
        });
        if !reuse {
            self.bp = None; // release the old stream/device first
            self.bp = Some(bitperfect::BpStream::open(
                self.bp_device.as_deref(),
                prep.sample_rate,
                prep.channels,
                prep.bits_per_sample,
                self.sample_buf.clone(),
                self.stereo_buf.clone(),
            ).map_err(|e| format!("Bit-perfect: {e}"))?);
        }
        let bp = self.bp.as_ref().unwrap();
        bp.resume();
        bp.start(prep, self.volume);
        self.bp_base = start;
        self.bp_played_at_track_start = Duration::ZERO; // fresh session: frames_played reset to 0
        self.rodio_stream_dirty = true; // exclusive mode may kill the rodio stream
        Ok(())
    }

    /// Queue `path` for gapless continuation on the open bit-perfect stream.
    /// Returns false (no gapless) if there is no stream, the file can't be
    /// prepared, or its rate/channels differ from the stream (a format change
    /// forces a device re-open, so the track boundary can't be gapless).
    fn bp_queue_next(&self, path: &Path) -> bool {
        let Some(bp) = self.bp.as_ref() else { return false };
        let Ok(prep) = bitperfect::prepare(path, Duration::ZERO) else { return false };
        if prep.sample_rate != bp.sample_rate || prep.channels != bp.channels { return false; }
        bp.queue_next(prep);
        true
    }

    /// Drop a queued gapless track that hasn't started yet.
    fn bp_clear_next(&self) {
        if let Some(bp) = self.bp.as_ref() { bp.clear_next(); }
    }

    /// If the bit-perfect device has crossed a gapless track boundary, advance
    /// the per-track position base and report it (true). Call in a loop.
    fn bp_poll_boundary(&mut self) -> bool {
        let Some(boundary) = self.bp.as_ref().and_then(|bp| bp.take_reached_boundary()) else {
            return false;
        };
        self.bp_played_at_track_start = boundary;
        self.bp_base = Duration::ZERO; // the new track starts from its beginning
        true
    }

    /// One-line description of the active bit-perfect stream for the UI.
    fn bp_describe(&self) -> Option<String> {
        self.bp.as_ref().map(|s| format!("{} → {}", s.describe(), s.device_name))
    }

    /// Drop the bit-perfect stream entirely (releases the audio device).
    fn close_bp(&mut self) {
        self.bp = None;
    }

    fn pause(&mut self) {
        // A seek may still be decoding on the worker; remember the intent so it
        // resumes paused.
        if let Some(ps) = self.pending_seek.as_mut() { ps.was_paused = true; }
        if self.bit_perfect {
            if let Some(ref bp) = self.bp
                && !bp.is_paused() {
                bp.pause();
                if let Some(started) = self.started_at.take() {
                    self.paused_elapsed += started.elapsed();
                }
            }
        } else if let Some(ref sink) = self.sink
            && !sink.is_paused() {
            sink.pause();
            if let Some(started) = self.started_at.take() {
                self.paused_elapsed += started.elapsed();
            }
        }
    }

    fn resume(&mut self) {
        if let Some(ps) = self.pending_seek.as_mut() { ps.was_paused = false; }
        if self.bit_perfect {
            if let Some(ref bp) = self.bp
                && bp.is_paused() {
                bp.resume();
                self.started_at = Some(Instant::now());
            }
        } else if let Some(ref sink) = self.sink
            && sink.is_paused() {
            sink.play();
            self.started_at = Some(Instant::now());
        }
    }

    fn stop(&mut self) {
        self.pending_seek = None; // abandon any in-flight background seek
        if let Some(sink) = self.sink.take() {
            sink.stop();
        }
        if let Some(ref bp) = self.bp { bp.stop_session(); }
        self.bp_base = Duration::ZERO;
        self.started_at = None;
        self.paused_elapsed = Duration::ZERO;
        self.current_duration = None;
    }

    fn is_finished(&self) -> bool {
        if self.bit_perfect {
            self.bp.as_ref().is_some_and(|bp| bp.is_finished())
        } else {
            self.sink.as_ref().is_some_and(|s| s.empty())
        }
    }

    fn elapsed(&self) -> Duration {
        if self.bit_perfect && let Some(ref bp) = self.bp {
            // Sample-accurate: frames delivered to the device, less the frames
            // that belonged to earlier gapless tracks in this session.
            return self.bp_base + bp.played().saturating_sub(self.bp_played_at_track_start);
        }
        let running = self.started_at.map(|t| t.elapsed()).unwrap_or(Duration::ZERO);
        self.paused_elapsed + running
    }

    fn set_volume(&mut self, vol: f32) {
        self.volume = vol;
        if let Some(ref sink) = self.sink {
            sink.set_volume(self.rodio_volume());
        }
        // Bit-perfect path gets the raw user volume — never ReplayGain.
        if let Some(ref bp) = self.bp { bp.set_volume(vol); }
    }

    /// Set the ReplayGain factor (linear) and apply it live to the rodio sink.
    fn set_replay_gain(&mut self, gain: f32) {
        self.replay_gain = gain;
        if let Some(ref sink) = self.sink {
            sink.set_volume(self.rodio_volume());
        }
    }

    /// Append `path` to the current rodio sink for gapless continuation — rodio
    /// plays queued sources back-to-back with no gap. Decoding is lazy, but the
    /// decoder is opened here so the hand-off never underruns.
    fn append_next(&self, path: &Path) -> Result<(), String> {
        let sink = self.sink.as_ref().ok_or("no sink")?;
        let file = File::open(path).map_err(|e| format!("Open failed: {e}"))?;
        let decoder = Decoder::new(BufReader::new(file))
            .map_err(|e| format!("Decode failed: {e}"))?;
        let tapped = SpectrumSource::new(decoder, self.sample_buf.clone(), self.stereo_buf.clone());
        if let Some(ref eq) = self.eq {
            sink.append(EqSource::new(tapped.convert_samples::<f32>(), eq.clone()));
        } else {
            sink.append(tapped);
        }
        Ok(())
    }

    /// Roll the normal-mode (wall-clock) position onto the next gapless track:
    /// carry any overflow past the old duration into the new track's elapsed.
    fn roll_normal_position(&mut self, next_duration: Option<Duration>) {
        let over = self.elapsed().saturating_sub(self.current_duration.unwrap_or(Duration::ZERO));
        self.paused_elapsed = over;
        self.started_at = Some(Instant::now());
        self.current_duration = next_duration;
    }

    /// Seek to `target`.
    ///
    /// Bit-perfect path: symphonia container-level seek on a fresh decode
    /// session — no decode-and-discard, fast even at 192 kHz.
    ///
    /// Normal mode, preferred path: `Sink::try_seek` — works for WAV and any
    /// format whose symphonia reader can seek without the byte-length hint (e.g. MP3).
    ///
    /// Normal mode, fallback path: stop sink, reopen file, consume N samples to
    /// reach `target`.  Handles FLAC, where symphonia 0.5.5 returns `Unseekable`
    /// because rodio's `ReadSeekSource::byte_len()` always returns `None`.
    fn seek_to(&mut self, path: &Path, target: Duration) {
        if self.bit_perfect {
            let was_paused = self.bp.as_ref().map(|bp| bp.is_paused()).unwrap_or(false);
            if let Err(e) = self.start_bp(path, target) {
                eprintln!("seek: {e}");
                return;
            }
            if was_paused {
                if let Some(ref bp) = self.bp { bp.pause(); }
                self.started_at = None;
            } else {
                self.started_at = Some(Instant::now());
            }
            self.paused_elapsed = target;
            return;
        }

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

        // --- Slow path: reopen + skip samples, ON A BACKGROUND THREAD ---
        // FLAC seeks decode-and-discard every sample up to `target`; deep into a
        // hi-res file that is tens of millions of samples. Doing it here would
        // freeze the UI for seconds ("not responding"), so we hand it to a
        // worker and install the resulting decoder when it's ready (poll_pending_seek).
        if let Some(sink) = self.sink.take() { sink.stop(); }
        let rx = Self::spawn_seek_worker(path, target); // drops any prior receiver

        // Reflect the target position immediately; time is frozen until the
        // seek lands (started_at = None), and is_finished() returns false while
        // a seek is pending, so no spurious auto-advance.
        self.pending_seek = Some(PendingSeek { rx, target, was_paused });
        self.paused_elapsed = target;
        self.started_at = None;
    }

    /// Spawn the background decode-and-skip worker for a normal-mode seek to
    /// `target`: open the file, build a rodio decoder, discard samples up to
    /// `target`, and hand the positioned decoder back over the channel. Every
    /// bit of decode cost lives on this thread, never the UI thread.
    fn spawn_seek_worker(
        path: &Path,
        target: Duration,
    ) -> std::sync::mpsc::Receiver<Option<Decoder<BufReader<File>>>> {
        let (tx, rx) = std::sync::mpsc::channel();
        let path_c = path.to_path_buf();
        std::thread::Builder::new()
            .name("rodio-seek".into())
            .spawn(move || {
                let seeked = (|| -> Option<Decoder<BufReader<File>>> {
                    let file = File::open(&path_c).ok()?;
                    let mut decoder = Decoder::new(BufReader::new(file)).ok()?;
                    let sr = decoder.sample_rate() as f64;
                    let ch = decoder.channels() as f64;
                    let skip = (target.as_secs_f64() * sr * ch) as u64;
                    let mut n = 0u64;
                    while n < skip {
                        if decoder.next().is_none() { break; }
                        n += 1;
                    }
                    Some(decoder)
                })();
                let _ = tx.send(seeked);
            })
            .ok();
        rx
    }

    /// Begin normal-mode playback of `path` positioned at `target`, doing the
    /// open + decode-skip entirely on the worker thread (poll_pending_seek
    /// installs the sink when it lands). No audio is emitted from position 0 and
    /// the UI never blocks — used by the bit-perfect toggle-off restart, where
    /// `target` can be deep into a hi-res file.
    fn play_seeked_async(
        &mut self,
        path: &Path,
        target: Duration,
        duration: Option<Duration>,
        was_paused: bool,
    ) {
        self.stop(); // release any bp stream/session and supersede a pending seek
        self.current_duration = duration;
        let rx = Self::spawn_seek_worker(path, target);
        self.pending_seek = Some(PendingSeek { rx, target, was_paused });
        self.paused_elapsed = target;
        self.started_at = None;
    }

    /// True while a background seek is decoding — the UI must keep ticking so
    /// poll_pending_seek() runs and installs the result.
    fn is_seeking(&self) -> bool { self.pending_seek.is_some() }

    /// Poll the background rodio seek; when the seeked decoder arrives, wire it
    /// into a fresh sink. Cheap — call every frame.
    fn poll_pending_seek(&mut self) {
        let Some(ps) = self.pending_seek.as_ref() else { return };
        match ps.rx.try_recv() {
            Ok(Some(decoder)) => {
                let PendingSeek { target, was_paused, .. } = self.pending_seek.take().unwrap();
                self.ensure_rodio_stream();
                let Ok(sink) = Sink::try_new(&self.stream_handle) else { return };
                sink.set_volume(self.rodio_volume());
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
            }
            Ok(None) => self.pending_seek = None,                         // seek failed
            Err(std::sync::mpsc::TryRecvError::Empty) => {}               // still working
            Err(std::sync::mpsc::TryRecvError::Disconnected) => self.pending_seek = None,
        }
    }
}

// ---------------------------------------------------------------------------
// App state
// ---------------------------------------------------------------------------

#[derive(PartialEq)]
enum PlayState { Stopped, Playing, Paused }

#[derive(PartialEq, Clone, Copy, Serialize, Deserialize)]
enum LoopMode { Sequential, RepeatAll, RepeatOne }

/// Column the playlist can be sorted by.
#[derive(PartialEq, Clone, Copy)]
enum SortKey { Title, Artist, Album, Duration }

impl SortKey {
    const ALL: [SortKey; 4] = [SortKey::Title, SortKey::Artist, SortKey::Album, SortKey::Duration];
    fn label(self) -> &'static str {
        match self {
            SortKey::Title => "Title", SortKey::Artist => "Artist",
            SortKey::Album => "Album", SortKey::Duration => "Time",
        }
    }
}

impl Default for LoopMode {
    fn default() -> Self { LoopMode::RepeatAll }
}

/// Persisted player preferences (`~/.moosik/player.json`).
#[derive(Serialize, Deserialize, Clone, PartialEq)]
struct PlayerPrefs {
    #[serde(default = "default_volume")] volume: f32,
    #[serde(default)] loop_mode: LoopMode,
}

fn default_volume() -> f32 { 0.8 }

impl Default for PlayerPrefs {
    fn default() -> Self { Self { volume: 0.8, loop_mode: LoopMode::RepeatAll } }
}

fn load_player_prefs(dir: &Path) -> PlayerPrefs {
    std::fs::read_to_string(dir.join("player.json"))
        .ok()
        .and_then(|s| serde_json::from_str::<PlayerPrefs>(&s).ok())
        .map(|mut p| { p.volume = p.volume.clamp(0.0, 1.0); p })
        .unwrap_or_default()
}

fn save_player_prefs(dir: &Path, p: &PlayerPrefs) {
    let _ = std::fs::create_dir_all(dir);
    if let Ok(json) = serde_json::to_string_pretty(p) {
        let _ = std::fs::write(dir.join("player.json"), json);
    }
}

// ---------------------------------------------------------------------------
// ReplayGain (loudness normalization) — rodio path only; bypassed in
// bit-perfect mode, since any gain multiply breaks bit-perfectness.
// ---------------------------------------------------------------------------

/// ReplayGain 2.0 reference loudness. Used as the target when normalising
/// from Moosik's own measured LUFS (for files without ReplayGain tags).
const RG_TARGET_LUFS: f32 = -18.0;

#[derive(PartialEq, Clone, Copy, Serialize, Deserialize, Default)]
enum RgMode {
    #[default]
    Off,
    Track,
    Album,
}

impl RgMode {
    fn label(self) -> &'static str {
        match self { RgMode::Off => "Off", RgMode::Track => "Track", RgMode::Album => "Album" }
    }
}

#[derive(Serialize, Deserialize, Clone)]
struct RgSettings {
    mode: RgMode,
    /// Cap gain by the track's peak so a boost can't clip.
    prevent_clip: bool,
}

impl Default for RgSettings {
    fn default() -> Self { Self { mode: RgMode::Off, prevent_clip: true } }
}

fn load_rg_settings(dir: &Path) -> RgSettings {
    std::fs::read_to_string(dir.join("replaygain.json"))
        .ok()
        .and_then(|s| serde_json::from_str(&s).ok())
        .unwrap_or_default()
}

fn save_rg_settings(dir: &Path, s: &RgSettings) {
    let _ = std::fs::create_dir_all(dir);
    if let Ok(json) = serde_json::to_string(s) {
        let _ = std::fs::write(dir.join("replaygain.json"), json);
    }
}

// ---------------------------------------------------------------------------
// Appearance — optional, persisted visual preferences
// ---------------------------------------------------------------------------

fn default_ui_scale() -> f32 { 1.0 }
fn default_true() -> bool { true }

/// Light / dark theme choice.
#[derive(Serialize, Deserialize, Clone, Copy, PartialEq, Default)]
enum ThemeMode {
    #[default]
    Dark,
    Light,
}

impl ThemeMode {
    fn is_dark(self) -> bool { self == ThemeMode::Dark }
    fn label(self) -> &'static str {
        match self { ThemeMode::Dark => "Dark", ThemeMode::Light => "Light" }
    }
}

/// User-configurable look-and-feel. Every field is optional to change and has a
/// default matching the app's original appearance, so an absent or partial
/// settings file (or a fresh install) reproduces the shipped defaults exactly.
#[derive(Serialize, Deserialize, Clone)]
struct Appearance {
    /// Colour ramp for the spectrum visualisers.
    #[serde(default)] spectrum_palette: spectrum::SpectrumPalette,
    /// Global UI / text scale (egui zoom factor). 1.0 = 100%.
    #[serde(default = "default_ui_scale")] ui_scale: f32,
    /// Tint the chrome (now-playing, seek fill, current row) with the accent
    /// pulled from the current track's cover art. When off, the fixed brand
    /// accent is used everywhere.
    #[serde(default = "default_true")] art_accent: bool,
    /// Light or dark theme.
    #[serde(default)] theme: ThemeMode,
}

impl Default for Appearance {
    fn default() -> Self {
        Self {
            spectrum_palette: spectrum::SpectrumPalette::default(),
            ui_scale: 1.0,
            art_accent: true,
            theme: ThemeMode::Dark,
        }
    }
}

/// UI-scale bounds — matches egui's own zoom range and keeps text legible.
const UI_SCALE_MIN: f32 = 0.7;
const UI_SCALE_MAX: f32 = 1.6;

fn load_appearance(dir: &Path) -> Appearance {
    std::fs::read_to_string(dir.join("appearance.json"))
        .ok()
        .and_then(|s| serde_json::from_str::<Appearance>(&s).ok())
        .map(|mut a| { a.ui_scale = a.ui_scale.clamp(UI_SCALE_MIN, UI_SCALE_MAX); a })
        .unwrap_or_default()
}

fn save_appearance(dir: &Path, a: &Appearance) {
    let _ = std::fs::create_dir_all(dir);
    if let Ok(json) = serde_json::to_string_pretty(a) {
        let _ = std::fs::write(dir.join("appearance.json"), json);
    }
}

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
    /// Art decoded and uploaded to the GPU, plus a per-track UI accent derived
    /// from the cover (already blended toward the brand accent).
    Loaded { texture: egui::TextureHandle, width: u32, height: u32, accent: Color32 },
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
                Some((tex, w, h, accent)) => ArtEntry::Loaded { texture: tex, width: w, height: h, accent },
                None                      => ArtEntry::NoArt,
            };
            self.entries.insert(path.to_path_buf(), entry);
        }
        match self.entries.get(path)? {
            ArtEntry::Loaded { texture, width, height, .. } =>
                Some((texture.id(), *width, *height)),
            ArtEntry::NoArt => None,
        }
    }

    /// The per-track UI accent derived from the cover art (a tint of the brand
    /// accent), or `None` if the track has no art or isn't loaded yet.
    fn accent(&self, path: &Path) -> Option<Color32> {
        match self.entries.get(path)? {
            ArtEntry::Loaded { accent, .. } => Some(*accent),
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
        -> Option<(egui::TextureHandle, u32, u32, Color32)>
    {
        let bytes = Self::extract_bytes(path)?;
        let img = image::load_from_memory(&bytes).ok()?.to_rgba8();
        let (w, h) = img.dimensions();
        let accent = Self::vibrant_accent(&img);
        let pixels: Vec<egui::Color32> = img.pixels()
            .map(|p| egui::Color32::from_rgba_unmultiplied(p.0[0], p.0[1], p.0[2], p.0[3]))
            .collect();
        let ci = egui::ColorImage { size: [w as usize, h as usize], pixels };
        let tex = ctx.load_texture(
            format!("art:{}", path.to_string_lossy()),
            ci,
            egui::TextureOptions::LINEAR,
        );
        Some((tex, w, h, accent))
    }

    /// Derive a UI accent from cover art: sample the image, weight each pixel by
    /// saturation × brightness so vivid colours win over muddy/dark ones,
    /// average, lift if too dark for a dark UI, then blend 55% toward the art
    /// colour and 45% toward Kugelblitz — so the result is always a *tint of the
    /// brand accent*, never an arbitrary clashing colour. Falls back to the pure
    /// brand accent for grayscale / low-colour art.
    fn vibrant_accent(img: &image::RgbaImage) -> Color32 {
        let (w, h) = img.dimensions();
        let step = ((w.max(h) / 64).max(1)) as usize; // cap at ~64 samples/axis
        let (mut r, mut g, mut b, mut wsum) = (0f32, 0f32, 0f32, 0f32);
        for y in (0..h as usize).step_by(step) {
            for x in (0..w as usize).step_by(step) {
                let p = img.get_pixel(x as u32, y as u32).0;
                if p[3] < 128 { continue; } // skip transparent
                let (rf, gf, bf) = (p[0] as f32 / 255.0, p[1] as f32 / 255.0, p[2] as f32 / 255.0);
                let max = rf.max(gf).max(bf);
                let min = rf.min(gf).min(bf);
                let sat = if max <= 0.0 { 0.0 } else { (max - min) / max };
                let weight = sat * max; // vivid AND not too dark
                r += rf * weight; g += gf * weight; b += bf * weight; wsum += weight;
            }
        }
        if wsum < 1e-3 {
            return pal::ACCENT; // grayscale / no vivid colour
        }
        let (mut rr, mut gg, mut bb) = (r / wsum, g / wsum, b / wsum);
        // Keep the accent bright enough to read on the dark chrome.
        let lum = 0.299 * rr + 0.587 * gg + 0.114 * bb;
        if lum < 0.35 {
            let boost = 0.35 / lum.max(1e-3);
            rr = (rr * boost).min(1.0); gg = (gg * boost).min(1.0); bb = (bb * boost).min(1.0);
        }
        let k = pal::ACCENT;
        let mix = |art: f32, brand: u8| -> u8 {
            ((art * 0.55 + (brand as f32 / 255.0) * 0.45).clamp(0.0, 1.0) * 255.0) as u8
        };
        Color32::from_rgb(mix(rr, k.r()), mix(gg, k.g()), mix(bb, k.b()))
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
    /// Last (volume, loop_mode) written to player.json, for change detection.
    saved_player: PlayerPrefs,
    player_save_at: Option<Instant>,
    // multi-select
    selected: HashSet<usize>,
    last_clicked: Option<usize>,
    /// Playlist filter query (title / artist / album substring). Empty = no filter.
    filter_query: String,
    /// Active playlist sort column + direction (None = manual / file order).
    sort_key: Option<SortKey>,
    sort_asc: bool,
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
    // bit-perfect mode
    bit_perfect: bool,
    bp_device: Option<String>,
    bp_devices: Option<Vec<bitperfect::DeviceCaps>>,
    bp_scan_rx: Option<std::sync::mpsc::Receiver<Vec<bitperfect::DeviceCaps>>>,
    // Frame limiter: wall-clock time the previous update() frame began.
    last_frame: Option<Instant>,
    // OS media controls (media keys + now-playing) and change-tracking so we
    // only push updates to the OS when something actually changes.
    media: media_controls::MediaOs,
    media_last_index: Option<usize>,
    media_last_state: Option<media_controls::PlaybackState>,
    media_last_push: Option<Instant>,
    // Appearance (palette, UI scale, accent source) — optional/persisted.
    appearance: Appearance,
    /// Pending text-size value edited by the slider; applied to
    /// `appearance.ui_scale` only when the user clicks Apply, so dragging the
    /// slider doesn't live-resize the UI (which fights the cursor).
    ui_scale_draft: f32,
    // ReplayGain
    rg: RgSettings,
    /// Last linear gain pushed to the engine — so update_replay_gain() is a
    /// cheap no-op until something actually changes.
    rg_last_applied: f32,
    /// Per-track UI accent (a tint of the brand accent pulled from the current
    /// cover art), refreshed each frame. Drives the now-playing bar, current-row
    /// marker, and seek fill so the chrome picks up the mood of what's playing.
    track_accent: Color32,
    // Gapless playback
    /// Playlist index appended/queued ahead for a seamless hand-off, once the
    /// current track nears its end. None when nothing is queued.
    gapless_next: Option<usize>,
    /// The `current_index` we've already made a prebuffer decision for, so the
    /// decision runs once per track rather than every frame.
    gapless_tried: Option<usize>,
}

impl MoosikApp {
    fn new(cc: &eframe::CreationContext) -> Self {
        setup_fonts(&cc.egui_ctx);
        let appearance = load_appearance(&moosik_dir());
        apply_theme(&cc.egui_ctx, appearance.theme.is_dark());
        let mut spectrum_window = SpectrumWindow::new();
        let mut engine = Engine::new(spectrum_window.sample_buf.clone(), spectrum_window.stereo_buf.clone());
        if let Some(ref mut e) = engine {
            e.eq = Some(spectrum_window.eq_state.clone());
        }
        let saved_paths = load_last_playlist();
        let playlist: Vec<Track> = saved_paths.into_iter().map(Track::load).collect();
        let playlist_store = load_playlist_store();
        let bp_settings = bitperfect::load_settings(&moosik_dir());
        if let Some(ref mut e) = engine {
            e.bit_perfect = bp_settings.enabled;
            e.bp_device = bp_settings.device.clone();
        }
        spectrum_window.bit_perfect = bp_settings.enabled;
        // Default the spectrum animation cap to the monitor's refresh rate
        // (clamped to the slider's range), falling back to 60 if unknown.
        if let Some(hz) = monitor_refresh_hz() {
            spectrum_window.max_fps = hz.clamp(1.0, 240.0);
        }
        let ui_scale_draft = appearance.ui_scale;
        let player_prefs = load_player_prefs(&moosik_dir());
        if let Some(ref mut e) = engine { e.set_volume(player_prefs.volume); }
        MoosikApp {
            playlist,
            current_index: None,
            play_state: PlayState::Stopped,
            engine,
            volume: player_prefs.volume,
            seek_pos: 0.0,
            seeking: false,
            status_msg: String::new(),
            spectrum_window,
            info_open: false,
            loop_mode: player_prefs.loop_mode,
            saved_player: player_prefs,
            player_save_at: None,
            selected: HashSet::new(),
            last_clicked: None,
            filter_query: String::new(),
            sort_key: None,
            sort_asc: true,
            drag_src: None,
            drag_over_row: None,
            playlist_store,
            active_saved_playlist: None,
            show_save_playlist_input: false,
            playlist_name_buf: String::new(),
            art_cache: ArtCache::new(),
            art_hover: None,
            bit_perfect: bp_settings.enabled,
            bp_device: bp_settings.device,
            bp_devices: None,
            // Kick off the device scan at startup so the picker is ready.
            bp_scan_rx: Some(bitperfect::spawn_device_scan()),
            last_frame: None,
            media: media_controls::MediaOs::new(cc),
            media_last_index: None,
            media_last_state: None,
            media_last_push: None,
            appearance,
            ui_scale_draft,
            rg: load_rg_settings(&moosik_dir()),
            rg_last_applied: 1.0,
            track_accent: pal::ACCENT,
            gapless_next: None,
            gapless_tried: None,
        }
    }

    /// The accent for the current track: its cover-derived tint, or the brand
    /// accent when the track has no art (or it hasn't decoded yet). On the light
    /// theme the cover tint is darkened so it reads against light chrome.
    fn current_accent(&self) -> Color32 {
        let dark = self.appearance.theme.is_dark();
        if !self.appearance.art_accent {
            return pal::accent(dark);
        }
        let base = self.current_index
            .and_then(|i| self.playlist.get(i))
            .and_then(|t| self.art_cache.accent(&t.path))
            .unwrap_or(pal::accent(dark));
        if dark { base } else { dim_for_light(base) }
    }

    /// Sort the playlist by a column. Re-selecting the same column flips
    /// direction. The currently-playing track is followed to its new position;
    /// selection and any queued gapless track are cleared since indices change.
    fn sort_playlist(&mut self, key: SortKey) {
        let asc = if self.sort_key == Some(key) { !self.sort_asc } else { true };
        let cur_path = self.current_index
            .and_then(|i| self.playlist.get(i))
            .map(|t| t.path.clone());
        self.playlist.sort_by(|a, b| {
            let o = match key {
                SortKey::Title    => a.title.to_lowercase().cmp(&b.title.to_lowercase()),
                SortKey::Artist   => a.artist.to_lowercase().cmp(&b.artist.to_lowercase()),
                SortKey::Album    => a.album.to_lowercase().cmp(&b.album.to_lowercase()),
                SortKey::Duration => a.duration.cmp(&b.duration),
            };
            if asc { o } else { o.reverse() }
        });
        self.current_index = cur_path
            .and_then(|p| self.playlist.iter().position(|t| t.path == p));
        self.selected.clear();
        self.last_clicked = None;
        self.flush_gapless();
        self.sort_key = Some(key);
        self.sort_asc = asc;
    }

    /// Save volume + loop mode when they change, throttled so a volume-slider
    /// drag doesn't hammer the disk.
    fn persist_player_prefs_if_changed(&mut self) {
        let cur = PlayerPrefs { volume: self.volume, loop_mode: self.loop_mode };
        if cur == self.saved_player { return; }
        let ready = self.player_save_at
            .map(|t| t.elapsed().as_secs_f32() > 0.6)
            .unwrap_or(true);
        if ready {
            save_player_prefs(&moosik_dir(), &cur);
            self.saved_player = cur;
            self.player_save_at = Some(Instant::now());
        }
    }

    /// A form of the per-track accent for playheads / hovered handles that need
    /// to pop against the accent-filled progress — brightened on dark, deepened
    /// on light.
    fn track_accent_bright(&self) -> Color32 {
        let a = self.track_accent;
        if self.appearance.theme.is_dark() {
            let m = |c: u8| (c as f32 * 0.55 + 255.0 * 0.45) as u8;
            Color32::from_rgb(m(a.r()), m(a.g()), m(a.b()))
        } else {
            let m = |c: u8| (c as f32 * 0.72) as u8;
            Color32::from_rgb(m(a.r()), m(a.g()), m(a.b()))
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
        // Explicit track change rebuilds the stream — drop any gapless queue.
        self.gapless_next = None;
        self.gapless_tried = None;
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
                if self.bit_perfect
                    && let Some(desc) = self.engine.as_ref().and_then(|e| e.bp_describe()) {
                    self.status_msg = format!("💎 {desc} — {}", self.playlist[idx].title);
                }
            }
            Err(e) if self.bit_perfect => {
                // Device rejected the format (or vanished) — drop to normal
                // mode so playback keeps working, and tell the user why.
                // Runtime-only: the saved preference is left untouched.
                self.apply_bit_perfect_runtime(false);
                let retried = self.engine.as_mut()
                    .map(|eng| eng.play_file(&path, duration))
                    .unwrap_or(Err("No audio engine".to_string()));
                match retried {
                    Ok(()) => {
                        self.play_state = PlayState::Playing;
                        self.status_msg = format!("💎 disabled ({e}) — playing in normal mode");
                    }
                    Err(e2) => {
                        self.play_state = PlayState::Stopped;
                        self.status_msg = format!("Error: {e2}");
                        return;
                    }
                }
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

    // ── Bit-perfect helpers ─────────────────────────────────────────────────

    fn save_bp_settings(&self) {
        bitperfect::save_settings(&moosik_dir(), &bitperfect::BpSettings {
            enabled: self.bit_perfect,
            device: self.bp_device.clone(),
        });
    }

    /// Propagate the bit-perfect flag to every component without persisting —
    /// used by the automatic fallback so a transient device failure doesn't
    /// overwrite the user's saved preference. Does not restart playback.
    fn apply_bit_perfect_runtime(&mut self, on: bool) {
        self.bit_perfect = on;
        self.spectrum_window.bit_perfect = on;
        if let Some(ref mut engine) = self.engine {
            engine.bit_perfect = on;
            if !on { engine.close_bp(); } // release the device
        }
    }

    /// Propagate the bit-perfect flag and persist it (explicit user action).
    fn apply_bit_perfect(&mut self, on: bool) {
        self.apply_bit_perfect_runtime(on);
        self.save_bp_settings();
    }

    /// Restart the current track in the engine's current mode, preserving
    /// position and pause state. Used when toggling bit-perfect or switching
    /// output device mid-track.
    fn restart_current_track(&mut self) -> Result<(), String> {
        let Some(idx) = self.current_index else { return Ok(()) };
        if self.play_state == PlayState::Stopped { return Ok(()) }
        let pos        = self.elapsed();
        let was_paused = self.play_state == PlayState::Paused;
        let path       = self.playlist[idx].path.clone();
        let dur        = self.playlist[idx].duration;
        let engine = self.engine.as_mut().ok_or("No audio engine")?;

        // Normal mode with a real position (e.g. toggling bit-perfect OFF
        // mid-track): open + seek entirely off the UI thread. Skips the
        // play-from-0 blip and keeps every decode off the UI thread — a deep
        // hi-res FLAC seek on the UI thread here is what froze the app.
        if !engine.bit_perfect && pos > Duration::ZERO {
            engine.play_seeked_async(&path, pos, dur, was_paused);
            self.spectrum_window.on_seek(pos.as_secs_f64());
            return Ok(());
        }

        engine.play_file(&path, dur)?;
        if pos > Duration::ZERO {
            engine.seek_to(&path, pos);
            self.spectrum_window.on_seek(pos.as_secs_f64());
        }
        if was_paused {
            engine.pause();
            self.play_state = PlayState::Paused;
        }
        Ok(())
    }

    /// Toggle bit-perfect mode, restarting the current track in the new mode.
    /// Reverts the toggle (with a status message) if the device refuses.
    fn toggle_bit_perfect(&mut self) {
        let new_bp = !self.bit_perfect;
        self.apply_bit_perfect(new_bp);
        match self.restart_current_track() {
            Ok(()) => {
                if new_bp {
                    self.status_msg = match self.engine.as_ref().and_then(|e| e.bp_describe()) {
                        Some(desc) => format!("💎 Bit-perfect: {desc}"),
                        None => "💎 Bit-perfect enabled".to_string(),
                    };
                } else {
                    self.status_msg = "Bit-perfect off".to_string();
                }
            }
            Err(e) => {
                self.apply_bit_perfect(!new_bp);
                let _ = self.restart_current_track();
                self.status_msg = format!("Bit-perfect unavailable: {e}");
            }
        }
    }

    /// Select the bit-perfect output device (None = system default) and, if
    /// playing in bit-perfect mode, move playback onto it.
    fn select_bp_device(&mut self, device: Option<String>) {
        let old = self.bp_device.clone();
        self.bp_device = device.clone();
        if let Some(ref mut engine) = self.engine {
            engine.bp_device = device.clone();
        }
        self.save_bp_settings();
        if self.bit_perfect
            && let Err(e) = self.restart_current_track() {
            // New device refused the current track — roll back.
            self.bp_device = old.clone();
            if let Some(ref mut engine) = self.engine { engine.bp_device = old; }
            self.save_bp_settings();
            let _ = self.restart_current_track();
            self.status_msg = format!("Device unavailable: {e}");
            return;
        }
        self.status_msg = match (&self.bp_device, self.bit_perfect) {
            (Some(n), _) => format!("Output device: {n}"),
            (None, _)    => "Output device: system default".to_string(),
        };
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
        self.gapless_next = None;
        self.gapless_tried = None;
        if let Some(ref mut engine) = self.engine {
            engine.stop();
        }
        self.play_state = PlayState::Stopped;
        self.seek_pos = 0.0;
        self.status_msg = String::new();
        self.spectrum_window.on_stop();
    }

    // ── Gapless playback ────────────────────────────────────────────────────

    /// The index that will play after the current one, per the loop mode — the
    /// track to prebuffer for a seamless hand-off. None = nothing follows.
    fn upcoming_index(&self) -> Option<usize> {
        let cur = self.current_index?;
        if self.playlist.is_empty() { return None; }
        match self.loop_mode {
            LoopMode::RepeatOne => Some(cur),
            LoopMode::RepeatAll => Some((cur + 1) % self.playlist.len()),
            LoopMode::Sequential => (cur + 1 < self.playlist.len()).then_some(cur + 1),
        }
    }

    /// Once per track, when it nears its end, prebuffer the next one onto the
    /// same stream so playback continues with no gap. Requires a known duration
    /// (for the lead timing); bit-perfect additionally requires the next track
    /// to share the current rate/channels, else it's left to end normally.
    fn try_arm_gapless(&mut self) {
        let Some(cur) = self.current_index else { return };
        if self.gapless_tried == Some(cur) { return; }
        let Some(dur) = self.playlist.get(cur).and_then(|t| t.duration) else { return };
        let bp = self.engine.as_ref().map(|e| e.bit_perfect).unwrap_or(false);
        let lead = Duration::from_secs(if bp { 5 } else { 2 });
        if dur.saturating_sub(self.elapsed()) > lead { return; }

        // Decide exactly once for this track, whatever the outcome.
        self.gapless_tried = Some(cur);
        let Some(next) = self.upcoming_index() else { return };
        let path = self.playlist[next].path.clone();
        let armed = self.engine.as_ref().map(|e| {
            if bp { e.bp_queue_next(&path) } else { e.append_next(&path).is_ok() }
        }).unwrap_or(false);
        if armed { self.gapless_next = Some(next); }
    }

    /// Roll the UI/metadata over to the gapless-queued track once the device has
    /// crossed the boundary. The audio is already flowing seamlessly; this only
    /// updates the displayed track, spectrum, ReplayGain source, and position.
    fn gapless_rollover(&mut self) {
        let Some(next) = self.gapless_next.take() else { return };
        if next >= self.playlist.len() {
            // Playlist shrank under a queued hand-off — advance cleanly instead.
            self.gapless_tried = None;
            self.next_track();
            return;
        }
        let next_dur = self.playlist[next].duration;
        let bp = self.engine.as_ref().map(|e| e.bit_perfect).unwrap_or(false);
        if let Some(e) = self.engine.as_mut() {
            if bp {
                e.current_duration = next_dur; // position base advanced in bp_poll_boundary
            } else {
                e.roll_normal_position(next_dur);
            }
        }
        self.current_index = Some(next);
        self.gapless_tried = None; // let the new current arm its own successor
        self.seek_pos = 0.0;
        let sr = self.playlist[next].sample_rate
            .or_else(|| self.engine.as_ref().map(|e| e.last_sample_rate))
            .unwrap_or(44_100);
        let path = self.playlist[next].path.clone();
        let title = self.playlist[next].title.clone();
        self.spectrum_window.on_play(&path, sr);
        self.status_msg = if bp {
            self.engine.as_ref().and_then(|e| e.bp_describe())
                .map(|d| format!("💎 {d} — {title}"))
                .unwrap_or_else(|| format!("Playing: {title}"))
        } else {
            format!("Playing: {title}")
        };
    }

    /// Discard a queued gapless hand-off (the user changed what plays next).
    /// Bit-perfect just drops the queued decode; normal mode must rebuild the
    /// current track without its appended successor (a brief gap, but only on an
    /// explicit action near a track boundary).
    fn flush_gapless(&mut self) {
        if self.gapless_next.take().is_none() { return; }
        self.gapless_tried = None;
        let bp = self.engine.as_ref().map(|e| e.bit_perfect).unwrap_or(false);
        if bp {
            if let Some(e) = self.engine.as_ref() { e.bp_clear_next(); }
        } else if self.play_state != PlayState::Stopped
            && let Some(idx) = self.current_index {
            let path = self.playlist[idx].path.clone();
            let dur = self.playlist[idx].duration;
            let pos = self.elapsed();
            let was_paused = self.play_state == PlayState::Paused;
            if let Some(e) = self.engine.as_mut() {
                e.play_seeked_async(&path, pos, dur, was_paused);
            }
        }
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

    /// Seek the current track to `target` (clamped to its duration), keeping
    /// the spectrum and seek-bar position in sync. Shared by the seek bar and
    /// OS media-key seeking.
    fn seek_to(&mut self, target: Duration) {
        let Some(idx) = self.current_index else { return };
        // A seek reshapes the current stream — discard any gapless queue so the
        // engine and our roll-over bookkeeping can't diverge.
        self.flush_gapless();
        let dur = self.playlist[idx].duration;
        let target = dur.map_or(target, |d| target.min(d));
        let path = self.playlist[idx].path.clone();
        if let Some(ref mut engine) = self.engine {
            engine.seek_to(&path, target);
        }
        self.spectrum_window.on_seek(target.as_secs_f64());
        self.seek_pos = dur
            .map(|d| (target.as_secs_f32() / d.as_secs_f32().max(0.001)).clamp(0.0, 1.0))
            .unwrap_or(0.0);
    }

    /// Handle media-key / OS transport events drained from `self.media`.
    fn handle_media_events(&mut self) {
        use media_controls::{MediaControlEvent as E, SeekDirection};
        let events: Vec<E> = self.media.events().collect();
        for ev in events {
            match ev {
                E::Play => if self.play_state != PlayState::Playing { self.toggle_play_pause(); },
                E::Pause => if self.play_state == PlayState::Playing { self.toggle_play_pause(); },
                E::Toggle => self.toggle_play_pause(),
                E::Next => self.next_track(),
                E::Previous => self.prev_track(),
                E::Stop => self.stop(),
                E::Seek(dir) | E::SeekBy(dir, _) => {
                    let step = match ev {
                        E::SeekBy(_, d) => d,
                        _ => Duration::from_secs(5),
                    };
                    let cur = self.elapsed();
                    let target = match dir {
                        SeekDirection::Forward => cur + step,
                        SeekDirection::Backward => cur.saturating_sub(step),
                    };
                    self.seek_to(target);
                }
                E::SetPosition(pos) => self.seek_to(pos.0),
                _ => {} // SetVolume / OpenUri / Raise / Quit — not handled
            }
        }
    }

    /// Push now-playing metadata (on track change) and playback state /
    /// progress (on state change or ~once a second) to the OS controls.
    fn sync_media_os(&mut self) {
        use media_controls::PlaybackState;
        let state = match self.play_state {
            PlayState::Playing => PlaybackState::Playing,
            PlayState::Paused => PlaybackState::Paused,
            PlayState::Stopped => PlaybackState::Stopped,
        };

        if self.current_index != self.media_last_index {
            self.media_last_index = self.current_index;
            if let Some(t) = self.current_index.and_then(|i| self.playlist.get(i)) {
                let (title, artist, album, dur) =
                    (t.display_title().to_string(), t.artist.clone(), t.album.clone(), t.duration);
                self.media.set_metadata(&title, &artist, &album, dur);
            }
            self.media_last_state = None; // force a playback push to follow
        }

        let due = self.media_last_push.map(|t| t.elapsed().as_millis() >= 1000).unwrap_or(true);
        if self.media_last_state != Some(state) || due {
            self.media_last_state = Some(state);
            self.media_last_push = Some(Instant::now());
            let progress = (state != PlaybackState::Stopped).then(|| self.elapsed());
            self.media.set_playback(state, progress);
        }
    }

    // ── ReplayGain ───────────────────────────────────────────────────────────

    /// Linear ReplayGain factor for the current track under the current mode.
    /// 1.0 (no change) when Off, in bit-perfect mode, or when no gain source
    /// is available yet (untagged track whose loudness scan hasn't finished).
    fn compute_replay_gain(&self) -> f32 {
        if self.rg.mode == RgMode::Off || self.bit_perfect {
            return 1.0;
        }
        let Some(t) = self.current_index.and_then(|i| self.playlist.get(i)) else { return 1.0 };

        // Prefer the requested tag (other as fallback); else Moosik's measured LUFS.
        let (tag_gain, tag_peak) = match self.rg.mode {
            RgMode::Album => (t.rg_album_gain.or(t.rg_track_gain), t.rg_album_peak.or(t.rg_track_peak)),
            _ => (t.rg_track_gain.or(t.rg_album_gain), t.rg_track_peak.or(t.rg_album_peak)),
        };

        let analysis = self.spectrum_window.track_analysis.as_ref();
        let (mut gain_db, peak_db) = if let Some(g) = tag_gain {
            (g, tag_peak.map(|p| 20.0 * p.log10()))
        } else if let Some(a) = analysis.filter(|a| a.integrated_lufs.is_finite()) {
            (RG_TARGET_LUFS - a.integrated_lufs, Some(a.peak_dbfs))
        } else {
            return 1.0; // measurement not ready yet — unity until it lands
        };

        if self.rg.prevent_clip
            && let Some(pk) = peak_db {
            gain_db = gain_db.min(-pk); // keep peak ≤ 0 dBFS
        }
        gain_db = gain_db.clamp(-24.0, 12.0);
        10f32.powf(gain_db / 20.0)
    }

    /// Recompute and, if changed, apply the ReplayGain factor. Cheap enough to
    /// call every frame — catches track change, mode change, bit-perfect
    /// toggle, and the measured LUFS arriving mid-track.
    fn update_replay_gain(&mut self) {
        let g = self.compute_replay_gain();
        if (g - self.rg_last_applied).abs() > 1e-4 {
            self.rg_last_applied = g;
            if let Some(ref mut e) = self.engine { e.set_replay_gain(g); }
        }
    }

    /// Currently-applied gain in dB (for the UI). 0.0 = unity.
    fn rg_applied_db(&self) -> f32 {
        20.0 * self.rg_last_applied.max(1e-6).log10()
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
    ui.label(egui::RichText::new(label).color(pal::text_dim(ui.visuals().dark_mode)).size(12.0));
    ui.label(egui::RichText::new(value).size(12.0));
    ui.end_row();
}

fn section(ui: &mut egui::Ui, heading: &str) {
    ui.separator();
    ui.label(egui::RichText::new(heading).strong().size(13.0).color(pal::accent(ui.visuals().dark_mode)));
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
                .size(12.0).color(pal::accent(ui.visuals().dark_mode)));
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
        // --- Frame limiter ---------------------------------------------------
        // eframe + an immediate child viewport (the spectrum window) free-run
        // the repaint loop: the child requests a repaint every frame, which
        // overrides request_repaint_after, so the window was redrawing at
        // ~900 fps and burning ~30% CPU. request_repaint_after is only an
        // upper bound on the wait, so it cannot cap this on its own. Park the
        // UI thread to the target frame interval instead — this yields the CPU
        // (the thread sleeps, it does not spin) and hard-caps the real rate.
        //
        // Target = spectrum's max_fps while that window is open (it drives the
        // animation), else 60 fps for the main window's seek bar.
        let spectrum_open = self.spectrum_window.open;
        let seeking = self.engine.as_ref().is_some_and(|e| e.is_seeking());
        let animating = self.play_state == PlayState::Playing
            || seeking
            || self.spectrum_window.analyzer.is_analyzing.load(std::sync::atomic::Ordering::Relaxed);
        // Drive frames while anything is animating (incl. a pending background
        // seek), or just to cap the spin of an open (but idle) spectrum viewport.
        let want_frames = animating || spectrum_open;
        if want_frames {
            let target_fps = match (spectrum_open, animating) {
                (true, true)  => self.spectrum_window.max_fps.max(1.0), // live spectrum
                (true, false) => 30.0,  // open but static — just cap the spin
                (false, _)    => 60.0,  // main-window seek bar / progress
            };
            let frame_time = Duration::from_secs_f32(1.0 / target_fps);
            if let Some(last) = self.last_frame {
                let dt = last.elapsed();
                if dt < frame_time {
                    std::thread::sleep(frame_time - dt);
                }
            }
            self.last_frame = Some(Instant::now());
            // Keep the loop running at our paced rate. We've already slept, so
            // this just schedules the next frame; when idle we skip it and fall
            // back to eframe's reactive (event-driven) mode → ~0% CPU.
            ctx.request_repaint();
        } else {
            self.last_frame = None;
        }

        // --- Typography / UI scale ---
        // Apply the configured zoom factor (egui scales point sizes too, so all
        // fixed `.size()` text scales proportionally). set_zoom_factor only
        // triggers a relayout when the value actually changes, so calling it
        // every frame is cheap.
        let scale = self.appearance.ui_scale.clamp(UI_SCALE_MIN, UI_SCALE_MAX);
        if (ctx.zoom_factor() - scale).abs() > f32::EPSILON {
            ctx.set_zoom_factor(scale);
        }

        // --- OS media-key / transport events ---
        self.handle_media_events();

        // --- Search shortcuts (work regardless of focus) ---
        // Ctrl+F focuses the playlist filter; Esc clears it (and drops focus).
        let filter_id = egui::Id::new("playlist_filter");
        let (ctrl_f, esc) = ctx.input(|i| (
            i.modifiers.ctrl && i.key_pressed(egui::Key::F),
            i.key_pressed(egui::Key::Escape),
        ));
        if ctrl_f {
            ctx.memory_mut(|m| m.request_focus(filter_id));
        }
        if esc {
            let (filter_focused, nothing_focused) =
                ctx.memory(|m| (m.has_focus(filter_id), m.focused().is_none()));
            // Clear when the filter box itself has focus, or when nothing else is
            // focused (so Esc while typing in another field is left alone).
            if filter_focused || (nothing_focused && !self.filter_query.is_empty()) {
                self.filter_query.clear();
                ctx.memory_mut(|m| m.surrender_focus(filter_id));
            }
        }

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
                if self.current_index.is_some() {
                    self.seek_to(pos);
                }
            }
            if arrow_right {
                let pos = self.elapsed() + Duration::from_secs(5);
                let capped = self.current_duration().map(|d| pos.min(d)).unwrap_or(pos);
                if self.current_index.is_some() {
                    self.seek_to(capped);
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

        // --- collect finished bit-perfect device scan ---
        if let Some(ref rx) = self.bp_scan_rx
            && let Ok(devices) = rx.try_recv() {
            self.bp_devices = Some(devices);
            self.bp_scan_rx = None;
        }

        // --- install a completed background seek, if any ---
        if let Some(ref mut engine) = self.engine {
            engine.poll_pending_seek();
        }

        // --- gapless roll-over + auto-advance when a track finishes ---
        if self.play_state == PlayState::Playing {
            // Bit-perfect: the device reports exact frame boundaries as it plays
            // through gaplessly-chained tracks (possibly several per frame).
            while self.engine.as_mut().map(|e| e.bp_poll_boundary()).unwrap_or(false) {
                self.gapless_rollover();
            }
            // Normal mode: no per-source callback, so detect the boundary by time.
            let normal_crossed = self.gapless_next.is_some()
                && self.engine.as_ref().map(|e| !e.bit_perfect).unwrap_or(false)
                && self.current_index
                    .and_then(|i| self.playlist.get(i)).and_then(|t| t.duration)
                    .map(|d| self.elapsed() >= d).unwrap_or(false);
            if normal_crossed {
                self.gapless_rollover();
            }

            // A truly finished stream (nothing was queued) advances the old way.
            let finished = self.engine.as_ref().map(|e| e.is_finished()).unwrap_or(false);
            if finished {
                self.next_track();
            }

            // Prebuffer the next track once we're near the end of this one.
            self.try_arm_gapless();
        }

        // sync volume to engine on startup
        if let Some(ref mut engine) = self.engine
            && (engine.volume - self.volume).abs() > 0.001 {
            engine.set_volume(self.volume);
        }

        // Repaint pacing is handled by the frame limiter at the top of update().

        // Advance spectrum analyzer and render window
        let elapsed_secs = self.elapsed().as_secs_f64();
        let is_playing = self.play_state == PlayState::Playing;
        // Push the current appearance into the spectrum window so its visualisers
        // (incl. the spectrogram fed inside tick()) use the selected palette.
        self.spectrum_window.palette_kind = self.appearance.spectrum_palette;
        self.spectrum_window.palette_accent = self.track_accent;
        self.spectrum_window.tick(elapsed_secs, is_playing);
        // Update current art for the spectrum window (loads it if needed), then
        // refresh the per-track accent now that the art is available.
        self.spectrum_window.current_art = self.current_index
            .and_then(|i| self.playlist.get(i))
            .and_then(|t| self.art_cache.get_or_load(&t.path, ctx));
        self.track_accent = self.current_accent();
        self.spectrum_window.palette_accent = self.track_accent;
        self.spectrum_window.show(ctx);
        // The palette selector lives in the spectrum window; persist any change
        // it made back into the Appearance settings.
        if self.spectrum_window.palette_kind != self.appearance.spectrum_palette {
            self.appearance.spectrum_palette = self.spectrum_window.palette_kind;
            save_appearance(&moosik_dir(), &self.appearance);
        }

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
                    let dark = ui.visuals().dark_mode;
                    ui.vertical(|ui| {
                        ui.label(RichText::new(&track.title).size(18.0).strong().color(pal::text_strong(dark)));
                        ui.label(
                            RichText::new(format!("{} — {}", track.artist, track.album))
                                .size(13.0)
                                .color(pal::text_dim(dark)),
                        );
                    });
                    ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
                        ui.add_space(12.0);
                        let dark = ui.visuals().dark_mode;
                        let (state_icon, state_col) = match self.play_state {
                            PlayState::Playing => ("▶ Playing", self.track_accent),
                            PlayState::Paused  => ("⏸ Paused",  pal::amber(dark)),
                            PlayState::Stopped => ("⏹ Stopped", pal::muted(dark)),
                        };
                        ui.label(RichText::new(state_icon).size(13.0).color(state_col));
                    });
                });
            } else {
                ui.add_space(4.0);
                ui.horizontal(|ui| {
                    ui.add_space(12.0);
                    ui.label(RichText::new("No track loaded").size(16.0).color(pal::text_faint(ui.visuals().dark_mode)));
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
            let dark = ui.visuals().dark_mode;
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
                        self.track_accent
                    } else {
                        pal::wave_unplayed(dark)
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
                            egui::Stroke::new(1.5, pal::warn(dark)),
                        );
                    }
                }
                // Playhead line
                painter.line_segment(
                    [egui::Pos2::new(fill_x, row_rect.top() + 2.0),
                     egui::Pos2::new(fill_x, row_rect.bottom() - 2.0)],
                    egui::Stroke::new(2.0, self.track_accent_bright()),
                );
            } else {
                // Fallback: thin bar
                painter.rect_filled(
                    egui::Rect::from_min_max(
                        egui::Pos2::new(track_x0, track_y - 2.0),
                        egui::Pos2::new(track_x1, track_y + 2.0),
                    ),
                    2.0,
                    pal::track_bg(dark),
                );
                if fill_x > track_x0 {
                    painter.rect_filled(
                        egui::Rect::from_min_max(
                            egui::Pos2::new(track_x0, track_y - 2.0),
                            egui::Pos2::new(fill_x,   track_y + 2.0),
                        ),
                        2.0,
                        self.track_accent,
                    );
                }
                let hot = seek_resp.hovered() || seek_resp.dragged();
                let handle_r = if hot { 8.0_f32 } else { 6.0_f32 };
                painter.circle_filled(egui::Pos2::new(fill_x, track_y), handle_r,
                    if hot { self.track_accent_bright() } else { pal::text_strong(dark) });
            }
            // Time labels
            painter.text(
                egui::Pos2::new(row_rect.left() + h_pad, track_y),
                egui::Align2::LEFT_CENTER,
                &elapsed_str,
                egui::FontId::monospace(11.0),
                pal::text(dark),
            );
            painter.text(
                egui::Pos2::new(row_rect.right() - h_pad, track_y),
                egui::Align2::RIGHT_CENTER,
                &total_str,
                egui::FontId::monospace(11.0),
                pal::text(dark),
            );

            // 4. Commit seek on drag-release or click.
            if seek_resp.drag_stopped() || seek_resp.clicked() {
                if let (Some(dur), Some(_idx)) = (total, self.current_index) {
                    let target_secs = self.seek_pos * dur.as_secs_f32();
                    self.seek_to(Duration::from_secs_f32(target_secs));
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
                    // The queued gapless track may no longer be the right next.
                    self.flush_gapless();
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
                        .color(pal::text_dim(ui.visuals().dark_mode)),
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
                        ui.label(RichText::new(label).size(11.0).color(pal::text_dim(ui.visuals().dark_mode)));
                    }
                }

                if self.play_state != PlayState::Stopped {
                    let lufs = self.spectrum_window.momentary_lufs;
                    if lufs.is_finite() {
                        ui.add_space(6.0);
                        ui.label(RichText::new(format!("{:.1} LUFS", lufs))
                            .size(11.0).color(pal::ok(ui.visuals().dark_mode)));
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
                        RichText::new("📊 Spectrum").size(13.0).color(pal::accent(ui.visuals().dark_mode))
                    } else {
                        RichText::new("📊 Spectrum").size(13.0)
                    };
                    if ui.button(spectrum_label).clicked() {
                        self.spectrum_window.open = !self.spectrum_window.open;
                    }
                    ui.add_space(8.0);
                    let info_enabled = self.current_index.is_some();
                    let info_label = if self.info_open {
                        RichText::new("ℹ Info").size(13.0).color(pal::accent(ui.visuals().dark_mode))
                    } else {
                        RichText::new("ℹ Info").size(13.0)
                    };
                    if ui.add_enabled(info_enabled, egui::Button::new(info_label)).clicked() {
                        self.info_open = !self.info_open;
                    }
                    ui.add_space(8.0);

                    // ── Bit-perfect: device picker (▾) + toggle ─────────────
                    // right_to_left layout: the toggle is added first so it
                    // sits to the right of the picker.
                    let bp_label = if self.bit_perfect {
                        RichText::new("💎 Bit-Perfect").size(13.0).color(pal::ok(ui.visuals().dark_mode))
                    } else {
                        RichText::new("💎 Bit-Perfect").size(13.0)
                    };
                    let mut hover = String::from(
                        "Direct device output at the file's native sample rate.\n\
                         Decoded at full precision (24-bit safe), EQ bypassed.");
                    if self.volume < 0.999 {
                        hover.push_str("\n⚠ Volume below 100% rescales samples — set volume to max for true bit-perfect.");
                    }
                    if ui.button(bp_label).on_hover_text(hover).clicked() {
                        self.toggle_bit_perfect();
                    }

                    ui.menu_button(RichText::new("🔈▾").size(13.0), |ui| {
                        ui.set_min_width(320.0);
                        ui.label(RichText::new("Bit-perfect output device").strong().size(12.0));
                        ui.separator();

                        let track_sr = self.current_index
                            .and_then(|i| self.playlist.get(i))
                            .and_then(|t| t.sample_rate);

                        let mut pick: Option<Option<String>> = None;
                        if ui.selectable_label(self.bp_device.is_none(), "System default").clicked() {
                            pick = Some(None);
                            ui.close_menu();
                        }
                        match self.bp_devices {
                            None => { ui.add_space(2.0); ui.spinner(); ui.label(RichText::new("Scanning devices…").size(11.0)); }
                            Some(ref devs) if devs.is_empty() => {
                                ui.label(RichText::new("No output devices found").size(11.0).color(pal::text_dim(ui.visuals().dark_mode)));
                            }
                            Some(ref devs) => {
                                for d in devs {
                                    let selected = self.bp_device.as_deref() == Some(d.name.as_str());
                                    let mut name = d.name.clone();
                                    if d.is_default { name.push_str("  (default)"); }
                                    // Flag devices that can't do the current track's rate.
                                    let caps = match track_sr {
                                        Some(sr) if !d.supports_rate(sr) =>
                                            format!("{}   ⚠ no {}", d.summary(), fmt_hz(sr)),
                                        _ => d.summary(),
                                    };
                                    let resp = ui.selectable_label(selected, name)
                                        .on_hover_text(&caps);
                                    ui.label(RichText::new(caps).size(10.0).color(pal::text_faint(ui.visuals().dark_mode)));
                                    if resp.clicked() {
                                        pick = Some(Some(d.name.clone()));
                                        ui.close_menu();
                                    }
                                }
                            }
                        }
                        ui.separator();
                        if ui.button(RichText::new("⟳ Rescan").size(12.0)).clicked() {
                            self.bp_devices = None;
                            self.bp_scan_rx = Some(bitperfect::spawn_device_scan());
                        }
                        if let Some(dev) = pick {
                            self.select_bp_device(dev);
                        }
                    }).response.on_hover_text("Choose the output device for bit-perfect playback");

                    ui.add_space(8.0);

                    // ── ReplayGain (loudness normalization) ─────────────────
                    let rg_active = self.rg.mode != RgMode::Off;
                    let rg_label = if rg_active {
                        RichText::new(format!("🔊 RG {}", self.rg.mode.label()))
                            .size(13.0).color(pal::ok(ui.visuals().dark_mode))
                    } else {
                        RichText::new("🔊 RG").size(13.0)
                    };
                    let rg_hover = if self.bit_perfect && rg_active {
                        "ReplayGain is bypassed in bit-perfect mode — a gain change breaks bit-perfectness.".to_string()
                    } else if rg_active {
                        format!("Loudness normalization: {} · currently {:+.1} dB", self.rg.mode.label(), self.rg_applied_db())
                    } else {
                        "Loudness normalization (off) — plays original levels".to_string()
                    };
                    ui.menu_button(rg_label, |ui| {
                        ui.set_min_width(250.0);
                        ui.label(RichText::new("ReplayGain — loudness normalization").strong().size(12.0));
                        if self.bit_perfect {
                            ui.label(RichText::new("💎 Bypassed in bit-perfect mode")
                                .size(11.0).color(pal::ok(ui.visuals().dark_mode)));
                        }
                        ui.separator();
                        let mut changed = false;
                        for mode in [RgMode::Off, RgMode::Track, RgMode::Album] {
                            if ui.selectable_label(self.rg.mode == mode, mode.label()).clicked() {
                                self.rg.mode = mode;
                                changed = true;
                            }
                        }
                        ui.separator();
                        if ui.checkbox(&mut self.rg.prevent_clip, "Prevent clipping (cap gain by peak)").changed() {
                            changed = true;
                        }
                        ui.separator();
                        if self.rg.mode == RgMode::Off {
                            ui.label(RichText::new("Off — original file levels")
                                .size(11.0).color(pal::text_dim(ui.visuals().dark_mode)));
                        } else {
                            let src = self.current_index.and_then(|i| self.playlist.get(i)).map(|t| {
                                let tagged = match self.rg.mode {
                                    RgMode::Album => t.rg_album_gain.or(t.rg_track_gain),
                                    _ => t.rg_track_gain.or(t.rg_album_gain),
                                }.is_some();
                                if tagged { "from ReplayGain tag" } else { "from measured loudness" }
                            }).unwrap_or("—");
                            ui.label(RichText::new(format!("Applied: {:+.1} dB  ({src})", self.rg_applied_db()))
                                .size(11.0).color(pal::text_dim(ui.visuals().dark_mode)));
                            ui.label(RichText::new(format!("Untagged target: {:.0} LUFS", RG_TARGET_LUFS))
                                .size(10.0).color(pal::text_faint(ui.visuals().dark_mode)));
                        }
                        if changed {
                            save_rg_settings(&moosik_dir(), &self.rg);
                        }
                    }).response.on_hover_text(rg_hover);

                    ui.add_space(8.0);

                    // ── Appearance (text size · accent) ─────────────────────
                    // The spectrum palette lives in the spectrum window itself
                    // (next to the View controls), since it only affects that view.
                    ui.menu_button(RichText::new("🎨 Look").size(13.0), |ui| {
                        ui.set_min_width(230.0);
                        ui.label(RichText::new("Appearance").strong().size(12.0));
                        ui.separator();
                        let mut changed = false;

                        ui.label(RichText::new("Theme")
                            .size(11.0).color(pal::text_dim(ui.visuals().dark_mode)));
                        ui.horizontal(|ui| {
                            for tm in [ThemeMode::Dark, ThemeMode::Light] {
                                if ui.selectable_label(self.appearance.theme == tm, tm.label()).clicked()
                                    && self.appearance.theme != tm
                                {
                                    self.appearance.theme = tm;
                                    apply_theme(ctx, tm.is_dark());
                                    changed = true;
                                }
                            }
                        });

                        ui.add_space(6.0);
                        ui.separator();
                        ui.label(RichText::new("Text size")
                            .size(11.0).color(pal::text_dim(ui.visuals().dark_mode)));
                        // Edit a draft, then commit on Apply — a live zoom change
                        // mid-drag resizes the menu under the cursor.
                        ui.add(
                            Slider::new(&mut self.ui_scale_draft, UI_SCALE_MIN..=UI_SCALE_MAX)
                                .custom_formatter(|v, _| format!("{:.0}%", v * 100.0))
                                .custom_parser(|s| {
                                    s.trim().trim_end_matches('%').parse::<f64>().ok().map(|v| v / 100.0)
                                }),
                        );
                        ui.horizontal(|ui| {
                            let pending = (self.ui_scale_draft - self.appearance.ui_scale).abs() > 1e-3;
                            if ui.add_enabled(pending, egui::Button::new("Apply")).clicked() {
                                self.appearance.ui_scale =
                                    self.ui_scale_draft.clamp(UI_SCALE_MIN, UI_SCALE_MAX);
                                save_appearance(&moosik_dir(), &self.appearance);
                            }
                            if ui.small_button("Reset").clicked() {
                                self.ui_scale_draft = 1.0;
                                self.appearance.ui_scale = 1.0;
                                save_appearance(&moosik_dir(), &self.appearance);
                            }
                            ui.label(RichText::new(format!("now {:.0}%", self.appearance.ui_scale * 100.0))
                                .size(10.0).color(pal::text_faint(ui.visuals().dark_mode)));
                        });

                        ui.add_space(6.0);
                        ui.separator();
                        if ui.checkbox(&mut self.appearance.art_accent,
                            "Tint UI with album-art accent").changed() {
                            changed = true;
                        }
                        ui.label(RichText::new("Off → fixed brand accent")
                            .size(10.0).color(pal::text_faint(ui.visuals().dark_mode)));

                        if changed {
                            save_appearance(&moosik_dir(), &self.appearance);
                        }
                    }).response.on_hover_text("Palette, text size, and accent — all optional");
                });
            });

            if !self.status_msg.is_empty() {
                ui.add_space(4.0);
                ui.horizontal(|ui| {
                    ui.add_space(10.0);
                    ui.label(RichText::new(&self.status_msg).size(11.0).color(pal::text_dim(ui.visuals().dark_mode)));
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
                    .size(13.0).color(pal::text_dim(ui.visuals().dark_mode)));
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
            const ROW_H: f32 = 32.0;
            let mut new_drop_row = n;
            let mut last_row_bottom: Option<(f32, f32, f32)> = None; // (left, right, y)

            // ── Search / filter ─────────────────────────────────────────
            ui.horizontal(|ui| {
                ui.label("🔍");
                ui.add(egui::TextEdit::singleline(&mut self.filter_query)
                    .id(egui::Id::new("playlist_filter"))
                    .hint_text(RichText::new("Filter by title, artist, or album…  (Ctrl+F)")
                        .color(pal::text_faint(ui.visuals().dark_mode)))
                    .desired_width(240.0));
                if !self.filter_query.is_empty()
                    && ui.small_button("✕").on_hover_text("Clear filter (Esc)").clicked() {
                    self.filter_query.clear();
                }

                // ── Sort columns (pushed to the right) ───────────────────
                ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
                    let mut sort_click: Option<SortKey> = None;
                    // right_to_left: add in reverse so they read Title→Time.
                    for key in SortKey::ALL.iter().rev() {
                        let active = self.sort_key == Some(*key);
                        let arrow = if active { if self.sort_asc { " ▲" } else { " ▼" } } else { "" };
                        if ui.selectable_label(active,
                            RichText::new(format!("{}{}", key.label(), arrow)).size(12.0))
                            .on_hover_text("Sort by this column (click again to reverse)")
                            .clicked()
                        {
                            sort_click = Some(*key);
                        }
                    }
                    ui.label(RichText::new("Sort:").size(11.0)
                        .color(pal::text_faint(ui.visuals().dark_mode)));
                    if let Some(k) = sort_click { self.sort_playlist(k); }
                });
            });
            ui.add_space(2.0);

            // Real playlist indices to display: all rows, or only those matching
            // the filter. Filtering disables drag-reorder (an ambiguous op on a
            // subset), so with no filter this is the identity 0..n and every
            // path below behaves exactly as before.
            let q = self.filter_query.trim().to_lowercase();
            let filtering = !q.is_empty();
            let visible: Vec<usize> = if filtering {
                (0..n).filter(|&i| {
                    let t = &self.playlist[i];
                    t.title.to_lowercase().contains(&q)
                        || t.artist.to_lowercase().contains(&q)
                        || t.album.to_lowercase().contains(&q)
                }).collect()
            } else {
                (0..n).collect()
            };
            let vn = visible.len();

            egui::ScrollArea::vertical().auto_shrink([false, false]).show_rows(ui, ROW_H, vn, |ui, rows| {
                if self.playlist.is_empty() {
                    ui.add_space(40.0);
                    ui.vertical_centered(|ui| {
                        ui.label(RichText::new("No tracks loaded").size(15.0).color(pal::text_faint(ui.visuals().dark_mode)));
                        ui.add_space(8.0);
                        ui.label(RichText::new("Click \"+ Add Files\" to get started").size(12.0).color(pal::text_faint(ui.visuals().dark_mode)));
                    });
                    return;
                }
                if visible.is_empty() {
                    ui.add_space(40.0);
                    ui.vertical_centered(|ui| {
                        ui.label(RichText::new(format!("No tracks match “{}”", self.filter_query.trim()))
                            .size(14.0).color(pal::text_faint(ui.visuals().dark_mode)));
                    });
                    return;
                }

                let available_w = ui.available_width();

                // A drag can only target rows that are on screen (there is no
                // auto-scroll), so scanning the visible range is sufficient.
                // If the pointer is above the first visible row, drop there.
                if dragging
                    && let Some(pp) = pointer_pos
                    && pp.y < ui.max_rect().top() {
                    new_drop_row = rows.start;
                }
                let rows_end = rows.end;

                for row in rows {
                    let i = visible[row];
                    let track_title  = self.playlist[i].display_title().to_string();
                    let track_artist = self.playlist[i].artist.clone();
                    let track_dur    = self.playlist[i].duration;
                    let is_current        = self.current_index == Some(i);
                    let is_selected       = self.selected.contains(&i);
                    let is_being_dragged  = self.drag_src == Some(i);
                    let dark = ui.visuals().dark_mode;

                    let base_color = if is_selected {
                        pal::row_selected(dark)
                    } else if is_current {
                        pal::row_current(dark)
                    } else if i % 2 == 0 {
                        pal::row_even(dark)
                    } else {
                        pal::row_odd(dark)
                    };

                    let (rect, response) = ui.allocate_exact_size(
                        Vec2::new(available_w, ROW_H),
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
                                egui::Stroke::new(2.0, pal::accent(dark)),
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
                        // A slim accent bar marks the track that's playing.
                        if is_current {
                            let bar = egui::Rect::from_min_max(
                                rect.left_top(),
                                egui::Pos2::new(rect.left() + 3.0, rect.bottom()),
                            );
                            ui.painter().rect_filled(bar, 0.0, self.track_accent);
                        }

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
                            if is_current { self.track_accent } else { pal::text_faint(dark) },
                        );

                        // Title
                        let title_x = inner.min.x + 42.0 + art_offset;
                        let title_w = (inner.width() * 0.45 - art_offset).max(0.0);
                        ui.painter().text(
                            egui::Pos2::new(title_x, cy),
                            egui::Align2::LEFT_CENTER,
                            &track_title,
                            egui::FontId::proportional(13.0),
                            if is_current { pal::text_strong(dark) } else { pal::text(dark) },
                        );

                        // Artist
                        ui.painter().text(
                            egui::Pos2::new(title_x + title_w + 8.0, cy),
                            egui::Align2::LEFT_CENTER,
                            &track_artist,
                            egui::FontId::proportional(12.0),
                            pal::text_dim(dark),
                        );

                        // Duration
                        if let Some(dur) = track_dur {
                            ui.painter().text(
                                egui::Pos2::new(inner.max.x - 8.0, cy),
                                egui::Align2::RIGHT_CENTER,
                                Self::format_duration(dur),
                                egui::FontId::monospace(12.0),
                                pal::text_faint(dark),
                            );
                        }

                        if response.hovered() && !is_being_dragged {
                            let hl = if dark {
                                Color32::from_rgba_unmultiplied(255, 255, 255, 8)
                            } else {
                                Color32::from_rgba_unmultiplied(0, 0, 0, 12)
                            };
                            ui.painter().rect_filled(rect, 0.0, hl);
                        }
                    }

                    // Drag start (reorder is disabled while filtering — a subset
                    // has no unambiguous drop position in the full list).
                    if response.drag_started() && !dragging && !filtering {
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

                // Pointer below every visible row's center → the drop target is
                // the first row just past the visible range (matches the old
                // full-scan behavior; equals n when scrolled to the bottom).
                if dragging && pointer_pos.is_some() && new_drop_row == n && rows_end < n {
                    new_drop_row = rows_end;
                }

                // Insertion line at end of list
                if dragging
                    && let (Some((rx0, rx1, ry)), Some(drop)) = (last_row_bottom, self.drag_over_row) {
                    let src = self.drag_src;
                    let no_op = src.map(|s| s + 1) == Some(n);
                    if drop == n && !no_op {
                        ui.painter().line_segment(
                            [egui::Pos2::new(rx0, ry), egui::Pos2::new(rx1, ry)],
                            egui::Stroke::new(2.0, pal::accent(ui.visuals().dark_mode)),
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

        // Reflect final playback state of this frame to the OS controls.
        self.sync_media_os();
        // Apply ReplayGain (idempotent; catches track change, mode change,
        // bit-perfect toggle, and the measured LUFS landing mid-track).
        self.update_replay_gain();
        // Persist volume / loop mode if they changed this frame (throttled).
        self.persist_player_prefs_if_changed();
    }

    fn on_exit(&mut self, _gl: Option<&eframe::glow::Context>) {
        save_last_playlist(&self.playlist);
        self.spectrum_window.art_settings.save();
        // Final flush of volume / loop mode (a change inside the throttle window
        // may not have been written yet).
        save_player_prefs(&moosik_dir(), &PlayerPrefs {
            volume: self.volume, loop_mode: self.loop_mode,
        });
    }
}
