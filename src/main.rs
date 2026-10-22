mod audio_formats;
mod audio_probe;
#[cfg(test)]
mod audio_format_tests;
mod bitperfect;

/// A counting allocator, armed on one thread at a time. Test builds only.
///
/// "The realtime callback does not allocate" is the kind of claim that is true
/// when written and false three refactors later, and no amount of reading
/// catches the `Vec` that grew a capacity. This measures it: arm the probe on
/// the thread about to run the callback, run it, and count.
///
/// Per-thread by construction, so an armed measurement is not polluted by the
/// rest of a parallel test run. The `const` initialiser matters — a lazily
/// initialised thread-local would itself allocate, from inside the allocator.
#[cfg(test)]
mod alloc_probe {
    use std::alloc::{GlobalAlloc, Layout, System};
    use std::cell::Cell;

    thread_local! {
        /// `(armed, allocations since arming)`.
        static STATE: Cell<(bool, u64)> = const { Cell::new((false, 0)) };
    }

    pub struct Counting;

    #[inline]
    fn note() {
        // `try_with` rather than `with`: during thread-local destruction the
        // slot is gone, and panicking inside the allocator is not recoverable.
        let _ = STATE.try_with(|s| {
            let (armed, n) = s.get();
            if armed {
                s.set((true, n + 1));
            }
        });
    }

    unsafe impl GlobalAlloc for Counting {
        unsafe fn alloc(&self, l: Layout) -> *mut u8 {
            note();
            unsafe { System.alloc(l) }
        }
        unsafe fn alloc_zeroed(&self, l: Layout) -> *mut u8 {
            note();
            unsafe { System.alloc_zeroed(l) }
        }
        unsafe fn realloc(&self, p: *mut u8, l: Layout, new: usize) -> *mut u8 {
            note();
            unsafe { System.realloc(p, l, new) }
        }
        unsafe fn dealloc(&self, p: *mut u8, l: Layout) {
            unsafe { System.dealloc(p, l) }
        }
    }

    /// Start counting on this thread. Pass the result to [`disarm`].
    pub fn arm() -> u64 {
        STATE.with(|s| s.set((true, 0)));
        0
    }

    /// Stop counting and return how many allocations happened while armed.
    pub fn disarm(_before: u64) -> u64 {
        STATE.with(|s| {
            let (_, n) = s.get();
            s.set((false, 0));
            n
        })
    }
}

#[cfg(test)]
#[global_allocator]
static ALLOC_PROBE: alloc_probe::Counting = alloc_probe::Counting;

mod dsd;
#[cfg(test)]
mod egui_layout_cache_tests;

#[cfg(test)]
mod appearance_tests {
    /// An appearance.json written before the translation display existed
    /// still loads, with the translation shown as stored.
    #[test]
    fn an_older_appearance_file_shows_translations_as_stored() {
        let old = r#"{"ui_scale":1.0,"art_accent":true,"theme":"Dark","show_play_count":false,"ui_font":null}"#;
        let a: super::Appearance = serde_json::from_str(old).expect("an old appearance.json");
        assert_eq!(a.lyrics_translation, super::lyrics::zh::ChineseDisplay::Original);
        let round: super::Appearance = serde_json::from_str(&serde_json::to_string(&super::Appearance {
            lyrics_translation: super::lyrics::zh::ChineseDisplay::Simplified, ..Default::default()
        }).unwrap()).unwrap();
        assert_eq!(round.lyrics_translation, super::lyrics::zh::ChineseDisplay::Simplified);
    }

    /// The lyric text size: an older file gets the default; a stored size
    /// survives a save and a load into a new window (a restart); anything
    /// unusable becomes a usable size without resetting the other fields.
    #[test]
    fn the_lyric_text_size_survives_a_restart_and_bad_values() {
        use super::lyrics::ui::{LyricsWindow, TEXT_SIZE_DEFAULT, TEXT_SIZE_MAX, TEXT_SIZE_MIN};
        let old = r#"{"ui_scale":1.0,"art_accent":true,"theme":"Dark","show_play_count":false,"ui_font":null}"#;
        let a: super::Appearance = serde_json::from_str(old).unwrap();
        assert_eq!(a.lyrics_text_size, TEXT_SIZE_DEFAULT);

        let dir = std::env::temp_dir().join(format!("moosik_appearance_{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&dir);
        let saved = super::Appearance { lyrics_text_size: 26.5, ui_scale: 1.2, ..Default::default() };
        super::save_appearance(&dir, &saved);
        let loaded = super::load_appearance(&dir);
        let mut w = LyricsWindow::default();
        w.set_text_size(loaded.lyrics_text_size);
        assert_eq!(w.text_size(), 26.5);
        assert_eq!(loaded.ui_scale, 1.2);
        let _ = std::fs::remove_dir_all(&dir);

        for (raw, want) in [
            ("1e9", TEXT_SIZE_MAX),
            ("-4", TEXT_SIZE_MIN),
            ("\"big\"", TEXT_SIZE_DEFAULT),
            ("null", TEXT_SIZE_DEFAULT),
            ("[20]", TEXT_SIZE_DEFAULT),
        ] {
            let text = format!(r#"{{"ui_scale":1.3,"lyrics_text_size":{raw}}}"#);
            let a: super::Appearance = serde_json::from_str(&text).unwrap_or_else(|e| panic!("{raw}: {e}"));
            assert_eq!(a.lyrics_text_size, want, "{raw}");
            assert_eq!(a.ui_scale, 1.3, "{raw}: a bad size reset another field");
        }
        let mut w = LyricsWindow::default();
        w.set_text_size(f32::NAN);
        assert_eq!(w.text_size(), TEXT_SIZE_DEFAULT);
        w.set_text_size(f32::INFINITY);
        assert_eq!(w.text_size(), TEXT_SIZE_DEFAULT);
    }

    /// A change is written once it is settled, and only if it is a change.
    #[test]
    fn the_lyric_text_size_is_saved_once_a_drag_ends() {
        assert_eq!(super::lyrics_text_size_to_save(20.0, 17.0, false), None, "saved mid-drag");
        assert_eq!(super::lyrics_text_size_to_save(20.0, 17.0, true), Some(20.0));
        assert_eq!(super::lyrics_text_size_to_save(17.0, 17.0, true), None, "saved without a change");
    }
}
mod fonts;
mod frame_log;
#[macro_use]
mod log;
mod lyrics;
mod media_controls;
mod spectrum;
mod tags;

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

/// Install the UI font (`chosen`, by family name) plus the CJK fallbacks.
///
/// Called again whenever the choice changes, so the whole set is rebuilt from
/// scratch rather than mutated — egui owns the font atlas and a partial update
/// would leave stale faces in the family lists.
fn setup_fonts(ctx: &egui::Context, chosen: Option<&str>) {
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

    // The chosen font goes in front of egui's own, so it is what the UI reads
    // as. The CJK fallbacks below are still *appended*, so picking a Latin-only
    // font does not turn Japanese tags into tofu — egui walks the family list
    // for any glyph the first face lacks.
    if let Some(name) = chosen
        && let Some(entry) = fonts::find(name)
        && let Ok(data) = std::fs::read(&entry.path)
    {
        let key = "ui_choice".to_string();
        fonts.font_data.insert(key.clone(), std::sync::Arc::new(egui::FontData {
            font: std::borrow::Cow::Owned(data),
            index: entry.index,
            tweak: Default::default(),
        }));
        for fam in [egui::FontFamily::Proportional, egui::FontFamily::Monospace] {
            fonts.families.entry(fam).or_default().insert(0, key.clone());
        }
    }

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
                break; // found one for this script, skip remaining paths
            }
        }
    }

    // Chinese text — the lyrics' translation — gets families of its own. In the
    // general chain above Japanese comes first, so a Simplified line takes
    // every character Meiryo has from Meiryo, in Japanese forms, and the rest
    // from JhengHei: one line in two typefaces (a Traditional line is all
    // Meiryo). These put one Chinese face first for all of it, the other second
    // for anything it lacks, then Japanese and Korean. The chosen UI font stays
    // in front, as everywhere. egui's own faces come last, after the Chinese
    // ones: before them they supplied the full-width ！？（） in a Latin face,
    // heavy and out of place in a Chinese line; after them they remain the
    // fallback for emoji. Nothing else uses these families, so the UI and
    // Japanese text are drawn exactly as before. Same font data, no copies.
    let base = egui::FontDefinitions::default().families
        .get(&egui::FontFamily::Proportional).cloned().unwrap_or_default();
    let has = |k: &str| fonts.font_data.contains_key(k);
    for (family, order) in [
        (lyrics::ui::ZH_HANT, ["zh_tw", "zh_cn", "jp", "ko", "cjk"]),
        (lyrics::ui::ZH_HANS, ["zh_cn", "zh_tw", "jp", "ko", "cjk"]),
    ] {
        let mut list: Vec<String> = Vec::new();
        if has("ui_choice") { list.push("ui_choice".into()); }
        list.extend(order.iter().filter(|k| has(k)).map(|k| k.to_string()));
        list.extend(base.iter().cloned());
        fonts.families.insert(egui::FontFamily::Name(family.into()), list);
    }

    // A family naming this installation, never drawn with: what lays text out
    // by heights it measured earlier reads it back from the fonts actually in
    // use (`lyrics::ui::fonts_rev`), and so knows when they were replaced —
    // the faces can change under the same names and sizes ("ui_choice" is
    // whichever font was picked). New definitions take effect at egui's next
    // pass, not at this call, which is why it is read back, not remembered.
    static REV: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);
    let rev = REV.fetch_add(1, std::sync::atomic::Ordering::Relaxed) + 1;
    fonts.families.insert(egui::FontFamily::Name(format!("{}{rev}", lyrics::ui::FONTS_REV).into()), base);

    // Always installed, even with no CJK font found: the named families must
    // exist, and without any font of ours they are egui's own.
    ctx.set_fonts(fonts);
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

    // Pin egui to the theme the user picked before installing the style.
    //
    // egui defaults `theme_preference` to `System` (`Memory::Options`), and on
    // any frame where the OS reports its theme it re-applies its *own*
    // `dark_style`/`light_style` over whatever `set_style` last put there. On a
    // machine whose Windows is set to Light that lands on the first frame: the
    // app starts in egui's stock light visuals while the settings panel still
    // reads "Dark", and toggling the switch appears to fix it only because
    // this function runs again. Declaring the preference stops egui resolving
    // a theme we did not ask for; setting both slots means that if it ever
    // does resolve one, it finds this style in both.
    let theme = if dark { egui::Theme::Dark } else { egui::Theme::Light };
    ctx.set_theme(egui::ThemePreference::from(theme));
    ctx.set_visuals_of(egui::Theme::Dark, style.visuals.clone());
    ctx.set_visuals_of(egui::Theme::Light, style.visuals.clone());
    ctx.set_style(style);
    mlog!("theme   applied {}", if dark { "dark" } else { "light" });
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

/// Leave the audio thread room to run.
///
/// The superlet pre-process is rayon-parallel and saturating: on a default
/// global pool it takes one worker per hardware thread and holds them all, at
/// normal priority, for minutes on end. The output callback is one more thread
/// wanting the same cores, and it has a hard deadline — miss it and the device
/// plays whatever is left in its buffer, which is audible.
///
/// Reserving two threads costs a little throughput (the analyser is the only
/// heavy rayon user, and it scales sub-linearly at the top end anyway) and buys
/// the scheduler somewhere to put the render thread that isn't "preempt a
/// worker mid-FFT". `build_global` can only be called once and only before the
/// pool is first used, so it happens here, before anything spawns.
fn cap_worker_threads() {
    let cores = std::thread::available_parallelism().map(|n| n.get()).unwrap_or(4);
    let workers = cores.saturating_sub(2).max(1);
    // Failure means the pool was already built — nothing to do, and not worth
    // failing a launch over.
    let _ = rayon::ThreadPoolBuilder::new().num_threads(workers).build_global();
}

fn main() -> eframe::Result {
    log::init();
    log::install_panic_hook();
    raise_timer_resolution();
    cap_worker_threads();
    let icon = eframe::icon_data::from_png_bytes(
        include_bytes!("../assets/icon.png")
    ).expect("invalid icon PNG");
    mlog!("icon    window icon decoded, {}x{}", icon.width, icon.height);
    // Adapter enumeration and the one-off GPU probe, off the UI thread. Both
    // are first-call-only and both can be slow on unfamiliar hardware; done
    // lazily they landed inside a repaint and looked like a hang.
    spectrum::gpu_calib::warm();

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
    codec: &'static str,
    lossless: Option<bool>,
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
            codec: m.codec, lossless: m.lossless,
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
    codec: &'static str,
    lossless: Option<bool>,
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

/// Read a file's tags with lofty. DSD containers aren't a lofty file type,
/// but both carry a standard ID3v2 blob (DSF via a header pointer, DFF via an
/// `ID3 ` chunk) — extract it and hand it to lofty's ID3v2-capable MPEG
/// reader with properties disabled, so titles, ReplayGain TXXX frames and
/// cover art all flow through the exact same pipeline as every other format.
fn probe_tagged(path: &Path) -> Option<lofty::file::TaggedFile> {
    if dsd::is_dsd_path(path) {
        let info = dsd::parse_file(path).ok()?;
        let blob = dsd::read_id3_blob(path, &info)?;
        Probe::with_file_type(std::io::Cursor::new(blob), lofty::file::FileType::Mpeg)
            .options(lofty::config::ParseOptions::new().read_properties(false))
            .read().ok()
    } else {
        Probe::open(path).ok()?.read().ok()
    }
}

fn read_metadata(path: &PathBuf) -> TrackMeta {
    let fallback_title = path.file_stem()
        .and_then(|s| s.to_str()).unwrap_or("Unknown").to_string();
    let file_size = std::fs::metadata(path).map(|m| m.len()).unwrap_or(0);

    let dsd_info = dsd::is_dsd_path(path).then(|| dsd::parse_file(path).ok()).flatten();
    let tagged = probe_tagged(path);
    // Lofty remains the tag editor. The decoder probe supplies audio identity
    // and read-only fallback metadata for containers Lofty does not handle.
    let probed = if dsd_info.is_none() { audio_probe::probe(path) } else { None };
    let codec = if dsd_info.is_some() { "DSD" }
        else { probed.as_ref().map_or(codec_name(path), |p| p.codec) };
    let lossless = if dsd_info.is_some() { Some(true) }
        else { probed.as_ref().and_then(|p| p.lossless) };

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
        (probed.as_ref().and_then(|p| p.title.clone()).unwrap_or(fallback_title),
         probed.as_ref().and_then(|p| p.artist.clone()).unwrap_or_else(|| "Unknown Artist".into()),
         probed.as_ref().and_then(|p| p.album.clone()).unwrap_or_else(|| "Unknown Album".into()), None, None, None)
    };

    let (duration, sample_rate, channels, bit_depth, bitrate) = if let Some(ref i) = dsd_info {
        // Stream properties come from the DSD header, not lofty.
        (
            Some(i.duration()),
            Some(i.sample_rate),
            Some(i.channels as u8),
            Some(1),
            Some(i.sample_rate / 1000 * i.channels), // 1 bit × rate × channels
        )
    } else if let Some(ref t) = tagged {
        let p = t.properties();
        let secs = p.duration().as_secs();
        (
            if secs > 0 { Some(Duration::from_secs(secs)) } else { None },
            p.sample_rate(), p.channels(), p.bit_depth(), p.audio_bitrate(),
        )
    } else {
        probed.as_ref().map_or((None, None, None, None, None), |p|
            (p.duration, p.sample_rate, p.channels, p.bit_depth, None))
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
                duration, sample_rate, channels, bit_depth, bitrate, file_size, codec, lossless,
                rg_track_gain, rg_album_gain, rg_track_peak, rg_album_peak }
}

// ---------------------------------------------------------------------------
// Audio info helpers
// ---------------------------------------------------------------------------

fn codec_name(path: &Path) -> &'static str {
    audio_formats::container_label(path)
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
/// Why a background seek did not produce a positioned decoder.
///
/// One `None` used to carry all of these. They are not the same thing to a
/// listener and they are not the same thing to the engine: a file that cannot
/// be opened is a broken playlist entry, a container that will not seek is a
/// limitation of the format, and a thread that would not start is this process
/// running out of something. Reporting all three as "this file could not be
/// seeked" was, for two of the three, simply untrue.
#[derive(Clone, Debug, PartialEq)]
enum SeekError {
    /// The file could not be opened.
    Open(String),
    /// The bytes are not something we can decode.
    Decode(String),
    /// The decoder ran out of samples before reaching the target: the seek was
    /// past the end of the audio the file actually contains.
    PastEnd { landed: Duration },
    /// The worker thread could not be started.
    Spawn(String),
    /// The worker ended without sending anything — it panicked.
    WorkerLost,
    /// Rebuilding the decimated-DSD source failed.
    Dsd(String),
    /// There is no output device to install the result on.
    Output(String),
    /// The seek was superseded before it landed.
    ///
    /// Not a failure the listener needs told about: another seek, a track
    /// change or a stop replaced this one, and the worker stopped decoding
    /// samples nobody is waiting for. It exists so the outcome is explicit
    /// rather than a receiver that quietly disconnects and is reported as a
    /// worker that "ended without a result".
    Cancelled,
}

impl SeekError {
    fn describe(&self) -> String {
        match self {
            SeekError::Open(e) => format!("this file could not be opened: {e}"),
            SeekError::Decode(e) => format!("this file could not be decoded: {e}"),
            SeekError::PastEnd { landed } => format!(
                "the file ends at {:.0}:{:02.0}, before that point",
                landed.as_secs() / 60,
                landed.as_secs() % 60
            ),
            SeekError::Spawn(e) => format!("the seek could not be started: {e}"),
            SeekError::WorkerLost => "the seek worker stopped without a result".into(),
            SeekError::Dsd(e) => format!("this DSD file could not be repositioned: {e}"),
            SeekError::Output(e) => format!("the output could not be opened: {e}"),
            SeekError::Cancelled => "the seek was replaced before it landed".into(),
        }
    }

    /// The typed failure to publish when a seek that had already begun
    /// reopening the output cannot finish.
    ///
    /// Every arm keeps the sentence `describe` would have given, so the panel
    /// says the same thing the status line does — and says it instead of
    /// staying on "Verifying output…" for the rest of the session.
    fn to_reason(&self) -> bitperfect::state::FailureReason {
        use bitperfect::state::FailureReason;
        match self {
            SeekError::Output(_) => FailureReason::DeviceUnavailable(self.describe()),
            SeekError::Spawn(_) | SeekError::WorkerLost => {
                FailureReason::SessionThreadStart(self.describe())
            }
            SeekError::Open(_) => FailureReason::SourceOpen(self.describe()),
            SeekError::Decode(_) => FailureReason::Decode(self.describe()),
            SeekError::PastEnd { .. } | SeekError::Dsd(_) | SeekError::Cancelled => {
                FailureReason::Seek(self.describe())
            }
        }
    }

    /// Whether this is something the listener needs to be told.
    ///
    /// A superseded seek is not: it was replaced by something the listener did
    /// themselves, and the thing that replaced it has its own outcome.
    fn is_worth_reporting(&self) -> bool {
        !matches!(self, SeekError::Cancelled)
    }
}

/// The time at the start of frame `frames`, at `rate`.
///
/// One place, because a position derived from a sample count rather than a
/// frame count is how the seek worker came to land between two channels.
fn frame_time(frames: u64, rate: u32) -> Duration {
    Duration::from_secs_f64(frames as f64 / rate.max(1) as f64)
}

/// What the seek worker produces: a decoder, where it actually landed, and
/// what the source turned out to be.
///
/// The landed position is the part that was missing. The worker stops
/// discarding samples when the decoder runs out, so a seek past the end of the
/// audio returned a decoder sitting at EOF — and the caller, which had only
/// asked for a target, published the target. The slider moved to a position
/// the file does not contain and the track ended immediately, which read as a
/// track that had failed.
struct SeekLanding {
    decoder: Decoder<BufReader<File>>,
    /// Where the decoder is, which is not always where it was asked to go.
    landed: Duration,
    /// What the file actually is — carried from the thread that opened it, so
    /// nothing downstream has to guess or fabricate it.
    source: bitperfect::format::SourceFormat,
}

struct PendingSeek {
    rx: std::sync::mpsc::Receiver<Result<SeekLanding, SeekError>>,
    was_paused: bool,
    /// The file being sought within, so a landing can be attributed and a
    /// failure can name it.
    path: PathBuf,
    /// Where playback was before the seek began, so a failure can put the
    /// engine back somewhere coherent instead of leaving it at a position
    /// nothing is playing from.
    from: Duration,
    /// The position currently being shown on the strength of the request
    /// alone, if any. `None` when the caller left the display where it was.
    provisional: Option<Duration>,
}

/// rodio adapter over the decimated-DSD stream, for the fallback path when a
/// device can't take the DoP carrier rate. Seeking is instant (DSD is
/// byte-addressable), so `Sink::try_seek` always takes the fast path.
struct DsdRodioSource(dsd::decimate::DsdFilePcmSource);

impl Iterator for DsdRodioSource {
    type Item = f32;
    fn next(&mut self) -> Option<f32> { self.0.next() }
}

impl Source for DsdRodioSource {
    fn current_frame_len(&self) -> Option<usize> { None }
    fn channels(&self) -> u16 { self.0.channels() as u16 }
    fn sample_rate(&self) -> u32 { self.0.out_rate() }
    fn total_duration(&self) -> Option<Duration> { Some(self.0.duration()) }
    fn try_seek(&mut self, pos: Duration) -> Result<(), rodio::source::SeekError> {
        self.0.seek_to_time(pos)
            .map_err(|e| rodio::source::SeekError::Other(Box::new(e)))
    }
}

/// A whole-track value-exactness scan in flight.
///
/// The result carries the identity of the file it was computed for, not just
/// the generation it was started in. The generation alone was not enough: the
/// poller built the cache key from *whatever was playing when the result
/// arrived*, so a track change between start and finish filed one file's
/// verdict under another file's key — and the wrong verdict then survived in
/// the cache for the rest of the session.
struct Q31Scan {
    rx: std::sync::mpsc::Receiver<Q31Outcome>,
    /// The session this scan belongs to.
    generation: u64,
    /// The *audio* generation it was started under.
    ///
    /// The session generation alone was not enough. It is the UI's counter and
    /// it does not move at a gapless hand-off until the UI folds one, so a
    /// verdict computed for the track that was playing when the scan began
    /// could be applied to the track that had since taken its place — a claim
    /// about one file, printed against another. Both clocks have to agree.
    audio_generation: u64,
    /// The file it was started for, identified the same way the cache is.
    key: bitperfect::q31::Q31Key,
}

/// What a scan came back with. Every arm updates visible state: a scan that
/// failed, was cancelled, or whose thread vanished used to leave the panel
/// saying "checking" for the rest of the track.
enum Q31Outcome {
    Verdict(bitperfect::q31::Q31Verdict),
    Failed(String),
}

/// A witness that something the engine held was actually let go.
///
/// A cleanup test on an engine that never opened a device asserts nothing:
/// `bp.is_none()` was already true before the failure it is meant to be about,
/// and every such assertion passes against code that cleans up nothing. The
/// real handles — a rodio `Sink`, a `BpStream`, an ASIO or ALSA session — need
/// a sound card to construct, so a test plants one of these in the slot a
/// half-open would have filled instead. They are dropped by the *same*
/// statements that drop the real handles, so a cleanup step that stops running
/// stops being witnessed.
#[cfg(test)]
#[derive(Debug)]
struct Sentinel(std::sync::Arc<std::sync::atomic::AtomicUsize>);

#[cfg(test)]
impl Drop for Sentinel {
    fn drop(&mut self) {
        self.0.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
    }
}

struct Engine {
    // ── rodio path (normal mode) ───────────────────────────────────────────
    /// Opened lazily on first normal-mode playback. Keeping it closed until
    /// needed avoids the launch-time "device is no longer available" error:
    /// the background device scan probes WASAPI *exclusive* formats, which
    /// invalidates any shared-mode stream that happens to be open.
    rodio_out: Option<(OutputStream, OutputStreamHandle)>,
    sink: Option<Sink>,
    // ── bit-perfect path (direct device stream fed by a symphonia thread) ──
    bp: Option<bitperfect::BpStream>,
    /// Cancels the seek worker currently in flight, if any.
    ///
    /// Dropping the receiver stops the *result* arriving; it does not stop the
    /// work. A deep seek into a hi-res FLAC decodes and discards tens of
    /// millions of samples, and a superseded worker went on doing that — a
    /// core, for seconds, for a position nobody was waiting for. This is how
    /// it is told.
    seek_cancel: std::sync::Arc<std::sync::atomic::AtomicBool>,
    /// Witnesses for the four things an open can take, planted by tests.
    ///
    /// Not state: nothing reads them, and in a release build they do not
    /// exist. They are here so that "this was released" is a fact about a
    /// value being dropped rather than about a field that was already empty.
    #[cfg(test)]
    held_sink: Option<Sentinel>,
    #[cfg(test)]
    held_bp: Option<Sentinel>,
    #[cfg(test)]
    held_native: Option<Sentinel>,
    #[cfg(test)]
    held_scan: Option<Sentinel>,
    /// Test seams for route-dependent behaviour with no sound card: the
    /// shared route's sink is an idle `rodio` sink (nothing is opened), and
    /// `on_bp_stream` can be told that a device route (DoP, native) is the
    /// session's owner. Nothing in a release build.
    #[cfg(test)]
    idle_output: bool,
    #[cfg(test)]
    device_route: Option<bool>,
    pub bit_perfect: bool,
    /// What the output path is actually doing: the request, the open route,
    /// and whether that route is carrying the source exactly.
    ///
    /// Three separate facts. `bit_perfect` above is only the first of them,
    /// and until 1.4.3 it was the only one recorded — which is how a DSD track
    /// decimated to processed PCM kept a green diamond.
    pub bp_state: bitperfect::state::OutputSessionState,
    /// The result of a whole-track value-exactness scan, when one is running.
    ///
    /// The scan carries the generation it was started for; a result from a
    /// superseded session is dropped rather than applied to its successor.
    q31_rx: Option<Q31Scan>,
    /// Cancels the running scan. A track change should not leave a worker
    /// grinding through a file nobody is listening to.
    q31_cancel: std::sync::Arc<std::sync::atomic::AtomicBool>,
    /// Verdicts already reached, keyed on the file's identity.
    ///
    /// Durable: loaded at startup and written back atomically, so a track that
    /// has been proved once is not re-decoded in full on every launch. A
    /// whole-track scan is minutes of decode for a library of hi-res Float32
    /// masters, which is why the label so often never arrived before the track
    /// ended.
    ///
    /// Still advisory. `Q31Guard` re-checks every packet regardless, so a
    /// stale entry can cost a label and can never put an unrepresentable
    /// sample on the wire.
    q31_cache: bitperfect::q31::Q31Cache,
    /// True while the loaded track is DSD, played via DoP on `bp` — always
    /// bit-perfect regardless of the `bit_perfect` (PCM) toggle, since DoP is
    /// the only path that can carry it at all. Volume/EQ are hard-bypassed.
    pub dsd_mode: bool,
    /// Whether the DoP track now playing ends on a half-full carrier frame.
    ///
    /// A carrier frame holds two DSD bytes per channel, so an odd frame count
    /// leaves the last one padded with DSD silence. Such a boundary is not
    /// gapless — see `bp_queue_next_dop`.
    dsd_odd_tail: Option<bool>,
    /// True while a DSD track plays as *native* DSD — raw bits, no DoP
    /// carrier, the only bit-perfect route to DSD512. ASIO on Windows
    /// (asio-dsd builds), ALSA on Linux (alsa-dsd builds). Volume/EQ are
    /// hard-bypassed, like DoP.
    pub dsd_native: bool,
    /// What the native-DSD feeder is doing, when one is running.
    ///
    /// The feeder is the only producer on that path and it used to report
    /// nothing at all: a read error, a truncated file, a torn frame and a
    /// panic all arrived at the UI either as a clean end of track or as
    /// silence with the claim intact. Faults here are stamped with the
    /// generation they were raised in, so a feeder still unwinding after its
    /// session was replaced cannot fault the track that replaced it.
    #[cfg(any(
        all(windows, feature = "asio-dsd"),
        all(target_os = "linux", feature = "alsa-dsd"),
    ))]
    feed_status: std::sync::Arc<bitperfect::native_dsd::FeedStatus>,
    /// Selected ASIO driver for native DSD; None = DSD goes over DoP.
    /// Persisted in bitperfect.json even on builds without the feature.
    pub asio_driver: Option<String>,
    /// Selected ALSA hw: device for native DSD on Linux; None = DoP.
    /// Persisted like asio_driver.
    pub alsa_dsd_device: Option<String>,
    /// The open native-DSD stream (kept across same-rate tracks, like `bp`).
    #[cfg(all(windows, feature = "asio-dsd"))]
    asio: Option<bitperfect::asio_dsd::AsioDsdStream>,
    #[cfg(all(target_os = "linux", feature = "alsa-dsd"))]
    alsa: Option<bitperfect::alsa_dsd::AlsaDsdStream>,
    /// True while a DSD track plays through the *fallback* path: DoP was
    /// unavailable (device can't take the carrier rate), so the bitstream is
    /// decimated to PCM and played through the normal rodio path — volume,
    /// EQ and ReplayGain all apply, exactly like any PCM track.
    pub dsd_fallback: bool,
    /// Why DoP was unavailable, for the status line (set with dsd_fallback).
    pub dsd_fallback_note: Option<String>,
    /// Target PCM rate for the playback fallback decimation. A playback
    /// setting (`SpectrumWindow::dsd_fallback_rate`), separate from the
    /// spectrum's preprocessing rate; the live tap on this route runs at what
    /// it produces, and the spectrum is told so (`spectrum_live_rate`).
    pub dsd_fallback_target: u32,
    /// "DSD64" / "DSD128" / … label for the currently loaded DSD track.
    pub dsd_label: Option<String>,
    /// Selected bit-perfect output device name; None = system default.
    pub bp_device: Option<String>,
    /// The source format most recently established by decoding, and the file
    /// it belongs to.
    ///
    /// Kept so that a track which fails its exact open and is retried through
    /// the shared mixer is still described by what it *is*. The shared path
    /// has only `rodio`'s rate and channel count to go on and used to publish
    /// every fallback as Float32, so a 24-bit FLAC that the DAC refused was
    /// reported to the user as a float source.
    last_prepared: Option<(PathBuf, bitperfect::format::SourceFormat)>,
    /// The track that is playing, and what it is — whatever route carries it.
    ///
    /// `last_prepared` is not this and was being used as though it were. It is
    /// set on the exact route and on a completed seek, and nowhere else, so
    /// after ordinary shared playback or a DSD fallback it holds *a different
    /// file*: whichever one last went through `prepare`. Anything that reached
    /// for "the current track" through it got the previous exact track, or
    /// nothing at all — and `resume` after a failed seek reopened that.
    ///
    /// The `MediaSource` is optional because it genuinely is: a file the
    /// shared mixer will play but `prepare` cannot describe has a path and no
    /// identity, and saying so is better than inventing one.
    current: Option<CurrentTrack>,
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
    /// The shared-route track appended behind the one playing, and what it is.
    ///
    /// `None` inside the pair means `prepare` could not say — an unknown
    /// identity, which is left as one rather than filled in with a guess.
    queued_next: Option<QueuedShared>,
    /// The audio generation the UI has folded up to.
    ///
    /// The render thread crosses a gapless boundary at the sample and the UI
    /// finds out a frame later, so between those two moments the stream's
    /// evidence is about a track `bp_state` has not rolled over to yet.
    /// Comparing this against `BpStream::audio_generation` is what stops that
    /// evidence being charged to the track that just finished — a second lock
    /// on the same door as folding boundaries before polling, and the one that
    /// does not depend on a caller keeping two statements in order.
    bp_audio_gen: u64,
    /// The same, for the native ASIO/ALSA session.
    ///
    /// Native routes have no gapless hand-off inside one stream today, so this
    /// only moves when a session starts — but the evidence is read by it, so
    /// that a callback still unwinding from a track that has been replaced
    /// cannot have its dropout charged to the replacement.
    native_audio_gen: u64,
    /// A seek that failed without waiting for a worker, parked for the tick.
    ///
    /// The synchronous paths cannot return a result — their callers are UI
    /// handlers that have already committed to the seek — so the failure goes
    /// here and is collected by the same tick that collects the asynchronous
    /// ones. Typed, so the tick has the same thing to report either way.
    seek_failure: Option<SeekError>,
    /// Set once a bit-perfect stream has grabbed the output device. On Windows,
    /// WASAPI exclusive mode suspends the long-lived rodio (shared-mode)
    /// OutputStream and cpal doesn't recover it when exclusive mode is released
    /// — so the stream is recreated before the next normal-mode sink, else the
    /// rodio path is silent forever after bit-perfect has run once.
    rodio_stream_dirty: bool,
}

impl Engine {
    fn new(sample_buf: SampleBuf, stereo_buf: StereoBuf) -> Option<Self> {
        Some(Engine {
            rodio_out: None,
            sink: None,
            bp: None,
            seek_cancel: std::sync::Arc::new(std::sync::atomic::AtomicBool::new(false)),
            #[cfg(test)]
            held_sink: None,
            #[cfg(test)]
            held_bp: None,
            #[cfg(test)]
            held_native: None,
            #[cfg(test)]
            held_scan: None,
            #[cfg(test)]
            idle_output: false,
            #[cfg(test)]
            device_route: None,
            bp_state: bitperfect::state::OutputSessionState::default(),
            q31_rx: None,
            q31_cancel: std::sync::Arc::new(std::sync::atomic::AtomicBool::new(false)),
            q31_cache: {
                let c = bitperfect::q31::Q31Cache::load(&moosik_dir());
                if c.is_empty() {
                    crate::mlog!("q31     no cached verdicts — each track is proved from scratch");
                } else {
                    crate::mlog!("q31     {} cached verdict(s) recovered", c.len());
                }
                c
            },
            bit_perfect: false,
            dsd_mode: false,
            dsd_native: false,
            #[cfg(any(
                all(windows, feature = "asio-dsd"),
                all(target_os = "linux", feature = "alsa-dsd"),
            ))]
            feed_status: std::sync::Arc::new(bitperfect::native_dsd::FeedStatus::new()),
            asio_driver: None,
            alsa_dsd_device: None,
            #[cfg(all(windows, feature = "asio-dsd"))]
            asio: None,
            #[cfg(all(target_os = "linux", feature = "alsa-dsd"))]
            alsa: None,
            dsd_odd_tail: None,
            dsd_fallback: false,
            dsd_fallback_note: None,
            dsd_fallback_target: dsd::decimate::DEFAULT_FALLBACK_RATE,
            dsd_label: None,
            bp_device: None,
            last_prepared: None,
            current: None,
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
            queued_next: None,
            bp_audio_gen: 0,
            native_audio_gen: 0,
            seek_failure: None,
            rodio_stream_dirty: false,
        })
    }

    /// The rodio output handle, opening the stream on first use and
    /// recreating it after an exclusive-mode session may have invalidated it
    /// (`rodio_stream_dirty`). Lazy so nothing holds the device at launch
    /// while the capability scan probes exclusive formats.
    /// Open (or reuse) the shared-mode output stream.
    ///
    /// Every exclusive/native handle is released first. `rodio` cannot open a
    /// device this process is still holding exclusively, and the error it
    /// produces then looks like a missing device rather than a self-inflicted
    /// one. 1.4.2 reached this function straight from the DSD-exact path with
    /// the WASAPI-exclusive stream still open and its Exact state still
    /// published.
    fn rodio_handle(&mut self) -> Result<OutputStreamHandle, String> {
        self.release_exclusive_handles();
        if self.rodio_stream_dirty {
            self.rodio_out = None;
            self.rodio_stream_dirty = false;
        }
        if self.rodio_out.is_none() {
            let (stream, handle) = OutputStream::try_default()
                .map_err(|e| format!("audio output: {e}"))?;
            self.rodio_out = Some((stream, handle));
        }
        Ok(self.rodio_out.as_ref().unwrap().1.clone())
    }

    /// Effective rodio sink volume: user volume × ReplayGain.
    fn rodio_volume(&self) -> f32 { self.volume * self.replay_gain }

    /// True when the active session lives on one of our own device streams
    /// rather than the `rodio` sink.
    ///
    /// This decides which object owns pause, resume, seek, position,
    /// end-of-track, ReplayGain, EQ, the spectrum tap, auto-advance and the
    /// gapless hand-off. Getting it wrong does not mis-label anything — it
    /// sends every one of those operations to the wrong object.
    ///
    /// It is derived from two facts that are both about what is *open*: the
    /// transport the session published, and the handle that transport implies.
    /// Neither is a preference. The predicate it replaces was
    /// `dsd_native || dsd_mode || (bit_perfect && !dsd_fallback)` — three
    /// request flags, so a saved "bit-perfect on" made the engine route
    /// through a `bp` stream that had never opened, or had died, and every
    /// one of those operations went to a stream with nobody behind it.
    ///
    /// Both halves are required. The transport alone would trust a state
    /// nobody had cleared; the handle alone cannot distinguish an open stream
    /// that is carrying this session from one left over from the last.
    /// **Ownership**, not liveness. A backend that has died still owns the
    /// session it was carrying, and is the only object that can say how it
    /// ended.
    ///
    /// This asked whether the stream was still *working*, and every caller
    /// that routes a question to the object holding the session asked it. The
    /// consequence was the worst kind of quiet: the moment a WASAPI backend
    /// died or an ASIO driver requested a reset, this went false, `completion`
    /// stopped asking the object that knew, fell through to the shared-sink
    /// branch, found no sink, and answered `Running`. The track never ended,
    /// the playlist never advanced, `halt_on_failure` never ran, and the
    /// exclusive handle stayed open — for as long as the process lived. The
    /// failure was recorded, correctly, in a place nobody was reading.
    ///
    /// Liveness is a separate question with one job: whether an open stream
    /// may be *reused* for the next track. `backend_dead` and
    /// `reset_requested` are read at those decisions, and only there.
    fn on_bp_stream(&self) -> bool {
        #[cfg(test)]
        if let Some(v) = self.device_route {
            return v;
        }
        session_owner(
            &self.bp_state.transport,
            self.bp.is_some(),
            self.native_handle_open(),
        )
    }

    /// Whether a native handle exists at all, alive or not.
    #[allow(unreachable_code)]
    fn native_handle_open(&self) -> bool {
        #[cfg(all(windows, feature = "asio-dsd"))]
        return self.asio.is_some();
        #[cfg(all(target_os = "linux", feature = "alsa-dsd"))]
        return self.alsa.is_some();
        false
    }

    /// True when the active session lives on a native-DSD stream.
    ///
    /// Same rule as `on_bp_stream`, for the branch that has to come first:
    /// pause, resume, position and end-of-track all reach the ASIO/ALSA
    /// handle rather than the ring. `dsd_native` alone is a request flag and
    /// stayed true after the handle had gone.
    fn on_native_stream(&self) -> bool {
        use bitperfect::Transport;
        matches!(
            self.bp_state.transport,
            Transport::AsioNativeDsd { .. }
                | Transport::AsioPcm { .. }
                | Transport::AlsaNativeDsd { .. }
        ) && self.native_handle_open()
    }

    /// Whether a native-DSD stream is open **and its backend has not died**.
    ///
    /// The reuse predicate, and nothing else. Deciding whether to hand the
    /// next track to an already-open driver is the one question where a
    /// requested reset or a dead writer is the right answer; deciding who owns
    /// the session that is ending is not.
    ///
    /// Compiled to `false` without the matching feature, where `dsd_native`
    /// can never be true anyway.
    #[allow(unreachable_code)]
    fn native_handle_live(&self) -> bool {
        #[cfg(all(windows, feature = "asio-dsd"))]
        return self.asio.as_ref().is_some_and(|a| !a.reset_requested());
        #[cfg(all(target_os = "linux", feature = "alsa-dsd"))]
        return self.alsa.as_ref().is_some_and(|a| !a.failed());
        false
    }

    // ── Native-DSD session accessors ─────────────────────────────────────────
    // One backend per platform: ASIO (Windows, asio-dsd) or ALSA (Linux,
    // alsa-dsd). Compiled to no-ops without the matching feature; `dsd_native`
    // can then never be true, so the no-op returns are unreachable in practice.

    /// True when a native-DSD output is configured for this platform —
    /// the gate for trying the native transport first.
    #[allow(unreachable_code)]
    fn native_dsd_selected(&self) -> bool {
        #[cfg(windows)]
        return self.asio_driver.is_some();
        #[cfg(target_os = "linux")]
        return self.alsa_dsd_device.is_some();
        false
    }

    /// Pause the native stream if it was playing; true if state changed.
    fn native_pause(&self) -> bool {
        #[cfg(all(windows, feature = "asio-dsd"))]
        if let Some(a) = &self.asio && !a.is_paused() { a.pause(); return true; }
        #[cfg(all(target_os = "linux", feature = "alsa-dsd"))]
        if let Some(a) = &self.alsa && !a.is_paused() { a.pause(); return true; }
        false
    }

    /// Resume the native stream if it was paused; true if state changed.
    fn native_resume(&self) -> bool {
        #[cfg(all(windows, feature = "asio-dsd"))]
        if let Some(a) = &self.asio && a.is_paused() { a.resume(); return true; }
        #[cfg(all(target_os = "linux", feature = "alsa-dsd"))]
        if let Some(a) = &self.alsa && a.is_paused() { a.resume(); return true; }
        false
    }

    fn native_is_paused(&self) -> bool {
        #[cfg(all(windows, feature = "asio-dsd"))]
        if let Some(a) = &self.asio { return a.is_paused(); }
        #[cfg(all(target_os = "linux", feature = "alsa-dsd"))]
        if let Some(a) = &self.alsa { return a.is_paused(); }
        false
    }

    fn native_played(&self) -> Duration {
        #[cfg(all(windows, feature = "asio-dsd"))]
        if let Some(a) = &self.asio { return a.played(); }
        #[cfg(all(target_os = "linux", feature = "alsa-dsd"))]
        if let Some(a) = &self.alsa { return a.played(); }
        Duration::ZERO
    }

    /// Release the native-DSD output (ASIO drivers and ALSA hw: devices both
    /// hold the hardware exclusively, so this must happen before a
    /// WASAPI/cpal/rodio path can open the same DAC).
    fn native_close(&mut self) {
        #[cfg(all(windows, feature = "asio-dsd"))]
        { self.asio = None; }
        #[cfg(all(target_os = "linux", feature = "alsa-dsd"))]
        { self.alsa = None; }
        #[cfg(test)]
        { self.held_native = None; }
    }

    /// Release the exact-output stream, and with it the endpoint it holds.
    ///
    /// A function rather than an assignment because there are eight places
    /// that let this go and every one of them has to let go of the same
    /// things. In a test build that includes the witness, which is what makes
    /// "this was released" provable on an engine that never had a device.
    fn release_bp(&mut self) {
        self.bp = None;
        #[cfg(test)]
        { self.held_bp = None; }
    }

    /// Stop and release the shared-mixer sink.
    ///
    /// Stopping matters as much as dropping: a sink that is dropped while
    /// still playing goes on until its queue empties.
    /// Release the shared sink — and with it, the shared analysis route.
    ///
    /// The one authoritative place a shared session ends, because there are
    /// several paths that end one and every one of them comes through here: a
    /// stop, a halt, a failed open, a rebuild, and the slow background seek,
    /// which releases the sink and then spends seconds decoding on a worker
    /// thread before anything replaces it.
    ///
    /// The invalidation cannot be left to the source's own `Drop`. `Sink::stop`
    /// sets an atomic that the mixer notices asynchronously, so the source may
    /// outlive this call by a callback or two; its `Drop` then runs on the audio
    /// thread and takes the buffer lock with `try_lock`, which is allowed to
    /// miss. Doing it here, synchronously and unconditionally, is what makes a
    /// seek report "no live PCM" the moment it starts rather than whenever the
    /// mixer gets round to it.
    fn release_sink(&mut self) -> bool {
        let mut released = false;
        if let Some(sink) = self.sink.take() {
            sink.stop();
            released = true;
        }
        #[cfg(test)]
        {
            released |= self.held_sink.take().is_some();
        }
        // Once, and only when a sink actually existed: an exact or native route
        // that happens to pass through here is not the shared route, and its
        // display claim is not this function's to revoke. Written as one call
        // rather than one per branch so a test that reaches it through the
        // witness is exercising the same line production reaches.
        //
        // Returned so the wider teardowns below can advance the route exactly
        // once. They used to release the sink — which advanced it — and then
        // advance it again unconditionally, which is harmless but meant the
        // count was not the thing it claimed to be.
        if released {
            spectrum::invalidate_live_pcm_route(&self.stereo_buf, &self.sample_buf);
        }
        released
    }

    /// Start `path` on this platform's native-DSD backend.
    #[allow(unreachable_code, unused_variables)]
    fn start_native(&mut self, path: &Path, opening: Opening) -> Result<(), String> {
        #[cfg(all(windows, feature = "asio-dsd"))]
        return self.start_asio_native(path, opening);
        #[cfg(all(target_os = "linux", feature = "alsa-dsd"))]
        return self.start_alsa_native(path, opening);
        Err("native DSD is not compiled into this build".into())
    }

    /// Which family of route `path` would be started on right now.
    ///
    /// The one place the question is answered, so that `play_file` and
    /// `restart_current_track` cannot disagree about it — and they did.
    ///
    /// It is a question about the *track*. A bit-perfect preference is a
    /// preference about PCM: it says nothing about a DSD file, which runs its
    /// own ladder — native, then DoP, then decimation — whether the toggle is
    /// on or off. And a configured native-DSD driver says nothing about a PCM
    /// file, which does not go near it.
    fn route_for(&self, path: &Path) -> Route {
        if dsd::is_dsd_path(path) {
            // Native, DoP and the decimating fallback are all tried by
            // `play_file`, in that order. Even the fallback has to be reached
            // through it: `play_seeked_async` opens the file with `rodio`,
            // which cannot read a DSD container at all.
            return Route::Device;
        }
        if self.bit_perfect { Route::Device } else { Route::Shared }
    }

    /// Open `path` and be at `target`, in that order and as one step.
    ///
    /// The position is part of opening, not something done to a route that is
    /// already running. It was the latter: `play_file` opened every route at
    /// zero and the caller seeked afterwards, so a restart to thirty seconds —
    /// a bit-perfect toggle, an output-device change, anything that rebuilds
    /// the stream mid-track — emitted the opening of the track first. Audibly,
    /// on every one of them. And if the seek then failed, that opening was all
    /// the listener got, from a device the player was still holding.
    ///
    /// Every route reaches its position before it can sound: the PCM and DoP
    /// paths hand `target` to `prepare`, which seeks the container before a
    /// device is touched; the native path hands it to its own opener; the
    /// decimated fallback is byte-addressable and starts at the offset; and
    /// the shared mixer holds the sink silent until `try_seek` has moved it.
    ///
    /// Returns where the route actually landed, which is not always what was
    /// asked for — a FLAC without a seek table lands on the nearest frame it
    /// can reach — and it is the landing the seek bar and the spectrum follow.
    fn play_file_at(
        &mut self,
        path: &PathBuf,
        duration: Option<Duration>,
        opening: Opening,
    ) -> Result<Duration, bitperfect::OpenError> {
        self.stop();
        if dsd::is_dsd_path(path) {
            // Best transport first: native DSD (ASIO on Windows, ALSA on
            // Linux, with an output selected — the only route to DSD512),
            // then bit-perfect DoP, then decimated PCM through the normal
            // path so the track is never simply unplayable.
            let native_err = if self.native_dsd_selected() {
                match self.start_native(path, opening) {
                    Ok(()) => {
                        self.dsd_native = true;
                        // Native DSD attaches no tap of any kind: the bits go
                        // to the driver as DSD and never become PCM anywhere in
                        // this process. Published rather than inferred, so the
                        // spectrum says "no live PCM" instead of drawing
                        // whatever the previous track happened to leave behind.
                        spectrum::invalidate_live_pcm_route(
                            &self.stereo_buf,
                            &self.sample_buf,
                        );
                        None
                    }
                    Err(e) => {
                        let e = e.to_string();
                        // Always audible in the console — a silent fall-through
                        // to DoP looks like "native just doesn't work".
                        crate::mlog!("[native-dsd] native DSD failed, trying DoP: {e}");
                        Some(e)
                    }
                }
            } else {
                None
            };
            if !self.dsd_native {
                match self.start_dop(path, opening) {
                    Ok(()) => self.dsd_mode = true,
                    // Decimation is a processed route, and it happens only
                    // when the *endpoint refused the carrier format* and the
                    // policy permits processing. An unreadable file, a refused
                    // seek, a self-contradictory `MOOSIK_BP_FORMAT` or a COM
                    // failure all stop here with their own reason: decimating
                    // in answer to any of them obeys a request nobody made and
                    // hides the real problem behind a working-looking
                    // fallback.
                    Err(e) if !e.is_device_limitation() => return Err(e),
                    // Strict was asked never to transform. A DSD file the
                    // device cannot carry is exactly the case it exists for.
                    Err(e) if !self.bp_state.policy.allows_processed() => {
                        return Err(bitperfect::OpenError::config(format!(
                            "{}; decimating to PCM would process the audio, which this \
                             output policy does not permit",
                            e.message()
                        )));
                    }
                    Err(dop_err) => {
                        let reason = match native_err {
                            Some(n) => format!("{n}; DoP: {}", dop_err.message()),
                            None => dop_err.message().to_string(),
                        };
                        self.start_dsd_fallback(path, opening).map_err(|e| {
                            bitperfect::OpenError::source(format!(
                                "{reason}; PCM fallback also failed: {e}"
                            ))
                        })?;
                        self.dsd_fallback = true;
                        // Decimated to PCM and played through the ordinary
                        // path: volume, EQ and ReplayGain all apply. 1.4.2 kept
                        // a green diamond here purely because the global
                        // preference was still on.
                        // The source stays the DSD file. Describing it as the
                        // decimated PCM carrier is what made a DSD64 track
                        // report itself as ordinary PCM.
                        let media = dsd::parse_file(path).ok().map(|i| {
                            bitperfect::state::MediaSource::dsd(
                                bitperfect::format::SourceFormat {
                                    kind: bitperfect::format::PcmKind::Integer { valid_bits: 1 },
                                    sample_rate: i.sample_rate,
                                    channels: i.channels as u16,
                                    layout: bitperfect::format::ChannelLayout::UNSPECIFIED,
                                },
                                i.rate_label(),
                            )
                        });
                        self.bp_note_processed(
                            media,
                            bitperfect::state::TransformDescription::DsdDecimated {
                                pcm_rate: self.last_sample_rate,
                            },
                            reason.clone());
                        self.dsd_fallback_note = Some(reason);
                    }
                }
            }
        } else if self.bit_perfect {
            self.start_bp(path, opening)?;
        } else {
            let landed = self
                .play_file_shared_at(path, duration, opening)
                .map_err(|e| shared_open_error(&e))?;
            return Ok(landed);
        }
        // Where the route actually is. `start_bp` takes the container's own
        // answer, which can be seconds from the request on a file with no seek
        // table; the DSD routes are byte-addressable and land exactly.
        let landed = self.bp_base;
        self.started_at = Some(Instant::now());
        self.paused_elapsed = landed;
        self.current_duration = duration;
        Ok(landed)
    }

    /// The ordinary shared-mode `rodio` route, opened at `start`.
    ///
    /// Reached both when bit-perfect is off and when an exact route refused
    /// this particular track. It never changes `bit_perfect`: the preference
    /// belongs to the user, and a track that cannot take the exact route says
    /// nothing about the next one.
    ///
    /// A `rodio` sink plays the moment a source is appended to it, so the
    /// only way to open at a position without emitting the beginning of the
    /// track is to hold the sink paused across the append and the seek and
    /// release it afterwards. Appending and then seeking is what the restart
    /// used to do, and what the listener heard was the first fraction of a
    /// second of the track every time they changed output device mid-song.
    fn play_file_shared_at(
        &mut self,
        path: &PathBuf,
        duration: Option<Duration>,
        opening: Opening,
    ) -> Result<Duration, String> {
        let start = opening.at;
        let handle = self.rodio_handle()?;
        let file = File::open(path).map_err(|e| format!("Open failed: {e}"))?;
        let decoder = Decoder::new(BufReader::new(file))
            .map_err(|e| format!("Decode failed: {e}"))?;
        let sink = Sink::try_new(&handle)
            .map_err(|e| format!("Sink failed: {e}"))?;
        // Silent until it is where it is meant to be, and — if the listener
        // was paused — silent afterwards too. A `rodio` sink plays the moment
        // a source is appended, so this is the only place the decision can be
        // made: after the append it is already too late by a buffer.
        if opening.paused || start > Duration::ZERO {
            sink.pause();
        }
        sink.set_volume(self.rodio_volume());
        self.last_sample_rate = decoder.sample_rate();
        let sink_channels = decoder.channels();
        let tapped = SpectrumSource::new(decoder, self.sample_buf.clone(), self.stereo_buf.clone());
        if let Some(ref eq) = self.eq {
            sink.append(EqSource::new(tapped.convert_samples::<f32>(), eq.clone()));
        } else {
            sink.append(tapped);
        }
        // And now, with the source in place and nothing audible yet, move it.
        let landed = if start > Duration::ZERO {
            match sink.try_seek(start) {
                Ok(()) => {
                    // Released only if it was never asked to stay paused.
                    if !opening.paused {
                        sink.play();
                    }
                    start
                }
                Err(e) => {
                    // Nothing was heard and nothing is kept. The caller's
                    // transaction releases the rest.
                    sink.stop();
                    return Err(format!("this file will not seek: {e}"));
                }
            }
        } else {
            Duration::ZERO
        };
        self.sink = Some(sink);
        // Ordinary shared playback: EQ, ReplayGain and the volume slider
        // are all in the path, and the badge says so. With the preference
        // still on, 1.4.2 rendered this green.
        // What the file is, where that can be established — never what
        // `rodio` happens to hand the mixer.
        //
        // The fallback arm did exactly what the comment above it said not to:
        // any track the exact route had not already prepared was published as
        // `Float32` at the sink's channel count, because that is what the
        // mixer converts everything to. A 24-bit FLAC the DAC had refused was
        // then described to the listener as a float source — a specific claim
        // about the recording, derived from the playback path, and wrong.
        //
        // `prepare` is asked instead, and if it will not read the container
        // the answer is that nothing is known. Unknown is a real answer and
        // the panel has a word for it; an invented format does not.
        let media = self.shared_source_for(path);
        let _ = sink_channels; // the mixer's shape, not the file's
        // Recorded before it is published, on the one route that had no record
        // of what it was playing at all.
        self.note_current(path, media.clone(), landed);
        self.bp_note_processed(
            media,
            bitperfect::state::TransformDescription::Identity,
            if self.bit_perfect {
                "playing through the shared mixer for this track"
            } else {
                "bit-perfect output is off"
            });
        self.started_at = Some(Instant::now());
        self.paused_elapsed = landed;
        self.current_duration = duration;
        Ok(landed)
    }

/// Open the best route this source and this policy allow.
///
/// The ladder, in order, and it stops at the first rung that holds:
///
/// 1. the source's own representation, exactly;
/// 2. for a Float32 source whose samples all land on the Q1.31 lattice, a
///    value-exact integer conversion;
/// 3. for a Float32 source the user has permitted processing on, a
///    deterministic rounded conversion.
///
/// Only a *device* rejection advances the ladder. A configuration error, an
/// unreadable file or a self-contradictory request stops here with its own
/// reason — trying a lossier route in answer to a typo obeys a request
/// nobody made.
///
/// Rung 2 is never entered on a claim: the stream opens as processed and is
/// upgraded only when a complete track scan says every sample was exactly
/// representable. Value-exactness proved from a priming packet would be the
/// same class of error this release exists to remove.
fn open_pcm_route(
    &mut self,
    source: bitperfect::format::SourceFormat,
) -> Result<bitperfect::BpStream, bitperfect::OpenError> {
    let policy = self.bp_state.policy;
    let device = self.bp_device.clone();
    let (sample_buf, stereo_buf) = (self.sample_buf.clone(), self.stereo_buf.clone());
    // The ladder rule lives in `walk_ladder`, which knows nothing about
    // devices — so the routing that decides what an integer-only DAC is
    // offered for a Float32 file can be exercised without one.
    let stream = bitperfect::walk_ladder(source.kind, policy, |plan| {
        let r = bitperfect::BpStream::open(
            device.as_deref(),
            source,
            plan,
            false,
            sample_buf.clone(),
            stereo_buf.clone(),
        );
        if let Err(ref e) = r
            && e.is_device_limitation()
        {
            crate::mlog!(
                "bp      {plan:?} refused by the device: {e}; trying the next                               route this policy allows"
            );
        }
        r
    })?;
    stream.set_policy(policy);
    Ok(stream)
}
    /// Decode `path` from `start` and play it on the bit-perfect stream,
    /// (re)opening the cpal stream if the device or stream format changed.
    fn start_bp(&mut self, path: &Path, opening: Opening) -> Result<(), bitperfect::OpenError> {
        let start = opening.at;
        self.native_close(); // a native-DSD output may hold the device exclusively
        // Timed because this decodes: `prepare` opens the file and seeks it,
        // and a FLAC without a seek table is scanned from the beginning. On the
        // UI thread that is a stall the length of the scan.
        let t_prep = std::time::Instant::now();
        // Typed all the way up. Flattening this to a string is what let an
        // unreadable file and a refused seek arrive at the caller looking
        // exactly like a DAC that could not take the format.
        let prep = bitperfect::prepare(path, start).map_err(bitperfect::OpenError::from)?;
        // Recorded before the device is touched, so a route that the endpoint
        // then refuses still leaves the engine knowing what the file actually
        // is.
        self.last_prepared = Some((path.to_path_buf(), prep.source));
        let prep_ms = t_prep.elapsed().as_secs_f64() * 1e3;
        if prep_ms > 100.0 {
            mlog!("bp      SLOW prepare: {prep_ms:.0} ms to seek to {:.2}s", start.as_secs_f64());
        }
        mlog!(
            "bp      source {} (declared {:?} bits) — {}",
            prep.source.describe(), prep.bits_per_sample,
            path.file_name().map(|n| n.to_string_lossy().into_owned()).unwrap_or_default(),
        );
        self.last_sample_rate = prep.sample_rate;

        // The whole negotiated format has to still prove the claim for this
        // source, not merely share a rate and a channel count with it. A
        // 16-bit stream carried a following 24-bit track and truncated it,
        // silently, with the diamond lit; a DoP stream could never carry PCM
        // at all. `StreamKey::can_carry` is the single predicate for both, and
        // the gapless queue below asks the same question.
        // The endpoint the *next* open would attach to, resolved independently
        // of whatever is currently running. Deriving it from the open stream
        // made a changed Windows default endpoint invisible: the key was built
        // from the stream and then compared back to it, so the comparison
        // could only ever succeed.
        let target = bitperfect::resolve_endpoint(self.bp_device.as_deref());

        // The whole negotiated format has to still prove the claim for this
        // source, not merely share a rate and a channel count with it. A
        // 16-bit stream carried a following 24-bit track and truncated it,
        // silently, with the diamond lit; a DoP stream could never carry PCM
        // at all. `StreamKey::can_carry` is the single predicate, and the
        // gapless queue asks it too.
        let reuse = match (&self.bp, &target) {
            (Some(s), Some(t)) =>
                // A backend whose thread has ended can carry nothing, whatever
                // the session state believes.
                !s.backend_dead()
                    && self.bp_state.backend_alive
                    && s.requested_device == self.bp_device
                    && s.can_carry(&s.key_for(t, &prep.source)),
            // An endpoint that cannot be resolved is not a licence to keep
            // using the old one.
            _ => false,
        };
        if !reuse {
            // The claim goes before the stream does: for the whole negotiation
            // window the honest answer is "verifying", not the answer from the
            // track that just ended.
            self.bp_state.begin_open();
            self.release_bp(); // release the old stream/device first
            self.bp = Some(self.open_pcm_route(prep.source)?);
        }
        let source = bitperfect::state::MediaSource::pcm(prep.source);
        // The position the container actually reached, not the one that was
        // asked for. Moving the time base and the spectrum to the requested
        // position while the decoder sat somewhere else made the display and
        // the audio disagree with nothing anywhere reporting it.
        let landed = if start > Duration::ZERO { prep.seeked_to } else { Duration::ZERO };
        let needs_proof = self.bp.as_ref().map(|s| s.plan) == Some(bitperfect::PayloadPlan::Q31);
        let bp = self.bp.as_ref().unwrap();
        // The pause flag is taken before `start`, not after it. `start` opens
        // the generation the render thread begins pulling from, so a route
        // paused a statement later has already put a buffer on the wire.
        bring_up(
            &mut StreamBringUp {
                route: bp,
                start: Some(|| bp.start(prep).map_err(bitperfect::OpenError::from)),
            },
            opening.paused,
        )?;
        // `start` opens a generation, on a fresh stream and on a reused one
        // alike; the UI is folded up to it from here.
        self.bp_audio_gen = bp.audio_generation();
        // And only now is this what is playing.
        //
        // It was recorded before `start`, so a route that failed to start left
        // the engine naming a track it had never played — which is the record
        // `resume` reopens after a failed seek, and the one every surface asks
        // "what is this?" of.
        self.note_current(path, Some(source.clone()), landed);
        self.bp_note_opened(source);
        // Playing already, as processed. The scan decides whether it earns the
        // stronger label; it never gates the audio.
        if needs_proof {
            self.begin_q31_scan(path);
        }
        self.bp_base = landed;
        self.bp_played_at_track_start = Duration::ZERO; // fresh session: frames_played reset to 0
        // The shared stream is *preempted*, not released.
        //
        // WASAPI exclusive mode takes the endpoint from whatever had it, and
        // that is the intended behaviour of this route: the listener asked for
        // the DAC's own words, and shared mixing is what they asked to be rid
        // of. This flag records that our own `rodio` stream may have been
        // killed as a result and must be rebuilt before it is used again — it
        // is not a claim that anything here released it. `rodio_handle`
        // rebuilds it on the next shared track.
        self.rodio_stream_dirty = true;
        Ok(())
    }

    /// Everything about a DSD file that must be settled before a device is
    /// touched.
    ///
    /// A container that declares more audio than the file holds runs off its
    /// own end mid-track. The feeder catches it — but by then an ASIO engine
    /// is running, the DAC is locked to a DSD stream, and the failure arrives
    /// as a fault on a route that was already claiming to be exact. Refusing
    /// here means the fallback happens with the device untouched.
    ///
    /// A function rather than an inline check because "no device was opened"
    /// is the property worth testing, and it cannot be tested against a
    /// hardware call.
    fn native_preflight(info: &dsd::DsdInfo) -> Result<(), String> {
        if info.is_truncated() {
            return Err(format!(
                "DSD: this file is truncated — it declares {} bytes of audio and contains {}",
                info.declared_data_len, info.data_len
            ));
        }
        Ok(())
    }

    /// Run the pre-open gate, and open only if it passes.
    ///
    /// Both native routes go through this, so the ordering is a property of
    /// one function rather than a habit repeated at two call sites — and a
    /// test can pass an opener that counts.
    fn open_native_checked<T>(
        info: &dsd::DsdInfo,
        open: impl FnOnce() -> Result<T, String>,
    ) -> Result<T, String> {
        Self::native_preflight(info)?;
        open()
    }

    /// Decode `path` (a `.dsf`/`.dff` file) and play it as DoP on the
    /// bit-perfect stream, (re)opening it at the DoP carrier rate if the
    /// device, rate, or channel count changed. `start` seeks within the
    /// track (each DoP PCM frame is 16 DSD samples, so the seek target is
    /// converted to a PCM-frame offset and applied before the device format or
    /// any engine state is touched — a failed seek changes nothing).
    fn start_dop(&mut self, path: &Path, opening: Opening) -> Result<(), bitperfect::OpenError> {
        let start = opening.at;
        // The source is prepared in full — opened, parsed, and seeked — before
        // anything with side effects is touched. Doing the seek after the
        // device was (re)opened meant a failure could leave an idle exclusive
        // stream holding the DAC and `dsd_label`/`last_sample_rate` describing
        // an attempt that never started.
        //
        // A file that will not open, parse or seek is not a device problem: the
        // decimated-PCM route reads the same bytes and fails the same way.
        // Scope here is synchronous only — open, parse, reader construction and
        // the seek. Decode failures arrive later, on the decode thread.
        let mut stream = dsd::dop::open_dop_stream(path)
            .map_err(|e| bitperfect::OpenError::source(format!("DSD: {e}")))?;
        // A container that declares more audio than the file holds will run
        // off its own end mid-track. That is a source problem, and it is
        // refused *before* a device is opened rather than discovered halfway
        // through — at which point it is a fault on a route that was already
        // claiming to be exact.
        if stream.info().is_truncated() {
            let i = stream.info();
            return Err(bitperfect::OpenError::source(format!(
                "DSD: this file is truncated — it declares {} bytes of audio and contains {}",
                i.declared_data_len, i.data_len
            )));
        }
        let carrier = stream.carrier_rate();
        let channels = stream.info().channels as u16;
        let dsd_rate = stream.info().sample_rate;
        let dsd_layout = stream.info().layout;
        let dsd_truncated = stream.info().is_truncated();
        let label = stream.info().rate_label();
        if start > Duration::ZERO {
            let pcm_frame = (start.as_secs_f64() * carrier as f64) as u64;
            stream.seek_to_pcm_frame(pcm_frame)
                .map_err(|e| bitperfect::OpenError::seek(format!("DSD seek: {e}")))?;
        }

        // Only now does anything change.
        // What is playing is recorded *after* the route is open, and with the
        // source parsed from the file. Recording it here published two
        // untruths at once: a current track with an unknown source on a route
        // that had succeeded, and — worse — a current track at all on a route
        // that was about to fail, so `resume` after a failed seek reopened a
        // file this session never managed to play.
        self.native_close(); // a native-DSD output may hold the device exclusively
        self.dsd_odd_tail = Some(stream.info().total_frames() % 2 == 1);
        self.dsd_label = Some(label);
        self.last_sample_rate = carrier;

        // The *carrier* the device negotiates against: a DoP frame is a 24-bit
        // integer word whatever the DSD rate. This is deliberately not the
        // source — the file is 1-bit DSD, and describing it to the panel as a
        // 24-bit integer source is how 1.4.2 reported a DSD64 file as PCM.
        // The layout, though, is the *source's*. A speaker mask says which
        // speaker each channel reaches, and that is a property of the
        // recording, not of the word width it travels in — so the DSD file's
        // parsed mask is what the device must be negotiated for and validated
        // against. Sending `UNSPECIFIED` meant a multichannel DSD was
        // negotiated against whatever default the channel count implies, and
        // the default for three channels is FL/FR/FC: a FL/FR/LFE recording
        // put its LFE content in the centre speaker, validated, with the
        // claim intact.
        //
        // `dsd_layout` is `None` when the container did not say or said
        // something self-contradictory, and `UNSPECIFIED` is then the honest
        // request — above stereo, `source_blocks_exact_claim` already
        // withholds the claim for an unknown layout.
        let dop_carrier_format = bitperfect::format::SourceFormat {
            kind: bitperfect::format::PcmKind::Integer { valid_bits: 24 },
            sample_rate: carrier,
            channels,
            layout: bitperfect::format::ChannelLayout(dsd_layout.unwrap_or(0)),
        };
        // The real source, kept separate and shown to the user.
        let media = bitperfect::state::MediaSource::dsd(
            bitperfect::format::SourceFormat {
                kind: bitperfect::format::PcmKind::Integer { valid_bits: 1 },
                sample_rate: dsd_rate,
                channels,
                layout: bitperfect::format::ChannelLayout(dsd_layout.unwrap_or(0)),
            },
            self.dsd_label.clone().unwrap_or_else(|| "DSD".into()),
        );
        let media = if dsd_truncated { media.truncated() } else { media };

        let target = bitperfect::resolve_endpoint(self.bp_device.as_deref());
        let reuse = match (&self.bp, &target) {
            (Some(s), Some(t)) =>
                s.dop
                    && !s.backend_dead()
                    && self.bp_state.backend_alive
                    && s.requested_device == self.bp_device
                    && s.can_carry(&s.key_for(t, &dop_carrier_format)),
            _ => false,
        };
        if !reuse {
            self.bp_state.begin_open();
            self.release_bp(); // release the old stream/device first
            self.bp = Some(bitperfect::BpStream::open(
                self.bp_device.as_deref(),
                dop_carrier_format,
                bitperfect::PayloadPlan::Identity,
                true, // dop
                self.sample_buf.clone(),
                self.stereo_buf.clone(),
            )?);
        }
        let bp = self.bp.as_ref().unwrap();
        // The pause flag is taken before `start_dop`, not after it. This route
        // ignored the intent entirely and resumed unconditionally: a paused
        // restart onto a DoP DAC opened the generation playing, and the pause
        // that `apply_restart` applied afterwards arrived a statement later —
        // milliseconds of the DAC's own sound after a silent gap.
        bring_up(
            &mut StreamBringUp {
                route: bp,
                start: Some(|| bp.start_dop(stream).map_err(bitperfect::OpenError::from)),
            },
            opening.paused,
        )?;
        // `start_dop` opens a generation; the UI is folded up to it.
        self.bp_audio_gen = bp.audio_generation();
        // Open, playing, and the source is the DSD file rather than the 24-bit
        // carrier it rides on.
        self.note_current(path, Some(media.clone()), start);
        self.bp_note_opened(media);
        self.bp_base = start;
        self.bp_played_at_track_start = Duration::ZERO;
        // The shared stream is *preempted*, not released.
        //
        // WASAPI exclusive mode takes the endpoint from whatever had it, and
        // that is the intended behaviour of this route: the listener asked for
        // the DAC's own words, and shared mixing is what they asked to be rid
        // of. This flag records that our own `rodio` stream may have been
        // killed as a result and must be rebuilt before it is used again — it
        // is not a claim that anything here released it. `rodio_handle`
        // rebuilds it on the next shared track.
        self.rodio_stream_dirty = true;
        Ok(())
    }

    /// Queue `path` for gapless continuation on the open bit-perfect stream.
    /// Returns false (no gapless) if there is no stream, the file can't be
    /// prepared, or its rate/channels differ from the stream (a format change
    /// forces a device re-open, so the track boundary can't be gapless).
    /// Play a DSD file as raw native DSD through the selected ASIO driver —
    /// no DoP carrier, so DSD512 works wherever the driver does. Reuses the
    /// open ASIO stream across same-rate tracks; a rate/driver change
    /// re-negotiates from scratch.
    #[cfg(all(windows, feature = "asio-dsd"))]
    fn start_asio_native(&mut self, path: &Path, opening: Opening) -> Result<(), String> {
        let start = opening.at;
        use bitperfect::asio_dsd::AsioDsdStream;
        use bitperfect::native_dsd::feed_loop;
        use std::sync::atomic::AtomicBool;
        use std::sync::Arc;

        let driver = self.asio_driver.clone().ok_or("no ASIO driver selected")?;
        // Release any WASAPI-exclusive stream first — it holds the DAC, and
        // the DAC's own ASIO driver can't open hardware someone else owns.
        // (The mirror of asio_close() in start_bp/start_dop.)
        self.release_bp();
        self.rodio_stream_dirty = true; // the ASIO driver may invalidate shared streams too
        let info = dsd::parse_file(path).map_err(|e| format!("DSD: {e}"))?;
        // Before the driver is touched. A container that declares more audio
        // than the file holds runs off its own end mid-track; the feeder
        // catches it, but by then an ASIO engine is open, the DAC is locked to
        // a DSD stream, and the failure arrives as a fault on a route that was
        // already claiming to be exact.
        Self::native_preflight(&info)?;
        let channels = info.channels as u16;
        self.dsd_label = Some(info.rate_label());

        // Liveness belongs here and only here: this is the one decision a
        // requested reset is an answer to.
        let live = self.native_handle_live();
        let reuse = live
            && self.asio.as_ref().is_some_and(|a| {
                a.dsd_rate == info.sample_rate
                    && a.channels == channels
                    && a.driver_name == driver
            });
        if !reuse {
            self.asio = None; // release the driver before re-opening
            // The gate and the open, in that order, as one call — see
            // `open_native_checked`.
            self.asio = Some(Self::open_native_checked(&info, || {
                AsioDsdStream::open(&driver, info.sample_rate, channels)
                    .map_err(|e| format!("native DSD: {e}"))
            })?);
        }
        // Everything from here to `start_session` can fail with the device
        // already open and exclusive.
        //
        // `?` on any of it used to return with the handle still held, while
        // the caller fell back to DoP or to the shared mixer — both of which
        // then could not open the device that this process was itself
        // holding. The failure was reported as the device being busy, which
        // was true, and unhelpful.
        macro_rules! release_on_err {
            ($e:expr) => {
                match $e {
                    Ok(v) => v,
                    Err(e) => {
                        self.asio = None;
                        return Err(e);
                    }
                }
            };
        }

        let mut reader = release_on_err!(dsd::open_reader(path));
        if start > Duration::ZERO {
            let frame = (start.as_secs_f64() * info.sample_rate as f64 / 8.0) as u64;
            release_on_err!(
                reader
                    .seek_to_frame(frame)
                    .map_err(|e| format!("DSD seek: {e}"))
            );
        }
        let stream = self.asio.as_ref().unwrap();

        // ~1 s of interleaved DSD byte-frames, whole frames only.
        let ring_frames = (info.sample_rate as usize / 8).max(32_768);
        let (prod, cons) =
            bitperfect::frame_ring::channel_ring::<u8>(channels as usize, ring_frames);
        let done = Arc::new(AtomicBool::new(false));
        let decode_eof = Arc::new(AtomicBool::new(false));
        let session_stop = Arc::new(AtomicBool::new(false));
        {
            let done = Arc::clone(&done);
            let decode_eof = Arc::clone(&decode_eof);
            let stop = Arc::clone(&session_stop);
            let lsb = stream.lsb_first;
            let status = std::sync::Arc::clone(&self.feed_status);
            let generation = status.begin_generation();
            // The session is installed only once its feeder exists; a thread
            // that could not start used to leave the session priming forever.
            let spawned = std::thread::Builder::new()
                .name("asio-dsd-decode".into())
                .spawn(move || {
                    // The outcome is not discarded. A feeder that stops for
                    // any reason other than reaching the end of the track
                    // faults its own generation on the way out, including on
                    // unwind, and never sets `decode_eof` — so the drain
                    // reports a failure and the playlist does not advance
                    // into the same file again.
                    if let Err(e) = feed_loop(
                        reader, lsb, prod, done, decode_eof, stop, status, generation,
                    ) {
                        crate::mlog!("[native-dsd] feed ended early: {e:?}");
                    }
                })
                .map_err(|e| format!("DSD feed thread could not start: {e}"));
            // The device is open and exclusive by this point; a feeder that
            // could not start must not leave it that way.
            if let Err(e) = spawned {
                self.asio = None;
                return Err(e);
            }
        }
        let stream = self.asio.as_ref().unwrap();
        // Before `start_session`, for the reason the DoP route has: the
        // session is what the callback pulls from, and this route resumed
        // unconditionally. `start_session` cannot fail, so the mapping below
        // is a type conversion and not a path.
        bring_up(
            &mut StreamBringUp {
                route: stream,
                start: Some(|| {
                    stream.start_session(cons, done, decode_eof, session_stop);
                    Ok(())
                }),
            },
            opening.paused,
        )
        .map_err(|e| e.to_string())?;
        // The session opened a generation; the UI reads evidence by it.
        let started_at = stream.audio_generation();
        self.native_audio_gen = started_at;

        self.last_sample_rate = info.sample_rate / 8; // byte rate; display uses dsd_label
        self.bp_base = start;
        self.bp_played_at_track_start = Duration::ZERO;
        // Payload-exactness on this route rests on the driver actually being
        // in DSD mode. `kAsioSetIoFormat` succeeding says the call succeeded,
        // not that the mode stuck — a driver that implements the getter can be
        // asked, and one that does not cannot. Playback is the same either
        // way; the claim is not. An unverifiable readback is amber
        // Unverified, never a green diamond, and a *mismatched* readback never
        // gets this far because `host_setup` refuses to open at all.
        //
        // The rate is the second half of the same question, and it was not
        // being asked. The driver is offered the rate in the form the DSD
        // specification defines and, if that is refused, in two divided forms
        // that some drivers want and that no specification describes. A driver
        // that accepts `rate/8` has accepted a number; what it understood by
        // it is written down nowhere. Playback on that route is the same, and
        // the claim is not.
        let (verified_mode, spec_rate) = self
            .asio
            .as_ref()
            .map(|a| (a.io_format_verified, a.rate_spec_form))
            .unwrap_or((false, false));
        let unverified = match (verified_mode, spec_rate) {
            (true, true) => None,
            (false, true) => Some(
                "this ASIO driver does not implement kAsioGetIoFormat, so the DSD output \
                 mode was set but could not be read back and confirmed"
                    .to_string(),
            ),
            (true, false) => Some(
                "this ASIO driver would not take the DSD rate in the form the specification \
                 defines and accepted an undocumented divided form instead, so what it \
                 understood by the rate cannot be confirmed"
                    .to_string(),
            ),
            (false, false) => Some(
                "this ASIO driver neither implements kAsioGetIoFormat nor takes the DSD rate \
                 in the form the specification defines, so neither the output mode nor the \
                 rate could be confirmed"
                    .to_string(),
            ),
        };
        self.publish_native_dsd(
            path,
            start,
            bitperfect::Transport::AsioNativeDsd {
                driver: bitperfect::state::EndpointIdentity::from_name(driver.clone()),
            },
            &info,
            channels,
            unverified.as_deref(),
        );
        Ok(())
    }

    /// Play a DSD file as raw native DSD through the selected ALSA hw:
    /// device — the Linux mirror of `start_asio_native`, over the kernel's
    /// DSD_U32_BE/U16/U8 formats. Reuses the open stream across same-rate
    /// tracks; a rate/device change re-negotiates from scratch.
    #[cfg(all(target_os = "linux", feature = "alsa-dsd"))]
    fn start_alsa_native(&mut self, path: &Path, opening: Opening) -> Result<(), String> {
        let start = opening.at;
        use bitperfect::alsa_dsd::AlsaDsdStream;
        use bitperfect::native_dsd::feed_loop;
        use std::sync::atomic::AtomicBool;
        use std::sync::Arc;

        let device = self.alsa_dsd_device.clone().ok_or("no ALSA device selected")?;
        // Release anything of ours that may hold the DAC: the cpal
        // bit-perfect stream, and the rodio (shared) stream — on the same
        // card even a shared handle can block direct hw: access. (Other
        // apps' handles we can't drop; the open error says so honestly.)
        self.release_bp();
        self.rodio_out = None;
        self.rodio_stream_dirty = false; // already gone; reopen lazily later
        let info = dsd::parse_file(path).map_err(|e| format!("DSD: {e}"))?;
        // Before the device is touched — see `start_asio_native`.
        Self::native_preflight(&info)?;
        let channels = info.channels as u16;
        self.dsd_label = Some(info.rate_label());

        // See the ASIO counterpart: reuse is the liveness question.
        let live = self.native_handle_live();
        let reuse = live
            && self.alsa.as_ref().is_some_and(|a| {
                a.dsd_rate == info.sample_rate
                    && a.channels == channels
                    && a.device_name == device
            });
        if !reuse {
            self.alsa = None; // release the device before re-opening
            // The gate and the open, in that order, as one call — see
            // `open_native_checked`.
            self.alsa = Some(Self::open_native_checked(&info, || {
                AlsaDsdStream::open(&device, info.sample_rate, channels)
                    .map_err(|e| format!("native DSD: {e}"))
            })?);
        }
        // Everything from here to `start_session` can fail with the device
        // already open and exclusive.
        //
        // `?` on any of it used to return with the handle still held, while
        // the caller fell back to DoP or to the shared mixer — both of which
        // then could not open the device that this process was itself
        // holding. The failure was reported as the device being busy, which
        // was true, and unhelpful.
        macro_rules! release_on_err {
            ($e:expr) => {
                match $e {
                    Ok(v) => v,
                    Err(e) => {
                        self.alsa = None;
                        return Err(e);
                    }
                }
            };
        }

        let mut reader = release_on_err!(dsd::open_reader(path));
        if start > Duration::ZERO {
            let frame = (start.as_secs_f64() * info.sample_rate as f64 / 8.0) as u64;
            release_on_err!(
                reader
                    .seek_to_frame(frame)
                    .map_err(|e| format!("DSD seek: {e}"))
            );
        }
        let stream = self.alsa.as_ref().unwrap();

        // ~1 s of interleaved DSD byte-frames, whole frames only.
        let ring_frames = (info.sample_rate as usize / 8).max(32_768);
        let (prod, cons) =
            bitperfect::frame_ring::channel_ring::<u8>(channels as usize, ring_frames);
        let done = Arc::new(AtomicBool::new(false));
        let decode_eof = Arc::new(AtomicBool::new(false));
        let session_stop = Arc::new(AtomicBool::new(false));
        {
            let done = Arc::clone(&done);
            let decode_eof = Arc::clone(&decode_eof);
            let stop = Arc::clone(&session_stop);
            let lsb = stream.lsb_first;
            let status = std::sync::Arc::clone(&self.feed_status);
            let generation = status.begin_generation();
            // The session is installed only once its feeder exists; a thread
            // that could not start used to leave the session priming forever.
            let spawned = std::thread::Builder::new()
                .name("alsa-dsd-decode".into())
                .spawn(move || {
                    // The outcome is not discarded. A feeder that stops for
                    // any reason other than reaching the end of the track
                    // faults its own generation on the way out, including on
                    // unwind, and never sets `decode_eof` — so the drain
                    // reports a failure and the playlist does not advance
                    // into the same file again.
                    if let Err(e) = feed_loop(
                        reader, lsb, prod, done, decode_eof, stop, status, generation,
                    ) {
                        crate::mlog!("[native-dsd] feed ended early: {e:?}");
                    }
                })
                .map_err(|e| format!("DSD feed thread could not start: {e}"));
            // The device is open and exclusive by this point; a feeder that
            // could not start must not leave it that way.
            if let Err(e) = spawned {
                self.alsa = None;
                return Err(e);
            }
        }
        let stream = self.alsa.as_ref().unwrap();
        // Before `start_session`, for the reason the DoP route has: the
        // session is what the writer pulls from, and this route resumed
        // unconditionally. `start_session` cannot fail, so the mapping below
        // is a type conversion and not a path.
        bring_up(
            &mut StreamBringUp {
                route: stream,
                start: Some(|| {
                    stream.start_session(cons, done, decode_eof, session_stop);
                    Ok(())
                }),
            },
            opening.paused,
        )
        .map_err(|e| e.to_string())?;
        // The session opened a generation; the UI reads evidence by it.
        let started_at = stream.audio_generation();
        self.native_audio_gen = started_at;

        self.last_sample_rate = info.sample_rate / 8; // byte rate; display uses dsd_label
        self.bp_base = start;
        self.bp_played_at_track_start = Duration::ZERO;
        self.publish_native_dsd(
            path,
            start,
            bitperfect::Transport::AlsaNativeDsd {
                endpoint: bitperfect::state::EndpointIdentity::from_name(device.clone()),
            },
            &info,
            channels,
            Some("native ALSA DSD end-to-end exactness is not verified in this release"),
        );
        Ok(())
    }

    /// Decimate a DSD file to PCM and play it through the rodio path — the
    /// fallback when the device can't take the DoP carrier rate. This is a
    /// normal-mode session: volume, EQ and ReplayGain apply, the spectrum tap
    /// feeds the realtime display, and seeks are instant (`try_seek` on the
    /// source, DSD being byte-addressable).
    /// The decimated-PCM route, used when neither native DSD nor DoP is
    /// available. Records the track like every other route.
    fn start_dsd_fallback(&mut self, path: &Path, opening: Opening) -> Result<(), String> {
        let start = opening.at;
        let out = self.start_dsd_fallback_inner(path, opening);
        if out.is_ok() {
            // The source is DSD whatever carrier it ends up on; the transform
            // is published by the caller, which knows which route this is.
            let source = dsd::parse_file(path)
                .ok()
                .map(|i| {
                    bitperfect::state::MediaSource::dsd(
                        bitperfect::format::SourceFormat {
                            kind: bitperfect::format::PcmKind::Integer { valid_bits: 1 },
                            sample_rate: i.sample_rate,
                            channels: i.channels as u16,
                            layout: bitperfect::format::ChannelLayout(
                                i.layout.unwrap_or(0),
                            ),
                        },
                        i.rate_label(),
                    )
                });
            self.note_current(path, source, start);
        }
        out
    }

    /// The decimated PCM a DSD file plays as on the fallback route, at the
    /// *playback* fallback rate — the one source for a start, a restart, a
    /// seek and a gapless append, so none of them can follow another setting.
    fn open_fallback_pcm(&self, path: &Path) -> Result<dsd::decimate::DsdFilePcmSource, String> {
        dsd::decimate::open_pcm_source(path, self.dsd_fallback_target)
            .map_err(|e| format!("DSD decimate: {e}"))
    }

    /// A sink on the shared mixer (the test seam's idle sink under test).
    fn new_shared_sink(&mut self) -> Result<Sink, String> {
        #[cfg(test)]
        if self.idle_output {
            return Ok(Sink::new_idle().0);
        }
        let handle = self.rodio_handle()?;
        Sink::try_new(&handle).map_err(|e| format!("Sink failed: {e}"))
    }

    /// What the route that is open *now* feeds the live tap for `path`: its
    /// PCM rate, or `None` when nothing is open (a stopped or abandoned
    /// session, a seek still decoding off-thread) or the route attaches no
    /// tap (DoP, native DSD). Read after a route has been installed, so it
    /// describes what became active, not what was asked for.
    fn live_tap_now(&self, path: &Path) -> Option<u32> {
        if self.sink.is_none() && !self.on_bp_stream() {
            return None;
        }
        self.spectrum_live_rate(path, self.last_sample_rate)
    }

    /// The rate of the PCM the spectrum's live tap carries while `path` plays
    /// on the current route, or `None` where the route feeds no live PCM.
    ///
    /// `reported` is the rate the route reports for a PCM file. A DSD file on
    /// a device stream (DoP, native) feeds no PCM to the tap; on the shared
    /// mixer it plays as the fallback decimation, at the rate that produces.
    fn spectrum_live_rate(&self, path: &Path, reported: u32) -> Option<u32> {
        if !dsd::is_dsd_path(path) {
            return Some(reported);
        }
        if self.on_bp_stream() {
            return None;
        }
        dsd::parse_file(path)
            .ok()
            .map(|i| dsd::decimate::analysis_rate_for(i.sample_rate, self.dsd_fallback_target))
    }

    fn start_dsd_fallback_inner(
        &mut self,
        path: &Path,
        opening: Opening,
    ) -> Result<(), String> {
        let start = opening.at;
        let mut src = self.open_fallback_pcm(path)?;
        if start > Duration::ZERO {
            src.seek_to_time(start).map_err(|e| format!("DSD seek: {e}"))?;
        }
        self.dsd_label = dsd::parse_file(path).ok().map(|i| i.rate_label());
        self.last_sample_rate = src.out_rate();
        let sink = self.new_shared_sink()?;
        // Before the append, because that is when a `rodio` sink starts.
        if opening.paused {
            sink.pause();
        }
        sink.set_volume(self.rodio_volume());
        let tapped = SpectrumSource::new(
            DsdRodioSource(src), self.sample_buf.clone(), self.stereo_buf.clone());
        if let Some(ref eq) = self.eq {
            sink.append(EqSource::new(tapped.convert_samples::<f32>(), eq.clone()));
        } else {
            sink.append(tapped);
        }
        self.sink = Some(sink);
        Ok(())
    }

    /// PCM gapless only — a DSD `path` is refused here (it can't ride a PCM
    /// stream); `bp_queue_next_dop` handles DSD→DSD gapless instead.
    fn bp_queue_next(&self, path: &Path) -> bool {
        if self.dsd_mode { return false; }
        let Some(bp) = self.bp.as_ref() else { return false };
        let Ok(prep) = bitperfect::prepare(path, Duration::ZERO) else { return false };
        // Exactly the predicate the reopen path uses. Before 1.4.3 this one was
        // looser — rate and channel count only — so the depth change that
        // forced a reopen mid-playlist was allowed to slide through the
        // gapless boundary instead, which is the worst place for it.
        if bp.backend_dead() { return false; }
        let Some(target) = bitperfect::resolve_endpoint(self.bp_device.as_deref())
            else { return false };
        if !bp.can_carry(&bp.key_for(&target, &prep.source)) { return false; }
        bp.queue_next(prep);
        true
    }

    /// Queue the next DSD track for gapless DoP continuation on the open
    /// stream. Returns false (no gapless — advance re-opens the device) unless
    /// there's an open DoP stream and the next file's carrier rate and channel
    /// count match it exactly.
    fn bp_queue_next_dop(&self, path: &Path) -> bool {
        if !self.dsd_mode { return false; }
        let Some(bp) = self.bp.as_ref() else { return false };
        if !bp.dop { return false; }
        let Ok(stream) = dsd::dop::open_dop_stream(path) else { return false };
        // A file that declares more audio than it contains runs off its own
        // end. Mid-playlist that is indistinguishable from a track finishing,
        // which is the one place it must not be allowed to look like — so it
        // is refused here, and the boundary refuses it again against the
        // stream that is actually open.
        if stream.info().is_truncated() { return false; }
        // A DoP carrier frame holds *two* DSD bytes per channel. A file whose
        // frame count is odd therefore ends on a half-full carrier frame,
        // which `read_payload` completes with DSD silence — a real, if
        // inaudible, insertion.
        //
        // Carrying that half-frame across the boundary would mean holding the
        // outgoing file's last byte until the incoming file's first arrived,
        // and pairing bytes from two different recordings inside one carrier
        // frame. This build does not do that: the boundary is declared
        // non-gapless instead, the stream ends normally, and the next track
        // opens its own session. The cost is one track boundary's worth of
        // gap on odd-length DSD files; the alternative is a carrier frame that
        // belongs to neither track.
        if self.dsd_odd_tail == Some(true) {
            crate::mlog!(
                "bp      DoP: the current track ends on a half-full carrier frame, so \
                 this boundary is not gapless"
            );
            return false;
        }
        let next = bitperfect::format::SourceFormat {
            kind: bitperfect::format::PcmKind::Integer { valid_bits: 24 },
            sample_rate: stream.carrier_rate(),
            channels: stream.info().channels as u16,
            // The next track's own speaker mask, so a layout change across a
            // gapless boundary forces a reopen instead of silently reassigning
            // every channel.
            layout: bitperfect::format::ChannelLayout(stream.info().layout.unwrap_or(0)),
        };
        let Some(target) = bitperfect::resolve_endpoint(self.bp_device.as_deref())
            else { return false };
        if !bp.can_carry(&bp.key_for(&target, &next)) { return false; }
        bp.queue_next_dop(stream);
        true
    }

    /// Drop a queued gapless track that hasn't started yet.
    fn bp_clear_next(&self) {
        if let Some(bp) = self.bp.as_ref() { bp.clear_next(); }
    }

    /// If the bit-perfect device has crossed a gapless track boundary, roll
    /// the whole session over to the new track and report it (true). Call in
    /// a loop.
    ///
    /// Everything the boundary carries is applied in one step: position base,
    /// source, transform, policy and generation. The boundary used to carry a
    /// frame count and nothing else, so the *title* changed at the hand-off
    /// while the source, the transform and the fidelity went on describing the
    /// track that had just ended — a 24-bit track following a 16-bit one, or a
    /// processed rung following an exact one, kept the previous track's badge
    /// for its whole life.
    fn bp_poll_boundary(&mut self) -> bool {
        let Some((at, boundary)) = self.bp.as_ref().and_then(|bp| bp.take_reached_boundary())
        else {
            return false;
        };
        self.bp_played_at_track_start = at;
        self.bp_base = Duration::ZERO; // the new track starts from its beginning
        // Folded up to here: evidence stamped with this generation is now
        // about the track `bp_state` is about to describe.
        self.bp_audio_gen = boundary.generation;
        // The next hand-off is judged on *this* track's frame count. Recording
        // it only when a DoP session opened meant that after one gapless
        // rollover it still described whichever file had opened the stream.
        if self.dsd_mode {
            self.dsd_odd_tail = Some(boundary.odd_tail);
        }

        // The transport does not change at a gapless boundary — it is the
        // same open stream — so it is read, not rebuilt.
        let transport = self.bp_state.transport.clone();
        let fidelity = Self::fidelity_for(&boundary.source, boundary.plan, &transport);
        let mut processing = bitperfect::state::ProcessingState::transparent_locked();
        processing.transform = boundary.transform.clone();
        self.bp_state.set_source(boundary.source.clone());
        // The track being played has changed, so the answer to "what is
        // playing" has to change with it — this is the audible boundary.
        if !boundary.path.as_os_str().is_empty() {
            let p = boundary.path.clone();
            self.note_current(&p, Some(boundary.source.clone()), Duration::ZERO);
        }
        // One operation. A new generation begins with it, so a scan or a fault
        // still in flight from the track that just ended cannot land on this
        // one.
        self.bp_state.roll_over(fidelity, processing);

        // The successor is a different file, so its value-exactness is a
        // different question. The outgoing track's scan is retired and, if
        // this route is the conversion, a new one begins under the new
        // track's own identity — without which the incoming track simply
        // inherited whatever the outgoing one had proved.
        self.retire_q31_scan();
        if boundary.plan == bitperfect::PayloadPlan::Q31 && !boundary.path.as_os_str().is_empty() {
            let path = boundary.path.clone();
            self.begin_q31_scan(&path);
        }
        crate::mlog!(
            "bp      gapless boundary at {:.3}s: {} under {:?} (decode generation {}, \
             policy {:?} as opened)",
            at.as_secs_f64(),
            boundary.source.describe(),
            boundary.plan,
            boundary.generation,
            boundary.policy,
        );
        true
    }

    /// One-line description of the active bit-perfect stream for the UI.
    fn bp_describe(&self) -> Option<String> {
        #[cfg(all(windows, feature = "asio-dsd"))]
        if self.dsd_native && let Some(a) = &self.asio {
            let label = self.dsd_label.as_deref().unwrap_or("DSD");
            let mut line = format!(
                "{label} native ({}) · {}ch → ASIO: {}",
                dsd::fmt_mhz(a.dsd_rate), a.channels, a.driver_name,
            );
            let dropped = a.underruns();
            if dropped > 0 {
                line.push_str(&format!(" · ⚠ {dropped} dropout(s)"));
            }
            return Some(line);
        }
        #[cfg(all(target_os = "linux", feature = "alsa-dsd"))]
        if self.dsd_native && let Some(a) = &self.alsa {
            let label = self.dsd_label.as_deref().unwrap_or("DSD");
            return Some(format!(
                "{label} native ({}) · {}ch · {} → ALSA: {}",
                dsd::fmt_mhz(a.dsd_rate), a.channels, a.format_label, a.device_name,
            ));
        }
        self.bp.as_ref().map(|s| {
            let mut line = if s.dop {
                let label = self.dsd_label.as_deref().unwrap_or("DSD");
                format!("{label} via DoP · {} → {}", s.describe(), s.device_name())
            } else {
                format!("{} → {}", s.describe(), s.device_name())
            };
            // Only ever shown once something has actually gone wrong: a dropout
            // is otherwise invisible after the fact, and "did that pop come
            // from Moosik or from the DAC?" is not a question guesswork should
            // answer.
            let dropped = s.underruns();
            if dropped > 0 {
                line.push_str(&format!(" · ⚠ {dropped} dropout(s)"));
            }
            let missed = s.lock_misses();
            if missed > 0 {
                line.push_str(&format!(" · ⚠ {missed} late callback(s)"));
            }
            line
        })
    }

/// Drop every direct/exclusive output handle and publish the idle state.
///
/// Both halves matter. Releasing the handle without clearing the state
/// leaves a green diamond describing a device nothing is connected to;
/// clearing the state without releasing the handle leaves an exclusive
/// stream holding the endpoint so nothing else can open it.
fn close_bp(&mut self) {
    self.release_bp();
    self.native_close();
    self.retire_q31_scan();
    self.bp_state.stopped();
}

/// Release every handle that could hold the endpoint, without touching the
/// session state.
///
/// Used on the way into shared/processed playback: `rodio` cannot open a
/// device this process is still holding exclusively, and the failure it
/// produces looks like a missing device rather than a self-inflicted one.
fn release_exclusive_handles(&mut self) {
    self.release_bp();
    self.native_close();
}

/// Publish the route that was just opened, with the fidelity it can
/// actually prove.
///
/// The claim comes from the transport and the format the backend reported,
/// never from the user's preference. `can_carry_exact` is a precondition
/// list — WASAPI exclusive PCM/DoP and ASIO — and everything else plays
/// while saying why it is not claiming.
/// Why a source's own description prevents an exact claim, if it does.
///
/// A container that declares "six channels" without saying which speakers
/// they are gives no way to check that the samples reach the right ones.
/// Above stereo that is not a detail: the difference between L/R/LFE and
/// L/R/C is silent in a spectrum and obvious in a room.
fn source_blocks_exact_claim(source: &bitperfect::state::MediaSource) -> Option<String> {
    if source.truncated {
        return Some(
            "the file ends before the audio its header declares, so it cannot deliver \
                 the whole track"
                .into(),
        );
    }
    if source.format.channels > 2 && !source.format.layout.is_specified() {
        return Some(format!(
            "the source declares {} channels but does not say which speakers they are, \
                 so the channel mapping cannot be verified",
            source.format.channels
        ));
    }
    None
}

/// What a route carrying `source` under `plan` on `transport` may claim.
///
/// The only place this is decided. The open path and the gapless rollover both
/// call it, so a track that arrives at a boundary is judged by exactly the
/// same rule as one that arrives at an open — which is what stops a hand-off
/// inheriting a claim.
///
/// Nothing here reads a preference, and nothing here reads a scan. A
/// conversion is Processed at the moment it opens, whatever it may later turn
/// out to have been; the value-exact upgrade is applied afterwards, by
/// `apply_q31_verdict`, and only when a complete scan and the running
/// conversion agree.
fn fidelity_for(
    source: &bitperfect::state::MediaSource,
    plan: bitperfect::PayloadPlan,
    transport: &bitperfect::Transport,
) -> bitperfect::state::Fidelity {
    use bitperfect::state::Fidelity;
    // A known transform comes first, and this order is the point.
    //
    // The unverifiable-source check ran ahead of it, so a six-channel 64-bit
    // float file with no speaker layout came out **Unverified** — "nothing
    // Moosik did altered it, but the path downstream cannot be proved" — of a
    // route that had just narrowed it to `f32` and rounded it into Q1.31.
    // Being unsure about the channel mapping does not make the conversion stop
    // having happened. Processed is the stronger and more specific statement,
    // and the layout doubt is carried alongside it rather than instead of it.
    if plan != bitperfect::PayloadPlan::Identity {
        // Why, and it is not always the same why.
        //
        // A Float32 source takes this route because the device would not give
        // us a Float32 exclusive format — a fact about the DAC, and a
        // different DAC could avoid it. A Float64 source takes it because no
        // device format anywhere carries a `double`: it is narrowed to `f32`
        // in the decoder before any of this, and would be on hardware that
        // does not exist yet. Telling the listener their DAC was the problem
        // was wrong in the second case, and invited them to go looking for one
        // that would fix it.
        let mut reason = if matches!(
            source.format.kind,
            bitperfect::format::PcmKind::Float64
        ) {
            "a 64-bit float source has no exact representation in any device format, so it \
             is narrowed to 32-bit float and converted to 32-bit integer"
                .to_string()
        } else {
            "the device has no Float32 exclusive format, so samples are converted to \
             32-bit integer"
                .to_string()
        };
        // Both facts, when both are true. The transform is what happened to
        // the samples; this is what cannot be established about where they go.
        if let Some(doubt) = Self::source_blocks_exact_claim(source) {
            reason.push_str("; and ");
            reason.push_str(&doubt);
        }
        Fidelity::Processed {
            transform: plan.describe(source.format.kind),
            reason,
        }
    } else if let Some(why) = Self::source_blocks_exact_claim(source) {
        // No transform of ours, and something about the source that cannot be
        // checked: this is the case `Unverified` is for.
        Fidelity::Unverified { reason: why }
    } else if transport.can_carry_exact() {
        Fidelity::PayloadExact
    } else {
        Fidelity::Unverified {
            reason: match transport {
                bitperfect::Transport::CpalDirect { .. } => {
                    "cpal requested this format, but ALSA, PipeWire, Pulse, dmix or \
                     CoreAudio may still resample or mix on the way to the hardware, and \
                     this process cannot see which route it got"
                        .into()
                }
                _ => "this route does not report enough to prove exactness".into(),
            },
        }
    }
}

fn bp_note_opened(&mut self, source: bitperfect::state::MediaSource) {
    use bitperfect::state::{CarrierFormat, CarrierKind, ProcessingState};
    let Some(bp) = self.bp.as_ref() else { return };
    let (transport, output, dop) = (bp.transport.clone(), bp.output.clone(), bp.dop);
    let plan = bp.plan;
    let carrier = dop.then_some(CarrierFormat {
        kind: CarrierKind::Dop,
        rate: output.sample_rate,
        channels: output.channels,
        bits: output.valid_bits,
    });

    let fidelity = Self::fidelity_for(&source, plan, &transport);

    let mut processing = ProcessingState::transparent_locked();
    processing.transform = plan.describe(source.format.kind);
    self.bp_state.set_source(source);
    self.bp_state
        .opened(transport, fidelity, Some(output), carrier, processing);
}

/// Start proving — or look up — whether every sample of `path` is exactly
/// representable in Q1.31.
///
/// Nothing is claimed here. The route is already playing as processed; a
/// positive verdict upgrades it to value-exact, and a negative one leaves
/// it exactly where it is with an accurate reason.
fn begin_q31_scan(&mut self, path: &Path) {
    use bitperfect::q31::Q31Key;
    // Retire every earlier scan before starting this one. Cancelling the old
    // worker *and* dropping its receiver, on every track transition, is what
    // stops a result computed for a previous file arriving here at all.
    self.retire_q31_scan();

    let generation = self.bp_state.generation;
    let audio_generation = self.bp_audio_gen;
    let Some(key) = Q31Key::of(path, 0) else {
        // No identity, so no cache entry could ever be validated and no
        // verdict could ever be trusted. Say so rather than leaving the panel
        // waiting.
        self.note_q31_unresolved("this file's identity could not be read, so exactness \
                                  cannot be proved for it");
        return;
    };

    if let Some(verdict) = self.q31_cache.get(&key) {
        // A cached verdict still does not put a sample on the wire: the
        // conversion re-checks every packet regardless.
        self.apply_q31_verdict(generation, audio_generation, verdict);
        return;
    }

    let (tx, rx) = std::sync::mpsc::channel();
    let cancel = std::sync::Arc::clone(&self.q31_cancel);
    let p = path.to_path_buf();
    match std::thread::Builder::new()
        .name("q31-scan".into())
        .spawn(move || {
            // Both arms are sent. Dropping the error arm left the receiver
            // silent and the session labelled "checking" forever.
            let msg = match bitperfect::scan_track_q31(&p, &cancel) {
                Ok(v) => Q31Outcome::Verdict(v),
                Err(e) => Q31Outcome::Failed(e.message().to_string()),
            };
            let _ = tx.send(msg);
        }) {
        Ok(_) => {
            self.q31_rx = Some(Q31Scan {
                rx,
                generation,
                audio_generation,
                key,
            });
        }
        Err(e) => {
            self.note_q31_unresolved(format!("the exactness scan could not start: {e}"));
        }
    }
}

/// Cancel and drop any scan in flight.
///
/// Called on every track transition — a new track, a seek, a stop, a route
/// Cancel whatever seek worker is running and hand out a fresh flag.
///
/// Called at the start of every spawn and by every path that abandons a seek,
/// so a superseded worker learns it has been superseded rather than decoding
/// to the end of a discard loop nobody will read.
fn new_seek_cancel(&mut self) -> std::sync::Arc<std::sync::atomic::AtomicBool> {
    use std::sync::atomic::Ordering;
    self.seek_cancel.store(true, Ordering::Relaxed);
    self.seek_cancel = std::sync::Arc::new(std::sync::atomic::AtomicBool::new(false));
    std::sync::Arc::clone(&self.seek_cancel)
}

/// Tell any seek worker in flight to stop, without starting another.
fn cancel_seek_worker(&mut self) {
    self.seek_cancel.store(true, std::sync::atomic::Ordering::Relaxed);
}

/// change — so no worker outlives the session that asked for it and no
/// receiver can deliver a result that belongs to a file nobody is playing.
fn retire_q31_scan(&mut self) {
    self.q31_cancel
        .store(true, std::sync::atomic::Ordering::Relaxed);
    self.q31_rx = None;
    #[cfg(test)]
    { self.held_scan = None; }
    self.q31_cancel = std::sync::Arc::new(std::sync::atomic::AtomicBool::new(false));
}

/// Record that value-exactness will not be decided for this track.
///
/// The route keeps playing and keeps its honest Processed label; what changes
/// is that the reason stops saying a check is under way when none is.
fn note_q31_unresolved(&mut self, why: impl Into<String>) {
    use bitperfect::state::Fidelity;
    let why = why.into();
    crate::mlog!("bp      value-exactness undecided: {why}");
    if let Fidelity::Processed { transform, .. } = &self.bp_state.fidelity {
        let transform = transform.clone();
        self.bp_state.fidelity = Fidelity::Processed {
            transform,
            reason: format!(
                "samples are converted to 32-bit integer; {why}"
            ),
        };
    }
}

/// Fold a completed scan into the session, if it is still the session that
/// asked for it.
fn poll_q31_scan(&mut self, path: Option<&Path>) {
    use std::sync::mpsc::TryRecvError;
    let Some(scan) = self.q31_rx.as_ref() else {
        return;
    };
    let (generation, audio_generation, key) =
        (scan.generation, scan.audio_generation, scan.key.clone());
    let outcome = match scan.rx.try_recv() {
        Ok(o) => o,
        Err(TryRecvError::Empty) => return,
        // The worker thread is gone without having sent anything — it
        // panicked, or the process is tearing down. Either way nothing else
        // is coming, and the session must stop saying a check is under way.
        Err(TryRecvError::Disconnected) => {
            self.q31_rx = None;
            self.note_q31_unresolved("the exactness scan ended without a result");
            return;
        }
    };
    self.q31_rx = None;

    match outcome {
        Q31Outcome::Failed(why) => {
            self.note_q31_unresolved(format!("the exactness scan could not finish: {why}"));
        }
        Q31Outcome::Verdict(verdict) => {
            // Cached under the key the scan was *started* with. Rebuilding the
            // key from whatever is playing now is how one file's verdict got
            // filed under another file's identity.
            self.q31_cache.insert(key.clone(), verdict.clone());
            // Written through rather than at exit: a player is as likely to be
            // killed as closed, and a verdict that only exists in memory is
            // one that has to be re-earned by decoding the whole file again.
            self.q31_cache.save(&moosik_dir());

            // And applied only if that identity is still what is playing. The
            // generation check below covers the ordinary case; this covers the
            // one where a track is replaced by a different file within the
            // same generation.
            let still_current = path
                .and_then(|p| bitperfect::q31::Q31Key::of(p, 0))
                .map(|now| now == key)
                .unwrap_or(false);
            if !still_current {
                crate::mlog!(
                    "bp      value-exactness verdict discarded: it was computed for a \
                     file that is no longer playing"
                );
                return;
            }
            self.apply_q31_verdict(generation, audio_generation, verdict);
        }
    }
}

fn apply_q31_verdict(
    &mut self,
    generation: u64,
    audio_generation: u64,
    verdict: bitperfect::q31::Q31Verdict,
) {
    use bitperfect::state::{Fidelity, TransformDescription};
    if !self.bp_state.accepts(generation) {
        crate::mlog!(
            "bp      value-exactness verdict ignored: it belongs to a session \
                          that has already been replaced"
        );
        return;
    }
    // And the audio clock, which is a different clock — read from the stream,
    // not from the copy this side has folded.
    //
    // The session generation is the UI's, and it does not move at a gapless
    // hand-off until the UI folds one; `bp_audio_gen` is that folded copy and
    // lags by up to a frame. The device crosses the boundary at the sample. In
    // that window the folded clock still names the outgoing track while the
    // incoming one is what is audible, so a verdict computed for the outgoing
    // track would be printed against the incoming one — and it would *stay*
    // printed, because a verdict is a label on the session, not a transient.
    //
    // Read from the live stream, and read again after the evidence below, so
    // that a crossing during the check is caught rather than straddled.
    let live_before = self.bp.as_ref().map(|s| s.audio_generation());
    if live_before != Some(audio_generation) {
        crate::mlog!(
            "bp      value-exactness verdict ignored: the device has moved on to \
             another track since the scan began"
        );
        return;
    }
    if self.bp_state.fidelity.is_faulted() {
        return;
    }
    // Two independent conditions, and the label needs both.
    //
    // The scan read the whole file and says every sample lands on the Q1.31
    // lattice. The running conversion has been checking each sample as it
    // publishes it and has not had to round one. A scan can be stale — the
    // file may have been rewritten under it — and the conversion has only seen
    // what has played so far, so neither is sufficient alone.
    // Read against the generation the verdict was computed for, not against
    // whatever the stream would answer now — and only believed when the read
    // caught a quiet instant.
    //
    // A rounding count that is being published *right now* has had its window
    // opened and its store not yet made: it reads as zero, and zero is exactly
    // the condition this label is granted on. An unsettled read is therefore
    // not evidence of cleanliness; it is evidence that the question cannot be
    // answered at this instant, and the answer to that is to leave the track
    // labelled Processed and ask again on the next tick.
    let observed_clean = self
        .bp
        .as_ref()
        .map(|s| {
            let ev = s.evidence_at(audio_generation);
            ev.settled && ev.off_grid == 0
        })
        .unwrap_or(false);
    let on_the_route = self.bp.as_ref().map(|s| s.plan) == Some(bitperfect::PayloadPlan::Q31);

    // The other half of the before/after. Everything above was read while the
    // device was on `audio_generation`; if it has moved since, the reads were
    // straddling a hand-off and none of them is about one track.
    let live_after = self.bp.as_ref().map(|s| s.audio_generation());
    if live_after != Some(audio_generation) {
        crate::mlog!(
            "bp      value-exactness verdict ignored: the device crossed into \
             another track while it was being checked"
        );
        return;
    }

    // A third, and it is not about the conversion at all.
    //
    // Value-exact means the representation changed and no number did. A 64-bit
    // float source has already had its numbers changed before anything here
    // sees them: the decoder narrows it to `f32` on the way into the canonical
    // buffer, and both the scan and the running conversion look at what came
    // out of that. Every sample can land perfectly on the Q1.31 lattice and
    // the claim still be false — so this route was handing a green-adjacent
    // "value-exact" label to a source that had been rounded twice.
    let already_narrowed = self.bp_state.source.as_ref().is_some_and(|s| {
        matches!(s.format.kind, bitperfect::format::PcmKind::Float64)
    });

    if value_exact_allowed(
        verdict.value_exact,
        observed_clean,
        on_the_route,
        already_narrowed,
    ) {
        crate::mlog!("bp      value-exact: {}", verdict.describe());
        self.bp_state.fidelity = Fidelity::ValueExact {
            transform: TransformDescription::FloatToQ31ValueExact,
        };
        let mut processing = self.bp_state.processing.clone();
        processing.transform = TransformDescription::FloatToQ31ValueExact;
        self.bp_state.set_processing(processing);
        return;
    }

    // Not value-exact, and the reason stops saying a check is under way.
    let why = if already_narrowed {
        "a 64-bit float source is narrowed to 32-bit float before this conversion sees it, \
         so the numbers have already changed"
            .to_string()
    } else if !on_the_route {
        "this route is not the integer conversion".to_string()
    } else if !verdict.value_exact {
        format!("a complete scan found samples that are not exactly representable: {}",
                verdict.describe())
    } else {
        "the running conversion has already had to round a sample".to_string()
    };
    crate::mlog!("bp      not value-exact: {why}");
    self.note_q31_unresolved(why);
}

/// Fold any realtime integrity fault into the session claim.
///
/// The audio threads publish a `u8` and never a string; this is where it
/// becomes a typed reason. Called from the UI tick, so a dropout revokes
/// the claim within a frame of happening rather than at the end of the
/// track. Faults are generation-scoped: one raised by a session that has
/// since been replaced is ignored rather than applied to its successor.
fn bp_poll_integrity(&mut self) {
    let generation = self.bp_state.generation;
    // One stamped read, for every consumer in this tick.
    //
    // It used to be a generation comparison followed by three separate
    // getters, which is three chances for the device to cross a gapless
    // boundary in between — and the fields that came back could then describe
    // two different tracks. A check placed before the reads cannot be about
    // what happens after them.
    let evidence = self
        .bp
        .as_ref()
        .map(|bp| bp.evidence_at(self.bp_audio_gen));
    self.bp_check_value_exact_still_holds(evidence);
    // Two records, into two fields.
    //
    // Both used to be pushed at `SessionState::fault`, recoverable one first
    // so that it kept its place in that method's first-wins ordering. It kept
    // more than its place: `fault` holds one reason and refuses the second, so
    // the fatal one — the account of why the audio stopped — was silently
    // dropped every time a dropout had preceded it. `revoked` is its own field
    // now, so the loss keeps its place in the detail and the badge says what
    // ended the track.
    let mut codes: Vec<u8> = Vec::new();
    let mut losses: Vec<u8> = Vec::new();
    if let Some(ev) = evidence {
        // Both fields came out of the same stamped read, so they are about the
        // same track or they are about nothing. A boundary crossing between
        // them is no longer a thing that can happen, rather than a thing a
        // check tries to get ahead of.
        losses.push(ev.revoked);
        codes.push(ev.fatal);
    }
    #[cfg(all(windows, feature = "asio-dsd"))]
    if let Some(a) = self.asio.as_ref() {
        let ev = a.evidence_at(self.native_audio_gen);
        losses.push(ev.revoked);
        codes.push(ev.fatal);
    }
    #[cfg(all(target_os = "linux", feature = "alsa-dsd"))]
    if let Some(a) = self.alsa.as_ref() {
        let ev = a.evidence_at(self.native_audio_gen);
        losses.push(ev.revoked);
        codes.push(ev.fatal);
    }
    // The native-DSD feeder is a producer with no backend of its own, so its
    // faults arrive here or nowhere.
    #[cfg(any(
        all(windows, feature = "asio-dsd"),
        all(target_os = "linux", feature = "alsa-dsd"),
    ))]
    if self.dsd_native {
        codes.push(self.feed_status.fault());
    }
    apply_integrity(&mut self.bp_state, generation, &losses, &codes);
}

/// Withdraw a value-exact claim the moment the running conversion contradicts
/// it.
///
/// The claim has two conditions and both have to keep holding. A cached
/// verdict is checked once, when it arrives; the conversion goes on producing
/// evidence for the rest of the track. A verdict that was true of the file
/// when it was scanned — and stale by the time it was used, because the file
/// was rewritten underneath it — would otherwise leave a green-adjacent
/// value-exact label on a track that is being rounded, which is precisely the
/// claim-without-evidence this release exists to remove.
///
/// Same generation only: a scan and a stream that belong to different sessions
/// have nothing to say about each other.
fn bp_check_value_exact_still_holds(&mut self, evidence: Option<bitperfect::Evidence>) {
    use bitperfect::state::{Fidelity, TransformDescription};
    if !matches!(self.bp_state.fidelity, Fidelity::ValueExact { .. }) {
        return;
    }
    // From the same stamped read the rest of this tick uses, rather than a
    // fresh getter that could answer about a different track.
    let on_the_route = self.bp.as_ref().map(|s| s.plan) == Some(bitperfect::PayloadPlan::Q31);
    let off_grid = match evidence {
        // The route is not the conversion any more, so the claim is not this
        // session's to keep either.
        Some(ev) if on_the_route => ev.off_grid,
        _ => 0,
    };
    if off_grid == 0 {
        return;
    }
    crate::mlog!(
        "bp      value-exact withdrawn: the conversion has had to round {off_grid} \
         sample(s), whatever the scan concluded"
    );
    self.bp_state.fidelity = Fidelity::Processed {
        transform: TransformDescription::FloatToQ31Processed,
        reason: format!(
            "a cached verdict said every sample was exactly representable, but {off_grid} \
             sample(s) have needed rounding"
        ),
    };
    let mut processing = self.bp_state.processing.clone();
    processing.transform = TransformDescription::FloatToQ31Processed;
    self.bp_state.set_processing(processing);
}

/// Whether an EQ stage is actually in the rodio path right now.
///
/// Conservative on contention: if the audio thread holds the EQ lock the
/// answer is "active", because over-reporting processing is the safe
/// direction and under-reporting it is the defect this release exists to
/// fix.
fn eq_is_active(&self) -> bool {
    match self.eq.as_ref() {
        None => false,
        Some(h) => match h.try_lock() {
            Ok(eq) => eq.enabled && eq.bands.iter().any(|b| b.enabled),
            Err(_) => true,
        },
    }
}

/// The processing state of the ordinary shared/rodio path, as it is right
/// now.
///
/// Read from the live engine rather than assumed: 1.4.2 hard-coded "EQ
/// active" here, so a processed fallback claimed an EQ stage whether or not
/// one was configured.
fn shared_processing(&self) -> bitperfect::state::ProcessingState {
    bitperfect::state::ProcessingState {
        gain: self.volume,
        gain_locked: false,
        eq_active: self.eq_is_active(),
        replaygain_active: self.replay_gain != 1.0,
        transform: bitperfect::state::TransformDescription::Identity,
    }
}

/// Publish the ordinary shared/processed route, with the endpoint `rodio`
/// actually resolved rather than the bit-perfect device that was selected.
fn bp_note_processed(
    &mut self,
    source: Option<bitperfect::state::MediaSource>,
    transform: bitperfect::state::TransformDescription,
    reason: impl Into<String>,
) {
    use bitperfect::state::{EndpointIdentity, Fidelity, Transport};
    // `rodio` does not report which endpoint it opened, so the honest
    // answer is that it did not report one — not the device the user
    // picked for the bit-perfect path, which is a different question.
    let endpoint = EndpointIdentity::new("", "system default (not reported by the backend)");
    let mut processing = self.shared_processing();
    processing.transform = transform.clone();
    if let Some(s) = source {
        self.bp_state.set_source(s);
    }
    self.bp_state.opened(
        Transport::Shared { endpoint },
        Fidelity::Processed {
            transform,
            reason: reason.into(),
        },
        None,
        None,
        processing,
    );
}

/// Publish a native-DSD route: the source is the DSD file, the carrier is
/// the raw DSD byte stream, and they are shown separately.
///
/// `unverified` carries the reason a transport that could otherwise claim
/// exactness is not doing so; `None` means it claims.
/// Publish an open native-DSD session — and record the track it is
/// playing.
///
/// Both native routes end here, which is why the record is taken here:
/// `ASIO` and `ALSA` were the two routes that opened a device, played a
/// file and never wrote down which file it was. `Engine::current` is what
/// `resume` reopens after a failed seek, so on a native route it reopened
/// whatever the last *other* route had left behind.
fn publish_native_dsd(
    &mut self,
    path: &Path,
    start: Duration,
    transport: bitperfect::Transport,
    info: &dsd::DsdInfo,
    channels: u16,
    unverified: Option<&str>,
) {
    use bitperfect::state::{
        CarrierFormat, CarrierKind, Fidelity, MediaSource, OutputFormat, ProcessingState,
    };
    let endpoint = transport.endpoint().cloned().unwrap_or_default();
    let source = MediaSource::dsd(
        bitperfect::format::SourceFormat {
            kind: bitperfect::format::PcmKind::Integer { valid_bits: 1 },
            sample_rate: info.sample_rate,
            channels,
            layout: bitperfect::format::ChannelLayout(info.layout.unwrap_or(0)),
        },
        info.rate_label(),
    );
    let source = if info.is_truncated() {
        source.truncated()
    } else {
        source
    };
    let fidelity = match unverified
        .map(str::to_string)
        .or_else(|| Self::source_blocks_exact_claim(&source))
    {
        None => Fidelity::PayloadExact,
        Some(why) => Fidelity::Unverified { reason: why },
    };
    // The file, the parsed source and where playback actually starts from.
    self.note_current(path, Some(source.clone()), start);
    self.bp_state.set_source(source);
    self.bp_state.opened(
        transport,
        fidelity,
        Some(OutputFormat {
            endpoint,
            sample_rate: info.sample_rate,
            channels,
            layout: bitperfect::format::ChannelLayout::UNSPECIFIED,
            container_bits: 1,
            valid_bits: 1,
            integer: true,
            // Native DSD runs its own drain inside the backend callback.
            buffer_frames: 0,
            label: "native DSD".into(),
        }),
        Some(CarrierFormat {
            kind: CarrierKind::NativeDsd,
            rate: info.sample_rate,
            channels,
            bits: 1,
        }),
        ProcessingState::transparent_locked(),
    );
}

/// Refresh the live processing description without changing the route.
///
/// Called whenever volume, EQ or ReplayGain changes, in the same operation
/// that changes the audio, so the panel can never describe a processing
/// chain the engine is no longer running.
fn bp_refresh_processing(&mut self) {
    if self.bp_state.transport.has_gain_stage() && self.bp_state.playback.is_active() {
        let mut p = self.shared_processing();
        p.transform = self.bp_state.processing.transform.clone();
        self.bp_state.set_processing(p);
    }
}
    fn pause(&mut self) {
        // A seek may still be decoding on the worker; remember the intent so it
        // resumes paused.
        if let Some(ps) = self.pending_seek.as_mut() { ps.was_paused = true; }
        // Pause keeps the transport and the fidelity claim: nothing has been
        // altered, the device is simply being fed protocol silence.
        if self.bp_state.playback.is_active() {
            self.bp_state.set_playback(bitperfect::state::PlaybackState::Paused);
        }
        if self.on_native_stream() {
            if self.native_pause()
                && let Some(started) = self.started_at.take() {
                self.paused_elapsed += started.elapsed();
            }
        } else if self.on_bp_stream() {
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
        if self.bp_state.playback.is_active() {
            self.bp_state.set_playback(bitperfect::state::PlaybackState::Playing);
        }
        if self.on_native_stream() {
            if self.native_resume() {
                self.started_at = Some(Instant::now());
            }
        } else if self.on_bp_stream() {
            if let Some(ref bp) = self.bp
                && bp.is_paused() {
                bp.resume();
                self.started_at = Some(Instant::now());
            }
        } else if let Some(ref sink) = self.sink {
            if sink.is_paused() {
                sink.play();
                self.started_at = Some(Instant::now());
            }
        } else if let Some((path, at)) = self.resume_target() {
            // There is nothing to resume, and doing nothing is the wrong
            // answer.
            //
            // A failed seek releases the sink — it has to, the old decoder is
            // already gone — so afterwards this method had no sink, no
            // bit-perfect stream and no native handle, fell off the end, and
            // returned. Pressing play did nothing at all, forever, with the
            // position sitting on screen and no way back to the audio short of
            // reselecting the track.
            //
            // Reopened from `current`, not from `last_prepared`. The latter is
            // written by the exact route and by a completed seek and by
            // nothing else, so on a shared track it names whichever file last
            // took the exact route — a different track, quite possibly from
            // earlier in the playlist, which is what would have started
            // playing.
            let duration = self.current_duration;
            crate::mlog!(
                "seek    resuming a released sink: reopening {} at {:.2}s",
                path.display(),
                at.as_secs_f64()
            );
            self.play_seeked_async(&path, at, duration, false);
        }
    }

    /// Stop playback and drop every active claim.
    ///
    /// Both halves are required. 1.4.2 stopped the *session* but left the
    /// exclusive/native handles open and the session state saying `streaming`,
    /// so a stopped player kept showing a green diamond describing a device it
    /// was no longer sending anything to — and kept holding that device open
    /// against every other application on the machine.
    ///
    /// The user's request and policy are untouched: stopping is not a
    /// preference change.
    fn stop(&mut self) {
        self.cancel_seek_worker(); // and tell it to stop, not merely stop listening
        self.pending_seek = None; // abandon any in-flight background seek
        let ended_shared_route = self.release_sink();
        // Release, not merely idle: an exclusive stream that stays open holds
        // the endpoint.
        self.release_bp();
        self.native_close();
        // A scan outliving the track it was started for is how a verdict for
        // one file reached another.
        self.retire_q31_scan();
        // Exactly once for the whole teardown. `release_sink` above already
        // advanced the route if there was a shared sink to release; a
        // bit-perfect or native route has no sink, so this is where those end.
        // Said out loud rather than left to a tap's own drop: a drop takes the
        // buffer lock without blocking and may simply miss it, and a stopped
        // player showing a live spectrum is exactly the kind of stale truth
        // this engine spends most of its teardown avoiding.
        if !ended_shared_route {
            spectrum::invalidate_live_pcm_route(&self.stereo_buf, &self.sample_buf);
        }
        self.dsd_mode = false;
        self.dsd_native = false;
        self.dsd_fallback = false;
        self.dsd_fallback_note = None;
        self.bp_base = Duration::ZERO;
        self.started_at = None;
        self.paused_elapsed = Duration::ZERO;
        self.current_duration = None;
        self.bp_state.stopped();
    }

    /// Whether playback is over, for any reason — what "stop waiting" means.
    fn is_finished(&self) -> bool {
        self.completion().is_over()
    }

/// *How* the current session ended, from whichever object owns it.
///
/// The distinction this carries is the one that was missing. Every fatal
/// path — a read error, a decode error, a dead backend, a panicked thread,
/// a driver reset — used to arrive at the UI as the same `finished` flag
/// the end of a track arrives on, because that flag's real job was "stop
/// waiting for audio". The auto-advance then treated it as "the track
/// ended" and started the next one, which under Repeat One is the *same*
/// file: a file that fails deterministically reopened forever.
fn completion(&self) -> bitperfect::Completion {
    if self.on_native_stream() {
        return self.native_completion();
    }
    if self.on_bp_stream() {
        return self
            .bp
            .as_ref()
            .map(|bp| bp.completion())
            .unwrap_or(bitperfect::Completion::Running);
    }
    // The shared decoder has no failure channel: `rodio`'s `Decoder` yields
    // `None` for a corrupt frame exactly as it does for the end of the file,
    // so an empty sink alone cannot tell "played out" from "gave up". Taken
    // as a clean end — which is what it was — a file that failed to decode
    // advanced the playlist, and under Repeat One advanced into itself.
    //
    // What is observable is how much of the track was played, and the rule
    // that draws a conclusion from it is `shared_completion`, on its own so
    // that the conclusion can be tested without an audio device.
    let Some(sink) = self.sink.as_ref() else {
        return bitperfect::Completion::Running;
    };
    shared_completion(
        sink.empty(),
        !(self.started_at.is_none() && self.paused_elapsed == Duration::ZERO),
        self.elapsed(),
        self.current_duration,
        self.bp_state.generation,
    )
}
    /// The native-DSD backend's terminal state, or `Running` without one.
    #[allow(unreachable_code)]
    fn native_completion(&self) -> bitperfect::Completion {
        #[cfg(all(windows, feature = "asio-dsd"))]
        if let Some(a) = &self.asio {
            return a.completion();
        }
        #[cfg(all(target_os = "linux", feature = "alsa-dsd"))]
        if let Some(a) = &self.alsa {
            return a.completion();
        }
        bitperfect::Completion::Running
    }

    fn elapsed(&self) -> Duration {
        if self.on_native_stream() {
            return self.bp_base + self.native_played().saturating_sub(self.bp_played_at_track_start);
        }
        if self.on_bp_stream() && let Some(ref bp) = self.bp {
            // Sample-accurate: frames delivered to the device, less the frames
            // that belonged to earlier gapless tracks in this session.
            return self.bp_base + bp.played().saturating_sub(self.bp_played_at_track_start);
        }
        let running = self.started_at.map(|t| t.elapsed()).unwrap_or(Duration::ZERO);
        self.paused_elapsed + running
    }

    /// End playback because the session failed.
    ///
    /// Every handle is released — a failed session holds nothing useful, and
    /// an exclusive handle left open is one no other application can take
    /// either — but the *reason* is kept. `stop()` would clear it, and an idle
    /// player with no account of why it is idle is the state this whole
    /// distinction exists to avoid.
    fn halt(&mut self, reason: u8) {
        self.cancel_seek_worker();
        self.pending_seek = None;
        let ended_shared_route = self.release_sink();
        if let Some(r) = bitperfect::fault::to_reason(reason) {
            self.bp_state.halted(r);
        }
        self.release_bp();
        self.native_close();
        self.retire_q31_scan();
        // Exactly once for the whole teardown. `release_sink` above already
        // advanced the route if there was a shared sink to release; a
        // bit-perfect or native route has no sink, so this is where those end.
        // Said out loud rather than left to a tap's own drop: a drop takes the
        // buffer lock without blocking and may simply miss it, and a stopped
        // player showing a live spectrum is exactly the kind of stale truth
        // this engine spends most of its teardown avoiding.
        if !ended_shared_route {
            spectrum::invalidate_live_pcm_route(&self.stereo_buf, &self.sample_buf);
        }
        self.dsd_mode = false;
        self.dsd_native = false;
        self.dsd_fallback = false;
        self.started_at = None;
        self.current_duration = None;
    }

    /// One attempt at taking a route, as a transaction.
    ///
    /// `body` may acquire anything an open acquires — a sink, an exact stream,
    /// a native session, a scan thread — and it may fail at any of the dozen
    /// places inside it that end in a `?`. What this guarantees is that if it
    /// returns an error, nothing it took survives: the transaction either
    /// commits or it undoes itself, and no caller has to remember which of the
    /// half-dozen things a particular failure might have left behind.
    ///
    /// That memory is exactly what the callers did not have. `play_index`
    /// published the typed reason and released nothing, so a DSD open that
    /// failed after the ASIO driver was loaded left the driver loaded and the
    /// endpoint held. A seek that got as far as reopening the output and then
    /// failed left the stream half-installed and the transport stuck on
    /// "verifying". Each of them was a different subset of the same six
    /// steps.
    ///
    /// `reason` maps the failure to the typed one the panel shows, so the
    /// account the listener gets is the failure's own and not
    /// `DeviceUnavailable` for everything.
    fn open_attempt<T, E>(
        &mut self,
        body: impl FnOnce(&mut Self) -> Result<T, E>,
        reason: impl FnOnce(&E) -> bitperfect::state::FailureReason,
    ) -> Result<T, E> {
        match body(self) {
            Ok(v) => Ok(v),
            Err(e) => {
                self.abort_open(reason(&e));
                Err(e)
            }
        }
    }

    /// Undo an open that got far enough to hold something.
    ///
    /// Every error path after a device or direct handle exists has to leave
    /// the same state: no audible stream, no held DAC, no scan in flight, no
    /// current track, and the typed reason on screen. Doing that in each
    /// caller is how one of them came to `?` out with audio playing at
    /// position zero and an exclusive handle still open, which is a device no
    /// other application can take and a track the listener never asked to hear
    /// from the beginning.
    ///
    /// Distinct from `halt`, which is for a session that really played and
    /// then failed: that one keeps the current track and reports an integrity
    /// reason. This is for a route that never started, so it also drops
    /// `current` — a route that never played is not what is playing.
    fn abort_open(&mut self, reason: bitperfect::state::FailureReason) {
        self.cancel_seek_worker();
        self.pending_seek = None;
        let ended_shared_route = self.release_sink();
        self.release_bp();
        self.native_close();
        self.retire_q31_scan();
        // Exactly once for the whole teardown. `release_sink` above already
        // advanced the route if there was a shared sink to release; a
        // bit-perfect or native route has no sink, so this is where those end.
        // Said out loud rather than left to a tap's own drop: a drop takes the
        // buffer lock without blocking and may simply miss it, and a stopped
        // player showing a live spectrum is exactly the kind of stale truth
        // this engine spends most of its teardown avoiding.
        if !ended_shared_route {
            spectrum::invalidate_live_pcm_route(&self.stereo_buf, &self.sample_buf);
        }
        self.dsd_mode = false;
        self.dsd_native = false;
        self.dsd_fallback = false;
        self.started_at = None;
        self.current_duration = None;
        self.current = None;
        self.bp_state.failed(reason);
    }

    /// Apply a new user volume.
    ///
    /// Refuses while the active route has no gain stage, and says so by
    /// returning `false`, so no caller can move the saved value on a route
    /// where it would have no effect.
    fn set_volume(&mut self, vol: f32) -> bool {
        if self.volume_is_locked() {
            return false;
        }
        self.volume = vol;
        if let Some(ref sink) = self.sink {
            sink.set_volume(self.rodio_volume());
        }
        self.bp_refresh_processing();
        true
    }

    /// Whether the volume control has anything to act on right now.
    ///
    /// One predicate, derived from the session state, shared by the slider,
    /// the keyboard, the media keys and anything added later. 1.4.2 disabled
    /// only the slider, so ArrowUp/ArrowDown still overwrote the saved volume
    /// while an exact route was open — the value was gone by the time normal
    /// playback resumed.
    fn volume_is_locked(&self) -> bool {
        self.bp_state.volume_locked()
    }

    /// Set the ReplayGain factor (linear) and apply it live to the rodio sink.
    fn set_replay_gain(&mut self, gain: f32) {
        self.replay_gain = gain;
        if let Some(ref sink) = self.sink {
            sink.set_volume(self.rodio_volume());
        }
        // The panel describes the processing chain that is running now, so it
        // changes in the same operation the audio does.
        self.bp_refresh_processing();
    }

    /// Append `path` to the current rodio sink for gapless continuation — rodio
    /// plays queued sources back-to-back with no gap. Decoding is lazy, but the
    /// decoder is opened here so the hand-off never underruns. A DSD `path`
    /// is appended as its decimated-PCM fallback source (used when the current
    /// session is already on the rodio path — e.g. DSD fallback → DSD).
    fn append_next(&mut self, path: &Path) -> Result<(), String> {
        use bitperfect::state::TransformDescription;
        let source = self.queued_source_for(path);
        let sink = self.sink.as_ref().ok_or("no sink")?;
        // The one sentence that is true of every track on this route,
        // whatever else is or is not known about it.
        const MIXER: &str = "the shared mixer: the volume control, the equaliser and \
                             ReplayGain are all in the path, and the samples are \
                             converted to float before the OS mixes them";
        if dsd::is_dsd_path(path) {
            let src = self.open_fallback_pcm(path)?;
            // The rate the decimator actually produced, not the one that was
            // asked for.
            let pcm_rate = src.out_rate();
            let tapped = SpectrumSource::new(
                DsdRodioSource(src), self.sample_buf.clone(), self.stereo_buf.clone());
            if let Some(ref eq) = self.eq {
                sink.append(EqSource::new(tapped.convert_samples::<f32>(), eq.clone()));
            } else {
                sink.append(tapped);
            }
            self.queued_next = Some(QueuedShared {
                path: path.to_path_buf(),
                source,
                transform: TransformDescription::DsdDecimated { pcm_rate },
                reason: format!("decimated to {pcm_rate} Hz PCM, then {MIXER}"),
            });
            return Ok(());
        }
        let file = File::open(path).map_err(|e| format!("Open failed: {e}"))?;
        let decoder = Decoder::new(BufReader::new(file))
            .map_err(|e| format!("Decode failed: {e}"))?;
        let tapped = SpectrumSource::new(decoder, self.sample_buf.clone(), self.stereo_buf.clone());
        if let Some(ref eq) = self.eq {
            sink.append(EqSource::new(tapped.convert_samples::<f32>(), eq.clone()));
        } else {
            sink.append(tapped);
        }
        // Held until the device crosses into it. The shared route has no
        // frame-exact boundary to report, so the queue is where the
        // successor's identity has to live: without it the panel went on
        // describing the previous file — its format, its transform, its
        // fidelity — for the whole of the next track.
        let reason = match &source {
            Some(_) => MIXER.to_string(),
            // Unknown is said out loud rather than left to be inferred from a
            // missing source.
            None => format!("{MIXER}; and this build cannot read the container, so what \
                             the file is could not be established"),
        };
        self.queued_next = Some(QueuedShared {
            path: path.to_path_buf(),
            source,
            transform: TransformDescription::Identity,
            reason,
        });
        Ok(())
    }

    /// What a file played through the shared mixer actually is.
    ///
    /// The exact route's answer if it has one for this file, otherwise the
    /// container's — and `None` if neither can say. `None` is a real answer:
    /// the panel has a word for an unknown source, and it is the truth.
    ///
    /// What it must never be is the *mixer's* format. `rodio` converts
    /// everything to 32-bit float, and the fallback arm here published
    /// exactly that, at the sink's channel count, for any track the exact
    /// route had not already prepared. A 24-bit FLAC the DAC had refused was
    /// then described to the listener as a float source: a specific claim
    /// about the recording, read off the playback path, and wrong.
    fn shared_source_for(&self, path: &Path) -> Option<bitperfect::state::MediaSource> {
        match self.last_prepared.as_ref() {
            Some((p, src)) if p == path => Some(bitperfect::state::MediaSource::pcm(*src)),
            _ => self.queued_source_for(path),
        }
    }

    /// What the next shared-route track actually is, as far as can be told
    /// without opening a second decoder for it.
    ///
    /// `prepare` reads the container's own header, which is the answer worth
    /// having. If it will not — a format symphonia declines, a file that has
    /// moved — the identity is left unknown rather than guessed, and the
    /// rollover below leaves the previous one alone rather than replacing a
    /// true description with a fabricated one.
    /// What the next shared-route track actually is, as far as can be told
    /// without opening a second decoder for it.
    ///
    /// A `MediaSource`, not a `SourceFormat`. The distinction is the whole
    /// point on this route: a DSD file played through the decimating fallback
    /// *is* a DSD source, and describing it by the PCM format the fallback
    /// produces would report the carrier as the recording. `prepare` cannot
    /// read a DSD container at all, so the previous version returned `None`
    /// for every one of them.
    fn queued_source_for(&self, path: &Path) -> Option<bitperfect::state::MediaSource> {
        if dsd::is_dsd_path(path) {
            return dsd::parse_file(path).ok().map(|i| {
                bitperfect::state::MediaSource::dsd(
                    bitperfect::format::SourceFormat {
                        kind: bitperfect::format::PcmKind::Integer { valid_bits: 1 },
                        sample_rate: i.sample_rate,
                        channels: i.channels as u16,
                        layout: bitperfect::format::ChannelLayout(i.layout.unwrap_or(0)),
                    },
                    i.rate_label(),
                )
            });
        }
        bitperfect::prepare(path, Duration::ZERO)
            .ok()
            .map(|p| bitperfect::state::MediaSource::pcm(p.source))
    }

    /// Promote the queued shared-route track to being the one playing.
    ///
    /// Called at the audible boundary, by the same rollover that moves the
    /// title and the spectrum.
    fn roll_shared_source(&mut self) {
        let Some(QueuedShared { path, source, transform, reason }) =
            self.queued_next.take()
        else {
            return;
        };
        // The track has changed whether or not its identity is known.
        self.note_current(&path, source.clone(), Duration::ZERO);

        match source {
            Some(media) => {
                // `last_prepared` is only ever consulted as a PCM identity,
                // so a DSD file on the fallback does not belong in it: the
                // carrier it is decimated to is not the recording.
                if media.dsd_label.is_none() {
                    self.last_prepared = Some((path, media.format));
                } else {
                    self.last_prepared = None;
                }
                self.bp_state.set_source(media);
            }
            // Unknown replaces the old truth rather than leaving it standing.
            //
            // Returning early here left the *previous* track's source, format
            // and transform on the panel for the whole of a track nothing
            // could describe — which is worse than saying nothing, because it
            // is a specific claim about the wrong file.
            None => {
                self.last_prepared = None;
                self.bp_state.clear_source();
            }
        }

        // The route did not change — it is the same sink — so playback stays
        // Processed across the rollover, in every case and without asking.
        //
        // It asked `fidelity_for`, which decides whether a *device* route
        // preserved the samples, and on this route the answer is known in
        // advance: it did not. A track whose channel layout could not be
        // checked came back `Unverified` — the badge that means nothing
        // Moosik did altered the audio — of a path that converts to float and
        // applies the volume slider, the equaliser and ReplayGain. A DSD file
        // on the decimating fallback came back `Identity`, of a path that had
        // just resampled it.
        //
        // The transform and the reason were decided in `append_next`, by the
        // code that knew which of the two it had appended, and they are
        // applied here unchanged.
        let fidelity = bitperfect::state::Fidelity::Processed {
            transform: transform.clone(),
            reason,
        };
        let mut processing = self.shared_processing();
        // Both records, not one. `Fidelity` carries the transform for the
        // badge and `ProcessingState` carries it for the panel's own account
        // of what is in the path — and the rollover set only the first, so a
        // DSD file that had just been decimated onto the mixer described its
        // processing as an identity transform on the surface that exists to
        // list what is being done to the audio.
        processing.transform = transform;
        self.bp_state.roll_over(fidelity, processing);
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
    /// Seek, and report where playback actually ended up.
    ///
    /// It returned `Ok(())` and the caller then moved the spectrum and the
    /// seek bar to the position it had *asked* for. On the bit-perfect path
    /// that is already known to be wrong — `Prepared::seeked_to` is the frame
    /// the container actually reached, and a FLAC without a seek table can
    /// land seconds away — so the display and the audio disagreed with nothing
    /// anywhere reporting it.
    fn seek_to(&mut self, path: &Path, target: Duration) -> Result<Duration, String> {
        if self.on_bp_stream() {
            // The pause intent is read *before* the reopen and handed to it,
            // rather than read before and re-applied after. Re-applying it
            // afterwards is a pause that arrives one statement — one buffer —
            // late, which on an exclusive device is audible: a listener who
            // paused and then dragged the seek bar heard the new position
            // start playing before it stopped again.
            let opening = Opening {
                at: target,
                paused: if self.dsd_native {
                    self.native_is_paused()
                } else {
                    self.bp.as_ref().map(|bp| bp.is_paused()).unwrap_or(false)
                },
            };
            // All three device routes reopen the output, so all three can
            // fail with a handle already taken — and all three used to return
            // the error and leave it taken, with `bp_state` still on the
            // "verifying" transport the reopen had set.
            //
            // Each keeps its own kind of failure. A native backend that will
            // not open is a backend failure; DoP and PCM already carry typed
            // reasons of their own and are not rewritten here.
            let result = self.open_attempt(
                |e| {
                    if e.dsd_native {
                        e.start_native(path, opening)
                            .map_err(bitperfect::OpenError::backend)
                    } else if e.dsd_mode {
                        e.start_dop(path, opening)
                    } else {
                        e.start_bp(path, opening)
                    }
                },
                |e| e.to_reason(),
            );
            // Returned rather than logged and swallowed: the callers update
            // the spectrum and the seek bar, and used to do so even when the
            // decoder never moved.
            if let Err(e) = result {
                let why = e.to_string();
                crate::mlog!("seek: {why}");
                return Err(why);
            }
            if opening.paused {
                self.started_at = None;
            } else {
                self.started_at = Some(Instant::now());
            }
            // `start_bp`/`start_dop` set `bp_base` from the container's own
            // answer. Overwriting it with the request here is what put the
            // requested position on the seek bar while the decoder sat
            // somewhere else.
            let landed = self.bp_base;
            self.paused_elapsed = landed;
            return Ok(landed);
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
            // `Sink::try_seek` reports success and nothing else — there is no
            // actual position to be had from it. The request is the only
            // answer available on this path, and saying so here is the point:
            // it is not evidence of where the decoder is, it is the absence of
            // any.
            return Ok(target);
        }

        // --- DSD fallback: rebuild in place (byte-addressable, instant) ---
        // Normally unreachable — try_seek forwards to the DSD source and the
        // fast path succeeds — but if it ever fails, the rodio-Decoder slow
        // path below can't open a DSD file, so rebuild the sink directly.
        if self.dsd_fallback {
            let from = self.elapsed();
            self.release_sink();
            match self.open_attempt(
                |e| {
                    e.start_dsd_fallback(
                        path,
                        Opening {
                            at: target,
                            paused: was_paused,
                        },
                    )
                },
                // A rebuild can fail for reasons that are not the seek's: the
                // file has gone, the decimator will not open it, the mixer has
                // no output. Mapping all of them to `Seek` told the listener
                // the container refused a position when the file was missing.
                |e| dsd_rebuild_reason(e),
            ) {
                Ok(()) => {
                    self.paused_elapsed = target;
                    if was_paused {
                        if let Some(ref s) = self.sink { s.pause(); }
                        self.started_at = None;
                    } else {
                        self.started_at = Some(Instant::now());
                    }
                }
                // Reported as a failure, because it is one.
                //
                // This returned `Ok(self.elapsed())` — a rebuild that could
                // not happen, described to the caller as a seek that landed
                // wherever playback had stopped. The seek bar moved there, the
                // spectrum followed, and the only sign that nothing was
                // playing was that nothing was playing. `open_attempt` above
                // has already released everything; what is left is to say so.
                Err(e) => {
                    let _ = from;
                    return Err(e);
                }
            }
            return Ok(target);
        }

        // --- Slow path: reopen + skip samples, ON A BACKGROUND THREAD ---
        // FLAC seeks decode-and-discard every sample up to `target`; deep into a
        // hi-res file that is tens of millions of samples. Doing it here would
        // freeze the UI for seconds ("not responding"), so we hand it to a
        // worker and install the resulting decoder when it's ready (poll_pending_seek).
        let from = self.elapsed();
        self.release_sink();
        // drops any prior receiver
        let cancel = self.new_seek_cancel();
        let rx = match Self::spawn_seek_worker(path, target, cancel) {
            Ok(rx) => rx,
            Err(e) => {
                // Same correction as the rebuild above: a seek that never
                // started is not a seek that landed. `abandon_seek` puts the
                // engine back where playback actually was — nothing was taken
                // here, so there is nothing to release — and the caller is
                // told.
                let why = e.describe();
                self.abandon_seek(from, e);
                return Err(why);
            }
        };

        // Reflect the target position immediately; time is frozen until the
        // seek lands (started_at = None), and is_finished() returns false while
        // a seek is pending, so no spurious auto-advance. `from` is kept so
        // that a seek which does not land can put the engine back.
        self.pending_seek = Some(PendingSeek {
            rx,
            was_paused,
            path: path.to_path_buf(),
            from,
            // The display below is moved to the request, so the request is
            // what is provisional until something lands.
            provisional: Some(target),
        });
        self.paused_elapsed = target;
        self.started_at = None;
        // Nothing has landed yet — the worker is still decoding. The caller
        // gets the target as a provisional position and `SeekOutcome::Landed`
        // corrects it, which is the one path that knows.
        Ok(target)
    }

    /// Put the engine back somewhere coherent after a seek that will not land,
    /// and park the reason for the tick to collect.
    ///
    /// Every failing seek path used to stop here in a different half-state: no
    /// sink, the clock frozen, and the position showing wherever the slider
    /// had been dragged to — a place nothing was playing from and nothing ever
    /// would. The listener saw silence and a moved slider, which is what a
    /// successful seek into a quiet passage looks like.
    fn abandon_seek(&mut self, from: Duration, why: SeekError) {
        self.cancel_seek_worker();
        self.pending_seek = None;
        // The track is still the track. Only the position moves back — to
        // where playback actually was, which is what `resume` will reopen at.
        if let Some(c) = self.current.as_mut() {
            c.at = from;
        }
        self.release_sink();
        // Back where playback actually was, stopped, with the clock consistent
        // with that position.
        self.paused_elapsed = from;
        self.started_at = None;
        // And the shared route is no longer running anything. Leaving
        // `bp_state` describing an open shared session was the last piece of
        // the half-state: the panel went on naming a transport, a fidelity and
        // a source for audio that had been stopped, so every surface agreed
        // that a track was playing except the speakers.
        if !self.on_bp_stream() && !self.on_native_stream() {
            // Stopped, not faulted: nothing is playing, and the reason travels
            // separately to the tick that reports it. Saying so retires the
            // generation too, so a scan or a fault still in flight from the
            // session this seek destroyed cannot land on whatever comes next.
            self.bp_state.stopped();
        } else if matches!(self.bp_state.transport, bitperfect::Transport::Opening) {
            // A reopen that got as far as `begin_open` and then failed.
            //
            // It stayed `Opening` — "Verifying output…" — for the rest of the
            // session, because the handle it had been about to replace was
            // still there and the branch above only fires when nothing is
            // held. Nothing was ever going to arrive to replace it. The typed
            // reason from the seek is published instead, so the panel says
            // what happened rather than that it is still thinking about it.
            self.bp_state.failed(why.to_reason());
        }
        self.seek_failure = Some(why);
    }

    /// Spawn the background decode-and-skip worker for a normal-mode seek to
    /// `target`: open the file, build a rodio decoder, discard samples up to
    /// `target`, and hand the positioned decoder back over the channel. Every
    /// bit of decode cost lives on this thread, never the UI thread.
    fn spawn_seek_worker(
        path: &Path,
        target: Duration,
        cancel: std::sync::Arc<std::sync::atomic::AtomicBool>,
    ) -> Result<std::sync::mpsc::Receiver<Result<SeekLanding, SeekError>>, SeekError> {
        let (tx, rx) = std::sync::mpsc::channel();
        let path_c = path.to_path_buf();
        std::thread::Builder::new()
            .name("rodio-seek".into())
            .spawn(move || {
                let seeked = (|| -> Result<SeekLanding, SeekError> {
                    let file = File::open(&path_c)
                        .map_err(|e| SeekError::Open(e.to_string()))?;
                    let mut decoder = Decoder::new(BufReader::new(file))
                        .map_err(|e| SeekError::Decode(e.to_string()))?;
                    let rate = decoder.sample_rate();
                    let channels = decoder.channels();
                    let ch = channels.max(1) as u64;
                    // Whole frames, and the frame count first.
                    //
                    // The skip was computed straight in samples —
                    // `target * rate * channels`, truncated — so on stereo it
                    // was as likely as not to be odd. An odd number of samples
                    // discarded leaves the decoder standing on a right-channel
                    // sample, and every frame after the seek is assembled from
                    // the right of one frame and the left of the next: the
                    // channels swap, for the rest of the track, silently.
                    let frames = (target.as_secs_f64() * rate.max(1) as f64).floor() as u64;
                    let samples = frames * ch;
                    let mut n = 0u64;
                    while n < samples {
                        // Checked inside the discard loop, because that loop is
                        // the whole cost: a deep seek into a hi-res FLAC decodes
                        // and throws away tens of millions of samples, and a
                        // worker whose seek has already been superseded went on
                        // doing it — a core, for seconds, for a position nobody
                        // is waiting for. Every 4096 samples, so the atomic is
                        // not in the inner cost.
                        if n.is_multiple_of(4096)
                            && cancel.load(std::sync::atomic::Ordering::Relaxed)
                        {
                            return Err(SeekError::Cancelled);
                        }
                        if decoder.next().is_none() {
                            // The file ended before the target did. Returning
                            // the decoder anyway handed back something sitting
                            // at EOF while the caller published the position it
                            // had asked for.
                            //
                            // Reported at the last *whole* frame: a file that
                            // ends mid-frame has no position at the fragment.
                            let landed = frame_time(n / ch, rate);
                            return Err(SeekError::PastEnd { landed });
                        }
                        n += 1;
                    }
                    Ok(SeekLanding {
                        // Every sample the worker discarded is a sample the
                        // decoder is now past, so this is where it is — not
                        // where it was asked to be.
                        landed: frame_time(frames, rate),
                        source: bitperfect::format::SourceFormat {
                            // `rodio` hands back `f32` whatever the container
                            // held; that is what this decoder produces, and
                            // claiming the container's own format for it would
                            // describe a route that does not exist.
                            kind: bitperfect::format::PcmKind::Float32,
                            sample_rate: rate,
                            channels,
                            layout: bitperfect::format::ChannelLayout::UNSPECIFIED,
                        },
                        decoder,
                    })
                })();
                let _ = tx.send(seeked);
            })
            // `.ok()` discarded this, and a thread that never started left a
            // receiver that disconnects — reported, eventually, as a worker
            // that "ended without a result", which is not what happened.
            .map_err(|e| SeekError::Spawn(e.to_string()))?;
        Ok(rx)
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
        // DSD can't go through the rodio-Decoder seek worker; its fallback
        // source seeks instantly, so build it synchronously at the position.
        if dsd::is_dsd_path(path) {
            // Typed like every other seek. The failure used to be discarded by
            // an `if ... .is_ok()` with no `else`: the player went silent at
            // the old position with nothing said and nothing published, which
            // is indistinguishable from a seek that worked into silence.
            // The pause intent goes *into* the rebuild: a sink is paused
            // before its source is appended or not at all, because appending
            // is when it starts.
            match self.start_dsd_fallback(
                path,
                Opening {
                    at: target,
                    paused: was_paused,
                },
            ) {
                Ok(()) => {
                    self.dsd_fallback = true;
                    self.paused_elapsed = target;
                    if was_paused {
                        self.started_at = None;
                    } else {
                        self.started_at = Some(Instant::now());
                    }
                    // Published only now, on the path that actually installed
                    // a sink.
                    self.bp_note_processed(
                        None,
                        bitperfect::state::TransformDescription::DsdDecimated {
                            pcm_rate: self.last_sample_rate,
                        },
                        "the DSD route was unavailable for this seek");
                }
                Err(e) => self.abandon_seek(Duration::ZERO, SeekError::Dsd(e)),
            }
            return;
        }
        let cancel = self.new_seek_cancel();
        let rx = match Self::spawn_seek_worker(path, target, cancel) {
            Ok(rx) => rx,
            Err(e) => {
                self.abandon_seek(Duration::ZERO, e);
                return;
            }
        };
        self.pending_seek = Some(PendingSeek {
            rx,
            was_paused,
            path: path.to_path_buf(),
            // `stop()` above already released the previous track, so there is
            // no earlier position to return to: a failure here leaves the
            // engine stopped at the beginning, which is coherent.
            from: Duration::ZERO,
            provisional: Some(target),
        });
        self.paused_elapsed = target;
        self.started_at = None;
    }

    /// True while a background seek is decoding — the UI must keep ticking so
    /// poll_pending_seek() runs and installs the result.
    fn is_seeking(&self) -> bool { self.pending_seek.is_some() }

    /// Record what is playing, on whatever route.
    ///
    /// Called by every start. There is one of these rather than a flag per
    /// route because "which file is playing" is one question, and answering it
    /// from `last_prepared` — which only some routes write — is how `resume`
    /// came to reopen a track the listener had moved on from.
    fn note_current(
        &mut self,
        path: &Path,
        source: Option<bitperfect::state::MediaSource>,
        at: Duration,
    ) {
        self.current = Some(CurrentTrack {
            path: path.to_path_buf(),
            source,
            at,
        });
    }

    /// The track that is playing, if any.
    fn current_track(&self) -> Option<&CurrentTrack> {
        self.current.as_ref()
    }

    /// What `resume` would reopen, when there is nothing left to un-pause.
    ///
    /// A function rather than an expression inside `resume` because it is the
    /// decision worth testing and `resume` itself ends in
    /// `play_seeked_async`, which needs an output device. This is the part
    /// that was wrong: it read `last_prepared`, which the exact route and a
    /// completed seek write and nothing else does, so on a shared track it
    /// named whichever file last took the exact route.
    fn resume_target(&self) -> Option<(PathBuf, Duration)> {
        self.current
            .as_ref()
            .map(|c| (c.path.clone(), c.at))
    }

    /// Poll the background rodio seek; when the seeked decoder arrives, wire it
    /// into a fresh sink. Cheap — call every frame.
    ///
    /// Returns what happened, so the caller can say so. Every failure used to
    /// be a bare `return` or a silent `self.pending_seek = None`: a device
    /// that would not open left the pending seek in place and the poll retried
    /// it every frame forever, and a container that refused the seek cleared
    /// it with no message and no state — the player simply stopped, mid-track,
    /// with the old position still on the slider.
    ///
    /// Session state is published **only** on the path that actually installed
    /// a sink. Publishing it earlier described a shared, processed route that
    /// did not exist yet.
    fn poll_pending_seek(&mut self) -> SeekOutcome {
        // A synchronous failure parked by `play_seeked_async` is reported
        // through the same channel as an asynchronous one, so the caller has
        // one thing to handle rather than two.
        if let Some(why) = self.seek_failure.take() {
            return SeekOutcome::Failed(why);
        }
        let Some(ps) = self.pending_seek.as_ref() else {
            return SeekOutcome::Idle;
        };
        let provisional = ps.provisional;
        let received = match ps.rx.try_recv() {
            Ok(r) => r,
            Err(std::sync::mpsc::TryRecvError::Empty) => {
                return match provisional {
                    Some(at) => SeekOutcome::Provisional(at),
                    None => SeekOutcome::Working,
                };
            }
            Err(std::sync::mpsc::TryRecvError::Disconnected) => Err(SeekError::WorkerLost),
        };
        let PendingSeek { was_paused, path, from, .. } = self.pending_seek.take().unwrap();
        let landing = match received {
            Ok(l) => l,
            Err(e) => {
                self.abandon_seek(from, e.clone());
                // Taken straight back out: the tick is asking now, and parking
                // it would report the same failure a frame later as well.
                self.seek_failure = None;
                return SeekOutcome::Failed(e);
            }
        };

        // Nothing below is committed to the engine until the sink exists. The
        // device is the last thing that can refuse, so it is the last thing
        // asked — and if it refuses, the engine goes back rather than sitting
        // at a position with no audio behind it.
        #[cfg(test)]
        let idle = self.idle_output.then(|| Sink::new_idle().0);
        #[cfg(not(test))]
        let idle: Option<Sink> = None;
        let sink = match idle {
            Some(s) => s,
            None => {
                let handle = match self.rodio_handle() {
                    Ok(h) => h,
                    Err(e) => {
                        self.abandon_seek(from, SeekError::Output(e.clone()));
                        self.seek_failure = None;
                        return SeekOutcome::Failed(SeekError::Output(e));
                    }
                };
                match Sink::try_new(&handle) {
                    Ok(s) => s,
                    Err(e) => {
                        let e = SeekError::Output(e.to_string());
                        self.abandon_seek(from, e.clone());
                        self.seek_failure = None;
                        return SeekOutcome::Failed(e);
                    }
                }
            }
        };
        let SeekLanding { decoder, landed, source } = landing;
        // What the *file* is, not what `rodio` hands back.
        //
        // The worker reports `Float32` because that is what its decoder
        // produces, whatever the container held — so publishing it described
        // every seeked track as a 32-bit float source. A 16-bit FLAC became a
        // float file by being seeked within.
        //
        // `current` is asked first because it is the authoritative record of
        // what is playing and is written by every route; `last_prepared` only
        // by the exact one. The worker's own answer is the last resort, for a
        // file nothing has been able to describe.
        let known_media = self
            .current_track()
            .filter(|c| c.path == path)
            .and_then(|c| c.source.clone());
        let source = match (&known_media, self.last_prepared.as_ref()) {
            (Some(m), _) => m.format,
            (None, Some((known, prepared))) if *known == path => *prepared,
            _ => source,
        };
        let media = known_media
            .unwrap_or_else(|| bitperfect::state::MediaSource::pcm(source));
        // Before the append, because appending is when a `rodio` sink starts.
        //
        // This paused *after* it, which is a buffer of the new position played
        // to a listener who had paused before dragging the bar — and on this
        // path the buffer is whatever `rodio` had already pulled, so the
        // amount is not even bounded by anything the player chooses.
        if was_paused {
            sink.pause();
        }
        sink.set_volume(self.rodio_volume());
        let tapped = SpectrumSource::new(
            decoder,
            self.sample_buf.clone(),
            self.stereo_buf.clone(),
        );
        if let Some(ref eq) = self.eq {
            sink.append(EqSource::new(tapped.convert_samples::<f32>(), eq.clone()));
        } else {
            sink.append(tapped);
        }
        if was_paused {
            self.started_at = None;
        } else {
            self.started_at = Some(Instant::now());
        }
        // Where the decoder is, not where the slider was dragged to.
        self.paused_elapsed = landed;
        self.last_sample_rate = source.sample_rate;
        self.sink = Some(sink);
        self.last_prepared = Some((path.clone(), source));
        self.note_current(&path, Some(media.clone()), landed);

        // Now, and not before: there is a shared route, and it is processed.
        self.bp_note_processed(
            Some(media),
            bitperfect::state::TransformDescription::Identity,
            "seeked through the shared mixer",
        );
        SeekOutcome::Landed(landed)
    }
}

// ---------------------------------------------------------------------------
// App state
// ---------------------------------------------------------------------------

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
enum PlayState { Stopped, Playing, Paused }

#[derive(PartialEq, Clone, Copy, Serialize, Deserialize)]
enum LoopMode { Sequential, RepeatAll, RepeatOne }

/// How far short of its declared length a shared track may end before the
/// shortfall is treated as a decode failure rather than as the end of the
/// track.
///
/// Generous on purpose. Durations come from tags and container headers and are
/// routinely a little wrong; what this is for is a decode that stopped in the
/// middle, not the last fraction of a second.
const SHARED_SHORTFALL_TOLERANCE: Duration = Duration::from_secs(3);

/// Whether the value-exact label may be applied.
///
/// Four conditions, and the fourth is not about the conversion at all. Pulled
/// out because it is the rule that matters and it was welded to an open
/// `BpStream`, which needs a device: the interesting cases could not be
/// written down.
///
/// * `scan_says` — a complete decode of the track found every sample on the
///   Q1.31 lattice. Can be stale: the file may have been rewritten under it.
/// * `observed_clean` — the running conversion has not had to round one. Can
///   only speak for what has played.
/// * `on_the_route` — this is the integer conversion. On any other route the
///   question does not arise.
/// * `already_narrowed` — the source is 64-bit float, so its numbers were
///   changed before either of the first two saw them. Value-exact means "the
///   representation changed and no number did"; of a source narrowed to `f32`
///   in the decoder that is false however clean the lattice looks afterwards.
fn value_exact_allowed(
    scan_says: bool,
    observed_clean: bool,
    on_the_route: bool,
    already_narrowed: bool,
) -> bool {
    scan_says && observed_clean && on_the_route && !already_narrowed
}

/// How a shared-mixer track ended, from the only three things anyone can see:
/// whether the sink has run dry, whether it ever started, and how much played.
///
/// Separated from `Engine::completion` because the interesting part is the
/// rule, and the rule is untestable while it is welded to a `rodio::Sink` —
/// which needs an output device that a test machine may not have. Everything
/// here is a value.
///
/// The failure this returns is `SHARED_ENDED_EARLY`, which is what was
/// observed. It was `SOURCE_READ` — a claim that a read error occurred, which
/// nothing had seen and which may well be false: a truncated file, a decoder
/// that gave up on a bad frame and a container whose declared duration is
/// simply wrong all look identical from here. Only the last of those is not a
/// failure, and the tolerance is what covers it.
/// What happened to an attempt to open a track.
#[derive(Debug, PartialEq)]
enum OpenOutcome {
    /// The requested route worked, and landed here.
    Exact(Duration),
    /// This track — and only this track — fell back to the shared mixer, and
    /// landed here.
    Shared(Duration),
    /// The fallback was permitted and itself failed.
    SharedFailed(String),
    /// Nothing is playing, and this is why.
    Stopped(bitperfect::state::FailureReason),
}

/// What a restart actually achieved.
///
/// Every caller — the bit-perfect toggle, the output-device picker, the
/// ASIO/ALSA picker, and the rollback one of them performs — gets one of these
/// and applies it the same way. They used to each interpret a
/// `Result<(), String>`: some set the play state, some did not; one discarded
/// the error entirely; none of them agreed about the position, the pause
/// state, or who owned the status line.
#[derive(Debug, PartialEq)]
enum RestartOutcome {
    /// The route the track asked for, at the position it reached.
    Exact { landed: Duration },
    /// This track alone fell back to the shared mixer, at the position it
    /// reached. The preference is untouched.
    Shared { landed: Duration },
    /// Nothing is playing, nothing is held, and this is why.
    Stopped(bitperfect::state::FailureReason),
}

/// Take a track as one transaction: open, then reach the position, or undo
/// both.
///
/// Every route into playback goes through here — starting a track from the
/// playlist, restarting one after a toggle or a device change — because the
/// two differ only in the position they are aiming at, and everything they
/// used to differ in was a divergence rather than a decision. `play_index` had
/// the policy ladder written out inline and released nothing when it failed;
/// the restart had the ladder in a reducer and released everything. They now
/// have one of each.
///
/// The two halves were separate. `play_file` starts audible at position zero
/// and `seek_to` moves it; a seek that failed left the `?` operator to return
/// an error with the track playing from the beginning and an exclusive handle
/// still held — a device no other application can take, and a track the
/// listener never asked to hear from the start. Either the restart reaches the
/// position it is restarting to, or it leaves nothing behind.
///
/// There is no second half. The position is an argument to the open, not an
/// operation performed on a route that is already sounding — which is what it
/// was, and what made a restart to thirty seconds emit the first fraction of a
/// second of the track before it got there.
///
/// `open` and `open_shared` are injected and both take the target. Production
/// passes the engine's own `play_file_at` and `play_file_shared_at`; a test
/// passes closures, which is the only way to drive an orchestration whose
/// every real step needs a sound card — and the closures see the position they
/// were asked to open at, so "it opened there" is something a test can hold.
fn open_track(
    engine: &mut Engine,
    opening: Opening,
    is_dsd: bool,
    open: impl FnOnce(&mut Engine, Opening) -> Result<Duration, bitperfect::OpenError>,
    open_shared: impl FnOnce(&mut Engine, Opening) -> Result<Duration, String>,
) -> RestartOutcome {
    let policy = engine.bp_state.policy;
    // The exact attempt, as a transaction: if it fails it has already given
    // back the endpoint, so the shared retry below opens a mixer on a device
    // nothing is holding rather than on one this same call still owns.
    let opened = engine.open_attempt(|e| open(e, opening), |e| e.to_reason());
    let outcome = resolve_open(opened, policy, is_dsd, || open_shared(engine, opening));
    match outcome {
        OpenOutcome::Exact(landed) => RestartOutcome::Exact { landed },
        OpenOutcome::Shared(landed) => RestartOutcome::Shared { landed },
        OpenOutcome::SharedFailed(why) => {
            // The mixer's own kind of failure, not `DeviceUnavailable` for all
            // of them: a missing file and a missing DAC are different things to
            // be told.
            let reason = shared_failure_reason(&why);
            engine.abort_open(reason.clone());
            RestartOutcome::Stopped(reason)
        }
        // Already released by the transaction above; what is left is to say
        // what happened.
        OpenOutcome::Stopped(reason) => RestartOutcome::Stopped(reason),
    }
}

/// How a route is asked to come up.
///
/// Two facts, carried together from the moment a restart is captured to the
/// moment a device is opened, because they are decided together and applying
/// one without the other is what went wrong. A listener who paused, changed
/// output device, and heard the track start again did not ask for that.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
struct Opening {
    /// Where the route must be before it can sound.
    at: Duration,
    /// Whether it must come up *already* paused.
    ///
    /// Not "pause it afterwards". A route that is started and paused a
    /// statement later has already emitted, and a statement is milliseconds of
    /// audio; on an exclusive device those milliseconds are the DAC's first
    /// sound after a silent gap. Nothing about the final state records the
    /// difference, which is why it survived: an engine that played for a
    /// moment and then paused looks exactly like one that never played.
    paused: bool,
}

impl Opening {
    fn at(at: Duration) -> Self {
        Opening { at, paused: false }
    }
}

/// The steps of bringing a route up, in the order that matters.
///
/// A trait rather than three inline calls because the ordering is the whole
/// property and the ordering is invisible afterwards. Production implements it
/// over a `BpStream`, a `rodio` sink, or a native session; a test implements it
/// over a recorder, and asserts that nothing which can produce sound happens
/// before the pause.
trait RouteBringUp {
    /// Take the route's pause flag before anything can emit.
    fn pause(&mut self);
    /// Release it.
    fn resume(&mut self);
    /// The step after which the route can produce sound.
    fn begin_output(&mut self) -> Result<(), bitperfect::OpenError>;
}

/// Bring `route` up, paused if it was asked to be.
///
/// Every production opener goes through here, so "paused means no sample is
/// emitted" is a property of one decision rather than of six call sites that
/// each had to remember it — and five of them did not.
fn bring_up<R: RouteBringUp>(route: &mut R, paused: bool) -> Result<(), bitperfect::OpenError> {
    if paused {
        route.pause();
    } else {
        route.resume();
    }
    route.begin_output()
}

/// The pause flag of an open route, wherever that route's stream keeps it.
///
/// Three unrelated stream types own the same flag and read it in the same
/// place — the callback, before it decides whether to emit the payload or
/// silence — so the flag is the thing worth abstracting over, and the call
/// that starts the output is not: it takes different arguments on every route
/// and returns different things.
trait RoutePause {
    fn pause(&self);
    fn resume(&self);
}

impl RoutePause for bitperfect::BpStream {
    fn pause(&self) {
        bitperfect::BpStream::pause(self);
    }
    fn resume(&self) {
        bitperfect::BpStream::resume(self);
    }
}

#[cfg(all(windows, feature = "asio-dsd"))]
impl RoutePause for bitperfect::asio_dsd::AsioDsdStream {
    fn pause(&self) {
        bitperfect::asio_dsd::AsioDsdStream::pause(self);
    }
    fn resume(&self) {
        bitperfect::asio_dsd::AsioDsdStream::resume(self);
    }
}

#[cfg(all(target_os = "linux", feature = "alsa-dsd"))]
impl RoutePause for bitperfect::alsa_dsd::AlsaDsdStream {
    fn pause(&self) {
        bitperfect::alsa_dsd::AlsaDsdStream::pause(self);
    }
    fn resume(&self) {
        bitperfect::alsa_dsd::AlsaDsdStream::resume(self);
    }
}

/// An open route, plus the single call after which it can put a payload on
/// the wire.
///
/// One adapter for every device route rather than one per route: the ordering
/// is the property, the ordering is the same everywhere, and a second copy of
/// it is exactly how the exact-output route ended up being the only one that
/// had it right. `start` is a `FnOnce` because the starting call is all these
/// routes do not share — `start`, `start_dop` and `start_session` take
/// different payloads and return different types — and because it must
/// consume the payload it hands over.
struct StreamBringUp<'a, T: RoutePause, F> {
    route: &'a T,
    /// Taken by `begin_output`, which is the step that can emit.
    start: Option<F>,
}

impl<T, F> RouteBringUp for StreamBringUp<'_, T, F>
where
    T: RoutePause,
    F: FnOnce() -> Result<(), bitperfect::OpenError>,
{
    fn pause(&mut self) {
        self.route.pause();
    }
    fn resume(&mut self) {
        self.route.resume();
    }
    fn begin_output(&mut self) -> Result<(), bitperfect::OpenError> {
        (self.start.take().expect("begin_output runs once"))()
    }
}

/// What kind of failure the shared mixer just had.
///
/// The shared route reports strings, and every one of them used to arrive at
/// the caller as a backend error or as `DeviceUnavailable` — so a file that had
/// been deleted, a container `rodio` cannot decode, and a missing output device
/// were all shown to the listener as the DAC being gone. The class is what the
/// panel names and what decides whether anything may be retried, so it is worth
/// keeping.
fn shared_open_error(why: &str) -> bitperfect::OpenError {
    let w = why.to_ascii_lowercase();
    if w.contains("open failed") || w.contains("decode failed") {
        // The file, not the output: it is missing, unreadable, or in a
        // container `rodio` has no decoder for.
        bitperfect::OpenError::source(why.to_string())
    } else if w.contains("will not seek") {
        bitperfect::OpenError::seek(why.to_string())
    } else {
        // A sink that will not build, or no output device at all.
        bitperfect::OpenError::backend(why.to_string())
    }
}

/// The same distinction, for the rung that has already fallen back.
fn shared_failure_reason(why: &str) -> bitperfect::state::FailureReason {
    shared_open_error(why).to_reason()
}

/// What the status line says after the bit-perfect toggle.
///
/// A reducer because the caller got the condition wrong and the condition is
/// the whole of it: it asked whether a track was *selected*, not whether one
/// was *playing*. A stopped player still has a current index — that is how
/// Play knows what to start — so turning bit-perfect on while stopped printed
/// "Playing: …", owned as the now-playing line, for a player that was not
/// playing. The next tick then refused to refresh it, correctly, because a
/// stopped player's line is not the app's to regenerate; so the false sentence
/// stayed until something else replaced it.
#[derive(Clone, PartialEq, Debug)]
enum ToggleStatus {
    /// The now-playing line for this track, owned so the route may still
    /// change it as the session settles.
    NowPlaying(usize),
    /// Something the listener has just been told.
    Transient(String),
}

fn toggle_status(on: bool, play_state: PlayState, current: Option<usize>) -> ToggleStatus {
    // A selected index is not an active session.
    let playing = match play_state {
        PlayState::Stopped => None,
        PlayState::Playing | PlayState::Paused => current,
    };
    match (on, playing) {
        (true, Some(i)) => ToggleStatus::NowPlaying(i),
        (true, None) => {
            ToggleStatus::Transient("◇ Bit-perfect requested — no active output".into())
        }
        (false, _) => ToggleStatus::Transient("Bit-perfect off".into()),
    }
}

/// What a completion means to a session that is paused.
///
/// The paused branch of the tick exists to notice failures, not endings: a
/// paused session still holds an exclusive device, and a driver reset while
/// paused left the player sitting on it until the listener pressed play. A
/// paused track that reached the end of its source has not ended as far as the
/// listener is concerned, and advancing the playlist while they are away from
/// the keyboard would be the same defect pointing the other way.
#[derive(Clone, Copy, PartialEq, Debug)]
enum PausedTick {
    /// Stay where we are.
    Nothing,
    /// The session failed. Halt once, release the device, say why.
    Halt(u8),
}

fn paused_completion(c: bitperfect::Completion) -> PausedTick {
    match c.failure() {
        Some(reason) => PausedTick::Halt(reason),
        // Running, or a clean end. Neither advances anything from here.
        None => PausedTick::Nothing,
    }
}

/// Where an asynchronous seek that cannot land leaves the player.
///
/// One answer, written down once, because there were two and they disagreed.
/// **Paused, at the position playback actually reached.** The engine's
/// `abandon_seek` keeps the current track and puts the position back to
/// `from`, so the track can be resumed from where it was; nothing is running,
/// so nothing claims to be playing; and the status line says the seek failed.
///
/// Fully stopping would also have been coherent — it is what the synchronous
/// device seek does, because that one tears the device down and clears the
/// current track — but it is not what this path's engine state supports, and
/// having the two paths differ silently is what this constant exists to stop.
fn async_seek_failure_state() -> PlayState {
    PlayState::Paused
}

/// Why a decimated-DSD rebuild failed, kept as the kind of failure it was.
///
/// Every one of them used to arrive as `Seek`, so a file that had been deleted
/// under the player was reported as a container refusing a position. The
/// message is the rebuild's own either way; what changes is the class the
/// panel and the fallback ladder read.
fn dsd_rebuild_reason(why: &str) -> bitperfect::state::FailureReason {
    use bitperfect::state::FailureReason;
    let w = why.to_ascii_lowercase();
    if w.contains("dsd seek") {
        FailureReason::Seek(why.to_string())
    } else if w.contains("decimate") || w.contains("open failed") || w.contains("dsd:") {
        FailureReason::SourceOpen(why.to_string())
    } else if w.contains("sink failed") || w.contains("no output") || w.contains("output device") {
        FailureReason::DeviceUnavailable(why.to_string())
    } else {
        FailureReason::Decode(why.to_string())
    }
}

/// Everything a restart needs to know, taken before anything is torn down.
///
/// A restart reads live state — which track, where in it, whether it was
/// paused — and a rollback is a restart that runs *after* an attempt has
/// already changed all three. Reading it again at that point reads the
/// wreckage of the attempt rather than the state being restored.
#[derive(Clone, Debug, PartialEq)]
struct RestartRequest {
    index: usize,
    path: PathBuf,
    duration: Option<Duration>,
    /// Where playback was. A rollback that restores the track and not the
    /// position has put the listener back at the beginning of it.
    target: Duration,
    was_paused: bool,
}

/// What happened when playback was moved to another output.
#[derive(Debug, PartialEq)]
enum SwitchOutcome {
    /// The new device took the track.
    Moved,
    /// It did not, and the previous one took it back — at the position and
    /// pause state captured before the attempt.
    RolledBack { why: String },
    /// Neither will play it. Both reasons: the first says why the move
    /// failed, the second says why the listener is now looking at a stopped
    /// player, and either alone is an incomplete account of what happened.
    Stranded { why: String, back: String },
}

/// Move playback to another output, and put it back if it will not go.
///
/// The sequence, with both restarts injected, because both of them reach a
/// sound card and neither of them is the part that was wrong. What was wrong
/// was the shape: the rollback's error was discarded with `let _ =`, and the
/// rollback itself re-read state the failed attempt had already changed.
fn switch_output<C>(
    ctx: &mut C,
    snapshot: Option<&RestartRequest>,
    restart_new: impl FnOnce(&mut C) -> Result<(), String>,
    roll_back: impl FnOnce(&mut C, &RestartRequest) -> Result<(), String>,
) -> SwitchOutcome {
    let Err(why) = restart_new(ctx) else {
        return SwitchOutcome::Moved;
    };
    // Nothing was playing, so there is nothing to put back — but the move
    // still failed, and saying so is the whole of what is left to do.
    let Some(req) = snapshot else {
        return SwitchOutcome::RolledBack { why };
    };
    match roll_back(ctx, req) {
        Ok(()) => SwitchOutcome::RolledBack { why },
        Err(back) => SwitchOutcome::Stranded { why, back },
    }
}

/// Everything the app itself has to do about a restart outcome.
///
/// Separated from doing it so that "what every caller gets" is one value that
/// can be looked at, rather than a sequence of assignments each caller was
/// free to perform differently — which is what they did.
#[derive(Debug, PartialEq)]
struct RestartEffect {
    play_state: PlayState,
    /// Where to move the spectrum and the seek bar, if anywhere.
    seek_to: Option<Duration>,
    /// A line for the status bar, owned as a transient message.
    status: Option<String>,
    /// The error to hand back to the caller, if the restart did not happen.
    error: Option<String>,
}

/// What a restart outcome means for the app's own state.
///
/// Paused stays paused: a restart is not a request to start playing, and a
/// listener who paused, changed output device, and found the music playing
/// again did not ask for that.
fn restart_effect(
    outcome: &RestartOutcome,
    was_paused: bool,
    title: Option<&str>,
) -> RestartEffect {
    match outcome {
        RestartOutcome::Stopped(reason) => RestartEffect {
            play_state: PlayState::Stopped,
            seek_to: None,
            status: None,
            error: Some(reason.message().to_string()),
        },
        RestartOutcome::Exact { landed } | RestartOutcome::Shared { landed } => {
            let fell_back = matches!(outcome, RestartOutcome::Shared { .. });
            RestartEffect {
                play_state: if was_paused {
                    PlayState::Paused
                } else {
                    PlayState::Playing
                },
                seek_to: (*landed > Duration::ZERO).then_some(*landed),
                status: fell_back.then(|| match title {
                    Some(t) => format!("⚠ This track played through the shared mixer — {t}"),
                    None => "⚠ This track played through the shared mixer".to_string(),
                }),
                error: None,
            }
        }
    }
}

/// Decide what a failed open means, and carry the decision out.
///
/// One reducer, so that starting a track and restarting one cannot disagree
/// about policy — and they did. `play_index` had the three-way rule inline;
/// `restart_current_track` had none, so it simply propagated the error, and
/// the toggle that had caused the restart was then reverted by its caller. A
/// listener who turned bit-perfect on during a track the DAC could not carry
/// exactly had the switch flipped back for them, and the preference they had
/// expressed was gone.
///
/// The three answers, none of which is "bit-perfect is switched on":
///
/// * **The failure has to be the endpoint's answer to a format question.** A
///   file that will not open, a container that refused a seek, a
///   self-contradictory `MOOSIK_BP_FORMAT`, a COM error or a thread that would
///   not start are not format rejections. The mixer would hit the same wall or
///   paper over a bug, and either way the listener is shown a downgrade
///   instead of the problem.
/// * **The policy has to permit processing.** Shared output has a volume
///   stage, an EQ, ReplayGain and a resampler in it. Under Strict that is
///   exactly what was forbidden, so it stops and says so.
/// * **DSD never reaches here.** It runs its own ladder inside `play_file` —
///   native, then DoP, then decimated PCM — and a raw DSD bitstream cannot go
///   through rodio's decoders anyway, so a DSD failure arriving here means
///   every permitted route already failed.
///
/// Nothing in this function can touch the requested preference, because it is
/// not given it. A track that cannot take the exact route says nothing about
/// the next one, and the next one starts again from the top of the ladder.
///
/// `open_shared` is the retry, injected rather than called through `self`, so
/// the decision can be driven without an audio device.
fn resolve_open(
    result: Result<Duration, bitperfect::OpenError>,
    policy: bitperfect::state::OutputPolicy,
    is_dsd: bool,
    open_shared: impl FnOnce() -> Result<Duration, String>,
) -> OpenOutcome {
    let e = match result {
        Ok(landed) => return OpenOutcome::Exact(landed),
        Err(e) => e,
    };
    if is_dsd || !e.is_device_limitation() || !policy.allows_processed() {
        crate::mlog!(
            "bp      open failed ({}): {e} — not retried through the shared mixer",
            e.kind()
        );
        return OpenOutcome::Stopped(e.to_reason());
    }
    match open_shared() {
        Ok(landed) => OpenOutcome::Shared(landed),
        Err(why) => OpenOutcome::SharedFailed(why),
    }
}

/// Fold one poll's worth of realtime evidence into a session state.
///
/// Two records, two destinations, and the order between them does not matter
/// because they no longer share a slot. `losses` are recoverable — the route
/// stopped being what it claimed and the music continues — and `codes` are
/// terminal. Both are first-wins within a session and both are scoped to
/// `generation`, so evidence from a track that has already rolled over cannot
/// land on the one playing.
///
/// Extracted so the sequence a listener actually experiences — a dropout on
/// one frame, the write failure that ends the track on the next — can be
/// driven without a device.
fn apply_integrity(
    state: &mut bitperfect::state::OutputSessionState,
    generation: u64,
    losses: &[u8],
    codes: &[u8],
) {
    for &code in losses {
        if let Some(reason) = bitperfect::fault::to_reason(code) {
            state.revoke(generation, reason);
        }
    }
    for &code in codes {
        if let Some(reason) = bitperfect::fault::to_reason(code) {
            state.fault(generation, reason);
        }
    }
}

/// The now-playing line and the headline it was rendered from.
///
/// One function returns both, because storing one without the other is what
/// let a line outlive the claim it named. The line carries the track title;
/// the headline does not, and it is the headline that is compared against the
/// session on every tick — a title does not change when a route does.
fn now_playing_status(
    presentation: Option<&bitperfect::state::Presentation>,
    title: &str,
) -> (String, String) {
    match presentation {
        Some(p) if p.badge != bitperfect::state::Badge::Off => {
            (format!("{} — {title}", p.headline), p.headline.clone())
        }
        // Bit-perfect is off: the line names no route, because there is no
        // claim to name. The fingerprint is still the session's own headline —
        // storing the rendered line here would make the comparison against the
        // live headline fail on every tick, which is the same defect this
        // function exists to remove, only quieter.
        Some(p) => (format!("Playing: {title}"), p.headline.clone()),
        // No session at all: nothing to go stale, and the line is its own
        // fingerprint.
        None => {
            let line = format!("Playing: {title}");
            (line.clone(), line)
        }
    }
}

/// The status line, and who owns it.
///
/// The text and the owner were two fields, written independently by a dozen
/// places, and their pairing was maintained by remembering to. It was not
/// remembered: the ASIO PCM probe wrote the text and left whatever owner was
/// there, so a probe result inherited the now-playing line's claim to be
/// regenerated and was wiped by the next tick; and a seek landing took
/// ownership whether or not it had put anything up, so an unrelated message
/// the listener had just been given became the app's to overwrite.
///
/// They are one value now, in a module of their own, with three ways in and
/// private fields — so a caller that has no headline to offer cannot become
/// the now-playing line by accident, which is what "inherited the claim"
/// means.
mod status {
    /// Who put the current line on the status bar.
    ///
    /// The only owner whose line may be regenerated behind the listener's back
    /// is the now-playing line, because that line embeds the fidelity headline
    /// and goes stale when the route changes under it. Everything else is
    /// something the app or the listener has just been told, and overwriting it
    /// loses the message.
    ///
    /// This was a bare `Option<Badge>`, which got both halves wrong. It compared
    /// badges, so a headline that changed *within* one badge — a dropout replaced
    /// by the backend write failure that ended the track, both `Faulted` — was
    /// never refreshed. And it stayed set through a stop, so the refresh went on
    /// regenerating "Playing: …" for a player that had stopped.
    #[derive(Clone, PartialEq, Debug, Default)]
    pub enum StatusOwner {
        /// Nobody: the line is empty or belongs to something that has
        /// finished.
        #[default]
        None,
        /// The now-playing line, and the exact headline it was rendered from.
        ///
        /// The whole headline, not the badge it belongs to: the badge is a
        /// category and the headline is the sentence, and it is the sentence
        /// that is on screen.
        NowPlaying { headline: String },
        /// A message the app or the listener has just been given — a seek
        /// failure, a device error, a toggle that could not be honoured. Not
        /// ours to overwrite.
        Transient,
    }

    /// The status line: what it says, and who it belongs to.
    #[derive(Default, Debug)]
    pub struct StatusLine {
        text: String,
        owner: StatusOwner,
    }

    impl StatusLine {
        /// What is on screen.
        pub fn text(&self) -> &str {
            &self.text
        }

        /// Who owns it. For assertions; nothing in the app branches on this.
        #[cfg(test)]
        pub fn owner(&self) -> &StatusOwner {
            &self.owner
        }

        /// Put the now-playing line up and take ownership of it.
        ///
        /// The line and the headline it was rendered from are set together,
        /// because a headline stored without its line — or a line stored under
        /// somebody else's headline — is a line that outlives the claim it
        /// names. `headline` is the fidelity sentence, not the rendered line:
        /// the rendered line also carries the track title, which does not
        /// change when the route does, so comparing whole lines would refresh
        /// on the first tick of every track and never afterwards.
        pub fn now_playing(&mut self, line: String, headline: String) {
            self.text = line;
            self.owner = StatusOwner::NowPlaying { headline };
        }

        /// Put up a message that is nobody's to regenerate.
        ///
        /// A seek failure, a device error, a probe result, a toggle that could
        /// not be honoured: something the listener has just been told, and
        /// overwriting it loses the only account of what happened.
        pub fn transient(&mut self, msg: impl Into<String>) {
            self.text = msg.into();
            self.owner = StatusOwner::Transient;
        }

        /// Give the line back. Stopping is not a claim about anything.
        pub fn clear(&mut self) {
            self.text.clear();
            self.owner = StatusOwner::None;
        }

        /// Regenerate the now-playing line if the claim it names has changed.
        ///
        /// Returns whether it did. `line` is only called when it will be used,
        /// because rendering it reaches into the session state.
        ///
        /// Two conditions, and the old code had neither. It fired on badge
        /// changes, so a dropout replaced by the backend write failure that
        /// ended the track — two sentences under one `Faulted` badge — left the
        /// first one on screen. And it fired regardless of whether anything was
        /// playing, so a stopped player got "Playing: …" put back by the next
        /// tick.
        pub fn refresh(
            &mut self,
            live_headline: &str,
            playing: bool,
            line: impl FnOnce() -> String,
        ) -> bool {
            let StatusOwner::NowPlaying { headline } = &self.owner else {
                return false;
            };
            if !playing || headline == live_headline {
                return false;
            }
            self.text = line();
            self.owner = StatusOwner::NowPlaying {
                headline: live_headline.to_string(),
            };
            true
        }

        /// A seek has landed.
        ///
        /// Ownership moves only if this landing actually replaced a notice it
        /// had put up. It used to move unconditionally, so a fast seek — one
        /// that lands before a "Seeking to …" notice is ever shown — took over
        /// whatever message happened to be on the bar and made it the app's to
        /// regenerate. The next tick then replaced the listener's message with
        /// a now-playing line.
        pub fn landed(&mut self, replaced_notice: bool, line: String, headline: String) {
            if replaced_notice {
                self.now_playing(line, headline);
            }
        }
    }
}

use status::StatusLine;
#[cfg(test)]
use status::StatusOwner;

/// Whether a device object owns the session `transport` describes.
///
/// Two inputs, and neither of them is liveness — which is the whole repair.
/// A function that cannot see whether a backend is still working cannot make
/// ownership depend on it, and ownership is what decides who gets asked how
/// the session ended. A dead backend is the *only* object that knows, and the
/// moment it stopped counting as the owner its answer stopped being read.
fn session_owner(transport: &bitperfect::Transport, bp_open: bool, native_open: bool) -> bool {
    use bitperfect::Transport;
    match transport {
        Transport::WasapiExclusivePcm { .. }
        | Transport::WasapiExclusiveDop { .. }
        | Transport::CpalDirect { .. } => bp_open,
        Transport::AsioPcm { .. }
        | Transport::AsioNativeDsd { .. }
        | Transport::AlsaNativeDsd { .. } => native_open,
        // The shared mixer is not a device object: the sink answers for it.
        Transport::Shared { .. }
        | Transport::Inactive
        | Transport::Opening
        | Transport::Dead { .. } => false,
    }
}

/// What a reopened route means for the spectrum, for the track that was
/// already playing: the position, and what the route now feeds the live tap.
///
/// Live-side only: the pre-process (its rate, identity, job and cache) is not
/// asked, so a same-track output change, restart, rollback or seek never
/// cancels or relaunches an analysis. Called after the route is installed —
/// by `apply_restart` (every restart, including a rollback) and when an
/// off-thread seek lands or fails — so it describes the source that actually
/// became active.
fn restart_spectrum(
    engine: Option<&Engine>,
    window: &mut SpectrumWindow,
    path: Option<&Path>,
    seek_to: Option<Duration>,
) {
    let tap = engine.zip(path).and_then(|(e, p)| e.live_tap_now(p));
    window.on_route_restart(seek_to.map(|at| at.as_secs_f64()), tap);
}

/// Resume the paused track, then describe the route that resume installed.
///
/// A resume can reopen: after a failed seek the sink is released, and
/// `Engine::resume` reopens it — synchronously for a DSD fallback, which
/// publishes no `Landed` for anything to answer, or off-thread for PCM, whose
/// landing refreshes again. So the tap is read back afterwards, from what is
/// installed, never assumed. Live-side only, like `restart_spectrum`: no
/// position change, no track start, no pre-process request. An ordinary
/// pause/resume finds the same tap and changes nothing.
fn resume_playback(engine: &mut Engine, window: &mut SpectrumWindow, path: Option<&Path>) {
    engine.resume();
    window.set_live_route(path.and_then(|p| engine.live_tap_now(p)));
}

/// The rate a DSD track falls back to for playback: the window's playback
/// setting, never its preprocessing one, which is analysis only.
fn playback_fallback_rate(w: &SpectrumWindow) -> u32 {
    w.dsd_fallback_rate
}

/// Arm the engine's playback fallback decimation for the next track start.
fn arm_dsd_fallback(engine: &mut Engine, rate: u32) {
    engine.dsd_fallback_target = rate;
}

/// What the status line should say when a seek reports in, or `None` to leave
/// it alone.
///
/// `notice_up` tracks whether the line is currently the provisional "Seeking
/// to …" notice, so the resolution knows whether the line is its to take
/// down. Nothing took it down: `Provisional` put it up to say that the
/// position on screen was a request rather than a landing, and it stayed
/// there after the landing — the one moment at which it is false — until
/// something unrelated happened to overwrite it.
fn seek_status_line(
    outcome: &SeekOutcome,
    notice_up: &mut bool,
    now_playing: impl FnOnce() -> String,
) -> Option<String> {
    match outcome {
        SeekOutcome::Provisional(at) => {
            *notice_up = true;
            Some(format!(
                "Seeking to {}:{:02}…",
                at.as_secs() / 60,
                at.as_secs() % 60
            ))
        }
        SeekOutcome::Landed(_) => {
            if std::mem::take(notice_up) {
                Some(now_playing())
            } else {
                None
            }
        }
        SeekOutcome::Failed(why) if why.is_worth_reporting() => {
            // Replaces the notice rather than queueing behind it.
            *notice_up = false;
            Some(format!("⚠ Seek failed: {}", why.describe()))
        }
        // A seek the listener replaced themselves. The thing that replaced it
        // has its own outcome and its own line; saying this one failed would
        // be reporting the player's own bookkeeping back at them.
        SeekOutcome::Failed(_) => {
            *notice_up = false;
            None
        }
        SeekOutcome::Idle | SeekOutcome::Working => None,
    }
}

fn shared_completion(
    sink_empty: bool,
    started: bool,
    elapsed: Duration,
    duration: Option<Duration>,
    generation: u64,
) -> bitperfect::Completion {
    if !sink_empty {
        return bitperfect::Completion::Running;
    }
    // Never before playback started: an appended sink reports empty for an
    // instant, and there is no elapsed time to compare against yet.
    if !started {
        return bitperfect::Completion::Running;
    }
    match duration {
        // Tag durations are approximate and some containers do not declare one
        // at all, so the margin is generous: this is meant to catch a decode
        // that stopped in the middle, not to police the last fraction of a
        // second. A track with no declared duration cannot be judged this way
        // at all, and gets the benefit of the doubt.
        Some(d) if elapsed + SHARED_SHORTFALL_TOLERANCE < d => {
            bitperfect::Completion::Failed {
                reason: bitperfect::fault::SHARED_ENDED_EARLY,
                generation,
            }
        }
        _ => bitperfect::Completion::CleanEof,
    }
}

/// Where a track is headed: a device session, or the shared mixer.
///
/// `Device` is the whole exact/native family — WASAPI or CPAL exclusive PCM,
/// DoP, ASIO and ALSA native DSD — because what every caller of this actually
/// wants to know is whether `play_file` has to run its ladder or whether the
/// destination is the sink that is already open. Which rung of the ladder is
/// reached is `play_file`'s business and depends on what the device says.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
enum Route {
    /// The shared `rodio` mixer.
    Shared,
    /// A device session, opened for this track.
    Device,
}

/// A track appended to the shared sink and not yet audible.
///
/// It carried a path and an optional source, and the rollover then asked
/// `fidelity_for` what to say about it — a function whose job is to decide
/// whether a *device* route preserved the samples. On the shared mixer it can
/// only ever get that wrong: the mixer converts to float, applies the volume
/// slider, the equaliser and ReplayGain, and hands the result to the OS to
/// mix with every other application on the machine. A track whose channel
/// layout could not be checked came out `Unverified` — the badge for "nothing
/// we did altered it" — of exactly that path.
///
/// So the route travels with the track. It is decided in `append_next`, which
/// is the code that knows whether it appended a decoder or a decimated DSD
/// stream, and it is applied unchanged at the boundary.
#[derive(Clone, Debug)]
struct QueuedShared {
    path: PathBuf,
    /// What the file is, where that could be established. `None` is a real
    /// answer and replaces the previous track's identity rather than leaving
    /// it standing.
    source: Option<bitperfect::state::MediaSource>,
    /// What the shared route is doing to it.
    transform: bitperfect::state::TransformDescription,
    /// Why the route is Processed, in the words for this track.
    reason: String,
}

/// The track the engine is playing, independent of how it is being played.
#[derive(Clone, Debug)]
struct CurrentTrack {
    path: PathBuf,
    /// What the file is, when that could be established. `None` is a real
    /// answer — a container `prepare` will not read, played through the shared
    /// mixer — and is published as an unknown source rather than a guess.
    source: Option<bitperfect::state::MediaSource>,
    /// Where playback of this track actually reached, updated as it moves.
    /// A failed seek restores this rather than the position the pointer was
    /// dropped on.
    at: Duration,
}

/// What a poll of a background seek found.
///
/// Typed because every one of these used to be indistinguishable from the
/// others at the call site: "still working", "the file cannot be seeked" and
/// "the output device would not open" were all either a bare `return` or a
/// silent clear.
#[derive(Clone, PartialEq, Debug)]
enum SeekOutcome {
    /// No seek is in flight.
    Idle,
    /// Still decoding, and the position on screen is the *request* rather than
    /// anything anyone has reached.
    ///
    /// This distinction was being papered over. The slow path sets the clock,
    /// the slider and the spectrum to the target the moment the worker is
    /// spawned — it has to show something, and it cannot show the landing,
    /// because nothing has landed — while the comment above `poll_pending_seek`
    /// claimed nothing was committed until the sink existed. Both cannot be
    /// true. What is true is that the position is provisional until
    /// `Landed` replaces it or `Failed` withdraws it, and that is now a state
    /// the UI can see and say.
    Provisional(Duration),
    /// Still decoding, with nothing provisional on screen.
    Working,
    /// A sink is installed and playing from this position.
    Landed(Duration),
    /// The seek did not happen, and will not.
    Failed(SeekError),
}

/// What plays after `current`, under `mode`, in a playlist of `len`.
///
/// One function, used both to prebuffer the next track and to choose it when
/// the current one ends, so the two can never disagree about what "next" is.
///
/// It is reached only from `TickAction::Advance`. A failed session never gets
/// here at all, which is the property that stops Repeat One reopening a file
/// that fails deterministically.
fn next_in_playlist(mode: LoopMode, current: Option<usize>, len: usize) -> Option<usize> {
    if len == 0 {
        return None;
    }
    match (mode, current) {
        (LoopMode::RepeatOne, Some(i)) => Some(i),
        (LoopMode::RepeatAll, Some(i)) => Some((i + 1) % len),
        (LoopMode::RepeatAll, None) => Some(0),
        (LoopMode::Sequential, Some(i)) => (i + 1 < len).then_some(i + 1),
        (LoopMode::RepeatOne, None) | (LoopMode::Sequential, None) => None,
    }
}

/// Column the playlist can be sorted by. `Plays` / `Recent` sort by the
/// persisted play statistics (most-played / most-recently-played first).
#[derive(PartialEq, Clone, Copy)]
enum SortKey { Title, Artist, Album, Duration, Plays, Recent }

impl SortKey {
    const ALL: [SortKey; 6] = [
        SortKey::Title, SortKey::Artist, SortKey::Album,
        SortKey::Duration, SortKey::Plays, SortKey::Recent,
    ];
    fn label(self) -> &'static str {
        match self {
            SortKey::Title => "Title", SortKey::Artist => "Artist",
            SortKey::Album => "Album", SortKey::Duration => "Time",
            SortKey::Plays => "Plays", SortKey::Recent => "Recent",
        }
    }
    /// These default to descending (highest / most-recent first).
    fn default_desc(self) -> bool {
        matches!(self, SortKey::Plays | SortKey::Recent)
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

/// Seconds since the Unix epoch (0 if the clock is before it, which never
/// happens in practice).
fn now_unix() -> u64 {
    std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|d| d.as_secs())
        .unwrap_or(0)
}

/// A coarse "… ago" string for a Unix timestamp (no calendar library needed).
fn fmt_ago(unix: u64) -> String {
    let now = now_unix();
    if unix == 0 || unix > now { return "just now".to_string(); }
    let s = now - unix;
    match s {
        0..=59            => "just now".to_string(),
        60..=3599         => format!("{} min ago", s / 60),
        3600..=86_399     => format!("{} h ago", s / 3600),
        86_400..=2_591_999 => format!("{} days ago", s / 86_400),
        _                 => format!("{} months ago", s / 2_592_000),
    }
}

// ── Play statistics (~/.moosik/stats.json) ─────────────────────────────────

#[derive(Serialize, Deserialize, Clone, Default)]
struct PlayStat {
    #[serde(default)] count: u32,
    /// Unix seconds of the most recent play.
    #[serde(default)] last: u64,
}

#[derive(Serialize, Deserialize, Clone, Default)]
struct PlayStats {
    /// Key: absolute track path as a string.
    #[serde(default)] plays: HashMap<String, PlayStat>,
}

impl PlayStats {
    fn load(dir: &Path) -> Self {
        std::fs::read_to_string(dir.join("stats.json"))
            .ok().and_then(|s| serde_json::from_str(&s).ok()).unwrap_or_default()
    }
    fn save(&self, dir: &Path) {
        let _ = std::fs::create_dir_all(dir);
        if let Ok(json) = serde_json::to_string(self) {
            let _ = std::fs::write(dir.join("stats.json"), json);
        }
    }
    fn record(&mut self, path: &Path) {
        let e = self.plays.entry(path.to_string_lossy().into_owned()).or_default();
        e.count += 1;
        e.last = now_unix();
    }
    fn get(&self, path: &Path) -> Option<&PlayStat> {
        self.plays.get(&*path.to_string_lossy())
    }
}

// ── Bookmarks (~/.moosik/bookmarks.json) ───────────────────────────────────

#[derive(Serialize, Deserialize, Clone, Default)]
struct Bookmarks {
    /// Key: absolute track path → sorted list of positions (seconds).
    #[serde(default)] marks: HashMap<String, Vec<f32>>,
}

impl Bookmarks {
    fn load(dir: &Path) -> Self {
        std::fs::read_to_string(dir.join("bookmarks.json"))
            .ok().and_then(|s| serde_json::from_str(&s).ok()).unwrap_or_default()
    }
    fn save(&self, dir: &Path) {
        let _ = std::fs::create_dir_all(dir);
        if let Ok(json) = serde_json::to_string(self) {
            let _ = std::fs::write(dir.join("bookmarks.json"), json);
        }
    }
    fn for_track(&self, path: &Path) -> Option<&Vec<f32>> {
        self.marks.get(&*path.to_string_lossy())
    }
    fn add(&mut self, path: &Path, secs: f32) {
        let v = self.marks.entry(path.to_string_lossy().into_owned()).or_default();
        // Ignore a near-duplicate within 1 s.
        if v.iter().any(|&x| (x - secs).abs() < 1.0) { return; }
        v.push(secs);
        v.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    }
    fn remove(&mut self, path: &Path, secs: f32) {
        if let Some(v) = self.marks.get_mut(&*path.to_string_lossy()) {
            v.retain(|&x| (x - secs).abs() >= 0.01);
            if v.is_empty() { self.marks.remove(&*path.to_string_lossy()); }
        }
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
    /// Show each track's play count in the playlist rows.
    #[serde(default)] show_play_count: bool,
    /// Family name of the chosen UI font, or `None` for egui's own.
    ///
    /// Stored by family name rather than by path so the setting survives a font
    /// being reinstalled or the app moving between machines; a name that no
    /// longer resolves falls back to the default instead of failing.
    #[serde(default)] ui_font: Option<String>,
    /// How the lyrics window shows a translation: as stored, or converted to
    /// Traditional or Simplified Chinese for display (never for saving).
    #[serde(default)] lyrics_translation: lyrics::zh::ChineseDisplay,
    /// Lyric text size, in points. A file from before it existed, or one
    /// holding something unusable, gets the default without losing the
    /// rest of the appearance.
    #[serde(default = "default_lyrics_text_size", deserialize_with = "lenient_lyrics_text_size")]
    lyrics_text_size: f32,
}

impl Default for Appearance {
    fn default() -> Self {
        Self {
            spectrum_palette: spectrum::SpectrumPalette::default(),
            ui_scale: 1.0,
            art_accent: true,
            theme: ThemeMode::Dark,
            show_play_count: false,
            ui_font: None,
            lyrics_translation: lyrics::zh::ChineseDisplay::Original,
            lyrics_text_size: lyrics::ui::TEXT_SIZE_DEFAULT,
        }
    }
}

fn default_lyrics_text_size() -> f32 { lyrics::ui::TEXT_SIZE_DEFAULT }

/// Any JSON value: a finite number is kept within range, anything else is
/// the default. Never an error, so one bad field cannot reset the others.
fn lenient_lyrics_text_size<'de, D: serde::Deserializer<'de>>(d: D) -> Result<f32, D::Error> {
    let v = <serde_json::Value as serde::Deserialize>::deserialize(d)?;
    Ok(v.as_f64().map_or(lyrics::ui::TEXT_SIZE_DEFAULT, |x| lyrics::ui::sane_text_size(x as f32)))
}

/// What to write for the lyric text size, if anything: a settled change.
fn lyrics_text_size_to_save(window: f32, saved: f32, settled: bool) -> Option<f32> {
    (settled && window.to_bits() != saved.to_bits()).then_some(window)
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
        let tagged = probe_tagged(path)?;
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
    status: StatusLine,
    /// Whether `status_msg` is currently the provisional "Seeking to …"
    /// notice, so `Landed` and `Failed` know the line is theirs to withdraw
    /// and do not overwrite something the app has since said.
    seek_notice: bool,
    /// Reserved: the seek notice below is the only remaining flag.
    /// rendered from.
    spectrum_window: SpectrumWindow,
    lyrics_window: lyrics::ui::LyricsWindow,
    tag_window: tags::ui::TagWindow,
    info_open: bool,
    loop_mode: LoopMode,
    /// Last (volume, loop_mode) written to player.json, for change detection.
    saved_player: PlayerPrefs,
    player_save_at: Option<Instant>,
    /// Sleep timer: stop playback at this instant, or at the end of the current
    /// track (when `sleep_end_of_track`).
    sleep_deadline: Option<Instant>,
    sleep_end_of_track: bool,
    /// A-B repeat within the current track (loops between two positions).
    ab_a: Option<Duration>,
    ab_b: Option<Duration>,
    /// Persisted play counts and per-track bookmarks.
    stats: PlayStats,
    bookmarks: Bookmarks,
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
    /// System fonts for the picker, filled the first time the Look menu opens.
    font_list: Vec<fonts::FontEntry>,
    /// Substring filter for the font list — a machine can easily have a couple
    /// of hundred families, which is not a thing to scroll through.
    font_filter: String,
}

impl MoosikApp {
    fn new(cc: &eframe::CreationContext) -> Self {
        // Read the font choice before anything draws, so the window never
        // flashes the default face on the way to the chosen one. The full
        // `Appearance` is loaded again below; this only needs the one field.
        setup_fonts(&cc.egui_ctx, load_appearance(&moosik_dir()).ui_font.as_deref());
        // Give the ASIO host the real main-window handle for driver init —
        // drivers parent hidden notification windows to it.
        #[cfg(all(windows, feature = "asio-dsd"))]
        {
            use raw_window_handle::{HasWindowHandle, RawWindowHandle};
            if let Ok(h) = cc.window_handle()
                && let RawWindowHandle::Win32(w) = h.as_raw() {
                bitperfect::asio_dsd::set_app_hwnd(w.hwnd.get() as *mut std::ffi::c_void);
            }
        }
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
            e.asio_driver = bp_settings.asio_driver.clone();
            e.alsa_dsd_device = bp_settings.alsa_dsd_device.clone();
            // The persisted request is a request, and nothing more. Loading it
            // must leave the session Idle with the volume unlocked and no
            // diamond — 1.4.2 set `bit_perfect` here and left the session state
            // untouched, so the two disagreed from the first frame.
            e.bp_state.requested = bp_settings.enabled;
            e.bp_state.policy = bp_settings.policy;
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
        if let Some(ref mut e) = engine { arm_dsd_fallback(e, playback_fallback_rate(&spectrum_window)); }
        MoosikApp {
            playlist,
            current_index: None,
            play_state: PlayState::Stopped,
            engine,
            volume: player_prefs.volume,
            seek_pos: 0.0,
            seeking: false,
            status: StatusLine::default(),
            seek_notice: false,
            spectrum_window,
            lyrics_window: {
                let mut w = lyrics::ui::LyricsWindow::default();
                w.translation_display = appearance.lyrics_translation;
                w.set_text_size(appearance.lyrics_text_size);
                w
            },
            tag_window: tags::ui::TagWindow::default(),
            info_open: false,
            loop_mode: player_prefs.loop_mode,
            saved_player: player_prefs,
            player_save_at: None,
            sleep_deadline: None,
            sleep_end_of_track: false,
            ab_a: None,
            ab_b: None,
            stats: PlayStats::load(&moosik_dir()),
            bookmarks: Bookmarks::load(&moosik_dir()),
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
            font_list: Vec::new(),
            font_filter: String::new(),
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
        let asc = if self.sort_key == Some(key) { !self.sort_asc } else { !key.default_desc() };
        let cur_path = self.current_index
            .and_then(|i| self.playlist.get(i))
            .map(|t| t.path.clone());
        // Snapshot the stats so the comparator doesn't borrow self while the
        // playlist is being sorted in place.
        let stats = self.stats.clone();
        let stat = |p: &Path| stats.get(p).cloned().unwrap_or_default();
        self.playlist.sort_by(|a, b| {
            let o = match key {
                SortKey::Title    => a.title.to_lowercase().cmp(&b.title.to_lowercase()),
                SortKey::Artist   => a.artist.to_lowercase().cmp(&b.artist.to_lowercase()),
                SortKey::Album    => a.album.to_lowercase().cmp(&b.album.to_lowercase()),
                SortKey::Duration => a.duration.cmp(&b.duration),
                SortKey::Plays    => stat(&a.path).count.cmp(&stat(&b.path).count),
                SortKey::Recent   => stat(&a.path).last.cmp(&stat(&b.path).last),
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
            .add_filter("Audio (containers: audio track only)", &audio_formats::import_extensions())
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
                } else if audio_formats::is_importable(&path) {
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
        let is_dsd = dsd::is_dsd_path(&path);
        let title = self.playlist[idx].title.clone();
        let rate = playback_fallback_rate(&self.spectrum_window);
        let Some(engine) = self.engine.as_mut() else {
            self.play_state = PlayState::Stopped;
            self.set_transient("Error: no audio engine");
            return;
        };
        // The spectrum learns the tap's actual rate from the engine below.
        arm_dsd_fallback(engine, rate);
        // The orchestration the restart uses, at position zero. Starting a
        // track was the one open outside it: the policy ladder was written out
        // here a second time, and a failure published the typed reason and
        // released nothing — so a DSD open that failed after the ASIO driver
        // was loaded left the driver loaded and the endpoint held by a player
        // that had stopped.
        let (p1, p2) = (path.clone(), path.clone());
        let outcome = open_track(
            engine,
            // A track chosen from the playlist starts playing: nothing about
            // that request is paused.
            Opening::at(Duration::ZERO),
            is_dsd,
            move |e, o| e.play_file_at(&p1, duration, o),
            move |e, o| e.play_file_shared_at(&p2, duration, o),
        );
        // And one description of what the outcome does to the app, shared with
        // every other caller — `play_state` is set here and nowhere else in
        // this function.
        let effect = restart_effect(&outcome, false, Some(&title));
        self.play_state = effect.play_state;
        match outcome {
            RestartOutcome::Exact { .. } => {
                self.set_now_playing(idx);
                if let Some(e) = self.engine.as_ref().filter(|e| e.dsd_fallback) {
                    let label = e.dsd_label.as_deref().unwrap_or("DSD");
                    let note = e.dsd_fallback_note.as_deref().unwrap_or("DoP unavailable");
                    let line = format!(
                        "⚠ {note} — playing {label} as {} PCM — {}",
                        fmt_hz(e.last_sample_rate), self.playlist[idx].title,
                    );
                    self.set_transient(line);
                }
                self.on_track_started(&path);
            }
            // This track could not take the exact route. That is a fact about
            // this track, not a decision about the next one: the preference
            // survives untouched and the next track starts again from the top
            // of the ladder.
            RestartOutcome::Shared { .. } => {
                if let Some(line) = effect.status {
                    self.set_transient(line);
                }
                self.on_track_started(&path);
            }
            // Released by the transaction, with the reason the failure
            // actually was — not `DeviceUnavailable` for everything, which is
            // how a configuration typo was reported as a missing DAC.
            RestartOutcome::Stopped(reason) => {
                self.set_transient(format!("Error: {}", reason.message()));
                return;
            }
        }
        self.seek_pos = 0.0;
        self.seeking = false;
        let sr = self.engine.as_ref().map(|e| e.last_sample_rate).unwrap_or(44_100);
        let live = self.engine.as_ref().map_or(Some(sr), |e| e.spectrum_live_rate(&path, sr));
        self.spectrum_window.on_play_live(&self.playlist[idx].path, sr, live);
    }

    // ── Bit-perfect helpers ─────────────────────────────────────────────────

    /// The status line for a track that has just started.
    ///
    /// Derived from the session's own `Presentation`, like every other visible
    /// surface. There is no diamond in this function: `Presentation` decides
    /// whether one is warranted, and it grants one only for `PayloadExact`.
    /// Four separate call sites used to format `"💎 {desc}"` from a `bp_describe()`
    /// that answers "what is the stream", never "is it exact" — so a
    /// decimated, resampled or rounded route printed a green diamond as long
    /// as it was a device stream.
    /// Fold every boundary the device has already crossed into the UI.
    ///
    /// Called immediately before anything reads integrity state, and again in
    /// the gapless section for anything crossed since. Draining twice is free:
    /// the second call finds nothing unless the device crossed in between, and
    /// then folding it is exactly right.
    fn fold_reached_boundaries(&mut self) {
        if self.play_state != PlayState::Playing {
            return;
        }
        // A-B repeat keeps playback inside one track; there is no hand-off to
        // fold and the section below skips it for the same reason.
        if self.ab_a.is_some() && self.ab_b.is_some() {
            return;
        }
        while self.engine.as_mut().map(|e| e.bp_poll_boundary()).unwrap_or(false) {
            self.gapless_rollover();
        }
    }

    /// Put the now-playing line up and take ownership of it.
    ///
    /// The line and the headline it was rendered from come from one function
    /// and are stored together, so no caller can set one without the other.
    /// The headline stored is the fidelity sentence — `set_now_playing` used
    /// to store the whole rendered line, title and all, against which the live
    /// headline never compares equal.
    fn set_now_playing(&mut self, idx: usize) {
        let (line, headline) = self.now_playing_pair(idx);
        self.status.now_playing(line, headline);
    }

    /// Put a message up that is nobody's to regenerate.
    fn set_transient(&mut self, msg: impl Into<String>) {
        self.status.transient(msg);
    }

    /// The now-playing line, and the headline it was rendered from.
    fn now_playing_pair(&self, idx: usize) -> (String, String) {
        let title = self
            .playlist
            .get(idx)
            .map(|t| t.title.clone())
            .unwrap_or_default();
        let presentation = self.engine.as_ref().map(|e| e.bp_state.presentation());
        now_playing_status(presentation.as_ref(), &title)
    }

    fn now_playing_line(&self, idx: usize) -> String {
        self.now_playing_pair(idx).0
    }

    fn save_bp_settings(&self) {
        bitperfect::save_settings(&moosik_dir(), &bitperfect::BpSettings {
            enabled: self.bit_perfect,
            device: self.bp_device.clone(),
            asio_driver: self.engine.as_ref().and_then(|e| e.asio_driver.clone()),
            alsa_dsd_device: self.engine.as_ref().and_then(|e| e.alsa_dsd_device.clone()),
            policy: self.engine.as_ref().map(|e| e.bp_state.policy).unwrap_or_default(),
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
            // The preference and the claim are separate facts. Recording the
            // request here never colours anything green on its own; only an
            // open route with proven integrity does that.
            engine.bp_state.requested = on;
            if !on { engine.close_bp(); } // release the device
        }
    }

    /// Push the EQ state into the engine's published processing chain.
    ///
    /// The panel describes what is in the path *now*. Enabling a band, or
    /// switching the EQ off entirely, changes that — and nothing recomputed
    /// it, so the panel went on reporting whatever the chain looked like when
    /// the track started.
    fn on_eq_changed(&mut self) {
        if let Some(e) = self.engine.as_mut() {
            e.bp_refresh_processing();
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
        match self.capture_restart() {
            Some(req) => self.restart_to(&req),
            None => Ok(()),
        }
    }

    /// Everything a restart needs, read from live state.
    ///
    /// Taken as one value and before anything is torn down, because a rollback
    /// happens *after* an attempt that has already changed the state it would
    /// otherwise read. `None` means there is nothing to restart.
    fn capture_restart(&self) -> Option<RestartRequest> {
        let idx = self.current_index?;
        if self.play_state == PlayState::Stopped {
            return None;
        }
        let track = self.playlist.get(idx)?;
        Some(RestartRequest {
            index: idx,
            path: track.path.clone(),
            duration: track.duration,
            target: self.elapsed(),
            was_paused: self.play_state == PlayState::Paused,
        })
    }

    /// Restart to a captured request.
    ///
    /// Deliberately does not consult `play_state`: that check belongs to
    /// `capture_restart`, at the moment the request is taken, and having it
    /// here as well is what made the rollback in `select_bp_device` do
    /// nothing. The failed attempt had just set the app to `Stopped`, so the
    /// rollback's first statement was `return Ok(())` — the listener was told
    /// the previous output had taken the track back, and it had never been
    /// asked.
    fn restart_to(&mut self, req: &RestartRequest) -> Result<(), String> {
        let pos = req.target;
        let was_paused = req.was_paused;
        let path = req.path.clone();
        let dur = req.duration;
        let engine = self.engine.as_mut().ok_or("No audio engine")?;

        // Rodio-session restart with a real position (e.g. toggling
        // bit-perfect OFF mid-track): open + seek entirely off the UI thread.
        // Skips the play-from-0 blip and keeps every decode off the UI thread
        // — a deep hi-res FLAC seek on the UI thread here is what froze the
        // app. Device sessions (bp/DoP/native DSD) take the play_file+seek
        // path below instead.
        //
        // Where the track is going, from the planner `play_file` uses, asked
        // about this track.
        //
        // The predicate here was `bit_perfect || native_dsd_selected()`, which
        // is a question about two global preferences and is wrong in both
        // directions. A PCM file with the toggle off and an ASIO driver
        // configured for DSD was sent down the blocking `play_file` + seek
        // path, because a setting about DSD had been read as a setting about
        // this track — a deep FLAC seek on the UI thread, which is the freeze
        // the shortcut exists to avoid. And a DSD file with the toggle off and
        // no native driver was sent to `play_seeked_async`, which opens the
        // file with `rodio`: it cannot read a DSD container, so the restart
        // failed on a track that was playing perfectly well a moment earlier.
        //
        // Neither branch touches the preference. A track the device refuses
        // falls back for itself, in `play_file`, and what the listener asked
        // for stays what they asked for.
        let route = engine.route_for(&path);
        if route == Route::Shared && !engine.on_bp_stream() && pos > Duration::ZERO {
            engine.play_seeked_async(&path, pos, dur, was_paused);
            // The seek is asynchronous, so the landing is provisional and
            // `SeekOutcome::Landed` corrects it. Everything else the app owns
            // is applied through the same function as every other restart.
            return self.apply_restart(RestartOutcome::Exact { landed: pos }, was_paused);
        }

        // Open and reach the position, or leave nothing behind. The two used
        // to be separate, and a seek that failed returned an error with the
        // track playing from zero and an exclusive handle still held.
        let is_dsd = dsd::is_dsd_path(&path);
        let p1 = path.clone();
        let p2 = path.clone();
        // The pause intent travels with the position, all the way to the
        // device. It used to be applied by the caller afterwards, so every
        // paused restart played a buffer first.
        let outcome = open_track(
            engine,
            Opening {
                at: pos,
                paused: was_paused,
            },
            is_dsd,
            move |e, o| e.play_file_at(&p1, dur, o),
            move |e, o| e.play_file_shared_at(&p2, dur, o),
        );
        self.apply_restart(outcome, was_paused)
    }

    /// Move a playing DSD track onto a newly chosen native driver, and put it
    /// back if it will not go.
    ///
    /// The same shape as the output-device picker, because it is the same
    /// operation: a setting changes, playback has to follow it, and if it
    /// cannot then the setting and the playback both have to go back. These
    /// two did neither. They changed the driver, reported a failure, and left
    /// the listener stopped on a configuration that had just been shown not to
    /// work — with the previous driver, which had been working a moment
    /// earlier, discarded.
    ///
    /// `restore` puts the configured driver back; it is a closure because ASIO
    /// and ALSA keep theirs in different fields and nothing else about this
    /// differs.
    fn switch_native_output(&mut self, label: &str, restore: impl FnOnce(&mut Engine)) {
        let playing = self
            .engine
            .as_ref()
            .is_some_and(|e| e.dsd_native || e.dsd_mode || e.dsd_fallback);
        if !playing {
            return;
        }
        // Before the attempt, for the same reason as the output picker: the
        // attempt is what destroys the state a rollback would have to read.
        let snapshot = self.capture_restart();
        let outcome = switch_output(
            self,
            snapshot.as_ref(),
            |app| app.restart_current_track(),
            |app, req| {
                if let Some(e) = app.engine.as_mut() {
                    restore(e);
                }
                app.save_bp_settings();
                app.restart_to(req)
            },
        );
        match outcome {
            SwitchOutcome::Moved => {}
            SwitchOutcome::RolledBack { why } => {
                self.set_transient(format!("{label} unavailable: {why}"));
            }
            SwitchOutcome::Stranded { why, back } => {
                self.set_transient(format!(
                    "{label} unavailable: {why} — and the previous output would \
                     not take the track back either: {back}"
                ));
            }
        }
    }

    /// Apply a restart outcome — all of it, in one place.
    ///
    /// Route and handles are the transaction's business; this is everything
    /// the app owns: the play state, the pause state, the position on the
    /// spectrum, and who owns the status line. Every caller of
    /// `restart_current_track` goes through here, so none of them can apply
    /// half of it.
    /// The current track's path, if any.
    fn current_track_path(&self) -> Option<PathBuf> {
        self.current_index
            .and_then(|i| self.playlist.get(i))
            .map(|t| t.path.clone())
    }

    fn apply_restart(
        &mut self,
        outcome: RestartOutcome,
        was_paused: bool,
    ) -> Result<(), String> {
        let title = self
            .current_index
            .and_then(|i| self.playlist.get(i))
            .map(|t| t.title.clone());
        let effect = restart_effect(&outcome, was_paused, title.as_deref());

        let state = effect.play_state;
        self.play_state = state;
        if state == PlayState::Stopped {
            // The transaction has already released everything; the app follows
            // it there rather than being left describing a session that no
            // longer exists.
            self.gapless_next = None;
            self.gapless_tried = None;
        }
        // Bookkeeping, not the pause.
        //
        // The route already came up paused: the intent travelled with the
        // position into every opener, which is the only place it can be
        // applied without a buffer going out first. What is left here is the
        // session's own record — the playback state the panel reads and the
        // elapsed-time accounting — and `Engine::pause` is where that lives.
        // This call used to *be* the pause, one statement after the route
        // started, and a statement is audible.
        if state == PlayState::Paused
            && let Some(e) = self.engine.as_mut()
        {
            e.pause();
        }
        let path = self.current_track_path();
        restart_spectrum(
            self.engine.as_ref(),
            &mut self.spectrum_window,
            path.as_deref(),
            effect.seek_to,
        );
        if let Some(line) = effect.status {
            self.set_transient(line);
        }
        match effect.error {
            Some(why) => Err(why),
            None => Ok(()),
        }
    }

    /// Toggle bit-perfect mode, restarting the current track in the new mode.
    ///
    /// It does **not** revert the toggle if the device refuses. It used to,
    /// and that is the one thing a listener who has just asked for exact
    /// output did not ask for: their setting overwritten because one file
    /// could not take it. Under Automatic or HQ the track falls back for
    /// itself and the toggle stands; under Strict it stops and the toggle
    /// still stands.
    fn toggle_bit_perfect(&mut self) {
        let new_bp = !self.bit_perfect;
        self.apply_bit_perfect(new_bp);
        match self.restart_current_track() {
            Ok(()) => {
                // What the route turned out to be, not what was asked for.
                // Printing a diamond because the toggle went on is the whole
                // defect: the request is not the achievement.
                // A toggle that landed on a live track leaves the now-playing
                // line up — and it is the now-playing line, so it is owned as
                // one and stays true as the route settles. It used to be
                // written as a transient, which froze the sentence at the
                // moment of the toggle.
                //
                // The decision is `toggle_status`, and the condition it gets
                // right is the one this call site had wrong: a *selected*
                // track is not a *playing* one.
                match toggle_status(new_bp, self.play_state, self.current_index) {
                    ToggleStatus::NowPlaying(i) => self.set_now_playing(i),
                    ToggleStatus::Transient(msg) => self.set_transient(msg),
                }
            }
            Err(e) => {
                // The preference stands.
                //
                // This used to flip the toggle back and restart again, which
                // is the one thing a listener who has just asked for exact
                // output did not ask for: their setting was overwritten
                // because one track could not take it. A track that the device
                // refuses says nothing about the next one, and under Automatic
                // or HQ it does not even stop — `restart_current_track` has
                // already put it on the mixer for this track alone.
                self.play_state = PlayState::Stopped;
                self.set_transient(format!("Bit-perfect unavailable for this track: {e}"));
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
        // DSD plays on the picked device regardless of the bit-perfect toggle,
        // so a device change must move a running DoP session too.
        let device_in_use = self.bit_perfect
            || self.engine.as_ref().is_some_and(|e| e.dsd_mode);
        if device_in_use {
            // Taken before the attempt, because the attempt is what destroys
            // the state a rollback would otherwise have to read back.
            let snapshot = self.capture_restart();
            let old_for_rollback = old.clone();
            let outcome = switch_output(
                self,
                snapshot.as_ref(),
                |app| app.restart_current_track(),
                |app, req| {
                    // The device setting goes back first, so the restart below
                    // opens on the output being restored rather than on the
                    // one that just refused the track.
                    app.bp_device = old_for_rollback.clone();
                    if let Some(ref mut engine) = app.engine {
                        engine.bp_device = old_for_rollback;
                    }
                    app.save_bp_settings();
                    app.restart_to(req)
                },
            );
            match outcome {
                SwitchOutcome::Moved => {}
                SwitchOutcome::RolledBack { why } => {
                    self.set_transient(format!("Device unavailable: {why}"));
                    return;
                }
                SwitchOutcome::Stranded { why, back } => {
                    self.set_transient(format!(
                        "Device unavailable: {why} — and the previous output \
                         would not take the track back either: {back}"
                    ));
                    return;
                }
            }
        }
        self.set_transient(match (&self.bp_device, self.bit_perfect) {
            (Some(n), _) => format!("Output device: {n}"),
            (None, _) => "Output device: system default".to_string(),
        });
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
                let path = self.current_track_path();
                if let Some(ref mut engine) = self.engine {
                    resume_playback(engine, &mut self.spectrum_window, path.as_deref());
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
        // Stopping gives the line back. It used to keep it, so the refresh
        // below went on regenerating "Playing: …" for a player that had
        // stopped.
        self.status.clear();
        self.spectrum_window.on_stop();
    }

    // ── Gapless playback ────────────────────────────────────────────────────

    /// The index that will play after the current one, per the loop mode — the
    /// track to prebuffer for a seamless hand-off. None = nothing follows.
    fn upcoming_index(&self) -> Option<usize> {
        let cur = self.current_index?;
        next_in_playlist(self.loop_mode, Some(cur), self.playlist.len())
    }

    /// Once per track, when it nears its end, prebuffer the next one onto the
    /// same stream so playback continues with no gap. Requires a known duration
    /// (for the lead timing); bit-perfect additionally requires the next track
    /// to share the current rate/channels, else it's left to end normally.
    fn try_arm_gapless(&mut self) {
        let Some(cur) = self.current_index else { return };
        if self.gapless_tried == Some(cur) { return; }
        let Some(dur) = self.playlist.get(cur).and_then(|t| t.duration) else { return };
        let bp = self.engine.as_ref().map(|e| e.on_bp_stream()).unwrap_or(false);
        let lead = Duration::from_secs(if bp { 5 } else { 2 });
        if dur.saturating_sub(self.elapsed()) > lead { return; }

        // Decide exactly once for this track, whatever the outcome.
        self.gapless_tried = Some(cur);
        let Some(next) = self.upcoming_index() else { return };
        let path = self.playlist[next].path.clone();
        let next_is_dsd = dsd::is_dsd_path(&path);
        let armed = self.engine.as_mut().map(|e| {
            if e.dsd_native {
                // Native DSD (ASIO/ALSA): no gapless queue yet (first
                // hardware-validated cut keeps one session per file) — the
                // advance re-opens, reusing the output when the rate matches.
                false
            } else if e.dsd_mode {
                // DoP session: only another DSD file at the same carrier can
                // continue it (bp_queue_next_dop checks the rate) — a PCM
                // successor needs a device re-open, so no gapless.
                next_is_dsd && e.bp_queue_next_dop(&path)
            } else if e.dsd_fallback {
                // Fallback sink: another DSD track appends as decimated PCM.
                // A PCM successor restarts cleanly instead (play_index resets
                // the DSD flags and, with 💎 on, reclaims the device).
                next_is_dsd && e.append_next(&path).is_ok()
            } else if bp {
                // PCM bit-perfect stream: DSD can't ride it.
                !next_is_dsd && e.bp_queue_next(&path)
            } else {
                // Plain rodio: PCM only — a DSD successor restarts through
                // play_index so it gets its DoP-first treatment and the
                // right status/flags rather than silently degrading.
                !next_is_dsd && e.append_next(&path).is_ok()
            }
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
        // DoP (DSD) uses the same frame-boundary machinery as PCM bit-perfect;
        // DSD fallback does not, even with the global toggle on — it's really
        // on the rodio path (on_bp_stream(), not the raw flags, is the truth).
        let bp = self.engine.as_ref().map(|e| e.on_bp_stream()).unwrap_or(false);
        if let Some(e) = self.engine.as_mut() {
            if bp {
                e.current_duration = next_dur; // position base advanced in bp_poll_boundary
            } else {
                e.roll_normal_position(next_dur);
                // The shared route has no frame-exact boundary to report, so
                // this is where its successor stops being queued and starts
                // being the track: source, fidelity, processing chain and the
                // prepared identity all move together, at the moment the
                // listener hears the change. They did not move at all, and the
                // panel described the previous file for the whole of the next
                // one.
                e.roll_shared_source();
            }
        }
        self.current_index = Some(next);
        self.gapless_tried = None; // let the new current arm its own successor
        self.seek_pos = 0.0;
        self.on_track_started(&self.playlist[next].path.clone());
        let sr = self.playlist[next].sample_rate
            .or_else(|| self.engine.as_ref().map(|e| e.last_sample_rate))
            .unwrap_or(44_100);
        let path = self.playlist[next].path.clone();
        let title = self.playlist[next].title.clone();
        let live = self.engine.as_ref().map_or(Some(sr), |e| e.spectrum_live_rate(&path, sr));
        self.spectrum_window.on_play_live(&path, sr, live);
        let _ = (bp, &title);
        self.set_now_playing(next);
    }

    /// Discard a queued gapless hand-off (the user changed what plays next).
    /// Bit-perfect just drops the queued decode; normal mode must rebuild the
    /// current track without its appended successor (a brief gap, but only on an
    /// explicit action near a track boundary).
    fn flush_gapless(&mut self) {
        if let Some(e) = self.engine.as_mut() {
            e.queued_next = None;
        }
        if self.gapless_next.take().is_none() { return; }
        self.gapless_tried = None;
        // bp_clear_next drops both the PCM and DoP gapless queues; DSD fallback
        // queued via append_next onto self.sink instead, so it must take the
        // rodio branch below even with the global toggle on (on_bp_stream()
        // reflects the actual session, not just the flags).
        let bp = self.engine.as_ref().map(|e| e.on_bp_stream()).unwrap_or(false);
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

    /// Called whenever a track begins playing (explicit, auto-advance, or
    /// gapless roll-over): bump its play count and reset the per-track A-B loop.
    fn on_track_started(&mut self, path: &Path) {
        self.stats.record(path);
        self.stats.save(&moosik_dir());
        self.ab_a = None;
        self.ab_b = None;
    }

    /// Cycle the A-B repeat state: unset → set A (now) → set B (now) → clear.
    fn cycle_ab_repeat(&mut self) {
        let pos = self.elapsed();
        match (self.ab_a, self.ab_b) {
            (None, _) => { self.ab_a = Some(pos); self.ab_b = None; }
            (Some(a), None) => {
                if pos > a {
                    self.ab_b = Some(pos);
                    self.flush_gapless(); // stay on this track while looping
                } else {
                    self.ab_a = Some(pos); // B must be after A — move A instead
                }
            }
            (Some(_), Some(_)) => { self.ab_a = None; self.ab_b = None; }
        }
    }

    /// Stop on a failed session, once, and stay stopped.
    ///
    /// The reason is latched into the status line and playback ends. Nothing
    /// here reopens anything: the whole point of separating a failure from an
    /// ending is that only the user restarts a track that failed. Under
    /// Repeat One the alternative is an unbounded reopen loop on the same
    /// file, which is what a user's log showed.
    fn halt_on_failure(&mut self, reason: u8) {
        if self.play_state == PlayState::Stopped {
            return; // already halted; say it once
        }
        let why = bitperfect::fault::describe(reason);
        crate::mlog!("bp      playback stopped: {why}");
        self.play_state = PlayState::Stopped;
        self.gapless_next = None;
        self.gapless_tried = None;
        if let Some(e) = self.engine.as_mut() {
            e.halt(reason);
        }
        self.set_transient(format!("⚠ Playback stopped: {why}"));
    }

    fn next_track(&mut self) {
        match next_in_playlist(self.loop_mode, self.current_index, self.playlist.len()) {
            Some(idx) => self.play_index(idx),
            // Sequential, at the end of the list.
            None if self.loop_mode == LoopMode::Sequential && self.current_index.is_some() => {
                self.stop()
            }
            None => {}
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
        // Logged *before* each stage rather than after, so that if one of them
        // never returns the last line in the log names it. A hang leaves no
        // completion record by definition, and this path runs on the UI thread
        // — `engine.seek_to` opens and seeks a decoder here, which is the same
        // shape as the deep-FLAC stall already noted in `restart_stream`.
        let t0 = std::time::Instant::now();
        mlog!("seek    → {:.2}s: flushing gapless", target.as_secs_f64());
        // A seek reshapes the current stream — discard any gapless queue so the
        // engine and our roll-over bookkeeping can't diverge.
        self.flush_gapless();
        let dur = self.playlist[idx].duration;
        let target = dur.map_or(target, |d| target.min(d));
        let path = self.playlist[idx].path.clone();
        // The request until the engine says otherwise. On the asynchronous
        // path it stays the request — nothing has landed yet — and
        // `SeekOutcome::Landed` corrects it a frame or two later.
        let mut landed = target;
        if let Some(ref mut engine) = self.engine {
            mlog!("seek    engine.seek_to ({:.0} ms in)", t0.elapsed().as_secs_f64() * 1e3);
            // Nothing below runs on failure. Moving the spectrum and the seek
            // bar regardless is how the UI came to claim a position the decoder
            // never reached.
            match engine.seek_to(&path, target) {
                // Where the decoder is. On the bit-perfect path that is
                // `Prepared::seeked_to` — the frame the container actually
                // reached — and on a FLAC without a seek table it can be
                // seconds from the request. Publishing the request moved the
                // spectrum and the bar to a position nothing was playing from.
                Ok(at) => landed = at,
                Err(e) => {
                    mlog!("seek    failed after {:.0} ms: {e}", t0.elapsed().as_secs_f64() * 1e3);
                    // Every failing seek path now releases what it took and
                    // leaves the engine stopped, so the app stops with it. It
                    // used to stay `Playing` over an engine with no sink and
                    // no stream, which is the state where the clock keeps
                    // running and the playlist advances off the end of a track
                    // nobody is hearing.
                    self.play_state = PlayState::Stopped;
                    self.set_transient(format!("Seek failed: {e}"));
                    // The engine has put itself back; the bar and the spectrum
                    // follow it rather than staying where the pointer was.
                    let at = self.elapsed();
                    self.spectrum_window.on_seek(at.as_secs_f64());
                    self.seek_pos = dur
                        .map(|d| (at.as_secs_f32() / d.as_secs_f32().max(0.001)).clamp(0.0, 1.0))
                        .unwrap_or(0.0);
                    return;
                }
            }
        }
        mlog!("seek    spectrum on_seek ({:.0} ms in)", t0.elapsed().as_secs_f64() * 1e3);
        self.spectrum_window.on_seek(landed.as_secs_f64());
        mlog!("seek    done in {:.0} ms", t0.elapsed().as_secs_f64() * 1e3);
        self.seek_pos = dur
            .map(|d| (landed.as_secs_f32() / d.as_secs_f32().max(0.001)).clamp(0.0, 1.0))
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
    /// 1.0 (no change) when Off, when the *active* session is on the
    /// bit-perfect device stream (PCM bit-perfect, or DSD via DoP — never a
    /// gain multiply), or when no gain source is available yet (untagged
    /// track whose loudness scan hasn't finished). Deliberately checks the
    /// session's actual routing, not just the global bit-perfect toggle: a
    /// DSD track that fell back to decimated PCM plays on the ordinary rodio
    /// sink and should get RG like any other PCM track, even with 💎 on.
    fn compute_replay_gain(&self) -> f32 {
        let on_bp = self.engine.as_ref().map(|e| e.on_bp_stream()).unwrap_or(self.bit_perfect);
        if self.rg.mode == RgMode::Off || on_bp {
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
    // A test build: the private test profile root (see `spectrum::test_profile`).
    #[cfg(test)]
    return crate::spectrum::test_profile::root().join(".moosik");
    #[cfg(not(test))]
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

fn show_track_info(ui: &mut egui::Ui, t: &Track, stat: Option<PlayStat>, spectral_ceiling: Option<SpectralCeiling>, analysis: Option<&TrackAnalysis>) {
    use egui::Grid;

    let codec  = t.codec;
    let lossy  = t.lossless != Some(true);
    let fname  = t.path.file_name().and_then(|s| s.to_str()).unwrap_or("?");

    Grid::new("info_grid").num_columns(2).spacing([12.0, 4.0]).striped(true).show(ui, |ui| {
        // ── File ────────────────────────────────────────────────────────
        section(ui, "File");
        row(ui, "Filename", fname);
        row(ui, "Size",     &fmt_size(t.file_size));
        row(ui, "Format",   codec);
        row(ui, "Lossless", match t.lossless { Some(true) => "Yes", Some(false) => "No (lossy)", None => "Unknown" });

        // ── History ─────────────────────────────────────────────────────
        section(ui, "History");
        let s = stat.unwrap_or_default();
        row(ui, "Play count", &s.count.to_string());
        row(ui, "Last played", &if s.count == 0 { "Never".to_string() } else { fmt_ago(s.last) });

        // ── Stream ──────────────────────────────────────────────────────
        section(ui, "Stream");
        let is_dsd = t.bit_depth == Some(1);
        row(ui, "Sample Rate", &t.sample_rate.map(|sr| if is_dsd {
                                    format!("{} Hz  ({} — {})", sr, dsd::rate_label(sr), dsd::fmt_mhz(sr))
                                } else {
                                    format!("{} Hz  ({})", sr, fmt_hz(sr))
                                }).unwrap_or_else(|| "Unknown".into()));
        row(ui, "Bit Depth",   &t.bit_depth.map(|b| if is_dsd { format!("{b} bit  (DSD)") }
                                                    else { format!("{b} bit") })
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

        // Raw stream throughput
        if let (Some(sr), Some(ch), Some(bd)) = (t.sample_rate, t.channels, t.bit_depth) {
            let kbps = sr as u64 * ch as u64 * bd as u64 / 1000;
            row(ui, if is_dsd { "DSD throughput" } else { "PCM throughput" },
                &format!("{kbps} kbps  ({sr} × {ch} ch × {bd} bit)"));
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
            // Bit-based so 1-bit DSD doesn't truncate to zero.
            let uncompressed = sr as u64 * ch as u64 * bd as u64 * dur.as_secs() / 8;
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
        frame_log::root_pass();
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
        // Every boundary the device has already crossed, folded first.
        //
        // The evidence below is scoped to the session generation it was raised
        // in, and the rollover is what advances that generation. Polling
        // before folding meant a callback that crossed A→B and dropped out in
        // B had its dropout applied while the state still said A: the track
        // that had finished cleanly was marked as having lost its claim, and
        // the track that had actually dropped out started clean a few
        // statements later. Both surfaces were wrong, in opposite directions.
        self.fold_reached_boundaries();

        // Fold any realtime integrity fault into the session claim before
        // anything draws. A dropout revokes the diamond within a frame of
        // happening rather than at the end of the track.
        if let Some(ref mut engine) = self.engine {
            engine.bp_poll_integrity();
            let now = self.current_index
                .and_then(|i| self.playlist.get(i))
                .map(|t| t.path.clone());
            engine.poll_q31_scan(now.as_deref());
        }

        // And the cached now-playing line, which names a route that has just
        // had the chance to change under it.
        //
        // Compared on the whole headline rather than on the badge: a dropout
        // replaced by the backend write failure that ended the track is two
        // different sentences under one `Faulted` badge, and comparing badges
        // left the first one on screen. And only while playing — the refresh
        // used to fire regardless, so a stopped player got "Playing: …" put
        // back on the bar by the next tick.
        if let Some(idx) = self.current_index {
            let live = self
                .engine
                .as_ref()
                .map(|e| e.bp_state.presentation().headline)
                .unwrap_or_default();
            let playing = self.play_state != PlayState::Stopped;
            let rendered = self.now_playing_line(idx);
            self.status.refresh(&live, playing, || rendered);
        }

        let spectrum_open = self.spectrum_window.open;
        let seeking = self.engine.as_ref().is_some_and(|e| e.is_seeking());
        let animating = self.play_state == PlayState::Playing
            || seeking
            // Owned or waiting, cancelled or not: the spectrum's tick has to keep
            // running until a finished job is consumed and the next one started,
            // or a request made while paused would never begin.
            || self.spectrum_window.analyzer.analysis_active();
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
            // The keyboard goes through the same predicate as the slider. It
            // used to write `self.volume` first and ask afterwards, so an
            // exact route silently lost the user's saved level.
            if arrow_up || arrow_down {
                let step = if arrow_up { 0.05 } else { -0.05 };
                let want = (self.volume + step).clamp(0.0, 1.0);
                let applied = self.engine.as_mut().is_some_and(|e| e.set_volume(want));
                if applied {
                    self.volume = want;
                } else {
                    let why = self.engine.as_ref()
                        .and_then(|e| e.bp_state.volume_lock_reason())
                        .unwrap_or_else(|| "Volume is locked on this route".into());
                    self.set_transient(why);
                }
            }
        }

        // --- collect finished bit-perfect device scan ---
        if let Some(ref rx) = self.bp_scan_rx
            && let Ok(devices) = rx.try_recv() {
            self.bp_devices = Some(devices);
            self.bp_scan_rx = None;
        }

        // The EQ is edited in the spectrum window, which has no way to call
        // back into the engine. Recomputing the published chain here costs two
        // atomics and a `try_lock` per frame and keeps the panel describing
        // what is in the path *now* rather than what was in it when the track
        // started.
        self.on_eq_changed();

        // --- install a completed background seek, if any ---
        let seek = self
            .engine
            .as_mut()
            .map(|e| e.poll_pending_seek())
            .unwrap_or(SeekOutcome::Idle);
        // The status line, through the one function that decides it, so that
        // "put the notice up" and "take it down again" cannot drift apart.
        if !matches!(seek, SeekOutcome::Idle) {
            let (resumed_line, resumed_headline) = match self.current_index {
                Some(i) => self.now_playing_pair(i),
                None => (String::new(), String::new()),
            };
            let mut notice = self.seek_notice;
            let line = seek_status_line(&seek, &mut notice, || resumed_line);
            self.seek_notice = notice;
            match seek {
                // A landing puts the now-playing line back — but only if it
                // had put a notice up to replace. A seek fast enough to land
                // before "Seeking to …" is ever shown produces no line here,
                // and taking ownership anyway handed the app whatever message
                // the listener was looking at, to be regenerated away on the
                // next tick.
                SeekOutcome::Landed(_) => {
                    self.status
                        .landed(line.is_some(), line.unwrap_or_default(), resumed_headline);
                }
                // A failure, and a provisional notice, are both things the
                // listener has just been told.
                SeekOutcome::Failed(_) | SeekOutcome::Provisional(_) => {
                    if let Some(line) = line {
                        self.set_transient(line);
                    }
                }
                // Nothing to say, and nothing changes hands.
                SeekOutcome::Idle | SeekOutcome::Working => {
                    debug_assert!(line.is_none());
                }
            }
        }
        match seek {
            SeekOutcome::Idle | SeekOutcome::Working | SeekOutcome::Provisional(_) => {}
            SeekOutcome::Landed(at) => {
                // The spectrum follows where the decoder actually landed, not
                // where the slider was dragged to — and so does the bar, which
                // was left showing the request. The source that landed is the
                // one now feeding the tap.
                let path = self.current_track_path();
                restart_spectrum(
                    self.engine.as_ref(),
                    &mut self.spectrum_window,
                    path.as_deref(),
                    Some(at),
                );
                self.seek_pos = self
                    .current_index
                    .and_then(|i| self.playlist.get(i))
                    .and_then(|t| t.duration)
                    .map(|d| (at.as_secs_f32() / d.as_secs_f32().max(0.001)).clamp(0.0, 1.0))
                    .unwrap_or(self.seek_pos);
            }
            SeekOutcome::Failed(why) => {
                // Said once, and playback stops rather than sitting silent
                // with the old position still on the slider. The engine has
                // already been put back where it was; every surface follows it
                // there, including the two that used to be left describing the
                // position the pointer was dropped on.
                crate::mlog!("seek failed: {}", why.describe());
                self.play_state = async_seek_failure_state();
                let at = self.elapsed();
                // Nothing is open after an abandoned seek: the tap goes too.
                let path = self.current_track_path();
                restart_spectrum(
                    self.engine.as_ref(),
                    &mut self.spectrum_window,
                    path.as_deref(),
                    Some(at),
                );
                self.seek_pos = self
                    .current_index
                    .and_then(|i| self.playlist.get(i))
                    .and_then(|t| t.duration)
                    .map(|d| (at.as_secs_f32() / d.as_secs_f32().max(0.001)).clamp(0.0, 1.0))
                    .unwrap_or(0.0);
            }
        }

        // --- gapless roll-over + auto-advance when a track finishes ---
        if self.play_state == PlayState::Playing {
            // A-B repeat keeps playback inside the current track: loop back to A
            // at B, and skip all advance / gapless logic while it's active.
            if let (Some(a), Some(b)) = (self.ab_a, self.ab_b) {
                if self.elapsed() >= b {
                    self.seek_to(a);
                }
            } else {
                // Bit-perfect: the device reports exact frame boundaries as it
                // plays through gaplessly-chained tracks (possibly several per
                // frame). Anything crossed since the fold at the top of this
                // tick is picked up here; the next tick folds before it polls,
                // so no boundary is ever read past.
                self.fold_reached_boundaries();
                // Normal mode: no per-source callback, so detect the boundary by time.
                // Excludes real bp/DoP sessions: those report frame-exact
                // boundaries above (bp_poll_boundary), and letting the clock
                // fire first would roll the UI over before the position base
                // advances (early title switch + position glitch). DSD fallback
                // is NOT excluded — on_bp_stream() is false for it even with
                // the global 💎 toggle on, since it's really on the rodio path
                // and the clock is the only boundary signal it has.
                // Timed by the track's declared duration, not observed.
                //
                // `rodio` reports nothing when one appended source ends and
                // the next begins, so on this route there is no boundary to
                // detect — only a clock and a number from a tag. A wrong or
                // missing duration moves the title, the spectrum and the
                // published source at the wrong moment, and there is nothing
                // here that could notice. The bit-perfect path is different in
                // kind: it counts frames the device has actually taken and the
                // render thread crosses at the sample.
                let normal_crossed = self.gapless_next.is_some()
                    && self.engine.as_ref().map(|e| !e.on_bp_stream()).unwrap_or(false)
                    && self.current_index
                        .and_then(|i| self.playlist.get(i)).and_then(|t| t.duration)
                        .map(|d| self.elapsed() >= d).unwrap_or(false);
                if normal_crossed {
                    self.gapless_rollover();
                }

                // A stream that reached the end of its source advances —
                // unless the sleep timer is set to stop at the end of this
                // track. A stream that *failed* stops, once, visibly, and is
                // never reopened by anything but the user: this is the gate
                // that turned a deterministically failing file into hundreds
                // of consecutive reopens under Repeat One.
                let completion = self
                    .engine
                    .as_ref()
                    .map(|e| e.completion())
                    .unwrap_or(bitperfect::Completion::Running);
                match bitperfect::on_completion(completion, self.sleep_end_of_track) {
                    bitperfect::TickAction::Nothing => {}
                    bitperfect::TickAction::Advance => self.next_track(),
                    bitperfect::TickAction::StopAtEndOfTrack => {
                        self.sleep_end_of_track = false;
                        self.stop();
                    }
                    bitperfect::TickAction::Halt(reason) => self.halt_on_failure(reason),
                }

                // Prebuffer the next track once we're near the end of this one.
                // Suppressed when we'll stop at the end anyway.
                // Nothing to prebuffer onto a session that is already over —
                // and on a failed one, arming a successor is how the same
                // failure gets a second chance to look like progress.
                let over = self.engine.as_ref().map(|e| e.is_finished()).unwrap_or(true);
                if !self.sleep_end_of_track && !over {
                    self.try_arm_gapless();
                }
            }
        }

        // --- a backend that dies while paused still dies ---
        //
        // Everything above is inside `play_state == Playing`, which is right
        // for advancing a playlist and wrong for noticing a failure. A paused
        // session still holds an exclusive device, and a driver reset or a
        // dead render thread while paused left the player sitting on it: the
        // failure was recorded, nothing read it, and the handle was released
        // only when the listener happened to press play. Auto-advance stays
        // where it is — a paused track has not ended — but a failure is not an
        // ending and does not wait for one.
        if self.play_state == PlayState::Paused {
            let completion = self
                .engine
                .as_ref()
                .map(|e| e.completion())
                .unwrap_or(bitperfect::Completion::Running);
            match paused_completion(completion) {
                PausedTick::Nothing => {}
                PausedTick::Halt(reason) => self.halt_on_failure(reason),
            }
        }

        // --- Sleep timer (fixed deadline) ---
        if let Some(dl) = self.sleep_deadline
            && Instant::now() >= dl {
            self.sleep_deadline = None;
            if self.play_state == PlayState::Playing {
                self.toggle_play_pause(); // pause
            }
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
        // Whether EQ is actually out of the current audio path — drives the
        // "is the plotted spectrum EQ-modified" overlay logic. Pushed every
        // frame (not just on toggle) so it tracks the *active session*, not
        // the global bit-perfect toggle: a DSD track played via DoP has EQ
        // bypassed regardless of that toggle, while a DSD track that fell
        // back to PCM has EQ applied even with the toggle on.
        self.spectrum_window.bit_perfect =
            self.engine.as_ref().map(|e| e.on_bp_stream()).unwrap_or(self.bit_perfect);
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
        // A click in the whole-track view seeks through the same path as the
        // seek bar and the arrow keys, and only with a current track.
        if let Some(t) = self.spectrum_window.take_seek_request()
            && self.current_index.is_some()
            && t.is_finite()
        {
            self.seek_to(Duration::from_secs_f64(t.max(0.0)));
        }
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
                        show_track_info(ui, &track, self.stats.get(&track.path).cloned(), self.spectrum_window.spectral_ceiling.clone(), self.spectrum_window.track_analysis.as_ref());
                    });
                });
            });
            if close { self.info_open = false; }
        }

        // ---------------------------------------------------------------
        // Lyrics window (own viewport, so it can live on a second screen)
        // ---------------------------------------------------------------
        if self.lyrics_window.is_open() {
            let pos = self.elapsed();
            let tref = self.current_index.map(|i| {
                let t = &self.playlist[i];
                lyrics::ui::TrackRef {
                    path: &t.path, title: &t.title, artist: &t.artist,
                    album: &t.album, duration: t.duration,
                }
            });
            // The window never touches playback; it reports what it wants.
            if let lyrics::ui::LyricsAction::Seek(to) =
                self.lyrics_window.ui(ctx, tref, pos)
            {
                self.seek_to(to);
            }
            // Its persisted preferences, written when they change — the text
            // size once a slider drag is let go, not once per step of it.
            let size = lyrics_text_size_to_save(
                self.lyrics_window.text_size(),
                self.appearance.lyrics_text_size,
                self.lyrics_window.text_size_settled(),
            );
            if self.lyrics_window.translation_display != self.appearance.lyrics_translation || size.is_some() {
                self.appearance.lyrics_translation = self.lyrics_window.translation_display;
                if let Some(s) = size {
                    self.appearance.lyrics_text_size = s;
                }
                save_appearance(&moosik_dir(), &self.appearance);
            }
        }

        // ---------------------------------------------------------------
        // Tag editor (own viewport)
        // ---------------------------------------------------------------
        if self.tag_window.open
            && let Some(i) = self.current_index
        {
            let (path, title) = (self.playlist[i].path.clone(), self.playlist[i].title.clone());
            if self.tag_window.ui(ctx, Some(&path), &title) {
                // Tags changed on disk — the playlist row is now stale.
                self.playlist[i] = Track::load(path);
            }
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

            // A-B repeat markers on the track.
            if let Some(dur) = total {
                let ds = dur.as_secs_f32().max(0.001);
                let mark = |pos: Duration, label: &str, col: Color32| {
                    let t = (pos.as_secs_f32() / ds).clamp(0.0, 1.0);
                    let x = track_x0 + t * (track_x1 - track_x0);
                    painter.line_segment(
                        [egui::Pos2::new(x, row_rect.top() + 3.0),
                         egui::Pos2::new(x, row_rect.bottom() - 3.0)],
                        egui::Stroke::new(2.0, col),
                    );
                    painter.text(egui::Pos2::new(x, row_rect.top() + 1.0),
                        egui::Align2::CENTER_TOP, label, egui::FontId::monospace(9.0), col);
                };
                let ab_col = pal::amber(dark);
                if let Some(a) = self.ab_a { mark(a, "A", ab_col); }
                if let Some(b) = self.ab_b { mark(b, "B", ab_col); }
            }

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

                ui.add_space(8.0);

                // ── A-B repeat ───────────────────────────────────────────
                let (ab_label, ab_on) = match (self.ab_a, self.ab_b) {
                    (None, _)          => ("A–B", false),
                    (Some(_), None)    => ("A‥",  true),
                    (Some(_), Some(_)) => ("A↔B", true),
                };
                let ab_txt = if ab_on {
                    RichText::new(ab_label).size(15.0).color(pal::accent(ui.visuals().dark_mode))
                } else {
                    RichText::new(ab_label).size(15.0)
                };
                let ab_tip = match (self.ab_a, self.ab_b) {
                    (None, _) => "A-B repeat — click to set point A".to_string(),
                    (Some(a), None) => format!("A = {} — click to set B (or clear)",
                        Self::format_duration(a)),
                    (Some(a), Some(b)) => format!("Looping {}–{} — click to clear",
                        Self::format_duration(a), Self::format_duration(b)),
                };
                if ui.add_enabled(self.current_index.is_some(),
                    egui::Button::new(ab_txt).min_size(Vec2::new(46.0, 36.0)))
                    .on_hover_text(ab_tip).clicked() {
                    self.cycle_ab_repeat();
                }

                // ── Bookmarks ────────────────────────────────────────────
                ui.menu_button(RichText::new("🔖").size(16.0), |ui| {
                    ui.set_min_width(200.0);
                    let cur = self.current_index.map(|i| self.playlist[i].path.clone());
                    if ui.add_enabled(cur.is_some(),
                        egui::Button::new("＋ Bookmark current position")).clicked() {
                        if let Some(ref p) = cur {
                            self.bookmarks.add(p, self.elapsed().as_secs_f32());
                            self.bookmarks.save(&moosik_dir());
                        }
                        ui.close_menu();
                    }
                    if let Some(ref p) = cur
                        && let Some(marks) = self.bookmarks.for_track(p).cloned()
                        && !marks.is_empty() {
                        ui.separator();
                        let mut jump: Option<f32> = None;
                        let mut remove: Option<f32> = None;
                        for m in marks {
                            ui.horizontal(|ui| {
                                if ui.button(format!("↪ {}",
                                    Self::format_duration(Duration::from_secs_f32(m)))).clicked() {
                                    jump = Some(m);
                                }
                                if ui.small_button("✕").on_hover_text("Remove").clicked() {
                                    remove = Some(m);
                                }
                            });
                        }
                        if let Some(m) = jump {
                            self.seek_to(Duration::from_secs_f32(m));
                            ui.close_menu();
                        }
                        if let Some(m) = remove {
                            self.bookmarks.remove(p, m);
                            self.bookmarks.save(&moosik_dir());
                        }
                    }
                }).response.on_hover_text("Bookmarks for the current track");

                // ── Sleep timer ──────────────────────────────────────────
                let sleep_on = self.sleep_deadline.is_some() || self.sleep_end_of_track;
                let sleep_txt = if sleep_on {
                    RichText::new("💤").size(16.0).color(pal::accent(ui.visuals().dark_mode))
                } else {
                    RichText::new("💤").size(16.0)
                };
                let sleep_tip = if let Some(dl) = self.sleep_deadline {
                    format!("Sleep in {}", Self::format_duration(dl.saturating_duration_since(Instant::now())))
                } else if self.sleep_end_of_track {
                    "Sleep at end of track".to_string()
                } else {
                    "Sleep timer".to_string()
                };
                ui.menu_button(sleep_txt, |ui| {
                    ui.set_min_width(150.0);
                    if ui.selectable_label(!sleep_on, "Off").clicked() {
                        self.sleep_deadline = None;
                        self.sleep_end_of_track = false;
                        ui.close_menu();
                    }
                    for mins in [15u64, 30, 45, 60, 90] {
                        if ui.selectable_label(false, format!("{mins} minutes")).clicked() {
                            self.sleep_deadline = Some(Instant::now() + Duration::from_secs(mins * 60));
                            self.sleep_end_of_track = false;
                            ui.close_menu();
                        }
                    }
                    if ui.selectable_label(self.sleep_end_of_track, "End of track").clicked() {
                        self.sleep_end_of_track = true;
                        self.sleep_deadline = None;
                        ui.close_menu();
                    }
                    if let Some(dl) = self.sleep_deadline {
                        ui.separator();
                        ui.label(RichText::new(format!("⏳ {} left",
                            Self::format_duration(dl.saturating_duration_since(Instant::now()))))
                            .size(11.0).color(pal::text_dim(ui.visuals().dark_mode)));
                    }
                }).response.on_hover_text(sleep_tip);

                ui.add_space(20.0);

                // Volume.
                //
                // Disabled outright while an exact route is open. There is no
                // gain on that path to move — no multiply exists between the
                // decoder and the device — so a slider that appeared to work
                // was reporting a state the audio could not be in. The saved
                // preference is untouched and comes back the moment normal
                // playback resumes.
                let lock = self.engine.as_ref()
                    .map(|e| (e.bp_state.volume_locked(), e.bp_state.volume_lock_reason()))
                    .unwrap_or((false, None));
                let (vol_locked, lock_reason) = lock;
                ui.label(RichText::new("🔊").size(16.0));
                let mut shown = if vol_locked { 1.0 } else { self.volume };
                let vol_slider = ui.add_enabled(
                    !vol_locked,
                    Slider::new(&mut shown, 0.0..=1.0)
                        .show_value(false)
                        .trailing_fill(true),
                );
                if vol_locked {
                    vol_slider.on_disabled_hover_text(
                        lock_reason.unwrap_or_else(|| "Volume is locked on this route".into()));
                } else if vol_slider.changed() {
                    // `set_volume` owns the predicate and refuses if the route
                    // has no gain stage, so the saved value cannot be moved by
                    // a control that would have no effect.
                    let applied = self.engine.as_mut().is_some_and(|e| e.set_volume(shown));
                    if applied {
                        self.volume = shown;
                    }
                }
                let dsd_active = self.engine.as_ref().is_some_and(|e| e.dsd_mode || e.dsd_native);
                ui.label(
                    RichText::new(if vol_locked {
                                      "100% · locked on this route".to_string()
                                  } else {
                                      format!("{}%", (self.volume * 100.0) as u32)
                                  })
                        .size(12.0)
                        .color(pal::text_dim(ui.visuals().dark_mode)),
                );

                // Sample rate + PCM bitrate indicator (or DSD/DoP equivalent)
                if self.play_state != PlayState::Stopped {
                    let sr = self.engine.as_ref().map(|e| e.last_sample_rate).unwrap_or(0);
                    let dsd_fallback = self.engine.as_ref().is_some_and(|e| e.dsd_fallback);
                    if dsd_active || dsd_fallback {
                        let dsd_label = self.engine.as_ref()
                            .and_then(|e| e.dsd_label.clone())
                            .unwrap_or_else(|| "DSD".to_string());
                        let native = self.engine.as_ref().is_some_and(|e| e.dsd_native);
                        let text = if native {
                            let backend = if cfg!(windows) { "ASIO" } else { "ALSA" };
                            format!("{dsd_label} native ({backend})")
                        } else if dsd_active {
                            format!("{dsd_label} via DoP · {}", fmt_hz(sr))
                        } else {
                            format!("{dsd_label} → {} PCM (no DoP)", fmt_hz(sr))
                        };
                        ui.add_space(8.0);
                        ui.label(RichText::new(text)
                            .size(11.0).color(pal::text_dim(ui.visuals().dark_mode)));
                    } else if sr > 0 {
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
                    let lyr_label = if self.lyrics_window.is_open() {
                        RichText::new("🎤 Lyrics").size(13.0).color(pal::accent(ui.visuals().dark_mode))
                    } else {
                        RichText::new("🎤 Lyrics").size(13.0)
                    };
                    if ui.add_enabled(info_enabled, egui::Button::new(lyr_label))
                        .on_hover_text(
                            "Lyrics for the playing track.\n\nReads a .lrc beside the file \
                             or the file's own tags, and can look the track up on LRCLIB. \
                             Saved lyrics go to a .lrc — the audio file is never modified.")
                        .clicked()
                    {
                        self.lyrics_window.toggle();
                    }
                    ui.add_space(8.0);
                    let tag_label = if self.tag_window.open {
                        RichText::new("🏷 Tags").size(13.0).color(pal::accent(ui.visuals().dark_mode))
                    } else {
                        RichText::new("🏷 Tags").size(13.0)
                    };
                    if ui.add_enabled(info_enabled, egui::Button::new(tag_label))
                        .on_hover_text(
                            "Edit the playing track's tags, and see every tag it carries.\n\n\
                             Writes go to a copy which then replaces the original, and are \
                             read back and checked. DSD files are read-only here.")
                        .clicked()
                    {
                        self.tag_window.open = !self.tag_window.open;
                    }
                    ui.add_space(8.0);

                    // ── Bit-perfect: device picker (▾) + toggle ─────────────
                    // right_to_left layout: the toggle is added first so it
                    // sits to the right of the picker.
                    // Every visible surface derives from one presentation
                    // object. The glyph and the words carry the state as well
                    // as the colour does, so it survives a monochrome
                    // screenshot and a colour-blind reader — and "requested"
                    // can never render as "achieved".
                    let mut present = self.engine.as_ref()
                        .map(|e| e.bp_state.presentation())
                        .unwrap_or_else(|| {
                            bitperfect::state::OutputSessionState::default().presentation()
                        });
                    // The backend's own account of the open stream — driver
                    // name, dropout count — as one more *detail* line. It
                    // answers "what is this stream", never "is it exact", and
                    // was previously the source of four hard-coded diamonds
                    // because those two questions were not separated.
                    if let Some(desc) = self.engine.as_ref().and_then(|e| e.bp_describe()) {
                        present.detail.push(desc);
                    }
                    let dark = ui.visuals().dark_mode;
                    let tint = match present.badge {
                        bitperfect::state::Badge::PayloadExact => Some(pal::ok(dark)),
                        bitperfect::state::Badge::Pending => Some(pal::text_dim(dark)),
                        bitperfect::state::Badge::ValueExact => Some(pal::accent(dark)),
                        bitperfect::state::Badge::Processed
                        | bitperfect::state::Badge::Unverified
                        | bitperfect::state::Badge::Faulted
                        | bitperfect::state::Badge::Failed => Some(pal::warn(dark)),
                        bitperfect::state::Badge::Off
                        | bitperfect::state::Badge::Requested => None,
                    };
                    let caption = format!("{} Bit-Perfect", present.badge.glyph());
                    let bp_label = match tint {
                        Some(c) => RichText::new(caption).size(13.0).color(c),
                        None => RichText::new(caption).size(13.0),
                    };

                    let mut hover = present.headline.clone();
                    for line in &present.detail {
                        hover.push('\n');
                        hover.push_str(line);
                    }
                    if dsd_active {
                        hover.push_str(
                            "\n\nDSD always takes a dedicated route — this toggle only affects \
                             PCM tracks.");
                    }
                    hover.push_str(
                        "\n\nExactness is claimed only as far as the audio driver. What the \
                         USB link and the DAC do after that is not observable from here.");

                    if ui.add_enabled(!dsd_active, egui::Button::new(bp_label))
                        .on_hover_text(hover).clicked() {
                        self.toggle_bit_perfect();
                    }

                    ui.menu_button(RichText::new("🔈▾").size(13.0), |ui| {
                        ui.set_min_width(320.0);
                        // What to do when an exact route is impossible. The
                        // choice is the user's, it is per-track in effect, and
                        // it never changes the bit-perfect request itself.
                        ui.label(RichText::new("When exact output is impossible")
                            .strong().size(12.0));
                        let current = self.engine.as_ref()
                            .map(|e| e.bp_state.policy)
                            .unwrap_or_default();
                        let mut chosen = current;
                        for p in [bitperfect::state::OutputPolicy::StrictExact,
                                  bitperfect::state::OutputPolicy::PreferExact,
                                  bitperfect::state::OutputPolicy::HqProcessed] {
                            ui.radio_value(&mut chosen, p, p.label())
                                .on_hover_text(p.describe());
                        }
                        if chosen != current {
                            if let Some(ref mut e) = self.engine {
                                // A policy change is an explicit user action and
                                // is persisted; it starts applying at the next
                                // track rather than reopening under playback.
                                e.bp_state.policy = chosen;
                            }
                            self.save_bp_settings();
                        }
                        ui.separator();

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

                        // ── Native DSD via ASIO (asio-dsd builds) ─────────
                        #[cfg(all(windows, feature = "asio-dsd"))]
                        {
                            ui.separator();
                            ui.label(RichText::new("Native DSD (ASIO)").strong().size(12.0));
                            let current = self.engine.as_ref().and_then(|e| e.asio_driver.clone());
                            let mut pick_asio: Option<Option<String>> = None;
                            if ui.selectable_label(current.is_none(), "Off — DSD plays via DoP")
                                .clicked() {
                                pick_asio = Some(None);
                                ui.close_menu();
                            }
                            let drivers = bitperfect::asio_dsd::list_asio_drivers();
                            if drivers.is_empty() {
                                ui.label(RichText::new("No ASIO drivers installed")
                                    .size(11.0).color(pal::text_dim(ui.visuals().dark_mode)));
                            }
                            let mut probe: Option<String> = None;
                            for d in &drivers {
                                let sel = current.as_deref() == Some(d.as_str());
                                if ui.selectable_label(sel, d)
                                    .on_hover_text("Route DSD tracks as raw native DSD through \
                                                    this ASIO driver — bit-perfect up to DSD512, \
                                                    no DoP carrier-rate ceiling.")
                                    .clicked() {
                                    pick_asio = Some(Some(d.clone()));
                                    ui.close_menu();
                                }
                            }

                            // A diagnostic, and labelled as one.
                            //
                            // Moosik does not play PCM over ASIO: there is no
                            // renderer, so there is no setting, no transport
                            // and no claim. What there is, is an answer to
                            // "what would this driver take?" that comes from
                            // the driver rather than from guesswork — which is
                            // the question that has to be settled before a
                            // renderer is worth writing. It opens the driver,
                            // asks, and closes it again.
                            if !drivers.is_empty() {
                                ui.separator();
                                ui.label(
                                    RichText::new("PCM capability probe (diagnostic)")
                                        .size(11.0)
                                        .color(pal::text_dim(ui.visuals().dark_mode)),
                                );
                                for d in &drivers {
                                    if ui
                                        .small_button(format!("Probe {d}"))
                                        .on_hover_text(
                                            "Ask this driver what it can do with ordinary PCM \
                                             and write the answer to the log.\n\nMoosik does \
                                             not play PCM over ASIO — this changes nothing \
                                             about playback.",
                                        )
                                        .clicked()
                                    {
                                        probe = Some(d.clone());
                                        ui.close_menu();
                                    }
                                }
                            }
                            if let Some(d) = probe {
                                // A probe result is a message, not a claim
                                // about what is playing. Written straight to
                                // the text, it inherited whatever owner was
                                // there — including the now-playing line's —
                                // and the next tick regenerated it away.
                                let line = match bitperfect::asio_dsd::probe_pcm(&d) {
                                        Ok(caps) => {
                                            let line = caps.describe();
                                            crate::mlog!("[asio-pcm] {line}");
                                            format!("ASIO PCM probe — {line}")
                                        }
                                        Err(e) => {
                                            crate::mlog!("[asio-pcm] probe failed: {e}");
                                            format!("ASIO PCM probe failed: {e}")
                                        }
                                };
                                self.set_transient(line);
                            }
                            if let Some(sel) = pick_asio {
                                let old = self.engine.as_ref()
                                    .and_then(|e| e.asio_driver.clone());
                                if let Some(e) = self.engine.as_mut() { e.asio_driver = sel.clone(); }
                                self.save_bp_settings();
                                self.set_transient(match &sel {
                                    Some(n) => format!("Native DSD via ASIO: {n}"),
                                    None => "DSD output: DoP".to_string(),
                                });
                                self.switch_native_output("Native DSD", move |e| {
                                    e.asio_driver = old;
                                });
                            }
                        }

                        // ── Native DSD via ALSA (alsa-dsd builds) ─────────
                        #[cfg(all(target_os = "linux", feature = "alsa-dsd"))]
                        {
                            ui.separator();
                            ui.label(RichText::new("Native DSD (ALSA)").strong().size(12.0));
                            let current = self.engine.as_ref().and_then(|e| e.alsa_dsd_device.clone());
                            let mut pick_alsa: Option<Option<String>> = None;
                            if ui.selectable_label(current.is_none(), "Off — DSD plays via DoP")
                                .clicked() {
                                pick_alsa = Some(None);
                                ui.close_menu();
                            }
                            let devices = bitperfect::alsa_dsd::list_dsd_devices();
                            if devices.is_empty() {
                                ui.label(RichText::new("No direct hw: playback devices found")
                                    .size(11.0).color(pal::text_dim(ui.visuals().dark_mode)));
                            }
                            for d in devices {
                                let sel = current.as_deref() == Some(d.id.as_str());
                                let resp = ui.selectable_label(sel, &d.id)
                                    .on_hover_text("Route DSD tracks as raw native DSD \
                                                    (DSD_U32_BE/U16/U8) through this device — \
                                                    no DoP carrier-rate ceiling. Needs a card \
                                                    whose driver advertises DSD formats, and \
                                                    exclusive hw: access.");
                                if !d.desc.is_empty() {
                                    ui.label(RichText::new(&d.desc).size(10.0)
                                        .color(pal::text_faint(ui.visuals().dark_mode)));
                                }
                                if resp.clicked() {
                                    pick_alsa = Some(Some(d.id.clone()));
                                    ui.close_menu();
                                }
                            }
                            if ui.button(RichText::new("⟳ Rescan").size(12.0)).clicked() {
                                bitperfect::alsa_dsd::rescan_devices();
                            }
                            if let Some(sel) = pick_alsa {
                                let old = self.engine.as_ref()
                                    .and_then(|e| e.alsa_dsd_device.clone());
                                if let Some(e) = self.engine.as_mut() { e.alsa_dsd_device = sel.clone(); }
                                self.save_bp_settings();
                                self.set_transient(match &sel {
                                    Some(n) => format!("Native DSD via ALSA: {n}"),
                                    None => "DSD output: DoP".to_string(),
                                });
                                self.switch_native_output("Native DSD", move |e| {
                                    e.alsa_dsd_device = old;
                                });
                            }
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
                    // Reflects the *active session's* routing, not just the global
                    // toggle — a DSD track that fell back to PCM plays on the
                    // ordinary sink and gets RG even with 💎 on for PCM tracks.
                    let on_bp_stream = self.engine.as_ref().is_some_and(|e| e.on_bp_stream());
                    let rg_hover = if on_bp_stream && rg_active {
                        "ReplayGain is bypassed on the bit-perfect device stream — a gain change breaks bit-perfectness.".to_string()
                    } else if rg_active {
                        format!("Loudness normalization: {} · currently {:+.1} dB", self.rg.mode.label(), self.rg_applied_db())
                    } else {
                        "Loudness normalization (off) — plays original levels".to_string()
                    };
                    ui.menu_button(rg_label, |ui| {
                        ui.set_min_width(250.0);
                        ui.label(RichText::new("ReplayGain — loudness normalization").strong().size(12.0));
                        // A device stream has no gain stage, which is a fact
                        // about the *route* and true whether or not the route
                        // is exact. The diamond is a fact about the
                        // *fidelity*, so this line states the first and lets
                        // the presentation decide the second.
                        if on_bp_stream {
                            let exact = self
                                .engine
                                .as_ref()
                                .is_some_and(|e| e.bp_state.shows_diamond());
                            let (glyph, colour) = if exact {
                                ("\u{1F48E}", pal::ok(ui.visuals().dark_mode))
                            } else {
                                ("\u{25C7}", pal::text_dim(ui.visuals().dark_mode))
                            };
                            ui.label(
                                RichText::new(format!(
                                    "{glyph} Bypassed on the output device stream"
                                ))
                                .size(11.0)
                                .color(colour),
                            );
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
                    // ── Session log ─────────────────────────────────────────
                    // Reachable without a console, because the build that most
                    // needs it is the one someone double-clicked. A bug report
                    // that can attach this file is worth more than any amount
                    // of asking what happened.
                    // Gated on a file actually existing, not on the absence of
                    // a recorded error — those are only the same thing while
                    // every failure path remembers to record one.
                    let log_ok = log::path().is_some();
                    // Two tooltip paths, not one: egui only shows
                    // `on_hover_text` on an *enabled* widget, so hanging the
                    // failure reason there meant the disabled button explained
                    // nothing at all.
                    let log_tip = match log::path() {
                        Some(p) => format!("Open this session's log
{}", p.display()),
                        None => "Open the log folder".to_string(),
                    };
                    let log_why = match log::error() {
                        Some(e) => format!("No log file this session — {e}"),
                        None => "No log file this session".to_string(),
                    };
                    if ui.add_enabled(log_ok, egui::Button::new(RichText::new("🗎 Log").size(13.0)))
                        .on_hover_text(log_tip)
                        .on_disabled_hover_text(log_why)
                        .clicked()
                    {
                        let dir = log::dir();
                        let _ = std::fs::create_dir_all(&dir);
                        #[cfg(windows)]
                        // Never while a local translation runtime's pipe ends are
                        // inheritable, or Explorer could hold them open.
                        let _ = crate::lyrics::local_ai::contain::ordinary_spawn(
                            std::process::Command::new("explorer").arg(&dir));
                        #[cfg(target_os = "macos")]
                        let _ = std::process::Command::new("open").arg(&dir).spawn();
                        #[cfg(all(unix, not(target_os = "macos")))]
                        let _ = std::process::Command::new("xdg-open").arg(&dir).spawn();
                    }

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
                        ui.label(RichText::new("Font")
                            .size(11.0).color(pal::text_dim(ui.visuals().dark_mode)));
                        // Enumerating every system font means reading a few
                        // hundred files, so it happens once, when the menu is
                        // first opened, rather than every frame.
                        if self.font_list.is_empty() { self.font_list = fonts::list(); }
                        let current = self.appearance.ui_font.clone();
                        let mut pick: Option<Option<String>> = None;
                        // A plain scrolling list, not a ComboBox. A ComboBox
                        // popup is its own egui Area, so a click inside it reads
                        // as a click *outside* this menu — the menu closes and
                        // swallows the click, and the font silently never
                        // changed. Widgets placed directly in the menu do not
                        // have that problem.
                        ui.horizontal(|ui| {
                            ui.add(egui::TextEdit::singleline(&mut self.font_filter)
                                   .desired_width(150.0).hint_text("filter…"));
                            if ui.small_button("✖").clicked() { self.font_filter.clear(); }
                        });
                        egui::ScrollArea::vertical().max_height(190.0)
                            .id_salt("ui_font_list")
                            .auto_shrink([false, false])
                            .show(ui, |ui| {
                                if ui.selectable_label(current.is_none(), "Default").clicked() {
                                    pick = Some(None);
                                }
                                ui.separator();
                                let needle = self.font_filter.to_lowercase();
                                for f in &self.font_list {
                                    if !needle.is_empty()
                                        && !f.family.to_lowercase().contains(&needle) { continue; }
                                    let sel = current.as_deref() == Some(f.family.as_str());
                                    if ui.selectable_label(sel, &f.family).clicked() {
                                        pick = Some(Some(f.family.clone()));
                                    }
                                }
                            });
                        if let Some(choice) = pick {
                            self.appearance.ui_font = choice;
                            // Rebuild the whole set: the CJK fallbacks have to be
                            // reinstalled behind the new primary, or Japanese
                            // tags turn to tofu the moment a Latin font is picked.
                            setup_fonts(ctx, self.appearance.ui_font.as_deref());
                            changed = true;
                        }
                        ui.label(RichText::new(
                            "CJK fallbacks stay active whatever you pick")
                            .size(10.0).color(pal::text_faint(ui.visuals().dark_mode)));

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

                        ui.add_space(6.0);
                        ui.separator();
                        if ui.checkbox(&mut self.appearance.show_play_count,
                            "Show play counts in playlist").changed() {
                            changed = true;
                        }

                        if changed {
                            save_appearance(&moosik_dir(), &self.appearance);
                        }
                    }).response.on_hover_text("Palette, text size, and accent — all optional");
                });
            });

            // The negotiated output, kept where a transient status message
            // cannot reach it.
            //
            // Until 1.4.3 there was one `status_msg` and everything wrote to
            // it, so "playing at 24-bit through WASAPI exclusive" survived
            // until the next tag save or cache eviction wanted to say something
            // — and the state that matters most is exactly the one you want to
            // still be able to read a minute later.
            let present = self.engine.as_ref().map(|e| e.bp_state.presentation());
            if let Some(present) = present
                && self.play_state != PlayState::Stopped
            {
                let dark = ui.visuals().dark_mode;
                let head = match present.badge {
                    bitperfect::state::Badge::PayloadExact => pal::ok(dark),
                    bitperfect::state::Badge::ValueExact => pal::accent(dark),
                    bitperfect::state::Badge::Processed
                    | bitperfect::state::Badge::Unverified
                    | bitperfect::state::Badge::Faulted
                    | bitperfect::state::Badge::Failed => pal::warn(dark),
                    _ => pal::text_dim(dark),
                };
                ui.add_space(4.0);
                ui.horizontal(|ui| {
                    ui.add_space(10.0);
                    ui.label(RichText::new(&present.headline).size(11.0).color(head));
                });
                for line in &present.detail {
                    ui.horizontal(|ui| {
                        ui.add_space(10.0);
                        ui.label(RichText::new(line).size(11.0).color(pal::text_dim(dark)));
                    });
                }
            }

            if !self.status.text().is_empty() {
                ui.add_space(4.0);
                ui.horizontal(|ui| {
                    ui.add_space(10.0);
                    ui.label(RichText::new(self.status.text()).size(11.0).color(pal::text_dim(ui.visuals().dark_mode)));
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
                    let play_count   = self.stats.get(&self.playlist[i].path).map(|s| s.count).unwrap_or(0);
                    let show_count   = self.appearance.show_play_count;
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

                        // Play count (optional) — sits left of the duration.
                        let dur_right = if show_count { inner.max.x - 52.0 } else { inner.max.x - 8.0 };
                        if show_count {
                            ui.painter().text(
                                egui::Pos2::new(inner.max.x - 8.0, cy),
                                egui::Align2::RIGHT_CENTER,
                                if play_count > 0 { format!("▶{play_count}") } else { "–".to_string() },
                                egui::FontId::proportional(11.0),
                                if play_count > 0 { pal::text_dim(dark) } else { pal::text_faint(dark) },
                            );
                        }

                        // Duration
                        if let Some(dur) = track_dur {
                            ui.painter().text(
                                egui::Pos2::new(dur_right, cy),
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
        // A lyric size dragged up to the moment of closing.
        if let Some(s) = lyrics_text_size_to_save(self.lyrics_window.text_size(), self.appearance.lyrics_text_size, true) {
            self.appearance.lyrics_text_size = s;
            save_appearance(&moosik_dir(), &self.appearance);
        }
    }
}

// ---------------------------------------------------------------------------
// Tests — DSD metadata integration (the parsers themselves are tested in dsd/)
// ---------------------------------------------------------------------------

#[cfg(test)]
#[path = "main_tests.rs"]
mod tests;
