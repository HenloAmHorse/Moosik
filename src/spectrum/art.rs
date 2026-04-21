use egui::{Color32, Pos2, Rect, Shape};
use egui::epaint::Vertex;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::path::PathBuf;

// ---------------------------------------------------------------------------
// Album art display — types and per-track settings store
// ---------------------------------------------------------------------------

#[derive(Clone, PartialEq, Debug, Serialize, Deserialize)]
pub enum ArtFit { Preserve, Stretch }

/// How the album art is composited onto the spectrum plot.
#[derive(Clone, PartialEq, Debug, Serialize, Deserialize)]
pub enum ArtSpectrumMode {
    /// Art not shown.
    Hidden,
    /// Art shown at reduced opacity behind bars.
    Transparent,
    /// Art shown only inside bar rectangles; bars become the mask.
    Mask,
}

/// Sub-mode when `ArtSpectrumMode::Mask` is active.
#[derive(Clone, PartialEq, Debug, Serialize, Deserialize)]
pub enum ArtMaskMode {
    /// Bar amplitude controls brightness — tall/bright bars → vivid art.
    Dynamic,
    /// Fixed brightness everywhere bars exist; controlled by slider.
    Fixed,
}

#[derive(Clone, Serialize, Deserialize)]
pub struct ArtDisplaySettings {
    pub spectrum_mode:     ArtSpectrumMode,
    pub fit:               ArtFit,
    /// Opacity for Transparent mode (0 = invisible, 1 = fully opaque).
    pub transparency:      f32,
    pub mask_mode:         ArtMaskMode,
    /// Brightness for Fixed mask sub-mode (0–1).
    pub mask_brightness:   f32,
}

impl Default for ArtDisplaySettings {
    fn default() -> Self {
        Self {
            spectrum_mode:   ArtSpectrumMode::Hidden,
            fit:             ArtFit::Preserve,
            transparency:    0.25,
            mask_mode:       ArtMaskMode::Dynamic,
            mask_brightness: 0.8,
        }
    }
}

/// Persisted global + per-track album art display preferences.
#[derive(Serialize, Deserialize)]
pub struct ArtSettingsStore {
    #[serde(default)] pub global:               ArtDisplaySettings,
    /// Key: absolute track path as string.
    #[serde(default)] pub per_track:            HashMap<String, ArtDisplaySettings>,
    /// Show thumbnails in the playlist.
    #[serde(default = "default_true")] pub playlist_show: bool,
    /// Show a placeholder icon when a track has no embedded art.
    #[serde(default)] pub playlist_placeholder: bool,
    /// Show a placeholder in the spectrum window when no art is available.
    #[serde(default)] pub spectrum_placeholder: bool,
}

fn default_true() -> bool { true }

impl Default for ArtSettingsStore {
    fn default() -> Self {
        Self {
            global: ArtDisplaySettings::default(),
            per_track: HashMap::new(),
            playlist_show: true,
            playlist_placeholder: false,
            spectrum_placeholder: false,
        }
    }
}

impl ArtSettingsStore {
    fn settings_path() -> PathBuf {
        home_dir().join(".moosik").join("art_settings.json")
    }

    pub fn load() -> Self {
        std::fs::read_to_string(Self::settings_path())
            .ok()
            .and_then(|s| serde_json::from_str(&s).ok())
            .unwrap_or_default()
    }

    pub fn save(&self) {
        let path = Self::settings_path();
        if let Some(p) = path.parent() { let _ = std::fs::create_dir_all(p); }
        if let Ok(json) = serde_json::to_string_pretty(self) {
            let _ = std::fs::write(path, json);
        }
    }

    /// Returns the effective settings for a track: per-track override if set,
    /// otherwise the global defaults.
    pub fn settings_for(&self, track_path: &str) -> &ArtDisplaySettings {
        self.per_track.get(track_path).unwrap_or(&self.global)
    }

    pub fn has_override(&self, track_path: &str) -> bool {
        self.per_track.contains_key(track_path)
    }

    /// Copy current global settings into a per-track slot so the user can
    /// customise this track without affecting others.
    pub fn make_per_track(&mut self, track_path: &str) {
        if !self.has_override(track_path) {
            self.per_track.insert(track_path.to_string(), self.global.clone());
        }
    }

    /// Remove the per-track override, reverting to global settings.
    pub fn reset_to_global(&mut self, track_path: &str) {
        self.per_track.remove(track_path);
    }
}

fn home_dir() -> PathBuf {
    std::env::var("USERPROFILE")
        .or_else(|_| std::env::var("HOME"))
        .map(PathBuf::from)
        .unwrap_or_else(|_| PathBuf::from("."))
}

/// Compute where the art image should sit inside `plot_rect`.
pub fn art_dest_rect(plot_rect: Rect, art_w: u32, art_h: u32, fit: &ArtFit) -> Rect {
    match fit {
        ArtFit::Stretch => plot_rect,
        ArtFit::Preserve => {
            if art_w == 0 || art_h == 0 { return plot_rect; }
            let art_aspect  = art_w as f32 / art_h as f32;
            let plot_aspect = plot_rect.width() / plot_rect.height().max(1.0);
            if art_aspect > plot_aspect {
                let new_h = plot_rect.width() / art_aspect;
                let pad   = (plot_rect.height() - new_h) / 2.0;
                Rect::from_min_max(
                    Pos2::new(plot_rect.left(),  plot_rect.top()    + pad),
                    Pos2::new(plot_rect.right(), plot_rect.bottom() - pad),
                )
            } else {
                let new_w = plot_rect.height() * art_aspect;
                let pad   = (plot_rect.width() - new_w) / 2.0;
                Rect::from_min_max(
                    Pos2::new(plot_rect.left()  + pad, plot_rect.top()),
                    Pos2::new(plot_rect.right() - pad, plot_rect.bottom()),
                )
            }
        }
    }
}

/// Draw the album art as a textured mesh shaped like the spectrum bars.
/// Replaces the normal bar draw call when `ArtSpectrumMode::Mask` is active.
pub fn draw_bars_art_mask(
    painter:    &egui::Painter,
    mags:       &[f32],
    rect:       Rect,
    gap:        f32,
    tex_id:     egui::TextureId,
    art_w:      u32,
    art_h:      u32,
    fit:        &ArtFit,
    mask_mode:  &ArtMaskMode,
    brightness: f32,
) {
    let n = mags.len();
    if n == 0 { return; }

    let art_rect  = art_dest_rect(rect, art_w, art_h, fit);
    let aw        = art_rect.width().max(1.0);
    let ah        = art_rect.height().max(1.0);

    let ppp       = painter.ctx().pixels_per_point();
    let phys_left = (rect.left()  * ppp).round() as i32;
    let phys_w    = ((rect.right() * ppp).round() as i32 - phys_left).max(1);
    let phys_gap  = (gap * ppp).round().max(0.0) as i32;
    let min_col_w = (1 + phys_gap).max(1);
    let draw_n    = ((phys_w / min_col_w) as usize).min(n).max(1);

    let mut mesh = egui::Mesh::with_texture(tex_id);
    mesh.reserve_triangles(draw_n * 2);
    mesh.reserve_vertices(draw_n * 4);

    for i in 0..draw_n {
        let src_lo = (i * n) / draw_n;
        let src_hi = (((i + 1) * n) / draw_n).min(n);
        let v = mags[src_lo..src_hi].iter().cloned().fold(0.0_f32, f32::max);
        if v <= 0.0 { continue; }

        let h   = v * rect.height();
        let px0 = phys_left + (i as i32 * phys_w) / draw_n as i32;
        let px1 = (phys_left + ((i + 1) as i32 * phys_w) / draw_n as i32 - phys_gap).max(px0 + 1);
        let x0  = px0 as f32 / ppp;
        let x1  = px1 as f32 / ppp;
        let y0  = rect.bottom() - h;  // bar top
        let y1  = rect.bottom();       // bar bottom

        // Map bar corners into art texture UV space
        let u0 = ((x0 - art_rect.left()) / aw).clamp(0.0, 1.0);
        let u1 = ((x1 - art_rect.left()) / aw).clamp(0.0, 1.0);
        let v0 = ((y0 - art_rect.top())  / ah).clamp(0.0, 1.0);
        let v1 = ((y1 - art_rect.top())  / ah).clamp(0.0, 1.0);

        let alpha = match mask_mode {
            ArtMaskMode::Dynamic => (v * 255.0) as u8,
            ArtMaskMode::Fixed   => (brightness * 255.0) as u8,
        };
        let tint = Color32::from_white_alpha(alpha);

        let base = mesh.vertices.len() as u32;
        mesh.vertices.push(Vertex { pos: Pos2::new(x0, y0), uv: Pos2::new(u0, v0), color: tint });
        mesh.vertices.push(Vertex { pos: Pos2::new(x1, y0), uv: Pos2::new(u1, v0), color: tint });
        mesh.vertices.push(Vertex { pos: Pos2::new(x1, y1), uv: Pos2::new(u1, v1), color: tint });
        mesh.vertices.push(Vertex { pos: Pos2::new(x0, y1), uv: Pos2::new(u0, v1), color: tint });
        mesh.indices.extend_from_slice(&[base, base+1, base+2, base, base+2, base+3]);
    }
    painter.add(Shape::mesh(mesh));
}
