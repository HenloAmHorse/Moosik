//! The spectrum's colour ramps: which ramp is chosen, and the colour it gives
//! a normalised magnitude. Stateless; drawing, textures and the picker stay
//! in `spectrum.rs`, which re-exports these types at their old paths.

use egui::Color32;
use serde::{Deserialize, Serialize};

/// Colour ramp for the spectrum visualisers. `Classic` is the original rainbow
/// heat map; the others are cohesive alternatives, and `Accent` follows the
/// current track's album-art accent so the graph matches the surrounding chrome.
/// Selectable (and persisted) from the app's Appearance menu.
#[derive(Clone, Copy, PartialEq, Eq, Debug, Serialize, Deserialize)]
pub enum SpectrumPalette {
    Classic,
    Kugelblitz,
    Ice,
    Magma,
    Aurora,
    Mono,
    Accent,
}

impl Default for SpectrumPalette {
    fn default() -> Self { SpectrumPalette::Classic }
}

impl SpectrumPalette {
    /// All variants, in menu order.
    pub const ALL: [SpectrumPalette; 7] = [
        SpectrumPalette::Classic,
        SpectrumPalette::Kugelblitz,
        SpectrumPalette::Ice,
        SpectrumPalette::Magma,
        SpectrumPalette::Aurora,
        SpectrumPalette::Mono,
        SpectrumPalette::Accent,
    ];

    pub fn label(self) -> &'static str {
        match self {
            SpectrumPalette::Classic    => "Classic",
            SpectrumPalette::Kugelblitz => "Kugelblitz",
            SpectrumPalette::Ice        => "Ice",
            SpectrumPalette::Magma      => "Magma",
            SpectrumPalette::Aurora     => "Aurora",
            SpectrumPalette::Mono       => "Mono",
            SpectrumPalette::Accent     => "Album accent",
        }
    }
}

/// A resolved palette: the chosen ramp plus the accent it may need (used only
/// by `SpectrumPalette::Accent`). Small and `Copy`, so passed by value/ref freely.
#[derive(Clone, Copy)]
pub struct Palette {
    pub kind:   SpectrumPalette,
    pub accent: Color32,
}

impl Palette {
    pub fn new(kind: SpectrumPalette, accent: Color32) -> Self { Self { kind, accent } }

    /// Colour for a normalised bar / point magnitude (0–1).
    pub fn bar(&self, t: f32) -> Color32 { self.sample(t) }
    /// Colour for a spectrogram / waterfall cell (0–1).
    pub fn heat(&self, t: f32) -> Color32 { self.sample(t) }
    /// A representative "hot" colour, for line / filled outline styles.
    pub fn line(&self) -> Color32 { self.sample(0.82) }

    fn sample(&self, t: f32) -> Color32 {
        let t = t.clamp(0.0, 1.0);
        match self.kind {
            SpectrumPalette::Classic => sample_stops(&[
                (0.0, (0, 0, 0)), (0.2, (0, 0, 140)), (0.4, (0, 220, 255)),
                (0.6, (255, 220, 0)), (0.8, (255, 70, 0)), (1.0, (255, 255, 220)),
            ], t),
            SpectrumPalette::Kugelblitz => sample_stops(&[
                (0.0, (10, 12, 24)), (0.45, (60, 90, 200)),
                (0.78, (148, 177, 255)), (1.0, (232, 240, 255)),
            ], t),
            SpectrumPalette::Ice => sample_stops(&[
                (0.0, (2, 6, 20)), (0.35, (0, 70, 150)),
                (0.7, (0, 190, 220)), (1.0, (224, 255, 255)),
            ], t),
            SpectrumPalette::Magma => sample_stops(&[
                (0.0, (2, 0, 6)), (0.25, (70, 16, 96)), (0.5, (160, 44, 96)),
                (0.75, (240, 110, 60)), (1.0, (255, 246, 200)),
            ], t),
            SpectrumPalette::Aurora => sample_stops(&[
                (0.0, (2, 12, 10)), (0.35, (0, 96, 76)),
                (0.66, (44, 200, 126)), (1.0, (204, 255, 190)),
            ], t),
            SpectrumPalette::Mono => sample_stops(&[
                (0.0, (8, 8, 10)), (1.0, (240, 242, 248)),
            ], t),
            SpectrumPalette::Accent => {
                let a = self.accent;
                // A low tint of the accent (keeps a little hue out of the floor),
                // the accent itself in the body, pushed toward white at the peak.
                let lo = (a.r() / 6, a.g() / 6, (a.b() / 5).max(14));
                let mid = (a.r(), a.g(), a.b());
                let hi = (
                    (a.r() as u16 + 170).min(255) as u8,
                    (a.g() as u16 + 170).min(255) as u8,
                    (a.b() as u16 + 170).min(255) as u8,
                );
                sample_stops(&[(0.0, (6, 7, 12)), (0.28, lo), (0.7, mid), (1.0, hi)], t)
            }
        }
    }
}

/// Linearly interpolate a colour from an ascending list of `(pos, rgb)` stops.
fn sample_stops(stops: &[(f32, (u8, u8, u8))], t: f32) -> Color32 {
    let rgb = |c: (u8, u8, u8)| Color32::from_rgb(c.0, c.1, c.2);
    match stops.first() {
        None => return Color32::BLACK,
        Some(&(p0, c0)) if t <= p0 => return rgb(c0),
        _ => {}
    }
    let &(pl, cl) = stops.last().unwrap();
    if t >= pl { return rgb(cl); }
    for w in stops.windows(2) {
        let (p0, c0) = w[0];
        let (p1, c1) = w[1];
        if t >= p0 && t <= p1 {
            let s = if (p1 - p0).abs() < 1e-6 { 0.0 } else { (t - p0) / (p1 - p0) };
            let lerp = |a: u8, b: u8| (a as f32 + (b as f32 - a as f32) * s).round() as u8;
            return Color32::from_rgb(
                lerp(c0.0, c1.0), lerp(c0.1, c1.1), lerp(c0.2, c1.2),
            );
        }
    }
    rgb(cl)
}
