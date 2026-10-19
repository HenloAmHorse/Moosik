//! The whole-track view's place in the Spectrum window.
//!
//! It takes over the plot area while "Whole track" is selected in Pre-process
//! mode and gives it back untouched when it is not: the live views keep their
//! own state and are not drawn meanwhile. The data contract and the raster are
//! in [`super::wholetrack`]; this file is layout, input and textures.
//!
//! What runs when:
//!
//! * The requested camera and the presented camera are separate. Cheap
//!   exact changes commit immediately. Expensive changes prepare on the
//!   worker while the last complete, same-size image AND its axes/readout
//!   remain visible. No old image is stretched into a different camera.
//! * Working uploads are suppressed during that wait; all planes commit
//!   together. Input keeps accumulating against the requested extent, but
//!   inspecting/clicking the visible image uses the presented camera.
//! * A new source, plane, palette or raster size cannot retain an incompatible
//!   image. First preparation still shows loading when there is no image.
use std::sync::Arc;

use eframe::egui::{self, Color32, Pos2, Rect, Stroke};

use super::cache::PreFrames;
use super::palette::Palette;
use super::wholetrack::{
    block_max, code_db, fit_raster, span, Geometry, Job, Origin, Raster, Snapshot,
    Unavailable, View, Worker, NO_DATA, OUTSIDE, OVERVIEW_MAX,
};
use super::wholetrack_ring::{Picture, Ring, RingStats, SYNC_CELLS};
use super::wholetrack::{column_max_held, held_cost, Accel, Level};

/// What a plane already has built for the UI thread to read through: its
/// installed sparse-table level, and its accelerators (for the pyramid).
type Held = (Option<Arc<Level>>, Option<Arc<Accel>>);

/// Complete previous presentation while the requested camera is prepared.
/// Working CPU rings may change, but their textures cannot be uploaded
/// until this presentation is retired at a complete-camera commit.
struct Presented {
    pictures: Vec<Picture>,
    plane: Plane,
    colours: PalKey,
}

/// Tests: a ring's texture, painted left edge, width, height, and the
/// texture row its top camera row is drawn from.
#[cfg(test)]
pub(crate) type RingTexture = (egui::TextureId, i64, usize, usize, usize);

/// Tests: what a ring drew for the painted camera — the camera, its width
/// and height, and each column's codes (`None`: drawn as loading).
#[cfg(test)]
pub(crate) type Drawn = (View, usize, usize, Vec<Option<Vec<u32>>>);

/// Most bytes the shown planes' pyramids may take together; past it a plane
/// is read frame by frame, as before (exact either way, only slower).
const PYRAMID_BUDGET: usize = 160 << 20;

/// Which planes are drawn.
#[derive(Clone, Copy, PartialEq, Eq, Debug, Default)]
pub enum Plane {
    #[default]
    Mix,
    Left,
    Right,
    /// Left above Right, one time and frequency axis.
    Split,
}

impl Plane {
    const ALL: [Plane; 4] = [Plane::Mix, Plane::Left, Plane::Right, Plane::Split];
    fn label(self) -> &'static str {
        match self {
            Plane::Mix => "Mix",
            Plane::Left => "Left",
            Plane::Right => "Right",
            Plane::Split => "Split L/R",
        }
    }
    pub fn needs_channels(self) -> bool {
        self != Plane::Mix
    }
}

/// What the window has to do on the view's behalf.
#[derive(Clone, Debug, PartialEq)]
pub enum Action {
    /// Seek playback to this many seconds, through the player's own seek.
    Seek(f64),
    /// Run the existing explicit channel analysis for this track.
    AnalyseChannels,
    /// The existing Re-analyze.
    Reanalyze,
}

/// A raster on the screen: its codes, kept for recolouring, and its texture.
struct Shown {
    raster: Raster,
    tex: egui::TextureHandle,
    /// The palette it was coloured with.
    colours: PalKey,
}

#[derive(Clone, Copy, PartialEq)]
struct PalKey(super::palette::SpectrumPalette, Color32);

/// Codes have a finite palette. Reuse their exact colours while both the
/// palette and the stored analysis' normalization are unchanged.
struct ColourTable {
    key: PalKey,
    max_code: u16,
    inv_max: u32,
    colours: Vec<Color32>,
}

impl PalKey {
    fn of(p: &Palette) -> Self {
        PalKey(p.kind, p.accent)
    }
}

/// Figures for the debug overlay and the tests.
#[derive(Clone, Copy, Default, Debug, PartialEq)]
pub struct Usage {
    /// Raster codes held on the CPU, bytes.
    pub cpu_bytes: usize,
    /// Textures uploaded, bytes (RGBA).
    pub gpu_bytes: usize,
    /// Rasters requested, and the last one's worker time.
    pub requests: u64,
    pub last_raster_ms: f64,
    /// Colour mappings done (each is a full upload).
    pub recolours: u64,
    /// Frames that completed requested demand, and frames still preparing
    /// it (including those displaying an honest complete previous view).
    pub exact_frames: u64,
    pub catching_up_frames: u64,
    /// Waiting frames that showed a complete previous camera, not loading.
    pub retained_frames: u64,
    /// The main rings' counters, summed over planes.
    pub ring: RingStats,
}

#[derive(Default)]
pub struct WholeTrackView {
    pub open: bool,
    pub plane: Plane,
    /// The playhead, in seconds. Set by the window's tick.
    pub playhead: f64,
    view: Option<View>,
    snapshot: Option<Snapshot>,
    unavailable: Option<Unavailable>,
    epoch: u64,
    seq: u64,
    /// What was last asked of the worker, so an unchanged frame asks nothing.
    asked: Option<(u64, View, (usize, usize), Plane)>,
    /// Results of jobs numbered below this were for rings since replaced.
    ring_since: u64,
    worker: Worker,
    /// The main picture, one ring per plane drawn (see `wholetrack_ring`).
    rings: Vec<Ring>,
    presented: Option<Presented>,
    rings_pal: Option<PalKey>,
    colour_table: Option<ColourTable>,
    overview: Option<Shown>,
    overview_asked: bool,
    /// Per plane shown, what the worker keeps to read wide columns quickly
    /// (a block-maximum pyramid within the budget, a sparse-table level).
    accel: Vec<(Arc<PreFrames>, Arc<Accel>)>,
    /// Follow playback (the preference, saved with the window's settings),
    /// and whether a pan, wheel zoom or the overview strip has set it aside
    /// until "Resume follow".
    pub follow: bool,
    pub follow_suspended: bool,
    pub usage: Usage,
    /// Pointer readout, recomputed while hovering.
    readout: String,
    /// The gesture list is expanded.
    show_help: bool,
    /// Where the plot was last laid out, for the gesture tests.
    #[cfg(test)]
    pub(crate) plot_for_test: Option<Rect>,
    #[cfg(test)]
    pub(crate) sub_for_test: Vec<Rect>,
    /// Tests: read at most this many cells on the UI thread (0 sends every
    /// camera change to the worker).
    #[cfg(test)]
    pub(crate) sync_cells_for_test: Option<usize>,
    /// Tests: the camera this frame asked for and painted, recorded
    /// independently of the rings (the rings say what they drew of it).
    #[cfg(test)]
    pub(crate) painted_for_test: Option<View>,
    #[cfg(test)]
    pub(crate) requested_for_test: Option<View>,
}

/// The plot's gestures, as the hint and the help list them. The input code
/// in `ui` is the authority; this is kept beside it so the two say the same.
pub(crate) const GESTURES: [(&str, &str); 6] = [
    ("Wheel", "zoom time: around the playhead while following (and it keeps following), around the pointer otherwise"),
    ("Shift + wheel", "zoom frequency, around the pointer (following carries on)"),
    ("Drag", "pan (does not seek; sets following aside until Resume follow)"),
    ("Click", "seek playback to that time (paused stays paused)"),
    ("Middle click", "fit the whole track (does not seek)"),
    ("Overview strip", "click or drag to move the time window"),
];

const AXIS_W: f32 = 52.0;
const LEGEND_W: f32 = 64.0;
const TIME_H: f32 = 16.0;
const STRIP_H: f32 = 40.0;
/// Smallest span the view zooms to, in frames and in bars.
const MIN_FRAMES: f64 = 8.0;
const MIN_BARS: f64 = 4.0;
/// Where following keeps the playhead: a third of the way across, so twice
/// as much of what is coming is shown as of what has passed.
pub(crate) const FOLLOW_AT: f64 = 1.0 / 3.0;

impl WholeTrackView {
    /// A view with following on, the preference's default.
    pub fn new() -> Self {
        WholeTrackView { follow: true, ..Default::default() }
    }

    /// Whether the channels should be resident for this view.
    ///
    /// The preference alone: the window asks this only while the view is
    /// actually on screen (see `SpectrumWindow::whole_track_shown`), so a
    /// closed Spectrum window or another mode never acquires channels for it.
    pub fn wants_channels(&self) -> bool {
        self.open && self.plane.needs_channels()
    }

    /// Turn the view off (the preference) and release what it holds.
    pub fn close(&mut self) {
        self.open = false;
        self.retire();
    }

    /// Release everything held for drawing — snapshot (planes and pin),
    /// rasters, textures, queued results — and cancel work in flight, keeping
    /// the preferences (on/off, plane). Called every tick while the view is
    /// not on screen: the parent window closed, another mode, or the toggle
    /// off. Cheap when there is nothing to release.
    ///
    /// A raster thread already running keeps its own `Arc`s of the planes
    /// until it notices the cancel at its next column and exits; it holds no
    /// store pin. Nothing here waits for it. Its result, if it still sends
    /// one, is dropped on the next call, and its thread is reaped then.
    pub fn retire(&mut self) {
        if self.snapshot.is_some() || !self.rings.is_empty() || self.overview.is_some() || self.asked.is_some() {
            self.drop_state();
            // The view belonged to the released snapshot; reopening fits
            // whatever is installed then.
            self.view = None;
        }
        self.unavailable = None;
        // Always, not only while busy: a completion can sit in the queue
        // after its thread was reaped. Dropping it here frees its raster.
        // Nonblocking; at most one thread can have sent.
        let _ = self.worker.poll(self.epoch);
    }

    /// Whether anything is held for drawing (tests and the debug overlay).
    #[cfg(test)]
    pub(crate) fn holds_anything(&self) -> bool {
        self.snapshot.is_some() || !self.rings.is_empty() || self.overview.is_some() || self.worker.busy()
    }

    fn drop_state(&mut self) {
        self.epoch += 1;
        self.worker.invalidate();
        self.snapshot = None;
        self.rings.clear();
        self.presented = None;
        self.colour_table = None;
        self.accel.clear();
        self.overview = None;
        self.overview_asked = false;
        self.asked = None;
        self.usage.cpu_bytes = 0;
        self.usage.gpu_bytes = 0;
    }

    /// Bring the snapshot in line with what the analyser has installed.
    fn sync(&mut self, a: &super::SpectrumAnalyzer) {
        let want_lr = self.plane.needs_channels();
        let fresh = match &self.snapshot {
            Some(s) => {
                s.revision != a.pre_revision()
                    || (want_lr && s.lr.is_none() && a.pre_channels_resident())
                    || (!want_lr && s.lr.is_some())
            }
            None => true,
        };
        if !fresh {
            return;
        }
        match Snapshot::capture(a, want_lr) {
            Ok(s) => {
                let same = self.snapshot.as_ref().is_some_and(|o| o.same_planes(&s));
                if same {
                    self.snapshot = Some(s);
                    return;
                }
                let keep_view = self.snapshot.as_ref().is_some_and(|o| {
                    Arc::ptr_eq(&o.mix, &s.mix) && o.geometry == s.geometry
                });
                self.drop_state();
                if !keep_view {
                    self.view = Some(View::full(s.frames(), s.geometry.bars));
                }
                self.snapshot = Some(s);
                self.unavailable = None;
            }
            Err(u) => {
                if self.snapshot.is_some() {
                    self.drop_state();
                }
                self.unavailable = Some(u);
            }
        }
    }

    /// The planes to draw, in order, or why they cannot be.
    fn planes(&self, a: &super::SpectrumAnalyzer) -> Result<Vec<(&'static str, Arc<PreFrames>)>, String> {
        let s = self.snapshot.as_ref().ok_or_else(String::new)?;
        let lr = || -> Result<_, String> {
            s.lr.clone().ok_or_else(|| channel_note(a.pre_channels()))
        };
        Ok(match self.plane {
            Plane::Mix => vec![("Mix", s.mix.clone())],
            Plane::Left => vec![("Left", lr()?.0)],
            Plane::Right => vec![("Right", lr()?.1)],
            Plane::Split => {
                let (l, r) = lr()?;
                vec![("Left", l), ("Right", r)]
            }
        })
    }

    pub fn ui(
        &mut self,
        ui: &mut egui::Ui,
        a: &super::SpectrumAnalyzer,
        pal: &Palette,
        analysing: Option<usize>,
        debug: bool,
    ) -> Option<Action> {
        self.sync(a);
        let mut action = None;
        let dark = ui.visuals().dark_mode;

        // ── controls ────────────────────────────────────────────────────────
        let before = self.plane;
        ui.horizontal_wrapped(|ui| {
            ui.label("Plane:");
            for p in Plane::ALL {
                ui.selectable_value(&mut self.plane, p, p.label());
            }
            ui.separator();
            let has = self.snapshot.is_some();
            if ui.add_enabled(has, egui::Button::new("Fit track").small())
                .on_hover_text("Show the whole track and every bar (a middle click on the plot does the same)")
                .clicked()
            {
                self.fit();
            }
            for (lbl, kt, kf, tip) in [
                ("Time −", 2.0, 1.0, "Zoom out in time"),
                ("Time +", 0.5, 1.0, "Zoom in in time"),
                ("Freq −", 1.0, 2.0, "Zoom out in frequency"),
                ("Freq +", 1.0, 0.5, "Zoom in in frequency"),
            ] {
                if ui.add_enabled(has, egui::Button::new(lbl).small()).on_hover_text(tip).clicked()
                    && let Some(v) = self.view
                {
                    // Time, while following: around the playhead, so it keeps
                    // its place and following carries on. Otherwise the middle.
                    let t = match (kt != 1.0, self.following(&v), &self.snapshot) {
                        (true, true, Some(s)) => self.playhead * s.geometry.frame_rate(),
                        _ => (v.t0 + v.t1) * 0.5,
                    };
                    self.zoom_at((t, (v.b0 + v.b1) * 0.5), kt, kf);
                }
            }
            ui.separator();
            ui.toggle_value(&mut self.show_help, "Mouse help")
                .on_hover_text("List what the mouse does on the plot");
            ui.separator();
            if ui.checkbox(&mut self.follow, "Follow playback")
                .on_hover_text("Scroll the time window with the playhead, continuously, a third of the way across. \
                    Zooming in time keeps following (around the playhead), and so does a frequency zoom. A drag or the \
                    overview strip sets it aside until Resume follow.")
                .changed()
            {
                self.follow_suspended = false;
            }
            if self.follow && self.follow_suspended
                && ui.button("Resume follow")
                    .on_hover_text("Follow the playhead again from where it is now, at this zoom")
                    .clicked()
            {
                self.follow_suspended = false;
            }
        });
        if self.show_help {
            egui::Grid::new("wholetrack_gestures").num_columns(2).spacing([12.0, 2.0]).show(ui, |ui| {
                for (what, does) in GESTURES {
                    ui.label(egui::RichText::new(what).strong());
                    ui.label(does);
                    ui.end_row();
                }
            });
        } else {
            ui.label(
                egui::RichText::new(if self.follow && !self.follow_suspended {
                    "Wheel: time zoom (keeps following) · Shift+wheel: frequency zoom · Drag: pan · Click: seek · Middle click: fit"
                } else {
                    "Wheel: time zoom · Shift+wheel: frequency zoom · Drag: pan · Click: seek · Middle click: fit"
                })
                    .color(super::txt_dim(dark)),
            );
        }
        if self.plane != before {
            // Another plane is other data: nothing of the old one is shown.
            self.epoch += 1;
            self.worker.invalidate();
            self.rings.clear();
            self.presented = None;
            // The strip is the Mix whatever the plane; one still on its way
            // was for the old epoch and is refused, so it is asked again.
            self.overview_asked = self.overview.is_some();
            self.usage.cpu_bytes = self.overview.as_ref().map_or(0, |o| o.raster.bytes());
            self.asked = None;
            self.sync(a);
        }

        // ── finished work ───────────────────────────────────────────────────
        let mut landed = Vec::new();
        for mut d in self.worker.poll(self.epoch) {
            if let Some(o) = d.overview.take() {
                self.overview = Some(upload(ui.ctx(), "moosik_wholetrack_overview", o, pal, self.snapshot.as_ref()));
                self.usage.recolours += 1;
            }
            // Columns for rings since replaced (another size) are not used.
            if d.seq >= self.ring_since && !d.cols.is_empty() && d.colmax.len() == self.rings.len() {
                self.usage.last_raster_ms = d.elapsed.as_secs_f64() * 1e3;
                landed.push(d);
            }
        }
        // A palette change recolours what is shown; nothing is rasterised.
        let key = PalKey::of(pal);
        let snap = self.snapshot.as_ref();
        if let Some(s) = self.overview.as_mut()
            && s.colours != key
        {
            s.tex.set(colour(&s.raster, pal, snap), egui::TextureOptions::NEAREST);
            s.colours = key;
            self.usage.recolours += 1;
        }
        if self.rings_pal != Some(key) {
            self.rings.iter_mut().for_each(Ring::mark_all_dirty);
            if self.rings_pal.is_some() && !self.rings.is_empty() {
                self.usage.recolours += self.rings.len() as u64;
            }
            self.rings_pal = Some(key);
        }

        // ── nothing to draw ─────────────────────────────────────────────────
        let Some(snap) = self.snapshot.as_ref() else {
            ui.add_space(12.0);
            let (msg, offer) = match (&self.unavailable, analysing) {
                (_, Some(pct)) => (format!("Analysing this track… {pct}%. The whole-track view opens when it finishes."), false),
                (Some(Unavailable::NoRecord), _) => (
                    "This is a legacy analysis that does not record the settings it was made at, \
                     so its time and frequency axes cannot be stated truthfully. It is not drawn. \
                     Re-analyze to view this track."
                        .to_string(),
                    true,
                ),
                (Some(Unavailable::Inconsistent(e)), _) => (format!("The installed analysis cannot be drawn: {e}."), true),
                _ => ("No completed analysis for this track at these settings.".to_string(), true),
            };
            ui.label(egui::RichText::new(msg).color(super::txt_dim(dark)));
            if offer && ui.button("Re-analyze").clicked() {
                action = Some(Action::Reanalyze);
            }
            return action;
        };
        let g = snap.geometry;
        let info = provenance(snap);
        ui.label(egui::RichText::new(info).size(10.5).color(super::txt_faint(dark)));

        let planes = match self.planes(a) {
            Ok(p) => p,
            Err(why) => {
                ui.add_space(12.0);
                ui.label(egui::RichText::new(why).color(super::txt_dim(dark)));
                if matches!(a.pre_channels(), super::PreChannels::Missing | super::PreChannels::Refused(_)) {
                    ui.label(
                        egui::RichText::new(
                            "Left and Right are only ever shown from their own analysis; \
                             they are never made from the Mix.",
                        )
                        .size(10.5)
                        .color(super::txt_faint(dark)),
                    );
                    if ui
                        .add_enabled(analysing.is_none(), egui::Button::new("Analyse Left/Right for this track"))
                        .on_hover_text("The same action as Channels → This track: the Mix stays as it is.")
                        .clicked()
                    {
                        action = Some(Action::AnalyseChannels);
                    }
                }
                return action;
            }
        };

        // ── layout ──────────────────────────────────────────────────────────
        let readout_h = 14.0;
        let avail = ui.available_size();
        let (rect, _) = ui.allocate_exact_size(avail, egui::Sense::hover());
        if !ui.is_rect_visible(rect) {
            return action;
        }
        let painter = ui.painter_at(rect);
        painter.rect_filled(rect, 2.0, Color32::from_rgb(8, 10, 16));
        let strip = Rect::from_min_max(
            Pos2::new(rect.left() + AXIS_W, rect.top() + 2.0),
            Pos2::new(rect.right() - LEGEND_W, rect.top() + STRIP_H),
        );
        let plot = Rect::from_min_max(
            Pos2::new(rect.left() + AXIS_W, strip.bottom() + 6.0),
            Pos2::new(rect.right() - LEGEND_W, rect.bottom() - TIME_H - readout_h),
        );
        if plot.width() < 40.0 || plot.height() < 40.0 {
            return action;
        }
        #[cfg(test)]
        {
            self.plot_for_test = Some(plot);
        }
        let n = planes.len();
        let gap = 4.0;
        let ph = (plot.height() - gap * (n - 1) as f32) / n as f32;
        // One texel per physical pixel: each plane's rectangle starts on the
        // pixel grid and is exactly its raster's size (unless the raster had
        // to be smaller than the plot, which then scales it).
        let ppp = ui.ctx().pixels_per_point();
        let max_side = ui.ctx().input(|i| i.max_texture_side);
        let size = fit_raster(plot.width() * ppp, (plot.height() - gap * (n - 1) as f32) * ppp, n, max_side);
        let align = |v: f32| (v * ppp).round() / ppp;
        let one_to_one = size.0 as f32 >= (plot.width() * ppp).floor() && size.1 as f32 >= (ph * ppp).floor();
        let sub: Vec<Rect> = (0..n)
            .map(|i| {
                let top = plot.top() + i as f32 * (ph + gap);
                if one_to_one {
                    let min = Pos2::new(align(plot.left()), align(top));
                    Rect::from_min_size(min, egui::vec2(size.0 as f32 / ppp, size.1 as f32 / ppp))
                } else {
                    Rect::from_min_max(Pos2::new(plot.left(), top), Pos2::new(plot.right(), top + ph))
                }
            })
            .collect();
        #[cfg(test)]
        { self.sub_for_test = sub.clone(); }

        // ── input ───────────────────────────────────────────────────────────
        let id = ui.id().with("wholetrack_plot");
        let resp = ui.interact(plot, id, egui::Sense::click_and_drag());
        let strip_resp = ui.interact(strip, id.with("strip"), egui::Sense::click_and_drag());
        let frames = snap.frames() as f64;
        let mut view = self.view.unwrap_or(View::full(snap.frames(), g.bars));
        // The camera drawn last frame (on the pixel grid), which is what the
        // pointer is over.
        // The pointer is over the last complete presentation, not over the
        // requested-but-unfinished camera. Accumulating wheel input still
        // uses the requested extent below.
        if self.presented.as_ref().is_some_and(|p| p.plane != self.plane || p.colours != key
            || p.pictures.len() != n || p.pictures.iter().any(|r| (r.w,r.h) != size)) {
            self.presented = None;
        }
        let requested_drawn = snap_view(view, size, frames, g.bars as f64);
        let drawn = self.presented.as_ref().map(|p| p.pictures[0].view)
            .or_else(|| self.rings.first().and_then(|r| r.painted)).unwrap_or(requested_drawn);
        let mut manual = false;
        let ph_frame = self.playhead * g.frame_rate();
        // Following is not set aside by a time zoom: it zooms around the
        // playhead, which keeps its place a third of the way across.
        let following = self.following(&view);
        if let Some(p) = resp.hover_pos() {
            // This frame's own wheel movement, not egui's smoothed scroll: a
            // smoothed notch keeps arriving for frames after it, by when Shift
            // may be up and a frequency zoom would turn into a time zoom.
            let (scroll, zoom, shift) = ui.input(|i| (i.raw_scroll_delta, i.zoom_delta(), i.modifiers.shift));
            let dy = if scroll.y != 0.0 { scroll.y } else { scroll.x };
            let k = (-dy as f64 * 0.003).exp() / zoom as f64;
            if k != 1.0 {
                let sr = sub.iter().find(|r| r.contains(p)).copied().unwrap_or(plot);
                let mut at = to_data(&drawn, sr, p);
                let (kt, kf) = if shift { (1.0, k) } else { (k, 1.0) };
                if !shift && following {
                    at.0 = ph_frame;
                }
                // A time zoom leaves the frequency window exactly as asked
                // (not as snapped to the row grid), and vice versa.
                let from = if shift { View { t0: view.t0, t1: view.t1, ..requested_drawn } } else { View { b0: view.b0, b1: view.b1, ..requested_drawn } };
                view = zoom_view(from, at, kt, kf, frames, g.bars as f64);
                // A time zoom while not following is inspection and stays
                // where the pointer put it; while following it carries on.
            }
        }
        // The gestures (see `GESTURES`): a primary drag pans, a primary click
        // seeks, a middle click fits. Each is decided on its own release, so a
        // click is never held back waiting to see whether a second follows.
        if resp.dragged_by(egui::PointerButton::Primary) {
            let d = resp.drag_delta();
            let sr = sub[0];
            let dt = -(d.x as f64) / sr.width() as f64 * (view.t1 - view.t0);
            let db = (d.y as f64) / sr.height() as f64 * (view.b1 - view.b0);
            view = clamp_view(View { t0: view.t0 + dt, t1: view.t1 + dt, b0: view.b0 + db, b1: view.b1 + db }, frames, g.bars as f64);
            manual |= d.x != 0.0;
        }
        if resp.middle_clicked() {
            view = View::full(snap.frames(), g.bars);
        } else if resp.clicked_by(egui::PointerButton::Primary)
            && let Some(p) = resp.interact_pointer_pos()
            // egui's touch slop extends the hit area a few points past the
            // plot; a click there is on the axis, not a time.
            && plot.contains(p)
        {
            let sr = sub.iter().find(|r| r.contains(p)).copied().unwrap_or(plot);
            let t = to_data(&drawn, sr, p).0.clamp(0.0, frames - 1.0).max(0.0);
            action = Some(Action::Seek(g.frame_time(t.floor())));
        }
        if (strip_resp.dragged_by(egui::PointerButton::Primary) || strip_resp.clicked_by(egui::PointerButton::Primary))
            && let Some(p) = strip_resp.interact_pointer_pos()
        {
            let c = ((p.x - strip.left()) / strip.width()) as f64 * frames;
            let half = (view.t1 - view.t0) * 0.5;
            view = clamp_view(View { t0: c - half, t1: c + half, ..view }, frames, g.bars as f64);
            manual = true;
        }
        if manual && self.follow {
            self.follow_suspended = true;
        }
        // Following: the time window moves with the playhead, keeping its
        // span and the frequency window. Every frame, not by pages.
        if self.following(&view) {
            let span_t = view.t1 - view.t0;
            let t0 = ph_frame - FOLLOW_AT * span_t;
            view = clamp_view(View { t0, t1: t0 + span_t, ..view }, frames, g.bars as f64);
        }
        self.view = Some(view);
        // What is drawn: the camera on the pixel grid (within half a pixel of
        // the one asked for), so that one scrolled by whole pixels finds most
        // of its columns already made.
        let view = snap_view(view, size, frames, g.bars as f64);

        // ── bring the picture to this camera ────────────────────────────────
        // The strip is always the Mix, drawn once per snapshot, by the worker.
        if !self.overview_asked {
            let w = ((strip.width() * ppp) as usize).clamp(1, OVERVIEW_MAX[0].min(max_side));
            let h = ((STRIP_H * ppp) as usize).clamp(1, OVERVIEW_MAX[1]);
            self.seq += 1;
            self.worker.request(Job {
                epoch: self.epoch,
                seq: self.seq,
                planes: Vec::new(),
                data_bars: g.data_bars,
                overview: Some((snap.mix.clone(), (w, h))),
                cols: Vec::new(),
                accel: Vec::new(),
                window: (0, 0),
            });
            self.overview_asked = true;
        }
        if self.rings.len() != n || self.rings.first().is_some_and(|r| (r.w, r.h) != size) {
            self.rings = (0..n).map(|_| Ring::new(size.0, size.1, g.data_bars)).collect();
            self.presented = None;
            self.ring_since = self.seq + 1;
            self.asked = None;
            landed.clear();
        }
        let held: Vec<Held> = planes.iter()
            .map(|(_, p)| match self.accel.iter().find(|(q, _)| Arc::ptr_eq(p, q)) {
                Some((_, a)) => (a.level.lock().ok().and_then(|l| l.clone()), Some(a.clone())),
                None => (None, None),
            }).collect();
        let mut plans: Vec<_> = self.rings.iter().map(|r| r.plan(view, snap.frames())).collect();
        let cost = |plans: &[super::wholetrack_ring::Plan]| -> usize {
            plans.iter().zip(&held).map(|(p, (l, a))| p.read_spans()
                .map(|sp| held_cost(l.as_deref(), a.as_ref().and_then(|a| a.pyramid.get()), sp, g.data_bars))
                .sum::<usize>()).sum()
        };
        #[cfg(test)]
        let sync_cells = self.sync_cells_for_test.unwrap_or(SYNC_CELLS);
        #[cfg(not(test))]
        let sync_cells = SYNC_CELLS;
        // Snapshot GPU ownership before any row/column is changed. Only a
        // complete image at this exact size is eligible. No matrix clone.
        if self.presented.is_none() && cost(&plans) > sync_cells {
            let pictures: Option<Vec<_>> = self.rings.iter().map(Ring::picture).collect();
            if let Some(pictures) = pictures.filter(|p| !p.is_empty() && p.iter().all(|r| r.view == p[0].view)) {
                self.presented = Some(Presented { pictures, plane: self.plane, colours: key });
            }
        }
        for r in &mut self.rings {
            r.set_rows(&view, g.bars);
        }
        // Columns the worker read: stored by their span, whatever camera
        // asked for them; a column is used only where a camera needs exactly
        // that span (see `Ring::plan`).
        let had_landed = !landed.is_empty();
        for d in landed {
            for (r, m) in self.rings.iter_mut().zip(&d.colmax) {
                let now = r.plan(view, snap.frames());
                r.put_landed_columns(&now, &d.cols, m);
                r.stats.cells_async += d.cols.iter().map(|(_, (a, b))| (b - a) * g.data_bars).sum::<usize>() as u64;
            }
        }
        if had_landed { plans = self.rings.iter().map(|r| r.plan(view, snap.frames())).collect(); }
        // What reading the missing columns here costs through what is already
        // built for each plane (a level answers in up to three cheap rows, see
        // `held_cost`); nothing is built on this thread.
        let cells = cost(&plans);
        if cells <= sync_cells && (cells == 0 || sync_cells > 0) {
            for (((r, p), (_, plane)), (l, a)) in self.rings.iter_mut().zip(&plans).zip(&planes).zip(&held) {
                let pyr = a.as_ref().and_then(|a| a.pyramid.get());
                r.fill_with(p, |sp, out| column_max_held(plane, l.as_deref(), pyr, sp, out));
            }
            self.asked = None;
        } else {
            // Too much to read on this thread: the worker reads it. Meanwhile
            // this camera is drawn with what is held for it, and the rest as
            // loading — never another camera's picture moved into place.
            for (r, p) in self.rings.iter_mut().zip(&plans) {
                r.present(p);
            }
            let want = (self.epoch, view, size, self.plane);
            // Asked again when the worker is idle and columns are still
            // missing (a superseded job's columns can have landed instead).
            if self.asked != Some(want) || !self.worker.busy() {
                self.seq += 1;
                let p = &plans[0];
                self.worker.request(Job {
                    epoch: self.epoch,
                    seq: self.seq,
                    planes: planes.iter().map(|(_, p)| p.clone()).collect(),
                    data_bars: g.data_bars,
                    overview: None,
                    accel: accel_for(&mut self.accel, &planes, g.data_bars),
                    cols: p.missing.iter().map(|&x| (((p.base + x as i64).rem_euclid(size.0 as i64)) as usize, p.spans[x])).collect(),
                    window: (p.spans.iter().map(|s| s.0).min().unwrap_or(0), p.spans.iter().map(|s| s.1).max().unwrap_or(0)),
                });
                self.usage.requests += 1;
                self.asked = Some(want);
                for r in &mut self.rings {
                    r.stats.async_jobs += 1;
                }
            }
        }
        drop(held);
        let exact = self.rings.iter().all(|r| r.shown == Some(view));
        if exact { self.presented = None; }
        if self.presented.is_some() { self.usage.retained_frames += 1; }
        if exact { self.usage.exact_frames += 1 } else { self.usage.catching_up_frames += 1 }
        #[cfg(test)]
        {
            self.requested_for_test = Some(view);
            self.painted_for_test = Some(self.presented.as_ref().map_or(view, |p| p.pictures[0].view));
        }
        let snap = self.snapshot.as_ref().expect("checked above");
        if self.colour_table.as_ref().is_none_or(|t| t.key != key || t.max_code != snap.max_code || t.inv_max != snap.inv_max.to_bits()) {
            self.colour_table = Some(ColourTable { key, max_code: snap.max_code, inv_max: snap.inv_max.to_bits(), colours: palette_lut(pal, snap) });
        }
        let lut = &self.colour_table.as_ref().expect("palette prepared").colours;
        for r in self.rings.iter_mut().filter(|_| self.presented.is_none()) {
            r.upload(ui.ctx(), "moosik_wholetrack", |c, gx, y| match c {
                OUTSIDE => Color32::TRANSPARENT,
                NO_DATA => hatch(gx.rem_euclid(8) as usize, y),
                c => lut[(c as usize).min(lut.len() - 1)],
            });
        }
        self.usage.ring = self.rings.iter().fold(RingStats::default(), |a, r| add_stats(a, r.stats));
        self.usage.cpu_bytes = self.rings.iter().map(Ring::cpu_bytes).sum::<usize>()
            + self.accel.iter().map(|(_, a)| a.bytes()).sum::<usize>()
            + self.overview.as_ref().map_or(0, |o| o.raster.bytes())
            + self.colour_table.as_ref().map_or(0, |t| t.colours.capacity() * std::mem::size_of::<Color32>());
        // The presentation holds another handle to the SAME unchanged
        // textures; do not count those allocations twice.
        self.usage.gpu_bytes = self.rings.iter().map(Ring::gpu_bytes).sum::<usize>()
            + self.overview.as_ref().map_or(0, |o| o.raster.w * o.raster.h * 4);

        // Every visible datum (axes, strip, playhead and hover included)
        // belongs to this presentation. Demand stays in self.view.
        let view = self.presented.as_ref().map_or(view, |p| p.pictures[0].view);

        // ── draw ────────────────────────────────────────────────────────────
        let grey = Color32::from_gray(90);
        if let Some(o) = &self.overview {
            painter.image(o.tex.id(), strip, uv1(), Color32::WHITE);
        }
        let vx0 = strip.left() + (view.t0 / frames) as f32 * strip.width();
        let vx1 = strip.left() + (view.t1 / frames) as f32 * strip.width();
        painter.rect_stroke(
            Rect::from_min_max(Pos2::new(vx0, strip.top()), Pos2::new(vx1.max(vx0 + 2.0), strip.bottom())),
            0.0,
            Stroke::new(1.0, Color32::from_rgba_unmultiplied(255, 255, 255, 200)),
            egui::StrokeKind::Inside,
        );
        let px = strip.left() + (ph_frame / frames) as f32 * strip.width();
        if (strip.left()..=strip.right()).contains(&px) {
            painter.line_segment([Pos2::new(px, strip.top()), Pos2::new(px, strip.bottom())], Stroke::new(1.0, Color32::WHITE));
        }
        super::text_inside(&painter, Pos2::new(rect.left() + 4.0, strip.center().y), egui::Align2::LEFT_CENTER,
            "Track".into(), egui::FontId::monospace(9.0), grey);

        for (i, (r, (name, _))) in sub.iter().zip(&planes).enumerate() {
            let pp = painter.with_clip_rect(*r);
            pp.rect_filled(*r, 0.0, Color32::from_rgb(4, 5, 8));
            if let Some(picture) = self.presented.as_ref().and_then(|p| p.pictures.get(i)) {
                picture.draw(&pp, *r);
            } else if let Some(ring) = self.rings.get(i) {
                ring.draw(&pp, *r);
            }
            // The playhead: a painted line, nothing else.
            let x = r.left() + ((ph_frame - view.t0) / (view.t1 - view.t0)) as f32 * r.width();
            if (r.left()..=r.right()).contains(&x) {
                pp.line_segment([Pos2::new(x, r.top()), Pos2::new(x, r.bottom())], Stroke::new(1.5, Color32::WHITE));
            }
            freq_axis(&painter, *r, &view, &g);
            if n > 1 {
                super::text_inside(&painter, Pos2::new(r.left() + 4.0, r.top() + 2.0), egui::Align2::LEFT_TOP,
                    name.to_string(), egui::FontId::proportional(11.0), Color32::from_rgba_unmultiplied(255, 255, 255, 200));
            }
        }
        time_axis(&painter, plot, &view, &g);
        legend(&painter, Rect::from_min_max(Pos2::new(plot.right() + 6.0, plot.top()), Pos2::new(rect.right() - 2.0, plot.bottom())), pal);

        // ── what the pixels are ─────────────────────────────────────────────
        let shown = self.presented.as_ref().map(|p| {
            let b = p.pictures[0].block;
            (b.0 <= 1 && b.1 <= 1, b)
        }).or_else(|| self.rings.first().filter(|r| r.painted.is_some()).map(|r| {
            let b = r.max_block();
            (b.0 <= 1 && b.1 <= 1, b)
        }));
        let mode = match shown {
            Some((true, _)) => "native cells, magnified (nearest; no interpolation)".to_string(),
            Some((false, (f, b))) => format!("each pixel is the maximum of up to {} × {}", count(f, "frame"), count(b, "bar")),
            None => "rasterising…".to_string(),
        };
        if !exact && shown.is_some() {
            super::text_inside(&painter, Pos2::new(plot.right() - 4.0, plot.top() + 2.0), egui::Align2::RIGHT_TOP,
                if self.presented.is_some() { "Preparing view… showing previous view" } else { "loading…" }.into(), egui::FontId::monospace(9.0), Color32::from_gray(200));
        }

        // ── hover readout, from the stored codes ────────────────────────────
        self.readout = mode;
        if let Some(p) = resp.hover_pos()
            && let Some(i) = sub.iter().position(|r| r.contains(p))
        {
            let r = sub[i];
            let (w, h) = size;
            let x = (((p.x - r.left()) / r.width()) * w as f32) as usize;
            let y = (((p.y - r.top()) / r.height()) * h as f32) as usize;
            let plane = &planes[i].1;
            self.readout = hover(plane, &view, (w, h), (x.min(w - 1), y.min(h - 1)), snap, planes[i].0);
        }
        super::text_inside(&painter, Pos2::new(plot.left(), rect.bottom() - 1.0), egui::Align2::LEFT_BOTTOM,
            self.readout.clone(), egui::FontId::monospace(10.0), Color32::from_gray(170));
        // F3: what this view holds and costs, measured here, for the record.
        if debug {
            let u = self.usage;
            let line = format!(
                "worker {:.1} ms · view CPU {:.1} MiB · textures {:.1} MiB · worker jobs {} · {} · exact frames {} · catching up {} · \
                 cells read here {} / by the worker {} · columns {} · uploads {} whole, {} partial ({:.1} MiB) · \
                 source planes this view holds (decoded, shared with the player, not the view's own): {}",
                u.last_raster_ms, u.cpu_bytes as f64 / 1048576.0, u.gpu_bytes as f64 / 1048576.0, u.requests,
                if self.worker.busy() { "busy" } else { "idle" },
                u.exact_frames, u.catching_up_frames, u.ring.cells_sync, u.ring.cells_async, u.ring.columns,
                u.ring.uploads_full, u.ring.uploads_partial, u.ring.upload_bytes as f64 / 1048576.0,
                source_planes(snap),
            );
            super::text_inside(&painter, Pos2::new(plot.left(), plot.top() + 2.0), egui::Align2::LEFT_TOP,
                line, egui::FontId::monospace(10.0), Color32::from_rgb(255, 220, 80));
        }
        action
    }

}

/// What the worker keeps per plane shown, while it is shown. A pyramid is
/// allowed while the planes' pyramids fit the budget (about one byte per
/// frame per analysed bar each).
fn accel_for(store: &mut Vec<(Arc<PreFrames>, Arc<Accel>)>, planes: &[(&'static str, Arc<PreFrames>)], data_bars: usize) -> Vec<Arc<Accel>> {
    store.retain(|(p, _)| planes.iter().any(|(_, q)| Arc::ptr_eq(p, q)));
    let mut total: usize = store.iter().filter(|(_, a)| a.pyramid_ok).map(|(p, _)| p.len() * data_bars).sum();
    planes.iter().map(|(_, p)| {
        if let Some((_, a)) = store.iter().find(|(q, _)| Arc::ptr_eq(p, q)) {
            return a.clone();
        }
        let need = p.len() * data_bars;
        let pyramid_ok = total + need <= PYRAMID_BUDGET;
        if pyramid_ok {
            total += need;
        }
        let a = Arc::new(Accel { pyramid_ok, ..Default::default() });
        store.push((p.clone(), a.clone()));
        a
    }).collect()
}

impl WholeTrackView {
    /// Whether the camera follows the playhead now: on, not set aside, and
    /// zoomed in (the whole track has nowhere to scroll).
    fn following(&self, view: &View) -> bool {
        let frames = self.snapshot.as_ref().map_or(0.0, |s| s.frames() as f64);
        self.follow && !self.follow_suspended && view.t1 - view.t0 < frames
    }

    fn fit(&mut self) {
        if let Some(s) = &self.snapshot {
            self.view = Some(View::full(s.frames(), s.geometry.bars));
        }
    }

    fn zoom_at(&mut self, at: (f64, f64), kt: f64, kf: f64) {
        if let (Some(v), Some(s)) = (self.view, &self.snapshot) {
            self.view = Some(zoom_view(v, at, kt, kf, s.frames() as f64, s.geometry.bars as f64));
        }
    }

    /// The worker, for the tests.
    #[cfg(test)]
    pub(crate) fn worker(&mut self) -> &mut Worker {
        &mut self.worker
    }

    #[cfg(test)]
    pub(crate) fn view(&self) -> Option<View> {
        self.view
    }

    #[cfg(test)]
    pub(crate) fn readout_for_test(&self) -> &str { &self.readout }

    #[cfg(test)]
    pub(crate) fn colours_for_test(&self) -> Option<&[Color32]> { self.colour_table.as_ref().map(|t| t.colours.as_slice()) }

    #[cfg(test)]
    pub(crate) fn set_view_for_test(&mut self, v: View) {
        self.view = Some(v);
    }

    #[cfg(test)]
    pub(crate) fn unavailable_for_test(&self) -> Option<Unavailable> {
        self.unavailable.clone()
    }

    #[cfg(test)]
    pub(crate) fn snapshot(&self) -> Option<&Snapshot> {
        self.snapshot.as_ref()
    }

    /// Complete working rasters, not a retained previous presentation.
    #[cfg(test)]
    pub(crate) fn shown(&self) -> Vec<Raster> {
        let v: Vec<Raster> = self.rings.iter().filter_map(Ring::raster).collect();
        if v.len() == self.rings.len() { v } else { Vec::new() }
    }

    /// What the view holds and has done so far, for the measurement log
    /// (JSON; see `frame_log`).
    pub fn log_note(&self) -> String {
        let (u, r) = (&self.usage, self.usage.ring);
        let v = self.view.map_or("null".to_string(), |v| format!("[{:.3},{:.3},{:.3},{:.3}]", v.t0, v.t1, v.b0, v.b1));
        let visible = self.presented.as_ref().map(|p| p.pictures[0].view)
            .or_else(|| self.rings.first().and_then(|r| r.painted));
        let pv = visible.map_or("null".to_string(), |v| format!("[{:.3},{:.3},{:.3},{:.3}]", v.t0, v.t1, v.b0, v.b1));
        format!(
            "{{\"open\":{},\"view\":{v},\"presented_view\":{pv},\"follow\":{},\"suspended\":{},\"exact_frames\":{},\"other_frames\":{},\
             \"retained_frames\":{},\"requests\":{},\"cells_sync\":{},\"cells_async\":{},\"columns\":{},\"rerows\":{},\"uploads_full\":{},\
             \"uploads_partial\":{},\"upload_bytes\":{},\"async_jobs\":{},\"cpu_bytes\":{},\"gpu_bytes\":{}}}",
            self.open, self.follow, self.follow_suspended, u.exact_frames, u.catching_up_frames, u.retained_frames, u.requests, r.cells_sync,
            r.cells_async, r.columns, r.rerows, r.uploads_full, r.uploads_partial, r.upload_bytes, r.async_jobs, u.cpu_bytes,
            u.gpu_bytes,
        )
    }

    /// Each ring's texture, its painted left edge (pixels) and width, and the
    /// rectangle it was drawn into, for the mapping test.
    #[cfg(test)]
    pub(crate) fn ring_textures_for_test(&self) -> Vec<RingTexture> {
        if let Some(p) = &self.presented {
            return p.pictures.iter().map(|r| (r.tex.id(), r.base, r.w, r.h, r.top)).collect();
        }
        self.rings.iter().filter_map(|r| Some((r.tex.as_ref()?.id(), r.base, r.w, r.h, r.top_texture_row()))).collect()
    }

    /// The camera painted last frame, if every ring drew all of it as data.
    /// Taken from the frame's own record of what it asked for, not from the
    /// rings: a ring that still holds an older camera is not evidence.
    #[cfg(test)]
    pub(crate) fn exact_camera(&self) -> Option<View> {
        if self.presented.is_some() { return None; }
        let v = self.painted_for_test?;
        self.rings.iter().all(|r| r.painted == Some(v) && r.pending.is_empty() && r.shown == Some(v)).then_some(v)
    }

    /// What each plane's ring drew last frame for the painted camera, column
    /// by column (`None`: drawn as loading). Panics if a ring painted any
    /// other camera than the one the frame asked for.
    #[cfg(test)]
    pub(crate) fn drawn(&self) -> Vec<Drawn> {
        let v = self.painted_for_test.expect("nothing painted");
        if let Some(p) = &self.presented {
            return p.pictures.iter().map(|r| {
                assert_eq!(r.view, v, "retained image and visible axes disagree");
                (r.view, r.w, r.h, r.columns.clone())
            }).collect();
        }
        self.rings.iter().map(|r| {
            let (pv, cols) = r.drawn().expect("a ring painted nothing");
            assert_eq!(pv, v, "a ring painted another camera than the one asked for");
            (v, r.w, r.h, cols)
        }).collect()
    }
}

fn channel_note(c: &super::PreChannels) -> String {
    match c {
        super::PreChannels::Missing => "This analysis has no Left/Right planes.".into(),
        super::PreChannels::OnDisk | super::PreChannels::Loading => "Loading Left/Right…".into(),
        super::PreChannels::Refused(e) => format!("Left/Right could not be read: {e}"),
        super::PreChannels::Resident(_) => "Left/Right are being prepared…".into(),
    }
}

fn provenance(s: &Snapshot) -> String {
    let g = &s.geometry;
    let origin = match &s.origin {
        Origin::Store { generation, .. } => {
            format!("cached analysis (generation {}…)", &format!("{generation:032x}")[..12])
        }
        Origin::Memory => "analysis shown from memory (not cached)".into(),
        Origin::Legacy { .. } => "LEGACY analysis — not checked against the audio file as it is now".into(),
    };
    let scale = match g.scale {
        super::freq_scale::FreqScale::Log => "log".to_string(),
        super::freq_scale::FreqScale::Erb => "ERB".to_string(),
        super::freq_scale::FreqScale::Blend(t) => format!("log/ERB blend {t:.2}"),
        super::freq_scale::FreqScale::Lens { .. } => "lens".to_string(),
    };
    let nyq = if g.data_bars < g.bars {
        format!("; bars above {:.0} Hz (Nyquist) hold no data", g.sample_rate as f32 / 2.0)
    } else {
        String::new()
    };
    format!(
        "{} · {} · {} bars, {scale} scale, {:.0} Hz–{:.0} Hz{nyq} · {:.2} frames/s (hop {} at {} Hz) · {}",
        g.transform.label(),
        origin,
        g.bars,
        g.lo,
        g.axis_hi,
        g.frame_rate(),
        g.hop,
        g.sample_rate,
        "levels: stored display codes, −80…0 dB clipped, not dBFS; no smoothing or ISO 226",
    )
}

fn uv1() -> Rect {
    Rect::from_min_max(Pos2::ZERO, Pos2::new(1.0, 1.0))
}

/// `v` with its left edge on the pixel grid of a `w`-pixel raster (within
/// half a pixel), still inside the track.
/// The camera on the pixel grid of a `w × h` raster: the time window's left
/// edge on a whole column, the frequency window's bottom edge on a whole row
/// (each within half a pixel of where it was asked to be), so that a pan by
/// whole pixels finds its columns and rows already made. Spans are kept.
pub(crate) fn snap_view(v: View, (w, h): (usize, usize), frames: f64, bars: f64) -> View {
    let mut out = v;
    let span_t = v.t1 - v.t0;
    if span_t < frames && w > 0 {
        let d = span_t / w as f64;
        out.t0 = (v.t0 / d).round() * d;
        out.t1 = out.t0 + span_t;
    }
    let span_b = v.b1 - v.b0;
    if span_b < bars && h > 0 {
        let d = span_b / h as f64;
        out.b0 = (v.b0 / d).round() * d;
        out.b1 = out.b0 + span_b;
    }
    if out == v { v } else { clamp_view(out, frames, bars) }
}

fn palette_lut(pal: &Palette, snap: &Snapshot) -> Vec<Color32> {
    (0..=snap.max_code as usize).map(|c| pal.heat(c as f32 * snap.inv_max)).collect()
}

fn add_stats(a: RingStats, b: RingStats) -> RingStats {
    RingStats {
        cells_sync: a.cells_sync + b.cells_sync,
        cells_async: a.cells_async + b.cells_async,
        columns: a.columns + b.columns,
        rerows: a.rerows + b.rerows,
        uploads_full: a.uploads_full + b.uploads_full,
        uploads_partial: a.uploads_partial + b.uploads_partial,
        upload_bytes: a.upload_bytes + b.upload_bytes,
        sync_frames: a.sync_frames + b.sync_frames,
        async_jobs: a.async_jobs + b.async_jobs,
    }
}

/// Frame and bar position of a screen point.
fn to_data(v: &View, r: Rect, p: Pos2) -> (f64, f64) {
    let fx = ((p.x - r.left()) / r.width()) as f64;
    let fy = ((r.bottom() - p.y) / r.height()) as f64;
    (v.t0 + fx * (v.t1 - v.t0), v.b0 + fy * (v.b1 - v.b0))
}

/// Keep a view's spans within limits and inside the track.
pub(crate) fn clamp_view(v: View, frames: f64, bars: f64) -> View {
    let fit = |a0: f64, a1: f64, min: f64, max: f64| {
        let span = (a1 - a0).clamp(min.min(max), max);
        let a0 = a0.clamp(0.0, (max - span).max(0.0));
        (a0, a0 + span)
    };
    let (t0, t1) = fit(v.t0, v.t1, MIN_FRAMES, frames);
    let (b0, b1) = fit(v.b0, v.b1, MIN_BARS, bars);
    View { t0, t1, b0, b1 }
}

/// Zoom by `kt` in time and `kf` in frequency, keeping the data point `at`
/// where it is on screen.
pub(crate) fn zoom_view(v: View, at: (f64, f64), kt: f64, kf: f64, frames: f64, bars: f64) -> View {
    let (t, b) = at;
    let z = View {
        t0: t - (t - v.t0) * kt,
        t1: t + (v.t1 - t) * kt,
        b0: b - (b - v.b0) * kf,
        b1: b + (v.b1 - b) * kf,
    };
    // Clamping the span first and then sliding keeps the anchor fixed except
    // where an edge of the track stops it.
    let span_t = (z.t1 - z.t0).clamp(MIN_FRAMES.min(frames), frames);
    let span_b = (z.b1 - z.b0).clamp(MIN_BARS.min(bars), bars);
    let ft = (t - v.t0) / (v.t1 - v.t0);
    let fb = (b - v.b0) / (v.b1 - v.b0);
    clamp_view(
        View { t0: t - ft * span_t, t1: t - ft * span_t + span_t, b0: b - fb * span_b, b1: b - fb * span_b + span_b },
        frames,
        bars,
    )
}

fn upload(ctx: &egui::Context, name: &str, r: Raster, pal: &Palette, snap: Option<&Snapshot>) -> Shown {
    let tex = ctx.load_texture(name, colour(&r, pal, snap), egui::TextureOptions::NEAREST);
    Shown { raster: r, tex, colours: PalKey::of(pal) }
}

/// Codes to colours. Data through the palette, exactly `code / max_code`;
/// [`NO_DATA`] as a hatch no palette uses; [`OUTSIDE`] transparent.
fn colour(r: &Raster, pal: &Palette, snap: Option<&Snapshot>) -> egui::ColorImage {
    let max = snap.map_or(4095, |s| s.max_code) as usize;
    let inv = snap.map_or(1.0 / 4095.0, |s| s.inv_max);
    let lut: Vec<Color32> = (0..=max).map(|c| pal.heat(c as f32 * inv)).collect();
    let mut px = vec![Color32::TRANSPARENT; r.w * r.h];
    for (i, (&c, out)) in r.codes.iter().zip(px.iter_mut()).enumerate() {
        *out = match c {
            OUTSIDE => Color32::TRANSPARENT,
            NO_DATA => hatch(i % r.w, i / r.w),
            c => lut[(c as usize).min(max)],
        };
    }
    egui::ColorImage { size: [r.w, r.h], pixels: px }
}

fn hatch(x: usize, y: usize) -> Color32 {
    if (x + y) % 8 < 4 { Color32::from_rgb(52, 46, 58) } else { Color32::from_rgb(30, 28, 34) }
}

fn count(n: usize, unit: &str) -> String {
    if n == 1 { format!("1 {unit}") } else { format!("{n} {unit}s") }
}

/// The distinct decoded planes the snapshot holds — Mix, and Left and Right
/// when present — each counted once however many `Arc`s point at it (the
/// overview strip draws the same Mix).
fn source_planes(s: &Snapshot) -> String {
    let mib = |p: &PreFrames| p.capacity_bytes() as f64 / 1048576.0;
    match &s.lr {
        Some((l, r)) => {
            let lr = mib(l) + mib(r);
            format!("Mix {:.1} + L/R {:.1} = {:.1} MiB", mib(&s.mix), lr, mib(&s.mix) + lr)
        }
        None => format!("Mix {:.1} MiB", mib(&s.mix)),
    }
}

/// Frequency ticks that fit: the standard ones, skipping any closer than a
/// label's height to the last drawn, and when fewer than two fit (a deep
/// frequency zoom), the frequencies at the plot's top and bottom edges.
pub(crate) fn freq_ticks(r_height: f32, v: &View, g: &Geometry) -> Vec<(f32, f32, String)> {
    const MIN_GAP: f32 = 13.0;
    let nyq = g.sample_rate as f32 / 2.0;
    let y_of = |b: f64| r_height - ((b - v.b0) / (v.b1 - v.b0)) as f32 * r_height;
    let mut out: Vec<(f32, f32, String)> = Vec::new();
    for &f in super::FREQ_TICKS {
        if f < g.lo || f > g.axis_hi || f > nyq {
            continue;
        }
        let b = g.bar_pos(f);
        if b < v.b0 || b > v.b1 {
            continue;
        }
        let y = y_of(b);
        if out.last().is_some_and(|(_, ly, _)| (ly - y).abs() < MIN_GAP) {
            continue;
        }
        out.push((f, y, super::tick_label(f)));
    }
    if out.len() < 2 {
        let at = |b: f64| g.scale.freq_at((b / g.bars as f64) as f32, g.lo, g.axis_hi);
        let label = |f: f32| if f >= 1000.0 { format!("{:.2}k", f / 1000.0) } else { format!("{f:.0}") };
        let (top, bottom) = (at(v.b1), at(v.b0));
        out.retain(|(_, y, _)| *y > MIN_GAP && *y < r_height - MIN_GAP);
        out.push((top, 6.0, label(top)));
        out.push((bottom, r_height - 6.0, label(bottom)));
    }
    out
}

fn freq_axis(p: &egui::Painter, r: Rect, v: &View, g: &Geometry) {
    let grey = Color32::from_gray(90);
    for (_, y, text) in freq_ticks(r.height(), v, g) {
        let y = r.top() + y;
        p.line_segment([Pos2::new(r.left() - 4.0, y), Pos2::new(r.left(), y)], Stroke::new(1.0, Color32::from_gray(60)));
        super::text_inside(p, Pos2::new(r.left() - 6.0, y), egui::Align2::RIGHT_CENTER,
            text, egui::FontId::monospace(9.0), grey);
    }
    super::text_inside(p, Pos2::new(r.left() - AXIS_W + 2.0, r.top()), egui::Align2::LEFT_TOP, "Hz".into(),
        egui::FontId::monospace(9.0), grey);
}

fn fmt_time(t: f64, step: f64) -> String {
    let m = (t / 60.0).floor();
    let s = t - m * 60.0;
    if step < 0.1 {
        format!("{m:.0}:{s:06.3}")
    } else if step < 1.0 {
        format!("{m:.0}:{s:04.1}")
    } else {
        format!("{m:.0}:{s:02.0}")
    }
}

fn time_axis(p: &egui::Painter, r: Rect, v: &View, g: &Geometry) {
    let grey = Color32::from_gray(90);
    let (s0, s1) = (g.frame_time(v.t0), g.frame_time(v.t1));
    let per_px = (s1 - s0) / r.width().max(1.0) as f64;
    const STEPS: [f64; 17] = [0.005, 0.01, 0.02, 0.05, 0.1, 0.2, 0.5, 1.0, 2.0, 5.0, 10.0, 15.0, 30.0, 60.0, 120.0, 300.0, 600.0];
    let step = STEPS.iter().copied().find(|s| s / per_px >= 70.0).unwrap_or(1200.0);
    let mut t = (s0 / step).ceil() * step;
    while t <= s1 {
        let x = r.left() + ((t - s0) / (s1 - s0)) as f32 * r.width();
        p.line_segment([Pos2::new(x, r.bottom()), Pos2::new(x, r.bottom() + 3.0)], Stroke::new(1.0, Color32::from_gray(60)));
        super::text_inside(p, Pos2::new(x, r.bottom() + 3.0), egui::Align2::CENTER_TOP, fmt_time(t, step),
            egui::FontId::monospace(9.0), grey);
        t += step;
    }
}

fn legend(p: &egui::Painter, r: Rect, pal: &Palette) {
    let bar = Rect::from_min_max(Pos2::new(r.left(), r.top() + 12.0), Pos2::new(r.left() + 10.0, r.bottom() - 24.0));
    let n = 64;
    for i in 0..n {
        let v = 1.0 - i as f32 / n as f32;
        let y0 = bar.top() + i as f32 / n as f32 * bar.height();
        let y1 = bar.top() + (i + 1) as f32 / n as f32 * bar.height();
        p.rect_filled(Rect::from_min_max(Pos2::new(bar.left(), y0), Pos2::new(bar.right(), y1 + 0.5)), 0.0, pal.heat(v));
    }
    let grey = Color32::from_gray(110);
    let f = egui::FontId::monospace(9.0);
    super::text_inside(p, Pos2::new(r.left(), r.top()), egui::Align2::LEFT_TOP, "level".into(), f.clone(), grey);
    for (db, lbl) in [(0.0, "≥0 dB"), (-20.0, "−20"), (-40.0, "−40"), (-60.0, "−60"), (-80.0, "≤−80")] {
        let y = bar.top() + (-db / 80.0) * bar.height();
        super::text_inside(p, Pos2::new(bar.right() + 3.0, y), egui::Align2::LEFT_CENTER, lbl.into(), f.clone(), grey);
    }
    let sw = Rect::from_min_max(Pos2::new(r.left(), r.bottom() - 12.0), Pos2::new(r.left() + 10.0, r.bottom() - 2.0));
    p.rect_filled(sw, 0.0, hatch(0, 0));
    super::text_inside(p, Pos2::new(sw.right() + 3.0, sw.center().y), egui::Align2::LEFT_CENTER, "no data".into(), f, grey);
}

/// The readout for pixel `(x, y)` of a `w × h` raster of `view`, read from the
/// plane itself: the block the pixel stands for, and its largest cell.
pub(crate) fn hover(
    plane: &PreFrames,
    view: &View,
    (w, h): (usize, usize),
    (x, y): (usize, usize),
    s: &Snapshot,
    name: &str,
) -> String {
    let g = &s.geometry;
    let (f0, f1) = span(view.t0, view.t1, w, x, plane.len());
    let (g0, g1) = span(view.b0, view.b1, h, h - 1 - y, plane.bars());
    if f1 <= f0 || g1 <= g0 {
        return format!("{name}: outside the track");
    }
    let time = |k: usize| g.frame_time(k as f64);
    let when = if f1 - f0 == 1 {
        format!("frame {f0} @ {:.3} s", time(f0))
    } else {
        format!("frames {f0}–{} ({:.3}–{:.3} s)", f1 - 1, time(f0), time(f1))
    };
    if g0 >= g.data_bars {
        let (lo, _) = g.bar_edges(g0);
        return format!("{name}: {when} · above Nyquist ({lo:.0} Hz and up): no analysed data");
    }
    let Some((code, kf, kb)) = block_max(plane, (f0, f1), (g0, g1), g.data_bars) else {
        return format!("{name}: {when} · no cell");
    };
    let (lo, hi) = g.bar_edges(kb);
    let db = code_db(code, s.inv_max);
    let level = if code == 0 {
        "≤ −80 dB (floor, clipped — not proof of silence)".to_string()
    } else if code == s.max_code {
        "≥ 0 dB (top, clipped)".to_string()
    } else {
        format!("{db:.1} dB display level")
    };
    let block = if f1 - f0 > 1 || g1 - g0 > 1 {
        format!(" · max of {} × {}, at frame {kf}", count(f1 - f0, "frame"), count(g1 - g0, "bar"))
    } else {
        String::new()
    };
    format!(
        "{name}: {when} · bar {kb} {lo:.1}–{hi:.1} Hz (centre {:.1}) · code {code}/{} → {level}{block} · {}",
        g.bar_center(kb),
        s.max_code,
        g.transform.anchor_note(time(kf), g.sample_rate),
    )
}
