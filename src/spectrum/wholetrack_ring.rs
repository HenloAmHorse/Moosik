//! The whole-track view's main picture, kept up to date one pixel column at
//! a time — exactly.
//!
//! # Why columns
//!
//! A pixel's code depends on two things only: the frames its column spans
//! and the bars its row spans ([`span`]). So a column's per-bar maxima
//! (`colmax`, one value per analysed bar) are a function of its frame span
//! alone, and every pixel of it follows from them and its row span
//! ([`pixel_code`]). Keyed by the span, never by the camera that asked:
//!
//! * when only the time window moves (a pan, the follow camera), each column
//!   whose span is unchanged keeps its maxima and its codes, and only the
//!   columns whose span is new read the plane;
//! * when only the frequency window moves (Shift+wheel, a vertical pan), no
//!   column reads the plane: the rows are re-reduced from the maxima;
//! * when the time scale changes (wheel), every span is new and every column
//!   is read again, as before.
//!
//! A column is reused only if its span is *equal* to the one its maxima were
//! made for. Nothing is assumed about how floating-point spans move with the
//! camera, so the result is the reference raster of the current camera by
//! construction, whatever the camera — a fractional offset just reuses
//! fewer columns. The follow camera and pans are snapped to the pixel grid
//! (see `WholeTrackView`), which is what makes the spans recur.
//!
//! # Where the columns live
//!
//! In a ring of `w` slots: camera column `x` sits in slot `(base + x) mod w`,
//! where `base` is the camera's left edge in pixel units. A one-pixel scroll
//! moves `base` by one, frees one slot and fills it; the texture has the same
//! ring layout, so only the columns that changed are uploaded.
//!
//! The rows are a ring too, the same way: a row's code depends only on its
//! bar span, so rows are keyed by it in `h` row slots — the camera's `j`-th
//! row from the bottom in slot `(base_b + j) mod h`, `base_b` being the
//! frequency window's bottom edge in row units (the frequency window sits on
//! the row grid as the time window sits on the column grid). A vertical pan
//! re-reduces only the rows whose span is new, from the held maxima, and
//! uploads only those rows. The texture keeps slot `rs` in its row
//! `h − 1 − rs`, so camera rows run down it contiguously, wrapping; it is
//! drawn as up to four rectangles split at the two wraps.
//!
//! # What is drawn
//!
//! Only the camera asked for this frame (the *painted* camera), and of it only
//! the columns whose maxima the ring holds for exactly their span. A column it
//! does not hold yet (the worker is reading it) is drawn as **loading**, never
//! as another camera's picture moved or scaled into place: a time zoom, a seek
//! or a long drag that needs more than [`SYNC_CELLS`] shows loading columns
//! until the worker's columns land, and every column drawn as data is the
//! reference for its span.
use eframe::egui::{self, Color32, Pos2, Rect};

#[cfg(test)]
use super::cache::PreFrames;
#[cfg(test)]
use super::wholetrack::column_max;
use super::wholetrack::{pixel_code, span, View, OUTSIDE};
#[cfg(test)]
use super::wholetrack::Raster;

/// Source cells the UI thread reads itself in one frame, counted through the
/// accelerators it can use as they are (see `wholetrack::held_cost`). A
/// camera change that needs more goes to the worker, and its columns show as
/// loading until they land. About a millisecond of reading.
pub const SYNC_CELLS: usize = 2 << 20;

/// What a loading column is filled with: flat, darker than any data colour's
/// floor and unlike the no-data hatch, with "loading…" written over the plot.
pub const LOADING: Color32 = Color32::from_rgb(26, 29, 40);

/// What a camera needs from the ring.
pub struct Plan {
    pub view: View,
    pub base: i64,
    /// Each camera column's frame span.
    pub spans: Vec<(usize, usize)>,
    /// Camera columns whose maxima the ring does not hold.
    pub missing: Vec<usize>,
}

impl Plan {
    /// Missing spans in screen order, once per distinct interval. Magnified
    /// neighbouring columns often show the same stored frame: the query is
    /// shared, although every destination column still has its own codes.
    pub fn read_spans(&self) -> impl Iterator<Item = (usize, usize)> + '_ {
        let mut last = None;
        self.missing.iter().filter_map(move |&x| {
            let sp = self.spans[x];
            if last == Some(sp) {
                None
            } else {
                last = Some(sp);
                Some(sp)
            }
        })
    }
}

/// Counters for the debug overlay, the tests and the benchmark.
#[derive(Clone, Copy, Default, Debug, PartialEq)]
pub struct RingStats {
    /// Source cells read on this thread and by the worker.
    pub cells_sync: u64,
    pub cells_async: u64,
    /// Columns recomputed (maxima), and rows re-reduced without reading
    /// (each row of every held column; one count per row).
    pub columns: u64,
    pub rerows: u64,
    /// Texture uploads: whole, partial, and the bytes they carried.
    pub uploads_full: u64,
    pub uploads_partial: u64,
    pub upload_bytes: u64,
    /// Camera changes satisfied on this thread, and sent to the worker.
    pub sync_frames: u64,
    pub async_jobs: u64,
}

/// A complete GPU presentation, without copying the source or CPU ring.
/// Its texture must not be updated until this presentation is retired.
/// Never used at another raster size.
pub struct Picture {
    pub view: View,
    pub w: usize,
    pub h: usize,
    pub base: i64,
    pub top: usize,
    pub tex: egui::TextureHandle,
    pub block: (usize, usize),
    #[cfg(test)]
    pub columns: Vec<Option<Vec<u32>>>,
}

impl Picture {
    pub fn draw(&self, p: &egui::Painter, r: Rect) {
        draw_image(self.tex.id(), self.w, self.h, self.base, self.top, p, r);
    }
}

fn draw_image(tex: egui::TextureId, width: usize, height: usize, base: i64, top: usize, p: &egui::Painter, r: Rect) {
    let (w, h) = (width as f32, height as f32);
    let s0 = base.rem_euclid(width as i64) as f32;
    let t0 = top as f32;
    let xs = [(0.0, w - s0, s0), (w - s0, w, s0 - w)];
    let ys = [(0.0, h - t0, t0), (h - t0, h, t0 - h)];
    for &(x0, x1, dx) in &xs {
        for &(y0, y1, dy) in &ys {
            if x1 <= x0 || y1 <= y0 { continue; }
            let dest = Rect::from_min_max(
                Pos2::new(r.left() + x0 / w * r.width(), r.top() + y0 / h * r.height()),
                Pos2::new(r.left() + x1 / w * r.width(), r.top() + y1 / h * r.height()));
            let uv = Rect::from_min_max(Pos2::new((x0 + dx) / w, (y0 + dy) / h), Pos2::new((x1 + dx) / w, (y1 + dy) / h));
            p.image(tex, dest, uv, Color32::WHITE);
        }
    }
}

pub struct Ring {
    pub w: usize,
    pub h: usize,
    data_bars: usize,
    /// Per row slot: the bar span its codes are for; the frequency window
    /// they were set for, and its bottom edge in rows.
    rows: Vec<Option<(usize, usize)>>,
    rows_key: Option<(u64, u64)>,
    pub base_b: i64,
    /// Row slots whose texture row is stale.
    row_dirty: Vec<bool>,
    /// Per slot: the frame span its maxima and codes are for.
    span: Vec<Option<(usize, usize)>>,
    /// Per slot, `data_bars` maxima each.
    colmax: Vec<u16>,
    /// Row-slot-major, `w` per row slot, in column-slot order.
    codes: Vec<u32>,
    /// Slots whose texture column is stale.
    dirty: Vec<bool>,
    all_dirty: bool,
    /// The camera drawn this frame (`painted`), the camera columns of it the
    /// ring does not hold (drawn as loading), and `shown`: the painted camera
    /// when nothing of it is pending. `base` is the painted camera's left edge
    /// in pixels.
    pub painted: Option<View>,
    pub pending: Vec<usize>,
    pub shown: Option<View>,
    pub base: i64,
    pub tex: Option<egui::TextureHandle>,
    pub stats: RingStats,
}

impl Ring {
    pub fn new(w: usize, h: usize, data_bars: usize) -> Self {
        Ring {
            w,
            h,
            data_bars,
            rows: vec![None; h],
            rows_key: None,
            base_b: 0,
            row_dirty: vec![false; h],
            span: vec![None; w],
            colmax: vec![0; w * data_bars],
            codes: vec![OUTSIDE; w * h],
            dirty: vec![true; w],
            all_dirty: true,
            painted: None,
            pending: Vec::new(),
            shown: None,
            base: 0,
            tex: None,
            stats: RingStats::default(),
        }
    }

    /// Bytes this ring holds on the CPU (maxima and codes) and on the GPU.
    pub fn cpu_bytes(&self) -> usize {
        self.colmax.capacity() * 2 + self.codes.capacity() * 4
    }
    pub fn gpu_bytes(&self) -> usize {
        if self.tex.is_some() { self.w * self.h * 4 } else { 0 }
    }

    fn slot(&self, base: i64, x: usize) -> usize {
        (base + x as i64).rem_euclid(self.w as i64) as usize
    }

    /// The row slot of the camera's `j`-th row from the bottom.
    fn rslot(&self, base_b: i64, j: usize) -> usize {
        (base_b + j as i64).rem_euclid(self.h as i64) as usize
    }

    /// The row slot camera row `y` (top to bottom) shows.
    fn row_of(&self, y: usize) -> usize {
        self.rslot(self.base_b, self.h - 1 - y)
    }

    /// The pixel-grid index of a camera's left edge.
    pub fn base_of(view: &View, w: usize) -> i64 {
        let d = (view.t1 - view.t0) / w.max(1) as f64;
        if d > 0.0 { (view.t0 / d).round() as i64 } else { 0 }
    }

    /// What `view` needs that the ring does not hold.
    pub fn plan(&self, view: View, frames: usize) -> Plan {
        let base = Self::base_of(&view, self.w);
        let spans: Vec<(usize, usize)> = (0..self.w).map(|x| span(view.t0, view.t1, self.w, x, frames)).collect();
        let missing = (0..spans.len()).filter(|&x| self.span[self.slot(base, x)] != Some(spans[x])).collect();
        Plan { view, base, spans, missing }
    }

    /// Bring the rows in line with the frequency window: a row slot whose
    /// bar span changes is re-reduced across every held column (no plane is
    /// read); one whose span stays keeps its codes.
    pub fn set_rows(&mut self, view: &View, bars: usize) {
        let key = (view.b0.to_bits(), view.b1.to_bits());
        if self.rows_key == Some(key) {
            return;
        }
        let h = self.h;
        let d = (view.b1 - view.b0) / h.max(1) as f64;
        let base_b = if d > 0.0 { (view.b0 / d).round() as i64 } else { 0 };
        for j in 0..h {
            let want = span(view.b0, view.b1, h, j, bars);
            let rs = self.rslot(base_b, j);
            if self.rows[rs] != Some(want) {
                self.rows[rs] = Some(want);
                self.recode_row(rs);
            }
        }
        self.base_b = base_b;
        self.rows_key = Some(key);
        for v in [&mut self.shown, &mut self.painted].into_iter().flatten() {
            v.b0 = view.b0;
            v.b1 = view.b1;
        }
    }

    fn column_maxima(&self, s: usize) -> Option<&[u16]> {
        let db = self.data_bars;
        let outside = self.span[s].is_none_or(|(f0, f1)| f1 <= f0);
        (!outside).then(|| &self.colmax[s * db..(s + 1) * db])
    }

    /// Re-reduce one row slot across every column slot.
    fn recode_row(&mut self, rs: usize) {
        let (db, w) = (self.data_bars, self.w);
        let Some(r) = self.rows[rs] else { return };
        for s in 0..w {
            let c = pixel_code(self.column_maxima(s), r, db);
            self.codes[rs * w + s] = c;
        }
        self.row_dirty[rs] = true;
        self.stats.rerows += 1;
    }

    /// Re-reduce one column slot across every row slot.
    fn recode(&mut self, s: usize) {
        let (db, w) = (self.data_bars, self.w);
        for rs in 0..self.h {
            if let Some(r) = self.rows[rs] {
                let c = pixel_code(self.column_maxima(s), r, db);
                self.codes[rs * w + s] = c;
            }
        }
        self.dirty[s] = true;
    }

    /// Store one column's maxima for `span` in slot `s`.
    pub fn put(&mut self, s: usize, sp: (usize, usize), colmax: &[u16]) {
        self.store_column(s, sp, colmax);
        self.recode(s);
    }

    fn store_column(&mut self, s: usize, sp: (usize, usize), colmax: &[u16]) {
        let db = self.data_bars;
        self.colmax[s * db..(s + 1) * db].copy_from_slice(colmax);
        self.span[s] = Some(sp);
        self.stats.columns += 1;
    }

    /// Retained only for reproducing the rejected locality experiment.
    /// Matched measurements found it slower than scalar coding.
    #[cfg(test)]
    fn recode_slots(&mut self, slots: &[usize]) {
        if slots.len() < 8 {
            for &s in slots { self.recode(s); }
            return;
        }
        for band in slots.chunks(32) {
            for rs in 0..self.h {
                let Some(row) = self.rows[rs] else { continue };
                for &s in band {
                    self.codes[rs * self.w + s] = pixel_code(self.column_maxima(s), row, self.data_bars);
                }
            }
        }
        for &s in slots { self.dirty[s] = true; }
    }

    pub fn put_landed_columns(&mut self, plan: &Plan, columns: &[(usize, (usize, usize))], maxima: &[u16]) {
        for (k, &(s, sp)) in columns.iter().enumerate() {
            self.put_landed(plan, s, sp, &maxima[k * self.data_bars..(k + 1) * self.data_bars]);
        }
    }

    /// Store a worker column only if this slot needs that exact span in the
    /// current camera. Obsolete completions must not spend time recoding or
    /// uploading invisible data, or replace a current column with stale data.
    pub fn put_landed(&mut self, plan: &Plan, s: usize, sp: (usize, usize), colmax: &[u16]) {
        let x = (s as i64 - plan.base).rem_euclid(self.w as i64) as usize;
        let need = plan.spans[x];
        // An obsolete zoom result cannot help this camera. Installing it
        // would recode/upload stale columns only to replace them again on
        // the next frame, making the async/loading path especially costly.
        if sp != need {
            return;
        }
        self.put(s, sp, colmax);
    }

    /// Compute every missing column of `plan` here, frame by frame from
    /// `plane`, and paint the camera.
    #[cfg(test)]
    pub fn fill(&mut self, plane: &PreFrames, plan: &Plan) {
        self.fill_with(plan, |sp, out| {
            column_max(plane, sp, out);
            sp.1 - sp.0
        });
    }

    /// Compute every missing column of `plan` here with `read` (a column's
    /// span into its maxima; returns the source rows it read), and paint the
    /// camera.
    pub fn fill_with(&mut self, plan: &Plan, mut read: impl FnMut((usize, usize), &mut [u16]) -> usize) {
        if plan.missing.is_empty() {
            self.stats.sync_frames += 1;
            self.present(plan);
            return;
        }
        let db = self.data_bars;
        let mut buf = vec![0u16; db];
        let mut rows = 0;
        let mut last = None;
        for &x in &plan.missing {
            let sp = plan.spans[x];
            if last != Some(sp) {
                rows += read(sp, &mut buf);
                last = Some(sp);
            }
            let s = self.slot(plan.base, x);
            self.put(s, sp, &buf);
        }
        self.stats.cells_sync += (rows * db) as u64;
        self.stats.sync_frames += 1;
        self.present(plan);
    }

    /// Paint `plan`'s camera: what of it the ring holds is drawn as data, the
    /// rest as loading.
    pub fn present(&mut self, plan: &Plan) {
        self.painted = Some(plan.view);
        self.base = plan.base;
        self.pending = (0..plan.spans.len()).filter(|&x| self.span[self.slot(plan.base, x)] != Some(plan.spans[x])).collect();
        self.shown = self.pending.is_empty().then_some(plan.view);
    }

    /// Paint `plan`'s camera; every column must be held.
    #[cfg(test)]
    pub fn show(&mut self, plan: &Plan) {
        self.present(plan);
        debug_assert!(self.pending.is_empty());
    }

    pub fn mark_all_dirty(&mut self) {
        self.all_dirty = true;
    }

    /// Capture only a fully uploaded, exact picture, before changing codes.
    /// The caller suppresses working uploads until this picture is retired.
    pub fn picture(&self) -> Option<Picture> {
        let view = self.shown?;
        if self.painted != Some(view) || !self.pending.is_empty() || self.all_dirty
            || self.dirty.iter().any(|&d| d) || self.row_dirty.iter().any(|&d| d) { return None; }
        Some(Picture { view, w: self.w, h: self.h, base: self.base,
            top: self.h - 1 - self.row_of(0), tex: self.tex.as_ref()?.clone(), block: self.max_block(),
            #[cfg(test)] columns: self.drawn()?.1,
        })
    }

    /// Upload what changed, coloured by `colour(code, global column, global
    /// row)` — the globals anchor anything drawn to the data (the no-data
    /// hatch). Stale rows go up as row bands, stale columns as column bands;
    /// past half of either, the whole texture once.
    pub fn upload(&mut self, ctx: &egui::Context, name: &str, colour: impl Fn(u32, i64, usize) -> Color32) {
        let (w, h) = (self.w, self.h);
        let (base, base_b) = (self.base, self.base_b);
        let gcol = |s: usize| base + (s as i64 - base).rem_euclid(w as i64);
        let grow = |rs: usize| (base_b + (rs as i64 - base_b).rem_euclid(h as i64)).rem_euclid(1 << 30) as usize;
        // Texture rows tr0..tr0+nr, column slots s0..s0+n.
        let image = |tr0: usize, nr: usize, s0: usize, n: usize, codes: &[u32]| {
            let mut px = Vec::with_capacity(n * nr);
            for tr in tr0..tr0 + nr {
                let rs = h - 1 - tr;
                for s in s0..s0 + n {
                    px.push(colour(codes[rs * w + s], gcol(s), grow(rs)));
                }
            }
            egui::ColorImage { size: [n, nr], pixels: px }
        };
        let rows_dirty = self.row_dirty.iter().filter(|&&d| d).count();
        let cols_dirty = self.dirty.iter().filter(|&&d| d).count();
        if self.tex.is_none() || self.all_dirty || rows_dirty * 2 > h || cols_dirty * 2 > w {
            let img = image(0, h, 0, w, &self.codes);
            match &mut self.tex {
                Some(t) => t.set(img, egui::TextureOptions::NEAREST),
                None => self.tex = Some(ctx.load_texture(name, img, egui::TextureOptions::NEAREST)),
            }
            self.stats.uploads_full += 1;
            self.stats.upload_bytes += (w * h * 4) as u64;
            self.all_dirty = false;
            self.dirty.iter_mut().for_each(|d| *d = false);
            self.row_dirty.iter_mut().for_each(|d| *d = false);
            return;
        }
        // Row bands, in texture order.
        let mut tr = 0;
        while tr < h {
            if !self.row_dirty[h - 1 - tr] {
                tr += 1;
                continue;
            }
            let t0 = tr;
            while tr < h && self.row_dirty[h - 1 - tr] {
                self.row_dirty[h - 1 - tr] = false;
                tr += 1;
            }
            let img = image(t0, tr - t0, 0, w, &self.codes);
            if let Some(t) = &mut self.tex {
                t.set_partial([0, t0], img, egui::TextureOptions::NEAREST);
            }
            self.stats.uploads_partial += 1;
            self.stats.upload_bytes += ((tr - t0) * w * 4) as u64;
        }
        let mut s = 0;
        while s < w {
            if !self.dirty[s] {
                s += 1;
                continue;
            }
            let s0 = s;
            while s < w && self.dirty[s] {
                self.dirty[s] = false;
                s += 1;
            }
            let img = image(0, h, s0, s - s0, &self.codes);
            if let Some(t) = &mut self.tex {
                t.set_partial([s0, 0], img, egui::TextureOptions::NEAREST);
            }
            self.stats.uploads_partial += 1;
            self.stats.upload_bytes += ((s - s0) * h * 4) as u64;
        }
    }

    /// Draw the painted camera into `r` (one texel per pixel): the held
    /// columns from the texture, the pending ones as loading.
    pub fn draw(&self, p: &egui::Painter, r: Rect) {
        let (Some(tex), Some(_)) = (&self.tex, &self.painted) else { return };
        let w = self.w as f32;
        // Camera column 0 is texture column s0; camera row 0 is texture row t0.
        draw_image(tex.id(), self.w, self.h, self.base, self.h - 1 - self.row_of(0), p, r);
        // Pending columns, in runs, covered whole.
        let col = |x: usize| r.left() + x as f32 / w * r.width();
        let mut i = 0;
        while i < self.pending.len() {
            let x0 = self.pending[i];
            let mut x1 = x0 + 1;
            while i + 1 < self.pending.len() && self.pending[i + 1] == x1 {
                i += 1;
                x1 += 1;
            }
            p.rect_filled(Rect::from_min_max(Pos2::new(col(x0), r.top()), Pos2::new(col(x1), r.bottom())), 0.0, LOADING);
            i += 1;
        }
    }

    /// The texture row camera row 0 (the top) is drawn from.
    #[cfg(test)]
    pub fn top_texture_row(&self) -> usize {
        self.h - 1 - self.row_of(0)
    }

    /// What `draw` puts on screen, camera column by camera column: a held
    /// column's codes (top to bottom), or `None` where it draws loading.
    #[cfg(test)]
    pub fn drawn(&self) -> Option<(View, Vec<Option<Vec<u32>>>)> {
        let view = self.painted?;
        let cols = (0..self.w)
            .map(|x| {
                (!self.pending.contains(&x)).then(|| {
                    let s = self.slot(self.base, x);
                    (0..self.h).map(|y| self.codes[self.row_of(y) * self.w + s]).collect()
                })
            })
            .collect();
        Some((view, cols))
    }

    /// The shown camera as an ordinary raster (camera order), for the tests
    /// and the reference comparison.
    #[cfg(test)]
    pub fn raster(&self) -> Option<Raster> {
        let view = self.shown?;
        let (w, h) = (self.w, self.h);
        let mut codes = vec![OUTSIDE; w * h];
        let mut max_block = (0, 0);
        for x in 0..w {
            let s = self.slot(self.base, x);
            if let Some((f0, f1)) = self.span[s] {
                max_block.0 = max_block.0.max(f1.saturating_sub(f0));
            }
            for y in 0..h {
                codes[y * w + x] = self.codes[self.row_of(y) * w + s];
            }
        }
        for &(r0, r1) in self.rows.iter().flatten() {
            max_block.1 = max_block.1.max(r1 - r0);
        }
        Some(Raster { w, h, view, codes, max_block })
    }

    /// The largest block a shown pixel reduces, in frames and bars, without
    /// building a raster.
    pub fn max_block(&self) -> (usize, usize) {
        let f = self.span.iter().flatten().map(|(a, b)| b.saturating_sub(*a)).max().unwrap_or(0);
        let b = self.rows.iter().flatten().map(|(a, b)| b - a).max().unwrap_or(0);
        (f, b)
    }
}

#[cfg(test)]
#[path = "wholetrack_ring_tests.rs"]
mod tests;
