//! The whole-track view: what it may show, and how it turns stored codes into
//! pixels.
//!
//! # What the data is
//!
//! A pre-process analysis is a `frames × bars` matrix of **quantised display
//! levels** ([`PreFrames`]): `v = code / max_code`, shown as `−80 + 80·v` dB on
//! the display's own reference. Both ends are clipped, so code 0 means "at or
//! below the floor", not silence, and the top code means "at or above 0". These
//! are not calibrated dBFS or a power spectral density, and they precede the
//! live display's smoothing and its ISO 226 weighting, neither of which is
//! applied here.
//!
//! # Where its axes come from
//!
//! Only from the analysis record (`identity::AnalysisSettings::record`) the
//! installed matrix was verified against — never from the sliders, which can
//! have moved since. A matrix with no record (a nominal legacy file) has no
//! known geometry and is not drawn; the view says so and offers Re-analyze.
//!
//! * **Frequency.** Bar `i` of `n` is the grid cell `[freq_at(i/n),
//!   freq_at((i+1)/n)]` of the record's [`FreqScale`] over the range the
//!   producer actually spread the bars across. The superlet clamps that range at
//!   Nyquist; the FFT routes do not, so on an FFT analysis whose upper limit is
//!   above Nyquist the bars wholly above it hold no analysed content (the
//!   producer reads the edge bin for them). Those are shown as *no data*, not
//!   as levels.
//! * **Time.** Frame `k` sits at `k·hop / sr` on the player's timeline, which is
//!   where the live display shows it. What that instant *means* differs by
//!   transform and is stated rather than corrected: an FFT frame is the window
//!   **starting** there (`fft` samples long); a superlet frame is the response
//!   **centred** there.
//!
//! # Pixels
//!
//! Each pixel covers a half-open block of frames × bars ([`spans`]). Magnified,
//! a pixel is exactly one stored cell (nearest, no interpolation). Reduced, it
//! is the **maximum** of every cell its block intersects, so a one-frame
//! transient cannot fall between pixels. No pixel is ever more than the cells
//! that produced it.
use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};
use std::sync::{mpsc, Arc};
use std::time::{Duration, Instant};

use super::cache::PreFrames;
use super::freq_scale::FreqScale;

/// A pixel outside the track (before its first frame, after its last, or
/// outside the bar range).
pub const OUTSIDE: u32 = u32::MAX;
/// A pixel over bars that hold no analysed content (above Nyquist on an FFT
/// analysis). Not the floor: the floor is code 0.
pub const NO_DATA: u32 = u32::MAX - 1;

/// Most pixels the main view rasterises across every plane it shows.
///
/// 4 Mi pixels is 16 MiB of codes on the CPU and 16 MiB of texture on the GPU,
/// plus a transient 16 MiB colour image while it is uploaded. Enforced by
/// [`fit_raster`], whatever the window size.
pub const MAX_RASTER_PIXELS: usize = 4 << 20;
/// The overview strip's largest size, in pixels.
pub const OVERVIEW_MAX: [usize; 2] = [4096, 48];

/// The levels' fixed display scale.
pub const FLOOR_DB: f32 = -80.0;
pub const TOP_DB: f32 = 0.0;

/// Display dB of a code at `inv_max` (`1 / max_code`).
pub fn code_db(code: u16, inv_max: f32) -> f32 {
    FLOOR_DB + (TOP_DB - FLOOR_DB) * (code as f32 * inv_max)
}

/// Which transform produced the matrix, as far as the axes need to know.
#[derive(Clone, Copy, PartialEq, Debug)]
pub enum Transform {
    /// An FFT route: `size`-sample windows starting at each frame.
    Fft { size: usize, pad: usize, mapping: Mapping },
    /// The superlet: responses centred on each frame.
    Superlet,
}

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum Mapping {
    Flat,
    Gaussian,
    /// Constant-Q *weighting* of FFT bins. Not a constant-Q transform.
    CqWeighted,
}

impl Transform {
    pub fn label(self) -> &'static str {
        match self {
            Transform::Fft { mapping: Mapping::Flat, .. } => "FFT, flat bar mapping",
            Transform::Fft { mapping: Mapping::Gaussian, .. } => "FFT, Gaussian bar mapping",
            Transform::Fft { mapping: Mapping::CqWeighted, .. } => {
                "FFT, constant-Q-weighted bar mapping (not a CQT)"
            }
            Transform::Superlet => "Superlet",
        }
    }

    /// What frame time `t` stands for, in words, for the hover.
    pub fn anchor_note(self, t: f64, sample_rate: u32) -> String {
        match self {
            Transform::Fft { size, .. } => format!(
                "FFT window {:.3}–{:.3} s (starts at the frame time)",
                t,
                t + size as f64 / sample_rate.max(1) as f64
            ),
            Transform::Superlet => format!("superlet response centred at {t:.3} s"),
        }
    }
}

/// The frequency and time geometry of one analysis, from its record.
#[derive(Clone, Copy, PartialEq, Debug)]
pub struct Geometry {
    pub sample_rate: u32,
    pub hop: usize,
    pub bars: usize,
    /// The record's lower and upper limits.
    pub lo: f32,
    pub hi: f32,
    /// The upper end the bars are actually spread to: `hi` for the FFT routes,
    /// `min(hi, sr/2)` for the superlet.
    pub axis_hi: f32,
    pub scale: FreqScale,
    pub transform: Transform,
    /// Bars `0..data_bars` hold analysed content; the rest are above Nyquist.
    pub data_bars: usize,
}

fn hex_f32(v: &str) -> Option<f32> {
    u32::from_str_radix(v, 16).ok().map(f32::from_bits).filter(|x| x.is_finite())
}

/// The inverse of `identity::scale`.
fn parse_scale(s: &str) -> Option<FreqScale> {
    let mut it = s.split(':');
    let scale = match it.next()? {
        "log" => FreqScale::Log,
        "erb" => FreqScale::Erb,
        "blend" => FreqScale::Blend(hex_f32(it.next()?)?),
        "lens" => FreqScale::Lens {
            tilt: hex_f32(it.next()?)?,
            hz: hex_f32(it.next()?)?,
            oct: hex_f32(it.next()?)?,
            gain: hex_f32(it.next()?)?,
        },
        _ => return None,
    };
    it.next().is_none().then_some(scale)
}

impl Geometry {
    /// Read the geometry a record states. Every field the axes need must be
    /// present and sane, or there is no geometry.
    pub fn parse(record: &str) -> Result<Self, String> {
        let get = |k: &str| {
            record
                .split(';')
                .filter_map(|f| f.split_once('='))
                .find(|(key, _)| *key == k)
                .map(|(_, v)| v)
        };
        let need = |k: &'static str| get(k).ok_or_else(|| format!("the record has no `{k}`"));
        if !record.starts_with(super::identity::SCHEMA) {
            return Err("the record is not one this build writes".into());
        }
        let num = |k: &'static str| -> Result<usize, String> {
            need(k)?.parse::<usize>().map_err(|_| format!("`{k}` is not a number"))
        };
        let freq = |k: &'static str| -> Result<f32, String> {
            hex_f32(need(k)?).filter(|f| *f > 0.0).ok_or_else(|| format!("`{k}` is not a frequency"))
        };
        let sample_rate = num("sr")? as u32;
        let hop = num("hop")?;
        let bars = num("bars")?;
        let (lo, hi) = (freq("lo")?, freq("hi")?);
        let scale = parse_scale(need("scale")?).ok_or("the record's scale is not readable")?;
        if sample_rate == 0 || hop == 0 || bars == 0 || lo >= hi {
            return Err("the record's geometry is degenerate".into());
        }
        let nyquist = sample_rate as f32 / 2.0;
        let (transform, axis_hi) = match need("map")? {
            "superlet" => (Transform::Superlet, hi.min(nyquist)),
            m => {
                let mapping = match m {
                    "flat" => Mapping::Flat,
                    "gauss" => Mapping::Gaussian,
                    "cqw" => Mapping::CqWeighted,
                    _ => return Err(format!("unknown bar mapping `{m}`")),
                };
                (Transform::Fft { size: num("fft")?, pad: num("pad")?, mapping }, hi)
            }
        };
        if lo >= axis_hi {
            return Err("the analysed range is empty below Nyquist".into());
        }
        let mut g = Geometry { sample_rate, hop, bars, lo, hi, axis_hi, scale, transform, data_bars: bars };
        g.data_bars = (0..bars).take_while(|&i| g.bar_edges(i).0 < nyquist).count();
        Ok(g)
    }

    /// Frames per second of the matrix.
    pub fn frame_rate(&self) -> f64 {
        self.sample_rate as f64 / self.hop as f64
    }

    /// Where frame `k` sits on the player's timeline, in seconds.
    pub fn frame_time(&self, k: f64) -> f64 {
        k / self.frame_rate()
    }

    /// The grid cell bar `i` covers, in Hz.
    pub fn bar_edges(&self, i: usize) -> (f32, f32) {
        let n = self.bars as f32;
        (
            self.scale.freq_at(i as f32 / n, self.lo, self.axis_hi),
            self.scale.freq_at((i + 1) as f32 / n, self.lo, self.axis_hi),
        )
    }

    /// The bar's centre frequency, exactly as the producer computed it.
    pub fn bar_center(&self, i: usize) -> f32 {
        self.scale.bar_center(i, self.bars, self.lo, self.axis_hi)
    }

    /// `f` in bar units (0 at the bottom edge of bar 0, `bars` at the top of
    /// the last), for placing a frequency label.
    pub fn bar_pos(&self, f: f32) -> f64 {
        self.scale.position_of(f, self.lo, self.axis_hi) as f64 * self.bars as f64
    }
}

/// A visible region, in frame and bar units. `b0` is the bottom.
#[derive(Clone, Copy, PartialEq, Debug)]
pub struct View {
    pub t0: f64,
    pub t1: f64,
    pub b0: f64,
    pub b1: f64,
}

impl View {
    pub fn full(frames: usize, bars: usize) -> Self {
        View { t0: 0.0, t1: frames as f64, b0: 0.0, b1: bars as f64 }
    }
}

/// The block of cells pixel `i` of `n` covers along one axis that runs from
/// `a0` to `a1`, clipped to `0..limit`. Empty (`lo == hi`) outside.
///
/// Magnified (`a1 − a0 ≤ n`) a pixel is the one cell under its centre.
/// Reduced, it is every cell its interval intersects; neighbouring pixels may
/// share a boundary cell, and no cell between them is skipped.
pub fn span(a0: f64, a1: f64, n: usize, i: usize, limit: usize) -> (usize, usize) {
    let d = (a1 - a0) / n.max(1) as f64;
    let p0 = a0 + i as f64 * d;
    let (lo, hi) = if d <= 1.0 {
        let c = (p0 + 0.5 * d).floor();
        (c, c + 1.0)
    } else {
        (p0.floor(), (p0 + d).ceil())
    };
    let clip = |v: f64| v.clamp(0.0, limit as f64) as usize;
    let (lo, hi) = (clip(lo), clip(hi));
    if lo >= hi { (0, 0) } else { (lo, hi) }
}

/// Every pixel's block along one axis.
pub fn spans(a0: f64, a1: f64, n: usize, limit: usize) -> Vec<(usize, usize)> {
    (0..n).map(|i| span(a0, a1, n, i, limit)).collect()
}

/// Rows run top to bottom, bars bottom to top: row `y` is the span of the
/// mirrored index.
fn row_span(view: &View, h: usize, y: usize, limit: usize) -> (usize, usize) {
    span(view.b0, view.b1, h, h - 1 - y, limit)
}

/// The largest cell in a block, and where it is. `None` when the block is
/// empty or lies wholly above `data_bars`.
pub fn block_max(
    plane: &PreFrames,
    (f0, f1): (usize, usize),
    (g0, g1): (usize, usize),
    data_bars: usize,
) -> Option<(u16, usize, usize)> {
    let g1 = g1.min(data_bars);
    let mut best: Option<(u16, usize, usize)> = None;
    for f in f0..f1 {
        let row = plane.row(f)?;
        for (g, &c) in row[g0.min(g1)..g1].iter().enumerate() {
            if best.is_none_or(|(b, ..)| c > b) {
                best = Some((c, f, g0 + g));
            }
        }
    }
    best
}

/// The per-bar maxima of one pixel column: `out[b]` is the largest code of
/// bar `b` over frames `f0..f1`, for every analysed bar (`out.len()` is the
/// analysed bar count). Cost: `(f1 − f0) × out.len()` cells, read row by row.
pub fn column_max(plane: &PreFrames, (f0, f1): (usize, usize), out: &mut [u16]) {
    let Some(first) = (f0 < f1).then(|| plane.row(f0)).flatten() else {
        out.fill(0);
        return;
    };
    // Magnified columns are a single stored row: copy it, rather than clear
    // the destination and run a maximum against zero for every bar.
    out.copy_from_slice(&first[..out.len()]);
    for f in f0 + 1..f1 {
        let Some(row) = plane.row(f) else { break };
        for (m, &c) in out.iter_mut().zip(row) {
            if c > *m {
                *m = c;
            }
        }
    }
}

/// The smallest block of the pyramid: 2^2 = 4 frames. Smaller blocks would
/// save little reading and cost the most memory.
pub const PYRAMID_FROM: u32 = 2;

/// Maxima of aligned blocks of frames, per bar, at every level from
/// [`PYRAMID_FROM`] up: block `b` of level `j` is frames `b·2^j ..
/// (b+1)·2^j`. Built once per plane, from the plane, and read only for
/// blocks lying wholly inside a column's frames — so a column read through
/// it is the same maximum, cell for cell, as reading every frame. It holds
/// about half as many rows as the plane.
pub struct Pyramid {
    bars: usize,
    /// `levels[k]` is level `PYRAMID_FROM + k`, frame-major (`bars` per block).
    levels: Vec<Vec<u16>>,
}

impl Pyramid {
    /// Build from `plane` (bars `0..data_bars`), or `None` once `cancel` is set.
    pub fn build(plane: &PreFrames, data_bars: usize, cancel: &AtomicBool) -> Option<Pyramid> {
        let (frames, db) = (plane.len(), data_bars.min(plane.bars()));
        let mut levels: Vec<Vec<u16>> = Vec::new();
        let first = 1usize << PYRAMID_FROM;
        let n = frames.div_ceil(first);
        let mut l0 = vec![0u16; n * db];
        for b in 0..n {
            if cancel.load(Ordering::Relaxed) {
                return None;
            }
            column_max(plane, (b * first, ((b + 1) * first).min(frames)), &mut l0[b * db..(b + 1) * db]);
        }
        levels.push(l0);
        loop {
            let prev = levels.last().unwrap();
            let rows = prev.len() / db.max(1);
            if rows <= 1 {
                break;
            }
            let n = rows.div_ceil(2);
            let mut next = vec![0u16; n * db];
            for b in 0..n {
                let out = &mut next[b * db..(b + 1) * db];
                out.copy_from_slice(&prev[2 * b * db..(2 * b + 1) * db]);
                if 2 * b + 1 < rows {
                    for (m, &c) in out.iter_mut().zip(&prev[(2 * b + 1) * db..(2 * b + 2) * db]) {
                        if c > *m {
                            *m = c;
                        }
                    }
                }
            }
            levels.push(next);
        }
        Some(Pyramid { bars: db, levels })
    }

    pub fn bytes(&self) -> usize {
        self.levels.iter().map(|l| l.capacity() * 2).sum()
    }

    /// The largest block starting at `f` that ends by `f1`: (level index, size).
    fn block_at(&self, f: usize, f1: usize) -> Option<(usize, usize)> {
        let left = f1.checked_sub(f)?;
        if left == 0 || self.levels.is_empty() { return None; }
        // Largest aligned power of two wholly inside the query. Searching
        // every level for every block made planning a wide query quadratic
        // in the number of levels, before any codes had even been read.
        let k = f.trailing_zeros().min(left.ilog2())
            .min(PYRAMID_FROM + self.levels.len() as u32 - 1);
        (k >= PYRAMID_FROM).then(|| ((k - PYRAMID_FROM) as usize, 1usize << k))
    }

    /// How many rows (blocks and single frames) [`Self::column_max`] reads
    /// for frames `f0..f1`.
    pub fn rows_for(&self, (f0, f1): (usize, usize)) -> usize {
        let (mut f, mut n) = (f0, 0);
        while f < f1 {
            f += self.block_at(f, f1).map_or(1, |(_, size)| size);
            n += 1;
        }
        n
    }

    /// [`column_max`] of frames `f0..f1`, from the largest aligned blocks
    /// that fit wholly inside, and single frames where none does.
    pub fn column_max(&self, plane: &PreFrames, (f0, f1): (usize, usize), out: &mut [u16]) {
        let db = self.bars;
        out.fill(0);
        let take = |m: &mut [u16], src: &[u16]| {
            for (a, &c) in m.iter_mut().zip(src) {
                if c > *a { *a = c; }
            }
        };
        let mut f = f0;
        while f < f1 {
            match self.block_at(f, f1) {
                Some((k, size)) => {
                    let b = f / size;
                    take(out, &self.levels[k][b * db..(b + 1) * db]);
                    f += size;
                }
                None => {
                    let Some(row) = plane.row(f) else { break };
                    take(out, row);
                    f += 1;
                }
            }
        }
    }
}

/// One level of a sparse table over a range of frames: row `f − start` is
/// the per-bar maximum of frames `f .. f + 2^k` (clipped at the end of the
/// track). A column whose width `L` is between `2^k` and `3 * 2^k` is the
/// maximum of up to three such rows, all lying wholly inside the column and
/// together covering it, so the result
/// is exactly the frame-by-frame maximum. The extra block covers rounded
/// 8/9-frame columns without rebuilding a four-frame level on every notch.
pub struct Level {
    pub k: u32,
    pub start: usize,
    pub end: usize,
    db: usize,
    rows: Vec<u16>,
}

impl Level {
    /// Rows `start..end` at `k`, by doubling from the plane; `None` once
    /// `cancel` is set.
    pub fn build(plane: &PreFrames, data_bars: usize, k: u32, start: usize, end: usize, cancel: &AtomicBool) -> Option<Level> {
        let db = data_bars.min(plane.bars());
        let n = (end - start + (1usize << k) - 1).min(plane.len() - start);
        let mut rows = vec![0u16; n * db];
        for i in 0..n {
            if i % 256 == 0 && cancel.load(Ordering::Relaxed) {
                return None;
            }
            rows[i * db..(i + 1) * db].copy_from_slice(&plane.row(start + i)?[..db]);
        }
        for step in 0..k {
            if cancel.load(Ordering::Relaxed) {
                return None;
            }
            let s = 1usize << step;
            // Ascending: row i + s is read before this pass changes it.
            for i in 0..n.saturating_sub(s) {
                let (a, b) = rows.split_at_mut((i + s) * db);
                for (m, &c) in a[i * db..(i + 1) * db].iter_mut().zip(&b[..db]) {
                    if c > *m {
                        *m = c;
                    }
                }
            }
        }
        rows.truncate((end - start) * db);
        rows.shrink_to_fit();
        Some(Level { k, start, end, db, rows })
    }

    pub fn bytes(&self) -> usize {
        self.rows.capacity() * 2
    }

    /// Whether it answers a column of frames `f0..f1`.
    pub fn answers(&self, (f0, f1): (usize, usize)) -> bool {
        let (l, b) = (f1.saturating_sub(f0), 1usize << self.k);
        b <= l && l <= 3 * b && f0 >= self.start && f1 - b < self.end
    }

    /// Stored rows read for an interval that this level answers. The final
    /// block ends exactly at f1; overlap is harmless for integer maxima.
    pub fn rows_for(&self, (f0, f1): (usize, usize)) -> usize {
        (f1 - f0).div_ceil(1usize << self.k)
    }

    /// [`column_max`] of a column it [`answers`](Self::answers).
    pub fn column_max(&self, (f0, f1): (usize, usize), out: &mut [u16]) {
        let db = self.db;
        let block = 1usize << self.k;
        let a = f0 - self.start;
        let first = &self.rows[a * db..(a + 1) * db];
        let reads = self.rows_for((f0, f1));
        if reads == 1 {
            out.copy_from_slice(first);
        } else {
            let z = f1 - block - self.start;
            let last = &self.rows[z * db..(z + 1) * db];
            if reads == 2 {
                for ((m, &x), &y) in out.iter_mut().zip(first).zip(last) {
                    *m = x.max(y);
                }
            } else {
                let b = a + block;
                let middle = &self.rows[b * db..(b + 1) * db];
                for (((m, &x), &y), &z) in out.iter_mut().zip(first).zip(middle).zip(last) {
                    *m = x.max(y).max(z);
                }
            }
        }
    }
}

/// Most bytes one plane's sparse-table level may take; a wider time window
/// is read through the pyramid (or frame by frame) instead.
///
/// This bounds the **installed** level. Building a replacement first
/// releases the installed one (see `columns`), so the worker never holds two;
/// a UI-thread fill that took the old level before it was released keeps it
/// alive until that fill ends (one frame), so the transient peak per plane is
/// at most two levels for that long, and one otherwise.
pub const LEVEL_BUDGET: usize = 64 << 20;

/// Source rows a column of frames `span` reads through what is already built
/// (nothing is built here): one to three through a level that answers it, the
/// pyramid's blocks and single frames, or every frame.
pub fn held_rows(level: Option<&Level>, pyramid: Option<&Pyramid>, span: (usize, usize)) -> usize {
    match (level, pyramid) {
        (Some(l), _) if l.answers(span) => l.rows_for(span),
        (_, Some(p)) => p.rows_for(span),
        _ => span.1.saturating_sub(span.0),
    }
}

/// What reading a column of frames `span` through what is already built
/// costs, in plane-scan cells (the unit of `wholetrack_ring::SYNC_CELLS`). A
/// level's rows are counted at a quarter of a cell each: measured at the
/// owner's geometry (1584 columns × 1024 bars, release build), a refill
/// through a level read 3.2 Mi cells in 0.35 ms where a frame scan read
/// 9.7 Mi in 4.1 ms — about 0.11 against 0.42 ns a cell. Without it a time
/// zoom inside the level's octave went to the worker at ordinary widths and
/// drew every column as loading until it landed. The pyramid is counted at
/// full cost (not measured).
pub fn held_cost(level: Option<&Level>, pyramid: Option<&Pyramid>, span: (usize, usize), data_bars: usize) -> usize {
    match level {
        Some(l) if l.answers(span) => (l.rows_for(span) * data_bars).div_ceil(4),
        // This is a copy from the same frame-major storage, not a scan.
        // Repeated magnified spans are charged once by Plan::read_spans.
        _ if span.1.saturating_sub(span.0) == 1 => data_bars.div_ceil(4),
        _ => held_rows(level, pyramid, span) * data_bars,
    }
}

/// [`column_max`] through what is already built; the rows it read. The
/// same maxima as reading every frame (see [`Level`] and [`Pyramid`]).
pub fn column_max_held(plane: &PreFrames, level: Option<&Level>, pyramid: Option<&Pyramid>, span: (usize, usize), out: &mut [u16]) -> usize {
    if span.1.saturating_sub(span.0) == 1 {
        // Keep the copy cost honest even when a wide-view pyramid exists:
        // do not route this back through its clear-and-reduce scan.
        column_max(plane, span, out);
        return 1;
    }
    match (level, pyramid) {
        (Some(l), _) if l.answers(span) => l.column_max(span, out),
        (_, Some(p)) => p.column_max(plane, span, out),
        _ => column_max(plane, span, out),
    }
    held_rows(level, pyramid, span)
}

/// What a plane keeps to read wide columns quickly: its pyramid (when the
/// view's budget allows one) and the last sparse-table level built.
#[derive(Default)]
pub struct Accel {
    pub pyramid_ok: bool,
    pub pyramid: std::sync::OnceLock<Pyramid>,
    pub level: std::sync::Mutex<Option<Arc<Level>>>,
    /// The most this has held installed at once, bytes.
    pub peak: std::sync::atomic::AtomicUsize,
}

impl Accel {
    /// Bytes installed now (pyramid and level).
    pub fn bytes(&self) -> usize {
        self.pyramid.get().map_or(0, Pyramid::bytes)
            + self.level.lock().map(|l| l.as_ref().map_or(0, |l| l.bytes())).unwrap_or(0)
    }

    fn note_peak(&self) {
        self.peak.fetch_max(self.bytes(), Ordering::Relaxed);
    }
}

/// The per-bar maxima of `cols` from `plane`, through whichever of `accel`
/// fits each column: the sparse-table level for this zoom's octave (built
/// here over the window, with a margin, when missing and within budget), the
/// pyramid for wide columns, or the frames themselves. `None` once cancelled.
fn columns(plane: &PreFrames, accel: Option<&Accel>, cols: &[(usize, (usize, usize))], window: (usize, usize), db: usize, cancel: &AtomicBool) -> Option<Vec<u16>> {
    let mut out = vec![0u16; cols.len() * db];
    let spans: Vec<(usize, usize)> = cols.iter().map(|&(_, s)| s).filter(|(a, b)| b > a).collect();
    let level = accel.and_then(|acc| {
        // Rounding an intersected pixel can produce adjacent lengths (7/8,
        // for example). The median may choose an eight-frame block which
        // cannot answer any seven-frame interval. Choose the shortest one.
        let k = spans.iter().map(|(a, b)| b - a).min()?.ilog2();
        if k < 1 {
            return None;
        }
        let (mut lo, mut hi) = (spans.iter().map(|s| s.0).min()?, spans.iter().map(|s| s.1).max()?);
        if window.1 > window.0 {
            (lo, hi) = (lo.min(window.0), hi.max(window.1.min(plane.len())));
        }
        let held = acc.level.lock().ok()?.clone();
        if let Some(l) = held.filter(|l| spans.iter().all(|&s| l.answers(s))) {
            return Some(l);
        }
        let margin = (hi - lo) / 4;
        let (start, end) = (lo.saturating_sub(margin), (hi + margin).min(plane.len()));
        if (end - start) * db * 2 > LEVEL_BUDGET {
            return None;
        }
        // The installed level goes first, so the worker never holds two
        // (see `LEVEL_BUDGET`). (`held`, refused above, is already dropped.)
        *acc.level.lock().ok()? = None;
        let l = Arc::new(Level::build(plane, db, k, start, end, cancel)?);
        *acc.level.lock().ok()? = Some(l.clone());
        acc.note_peak();
        Some(l)
    });
    if cancel.load(Ordering::Relaxed) {
        return None;
    }
    let wide = !spans.is_empty() && spans.iter().map(|(a, b)| b - a).sum::<usize>() / spans.len() >= PYRAMID_MIN_SPAN;
    let pyramid = match accel {
        Some(acc) if wide && acc.pyramid_ok => match acc.pyramid.get() {
            Some(p) => Some(p),
            None => {
                let p = Pyramid::build(plane, db, cancel)?;
                let p = acc.pyramid.get_or_init(|| p);
                acc.note_peak();
                Some(p)
            }
        },
        _ => None,
    };
    for (k, &(_, span)) in cols.iter().enumerate() {
        if cancel.load(Ordering::Relaxed) {
            return None;
        }
        let o = &mut out[k * db..(k + 1) * db];
        column_max_held(plane, level.as_deref(), pyramid, span, o);
    }
    Some(out)
}

/// One pixel's code from its column's per-bar maxima (`None`: the column is
/// outside the track) and its row's bar span, exactly as [`rasterize`]
/// decides it.
pub fn pixel_code(colmax: Option<&[u16]>, (r0, r1): (usize, usize), data_bars: usize) -> u32 {
    match colmax {
        None => OUTSIDE,
        Some(_) if r1 <= r0 => OUTSIDE,
        Some(_) if r0 >= data_bars => NO_DATA,
        Some(m) => m[r0..r1.min(data_bars)].iter().copied().max().unwrap_or(0) as u32,
    }
}

/// A rasterised view of one plane.
#[derive(Clone, Debug, PartialEq)]
pub struct Raster {
    pub w: usize,
    pub h: usize,
    pub view: View,
    /// `w × h` codes, row-major from the top; [`OUTSIDE`] or [`NO_DATA`] where
    /// there is no level.
    pub codes: Vec<u32>,
    /// Largest block any pixel reduced, in frames and in bars. `(1, 1)` (or
    /// less) means every pixel is one native cell.
    pub max_block: (usize, usize),
}

impl Raster {
    #[cfg(test)]
    pub fn native(&self) -> bool {
        self.max_block.0 <= 1 && self.max_block.1 <= 1
    }
    pub fn bytes(&self) -> usize {
        self.codes.capacity() * std::mem::size_of::<u32>()
    }
}

/// Rasterise `view` of `plane` at `w × h`. Stops early, returning `None`, once
/// `cancel` is set.
///
/// The cost is the cells in view plus the pixels: each column's frames are
/// reduced into one per-bar maximum, then each row reduces its bars from that.
pub fn rasterize(
    plane: &PreFrames,
    view: View,
    w: usize,
    h: usize,
    data_bars: usize,
    cancel: &AtomicBool,
) -> Option<Raster> {
    let (frames, bars) = (plane.len(), plane.bars());
    let data_bars = data_bars.min(bars);
    let cols = spans(view.t0, view.t1, w, frames);
    let rows: Vec<(usize, usize)> = (0..h).map(|y| row_span(&view, h, y, bars)).collect();
    let (g_lo, g_hi) = rows.iter().filter(|r| r.1 > r.0).fold((bars, 0), |(a, b), r| (a.min(r.0), b.max(r.1)));
    let g_hi = g_hi.min(data_bars);
    let mut codes = vec![OUTSIDE; w * h];
    let mut colmax = vec![0u16; g_hi.saturating_sub(g_lo)];
    let mut max_block = (0, 0);
    for &(r0, r1) in &rows {
        max_block.1 = max_block.1.max(r1 - r0);
    }
    for (x, &(f0, f1)) in cols.iter().enumerate() {
        if cancel.load(Ordering::Relaxed) {
            return None;
        }
        if f1 <= f0 {
            continue;
        }
        max_block.0 = max_block.0.max(f1 - f0);
        colmax.iter_mut().for_each(|v| *v = 0);
        if g_hi > g_lo {
            for f in f0..f1 {
                let row = &plane.row(f)?[g_lo..g_hi];
                for (m, &c) in colmax.iter_mut().zip(row) {
                    if c > *m {
                        *m = c;
                    }
                }
            }
        }
        for (y, &(r0, r1)) in rows.iter().enumerate() {
            if r1 <= r0 {
                continue;
            }
            let px = &mut codes[y * w + x];
            if r0 >= data_bars {
                *px = NO_DATA;
                continue;
            }
            let (a, b) = (r0 - g_lo, r1.min(data_bars) - g_lo);
            *px = colmax[a..b].iter().copied().max().unwrap_or(0) as u32;
        }
    }
    Some(Raster { w, h, view, codes, max_block })
}

/// A raster size for a plot of `w × h` physical pixels showing `planes`
/// stacked planes: within the texture limit per side, and within
/// [`MAX_RASTER_PIXELS`] in total, keeping the aspect.
pub fn fit_raster(w: f32, h: f32, planes: usize, max_side: usize) -> (usize, usize) {
    let planes = planes.max(1);
    let mut w = (w.max(1.0) as usize).min(max_side).max(1);
    let mut h = ((h.max(1.0) / planes as f32) as usize).min(max_side).max(1);
    let budget = MAX_RASTER_PIXELS / planes;
    if w * h > budget {
        let k = (budget as f64 / (w * h) as f64).sqrt();
        w = ((w as f64 * k) as usize).max(1);
        h = ((h as f64 * k) as usize).max(1);
    }
    (w, h)
}

// ---------------------------------------------------------------------------
// The snapshot
// ---------------------------------------------------------------------------

/// Where the installed analysis lives.
#[derive(Clone, Debug, PartialEq)]
pub enum Origin {
    /// A cache-store generation, pinned for as long as the snapshot lives.
    Store { stem: String, generation: u128 },
    /// A finished analysis shown from memory only (it could not be published
    /// or pinned).
    Memory,
    /// A legacy file the user chose. Its record states its settings; nothing
    /// checked it against the audio file as it is now.
    Legacy { path: std::path::PathBuf },
}

/// Everything the view draws from, captured once and never changed.
///
/// Holding it keeps the planes and the store pin alive; it never copies a
/// plane. A new snapshot is taken whenever the analyser's revision moves or
/// the channels the view needs become resident or go away.
pub struct Snapshot {
    /// `SpectrumAnalyzer::pre_revision` at capture.
    pub revision: u64,
    pub origin: Origin,
    /// The store pin, when there is one. Held, never read.
    #[allow(dead_code)]
    pub pin: Option<Arc<super::store::Installed>>,
    pub mix: Arc<PreFrames>,
    /// Left and right, only when resident and asked for.
    pub lr: Option<(Arc<PreFrames>, Arc<PreFrames>)>,
    pub geometry: Geometry,
    pub inv_max: f32,
    pub max_code: u16,
}

/// Why there is nothing to draw.
#[derive(Clone, Debug, PartialEq)]
pub enum Unavailable {
    /// No analysis is installed.
    Empty,
    /// The installed analysis records no settings (a nominal legacy file).
    NoRecord,
    /// The record disagrees with the matrix, or cannot be read.
    Inconsistent(String),
}

impl Snapshot {
    /// Capture what the analyser has installed now.
    pub fn capture(a: &super::SpectrumAnalyzer, want_lr: bool) -> Result<Snapshot, Unavailable> {
        let mix = a.pre_frames.clone();
        if mix.is_empty() {
            return Err(Unavailable::Empty);
        }
        let record = a.pre_record().ok_or(Unavailable::NoRecord)?;
        let geometry = Geometry::parse(record).map_err(Unavailable::Inconsistent)?;
        if geometry.bars != mix.bars() {
            return Err(Unavailable::Inconsistent(format!(
                "the record states {} bars and the analysis holds {}",
                geometry.bars,
                mix.bars()
            )));
        }
        if geometry.sample_rate != a.sample_rate
            || (geometry.frame_rate() - a.pre_frame_rate).abs() > 1e-9 * geometry.frame_rate()
        {
            return Err(Unavailable::Inconsistent(format!(
                "the record states {:.4} frames/s at {} Hz and the display is timing it at {:.4} frames/s at {} Hz",
                geometry.frame_rate(),
                geometry.sample_rate,
                a.pre_frame_rate,
                a.sample_rate
            )));
        }
        let (origin, pin) = match a.pre_cache() {
            Some(super::CacheRef::Store { installed, .. }) => (
                Origin::Store { stem: installed.stem.clone(), generation: installed.generation },
                Some(installed.clone()),
            ),
            Some(super::CacheRef::Legacy { path, .. }) => (Origin::Legacy { path: path.clone() }, None),
            None => (Origin::Memory, None),
        };
        let lr = want_lr
            .then(|| a.pre_channels().resident().map(|s| s.shared()))
            .flatten()
            .filter(|(l, r)| l.len() == mix.len() && l.bars() == mix.bars() && r.len() == mix.len());
        Ok(Snapshot {
            revision: a.pre_revision(),
            origin,
            pin,
            inv_max: mix.inv_max(),
            max_code: mix.max_code(),
            mix,
            lr,
            geometry,
        })
    }

    pub fn frames(&self) -> usize {
        self.mix.len()
    }

    /// Whether `other` shows the same matrices: same mix, same channels.
    pub fn same_planes(&self, other: &Snapshot) -> bool {
        Arc::ptr_eq(&self.mix, &other.mix)
            && match (&self.lr, &other.lr) {
                (None, None) => true,
                (Some(a), Some(b)) => Arc::ptr_eq(&a.0, &b.0) && Arc::ptr_eq(&a.1, &b.1),
                _ => false,
            }
    }
}

// ---------------------------------------------------------------------------
// The worker
// ---------------------------------------------------------------------------

/// What to rasterise: one or more planes at one view, and optionally the
/// whole-track overview strip of one plane (the Mix).
pub struct Job {
    pub epoch: u64,
    pub seq: u64,
    pub planes: Vec<Arc<PreFrames>>,
    pub data_bars: usize,
    /// The overview strip's plane and size, when the strip is wanted too.
    pub overview: Option<(Arc<PreFrames>, (usize, usize))>,
    /// The per-bar maxima wanted: these columns (ring slot, frame span), for
    /// every plane.
    pub cols: Vec<(usize, (usize, usize))>,
    /// Per plane, what it keeps to read wide columns quickly (see `Accel`).
    pub accel: Vec<Arc<Accel>>,
    /// The asking camera's frames (empty for none): a level built for this
    /// job covers them, not only `cols`, so a request for a few leftover
    /// columns does not rebuild the level over just those and leave the
    /// next zoom step of the same camera unanswered.
    pub window: (usize, usize),
}

/// Columns at least this many frames wide on average are read through the
/// plane's pyramid (narrower ones gain little: their edges are most of them).
pub const PYRAMID_MIN_SPAN: usize = 16;

pub struct Done {
    pub epoch: u64,
    pub seq: u64,
    pub overview: Option<Raster>,
    pub elapsed: Duration,
    /// For a column job: the columns, and per plane their maxima
    /// (`cols.len() × data_bars`, column-major).
    pub cols: Vec<(usize, (usize, usize))>,
    pub colmax: Vec<Vec<u16>>,
}

struct InFlight {
    epoch: u64,
    cancel: Arc<AtomicBool>,
    thread: std::thread::JoinHandle<()>,
}

/// At most one rasterisation running and one waiting.
///
/// A newer request replaces the waiting one, never queues behind it, so a drag
/// cannot build a backlog. The running one is left to finish for the same
/// epoch (it is bounded, and its result is still a better picture than none);
/// it is cancelled when the epoch moves on, and whatever it sends then is
/// refused by [`Worker::poll`].
pub struct Worker {
    tx: mpsc::Sender<Done>,
    rx: mpsc::Receiver<Done>,
    in_flight: Option<InFlight>,
    pending: Option<Job>,
    /// Results refused because their epoch had moved on.
    pub discarded: u64,
    /// Threads ever started, and the most alive at once.
    pub started: u64,
    live: Arc<AtomicUsize>,
    pub peak_live: usize,
    /// Test only: hold each worker just before it sends.
    #[cfg(test)]
    pub hold: Option<Arc<AtomicBool>>,
    /// Test only: set by a held worker once its raster is done.
    #[cfg(test)]
    pub parked: Arc<AtomicBool>,
    /// Test only: runs inside `poll`, after the drain and before the reap.
    #[cfg(test)]
    pub between_drain_and_reap: Option<Box<dyn FnMut()>>,
}

impl Default for Worker {
    fn default() -> Self {
        let (tx, rx) = mpsc::channel();
        Worker {
            tx,
            rx,
            in_flight: None,
            pending: None,
            discarded: 0,
            started: 0,
            live: Arc::new(AtomicUsize::new(0)),
            peak_live: 0,
            #[cfg(test)]
            hold: None,
            #[cfg(test)]
            parked: Arc::new(AtomicBool::new(false)),
            #[cfg(test)]
            between_drain_and_reap: None,
        }
    }
}

impl Worker {
    /// Ask for `job`. Starts it now if nothing is running; otherwise it waits,
    /// replacing whatever was waiting.
    pub fn request(&mut self, mut job: Job) {
        if let Some(f) = &self.in_flight {
            if f.epoch != job.epoch {
                f.cancel.store(true, Ordering::Relaxed);
            }
            if let Some(old) = self.pending.take()
                && old.epoch == job.epoch
                && job.overview.is_none()
            {
                // The strip still has to be drawn once.
                job.overview = old.overview;
            }
            self.pending = Some(job);
        } else {
            self.spawn(job);
        }
    }

    /// Cancel whatever is running and forget what is waiting.
    pub fn invalidate(&mut self) {
        if let Some(f) = &self.in_flight {
            f.cancel.store(true, Ordering::Relaxed);
        }
        self.pending = None;
    }

    /// Test only: take one queued result, if any (consumes it).
    #[cfg(test)]
    pub fn take_queued(&self) -> Option<Done> {
        self.rx.try_recv().ok()
    }

    /// Whether anything is running or waiting.
    pub fn busy(&self) -> bool {
        self.in_flight.is_some() || self.pending.is_some()
    }

    /// Collect finished work for `epoch`, then start the waiting job if the
    /// slot is free. Results for any other epoch are dropped.
    pub fn poll(&mut self, epoch: u64) -> Vec<Done> {
        // Whether the thread has exited is read BEFORE the queue is drained.
        // A thread sends before it exits, so if it had exited by now its
        // result is already queued and the drain below takes it. Reading it
        // after the drain instead let a result sent between the two be left
        // queued while the slot was reaped and `busy()` turned false — and a
        // retired view, which only polled while busy, kept that raster.
        let exited = self.in_flight.as_ref().is_some_and(|f| f.thread.is_finished());
        let mut out = Vec::new();
        while let Ok(d) = self.rx.try_recv() {
            if d.epoch == epoch {
                out.push(d);
            } else {
                self.discarded += 1;
            }
        }
        #[cfg(test)]
        if let Some(f) = self.between_drain_and_reap.as_mut() {
            f();
        }
        // The slot is held from spawn until the thread has exited, so a slow
        // exit can never let a second thread start beside it.
        if exited {
            self.in_flight = None;
        }
        // A waiting job for another epoch is taken and dropped here.
        if self.in_flight.is_none()
            && let Some(job) = self.pending.take()
            && job.epoch == epoch
        {
            self.spawn(job);
        }
        out
    }

    fn spawn(&mut self, job: Job) {
        let cancel = Arc::new(AtomicBool::new(false));
        let tx = self.tx.clone();
        let c = cancel.clone();
        let live = self.live.clone();
        let n = live.fetch_add(1, Ordering::SeqCst) + 1;
        self.peak_live = self.peak_live.max(n);
        self.started += 1;
        #[cfg(test)]
        let (hold, parked) = (self.hold.clone(), self.parked.clone());
        let epoch = job.epoch;
        let thread = std::thread::Builder::new()
            .name("moosik-wholetrack".into())
            .spawn(move || {
                struct Leave(Arc<AtomicUsize>);
                impl Drop for Leave {
                    fn drop(&mut self) {
                        self.0.fetch_sub(1, Ordering::SeqCst);
                    }
                }
                let _leave = Leave(live);
                let t = Instant::now();
                let mut colmax = Vec::new();
                for (i, p) in job.planes.iter().enumerate() {
                    match columns(p, job.accel.get(i).map(|a| &**a), &job.cols, job.window, job.data_bars, &c) {
                        Some(out) => colmax.push(out),
                        None => return,
                    }
                }
                let overview = match &job.overview {
                    Some((p, (ow, oh))) => {
                        let full = View::full(p.len(), p.bars());
                        match rasterize(p, full, *ow, *oh, job.data_bars, &c) {
                            Some(r) => Some(r),
                            None => return,
                        }
                    }
                    None => None,
                };
                // The planes are released before the result is handed over.
                drop(job.planes);
                drop(job.overview);
                drop(job.accel);
                #[cfg(test)]
                if let Some(h) = hold {
                    parked.store(true, Ordering::SeqCst);
                    while h.load(Ordering::SeqCst) {
                        std::thread::yield_now();
                    }
                }
                // A dropped receiver (the view went away) makes this a no-op.
                let _ = tx.send(Done {
                    epoch,
                    seq: job.seq,
                    overview,
                    elapsed: t.elapsed(),
                    cols: job.cols,
                    colmax,
                });
            });
        match thread {
            Ok(thread) => self.in_flight = Some(InFlight { epoch, cancel, thread }),
            Err(e) => {
                self.live.fetch_sub(1, Ordering::SeqCst);
                crate::mlog!("[wholetrack] could not start the raster worker: {e}");
            }
        }
    }
}

#[cfg(test)]
#[path = "wholetrack_tests.rs"]
mod tests;
