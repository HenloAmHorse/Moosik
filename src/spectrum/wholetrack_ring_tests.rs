//! The ring renderer against a frozen reference, along deliberate camera
//! trajectories, plus the counterexamples that rule out the cheaper designs.
//!
//! `frozen` is a copy of the batch-154 raster (`span` and `rasterize` as they
//! stood at 878d270), kept here so that no later change to the production
//! code can move the reference with it. Every comparison is code for code
//! against the reference raster of the camera actually drawn.
use super::*;
use crate::spectrum::cache::PreFrames;
use crate::spectrum::wholetrack::View;

/// Reproduce the rejected banded-coding experiment against scalar stores.
/// No GPU or event loop; this is coding cost, not production frame rate.
#[test]
#[ignore = "component timing: run alone in release"]
fn ring_column_coding_cost_probe() {
    use std::time::Instant;
    for (w, h) in [(1584usize, 800usize), (2448, 1107)] {
        let db = 1024;
        let view = View { t0: 17.0, t1: 17.0 + w as f64, b0: 0.0, b1: db as f64 };
        let mut seed = 81u32;
        let maxima: Vec<u16> = (0..w * db).map(|_| {
            seed ^= seed << 13;
            seed ^= seed >> 17;
            seed ^= seed << 5;
            (seed & 4095) as u16
        }).collect();
        let mut scalar = Ring::new(w, h, db);
        let mut banded = Ring::new(w, h, db);
        scalar.set_rows(&view, db);
        banded.set_rows(&view, db);
        let plan = scalar.plan(view, w + 34);
        let columns: Vec<_> = (0..w).map(|x| (scalar.slot(plan.base, x), plan.spans[x])).collect();
        let mut scalar_ms = Vec::new();
        let mut banded_ms = Vec::new();
        // Alternate which path runs first rather than favoring the last
        // one with an already-hot source on every sample.
        for sample in 0..8 {
            for mode in if sample % 2 == 0 { [false, true] } else { [true, false] } {
                let start = Instant::now();
                for _ in 0..12 {
                    let m = std::hint::black_box(&maxima);
                    if mode {
                        let slots: Vec<_> = columns.iter().map(|&(s, _)| s).collect();
                        for (k, &(s, sp)) in columns.iter().enumerate() {
                            banded.store_column(s, sp, &m[k * db..(k + 1) * db]);
                        }
                        banded.recode_slots(&slots);
                        std::hint::black_box(&banded.codes);
                    } else {
                        for (k, &(s, sp)) in columns.iter().enumerate() { scalar.put(s, sp, &m[k * db..(k + 1) * db]); }
                        std::hint::black_box(&scalar.codes);
                    }
                }
                let ms = start.elapsed().as_secs_f64() * 1000.0 / 12.0;
                if mode { banded_ms.push(ms); } else { scalar_ms.push(ms); }
            }
            assert_eq!(scalar.codes, banded.codes);
        }
        println!("{}", serde_json::json!({"w":w,"h":h,"bars":db,"scalar_ms":scalar_ms,"banded_ms":banded_ms}));
    }
}

#[test]
fn batched_completion_matches_individual_columns_through_wraps_and_stale_results() {
    let p = plane(512, 16, 71);
    for w in [7, 31, 32, 33, 65] {
        let mut scalar = Ring::new(w, 43, 13);
        let mut batched = Ring::new(w, 43, 13);
        for (j, v) in [
            View { t0: 40.5, t1: 170.5, b0: 0.0, b1: 16.0 },
            View { t0: 48.25, t1: 178.25, b0: 2.25, b1: 15.25 },
            View { t0: 3.0, t1: 11.0, b0: 0.0, b1: 16.0 },
        ].into_iter().enumerate() {
            scalar.set_rows(&v, 16);
            batched.set_rows(&v, 16);
            let plan = scalar.plan(v, p.len());
            let columns: Vec<_> = plan.missing.iter().enumerate().map(|(k, &x)| {
                let sp = plan.spans[x];
                (scalar.slot(plan.base, x), if (j + k) % 3 == 0 { (400, 401) } else { sp })
            }).collect();
            let mut maxima = vec![0; columns.len() * 13];
            for (k, &(_, sp)) in columns.iter().enumerate() {
                crate::spectrum::wholetrack::column_max(&p, sp, &mut maxima[k * 13..(k + 1) * 13]);
            }
            scalar.dirty.fill(false);
            batched.dirty.fill(false);
            for (k, &(s, sp)) in columns.iter().enumerate() { scalar.put_landed(&plan, s, sp, &maxima[k * 13..(k + 1) * 13]); }
            batched.put_landed_columns(&plan, &columns, &maxima);
            assert_eq!(batched.codes, scalar.codes, "w={w}, step={j}");
            assert_eq!(batched.span, scalar.span);
            assert_eq!(batched.dirty, scalar.dirty);
            assert_eq!(batched.stats.columns, scalar.stats.columns);
            let plan = batched.plan(v, p.len());
            batched.fill(&p, &plan);
            assert_eq!(batched.raster().unwrap().codes, frozen::raster(&p, v, w, 43, 13));
            let plan = scalar.plan(v, p.len());
            scalar.fill(&p, &plan);
        }
    }
}

/// Deep magnification maps many screen columns to one stored frame. Charge
/// and execute one source query per distinct span, not per destination.
#[test]
fn magnified_columns_share_queries_but_keep_the_exact_camera() {
    let p = plane(256, 64, 19);
    let mut r = Ring::new(1584, 60, 64);
    let v = View { t0: 100.25, t1: 120.75, b0: 0.0, b1: 64.0 };
    r.set_rows(&v, 64);
    let plan = r.plan(v, p.len());
    let queries = plan.read_spans().count();
    assert!(queries <= 22, "magnified source reads grew with window width");
    let mut calls = 0;
    r.fill_with(&plan, |sp, out| {
        calls += 1;
        crate::spectrum::wholetrack::column_max(&p, sp, out);
        sp.1 - sp.0
    });
    assert_eq!(calls, queries);
    assert_eq!(r.stats.cells_sync, (queries * 64) as u64);
    let got = r.raster().unwrap();
    assert_eq!(got.codes, frozen::raster(&p, v, r.w, r.h, 64));
}

/// An obsolete completion must not recode/upload columns for a camera that
/// is no longer visible, even if its destination is currently missing.
#[test]
fn an_obsolete_completion_does_not_dirty_a_missing_current_column() {
    let mut r = Ring::new(16, 8, 4);
    let v = View { t0: 100.0, t1: 116.0, b0: 0.0, b1: 4.0 };
    r.set_rows(&v, 4);
    let plan = r.plan(v, 256);
    let s = r.slot(plan.base, 0);
    r.dirty.fill(false);
    r.put_landed(&plan, s, (0, 1), &[4095; 4]);
    assert_eq!(r.stats.columns, 0);
    assert!(r.span[s].is_none());
    assert!(!r.dirty[s]);
    r.put_landed(&plan, s, plan.spans[0], &[100; 4]);
    assert_eq!(r.stats.columns, 1);
    assert_eq!(r.span[s], Some(plan.spans[0]));
}

mod frozen {
    use crate::spectrum::cache::PreFrames;
    use crate::spectrum::wholetrack::View;
    pub const OUTSIDE: u32 = u32::MAX;
    pub const NO_DATA: u32 = u32::MAX - 1;

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

    /// Pixel by pixel, cell by cell: no column reuse, no row reuse.
    pub fn raster(p: &PreFrames, v: View, w: usize, h: usize, data_bars: usize) -> Vec<u32> {
        let mut out = vec![OUTSIDE; w * h];
        for y in 0..h {
            let (g0, g1) = span(v.b0, v.b1, h, h - 1 - y, p.bars());
            for x in 0..w {
                let (f0, f1) = span(v.t0, v.t1, w, x, p.len());
                if f1 <= f0 || g1 <= g0 {
                    continue;
                }
                if g0 >= data_bars {
                    out[y * w + x] = NO_DATA;
                    continue;
                }
                let mut m = 0u16;
                for f in f0..f1 {
                    for &c in &p.row(f).unwrap()[g0..g1.min(data_bars)] {
                        m = m.max(c);
                    }
                }
                out[y * w + x] = m as u32;
            }
        }
        out
    }
}

/// A plane with isolated one-cell peaks (so any extra cell in a maximum
/// shows), a ramp, and a few bright rows and columns.
fn plane(frames: usize, bars: usize, seed: u32) -> PreFrames {
    let mut x = seed.max(1);
    let mut codes = Vec::with_capacity(frames * bars);
    for f in 0..frames {
        for b in 0..bars {
            x ^= x << 13;
            x ^= x >> 17;
            x ^= x << 5;
            let mut c = (x % 1000) as u16;
            if x.is_multiple_of(97) { c = 3000 + (x % 1000) as u16; }
            if f % 211 == 0 { c = c.max(2500); }
            if b == bars / 3 { c = c.max(((f * 7) % 4000) as u16); }
            codes.push(c.min(4095));
        }
    }
    PreFrames::from_v4_codes(codes, frames, bars)
}

/// Draw `v` exactly as the view does when the work fits on the UI thread.
fn draw(r: &mut Ring, p: &PreFrames, v: View) {
    r.set_rows(&v, p.bars());
    let plan = r.plan(v, p.len());
    if plan.missing.is_empty() { r.show(&plan) } else { r.fill(p, &plan) }
}

fn check(r: &mut Ring, p: &PreFrames, v: View, data_bars: usize, what: &str) {
    draw(r, p, v);
    let got = r.raster().expect("nothing shown");
    assert_eq!(got.view, v, "{what}: another camera is shown");
    let want = frozen::raster(p, v, r.w, r.h, data_bars);
    if got.codes != want {
        let i = got.codes.iter().zip(&want).position(|(a, b)| a != b).unwrap();
        panic!("{what}: pixel ({}, {}) is {} and should be {} at {v:?}", i % r.w, i / r.w, got.codes[i], want[i]);
    }
}

/// The four regimes (time and frequency each magnified or reduced), at odd
/// sizes and non-integer scales, scrolled a pixel at a time across several
/// ring wraps and back, with fractional and boundary offsets.
#[test]
fn every_regime_scrolled_and_reversed_matches_the_reference() {
    let (frames, bars) = (6000, 64);
    let p = plane(frames, bars, 5);
    let (w, h) = (97, 31);
    for (tspan, bspan) in [(41.3, 13.7), (41.3, 57.9), (803.7, 13.7), (803.7, 57.9), (97.0 * 3.0, 31.0 * 2.0)] {
        let mut r = Ring::new(w, h, bars);
        let d = tspan / w as f64;
        let t0 = 100.0;
        let what = format!("t {tspan} b {bspan}");
        // Forward a pixel at a time across three wraps, then back.
        for step in (0..3 * w as i64).chain((0..3 * w as i64).rev()) {
            let base = (t0 / d).round() + step as f64;
            let v = View { t0: base * d, t1: base * d + tspan, b0: 2.5, b1: 2.5 + bspan };
            check(&mut r, &p, v, bars, &format!("{what} step {step}"));
        }
        // Fractional offsets, and offsets a hair either side of a cell edge.
        for off in [0.25, 0.5, 0.999_999, 1e-9, -1e-9, 0.0] {
            for edge in [1000.0, 1001.0, 2047.0] {
                let v = View { t0: edge + off, t1: edge + off + tspan, b0: 1.0 + off, b1: 1.0 + off + bspan };
                check(&mut r, &p, v, bars, &format!("{what} offset {off} at {edge}"));
            }
        }
        // Reuse: a one-pixel scroll on the grid recomputes about one column.
        let base = 900.0;
        let v0 = View { t0: base * d, t1: base * d + tspan, b0: 0.0, b1: bspan };
        draw(&mut r, &p, v0);
        let before = r.stats.columns;
        for k in 1..=20 {
            let v = View { t0: (base + k as f64) * d, t1: (base + k as f64) * d + tspan, ..v0 };
            check(&mut r, &p, v, bars, &format!("{what} reuse {k}"));
        }
        let per_step = (r.stats.columns - before) as f64 / 20.0;
        assert!(per_step <= 2.0, "{what}: a one-pixel scroll recomputed {per_step} columns");
    }
}

/// Track edges (columns before the first frame and after the last are
/// outside), above-Nyquist bars (no data), and a random walk of combined
/// pans and zooms in both axes with resizes between.
#[test]
fn edges_no_data_and_combined_pan_zoom_walks_match_the_reference() {
    let (frames, bars) = (3000, 48);
    let p = plane(frames, bars, 9);
    for data_bars in [bars, 31] {
        let mut x = 12345u64;
        let mut rnd = || {
            x = x.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
            (x >> 33) as f64 / (1u64 << 31) as f64
        };
        for (w, h) in [(64, 20), (151, 47), (33, 9)] {
            let mut r = Ring::new(w, h, data_bars);
            // Past both ends of the track.
            for v in [
                View { t0: -50.5, t1: 200.0, b0: 0.0, b1: bars as f64 },
                View { t0: frames as f64 - 120.25, t1: frames as f64 + 80.0, b0: 10.0, b1: bars as f64 + 6.0 },
            ] {
                check(&mut r, &p, v, data_bars, &format!("edge {v:?} data {data_bars}"));
            }
            let mut v = View { t0: 500.0, t1: 900.0, b0: 0.0, b1: bars as f64 };
            for i in 0..150 {
                let a = rnd();
                let span_t = v.t1 - v.t0;
                let span_b = v.b1 - v.b0;
                v = match (a * 5.0) as usize {
                    0 => View { t0: v.t0 + (rnd() - 0.5) * span_t * 0.1, t1: v.t1 + (rnd() - 0.5) * span_t * 0.1, ..v },
                    1 => { let k = 0.6 + rnd() * 0.8; let c = (v.t0 + v.t1) / 2.0; View { t0: c - span_t * k / 2.0, t1: c + span_t * k / 2.0, ..v } }
                    2 => { let k = 0.6 + rnd() * 0.8; let c = (v.b0 + v.b1) / 2.0; View { b0: c - span_b * k / 2.0, b1: c + span_b * k / 2.0, ..v } }
                    3 => { let s = (rnd() * 7.0).round() - 3.0; let d = span_t / w as f64; View { t0: v.t0 + s * d, t1: v.t1 + s * d, ..v } }
                    _ => View { b0: v.b0 + (rnd() - 0.5) * 3.0, b1: v.b1 + (rnd() - 0.5) * 3.0, ..v },
                };
                v.t1 = v.t1.max(v.t0 + 2.0);
                v.b1 = v.b1.max(v.b0 + 1.0);
                check(&mut r, &p, v, data_bars, &format!("walk {i} {w}x{h} data {data_bars}"));
            }
        }
    }
}

/// Counterexample: the old picture moved by a fraction of a pixel (what
/// redrawing the previous texture at the new camera shows) is not the new
/// camera's raster. Following by drawing yesterday's texture in today's
/// place is ruled out.
#[test]
fn a_shifted_raster_is_not_the_shifted_camera() {
    let p = plane(4000, 32, 3);
    let (w, h) = (80, 16);
    let a = View { t0: 1000.0, t1: 1000.0 + 80.0 * 7.3, b0: 0.0, b1: 32.0 };
    let old = frozen::raster(&p, a, w, h, 32);
    let d = (a.t1 - a.t0) / w as f64;
    let mut differs = 0;
    for frac in [0.25, 0.5, 0.75] {
        let b = View { t0: a.t0 + frac * d, t1: a.t1 + frac * d, ..a };
        let new = frozen::raster(&p, b, w, h, 32);
        // Nearest sampling of the old texture at the new camera: pixel x shows
        // old column round(x + frac).
        for y in 0..h {
            for x in 0..w - 1 {
                let src = (x as f64 + frac).round() as usize;
                differs += (old[y * w + src] != new[y * w + x]) as usize;
            }
        }
    }
    assert!(differs > 0, "the counterexample did not show a difference");
}

/// Counterexample: maxima over aligned power-of-two blocks (a max mip map)
/// that overlap a pixel's interval include cells outside it, and so differ
/// from the exact maximum; a one-cell peak just outside an interval shows.
#[test]
fn overlapping_aligned_block_maxima_are_not_the_exact_maximum() {
    let p = plane(4000, 32, 21);
    let (w, h) = (90, 32);
    let v = View { t0: 1003.3, t1: 1003.3 + 90.0 * 5.6, b0: 0.0, b1: 32.0 };
    let exact = frozen::raster(&p, v, w, h, 32);
    let level = 4usize; // 16-frame blocks; the pixels span 5.6
    let mut differs = 0;
    for x in 0..w {
        let (f0, f1) = frozen::span(v.t0, v.t1, w, x, p.len());
        let (b0, b1) = (f0 / level * level, f1.div_ceil(level) * level);
        for y in 0..h {
            let (g0, g1) = frozen::span(v.b0, v.b1, h, h - 1 - y, 32);
            let m = (b0..b1.min(p.len())).flat_map(|f| p.row(f).unwrap()[g0..g1].to_vec()).max().unwrap() as u32;
            differs += (m != exact[y * w + x]) as usize;
        }
    }
    assert!(differs > 0, "the counterexample did not show a difference");
}

/// The production span, at the scales and offsets these tests use, against
/// the same interval worked out in exact rational arithmetic (every f64 is a
/// dyadic rational). Where floating point and exact arithmetic would put a
/// boundary cell on different sides, both the reference and the renderer
/// follow the floating-point rule: this test counts such places so that the
/// report can say whether the grids above contain any.
#[test]
fn float_spans_against_exact_arithmetic_on_the_test_grids() {
    fn exact(a0: f64, a1: f64, n: usize, i: usize, limit: usize) -> (usize, usize) {
        // a0, a1 as integers over 2^60 (the test values are well within range).
        let q = |v: f64| (v * (1u64 << 60) as f64) as i128;
        let (a0, a1, n, i) = (q(a0), q(a1), n as i128, i as i128);
        let one = 1i128 << 60;
        // p0 = a0 + i (a1 − a0) / n, kept over the denominator n.
        let num0 = a0 * n + i * (a1 - a0);
        let num1 = num0 + (a1 - a0);
        let den = one * n;
        let floor = |x: i128| x.div_euclid(den);
        let ceil = |x: i128| -((-x).div_euclid(den));
        let (lo, hi) = if a1 - a0 <= one * n {
            // The cell under the pixel's centre, p0 + d/2.
            let c = (num0 * 2 + (a1 - a0)).div_euclid(den * 2);
            (c, c + 1)
        } else {
            (floor(num0), ceil(num1))
        };
        let clip = |v: i128| v.clamp(0, limit as i128) as usize;
        let (lo, hi) = (clip(lo), clip(hi));
        if lo >= hi { (0, 0) } else { (lo, hi) }
    }
    let mut checked = 0;
    let mut disagree = Vec::new();
    for (tspan, w) in [(41.3, 97), (803.7, 97), (291.0, 97), (400.0, 64), (90.0 * 5.6, 90)] {
        let d = tspan / w as f64;
        for base in 0..400i64 {
            let t0 = base as f64 * d;
            for x in 0..w {
                checked += 1;
                let f = crate::spectrum::wholetrack::span(t0, t0 + tspan, w, x, 6000);
                let e = exact(t0, t0 + tspan, w, x, 6000);
                if f != e && disagree.len() < 5 {
                    disagree.push((tspan, base, x, f, e));
                }
            }
        }
    }
    println!("spans checked {checked}; float/exact disagreements (first five): {disagree:?}");
}

/// The pyramid gives the same per-bar maxima as reading every frame, for
/// spans of every length and alignment, at the ends of the track included,
/// for planes whose length is not a power of two.
#[test]
fn pyramid_column_maxima_equal_frame_by_frame_maxima() {
    use crate::spectrum::wholetrack::{column_max, Pyramid};
    use std::sync::atomic::AtomicBool;
    for (frames, bars, data_bars) in [(5003, 24, 24), (777, 16, 11), (64, 8, 8)] {
        let p = plane(frames, bars, frames as u32);
        let py = Pyramid::build(&p, data_bars, &AtomicBool::new(false)).unwrap();
        let mut x = 99u64;
        let mut spans: Vec<(usize, usize)> = vec![(0, frames), (0, 1), (frames - 1, frames), (3, 4), (4, 8), (0, 4), (1, frames - 1)];
        for _ in 0..2000 {
            x = x.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
            let a = (x >> 33) as usize % frames;
            x = x.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
            let len = 1 + (x >> 33) as usize % 300;
            spans.push((a, (a + len).min(frames)));
        }
        let (mut want, mut got) = (vec![0u16; data_bars], vec![0u16; data_bars]);
        for sp in spans {
            column_max(&p, sp, &mut want);
            py.column_max(&p, sp, &mut got);
            assert_eq!(got, want, "{frames} frames, span {sp:?}");
        }
    }
}

/// A sparse-table level gives the frame-by-frame maximum for every column it
/// says it answers (widths within its octave, inside its range), at the ends
/// of the track included; and it answers nothing outside them.
#[test]
fn sparse_level_column_maxima_equal_frame_by_frame_maxima() {
    use crate::spectrum::wholetrack::{column_max, Level};
    use std::sync::atomic::AtomicBool;
    for (frames, bars, data_bars) in [(3001, 24, 24), (517, 16, 9)] {
        let p = plane(frames, bars, frames as u32 + 7);
        for k in [2u32, 3, 5] {
            for (start, end) in [(0, frames), (100, 400), (frames - 200, frames)] {
                let l = Level::build(&p, data_bars, k, start, end, &AtomicBool::new(false)).unwrap();
                let (mut want, mut got) = (vec![0u16; data_bars], vec![0u16; data_bars]);
                let mut answered = 0;
                for f0 in (0..frames).step_by(7) {
                    for len in [1usize << k, (1 << k) + 1, (3 << k) / 2, 2 << k, (2 << k) + 1, (1 << k) - 1] {
                        let sp = (f0, (f0 + len).min(frames));
                        if !l.answers(sp) {
                            assert!(sp.1 - sp.0 < 1 << k || sp.1 - sp.0 > 2 << k || sp.0 < start || sp.1 - (1 << k) >= end, "{sp:?} refused");
                            continue;
                        }
                        answered += 1;
                        column_max(&p, sp, &mut want);
                        l.column_max(sp, &mut got);
                        assert_eq!(got, want, "k {k} range {start}..{end} span {sp:?}");
                    }
                }
                assert!(answered > 0);
            }
        }
    }
}

/// A vertical pan by whole rows re-reduces only the rows that come into
/// view, uploads only them, and the picture is still the reference — for
/// both directions, magnified and reduced, and across the ring's wrap.
#[test]
fn a_vertical_pan_by_whole_rows_redoes_only_the_new_rows() {
    let p = plane(3000, 512, 11);
    let ctx = egui::Context::default();
    for (span_b, h) in [(40.0, 120usize), (300.0, 100)] {
        let (w, db) = (160, 512);
        let mut r = Ring::new(w, h, db);
        let row = span_b / h as f64;
        let mut v = View { t0: 300.0, t1: 1500.0, b0: 20.0 * row, b1: 20.0 * row + span_b };
        check(&mut r, &p, v, db, "start");
        r.upload(&ctx, "t", |c, _, _| Color32::from_gray((c % 251) as u8));
        for k in [3i64, -7, 1, 12, -12, 60] {
            let (rerows, bytes) = (r.stats.rerows, r.stats.upload_bytes);
            v.b0 += k as f64 * row;
            v.b1 += k as f64 * row;
            check(&mut r, &p, v, db, &format!("pan {k} rows"));
            assert_eq!(r.stats.rerows - rerows, k.unsigned_abs(), "pan {k}: rows re-reduced");
            r.upload(&ctx, "t", |c, _, _| Color32::from_gray((c % 251) as u8));
            let up = r.stats.upload_bytes - bytes;
            let want = if (k.unsigned_abs() as usize) * 2 > h { (w * h * 4) as u64 } else { k.unsigned_abs() * (w * 4) as u64 };
            assert_eq!(up, want, "pan {k}: bytes uploaded");
        }
    }
}
