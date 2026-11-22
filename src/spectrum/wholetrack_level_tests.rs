//! Fixed-window maxima against an independent cell scan and the previous
//! doubling builder. The ignored benchmark compares construction only.
use super::*;

/// Retained pre-lab builder: the reference for the arithmetic and clipping
/// that the linear builder replaces, not the expected-value oracle below.
fn doubling_level(plane: &PreFrames, data_bars: usize, k: u32, start: usize, end: usize, cancel: &AtomicBool) -> Option<Level> {
    let db = data_bars.min(plane.bars());
    let n = (end - start + (1usize << k) - 1).min(plane.len() - start);
    let mut rows = vec![0u16; n * db];
    for i in 0..n {
        if i % 256 == 0 && cancel.load(Ordering::Relaxed) { return None; }
        rows[i * db..(i + 1) * db].copy_from_slice(&plane.row(start + i)?[..db]);
    }
    for step in 0..k {
        if cancel.load(Ordering::Relaxed) { return None; }
        let s = 1usize << step;
        for i in 0..n.saturating_sub(s) {
            let (a, b) = rows.split_at_mut((i + s) * db);
            for (m, &c) in a[i * db..(i + 1) * db].iter_mut().zip(&b[..db]) {
                if c > *m { *m = c; }
            }
        }
    }
    rows.truncate((end - start) * db);
    rows.shrink_to_fit();
    Some(Level { k, start, end, db, rows })
}

fn codes(frames: usize, bars: usize) -> Vec<u16> {
    let mut seed = 0x1625_1551u32;
    (0..frames * bars).map(|i| {
        seed ^= seed << 13;
        seed ^= seed >> 17;
        seed ^= seed << 5;
        match i % 37 {
            0 => 0,
            1 => u16::MAX,
            _ => seed as u16,
        }
    }).collect()
}

/// Every answer is a rectangle scanned directly in the original flat codes.
/// No block maxima, level query, or previous builder helps compute it.
fn brute(codes: &[u16], frames: usize, bars: usize, db: usize, k: u32, start: usize, end: usize) -> Vec<u16> {
    let db = db.min(bars);
    let width = 1usize << k;
    let mut result = Vec::with_capacity((end - start) * db);
    for first in start..end {
        for b in 0..db {
            let mut best = 0u16;
            for f in first..first.saturating_add(width).min(frames) {
                best = best.max(codes[f * bars + b]);
            }
            result.push(best);
        }
    }
    result
}

#[test]
fn linear_levels_match_cell_scan_and_doubling_at_every_boundary() {
    let no = AtomicBool::new(false);
    for frames in [0usize, 1, 2, 5, 9, 17, 65] {
        for bars in [1usize, 3, 7, 17] {
            let source = codes(frames, bars);
            let plane = PreFrames::from_v4_codes(source.clone(), frames, bars);
            for k in [0u32, 1, 2, 3, 4, 6, 8] {
                for (start, end) in [(0, frames), (frames / 3, frames * 2 / 3),
                                     (frames.saturating_sub(1), frames), (frames / 2, frames / 2)] {
                    for db in [0usize, 1, bars, bars + 5] {
                        let got = Level::build(&plane, db, k, start, end, &no).unwrap();
                        let old = doubling_level(&plane, db, k, start, end, &no).unwrap();
                        let want = brute(&source, frames, bars, db, k, start, end);
                        assert_eq!((got.k, got.start, got.end, got.db), (k, start, end, db.min(bars)));
                        assert_eq!(got.rows, want, "frames {frames}, bars {bars}, db {db}, k {k}, {start}..{end}");
                        assert_eq!(got.rows, old.rows, "the previous builder disagrees at k {k}, {start}..{end}");
                    }
                }
            }
        }
    }
}

#[test]
fn clipped_last_blocks_and_nonzero_source_offsets_exclude_earlier_peaks() {
    let (frames, bars) = (43usize, 5usize);
    let no = AtomicBool::new(false);
    for descending in [false, true] {
        let source: Vec<u16> = (0..frames).flat_map(|f| (0..bars).map(move |b| {
            let row = if descending { frames - f } else { f };
            (row * 100 + b * 7) as u16
        })).collect();
        let plane = PreFrames::from_v4_codes(source.clone(), frames, bars);
        for start in [0usize, 1, 3, 8, 17, 35] {
            for k in 0..=7 {
                let got = Level::build(&plane, bars, k, start, frames, &no).unwrap();
                assert_eq!(got.rows, brute(&source, frames, bars, bars, k, start, frames),
                           "descending {descending}, k {k}, source offset {start}");
            }
        }
    }
}

#[test]
fn a_wide_level_keeps_only_the_requested_rows() {
    let (frames, bars, start, end) = (2049usize, 7usize, 37usize, 40usize);
    let source = codes(frames, bars);
    let plane = PreFrames::from_v4_codes(source.clone(), frames, bars);
    let got = Level::build(&plane, bars, 20, start, end, &AtomicBool::new(false)).unwrap();
    assert_eq!(got.rows, brute(&source, frames, bars, bars, 20, start, end));
    assert_eq!(got.bytes(), (end - start) * bars * 2, "the source halo remained allocated");
}

#[test]
fn linear_level_cancellation_covers_both_scans_and_the_small_window_path() {
    let plane = PreFrames::from_v4_codes(codes(1537, 7), 1537, 7);
    assert!(Level::build(&plane, 7, 6, 13, 1213, &AtomicBool::new(true)).is_none());
    for k in [0u32, 1, 2, 3, 6, 10] {
        let mut polls = 0usize;
        let complete = Level::build_cancellable(&plane, 7, k, 13, 1213, || { polls += 1; false });
        assert!(complete.is_some());
        assert!(polls >= 6, "the build did not poll during its work");
        // Set the actual flag at each polling point, including the final one.
        // This deterministically reaches cancellation during both scans.
        for at in 1..=polls {
            let flag = AtomicBool::new(false);
            let mut seen = 0usize;
            let result = Level::build_cancellable(&plane, 7, k, 13, 1213, || {
                seen += 1;
                if seen == at { flag.store(true, Ordering::Relaxed); }
                flag.load(Ordering::Relaxed)
            });
            assert!(result.is_none(), "k {k}: cancellation at poll {at} returned a level");
            assert_eq!(seen, at, "k {k}: cancellation was not returned at once");
        }
    }
}

#[test]
#[ignore = "algorithm lab: construction benchmark, run centrally in release"]
fn algorithm_lab_wholetrack_fixed_window_max() {
    let frames: usize = std::env::var("MOOSIK_ALGORITHM_LEVEL_FRAMES").ok().and_then(|v| v.parse().ok()).unwrap_or(16_384);
    let bars: usize = std::env::var("MOOSIK_ALGORITHM_LEVEL_BARS").ok().and_then(|v| v.parse().ok()).unwrap_or(1024);
    let repeats: usize = std::env::var("MOOSIK_ALGORITHM_LEVEL_REPEATS").ok().and_then(|v| v.parse().ok()).unwrap_or(3);
    assert!(frames >= 4 && bars > 0 && repeats > 0);
    let plane = PreFrames::from_v4_codes(codes(frames, bars), frames, bars);
    let (start, end) = (frames / 4, frames * 3 / 4);
    let no = AtomicBool::new(false);
    for k in [0u32, 1, 2, 3, 4, 6, 8, 10, 12] {
        let a = doubling_level(&plane, bars, k, start, end, &no).unwrap();
        let b = Level::build(&plane, bars, k, start, end, &no).unwrap();
        assert!(a.rows == b.rows, "k {k}: benchmark builders differ");
        drop((a, b));
        let (mut old_s, mut new_s) = (0.0, 0.0);
        for repeat in 0..repeats {
            // Alternate order to avoid consistently favouring a warm source.
            for old_first in [repeat % 2 == 0, repeat % 2 != 0] {
                let t = Instant::now();
                let level = if old_first { doubling_level(&plane, bars, k, start, end, &no) }
                            else { Level::build(&plane, bars, k, start, end, &no) }.unwrap();
                std::hint::black_box(&level.rows);
                let elapsed = t.elapsed().as_secs_f64();
                if old_first { old_s += elapsed; } else { new_s += elapsed; }
            }
        }
        let halo = (end - start + (1usize << k) - 1).min(frames - start);
        println!("algorithm_lab_wholetrack k={k} window={} frames={frames} bars={bars} rows={} old_ms={:.3} linear_ms={:.3} ratio={:.3} old_source_buffer_bytes={} linear_code_buffers_bytes={}",
                 1usize << k, end - start, old_s * 1000.0 / repeats as f64, new_s * 1000.0 / repeats as f64,
                 new_s / old_s, halo * bars * 2, (end - start) * bars * 2 + if k > 1 { bars * 2 } else { 0 });
    }
}

#[test]
#[ignore = "algorithm lab: coherent v4 patterns, run alone"]
fn algorithm_lab_wholetrack_patterns() {
    let frames: usize = std::env::var("MOOSIK_ALGORITHM_LEVEL_FRAMES").ok().and_then(|v| v.parse().ok()).unwrap_or(16_384);
    let bars: usize = std::env::var("MOOSIK_ALGORITHM_LEVEL_BARS").ok().and_then(|v| v.parse().ok()).unwrap_or(1024);
    let no = AtomicBool::new(false);
    let (start, end) = (frames / 4, frames * 3 / 4);
    for kind in ["random_v4", "coherent_harmonics", "silence", "rising"] {
        let data = match kind {
            "random_v4" => codes(frames, bars).into_iter().map(|v| v & 4095).collect(),
            "coherent_harmonics" => (0..frames * bars).map(|i| {
                let (f, b) = (i / bars, i % bars);
                let distance = (b % 87).abs_diff(43);
                let crest = 4095usize.saturating_sub(distance * 120);
                let t = (f / 3 + b * 7) % 4096;
                crest.saturating_sub(t.abs_diff(2048) / 2) as u16
            }).collect(),
            "silence" => vec![0; frames * bars],
            "rising" => (0..frames * bars).map(|i| ((i / bars) * 4095 / frames.max(1)) as u16).collect(),
            _ => unreachable!(),
        };
        let plane = PreFrames::from_v4_codes(data, frames, bars);
        for k in [0u32, 1, 2, 3, 4, 8, 12] {
            let expected = doubling_level(&plane, bars, k, start, end, &no).unwrap();
            assert_eq!(Level::build(&plane, bars, k, start, end, &no).unwrap().rows, expected.rows);
            drop(expected);
            for rep in 0..3 {
                for offset in 0..2 {
                    let arm = (rep + offset) % 2;
                    let at = Instant::now();
                    let out = match arm {
                        0 => doubling_level(&plane, bars, k, start, end, &no),
                        _ => Level::build(&plane, bars, k, start, end, &no),
                    }.unwrap();
                    let elapsed = at.elapsed();
                    std::hint::black_box(&out.rows);
                    println!("LAB_LEVEL_PATTERN kind={kind} frames={frames} bars={bars} k={k} rep={rep} arm={} ms={:.6}", ["old", "linear"][arm], elapsed.as_secs_f64()*1000.0);
                }
            }
        }
    }
}
