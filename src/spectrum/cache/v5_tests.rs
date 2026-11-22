//! The compact (v5) container through the cache's own entry points.
use super::*;

/// Rows with slow structure and noise on top, as an analysis has: a smooth
/// shape the model can follow, and texture it cannot.
fn rows(frames: usize, bars: usize, tag: usize) -> PreFrames {
    let mut s = (tag as u64 * 0x9E37_79B9 + 1) | 1;
    let rows: Vec<Vec<f32>> = (0..frames)
        .map(|t| (0..bars).map(|b| {
            s ^= s << 13;
            s ^= s >> 7;
            s ^= s << 17;
            let noise = ((s >> 40) % 1000) as f32 / 8000.0;
            (0.45 + 0.25 * ((t as f32 * 0.02 + tag as f32).sin() * (b as f32 * 0.07).cos()) + noise).clamp(0.0, 1.0)
        }).collect())
        .collect();
    PreFrames::from_analysis(&rows)
}

/// A compact file reads back to the rows that were written, through the
/// ordinary decoder, with and without a settings record; and it is smaller
/// than the plain one for the same rows.
#[test]
fn a_compact_file_reads_back_through_the_ordinary_decoder() {
    // Large enough that the compact form's tables (about 46 KB) are paid for.
    let f = rows(3000, 128, 1);
    for record in ["", "superlet;bars=64;test"] {
        let bytes = f.to_v5_with(record.as_bytes(), None).expect("these rows take the compact form");
        assert_eq!(decode(&bytes, 128, 2048).as_ref(), Ok(&f));
        assert_eq!(record_of(&bytes), (!record.is_empty()).then_some(record));
        assert!(is_compact(&bytes) && !is_compact(&f.to_v4_with(record.as_bytes())));
        assert!(bytes.len() < f.to_v4_with(record.as_bytes()).len());
    }
}

/// A plane coded against a reference needs that reference: none, or one of
/// another shape, is refused by name; another plane of the same shape by the
/// checksum. A plane coded without one ignores whatever is offered.
#[test]
fn a_reference_is_required_exactly_when_one_was_used() {
    let (mix, left) = (rows(700, 48, 1), rows(700, 48, 2));
    let limits = Limits::default();
    let with = left.to_v5_with(&[], Some(&mix)).expect("compact");
    assert_eq!(decode_with_reference(&with, 48, 2048, limits, Some(&mix)).as_ref(), Ok(&left));
    assert_eq!(decode_with_reference(&with, 48, 2048, limits, None), Err(CacheError::NeedsReference));
    assert_eq!(decode_with_reference(&with, 48, 2048, limits, Some(&rows(701, 48, 1))), Err(CacheError::NeedsReference));
    assert_eq!(decode_with_reference(&with, 48, 2048, limits, Some(&rows(700, 48, 9))), Err(CacheError::Checksum));
    let without = left.to_v5_with(&[], None).expect("compact");
    assert_eq!(decode_with_reference(&without, 48, 2048, limits, Some(&mix)).as_ref(), Ok(&left));
    // A reference of another shape is not a reason to write something wrong.
    assert!(left.to_v5_with(&[], Some(&rows(700, 32, 1))).is_none());
}

/// The header is held to the same rules as v4's: lengths that add up, a known
/// depth, reserved bytes zero, the display's bar count, the cell limit.
#[test]
fn a_compact_header_is_checked_like_a_plain_one() {
    let f = rows(600, 32, 3);
    let good = f.to_v5_with(b"rec", None).expect("compact");
    let limits = Limits::default();
    let with = |at: usize, v: u8| {
        let mut b = good.clone();
        b[at] = v;
        decode_with(&b, 32, 2048, limits)
    };
    assert_eq!(with(12, 13), Err(CacheError::UnsupportedBits(13)));
    assert_eq!(with(20, 2), Err(CacheError::Reserved));
    assert_eq!(with(23, 1), Err(CacheError::Reserved));
    assert_eq!(with(13, 4), Err(CacheError::Reserved));
    assert!(matches!(with(16, good[16] ^ 1), Err(CacheError::LengthMismatch { .. })));
    assert_eq!(decode_with(&good, 64, 2048, limits), Err(CacheError::BarsMismatch { want: 64, got: 32 }));
    let small = Limits { max_cells: 600 * 32 - 1, ..limits };
    assert!(matches!(decode_with(&good, 32, 2048, small), Err(CacheError::TooLarge { .. })));
    for cut in [0, 11, 23, 24, good.len() - 1] {
        assert!(decode_with(&good[..cut], 32, 2048, limits).is_err(), "cut at {cut}");
    }
}

/// Rows kept at another precision (a legacy cache's 16 bits) are not written
/// compact, and the setting off writes exactly the plain form.
#[test]
fn only_twelve_bit_rows_are_written_compact() {
    let f = rows(300, 16, 4);
    assert!(!compact(), "the compact setting is off unless a user turns it on");
    assert_eq!(f.to_bytes_with(b"r", None), f.to_v4_with(b"r"));
    // Stated on: the compact form where it is smaller, the plain one where
    // the plane is too small to pay for the compact form's tables.
    let big = rows(3000, 128, 5);
    assert!(is_compact(&big.to_bytes_as(b"r", None, true)));
    let tiny = rows(3, 16, 6);
    assert_eq!(tiny.to_bytes_as(b"r", None, true), tiny.to_v4_with(b"r"));
    let legacy = PreFrames::from_codes(vec![60_000u16; 300 * 16], 300, 16, LEGACY_MAX_CODE);
    assert!(legacy.to_v5_with(&[], None).is_none());
    assert!(PreFrames::default().to_v5_with(&[], None).is_none());
}

/// Compact coding keeps to the user's thread budget, through the cache's own
/// writer and reader: at budgets of 1, 2 and 6 every piece of its work runs
/// on the analysis pool, the pool has that size, no more pieces run at once
/// than that, and the bytes written are the same whatever the budget.
#[test]
fn compact_coding_keeps_to_the_thread_budget() {
    use crate::spectrum::gpu_calib;
    // Puts the previous worker selection back on drop.
    let _env = gpu_calib::isolated_test_env("v5_budget");
    let (mix, left) = (rows(3000, 128, 1), rows(3000, 128, 2));
    let me = std::thread::current().id();
    let calls = || v5::probe::CALLS.lock().unwrap_or_else(|e| e.into_inner());
    let mut files = Vec::new();
    for budget in [1usize, 2, 6] {
        gpu_calib::set_workers(budget);
        // The setter holds a budget to the machine's cores.
        let want = gpu_calib::workers();
        calls().retain(|c| c.0 != me);
        let bytes = left.to_bytes_as(b"r", Some(&mix), true);
        assert!(is_compact(&bytes), "budget {budget}: setup: not written compact");
        assert_eq!(decode_with_reference(&bytes, 128, 2048, Limits::default(), Some(&mix)).as_ref(), Ok(&left), "budget {budget}");
        let mine: Vec<_> = calls().iter().filter(|c| c.0 == me).copied().collect();
        assert!(mine.len() >= 4, "budget {budget}: setup: {} parallel sections seen", mine.len());
        for (_, peak, pool, stray) in &mine {
            assert_eq!(*stray, 0, "budget {budget}: work ran off the analysis pool");
            assert_eq!(*pool, want, "budget {budget}: a pool of {pool} threads ran it");
            assert!(*peak <= want, "budget {budget}: {peak} pieces at once");
        }
        assert!(want == 1 || mine.iter().any(|c| c.1 > 1), "budget {budget}: setup: nothing overlapped, so the limit was not exercised");
        files.push(bytes);
    }
    assert!(files.windows(2).all(|w| w[0] == w[1]), "the bytes depend on the budget");
}
