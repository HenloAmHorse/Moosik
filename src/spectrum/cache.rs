//! The pre-process cache: its on-disk format, and how it lives in memory.
//!
//! # Why 12 bits
//!
//! A cached value is a level on a fixed 80 dB axis, normalised to `0.0..=1.0`.
//! v2 and v3 stored it as a `u16`, which is 65 536 levels for a plot that is at
//! most a few thousand pixels tall. v4 stores 4 096, so the quantiser step is
//! `80 / 4095 = 0.01954 dB` and the worst rounding error is half of that:
//!
//! ```text
//! 80 / (2 × 4095) = 0.009768 dB
//! ```
//!
//! Measured over 11 tracks at the owner's settings, that costs nothing visible
//! and returns about 29 % of the file. **The 29 % is that corpus at those
//! settings**, not a guarantee, and an existing v2/v3 file does not shrink
//! because a v4 reader exists — only a fresh analysis writes v4.
//!
//! The bound above is the raw per-cell quantisation error and nothing more. It
//! is not a perceptual threshold, and it says nothing about what interpolation,
//! peak-hold decay or any other consumer does with a coarser grid.
//!
//! # Why the values stay quantised in memory
//!
//! `PreFrames` holds the codes, not decoded floats. A 248-second track at
//! 180 fps and 1024 bars is 44 645 rows; as `Vec<Vec<f32>>` that was 174 MiB
//! resident, and it existed so that a display which needs 1024 values per frame
//! could read one row. As `u16` codes it is 87 MiB, one allocation instead of
//! 44 645, and a row is a subslice rather than a decode.
//!
//! `u16` and not a packed 12-bit array on purpose: it holds a v4 code and a
//! legacy 16-bit code equally exactly, so a v2/v3 cache keeps every bit of the
//! precision it was written with rather than being requantised on load to
//! simplify the type.

use std::io::{Read, Write};
use std::path::Path;

/// Span of the level axis, in dB. A stored 0 means −80 dB.
///
/// Reached only through [`V4_MAX_QUANT_ERROR_DB`] and the tests that hold the
/// quantiser to it. Kept as a named constant because the bound is a claim this
/// module makes, and a claim with no expression of it in the code is one
/// nobody can check.
#[allow(dead_code)]
pub const AXIS_DB: f32 = 80.0;

/// Bit depth v4 writes.
pub const V4_BITS: u32 = 12;

/// Largest code v4 writes: `2^12 − 1`.
pub const V4_MAX_CODE: u16 = (1 << V4_BITS) - 1;

/// Largest code the legacy formats write.
pub const LEGACY_MAX_CODE: u16 = u16::MAX;

/// Worst-case rounding error of the v4 quantiser, in dB.
///
/// Half a step. This is the raw per-cell bound and is deliberately not
/// described as anything else.
#[allow(dead_code)]
pub const V4_MAX_QUANT_ERROR_DB: f32 = AXIS_DB / (2.0 * V4_MAX_CODE as f32);

/// Most frames a cache may declare. Matches the bound v3 enforced.
pub const MAX_FRAMES: usize = 500_000;

/// What one cache read is permitted to consume.
///
/// # Why this exists
///
/// Every dimension in a cache header is file-controlled, and checked
/// multiplication only stops the arithmetic overflowing — it does not stop a
/// 64-byte file declaring six million cells and getting six megabytes allocated
/// on the way to failing. The limits below are checked **before** any of the
/// three large allocations a read performs: the file bytes, the decompressed
/// planes, and the decoded matrix.
///
/// # The numbers, and what they cost
///
/// `max_cells` is deliberately the format's own implicit ceiling,
/// `MAX_FRAMES × MAX_BAR_COUNT` — 512 M cells, a gibibyte of `u16`. That is
/// what a legitimate 46-minute track at 180 fps and 1024 bars needs, so
/// tightening it would refuse real files. It bounds the damage; it is not a
/// snug fit, and nothing here pretends otherwise.
///
/// `max_expansion` is the ratio a compressed block is trusted to expand by. It
/// is *generous*: a digitally silent track's low plane is all zeros and LZ4
/// takes it to roughly 250:1, so a tight bound would refuse real material. 1024
/// still turns "six million cells from forty bytes" — a 160 000:1 claim — into
/// a refusal before anything is allocated.
#[derive(Clone, Copy, Debug)]
pub struct Limits {
    /// Largest file that will be read into memory at all.
    pub max_file_bytes: usize,
    /// Largest decoded matrix, in cells.
    pub max_cells: usize,
    /// Most a compressed block is trusted to expand, as a ratio.
    pub max_expansion: usize,
}

impl Default for Limits {
    fn default() -> Self {
        Self {
            max_file_bytes: 512 << 20,
            max_cells: MAX_FRAMES * crate::spectrum::MAX_BAR_COUNT,
            max_expansion: 1024,
        }
    }
}

// ---------------------------------------------------------------------------
// The v4 container
// ---------------------------------------------------------------------------

// v2: [magic, frames, bars, lz4(u16 LE interleaved)]
// v3: as v2, but the payload is zigzag-delta across frequency and split into a
//     high-byte plane followed by a low-byte plane.
const MAGIC_V2: u32 = 0x4D53_5032; // "MSP2"
const MAGIC_V3: u32 = 0x4D53_5033; // "MSP3"

/// v4 magic, "MSP4".
///
/// # Layout
///
/// All integers little-endian. Header is 24 bytes:
///
/// ```text
///  0  u32  magic          0x4D535034
///  4  u32  frames         1..=MAX_FRAMES
///  8  u32  bars           1..=MAX_BAR_COUNT
/// 12  u8   bits           12; any other value is unsupported, not an error to
///                         guess at
/// 13  u8   flags          reserved, must be 0
/// 14  u16  reserved       must be 0
/// 16  u32  lo_len         compressed length of the low plane, in bytes
/// 20  u32  hi_len         compressed length of the high plane, in bytes
/// 24  ..   lo block       LZ4 block, no size prefix
/// ..  ..   hi block       LZ4 block, no size prefix
/// ```
///
/// The file must be exactly `24 + lo_len + hi_len` bytes. Neither block carries
/// its own uncompressed size: the reader knows both from `frames × bars` and
/// passes them to the decompressor, so a corrupt file cannot ask for an
/// allocation by declaring one.
///
/// # Payload
///
/// Values are quantised to `round(v.clamp(0,1) × 4095)`, then within each row
/// zigzag-delta-coded across frequency:
///
/// * `prev` **resets to 0 at the start of every row**, so a row is decodable
///   without the rows before it and a torn frame cannot poison the rest;
/// * the difference is wrapped into a signed 12-bit value before zigzagging,
///   exactly as v3 wraps into `i16`, which keeps the common small steps small
///   and still reconstructs exactly;
/// * the resulting 12-bit code splits into a low byte and a high nibble.
///
/// The low plane is one byte per cell. The high plane is 4 bits per cell,
/// LSB-first, two cells per byte, packed **continuously across rows** — rows
/// are not padded to a byte boundary. Only the final byte can carry padding,
/// when the cell count is odd; it is written as zero and ignored on read.
const MAGIC_V4: u32 = 0x4D53_5034;

const V4_HEADER: usize = 24;

/// Why a cache could not be read. Every variant is a reason to recompute, never
/// a reason to trust part of the file.
#[derive(Debug, PartialEq, Eq)]
pub enum CacheError {
    TooShort,
    BadMagic(u32),
    /// A dimension outside what this build will allocate for.
    BadDimensions { frames: usize, bars: usize },
    /// Bars on disk are not the bars being displayed; the cache is for another
    /// setting.
    BarsMismatch { want: usize, got: usize },
    UnsupportedBits(u8),
    /// A reserved field was not zero — a newer writer, so do not guess.
    Reserved,
    /// The declared block lengths do not add up to the file.
    LengthMismatch { declared: usize, actual: usize },
    Decompress,
    /// A block decompressed to the wrong size.
    PayloadSize { want: usize, got: usize },
    /// A declared size exceeds what one read may consume. Refused **before**
    /// the allocation it would have caused.
    TooLarge { what: &'static str, got: usize, limit: usize },
    /// An allocation this size could not be reserved.
    OutOfMemory { bytes: usize },
}

impl std::fmt::Display for CacheError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            CacheError::TooShort => write!(f, "file shorter than a header"),
            CacheError::BadMagic(m) => write!(f, "unrecognised magic {m:#010x}"),
            CacheError::BadDimensions { frames, bars } => {
                write!(f, "implausible dimensions {frames}×{bars}")
            }
            CacheError::BarsMismatch { want, got } => {
                write!(f, "cache holds {got} bars, display wants {want}")
            }
            CacheError::UnsupportedBits(b) => write!(f, "unsupported depth {b}"),
            CacheError::Reserved => write!(f, "reserved field set; written by a newer version"),
            CacheError::LengthMismatch { declared, actual } => {
                write!(f, "declares {declared} bytes, file is {actual}")
            }
            CacheError::Decompress => write!(f, "decompression failed"),
            CacheError::PayloadSize { want, got } => {
                write!(f, "payload is {got} values, expected {want}")
            }
            CacheError::TooLarge { what, got, limit } => {
                write!(f, "{what} is {got}, over the {limit} this read allows")
            }
            CacheError::OutOfMemory { bytes } => {
                write!(f, "could not reserve {bytes} bytes")
            }
        }
    }
}

// ---------------------------------------------------------------------------
// Compact resident frames
// ---------------------------------------------------------------------------

/// A whole track's analysis, held as quantiser codes.
#[derive(Clone, Default, PartialEq)]
pub struct PreFrames {
    codes: Vec<u16>,
    bars: usize,
    frames: usize,
    /// Largest code, and therefore what returns the `0.0..=1.0` domain. 4095
    /// for v4, 65535 for a legacy cache kept at its own precision.
    max_code: u16,
    /// `1 / max_code`, stored so every consumer computes a value the same way.
    ///
    /// Not a micro-optimisation. The display multiplies by a reciprocal and a
    /// decoded row used to divide, which differ in the last bit, and two paths
    /// that disagree by one ULP make an exact comparison between "what was
    /// shown" and "what was stored" fail for a reason that has nothing to do
    /// with the cache.
    inv_max: f32,
}

/// Summarised, never dumped. A failing `assert_eq!` on a whole track would
/// otherwise print tens of millions of codes.
impl std::fmt::Debug for PreFrames {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "PreFrames {{ {} frames × {} bars, max_code {}, {} KiB }}",
            self.frames,
            self.bars,
            self.max_code,
            self.capacity_bytes() / 1024
        )
    }
}

impl PreFrames {
    /// Quantise a freshly analysed track to what v4 will store.
    ///
    /// Done here, once, rather than at write time, so the values the display
    /// shows immediately after an analysis are the same values it will show
    /// after a reload. Quantising only on the way to disk would have made a
    /// fresh result and a reloaded one differ by up to half a step.
    pub fn from_analysis(rows: &[Vec<f32>]) -> Self {
        let bars = rows.first().map(Vec::len).unwrap_or(0);
        if bars == 0 {
            return Self::default();
        }
        let frames = rows.iter().filter(|r| r.len() == bars).count();
        let mut codes = Vec::with_capacity(frames * bars);
        for row in rows.iter().filter(|r| r.len() == bars) {
            codes.extend(row.iter().map(|&v| quantise(v, V4_MAX_CODE)));
        }
        Self::assemble(codes, frames, bars, V4_MAX_CODE)
    }

    /// Build from codes already at a known depth. Used by the readers.
    fn from_codes(codes: Vec<u16>, frames: usize, bars: usize, max_code: u16) -> Self {
        debug_assert_eq!(codes.len(), frames * bars);
        Self::assemble(codes, frames, bars, max_code)
    }

    fn assemble(codes: Vec<u16>, frames: usize, bars: usize, max_code: u16) -> Self {
        Self { codes, bars, frames, max_code, inv_max: 1.0 / max_code as f32 }
    }

    /// Drop every row **and release the memory holding them**.
    ///
    /// `Vec::clear` sets the length to zero and keeps the capacity, which for a
    /// long track is 87 MiB retained by a cache nobody is going to read again.
    /// The whole point of clearing an obsolete analysis is to get that back, so
    /// the buffer is replaced rather than emptied.
    ///
    /// Still called `clear`, and still called through `clear_pre_frames`, so
    /// the source-level guard in `nothing_writes_the_cached_matrix_directly`
    /// keeps counting what it was written to count.
    pub fn clear(&mut self) {
        self.codes = Vec::new();
        self.frames = 0;
        self.bars = 0;
    }

    pub fn is_empty(&self) -> bool {
        self.frames == 0 || self.bars == 0
    }

    /// Number of frames.
    pub fn len(&self) -> usize {
        self.frames
    }

    pub fn bars(&self) -> usize {
        self.bars
    }

    /// Largest code in this cache's own depth.
    ///
    /// Used by the tests that check a legacy cache was not requantised on load.
    /// The display path uses [`PreFrames::inv_max`] instead, which is the same
    /// fact in the form every consumer needs it.
    #[allow(dead_code)]
    pub fn max_code(&self) -> u16 {
        self.max_code
    }

    /// One row of codes. `None` past the end.
    ///
    /// A subslice: no decode, no allocation, no scan of the rows before it.
    #[inline]
    pub fn row(&self, frame: usize) -> Option<&[u16]> {
        if frame >= self.frames {
            return None;
        }
        let base = frame * self.bars;
        self.codes.get(base..base + self.bars)
    }

    /// The `0.0..=1.0` value a code stands for, at this cache's depth.
    #[inline]
    pub fn value(&self, code: u16) -> f32 {
        code as f32 * self.inv_max
    }

    /// The reciprocal the values are built from, for a caller converting a whole
    /// row inline. Using anything else here reintroduces the ULP split.
    #[inline]
    pub fn inv_max(&self) -> f32 {
        self.inv_max
    }

    /// One row as floats, allocating.
    ///
    /// For tests. The display path deliberately does not use it — it reads
    /// codes and converts in place — so this is dead in a non-test build and
    /// says so rather than being kept alive by a caller invented for it.
    #[allow(dead_code)]
    pub fn row_vec(&self, frame: usize) -> Vec<f32> {
        self.row(frame)
            .map(|r| r.iter().map(|&c| self.value(c)).collect())
            .unwrap_or_default()
    }

    /// Largest value each bar reaches across the whole track.
    ///
    /// Replaces a full-track scan that took `&[Vec<f32>]`. One pass over the
    /// codes, no intermediate rows.
    pub fn peak_per_bar(&self) -> Vec<f32> {
        let mut peak = vec![0u16; self.bars];
        for row in self.codes.chunks_exact(self.bars.max(1)) {
            for (p, &c) in peak.iter_mut().zip(row) {
                if c > *p {
                    *p = c;
                }
            }
        }
        peak.into_iter().map(|c| self.value(c)).collect()
    }

    /// Bytes of payload — what the codes actually occupy.
    ///
    /// Distinct from [`PreFrames::capacity_bytes`] on purpose: a `Vec` that has
    /// been emptied reports zero length while still holding its buffer, so a
    /// length-derived figure is the wrong number to answer "did clearing this
    /// give the memory back".
    #[allow(dead_code)]
    pub fn payload_bytes(&self) -> usize {
        self.codes.len() * std::mem::size_of::<u16>()
    }

    /// Bytes of backing allocation — what the process is actually holding.
    pub fn capacity_bytes(&self) -> usize {
        self.codes.capacity() * std::mem::size_of::<u16>()
    }

    // ── encoding ────────────────────────────────────────────────────────────

    /// Serialise as v4. Only meaningful for 12-bit data.
    ///
    /// # What this deliberately does not keep
    ///
    /// The straightforward encoder holds four things at once: a low-byte plane
    /// (1 B/cell), an **unpacked** high-nibble plane (1 B/cell), the packed
    /// nibbles (0.5), and two compressed blocks that are then copied into the
    /// file. Two of those are avoidable and neither costs anything to avoid.
    ///
    /// * Nibbles pair by cell index across the whole plane, so they can be
    ///   packed *during* the scan that produces them. The unpacked plane never
    ///   needs to exist.
    /// * `compress_into` writes into a buffer the caller owns, so each block can
    ///   be compressed straight into the file rather than into a `Vec` that is
    ///   then copied in and dropped.
    ///
    /// `lz4_flex::compress` and `compress_into` both call
    /// `compress_into_sink_with_dict::<false>` with an empty dictionary and
    /// differ only in the sink they write through, so identical bytes are
    /// expected by construction. **Expected is not established**:
    /// `the_v4_encoder_is_byte_identical_to_the_staged_form` holds this to the
    /// previous encoder's exact output, and that test is what decides.
    pub fn to_v4(&self) -> Vec<u8> {
        let cells = self.frames * self.bars;
        let packed_len = cells.div_ceil(2);

        // One scan, two planes: the low bytes, and the high nibbles already
        // packed two to a byte. The parity that matters is the running cell
        // count, which `lo.len()` already is — nibbles pair across the plane,
        // not within a row.
        let mut lo: Vec<u8> = Vec::with_capacity(cells);
        let mut hi: Vec<u8> = Vec::with_capacity(packed_len);
        let mut pending = 0u8;
        for row in self.codes.chunks_exact(self.bars.max(1)) {
            let mut prev = 0i32;
            for &q in row {
                let q = q.min(V4_MAX_CODE) as i32;
                let d = wrap_signed(q - prev, V4_BITS);
                prev = q;
                let z = (((d << 1) ^ (d >> 31)) as u32) & (V4_MAX_CODE as u32);
                lo.push((z & 0xFF) as u8);
                let nib = ((z >> 8) as u8) & 0x0F;
                if lo.len() % 2 == 1 {
                    pending = nib;
                } else {
                    hi.push(pending | (nib << 4));
                }
            }
        }
        // An odd cell count leaves one nibble in hand, and its spare half-byte
        // is written as zero — exactly what packing the finished plane did.
        if cells % 2 == 1 {
            hi.push(pending);
        }
        debug_assert_eq!(hi.len(), packed_len, "the packed nibble plane is the wrong size");

        let mut out: Vec<u8> = Vec::with_capacity(
            V4_HEADER + lz4_flex::block::get_maximum_output_size(lo.len()),
        );
        out.resize(V4_HEADER, 0);

        // The low plane goes first and is released the moment its block is
        // written, so the high plane's pass never holds both.
        let at = out.len();
        out.resize(at + lz4_flex::block::get_maximum_output_size(lo.len()), 0);
        let lo_len = lz4_flex::compress_into(&lo, &mut out[at..])
            .expect("the slice is get_maximum_output_size");
        drop(lo);
        out.truncate(at + lo_len);

        let at = out.len();
        out.resize(at + lz4_flex::block::get_maximum_output_size(hi.len()), 0);
        let hi_len = lz4_flex::compress_into(&hi, &mut out[at..])
            .expect("the slice is get_maximum_output_size");
        drop(hi);
        out.truncate(at + hi_len);

        let h = &mut out[..V4_HEADER];
        h[0..4].copy_from_slice(&MAGIC_V4.to_le_bytes());
        h[4..8].copy_from_slice(&(self.frames as u32).to_le_bytes());
        h[8..12].copy_from_slice(&(self.bars as u32).to_le_bytes());
        h[12] = V4_BITS as u8;
        h[13] = 0; // flags
        h[14..16].copy_from_slice(&0u16.to_le_bytes()); // reserved
        h[16..20].copy_from_slice(&(lo_len as u32).to_le_bytes());
        h[20..24].copy_from_slice(&(hi_len as u32).to_le_bytes());

        // The buffer was sized for the worst case LZ4 could have produced; the
        // caller keeps this, and the sidecar keeps two of them, so give the
        // slack back. The copy this costs is smaller than the peak above.
        out.shrink_to_fit();
        out
    }
}

// ---------------------------------------------------------------------------
// Quantisation
// ---------------------------------------------------------------------------

/// A level in `0.0..=1.0` as a code.
///
/// Only NaN is special-cased, to zero: it carries no level and `clamp` would
/// propagate it. The infinities clamp like any other out-of-range value, so
/// `+inf` becomes full scale rather than silence — treating "louder than
/// anything" as "nothing at all" is the larger lie of the two, and the dB
/// conversion upstream already clamps, so neither should ever arrive here.
#[inline]
pub fn quantise(v: f32, max_code: u16) -> u16 {
    if v.is_nan() {
        return 0;
    }
    (v.clamp(0.0, 1.0) * max_code as f32).round() as u16
}

/// Wrap a difference into a signed `bits`-wide value.
#[inline]
fn wrap_signed(d: i32, bits: u32) -> i32 {
    let shift = 32 - bits;
    (d << shift) >> shift
}

// ---------------------------------------------------------------------------
// Reading
// ---------------------------------------------------------------------------

fn le_u32(b: &[u8], at: usize) -> u32 {
    u32::from_le_bytes([b[at], b[at + 1], b[at + 2], b[at + 3]])
}

/// Decode any supported cache under the default [`Limits`].
///
/// Production reaches the decoder through [`read`], which checks the file's
/// size from its metadata first; this entry point takes bytes that are already
/// in hand and is used by the tests.
#[allow(dead_code)]
pub fn decode(raw: &[u8], want_bars: usize, max_bars: usize) -> Result<PreFrames, CacheError> {
    decode_with(raw, want_bars, max_bars, Limits::default())
}

/// Decode under explicit limits.
///
/// `want_bars` is the display's bar count; a cache for a different one is
/// refused rather than stretched. Every limit is checked before the allocation
/// it governs, which is what lets a test observe a refusal by passing a tiny
/// limit instead of by allocating a gigabyte.
pub fn decode_with(
    raw: &[u8],
    want_bars: usize,
    max_bars: usize,
    limits: Limits,
) -> Result<PreFrames, CacheError> {
    if raw.len() > limits.max_file_bytes {
        return Err(CacheError::TooLarge {
            what: "file",
            got: raw.len(),
            limit: limits.max_file_bytes,
        });
    }
    if raw.len() < 12 {
        return Err(CacheError::TooShort);
    }
    let magic = le_u32(raw, 0);
    let frames = le_u32(raw, 4) as usize;
    let bars = le_u32(raw, 8) as usize;

    if frames == 0 || frames > MAX_FRAMES || bars == 0 || bars > max_bars {
        return Err(CacheError::BadDimensions { frames, bars });
    }
    if bars != want_bars {
        return Err(CacheError::BarsMismatch { want: want_bars, got: bars });
    }
    // Checked, because `frames × bars` is attacker-influenced and both bounds
    // above still permit half a billion cells.
    let cells = frames
        .checked_mul(bars)
        .ok_or(CacheError::BadDimensions { frames, bars })?;
    // Before anything is allocated for it.
    if cells > limits.max_cells {
        return Err(CacheError::TooLarge {
            what: "decoded cells",
            got: cells,
            limit: limits.max_cells,
        });
    }

    match magic {
        MAGIC_V4 => decode_v4(raw, frames, bars, cells, limits),
        MAGIC_V2 | MAGIC_V3 => {
            decode_legacy(raw, frames, bars, cells, magic == MAGIC_V3, limits)
        }
        other => Err(CacheError::BadMagic(other)),
    }
}

/// Refuse a decompression whose claimed output is out of all proportion to the
/// bytes present to produce it.
fn check_expansion(
    what: &'static str,
    compressed: usize,
    want: usize,
    limits: Limits,
) -> Result<(), CacheError> {
    let ceiling = compressed.saturating_mul(limits.max_expansion).max(1024);
    if want > ceiling {
        return Err(CacheError::TooLarge { what, got: want, limit: ceiling });
    }
    Ok(())
}

/// A `Vec<u16>` of exactly `n`, or an error rather than an abort.
fn reserve_codes(n: usize) -> Result<Vec<u16>, CacheError> {
    let mut v: Vec<u16> = Vec::new();
    v.try_reserve_exact(n)
        .map_err(|_| CacheError::OutOfMemory { bytes: n * 2 })?;
    Ok(v)
}

fn decode_v4(
    raw: &[u8],
    frames: usize,
    bars: usize,
    cells: usize,
    limits: Limits,
) -> Result<PreFrames, CacheError> {
    if raw.len() < V4_HEADER {
        return Err(CacheError::TooShort);
    }
    let bits = raw[12];
    if bits as u32 != V4_BITS {
        return Err(CacheError::UnsupportedBits(bits));
    }
    if raw[13] != 0 || raw[14] != 0 || raw[15] != 0 {
        return Err(CacheError::Reserved);
    }
    let lo_len = le_u32(raw, 16) as usize;
    let hi_len = le_u32(raw, 20) as usize;
    let declared = V4_HEADER
        .checked_add(lo_len)
        .and_then(|n| n.checked_add(hi_len))
        .ok_or(CacheError::LengthMismatch { declared: usize::MAX, actual: raw.len() })?;
    if declared != raw.len() {
        return Err(CacheError::LengthMismatch { declared, actual: raw.len() });
    }

    // Both output sizes come from the dimensions, never from the file, and both
    // are checked against the bytes actually present before the decompressor is
    // asked for the allocation.
    let hi_packed = cells.div_ceil(2);
    check_expansion("low plane", lo_len, cells, limits)?;
    check_expansion("high plane", hi_len, hi_packed, limits)?;
    let lo = lz4_flex::decompress(&raw[V4_HEADER..V4_HEADER + lo_len], cells)
        .map_err(|_| CacheError::Decompress)?;
    let hi_bytes =
        lz4_flex::decompress(&raw[V4_HEADER + lo_len..], hi_packed).map_err(|_| CacheError::Decompress)?;
    if lo.len() != cells {
        return Err(CacheError::PayloadSize { want: cells, got: lo.len() });
    }
    if hi_bytes.len() != hi_packed {
        return Err(CacheError::PayloadSize { want: hi_packed, got: hi_bytes.len() });
    }

    // Nibbles are read out of the packed buffer as they are needed. The
    // unpacked plane this used to build was a byte per cell, alive alongside
    // the packed buffer, the low plane and the codes — about a quarter of the
    // read's memory, held to save an index shift.
    //
    // `hi_packed` is `cells.div_ceil(2)` and has just been checked, so `i >> 1`
    // is in range for every `i < cells`. An odd cell count leaves the last high
    // nibble unread: that is the spare zero the encoder wrote.
    let mut codes = reserve_codes(cells)?;
    for r in 0..frames {
        let base = r * bars;
        let mut prev = 0i32;
        for i in base..base + bars {
            let packed = hi_bytes[i >> 1];
            let nib = if i % 2 == 0 { packed & 0x0F } else { packed >> 4 };
            let z = (lo[i] as u32) | ((nib as u32) << 8);
            let d = ((z >> 1) as i32) ^ -((z & 1) as i32);
            prev = (prev + d) & V4_MAX_CODE as i32;
            codes.push(prev as u16);
        }
    }
    Ok(PreFrames::from_codes(codes, frames, bars, V4_MAX_CODE))
}

fn decode_legacy(
    raw: &[u8],
    frames: usize,
    bars: usize,
    cells: usize,
    v3: bool,
    limits: Limits,
) -> Result<PreFrames, CacheError> {
    let want = cells.checked_mul(2).ok_or(CacheError::BadDimensions { frames, bars })?;
    // The legacy container prepends the uncompressed size. It is not trusted:
    // the expected size is computed from the dimensions and passed in, and a
    // file that decompresses to anything else is refused.
    if raw.len() < 16 {
        return Err(CacheError::TooShort);
    }
    let body = &raw[12..];
    check_expansion("legacy payload", body.len().saturating_sub(4), want, limits)?;
    let payload =
        lz4_flex::decompress(&body[4..], want).map_err(|_| CacheError::Decompress)?;
    if payload.len() != want {
        return Err(CacheError::PayloadSize { want, got: payload.len() });
    }

    let mut codes = reserve_codes(cells)?;
    if !v3 {
        for i in 0..cells {
            codes.push(u16::from_le_bytes([payload[i * 2], payload[i * 2 + 1]]));
        }
    } else {
        let (hi, lo) = payload.split_at(cells);
        for r in 0..frames {
            let base = r * bars;
            let mut prev = 0i32;
            for i in base..base + bars {
                let z = u16::from_le_bytes([lo[i], hi[i]]) as u32;
                let d = ((z >> 1) as i32) ^ -((z & 1) as i32);
                prev = (prev + d) & 0xFFFF;
                codes.push(prev as u16);
            }
        }
    }
    // Legacy precision is preserved: the codes stay 16-bit and the divisor
    // stays 65535, so opening an old cache does not requantise it.
    Ok(PreFrames::from_codes(codes, frames, bars, LEGACY_MAX_CODE))
}

// ---------------------------------------------------------------------------
// Writing
// ---------------------------------------------------------------------------

/// Points at which a test may make the write fail.
///
/// Real failures here are a full disk, a revoked permission, a reader holding
/// the destination open, or the process dying — none of which a test can
/// arrange reliably, and the previous attempt at one arranged nothing at all:
/// it created a directory at a path the timestamped temporary could never
/// collide with, so the write succeeded and the test asserted only that the
/// result still decoded.
#[cfg(test)]
pub(crate) mod fault {
    use std::cell::{Cell, RefCell};

    #[derive(Clone, Copy, PartialEq, Eq, Debug)]
    pub enum At {
        None,
        /// After some of the temporary is on disk, before the rest.
        MidWrite,
        /// After a complete temporary, before it is published.
        BeforePublish,
    }

    thread_local! {
        static POINT: Cell<At> = const { Cell::new(At::None) };
    }

    pub fn arm(at: At) {
        POINT.with(|p| p.set(at));
    }

    /// True once, if armed at `at`. Disarms itself so one arming fires once.
    pub fn fires(at: At) -> bool {
        POINT.with(|p| {
            if p.get() == at {
                p.set(At::None);
                true
            } else {
                false
            }
        })
    }

    /// Moments inside an operation at which a test may change state the
    /// operation then observes for itself.
    ///
    /// This is not the same thing as [`At`]. `At` injects a failure; a hook
    /// injects nothing — it lets a test move the world underneath an operation
    /// that is genuinely in flight, so what the operation does next is its own
    /// real behaviour rather than a substituted error.
    #[derive(Clone, Copy, PartialEq, Eq, Debug)]
    pub enum Hook {
        /// The temporary is written, before the publication decision.
        TempWritten,
        /// A file's length has been taken, before its contents are read.
        AfterMetadata,
        /// A mono cache has been encoded and nothing has touched the disk.
        ///
        /// Reaching this point *is* the observation: the encode is the one step
        /// the save wrapper's own cancellation check exists to skip, and it
        /// leaves no trace a test could otherwise look for.
        CacheEncoded,
        /// The same moment for a sidecar.
        SidecarEncoded,
    }

    type Callback = Box<dyn Fn()>;

    thread_local! {
        static HOOKS: RefCell<Vec<(Hook, Callback)>> = const { RefCell::new(Vec::new()) };
    }

    /// Arm `f` to run the next time `at` is reached on this thread, once.
    pub fn on(at: Hook, f: impl Fn() + 'static) {
        HOOKS.with(|h| h.borrow_mut().push((at, Box::new(f))));
    }

    /// Forget every armed hook on this thread.
    ///
    /// A hook that is armed at a moment the operation then *skips* is never
    /// consumed, and the next arming of the same moment would find the stale
    /// one first. A test that deliberately skips a moment clears afterwards.
    pub fn clear() {
        HOOKS.with(|h| h.borrow_mut().clear());
    }

    /// Run and remove the callback armed at `at`, if there is one.
    pub fn run(at: Hook) {
        let cb = HOOKS.with(|h| {
            let mut v = h.borrow_mut();
            v.iter().position(|(a, _)| *a == at).map(|i| v.remove(i).1)
        });
        if let Some(cb) = cb {
            cb();
        }
    }
}

#[cfg(not(test))]
pub(crate) mod fault {
    #[derive(Clone, Copy, PartialEq, Eq)]
    pub enum At {
        MidWrite,
        BeforePublish,
    }
    #[derive(Clone, Copy, PartialEq, Eq)]
    pub enum Hook {
        TempWritten,
        AfterMetadata,
        CacheEncoded,
        SidecarEncoded,
    }
    #[inline(always)]
    pub fn fires(_: At) -> bool {
        false
    }
    #[inline(always)]
    pub fn run(_: Hook) {}
}

/// How a write ended.
///
/// Distinguishing these is the point: a cancelled write and a published one
/// both leave the destination in a valid state, and a caller that cannot tell
/// them apart will report a cancelled save as a successful one.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum Written {
    /// The rename happened. The file at `path` is the new one.
    Published,
    /// Cancellation was observed before the commit decision. The destination
    /// is exactly as it was — its previous bytes, or still absent — and no
    /// temporary survives.
    Cancelled,
}

/// Write `bytes` to `path`, replacing any existing file only once the new one
/// is complete.
///
/// # The publication boundary
///
/// **The rename is the publication boundary.** Nothing before it is visible to
/// a reader: the bytes go to a uniquely named sibling, and until the rename
/// succeeds the destination is either absent or exactly its previous contents.
/// Anything that goes wrong before that point — a short write, a full disk, a
/// cancelled analysis — leaves the destination untouched and removes the
/// temporary.
///
/// Cancellation is carried through to that boundary, not merely checked
/// before the call. `cancelled` is consulted twice: once on entry, with the
/// encoding done and nothing yet on disk, and once immediately before the
/// rename. **That second check is the commit decision.** Cancellation that
/// arrives after it may be too late — the rename is a single step and nothing
/// here pretends to make it interruptible — but cancellation observed at or
/// before it is honoured, and the destination keeps its previous contents.
///
/// A cache is written at the end of an analysis that can take minutes, and the
/// old file is often still the one being displayed. Writing in place means a
/// crash, a full disk or an abort between `create` and the last `write_all`
/// leaves a truncated file where a valid one was, and the next run recomputes
/// the whole track. So the new file is built under a unique sibling name and
/// moved into place in one step.
///
/// `fs::rename` replaces an existing destination on Windows as well as on Unix.
/// If it fails — the destination held open by a reader, for instance — the
/// temporary is removed and the existing file is left exactly as it was.
pub fn write_atomic_cancellable(
    path: &Path,
    bytes: &[u8],
    cancelled: &dyn Fn() -> bool,
) -> std::io::Result<Written> {
    // The encoding is done and nothing exists yet, so this one is free.
    if cancelled() {
        return Ok(Written::Cancelled);
    }
    if let Some(parent) = path.parent() {
        std::fs::create_dir_all(parent)?;
    }
    let stamp = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|d| d.as_nanos())
        .unwrap_or(0);
    let tmp = path.with_extension(format!("tmp-{}-{stamp:x}", std::process::id()));

    let write = || -> std::io::Result<()> {
        let mut f = std::fs::File::create(&tmp)?;
        if fault::fires(fault::At::MidWrite) {
            // Half of it, then stop — the shape of a disk filling up.
            f.write_all(&bytes[..bytes.len() / 2])?;
            f.sync_all()?;
            return Err(std::io::Error::other("injected mid-write failure"));
        }
        f.write_all(bytes)?;
        // The temporary is complete and nothing is published. A test cancels
        // here, so the flag it flips changes while the operation is genuinely
        // in flight rather than before it ever started.
        fault::run(fault::Hook::TempWritten);
        f.sync_all()
    };
    if let Err(e) = write() {
        let _ = std::fs::remove_file(&tmp);
        return Err(e);
    }
    if fault::fires(fault::At::BeforePublish) {
        let _ = std::fs::remove_file(&tmp);
        return Err(std::io::Error::other("injected publication failure"));
    }
    // The commit decision. Everything above is invisible to a reader; the
    // rename below is not.
    if cancelled() {
        let _ = std::fs::remove_file(&tmp);
        return Ok(Written::Cancelled);
    }
    if let Err(e) = std::fs::rename(&tmp, path) {
        let _ = std::fs::remove_file(&tmp);
        return Err(e);
    }
    Ok(Written::Published)
}

/// [`write_atomic_cancellable`] with nothing to cancel it, for the tests that
/// are about something else. Production always has an abort flag to pass.
#[cfg(test)]
pub fn write_atomic(path: &Path, bytes: &[u8]) -> std::io::Result<()> {
    write_atomic_cancellable(path, bytes, &|| false).map(|_| ())
}

/// Read and decode a cache file under the default [`Limits`].
pub fn read(path: &Path, want_bars: usize, max_bars: usize) -> Result<PreFrames, CacheError> {
    read_with(path, want_bars, max_bars, Limits::default())
}

/// Read and decode under explicit limits.
///
/// The length is taken **from the open handle**, not from a separate `stat` of
/// the path, and the read is bounded on top of that. A length checked against
/// one file and then used to size a read of another is not a limit: between the
/// two calls the path can be replaced, or the file can grow, and the check
/// would have been made about something that is no longer there.
///
/// So: open, ask that handle how long it is, refuse it if the metadata already
/// exceeds the budget, and then read through a `Take` capped at one byte past
/// the budget. If that extra byte arrives, the file grew or was swapped after
/// its length was taken, and the read is refused having consumed
/// `max_file_bytes + 1` bytes and no more. **The overflow allowance is exactly
/// one byte** — never a second buffer's worth, and never however much the file
/// happens to have become.
pub fn read_with(
    path: &Path,
    want_bars: usize,
    max_bars: usize,
    limits: Limits,
) -> Result<PreFrames, CacheError> {
    let raw = read_bounded(path, limits)?;
    decode_with(&raw, want_bars, max_bars, limits)
}

/// Read a whole file into memory under `limits`, and no more than that.
///
/// Split out from [`read_with`] because the stereo sidecar is a different
/// format read from the same cache directory, and a second `read_to_end`
/// written next door is exactly how one of two readers ends up unbounded. The
/// bounding rule described on `read_with` lives here.
pub(crate) fn read_bounded(path: &Path, limits: Limits) -> Result<Vec<u8>, CacheError> {
    let mut f = std::fs::File::open(path).map_err(|_| CacheError::TooShort)?;
    let len = f.metadata().map_err(|_| CacheError::TooShort)?.len();
    if len > limits.max_file_bytes as u64 {
        return Err(CacheError::TooLarge {
            what: "file",
            got: len as usize,
            limit: limits.max_file_bytes,
        });
    }
    // A test may replace the file here, which is the whole reason the read
    // below is bounded rather than trusting the length above.
    fault::run(fault::Hook::AfterMetadata);

    let ceiling = limits.max_file_bytes.saturating_add(1);
    let reserve = (len as usize).saturating_add(1).min(ceiling);
    let mut raw = Vec::new();
    raw.try_reserve_exact(reserve)
        .map_err(|_| CacheError::OutOfMemory { bytes: reserve })?;
    Read::by_ref(&mut f)
        .take(ceiling as u64)
        .read_to_end(&mut raw)
        .map_err(|_| CacheError::TooShort)?;
    if raw.len() > limits.max_file_bytes {
        // `got` is what the read stopped at, which is a lower bound on the
        // file's real size — the point being that nothing larger was read.
        return Err(CacheError::TooLarge {
            what: "file",
            got: raw.len(),
            limit: limits.max_file_bytes,
        });
    }
    Ok(raw)
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    /// A directory of its own, removed on drop. Never the owner's real cache.
    struct TmpDir(std::path::PathBuf);

    impl TmpDir {
        fn new(tag: &str) -> Self {
            let stamp = std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .map(|d| d.as_nanos())
                .unwrap_or(0);
            let d = std::env::temp_dir()
                .join(format!("moosik_cache_test_{tag}_{}_{stamp:x}", std::process::id()));
            std::fs::create_dir_all(&d).unwrap();
            Self(d)
        }
        fn join(&self, name: &str) -> std::path::PathBuf {
            self.0.join(name)
        }
    }

    impl Drop for TmpDir {
        fn drop(&mut self) {
            let _ = std::fs::remove_dir_all(&self.0);
        }
    }

    const BARS: usize = 1024;

    fn rows(frames: usize, bars: usize) -> Vec<Vec<f32>> {
        (0..frames)
            .map(|f| {
                (0..bars)
                    .map(|b| {
                        let x = (f as f32 * 0.031 + b as f32 * 0.0117).sin() * 0.5 + 0.5;
                        x.clamp(0.0, 1.0)
                    })
                    .collect()
            })
            .collect()
    }

    /// The shapes a delta coder is most likely to get wrong.
    fn awkward(bars: usize) -> Vec<Vec<f32>> {
        vec![
            vec![0.0; bars],
            vec![1.0; bars],
            (0..bars).map(|b| if b % 2 == 0 { 0.0 } else { 1.0 }).collect(),
            (0..bars).map(|b| b as f32 / (bars.max(2) - 1) as f32).collect(),
            (0..bars).map(|b| 1.0 - b as f32 / (bars.max(2) - 1) as f32).collect(),
        ]
    }

    fn roundtrip(rows: &[Vec<f32>], bars: usize) -> PreFrames {
        let pf = PreFrames::from_analysis(rows);
        let bytes = pf.to_v4();
        decode(&bytes, bars, BARS).expect("v4 decode")
    }

    // ── the container ──────────────────────────────────────────────────────

    /// Two 4-bit values per byte, low nibble first — the encoder's former
    /// second pass.
    ///
    /// Test-only now: the encoder packs during its scan and the decoder reads
    /// nibbles in place, so nothing in production builds an unpacked plane. It
    /// survives here because the byte-identity test's reference encoder is the
    /// old encoder, and that one needed it.
    fn pack_nibbles(vals: &[u8]) -> Vec<u8> {
        let mut out = Vec::with_capacity(vals.len().div_ceil(2));
        let mut it = vals.chunks_exact(2);
        for c in &mut it {
            out.push((c[0] & 0x0F) | ((c[1] & 0x0F) << 4));
        }
        if let [last] = it.remainder() {
            out.push(last & 0x0F); // the spare nibble is written as zero
        }
        out
    }

    /// Codes chosen to exercise the delta-zigzag path rather than a signal.
    ///
    /// Includes both endpoints, both directions of a delta that wraps the 12-bit
    /// range, the largest jumps in each direction, and a run that is constant so
    /// the delta is zero.
    fn adversarial_codes(frames: usize, bars: usize) -> Vec<u16> {
        let mut v = Vec::with_capacity(frames * bars);
        for f in 0..frames {
            for b in 0..bars {
                let i = f * bars + b;
                v.push(match i % 9 {
                    0 => 0,                        // the floor
                    1 => V4_MAX_CODE,              // the ceiling: delta wraps up
                    2 => 0,                        // and straight back down
                    3 => V4_MAX_CODE / 2,
                    4 => V4_MAX_CODE / 2,          // a zero delta
                    5 => V4_MAX_CODE / 2 + 1,      // the smallest positive delta
                    6 => V4_MAX_CODE / 2 - 1,
                    7 => ((i * 2_654_435_761) % (V4_MAX_CODE as usize + 1)) as u16,
                    _ => V4_MAX_CODE,
                });
            }
        }
        v
    }

    /// The v4 encoder writes exactly the bytes the staged encoder wrote.
    ///
    /// The reference below is the shape `to_v4` had before it packed nibbles
    /// during the scan and compressed into the output: a low plane, a separate
    /// **unpacked** high-nibble plane, `pack_nibbles` over that, two
    /// `lz4_flex::compress` calls, and a copy of both blocks into the file.
    ///
    /// `lz4_flex::compress` and `compress_into` share one compression core and
    /// differ only in the sink, so identical output is expected by
    /// construction. **A source argument is not evidence**, so this compares the
    /// bytes over shapes chosen to break the packing: odd and even cell counts,
    /// odd and even bar counts, one frame, one bar, and code sequences that hit
    /// both endpoints and wrap the delta in both directions.
    #[test]
    fn the_v4_encoder_is_byte_identical_to_the_staged_form() {
        fn reference(pf: &PreFrames) -> Vec<u8> {
            let cells = pf.len() * pf.bars();
            let mut lo: Vec<u8> = Vec::with_capacity(cells);
            let mut hi: Vec<u8> = Vec::with_capacity(cells);
            for row in pf.codes.chunks_exact(pf.bars().max(1)) {
                let mut prev = 0i32;
                for &q in row {
                    let q = q.min(V4_MAX_CODE) as i32;
                    let d = wrap_signed(q - prev, V4_BITS);
                    prev = q;
                    let z = (((d << 1) ^ (d >> 31)) as u32) & (V4_MAX_CODE as u32);
                    lo.push((z & 0xFF) as u8);
                    hi.push((z >> 8) as u8);
                }
            }
            let lo_block = lz4_flex::compress(&lo);
            let hi_block = lz4_flex::compress(&pack_nibbles(&hi));

            let mut out = Vec::with_capacity(V4_HEADER + lo_block.len() + hi_block.len());
            out.extend_from_slice(&MAGIC_V4.to_le_bytes());
            out.extend_from_slice(&(pf.len() as u32).to_le_bytes());
            out.extend_from_slice(&(pf.bars() as u32).to_le_bytes());
            out.push(V4_BITS as u8);
            out.push(0);
            out.extend_from_slice(&0u16.to_le_bytes());
            out.extend_from_slice(&(lo_block.len() as u32).to_le_bytes());
            out.extend_from_slice(&(hi_block.len() as u32).to_le_bytes());
            out.extend_from_slice(&lo_block);
            out.extend_from_slice(&hi_block);
            out
        }

        let mut odd_cells = 0usize;
        let mut even_cells = 0usize;
        for &frames in &[1usize, 2, 3, 5, 8, 17, 33] {
            for &bars in &[1usize, 2, 3, 7, 8, 65, 253] {
                let pf = PreFrames::from_codes(
                    adversarial_codes(frames, bars), frames, bars, V4_MAX_CODE,
                );
                let got = pf.to_v4();
                let want = reference(&pf);
                assert_eq!(got, want, "{frames}x{bars}: v4 bytes changed");

                if (frames * bars) % 2 == 1 { odd_cells += 1 } else { even_cells += 1 }

                // And it is still readable as itself.
                let back = decode(&got, bars, BARS).expect("decode");
                assert_eq!(back, pf, "{frames}x{bars}: round trip");
            }
        }
        assert!(odd_cells > 0 && even_cells > 0,
                "setup: {odd_cells} odd and {even_cells} even cell counts");

        // Real magnitudes as well as adversarial codes.
        for bars in [1usize, 2, 3, 7, 64, 1023, 1024] {
            let pf = PreFrames::from_analysis(&awkward(bars));
            assert_eq!(pf.to_v4(), reference(&pf), "awkward bars={bars}");
        }
        let pf = PreFrames::from_analysis(&rows(97, 253));
        assert_eq!(pf.to_v4(), reference(&pf), "realistic shape");
    }

    #[test]
    fn v4_round_trips_every_awkward_shape() {
        // Bar counts that are and are not even, so the nibble plane's final
        // byte is padded in some cases and not others.
        for bars in [1usize, 2, 3, 7, 64, 1023, 1024] {
            let src = awkward(bars);
            let pf = PreFrames::from_analysis(&src);
            let back = roundtrip(&src, bars);
            assert_eq!(back.len(), src.len(), "bars={bars}");
            assert_eq!(back.bars(), bars);
            for f in 0..src.len() {
                assert_eq!(
                    back.row(f).unwrap(),
                    pf.row(f).unwrap(),
                    "bars={bars} row {f} did not survive"
                );
            }
        }
    }

    #[test]
    fn v4_round_trips_a_realistic_track() {
        let src = rows(97, 253); // neither dimension convenient
        let pf = PreFrames::from_analysis(&src);
        let back = roundtrip(&src, 253);
        assert_eq!(back, pf, "decoded frames differ from what was encoded");
    }

    #[test]
    fn the_quantiser_stays_inside_its_stated_bound() {
        // Across the whole range, not just at the ends.
        let mut worst = 0.0f32;
        for i in 0..=20_000 {
            let v = i as f32 / 20_000.0;
            let back = v_of(quantise(v, V4_MAX_CODE), V4_MAX_CODE);
            worst = worst.max((v - back).abs() * AXIS_DB);
        }
        assert!(
            worst <= V4_MAX_QUANT_ERROR_DB + 1e-4,
            "worst quantisation error {worst} dB exceeds the stated {V4_MAX_QUANT_ERROR_DB} dB"
        );
        // And the bound is the arithmetic one, not a rounded-up claim.
        assert!((V4_MAX_QUANT_ERROR_DB - 80.0 / (2.0 * 4095.0)).abs() < 1e-9);
    }

    fn v_of(code: u16, max: u16) -> f32 {
        code as f32 * (1.0 / max as f32)
    }

    #[test]
    fn the_endpoints_are_exact() {
        let pf = PreFrames::from_analysis(&[vec![0.0, 1.0]]);
        assert_eq!(pf.row_vec(0), vec![0.0, 1.0]);
        let back = roundtrip(&[vec![0.0, 1.0]], 2);
        assert_eq!(back.row_vec(0), vec![0.0, 1.0]);
    }

    #[test]
    fn out_of_range_and_non_finite_input_is_clamped_not_wrapped() {
        let src = vec![vec![-1.0f32, 2.0, f32::NAN, f32::INFINITY, 0.5]];
        let pf = PreFrames::from_analysis(&src);
        let got = pf.row_vec(0);
        assert_eq!(got[0], 0.0, "negative clamped to the floor");
        assert_eq!(got[1], 1.0, "above full scale clamped to the ceiling");
        assert_eq!(got[2], 0.0, "NaN is not a level");
        assert_eq!(got[3], 1.0, "infinity clamps rather than wrapping");
        assert!((got[4] - 0.5).abs() <= 1.0 / V4_MAX_CODE as f32);
    }

    // ── legacy ─────────────────────────────────────────────────────────────

    /// Build a real v2 file the way the shipped writer did.
    fn v2_blob(rows: &[Vec<f32>], bars: usize) -> Vec<u8> {
        let mut payload = Vec::new();
        for row in rows {
            for &v in row {
                payload.extend_from_slice(&((v * 65535.0).round() as u16).to_le_bytes());
            }
        }
        let mut blob = Vec::new();
        blob.extend_from_slice(&MAGIC_V2.to_le_bytes());
        blob.extend_from_slice(&(rows.len() as u32).to_le_bytes());
        blob.extend_from_slice(&(bars as u32).to_le_bytes());
        blob.extend_from_slice(&lz4_flex::compress_prepend_size(&payload));
        blob
    }

    /// Build a real v3 file the way the shipped writer did.
    fn v3_blob(rows: &[Vec<f32>], bars: usize) -> Vec<u8> {
        let count = rows.len() * bars;
        let mut hi = Vec::with_capacity(count);
        let mut lo = Vec::with_capacity(count);
        for row in rows {
            let mut prev: i32 = 0;
            for &v in row {
                let q = (v.clamp(0.0, 1.0) * 65535.0).round() as i32;
                let d = (q - prev) as i16 as i32;
                prev = q;
                let z = ((d << 1) ^ (d >> 31)) as u32 as u16;
                let [b0, b1] = z.to_le_bytes();
                lo.push(b0);
                hi.push(b1);
            }
        }
        let mut payload = hi;
        payload.append(&mut lo);
        let mut blob = Vec::new();
        blob.extend_from_slice(&MAGIC_V3.to_le_bytes());
        blob.extend_from_slice(&(rows.len() as u32).to_le_bytes());
        blob.extend_from_slice(&(bars as u32).to_le_bytes());
        blob.extend_from_slice(&lz4_flex::compress_prepend_size(&payload));
        blob
    }

    #[test]
    fn legacy_files_load_and_keep_their_own_precision() {
        let bars = 64;
        let src = rows(40, bars);
        for (name, blob) in [("v2", v2_blob(&src, bars)), ("v3", v3_blob(&src, bars))] {
            let pf = decode(&blob, bars, BARS).unwrap_or_else(|e| panic!("{name}: {e}"));
            assert_eq!(pf.max_code(), LEGACY_MAX_CODE, "{name} was requantised on load");
            let worst = src
                .iter()
                .enumerate()
                .flat_map(|(f, row)| {
                    let got = pf.row_vec(f);
                    row.iter().zip(got).map(|(a, b)| (a - b).abs()).collect::<Vec<_>>()
                })
                .fold(0.0f32, f32::max);
            // Half a 16-bit step, not half a 12-bit one. Opening an old cache
            // must not cost precision it already had.
            assert!(
                worst <= 1.0 / 65535.0 + 1e-7,
                "{name} lost {worst}, more than its own quantiser"
            );
        }
    }

    #[test]
    fn a_cache_for_another_bar_count_is_refused() {
        let blob = PreFrames::from_analysis(&rows(8, 32)).to_v4();
        assert_eq!(
            decode(&blob, 64, BARS),
            Err(CacheError::BarsMismatch { want: 64, got: 32 })
        );
        assert!(decode(&blob, 32, BARS).is_ok());
    }

    // ── malformed input ────────────────────────────────────────────────────

    #[test]
    fn malformed_input_is_refused_rather_than_guessed_at() {
        let good = PreFrames::from_analysis(&rows(8, 32)).to_v4();

        assert_eq!(decode(&[], 32, BARS), Err(CacheError::TooShort));
        assert_eq!(decode(&good[..11], 32, BARS), Err(CacheError::TooShort));

        let mut bad_magic = good.clone();
        bad_magic[0] ^= 0xFF;
        assert!(matches!(
            decode(&bad_magic, 32, BARS),
            Err(CacheError::BadMagic(_))
        ));

        // Dimensions beyond what this build will allocate for.
        for (frames, bars) in [(0u32, 32u32), (600_000, 32), (8, 0), (8, 99_999)] {
            let mut b = good.clone();
            b[4..8].copy_from_slice(&frames.to_le_bytes());
            b[8..12].copy_from_slice(&bars.to_le_bytes());
            assert!(
                matches!(decode(&b, bars as usize, BARS), Err(CacheError::BadDimensions { .. }))
                    || matches!(decode(&b, bars as usize, BARS), Err(CacheError::BarsMismatch { .. })),
                "accepted {frames}×{bars}"
            );
        }

        // A depth this build does not write.
        let mut bits = good.clone();
        bits[12] = 16;
        assert_eq!(decode(&bits, 32, BARS), Err(CacheError::UnsupportedBits(16)));

        // Reserved bytes set: a newer writer, so do not guess.
        for i in 13..16 {
            let mut r = good.clone();
            r[i] = 1;
            assert_eq!(decode(&r, 32, BARS), Err(CacheError::Reserved), "byte {i}");
        }

        // Declared block lengths that do not add up.
        let mut len = good.clone();
        len[16..20].copy_from_slice(&0xFFFF_FFFFu32.to_le_bytes());
        assert!(matches!(
            decode(&len, 32, BARS),
            Err(CacheError::LengthMismatch { .. })
        ));

        // Truncated body, header intact.
        let cut = &good[..good.len() - 3];
        assert!(matches!(
            decode(cut, 32, BARS),
            Err(CacheError::LengthMismatch { .. })
        ));

        // Corrupt compressed bytes.
        let mut corrupt = good.clone();
        let n = corrupt.len();
        for b in &mut corrupt[V4_HEADER..n.min(V4_HEADER + 40)] {
            *b ^= 0x5A;
        }
        assert!(
            decode(&corrupt, 32, BARS).is_err(),
            "a corrupted payload decoded as if it were fine"
        );
    }

    /// The case Codex demonstrated: a tiny file declaring six million cells.
    ///
    /// The old test asserted only that *some* error came back, and one did —
    /// after `lz4_flex` had been handed a 6.4 MB output buffer to fill. The
    /// point of a limit is that the refusal happens first.
    #[test]
    fn an_oversized_declaration_is_refused_before_the_allocation() {
        let src = rows(4, 16);
        let mut blob = PreFrames::from_analysis(&src).to_v4();
        blob[4..8].copy_from_slice(&400_000u32.to_le_bytes());

        // Under the shipped limits the expansion bound catches it: 6.4 M cells
        // cannot come out of the handful of bytes the file actually contains.
        match decode(&blob, 16, BARS) {
            Err(CacheError::TooLarge { what, got, limit }) => {
                assert_eq!(what, "low plane");
                assert_eq!(got, 400_000 * 16);
                assert!(limit < got, "the bound must be below the claim");
            }
            other => panic!("expected a size refusal, got {other:?}"),
        }

        // And with a cell budget smaller than the claim, it is refused earlier
        // still — before the expansion bound is even reached. A tiny limit is
        // how this is observed without allocating anything large.
        let tiny = Limits { max_cells: 1_000, ..Limits::default() };
        assert_eq!(
            decode_with(&blob, 16, BARS, tiny),
            Err(CacheError::TooLarge {
                what: "decoded cells",
                got: 400_000 * 16,
                limit: 1_000
            })
        );
    }

    /// Each limit refuses, and refuses at its own stage.
    #[test]
    fn every_limit_is_enforced_before_the_allocation_it_governs() {
        let blob = PreFrames::from_analysis(&rows(40, 64)).to_v4();

        // File bytes, before the header is even parsed.
        let by_file = Limits { max_file_bytes: 8, ..Limits::default() };
        assert_eq!(
            decode_with(&blob, 64, BARS, by_file),
            Err(CacheError::TooLarge {
                what: "file",
                got: blob.len(),
                limit: 8
            })
        );

        // Decoded cells, after the header and before any decompression.
        let by_cells = Limits { max_cells: 100, ..Limits::default() };
        assert_eq!(
            decode_with(&blob, 64, BARS, by_cells),
            Err(CacheError::TooLarge {
                what: "decoded cells",
                got: 40 * 64,
                limit: 100
            })
        );

        // Expansion, once the block lengths are known.
        let by_expansion = Limits { max_expansion: 1, ..Limits::default() };
        assert!(
            matches!(
                decode_with(&blob, 64, BARS, by_expansion),
                Err(CacheError::TooLarge { what: "low plane", .. })
            ),
            "an expansion of 1 should refuse any compressed plane"
        );

        // And the same file passes under the shipped limits.
        assert!(decode(&blob, 64, BARS).is_ok());
    }

    /// A real, digitally silent track compresses enormously. The expansion
    /// bound has to be loose enough not to refuse it.
    #[test]
    fn the_expansion_bound_does_not_refuse_real_silence() {
        let silent = vec![vec![0.0f32; 1024]; 4_000];
        let pf = PreFrames::from_analysis(&silent);
        let blob = pf.to_v4();
        let ratio = (4_000 * 1024) as f64 / blob.len() as f64;
        assert!(ratio > 100.0, "setup: silence should compress hard, got {ratio:.0}:1");
        assert_eq!(
            decode(&blob, 1024, BARS).expect("silence was refused"),
            pf,
            "a legitimate silent track must survive the expansion bound"
        );
    }

    /// A file too large to read is refused from its metadata, without being
    /// read into memory.
    #[test]
    fn an_oversized_file_is_refused_without_reading_it() {
        let d = TmpDir::new("big_file");
        let p = d.join("big.spectrumcache");
        write_atomic(&p, &PreFrames::from_analysis(&rows(64, 128)).to_v4()).unwrap();
        let on_disk = std::fs::metadata(&p).unwrap().len() as usize;
        let tiny = Limits { max_file_bytes: on_disk - 1, ..Limits::default() };
        assert_eq!(
            read_with(&p, 128, BARS, tiny),
            Err(CacheError::TooLarge {
                what: "file",
                got: on_disk,
                limit: on_disk - 1
            })
        );
        assert!(read_with(&p, 128, BARS, Limits::default()).is_ok());
    }

    // ── files ──────────────────────────────────────────────────────────────

    #[test]
    fn a_written_cache_reads_back_identically() {
        let d = TmpDir::new("roundtrip");
        let p = d.join("a.spectrumcache");
        let pf = PreFrames::from_analysis(&rows(64, 128));
        write_atomic(&p, &pf.to_v4()).unwrap();
        let back = read(&p, 128, BARS).unwrap();
        assert_eq!(back, pf);
    }

    /// Deterministic failure at each stage, asserting the destination's exact
    /// previous bytes.
    ///
    /// The previous version of this test blocked a path the timestamped
    /// temporary could never have used, so the write simply succeeded and the
    /// assertion — that the result decoded — held for the wrong reason.
    #[test]
    fn a_failed_write_leaves_the_previous_bytes_exactly() {
        for at in [fault::At::MidWrite, fault::At::BeforePublish] {
            let d = TmpDir::new("failed_write");
            let p = d.join("keep.spectrumcache");
            let original = PreFrames::from_analysis(&rows(16, 32)).to_v4();
            write_atomic(&p, &original).unwrap();
            let before = std::fs::read(&p).unwrap();
            assert_eq!(before, original);

            let replacement = PreFrames::from_analysis(&rows(8, 32)).to_v4();
            assert_ne!(replacement, original, "setup: the two must differ");

            fault::arm(at);
            let got = write_atomic(&p, &replacement);
            assert!(got.is_err(), "{at:?}: the injected failure did not surface");

            let after = std::fs::read(&p).unwrap();
            assert_eq!(
                after, before,
                "{at:?}: the destination changed despite a failed write"
            );
            assert!(strays(&d).is_empty(), "{at:?}: left {:?}", strays(&d));
        }
    }

    /// Cancellation observed while the write is in flight publishes nothing.
    ///
    /// The flag is the real one the production path uses, and it is flipped
    /// after the temporary is on disk — not before the call, and not by
    /// standing an injected I/O error in for a cancellation. The hook asserts
    /// the temporary exists at the moment it fires, so the test also proves
    /// *when* it cancelled.
    #[test]
    fn cancelling_while_the_write_is_in_flight_publishes_nothing() {
        for existing in [false, true] {
            let d = TmpDir::new("cancel_inflight");
            let p = d.join("c.spectrumcache");
            let original = PreFrames::from_analysis(&rows(16, 32)).to_v4();
            if existing {
                write_atomic(&p, &original).unwrap();
            }

            let flag = std::sync::Arc::new(std::sync::atomic::AtomicBool::new(false));
            let set = flag.clone();
            let dir = d.0.clone();
            fault::on(fault::Hook::TempWritten, move || {
                assert!(
                    !strays_in(&dir).is_empty(),
                    "setup: the temporary should be on disk by now"
                );
                set.store(true, std::sync::atomic::Ordering::Relaxed);
            });

            let replacement = PreFrames::from_analysis(&rows(8, 32)).to_v4();
            assert_ne!(replacement, original, "setup: the two must differ");
            let seen = flag.clone();
            let got = write_atomic_cancellable(&p, &replacement, &move || {
                seen.load(std::sync::atomic::Ordering::Relaxed)
            });

            assert_eq!(got.unwrap(), Written::Cancelled, "existing={existing}");
            assert!(
                flag.load(std::sync::atomic::Ordering::Relaxed),
                "setup: the hook never ran, so nothing was cancelled mid-flight"
            );
            if existing {
                assert_eq!(
                    std::fs::read(&p).unwrap(),
                    original,
                    "a cancelled write changed the destination"
                );
            } else {
                assert!(!p.exists(), "a cancelled write created a file");
            }
            assert!(strays(&d).is_empty(), "left {:?}", strays(&d));
        }
    }

    /// Cancellation already true on entry: nothing is created at all.
    #[test]
    fn cancelling_before_the_write_starts_creates_nothing() {
        let d = TmpDir::new("cancel_early");
        let bytes = PreFrames::from_analysis(&rows(8, 16)).to_v4();

        let fresh = d.join("fresh.spectrumcache");
        assert_eq!(
            write_atomic_cancellable(&fresh, &bytes, &|| true).unwrap(),
            Written::Cancelled
        );
        assert!(!fresh.exists(), "a cancelled write created a file");

        let held = d.join("held.spectrumcache");
        let original = PreFrames::from_analysis(&rows(16, 16)).to_v4();
        write_atomic(&held, &original).unwrap();
        assert_eq!(
            write_atomic_cancellable(&held, &bytes, &|| true).unwrap(),
            Written::Cancelled
        );
        assert_eq!(std::fs::read(&held).unwrap(), original);
        assert!(strays(&d).is_empty());
    }

    /// A predicate that never fires publishes, and says so.
    #[test]
    fn an_uncancelled_write_reports_publication() {
        let d = TmpDir::new("published");
        let p = d.join("p.spectrumcache");
        let pf = PreFrames::from_analysis(&rows(8, 16));
        assert_eq!(
            write_atomic_cancellable(&p, &pf.to_v4(), &|| false).unwrap(),
            Written::Published
        );
        assert_eq!(read(&p, 16, BARS).unwrap(), pf);
    }

    /// A file replaced with a much larger one *after* its length is taken is
    /// refused, and refused without reading what it became.
    #[test]
    fn a_file_that_grows_after_its_length_is_taken_is_refused() {
        let d = TmpDir::new("grew");
        let p = d.join("g.spectrumcache");
        let small = PreFrames::from_analysis(&rows(4, 16)).to_v4();
        write_atomic(&p, &small).unwrap();

        let budget = small.len() + 8;
        let swap = p.clone();
        fault::on(fault::Hook::AfterMetadata, move || {
            std::fs::write(&swap, vec![0u8; budget * 16]).expect("could not swap the file");
        });

        let limits = Limits { max_file_bytes: budget, ..Limits::default() };
        match read_with(&p, 16, BARS, limits) {
            Err(CacheError::TooLarge { what, got, limit }) => {
                assert_eq!(what, "file");
                assert_eq!(limit, budget);
                assert_eq!(
                    got,
                    budget + 1,
                    "the read should stop one byte past the budget, not consume the file"
                );
            }
            other => panic!("expected a refusal, got {other:?}"),
        }
    }

    /// The success path still replaces, and replaces completely.
    #[test]
    fn a_successful_write_replaces_the_previous_file() {
        let d = TmpDir::new("replace");
        let p = d.join("r.spectrumcache");
        let first = PreFrames::from_analysis(&rows(16, 32));
        write_atomic(&p, &first.to_v4()).unwrap();
        assert_eq!(read(&p, 32, BARS).unwrap(), first);

        let second = PreFrames::from_analysis(&rows(9, 32));
        write_atomic(&p, &second.to_v4()).unwrap();
        let back = read(&p, 32, BARS).unwrap();
        assert_eq!(back, second, "the replacement did not take");
        assert_ne!(back, first);
        assert!(strays(&d).is_empty());
    }

    fn strays(d: &TmpDir) -> Vec<String> {
        strays_in(&d.0)
    }

    fn strays_in(dir: &std::path::Path) -> Vec<String> {
        std::fs::read_dir(dir)
            .unwrap()
            .filter_map(|e| e.ok())
            .map(|e| e.file_name().to_string_lossy().to_string())
            .filter(|n| n.contains("tmp-"))
            .collect()
    }

    #[test]
    fn no_temporary_files_are_left_behind() {
        let d = TmpDir::new("no_temps");
        let p = d.join("t.spectrumcache");
        for _ in 0..3 {
            write_atomic(&p, &PreFrames::from_analysis(&rows(8, 16)).to_v4()).unwrap();
        }
        let strays: Vec<_> = std::fs::read_dir(&d.0)
            .unwrap()
            .filter_map(|e| e.ok())
            .map(|e| e.file_name().to_string_lossy().to_string())
            .filter(|n| n.contains("tmp-"))
            .collect();
        assert!(strays.is_empty(), "left {strays:?} behind");
    }

    // ── access patterns ────────────────────────────────────────────────────

    #[test]
    fn rows_are_addressable_in_any_order() {
        let src = rows(200, 64);
        let pf = PreFrames::from_analysis(&src);
        // First, last, and a deterministic scatter including backward seeks.
        let mut order: Vec<usize> = vec![0, 199, 1, 198, 100, 3, 197, 50, 0, 199];
        order.extend((0..200).rev().step_by(7));
        for f in order {
            let want: Vec<f32> = src[f]
                .iter()
                .map(|&v| v_of(quantise(v, V4_MAX_CODE), V4_MAX_CODE))
                .collect();
            assert_eq!(pf.row_vec(f), want, "row {f} read out of order");
        }
        assert!(pf.row(200).is_none(), "past the end must be None");
    }

    #[test]
    fn the_peak_scan_matches_a_row_by_row_scan() {
        let src = rows(120, 96);
        let pf = PreFrames::from_analysis(&src);
        let mut want = vec![0.0f32; 96];
        for f in 0..pf.len() {
            for (w, v) in want.iter_mut().zip(pf.row_vec(f)) {
                if v > *w {
                    *w = v;
                }
            }
        }
        assert_eq!(pf.peak_per_bar(), want);
    }

    #[test]
    fn a_fresh_result_and_a_reloaded_one_agree_exactly() {
        let d = TmpDir::new("fresh_vs_reload");
        let p = d.join("x.spectrumcache");
        let src = rows(64, 128);
        let fresh = PreFrames::from_analysis(&src);
        write_atomic(&p, &fresh.to_v4()).unwrap();
        let reloaded = read(&p, 128, BARS).unwrap();
        assert_eq!(fresh, reloaded, "a reload differs from what was just computed");
        for f in 0..fresh.len() {
            assert_eq!(fresh.row_vec(f), reloaded.row_vec(f), "row {f}");
        }
    }

    #[test]
    fn payload_size_is_two_bytes_a_cell() {
        let pf = PreFrames::from_analysis(&rows(1000, 1024));
        assert_eq!(pf.payload_bytes(), 1000 * 1024 * 2);
        assert!(
            pf.capacity_bytes() >= pf.payload_bytes(),
            "capacity cannot be below the payload it holds"
        );
    }

    /// Clearing must give the memory back, not merely report zero.
    ///
    /// `Vec::clear` keeps the buffer, so a cache that had been cleared still
    /// held 87 MiB for a long track while `len()` said nothing was there. A
    /// test on `is_empty()` or `len() == 0` passes in both worlds, which is why
    /// this one asserts on capacity.
    #[test]
    fn clearing_releases_the_backing_allocation() {
        let mut pf = PreFrames::from_analysis(&rows(2_000, 512));
        let held = pf.capacity_bytes();
        assert!(held >= 2_000 * 512 * 2, "setup: expected a real allocation");
        assert_eq!(pf.payload_bytes(), 2_000 * 512 * 2);

        pf.clear();

        assert_eq!(pf.capacity_bytes(), 0, "clearing kept {held} bytes");
        assert_eq!(pf.payload_bytes(), 0);
        assert!(pf.is_empty());
        assert_eq!(pf.len(), 0);
        assert!(pf.row(0).is_none());
        // And it is still usable afterwards.
        pf = PreFrames::from_analysis(&rows(4, 8));
        assert_eq!(pf.len(), 4);
    }
}

// ---------------------------------------------------------------------------
// Benchmark
// ---------------------------------------------------------------------------

/// Numbers for the Phase A+B handoff, on a fixture the shape of the longest
/// track in the study corpus: 44 645 frames × 1024 bars, which is 248 seconds
/// at 180 fps.
///
/// Run:
///   cargo test --release --locked --all-features cache::bench -- --ignored --nocapture
#[cfg(test)]
mod bench {
    use super::tests_support::*;
    use super::*;
    use std::time::Instant;

    const FRAMES: usize = 44_645;
    const BARS: usize = 1024;

    fn mib(b: usize) -> f64 {
        b as f64 / (1024.0 * 1024.0)
    }

    #[test]
    #[ignore]
    fn phase_ab_numbers() {
        println!("\nfixture: {FRAMES} frames × {BARS} bars = {} cells\n", FRAMES * BARS);

        // ── the baseline representation ─────────────────────────────────────
        let t = Instant::now();
        let baseline = bench_rows(FRAMES, BARS);
        let build = t.elapsed();
        let baseline_bytes = baseline.len() * std::mem::size_of::<Vec<f32>>()
            + baseline.iter().map(|r| r.capacity() * 4).sum::<usize>();
        println!(
            "  baseline Vec<Vec<f32>>      {:>9.1} MiB resident, {} allocations, built in {:.2}s",
            mib(baseline_bytes),
            baseline.len() + 1,
            build.as_secs_f64()
        );

        // ── convert ─────────────────────────────────────────────────────────
        let t = Instant::now();
        let pf = PreFrames::from_analysis(&baseline);
        let convert = t.elapsed();
        println!(
            "  PreFrames                   {:>9.1} MiB resident, 1 allocation, converted in {:.2}s",
            mib(pf.capacity_bytes()),
            convert.as_secs_f64()
        );
        println!(
            "  sum of the two payloads     {:>9.1} MiB  (arithmetic, not a measured process\n             {:>32}peak: it counts these two allocations only)",
            mib(baseline_bytes + pf.capacity_bytes()),
            ""
        );
        println!(
            "  resident saving             {:>9.1} MiB  ({:+.0}%)",
            mib(baseline_bytes) - mib(pf.capacity_bytes()),
            (pf.capacity_bytes() as f64 / baseline_bytes as f64 - 1.0) * 100.0
        );

        // ── on disk ─────────────────────────────────────────────────────────
        let t = Instant::now();
        let v4 = pf.to_v4();
        let enc = t.elapsed();
        let v3 = v3_blob_bench(&baseline, BARS);
        println!(
            "\n  --- on disk, smooth fixture (flatters v3: nothing in the low bits) ---"
        );
        println!("  v3 (16-bit)                 {:>9.1} MiB", mib(v3.len()));
        println!(
            "  v4 (12-bit)                 {:>9.1} MiB  ({:+.0}%), encoded in {:.2}s",
            mib(v4.len()),
            (v4.len() as f64 / v3.len() as f64 - 1.0) * 100.0,
            enc.as_secs_f64()
        );
        drop(baseline);

        // The same shape with a low-bit noise floor, which is what real
        // analyser output has and what the corpus measurement was made on.
        for amp in [0.0005f32, 0.002] {
            let noisy = bench_rows_with_noise(FRAMES, BARS, amp);
            let n3 = v3_blob_bench(&noisy, BARS).len();
            let n4 = PreFrames::from_analysis(&noisy).to_v4().len();
            drop(noisy);
            println!(
                "\n  --- on disk, noise floor {amp} (about {:.0} 16-bit steps) ---",
                amp * 65535.0
            );
            println!("  v3 (16-bit)                 {:>9.1} MiB", mib(n3));
            println!(
                "  v4 (12-bit)                 {:>9.1} MiB  ({:+.0}%)",
                mib(n4),
                (n4 as f64 / n3 as f64 - 1.0) * 100.0
            );
        }
        println!(
            "\n  The corpus figure -- 608.1 MiB v3 against 429.9 MiB packed 12-bit over\n               11 real tracks, -29% -- remains the one to quote. A synthetic fixture\n               can be made to show anything between these."
        );

        let t = Instant::now();
        let back = decode(&v4, BARS, BARS).expect("decode");
        let dec = t.elapsed();
        println!("  decode whole file           {:>9.2}s", dec.as_secs_f64());
        assert_eq!(back, pf);

        // ── row access ──────────────────────────────────────────────────────
        let mut sink = 0.0f32;
        let t = Instant::now();
        for f in 0..pf.len() {
            let row = pf.row(f).unwrap();
            sink += row[0] as f32 + row[BARS - 1] as f32;
        }
        let seq = t.elapsed();
        println!(
            "\n  --- row lookup only, not what a frame costs ---
  sequential  {:>9.1} ns/row over {} rows",
            seq.as_secs_f64() * 1e9 / pf.len() as f64,
            pf.len()
        );

        // A deterministic scatter, so the figure is a random-access one rather
        // than a prefetched walk.
        let mut idx = 0usize;
        let t = Instant::now();
        for i in 0..pf.len() {
            idx = (idx * 1_103_515_245 + 12_345 + i) % pf.len();
            let row = pf.row(idx).unwrap();
            sink += row[0] as f32;
        }
        let rand = t.elapsed();
        println!(
            "  random      {:>9.1} ns/row over {} rows",
            rand.as_secs_f64() * 1e9 / pf.len() as f64,
            pf.len()
        );

        // The catch-up figure that stood here timed a conversion loop, called
        // it a tick, and drew a percentage of the frame budget from it. Neither
        // the measurement nor the claim survived review. The real consumer --
        // smoothing, peaks and the waterfall -- is benchmarked against the
        // previous storage, on the same data, in
        // `spectrum::preprocess_consumer_bench`.

        println!("\n  (sink {sink:.3}, so nothing above is optimised away)\n");
    }
}

/// Fixture builders the benchmark shares with the tests.
#[cfg(test)]
mod tests_support {
    /// Rows with realistic structure: a slow spectral tilt plus per-frame
    /// motion, so the delta coder sees something like real material rather than
    /// noise or a constant.
    pub fn bench_rows(frames: usize, bars: usize) -> Vec<Vec<f32>> {
        bench_rows_with_noise(frames, bars, 0.0)
    }

    /// `noise` is the amplitude of a deterministic low-bit dither.
    ///
    /// It matters more than it looks. A smooth fixture is already almost free
    /// to delta-code, so dropping four bits from it saves nothing and the
    /// on-disk comparison flatters v3. Real analyser output is not smooth: its
    /// low bits are quantisation and estimator noise, which is exactly what a
    /// 12-bit grid discards and exactly where the measured saving came from.
    /// The benchmark therefore reports both, and neither is a substitute for
    /// the corpus figure.
    pub fn bench_rows_with_noise(frames: usize, bars: usize, noise: f32) -> Vec<Vec<f32>> {
        let mut seed = 0x2545_F491_4F6C_DD1Du64;
        (0..frames)
            .map(|f| {
                let t = f as f32 * 0.0007;
                (0..bars)
                    .map(|b| {
                        let x = b as f32 / bars as f32;
                        let tilt = 1.0 - x * 0.8;
                        let motion = (t + x * 9.0).sin() * 0.12;
                        seed ^= seed << 13;
                        seed ^= seed >> 7;
                        seed ^= seed << 17;
                        let n = ((seed >> 40) as f32 / 8_388_608.0) - 1.0;
                        (tilt * 0.7 + motion + n * noise).clamp(0.0, 1.0)
                    })
                    .collect()
            })
            .collect()
    }

    /// The v3 writer, verbatim, so the on-disk comparison is against what the
    /// shipped format actually produced rather than an estimate.
    pub fn v3_blob_bench(rows: &[Vec<f32>], bars: usize) -> Vec<u8> {
        let count = rows.len() * bars;
        let mut hi = Vec::with_capacity(count);
        let mut lo = Vec::with_capacity(count);
        for row in rows {
            let mut prev: i32 = 0;
            for &v in row {
                let q = (v.clamp(0.0, 1.0) * 65535.0).round() as i32;
                let d = (q - prev) as i16 as i32;
                prev = q;
                let z = ((d << 1) ^ (d >> 31)) as u32 as u16;
                let [b0, b1] = z.to_le_bytes();
                lo.push(b0);
                hi.push(b1);
            }
        }
        let mut payload = hi;
        payload.append(&mut lo);
        let mut blob = Vec::new();
        blob.extend_from_slice(&0x4D53_5033u32.to_le_bytes());
        blob.extend_from_slice(&(rows.len() as u32).to_le_bytes());
        blob.extend_from_slice(&(bars as u32).to_le_bytes());
        blob.extend_from_slice(&lz4_flex::compress_prepend_size(&payload));
        blob
    }
}
