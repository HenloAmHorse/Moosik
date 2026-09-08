//! TEMPORARY: the storage experiment, corrected to decode from stored bytes.
//!
//! Not product code. Sits beside `aslt_mono_study.rs`, which is preserved as
//! run; this file replaces the part of it that was measuring the wrong thing.
//!
//! # What was wrong
//!
//! The first study predicted mono as `(|L| + |R|) / 2` from the **raw** analyser
//! magnitudes and then scored the reconstruction against those same raw values.
//! A decoder has no raw magnitudes. It has the bytes on disk, and those have
//! been through a quantiser *and* an −80 dB clamp. So the residual it measured
//! was the error of a predictor nothing can run, and the reconstruction it
//! scored was one nothing can perform.
//!
//! Here the predictor is a single function, [`predict_mono_db`], and it is
//! reached only from decoded plane values. The encoder calls it on what it is
//! about to store, never on what it was given.
//!
//! # Floor and saturation
//!
//! A stored plane value is a level on a fixed 80 dB axis, and both ends of that
//! axis are lossy in a way quantisation is not:
//!
//! * **Floor.** 0 means "at or below −80 dB". The decoder reads it as exactly
//!   −80 dB because that is the only thing it can do, but the true value may be
//!   anywhere below. Two bands stored as 0 can have been 20 dB apart.
//! * **Ceiling.** Full scale means "at or above 0 dB", read as exactly 0 dB.
//!
//! What this does to the predictor is worth stating precisely, because the
//! obvious guess is wrong. Two channels both stored at the floor predict −80 dB,
//! not −74: the prediction is a *mean*, and the mean of two equal levels is that
//! level. Clamping both inputs and the truth the same way keeps them consistent,
//! so the common case costs nothing.
//!
//! The floor bites in the asymmetric case, and it bites upward. A channel that
//! is truly far below −80 dB is read as exactly −80 and so contributes far more
//! to the mean than it should. With L at −74 dB and R actually silent, the true
//! mono is −80.0 dB and stores at the floor, while the prediction from the
//! stored pair (−74, −80) is −76.5 dB — 3.5 dB too loud. The residual has to
//! carry that, and it is a genuine information loss rather than a rounding
//! error: nothing in the stored bytes distinguishes "−80 dB" from "silent".
//!
//! The residual is measured against the true mono's **stored** value, since that
//! is what a mono-plane format would have held, so the comparison is like for
//! like.

use super::*;

// ---------------------------------------------------------------------------
// The 80 dB axis
// ---------------------------------------------------------------------------

/// Span of the cache's level axis, in dB. A stored value of 0 is `-AXIS_DB`.
pub const AXIS_DB: f32 = 80.0;

/// dB for a decoded plane value in `0.0..=1.0`.
#[inline]
pub fn v_to_db(v: f32) -> f32 {
    v.clamp(0.0, 1.0) * AXIS_DB - AXIS_DB
}

/// Plane value for a level in dB, clamped at both ends of the axis.
#[inline]
pub fn db_to_v(db: f32) -> f32 {
    ((db + AXIS_DB) / AXIS_DB).clamp(0.0, 1.0)
}

/// Mono level predicted from two **stored** channel levels, in dB.
///
/// The one predictor. The encoder runs it on the values it is about to write
/// and the decoder runs it on the values it has read, so the residual is an
/// error against a number both sides agree on exactly.
///
/// The dB reference cancels: converting each level to a linear magnitude
/// divides by it and converting the mean back multiplies by it again. So this
/// needs no reference, and a superlet plane and an FFT plane go through the
/// identical function despite their different scaling.
///
/// Both inputs are levels on the axis, so both have already been floored. See
/// the module note: two floored channels predict −74 dB, not −80.
#[inline]
pub fn predict_mono_db(l_db: f32, r_db: f32) -> f32 {
    let p = (10f32.powf(l_db / 20.0) + 10f32.powf(r_db / 20.0)) * 0.5;
    if p <= 0.0 {
        return -AXIS_DB;
    }
    (20.0 * p.log10()).clamp(-AXIS_DB, 0.0)
}

// ---------------------------------------------------------------------------
// Plane codec — quantise, pack, compress, and the exact inverse of each
// ---------------------------------------------------------------------------

/// One magnitude plane at a chosen bit depth.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct PlaneCodec {
    pub bits: u32,
}

impl PlaneCodec {
    pub fn new(bits: u32) -> Self {
        assert!((1..=16).contains(&bits), "bit depth {bits} out of range");
        Self { bits }
    }

    #[inline]
    pub fn levels(&self) -> f32 {
        ((1u32 << self.bits) - 1) as f32
    }

    /// Plane value to stored code.
    #[inline]
    pub fn quantise(&self, v: f32) -> u16 {
        (v.clamp(0.0, 1.0) * self.levels()).round() as u16
    }

    /// Stored code back to a plane value. The exact inverse of the rounding,
    /// so `dequantise(quantise(v))` is `v` to within half a step and never
    /// leaves `0.0..=1.0`.
    #[inline]
    pub fn dequantise(&self, q: u16) -> f32 {
        (q as f32 / self.levels()).clamp(0.0, 1.0)
    }

    /// Worst-case level error this depth can introduce, in dB.
    pub fn max_step_error_db(&self) -> f32 {
        AXIS_DB / self.levels() * 0.5
    }
}

// ---------------------------------------------------------------------------
// Framing: several LZ4 blocks in one buffer, each recoverable
// ---------------------------------------------------------------------------

/// Concatenate blocks with an explicit compressed length each.
///
/// `lz4_flex::compress_prepend_size` writes the *uncompressed* size, which does
/// not say where the block ends, so two of them cannot be concatenated and read
/// back. This adds the length that is actually needed.
fn frame(blocks: &[Vec<u8>]) -> Vec<u8> {
    let mut out = Vec::new();
    for b in blocks {
        out.extend_from_slice(&(b.len() as u32).to_le_bytes());
        out.extend_from_slice(b);
    }
    out
}

fn unframe(bytes: &[u8]) -> Option<Vec<&[u8]>> {
    let mut out = Vec::new();
    let mut at = 0usize;
    while at < bytes.len() {
        if at + 4 > bytes.len() {
            return None;
        }
        let n = u32::from_le_bytes(bytes[at..at + 4].try_into().ok()?) as usize;
        at += 4;
        if at + n > bytes.len() {
            return None;
        }
        out.push(&bytes[at..at + n]);
        at += n;
    }
    Some(out)
}

// ---------------------------------------------------------------------------
// Bit packing, both ways
// ---------------------------------------------------------------------------

/// Pack `bits`-wide values LSB-first.
pub fn pack(vals: &[u16], bits: u32) -> Vec<u8> {
    let mask = if bits >= 16 { 0xFFFFu32 } else { (1u32 << bits) - 1 };
    let mut out = Vec::with_capacity(vals.len() * bits as usize / 8 + 2);
    let mut acc: u64 = 0;
    let mut have: u32 = 0;
    for &v in vals {
        acc |= ((v as u32 & mask) as u64) << have;
        have += bits;
        while have >= 8 {
            out.push((acc & 0xFF) as u8);
            acc >>= 8;
            have -= 8;
        }
    }
    if have > 0 {
        out.push((acc & 0xFF) as u8);
    }
    out
}

/// The exact inverse of [`pack`], for a known count.
///
/// `n` is required rather than derived: the last byte of a packed plane is
/// padded whenever `n * bits` is not a multiple of 8, so the byte length alone
/// over-reports the count. Getting this wrong is how a row length that is not
/// byte-aligned silently grows a trailing value.
pub fn unpack(bytes: &[u8], bits: u32, n: usize) -> Option<Vec<u16>> {
    let mask = if bits >= 16 { 0xFFFFu32 } else { (1u32 << bits) - 1 };
    let needed = (n * bits as usize).div_ceil(8);
    if bytes.len() < needed {
        return None;
    }
    let mut out = Vec::with_capacity(n);
    let mut acc: u64 = 0;
    let mut have: u32 = 0;
    let mut at = 0usize;
    for _ in 0..n {
        while have < bits {
            let b = if at < bytes.len() { bytes[at] } else { 0 };
            at += 1;
            acc |= (b as u64) << have;
            have += 8;
        }
        out.push((acc as u32 & mask) as u16);
        acc >>= bits;
        have -= bits;
    }
    Some(out)
}

// ---------------------------------------------------------------------------
// A whole plane: delta across frequency, byte-plane split, LZ4 — and back
// ---------------------------------------------------------------------------

/// Wrap a difference into a signed `bits`-wide value, as v3 wraps into `i16`.
#[inline]
fn wrap_signed(d: i32, bits: u32) -> i32 {
    let shift = 32 - bits;
    (d << shift) >> shift
}

/// Encode a plane: zigzag delta across frequency within each row, then split
/// into a low-byte plane and a high-bits plane, each LZ4-compressed.
///
/// The split is what v3 does and why it beats plain LZ4: interleaved, the noisy
/// low bits sit between every pair of compressible high bits and stop the coder
/// finding a run at all.
pub fn encode_plane(codes: &[u16], bits: u32, n_bars: usize) -> Vec<u8> {
    let n_bars = n_bars.max(1);
    let mut lo: Vec<u16> = Vec::with_capacity(codes.len());
    let mut hi: Vec<u16> = Vec::with_capacity(codes.len());
    for row in codes.chunks(n_bars) {
        let mut prev = 0i32;
        for &v in row {
            let d = wrap_signed(v as i32 - prev, bits);
            prev = v as i32;
            let z = ((d << 1) ^ (d >> 31)) as u32
                & if bits >= 16 { 0xFFFF } else { (1u32 << bits) - 1 };
            lo.push((z & 0xFF) as u16);
            hi.push((z >> 8) as u16);
        }
    }
    let hi_bits = bits.saturating_sub(8);
    let mut blocks = vec![lz4_flex::compress_prepend_size(&pack(&lo, 8.min(bits)))];
    if hi_bits > 0 {
        blocks.push(lz4_flex::compress_prepend_size(&pack(&hi, hi_bits)));
    }
    frame(&blocks)
}

/// The exact inverse of [`encode_plane`].
pub fn decode_plane(bytes: &[u8], bits: u32, n_bars: usize, n: usize) -> Option<Vec<u16>> {
    let n_bars = n_bars.max(1);
    let blocks = unframe(bytes)?;
    let lo_bits = 8.min(bits);
    let hi_bits = bits.saturating_sub(8);
    let lo = unpack(&lz4_flex::decompress_size_prepended(blocks.first()?).ok()?, lo_bits, n)?;
    let hi = if hi_bits > 0 {
        unpack(&lz4_flex::decompress_size_prepended(blocks.get(1)?).ok()?, hi_bits, n)?
    } else {
        vec![0u16; n]
    };

    let mut out = Vec::with_capacity(n);
    for r in 0..n.div_ceil(n_bars) {
        let mut prev = 0i32;
        let base = r * n_bars;
        for i in base..(base + n_bars).min(n) {
            let z = (lo[i] as u32) | ((hi[i] as u32) << 8);
            let d = ((z >> 1) as i32) ^ -((z & 1) as i32);
            prev = wrap_signed(prev + d, bits.max(1));
            // The wrap is modular; bring it back into range the same way v3
            // does, by masking to the stored width.
            let m = if bits >= 16 { 0xFFFFi32 } else { (1i32 << bits) - 1 };
            prev &= m;
            out.push(prev as u16);
        }
    }
    Some(out)
}

// ---------------------------------------------------------------------------
// Residual codecs
// ---------------------------------------------------------------------------

/// How the correction between the prediction and the truth is stored.
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum ResidualCodec {
    /// Uniform steps over ±`range` dB. Clips beyond it.
    Uniform { range: f32, bits: u32 },
    /// µ-law over ±`range` dB: fine near zero, coarse in the tail, no clipping
    /// inside the range.
    MuLaw { range: f32, bits: u32, mu: f32 },
}

impl ResidualCodec {
    pub fn bits(&self) -> u32 {
        match *self {
            ResidualCodec::Uniform { bits, .. } | ResidualCodec::MuLaw { bits, .. } => bits,
        }
    }

    /// Distance from the centre code to either end, in codes.
    ///
    /// A residual codec has to be able to say "no correction", and that needs a
    /// code at exactly zero. `2^bits` codes is an even count with no middle:
    /// spreading them over `-1..=1` puts the centre at `(2^bits - 1) / 2`, which
    /// is not an integer, and the nearest code decodes to a small non-zero
    /// residual. At 8 bits over ±80 dB that is a **0.31 dB bias applied to every
    /// perfectly predicted band in the file** — small, constant, and in one
    /// direction, which is the worst shape an error can have.
    ///
    /// So the codes are made an odd count by giving up one: `half` either side
    /// of an exact centre, with the top code unused.
    pub fn half(&self) -> f32 {
        (((1u32 << self.bits()) - 2) / 2) as f32
    }

    /// Number of distinct codes actually used.
    pub fn levels(&self) -> f32 {
        self.half() * 2.0
    }

    /// Residual in dB to a stored code.
    pub fn quantise(&self, res_db: f32) -> u16 {
        let u = match *self {
            ResidualCodec::Uniform { range, .. } => (res_db / range).clamp(-1.0, 1.0),
            ResidualCodec::MuLaw { range, mu, .. } => {
                let a = (res_db.abs() / range).min(1.0);
                let s = if res_db < 0.0 { -1.0 } else { 1.0 };
                s * (1.0 + mu * a).ln() / (1.0 + mu).ln()
            }
        };
        let h = self.half();
        ((u * h).round() + h).clamp(0.0, h * 2.0) as u16
    }

    /// Stored code back to a residual in dB.
    pub fn dequantise(&self, q: u16) -> f32 {
        let h = self.half();
        let u = (q as f32 - h) / h;
        match *self {
            ResidualCodec::Uniform { range, .. } => u * range,
            ResidualCodec::MuLaw { range, mu, .. } => {
                let a = u.abs().min(1.0);
                let s = if u < 0.0 { -1.0 } else { 1.0 };
                s * range * ((1.0 + mu).powf(a) - 1.0) / mu
            }
        }
    }

    pub fn label(&self) -> String {
        match *self {
            ResidualCodec::Uniform { range, bits } => {
                format!("uniform {bits}b +-{range:.0}dB")
            }
            ResidualCodec::MuLaw { range, bits, mu } => {
                format!("mu-law {bits}b +-{range:.0}dB mu={mu:.0}")
            }
        }
    }
}

// ---------------------------------------------------------------------------
// Candidate formats — every one encodes to bytes and decodes from them
// ---------------------------------------------------------------------------

/// What a candidate stores in place of a mono plane.
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum Candidate {
    /// Mono only, at `bits`. The baseline the others are measured against.
    MonoOnly { bits: u32 },
    /// L and R at `bits`, and mono stored independently at `mono_bits`.
    LrPlusMono { bits: u32, mono_bits: u32 },
    /// L and R at `bits`, and a residual against the shared predictor.
    LrPlusResidual { bits: u32, res: ResidualCodec },
    /// L and R only; mono is the prediction, with no correction at all.
    LrOnly { bits: u32 },
}

impl Candidate {
    pub fn label(&self) -> String {
        match *self {
            Candidate::MonoOnly { bits } => format!("mono only {bits}b"),
            Candidate::LrPlusMono { bits, mono_bits } => {
                format!("L/R {bits}b + mono {mono_bits}b")
            }
            Candidate::LrPlusResidual { bits, res } => {
                format!("L/R {bits}b + {}", res.label())
            }
            Candidate::LrOnly { bits } => format!("L/R {bits}b, predicted mono"),
        }
    }

    pub fn lr_bits(&self) -> Option<u32> {
        match *self {
            Candidate::MonoOnly { .. } => None,
            Candidate::LrPlusMono { bits, .. }
            | Candidate::LrPlusResidual { bits, .. }
            | Candidate::LrOnly { bits } => Some(bits),
        }
    }
}

/// One encoded track, as bytes.
pub struct Encoded {
    pub bytes: Vec<u8>,
    pub n_frames: usize,
    pub n_bars: usize,
}

impl Encoded {
    pub fn len(&self) -> usize {
        self.bytes.len()
    }
}

/// Encode `l`, `r` and `m` — all plane values in `0.0..=1.0` — under `cand`.
///
/// Every prediction inside is made from the values that have just been stored,
/// never from the inputs.
pub fn encode(
    cand: Candidate,
    l: &[f32],
    r: &[f32],
    m: &[f32],
    n_bars: usize,
) -> Encoded {
    let n = m.len();
    let mut blocks: Vec<Vec<u8>> = Vec::new();

    match cand {
        Candidate::MonoOnly { bits } => {
            let c = PlaneCodec::new(bits);
            let q: Vec<u16> = m.iter().map(|&v| c.quantise(v)).collect();
            blocks.push(encode_plane(&q, bits, n_bars));
        }
        Candidate::LrOnly { bits } => {
            let c = PlaneCodec::new(bits);
            blocks.push(encode_plane(
                &l.iter().map(|&v| c.quantise(v)).collect::<Vec<_>>(),
                bits,
                n_bars,
            ));
            blocks.push(encode_plane(
                &r.iter().map(|&v| c.quantise(v)).collect::<Vec<_>>(),
                bits,
                n_bars,
            ));
        }
        Candidate::LrPlusMono { bits, mono_bits } => {
            let c = PlaneCodec::new(bits);
            let cm = PlaneCodec::new(mono_bits);
            blocks.push(encode_plane(
                &l.iter().map(|&v| c.quantise(v)).collect::<Vec<_>>(),
                bits,
                n_bars,
            ));
            blocks.push(encode_plane(
                &r.iter().map(|&v| c.quantise(v)).collect::<Vec<_>>(),
                bits,
                n_bars,
            ));
            blocks.push(encode_plane(
                &m.iter().map(|&v| cm.quantise(v)).collect::<Vec<_>>(),
                mono_bits,
                n_bars,
            ));
        }
        Candidate::LrPlusResidual { bits, res } => {
            let c = PlaneCodec::new(bits);
            let ql: Vec<u16> = l.iter().map(|&v| c.quantise(v)).collect();
            let qr: Vec<u16> = r.iter().map(|&v| c.quantise(v)).collect();
            // The residual is taken against the prediction the decoder will
            // make, which means dequantising what was just quantised.
            let mut qres: Vec<u16> = Vec::with_capacity(n);
            for i in 0..n {
                let p = predict_mono_db(
                    v_to_db(c.dequantise(ql[i])),
                    v_to_db(c.dequantise(qr[i])),
                );
                // Against the *stored* truth, so this is like for like with a
                // mono plane at the same depth.
                let truth = v_to_db(c.dequantise(c.quantise(m[i])));
                qres.push(res.quantise(truth - p));
            }
            blocks.push(encode_plane(&ql, bits, n_bars));
            blocks.push(encode_plane(&qr, bits, n_bars));
            blocks.push(encode_plane(&qres, res.bits(), n_bars));
        }
    }

    Encoded {
        bytes: frame(&blocks),
        n_frames: if n_bars > 0 { n / n_bars } else { 0 },
        n_bars,
    }
}

/// Reconstruct the mono plane from stored bytes alone.
pub fn decode_mono(cand: Candidate, e: &Encoded) -> Option<Vec<f32>> {
    let n = e.n_frames * e.n_bars;
    let blocks = unframe(&e.bytes)?;
    match cand {
        Candidate::MonoOnly { bits } => {
            let c = PlaneCodec::new(bits);
            let q = decode_plane(blocks.first()?, bits, e.n_bars, n)?;
            Some(q.into_iter().map(|x| c.dequantise(x)).collect())
        }
        Candidate::LrOnly { bits } => {
            let c = PlaneCodec::new(bits);
            let ql = decode_plane(blocks.first()?, bits, e.n_bars, n)?;
            let qr = decode_plane(blocks.get(1)?, bits, e.n_bars, n)?;
            Some(
                (0..n)
                    .map(|i| {
                        db_to_v(predict_mono_db(
                            v_to_db(c.dequantise(ql[i])),
                            v_to_db(c.dequantise(qr[i])),
                        ))
                    })
                    .collect(),
            )
        }
        Candidate::LrPlusMono { mono_bits, .. } => {
            let cm = PlaneCodec::new(mono_bits);
            let q = decode_plane(blocks.get(2)?, mono_bits, e.n_bars, n)?;
            Some(q.into_iter().map(|x| cm.dequantise(x)).collect())
        }
        Candidate::LrPlusResidual { bits, res } => {
            let c = PlaneCodec::new(bits);
            let ql = decode_plane(blocks.first()?, bits, e.n_bars, n)?;
            let qr = decode_plane(blocks.get(1)?, bits, e.n_bars, n)?;
            let qres = decode_plane(blocks.get(2)?, res.bits(), e.n_bars, n)?;
            Some(
                (0..n)
                    .map(|i| {
                        let p = predict_mono_db(
                            v_to_db(c.dequantise(ql[i])),
                            v_to_db(c.dequantise(qr[i])),
                        );
                        db_to_v(p + res.dequantise(qres[i]))
                    })
                    .collect(),
            )
        }
    }
}

// ---------------------------------------------------------------------------
// Short round-trip cases
// ---------------------------------------------------------------------------

/// Small, exact cases, run before any corpus measurement.
///
/// A size measurement proves nothing about a codec: a coder that drops the
/// high bits compresses beautifully. Every case here encodes to bytes and
/// decodes from those bytes only, through the real packing, LZ4 and delta.
#[cfg(test)]
mod round_trip_tests {
    use super::*;

    const BARS: usize = 7; // deliberately not a multiple of 8 bits anywhere

    fn planes(v: &[(f32, f32, f32)]) -> (Vec<f32>, Vec<f32>, Vec<f32>) {
        (
            v.iter().map(|x| x.0).collect(),
            v.iter().map(|x| x.1).collect(),
            v.iter().map(|x| x.2).collect(),
        )
    }

    /// Encode then decode, and return the reconstructed mono plane.
    fn round_trip(cand: Candidate, l: &[f32], r: &[f32], m: &[f32], n_bars: usize) -> Vec<f32> {
        let e = encode(cand, l, r, m, n_bars);
        decode_mono(cand, &e).expect("decode failed")
    }

    /// True mono as a format at this depth could ever have stored it.
    fn stored_truth(bits: u32, m: &[f32]) -> Vec<f32> {
        let c = PlaneCodec::new(bits);
        m.iter().map(|&v| c.dequantise(c.quantise(v))).collect()
    }

    fn worst_db(a: &[f32], b: &[f32]) -> f32 {
        a.iter()
            .zip(b)
            .map(|(&x, &y)| (v_to_db(x) - v_to_db(y)).abs())
            .fold(0.0f32, f32::max)
    }

    // ── the plane codec itself ─────────────────────────────────────────────

    #[test]
    fn packing_survives_row_lengths_that_are_not_byte_aligned() {
        // 12 bits x 7 values is 84 bits: the last byte is half padding, and a
        // decoder that trusts the byte length reads an eighth value out of it.
        for bits in [4u32, 6, 8, 10, 11, 12, 16] {
            for n in [1usize, 3, 7, 7 * 5, 100] {
                // In u32: at 16 bits `max + 1` overflows a u16 to zero.
                let span = 1u32 << bits;
                let vals: Vec<u16> =
                    (0..n).map(|i| ((i as u32 * 37) % span) as u16).collect();
                let packed = pack(&vals, bits);
                let back = unpack(&packed, bits, n).expect("unpack");
                assert_eq!(back, vals, "bits={bits} n={n}");
                assert_eq!(
                    packed.len(),
                    (n * bits as usize).div_ceil(8),
                    "bits={bits} n={n}: packed size is not the bit count rounded up"
                );
            }
        }
    }

    #[test]
    fn a_plane_round_trips_through_delta_split_and_lz4() {
        for bits in [4u32, 8, 10, 12, 16] {
            let max = ((1u32 << bits) - 1) as u16;
            // Zero, full scale, alternating extremes, and a ramp — the shapes
            // the delta coder is most likely to get wrong.
            let mut codes: Vec<u16> = Vec::new();
            codes.extend(std::iter::repeat_n(0u16, BARS));
            codes.extend(std::iter::repeat_n(max, BARS));
            codes.extend((0..BARS).map(|i| if i % 2 == 0 { 0 } else { max }));
            // In u32. At 16 bits `i * max` overflows a u16 for i >= 2, which
            // panics in debug and wraps in release -- so the release build was
            // not testing the ramp it claimed to, it was testing whatever the
            // wrap produced.
            let ramp: Vec<u16> = (0..BARS)
                .map(|i| ((i as u32 * max as u32) / BARS as u32) as u16)
                .collect();
            // Assert the ramp is the ramp, so widening the arithmetic cannot
            // satisfy this test by quietly producing something else. It must
            // start at zero, rise, and stay inside the depth.
            assert_eq!(ramp.len(), BARS);
            assert_eq!(ramp[0], 0, "bits={bits}: the ramp must start at zero");
            for k in 1..BARS {
                assert!(
                    ramp[k] >= ramp[k - 1],
                    "bits={bits}: the ramp fell at {k}: {:?}",
                    ramp
                );
                assert!(
                    ramp[k] <= max,
                    "bits={bits}: the ramp left the depth at {k}: {} > {max}",
                    ramp[k]
                );
                assert_eq!(
                    ramp[k] as u32,
                    (k as u32 * max as u32) / BARS as u32,
                    "bits={bits}: the ramp is not the intended value at {k}"
                );
            }
            assert!(
                ramp[BARS - 1] > 0 || max == 0,
                "bits={bits}: the ramp never rose above zero"
            );
            codes.extend(ramp);
            let n = codes.len();
            let bytes = encode_plane(&codes, bits, BARS);
            let back = decode_plane(&bytes, bits, BARS, n).expect("decode_plane");
            assert_eq!(back, codes, "bits={bits}");
        }
    }

    #[test]
    fn the_quantiser_never_leaves_the_axis() {
        for bits in [4u32, 8, 12, 16] {
            let c = PlaneCodec::new(bits);
            for v in [-1.0f32, 0.0, 0.5, 1.0, 2.0, f32::NAN] {
                let q = c.quantise(v);
                let back = c.dequantise(q);
                assert!(
                    (0.0..=1.0).contains(&back),
                    "bits={bits} v={v} left the axis at {back}"
                );
            }
            assert_eq!(c.dequantise(c.quantise(0.0)), 0.0);
            assert_eq!(c.dequantise(c.quantise(1.0)), 1.0);
        }
    }

    // ── residual codecs ────────────────────────────────────────────────────

    #[test]
    fn residual_endpoints_and_zero_survive_their_codecs() {
        for res in [
            ResidualCodec::Uniform { range: 80.0, bits: 8 },
            ResidualCodec::Uniform { range: 12.0, bits: 4 },
            ResidualCodec::MuLaw { range: 80.0, bits: 8, mu: 63.0 },
            ResidualCodec::MuLaw { range: 80.0, bits: 6, mu: 255.0 },
        ] {
            let label = res.label();
            // Zero must survive exactly, or every quiet band acquires a bias.
            let z = res.dequantise(res.quantise(0.0));
            assert!(z.abs() < 1e-4, "{label}: zero came back as {z}");
            // The endpoints must reach the full range, not fall short of it.
            let hi = res.dequantise(res.quantise(80.0));
            let lo = res.dequantise(res.quantise(-80.0));
            let range = match res {
                ResidualCodec::Uniform { range, .. } | ResidualCodec::MuLaw { range, .. } => range,
            };
            assert!(
                (hi - range).abs() < range * 0.02,
                "{label}: +full came back as {hi}, want {range}"
            );
            assert!(
                (lo + range).abs() < range * 0.02,
                "{label}: -full came back as {lo}, want {}",
                -range
            );
            // Monotonic, so a larger residual never decodes smaller.
            let mut prev = f32::NEG_INFINITY;
            for k in 0..=res.levels() as u16 {
                let v = res.dequantise(k);
                assert!(v >= prev, "{label}: code {k} decoded below its predecessor");
                prev = v;
            }
        }
    }

    // ── the predictor, at the awkward values ───────────────────────────────

    #[test]
    fn the_floor_costs_nothing_when_both_channels_are_on_it() {
        // The obvious guess -- that two floored channels predict 6 dB above the
        // floor -- is wrong, and it was in this module's own notes until this
        // test was written. The prediction is a mean, and the mean of two equal
        // levels is that level. Clamping both inputs and the truth the same way
        // keeps them consistent.
        let p = predict_mono_db(-80.0, -80.0);
        assert!(
            (p - -80.0).abs() < 0.01,
            "two floored channels predicted {p}, expected the floor itself"
        );
    }

    #[test]
    fn the_floor_costs_real_decibels_when_only_one_channel_is_on_it() {
        // Where it does bite, it bites upward: a channel truly far below -80 dB
        // reads as exactly -80 and contributes far more to the mean than it
        // should. L at -74 dB against genuine silence is 3.5 dB of error that no
        // bit depth can remove, because nothing in the bytes distinguishes
        // "-80 dB" from "nothing at all".
        let p = predict_mono_db(-74.0, -80.0);
        assert!(
            (p - -76.5).abs() < 0.2,
            "expected about -76.5 from the stored pair, got {p}"
        );
        // The truth, had R really been silent: -74 dB halved is -80.02, which
        // stores at the floor.
        let truth = -80.0f32;
        let err = p - truth;
        assert!(
            err > 3.0,
            "the floor should leave the prediction several dB high, got {err}"
        );
        // And the residual must be able to carry exactly that back down.
        for res in [
            ResidualCodec::MuLaw { range: 80.0, bits: 8, mu: 63.0 },
            ResidualCodec::Uniform { range: 80.0, bits: 8 },
        ] {
            let corrected = p + res.dequantise(res.quantise(truth - p));
            assert!(
                (corrected - truth).abs() < 0.7,
                "{}: correction landed at {corrected}, want {truth}",
                res.label()
            );
        }
    }

    #[test]
    fn one_silent_channel_predicts_six_db_down() {
        // Mono of (x, 0) is x/2, which is 6 dB below x, at every level.
        for l in [-6.0f32, -20.0, -40.0, -70.0] {
            let p = predict_mono_db(l, -80.0);
            let want = 20.0 * ((10f32.powf(l / 20.0) + 1e-4) * 0.5).log10();
            assert!((p - want).abs() < 0.01, "l={l}: {p} vs {want}");
        }
    }

    #[test]
    fn saturation_is_reported_as_the_top_of_the_axis() {
        // Two channels at full scale predict +6 dB, which the axis cannot hold.
        let p = predict_mono_db(0.0, 0.0);
        assert_eq!(p, 0.0, "the prediction must clamp to the top of the axis");
        // The residual then has nothing to correct, because the stored truth is
        // also clamped there.
        assert_eq!(db_to_v(p), 1.0);
    }

    // ── the case that exposed the raw-versus-stored mismatch ───────────────

    #[test]
    fn the_two_ten_thousandths_case_is_reproduced_from_stored_values_only() {
        // L = .0002, R = 0, so true mono = .0001 in normalised magnitude. The
        // old study predicted from these raw numbers and scored against them,
        // and got an exact answer. A decoder sees none of them: it sees what
        // the -80 dB clamp and the quantiser left behind.
        let (l_raw, r_raw, m_raw) = (0.0002f32, 0.0f32, 0.0001f32);
        // As plane values, through the product's own conversion.
        let to_v = |mag: f32| -> f32 {
            let db = (20.0 * (mag * 0.25f32).log10()).max(-80.0);
            ((db + 80.0) / 80.0).clamp(0.0, 1.0)
        };
        let (vl, vr, vm) = (to_v(l_raw), to_v(r_raw), to_v(m_raw));

        // Every one of them is at or below the floor, so the axis cannot tell
        // them apart at all. That is the finding, not a failure.
        assert_eq!(vl, 0.0, "L is below the -80 dB floor");
        assert_eq!(vr, 0.0, "R is below the floor");
        assert_eq!(vm, 0.0, "true mono is below the floor");

        for bits in [8u32, 12, 16] {
            let cand = Candidate::LrPlusResidual {
                bits,
                res: ResidualCodec::MuLaw { range: 80.0, bits: 8, mu: 63.0 },
            };
            let out = round_trip(cand, &[vl], &[vr], &[vm], 1);
            assert_eq!(
                out.len(),
                1,
                "bits={bits}: the decoder returned the wrong length"
            );
            // The prediction is -74 dB; the residual must bring it back to the
            // floor, because that is where the stored truth is.
            let got = v_to_db(out[0]);
            assert!(
                got < -79.0,
                "bits={bits}: floored input reconstructed at {got} dB, not at the floor"
            );
        }
    }

    // ── whole formats ──────────────────────────────────────────────────────

    #[test]
    fn silence_reconstructs_as_silence() {
        let n = BARS * 4;
        let z = vec![0.0f32; n];
        for cand in [
            Candidate::MonoOnly { bits: 12 },
            Candidate::LrOnly { bits: 12 },
            Candidate::LrPlusMono { bits: 12, mono_bits: 12 },
            Candidate::LrPlusResidual {
                bits: 12,
                res: ResidualCodec::MuLaw { range: 80.0, bits: 8, mu: 63.0 },
            },
        ] {
            let out = round_trip(cand, &z, &z, &z, BARS);
            let worst = out.iter().map(|&v| v_to_db(v) + 80.0).fold(0.0f32, f32::max);
            assert!(
                worst < 1.0,
                "{}: silence reconstructed {worst} dB above the floor",
                cand.label()
            );
        }
    }

    #[test]
    fn identical_channels_reconstruct_as_themselves() {
        // With L = R = M, the prediction is exact before any residual, so this
        // isolates the plane codec from the predictor.
        let n = BARS * 6;
        let v: Vec<f32> = (0..n).map(|i| (i as f32 / n as f32) * 0.9 + 0.05).collect();
        for bits in [8u32, 12, 16] {
            let cand = Candidate::LrOnly { bits };
            let out = round_trip(cand, &v, &v, &v, BARS);
            let truth = stored_truth(bits, &v);
            let worst = worst_db(&out, &truth);
            let allowed = PlaneCodec::new(bits).max_step_error_db() * 2.0 + 0.01;
            assert!(
                worst <= allowed,
                "bits={bits}: identical channels drifted {worst} dB, allowed {allowed}"
            );
        }
    }

    #[test]
    fn a_residual_format_beats_a_predicted_one_on_the_cases_that_matter() {
        // Hard-panned and near-cancelling content, where the prediction is at
        // its worst. The residual must actually fix it; if it does not, the
        // third plane is not earning its bytes.
        let cases: Vec<(f32, f32, f32)> = vec![
            // (L, R, true mono) as plane values.
            (0.90, 0.10, 0.55), // hard panned left
            (0.10, 0.90, 0.55), // hard panned right
            (0.80, 0.80, 0.30), // near-antiphase: both loud, mono cancels
            (0.75, 0.75, 0.75), // in phase
            (0.60, 0.05, 0.20),
            (0.05, 0.60, 0.20),
            (0.00, 0.95, 0.83), // one channel at the floor
        ];
        let (l, r, m) = planes(&cases);
        let bits = 12;
        let predicted = round_trip(Candidate::LrOnly { bits }, &l, &r, &m, cases.len());
        let residual = round_trip(
            Candidate::LrPlusResidual {
                bits,
                res: ResidualCodec::MuLaw { range: 80.0, bits: 8, mu: 63.0 },
            },
            &l,
            &r,
            &m,
            cases.len(),
        );
        let truth = stored_truth(bits, &m);
        let p_worst = worst_db(&predicted, &truth);
        let r_worst = worst_db(&residual, &truth);
        assert!(
            p_worst > 5.0,
            "setup: the prediction alone should be badly wrong here, got {p_worst} dB"
        );
        assert!(
            r_worst < 1.0,
            "the residual left {r_worst} dB on cases the prediction gets wrong"
        );
    }

    #[test]
    fn an_independent_mono_plane_is_exact_at_its_own_depth() {
        let n = BARS * 5;
        let m: Vec<f32> = (0..n).map(|i| ((i * 7) % 100) as f32 / 100.0).collect();
        let l = vec![0.5f32; n];
        let r = vec![0.2f32; n];
        for mono_bits in [8u32, 12, 16] {
            let cand = Candidate::LrPlusMono { bits: 12, mono_bits };
            let out = round_trip(cand, &l, &r, &m, BARS);
            let truth = stored_truth(mono_bits, &m);
            assert_eq!(
                out.len(),
                truth.len(),
                "mono_bits={mono_bits}: wrong length"
            );
            for (i, (&a, &b)) in out.iter().zip(&truth).enumerate() {
                assert!(
                    (a - b).abs() < 1e-6,
                    "mono_bits={mono_bits} cell {i}: {a} vs {b} — an independently \
                     stored plane must come back exactly as stored"
                );
            }
        }
    }

    // ── downstream ─────────────────────────────────────────────────────────

    #[test]
    fn the_diff_reading_survives_twelve_bit_planes() {
        // Diff is right-minus-left in dB, so it is the difference of two
        // quantised values and can carry twice one plane's error. Check it is
        // no worse than that, and that the sign — which decides the colour and
        // the side — never flips on a difference worth seeing.
        let bits = 12;
        let c = PlaneCodec::new(bits);
        let n = 256;
        let mut worst = 0.0f32;
        for i in 0..n {
            let vl = i as f32 / (n - 1) as f32;
            for k in 0..n {
                let vr = k as f32 / (n - 1) as f32;
                let raw = crate::spectrum::channels::diff_fraction(vl, vr);
                let got = crate::spectrum::channels::diff_fraction(
                    c.dequantise(c.quantise(vl)),
                    c.dequantise(c.quantise(vr)),
                );
                worst = worst.max((raw - got).abs());
                if raw.abs() > 0.02 {
                    assert_eq!(
                        crate::spectrum::channels::louder(raw),
                        crate::spectrum::channels::louder(got),
                        "vl={vl} vr={vr}: the difference changed channel"
                    );
                }
            }
        }
        // `diff_fraction` is (right - left) * 80 / 20, so one dB of plane error
        // is 0.05 of a fraction, and two planes can each be off by half a step.
        let allowed = 2.0 * c.max_step_error_db() * 80.0 / crate::spectrum::channels::DIFF_FULL_SCALE_DB / 80.0
            * 80.0
            / 20.0
            + 1e-4;
        assert!(
            worst <= allowed,
            "diff drifted {worst} of full scale, allowed {allowed}"
        );
    }

    #[test]
    fn peak_selection_survives_twelve_bit_planes() {
        // The renderer folds groups of bars by taking a maximum. Quantisation
        // can reorder two bars inside half a step of each other, so the *index*
        // chosen is not stable — but the value it selects must be, or a peak
        // visibly drops.
        let bits = 12;
        let c = PlaneCodec::new(bits);
        let n = 4096;
        let raw: Vec<f32> = (0..n)
            .map(|i| {
                let x = i as f32 / n as f32;
                (x * 37.0).sin() * 0.35 + 0.5
            })
            .collect();
        let stored: Vec<f32> = raw.iter().map(|&v| c.dequantise(c.quantise(v))).collect();
        for chunk in 1..=8 {
            for w in raw.chunks(chunk).zip(stored.chunks(chunk)) {
                let a = w.0.iter().cloned().fold(f32::MIN, f32::max);
                let b = w.1.iter().cloned().fold(f32::MIN, f32::max);
                assert!(
                    (v_to_db(a) - v_to_db(b)).abs() <= c.max_step_error_db() + 1e-4,
                    "a group maximum moved {} dB",
                    (v_to_db(a) - v_to_db(b)).abs()
                );
            }
        }
    }
}

// ---------------------------------------------------------------------------
// The corrected corpus comparison
// ---------------------------------------------------------------------------

/// Every candidate encoded to bytes and decoded from them, on real tracks.
///
/// Replaces the storage half of `mono_study`, which predicted from raw analyser
/// magnitudes and scored against them. Here the only inputs to a reconstruction
/// are the bytes the candidate wrote.
///
/// Error is measured against the **true analysed mono**, in dB on the 80 dB
/// axis, so every candidate carries its own quantisation in its own number and
/// they are directly comparable. A candidate that stores mono at 8 bits is
/// charged for that; one that stores it at 16 is not.
///
/// Run:
///   cargo test --release corrected_corpus -- --ignored --nocapture
#[cfg(test)]
mod corpus {
    use super::*;
    use std::io::Write;

    fn candidates() -> Vec<Candidate> {
        let mu8 = ResidualCodec::MuLaw { range: 80.0, bits: 8, mu: 63.0 };
        let mu6 = ResidualCodec::MuLaw { range: 80.0, bits: 6, mu: 63.0 };
        let un8 = ResidualCodec::Uniform { range: 80.0, bits: 8 };
        vec![
            // Baselines: mono alone, at three depths.
            Candidate::MonoOnly { bits: 16 },
            Candidate::MonoOnly { bits: 12 },
            Candidate::MonoOnly { bits: 8 },
            // Stereo with no third plane at all.
            Candidate::LrOnly { bits: 16 },
            Candidate::LrOnly { bits: 12 },
            // Stereo plus an independently stored mono plane.
            Candidate::LrPlusMono { bits: 16, mono_bits: 16 },
            Candidate::LrPlusMono { bits: 12, mono_bits: 12 },
            // Stereo plus a residual, at the same L/R precision the candidate
            // would actually store.
            Candidate::LrPlusResidual { bits: 16, res: mu8 },
            Candidate::LrPlusResidual { bits: 12, res: mu8 },
            Candidate::LrPlusResidual { bits: 12, res: un8 },
            Candidate::LrPlusResidual { bits: 12, res: mu6 },
        ]
    }

    struct Acc {
        hist: crate::spectrum::aslt::mono_study::Hist,
        bytes: u64,
    }

    #[test]
    #[ignore]
    fn corrected_corpus() {
        let _ = std::fs::create_dir_all(OUT_DIR);
        let stamp = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_secs();
        let out = std::path::PathBuf::from(OUT_DIR).join(format!("corrected_{stamp}.txt"));
        let mut w = std::io::BufWriter::new(std::fs::File::create(&out).unwrap());
        eprintln!("corrected corpus -> {}", out.display());

        let s = crate::spectrum::load_spectrum_settings().expect("no spectrum.json");
        let p = Params {
            n_bars: s.bar_count,
            min_freq: s.min_freq,
            max_freq: s.max_freq,
            pre_fps: s.pre_fps,
            aslt_cfg: s.aslt_cfg.clone(),
            pad_factor: s.pad_factor,
            overlap: s.overlap,
            window_fn: s.window_fn.clone(),
            interp: s.interp_mode.clone(),
        };

        let _ = writeln!(
            w,
            "corrected storage comparison — decode from stored bytes only\n\
bars {} | {:.0}-{:.0} Hz | pre_fps {} | aslt {:?}\n\n\
Error is dB against the true analysed mono on the 80 dB axis, so each\n\
candidate carries its own quantisation. Sizes are real encoded bytes.",
            p.n_bars, p.min_freq, p.max_freq, p.pre_fps, p.aslt_cfg
        );

        let maps: Vec<Mapping> = match std::env::var("STUDY_MAPPINGS").as_deref() {
            Ok("superlet") => vec![Mapping::Superlet],
            Ok("cqt") => vec![Mapping::Cqt],
            _ => vec![Mapping::Superlet, Mapping::Cqt],
        };

        let mut files: Vec<std::path::PathBuf> = std::fs::read_dir(MUSIC_DIR)
            .expect("Test Music Files")
            .filter_map(|e| e.ok().map(|e| e.path()))
            .filter(|p| p.is_file())
            .collect();
        files.sort();
        let only: Option<String> = std::env::var("STUDY_ONLY").ok();

        let cands = candidates();
        // Pooled across the corpus, per mapping.
        let mut pooled: Vec<Vec<Acc>> = maps
            .iter()
            .map(|_| {
                cands
                    .iter()
                    .map(|_| Acc {
                        hist: crate::spectrum::aslt::mono_study::Hist::new(),
                        bytes: 0,
                    })
                    .collect()
            })
            .collect();
        let mut lr_bytes: Vec<u64> = maps.iter().map(|_| 0).collect();
        let mut cells: Vec<u64> = maps.iter().map(|_| 0).collect();

        for f in &files {
            if let Some(o) = &only
                && !f.to_string_lossy().contains(o.as_str())
            {
                continue;
            }
            let name = f.file_name().unwrap().to_string_lossy().to_string();
            let (l_sig, r_sig, sr) = match decode_stereo(f) {
                Ok(v) => v,
                Err(e) => {
                    let _ = writeln!(w, "\n### {name}\n  SKIPPED: {e}");
                    continue;
                }
            };
            let m_sig: Vec<f32> = l_sig.iter().zip(&r_sig).map(|(&a, &b)| (a + b) / 2.0).collect();
            eprintln!("=== {name}");
            let _ = writeln!(w, "\n\n### {name}  ({sr} Hz, {:.1} s)", l_sig.len() as f64 / sr as f64);

            for (mi, &mapping) in maps.iter().enumerate() {
                let t = std::time::Instant::now();
                let (la, db_ref) = run(&l_sig, sr, &p, mapping);
                let (ra, _) = run(&r_sig, sr, &p, mapping);
                let (ma, _) = run(&m_sig, sr, &p, mapping);
                if la.is_empty() || ra.is_empty() || ma.is_empty() {
                    continue;
                }
                let frames = la.len().min(ra.len()).min(ma.len());
                let bars = la[0].len();
                eprintln!("    {} {frames}x{bars} in {:.0}s", mapping.name(), t.elapsed().as_secs_f64());

                // Flatten to plane values once.
                let flat = |a: &Vec<Vec<f32>>| -> Vec<f32> {
                    let mut v = Vec::with_capacity(frames * bars);
                    for row in a.iter().take(frames) {
                        v.extend(row.iter().map(|&m| cache_v(m, db_ref)));
                    }
                    v
                };
                let vl = flat(&la);
                let vr = flat(&ra);
                let vm = flat(&ma);
                drop(la);
                drop(ra);
                drop(ma);
                cells[mi] += vm.len() as u64;

                // L+R alone, for the tradeoff table.
                let e_lr = encode(Candidate::LrOnly { bits: 12 }, &vl, &vr, &vm, bars);
                lr_bytes[mi] += e_lr.len() as u64;
                drop(e_lr);

                let _ = writeln!(w, "\n  --- {} --- {frames} x {bars}", mapping.name());
                for (ci, &cand) in cands.iter().enumerate() {
                    let e = encode(cand, &vl, &vr, &vm, bars);
                    let bytes = e.len() as u64;
                    let got = decode_mono(cand, &e).expect("decode");
                    drop(e);
                    assert_eq!(got.len(), vm.len(), "{}: length changed", cand.label());
                    let mut h = crate::spectrum::aslt::mono_study::Hist::new();
                    for (i, (&a, &b)) in got.iter().zip(&vm).enumerate() {
                        let err = v_to_db(a) - v_to_db(b);
                        h.push(err, i / bars, i % bars);
                        pooled[mi][ci].hist.push(err, i / bars, i % bars);
                    }
                    pooled[mi][ci].bytes += bytes;
                    let _ = writeln!(
                        w,
                        "    {:<34} {:>9.2} MB  mean {:>8.4}  p99 {:>7.3}  p99.9 {:>8.3}  \
p99.99 {:>8.3}  max {:>9.3}",
                        cand.label(),
                        bytes as f64 / 1_048_576.0,
                        h.mean(),
                        h.pct(0.99),
                        h.pct(0.999),
                        h.pct(0.9999),
                        h.max,
                    );
                    let _ = w.flush();
                }
            }
        }

        // ── pooled ─────────────────────────────────────────────────────────
        for (mi, &mapping) in maps.iter().enumerate() {
            let _ = writeln!(
                w,
                "\n\n## {} — pooled over the corpus ({} cells)\n\n\
candidate | MB | vs mono16 | mean dB | p99 | p99.9 | p99.99 | max | bound\n\
--- | --- | --- | --- | --- | --- | --- | --- | ---",
                mapping.name(),
                cells[mi]
            );
            let base = pooled[mi][0].bytes.max(1);
            for (ci, &cand) in cands.iter().enumerate() {
                let a = &pooled[mi][ci];
                // The analytically established bound, where there is one.
                let bound = match cand {
                    Candidate::MonoOnly { bits } | Candidate::LrPlusMono { mono_bits: bits, .. } => {
                        format!("{:.4} dB", PlaneCodec::new(bits).max_step_error_db())
                    }
                    Candidate::LrOnly { .. } => "none".to_string(),
                    Candidate::LrPlusResidual { bits, res } => {
                        // Half a residual step, plus the L/R quantisation the
                        // residual was measured against, which cancels — the
                        // residual is taken against the stored truth, so the
                        // remaining term is the stored truth vs the real one.
                        let step = match res {
                            ResidualCodec::Uniform { range, .. } => range / res.half(),
                            ResidualCodec::MuLaw { range, mu, .. } => {
                                let h = res.half();
                                range * ((1.0 + mu).powf(1.0 / h) - 1.0) / mu
                            }
                        };
                        format!(
                            "{:.4} dB near 0",
                            step * 0.5 + PlaneCodec::new(bits).max_step_error_db()
                        )
                    }
                };
                let _ = writeln!(
                    w,
                    "`{}` | {:.1} | {:+.0}% | {:.4} | {:.3} | {:.3} | {:.3} | {:.3} | {}",
                    cand.label(),
                    a.bytes as f64 / 1_048_576.0,
                    a.bytes as f64 / base as f64 * 100.0 - 100.0,
                    a.hist.mean(),
                    a.hist.pct(0.99),
                    a.hist.pct(0.999),
                    a.hist.pct(0.9999),
                    a.hist.max,
                    bound,
                );
            }
            let _ = writeln!(
                w,
                "\nL+R at 12 bits alone: {:.1} MB — the floor any stereo format pays \
before a third plane.",
                lr_bytes[mi] as f64 / 1_048_576.0
            );
        }
        let _ = w.flush();
        eprintln!("done -> {}", out.display());
    }
}
