//! Canonical audio payloads, the exact-output format policy, and the byte
//! writers that put a payload on the wire.
//!
//! # Why a canonical payload at all
//!
//! Until 1.4.2 the bit-perfect ring carried `f32`, and every backend
//! reconstructed device bytes by multiplying that float by a power of two.
//! That is exact for sources of 24 bits or fewer — an `f32` has 24 bits of
//! mantissa — and silently lossy for everything else. A 32-bit integer source
//! lost its low eight bits before it ever reached a device, and the diamond
//! stayed green while it happened. The DoP path had a subtler version of the
//! same problem: it packed marker bytes into a 24-bit word, converted *that*
//! to `f32`, and relied on the multiply being its exact inverse.
//!
//! The ring now carries a `u32` **canonical payload** whose meaning is fixed
//! by the session's [`PcmKind`]:
//!
//! * [`PcmKind::Integer`] — a left-aligned signed 32-bit sample, stored as raw
//!   bits. 16-bit sources are `v << 16`, 24-bit are `v << 8`, 32-bit are
//!   themselves. Left alignment is what makes every widening free: the same
//!   32 bits are simultaneously the correct 16-, 24- and 32-bit sample, so a
//!   writer never rescales, it only chooses where to cut.
//! * [`PcmKind::Float32`] — `f32::to_bits()`, moved to the device untouched.
//! * DoP — the raw 16-bit DSD payload in the low half, with **no marker**.
//!   The marker is applied at the output boundary by [`DopState`], because it
//!   is a property of the carrier frame the device is about to receive, not of
//!   the audio.
//!
//! # What "exact" means here
//!
//! Exact sample/payload words from Moosik to the Windows audio driver. That is
//! the whole claim. What the driver, the USB link and the DAC do afterwards is
//! outside anything this process can observe, and nothing in this module
//! should be read as evidence about it.

use std::fmt;

// ---------------------------------------------------------------------------
// Source description
// ---------------------------------------------------------------------------

/// The numeric family of a decoded source, and how much of it is real.
///
/// `valid_bits` is the source's own precision, not its storage width: a 24-bit
/// FLAC decoded into `i32` storage is `Integer { valid_bits: 24 }`. Where the
/// precision cannot be established, callers use the storage width, which is
/// conservative in the only direction that matters — it can widen the format
/// requirement, never narrow it.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum PcmKind {
    Integer {
        valid_bits: u8,
    },
    Float32,
    /// 64-bit float. No WASAPI format can carry one exactly, so this never
    /// reaches an exact route — it is a distinct variant rather than an error
    /// so the reason survives all the way to the status line.
    Float64,
}

impl PcmKind {
    pub fn describe(&self) -> String {
        match self {
            PcmKind::Integer { valid_bits } => format!("{valid_bits}-bit integer"),
            PcmKind::Float32 => "32-bit float".into(),
            PcmKind::Float64 => "64-bit float".into(),
        }
    }

    /// Same family, ignoring precision — the axis a stream may never be
    /// reused across.
    pub fn family(&self) -> Family {
        match self {
            PcmKind::Integer { .. } => Family::Integer,
            PcmKind::Float32 => Family::Float32,
            PcmKind::Float64 => Family::Float64,
        }
    }
}

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum Family {
    Integer,
    Float32,
    Float64,
}

/// A channel mask in WASAPI/symphonia `dwChannelMask` order. `0` means the
/// source declared no positional layout.
///
/// Carried because two streams with the same channel *count* and different
/// masks are different formats: three channels as L/R/LFE and three as L/R/C
/// put the same samples in different speakers. Reusing one stream for the
/// other is silent channel corruption, so the reuse predicate compares this
/// and not just the count.
#[derive(Clone, Copy, PartialEq, Eq, Debug, Default)]
pub struct ChannelLayout(pub u32);

impl ChannelLayout {
    pub const UNSPECIFIED: ChannelLayout = ChannelLayout(0);

    /// Whether the source actually named its speakers.
    ///
    /// A channel *count* is not a layout: three channels as L/R/LFE and three
    /// as L/R/C carry the same samples to different speakers, and a claim that
    /// the audio reached the DAC unaltered means nothing if it reached the
    /// wrong outputs.
    pub fn is_specified(&self) -> bool {
        self.0 != 0
    }
}

impl fmt::Display for ChannelLayout {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        if self.0 == 0 {
            write!(f, "unspecified")
        } else {
            write!(f, "{:#010x}", self.0)
        }
    }
}

/// Everything about a source that the output format must respect.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub struct SourceFormat {
    pub kind: PcmKind,
    pub sample_rate: u32,
    pub channels: u16,
    pub layout: ChannelLayout,
}

impl SourceFormat {
    /// One line for the persistent output panel, e.g.
    /// `192 kHz · stereo · 24-bit integer`.
    pub fn describe(&self) -> String {
        format!(
            "{} · {} · {}",
            super::fmt_khz(self.sample_rate),
            describe_channels(self.channels),
            self.kind.describe()
        )
    }
}

pub fn describe_channels(n: u16) -> String {
    match n {
        1 => "mono".into(),
        2 => "stereo".into(),
        n => format!("{n}ch"),
    }
}

// ---------------------------------------------------------------------------
// Canonical payload encoding (decode side — never realtime)
// ---------------------------------------------------------------------------

/// DSD's idle payload: `0b01101001` repeated. A sigma-delta DAC fed all-zero
/// bits sits at full-scale negative DC, so silence has to be this pattern and
/// not zero. Two bytes make one 16-bit DoP payload.
///
/// Derived from the container module's byte rather than restated, so the
/// render thread's silence and the file reader's tail padding cannot drift.
pub const DSD_SILENCE_PAYLOAD: u32 =
    (crate::dsd::dop::DSD_SILENCE as u32) << 8 | crate::dsd::dop::DSD_SILENCE as u32;

#[inline]
fn top_mask(valid_bits: u8) -> u32 {
    // `<< 32` panics in debug and wraps in release, so the full-width case is
    // spelled out rather than shifted into.
    if valid_bits >= 32 {
        u32::MAX
    } else {
        !0u32 << (32 - valid_bits as u32)
    }
}

/// Left-align a signed sample of `valid_bits` precision into canonical form.
///
/// The shift is the entire conversion: no multiply, no float, no rounding.
/// That is the property the whole exact path rests on, so it is one function
/// with one test rather than an idiom repeated per backend.
#[inline]
pub fn canon_from_signed(v: i32, valid_bits: u8) -> u32 {
    debug_assert!((1..=32).contains(&valid_bits));
    if valid_bits >= 32 {
        v as u32
    } else {
        ((v as u32) << (32 - valid_bits as u32)) & top_mask(valid_bits)
    }
}

#[inline]
pub fn canon_from_i16(v: i16) -> u32 {
    ((v as i32) << 16) as u32
}

/// `v` must already be sign-extended from 24 bits.
#[inline]
pub fn canon_from_i24(v: i32) -> u32 {
    ((v as u32) << 8) & 0xFFFF_FF00
}

#[inline]
pub fn canon_from_i32(v: i32) -> u32 {
    v as u32
}

#[inline]
pub fn canon_from_f32(v: f32) -> u32 {
    v.to_bits()
}

/// Offset-binary (unsigned) integer sources: symphonia hands `u8`/`u16`/`u24`/
/// `u32` buffers out of some containers, where the midpoint is `2^(n-1)`
/// rather than zero. Flipping the top bit converts to two's complement without
/// touching any other bit.
#[inline]
pub fn canon_from_unsigned(v: u32, storage_bits: u8) -> u32 {
    debug_assert!((2..=32).contains(&storage_bits));
    let signed = v ^ (1u32 << (storage_bits as u32 - 1));
    if storage_bits >= 32 {
        signed
    } else {
        (signed << (32 - storage_bits as u32)) & top_mask(storage_bits)
    }
}

/// Read a canonical payload back as the float the analyser wants.
///
/// Observational only. The result is copied into the spectrum's own scratch
/// and never returns to the ring — an `f32` cannot round-trip a 32-bit integer
/// payload, so a value that made this trip is no longer exact.
#[inline]
pub fn canon_to_f32(canon: u32, kind: PcmKind) -> f32 {
    match kind {
        PcmKind::Integer { .. } => (canon as i32) as f32 / 2_147_483_648.0,
        PcmKind::Float32 | PcmKind::Float64 => f32::from_bits(canon),
    }
}

/// Whether the bits below `valid_bits` are all zero, as a left-aligned
/// canonical sample from a source of that precision must be.
///
/// A source that declares 24 bits and delivers dirt in the low 8 is either
/// mis-declared or mis-decoded; either way the exact claim for it is not
/// provable, so the caller downgrades rather than asserting.
#[inline]
pub fn low_bits_are_clean(canon: u32, valid_bits: u8) -> bool {
    if valid_bits >= 32 {
        return true;
    }
    canon & !top_mask(valid_bits) == 0
}

// ---------------------------------------------------------------------------
// Device formats
// ---------------------------------------------------------------------------

/// How samples are laid out in the device buffer.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum DevFmt {
    I16,
    /// 24-bit packed into 3 bytes.
    I24,
    /// 24 valid bits, left-aligned in a 32-bit container.
    I24In32,
    /// 32 valid bits in a 32-bit container.
    I32,
    F32,
}

impl DevFmt {
    #[inline]
    pub fn bytes_per_sample(&self) -> usize {
        match self {
            DevFmt::I16 => 2,
            DevFmt::I24 => 3,
            DevFmt::I24In32 | DevFmt::I32 | DevFmt::F32 => 4,
        }
    }
}

/// Whether a candidate is an integer or float subtype, without depending on
/// `wasapi` — this module compiles on every platform.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum SampleTag {
    Int,
    Float,
}

pub struct Candidate {
    pub store_bits: usize,
    pub valid_bits: usize,
    pub tag: SampleTag,
    pub dev: DevFmt,
    /// Shown in the status line and the log.
    pub label: &'static str,
    /// Accepted by `MOOSIK_BP_FORMAT`.
    pub token: &'static str,
}

/// The five formats the exclusive path knows how to write. Indices are stable
/// and are what [`exact_candidates`] returns.
pub const CANDIDATES: [Candidate; 5] = [
    Candidate {
        store_bits: 16,
        valid_bits: 16,
        tag: SampleTag::Int,
        dev: DevFmt::I16,
        label: "16i excl",
        token: "16i",
    },
    Candidate {
        store_bits: 24,
        valid_bits: 24,
        tag: SampleTag::Int,
        dev: DevFmt::I24,
        label: "24i excl",
        token: "24i",
    },
    Candidate {
        store_bits: 32,
        valid_bits: 24,
        tag: SampleTag::Int,
        dev: DevFmt::I24In32,
        label: "24i/32 excl",
        token: "24i32",
    },
    Candidate {
        store_bits: 32,
        valid_bits: 32,
        tag: SampleTag::Int,
        dev: DevFmt::I32,
        label: "32i excl",
        token: "32i",
    },
    Candidate {
        store_bits: 32,
        valid_bits: 32,
        tag: SampleTag::Float,
        dev: DevFmt::F32,
        label: "32f excl",
        token: "32f",
    },
];

pub const IDX_I16: usize = 0;
pub const IDX_I24: usize = 1;
pub const IDX_I24_32: usize = 2;
pub const IDX_I32: usize = 3;
pub const IDX_F32: usize = 4;

/// Integer containers wide enough to carry a 24-bit DoP word, in the order
/// hardware has actually been verified to like: packed 24 first (confirmed on
/// an SMSL C200Pro), then the two 32-bit containers.
pub const DOP_CANDIDATES: [usize; 3] = [IDX_I24, IDX_I24_32, IDX_I32];

// ---------------------------------------------------------------------------
// The exactness policy
// ---------------------------------------------------------------------------

/// Every device format that can carry `kind` **without losing a bit**, best
/// first — or why none can.
///
/// There is deliberately no last-chance candidate. Before 1.4.3 each source
/// width had a full five-format fallback order ending in `16i`, so a 24-bit
/// file on a device that only offered 16-bit exclusive would negotiate 16-bit,
/// truncate eight bits per sample, and light the diamond. Every entry in every
/// list below is exact for its source; if no entry survives the answer is an
/// error, not a quieter format.
///
/// `dop` overrides the source family entirely: a DoP session's payload is a
/// 24-bit word with a marker in its top byte, so it needs an integer container
/// of at least 24 bits whatever the underlying DSD looks like.
pub fn exact_candidates(kind: PcmKind, dop: bool) -> Result<Vec<usize>, String> {
    if dop {
        return Ok(DOP_CANDIDATES.to_vec());
    }
    match kind {
        PcmKind::Integer { valid_bits: 0 } => Err("source declares zero valid bits".into()),
        // The narrowest exact container first, then wider ones — except above
        // 16 bits, where 24-in-32 precedes packed 24: a Realtek HDA driver
        // advertised packed 24 and then mishandled it, which is the noise
        // 1.4.1 shipped with.
        PcmKind::Integer { valid_bits } if valid_bits <= 16 => {
            Ok(vec![IDX_I16, IDX_I24_32, IDX_I32, IDX_I24])
        }
        PcmKind::Integer { valid_bits } if valid_bits <= 24 => {
            Ok(vec![IDX_I24_32, IDX_I24, IDX_I32])
        }
        PcmKind::Integer { valid_bits } if valid_bits <= 32 => {
            // Only a full 32 valid bits will do. `24i/32` has the same block
            // alignment and the same byte layout, and would look like it
            // worked while the device read 24 of the 32 bits.
            Ok(vec![IDX_I32])
        }
        PcmKind::Integer { valid_bits } => Err(format!(
            "{valid_bits}-bit integer is wider than any device format"
        )),
        PcmKind::Float32 => Ok(vec![IDX_F32]),
        PcmKind::Float64 => Err("64-bit float has no exact exclusive-mode representation — \
             every available device format would round it"
            .into()),
    }
}

/// Parse one `MOOSIK_BP_FORMAT` value into a candidate index.
pub fn parse_forced(value: &str) -> Result<usize, String> {
    let want = value.trim();
    CANDIDATES
        .iter()
        .position(|c| c.token == want)
        .ok_or_else(|| {
            format!(
                "MOOSIK_BP_FORMAT=\"{want}\" is not a device format \
                 (expected 16i, 24i, 24i32, 32i or 32f)"
            )
        })
}

/// Resolve the negotiation order, applying `MOOSIK_BP_FORMAT` **through** the
/// exactness matrix rather than around it.
///
/// An override is a debugging aid. It can pick among the formats that are
/// exact for this source; it cannot authorise one that is not. Pinning `16i`
/// for a 24-bit file used to be obeyed silently, which is the shape of a
/// mistake that survives a whole listening session before anyone notices.
pub fn negotiation_order(
    kind: PcmKind,
    dop: bool,
    forced: Option<usize>,
) -> Result<Vec<usize>, String> {
    let allowed = exact_candidates(kind, dop)?;
    match forced {
        None => Ok(allowed),
        Some(i) if allowed.contains(&i) => Ok(vec![i]),
        Some(i) => Err(format!(
            "MOOSIK_BP_FORMAT={} cannot carry {} exactly{} — this source needs one of: {}",
            CANDIDATES[i].token,
            if dop {
                "a DoP word".to_string()
            } else {
                kind.describe()
            },
            if dop {
                " (an integer container of at least 24 bits)"
            } else {
                ""
            },
            allowed
                .iter()
                .map(|&a| CANDIDATES[a].token)
                .collect::<Vec<_>>()
                .join(", "),
        )),
    }
}

// ---------------------------------------------------------------------------
// Writers (realtime — no allocation)
// ---------------------------------------------------------------------------

/// A validated canonical→device byte writer.
///
/// Constructing one proves the pairing is exact and same-family; writing with
/// it is then infallible, which is what lets the render loop call it with no
/// error path. A float source and an integer device format cannot produce a
/// `Writer` at all, so "silently convert F32 to integer under a green diamond"
/// is unrepresentable rather than merely unlikely.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub struct Writer {
    dev: DevFmt,
    /// Kept for the `Debug` output and for the invariant it records: this
    /// writer was proven exact for this family, once, at construction.
    #[allow(dead_code)]
    kind: PcmKind,
}

impl Writer {
    pub fn new(kind: PcmKind, dev: DevFmt) -> Result<Writer, String> {
        let idx = CANDIDATES
            .iter()
            .position(|c| c.dev == dev)
            .ok_or_else(|| format!("unknown device format {dev:?}"))?;
        let allowed = exact_candidates(kind, false)?;
        if !allowed.contains(&idx) {
            return Err(format!(
                "{} cannot be written to {} exactly",
                kind.describe(),
                CANDIDATES[idx].label
            ));
        }
        Ok(Writer { dev, kind })
    }

    /// A DoP writer: the payload family is fixed by the transport, not by a
    /// decoded source.
    pub fn for_dop(dev: DevFmt) -> Result<Writer, String> {
        if !DOP_CANDIDATES.iter().any(|&i| CANDIDATES[i].dev == dev) {
            return Err(format!("{dev:?} cannot carry a DoP word"));
        }
        Ok(Writer {
            dev,
            kind: PcmKind::Integer { valid_bits: 24 },
        })
    }

    #[inline]
    pub fn dev(&self) -> DevFmt {
        self.dev
    }
    #[inline]
    pub fn bytes_per_sample(&self) -> usize {
        self.dev.bytes_per_sample()
    }
    /// Write `canon` as device bytes into `out`, which must be exactly
    /// `canon.len() * bytes_per_sample()` long.
    ///
    /// Integer packing is shifts and byte copies throughout. The pre-1.4.3
    /// writers multiplied an `f32` by `2^23`/`2^31` and cast; that is exact
    /// for narrow sources, quietly wrong for wide ones, and a float-to-int
    /// cast saturates, so its failure mode is a full-scale sample.
    #[inline]
    pub fn write(&self, canon: &[u32], out: &mut [u8]) {
        debug_assert_eq!(out.len(), canon.len() * self.bytes_per_sample());
        match self.dev {
            DevFmt::I16 => {
                for (c, o) in canon.iter().zip(out.chunks_exact_mut(2)) {
                    o.copy_from_slice(&(((*c as i32) >> 16) as i16).to_le_bytes());
                }
            }
            DevFmt::I24 => {
                for (c, o) in canon.iter().zip(out.chunks_exact_mut(3)) {
                    o.copy_from_slice(&((*c as i32) >> 8).to_le_bytes()[..3]);
                }
            }
            // Both integer containers and the float one are the canonical word
            // itself: left-aligned in 32 bits is precisely what canonical form
            // means, and an `f32` payload is already its own bit pattern.
            DevFmt::I24In32 | DevFmt::I32 | DevFmt::F32 => {
                for (c, o) in canon.iter().zip(out.chunks_exact_mut(4)) {
                    o.copy_from_slice(&c.to_le_bytes());
                }
            }
        }
    }
}

// ---------------------------------------------------------------------------
// DoP marker ownership
// ---------------------------------------------------------------------------

/// The two alternating DoP marker bytes, single-sourced from the container
/// module so the encoder and the decoder cannot disagree about them.
pub use crate::dsd::dop::DOP_MARKERS;

/// The DoP marker phase, owned by the render thread.
///
/// Before 1.4.3 the marker was applied by the decode thread and travelled
/// through the ring inside each word, which made three things impossible to
/// get right at once. Silence inserted by the *render* thread — prefill,
/// pause, underrun, drain — had no marker source, so it went out as all-zero
/// PCM frames and a DoP DAC dropped lock. Nothing could guarantee one marker
/// per carrier frame across channels, because the ring's frame boundaries were
/// not enforced. And a pause consumed source words in order to produce
/// silence, destroying audio by not listening to it.
///
/// One phase counter on the thread that emits frames fixes all three: every
/// complete channel-frame gets exactly one marker, silence is generated with a
/// valid marker at the point of need, and the source payload is untouched
/// until a frame is genuinely going to the device.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct DopState {
    phase: u8,
}

impl DopState {
    pub fn new() -> Self {
        DopState { phase: 0 }
    }

    // Production never reads or writes the phase: the render thread owns one
    // counter for the life of the stream, so there is nothing to carry across
    // a file boundary and nothing to restart after a seek. That is the whole
    // improvement, and these exist only so the tests can pin it.
    #[cfg(test)]
    pub fn phase(&self) -> u8 {
        self.phase
    }
    #[cfg(test)]
    pub fn set_phase(&mut self, p: u8) {
        self.phase = p & 1;
    }

    /// Take the next frame's marker, pre-shifted into bits 23–16, and advance
    /// the phase. Public because two backends emit frames: WASAPI writes device
    /// bytes, cpal writes typed `i32` slots, and both must draw their marker
    /// from the same counter or the alternation breaks at a backend boundary.
    #[inline]
    pub fn next_marker(&mut self) -> u32 {
        let m = DOP_MARKERS[self.phase as usize] as u32;
        self.phase ^= 1;
        m << 16
    }

    /// Emit `total_frames` carrier frames of `channels` samples each into
    /// `out`, taking payload from `payload` for as far as it goes and marked
    /// DSD silence for the rest.
    ///
    /// `payload` holds interleaved 16-bit DSD payloads with no marker, and its
    /// length must be a whole number of frames. The returned count is how many
    /// frames came from `payload`; the rest is silence the caller treats as an
    /// underrun if a session was supposed to be feeding it.
    #[inline]
    pub fn emit(
        &mut self,
        payload: &[u32],
        total_frames: usize,
        channels: usize,
        writer: &Writer,
        out: &mut [u8],
    ) -> usize {
        debug_assert!(channels > 0);
        debug_assert_eq!(payload.len() % channels, 0, "payload must be whole frames");
        let bps = writer.bytes_per_sample();
        debug_assert_eq!(out.len(), total_frames * channels * bps);

        let from_payload = (payload.len() / channels).min(total_frames);
        let mut off = 0usize;

        for f in 0..total_frames {
            let marker = self.next_marker();
            for c in 0..channels {
                let word = if f < from_payload {
                    marker | (payload[f * channels + c] & 0xFFFF)
                } else {
                    marker | DSD_SILENCE_PAYLOAD
                };
                write_word(word, writer.dev(), &mut out[off..off + bps]);
                off += bps;
            }
        }
        from_payload
    }
}

/// One 24-bit DoP word into device bytes.
///
/// Split out so the DoP path and the canonical PCM path cannot drift: `word`
/// is already the final 24-bit value, so it is left-aligned to canonical form
/// exactly once, here.
#[inline]
fn write_word(word: u32, dev: DevFmt, out: &mut [u8]) {
    match dev {
        DevFmt::I24 => out.copy_from_slice(&word.to_le_bytes()[..3]),
        DevFmt::I24In32 | DevFmt::I32 => out.copy_from_slice(&(word << 8).to_le_bytes()),
        // Unreachable: `Writer::for_dop` refuses these.
        DevFmt::I16 | DevFmt::F32 => out.fill(0),
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------
//
// Expected bytes throughout are built from the *source value*, not by calling
// the production helpers a second time. A test that packs with `Writer::write`
// and then expects `Writer::write` proves only that the function is
// deterministic.

#[cfg(test)]
mod tests {
    use super::*;

    fn writer(kind: PcmKind, dev: DevFmt) -> Writer {
        Writer::new(kind, dev).unwrap_or_else(|e| panic!("{kind:?} -> {dev:?}: {e}"))
    }

    /// Every one of the 65,536 `i16` values, end to end.
    ///
    /// The pre-1.4.3 path was `i16 -> f32 (/2^15) -> *2^15 -> as i16`. That
    /// happens to round-trip for 16-bit, which is why it survived so long; the
    /// point here is that the replacement does too, without a float in sight.
    #[test]
    fn every_i16_survives_canonical_packing() {
        let k = PcmKind::Integer { valid_bits: 16 };
        let w16 = writer(k, DevFmt::I16);
        let w24 = writer(k, DevFmt::I24);
        let w32 = writer(k, DevFmt::I32);
        let w2432 = writer(k, DevFmt::I24In32);

        let mut b2 = [0u8; 2];
        let mut b3 = [0u8; 3];
        let mut b4 = [0u8; 4];

        for v in i16::MIN..=i16::MAX {
            let canon = canon_from_i16(v);

            w16.write(&[canon], &mut b2);
            assert_eq!(b2, v.to_le_bytes(), "16-bit round trip failed at {v}");

            // Widening is exact: the 16-bit value sits in the top of a 24-bit
            // field, i.e. the same number multiplied by 2^8.
            w24.write(&[canon], &mut b3);
            let want24 = ((v as i32) * 256) as u32;
            assert_eq!(
                b3,
                want24.to_le_bytes()[..3],
                "packed-24 widening failed at {v}"
            );

            // ...and in the top of a 32-bit field, multiplied by 2^16.
            let want32 = ((v as i64) * 65_536) as i32;
            w32.write(&[canon], &mut b4);
            assert_eq!(b4, want32.to_le_bytes(), "32-bit widening failed at {v}");
            w2432.write(&[canon], &mut b4);
            assert_eq!(b4, want32.to_le_bytes(), "24-in-32 widening failed at {v}");
        }
    }

    /// 20- and 24-bit values, dense around the edges and pseudorandom in the
    /// middle. 20-bit is the case a `bits_per_sample` of 20 produces, which no
    /// device format matches exactly — it rides a 24-bit container, and the
    /// arithmetic has to be right for a width that is nobody's container.
    #[test]
    fn twenty_and_twenty_four_bit_values_survive() {
        for &bits in &[20u8, 24u8] {
            let k = PcmKind::Integer { valid_bits: bits };
            let w24 = writer(k, DevFmt::I24);
            let w2432 = writer(k, DevFmt::I24In32);
            let w32 = writer(k, DevFmt::I32);
            let half = 1i64 << (bits - 1);

            let mut vals: Vec<i64> = vec![0, 1, -1, half - 1, -half, half - 2, -half + 1];
            // Deterministic pseudorandom sweep — a 64-bit LCG, no dependency.
            let mut s: u64 = 0x2545_F491_4F6C_DD1D;
            for _ in 0..20_000 {
                s = s
                    .wrapping_mul(6364136223846793005)
                    .wrapping_add(1442695040888963407);
                vals.push(((s >> 11) as i64 % (2 * half)) - half);
            }

            let mut b3 = [0u8; 3];
            let mut b4 = [0u8; 4];
            for v in vals {
                let canon = canon_from_signed(v as i32, bits);

                // The number, scaled to each container's full-scale, computed
                // by multiplication rather than by repeating the shift.
                let want24 = (v * (1i64 << (24 - bits))) as i32;
                let want32 = (v * (1i64 << (32 - bits))) as i32;

                w24.write(&[canon], &mut b3);
                assert_eq!(
                    b3,
                    (want24 as u32).to_le_bytes()[..3],
                    "{bits}-bit packed 24 at {v}"
                );
                w2432.write(&[canon], &mut b4);
                assert_eq!(b4, want32.to_le_bytes(), "{bits}-bit 24-in-32 at {v}");
                w32.write(&[canon], &mut b4);
                assert_eq!(b4, want32.to_le_bytes(), "{bits}-bit 32 at {v}");
            }
        }
    }

    /// The defect the canonical payload exists for: 32-bit integers that differ
    /// only in their lowest eight bits.
    ///
    /// `f32` has 24 bits of mantissa. The old transport carried these as
    /// floats, so every one of them collapsed onto the same value before the
    /// device saw it — and the diamond stayed lit.
    #[test]
    fn thirty_two_bit_low_bits_are_not_lost() {
        let k = PcmKind::Integer { valid_bits: 32 };
        let w = writer(k, DevFmt::I32);
        let mut seen = std::collections::HashSet::new();
        let mut b = [0u8; 4];

        let bases = [
            0i32,
            1,
            -1,
            i32::MIN,
            i32::MAX,
            0x100,
            0x101,
            0x0123_4500,
            -0x0123_4500,
        ];
        let mut values = std::collections::HashSet::new();
        for base in bases {
            for low in 0..=255i32 {
                let v = base.wrapping_add(low);
                let canon = canon_from_i32(v);
                w.write(&[canon], &mut b);
                assert_eq!(b, v.to_le_bytes(), "32-bit value {v} did not survive");
                // Distinct values must produce distinct bytes. Some bases
                // overlap by construction (0x100 + 1 is 0x101 + 0), so the
                // claim is about distinct *inputs*, not about the loop count.
                if values.insert(v) {
                    assert!(
                        seen.insert(b),
                        "two distinct 32-bit values produced the same bytes"
                    );
                }
            }
        }
        assert_eq!(seen.len(), values.len());

        // And the specific values the brief calls out, explicitly.
        for v in [0i32, 1, -1, i32::MIN, i32::MAX, 0x100, 0x101] {
            w.write(&[canon_from_i32(v)], &mut b);
            assert_eq!(b, v.to_le_bytes());
        }
    }

    /// Float payloads reach the device as their own bit pattern, including the
    /// values that any arithmetic would destroy.
    #[test]
    fn float_bit_patterns_survive_exactly() {
        let w = writer(PcmKind::Float32, DevFmt::F32);
        let mut b = [0u8; 4];

        let mut vals = vec![
            0.0f32,
            -0.0,
            1.0,
            -1.0,
            f32::MIN,
            f32::MAX,
            f32::MIN_POSITIVE,
            -f32::MIN_POSITIVE,
            f32::INFINITY,
            f32::NEG_INFINITY,
            f32::from_bits(0x0000_0001), // smallest subnormal
            f32::from_bits(0x0080_0000), // smallest normal
            f32::from_bits(0x7F80_0001), // signalling NaN payload
            f32::from_bits(0x7FC0_0000), // quiet NaN
            f32::from_bits(0xFFC0_1234), // negative NaN with a payload
        ];
        let mut s: u64 = 0x9E37_79B9_7F4A_7C15;
        for _ in 0..5000 {
            s = s.wrapping_mul(6364136223846793005).wrapping_add(1);
            vals.push(f32::from_bits((s >> 32) as u32));
        }

        for v in vals {
            let canon = canon_from_f32(v);
            w.write(&[canon], &mut b);
            assert_eq!(
                b,
                v.to_le_bytes(),
                "float bits {:#010x} did not survive",
                v.to_bits()
            );
            // Round-trip through the payload, bit for bit — NaN payloads and
            // signed zero included, which an equality check would miss.
            assert_eq!(f32::from_bits(u32::from_le_bytes(b)).to_bits(), v.to_bits());
        }

        // Negative zero is the one that reads as equal to positive zero and is
        // a different sample.
        assert_ne!(canon_from_f32(0.0), canon_from_f32(-0.0));
    }

    /// A writer that would narrow or cross families cannot be built at all.
    #[test]
    fn narrowing_and_cross_family_writers_are_unrepresentable() {
        // Narrowing.
        assert!(Writer::new(PcmKind::Integer { valid_bits: 24 }, DevFmt::I16).is_err());
        assert!(Writer::new(PcmKind::Integer { valid_bits: 32 }, DevFmt::I16).is_err());
        assert!(Writer::new(PcmKind::Integer { valid_bits: 32 }, DevFmt::I24).is_err());
        assert!(Writer::new(PcmKind::Integer { valid_bits: 32 }, DevFmt::I24In32).is_err());
        assert!(Writer::new(PcmKind::Integer { valid_bits: 17 }, DevFmt::I16).is_err());

        // Cross-family, both directions.
        assert!(Writer::new(PcmKind::Float32, DevFmt::I32).is_err());
        assert!(Writer::new(PcmKind::Float32, DevFmt::I24).is_err());
        assert!(Writer::new(PcmKind::Float32, DevFmt::I16).is_err());
        for b in 1..=32u8 {
            assert!(
                Writer::new(PcmKind::Integer { valid_bits: b }, DevFmt::F32).is_err(),
                "{b}-bit integer must not reach a float device format"
            );
        }

        // No exact writer exists for a 64-bit float at all.
        for d in [
            DevFmt::I16,
            DevFmt::I24,
            DevFmt::I24In32,
            DevFmt::I32,
            DevFmt::F32,
        ] {
            assert!(Writer::new(PcmKind::Float64, d).is_err());
        }

        // DoP refuses the two containers that cannot carry a marker.
        assert!(Writer::for_dop(DevFmt::I16).is_err());
        assert!(Writer::for_dop(DevFmt::F32).is_err());
        for d in [DevFmt::I24, DevFmt::I24In32, DevFmt::I32] {
            assert!(Writer::for_dop(d).is_ok());
        }
    }

    /// The whole policy matrix: every integer depth 1..=32 against every
    /// candidate, forced and unforced.
    #[test]
    fn the_policy_admits_only_exact_mappings() {
        for bits in 1..=32u8 {
            let k = PcmKind::Integer { valid_bits: bits };
            let allowed = exact_candidates(k, false).unwrap();
            assert!(!allowed.is_empty(), "{bits}-bit has no exact route");

            for (i, c) in CANDIDATES.iter().enumerate() {
                let admitted = allowed.contains(&i);
                // Exact iff the container is an integer with at least as many
                // valid bits as the source.
                let exact = c.tag == SampleTag::Int && c.valid_bits >= bits as usize;
                assert_eq!(
                    admitted, exact,
                    "{bits}-bit vs {}: admitted={admitted} exact={exact}",
                    c.label
                );

                // Forcing follows the same matrix, and a refusal names the
                // formats that would have worked.
                let forced = negotiation_order(k, false, Some(i));
                assert_eq!(forced.is_ok(), exact, "{bits}-bit forced {}", c.label);
                if let Ok(v) = forced {
                    assert_eq!(v, vec![i]);
                } else {
                    let e = negotiation_order(k, false, Some(i)).unwrap_err();
                    assert!(e.contains(CANDIDATES[allowed[0]].token), "{e}");
                }
            }
        }

        // Float32 reaches F32 and nothing else; F64 reaches nothing.
        let f = exact_candidates(PcmKind::Float32, false).unwrap();
        assert_eq!(f, vec![IDX_F32]);
        for i in 0..CANDIDATES.len() {
            assert_eq!(
                negotiation_order(PcmKind::Float32, false, Some(i)).is_ok(),
                i == IDX_F32
            );
        }
        assert!(exact_candidates(PcmKind::Float64, false).is_err());
        assert!(negotiation_order(PcmKind::Float64, false, None).is_err());
        assert!(negotiation_order(PcmKind::Float64, false, Some(IDX_F32)).is_err());

        // DoP: integer containers of at least 24 bits, whatever the source.
        for bits in 1..=32u8 {
            let k = PcmKind::Integer { valid_bits: bits };
            let d = exact_candidates(k, true).unwrap();
            assert_eq!(
                d,
                vec![IDX_I24, IDX_I24_32, IDX_I32],
                "{bits}-bit DoP order"
            );
            for &i in &d {
                assert!(CANDIDATES[i].tag == SampleTag::Int && CANDIDATES[i].store_bits >= 24);
            }
            assert!(negotiation_order(k, true, Some(IDX_I16)).is_err());
            assert!(negotiation_order(k, true, Some(IDX_F32)).is_err());
        }
    }

    /// The exact regression 1.4.1 shipped: a 24-bit source must never reach
    /// `16i`, and a 32-bit integer must never reach a float.
    #[test]
    fn the_two_narrowing_routes_that_used_to_exist_are_gone() {
        let o24 = exact_candidates(PcmKind::Integer { valid_bits: 24 }, false).unwrap();
        assert!(
            !o24.contains(&IDX_I16),
            "24-bit must never fall through to 16i"
        );
        assert!(!o24.contains(&IDX_F32));
        // ...and the 24-in-32 container still comes first, which is the fix
        // for the Realtek driver that mishandled packed 24.
        assert_eq!(o24[0], IDX_I24_32);

        let o32 = exact_candidates(PcmKind::Integer { valid_bits: 32 }, false).unwrap();
        assert_eq!(
            o32,
            vec![IDX_I32],
            "32-bit integer has exactly one exact container"
        );
    }

    /// Left-aligned canonical form is what makes widening free; a value whose
    /// low bits are dirty is not the width it claims.
    #[test]
    fn declared_precision_is_checked_not_assumed() {
        assert!(low_bits_are_clean(canon_from_i16(-1), 16));
        assert!(low_bits_are_clean(canon_from_i24(0x7F_FFFF), 24));
        assert!(low_bits_are_clean(canon_from_i32(i32::MIN), 32));
        // 24 declared, content in the low 8 → not actually 24-bit.
        assert!(!low_bits_are_clean(0x1234_5601, 24));
        assert!(!low_bits_are_clean(0x0000_0001, 31));
        assert!(
            low_bits_are_clean(0xFFFF_FFFF, 32),
            "32 bits have no unused low bits"
        );
    }

    /// Offset-binary sources land on the same canonical values as their signed
    /// equivalents.
    #[test]
    fn unsigned_sources_convert_to_the_same_canonical_form() {
        for v in 0..=u16::MAX {
            let signed = (v as i32 - 32_768) as i16;
            assert_eq!(
                canon_from_unsigned(v as u32, 16),
                canon_from_i16(signed),
                "u16 {v} should equal i16 {signed}"
            );
        }
        assert_eq!(canon_from_unsigned(0x80, 8), canon_from_signed(0, 8));
        assert_eq!(canon_from_unsigned(0x8000_0000, 32), canon_from_i32(0));
        assert_eq!(canon_from_unsigned(0, 32), canon_from_i32(i32::MIN));
    }

    // -----------------------------------------------------------------------
    // DoP
    // -----------------------------------------------------------------------

    /// Read a DoP word back out of device bytes, independently of the writer.
    fn word_from(bytes: &[u8], dev: DevFmt) -> u32 {
        match dev {
            DevFmt::I24 => bytes[0] as u32 | (bytes[1] as u32) << 8 | (bytes[2] as u32) << 16,
            DevFmt::I24In32 | DevFmt::I32 => {
                let v = u32::from_le_bytes([bytes[0], bytes[1], bytes[2], bytes[3]]);
                assert_eq!(
                    v & 0xFF,
                    0,
                    "a 24-bit word left-aligned in 32 has a zero low byte"
                );
                v >> 8
            }
            other => panic!("{other:?} cannot carry DoP"),
        }
    }

    /// The DoP contract, across both starting phases, every channel count and
    /// every allowed container: one marker per complete frame, the same marker
    /// on every channel of that frame, perfect alternation, and the payload
    /// arriving unaltered.
    #[test]
    fn dop_frames_are_marked_correctly_and_carry_the_payload_untouched() {
        for dev in [DevFmt::I24, DevFmt::I24In32, DevFmt::I32] {
            let w = Writer::for_dop(dev).unwrap();
            let bps = w.bytes_per_sample();
            for channels in 1..=8usize {
                for start_phase in 0..2u8 {
                    let mut st = DopState::new();
                    st.set_phase(start_phase);

                    // A payload where every channel and frame is identifiable.
                    let frames = 37usize; // prime, and odd, so phase flips across calls
                    let payload: Vec<u32> = (0..frames * channels)
                        .map(|i| ((i * 7 + 1) as u32) & 0xFFFF)
                        .collect();

                    let total = frames + 5; // five frames of forced silence
                    let mut out = vec![0u8; total * channels * bps];
                    let used = st.emit(&payload, total, channels, &w, &mut out);
                    assert_eq!(used, frames);

                    let mut expect_phase = start_phase as usize;
                    for fr in 0..total {
                        let marker = DOP_MARKERS[expect_phase];
                        for c in 0..channels {
                            let off = (fr * channels + c) * bps;
                            let word = word_from(&out[off..off + bps], dev);
                            assert_eq!(
                                (word >> 16) as u8,
                                marker,
                                "{dev:?} {channels}ch frame {fr} ch {c}: marker"
                            );
                            let want = if fr < frames {
                                payload[fr * channels + c]
                            } else {
                                0x6969
                            };
                            assert_eq!(
                                word & 0xFFFF,
                                want,
                                "{dev:?} {channels}ch frame {fr} ch {c}: payload"
                            );
                            assert_eq!(word & 0xFF00_0000, 0, "word wider than 24 bits");
                        }
                        expect_phase ^= 1;
                    }
                    // The next emit continues the alternation rather than
                    // restarting it.
                    assert_eq!(st.phase() as usize, expect_phase);
                }
            }
        }
    }

    /// Every silence path a render thread can take produces marked DSD
    /// silence, never a frame of PCM zeros.
    ///
    /// Prefill, pause, priming after a seek, underrun and drain all arrive here
    /// as "no payload available". Before 1.4.3 only the lead-in and the tail
    /// were marked, because they were the only silence the *decoder* knew
    /// about, and every other case dropped the DAC out of DSD lock.
    #[test]
    fn every_kind_of_dop_silence_is_still_a_valid_dop_frame() {
        for dev in [DevFmt::I24, DevFmt::I24In32, DevFmt::I32] {
            let w = Writer::for_dop(dev).unwrap();
            let bps = w.bytes_per_sample();
            for channels in 1..=8usize {
                let mut st = DopState::new();
                // Odd and even buffer lengths, so the phase lands on both.
                for &frames in &[1usize, 2, 3, 8, 33] {
                    let mut out = vec![0u8; frames * channels * bps];
                    let used = st.emit(&[], frames, channels, &w, &mut out);
                    assert_eq!(used, 0);
                    for fr in 0..frames {
                        for c in 0..channels {
                            let off = (fr * channels + c) * bps;
                            let word = word_from(&out[off..off + bps], dev);
                            assert!(
                                matches!((word >> 16) as u8, 0x05 | 0xFA),
                                "silence frame carried no marker"
                            );
                            assert_eq!(word & 0xFFFF, 0x6969, "silence payload must be 0x6969");
                        }
                    }
                    assert!(
                        out.iter().any(|&b| b != 0),
                        "an all-zero buffer would drop DoP lock"
                    );
                }
            }
        }
    }

    /// A partly-filled buffer is the underrun case: real payload, then marked
    /// silence, with the phase continuing straight through the join.
    #[test]
    fn an_underrun_splices_marked_silence_without_breaking_phase() {
        let w = Writer::for_dop(DevFmt::I24).unwrap();
        let channels = 2usize;
        let mut st = DopState::new();
        let payload = vec![0x1111u32, 0x2222, 0x3333, 0x4444, 0x5555, 0x6666]; // 3 frames
        let mut out = vec![0u8; 6 * channels * 3];
        assert_eq!(st.emit(&payload, 6, channels, &w, &mut out), 3);

        let markers: Vec<u8> = (0..6)
            .map(|fr| word_from(&out[fr * channels * 3..], DevFmt::I24) as u32)
            .map(|w| (w >> 16) as u8)
            .collect();
        assert_eq!(markers, vec![0x05, 0xFA, 0x05, 0xFA, 0x05, 0xFA]);
        // Frames 0..3 are audio, 3..6 are silence, and nothing in between is
        // an unmarked frame.
        assert_eq!(
            word_from(&out[channels * 3..], DevFmt::I24) & 0xFFFF,
            0x3333
        );
        assert_eq!(
            word_from(&out[2 * channels * 3..], DevFmt::I24) & 0xFFFF,
            0x5555
        );
        assert_eq!(
            word_from(&out[3 * channels * 3..], DevFmt::I24) & 0xFFFF,
            0x6969
        );
    }

    /// Whatever a volume or EQ setting says, there is no code path that could
    /// apply it to a DoP frame: `emit` takes no gain and the writer has no
    /// multiply. This pins that as a property rather than a convention.
    #[test]
    fn nothing_in_the_dop_path_can_scale_a_word() {
        let w = Writer::for_dop(DevFmt::I32).unwrap();
        let payload: Vec<u32> = (0..64u32).map(|i| (i * 1031) & 0xFFFF).collect();

        let mut a = vec![0u8; 32 * 2 * 4];
        let mut b = vec![0u8; 32 * 2 * 4];
        let mut s1 = DopState::new();
        let mut s2 = DopState::new();
        s1.emit(&payload, 32, 2, &w, &mut a);
        s2.emit(&payload, 32, 2, &w, &mut b);
        assert_eq!(
            a, b,
            "emitting the same payload twice must produce the same bytes"
        );

        // And the bytes are a pure function of marker + payload: reconstructing
        // them by hand from the source payload reproduces the buffer exactly.
        let mut want = Vec::with_capacity(a.len());
        for fr in 0..32usize {
            let marker = if fr % 2 == 0 { 0x05u32 } else { 0xFA };
            for c in 0..2usize {
                let word = (marker << 16) | payload[fr * 2 + c];
                want.extend_from_slice(&(word * 256).to_le_bytes());
            }
        }
        assert_eq!(a, want);
    }
}
