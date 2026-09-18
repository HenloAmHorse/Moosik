//! The stereo sidecar: left and right, stored beside a mono cache that still
//! stands on its own.
//!
//! # Why a sidecar and not a wider cache
//!
//! Mix is the default view, and the waterfall, the spectrogram explorer and the
//! octave meters all read it. Widening the cache format to three channels would
//! make every one of those depend on a file three times the size, invalidate
//! every cache the owner already has, and make a truncated stereo write cost
//! the mono spectrum as well. A sidecar costs one extra file and buys the
//! property that matters: **mono is never worse off**. A missing, truncated,
//! stale or refused sidecar is not an error — it is simply no channels, and
//! everything that reads mono carries on.
//!
//! # Identity
//!
//! A cache key is a hash of the path plus the settings that change the numbers.
//! Two analyses that share a key should be identical, but "should" is doing
//! work there: a code change, an interrupted write from an older build, or a
//! key that does not capture some new parameter all produce a sidecar that
//! *fits* a mono cache it did not come from. Indexing into it would draw a left
//! and right that belong to different audio than the Mix beside them.
//!
//! So the sidecar carries the mono cache's frame count, its bar count, and a
//! digest of its codes, and any disagreement refuses the sidecar rather than
//! stretching or trusting it.

use super::cache::{self, CacheError, Limits, PreFrames};
use std::path::{Path, PathBuf};

/// Sidecar magic, "MSPS".
pub const MAGIC: u32 = 0x4D53_5053;

/// Format version.
///
/// A sidecar is disposable — a mismatch recomputes rather than migrates — so
/// there is no migration path and no reader for older versions. v1 carried a
/// strided sample of the mono codes and nothing about the source; v2 carries
/// four separate identities. A v1 file is refused and rewritten by the next
/// analysis. **Mono caches are untouched by this: v2, v3 and v4 all still
/// load, and nothing is migrated or deleted.**
pub const VERSION: u32 = 2;

/// Which fingerprint algorithm produced the identities in a header.
///
/// Stored so the algorithm can be replaced without silently accepting files
/// whose identities were computed by the old one. A file naming an algorithm
/// this build does not implement is refused, not guessed at.
pub const FINGERPRINT_V1: u32 = 1;

/// The header, in order:
///
/// ```text
///  0  magic          u32
///  4  version        u32
///  8  frames         u32
/// 12  bars           u32
/// 16  fingerprint    u32   which algorithm produced the identities
/// 20  reserved       u32
/// 24  source         u64   which audio file
/// 32  params         u64   which settings
/// 40  mono           u64   the mono cache's full-content fingerprint
/// 48  payload        u64   the sidecar's own bytes
/// 56  left length    u32
/// 60  right length   u32
/// 64  payload
/// ```
///
/// Eight `u32` and four `u64`. Counting six `u32` here put the payload slice
/// eight bytes early, straight through the two lengths, and the integrity check
/// caught it on the first clean round trip — which is the one thing an
/// integrity check is guaranteed to be exercised by.
const HEADER: usize = 4 * 8 + 8 * 4;

/// Which channel a failure came from, so the log says which.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum Side {
    Left,
    Right,
}

impl std::fmt::Display for Side {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(match self {
            Side::Left => "left",
            Side::Right => "right",
        })
    }
}

/// What a sidecar has to agree with before it is believed.
///
/// Four things, kept apart because they fail for different reasons and mean
/// different things:
///
/// * **source** — which *path* this analysis was of; see [`Identity`];
/// * **parameters** — the settings it was analysed at;
/// * **mono content** — the numbers the accompanying mono cache holds;
/// * **payload integrity** — whether the sidecar's own bytes are intact.
///
/// The first two are not redundant with the third. **Identical mono does not
/// establish identical left and right**: two different stereo sources can share
/// a mono mix exactly — the trivial case is a track and its own channel-swapped
/// copy — so a sidecar that matched only on mono content could be paired with
/// the wrong stereo material. Binding the source and the parameters is what
/// closes that.
/// # What `source` is, and is not
///
/// **`source` is pathname identity, not content identity.** It is a hash of the
/// file's path, which is the same thing the cache filename is keyed on. It
/// establishes that a sidecar belongs to the analysis of *that path* at those
/// settings.
///
/// **It does not cover a file being replaced at the same path.** Overwrite a
/// track with different audio, keeping the name, and the old mono cache and the
/// old sidecar are still there, still agree with each other, and are both
/// accepted — every check here passes, because every one of them is about
/// whether the sidecar matches *that cache*, and it does. What is displayed is
/// then the previous track's analysis, Mix and channels alike.
///
/// An earlier version of this note claimed the mono fingerprint would refuse
/// the sidecar in that case. It does not: the two files were written together
/// and are consistent with each other. Nothing here detects that the *audio*
/// changed.
///
/// Recorded as a limitation. Source invalidation — noticing that a path now
/// holds different audio — is a question about the mono cache's own validity,
/// not about the sidecar, and is not designed here.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub struct Identity {
    /// The audio file's *path*. See the note above.
    pub source: u64,
    /// The settings it was analysed at.
    pub params: u64,
}

/// Why a sidecar was refused.
///
/// Every variant leads to the same place — draw Mix, offer to analyse stereo
/// again — but they are distinguished so the log can say which happened. A
/// file from another track and a file with a flipped bit are very different
/// bugs, and a single "rejected" would hide both.
#[derive(Debug, PartialEq, Eq)]
pub enum SidecarError {
    TooShort { got: usize, need: usize },
    BadMagic(u32),
    BadVersion(u32),
    /// The identities were computed by an algorithm this build does not have.
    BadFingerprintAlgo(u32),
    /// The sidecar describes a differently shaped analysis.
    Mismatch { what: &'static str, mono: usize, sidecar: usize },
    /// A different audio file, or the same file analysed at other settings.
    WrongSource { what: &'static str, want: u64, got: u64 },
    /// Same source, same settings, different numbers: another analysis of the
    /// same track — an older build, or a parameter the key does not capture.
    DifferentMonoContent { want: u64, got: u64 },
    /// The payload's own bytes do not hash to what the header says.
    ///
    /// A checksum, not a signature: it detects damage and mistakes, and claims
    /// nothing about deliberate forgery.
    Corrupt { want: u64, got: u64 },
    /// A channel's own payload would not decode.
    Channel { side: Side, err: CacheError },
    /// The declared block lengths do not fit the file.
    Truncated { declared: usize, got: usize },
    Io(CacheError),
}

impl std::fmt::Display for SidecarError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            SidecarError::TooShort { got, need } => {
                write!(f, "too short: {got} bytes, need {need}")
            }
            SidecarError::BadMagic(m) => write!(f, "not a sidecar (magic {m:#010x})"),
            SidecarError::BadVersion(v) => write!(f, "sidecar version {v} is not readable"),
            SidecarError::BadFingerprintAlgo(a) => {
                write!(f, "identities were computed by unknown algorithm {a}")
            }
            SidecarError::Mismatch { what, mono, sidecar } => {
                write!(f, "{what}: mono has {mono}, sidecar claims {sidecar}")
            }
            SidecarError::WrongSource { what, want, got } => {
                write!(f, "belongs to another {what} ({got:#018x} != {want:#018x})")
            }
            SidecarError::DifferentMonoContent { want, got } => {
                write!(f, "belongs to another analysis of this track \
                           ({got:#018x} != {want:#018x})")
            }
            SidecarError::Corrupt { want, got } => {
                write!(f, "payload is damaged ({got:#018x} != {want:#018x})")
            }
            SidecarError::Channel { side, err } => write!(f, "{side} channel: {err}"),
            SidecarError::Truncated { declared, got } => {
                write!(f, "declares {declared} bytes of payload, file holds {got}")
            }
            SidecarError::Io(e) => write!(f, "{e}"),
        }
    }
}

/// A matched pair, both the same shape as the mono cache they accompany.
///
/// There is no constructor that does not go through [`decode_with`] or
/// [`Stereo::new`], because a pair whose dimensions disagree with each other is
/// not something any consumer should have to check for.
#[derive(Clone, Default, PartialEq)]
pub struct Stereo {
    left: PreFrames,
    right: PreFrames,
}

impl std::fmt::Debug for Stereo {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "Stereo {{ {} frames x {} bars, {} KiB }}",
            self.len(),
            self.bars(),
            self.capacity_bytes() / 1024
        )
    }
}

impl Stereo {
    /// Pair two channels, or refuse if they do not describe the same analysis.
    pub fn new(left: PreFrames, right: PreFrames) -> Option<Self> {
        if left.len() != right.len() || left.bars() != right.bars() {
            return None;
        }
        Some(Self { left, right })
    }

    pub fn left(&self) -> &PreFrames {
        &self.left
    }

    pub fn right(&self) -> &PreFrames {
        &self.right
    }

    /// Rows, which both channels have the same number of.
    pub fn len(&self) -> usize {
        self.left.len()
    }

    pub fn is_empty(&self) -> bool {
        self.left.is_empty()
    }

    pub fn bars(&self) -> usize {
        self.left.bars()
    }

    /// One row of each channel, or `None` if the row does not exist.
    pub fn row(&self, frame: usize) -> Option<(&[u16], &[u16])> {
        Some((self.left.row(frame)?, self.right.row(frame)?))
    }

    /// What both channels are holding from the allocator.
    ///
    /// Capacity, not length: a cleared pair that kept its buffers would report
    /// zero on any length-based measure while still holding the memory, which
    /// is the bug `PreFrames::clear` was fixed for.
    pub fn capacity_bytes(&self) -> usize {
        self.left.capacity_bytes() + self.right.capacity_bytes()
    }

    /// Release both allocations. See `PreFrames::clear`.
    pub fn clear(&mut self) {
        self.left.clear();
        self.right.clear();
    }
}

/// Where the sidecar for a mono cache lives.
///
/// The same stem, so the two are paired by name and eviction can find one from
/// the other without parsing the key. A different extension, so a sidecar can
/// never be mistaken for a mono cache by a reader that only knows the old
/// format.
pub fn path_for(mono_cache: &Path) -> PathBuf {
    mono_cache.with_extension("stereocache")
}

/// The fingerprint mixing constants. Odd, and chosen for their bit patterns
/// rather than for anything numerological.
const FP_A: u64 = 0x9E37_79B1_85EB_CA87;
const FP_B: u64 = 0xC2B2_AE3D_27D4_EB4F;
const FP_C: u64 = 0x1656_67B1_9E37_79F9;

/// Fold one word into a running fingerprint.
#[inline]
fn fp_mix(h: u64, w: u64) -> u64 {
    let h = h ^ w.wrapping_mul(FP_B).rotate_left(31);
    h.rotate_left(27).wrapping_mul(FP_A).wrapping_add(FP_C)
}

/// A **full-content** fingerprint of a mono cache. Algorithm [`FINGERPRINT_V1`].
///
/// Every code is folded in — not a sample of them. The previous version hashed
/// one code in 64, which meant a change confined to unsampled cells was
/// invisible to it, and the v4 decoder does not close that gap: it checks
/// structure and sizes, and a validly decodable file with different numbers
/// passes all of them.
///
/// **Not cryptographic.** It detects accidental difference — a different
/// analysis, a damaged file, an older build's output — and claims nothing about
/// deliberate forgery. Anyone who can write to the cache directory can write a
/// header to match whatever they put in it.
///
/// It costs a full pass over 87 MiB at the reference size, tens of
/// milliseconds, which is why it runs on the sidecar reader's thread and never
/// on the one that draws.
///
/// The dimensions and the code ceiling are folded in first, so two analyses
/// that differ only in shape cannot collide.
pub fn fingerprint(mono: &PreFrames) -> u64 {
    let mut h = fp_mix(FP_A, mono.len() as u64);
    h = fp_mix(h, mono.bars() as u64);
    h = fp_mix(h, mono.max_code() as u64);
    for frame in 0..mono.len() {
        let Some(row) = mono.row(frame) else { continue };
        // Four codes to a word: the same bytes either way, four times fewer
        // rounds.
        let mut chunks = row.chunks_exact(4);
        for c in &mut chunks {
            let w = (c[0] as u64)
                | ((c[1] as u64) << 16)
                | ((c[2] as u64) << 32)
                | ((c[3] as u64) << 48);
            h = fp_mix(h, w);
        }
        for &code in chunks.remainder() {
            h = fp_mix(h, code as u64);
        }
    }
    h
}

/// The same fold over arbitrary bytes — the sidecar's own payload integrity.
///
/// Separate from [`fingerprint`] in *purpose*, not in algorithm: one answers
/// "is this the analysis I think it is", the other "did these bytes survive the
/// trip". Conflating them would mean a damaged file and a foreign file were the
/// same error.
pub fn payload_fingerprint(bytes: &[u8]) -> u64 {
    let mut h = fp_mix(FP_B, bytes.len() as u64);
    let mut chunks = bytes.chunks_exact(8);
    for c in &mut chunks {
        h = fp_mix(h, u64::from_le_bytes(c.try_into().expect("8 bytes")));
    }
    for &b in chunks.remainder() {
        h = fp_mix(h, b as u64);
    }
    h
}

/// Serialise a pair as a sidecar for the mono cache `mono_digest` came from.
///
/// The two channels are ordinary v4 payloads — the same encoder, the same
/// quantiser, the same tests — laid end to end behind a header that says which
/// mono analysis they belong to.
///
/// # Written once, not built and then copied
///
/// The obvious form builds the payload by concatenating the two channels, then
/// builds the file by concatenating the header and the payload — so at the end
/// the two channel buffers, the payload and the file are all alive, about
/// `3 × (left + right)` for a structure that is `left + right` of actual
/// content. On the corpus that is a 177–264 MiB peak for a 62–92 MB file.
///
/// Here the file is laid out once. Space for the header is reserved, each
/// channel is appended and released as it is appended, and the header is filled
/// in afterwards — the payload fingerprint is the only field that has to wait,
/// and it is computed over the payload region of the file itself.
///
/// **The bytes are identical to the concatenated form, by construction rather
/// than by argument**: the payload region of the output *is* left followed by
/// right, in that order, so `payload_fingerprint` folds over the same bytes in
/// the same sequence. The eight-byte word straddling the left/right boundary
/// and the trailing remainder are the same words, because there is only one
/// byte sequence involved. Nothing streams and nothing carries pending bytes.
pub fn encode(id: Identity, mono: &PreFrames, pair: &Stereo) -> Vec<u8> {
    fn put32(b: &mut [u8], at: usize, v: u32) {
        b[at..at + 4].copy_from_slice(&v.to_le_bytes());
    }
    fn put64(b: &mut [u8], at: usize, v: u64) {
        b[at..at + 8].copy_from_slice(&v.to_le_bytes());
    }

    let left = pair.left.to_v4();
    let right = pair.right.to_v4();
    let (left_len, right_len) = (left.len(), right.len());

    let mut out = Vec::with_capacity(HEADER + left_len + right_len);
    out.resize(HEADER, 0);
    out.extend_from_slice(&left);
    drop(left); // its bytes are in `out` now; holding it as well is the cost
    out.extend_from_slice(&right);
    drop(right);

    let payload = payload_fingerprint(&out[HEADER..]);
    let mono_fp = fingerprint(mono);

    let h = &mut out[..HEADER];
    put32(h, 0, MAGIC);
    put32(h, 4, VERSION);
    put32(h, 8, pair.len() as u32);
    put32(h, 12, pair.bars() as u32);
    put32(h, 16, FINGERPRINT_V1);
    put32(h, 20, 0); // reserved
    put64(h, 24, id.source);
    put64(h, 32, id.params);
    put64(h, 40, mono_fp);
    put64(h, 48, payload);
    put32(h, 56, left_len as u32);
    put32(h, 60, right_len as u32);
    out
}

fn u32_at(raw: &[u8], off: usize) -> usize {
    u32::from_le_bytes([raw[off], raw[off + 1], raw[off + 2], raw[off + 3]]) as usize
}

fn u64_at(raw: &[u8], off: usize) -> u64 {
    u64::from_le_bytes(raw[off..off + 8].try_into().expect("8 bytes"))
}

/// Decode a sidecar, checking it against the analysis it claims to accompany.
///
/// The checks run cheapest-first, and each is a different question:
///
/// 1. is this a sidecar at all, of a version and algorithm this build reads;
/// 2. is it the right *shape* — before anything is decompressed;
/// 3. is it of this **source**, at these **parameters**;
/// 4. is it of this **mono content**;
/// 5. are its own **bytes** intact;
/// 6. do the channels decode.
pub fn decode_with(
    raw: &[u8],
    id: Identity,
    mono: &PreFrames,
    max_bars: usize,
    limits: Limits,
) -> Result<Stereo, SidecarError> {
    if raw.len() < HEADER {
        return Err(SidecarError::TooShort { got: raw.len(), need: HEADER });
    }
    let magic = u32_at(raw, 0) as u32;
    if magic != MAGIC {
        return Err(SidecarError::BadMagic(magic));
    }
    let version = u32_at(raw, 4) as u32;
    if version != VERSION {
        return Err(SidecarError::BadVersion(version));
    }
    let algo = u32_at(raw, 16) as u32;
    if algo != FINGERPRINT_V1 {
        return Err(SidecarError::BadFingerprintAlgo(algo));
    }

    // Shape before contents: a sidecar for a differently shaped analysis is
    // refused here, before a byte of it is decompressed.
    let frames = u32_at(raw, 8);
    let bars = u32_at(raw, 12);
    if frames != mono.len() {
        return Err(SidecarError::Mismatch {
            what: "frames",
            mono: mono.len(),
            sidecar: frames,
        });
    }
    if bars != mono.bars() {
        return Err(SidecarError::Mismatch {
            what: "bars",
            mono: mono.bars(),
            sidecar: bars,
        });
    }

    // Source and settings, which mono content cannot establish on its own.
    for (what, want, got) in [
        ("source", id.source, u64_at(raw, 24)),
        ("parameter set", id.params, u64_at(raw, 32)),
    ] {
        if want != got {
            return Err(SidecarError::WrongSource { what, want, got });
        }
    }

    // The mono cache's own contents, in full.
    let want_mono = u64_at(raw, 40);
    let got_mono = fingerprint(mono);
    if want_mono != got_mono {
        return Err(SidecarError::DifferentMonoContent { want: want_mono, got: got_mono });
    }

    let left_len = u32_at(raw, 56);
    let right_len = u32_at(raw, 60);
    let declared = HEADER
        .checked_add(left_len)
        .and_then(|n| n.checked_add(right_len))
        .ok_or(SidecarError::Truncated { declared: usize::MAX, got: raw.len() })?;
    if declared > raw.len() {
        return Err(SidecarError::Truncated { declared, got: raw.len() });
    }

    // The bytes themselves, before anything trusts their contents.
    let want_payload = u64_at(raw, 48);
    let got_payload = payload_fingerprint(&raw[HEADER..declared]);
    if want_payload != got_payload {
        return Err(SidecarError::Corrupt { want: want_payload, got: got_payload });
    }

    let left = cache::decode_with(&raw[HEADER..HEADER + left_len], bars, max_bars, limits)
        .map_err(|err| SidecarError::Channel { side: Side::Left, err })?;
    let right = cache::decode_with(
        &raw[HEADER + left_len..declared],
        bars,
        max_bars,
        limits,
    )
    .map_err(|err| SidecarError::Channel { side: Side::Right, err })?;

    // The header said what shape to expect; the payloads have to agree with it,
    // or the header was describing something other than what it carried.
    for (side, ch) in [(Side::Left, &left), (Side::Right, &right)] {
        if ch.len() != frames {
            return Err(SidecarError::Channel {
                side,
                err: CacheError::LengthMismatch { declared: frames, actual: ch.len() },
            });
        }
    }
    Stereo::new(left, right).ok_or(SidecarError::Mismatch {
        what: "channels disagree with each other",
        mono: frames,
        sidecar: 0,
    })
}

/// Read a sidecar under the default limits, checked against `mono` and `id`.
pub fn read(
    path: &Path, id: Identity, mono: &PreFrames, max_bars: usize,
) -> Result<Stereo, SidecarError> {
    read_with(path, id, mono, max_bars, Limits::default())
}

/// Read under explicit limits.
///
/// The file is read through the same bounded path a mono cache is, so a sidecar
/// cannot be the way an oversized file gets into memory.
pub fn read_with(
    path: &Path,
    id: Identity,
    mono: &PreFrames,
    max_bars: usize,
    limits: Limits,
) -> Result<Stereo, SidecarError> {
    let raw = cache::read_bounded(path, limits).map_err(SidecarError::Io)?;
    decode_with(&raw, id, mono, max_bars, limits)
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    const BARS: usize = 4096;

    /// The identity most tests are not about. Real values come from the cache
    /// key; here they only have to be consistent.
    const ID: Identity = Identity { source: 0xA11CE, params: 0xB0B };

    fn rows(frames: usize, bars: usize, seed: u64) -> Vec<Vec<f32>> {
        // Mixed, not used raw: `seed | 1` turns 2 and 3 into the same stream,
        // which quietly made an earlier version of this file test a left and
        // right that were identical.
        let mut s = seed.wrapping_mul(0x9E37_79B9_7F4A_7C15) | 1;
        (0..frames)
            .map(|_| {
                (0..bars)
                    .map(|_| {
                        s ^= s << 13;
                        s ^= s >> 7;
                        s ^= s << 17;
                        (s >> 40) as f32 / 16_777_216.0
                    })
                    .collect()
            })
            .collect()
    }

    fn trio(frames: usize, bars: usize) -> (PreFrames, Stereo) {
        let mono = PreFrames::from_analysis(&rows(frames, bars, 1));
        let left = PreFrames::from_analysis(&rows(frames, bars, 2));
        let right = PreFrames::from_analysis(&rows(frames, bars, 3));
        (mono, Stereo::new(left, right).expect("same shape"))
    }

    /// The file is byte-identical to the build-then-concatenate form.
    ///
    /// The reference below is the shape `encode` had before it laid the file
    /// out in place: two channel payloads, concatenated, fingerprinted, and
    /// copied in behind a header. It exists only here, and it is the thing the
    /// new code has to reproduce exactly — header, fingerprints and payload.
    ///
    /// **Every left/right boundary remainder is covered.** `payload_fingerprint`
    /// folds eight bytes at a time and then a remainder, so the byte at which
    /// the left payload ends decides which word straddles the boundary and what
    /// the trailing remainder is. The shapes below are chosen to hit all eight
    /// values of `left_len % 8`, and the test fails if any is missed rather than
    /// passing quietly on seven.
    #[test]
    fn the_sidecar_is_byte_identical_to_the_concatenated_form() {
        fn reference(id: Identity, mono: &PreFrames, pair: &Stereo) -> Vec<u8> {
            let left = pair.left().to_v4();
            let right = pair.right().to_v4();
            let mut payload = Vec::with_capacity(left.len() + right.len());
            payload.extend_from_slice(&left);
            payload.extend_from_slice(&right);

            let mut out = Vec::with_capacity(HEADER + payload.len());
            out.extend_from_slice(&MAGIC.to_le_bytes());
            out.extend_from_slice(&VERSION.to_le_bytes());
            out.extend_from_slice(&(pair.len() as u32).to_le_bytes());
            out.extend_from_slice(&(pair.bars() as u32).to_le_bytes());
            out.extend_from_slice(&FINGERPRINT_V1.to_le_bytes());
            out.extend_from_slice(&0u32.to_le_bytes());
            out.extend_from_slice(&id.source.to_le_bytes());
            out.extend_from_slice(&id.params.to_le_bytes());
            out.extend_from_slice(&fingerprint(mono).to_le_bytes());
            out.extend_from_slice(&payload_fingerprint(&payload).to_le_bytes());
            out.extend_from_slice(&(left.len() as u32).to_le_bytes());
            out.extend_from_slice(&(right.len() as u32).to_le_bytes());
            out.extend_from_slice(&payload);
            out
        }

        let mut seen = [false; 8];
        let mut checked = 0usize;
        // Enough shapes and contents that the compressed left payload lands on
        // every remainder; each is also checked for byte equality regardless.
        for frames in [1usize, 2, 3, 5, 7, 8, 11, 13, 17, 23, 31, 40, 64, 97] {
            for bars in [1usize, 3, 7, 8, 16, 33, 64, 65] {
                let mono = PreFrames::from_analysis(&rows(frames, bars, 1));
                let left = PreFrames::from_analysis(&rows(frames, bars, 2));
                let right = PreFrames::from_analysis(&rows(frames, bars, 3));
                let pair = Stereo::new(left, right).expect("same shape");

                let got = encode(ID, &mono, &pair);
                let want = reference(ID, &mono, &pair);
                assert_eq!(got, want, "{frames}x{bars}: the file bytes changed");

                let left_len = u32_at(&got, 56);
                seen[left_len % 8] = true;
                checked += 1;

                // And it still reads back as itself.
                let back = decode_with(&got, ID, &mono, BARS, Limits::default())
                    .unwrap_or_else(|e| panic!("{frames}x{bars}: {e}"));
                assert_eq!(back, pair, "{frames}x{bars}");
            }
        }
        assert!(checked >= 100, "setup: only {checked} shapes");
        let missed: Vec<usize> = (0..8).filter(|&i| !seen[i]).collect();
        assert!(
            missed.is_empty(),
            "left/right boundary remainders not covered: {missed:?}",
        );
    }

    /// A damaged payload is still caught, wherever the damage lands.
    ///
    /// Laying the file out in place means the fingerprint is taken over the
    /// output buffer rather than over a separate payload vector. This holds it
    /// to the same job: a flipped bit anywhere in either channel, including the
    /// bytes either side of the boundary between them, is refused as corrupt
    /// rather than decoded.
    #[test]
    fn a_bit_flipped_anywhere_in_the_payload_is_refused() {
        let (mono, pair) = trio(40, 64);
        let blob = encode(ID, &mono, &pair);
        let left_len = u32_at(&blob, 56);
        let boundary = HEADER + left_len;
        let spots = [
            HEADER,                 // first byte of left
            HEADER + 1,
            boundary - 8,           // the word that straddles the boundary
            boundary - 1,           // last byte of left
            boundary,               // first byte of right
            boundary + 1,
            blob.len() - 8,
            blob.len() - 1,         // the trailing remainder
        ];
        for at in spots {
            let mut bad = blob.clone();
            bad[at] ^= 0x01;
            let got = decode_with(&bad, ID, &mono, BARS, Limits::default());
            assert!(
                matches!(got, Err(SidecarError::Corrupt { .. })),
                "a bit flipped at {at} was not refused as corrupt: {got:?}",
            );
        }
    }

    #[test]
    fn a_sidecar_round_trips_beside_its_mono_cache() {
        let (mono, pair) = trio(40, 64);
        let blob = encode(ID, &mono, &pair);
        let back = decode_with(&blob, ID, &mono, BARS, Limits::default()).expect("refused");
        assert_eq!(back, pair);
        assert_ne!(back.left(), back.right(), "setup: the channels must differ");
    }

    #[test]
    fn every_awkward_shape_round_trips() {
        for (frames, bars) in [(1, 1), (1, 512), (3, 7), (129, 64), (64, 129)] {
            let (mono, pair) = trio(frames, bars);
            let blob = encode(ID, &mono, &pair);
            let back = decode_with(&blob, ID, &mono, BARS, Limits::default())
                .unwrap_or_else(|e| panic!("{frames}x{bars}: {e}"));
            assert_eq!(back, pair, "{frames}x{bars}");
        }
    }

    #[test]
    fn dimensions_are_checked_before_the_payload_is_touched() {
        let (mono, pair) = trio(40, 64);
        let blob = encode(ID, &mono, &pair);

        let short = PreFrames::from_analysis(&rows(39, 64, 1));
        assert_eq!(
            decode_with(&blob, ID, &short, BARS, Limits::default()),
            Err(SidecarError::Mismatch { what: "frames", mono: 39, sidecar: 40 })
        );

        let narrow = PreFrames::from_analysis(&rows(40, 63, 1));
        assert_eq!(
            decode_with(&blob, ID, &narrow, BARS, Limits::default()),
            Err(SidecarError::Mismatch { what: "bars", mono: 63, sidecar: 64 })
        );
    }

    #[test]
    fn a_truncated_sidecar_is_refused_rather_than_half_read() {
        let (mono, pair) = trio(40, 64);
        let blob = encode(ID, &mono, &pair);
        for cut in [0, 8, HEADER - 1, HEADER, HEADER + 4, blob.len() - 1] {
            let got = decode_with(&blob[..cut], ID, &mono, BARS, Limits::default());
            assert!(got.is_err(), "a sidecar cut to {cut} bytes decoded");
        }
    }

    #[test]
    fn a_foreign_file_is_refused_by_magic_and_version() {
        let (mono, pair) = trio(8, 16);
        let mut blob = encode(ID, &mono, &pair);

        let mut foreign = blob.clone();
        foreign[..4].copy_from_slice(&0xDEAD_BEEFu32.to_le_bytes());
        assert_eq!(
            decode_with(&foreign, ID, &mono, BARS, Limits::default()),
            Err(SidecarError::BadMagic(0xDEAD_BEEF))
        );

        // A mono cache is not a sidecar, and says so rather than being parsed.
        let mono_blob = mono.to_v4();
        assert!(matches!(
            decode_with(&mono_blob, ID, &mono, BARS, Limits::default()),
            Err(SidecarError::BadMagic(_))
        ));

        blob[4..8].copy_from_slice(&7u32.to_le_bytes());
        assert_eq!(
            decode_with(&blob, ID, &mono, BARS, Limits::default()),
            Err(SidecarError::BadVersion(7))
        );
    }

    #[test]
    fn a_declared_length_past_the_file_is_refused() {
        let (mono, pair) = trio(16, 32);
        let mut blob = encode(ID, &mono, &pair);
        let len = blob.len();
        // The left length, at its documented offset.
        blob[56..60].copy_from_slice(&(u32::MAX / 2).to_le_bytes());
        match decode_with(&blob, ID, &mono, BARS, Limits::default()) {
            Err(SidecarError::Truncated { declared, got }) => {
                assert!(declared > len);
                assert_eq!(got, len);
            }
            got => panic!("expected a refusal, got {got:?}"),
        }
    }

    /// Damage inside a channel is caught by the payload check, before anything
    /// tries to decode it.
    #[test]
    fn damage_inside_a_channel_is_caught_by_the_payload_check() {
        let (mono, pair) = trio(16, 32);
        let blob = encode(ID, &mono, &pair);
        let left_len = u32_at(&blob, 56);

        for (what, at) in [("left", HEADER + left_len - 3), ("right", blob.len() - 3)] {
            let mut hurt = blob.clone();
            hurt[at] ^= 0xFF;
            assert!(
                matches!(
                    decode_with(&hurt, ID, &mono, BARS, Limits::default()),
                    Err(SidecarError::Corrupt { .. })
                ),
                "{what}: damaged bytes were not caught"
            );
        }
    }

    /// A file that is internally consistent but holds an undecodable channel
    /// names which one.
    ///
    /// The payload check cannot catch this: the hash was computed over the
    /// damaged bytes, so they are exactly the bytes that were written. Only the
    /// decoder can tell, and it has to say which side it was.
    #[test]
    fn an_undecodable_channel_names_which_one() {
        let (mono, pair) = trio(16, 32);
        let blob = encode(ID, &mono, &pair);
        let left_len = u32_at(&blob, 56);

        for (side, at) in [(Side::Left, HEADER + left_len - 3), (Side::Right, blob.len() - 3)] {
            let mut hurt = blob.clone();
            hurt[at] ^= 0xFF;
            // Re-stamp the payload hash, so the file agrees with itself.
            let declared = hurt.len();
            let fixed = payload_fingerprint(&hurt[HEADER..declared]);
            hurt[48..56].copy_from_slice(&fixed.to_le_bytes());

            // lz4 sometimes decodes damaged input to the right length, so this
            // may or may not be detected — but a wrong answer must never be
            // silently accepted as the *other* channel.
            if let Err(SidecarError::Channel { side: got, .. }) =
                decode_with(&hurt, ID, &mono, BARS, Limits::default())
            {
                assert_eq!(got, side, "the wrong channel was blamed");
            }
        }
    }

    #[test]
    fn the_fingerprint_moves_when_the_analysis_does() {
        let a = PreFrames::from_analysis(&rows(64, 128, 1));
        let b = PreFrames::from_analysis(&rows(64, 128, 2));
        assert_ne!(fingerprint(&a), fingerprint(&b));

        // Same numbers, same fingerprint: it must be a function of the content,
        // or a legitimate sidecar would be refused after a harmless reload.
        let a2 = PreFrames::from_analysis(&rows(64, 128, 1));
        assert_eq!(fingerprint(&a), fingerprint(&a2));

        // Shape is folded in first, so a differently shaped analysis cannot
        // collide with a differently filled one.
        assert_ne!(fingerprint(&a), fingerprint(&PreFrames::from_analysis(&rows(63, 128, 1))));
        assert_ne!(fingerprint(&a), fingerprint(&PreFrames::from_analysis(&rows(64, 127, 1))));
    }

    /// The gap the sampled digest left: a change confined to cells it did not
    /// look at. Every cell is folded in now, so there is no such cell.
    #[test]
    fn the_fingerprint_sees_a_change_in_any_single_cell() {
        let base = rows(6, 300, 1);
        let before = fingerprint(&PreFrames::from_analysis(&base));
        // Includes cells the old one-in-64 stride skipped by construction.
        for (row, bar) in [(0usize, 1usize), (0, 63), (2, 130), (5, 299), (3, 7)] {
            let mut r = base.clone();
            r[row][bar] = if r[row][bar] > 0.5 { 0.1 } else { 0.9 };
            assert_ne!(
                before,
                fingerprint(&PreFrames::from_analysis(&r)),
                "a change at row {row}, bar {bar} was invisible to the fingerprint"
            );
        }
    }

    /// Four identities, four refusals, each naming what disagreed.
    #[test]
    fn a_sidecar_is_bound_to_source_settings_content_and_its_own_bytes() {
        let (mono, pair) = trio(40, 64);
        let blob = encode(ID, &mono, &pair);
        assert!(decode_with(&blob, ID, &mono, BARS, Limits::default()).is_ok());

        // Another file, same settings.
        match decode_with(&blob, Identity { source: 99, ..ID }, &mono, BARS, Limits::default()) {
            Err(SidecarError::WrongSource { what, .. }) => assert_eq!(what, "source"),
            got => panic!("expected a source refusal, got {got:?}"),
        }
        // Same file, other settings.
        match decode_with(&blob, Identity { params: 99, ..ID }, &mono, BARS, Limits::default()) {
            Err(SidecarError::WrongSource { what, .. }) => assert_eq!(what, "parameter set"),
            got => panic!("expected a settings refusal, got {got:?}"),
        }
        // Same file, same settings, another analysis of it.
        let other = PreFrames::from_analysis(&rows(40, 64, 99));
        assert_eq!(other.len(), mono.len());
        assert!(matches!(
            decode_with(&blob, ID, &other, BARS, Limits::default()),
            Err(SidecarError::DifferentMonoContent { .. })
        ));
    }

    /// Mono content alone cannot identify a stereo source, so the check does
    /// not rest on it.
    ///
    /// Two different stereo pairs can share a mono mix exactly — a track and
    /// its own channel-swapped copy is the trivial case. A sidecar bound only
    /// to mono content would accept the wrong one of those.
    #[test]
    fn identical_mono_does_not_make_two_sources_interchangeable() {
        let (mono, pair) = trio(32, 48);
        // The swapped pair is a different analysis of a different arrangement,
        // and its mono mix is identical by construction.
        let swapped = Stereo::new(pair.right().clone(), pair.left().clone()).unwrap();
        assert_ne!(swapped, pair);

        let from_a = encode(Identity { source: 1, params: 7 }, &mono, &pair);
        let from_b = encode(Identity { source: 2, params: 7 }, &mono, &swapped);

        // Both agree with the same mono, and the mono fingerprint cannot tell
        // them apart — the source does.
        assert_eq!(u64_at(&from_a, 40), u64_at(&from_b, 40));
        let want_a = Identity { source: 1, params: 7 };
        assert_eq!(decode_with(&from_a, want_a, &mono, BARS, Limits::default()).unwrap(), pair);
        assert!(matches!(
            decode_with(&from_b, want_a, &mono, BARS, Limits::default()),
            Err(SidecarError::WrongSource { what: "source", .. })
        ));
    }

    /// A payload that still decodes, but is not the payload that was written.
    #[test]
    fn a_damaged_payload_is_caught_even_when_it_would_still_decode() {
        let (mono, pair) = trio(24, 48);
        let mut blob = encode(ID, &mono, &pair);
        let last = blob.len() - 1;
        blob[last] ^= 0x01;
        match decode_with(&blob, ID, &mono, BARS, Limits::default()) {
            Err(SidecarError::Corrupt { want, got }) => assert_ne!(want, got),
            got => panic!("expected a corruption refusal, got {got:?}"),
        }
    }

    /// An older sidecar is refused rather than misread, and nothing migrates.
    #[test]
    fn a_previous_format_version_is_refused() {
        let (mono, pair) = trio(16, 32);
        let mut blob = encode(ID, &mono, &pair);
        blob[4..8].copy_from_slice(&1u32.to_le_bytes());
        assert_eq!(
            decode_with(&blob, ID, &mono, BARS, Limits::default()),
            Err(SidecarError::BadVersion(1))
        );

        // And an identity computed by an algorithm this build does not have.
        let mut other = encode(ID, &mono, &pair);
        other[16..20].copy_from_slice(&7u32.to_le_bytes());
        assert_eq!(
            decode_with(&other, ID, &mono, BARS, Limits::default()),
            Err(SidecarError::BadFingerprintAlgo(7))
        );
    }

    #[test]
    fn the_sidecar_sits_beside_the_cache_it_belongs_to() {
        let mono = Path::new("/tmp/.moosik/cache/abc_b1024.spectrumcache");
        let side = path_for(mono);
        assert_eq!(side.extension().unwrap(), "stereocache");
        assert_eq!(side.file_stem(), mono.file_stem());
        assert_eq!(side.parent(), mono.parent());
        assert_ne!(side, mono.to_path_buf());
    }

    #[test]
    fn mismatched_channels_cannot_be_paired() {
        let l = PreFrames::from_analysis(&rows(8, 16, 1));
        let r = PreFrames::from_analysis(&rows(9, 16, 2));
        assert!(Stereo::new(l, r).is_none());

        let l = PreFrames::from_analysis(&rows(8, 16, 1));
        let r = PreFrames::from_analysis(&rows(8, 15, 2));
        assert!(Stereo::new(l, r).is_none());
    }

    #[test]
    fn a_written_sidecar_reads_back_through_the_bounded_path() {
        let dir = std::env::temp_dir().join(format!(
            "moosik_sidecar_{}_{}",
            std::process::id(),
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .map(|d| d.as_nanos())
                .unwrap_or(0),
        ));
        std::fs::create_dir_all(&dir).unwrap();
        let mono_path = dir.join("t.spectrumcache");
        let (mono, pair) = trio(32, 64);
        cache::write_atomic(&mono_path, &mono.to_v4()).unwrap();

        let side_path = path_for(&mono_path);
        cache::write_atomic(&side_path, &encode(ID, &mono, &pair)).unwrap();

        let loaded = cache::read(&mono_path, 64, BARS).unwrap();
        assert_eq!(loaded, mono);
        assert_eq!(read(&side_path, ID, &loaded, BARS).unwrap(), pair);

        // An oversized sidecar is refused by the same file budget a mono cache
        // is, and refused before it is read.
        let on_disk = std::fs::metadata(&side_path).unwrap().len() as usize;
        let tiny = Limits { max_file_bytes: on_disk - 1, ..Limits::default() };
        assert!(matches!(
            read_with(&side_path, ID, &loaded, BARS, tiny),
            Err(SidecarError::Io(CacheError::TooLarge { what: "file", .. }))
        ));

        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn clearing_a_pair_releases_both_allocations() {
        let (_, mut pair) = trio(500, 512);
        assert!(
            pair.capacity_bytes() >= 2 * 500 * 512 * 2,
            "setup: expected two real allocations"
        );
        pair.clear();
        assert_eq!(pair.capacity_bytes(), 0, "a cleared pair kept its buffers");
        assert!(pair.is_empty());
        assert_eq!(pair.len(), 0);
        assert!(pair.row(0).is_none());
    }
}
