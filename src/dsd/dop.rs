//! DoP (DSD over PCM, spec 1.1) encoding — Phase 1 of DSD support.
//!
//! DoP carries the untouched 1-bit DSD stream inside 24-bit PCM samples so it
//! survives any bit-transparent PCM path: each 24-bit word holds a marker
//! byte in bits 23–16 (alternating `0x05`/`0xFA` on successive words of the
//! same channel, synchronised across channels) and 16 DSD bits below it —
//! the older byte in bits 15–8, the newer in bits 7–0, MSB = oldest sample.
//! A DoP-aware DAC spots the marker sequence and unpacks the original bits;
//! anything else just plays very quiet high-frequency noise.
//!
//! The PCM carrier runs at `dsd_rate / 16` (DSD64 → 176.4 kHz, DSD128 →
//! 352.8 kHz, DSD256 → 705.6 kHz), so bit-perfect DoP only needs a device
//! that accepts ≥24-bit PCM at that rate.
//!
//! Input is the [`DsdReader`](super::DsdReader) byte stream: MSB-first,
//! channel-interleaved — exactly what both containers are normalised to.

use std::fs::File;
use std::io::{BufReader, Read, Seek};
use std::path::Path;

use super::DsdReader;

/// Convert one 24-bit DoP word (`0..=0xFF_FFFF`) to the ±1-normalised f32 the
/// bit-perfect ring buffer carries, by treating it as a signed 24-bit
/// two's-complement integer. A 24-bit integer is always exactly representable
/// in f32 (24 bits of mantissa precision) and — not by accident — the
/// existing power-of-two device writers (`× 2^23` for a 24-bit container,
/// `× 2^31` for 24-in-32/32-bit, which is the same value left-shifted 8 bits)
/// already round-trip a value scaled this way bit-exactly, so a DoP session
/// reuses the whole existing sample pipeline unchanged.
///
/// **Never** let this reach a float (32f) device format: the DAC must see the
/// literal integer bit pattern the marker bytes were packed into, not an
/// IEEE-754 encoding of the same number — [`DopPacker`]'s callers must
/// restrict device format negotiation to integer PCM only.
#[inline]
pub fn word_to_f32(word: u32) -> f32 {
    let signed = ((word << 8) as i32) >> 8; // sign-extend the low 24 bits
    signed as f32 / 8_388_608.0
}

/// The two alternating DoP marker bytes. The spec allows starting with
/// either; we start with `0x05`.
pub const DOP_MARKERS: [u8; 2] = [0x05, 0xFA];

/// DSD "silence": a repeating `01101001…` pattern that idles a sigma-delta
/// DAC near zero (all-zero bits would be full-scale negative DC). Used to pad
/// a stream that ends on an odd DSD byte.
pub const DSD_SILENCE: u8 = 0x69;

/// PCM carrier rate for a DSD rate: 16 DSD bits ride in each 24-bit sample.
pub fn carrier_rate(dsd_rate: u32) -> u32 {
    dsd_rate / 16
}

/// Build one DoP word from a marker and two DSD bytes (older first).
/// The 24-bit word sits in the low bits of the returned `u32`.
#[inline]
pub fn dop_word(marker: u8, older: u8, newer: u8) -> u32 {
    (marker as u32) << 16 | (older as u32) << 8 | newer as u32
}

/// Stateful packer: turns the interleaved DSD byte stream into interleaved
/// 24-bit DoP words, keeping the marker phase across calls.
pub struct DopPacker {
    channels: usize,
    /// Index into [`DOP_MARKERS`] for the next PCM frame.
    phase: usize,
}

impl DopPacker {
    pub fn new(channels: usize) -> Self {
        assert!(channels > 0);
        DopPacker { channels, phase: 0 }
    }

    pub fn channels(&self) -> usize {
        self.channels
    }

    /// The marker phase the *next* packed frame will use (0 = `0x05`,
    /// 1 = `0xFA`). Used to carry alternation across a gapless file swap so
    /// the DAC never sees two identical markers in a row at the boundary.
    pub fn phase(&self) -> usize {
        self.phase
    }

    /// Adopt a marker phase (e.g. continue from the previous file's packer).
    pub fn set_phase(&mut self, phase: usize) {
        self.phase = phase & 1;
    }

    /// Pack interleaved DSD bytes (2 DSD frames → 1 PCM frame) into `out`.
    /// `dsd.len()` must be a multiple of `2 × channels`; use
    /// [`pack_padded`](Self::pack_padded) for a stream tail. Returns the
    /// number of PCM frames appended.
    pub fn pack(&mut self, dsd: &[u8], out: &mut Vec<u32>) -> usize {
        let ch = self.channels;
        assert!(dsd.len() % (2 * ch) == 0, "partial PCM frame — use pack_padded for tails");
        let frames = dsd.len() / (2 * ch);
        out.reserve(frames * ch);
        for f in 0..frames {
            let marker = DOP_MARKERS[self.phase];
            self.phase ^= 1;
            let base = f * 2 * ch;
            for c in 0..ch {
                out.push(dop_word(marker, dsd[base + c], dsd[base + ch + c]));
            }
        }
        frames
    }

    /// Like [`pack`](Self::pack), but a trailing odd DSD frame is completed
    /// with [`DSD_SILENCE`] so the last 16-bit slot is full.
    pub fn pack_padded(&mut self, dsd: &[u8], out: &mut Vec<u32>) -> usize {
        let ch = self.channels;
        assert!(dsd.len() % ch == 0, "input must be whole DSD frames");
        let whole = dsd.len() / (2 * ch) * (2 * ch);
        let mut n = self.pack(&dsd[..whole], out);
        if whole < dsd.len() {
            let marker = DOP_MARKERS[self.phase];
            self.phase ^= 1;
            for c in 0..ch {
                out.push(dop_word(marker, dsd[whole + c], DSD_SILENCE));
            }
            n += 1;
        }
        n
    }

    /// Append `frames` PCM frames of DoP-encoded DSD silence (device warm-up,
    /// stream tail, or gap filler that keeps the DAC locked in DSD mode).
    pub fn pack_silence(&mut self, frames: usize, out: &mut Vec<u32>) -> usize {
        out.reserve(frames * self.channels);
        for _ in 0..frames {
            let marker = DOP_MARKERS[self.phase];
            self.phase ^= 1;
            for _ in 0..self.channels {
                out.push(dop_word(marker, DSD_SILENCE, DSD_SILENCE));
            }
        }
        frames
    }
}

/// A [`DsdReader`] + [`DopPacker`] pair: pulls the container's bit stream and
/// yields interleaved 24-bit DoP words ready for a 24-bit PCM device. This is
/// the unit the bit-perfect output path consumes in Phase 2.
pub struct DopStream<R: Read + Seek> {
    reader: DsdReader<R>,
    packer: DopPacker,
    /// Scratch for the raw DSD bytes of one read.
    buf: Vec<u8>,
}

impl<R: Read + Seek> DopStream<R> {
    pub fn new(reader: DsdReader<R>) -> Self {
        let channels = reader.info().channels as usize;
        DopStream { reader, packer: DopPacker::new(channels), buf: Vec::new() }
    }

    pub fn info(&self) -> &super::DsdInfo {
        self.reader.info()
    }

    /// PCM carrier rate of this stream.
    pub fn carrier_rate(&self) -> u32 {
        carrier_rate(self.reader.info().sample_rate)
    }

    /// Total PCM frames the stream will yield (including a padded tail).
    #[allow(dead_code)] // for a future progress/duration readout on the DoP path
    pub fn total_pcm_frames(&self) -> u64 {
        self.reader.total_frames().div_ceil(2)
    }

    /// The marker phase the next packed frame will use — read it off a
    /// finishing stream to seed the next one for gapless continuity.
    pub fn marker_phase(&self) -> usize {
        self.packer.phase()
    }

    /// Continue the marker alternation from `phase` (the value the previous
    /// gapless file left off at), so the boundary word alternates correctly.
    pub fn set_marker_phase(&mut self, phase: usize) {
        self.packer.set_phase(phase);
    }

    /// Append `frames` PCM frames of DoP-encoded DSD silence, advancing the
    /// marker phase so real audio packed afterwards stays continuous. Used to
    /// warm the DAC into DSD lock before the first audio of a session.
    pub fn warmup_silence(&mut self, frames: usize, out: &mut Vec<u32>) -> usize {
        self.packer.pack_silence(frames, out)
    }

    /// Read up to `max_frames` PCM frames of DoP words into `out` (appended,
    /// interleaved). Returns frames appended; 0 = end of stream.
    pub fn read_dop(&mut self, max_frames: usize, out: &mut Vec<u32>) -> std::io::Result<usize> {
        let ch = self.packer.channels();
        self.buf.clear();
        self.buf.resize(max_frames * 2 * ch, 0);
        // Fill as much as the reader will give — it may return short reads
        // near block boundaries, but only whole DSD frames.
        let mut got = 0usize;
        while got < max_frames * 2 {
            let n = self.reader.read_frames(&mut self.buf[got * ch..])?;
            if n == 0 {
                break;
            }
            got += n;
        }
        Ok(self.packer.pack_padded(&self.buf[..got * ch], out))
    }

    /// Seek to a PCM-frame position (each PCM frame = 16 DSD samples). The
    /// marker phase restarts, which any DoP decoder re-locks onto within a
    /// few samples.
    pub fn seek_to_pcm_frame(&mut self, frame: u64) -> std::io::Result<()> {
        self.reader.seek_to_frame(frame * 2)
    }
}

pub type DopFileStream = DopStream<BufReader<File>>;

/// Open a DSD file and wrap it as a ready-to-read DoP word stream.
pub fn open_dop_stream(path: &Path) -> Result<DopFileStream, String> {
    Ok(DopStream::new(super::open_reader(path)?))
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::super::{parse_dsf, tests::make_dsf, DSD64_RATE};
    use super::*;
    use std::io::Cursor;

    #[test]
    fn carrier_rates() {
        assert_eq!(carrier_rate(2_822_400), 176_400); // DSD64
        assert_eq!(carrier_rate(5_644_800), 352_800); // DSD128
        assert_eq!(carrier_rate(11_289_600), 705_600); // DSD256
        assert_eq!(carrier_rate(22_579_200), 1_411_200); // DSD512
    }

    #[test]
    fn word_layout() {
        // Marker in bits 23-16, older DSD byte in 15-8, newer in 7-0.
        assert_eq!(dop_word(0x05, 0xAB, 0xCD), 0x05ABCD);
        assert_eq!(dop_word(0xFA, 0x00, 0xFF), 0xFA00FF);
    }

    #[test]
    fn markers_alternate_and_channels_share_them() {
        let mut p = DopPacker::new(2);
        let mut out = Vec::new();
        // 4 DSD frames of 2ch = 2 PCM frames.
        let dsd = [0x11, 0x21, 0x12, 0x22, 0x13, 0x23, 0x14, 0x24];
        assert_eq!(p.pack(&dsd, &mut out), 2);
        assert_eq!(out, vec![
            0x05_11_12, 0x05_21_22, // frame 0: both channels marked 0x05
            0xFA_13_14, 0xFA_23_24, // frame 1: both channels marked 0xFA
        ]);
        // Phase carries across calls.
        assert_eq!(p.pack(&dsd[..4], &mut out), 1);
        assert_eq!(out[4], 0x05_11_12);
    }

    #[test]
    fn round_trip_preserves_every_dsd_bit() {
        let channels = 2;
        let dsd: Vec<u8> = (0..64u16).map(|i| (i * 37 + 11) as u8).collect();
        let mut p = DopPacker::new(channels);
        let mut words = Vec::new();
        p.pack(&dsd, &mut words);

        // Decode: strip markers, re-interleave the two byte lanes.
        let mut decoded = Vec::new();
        for pcm_frame in words.chunks(channels) {
            for w in pcm_frame {
                assert!(matches!((w >> 16) as u8, 0x05 | 0xFA));
            }
            for w in pcm_frame { decoded.push((w >> 8) as u8); } // older bytes
            for w in pcm_frame { decoded.push(*w as u8); }       // newer bytes
        }
        assert_eq!(decoded, dsd);
    }

    #[test]
    fn odd_tail_padded_with_dsd_silence() {
        let mut p = DopPacker::new(2);
        let mut out = Vec::new();
        // 3 DSD frames — the last PCM frame gets a silence byte.
        let dsd = [0x11, 0x21, 0x12, 0x22, 0x13, 0x23];
        assert_eq!(p.pack_padded(&dsd, &mut out), 2);
        assert_eq!(out, vec![
            0x05_11_12, 0x05_21_22,
            dop_word(0xFA, 0x13, DSD_SILENCE), dop_word(0xFA, 0x23, DSD_SILENCE),
        ]);
    }

    #[test]
    fn silence_frames_keep_marker_phase() {
        let mut p = DopPacker::new(1);
        let mut out = Vec::new();
        p.pack_silence(2, &mut out);
        assert_eq!(out, vec![0x05_69_69, 0xFA_69_69]);
        p.pack(&[0xAA, 0xBB], &mut out);
        assert_eq!(out[2], 0x05_AA_BB); // phase continued
    }

    #[test]
    fn dop_stream_end_to_end_over_dsf() {
        // 2ch DSF, block size 4, 2 block-rows = 8 DSD frames = 4 PCM frames.
        // DSF is LSB-first, so the packed bytes are the bit-reversed input.
        let blocks = vec![
            vec![vec![0x01, 0x02, 0x03, 0x04], vec![0x11, 0x12, 0x13, 0x14]],
            vec![vec![0x05, 0x06, 0x07, 0x08], vec![0x15, 0x16, 0x17, 0x18]],
        ];
        let file = make_dsf(2, DSD64_RATE, 1, 64, 4, &blocks, None);
        let info = parse_dsf(&mut Cursor::new(&file)).unwrap();
        let reader = DsdReader::new(Cursor::new(&file), info).unwrap();
        let mut s = DopStream::new(reader);
        assert_eq!(s.carrier_rate(), 176_400);
        assert_eq!(s.total_pcm_frames(), 4);

        let mut words = Vec::new();
        // Ask for more than exists — should deliver exactly 4 then 0.
        assert_eq!(s.read_dop(16, &mut words).unwrap(), 4);
        assert_eq!(s.read_dop(16, &mut words).unwrap(), 0);

        let r = |b: u8| b.reverse_bits();
        assert_eq!(words, vec![
            dop_word(0x05, r(0x01), r(0x02)), dop_word(0x05, r(0x11), r(0x12)),
            dop_word(0xFA, r(0x03), r(0x04)), dop_word(0xFA, r(0x13), r(0x14)),
            dop_word(0x05, r(0x05), r(0x06)), dop_word(0x05, r(0x15), r(0x16)),
            dop_word(0xFA, r(0x07), r(0x08)), dop_word(0xFA, r(0x17), r(0x18)),
        ]);
    }

    /// Confirms the "reuse the existing power-of-two device writers" claim:
    /// word -> f32 -> the exact arithmetic `fill_bytes`/`write_data` use for a
    /// 24-bit container (`× 2^23`) and for 24-in-32/32-bit (`× 2^31`) must
    /// both reproduce the original word bit-exactly, for every marker paired
    /// with every edge-case DSD byte.
    #[test]
    fn word_to_f32_round_trips_through_device_scaling() {
        for &marker in &DOP_MARKERS {
            for &b in &[0x00u8, 0x01, 0x7F, 0x80, 0xFE, 0xFF] {
                for &c in &[0x00u8, 0x01, 0x7F, 0x80, 0xFE, 0xFF] {
                    let word = dop_word(marker, b, c);
                    let v = word_to_f32(word);

                    // 24-bit container path (wasapi_out::fill_bytes DevFmt::I24).
                    let as_i24 = ((v as f64 * 8_388_608.0) as i32).clamp(-8_388_608, 8_388_607);
                    let signed = ((word << 8) as i32) >> 8;
                    assert_eq!(as_i24, signed, "24-bit scaling mismatch for word {word:#08x}");

                    // 24-in-32 / 32-bit container path (× 2^31 = signed << 8).
                    let as_i32 = (v as f64 * 2_147_483_648.0) as i32;
                    assert_eq!(as_i32, signed << 8, "32-bit scaling mismatch for word {word:#08x}");
                }
            }
        }
    }

    #[test]
    fn marker_phase_carries_across_a_gapless_swap() {
        // File A has an odd PCM-frame count so it ends on phase 1 (0xFA next);
        // seeding B with A's phase must make B start on 0xFA, preserving the
        // 05/FA alternation across the boundary.
        let a_blocks = vec![vec![vec![0x01, 0x02], vec![0x11, 0x12]]]; // 2 bytes/ch = 1 PCM frame
        let a = make_dsf(2, DSD64_RATE, 1, 16, 2, &a_blocks, None);
        let ai = parse_dsf(&mut Cursor::new(&a)).unwrap();
        let mut sa = DopStream::new(DsdReader::new(Cursor::new(a), ai).unwrap());
        let mut wa = Vec::new();
        assert_eq!(sa.read_dop(8, &mut wa).unwrap(), 1);
        assert_eq!((wa[0] >> 16) as u8, 0x05); // A's one frame is 0x05
        assert_eq!(sa.marker_phase(), 1);       // …so the next would be 0xFA

        let b_blocks = vec![vec![vec![0x03, 0x04], vec![0x13, 0x14]]];
        let b = make_dsf(2, DSD64_RATE, 1, 16, 2, &b_blocks, None);
        let bi = parse_dsf(&mut Cursor::new(&b)).unwrap();
        let mut sb = DopStream::new(DsdReader::new(Cursor::new(b), bi).unwrap());
        sb.set_marker_phase(sa.marker_phase());
        let mut wb = Vec::new();
        assert_eq!(sb.read_dop(8, &mut wb).unwrap(), 1);
        assert_eq!((wb[0] >> 16) as u8, 0xFA); // continues the alternation
    }

    #[test]
    fn warmup_silence_then_audio_is_phase_continuous() {
        let blocks = vec![vec![vec![0xAA, 0xBB], vec![0xCC, 0xDD]]];
        let file = make_dsf(2, DSD64_RATE, 1, 16, 2, &blocks, None);
        let info = parse_dsf(&mut Cursor::new(&file)).unwrap();
        let mut s = DopStream::new(DsdReader::new(Cursor::new(file), info).unwrap());
        let mut out = Vec::new();
        assert_eq!(s.warmup_silence(3, &mut out), 3);
        // 3 silence frames of 2ch → 6 words; frame markers 05, FA, 05
        // (each shared by both channels); audio then continues on FA.
        assert_eq!((out[0] >> 16) as u8, 0x05); // frame 0, ch0
        assert_eq!((out[2] >> 16) as u8, 0xFA); // frame 1, ch0
        assert_eq!((out[4] >> 16) as u8, 0x05); // frame 2, ch0
        assert_eq!(out[0] & 0xFFFF, u32::from(DSD_SILENCE) << 8 | u32::from(DSD_SILENCE));
        let base = out.len();
        assert_eq!(s.read_dop(8, &mut out).unwrap(), 1);
        assert_eq!((out[base] >> 16) as u8, 0xFA);
    }

    #[test]
    fn dop_stream_seek() {
        let blocks = vec![
            vec![vec![0x01, 0x02, 0x03, 0x04], vec![0x11, 0x12, 0x13, 0x14]],
        ];
        let file = make_dsf(2, DSD64_RATE, 1, 32, 4, &blocks, None);
        let info = parse_dsf(&mut Cursor::new(&file)).unwrap();
        let reader = DsdReader::new(Cursor::new(&file), info).unwrap();
        let mut s = DopStream::new(reader);
        s.seek_to_pcm_frame(1).unwrap();
        let mut words = Vec::new();
        assert_eq!(s.read_dop(4, &mut words).unwrap(), 1);
        let r = |b: u8| b.reverse_bits();
        // Marker phase restarts at 0x05 after a seek.
        assert_eq!(words, vec![
            dop_word(0x05, r(0x03), r(0x04)),
            dop_word(0x05, r(0x13), r(0x14)),
        ]);
    }
}
