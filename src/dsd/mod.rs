//! DSD (Direct Stream Digital) container parsing — Phase 0 of DSD support.
//!
//! Reads the two DSD container formats:
//!
//! * **DSF** (Sony "DSD Stream File", `.dsf`) — little-endian, fixed
//!   `DSD `/`fmt `/`data` chunk layout, audio stored in per-channel blocks
//!   (normally 4096 bytes/channel), bits **LSB-first** within each byte, and
//!   an optional ID3v2 tag whose offset the header points at.
//! * **DFF** (Philips DSDIFF, `.dff`) — big-endian EA-IFF-85 style `FRM8`
//!   container, audio byte-interleaved per channel, bits **MSB-first**, with
//!   an optional `ID3 ` chunk (written by foobar2000 and friends).
//!
//! [`DsdReader`] yields the raw 1-bit stream as channel-interleaved bytes
//! normalised to **MSB-first** (oldest sample in the most significant bit) —
//! the order DoP packing and the decimator both consume — so later phases
//! never care which container the bits came from.

pub mod decimate;
pub mod dop;

use std::fs::File;
use std::io::{Read, Seek, SeekFrom};
use std::path::Path;
use std::time::Duration;

/// DSD sampling rates are multiples of this (44.1 kHz × 64).
pub const DSD64_RATE: u32 = 2_822_400;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum DsdContainer {
    Dsf,
    Dff,
}

/// Everything the rest of the app needs to know about a DSD file, extracted
/// from its header without touching the audio data.
#[derive(Clone, Debug)]
pub struct DsdInfo {
    pub container: DsdContainer,
    /// 1-bit sample rate per channel in Hz (e.g. 2 822 400 for DSD64).
    pub sample_rate: u32,
    pub channels: u32,
    /// 1-bit samples per channel.
    pub sample_count: u64,
    /// File offset of the first audio byte.
    pub data_offset: u64,
    /// Total audio bytes (all channels, including any final-block padding).
    pub data_len: u64,
    /// Bit order *as stored*: true = oldest sample in the LSB (DSF).
    /// `DsdReader` normalises to MSB-first regardless.
    pub lsb_first: bool,
    /// Per-channel block size in bytes: DSF stores `[ch0 block][ch1 block]…`;
    /// 1 for DFF (plain byte interleave).
    pub block_size: u32,
    /// Offset + length of an embedded ID3v2 tag, if the file has one.
    pub id3: Option<(u64, u64)>,
}

impl DsdInfo {
    pub fn duration(&self) -> Duration {
        Duration::from_secs_f64(self.sample_count as f64 / self.sample_rate as f64)
    }

    /// Total frames the reader will yield (a frame = 1 byte per channel
    /// = 8 DSD samples per channel).
    pub fn total_frames(&self) -> u64 {
        self.sample_count.div_ceil(8)
    }

    pub fn rate_label(&self) -> String {
        rate_label(self.sample_rate)
    }
}

/// "DSD64" / "DSD128" / … for the 44.1k family, with a 48k-family suffix for
/// the rare 48 kHz-based streams, else the raw MHz.
pub fn rate_label(rate: u32) -> String {
    if rate % DSD64_RATE == 0 {
        format!("DSD{}", 64 * (rate / DSD64_RATE))
    } else if rate % 3_072_000 == 0 {
        format!("DSD{} (48k-family)", 64 * (rate / 3_072_000))
    } else {
        format!("DSD @ {:.4} MHz", rate as f64 / 1_000_000.0)
    }
}

/// "2.8224 MHz"-style rate for info displays.
pub fn fmt_mhz(rate: u32) -> String {
    format!("{} MHz", (rate as f64 / 1_000_000.0 * 10_000.0).round() / 10_000.0)
}

pub fn is_dsd_path(path: &Path) -> bool {
    matches!(
        path.extension().and_then(|e| e.to_str()).map(|e| e.to_ascii_lowercase()).as_deref(),
        Some("dsf") | Some("dff")
    )
}

// ---------------------------------------------------------------------------
// Parsing
// ---------------------------------------------------------------------------

fn read_exact_at<R: Read + Seek>(r: &mut R, off: u64, buf: &mut [u8]) -> Result<(), String> {
    r.seek(SeekFrom::Start(off)).map_err(|e| e.to_string())?;
    r.read_exact(buf).map_err(|e| e.to_string())
}

fn u32_le(b: &[u8]) -> u32 { u32::from_le_bytes(b[..4].try_into().unwrap()) }
fn u64_le(b: &[u8]) -> u64 { u64::from_le_bytes(b[..8].try_into().unwrap()) }
fn u64_be(b: &[u8]) -> u64 { u64::from_be_bytes(b[..8].try_into().unwrap()) }

/// Open and parse a `.dsf`/`.dff` file, dispatching on the magic bytes.
pub fn parse_file(path: &Path) -> Result<DsdInfo, String> {
    let mut f = File::open(path).map_err(|e| e.to_string())?;
    let mut magic = [0u8; 4];
    read_exact_at(&mut f, 0, &mut magic)?;
    match &magic {
        b"DSD " => parse_dsf(&mut f),
        b"FRM8" => parse_dff(&mut f),
        _ => Err("not a DSD file (no DSF/DSDIFF magic)".into()),
    }
}

/// Parse a Sony DSF header (fixed `DSD ` → `fmt ` → `data` layout).
pub fn parse_dsf<R: Read + Seek>(r: &mut R) -> Result<DsdInfo, String> {
    let file_len = r.seek(SeekFrom::End(0)).map_err(|e| e.to_string())?;

    // "DSD " chunk: magic, chunk size (28), total file size, metadata pointer.
    let mut hdr = [0u8; 28];
    read_exact_at(r, 0, &mut hdr)?;
    if &hdr[0..4] != b"DSD " {
        return Err("missing DSF 'DSD ' header".into());
    }
    let metadata_ptr = u64_le(&hdr[20..28]);

    // "fmt " chunk, always at offset 28.
    let mut fmt = [0u8; 52];
    read_exact_at(r, 28, &mut fmt)?;
    if &fmt[0..4] != b"fmt " {
        return Err("missing DSF 'fmt ' chunk".into());
    }
    let format_id = u32_le(&fmt[16..20]);
    if format_id != 0 {
        return Err(format!("unsupported DSF format id {format_id} (only raw DSD)"));
    }
    let channels = u32_le(&fmt[24..28]);
    let sample_rate = u32_le(&fmt[28..32]);
    let bits_per_sample = u32_le(&fmt[32..36]);
    let sample_count = u64_le(&fmt[36..44]);
    let block_size = u32_le(&fmt[44..48]);
    if channels == 0 || channels > 8 {
        return Err(format!("bad DSF channel count {channels}"));
    }
    if sample_rate < DSD64_RATE / 2 {
        return Err(format!("implausible DSD rate {sample_rate} Hz"));
    }
    // Per spec: 1 = oldest sample in the LSB, 8 = oldest in the MSB.
    let lsb_first = match bits_per_sample {
        1 => true,
        8 => false,
        b => return Err(format!("bad DSF bits-per-sample {b}")),
    };
    if block_size == 0 || block_size > 1 << 20 {
        return Err(format!("bad DSF block size {block_size}"));
    }

    // "data" chunk at offset 80: magic + u64 size (= audio bytes + 12).
    let mut data_hdr = [0u8; 12];
    read_exact_at(r, 80, &mut data_hdr)?;
    if &data_hdr[0..4] != b"data" {
        return Err("missing DSF 'data' chunk".into());
    }
    let data_offset = 92u64;
    // Cap at what the file actually holds (and below the ID3 tag, if any).
    let mut data_end = file_len;
    if metadata_ptr > data_offset && metadata_ptr <= file_len {
        data_end = metadata_ptr;
    }
    let data_len = u64_le(&data_hdr[4..12])
        .saturating_sub(12)
        .min(data_end.saturating_sub(data_offset));

    let id3 = (metadata_ptr > 0 && metadata_ptr < file_len)
        .then(|| (metadata_ptr, file_len - metadata_ptr));

    Ok(DsdInfo {
        container: DsdContainer::Dsf,
        sample_rate,
        channels,
        sample_count,
        data_offset,
        data_len,
        lsb_first,
        block_size,
        id3,
    })
}

/// Parse a DSDIFF (`FRM8`) container: walk the top-level chunks for `PROP`
/// (rate / channels / compression), the `DSD ` sound chunk, and `ID3 `.
pub fn parse_dff<R: Read + Seek>(r: &mut R) -> Result<DsdInfo, String> {
    let file_len = r.seek(SeekFrom::End(0)).map_err(|e| e.to_string())?;

    let mut hdr = [0u8; 16];
    read_exact_at(r, 0, &mut hdr)?;
    if &hdr[0..4] != b"FRM8" || &hdr[12..16] != b"DSD " {
        return Err("missing DSDIFF 'FRM8'/'DSD ' header".into());
    }
    let frm8_end = (12 + u64_be(&hdr[4..12])).min(file_len);

    let mut sample_rate = 0u32;
    let mut channels = 0u32;
    let mut data: Option<(u64, u64)> = None;
    let mut id3: Option<(u64, u64)> = None;

    // Top-level local chunks, 2-byte aligned.
    let mut pos = 16u64;
    while pos + 12 <= frm8_end {
        let mut ch = [0u8; 12];
        read_exact_at(r, pos, &mut ch)?;
        let id = [ch[0], ch[1], ch[2], ch[3]];
        let size = u64_be(&ch[4..12]);
        let body = pos + 12;

        match &id {
            b"PROP" => {
                let mut kind = [0u8; 4];
                read_exact_at(r, body, &mut kind)?;
                if &kind == b"SND " {
                    // Nested property chunks.
                    let prop_end = (body + size).min(frm8_end);
                    let mut p = body + 4;
                    while p + 12 <= prop_end {
                        let mut sc = [0u8; 12];
                        read_exact_at(r, p, &mut sc)?;
                        let sid = [sc[0], sc[1], sc[2], sc[3]];
                        let ssize = u64_be(&sc[4..12]);
                        match &sid {
                            b"FS  " => {
                                let mut fs = [0u8; 4];
                                read_exact_at(r, p + 12, &mut fs)?;
                                sample_rate = u32::from_be_bytes(fs);
                            }
                            b"CHNL" => {
                                let mut n = [0u8; 2];
                                read_exact_at(r, p + 12, &mut n)?;
                                channels = u16::from_be_bytes(n) as u32;
                            }
                            b"CMPR" => {
                                let mut c = [0u8; 4];
                                read_exact_at(r, p + 12, &mut c)?;
                                if &c != b"DSD " {
                                    return Err(format!(
                                        "compressed DSDIFF ({}) not supported — only uncompressed 'DSD '",
                                        String::from_utf8_lossy(&c).trim()
                                    ));
                                }
                            }
                            _ => {}
                        }
                        p += 12 + ssize + (ssize & 1);
                    }
                }
            }
            b"DSD " => data = Some((body, size.min(file_len.saturating_sub(body)))),
            b"ID3 " => id3 = Some((body, size.min(file_len.saturating_sub(body)))),
            _ => {}
        }
        pos = body + size + (size & 1);
    }

    let (data_offset, data_len) = data.ok_or("DSDIFF has no 'DSD ' sound chunk")?;
    if channels == 0 || channels > 8 {
        return Err(format!("bad DSDIFF channel count {channels}"));
    }
    if sample_rate < DSD64_RATE / 2 {
        return Err(format!("implausible DSD rate {sample_rate} Hz"));
    }
    let sample_count = data_len / channels as u64 * 8;

    Ok(DsdInfo {
        container: DsdContainer::Dff,
        sample_rate,
        channels,
        sample_count,
        data_offset,
        data_len,
        lsb_first: false, // DSDIFF is MSB-first
        block_size: 1,    // plain byte interleave
        id3,
    })
}

/// Extract the embedded ID3v2 tag bytes, if the file has one. The blob starts
/// with `ID3` and can be handed to lofty's ID3v2-capable readers as-is.
pub fn read_id3_blob(path: &Path, info: &DsdInfo) -> Option<Vec<u8>> {
    let (off, len) = info.id3?;
    if len < 10 || len > 256 << 20 {
        return None;
    }
    let mut f = File::open(path).ok()?;
    f.seek(SeekFrom::Start(off)).ok()?;
    let mut blob = vec![0u8; len as usize];
    f.read_exact(&mut blob).ok()?;
    blob.starts_with(b"ID3").then_some(blob)
}

// ---------------------------------------------------------------------------
// Reader — normalised 1-bit stream access
// ---------------------------------------------------------------------------

/// Streams the raw DSD data as channel-interleaved bytes (one *frame* =
/// `channels` bytes = 8 samples per channel), bit order normalised to
/// MSB-first. DSF's per-channel blocks are de-interleaved and its LSB-first
/// bytes reversed; DFF passes straight through.
pub struct DsdReader<R: Read + Seek> {
    src: R,
    info: DsdInfo,
    /// One block-row (`block_size × channels` bytes) for DSF de-interleave.
    block: Vec<u8>,
    /// Frames available / consumed in `block` (DSF only).
    block_frames: usize,
    block_pos: usize,
    /// Next block-row index to load (DSF only).
    next_block: u64,
    frames_read: u64,
}

pub type DsdFileReader = DsdReader<std::io::BufReader<File>>;

/// Open a DSD file for streaming (parses the header itself).
pub fn open_reader(path: &Path) -> Result<DsdFileReader, String> {
    let info = parse_file(path)?;
    let f = File::open(path).map_err(|e| e.to_string())?;
    DsdReader::new(std::io::BufReader::new(f), info)
}

impl<R: Read + Seek> DsdReader<R> {
    pub fn new(mut src: R, info: DsdInfo) -> Result<Self, String> {
        let block = if info.container == DsdContainer::Dsf {
            vec![0u8; info.block_size as usize * info.channels as usize]
        } else {
            // DFF reads are positional — start at the audio data. (DSF block
            // loads seek absolutely each time and don't need this.)
            src.seek(SeekFrom::Start(info.data_offset)).map_err(|e| e.to_string())?;
            Vec::new()
        };
        Ok(DsdReader {
            src,
            info,
            block,
            block_frames: 0,
            block_pos: 0,
            next_block: 0,
            frames_read: 0,
        })
    }

    pub fn info(&self) -> &DsdInfo { &self.info }
    pub fn total_frames(&self) -> u64 { self.info.total_frames() }
    #[allow(dead_code)] // diagnostics/progress reporting, not wired up yet
    pub fn frames_read(&self) -> u64 { self.frames_read }

    /// Fill `out` with as many whole frames as fit (`out.len()` should be a
    /// multiple of `channels`). Returns frames written; 0 = end of stream.
    pub fn read_frames(&mut self, out: &mut [u8]) -> std::io::Result<usize> {
        let ch = self.info.channels as usize;
        let want = (out.len() / ch).min((self.total_frames() - self.frames_read) as usize);
        let mut done = 0usize;

        if self.info.container == DsdContainer::Dff {
            // Fill fully — a short read() must not drop bytes mid-frame or the
            // channels would swap. A truncated tail frame (file ended mid-frame)
            // is dropped, which is safe: the stream is over.
            let mut filled = 0usize;
            while filled < want * ch {
                let n = self.src.read(&mut out[filled..want * ch])?;
                if n == 0 { break; }
                filled += n;
            }
            done = filled / ch;
        } else {
            while done < want {
                if self.block_pos >= self.block_frames && !self.load_block()? {
                    break;
                }
                let take = (want - done).min(self.block_frames - self.block_pos);
                let bs = self.info.block_size as usize;
                for f in 0..take {
                    let row = self.block_pos + f;
                    for c in 0..ch {
                        out[(done + f) * ch + c] = self.block[c * bs + row];
                    }
                }
                self.block_pos += take;
                done += take;
            }
        }

        if self.info.lsb_first {
            for b in &mut out[..done * ch] {
                *b = b.reverse_bits();
            }
        }
        self.frames_read += done as u64;
        Ok(done)
    }

    /// Jump to a frame position (a frame = 8 DSD samples). DSF seeks land
    /// exactly (block-addressed); DFF seeks are byte-exact by construction.
    pub fn seek_to_frame(&mut self, frame: u64) -> std::io::Result<()> {
        let frame = frame.min(self.total_frames());
        let ch = self.info.channels as u64;
        if self.info.container == DsdContainer::Dff {
            self.src.seek(SeekFrom::Start(self.info.data_offset + frame * ch))?;
        } else {
            let bs = self.info.block_size as u64;
            self.next_block = frame / bs;
            self.block_frames = 0; // force reload
            self.block_pos = frame as usize % bs as usize;
            // load_block resets block_pos, so stash the target and re-apply.
            let within = frame % bs;
            if within > 0 || frame < self.total_frames() {
                if self.load_block_io()? {
                    self.block_pos = within as usize;
                } else {
                    self.block_pos = self.block_frames; // at EOF
                }
            }
        }
        self.frames_read = frame;
        Ok(())
    }

    /// Load the next DSF block-row; returns false at end of stream.
    fn load_block(&mut self) -> std::io::Result<bool> {
        let ok = self.load_block_io()?;
        self.block_pos = 0;
        Ok(ok)
    }

    fn load_block_io(&mut self) -> std::io::Result<bool> {
        let bs = self.info.block_size as u64;
        let ch = self.info.channels as u64;
        let row_bytes = bs * ch;
        let row_off = self.next_block * row_bytes;
        if row_off >= self.info.data_len {
            self.block_frames = 0;
            return Ok(false);
        }
        self.src.seek(SeekFrom::Start(self.info.data_offset + row_off))?;
        let avail = (self.info.data_len - row_off).min(row_bytes) as usize;
        self.block[..avail].fill(0);
        self.src.read_exact(&mut self.block[..avail])?;
        // Frames in this row: full block unless the file ends mid-row, and
        // never past the declared per-channel sample count.
        let row_frames = (avail as u64 / ch).min(bs);
        let frames_before = self.next_block * bs;
        self.block_frames = row_frames.min(self.total_frames().saturating_sub(frames_before)) as usize;
        self.next_block += 1;
        Ok(self.block_frames > 0)
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
pub(crate) mod tests {
    use super::*;
    use std::io::Cursor;

    /// Build a minimal valid DSF file in memory.
    /// `blocks`: per-channel audio blocks, `blocks[row][channel]` of `block_size` bytes.
    /// (pub(crate): main.rs's metadata tests reuse it to build tagged fixtures.)
    pub(crate) fn make_dsf(
        channels: u32,
        sample_rate: u32,
        bits_per_sample: u32,
        sample_count: u64,
        block_size: u32,
        blocks: &[Vec<Vec<u8>>],
        id3: Option<&[u8]>,
    ) -> Vec<u8> {
        let mut audio = Vec::new();
        for row in blocks {
            assert_eq!(row.len(), channels as usize);
            for chb in row {
                assert_eq!(chb.len(), block_size as usize);
                audio.extend_from_slice(chb);
            }
        }
        let data_chunk_size = audio.len() as u64 + 12;
        let metadata_ptr = if id3.is_some() { 92 + audio.len() as u64 } else { 0 };
        let total = 92 + audio.len() as u64 + id3.map_or(0, |b| b.len() as u64);

        let mut f = Vec::new();
        f.extend_from_slice(b"DSD ");
        f.extend_from_slice(&28u64.to_le_bytes());
        f.extend_from_slice(&total.to_le_bytes());
        f.extend_from_slice(&metadata_ptr.to_le_bytes());
        f.extend_from_slice(b"fmt ");
        f.extend_from_slice(&52u64.to_le_bytes());
        f.extend_from_slice(&1u32.to_le_bytes()); // format version
        f.extend_from_slice(&0u32.to_le_bytes()); // format id: raw DSD
        f.extend_from_slice(&2u32.to_le_bytes()); // channel type: stereo
        f.extend_from_slice(&channels.to_le_bytes());
        f.extend_from_slice(&sample_rate.to_le_bytes());
        f.extend_from_slice(&bits_per_sample.to_le_bytes());
        f.extend_from_slice(&sample_count.to_le_bytes());
        f.extend_from_slice(&block_size.to_le_bytes());
        f.extend_from_slice(&0u32.to_le_bytes()); // reserved
        f.extend_from_slice(b"data");
        f.extend_from_slice(&data_chunk_size.to_le_bytes());
        f.extend_from_slice(&audio);
        if let Some(b) = id3 {
            f.extend_from_slice(b);
        }
        f
    }

    /// Build a minimal valid DFF file in memory.
    fn make_dff(channels: u16, sample_rate: u32, cmpr: &[u8; 4], audio: &[u8], id3: Option<&[u8]>) -> Vec<u8> {
        let mut prop = Vec::new();
        prop.extend_from_slice(b"SND ");
        prop.extend_from_slice(b"FS  ");
        prop.extend_from_slice(&4u64.to_be_bytes());
        prop.extend_from_slice(&sample_rate.to_be_bytes());
        prop.extend_from_slice(b"CHNL");
        let chnl_size = 2 + channels as u64 * 4;
        prop.extend_from_slice(&chnl_size.to_be_bytes());
        prop.extend_from_slice(&channels.to_be_bytes());
        for i in 0..channels {
            prop.extend_from_slice(if i == 0 { b"SLFT" } else { b"SRGT" });
        }
        if chnl_size % 2 == 1 {
            prop.push(0);
        }
        prop.extend_from_slice(b"CMPR");
        // 4-byte id + pascal string "not compressed" (14) + pad to even = 20
        let name = b"not compressed";
        prop.extend_from_slice(&(4 + 1 + name.len() as u64 + 1).to_be_bytes());
        prop.extend_from_slice(cmpr);
        prop.push(name.len() as u8);
        prop.extend_from_slice(name);
        prop.push(0);

        let mut body = Vec::new();
        body.extend_from_slice(b"DSD "); // form type
        body.extend_from_slice(b"FVER");
        body.extend_from_slice(&4u64.to_be_bytes());
        body.extend_from_slice(&0x01050000u32.to_be_bytes());
        body.extend_from_slice(b"PROP");
        body.extend_from_slice(&(prop.len() as u64).to_be_bytes());
        body.extend_from_slice(&prop);
        body.extend_from_slice(b"DSD ");
        body.extend_from_slice(&(audio.len() as u64).to_be_bytes());
        body.extend_from_slice(audio);
        if audio.len() % 2 == 1 {
            body.push(0);
        }
        if let Some(b) = id3 {
            body.extend_from_slice(b"ID3 ");
            body.extend_from_slice(&(b.len() as u64).to_be_bytes());
            body.extend_from_slice(b);
        }

        let mut f = Vec::new();
        f.extend_from_slice(b"FRM8");
        f.extend_from_slice(&(body.len() as u64).to_be_bytes());
        f.extend_from_slice(&body);
        f
    }

    #[test]
    fn dsf_header_fields() {
        let blocks = vec![vec![vec![0u8; 4], vec![0u8; 4]]];
        let file = make_dsf(2, DSD64_RATE, 1, 32, 4, &blocks, None);
        let info = parse_dsf(&mut Cursor::new(&file)).unwrap();
        assert_eq!(info.container, DsdContainer::Dsf);
        assert_eq!(info.sample_rate, DSD64_RATE);
        assert_eq!(info.channels, 2);
        assert_eq!(info.sample_count, 32);
        assert_eq!(info.data_offset, 92);
        assert_eq!(info.data_len, 8);
        assert!(info.lsb_first);
        assert_eq!(info.block_size, 4);
        assert!(info.id3.is_none());
        assert_eq!(info.total_frames(), 4);
    }

    #[test]
    fn dsf_id3_pointer() {
        let id3 = b"ID3\x04\x00\x00\x00\x00\x00\x00fake-tag-payload";
        let blocks = vec![vec![vec![0u8; 4], vec![0u8; 4]]];
        let file = make_dsf(2, DSD64_RATE, 1, 32, 4, &blocks, Some(id3));
        let info = parse_dsf(&mut Cursor::new(&file)).unwrap();
        let (off, len) = info.id3.unwrap();
        assert_eq!(off, 92 + 8);
        assert_eq!(len, id3.len() as u64);
        // Audio length must not swallow the tag.
        assert_eq!(info.data_len, 8);
    }

    #[test]
    fn dsf_rejects_garbage() {
        assert!(parse_dsf(&mut Cursor::new(b"RIFFxxxxxxxxxxxxxxxxxxxxxxxxxxxx".to_vec())).is_err());
        // Bad bits-per-sample.
        let blocks = vec![vec![vec![0u8; 4]]];
        let file = make_dsf(1, DSD64_RATE, 4, 32, 4, &blocks, None);
        assert!(parse_dsf(&mut Cursor::new(&file)).is_err());
        // Zero channels.
        let file = make_dsf(1, DSD64_RATE, 1, 32, 4, &blocks, None);
        let mut bad = file.clone();
        bad[52..56].copy_from_slice(&0u32.to_le_bytes()); // channel num = 0
        assert!(parse_dsf(&mut Cursor::new(&bad)).is_err());
    }

    #[test]
    fn dff_header_fields() {
        let audio = [0xAAu8, 0xBB, 0xCC, 0xDD]; // 2 frames of 2ch
        let file = make_dff(2, 2 * DSD64_RATE, b"DSD ", &audio, None);
        let info = parse_dff(&mut Cursor::new(&file)).unwrap();
        assert_eq!(info.container, DsdContainer::Dff);
        assert_eq!(info.sample_rate, 2 * DSD64_RATE);
        assert_eq!(info.channels, 2);
        assert_eq!(info.data_len, 4);
        assert_eq!(info.sample_count, 16); // 4 bytes / 2 ch × 8 bits
        assert!(!info.lsb_first);
        assert!(info.id3.is_none());
    }

    #[test]
    fn dff_id3_chunk_and_dst_rejection() {
        let id3 = b"ID3\x03\x00\x00\x00\x00\x00\x00hello";
        let file = make_dff(2, DSD64_RATE, b"DSD ", &[0u8; 4], Some(id3));
        let info = parse_dff(&mut Cursor::new(&file)).unwrap();
        let (off, len) = info.id3.unwrap();
        assert_eq!(len, id3.len() as u64);
        assert_eq!(&file[off as usize..(off + 3) as usize], b"ID3");

        let dst = make_dff(2, DSD64_RATE, b"DST ", &[0u8; 4], None);
        let err = parse_dff(&mut Cursor::new(&dst)).unwrap_err();
        assert!(err.contains("compressed"), "{err}");
    }

    #[test]
    fn duration_and_labels() {
        let info = DsdInfo {
            container: DsdContainer::Dsf,
            sample_rate: DSD64_RATE,
            channels: 2,
            sample_count: DSD64_RATE as u64 * 3, // exactly 3 s
            data_offset: 92,
            data_len: 0,
            lsb_first: true,
            block_size: 4096,
            id3: None,
        };
        assert_eq!(info.duration(), Duration::from_secs(3));
        assert_eq!(rate_label(2_822_400), "DSD64");
        assert_eq!(rate_label(5_644_800), "DSD128");
        assert_eq!(rate_label(11_289_600), "DSD256");
        assert_eq!(rate_label(22_579_200), "DSD512");
        assert_eq!(rate_label(3_072_000), "DSD64 (48k-family)");
        assert_eq!(fmt_mhz(2_822_400), "2.8224 MHz");
        assert_eq!(fmt_mhz(11_289_600), "11.2896 MHz");
    }

    #[test]
    fn dsf_reader_deinterleaves_and_reverses_bits() {
        // 2 channels, block size 4, two block-rows.
        // Channel bytes chosen so bit reversal is visible: 0x01 -> 0x80.
        let blocks = vec![
            vec![vec![0x01, 0x02, 0x03, 0x04], vec![0x11, 0x12, 0x13, 0x14]],
            vec![vec![0x05, 0x06, 0x07, 0x08], vec![0x15, 0x16, 0x17, 0x18]],
        ];
        let file = make_dsf(2, DSD64_RATE, 1, 64, 4, &blocks, None); // 64 samples = 8 frames
        let info = parse_dsf(&mut Cursor::new(&file)).unwrap();
        let mut rd = DsdReader::new(Cursor::new(&file), info).unwrap();
        let mut out = vec![0u8; 16];
        let n = rd.read_frames(&mut out).unwrap();
        assert_eq!(n, 8);
        let rev = |b: u8| b.reverse_bits();
        // Interleaved ch0,ch1 per frame, bit-reversed (DSF is LSB-first).
        let expect: Vec<u8> = vec![
            0x01, 0x11, 0x02, 0x12, 0x03, 0x13, 0x04, 0x14,
            0x05, 0x15, 0x06, 0x16, 0x07, 0x17, 0x08, 0x18,
        ].into_iter().map(rev).collect();
        assert_eq!(out, expect);
        assert_eq!(rd.read_frames(&mut out).unwrap(), 0); // EOF
    }

    #[test]
    fn dsf_reader_respects_sample_count_padding() {
        // Declared 40 samples (5 frames) but a full 8-frame block-row on disk:
        // the trailing padding must not be emitted.
        let blocks = vec![
            vec![vec![1, 2, 3, 4], vec![9, 10, 11, 12]],
            vec![vec![5, 0, 0, 0], vec![13, 0, 0, 0]],
        ];
        let file = make_dsf(2, DSD64_RATE, 1, 40, 4, &blocks, None);
        let info = parse_dsf(&mut Cursor::new(&file)).unwrap();
        let mut rd = DsdReader::new(Cursor::new(&file), info).unwrap();
        let mut out = vec![0u8; 32];
        assert_eq!(rd.read_frames(&mut out).unwrap(), 5);
        assert_eq!(rd.read_frames(&mut out).unwrap(), 0);
    }

    #[test]
    fn dff_reader_passthrough() {
        let audio = [0x01u8, 0x11, 0x02, 0x12]; // already interleaved, MSB-first
        let file = make_dff(2, DSD64_RATE, b"DSD ", &audio, None);
        let info = parse_dff(&mut Cursor::new(&file)).unwrap();
        let mut rd = DsdReader::new(Cursor::new(&file), info).unwrap();
        let mut out = vec![0u8; 8];
        let n = rd.read_frames(&mut out).unwrap();
        assert_eq!(n, 2);
        assert_eq!(&out[..4], &audio); // no bit reversal, no re-ordering
    }

    #[test]
    fn reader_seek() {
        let blocks = vec![
            vec![vec![0x01, 0x02, 0x03, 0x04], vec![0x11, 0x12, 0x13, 0x14]],
            vec![vec![0x05, 0x06, 0x07, 0x08], vec![0x15, 0x16, 0x17, 0x18]],
        ];
        let file = make_dsf(2, DSD64_RATE, 1, 64, 4, &blocks, None);
        let info = parse_dsf(&mut Cursor::new(&file)).unwrap();
        let mut rd = DsdReader::new(Cursor::new(&file), info).unwrap();
        rd.seek_to_frame(5).unwrap(); // into the second block-row
        let mut out = vec![0u8; 4];
        assert_eq!(rd.read_frames(&mut out).unwrap(), 2);
        assert_eq!(out, vec![0x06u8.reverse_bits(), 0x16u8.reverse_bits(),
                             0x07u8.reverse_bits(), 0x17u8.reverse_bits()]);

        // DFF byte-exact seek.
        let audio = [0x01u8, 0x11, 0x02, 0x12, 0x03, 0x13];
        let file = make_dff(2, DSD64_RATE, b"DSD ", &audio, None);
        let info = parse_dff(&mut Cursor::new(&file)).unwrap();
        let mut rd = DsdReader::new(Cursor::new(&file), info).unwrap();
        rd.seek_to_frame(2).unwrap();
        let mut out = vec![0u8; 2];
        assert_eq!(rd.read_frames(&mut out).unwrap(), 1);
        assert_eq!(out, vec![0x03, 0x13]);
    }

    #[test]
    fn magic_dispatch() {
        // parse_file dispatches on magic, not extension — verified indirectly
        // through the two parsers here; is_dsd_path handles the extension side.
        assert!(is_dsd_path(Path::new("/x/a.dsf")));
        assert!(is_dsd_path(Path::new("/x/a.DFF")));
        assert!(!is_dsd_path(Path::new("/x/a.flac")));
    }
}
