// Symphonia
// Copyright (c) 2019-2022 The Project Symphonia Developers.
//
// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at https://mozilla.org/MPL/2.0/.

use std::collections::{HashMap, VecDeque};
use std::convert::TryFrom;
use std::io::{Seek, SeekFrom};

use symphonia_core::audio::Layout;
use symphonia_core::codecs::{CodecParameters, CODEC_TYPE_FLAC, CODEC_TYPE_OPUS, CODEC_TYPE_NULL, CODEC_TYPE_VORBIS};
use symphonia_core::errors::{
    decode_error, end_of_stream_error, seek_error, unsupported_error, Error, Result, SeekErrorKind,
};
use symphonia_core::formats::{
    Cue, FormatOptions, FormatReader, Packet, SeekMode, SeekTo, SeekedTo, Track,
};
use symphonia_core::io::{BufReader, MediaSource, MediaSourceStream, ReadBytes};
use symphonia_core::meta::{Metadata, MetadataLog};
use symphonia_core::probe::Instantiate;
use symphonia_core::probe::{Descriptor, QueryDescriptor};
use symphonia_core::sample::SampleFormat;
use symphonia_core::support_format;
use symphonia_core::units::TimeBase;
use symphonia_utils_xiph::flac::metadata::{MetadataBlockHeader, MetadataBlockType};

use crate::codecs::codec_id_to_type;
use crate::ebml::{EbmlElement, ElementHeader, ElementIterator};
use crate::element_ids::{ElementType, ELEMENTS};
use crate::lacing::{extract_frames, read_xiph_sizes, Frame};
use crate::segment::{
    BlockGroupElement, ClusterElement, CuesElement, InfoElement, SeekHeadElement, TagsElement,
    TracksElement,
};

#[allow(dead_code)]
pub struct TrackState {
    /// Codec parameters.
    pub(crate) codec_params: CodecParameters,
    /// The track number.
    track_num: u32,
    /// Default frame duration in nanoseconds.
    pub(crate) default_frame_duration: Option<u64>,
    startup_trim: u32,
}

/// Matroska (MKV) and WebM demultiplexer.
///
/// `MkvReader` implements a demuxer for the Matroska and WebM formats.
pub struct MkvReader {
    /// Iterator over EBML element headers
    iter: ElementIterator<MediaSourceStream>,
    tracks: Vec<Track>,
    track_states: HashMap<u32, TrackState>,
    current_cluster: Option<ClusterState>,
    metadata: MetadataLog,
    cues: Vec<Cue>,
    frames: VecDeque<Frame>,
    timestamp_scale: u64,
    clusters: Vec<ClusterElement>,
    segment_start: u64,
}

#[derive(Debug)]
struct ClusterState {
    timestamp: Option<u64>,
    end: Option<u64>,
}

fn nanos_to_opus_frames(nanos: u64) -> Result<u32> {
    // Nanosecond metadata is integer-rounded. Nearest sample avoids an extra
    // discarded sample for ordinary 48 kHz padding (e.g. 648/48000 seconds).
    let frames = (u128::from(nanos) * 48_000 + 500_000_000) / 1_000_000_000;
    u32::try_from(frames).map_err(|_| Error::DecodeError("mkv: excessive Opus padding"))
}

fn vorbis_extra_data_from_codec_private(extra: &[u8]) -> Result<Box<[u8]>> {
    const VORBIS_PACKET_TYPE_IDENTIFICATION: u8 = 1;
    const VORBIS_PACKET_TYPE_SETUP: u8 = 5;

    // Private Data for this codec has the following layout:
    // - 1 byte that represents number of packets minus one;
    // - Xiph coded lengths of packets, length of the last packet must be deduced (as in Xiph lacing)
    // - packets in order:
    //    - The Vorbis identification header
    //    - Vorbis comment header
    //    - codec setup header

    let mut reader = BufReader::new(extra);
    let packet_count = reader.read_byte()? as usize;
    let packet_lengths = read_xiph_sizes(&mut reader, packet_count)?;

    let mut packets = Vec::new();
    for length in packet_lengths {
        packets.push(reader.read_boxed_slice_exact(length as usize)?);
    }

    let last_packet_length = extra.len() - reader.pos() as usize;
    packets.push(reader.read_boxed_slice_exact(last_packet_length)?);

    let mut ident_header = None;
    let mut setup_header = None;

    for packet in packets {
        match packet.first().copied() {
            Some(VORBIS_PACKET_TYPE_IDENTIFICATION) => {
                ident_header = Some(packet);
            }
            Some(VORBIS_PACKET_TYPE_SETUP) => {
                setup_header = Some(packet);
            }
            _ => {
                log::debug!("unsupported vorbis packet type");
            }
        }
    }

    // This is layout expected currently by Vorbis codec.
    Ok([
        ident_header.ok_or(Error::DecodeError("mkv: missing vorbis identification packet"))?,
        setup_header.ok_or(Error::DecodeError("mkv: missing vorbis setup packet"))?,
    ]
    .concat()
    .into_boxed_slice())
}

fn flac_extra_data_from_codec_private(codec_private: &[u8]) -> Result<Box<[u8]>> {
    let mut reader = BufReader::new(codec_private);

    let marker = reader.read_quad_bytes()?;
    if marker != *b"fLaC" {
        return decode_error("mkv (flac): missing flac stream marker");
    }

    let header = MetadataBlockHeader::read(&mut reader)?;

    loop {
        match header.block_type {
            MetadataBlockType::StreamInfo => {
                break Ok(reader.read_boxed_slice_exact(header.block_len as usize)?);
            }
            _ => reader.ignore_bytes(u64::from(header.block_len))?,
        }
    }
}

impl MkvReader {
    fn seek_track_by_ts_forward(&mut self, track_id: u32, ts: u64) -> Result<SeekedTo> {
        let actual_ts = 'out: loop {
            // Skip frames from the buffer until the given timestamp
            while let Some(frame) = self.frames.front() {
                if frame.timestamp + frame.duration >= ts && frame.track == track_id {
                    break 'out frame.timestamp;
                }
                else {
                    self.frames.pop_front();
                }
            }
            self.next_element()?
        };

        Ok(SeekedTo { track_id, required_ts: ts, actual_ts })
    }

    fn seek_track_by_ts(&mut self, track_id: u32, ts: u64) -> Result<SeekedTo> {
        if self.clusters.is_empty() {
            self.seek_track_by_ts_forward(track_id, ts)
        }
        else {
            let mut target_cluster = None;
            for cluster in &self.clusters {
                if cluster.timestamp > ts {
                    break;
                }
                target_cluster = Some(cluster);
            }
            let cluster = target_cluster.ok_or(Error::SeekError(SeekErrorKind::OutOfRange))?;

            let mut target_block = None;
            for block in cluster.blocks.iter() {
                if block.track as u32 != track_id {
                    continue;
                }
                if block.timestamp > ts {
                    break;
                }
                target_block = Some(block);
            }

            let pos = match target_block {
                Some(block) => block.pos,
                None => cluster.pos,
            };
            self.iter.seek(pos)?;

            // Restore cluster's metadata
            self.current_cluster =
                Some(ClusterState { timestamp: Some(cluster.timestamp), end: cluster.end });

            // Seek to a specified block inside the cluster.
            self.seek_track_by_ts_forward(track_id, ts)
        }
    }

    fn next_element(&mut self) -> Result<()> {
        if let Some(ClusterState { end: Some(end), .. }) = &self.current_cluster {
            // Make sure we don't read past the current cluster if its size is known.
            if self.iter.pos() >= *end {
                // log::debug!("ended cluster");
                self.current_cluster = None;
            }
        }

        // Each Cluster is being read incrementally so we need to keep track of
        // which cluster we are currently in.

        let header = match self.iter.read_child_header()? {
            Some(header) => header,
            None => {
                // If we reached here, it must be an end of stream.
                return end_of_stream_error();
            }
        };

        match header.etype {
            ElementType::Cluster => {
                self.current_cluster = Some(ClusterState { timestamp: None, end: header.end() });
            }
            ElementType::Timestamp => match self.current_cluster.as_mut() {
                Some(cluster) => {
                    cluster.timestamp = Some(self.iter.read_u64()?);
                }
                None => {
                    self.iter.ignore_data()?;
                    log::warn!("timestamp element outside of a cluster");
                    return Ok(());
                }
            },
            ElementType::SimpleBlock => {
                let cluster_ts = match self.current_cluster.as_ref() {
                    Some(ClusterState { timestamp: Some(ts), .. }) => *ts,
                    Some(_) => {
                        self.iter.ignore_data()?;
                        log::warn!("missing cluster timestamp");
                        return Ok(());
                    }
                    None => {
                        self.iter.ignore_data()?;
                        log::warn!("simple block element outside of a cluster");
                        return Ok(());
                    }
                };

                let data = self.iter.read_boxed_slice()?;
                extract_frames(
                    &data,
                    None,
                    &self.track_states,
                    cluster_ts,
                    self.timestamp_scale,
                    &mut self.frames,
                )?;
            }
            ElementType::BlockGroup => {
                let cluster_ts = match self.current_cluster.as_ref() {
                    Some(ClusterState { timestamp: Some(ts), .. }) => *ts,
                    Some(_) => {
                        self.iter.ignore_data()?;
                        log::warn!("missing cluster timestamp");
                        return Ok(());
                    }
                    None => {
                        self.iter.ignore_data()?;
                        log::warn!("block group element outside of a cluster");
                        return Ok(());
                    }
                };

                let group = self.iter.read_element_data::<BlockGroupElement>()?;
                let before = self.frames.len();
                extract_frames(
                    &group.data,
                    group.duration,
                    &self.track_states,
                    cluster_ts,
                    self.timestamp_scale,
                    &mut self.frames,
                )?;
                // Padding belongs to the whole Block, which may contain
                // several laced Opus packets. Never put an entire block's
                // trim on one packet or carry it into the next Block.
                if self.frames.get(before).is_some_and(|frame| {
                    self.track_states.get(&frame.track)
                        .is_some_and(|state| state.codec_params.codec == CODEC_TYPE_OPUS)
                }) {
                    apply_opus_block_padding(&mut self.frames, before, group.discard_padding)?;
                }
            }
            ElementType::Tags => {
                let tags = self.iter.read_element_data::<TagsElement>()?;
                self.metadata.push(tags.to_metadata());
                self.current_cluster = None;
            }
            _ if header.etype.is_top_level() => {
                self.current_cluster = None;
            }
            other => {
                log::debug!("ignored element {:?}", other);
                self.iter.ignore_data()?;
            }
        }

        Ok(())
    }
}

impl FormatReader for MkvReader {
    fn try_new(mut reader: MediaSourceStream, _options: &FormatOptions) -> Result<Self>
    where
        Self: Sized,
    {
        let is_seekable = reader.is_seekable();

        // Get the total length of the stream, if possible.
        let total_len = if is_seekable {
            let pos = reader.pos();
            let len = reader.byte_len().ok_or(Error::SeekError(SeekErrorKind::Unseekable))?;
            reader.seek(SeekFrom::Start(pos))?;
            log::info!("stream is seekable with len={} bytes.", len);
            Some(len)
        }
        else {
            None
        };

        let mut it = ElementIterator::new(reader, total_len);
        let ebml = it.read_element::<EbmlElement>()?;

        if !matches!(ebml.header.doc_type.as_str(), "matroska" | "webm") {
            return unsupported_error("mkv: not a matroska / webm file");
        }

        let segment_pos = match it.read_child_header()? {
            Some(ElementHeader { etype: ElementType::Segment, data_pos, .. }) => data_pos,
            _ => return unsupported_error("mkv: missing segment element"),
        };

        let mut segment_tracks = None;
        let mut info = None;
        let mut clusters = Vec::new();
        let mut metadata = MetadataLog::default();
        let mut current_cluster = None;

        let mut seek_positions = Vec::new();
        while let Ok(Some(header)) = it.read_child_header() {
            match header.etype {
                ElementType::SeekHead => {
                    let seek_head = it.read_element_data::<SeekHeadElement>()?;
                    for element in seek_head.seeks.into_vec() {
                        let tag = element.id as u32;
                        let etype = match ELEMENTS.get(&tag) {
                            Some((_, etype)) => *etype,
                            None => continue,
                        };
                        seek_positions.push((etype, segment_pos + element.position));
                    }
                }
                ElementType::Tracks => {
                    segment_tracks = Some(it.read_element_data::<TracksElement>()?);
                }
                ElementType::Info => {
                    info = Some(it.read_element_data::<InfoElement>()?);
                }
                ElementType::Cues => {
                    let cues = it.read_element_data::<CuesElement>()?;
                    for cue in cues.points.into_vec() {
                        clusters.push(ClusterElement {
                            timestamp: cue.time,
                            pos: segment_pos + cue.positions.cluster_position,
                            end: None,
                            blocks: Box::new([]),
                        });
                    }
                }
                ElementType::Tags => {
                    let tags = it.read_element_data::<TagsElement>()?;
                    metadata.push(tags.to_metadata());
                }
                ElementType::Cluster => {
                    // Set state for current cluster for the first call of `next_element`.
                    current_cluster = Some(ClusterState { timestamp: None, end: header.end() });

                    // Don't look forward into the stream since
                    // we can't be sure that we'll find anything useful.
                    break;
                }
                other => {
                    it.ignore_data()?;
                    log::debug!("ignored element {:?}", other);
                }
            }
        }

        if is_seekable {
            // Make sure we don't jump backwards unnecessarily.
            seek_positions.sort_by_key(|sp| sp.1);

            for (etype, pos) in seek_positions {
                it.seek(pos)?;

                // Safety: The element type or position may be incorrect. The element iterator will
                // validate the type (as declared in the header) of the element at the seeked
                // position against the element type asked to be read.
                match etype {
                    ElementType::Tracks => {
                        segment_tracks = Some(it.read_element::<TracksElement>()?);
                    }
                    ElementType::Info => {
                        info = Some(it.read_element::<InfoElement>()?);
                    }
                    ElementType::Tags => {
                        let tags = it.read_element::<TagsElement>()?;
                        metadata.push(tags.to_metadata());
                    }
                    ElementType::Cues => {
                        let cues = it.read_element::<CuesElement>()?;
                        for cue in cues.points.into_vec() {
                            clusters.push(ClusterElement {
                                timestamp: cue.time,
                                pos: segment_pos + cue.positions.cluster_position,
                                end: None,
                                blocks: Box::new([]),
                            });
                        }
                    }
                    _ => (),
                }
            }
        }

        let segment_tracks =
            segment_tracks.ok_or(Error::DecodeError("mkv: missing Tracks element"))?;

        if is_seekable {
            let mut reader = it.into_inner();
            reader.seek(SeekFrom::Start(segment_pos))?;
            it = ElementIterator::new(reader, total_len);
        }

        let info = info.ok_or(Error::DecodeError("mkv: missing Info element"))?;

        // TODO: remove this unwrap?
        let time_base = TimeBase::new(u32::try_from(info.timestamp_scale).unwrap(), 1_000_000_000);

        let mut tracks = Vec::new();
        let mut states = HashMap::new();
        for track in segment_tracks.tracks.into_vec() {
            let codec_type = codec_id_to_type(&track);

            let mut codec_params = CodecParameters::new();
            codec_params.with_time_base(time_base);

            if let Some(duration) = info.duration {
                codec_params.with_n_frames(duration as u64);
            }

            if let Some(audio) = track.audio {
                codec_params.with_sample_rate(audio.sampling_frequency.round() as u32);

                let format = audio.bit_depth.and_then(|bits| match bits {
                    8 => Some(SampleFormat::S8),
                    16 => Some(SampleFormat::S16),
                    24 => Some(SampleFormat::S24),
                    32 => Some(SampleFormat::S32),
                    _ => None,
                });

                if let Some(format) = format {
                    codec_params.with_sample_format(format);
                }

                if let Some(bits) = audio.bit_depth {
                    codec_params.with_bits_per_sample(bits as u32);
                }

                let layout = match audio.channels {
                    1 => Some(Layout::Mono),
                    2 => Some(Layout::Stereo),
                    3 => Some(Layout::TwoPointOne),
                    6 => Some(Layout::FivePointOne),
                    other => {
                        log::warn!(
                            "track #{} has custom number of channels: {}",
                            track.number,
                            other
                        );
                        None
                    }
                };

                if let Some(layout) = layout {
                    codec_params.with_channel_layout(layout);
                }

                if let Some(codec_type) = codec_type {
                    codec_params.for_codec(codec_type);
                    if let Some(codec_private) = track.codec_private {
                        let extra_data = match codec_type {
                            CODEC_TYPE_VORBIS => {
                                vorbis_extra_data_from_codec_private(&codec_private)?
                            }
                            CODEC_TYPE_FLAC => flac_extra_data_from_codec_private(&codec_private)?,
                            _ => codec_private,
                        };
                        codec_params.with_extra_data(extra_data);
                    }
                    if codec_type == CODEC_TYPE_OPUS {
                        // Opus always decodes at 48 kHz here; the header's
                        // original input rate is not its output clock.
                        codec_params.with_sample_rate(48_000);
                        let delay = if track.codec_delay != 0 {
                            nanos_to_opus_frames(track.codec_delay)?
                        } else {
                            codec_params.extra_data.as_deref()
                                .filter(|h| h.len() >= 19 && &h[..8] == b"OpusHead")
                                .map(|h| u32::from(u16::from_le_bytes([h[10], h[11]])))
                                .unwrap_or(0)
                        };
                        if let Some(h) = codec_params.extra_data.as_deref()
                            .filter(|h| h.len() >= 19 && &h[..8] == b"OpusHead") {
                            let header_delay = u32::from(u16::from_le_bytes([h[10], h[11]]));
                            if delay != header_delay {
                                return decode_error("mkv: CodecDelay disagrees with OpusHead pre-skip");
                            }
                        }
                        codec_params.with_delay(delay);
                    }
                }
            }

            let track_id = track.number as u32;
            tracks.push(Track {
                id: track_id,
                codec_params: codec_params.clone(),
                language: track.language,
            });

            states.insert(
                track_id,
                TrackState {
                    startup_trim: codec_params.delay.unwrap_or(0),
                    codec_params,
                    track_num: track_id,
                    default_frame_duration: track.default_duration,
                },
            );
        }

        Ok(Self {
            iter: it,
            tracks,
            track_states: states,
            current_cluster,
            metadata,
            cues: Vec::new(),
            frames: VecDeque::new(),
            timestamp_scale: info.timestamp_scale,
            clusters,
            segment_start: segment_pos,
        })
    }

    fn cues(&self) -> &[Cue] {
        &self.cues
    }

    fn metadata(&mut self) -> Metadata<'_> {
        self.metadata.metadata()
    }

    fn seek(&mut self, _mode: SeekMode, to: SeekTo) -> Result<SeekedTo> {
        if self.tracks.is_empty() {
            return seek_error(SeekErrorKind::Unseekable);
        }

        let (requested_track, required_ts) = match to {
            SeekTo::Time { time, track_id } => {
                let track = track_id.and_then(|id| self.tracks.iter().find(|t| t.id == id))
                    .or_else(|| if track_id.is_none() { self.default_track() } else { None })
                    .ok_or(Error::SeekError(SeekErrorKind::InvalidTrack))?;
                (track, track.codec_params.time_base.unwrap().calc_timestamp(time))
            }
            SeekTo::TimeStamp { ts, track_id } => {
                (self.tracks.iter().find(|t| t.id == track_id)
                    .ok_or(Error::SeekError(SeekErrorKind::InvalidTrack))?, ts)
            }
        };
        if requested_track.codec_params.codec == CODEC_TYPE_OPUS {
            // Correctness-first fallback: this reader has no packet-indexed
            // Opus SeekPreRoll implementation. Restart at the known beginning;
            // consumers decode/discard preceding audio with bounded memory.
            // This also works without Cues and avoids reapplying pre-skip to
            // an arbitrary packet. It costs O(seek position) decode time.
            let track_id = requested_track.id;
            self.iter.seek(self.segment_start)?;
            self.frames.clear();
            self.current_cluster = None;
            for state in self.track_states.values_mut() {
                state.startup_trim = state.codec_params.delay.unwrap_or(0);
            }
            return Ok(SeekedTo { track_id, required_ts, actual_ts: 0 });
        }

        // Buffered frames belong to the old read position, not the new seek.
        self.frames.clear();
        let sought = match to {
            SeekTo::Time { time, track_id } => {
                let track = match track_id {
                    Some(id) => self.tracks.iter().find(|track| track.id == id),
                    None => self.default_track(),
                };
                let track = track.ok_or(Error::SeekError(SeekErrorKind::InvalidTrack))?;
                let tb = track.codec_params.time_base.unwrap();
                let ts = tb.calc_timestamp(time);
                let track_id = track.id;
                self.seek_track_by_ts(track_id, ts)
            }
            SeekTo::TimeStamp { ts, track_id } => {
                match self.tracks.iter().find(|t| t.id == track_id) {
                    Some(_) => self.seek_track_by_ts(track_id, ts),
                    None => seek_error(SeekErrorKind::InvalidTrack),
                }
            }
        }?;
        for state in self.track_states.values_mut() {
            // A decoder reset after a mid-track seek must not drop the file's
            // startup delay again. Returning to the actual beginning must.
            state.startup_trim = if sought.actual_ts == 0 {
                state.codec_params.delay.unwrap_or(0)
            } else { 0 };
        }
        Ok(sought)
    }

    fn tracks(&self) -> &[Track] {
        &self.tracks
    }

    fn default_track(&self) -> Option<&Track> {
        self.tracks.iter().find(|t| t.codec_params.codec != CODEC_TYPE_NULL)
    }

    fn next_packet(&mut self) -> Result<Packet> {
        loop {
            if let Some(mut frame) = self.frames.pop_front() {
                if let Some(state) = self.track_states.get_mut(&frame.track) {
                    if state.codec_params.codec == CODEC_TYPE_OPUS {
                        let count = moosik_codec_opus::packet_duration_frames(&frame.data)? as u32;
                        let trim = state.startup_trim.min(count);
                        state.startup_trim -= trim;
                        // Both are prefixes in the same raw packet. Their
                        // union is the longer prefix, not a double subtraction.
                        frame.trim_start = frame.trim_start.max(trim);
                    }
                }
                return Ok(Packet::new_trimmed_from_boxed_slice(
                    frame.track,
                    frame.timestamp,
                    frame.duration,
                    frame.trim_start,
                    frame.trim_end,
                    frame.data,
                ));
            }
            self.next_element()?;
        }
    }

    fn into_inner(self: Box<Self>) -> MediaSourceStream {
        self.iter.into_inner()
    }
}

impl QueryDescriptor for MkvReader {
    fn query() -> &'static [Descriptor] {
        &[support_format!(
            "matroska",
            "Matroska / WebM",
            &["webm", "mkv"],
            &["video/webm", "video/x-matroska"],
            &[b"\x1A\x45\xDF\xA3"] // Top-level element Ebml element
        )]
    }

    fn score(_context: &[u8]) -> u8 {
        255
    }
}

/// Apply DiscardPadding only to the newly extracted laces of one Block.
/// Packet trim fields are PCM-frame counts at 48 kHz, not Matroska ticks.
/// Validate the complete request before changing any frame. Startup CodecDelay
/// is a separate encoded-stream prefix, merged by next_packet when it consumes
/// that prefix; this helper does not consume or restart the stream's delay.
fn apply_opus_block_padding(
    frames: &mut VecDeque<Frame>,
    first: usize,
    padding_nanos: i64,
) -> Result<()> {
    let mut remaining = nanos_to_opus_frames(padding_nanos.unsigned_abs())?;
    if remaining == 0 {
        return Ok(());
    }
    let available = |frame: &Frame| -> Result<u32> {
        let count = moosik_codec_opus::packet_duration_frames(&frame.data)? as u32;
        count.checked_sub(frame.trim_start)
            .and_then(|n| n.checked_sub(frame.trim_end))
            .ok_or(Error::DecodeError("mkv: Opus trim exceeds a lace duration"))
    };
    let total = frames.iter().skip(first).try_fold(0u64, |total, frame| {
        total.checked_add(u64::from(available(frame)?))
            .ok_or(Error::DecodeError("mkv: Opus block duration overflow"))
    })?;
    if u64::from(remaining) > total {
        return decode_error("mkv: DiscardPadding exceeds the Opus block duration");
    }
    if padding_nanos < 0 {
        for frame in frames.iter_mut().skip(first) {
            let trim = remaining.min(available(frame)?);
            frame.trim_start += trim;
            remaining -= trim;
            if remaining == 0 { break; }
        }
    } else {
        for frame in frames.iter_mut().skip(first).rev() {
            let trim = remaining.min(available(frame)?);
            frame.trim_end += trim;
            remaining -= trim;
            if remaining == 0 { break; }
        }
    }
    debug_assert_eq!(remaining, 0);
    Ok(())
}

#[cfg(test)]
mod opus_block_padding_tests {
    use super::*;

    fn frame() -> Frame {
        Frame {
            track: 1,
            timestamp: 0,
            duration: 20,
            data: Box::from([0xf8, 0xff, 0xfe]), // 960 frames at 48 kHz.
            trim_start: 0,
            trim_end: 0,
        }
    }

    fn nanos(samples: u32) -> i64 {
        ((u64::from(samples) * 1_000_000_000 + 24_000) / 48_000) as i64
    }

    fn trims(frames: &VecDeque<Frame>) -> Vec<(u32, u32)> {
        frames.iter().map(|frame| (frame.trim_start, frame.trim_end)).collect()
    }

    #[test]
    fn positive_padding_spans_laces_backwards_without_touching_queued_blocks() {
        let mut frames: VecDeque<_> = (0..4).map(|_| frame()).collect();
        frames[0].trim_start = 17; // A preceding Block already in the queue.
        apply_opus_block_padding(&mut frames, 1, nanos(1440)).unwrap();
        assert_eq!(trims(&frames), [(17, 0), (0, 0), (0, 480), (0, 960)]);
        // A following Block is not part of the old padding's ownership.
        frames.push_back(frame());
        assert_eq!(trims(&frames)[4], (0, 0));
    }

    #[test]
    fn negative_padding_spans_laces_forwards() {
        let mut frames: VecDeque<_> = (0..4).map(|_| frame()).collect();
        apply_opus_block_padding(&mut frames, 1, -nanos(1440)).unwrap();
        assert_eq!(trims(&frames), [(0, 0), (960, 0), (480, 0), (0, 0)]);
    }

    #[test]
    fn whole_block_padding_is_valid_but_excess_is_rejected_before_mutation() {
        for sign in [-1, 1] {
            let mut frames: VecDeque<_> = (0..3).map(|_| frame()).collect();
            assert!(apply_opus_block_padding(&mut frames, 0, sign * nanos(2881)).is_err());
            assert_eq!(trims(&frames), [(0, 0), (0, 0), (0, 0)]);
            apply_opus_block_padding(&mut frames, 0, sign * nanos(2880)).unwrap();
            assert!(trims(&frames).iter().all(|(start, end)| start + end == 960));
        }
    }

    #[test]
    fn nanosecond_roundtrip_keeps_sample_exact_trim_and_existing_opposite_trim() {
        assert_eq!(nanos_to_opus_frames(6_500_000).unwrap(), 312);
        assert_eq!(nanos_to_opus_frames(nanos(1) as u64).unwrap(), 1);
        let mut frames = VecDeque::from([frame()]);
        frames[0].trim_start = 312;
        apply_opus_block_padding(&mut frames, 0, nanos(648)).unwrap();
        assert_eq!(trims(&frames), [(312, 648)]);
        assert!(apply_opus_block_padding(&mut frames, 0, nanos(1)).is_err());
        assert_eq!(trims(&frames), [(312, 648)]);
    }
}
