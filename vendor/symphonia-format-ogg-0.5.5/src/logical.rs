// Symphonia
// Copyright (c) 2019-2022 The Project Symphonia Developers.
//
// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at https://mozilla.org/MPL/2.0/.

use std::collections::VecDeque;

use symphonia_core::codecs::CodecParameters;
use symphonia_core::errors::{decode_error, Result};
use symphonia_core::formats::Packet;

use super::common::SideData;
use super::mappings::Mapper;
use super::mappings::{MapResult, PacketParser};
use super::page::Page;
#[cfg(feature = "moosik-opus")]
use super::page::PageHeader;

use log::{debug, warn};

#[allow(dead_code)]
#[derive(Copy, Clone, Debug)]
struct Bound {
    seq: u32,
    ts: u64,
    delay: u64,
}

/// Opus packet duration needs only its first two bytes. Remember those bytes
/// across page boundaries without retaining compressed audio during probing.
#[cfg(feature = "moosik-opus")]
fn opus_page_duration(page: &Page<'_>, parser: &mut dyn PacketParser, prefix: &mut Vec<u8>) -> Option<u64> {
    if !page.header.is_continuation { prefix.clear(); }
    let mut packets = page.packets();
    let mut duration = 0u64;
    let mut valid = true;
    if page.header.is_continuation {
        if let Some(tail) = packets.next() {
            if prefix.is_empty() { valid = false; }
            else {
                let wanted = 2usize.saturating_sub(prefix.len());
                prefix.extend_from_slice(&tail[..tail.len().min(wanted)]);
                let value = parser.parse_next_packet_dur(prefix);
                if value == 0 || value > 5760 { valid = false; } else { duration += value; }
            }
            prefix.clear();
        } else {
            // No completed packet: preserve an already known TOC prefix.
            if let Some(partial) = packets.partial_packet() {
                if !prefix.is_empty() {
                    let wanted = 2usize.saturating_sub(prefix.len());
                    prefix.extend_from_slice(&partial[..partial.len().min(wanted)]);
                }
            }
            return Some(0);
        }
    }
    for packet in &mut packets {
        let value = parser.parse_next_packet_dur(packet);
        if value == 0 || value > 5760 { valid = false; }
        else { duration = duration.checked_add(value)?; }
    }
    if let Some(partial) = packets.partial_packet() {
        prefix.clear();
        prefix.extend_from_slice(&partial[..partial.len().min(2)]);
    }
    if valid { Some(duration) } else { None }
}

#[allow(dead_code)]
#[derive(Copy, Clone)]
struct PageInfo {
    seq: u32,
    absgp: u64,
}

#[derive(Default)]
pub struct InspectState {
    bound: Option<Bound>,
    parser: Option<Box<dyn PacketParser>>,
    #[cfg(feature = "moosik-opus")]
    opus_prefix: Vec<u8>,
}

#[cfg(feature = "moosik-opus")]
struct OpusTimeline {
    pre_skip: u64,
    initial_gp: Option<u64>,
    previous_gp: Option<u64>,
}

pub struct LogicalStream {
    mapper: Box<dyn Mapper>,
    packets: VecDeque<Packet>,
    part_buf: Vec<u8>,
    part_len: usize,
    prev_page_info: Option<PageInfo>,
    start_bound: Option<Bound>,
    end_bound: Option<Bound>,
    gapless: bool,
    #[cfg(feature = "moosik-opus")]
    opus: Option<OpusTimeline>,
}

impl LogicalStream {
    const MAX_PACKET_LEN: usize = 16 * 1024 * 1024;

    pub fn new(mapper: Box<dyn Mapper>, gapless: bool) -> Self {
        #[cfg(feature = "moosik-opus")]
        let opus = if mapper.codec_params().codec == symphonia_core::codecs::CODEC_TYPE_OPUS
        {
            Some(OpusTimeline {
                pre_skip: u64::from(mapper.codec_params().delay.unwrap_or(0)),
                initial_gp: None,
                previous_gp: None,
            })
        }
        else {
            None
        };
        LogicalStream {
            mapper,
            packets: Default::default(),
            part_buf: Default::default(),
            part_len: 0,
            prev_page_info: None,
            start_bound: None,
            end_bound: None,
            gapless,
            #[cfg(feature = "moosik-opus")]
            opus,
        }
    }

    /// Reset the logical stream after a page discontinuity.
    pub fn reset(&mut self) {
        self.part_len = 0;
        self.prev_page_info = None;
        self.packets.clear();
        self.mapper.reset();
        #[cfg(feature = "moosik-opus")]
        if let Some(opus) = &mut self.opus {
            // The stream's initial granule and header pre-skip survive seeks;
            // the previous page's position does not.
            opus.previous_gp = None;
        }
    }

    pub(crate) fn is_opus_gapless(&self) -> bool {
        #[cfg(feature = "moosik-opus")]
        { self.opus.is_some() }
        #[cfg(not(feature = "moosik-opus"))]
        { false }
    }

    pub(crate) fn needs_opus_end_probe(&self) -> bool {
        self.is_opus_gapless() && self.end_bound.is_none()
    }

    #[cfg(feature = "moosik-opus")]
    fn timestamp_opus_page(&mut self, header: &PageHeader, first: usize) -> Result<()> {
        if header.absgp == u64::MAX {
            return decode_error("ogg/opus: completed packets without a granule position");
        }
        let mut duration = 0u64;
        for packet in self.packets.iter().skip(first) {
            if packet.dur == 0 || packet.dur > 5760 {
                return decode_error("ogg/opus: invalid packet duration");
            }
            duration = match duration.checked_add(packet.dur) {
                Some(value) => value,
                None => return decode_error("ogg/opus: page duration overflow"),
            };
        }
        let opus = self.opus.as_mut().expect("Opus timeline");
        let first_audio_page = opus.initial_gp.is_none();
        let initial = if let Some(initial) = opus.initial_gp { initial } else {
            let initial = if header.absgp >= duration { header.absgp - duration }
                else if header.is_last_page && header.absgp >= opus.pre_skip { 0 }
                else { return decode_error("ogg/opus: invalid initial granule position"); };
            opus.initial_gp = Some(initial);
            self.start_bound = Some(Bound { seq: header.sequence, ts: initial, delay: opus.pre_skip });
            self.mapper.codec_params_mut().with_start_ts(0);
            initial
        };
        let cutoff = match initial.checked_add(opus.pre_skip) {
            Some(value) => value,
            None => return decode_error("ogg/opus: pre-skip position overflow"),
        };
        let raw_start = if !header.is_last_page {
            match header.absgp.checked_sub(duration) {
                Some(value) => value,
                None => return decode_error("ogg/opus: granule position precedes page duration"),
            }
        } else if first_audio_page { initial }
        else if let Some(previous) = opus.previous_gp { previous }
        else if let Some(end) = self.end_bound.filter(|end| end.seq == header.sequence) {
            match header.absgp.checked_add(end.delay).and_then(|end| end.checked_sub(duration)) {
                Some(value) => value,
                None => return decode_error("ogg/opus: invalid final page position"),
            }
        } else { return decode_error("ogg/opus: final page needs a preceding granule position"); };
        let raw_end = match raw_start.checked_add(duration) {
            Some(value) => value,
            None => return decode_error("ogg/opus: decoded position overflow"),
        };
        if header.is_last_page && (header.absgp < raw_start || header.absgp > raw_end || header.absgp < cutoff) {
            return decode_error("ogg/opus: invalid final granule position");
        }
        let audible_end = header.absgp.saturating_sub(cutoff);
        let mut raw = raw_start;
        for packet in self.packets.iter_mut().skip(first) {
            let decoded = packet.dur;
            let start_trim = cutoff.saturating_sub(raw).min(decoded);
            let end_trim = if header.is_last_page {
                (raw + decoded).saturating_sub(header.absgp).min(decoded - start_trim)
            } else { 0 };
            packet.trim_start = start_trim as u32;
            packet.trim_end = end_trim as u32;
            packet.dur = decoded - start_trim - end_trim;
            packet.ts = (raw + start_trim).saturating_sub(cutoff);
            if header.is_last_page { packet.ts = packet.ts.min(audible_end); }
            raw += decoded;
        }
        opus.previous_gp = Some(header.absgp);
        if header.is_last_page {
            let padding = raw_end - header.absgp;
            self.end_bound = Some(Bound { seq: header.sequence, ts: audible_end, delay: padding });
            self.mapper.codec_params_mut().with_n_frames(audible_end).with_padding(padding as u32);
        }
        Ok(())
    }

    #[cfg(feature = "moosik-opus")]
    fn inspect_opus_end_page(&mut self, mut state: InspectState, page: &Page<'_>) -> InspectState {
        if self.end_bound.is_some() { return state; }
        if state.parser.is_none() { state.parser = self.mapper.make_parser(); }
        let duration = match &mut state.parser {
            Some(parser) => opus_page_duration(page, &mut **parser, &mut state.opus_prefix),
            None => None,
        };
        if page.header.absgp == u64::MAX || page.num_packets() == 0 { return state; }
        let opus = self.opus.as_ref().expect("Opus timeline");
        if page.header.is_last_page {
            // A scan beginning on an isolated EOS page cannot infer its padding.
            // Leave the bound unset so the physical probe retries from the start.
            let initial = match opus.initial_gp { Some(value) => value, None => return state };
            let raw_start = state.bound.map(|bound| bound.ts).or_else(|| {
                self.start_bound.filter(|bound| bound.seq == page.header.sequence).map(|_| initial)
            });
            if let (Some(raw_start), Some(duration)) = (raw_start, duration) {
                if let (Some(raw_end), Some(cutoff)) = (raw_start.checked_add(duration), initial.checked_add(opus.pre_skip)) {
                    if page.header.absgp >= raw_start && page.header.absgp <= raw_end && page.header.absgp >= cutoff {
                        let end = page.header.absgp - cutoff;
                        let padding = raw_end - page.header.absgp;
                        if padding <= u32::MAX as u64 {
                            self.end_bound = Some(Bound { seq: page.header.sequence, ts: end, delay: padding });
                            self.mapper.codec_params_mut().with_n_frames(end).with_padding(padding as u32);
                        }
                    }
                }
            }
        }
        state.bound = Some(Bound { seq: page.header.sequence, ts: page.header.absgp, delay: 0 });
        state
    }

    /// Returns true if the stream is ready.
    pub fn is_ready(&self) -> bool {
        self.mapper.is_ready()
    }

    /// Get the `CodecParameters` for the logical stream.
    pub fn codec_params(&self) -> &CodecParameters {
        self.mapper.codec_params()
    }

    /// Reads a page.
    pub fn read_page(&mut self, page: &Page<'_>) -> Result<Vec<SideData>> {
        // Side data vector. This will not allocate unless data is pushed to it (normal case).
        let mut side_data = Vec::new();

        // If the last sequence number is available, detect non-monotonicity and discontinuities
        // in the stream. In these cases, clear any partial packet data.
        if let Some(last_ts) = &self.prev_page_info {
            if page.header.sequence < last_ts.seq {
                warn!("detected stream page non-monotonicity");
                self.part_len = 0;
                #[cfg(feature = "moosik-opus")]
                if let Some(opus) = &mut self.opus { opus.previous_gp = None; }
            }
            else if page.header.sequence - last_ts.seq > 1 {
                warn!(
                    "detected stream discontinuity of {} page(s)",
                    page.header.sequence - last_ts.seq
                );
                self.part_len = 0;
                #[cfg(feature = "moosik-opus")]
                if let Some(opus) = &mut self.opus { opus.previous_gp = None; }
            }
        }

        self.prev_page_info =
            Some(PageInfo { seq: page.header.sequence, absgp: page.header.absgp });

        let mut iter = page.packets();

        // If there is partial packet data buffered, a continuation page is expected.
        if !page.header.is_continuation && self.part_len > 0 {
            warn!("expected a continuation page");

            // Clear partial packet data.
            self.part_len = 0;
        }

        // If there is no partial packet data buffered, a continuation page is not expected.
        if page.header.is_continuation && self.part_len == 0 {
            // If the continuation page contains packets, drop the first packet since it would
            // require partial packet data to be complete. Otherwise, ignore this page entirely.
            if page.num_packets() > 0 {
                warn!("unexpected continuation page, ignoring incomplete first packet");
                iter.next();
            }
            else {
                warn!("unexpected continuation page, ignoring page");
                return Ok(side_data);
            }
        }

        let num_prev_packets = self.packets.len();

        for buf in &mut iter {
            // Get a packet with data from the partial packet buffer, the page, or both.
            let data = self.get_packet(buf);

            // Perform packet mapping. If the packet contains stream data, queue it onto the packet
            // queue. If it contains side data, then add it to the side data list. Ignore other
            // types of packet data.
            match self.mapper.map_packet(&data) {
                Ok(MapResult::StreamData { dur }) => {
                    // Create a packet.
                    self.packets.push_back(Packet::new_from_boxed_slice(
                        page.header.serial,
                        0,
                        dur,
                        data,
                    ));
                }
                Ok(MapResult::SideData { data }) => side_data.push(data),
                Err(e) => {
                    if self.is_opus_gapless() {
                        return Err(e);
                    }
                    warn!("mapping packet failed ({}), skipping", e)
                }
                _ => (),
            }
        }

        // If the page contains partial packet data, then save the partial packet data for later
        // as the packet will be completed on a later page.
        if let Some(buf) = iter.partial_packet() {
            self.save_partial_packet(buf)?;
        }

        // The number of packets from this page that were queued.
        let num_new_packets = self.packets.len() - num_prev_packets;

        #[cfg(feature = "moosik-opus")]
        if num_new_packets > 0 && self.opus.is_some() {
            self.timestamp_opus_page(&page.header, num_prev_packets)?;
            return Ok(side_data);
        }

        if num_new_packets > 0 {
            // Get the start delay.
            let start_delay = self.start_bound.as_ref().map_or(0, |b| b.delay);

            // Assign timestamps by first calculating the timestamp of one past the last sample in
            // in the last packet of this page, add the start delay.
            let mut page_end_ts =
                self.mapper.absgp_to_ts(page.header.absgp).saturating_add(start_delay);

            // If this is the last page, then add the end delay to the timestamp.
            if page.header.is_last_page {
                let end_delay = self.end_bound.as_ref().map_or(0, |b| b.delay);
                page_end_ts = page_end_ts.saturating_add(end_delay);
            }

            // Then, iterate over the newly added packets in reverse order and subtract their
            // cumulative duration at each iteration to get the timestamp of the first sample
            // in each packet.
            let mut page_dur = 0u64;

            for packet in self.packets.iter_mut().rev().take(num_new_packets) {
                page_dur = page_dur.saturating_add(packet.dur);
                packet.ts = page_end_ts.saturating_sub(page_dur);
            }

            if self.gapless {
                for packet in self.packets.iter_mut().rev().take(num_new_packets) {
                    symphonia_core::formats::util::trim_packet(
                        packet,
                        start_delay as u32,
                        self.end_bound.as_ref().map(|b| b.ts),
                    );
                }
            }
        }

        Ok(side_data)
    }

    /// Returns true if the logical stream has packets buffered.
    pub fn has_packets(&self) -> bool {
        !self.packets.is_empty()
    }

    /// Examine, but do not consume, the next packet.
    pub fn peek_packet(&self) -> Option<&Packet> {
        self.packets.front()
    }

    /// Consumes and returns the next packet.
    pub fn next_packet(&mut self) -> Option<Packet> {
        self.packets.pop_front()
    }

    /// Consumes the next packet.
    pub fn consume_packet(&mut self) {
        self.packets.pop_front();
    }

    /// Examine the first page of the non-setup codec bitstream to obtain the start time and start
    /// delay parameters.
    pub fn inspect_start_page(&mut self, page: &Page<'_>) {
        // The first Opus audio page is timestamped and trimmed when read,
        // before these already queued packets can be returned to the caller.
        if self.is_opus_gapless() {
            return;
        }
        if self.start_bound.is_some() {
            debug!("start page already found");
            return;
        }

        let mut parser = match self.mapper.make_parser() {
            Some(parser) => parser,
            _ => {
                debug!("failed to make start bound packet parser");
                return;
            }
        };

        // Calculate the page duration.
        let mut page_dur = 0u64;

        for buf in page.packets() {
            page_dur = page_dur.saturating_add(parser.parse_next_packet_dur(buf));
        }

        let page_end_ts = self.mapper.absgp_to_ts(page.header.absgp);

        // If the page timestamp is >= the page duration, then the stream starts at timestamp 0 or
        // a positive start time.
        let bound = if page_end_ts >= page_dur {
            Bound { seq: page.header.sequence, ts: page_end_ts - page_dur, delay: 0 }
        }
        else {
            // If the page timestamp < the page duration, then the difference is the start delay.
            Bound { seq: page.header.sequence, ts: 0, delay: page_dur - page_end_ts }
        };

        // Update codec parameters.
        let codec_params = self.mapper.codec_params_mut();

        codec_params.with_start_ts(bound.ts);

        if bound.delay > 0 {
            codec_params.with_delay(bound.delay as u32);
        }

        // Update start bound.
        self.start_bound = Some(bound);
    }

    /// Examines one or more of the last pages of the codec bitstream to obtain the end time and
    /// end delay parameters. To obtain the end delay, at a minimum, the last two pages are
    /// required. The state returned by each iteration of this function should be passed into the
    /// subsequent iteration.
    pub fn inspect_end_page(&mut self, mut state: InspectState, page: &Page<'_>) -> InspectState {
        #[cfg(feature = "moosik-opus")]
        if self.opus.is_some() {
            return self.inspect_opus_end_page(state, page);
        }
        if self.end_bound.is_some() {
            debug!("end page already found");
            return state;
        }

        // Get and/or create the sniffer state.
        let parser = match &mut state.parser {
            Some(parser) => parser,
            None => {
                state.parser = self.mapper.make_parser();

                if let Some(parser) = &mut state.parser {
                    parser
                }
                else {
                    debug!("failed to make end bound packet parser");
                    return state;
                }
            }
        };

        let start_delay = self.start_bound.as_ref().map_or(0, |b| b.delay);

        // The actual page end timestamp is the absolute granule position + the start delay.
        let page_end_ts = self
            .mapper
            .absgp_to_ts(page.header.absgp)
            .saturating_add(if self.gapless { 0 } else { start_delay });

        // Calculate the page duration. Note that even though only the last page uses this duration,
        // it is important to feed the packet parser so that the first packet of the final page
        // doesn't have a duration of 0 due to lapping on some codecs.
        let mut page_dur = 0u64;

        for buf in page.packets() {
            page_dur = page_dur.saturating_add(parser.parse_next_packet_dur(buf));
        }

        // The end delay can only be determined if this is the last page, and the timstamp of the
        // second last page is known.
        let end_delay = if page.header.is_last_page {
            if let Some(last_bound) = &state.bound {
                // The real ending timestamp of the decoded data is the timestamp of the previous
                // page plus the decoded duration of this page.
                let actual_page_end_ts = last_bound.ts.saturating_add(page_dur);

                // Any samples after the stated timestamp of this page are considered delay samples.
                actual_page_end_ts.saturating_sub(page_end_ts)
            }
            else {
                // Don't have the timestamp of the previous page so it is not possible to
                // calculate the end delay.
                0
            }
        }
        else {
            // Only the last page can have an end delay.
            0
        };

        let bound = Bound { seq: page.header.sequence, ts: page_end_ts, delay: end_delay };

        // If this is the last page, update the codec parameters.
        if page.header.is_last_page {
            let codec_params = self.mapper.codec_params_mut();

            // Do not report the end delay if gapless is enabled.
            let block_end_ts = bound.ts + if self.gapless { 0 } else { bound.delay };

            if block_end_ts > codec_params.start_ts {
                codec_params.with_n_frames(block_end_ts - codec_params.start_ts);
            }

            if bound.delay > 0 {
                codec_params.with_padding(bound.delay as u32);
            }

            self.end_bound = Some(bound)
        }

        // Update the state's bound.
        state.bound = Some(bound);

        state
    }

    /// Examine a page and return the start and end timestamps as a tuple.
    pub fn inspect_page(&mut self, page: &Page<'_>) -> (u64, u64) {
        #[cfg(feature = "moosik-opus")]
        if let Some(opus) = &self.opus {
            if page.header.absgp == u64::MAX || page.num_packets() == 0 {
                let start = opus.initial_gp.unwrap_or(0);
                return (start, start);
            }
            let mut prefix = Vec::new();
            let mut parser = self.mapper.make_parser().expect("Opus has a packet parser");
            let dur = opus_page_duration(page, &mut *parser, &mut prefix).unwrap_or(0);
            let padding = if page.header.is_last_page {
                self.end_bound.as_ref().map_or(0, |bound| bound.delay)
            } else { 0 };
            let end = page.header.absgp.saturating_sub(opus.pre_skip);
            let start = page.header.absgp.saturating_add(padding)
                .saturating_sub(dur).saturating_sub(opus.pre_skip)
                .max(opus.initial_gp.unwrap_or(0));
            return (start.min(end), end);
        }
        // Get the start delay.
        let start_delay = self.start_bound.as_ref().map_or(0, |b| b.delay);

        // Get the cumulative duration of all packets within this page.
        let mut page_dur = 0u64;

        if let Some(mut parser) = self.mapper.make_parser() {
            for buf in page.packets() {
                page_dur = page_dur.saturating_add(parser.parse_next_packet_dur(buf));
            }
        }

        // If this is the final page, get the end delay.
        let end_delay = if page.header.is_last_page {
            self.end_bound.as_ref().map_or(0, |b| b.delay)
        }
        else {
            0
        };

        // The total delay.
        let delay = start_delay + end_delay;

        // Add the total delay to the page end timestamp.
        let page_end_ts = self.mapper.absgp_to_ts(page.header.absgp).saturating_add(delay);

        // Get the page start timestamp of the page by subtracting the cumulative packet duration.
        let page_start_ts = page_end_ts.saturating_sub(page_dur);

        if !self.gapless {
            // If gapless playback is disabled, then report the start and end timestamps with the
            // delays incorporated.
            (page_start_ts, page_end_ts)
        }
        else {
            // If gapless playback is enabled, report the start and end timestamps without the
            // delays.
            (page_start_ts.saturating_sub(delay), page_end_ts.saturating_sub(delay))
        }
    }

    fn get_packet(&mut self, packet_buf: &[u8]) -> Box<[u8]> {
        if self.part_len == 0 {
            Box::from(packet_buf)
        }
        else {
            let mut buf = vec![0u8; self.part_len + packet_buf.len()];

            // Split packet buffer into two portions: saved and new.
            let (vec0, vec1) = buf.split_at_mut(self.part_len);

            // Copy and consume the saved partial packet.
            vec0.copy_from_slice(&self.part_buf[..self.part_len]);
            self.part_len = 0;

            // Read the remainder of the partial packet from the page.
            vec1.copy_from_slice(packet_buf);

            buf.into_boxed_slice()
        }
    }

    fn save_partial_packet(&mut self, buf: &[u8]) -> Result<()> {
        let new_part_len = self.part_len + buf.len();

        if new_part_len > self.part_buf.len() {
            // Do not exceed an a certain limit to prevent unbounded memory growth.
            if new_part_len > LogicalStream::MAX_PACKET_LEN {
                return decode_error("ogg: packet buffer would exceed max size");
            }

            // New partial packet buffer size, rounded up to the nearest 8K block.
            let new_buf_len = (new_part_len + (8 * 1024 - 1)) & !(8 * 1024 - 1);
            debug!("grow packet buffer to {} bytes", new_buf_len);

            self.part_buf.resize(new_buf_len, Default::default());
        }

        self.part_buf[self.part_len..new_part_len].copy_from_slice(buf);
        self.part_len = new_part_len;

        Ok(())
    }
}
