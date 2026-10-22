//! Native Opus decoding with explicit container timing ownership.
//!
//! Pre-skip, padding and seeking belong to the demuxer; see the crate README.

use std::ptr::NonNull;

use symphonia_core::audio::{
    AsAudioBufferRef, AudioBuffer, AudioBufferRef, Channels, Signal, SignalSpec,
};
use symphonia_core::codecs::{
    CodecDescriptor, CodecParameters, Decoder, DecoderOptions, FinalizeResult, CODEC_TYPE_OPUS,
};
use symphonia_core::errors::{decode_error, unsupported_error, Error, Result};
use symphonia_core::formats::Packet;
use symphonia_core::sample::SampleFormat;
use symphonia_core::support_codec;

const RATE: u32 = 48_000;
const MAX_FRAMES: usize = 5_760;
const MAX_PACKET_BYTES: usize = 61_440;

/// Validate an ordinary Opus packet and return its duration in 48 kHz frames.
///
/// This does not allocate or touch decoder state. Container readers can use it
/// to convert padding to frame trims without depending on container tick units.
pub fn packet_duration_frames(packet: &[u8]) -> Result<usize> {
    if packet.is_empty() || packet.len() > MAX_PACKET_BYTES {
        return decode_error("opus: empty or oversized audio packet");
    }
    // The byte cap is far below i32::MAX, so this conversion cannot truncate.
    let len = packet.len() as i32;
    // SAFETY: packet is non-empty and `len` is its checked exact byte length.
    let frames = unsafe {
        opusic_sys::opus_packet_get_nb_samples(packet.as_ptr(), len, RATE as i32)
    };
    if frames <= 0 || frames as usize > MAX_FRAMES {
        return decode_error("opus: invalid packet duration");
    }
    Ok(frames as usize)
}

#[derive(Clone, Copy, Debug)]
struct Header {
    channels: usize,
    gain_q8: i16,
}

fn header(bytes: &[u8]) -> Result<Header> {
    if bytes.len() < 19 || &bytes[..8] != b"OpusHead" {
        return decode_error("opus: missing or truncated OpusHead");
    }
    if bytes[8] > 15 {
        return unsupported_error("opus: incompatible OpusHead version");
    }
    if bytes[18] != 0 {
        return unsupported_error("opus: this decoder supports channel mapping family 0 only");
    }
    let channels = bytes[9] as usize;
    if !(1..=2).contains(&channels) {
        return unsupported_error("opus: mapping family 0 requires mono or stereo");
    }
    // The input sample rate and pre-skip are deliberately not decode settings:
    // the former is advisory, and the latter is handled by the demuxer.
    Ok(Header {
        channels,
        gain_q8: i16::from_le_bytes([bytes[16], bytes[17]]),
    })
}

/// A unique libopus allocation. No pointer is exposed outside this module.
struct State(NonNull<opusic_sys::OpusDecoder>);

// SAFETY: each State uniquely owns its allocation, all libopus operations require
// `&mut self`, and shared access never dereferences the pointer. Independent
// decoder states are thread-safe in the bundled reference library.
unsafe impl Send for State {}
// SAFETY: `&State` exposes no operation on the codec state; mutation is exclusive.
unsafe impl Sync for State {}

impl State {
    fn new(channels: usize) -> Result<Self> {
        let mut error = opusic_sys::OPUS_OK;
        // SAFETY: valid fixed sample rate, validated channel count, valid out ptr.
        let raw = unsafe {
            opusic_sys::opus_decoder_create(RATE as i32, channels as i32, &mut error)
        };
        let Some(raw) = NonNull::new(raw) else {
            return Err(Error::DecodeError("opus: could not allocate decoder"));
        };
        let state = Self(raw);
        if error != opusic_sys::OPUS_OK {
            // `state` also retires a non-null allocation on an unusual error.
            return decode_error("opus: decoder initialization failed");
        }
        Ok(state)
    }

    fn gain(&mut self, gain_q8: i16) -> Result<()> {
        // SAFETY: owned live decoder; SET_GAIN takes a promoted signed int.
        let status = unsafe {
            opusic_sys::opus_decoder_ctl(
                self.0.as_ptr(),
                opusic_sys::OPUS_SET_GAIN_REQUEST,
                i32::from(gain_q8),
            )
        };
        if status != opusic_sys::OPUS_OK {
            return decode_error("opus: setting output gain failed");
        }
        Ok(())
    }

    fn reset(&mut self) -> Result<()> {
        // SAFETY: owned live decoder; RESET_STATE takes no additional argument.
        let status = unsafe {
            opusic_sys::opus_decoder_ctl(self.0.as_ptr(), opusic_sys::OPUS_RESET_STATE)
        };
        if status != opusic_sys::OPUS_OK {
            return decode_error("opus: resetting decoder failed");
        }
        Ok(())
    }
}

impl Drop for State {
    fn drop(&mut self) {
        // SAFETY: pointer came from opus_decoder_create, is unique and live.
        unsafe { opusic_sys::opus_decoder_destroy(self.0.as_ptr()) };
    }
}

/// Symphonia decoder for ordinary mono/stereo Opus streams.
pub struct OpusDecoder {
    params: CodecParameters,
    state: State,
    header: Header,
    buffer: AudioBuffer<f32>,
    interleaved: Box<[f32]>,
    reset_failed: bool,
}

impl OpusDecoder {
    fn decode_packet(&mut self, packet: &Packet) -> Result<()> {
        if self.reset_failed {
            return decode_error("opus: decoder reset previously failed");
        }
        let expected = packet_duration_frames(&packet.data)?;
        // packet_duration_frames checked the exact byte count before conversion.
        let len = packet.data.len() as i32;
        let trim_start = packet.trim_start() as usize;
        let trim_end = packet.trim_end() as usize;
        if trim_start > expected || trim_end > expected - trim_start {
            return decode_error("opus: packet trimming exceeds decoded frame count");
        }
        // SAFETY: unique live state; packet length is bounded. The interleaved
        // allocation has MAX_FRAMES * channels writable f32 slots. libopus's
        // frame_size is per channel, not the length of the interleaved array.
        let frames = unsafe {
            opusic_sys::opus_decode_float(
                self.state.0.as_ptr(),
                packet.data.as_ptr(),
                len,
                self.interleaved.as_mut_ptr(),
                MAX_FRAMES as i32,
                0,
            )
        };
        if frames < 0 || frames as usize != expected {
            return decode_error("opus: packet decode failed or returned inconsistent duration");
        }
        let frames = frames as usize;
        self.buffer.render_reserved(Some(frames));
        for channel in 0..self.header.channels {
            let output = self.buffer.chan_mut(channel);
            for (frame, sample) in output.iter_mut().enumerate() {
                *sample = self.interleaved[frame * self.header.channels + channel];
            }
        }
        self.buffer.trim(trim_start, trim_end);
        Ok(())
    }
}

impl Decoder for OpusDecoder {
    fn try_new(params: &CodecParameters, _options: &DecoderOptions) -> Result<Self> {
        if params.codec != CODEC_TYPE_OPUS {
            return unsupported_error("opus: wrong codec type");
        }
        let Some(extra) = params.extra_data.as_deref() else {
            return decode_error("opus: OpusHead is required");
        };
        let header = header(extra)?;
        if let Some(channels) = params
            .channels
            .or_else(|| params.channel_layout.map(|layout| layout.into_channels()))
        {
            if channels.count() != header.channels {
                return decode_error("opus: container and OpusHead channel counts disagree");
            }
        }
        let channels = if header.channels == 1 {
            Channels::FRONT_LEFT
        } else {
            Channels::FRONT_LEFT | Channels::FRONT_RIGHT
        };
        let mut state = State::new(header.channels)?;
        state.gain(header.gain_q8)?;
        let mut params = params.clone();
        params
            .with_sample_rate(RATE)
            .with_sample_format(SampleFormat::F32)
            .with_channels(channels);
        Ok(Self {
            params,
            state,
            header,
            buffer: AudioBuffer::new(MAX_FRAMES as u64, SignalSpec::new(RATE, channels)),
            interleaved: vec![0.0; MAX_FRAMES * header.channels].into_boxed_slice(),
            reset_failed: false,
        })
    }

    fn supported_codecs() -> &'static [CodecDescriptor] {
        &[support_codec!(CODEC_TYPE_OPUS, "opus", "Opus (libopus)")]
    }

    fn reset(&mut self) {
        self.buffer.clear();
        self.reset_failed = self
            .state
            .reset()
            .and_then(|()| self.state.gain(self.header.gain_q8))
            .is_err();
    }

    fn codec_params(&self) -> &CodecParameters {
        &self.params
    }

    fn decode(&mut self, packet: &Packet) -> Result<AudioBufferRef<'_>> {
        self.buffer.clear();
        self.decode_packet(packet)?;
        Ok(self.buffer.as_audio_buffer_ref())
    }

    fn finalize(&mut self) -> FinalizeResult {
        FinalizeResult::default()
    }

    fn last_decoded(&self) -> AudioBufferRef<'_> {
        self.buffer.as_audio_buffer_ref()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn params(channels: u8, gain_q8: i16) -> CodecParameters {
        let mut bytes = b"OpusHead".to_vec();
        bytes.extend_from_slice(&[1, channels]);
        bytes.extend_from_slice(&312u16.to_le_bytes());
        bytes.extend_from_slice(&44_100u32.to_le_bytes());
        bytes.extend_from_slice(&gain_q8.to_le_bytes());
        bytes.push(0);
        let mut params = CodecParameters::new();
        params
            .for_codec(CODEC_TYPE_OPUS)
            .with_extra_data(bytes.into_boxed_slice());
        params
    }

    fn decoder(channels: u8, gain_q8: i16) -> OpusDecoder {
        OpusDecoder::try_new(&params(channels, gain_q8), &DecoderOptions::default()).unwrap()
    }

    fn silence() -> Packet {
        Packet::new_from_slice(0, 0, 960, &[0xf8, 0xff, 0xfe])
    }

    #[test]
    fn accepts_mono_and_stereo_at_48k_without_inventing_source_bit_depth() {
        for channels in [1, 2] {
            let mut decoder = decoder(channels, 0);
            let decoded = decoder.decode(&silence()).unwrap();
            assert_eq!(decoded.spec().rate, RATE);
            assert_eq!(decoded.spec().channels.count(), channels as usize);
            assert_eq!(decoded.frames(), 960);
            assert!(decoder.codec_params().bits_per_sample.is_none());
        }
    }

    #[test]
    fn packet_trim_is_the_only_skip_and_reset_does_not_add_header_preskip() {
        let mut decoder = decoder(2, 0);
        let mut packet = silence();
        packet.trim_start = 312;
        packet.trim_end = 17;
        assert_eq!(decoder.decode(&packet).unwrap().frames(), 960 - 312 - 17);
        assert_eq!(decoder.decode(&silence()).unwrap().frames(), 960);
        decoder.reset();
        assert_eq!(decoder.last_decoded().frames(), 0);
        assert_eq!(decoder.decode(&silence()).unwrap().frames(), 960);
    }

    #[test]
    fn fully_trimmed_packets_are_empty_and_do_not_trim_the_next_packet() {
        let mut decoder = decoder(1, 0);
        let mut packet = silence();
        packet.trim_start = 960;
        assert_eq!(decoder.decode(&packet).unwrap().frames(), 0);
        assert_eq!(decoder.decode(&silence()).unwrap().frames(), 960);
        assert_eq!(packet_duration_frames(&silence().data).unwrap(), 960);
        assert!(packet_duration_frames(&[0xfb, 63]).is_err());
    }

    #[test]
    fn malformed_packets_and_trims_clear_the_last_buffer() {
        let mut decoder = decoder(1, 0);
        decoder.decode(&silence()).unwrap();
        let mut packet = silence();
        packet.trim_start = 961;
        assert!(decoder.decode(&packet).is_err());
        assert_eq!(decoder.last_decoded().frames(), 0);
        let empty = Packet::new_from_slice(0, 0, 0, &[]);
        assert!(decoder.decode(&empty).is_err());
        assert_eq!(decoder.last_decoded().frames(), 0);
        let oversized = Packet::new_from_boxed_slice(
            0,
            0,
            0,
            vec![0; MAX_PACKET_BYTES + 1].into(),
        );
        assert!(decoder.decode(&oversized).is_err());
        assert_eq!(decoder.last_decoded().frames(), 0);
    }

    #[test]
    fn malformed_headers_and_unsupported_layouts_are_rejected() {
        for channels in [0, 3, 255] {
            assert!(OpusDecoder::try_new(&params(channels, 0), &DecoderOptions::default()).is_err());
        }
        for mutation in 0..4 {
            let mut params = params(2, 0);
            let mut extra = params.extra_data.take().unwrap().into_vec();
            match mutation {
                0 => extra.truncate(18),
                1 => extra[0] = b'X',
                2 => extra[8] = 16,
                3 => extra[18] = 1,
                _ => unreachable!(),
            }
            params.with_extra_data(extra.into_boxed_slice());
            assert!(OpusDecoder::try_new(&params, &DecoderOptions::default()).is_err());
        }
    }

    #[test]
    fn container_channel_disagreement_is_not_silently_downmixed() {
        let mut params = params(2, 0);
        params.with_channels(Channels::FRONT_LEFT);
        assert!(OpusDecoder::try_new(&params, &DecoderOptions::default()).is_err());
    }

    #[test]
    fn gain_is_configured_and_survives_reset() {
        for gain in [-1_536, 0, 1_536] {
            let mut decoder = decoder(2, gain);
            let mut actual: i32 = 0;
            // SAFETY: valid unique decoder; GET_GAIN requires a valid int out ptr.
            let status = unsafe {
                opusic_sys::opus_decoder_ctl(
                    decoder.state.0.as_ptr(),
                    opusic_sys::OPUS_GET_GAIN_REQUEST,
                    &mut actual as *mut i32,
                )
            };
            assert_eq!(status, opusic_sys::OPUS_OK);
            assert_eq!(actual, i32::from(gain));
            decoder.reset();
            assert!(!decoder.reset_failed);
            // SAFETY: same live owned decoder and valid out pointer after reset.
            unsafe {
                assert_eq!(
                    opusic_sys::opus_decoder_ctl(
                        decoder.state.0.as_ptr(),
                        opusic_sys::OPUS_GET_GAIN_REQUEST,
                        &mut actual as *mut i32,
                    ),
                    opusic_sys::OPUS_OK,
                );
            }
            assert_eq!(actual, i32::from(gain));
        }
    }
}
