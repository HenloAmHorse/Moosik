use core::fmt;
use core::time::Duration;
use symphonia::{
    core::{
        audio::{AudioBufferRef, SampleBuffer, SignalSpec},
        codecs::{Decoder, DecoderOptions, CODEC_TYPE_NULL, CODEC_TYPE_OPUS},
        errors::Error,
        formats::{FormatOptions, FormatReader, SeekedTo, Track},
        io::MediaSourceStream,
        meta::MetadataOptions,
        probe::Hint,
        units::{self, Time, TimeBase},
    },
    default::get_probe,
};

use crate::{source, Source};

use super::DecoderError;

// Decoder errors are not considered fatal.
// The correct action is to just get a new packet and try again.
// But a decode error in more than 3 consecutive packets is fatal.
const MAX_DECODE_RETRIES: usize = 3;

pub(crate) struct SymphoniaDecoder {
    decoder: Box<dyn Decoder>,
    current_frame_offset: usize,
    format: Box<dyn FormatReader>,
    total_duration: Option<Time>,
    buffer: SampleBuffer<i16>,
    spec: SignalSpec,
    track_id: u32,
    time_base: TimeBase,
}

impl SymphoniaDecoder {
    pub(crate) fn new(
        mss: MediaSourceStream,
        extension: Option<&str>,
    ) -> Result<Self, DecoderError> {
        match SymphoniaDecoder::init(mss, extension) {
            Err(e) => match e {
                Error::IoError(e) => Err(DecoderError::IoError(e.to_string())),
                Error::DecodeError(e) => Err(DecoderError::DecodeError(e)),
                Error::SeekError(e) => Err(DecoderError::IoError(format!("Container positioning failed: {e:?}"))),
                Error::Unsupported(_) => Err(DecoderError::UnrecognizedFormat),
                Error::LimitError(e) => Err(DecoderError::LimitError(e)),
                Error::ResetRequired => Err(DecoderError::ResetRequired),
            },
            Ok(Some(decoder)) => Ok(decoder),
            Ok(None) => Err(DecoderError::NoStreams),
        }
    }

    pub(crate) fn into_inner(self) -> MediaSourceStream {
        self.format.into_inner()
    }

    fn init(
        mss: MediaSourceStream,
        extension: Option<&str>,
    ) -> symphonia::core::errors::Result<Option<SymphoniaDecoder>> {
        let mut hint = Hint::new();
        if let Some(ext) = extension {
            hint.with_extension(ext);
        }
        let format_opts: FormatOptions = FormatOptions {
            enable_gapless: true,
            ..Default::default()
        };
        let metadata_opts: MetadataOptions = Default::default();
        let mut probed = get_probe().format(&hint, mss, &format_opts, &metadata_opts)?;

        let track = match selected_track(probed.format.as_ref()) {
            Some(track) => track,
            None => return Ok(None),
        };
        let track_id = track.id;
        let mut decoder = symphonia::default::get_codecs()
            .make(&track.codec_params, &DecoderOptions::default())?;
        let track_time_base = track.codec_params.time_base;
        let total_duration = track
            .codec_params
            .time_base
            .filter(|base| base.numer != 0 && base.denom != 0)
            .zip(track.codec_params.n_frames)
            .map(|(base, frames)| base.calc_time(frames));
        let (spec, buffer) = read_audio_buffer(
            probed.format.as_mut(),
            decoder.as_mut(),
            track_id,
        )?;
        let time_base = track_time_base
            .filter(|base| base.numer != 0 && base.denom != 0)
            .unwrap_or_else(|| TimeBase::new(1, spec.rate));
        Ok(Some(SymphoniaDecoder {
            decoder,
            current_frame_offset: 0,
            format: probed.format,
            total_duration,
            buffer,
            spec,
            track_id,
            time_base,
        }))
    }

    #[inline]
    fn get_buffer(decoded: AudioBufferRef, spec: &SignalSpec) -> SampleBuffer<i16> {
        let duration = units::Duration::from(decoded.capacity() as u64);
        let mut buffer = SampleBuffer::<i16>::new(duration, *spec);
        buffer.copy_interleaved_ref(decoded);
        buffer
    }
}

impl Source for SymphoniaDecoder {
    #[inline]
    fn current_frame_len(&self) -> Option<usize> {
        Some(self.buffer.samples().len())
    }

    #[inline]
    fn channels(&self) -> u16 {
        self.spec.channels.count() as u16
    }

    #[inline]
    fn sample_rate(&self) -> u32 {
        self.spec.rate
    }

    #[inline]
    fn total_duration(&self) -> Option<Duration> {
        self.total_duration
            .map(Duration::from)
    }

    fn try_seek(&mut self, pos: Duration) -> Result<(), source::SeekError> {
        use symphonia::core::formats::{SeekMode, SeekTo};

        let seek_beyond_end = self
            .total_duration()
            .is_some_and(|dur| dur.saturating_sub(pos).as_millis() < 1);

        let time = if seek_beyond_end {
            let time = self.total_duration.expect("if guarantees this is Some");
            skip_back_a_tiny_bit(time) // some decoders can only seek to just before the end
        } else {
            pos.as_secs_f64().into()
        };

        // make sure the next sample is for the right channel
        let to_skip = self.current_frame_offset % self.channels() as usize;

        let seek_res = self
            .format
            .seek(
                SeekMode::Accurate,
                SeekTo::Time {
                    time,
                    track_id: Some(self.track_id),
                },
            )
            .map_err(SeekError::BaseSeek)?;

        let target = if seek_beyond_end { Duration::from(time) } else { pos };
        self.refine_position(seek_res, target)?;
        self.current_frame_offset = self
            .current_frame_offset
            .saturating_add(to_skip)
            .min(self.buffer.len());

        Ok(())
    }
}

/// Error returned when the try_seek implementation of the symphonia decoder fails.
#[derive(Debug)]
pub enum SeekError {
    /// Could not get next packet while refining seek position
    Refining(symphonia::core::errors::Error),
    /// Format reader failed to seek
    BaseSeek(symphonia::core::errors::Error),
    /// Decoding failed retrying on the next packet failed
    Retrying(symphonia::core::errors::Error),
    /// Decoding failed on multiple consecutive packets
    Decoding(symphonia::core::errors::Error),
}
impl fmt::Display for SeekError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            SeekError::Refining(err) => {
                write!(
                    f,
                    "Could not get next packet while refining seek position: {:?}",
                    err
                )
            }
            SeekError::BaseSeek(err) => {
                write!(f, "Format reader failed to seek: {:?}", err)
            }
            SeekError::Retrying(err) => {
                write!(
                    f,
                    "Decoding failed retrying on the next packet failed: {:?}",
                    err
                )
            }
            SeekError::Decoding(err) => {
                write!(
                    f,
                    "Decoding failed on multiple consecutive packets: {:?}",
                    err
                )
            }
        }
    }
}
impl std::error::Error for SeekError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            SeekError::Refining(err) => Some(err),
            SeekError::BaseSeek(err) => Some(err),
            SeekError::Retrying(err) => Some(err),
            SeekError::Decoding(err) => Some(err),
        }
    }
}

impl SymphoniaDecoder {
    /// Note frame offset must be set after
    fn refine_position(&mut self, seek_res: SeekedTo, target: Duration) -> Result<(), source::SeekError> {
        self.decoder.reset();
        self.buffer.clear();
        self.current_frame_offset = 0;
        let mut errors = 0;
        // The Moosik Opus demuxers may deliberately seek from the known stream
        // beginning. In that case count decoded audible frames, not rounded
        // container timestamps (Matroska commonly uses millisecond ticks).
        let mut frames_from_start =
            (self.decoder.codec_params().codec == CODEC_TYPE_OPUS && seek_res.actual_ts == 0)
                .then(|| duration_to_frames(target, self.spec.rate));
        loop {
            let packet = self.format.next_packet().map_err(SeekError::Refining)?;
            if packet.track_id() != self.track_id {
                continue;
            }
            // Decode preceding packets too: codecs such as Opus require their
            // history after a reset. Demuxer-provided fully trimmed pre-roll is
            // a successful empty buffer, never an end-of-stream indication.
            let decoded = match self.decoder.decode(&packet) {
                Ok(decoded) => decoded,
                Err(error @ Error::DecodeError(_)) => {
                    errors += 1;
                    if errors > MAX_DECODE_RETRIES {
                        return Err(SeekError::Decoding(error).into());
                    }
                    continue;
                }
                Err(error) => return Err(SeekError::Decoding(error).into()),
            };
            errors = 0;
            let frames_to_pass = frames_from_start.unwrap_or_else(|| {
                ticks_to_frames(
                    seek_res.required_ts.saturating_sub(packet.ts()),
                    self.time_base,
                    decoded.spec().rate,
                )
            });
            if frames_to_pass >= decoded.frames() {
                if let Some(remaining) = &mut frames_from_start {
                    *remaining -= decoded.frames();
                }
                continue;
            }
            decoded.spec().clone_into(&mut self.spec);
            self.buffer = SymphoniaDecoder::get_buffer(decoded, &self.spec);
            self.current_frame_offset = frames_to_pass
                .saturating_mul(self.spec.channels.count())
                .min(self.buffer.len());
            return Ok(());
        }
    }
}

fn skip_back_a_tiny_bit(time: Time) -> Time {
    Duration::from(time)
        .saturating_sub(Duration::from_micros(100))
        .into()
}

impl Iterator for SymphoniaDecoder {
    type Item = i16;

    #[inline]
    fn next(&mut self) -> Option<i16> {
        if self.current_frame_offset >= self.buffer.len() {
            let (spec, buffer) = read_audio_buffer(
                self.format.as_mut(),
                self.decoder.as_mut(),
                self.track_id,
            )
            .ok()?;
            self.spec = spec;
            self.buffer = buffer;
            self.current_frame_offset = 0;
        }

        let sample = *self.buffer.samples().get(self.current_frame_offset)?;
        self.current_frame_offset += 1;

        Some(sample)
    }
}

fn selected_track(format: &dyn FormatReader) -> Option<&Track> {
    let codecs = symphonia::default::get_codecs();
    let supported = |track: &&Track| {
        track.codec_params.codec != CODEC_TYPE_NULL
            && codecs.get_codec(track.codec_params.codec).is_some()
    };
    format
        .default_track()
        .filter(supported)
        .or_else(|| format.tracks().iter().find(supported))
}

fn read_audio_buffer(
    format: &mut dyn FormatReader,
    decoder: &mut dyn Decoder,
    track_id: u32,
) -> symphonia::core::errors::Result<(SignalSpec, SampleBuffer<i16>)> {
    let mut errors = 0;
    loop {
        let packet = format.next_packet()?;
        if packet.track_id() != track_id {
            continue;
        }
        match decoder.decode(&packet) {
            Ok(decoded) => {
                errors = 0;
                if decoded.frames() == 0 {
                    continue;
                }
                let spec = *decoded.spec();
                return Ok((spec, SymphoniaDecoder::get_buffer(decoded, &spec)));
            }
            Err(error @ Error::DecodeError(_)) => {
                errors += 1;
                if errors > MAX_DECODE_RETRIES {
                    return Err(error);
                }
            }
            Err(error) => return Err(error),
        }
    }
}

fn ticks_to_frames(ticks: u64, base: TimeBase, rate: u32) -> usize {
    // Up to 64 + 32 + 32 bits: an exact u128 rational conversion, rounded up
    // so seeking does not return a sample before the requested timestamp.
    let product = u128::from(ticks) * u128::from(base.numer) * u128::from(rate);
    let denominator = u128::from(base.denom);
    let frames = product / denominator + u128::from(product % denominator != 0);
    frames.min(usize::MAX as u128) as usize
}

fn duration_to_frames(duration: Duration, rate: u32) -> usize {
    let frames = duration.as_nanos() * u128::from(rate) / 1_000_000_000;
    frames.min(usize::MAX as u128) as usize
}

#[cfg(test)]
mod moosik_tests {
    use super::*;
    use std::collections::VecDeque;
    use std::io::{Cursor, ErrorKind};
    use std::sync::{Arc, Mutex};
    use symphonia::core::audio::{AsAudioBufferRef, AudioBuffer, Channels, Signal};
    use symphonia::core::codecs::{CodecDescriptor, CodecParameters, FinalizeResult, CODEC_TYPE_PCM_S16LE};
    use symphonia::core::formats::{Cue, Packet, SeekMode, SeekTo};
    use symphonia::core::meta::{Metadata, MetadataLog};

    struct FakeFormat {
        packets: VecDeque<Packet>,
        tracks: Vec<Track>,
        metadata: MetadataLog,
        sought_track: Arc<Mutex<Option<u32>>>,
    }

    impl FormatReader for FakeFormat {
        fn try_new(_: MediaSourceStream, _: &FormatOptions) -> symphonia::core::errors::Result<Self> {
            unreachable!("constructed directly by private unit tests")
        }
        fn cues(&self) -> &[Cue] { &[] }
        fn metadata(&mut self) -> Metadata<'_> { self.metadata.metadata() }
        fn seek(&mut self, _: SeekMode, to: SeekTo) -> symphonia::core::errors::Result<SeekedTo> {
            let track_id = match to {
                SeekTo::Time { track_id, .. } => track_id,
                SeekTo::TimeStamp { track_id, .. } => Some(track_id),
            };
            *self.sought_track.lock().unwrap() = track_id;
            Ok(SeekedTo { track_id: 7, required_ts: 15, actual_ts: 0 })
        }
        fn tracks(&self) -> &[Track] { &self.tracks }
        fn next_packet(&mut self) -> symphonia::core::errors::Result<Packet> {
            self.packets.pop_front().ok_or_else(|| {
                Error::IoError(std::io::Error::new(ErrorKind::UnexpectedEof, "private fixture ended"))
            })
        }
        fn into_inner(self: Box<Self>) -> MediaSourceStream {
            MediaSourceStream::new(Box::new(Cursor::new(Vec::<u8>::new())), Default::default())
        }
    }

    struct FakeDecoder {
        params: CodecParameters,
        buffer: AudioBuffer<i16>,
        log: Arc<Mutex<Vec<u8>>>,
    }

    impl Decoder for FakeDecoder {
        fn try_new(_: &CodecParameters, _: &DecoderOptions) -> symphonia::core::errors::Result<Self> {
            unreachable!("constructed directly by private unit tests")
        }
        fn supported_codecs() -> &'static [CodecDescriptor] { &[] }
        fn reset(&mut self) {
            self.buffer.clear();
            self.log.lock().unwrap().push(254);
        }
        fn codec_params(&self) -> &CodecParameters { &self.params }
        fn decode(&mut self, packet: &Packet) -> symphonia::core::errors::Result<AudioBufferRef<'_>> {
            self.buffer.clear();
            let marker = packet.data[0];
            self.log.lock().unwrap().push(marker);
            if marker == 255 {
                return Err(Error::DecodeError("deliberately bad fixture packet"));
            }
            let frames = if marker == 0 { 0 } else { 480 };
            self.buffer.render_reserved(Some(frames));
            for (frame, sample) in self.buffer.chan_mut(0).iter_mut().enumerate() {
                *sample = i16::from(marker) * 1_000 + frame as i16;
            }
            Ok(self.buffer.as_audio_buffer_ref())
        }
        fn finalize(&mut self) -> FinalizeResult { FinalizeResult::default() }
        fn last_decoded(&self) -> AudioBufferRef<'_> { self.buffer.as_audio_buffer_ref() }
    }

    fn packet(track: u32, timestamp: u64, marker: u8) -> Packet {
        Packet::new_from_slice(track, timestamp, 10, &[marker])
    }

    fn fake(packets: Vec<Packet>) -> (SymphoniaDecoder, Arc<Mutex<Vec<u8>>>, Arc<Mutex<Option<u32>>>) {
        fake_with_codec(packets, CODEC_TYPE_PCM_S16LE)
    }

    fn fake_with_codec(
        packets: Vec<Packet>,
        codec: symphonia::core::codecs::CodecType,
    ) -> (SymphoniaDecoder, Arc<Mutex<Vec<u8>>>, Arc<Mutex<Option<u32>>>) {
        let spec = SignalSpec::new(48_000, Channels::FRONT_LEFT);
        let mut params = CodecParameters::new();
        params
            .for_codec(codec)
            .with_channels(Channels::FRONT_LEFT)
            .with_sample_rate(48_000)
            .with_time_base(TimeBase::new(1, 1_000))
            .with_n_frames(1_137);
        let log = Arc::new(Mutex::new(Vec::new()));
        let sought_track = Arc::new(Mutex::new(None));
        let decoder = SymphoniaDecoder {
            decoder: Box::new(FakeDecoder {
                params: params.clone(),
                buffer: AudioBuffer::new(480, spec),
                log: log.clone(),
            }),
            current_frame_offset: 0,
            format: Box::new(FakeFormat {
                packets: packets.into(),
                tracks: vec![
                    Track { id: 99, codec_params: CodecParameters::new(), language: None },
                    Track { id: 7, codec_params: params, language: None },
                ],
                metadata: MetadataLog::default(),
                sought_track: sought_track.clone(),
            }),
            total_duration: Some(Time::new(1, 0.137)),
            buffer: SampleBuffer::new(0, spec),
            spec,
            track_id: 7,
            time_base: TimeBase::new(1, 1_000),
        };
        (decoder, log, sought_track)
    }

    #[test]
    fn selected_track_ignores_an_unsupported_default() {
        let (source, _, _) = fake(Vec::new());
        assert_eq!(selected_track(source.format.as_ref()).unwrap().id, 7);
        assert_eq!(source.total_duration(), Some(Duration::from_millis(1_137)));
    }

    #[test]
    fn iteration_filters_other_tracks_and_continues_after_empty_audio() {
        let (mut source, log, _) = fake(vec![
            packet(99, 0, 255),
            packet(7, 0, 0),
            packet(99, 0, 255),
            packet(7, 0, 2),
        ]);
        assert_eq!(source.next(), Some(2_000));
        assert_eq!(*log.lock().unwrap(), vec![0, 2]);
    }

    #[test]
    fn seeking_decodes_history_and_uses_millisecond_time_base() {
        let (mut source, log, sought_track) = fake(vec![
            packet(99, 0, 255),
            packet(7, 0, 1),
            packet(7, 10, 2),
        ]);
        source.try_seek(Duration::from_millis(15)).unwrap();
        assert_eq!(*sought_track.lock().unwrap(), Some(7));
        assert_eq!(*log.lock().unwrap(), vec![254, 1, 2]);
        assert_eq!(source.next(), Some(2_240));
    }

    #[test]
    fn seeking_decodes_fully_trimmed_preroll_without_returning_eof() {
        let (mut source, log, _) = fake(vec![packet(7, 0, 0), packet(7, 15, 3)]);
        source.try_seek(Duration::from_millis(15)).unwrap();
        assert_eq!(*log.lock().unwrap(), vec![254, 0, 3]);
        assert_eq!(source.next(), Some(3_000));
    }

    #[test]
    fn opus_seek_from_start_counts_audio_not_rounded_container_timestamps() {
        let (mut source, log, _) = fake_with_codec(
            vec![packet(7, 0, 0), packet(7, 0, 1), packet(7, 11, 2)],
            CODEC_TYPE_OPUS,
        );
        source.try_seek(Duration::from_millis(15)).unwrap();
        assert_eq!(*log.lock().unwrap(), vec![254, 0, 1, 2]);
        assert_eq!(source.next(), Some(2_240));
    }

    #[test]
    fn consecutive_bad_audio_packets_have_a_bounded_retry_budget() {
        let (mut source, log, _) = fake(vec![packet(7, 0, 255); MAX_DECODE_RETRIES + 2]);
        assert_eq!(source.next(), None);
        assert_eq!(log.lock().unwrap().len(), MAX_DECODE_RETRIES + 1);
    }

    #[test]
    fn time_conversion_and_end_borrow_are_not_sample_rate_assumptions() {
        for ms in 0..=1_137 {
            assert_eq!(duration_to_frames(Duration::from_millis(ms), 48_000), ms as usize * 48);
        }
        assert_eq!(duration_to_frames(Duration::from_nanos(1_000_500), 48_000), 48);
        assert_eq!(ticks_to_frames(5, TimeBase::new(1, 1_000), 48_000), 240);
        assert_eq!(ticks_to_frames(1, TimeBase::new(1, 48_000), 48_000), 1);
        assert_eq!(ticks_to_frames(1, TimeBase::new(1, 90_000), 48_000), 1);
        assert_eq!(skip_back_a_tiny_bit(Time::new(0, 0.0)), Time::new(0, 0.0));
        let before_one = skip_back_a_tiny_bit(Time::new(1, 0.0));
        assert_eq!(before_one.seconds, 0);
        assert!((before_one.frac - 0.9999).abs() < 1e-12);
    }
}
