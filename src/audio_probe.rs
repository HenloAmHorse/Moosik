//! Header-only audio identity, shared by import metadata and native preparation.
//! Container names never decide whether a stream is lossless.
use std::path::Path;
use std::time::Duration;
use symphonia::core::codecs::*;
use symphonia::core::formats::{FormatOptions, FormatReader, Track};
use symphonia::core::io::MediaSourceStream;
use symphonia::core::meta::{MetadataOptions, StandardTagKey};
use symphonia::core::probe::Hint;

pub fn audio_track(format: &dyn FormatReader) -> Option<&Track> {
    let registry = symphonia::default::get_codecs();
    format
        .default_track()
        .filter(|t| registry.get_codec(t.codec_params.codec).is_some())
        .or_else(|| {
            format
                .tracks()
                .iter()
                .find(|t| registry.get_codec(t.codec_params.codec).is_some())
        })
}

pub struct Info {
    pub codec: &'static str,
    pub lossless: Option<bool>,
    pub duration: Option<Duration>,
    pub sample_rate: Option<u32>,
    pub channels: Option<u8>,
    pub bit_depth: Option<u8>,
    pub title: Option<String>,
    pub artist: Option<String>,
    pub album: Option<String>,
}

pub fn probe(path: &Path) -> Option<Info> {
    let source = std::fs::File::open(path).ok()?;
    let mut hint = Hint::new();
    if let Some(ext) = path.extension().and_then(|s| s.to_str()) {
        hint.with_extension(ext);
    }
    let mut probed = symphonia::default::get_probe()
        .format(
            &hint,
            MediaSourceStream::new(Box::new(source), Default::default()),
            &FormatOptions {
                enable_gapless: true,
                ..Default::default()
            },
            &MetadataOptions::default(),
        )
        .ok()?;
    let params = &audio_track(probed.format.as_ref())?.codec_params;
    let codec = symphonia::default::get_codecs()
        .get_codec(params.codec)?
        .long_name;
    let duration = params
        .time_base
        .filter(|tb| tb.numer != 0 && tb.denom != 0)
        .zip(params.n_frames)
        .and_then(|(tb, n)| {
            let time = tb.calc_time(n);
            Duration::try_from_secs_f64(time.seconds as f64 + time.frac).ok()
        });
    let mut info = Info {
        codec,
        lossless: lossless(params.codec),
        duration,
        sample_rate: params.sample_rate,
        channels: params
            .channels
            .or_else(|| params.channel_layout.map(|layout| layout.into_channels()))
            .and_then(|c| u8::try_from(c.count()).ok()),
        bit_depth: params.bits_per_sample.and_then(|b| u8::try_from(b).ok()),
        title: None,
        artist: None,
        album: None,
    };
    let mut absorb = |meta: &symphonia::core::meta::MetadataRevision| {
        for tag in meta.tags() {
            let slot = match tag.std_key {
                Some(StandardTagKey::TrackTitle) => &mut info.title,
                Some(StandardTagKey::Artist) => &mut info.artist,
                Some(StandardTagKey::Album) => &mut info.album,
                _ => continue,
            };
            let text = tag.value.to_string();
            if slot.is_none() && !text.trim().is_empty() {
                *slot = Some(text);
            }
        }
    };
    if let Some(metadata) = probed.metadata.get()
        && let Some(meta) = metadata.current()
    {
        absorb(meta);
    }
    if let Some(meta) = probed.format.metadata().current() {
        absorb(meta);
    }
    Some(info)
}

pub fn lossless(codec: CodecType) -> Option<bool> {
    match codec {
        CODEC_TYPE_FLAC | CODEC_TYPE_ALAC => Some(true),
        CODEC_TYPE_PCM_S8 | CODEC_TYPE_PCM_U8 | CODEC_TYPE_PCM_S16LE | CODEC_TYPE_PCM_S16BE
        | CODEC_TYPE_PCM_S24LE | CODEC_TYPE_PCM_S24BE | CODEC_TYPE_PCM_S32LE
        | CODEC_TYPE_PCM_S32BE | CODEC_TYPE_PCM_U16LE | CODEC_TYPE_PCM_U16BE
        | CODEC_TYPE_PCM_U24LE | CODEC_TYPE_PCM_U24BE | CODEC_TYPE_PCM_U32LE
        | CODEC_TYPE_PCM_U32BE | CODEC_TYPE_PCM_F32LE | CODEC_TYPE_PCM_F32BE
        | CODEC_TYPE_PCM_F64LE | CODEC_TYPE_PCM_F64BE => Some(true),
        CODEC_TYPE_AAC
        | CODEC_TYPE_OPUS
        | CODEC_TYPE_VORBIS
        | CODEC_TYPE_MP3
        | CODEC_TYPE_ADPCM_IMA_WAV
        | CODEC_TYPE_ADPCM_MS
        | CODEC_TYPE_PCM_ALAW
        | CODEC_TYPE_PCM_MULAW => Some(false),
        _ => None,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn lossless_is_a_codec_property_not_a_container_extension() {
        assert_eq!(lossless(CODEC_TYPE_ALAC), Some(true));
        assert_eq!(lossless(CODEC_TYPE_AAC), Some(false));
        assert_eq!(lossless(CODEC_TYPE_ADPCM_IMA_WAV), Some(false));
        assert_eq!(lossless(CODEC_TYPE_OPUS), Some(false));
        assert_eq!(lossless(CODEC_TYPE_NULL), None);
    }
}
