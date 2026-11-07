use symphonia::core::audio::SampleBuffer;
use symphonia::core::codecs::DecoderOptions;
use symphonia::core::formats::FormatOptions;
use symphonia::core::io::MediaSourceStream;
use symphonia::core::meta::MetadataOptions;
use symphonia::core::probe::Hint;
#[cfg(test)]
pub use rodio::{source, Source};
#[cfg(test)]
mod rodio_decoder_tests;

fn main() {
    let path = std::env::args_os().nth(1).expect("input path");
    let file = std::fs::File::open(&path).unwrap();
    let mut format = symphonia::default::get_probe().format(
        &Hint::new(), MediaSourceStream::new(Box::new(file), Default::default()),
        &FormatOptions { enable_gapless: true, ..Default::default() },
        &MetadataOptions::default(),
    ).unwrap().format;
    let track = format.tracks().iter().find(|t|
        symphonia::default::get_codecs().get_codec(t.codec_params.codec).is_some()
    ).expect("audio track");
    let id = track.id;
    let mut decoder = symphonia::default::get_codecs().make(&track.codec_params, &DecoderOptions::default()).unwrap();
    let mut samples = Vec::<f32>::new();
    let mut packets = 0;
    loop {
        let packet = match format.next_packet() {
            Ok(p) => p,
            Err(symphonia::core::errors::Error::IoError(e)) if e.kind() == std::io::ErrorKind::UnexpectedEof => break,
            Err(e) => panic!("packet {e}"),
        };
        if packet.track_id() != id { continue; }
        let decoded = decoder.decode(&packet).unwrap();
        let mut buffer = SampleBuffer::<f32>::new(decoded.capacity() as u64, *decoded.spec());
        buffer.copy_interleaved_ref(decoded);
        samples.extend_from_slice(buffer.samples());
        packets += 1;
    }
    let mut oracle = path.clone();
    oracle.push(".f32");
    let mut worst = 0.0f32;
    if let Ok(raw) = std::fs::read(oracle) {
        assert_eq!(raw.len(), samples.len() * 4, "decoded length disagrees with FFmpeg oracle");
        for (chunk, &sample) in raw.chunks_exact(4).zip(&samples) {
            let expected = f32::from_le_bytes(chunk.try_into().unwrap());
            assert!(sample.is_finite());
            worst = worst.max((sample - expected).abs());
        }
        assert!(worst < 2.0e-5, "Opus reference mismatch: {worst}");
    }
    println!("path={} packets={packets} samples={} worst_abs={worst}", std::path::Path::new(&path).display(), samples.len());
}
