//! Compile the actual patched module's private tests without importing all of
//! upstream rodio's unrelated example/benchmark development dependencies.
pub use rodio::decoder::DecoderError;
use symphonia::core::io::MediaSource;
#[path = "../../../../vendor/rodio-0.20.1/src/decoder/symphonia.rs"]
mod actual_decoder;

// This included module has a local SeekError type, unlike the linked rodio
// crate's identically sourced module. Preserve it as a boxed source error for
// this harness only; production uses rodio's original concrete conversion.
impl From<actual_decoder::SeekError> for rodio::source::SeekError {
    fn from(error: actual_decoder::SeekError) -> Self {
        Self::Other(Box::new(error))
    }
}

#[test]
fn the_actual_constructor_can_return_its_owned_source() {
    let bytes = include_bytes!("../../../../src/testdata/audio/tone.opus").to_vec();
    let length = bytes.len() as u64;
    let stream = symphonia::core::io::MediaSourceStream::new(
        Box::new(std::io::Cursor::new(bytes)), Default::default());
    let mut decoder = actual_decoder::SymphoniaDecoder::new(stream, Some("opus")).unwrap();
    assert!(decoder.next().is_some());
    let returned = decoder.into_inner();
    assert_eq!(returned.byte_len(), Some(length));
}
