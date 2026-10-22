use super::*;

fn decode(path: &Path, start: Duration) -> (Vec<u32>, Duration) {
    let mut prepared = prepare(path, start).unwrap();
    let landed = prepared.seeked_to;
    let mut output = std::mem::take(&mut prepared.primed);
    loop {
        let packet = match prepared.format.next_packet() {
            Ok(packet) => packet,
            Err(SymError::IoError(e)) if e.kind() == std::io::ErrorKind::UnexpectedEof => break,
            Err(error) => panic!("container: {error}"),
        };
        if packet.track_id() != prepared.track_id {
            continue;
        }
        let decoded = prepared.decoder.decode(&packet).unwrap();
        append_canonical(&decoded, &mut output);
    }
    (output, landed)
}

#[test]
fn opus_native_seek_matches_decoding_the_prefix_without_a_device() {
    for name in ["tone.opus", "tone.webm", "video-first.webm"] {
        let path = Path::new(env!("CARGO_MANIFEST_DIR"))
            .join("src/testdata/audio")
            .join(name);
        let (reference, _) = decode(&path, Duration::ZERO);
        assert_eq!(reference.len(), 109_152, "{name}");
        for milliseconds in [237, 0, 9, 18, 71, 282, 20, 510, 1000] {
            let position = Duration::from_millis(milliseconds);
            let (found, landed) = decode(&path, position);
            assert_eq!(landed, position, "{name}");
            assert_eq!(
                found,
                reference[milliseconds as usize * 48 * 2..],
                "{name}: {milliseconds}ms"
            );
        }
        let (past_end, landed) = decode(&path, Duration::from_secs(2));
        assert!(past_end.is_empty());
        assert_eq!(landed, Duration::from_millis(1137));
    }
}
