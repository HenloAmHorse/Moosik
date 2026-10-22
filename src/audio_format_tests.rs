//! Real encoded containers, no device, subprocess, owner cache or network.
use super::*;
use rodio::Source;
use symphonia::core::io::Monitor;

fn fixture(name: &str) -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("src/testdata/audio")
        .join(name)
}

#[test]
fn common_audio_containers_reach_the_shared_decoder_and_metadata() {
    for (name, lossless) in [
        ("tone-alac.m4a", true),
        ("tone-aac.m4a", false),
        ("tone.aiff", true),
        ("tone.caf", true),
        ("tone.opus", false),
        ("tone.webm", false),
    ] {
        let path = fixture(name);
        let decoded = Decoder::new(BufReader::new(File::open(&path).unwrap()))
            .unwrap_or_else(|e| panic!("{name}: {e}"));
        assert_eq!(decoded.sample_rate(), 48_000, "{name}");
        assert_eq!(decoded.channels(), 2, "{name}");
        let samples: Vec<i16> = decoded.collect();
        assert!(!samples.is_empty(), "{name}");
        assert!(samples.iter().any(|&v| v != 0), "{name}");
        assert_eq!(samples.len() % 2, 0, "{name}");
        if name != "tone-aac.m4a" {
            assert_eq!(samples.len(), 109_152, "{name}");
        }
        let info = audio_probe::probe(&path).expect("audio header");
        assert_eq!(info.lossless, Some(lossless), "{name}");
        assert_eq!(info.sample_rate, Some(48_000), "{name}");
        let track = Track::load(path);
        assert_eq!(track.lossless, Some(lossless), "{name}");
        assert_eq!(track.sample_rate, Some(48_000), "{name}");
        assert!(track.duration.is_some(), "duration missing: {name}");
    }
}

#[test]
fn opus_shared_seek_replays_the_same_decoded_samples() {
    for name in ["tone.opus", "tone.webm"] {
        let path = fixture(name);
        let open = || Decoder::new(BufReader::new(File::open(&path).unwrap())).unwrap();
        let reference: Vec<i16> = open().collect();
        let mut decoder = open();
        for milliseconds in [237, 0, 9, 18, 71, 282, 510, 20, 1000] {
            decoder
                .try_seek(Duration::from_millis(milliseconds))
                .unwrap_or_else(|e| panic!("{name}: {e}"));
            let expected_at = milliseconds as usize * 48 * 2;
            let found: Vec<i16> = decoder.by_ref().take(480).collect();
            assert_eq!(
                found,
                reference[expected_at..expected_at + 480],
                "{name} at {milliseconds}ms"
            );
        }
    }
}

#[test]
fn common_audio_containers_reach_native_preparation_without_a_device() {
    for name in [
        "tone-alac.m4a",
        "tone-aac.m4a",
        "tone.aiff",
        "tone.caf",
        "tone.opus",
        "tone.webm",
    ] {
        let p = bitperfect::prepare(&fixture(name), Duration::ZERO)
            .unwrap_or_else(|e| panic!("{name}: {e:?}"));
        assert_eq!(p.sample_rate, 48_000, "{name}");
        assert_eq!(p.source.channels, 2, "{name}");
    }
}

fn rewrite_ogg(mut bytes: Vec<u8>, pre_skip: Option<u16>, initial_offset: u64) -> Vec<u8> {
    let mut offset = 0;
    while offset < bytes.len() {
        assert_eq!(&bytes[offset..offset + 4], b"OggS");
        let segments = bytes[offset + 26] as usize;
        let header = 27 + segments;
        let body: usize = bytes[offset + 27..offset + header]
            .iter()
            .map(|&v| v as usize)
            .sum();
        let page = &mut bytes[offset..offset + header + body];
        if page[header..].starts_with(b"OpusHead") {
            if let Some(skip) = pre_skip {
                page[header + 10..header + 12].copy_from_slice(&skip.to_le_bytes());
            }
        } else if !page[header..].starts_with(b"OpusTags") {
            let gp = u64::from_le_bytes(page[6..14].try_into().unwrap());
            page[6..14].copy_from_slice(&(gp + initial_offset).to_le_bytes());
        }
        page[22..26].fill(0);
        let mut crc = symphonia::core::checksum::Crc32::new(0);
        crc.process_buf_bytes(page);
        page[22..26].copy_from_slice(&crc.crc().to_le_bytes());
        offset += header + body;
    }
    bytes
}

#[test]
fn opus_preskip_and_positive_initial_granules_do_not_lose_or_repeat_audio() {
    let raw = include_bytes!("testdata/audio/tone.opus").to_vec();
    let decode = |bytes: Vec<u8>| -> Vec<i16> {
        Decoder::new(std::io::Cursor::new(bytes)).unwrap().collect()
    };
    let reference = decode(raw.clone());
    // Same compressed data; only the container's required initial discard
    // increases. Decoder history must still see every compressed packet.
    let longer_skip = decode(rewrite_ogg(raw.clone(), Some(2000), 0));
    assert_eq!(longer_skip, reference[(2000 - 312) * 2..]);
    // Cropped streams may have a positive initial granule position. The
    // audible timeline remains zero-based and the decoded audio is unchanged.
    let cropped = decode(rewrite_ogg(raw, None, 96_000));
    assert_eq!(cropped, reference);
}

#[test]
fn tiny_opus_and_fully_trimmed_short_packets_are_not_false_eof() {
    for (name, frames) in [("tiny.opus", 144), ("short-packets.opus", 6576)] {
        let bytes = std::fs::read(fixture(name)).unwrap();
        let decoded: Vec<i16> = Decoder::new(std::io::Cursor::new(bytes.clone()))
            .unwrap()
            .collect();
        assert_eq!(decoded.len(), frames * 2, "{name}");
        assert!(decoded.iter().any(|&v| v != 0));
        if name == "short-packets.opus" {
            // The low-delay encoder's pre-skip is 120; make it span six
            // complete 2.5ms packets without changing compressed audio.
            let longer: Vec<i16> =
                Decoder::new(std::io::Cursor::new(rewrite_ogg(bytes, Some(720), 0)))
                    .unwrap()
                    .collect();
            assert_eq!(longer, decoded[(720 - 120) * 2..]);
        }
    }
}

#[test]
fn a_video_first_container_selects_and_keeps_only_its_audio_track() {
    let decode = |name| -> Vec<i16> {
        Decoder::new(BufReader::new(File::open(fixture(name)).unwrap()))
            .unwrap()
            .collect()
    };
    assert_eq!(decode("video-first.webm"), decode("tone.opus"));
    let prepared = bitperfect::prepare(&fixture("video-first.webm"), Duration::ZERO).unwrap();
    assert_eq!(prepared.source.channels, 2);
    assert_eq!(prepared.sample_rate, 48_000);
}
