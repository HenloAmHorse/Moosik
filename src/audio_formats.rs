//! Common local-audio import inventory and extension-based container labels.
//!
//! A listed extension is an invitation to probe a file, not proof of its
//! contents or codec. In particular MP4, Matroska and WebM are audio-container
//! imports: this does not add video playback, DRM support, or every codec those
//! containers can carry. Actual audio properties must come from the decoder.
//!
//! Keep this aligned with the shared Symphonia feature set. ALAC is a codec
//! inside M4A/MP4 (or CAF), not an invented standalone `.alac` file format.

use std::path::Path;

struct Format {
    extensions: &'static [&'static str],
    label: &'static str,
    importable: bool,
}

const FORMATS: &[Format] = &[
    Format {
        extensions: &["flac"],
        label: "FLAC",
        importable: true,
    },
    Format {
        extensions: &["wav", "wave"],
        label: "WAV / audio container",
        importable: true,
    },
    Format {
        extensions: &["mp3"],
        label: "MP3",
        importable: true,
    },
    Format {
        extensions: &["ogg", "oga"],
        label: "Ogg / audio container",
        importable: true,
    },
    Format {
        extensions: &["dsf"],
        label: "DSD (DSF)",
        importable: true,
    },
    Format {
        extensions: &["dff"],
        label: "DSD (DSDIFF)",
        importable: true,
    },
    Format {
        extensions: &["aac"],
        label: "AAC / ADTS",
        importable: true,
    },
    Format {
        extensions: &["m4a", "m4b"],
        label: "M4A / audio container",
        importable: true,
    },
    Format {
        extensions: &["mp4"],
        label: "MP4 / audio track only",
        importable: true,
    },
    // Enabled by the application's explicit Symphonia aiff/caf/alac features.
    Format {
        extensions: &["aif", "aiff"],
        label: "AIFF / audio container",
        importable: true,
    },
    Format {
        extensions: &["aifc"],
        label: "AIFF-C / audio container",
        importable: true,
    },
    Format {
        extensions: &["caf"],
        label: "CAF / audio container",
        importable: true,
    },
    // Symphonia's Matroska reader also probes MKA by its EBML marker.
    Format {
        extensions: &["mka", "mkv"],
        label: "Matroska / audio track only",
        importable: true,
    },
    Format {
        extensions: &["webm"],
        label: "WebM / audio track only",
        importable: true,
    },
    // The same bundled decoder is registered for shared playback, analysis
    // and native output. Current mapping support is explicitly mono/stereo.
    Format {
        extensions: &["opus"],
        label: "Ogg / Opus",
        importable: true,
    },
];

fn format_for(path: &Path) -> Option<&'static Format> {
    let extension = path.extension()?.to_str()?;
    FORMATS.iter().find(|format| {
        format
            .extensions
            .iter()
            .any(|known| extension.eq_ignore_ascii_case(known))
    })
}

/// Extensions for the file picker. Folder import uses the same inventory.
/// This small vector is created only when opening the picker, not per track.
pub fn import_extensions() -> Vec<&'static str> {
    FORMATS
        .iter()
        .filter(|format| format.importable)
        .flat_map(|format| format.extensions.iter().copied())
        .collect()
}

/// Whether the local-audio importer should offer to probe this path.
/// Case-insensitive and allocation-free; no file or directory is opened.
pub fn is_importable(path: &Path) -> bool {
    format_for(path).is_some_and(|format| format.importable)
}

/// Extension-based file/container label, never a measured codec verdict.
pub fn container_label(path: &Path) -> &'static str {
    format_for(path).map_or("Unknown", |format| format.label)
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::HashSet;

    #[test]
    fn picker_folder_and_labels_share_the_same_import_inventory() {
        let picker = import_extensions();
        let mut unique = HashSet::new();
        for format in FORMATS {
            for extension in format.extensions {
                assert!(unique.insert(*extension), "duplicate extension {extension}");
                let path = format!("track.{extension}");
                let upper = format!("TRACK.{}", extension.to_ascii_uppercase());
                assert_eq!(is_importable(Path::new(&path)), format.importable);
                assert_eq!(is_importable(Path::new(&upper)), format.importable);
                assert_eq!(picker.contains(extension), format.importable);
                assert_eq!(container_label(Path::new(&path)), format.label);
                assert_eq!(container_label(Path::new(&upper)), format.label);
            }
        }
        assert_eq!(
            picker.len(),
            picker.iter().copied().collect::<HashSet<_>>().len()
        );
    }

    #[test]
    fn common_audio_extensions_are_offered_but_video_codec_support_is_not_claimed() {
        for extension in [
            "flac", "wav", "wave", "mp3", "ogg", "oga", "dsf", "dff", "aac", "m4a", "m4b", "mp4",
            "aif", "aiff", "aifc", "caf", "mka", "mkv", "webm", "opus",
        ] {
            assert!(
                is_importable(Path::new(&format!("song.{extension}"))),
                "{extension}"
            );
        }
        for extension in ["mp4", "mka", "mkv", "webm"] {
            assert!(
                container_label(Path::new(&format!("song.{extension}")))
                    .contains("audio track only")
            );
        }
        for extension in ["m4a", "m4b", "mp4", "caf"] {
            assert!(
                !container_label(Path::new(&format!("song.{extension}"))).contains("ALAC"),
                "a container name guessed its codec"
            );
        }
    }

    #[test]
    fn unsupported_or_missing_extensions_are_not_silently_offered() {
        for path in ["song", "song.txt", "song.exe", "song.wma", "song.alac"] {
            assert!(!is_importable(Path::new(path)), "{path}");
        }
        assert_eq!(container_label(Path::new("song.alac")), "Unknown");
        assert_eq!(container_label(Path::new("song.OPUS")), "Ogg / Opus");
    }
}
