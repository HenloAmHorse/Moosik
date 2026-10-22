# Audio formats (1.5.5)

The picker and folder importer share one case-insensitive inventory. An extension
is an invitation to probe the file, not proof that every codec/profile in that
container is supported. This is audio playback; importing MP4/MKV/WebM does not
add a video renderer or DRM support.

| File/container | Audio decoder scope |
|---|---|
| FLAC, MP3, WAV/WAVE, Ogg/OGA | Existing decoders; actual content is probed |
| AAC (ADTS), M4A/M4B/MP4 | AAC-LC; ALAC enabled in its supported containers |
| AIFF/AIF/AIFF-C, CAF | Supported PCM/ALAC variants; not every compressed subtype |
| Opus, Ogg Opus, WebM/MKA/MKV | Bundled reference libopus, mono/stereo mapping family 0; supported other audio codecs remain available |
| DSF, DFF | Existing DSD paths, unchanged |

Unsupported profiles/mappings fail explicitly rather than being silently
downmixed or renamed. HE-AAC, WMA, APE and WavPack are not newly supported by this
round. Neither a remux nor decoding a lossy source into PCM makes that source
lossless. The information panel now uses the probed codec, not the extension,
for its lossless status; unknown is reported as unknown.

## Opus integration

Shared playback, analysis and native output use the same registered decoder.
The player does not invoke FFmpeg and does not require a separately installed
Opus DLL. Native Opus output is decoded 48 kHz float; the advisory original rate
in OpusHead is not its playback clock. The existing shared rodio/analysis path
still converts to i16; this round does not claim to remove that existing policy
or make a compressed source bit-identical to an original recording.

Demuxers own startup pre-skip and end padding. The decoder applies packet trims
once and honours header output gain. Fully trimmed packets are not EOF. A
video-first container selects a registered audio track and filters other tracks'
packets throughout playback.

**Seek tradeoff:** until indexed Opus pre-roll is independently validated, seeks
decode/discard from the known beginning with bounded packet memory. This keeps
decoder history and output consistent but may be slower for a long seek. It is
not an instantaneous random-access claim. Native preparation checks actual frame
counts; Matroska timestamps are converted through their real time base.

## Tests and limits

Checked-in fixtures are original synthetic stereo tones, not downloaded music.
Device-free tests cover shared decode, native preparation and seeks, positive
initial Ogg granules, pre-skip spanning packets, short EOS-only audio, multi-track
containers and production analysis/cache publication. A private FFmpeg float
decode is an independent Opus oracle; FFmpeg is only a development fixture tool.

This does not prove every encoder/container variant or an audio device. AAC-LC
is playable, but precise MP4 edit-list/encoder-delay trimming remains an upstream
limitation in the existing route; its decoded tail is not asserted sample-exact
against the source fixture. Nonstandard channel mappings and chained streams
need separate evidence. Tag editing remains restricted to types Lofty can write;
read-only container metadata does not authorize rewriting that container.

Building bundled libopus needs a C compiler and CMake. On the measured Windows
machine use the installed VS 2022 CMake and `CMAKE_GENERATOR=Visual Studio 17 2022`;
the older bundled CMake does not recognize the auto-detected VS 2026 generator.
This is a build-time requirement, not an end-user installation step.
