# Synthetic audio decoder fixtures

Generated locally with FFmpeg from
`aevalsrc=0.2*sin(2*PI*440*t)|0.15*sin(2*PI*997*t):s=48000:d=1.137`.
No commercial recording or downloaded media is included. These fixtures are
original test data under the project's license. Metadata title/artist explicitly
identify them as synthetic. All files are stereo, 48 kHz, 54,576 source frames.

Encoders: libopus (Ogg/WebM), ALAC (M4A), AAC-LC (M4A), signed 24-bit big-endian
PCM (AIFF), signed 24-bit little-endian PCM (CAF). AAC encoder delay and padding
are a separate codec/container concern; no lossy codec is described as lossless.

The private fixture generator and independent FFmpeg f32 oracle comparison are
recorded in the format research report. FFmpeg is not needed by the player or
these checked-in fixture tests.

`tiny.opus`: 0.003 seconds, 2.5 ms packets, first audio page is also EOS.
`short-packets.opus`: 0.137 seconds, 2.5 ms packets, 2500 us Ogg pages;
the first packet is fully discarded by the encoder's 120-frame pre-skip.
`video-first.webm`: a synthetic 16x16 black VP8 track precedes a stream-copy
of `tone.opus`; this tests selecting audio, not video rendering support.
