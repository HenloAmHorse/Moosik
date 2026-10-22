# Moosik Opus timing / audio-only import patch

Base: published `symphonia-format-mkv 0.5.5`, crates.io archive SHA-256
`122d786d2c43a49beb6f397551b4a050d8229eaa54c7ddf9ee4b98899b8742d0`.
Original source headers and MPL-2.0 licensing are retained.

The Opus path decodes at 48 kHz, validates CodecDelay against OpusHead, consumes
startup trim once and converts signed DiscardPadding from nanoseconds to sample
frames. Padding belongs to a Block and is distributed across its laced packets,
not carried into the next Block. Negative startup timestamps no longer underflow
an unsigned subtraction. The default track excludes unrecognized/video codecs.

Until an indexed SeekPreRoll implementation is validated, Opus seeks reset to
the known Segment beginning. Moosik/rodio decode and discard the preceding audio
with bounded packet memory. This also supports rewind without Cues, but seek cost
can grow with the requested position. It is not a fast-seek or video claim.

Modified files: Cargo.toml (local Opus packet-duration helper), src/segment.rs,
src/lacing.rs, src/demuxer.rs. Other audio codecs keep their existing decode path.
The consumer must interpret timestamps using CodecParameters.time_base, never
assume Matroska ticks are PCM frames. See docs/AUDIO_FORMATS.md for limitations.
