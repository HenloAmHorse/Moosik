# Moosik Opus decoder

This is original Moosik integration code, licensed with Moosik under AGPL-3.0-only
(see the repository's `LICENSE`). It is not copied from
`symphonia-adapter-libopus` or a different decoder implementation.

`opusic-sys = 0.7.5` bundles the reference libopus 1.6.1 by default and links it
statically. The build needs CMake and a platform C compiler; users of the built
player need neither CMake nor a separate Opus DLL. The dependency and its bundled
libopus retain their upstream BSD-3-Clause license and notices. They must be
included by the application's release notice generator. No experimental DRED,
OSCE, or hardening-disabling feature is requested here.

## Decoder / demuxer contract

* Symphonia 0.5 `Decoder`, registered for `CODEC_TYPE_OPUS`.
* Output is 48 kHz f32, regardless of the advisory original input rate in
  `OpusHead`. Output gain from that header is applied by libopus.
* This first integration explicitly accepts mapping family 0, mono or stereo.
  Multistream/surround and other mapping families fail with an unsupported-format
  error; they are never silently downmixed or interpreted as stereo.
* The **demuxer** owns pre-skip, seek pre-roll, final padding, timestamps, duration,
  and any edits/cropping. It passes frame trimming in `Packet::trim_start/end` at
  48 kHz. This decoder applies those trims once and does **not** additionally
  subtract `OpusHead.pre_skip` or `CodecParameters.delay`.
* Reset discards codec history and reapplies output gain. It does not reapply
  stream-start pre-skip; a seek must deliver sufficient preceding packets and
  supply the desired trimming through the demuxer.
* A packet may decode at most 5,760 frames/channel (120 ms). Empty packets are
  errors, not an instruction to fabricate packet-loss-concealment audio.
* Packets over 61,440 bytes are rejected following RFC 7845 section 6's defensive
  limit. Legal packets with excessive padding may therefore be rejected.
* Malformed trims are rejected, not saturated or applied to a later packet.
  Errors and resets clear the published buffer, as required by Symphonia.

The float decoder does not add an i16 intermediate. The existing rodio 0.20.1
consumer may subsequently convert to i16; that pre-existing consumer policy is
outside this small codec adapter and is not represented as bit-perfect decoding.

Primary specifications: [RFC 7845](https://www.rfc-editor.org/rfc/rfc7845.html),
[libopus decoder API](https://www.opus-codec.org/docs/opus_api-1.5/group__opus__decoder.html),
[opusic-sys](https://github.com/DoumanAsh/opusic-sys).
