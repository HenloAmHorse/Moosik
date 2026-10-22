# Moosik opt-in Ogg Opus timeline

Base: published `symphonia-format-ogg 0.5.5`, archive SHA-256
`2b4955c67c1ed3aa8ae8428d04ca8397fbef6a19b2b051e73b5da8b1435639cb`.
Original MPL-2.0 source headers are retained.

Feature `moosik-opus` gives Opus its own timeline: header pre-skip, positive
initial granules, first-and-last audio pages, end padding and zero-based audible
timestamps. Packet trim fields are the decoder's only skip authority. These
mandatory Opus trims do not depend on the generic optional gapless flag. Other
codec mappings keep the existing timeline.

An isolated EOS page cannot reveal its padding; the end probe retries a bounded-
memory scan from the physical stream beginning when needed. It scans compressed
pages, not decoded PCM. Opus seeks use the known first audio page and a correct
decode/discard-from-start fallback, not an unverified random-access pre-roll.
Source consumers must decode zero-output packets, not treat them as EOF.

Modified files: Cargo.toml, src/logical.rs, src/physical.rs, src/page.rs and
src/demuxer.rs. Indexed pre-roll and general chained/multiplexed Opus coverage
remain separate work; this patch does not assert all such cases are verified.
