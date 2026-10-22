# Moosik: shared opt-in native Opus registry

Base: the published `symphonia 0.5.5` crates.io archive (MPL-2.0).
All original license and source files are retained.

Changes: the `moosik-opus` feature and optional local leaf decoder dependency;
one registration in `default::register_enabled_codecs`. With the feature off,
the original registry is unchanged. This single registry is used by both
Moosik's native precision-preserving decoder and rodio's shared/analysis paths.
No upstream codec is replaced. No runtime FFmpeg executable or DLL is required.

The separately licensed `moosik-codec-opus` wrapper and bundled libopus have
their own provenance/license notices. Container trimming is verified separately;
registering a codec alone is not evidence that every Opus container or channel
mapping is supported.
