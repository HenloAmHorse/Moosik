# Locally patched `egui`

This directory is **derived from the official `egui 0.31.1` crates.io
archive** and wired in through `[patch.crates-io]` in the workspace
`Cargo.toml`.

## Provenance

| | |
|---|---|
| package | `egui` |
| version | `0.31.1` |
| official archive SHA-256 | `25dd34cec49ab55d85ebf70139cb1ccd29c977ef6b6ba4fe85489d6877ee9ef3` (equals the `Cargo.lock` checksum it replaces) |
| upstream VCS commit | `1669e52a7ccfc3489c1b0999b9ed48894a0b3887` (`.cargo_vcs_info.json`, `path_in_vcs: crates/egui`) — the same commit as the vendored `eframe` |
| upstream repository | <https://github.com/emilk/egui/tree/1669e52a7ccfc3489c1b0999b9ed48894a0b3887/crates/egui> |
| licence | MIT OR Apache-2.0 |

Extracted from the checksum-verified official `.crate`; every archive entry is
present. `LICENSE-MIT` and `LICENSE-APACHE` are added (copied from the vendored
`eframe`, which took them from the upstream repository at the same commit;
the published archive names them in `include` but does not carry them).
`Cargo.toml.orig` is the archive copy and not the active manifest.

## What is changed and why

egui keeps one `Fonts` per pixels-per-point, shared by every viewport at that
scale, and with it one galley (text-layout) cache. `GalleyCache::flush_cache`
keeps only what was used since the previous flush, and is documented as
"must be called once per frame". `Context::update_fonts_mut` calls
`Fonts::begin_pass` — and so flushes — at the start of **every** viewport's
pass. An immediate child viewport's pass runs inside its parent's pass, so a
frame with one flushes twice: once at the root's start and again at the
child's. Moosik draws its panels after its immediate children (Spectrum,
Lyrics, Tags, Info), so at the child's flush the frame had re-used nothing,
and every galley — the root's and the child's — was evicted and laid out
again on every frame while any child was open.

Measured in Moosik's headless lyrics harness (real egui passes, app fonts,
no GL): with the Lyrics window open the root's stand-in UI re-laid all 140 of
its galleys every pass (0 with Lyrics closed), and Lyrics re-laid all of its
own (≈ 45; ≈ 233 with furigana).

The change: a pass nested inside an outer pass that uses the **same**
`Fonts` does not call `Fonts::begin_pass`; the outer pass already did this
frame. "The same" means all three: an ancestor pass at the same
pixels-per-point, that ancestor's requested texture cap equal to this
pass's, **and** the shared `Fonts`' actual cap (`Fonts::max_texture_side`)
equal to this pass's request. The last is needed because another nested pass
may have recreated the shared `Fonts` at its own cap since the ancestor began
them; without it a grandchild or later sibling asking the ancestor's cap was
skipped onto the other cap's fonts.

## The contract established (and what is not claimed)

`Fonts::begin_pass` does more than flush: it recreates the fonts and atlas
when the scale or texture cap changed or the atlas is over 80 % full. Skipping
it in a nested pass skips those too. What the tests establish, with real root
and immediate-child passes:

- **Two cases, kept apart.** *The configured path* — eframe's glow renderer,
  one cap for every viewport — is the one Moosik runs and the one the
  optimisation is for. *The generic fallback* — any pass whose cap differs
  from the actual shared `Fonts` or from its ancestor's — begins its own
  fonts exactly as upstream does; nothing beyond that lifecycle is claimed
  for mixed caps (no rendering or pixel equality: these are lifecycle tests).
- **Renderer.** The patch applies only where a nested child has its parent's
  cap. Moosik's renderer is eframe's glow integration, which reads one cap
  from its one painter and gives it to every viewport, immediate children
  included (`glow_integration.rs`, checked against the vendored source by a
  test); the wgpu renderer is not enabled in this build and nothing is
  claimed for it. A nested child at the same scale but **another** cap is
  left to upstream: it begins (and so recreates) its fonts at its own pass.
- **Same scale and cap.** The child's pass uses exactly the `Fonts` and atlas
  the root began that frame.
- **Cap change.** Taken at the next outermost pass start: the root recreates,
  the child in that frame uses the new fonts, text is laid out once and then
  reused.
- **Atlas pressure.** With the atlas filled past the threshold by the root's
  galleys **before** the child runs, the atlas stays the one those galleys
  point into for the rest of the frame (unmodified egui's child pass replaced
  it there, beneath them) and is recreated at the next frame's start, where
  everything is laid out again. The cost: within one frame every viewport at
  that scale shares the headroom left above the threshold, as a single
  viewport's frame does upstream; a frame that by itself adds more glyphs
  than that overflows the atlas (egui's own overflow path) exactly as an
  unpatched single-viewport frame would.
- **New font definitions between frames.** Taken once at the root's start
  (`self.fonts.clear()`, unchanged); the child uses the root's new fonts.
- **New font definitions inside the root's pass, before a child.** Taken at
  the child's pass start, replacing fonts the root already laid text against
  — upstream's behaviour, in code ahead of the patch and unchanged by it,
  identical with and without the patch. An inherited limitation, not a new
  one. Moosik does not do this: its font picker runs after every immediate
  child.
- **Mixed caps across several nested passes.** Root 4096, a child 1024 (it
  recreates the shared fonts), then a grandchild and a later sibling asking
  4096: each begins on fonts of the cap it asked for. Before the actual-cap
  condition both began on the child's 1024 (`a_nested_pass_checks_the_shared_fonts_actual_cap`).
- **Scale.** A nested child at another scale has its own `Fonts` and still
  begins them at its own pass.
- **Retirement.** Text that stops being drawn is still let go: it survives at
  most one further root flush; the cache stays bounded with a child drawing
  new text every frame, at the same or another scale, and with the root
  alone.

Tests: `src/egui_layout_cache_tests.rs`. Run against the unmodified archive's
`context.rs`, three fail — unchanged text re-laid every frame with a child
open; the full atlas recreated by the child beneath the root's galleys; text
re-laid after a cap change — and the rest pass, the mid-pass font replacement
included (same behaviour). Mutants — the patch reverted, the cap condition
dropped, the actual-cap condition dropped (ancestor-only), nested passes
skipping regardless of scale, no pass flushing — are each caught.

## Files changed relative to the official archive

| file | change |
|---|---|
| `src/context.rs` | `update_fonts_mut`: `nested_in_same_fonts` (an outer pass at the same scale and texture cap), and `fonts.begin_pass` skipped only when it holds **and** the shared `Fonts`' actual cap equals the request (marked `MOOSIK PATCH`) |
| `PATCH.md`, `LICENSE-MIT`, `LICENSE-APACHE` | added |
