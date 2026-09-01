# Locally patched `eframe`

This directory is **derived from, and is not byte-identical to, the official
`eframe 0.31.1` crates.io archive.** It is vendored into Moosik and wired in
through `[patch.crates-io]` in the workspace `Cargo.toml`.

## Provenance

| | |
|---|---|
| package | `eframe` |
| version | `0.31.1` |
| official archive SHA-256 | `d0dfe0859f3fb1bc6424c57d41e10e9093fe938f426b691e42272c2f336d915c` |
| upstream VCS commit | `1669e52a7ccfc3489c1b0999b9ed48894a0b3887` (`.cargo_vcs_info.json`, `path_in_vcs: crates/eframe`) |
| upstream repository | <https://github.com/emilk/egui/tree/1669e52a7ccfc3489c1b0999b9ed48894a0b3887/crates/eframe> |
| licence | MIT OR Apache-2.0 |

The tree was extracted from the checksum-verified official `.crate`. All 32
archive entries remain present. Twenty-eight are byte-for-byte identical; four
are intentionally modified, as listed under "Files changed relative to the
official archive" below. `.cargo_vcs_info.json` remains verbatim.
`Cargo.toml.orig` remains the published archive copy but is **not** the active
manifest — it declares workspace-relative dependencies and is not standalone.
Cargo's generated `.cargo-ok` is not vendored, and no `.cargo-checksum.json` is
added: a path-based `[patch.crates-io]` does not verify one.

`LICENSE-MIT` and `LICENSE-APACHE` are added from the upstream repository at
the same commit. The published archive's `include` list names them but the
archive itself does not carry them, so they are restored here rather than
authored.

## What is changed and why

Upstream sleeps the event loop for 10 ms after painting a viewport whose
window is minimized, so that a minimized window does not spin a core
(<https://github.com/emilk/egui/issues/325>).

That is correct for a lone window. It is wrong when the minimized viewport is
the parent of a visible **immediate** child viewport. An immediate viewport is
painted inside its parent's pass, and eframe redirects the child's repaint
requests to the parent (`src/native/glow_integration.rs`, in
`request_repaint`, where a viewport with no ui callback is resolved to its
parent). Pacing the parent therefore paces the child, and sleeping because the
parent is minimized throttles a window the user is still looking at.

In Moosik this is visible directly: minimizing the player window while the
spectrum window stays open dropped the spectrum from ~170 FPS to ~80 FPS,
which is `1 / (0.010 + frame_time)`. Verified against a 50 ms mutant of the
same sleep, which produced ~19 FPS — the same relation at a different dose.

### The local policy

`src/native/minimized_policy.rs` (new) decides whether the sleep may be
waived. Both backends call it through a small adapter over their own viewport
map. The sleep is skipped **only** when all of the following hold:

1. the viewport just painted is minimized (unchanged upstream condition);
2. the `moosik_windows_linked_viewport` feature is enabled;
3. `target_os = "windows"`;
4. that viewport has a descendant reachable through an unbroken chain of
   *immediate* viewports — a deferred viewport anywhere along the way
   disqualifies it, because the descendant is then paced by that deferred
   viewport's own pass rather than by this one's;
5. the qualifying descendant's native window state is exactly
   `is_visible() == Some(true) && is_minimized() == Some(false)`.

Any unknown (`None`) native state keeps the upstream sleep. If nothing
qualifies, the upstream sleep runs exactly as before.

### Scope

- **Windows only.** The macOS mitigation that the sleep exists for is
  untouched: off Windows `may_waive_minimized_sleep` is a constant `false`.
- **Opt-in.** With the feature disabled, the policy module is not compiled and
  both call sites reduce to `false`, so the crate is upstream-equivalent.
- Both the glow and the wgpu backend are patched symmetrically, even though
  Moosik builds glow.

### Immediate viewports cannot be identified by `ViewportClass`

`ViewportClass::Immediate` is unusable for this decision in this version, so
the policy reads the ui callback instead.

- `Context::show_viewport_immediate` sets `viewport_ui_cb = None` but never
  assigns `ViewportState::class` (`egui-0.31.1/src/context.rs`, in
  `show_viewport_immediate`). The only assignment to that field in
  `context.rs` is `class = ViewportClass::Deferred`.
- `ViewportClass` defaults to `Root`, and that default is copied verbatim into
  `ViewportOutput.class`.
- eframe does set `ViewportClass::Immediate` when it renders an immediate
  viewport, but `initialize_or_update_viewport` writes the output's class back
  over it, and `handle_viewport_output` runs immediately before the sleep
  site.

So the recorded class of a live immediate viewport is `Root` by the time the
policy runs. `viewport_ui_cb.is_none()` is the signal eframe itself uses to
tell the two apart. Because the root viewport also has no callback, the policy
excludes `ViewportId::ROOT` and the viewport being painted explicitly. The
class is not consulted at all, so a stale class cannot cause a false negative.

## Files changed relative to the official archive

| file | change |
|---|---|
| `Cargo.toml` | adds the `moosik_windows_linked_viewport` feature |
| `src/native/mod.rs` | declares `minimized_policy` |
| `src/native/minimized_policy.rs` | **new** — the shared policy and its tests |
| `src/native/glow_integration.rs` | adapter, one call at the sleep site, adapter tests |
| `src/native/wgpu_integration.rs` | the same, symmetrically |
| `LICENSE-MIT`, `LICENSE-APACHE` | **added** from upstream at the pinned commit |

No other file is modified. In each backend exactly one line is replaced:
`if window.is_minimized() == Some(true) {` gains a `!waive_minimized_sleep &&`
guard.

## Tests

The policy is covered by unit tests inside `minimized_policy.rs`, and each
backend adapter by tests in its own module. They call the production decision
function; only the viewport graph is fabricated.

    cargo test --no-default-features --features glow,moosik_windows_linked_viewport
    cargo test --no-default-features --features wgpu,moosik_windows_linked_viewport

Dependency tests are not run by Moosik's own `cargo test`, so these must be
run from this directory.

## Upstream status

No upstream patch has been submitted. The behaviour above is a deliberate
local deviation, not a fix upstream has accepted. The relevant upstream
context is <https://github.com/emilk/egui/issues/325>, which motivated the
sleep, and egui's own documentation of immediate viewports as sharing their
parent's pass.
