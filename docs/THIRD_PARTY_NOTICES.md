# Third-party notices: how they were assembled

The notices themselves are in [`THIRD_PARTY_NOTICES.txt`](../THIRD_PARTY_NOTICES.txt),
with four companion files in [`third_party/`](../third_party). This page says what
they cover and how the original open items were settled for 1.5.3, with
the scope rechecked for 1.5.4, regenerated for 1.5.5 and carried forward,
checked, for 1.6.0. It records license
information; it is not a legal opinion.

**1.5.5:** the current locked Windows normal/build graph was enumerated
again, not inferred from the previous count. Its third-party set is the 298
packages from 1.5.4 plus `cmake` 0.1.58, `opusic-sys` 0.7.5,
`symphonia-codec-alac` 0.5.5 and `symphonia-format-caf` 0.5.5: **302 packages and
205 deduplicated texts**. The original `moosik-codec-opus` 0.1.0 leaf is Moosik
AGPL-3.0-only code, separately accounted for, not a 303rd third-party package.

## Scope

- **What is covered:** the Windows x86_64 release executable with the default
  features. That is the 302 third-party packages `cargo tree --locked` reaches
  from `moosik` for `x86_64-pc-windows-msvc` (normal and build dependencies),
  the data they embed, the embedded fonts and the Rust standard library.
- **Build-time-only packages** (build scripts, proc-macros) are marked and
  included anyway, so no judgement about which of them reach the executable
  is needed.
- **Checked:** all 295 registry package archives against their current
  `Cargo.lock` checksums, plus the published upstream archives of seven vendored
  third-party packages against retained provenance hashes. A local path patch
  has no registry checksum in `Cargo.lock`; these are distinct checks.
- **1.5.4 recheck:** dependency names, versions and lockfile checksums are
  unchanged from 1.5.3 except Moosik's own version. The current Windows
  normal/build graph was re-enumerated; the notices were regenerated from
  retained hash-checked evidence with rustc 1.94.0. There are 298 packages
  and 203 deduplicated texts, including the local-model setup license texts.
- **1.5.5 recheck:** the emitted 302-package inventory equals the complete
  third-party graph, not merely its size. There are 282 normal-graph packages
  and 20 reached only through build edges. The 1.5.4 text sources and exception
  dispositions were reused with their content hashes; new license texts were
  read from checksum-verified package archives. The Rust toolchain remains
  rustc 1.94.0. No new upstream retrieval was needed.
- **Optional local translation:** the executable includes setup/consent UI
  and upstream model/runtime license texts. Those texts are included in the
  notices. Model weights, llama.cpp executables and its OpenMP runtime are
  not bundled in the release assets; pack-specific notices are shown before
  explicit download/import and installed with each pack.
- **Not covered:** dev-dependencies and non-Windows targets. This is not a
  claim that the optional separately downloaded runtime is linked into Moosik.

## Sources of the texts

- **Upstream:** the license and notice files each package ships. Where a
  package archive has none, the files at the source commit the archive records
  are used, accepted only when that commit's manifest declares exactly the
  package's name and version.
- **Standard texts:** where neither exists, a standard license text is supplied
  from the Rust toolchain's `share/doc/rust/licenses/`. It is labelled in the
  notices as supplied by Moosik, not as an upstream file.
- **Dual-licensed packages:** every text the package provides is reproduced.
  Where Moosik relies on one option, the entry names it.
- **Legacy expressions:** a `/` in a license field such as `MIT/Apache-2.0` is
  Cargo's deprecated spelling of `OR`.

## Items settled for 1.5.3

| Item | Disposition |
|---|---|
| `dasp_sample` 0.11.0 (MIT OR Apache-2.0) | **Distributed under MIT.** The full MIT text comes from the packaged source commit. Its `LICENSE-APACHE` is only a short pointer and is reproduced as published. |
| `gpu-alloc` 0.6.2, `gpu-alloc-types` 0.3.1, `gpu-descriptor` 0.3.2, `gpu-descriptor-types` 0.2.0 (MIT OR Apache-2.0) | **Distributed under MIT.** The upstream `COPYING` (copyright line and dual-license statement) is reproduced. The license files it names do not exist at the packaged commit, so the standard MIT and Apache-2.0 texts are supplied and labelled. |
| `hexf-parse` 0.2.1 (CC0-1.0) | **Public-domain dedication; no notice required.** Listed, with a link to the CC0 1.0 legal code. |
| `ferrous-opencc`, `ferrous-opencc-compiler` 0.4.0 (Apache-2.0) | **Standard Apache-2.0 text supplied.** The packages carry no license, NOTICE or copyright file. No copyright holder is asserted beyond what the packages state. |
| OpenCC dictionaries (compiled into the executable by `ferrous-opencc`) | **Apache-2.0 per the tables' own headers, which are reproduced.** The OpenCC project's LICENSE is reproduced from its repository at a pinned commit; that repository has no NOTICE file at its root. The OpenCC release the tables came from is not recorded in the package, and the notices say so. |
| IPADIC (embedded by `lindera-ipadic` 6.0.0) | **The `COPYING` of the archive actually embedded is reproduced:** `mecab-ipadic-2.7.0-20250920`, archive SHA-256 `a7ba9f64…62db3d`. Its original bytes are in `third_party/`. The crate's NOTICE names the older 20070801 release; the notices state which data is embedded. The file's last line holds two bytes (0xF7 0xF7) with no verified rendering; they are kept in the original-bytes copy and noted. |
| Embedded fonts | Only egui's default fonts (`epaint_default_fonts` 0.31.1: Hack, Ubuntu Light, Noto Emoji, emoji-icon-font). Their OFL-1.1, Ubuntu Font Licence, Hack and MIT texts are reproduced from the package. System fonts are read at runtime, not distributed. |
| Vendored, modified `eframe`, `egui`, `rustfft` (MIT OR Apache-2.0) | **Distributed under MIT**, with the published archives' license files. Each `vendor/*/PATCH.md` lists what was changed. |
| `symphonia` 0.5.5 family (MPL-2.0) | 1.5.5 vendors modified registry/Ogg/Matroska sources; their Source Code Form and patch descriptions are retained under `vendor/`. Other family crates remain unmodified. The newly enabled ALAC and CAF archives record the same packaged workspace commit as the retained upstream MPL text. The release source snapshot must include the modified sources; an upstream-only link is not a substitute for those changes. |
| `rodio` 0.20.1 (MIT OR Apache-2.0) | Vendored decoder fixes; distributed under MIT with original texts and patch provenance retained. |
| `opusic-sys` 0.7.5 / bundled libopus 1.6.1 | Native static Opus dependency; the verified archive's `LICENSE` and bundled `opus/COPYING` are byte-identical and both source labels are retained, together with `opus/AUTHORS`. The BSD notice also remains in `third_party/opusic-sys-0.7.5-LICENSE.txt`. No Opus DLL is redistributed. |
| `cmake` 0.1.58 (MIT OR Apache-2.0) | Build-time Cargo helper for bundled libopus; both license texts come from its verified package archive. The CMake executable is a build tool, not bundled in the player. |
| Rust standard library | The toolchain's `COPYRIGHT-library.html`, unmodified, in `third_party/`. |
| Microsoft Visual C++ runtime | The executable imports `VCRUNTIME140.dll` and the Universal CRT. **Neither is redistributed;** the release notes state that the Visual C++ 2015-2022 Redistributable (x64) is required. |

## Items for 1.6.0

`Cargo.lock` gains no package over 1.5.5 (the new `windows-sys` features only
add API declarations), so the 1.5.5 notices were carried forward rather than
regenerated: `research/v160/notices160.py` refuses to run unless the locked
default Windows x86_64 normal/build graph is exactly the 302 packages the
notices list, `Cargo.lock` differs from v1.5.5 only in Moosik's own version and
the toolchain is the same rustc 1.94.0. It then changes the version lines and
adds the one new item, as section 1.5 of the notices:

| Item | Disposition |
|---|---|
| lv-chordia 1.1.0 (MIT, Copyright (c) 2023 Music X Lab) | **Ported code in the executable; weights not distributed.** `src/chords/` is a Rust port of its network, CQT front end and chord vocabulary. The upstream `LICENSE` from the pinned wheel (SHA-256 `292241f0…ee0cb0`) is reproduced byte for byte in `third_party/lv-chordia-1.1.0-LICENSE` (SHA-256 `24fe07e4…8e0512`). The five weight files are downloaded from that wheel only when the user asks, each checked against a pinned SHA-256; they are not in the release assets. |

## Regenerating

The notices are generated from the locked dependency graph and retained,
hash-checked evidence (package archives, the upstream retrievals with their
URLs, times and SHA-256s, and the IPADIC archive). Any change to `Cargo.lock`,
the vendored crates or the toolchain means regenerating them before a release.

The research-only assembler is `research/v155/notices155.py`. It reuses the
hash-pinned 1.5.4 assembler and its retained evidence without changing either,
checks the current graph and archives, and writes provenance plus the exact
package inventory to a fresh evidence directory. It refuses unknown graph
changes and never builds or launches the player. The two completed generations
were byte-identical; the earlier refused attempts are retained as evidence rather
than described as successful runs.

That assembler uses the maintainer's retained local research directories, which
are not part of the distributed source snapshot. It is not an end-user build
step: compiling Moosik from the corresponding source does not require those
private evidence directories or running the notices assembler.
