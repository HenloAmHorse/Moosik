# Third-party notices: how they were assembled

The notices themselves are in [`THIRD_PARTY_NOTICES.txt`](../THIRD_PARTY_NOTICES.txt),
with two companion files in [`third_party/`](../third_party). This page says what
they cover and how the original open items were settled for 1.5.3, with
the scope rechecked for 1.5.4. It records license
information; it is not a legal opinion.

## Scope

- **What is covered:** the Windows x86_64 release executable with the default
  features. That is the 298 third-party packages `cargo tree --locked` reaches
  from `moosik` for `x86_64-pc-windows-msvc` (normal and build dependencies),
  the data they embed, the embedded fonts and the Rust standard library.
- **Build-time-only packages** (build scripts, proc-macros) are marked and
  included anyway, so no judgement about which of them reach the executable
  is needed.
- **Checked:** every package archive against its `Cargo.lock` checksum.
- **1.5.4 recheck:** dependency names, versions and lockfile checksums are
  unchanged from 1.5.3 except Moosik's own version. The current Windows
  normal/build graph was re-enumerated; the notices were regenerated from
  retained hash-checked evidence with rustc 1.94.0. There are 298 packages
  and 203 deduplicated texts, including the local-model setup license texts.
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
| `symphonia` 0.5.5 family (MPL-2.0) | Used unmodified. The notices state where the Source Code Form is available (crates.io, same name and version, and the upstream repository). |
| Rust standard library | The toolchain's `COPYRIGHT-library.html`, unmodified, in `third_party/`. |
| Microsoft Visual C++ runtime | The executable imports `VCRUNTIME140.dll` and the Universal CRT. **Neither is redistributed;** the release notes state that the Visual C++ 2015-2022 Redistributable (x64) is required. |

## Regenerating

The notices are generated from the locked dependency graph and retained,
hash-checked evidence (package archives, the upstream retrievals with their
URLs, times and SHA-256s, and the IPADIC archive). Any change to `Cargo.lock`,
the vendored crates or the toolchain means regenerating them before a release.
