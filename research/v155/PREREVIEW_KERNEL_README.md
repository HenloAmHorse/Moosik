# Moosik 1.5.5 prereview: exact parallel CQT kernel preparation

Run the `Moosik-1.5.5-prereview-*.exe` in this folder after closing other
Moosik windows. This is the normal GUI player, not a benchmark runner or a
published release. It uses and saves the normal profile, not a separate
portable-profile mode. The agent has not launched it on the owner's profile.

This incremental candidate prepares different CQT kernels in parallel using
the existing admitted analysis slots and their scratch buffers. Each kernel's
f64 construction, forward FFT, support, window and subsequent output order
are unchanged. There is no larger memory allowance, new dependency, changed
cache revision, hidden resampling, shortened bass window or smoothing change.
It does not change ASLT, real-time FFT or translation models.

The inherited common-format support and its limitations remain: mono/stereo
native Opus, AAC-LC/ALAC, AIFF and CAF; WebM/MP4/Matroska import audio, not
video. This is not a YouTube client or universal codec-profile support.

Start with OWNER_CHECKLIST_zh-TW.md. Existing cache hits are unaffected; to
observe analysis speed, use a track/settings combination without a matching
cache. More of the selected analysis workers may be busy during preparation;
shorter wall time is not a promise of lower instantaneous CPU usage.

The corresponding `*-source.zip` is an exact tracked-source snapshot. Its
commit, the tested runtime commit, executable identity and companion hashes
are recorded in ARTIFACT.json and SHA256SUMS.txt. Cargo.lock binds remaining
registry dependencies; this is not a complete offline toolchain/dependency
cache. See the source README for build prerequisites.

Keep LICENSE, THIRD_PARTY_NOTICES.txt, third_party/ and corresponding source
with this prereview. The unchanged locked default Windows graph has 302
third-party packages and 205 deduplicated texts. Optional model weights and
runtimes are not bundled. The x64 VC++ 2015–2022 Redistributable remains
required; no separate FFmpeg or Opus DLL is required.

Full CQT bass responsiveness, GPU/multirate CQT and additional codec profiles
remain open. This artifact is not authorization to merge, push, tag or release.
