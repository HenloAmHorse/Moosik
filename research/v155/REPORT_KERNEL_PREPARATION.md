# CQT follow-up: exact parallel kernel preparation

2026-10-21, Windows x86_64 MSVC, i7-9700F, Rust/Cargo 1.94.0.
Prereview only: no merge, push, tag, version bump or release.

## Outcome and scope

Retain the bounded scheduling change in runtime commit
`6081ef6b2a907b3ba0ca11b35661d85b8e84e0d7`. Different CQT kernels are now
prepared using scratch owned by the already-admitted analysis slots, before
the source transforms reuse that scratch. The unchanged kernel constructor,
same forward plan and ordered bank are used; there is no dense clone or second
bank, larger allowance, new dependency, revised cache identity or DSP change.
No-workspace groups remain serial; calls outside an explicit pool never select
a global pool. Mapping, progress and coefficient observers remain caller-side.

The change saves **6.46% / 15.46% / 13.78%** core wall time at BPO12/48/192 in
the bounded paired screen below, relative to the previous frozen runtime.
These are this change's incremental measurements, not cumulative 1.5.4 gains,
GUI frame rates, real-music guarantees or reductions in total CPU operations.
It can keep more of the selected workers busy while analysis is running.

Full CQT bass responsiveness is unchanged. A faster calculation does not
shorten the centred window. GPU/multirate CQT, a separately explicit responsive
mode, additional codec profiles and YouTube integration remain unimplemented.
ASLT, live FFT, translation models and owner profiles are not changed here.

## Source and controls

- Round start: clean `2a6031a540744fc3cd54eb4a08503781d206ef2c`.
- Recovery: `refs/recovery/codex-v155-kernel-prep-start-20261021`.
- Candidate runtime/tests: `6081ef6b2a907b3ba0ca11b35661d85b8e84e0d7`.
- Frozen baseline probe: `84b6b45df0d759d91b86f59f5085e6ea446fdf44`.
  Its CQT source equals round-start 2a6031a; it was copied, not rebuilt and
  represented as an earlier executable.
- Baseline SHA-256:
  `6d628f3f301892a9ab92aa95e374ef8245779a8f81d39e4e04c413263770fbf9`.
- Candidate SHA-256:
  `9d313088e07969ba10c499581f589c19a2f8129d8085cf671bb0f68d725ddb27`.

The external artifact manifest binds the final ordinary GUI and its exact
tracked-source snapshot. Later result/checklist/document edits are not runtime
algorithm changes. Earlier frozen GUI artifacts are not overwritten.

## Paired measurement

Standalone probe includes the production core, using the ordinary app release
settings: opt-level 3, thin LTO, one codegen unit, panic-abort. Deterministic
30-second stereo chirp/noise at 48 kHz, 1024 logarithmic centres 20 Hz–24 kHz,
Blackman, 180 fps, 5400 returned frames, explicit six-worker pool. Fixture and
pool construction and post-timer digests are outside the core timer; planning,
allocation, kernels, transforms, folding/mapping and returned planes are inside.
Decode, disk cache, UI, playback, GPU and model inference are not measured.

Three alternating repetitions per arm/case: **18 accepted, 0 failed/excluded**.
The existing `run_cqt_ab.ps1` controller was used, with phase diagnostics off.
No player, model or compiler ran concurrently. Independent read-only review
checked every raw record, binary hash, full revision, order, geometry, the
complete emitted statistics object and output identity aids. The probe does
not emit every internal WorkStats field; app tests cover additional counters.

| BPO | Baseline min / median / max (s) | Candidate min / median / max (s) | Median time saved |
|---|---:|---:|---:|
| 12 | 4.078 / 4.080 / 4.144 | 3.754 / 3.816 / 3.817 | 6.46% |
| 48 | 6.494 / 6.529 / 6.546 | 5.463 / 5.520 / 5.566 | 15.46% |
| 192 | 19.380 / 19.410 / 19.450 | 16.552 / 16.735 / 16.760 | 13.78% |

All reported work statistics are identical across all six records per BPO:
kernel/input/inverse FFTs, batches, pruning counts, parallel groups, selected
slots, output size and explicit workspace. The latter remains 126331120 bytes
at 12 BPO and 265324000 at 48/192, below 268435456. This is explicit vector payload,
not peak process RSS; allocator, planner and scheduling metadata remain excluded.
Three FNV code-plane identity aids per BPO match across arms/repetitions. FNV
does not substitute for the bit/code tests below or constitute a pixel proof.

## Separate attribution profiles

One additional profiled sample per arm at 48/192 BPO: four runs, not new
confidence intervals or an instrumentation-overhead estimate. Timers conserve
exclusive wall time; overlapping inverse-task durations are not CPU seconds.

| BPO | Kernel construction + forward FFT, baseline → candidate (s) | Other observations |
|---|---:|---|
| 48 | 1.378 → 0.438 | Source 0.540→0.511; inverse region 4.176→4.252 |
| 192 | 6.300 → 3.793 | Source 4.617→4.458; inverse region 6.202→6.312; serial region 1.563→1.566 |

The intended kernel region accounts for the observed saving. Remaining source
and inverse costs are not relabelled as optimized. The largest memory-limited
group still has no parallel workspace and is deliberately untouched.

## Exactness, ownership and cancellation

Targeted app debug gate: **25 passed, 0 failed, 1 ignored** in 47.07 seconds.
This includes the existing independent direct-f64 definition and frozen full
complex-component/production-code comparisons with explicit pools and budgets.

Three added tests establish:

1. Each prepared kernel matches every complex component bit of the unchanged
   constructor plus the same forward plan. Actual N512/4096/32768 layouts,
   7/13-bin banks, 5/1-bin tails, mono/joint admission, two windows, exact admitted
   payload limits and moved-outside-pool serial fallback are exercised. Scratch
   pointer/length/capacity and inverse statistics remain unchanged.
2. Every slot's forward scratch is validated before dense preparation, including
   an unused last slot in a one-bin tail. A matching real-forward positive
   control succeeds after repair. Wrong length/direction and invalid batches
   fail separately. An initial mismatched-half fixture could have masked this
   check; it was corrected before the first build, not accepted as evidence.
3. A test-only observer cancels after at least one completed kernel; the partial
   bank is not returned. A one-bin schedule permits the slot's last poll and
   refuses the post-join poll. Retry succeeds with exact bits. All Rayon work
   joins before an error drops the partial bank; prior coefficients are inactive.

The original per-kernel arithmetic and FFT operation graph are unchanged.
Finite fixture comparisons are not a universal floating-point or pixel proof.
Cancelled analyses still return no Output/WorkStats; successful kernel counts
are actual entered forward calls, not intended batch lengths.

## Final integration gates and artifact

On runtime 6081ef6, with only documentation/checklists subsequently edited:

- Default-feature release tests, all targets: **1418 passed, 0 failed,
  42 ignored**, 32.65 seconds after the 6-minute compile. Three tests were added.
- Clippy, all targets/all features: exit 0, 19 unique primary-file/message
  warning identities, the same set as the prior frozen low-N gate; no new ones.
- No-default-features check: exit 0, the same five existing warnings.
- Normal Git whitespace checks: clean; no repository-wide formatter run.

A separate production-entry screen ran ASLT n1/n9 and CQT12/48/192 through the
actual preprocessor and cache write/read path. **5/5 succeeded**. Every Mix,
Left and Right plane SHA-256, input SHA, returned frame count and frame rate
match all three earlier 833871f baseline repetitions for that arm. Every code
read back from the published cache equals memory. No owner cache was used.

Single observations, not new robust speed claims: ASLT n1=2.195101 s,
CQT12=4.1717594 s, ASLT n9=16.3575224 s, CQT48=6.0532929 s,
CQT192=17.403487 s. These include the production pipeline and must not be mixed
with core-only medians or treated as matched-quality ASLT/CQT comparisons.

The final distinct ordinary GUI build/export, PE identity, source commit,
source archive and companion-file hashes are recorded by the external
ARTIFACT.json and handoff, not established merely by this report. The final
source-freeze commit adds documentation/checklists only after runtime 6081ef6.
No new dependency/feature/notice graph was introduced. The retained default
Windows 302-package/205-text notices remain applicable; modified third-party
sources accompany the GUI source snapshot. The agent does not launch the GUI
or make a merge, push, tag or release.

## Evidence and reproduction

External evidence root: `C:/Users/PC/Programming/moosik-v155-home/kernel-prep-20261021`.
It retains preflight and previous RUN_STATE, build/test logs, separately frozen
probes, 18 paired raw runs and four profile runs. `run_cqt_ab.ps1` records full
revision/hash provenance. The candidate build command is:

```powershell
cargo build --release --locked --offline --manifest-path research/v155/speed_probe/Cargo.toml
```

Coordinate builds and measurements sequentially. No owner media/profile/device
session or model launch is needed. App gates use the configured VS2022 CMake
generator and retained Lindera dictionary cache. Normal Git whitespace checks
are required; repository-wide formatting is not applied. See
[CQT_KERNEL_PREPARATION_PLAN.md](CQT_KERNEL_PREPARATION_PLAN.md) for the focused
candidate scope and [OWNER_CHECKLIST_KERNEL_zh-TW.md](OWNER_CHECKLIST_KERNEL_zh-TW.md)
for the short normal-use check.
