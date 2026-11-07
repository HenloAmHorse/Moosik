# 1.5.5 prereview: common formats and measured low-N CQT parallelism

2026-10-21, Windows x86_64 MSVC, i7-9700F, Rust/Cargo 1.94.0.
This is a prereview, not authorization to merge, push, tag or release.

## Outcome

Common-format implementation remains as recorded in
[REPORT_FORMAT_AND_PAIR.md](REPORT_FORMAT_AND_PAIR.md): native mapping-family-0
mono/stereo Opus in Ogg/WebM/Matroska, AAC-LC/ALAC, AIFF and CAF through import,
playback and analysis. This does not add a YouTube client, video playback,
HE-AAC, WMA, APE, WavPack or universal profile/container support. AAC-LC tail
trimming and correctness-first from-start Opus seek limitations remain explicit.

The new accepted performance change allows the **existing** bounded parallel
inverse route to handle N512/1024/2048 groups. Previously they ran serially even
inside the selected analysis pool. No kernel, support, plan, multiplication,
precision, frame clock, level mapping or numerical/cache revision changes.
No balanced partition, planner scorer, kernel-preparation parallelism or larger
memory allowance is included. N<512 remains serial.

Full CQT's long bass response is **not fixed or concealed by smoothing**. Faster
calculation does not shorten its centred analysis window. A separately named
responsive/variable-Q choice remains future work, not a silent substitution.

## Source checkpoints

- Round start: `3f79c95c87ed7d5e205eb8dd75892aa6ad16a9e9`.
- Recovery: `refs/recovery/codex-v155-preview-phase-start-20261020`.
- Notices: `45e7eb5a84d7ace07e3f7c8ad1798f7eb20fb8ea`.
- Opt-in diagnostics: `d5e52d0f6d11f635e18e371f5a32428f71119d65`.
- Diagnostic syntax correction/baseline: `797fd51` (full revision in probe provenance).
- Low-N runtime/gates: `84b6b45df0d759d91b86f59f5085e6ea446fdf44`.

The final GUI's exact build/source revision, PE identity, hashes and source
archive are bound by the external artifact manifest. Later report/controller
changes are not further algorithm changes. Earlier frozen players are preserved.

## Why this candidate, not a scheduling guess

Two coarse profiles around one disabled run at each BPO cover the same 48 kHz,
30 s, 1024-column, joint, six-worker core. These six runs are **not** an overhead
confidence bound. Disabled diagnostics do no phase timestamping/group-vector
allocation, but do retain an environment lookup and coarse disabled branches.

At 48 BPO, approximately 2.76 s of an 8.5 s core run is the serial combined
inverse/fold/progress region: N512 about0.58 s, N1024 about1.10 s and N2048
about1.08 s. Kernel construction/transforms cost about1.4 s, source fill/FFTs
about0.55 s, and the already-parallel inverse region about3.5 s.

At 192 BPO, the distribution changes: kernel preparation about6.35 s, source
fill/FFTs about4.65 s, parallel inverse about6.0 s and combined serial work
about2.65 s. The latter contains both N2048 and a memory-constrained N2097152
group. That largest group remains serial; its budget was not weakened.

Timers count non-overlapping monotonic wall regions. Overlapping task-wall sums
are reported separately, **not** as CPU seconds. Four profiled records conserve
exclusive wall sums and their group/work counters. Detailed scopes:
[CQT_PHASE_INSTRUMENTATION.md](CQT_PHASE_INSTRUMENTATION.md).

The existing ceil partition can underfill allocated slots, but simply making
more tasks does not reduce its ideal `ceil(batch/slots)` critical path. That
observation was not promoted into a measured speed claim or patched here.

## Measured A/B screen

Two separately frozen app-profile standalone executables include the same
production CQT source and default release settings: opt-level3, thin LTO, one
codegen unit, panic-abort, no leaf diagnostics. Explicit six-worker pool;
48 kHz, 30 s, 1024 logarithmic centres20 Hz–24 kHz, joint Mix/L/R, Blackman,
180 fps, 5400 returned rows. Each process generates the same deterministic
chirp/noise signals before the timer. No model, GPU, player or compiler ran
during measurements.

Three alternating baseline/candidate repetitions at each BPO: **18 accepted,
zero failed/excluded**. Independent read-only review checked frozen binary
hashes, raw logs, order, configuration, unchanged work counters and medians.

| BPO | Baseline min / median / max (s) | Candidate min / median / max (s) | Median time saved |
|---|---:|---:|---:|
| 12 | 6.271 / 6.368 / 6.437 | 4.035 / 4.067 / 4.095 | 36.13% |
| 48 | 8.522 / 8.533 / 8.716 | 6.534 / 6.550 / 6.582 | 23.24% |
| 192 | 20.194 / 20.309 / 20.314 | 19.432 / 19.528 / 19.701 | 3.84% |

The timer includes core setup/plans/kernels/convolution/mapping/returned planes,
not decode, disk cache, UI, audio playback, fixture/pool construction or the
post-timer code digest. It includes neither source-model inference nor a
real-music/device experiment. The small 192-BPO saving is not generalized to
other machines or files. These are **this incremental change's** savings, not
the cumulative 1.5.4-to-1.5.5 result.

All per-BPO three-plane FNV identity aids match across arms/repeats; they are
not a cryptographic or pixel proof. Existing work counters match except the
intended parallel-group admission count. Peak explicit workspace is126331120
bytes at12 BPO and265324000 at48/192 BPO in both arms, within268435456 bytes.
This is vector payload, not peak process RAM or a guarantee of lower CPU work.

Frozen baseline SHA-256:
`99b1472b9b563a0113376de59de7520511db6da99986200eb56e1341e6a2c595`.
Frozen candidate SHA-256:
`6d628f3f301892a9ab92aa95e374ef8245779a8f81d39e4e04c413263770fbf9`.

## Exactness and regression gates

New real-kernel fixtures assert that their geometry actually selects
N512/1024/2048, then test mono/joint, 7/13-bin batches, 5/1-bin tails, six allocated
slots, whole/requested inverses, edge/middle rows, production codes and complex
component bits. The independently retained full route/frozen-prereview and
direct f64 definition tests remain in force. Serial-only budgets and calls
outside a selected pool cannot admit parallel buffers or a global pool.

Cancellation checks demonstrate invalidation/reuse, not a forced cancellation
after a coefficient has already been written. No new forced partial-write
schedule or universal floating-point/pixel proof is claimed.

- Targeted default debug CQT:22 passed,0 failed,1 manual benchmark ignored.
- Final default-feature full release/all targets on84b6b45:1415 passed,0 failed,
  42 ignored;31.93 s test execution after compilation.
- All-target/all-feature clippy exits0;19 unique(file,message) warning identities
  equal20c0b02, zero additions/removals. No-default check exits0,5 existing warnings.
- Final production-entry screen:5/5 arms publish and read back their full
  Mix/Left/Right codes. Every three-plane SHA-256 matches the previous833871f
  production baseline; input bytes and frame counts also match.

That last screen includes decode/statistics/analysis/encoding/store publication,
excluding fixture generation, pool prewarm and verification readback. One fresh
process per arm; optimized default-feature **test** build, not a GUI session.
Its wall seconds are ASLTn1=2.1924264, ASLTn9=16.5924378, CQT12=4.4823911,
CQT48=6.9671426 and CQT192=19.7415371. These single observations are regression
and data-path checks, not another three-repetition speed claim. In particular,
the core A/B percentages above must not be relabelled as whole-player savings.

Parallelism can use more of the allowed cores while analysis runs; it does
not promise lower instantaneous CPU utilization or fewer total operations.
For example, this one CQT48 production run used25.66 CPU seconds while the
previous baseline's median was24.94; that is not a controlled CPU-work regression
campaign, but it prevents presenting this as a demonstrated reduction in CPU
work. Resource/worker limits are not an application-wide CPU-percentage cap.

## Notices and packaging

The locked Windows default normal/build graph is exactly302 third-party
packages:295 registry archives plus7 patched vendored upstream packages. The
original `moosik-codec-opus` leaf is Moosik AGPL code, not another third party.
All archive bindings were checked;205 deduplicated texts; no unresolved item.
Two completed generations are byte-identical. Notice SHA-256:
`5baf9fe94c901f3e6cfa184b8ba2c3f2027da80ada7b48e43d87a11f2622fb57`.

The prereview accompanies its exact source snapshot, app license, notices and
all four companion files, including modified MPL sources under `vendor/`.
This is not a legal opinion, installer, separate portable-profile mode or new
published release. Optional model weights/runtimes are not bundled. CMake and
native C/C++ tools are build requirements, not listener runtime dependencies.

## Evidence / development corrections

Evidence root:
`C:/Users/PC/Programming/moosik-v155-home/preview-20261020/`.
The directory retains the campaign's start-date name; this continuation is
2026-10-21. `phase-screen/` holds6 records, `low-n-ab/` holds18 records/logs,
`notices-d/` holds the authoritative notice graph/provenance/generation.
Controllers are retained in this research directory; no old campaign is rerun.

The first diagnostic build failed because a clock statement landed between a
while condition and its body. Independent review also caught it; fixed in a
separate commit, without amending history. That failed log remains. Earlier
notice attempts refused a patched-package checksum assumption and a CRLF source
anchor; both are preserved alongside the two successful deterministic runs.
Do not describe the first attempts as green.

## Remaining 1.5.5 work

1. Owner normal-use check of the new GUI, using
   [OWNER_CHECKLIST_FORMATS_zh-TW.md](OWNER_CHECKLIST_FORMATS_zh-TW.md).
2. Measured kernel-preparation parallelism may be worthwhile: retain ordered
   empty outer Vec slots, call the unchanged kernel builder on admitted inverse
   slots and reuse each slot's scratch before source work. No dense clones or
   extra budget; join before source fills. The constrained serial largest group
   has no inverse slots and needs a separately admitted proposal. This is a
   design opportunity, not implemented or timed here.
3. Quantify actual bass impulse/gated-tone/bandwidth responses against ASLT and
   design an explicitly named responsive/variable-Q option if useful. Preserve
   Full CQT and disclose the time/frequency tradeoff.
4. Faster indexed Opus seeking, additional unsupported codecs/profiles and
   YouTube integration are separate work, not hidden prerequisites for this exe.

Main/local origin-main, the shared dirty checkout, stash, public repository and
old frozen executables are protected. No merge, push, tag or release occurs.
