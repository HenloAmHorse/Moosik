# Exact parallel kernel preparation candidate

Base: `2a6031a540744fc3cd54eb4a08503781d206ef2c`. This note describes a
candidate, not a measured speed claim. The retained GUI artifact is unchanged.

## Scope

Only groups already admitted to `cqt::parallel::Workspace` prepare their
kernel bank in the explicitly current analysis pool. Each existing slot lends
its own already-budgeted FFT scratch. Fixed contiguous bin ranges fill an
ordered outer vector of initially empty vectors; finished dense kernel vectors
move into that bank without clones. Preparation joins before source FFTs can
reuse slot zero's scratch. Moving a workspace outside a pool uses serial
preparation and does not create or select a global pool.

The unchanged `kernel()` function constructs each kernel. Its original f64
arithmetic, window, support, frequency, normalization and forward FFT plan remain
unchanged. Source FFTs, inverse products, requested inverse algorithm, fold order,
progress and coefficient observers are unchanged. The no-workspace route remains
serial. There is no budget, inverse admission, precision, key, cache revision or
transform-support change.

The outer vector is metadata; its dense members are the existing kernel-bank
payload already charged by the plan. No extra dense buffers are introduced.
Preparation does not update inverse worker counters or inverse task timers.
`WorkStats::kernel_ffts` receives successful actual forward-call entries, counted
immediately before each FFT, rather than the intended batch length. On error no
`Output` or partially prepared bank is returned. The coarse Kernel phase now
includes the joined parallel preparation's wall time, not summed worker time.

## Safety and cancellation

Preparation invalidates the old coefficient slab's active range before work.
It validates the forward plan's length/direction, the batch shape, and **every**
allocated slot's forward scratch requirement before dense kernel allocation.
This is distinct from the existing inverse admission check. Errors and
cancellation join the scoped work before the partial bank is dropped. A final
caller-side poll rejects cancellation arriving after the last worker's poll.

## Focused gates added, not executed by this agent

- Same constructor + same forward plan: every complex kernel bit and output
  position for N=512/4096/32768, 7/13 bins, mono/joint admission, Hann and
  Blackman-Harris. Full banks and 5/1-bin tails. Exactly-sized admitted budgets,
  unchanged slot scratch pointer/length/capacity and untouched inverse stats.
- Move an admitted workspace outside its pool: cancellation polls assert no
  current Rayon worker, and the resulting bank remains bit-identical.
- Deliberately shorten only the final allocated slot's scratch for a one-bin
  tail. A host-independent forward stub refuses any executed FFT. A matching
  actual-forward positive control succeeds after scratch is repaired. Wrong
  plan length, inverse direction, empty and over-capacity batches are separate
  invalid cases.
- A private test-only post-publication observer forces cancellation after at
  least one completed kernel but before the complete 13-bin bank. No bank is
  returned; a retry reuses the workspace and matches the serial reference.
- A one-bin tail permits the slot's post-FFT poll, then cancels at the caller's
  post-join poll, so omitting that final check is observable.

Existing frozen full-complex and production-code pipeline gates also cover this
new preparation route; raw kernel comparisons alone do not replace them.

## Root-coordinated measurement / acceptance

First compile and run the focused/core frozen tests. Then compare the retained
84b6b45 standalone probe (runtime-equivalent to round-start 2a6031a) against this
candidate on the existing paired 48 kHz, 30 s, 1024-column, joint,
explicit-six-worker workload at BPO12/48/192, with three alternating repetitions
per arm/case: 18 runs. Keep phase diagnostics separate from timing runs.
Require unchanged coefficient bits, production codes, work counts and explicit
workspace bounds. Inspect kernel-region wall time and total wall time; no
benefit or a material regression means reject the candidate. The already-known
constant-Q time/frequency tradeoff and the comparison with ASLT are unaffected.
Only after acceptance should root run final app gates and freeze a new artifact.

No builds, timing runs, owner-profile access or commits were performed by this
agent while preparing the patch.
