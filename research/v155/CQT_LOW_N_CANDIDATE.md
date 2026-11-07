# Bounded low-N parallel admission candidate

2026-10-21. Candidate only; no agent-side build or measurement was run.

The runtime change is `MIN_PARALLEL_N: 4096 -> 512` in `cqt/parallel.rs`.
No slot/batch scorer, balanced partition, kernel preparation parallelization,
FFT graph, product, support, clock, gain, precision, code mapping or key revision
is changed. Existing memory admission and explicit-current-pool requirements
remain in force; a small budget still selects serial work. N<512 stays serial.

Reason for screening: root's coarse baseline profiles find material combined
serial inverse/fold/progress wall time in N512/1024/2048 groups. It is not a
claim that scheduling those groups will be faster: short task dispatch can
cost more than the inverse work it distributes. Keep this candidate separate
from planner or arithmetic experiments so the A/B answer is interpretable.

New gates:

- `genuine_low_n_parallel_groups_match_the_retained_full_complex_bits_and_codes`
  builds actual CQT kernels whose layout must be exactly N512/1024/2048, uses
  7/13 frequency centres and mono/joint inputs, and enters a six-worker pool.
  It compares every observed complex component bit and production v4 code
  against the retained complete product/inverse route. It requires actual
  low-N parallel entry, six allocated slots and unchanged FFT/work counts.
- `genuine_low_n_kernels_keep_full_bits_codes_budget_and_cancellation_with_partial_batches`
  checks the workspace with the same actual CQT kernels, pruning disabled and
  enabled where supported, and 7/13-bin full plus 5/1-bin tail batches. It
  compares coefficients/codes against a complete inverse using that same FFT
  plan, checks start/middle/end blocks, refuses a serial-only budget and
  implicit global-pool entry, hides cancelled slabs, and reuses the workspace
  successfully after cancellation. It separately requires N256 refusal.
- The existing coarse-diagnostics on/off test keeps its serial assertion. Its
  fixture now uses 12 BPO so treble kernels genuinely fall below N512. At 48
  BPO even a Nyquist-adjacent filter needs at least N512; increasing frequency
  cannot retain an honest sub-512 serial fixture there.

Suggested coordinated app gates (no build started by the agent):

```powershell
cargo test --release --locked genuine_low_n -- --test-threads=1
cargo test --release --locked spectrum::cqt::tests::coarse_phase_diagnostics_change_neither_coefficients_codes_work_nor_progress -- --exact --test-threads=1
```

Run the existing frozen-baseline and budget/cancellation CQT tests as well.
The new complete-inverse route reference complements rather than replaces
the independently frozen prereview and direct f64 definition gates.

For timing, use identical app-profile probe sources/build options in retained
baseline and candidate binaries, a quiet machine and explicit six-worker pool:
48 kHz, 30 s, 1024 columns, joint, 20 Hz–24 kHz, Blackman/180 fps, BPO48 and
BPO192. Three rotated repetitions per arm/case are a bounded speed gate.
Keep output identity aids and all raw runs, plus occasional opt-in phase
profiles to explain the result. A timing loss rejects this candidate; it does
not justify changing the budget, numerics or temporal response to make it win.
