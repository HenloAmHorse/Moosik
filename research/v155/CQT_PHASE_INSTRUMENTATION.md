# Opt-in CQT coarse phase screen

Instrumentation baseline only. No scheduling, support, precision, numerical
revision, cache key or output-mapping change is intended. Agent-side work did
not build or run these probes; results must be recorded separately.

`MOOSIK_CQT_PHASES=1` is read once at the CQT analysis boundary on every run.
Other values disable it. No static environment cache or per-task environment
lookup is used. An explicit internal flag serves the on/off equivalence test,
without mutating process environment while parallel tests run.

Successful enabled runs emit exactly one stderr line beginning `CQT_PHASES `,
followed by JSON schema `moosik-cqt-coarse-phases-v1`. Disabled and failed runs
emit no such completion record. Probe stdout JSON remains separate.

## Regions and accounting

All durations are monotonic-clock **nanoseconds**, not CPU-time samples.

- `setup_wall_ns`: layout, validation and returned-plane allocation.
- `planning_wall_ns`: FFT plans, group geometry and parallel admission.
- `allocation_wall_ns`: workspace/optional-request/serial-buffer admission and
  allocation; failed optional allocation attempts remain charged.
- `kernel_prepare_fft_wall_ns`: one region per batch, including kernel storage
  allocation, trig/window construction and kernel forward transforms.
- `source_fill_fft_wall_ns`: fill/validation and channel source FFTs per block.
- `parallel_inverse_region_wall_ns`: the complete dispatch/join/gather API
  region, once per block. Workers include requested-index setup and gathering
  into the coefficient slab.
- `parallel_inverse_worker_ns`: the **overlapping sum of task wall durations**.
  It is not added to the wall-region sum, is not process CPU time, and need
  not equal the dispatch region multiplied by allocated slots.
- `parallel_fold_progress_wall_ns`: coefficients, normalization, Mix, norms,
  mapping, observer and progress calls, as one region per block.
- `serial_inverse_fold_progress_wall_ns`: inverse and fold/progress together,
  once per serial block. No timers were added inside a per-bin or frame loop
  to pretend this region can be split cheaply.
- `unclassified_wall_ns`: the nonnegative difference between core total and
  named non-overlapping regions. Includes control, serial requested-index
  setup, drops and instrumentation bookkeeping outside those regions.

The core total stops before JSON serialization/stderr. The outer probe wall
timer naturally includes their small cost; it is not the same timing scope.
Group records retain FFT length, bins, slots, batch capacity, FFT counts and
actual explicit workspace, with the same region accounting. Phase maxima
count task invocations, not simultaneously executing CPU cores.

The disabled helper takes no timestamps and allocates no group vector. There
is still a one-per-run environment lookup and disabled branches at the coarse
boundaries. No universal zero-overhead claim is made. The planned on/off screen
is an observation, not a statistically robust overhead bound.

## Gates and a bounded measurement

After root coordinates a build, run the new app tests:

```powershell
cargo test --release --locked spectrum::cqt::phases::tests -- --test-threads=1
cargo test --release --locked spectrum::cqt::tests::coarse_phase_diagnostics_change_neither_coefficients_codes_work_nor_progress -- --exact --test-threads=1
```

The latter compares every complex component bit, v4 code, existing work
counter and progress value with diagnostics off/on. It also checks region
conservation, input-FFT/source-block counts and serial/parallel coverage.
Run the existing frozen-baseline CQT gates too; the on/off test is not their
replacement.

The existing app-profile standalone probe includes this same CQT source and
is the cheap first timing screen. Root's `run_phase_screen.ps1` runs 48 kHz,
30 seconds, 1024 columns, joint, six workers at BPO 48 and 192: enabled/off/
enabled, six runs total. It pins source/binary identity and preserves all raw
logs. Do not rerun the old 117-run campaign. Compare output identity aids in
every arm, and report single off-run overhead comparisons as descriptive.

Do not select a new slot/batch heuristic until these phase/group records show
which regions actually dominate. Balanced dispatch remains a separate later
candidate and must retain the same frozen-bit/code and budget gates.
