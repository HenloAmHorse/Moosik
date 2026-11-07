# CQT speed/preview: measured timing addendum

Date: 2026-10-20, Asia/Taipei. **Complete:117/117 accepted runs.**
This follows the software-only handoff in `REPORT_SPEED_PREVIEW.md`; that
earlier report and its frozen artifact copy correctly record timing as not
yet run at their handoff. Do not reinterpret them as performance evidence.

## Scope and provenance

The owner closed the old player and explicitly requested timing. No player,
model, compiler or other probe was present at the initial preflight. All
measurements are sequential; compilation is between, never during, campaigns.
The runners check player/model/compiler contention before each child; the
extended runner also checks after each child. They stop rather than kill a
contender. This is not a claim of eliminating every Windows background task.

- Machine: i7-9700F, eight logical CPUs, Windows11 10.0.26100, about32GiB RAM.
- App runtime source: `20c0b020740fb1af72763e09a85a4107bf1a720e`.
- Initial probe/runner source: `e90b10b4a616618eb7f471f312b94d0ff025fe11`.
- Extended probe/runner: `17cb2557ae29c010091710034f1521344f8bdba5`.
- rustc1.94.0 MSVC; edition2024, opt3, thinLTO, one codegen unit, panicabort.
- Same rustfft6.4.1 vendor, rayon1.12.0/core1.13.0 and num-complex0.4.6.
- Separate immutable evidence/probe copies under
  `C:/Users/PC/Programming/moosik-v155-home/timing-20261020/`.
  `PROVENANCE_INITIAL.json` and `PROVENANCE_EXTENDED.json` pin the binary,
  source and runner hashes. Original raw JSONL records are not rewritten.

The three arms in each one-binary comparison are:

1. `frozen`: unchanged first CQT prereview core from f9a6999, whole inverse,
   serial. The unchanged baseline body remains independently pinned.
2. `serial`: new requested-output core, outside a Rayon pool.
3. `pool6`: new core installed in an explicit six-worker pool, matching the
   owner's chosen worker limit. Allocated slots are not measured concurrency.

Inputs are deterministic synthetic chirp plus noise, distinct L/R seeds. The
timer includes planning, kernel preparation, transforms, normalization,
production-equivalent code mapping and output-plane allocation. Fixture and
pool creation, digesting, decode, cache encoding/IO, UI and audio devices are
excluded. Cancellation/progress callbacks are no-ops here, unlike production's
atomic/progress/ETA updates. Thus these are **transform-core**, not complete
analysis-lane, application CPU-percentage or full-song player timings. They
do not establish a speed comparison against ASLT.

Each case has three repeats per arm. Run order is frozen/serial/pool6,
pool6/serial/frozen, then frozen/serial/pool6. Tables give medians and observed
min–max, not confidence intervals. Time saved is `100*(1-new/old)`; speedup is
`old/new`. These percentages are not display/FFT frame-rate changes.

## Initial 90-run matrix — complete

1024 columns, **20Hz–20kHz**,48BPO,Blackman,180fps. This upper frequency is
explicitly NOT the product default24kHz/Nyquist-clamped geometry. Mono means
one Mix plane; joint computes L/R and derives Mix from their complex mean,
returning three planes.

All times below are seconds: **median [min–max]**.

| SR / input / mode | Frozen | New serial | New six-worker | Six-worker time saved |
|---|---:|---:|---:|---:|
|44.1k /2s /mono|2.769 [2.756–2.788]|2.231 [2.212–2.238]|1.790 [1.784–1.817]|35.4%|
|44.1k /2s /joint|3.915 [3.910–4.003]|3.100 [3.083–3.106]|2.201 [2.198–2.261]|43.8%|
|48k /2s /mono|3.083 [3.044–3.095]|2.203 [2.203–2.264]|1.835 [1.832–1.870]|40.5%|
|48k /2s /joint|4.401 [4.365–4.455]|2.896 [2.895–2.962]|2.168 [2.155–2.182]|50.7%|
|96k /2s /mono|6.702 [6.681–6.713]|4.710 [4.710–4.710]|3.886 [3.880–3.920]|42.0%|
|96k /2s /joint|9.787 [9.707–9.801]|6.287 [6.213–6.336]|4.767 [4.727–4.819]|51.3%|
|48k /10s /mono|7.133 [7.107–7.145]|4.405 [4.399–4.457]|2.780 [2.718–2.782]|61.0%|
|48k /10s /joint|12.644 [12.588–12.830]|7.394 [7.263–7.413]|4.055 [4.025–4.058]|67.9%|
|48k /30s /mono|16.829 [16.808–17.046]|9.747 [9.616–9.748]|5.067 [5.038–5.076]|69.9%|
|48k /30s /joint|32.477 [32.260–32.635]|18.214 [18.090–18.217]|8.608 [8.596–8.651]|73.5%|

Across these cases serial saves19.4–43.9%; six-worker saves35.4–73.5%.
The30s joint case is3.77x faster, not73.5x or a73.5% frame-rate increase.
Short inputs include planning/setup and should not be extrapolated as a
whole-track rate.

Root and a separate read-only agent validated90/90 unique expected keys,
three repeats per arm, exact row order, geometry, finite positive times,
frame/output dimensions and stable per-arm work statistics. The agent also
checked `pruned+whole=inverse`, serial no-pool state and timestamp order.
All9 FNV triples per workload match. These are compact code-array identity
aids, not cryptographic proofs or byte-identical cache files: this probe
does not encode cache files. Prior frozen-reference complex-bit/code-value
gates remain the separate numerical equivalence evidence.

## Long/default-geometry and extra stress follow-up

The optional maximum-frequency argument explicitly uses the same f64
Nyquist clamp and logarithmic centers as `cqt_pipeline`. It preserves the
original five-argument20k geometry. Three further joint cases,27runs, are
measured:48k120s1024/24k;96k30s1024/24k;48k30s2048/20k.

The first two use the product's default geometry. The last is **core stress
only**, not an owner-selectable CQT setting: CQT UI columns are16..1024, and
ASLT's2x grid does not multiply genuine CQT columns. BPO is support Q, not
display oversampling. No production setting was changed for this campaign.

All three cases completed9/9 each. The extended runner exited0 at12:12.

| SR / input / columns / range | Frozen | New serial | New six-worker | Six-worker time saved / speedup |
|---|---:|---:|---:|---:|
|48k /120s /1024 /20–24k|130.276 [127.370–130.925]|68.279 [68.152–68.316]|29.588 [29.353–30.007]|77.3% /4.40x|
|96k /30s /1024 /20–24k|74.627 [74.435–74.713]|39.243 [39.142–39.323]|19.540 [19.392–19.543]|73.8% /3.82x|
|48k /30s /2048 /20–20k (**stress only**)|66.629 [66.623–67.745]|36.260 [36.139–36.566]|17.161 [16.764–17.250]|74.2% /3.88x|

The serial arm alone saves47.6% here. The six-worker core finishes this120s
synthetic joint fixture in under30s at its median. This is observed headroom,
not an end-to-end or arbitrary-song completion guarantee. All nine code-plane
digest triples match; output size is132,710,400bytes (three21600x1024u16planes).

The96k case saves47.4% with the serial arm; the2048-column stress case saves
45.6% serial. Every measured case improves in both candidate arms. Across all
13workloads, the observed median time reductions are19.4–47.6% serial and
35.4–77.3% six-worker. Do not generalize this range to arbitrary music,
machines, BPO settings or whole applications.

## Final disposition

Initial session16536, intervening compiler31117 and extended session10091
all exited0. All117 scheduled measurements were retained, with no failed or
excluded timing run. Every workload's three code-array digest strings match
across all arms/repeats. Root validated complete matrices and an independent
read-only agent cross-checked keys, geometry, output sizes, work counters,
ordering, medians and ranges. Numerical gate scope remains as stated above.

At12:12:35 postflight no matching player/model/compiler/probe process was
present. Main and local origin/main remain330b8ac, the shared checkout stays
71be39b with15status entries, and the original stash is intact. No remote
operation occurred. Only research harness/report changes were committed;
`git diff20c0b02 -- src Cargo.toml Cargo.lock vendor` is empty.

The existing frozen20c0b02 player is still the test artifact; no new app build
is needed for these results. Its runtime and software gates have not changed,
so they were not rerun merely to time the same code. Original artifact files,
checksum list and earlier manifest remain unmodified; this measured report
and a separate timing checksum/manifest are added alongside them. The owner
can resume normal player use. Stop measurement work here: no new campaign,
automatic publication or extra soak is started.

## Boundaries

- Peak explicit workspace stays at/below256MiB; this is vector payload,
  not physical RAM, peak working set or total process commit.
- Requested-output calls count API entries, not wholly avoided dense FFTs;
  inner leaves may still run full transforms. Input FFT calls can increase
  when parallel admission uses smaller kernel batches.
- No live device, nativeDSD/DoP, UI pixel test, GPU speedup, physical-memory
  measurement, soak or real-music end-to-end claim comes from this probe.
- No runtime source, cache revision, frozen player executable, owner profile,
  main, shared checkout or public repository is changed by timing work.
- No merge, push, tag, release or additional optimization experiment is
  authorized or implied by this measurement.
