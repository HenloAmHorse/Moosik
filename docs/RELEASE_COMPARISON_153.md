# Release comparison: 1.5.2.1 and 1.5.3

Measured 2026-10-14 on one computer. These are the numbers for these inputs and
this setting on this machine, not a general speed-up: other tracks, settings
and computers will differ. 1.5.3 was released on 2026-10-15; which trees were
measured is stated in each section.

## What was compared

- **Arm A:** 1.5.2.1 as released (`4ca1300`).
- **Arm B:** the 1.5.3 closeout candidate (`6ebac37`; the later commits change
  documentation only). The analysis timings below are from that tree; the
  later cache repair (`c9d1df6`, `3e846f9`) changes cache publication and
  cleanup, not the analysis, and the load section was measured on it.
- **Harness:** a test-only module in each tree that calls production's
  `preprocess_file`, and, for loads, production's own cache lookup. One fresh
  process per measurement, release build. Wall time is measured around the
  call; process CPU is user + kernel time.
- **Setting:** superlet pre-processing, 9 wavelets, Q ratio 2, 1024 bars,
  180 fps, 20 Hz–24 kHz, CPU route (`MOOSIK_GPU=0`) with a 6-thread pool. Mono
  is the Mix only; joint is Left and Right with the Mix derived from them.
- **Machine:** Intel Core i7-9700F (8 cores), 32 GB RAM, Windows 11. Nothing
  else heavy ran during the measurements.

## Inputs

All inputs are 32-bit float stereo WAV files, written sample for sample from
retained decoded audio. "DSD" is PCM that the DSD decimator produced earlier,
so its timing is PCM analysis timing; **it does not include DSD decoding.**

| Input | Rate | Length |
|---|---|---|
| H96 | 96 kHz | 210.8 s |
| L44 | 44.1 kHz | 603.0 s |
| DSD-derived | 88.2 kHz | 391.1 s |

## Procedure (declared before the first run)

- **Runs:** two pairs per cell, in the order A B, then B A.
- **Extra pair:** one more pair, only for a cell where B was more than 5 %
  slower than A in wall time or CPU, B's peak commit was more than 10 % higher,
  or either arm's two runs were more than 5 % apart in wall time.
- **Triggers in this comparison:** the extra pair ran for H96 mono and L44
  joint. Both triggers were the spread between runs; neither was B being
  slower. That made 28 analyses, under the limit of 36.

## Pre-processing (analysis from scratch)

RTF is wall time divided by track length; below 1 is faster than real time.

| Input | Mode | 1.5.2.1 wall s (mean) | 1.5.3 wall s (mean) | 1.5.2.1 RTF | 1.5.3 RTF | 1.5.3 / 1.5.2.1 wall | CPU | Peak commit |
|---|---|---|---|---|---|---|---|---|
| H96 | mono | 241.3 | 151.7 | 1.145 | 0.720 | 0.63 | 0.62 | 0.93 |
| H96 | joint | 516.7 | 258.2 | 2.452 | **1.225** | 0.50 | 0.47 | 0.64 |
| L44 | mono | 251.8 | 161.3 | 0.418 | 0.268 | 0.64 | 0.62 | 0.81 |
| L44 | joint | 553.0 | 297.9 | 0.917 | 0.494 | 0.54 | 0.52 | 0.60 |
| DSD-derived | mono | 428.3 | 188.9 | 1.095 | 0.483 | 0.44 | 0.42 | 0.64 |
| DSD-derived | joint | 855.9 | 354.0 | 2.189 | 0.905 | 0.41 | 0.40 | 0.61 |

**Peak commit, the largest of each arm's runs:**

| Input | Mode | 1.5.2.1 | 1.5.3 |
|---|---|---|---|
| H96 | mono | 0.62 GiB | 0.58 GiB |
| H96 | joint | 1.02 GiB | 0.65 GiB |
| L44 | mono | 0.99 GiB | 0.79 GiB |
| L44 | joint | 2.05 GiB | 1.23 GiB |
| DSD-derived | mono | 0.87 GiB | 0.56 GiB |
| DSD-derived | joint | 1.80 GiB | 1.10 GiB |

**Same output.** For every input and mode, every run of both versions produced
byte-identical Mix, Left and Right codes. Separately, the final candidate's
codes were compared with the accepted reference outputs: seven synthetic edge
cases, the first 20 s of each input, and one case at the production memory
budget. **No code differed.** These are the quantised analysis codes the
display draws from, not a comparison of rendered pixels.

**Still slower than real time.** H96 joint analysis remains slower than real
time on this machine: RTF 1.23.

## Loading a finished analysis

**Re-measured 2026-10-15 for the final 1.5.3 candidate.** Two earlier load
measurements are superseded: the first compared only the Mix after a joint
load; the second (repair round, 120 loads, about 1.3x slower) was valid but
measured the repaired tree before the load path was made faster. In that
second table the "Mix" rows were Mix-only loads, not a pooled mean that
included the Mix part of joint loads; and peak working sets were unchanged at
the displayed precision, not byte-identical (the repaired tree's maxima were
0.32 to 0.49 MiB higher).

- **What was run:** two test-harness builds, not the release executables.
  1.5.2.1 is measurement branch `74da796`: `4ca1300` plus a test-only load
  harness (`src/spectrum/rel_measure.rs`) that writes every plane it loaded
  and records process CPU. 1.5.3 is measurement branch `b7985fc`: the
  parallel Left/Right candidate (`3f4c3fd`, the same tree as `757c3fb`) plus
  the same harness. The release source differs from `b7985fc` by removing the
  harness, removing one unused wrapper and marking three functions test-only;
  the load functions production calls are the same, but the tree is not the
  release tree.
- **Caches:** the six caches saved for the repair-round measurement (one joint
  analysis per input and version), reused, each file checked against its
  recorded SHA-256 first. Their Mix, Left and Right reference planes equal
  the earlier timing campaign's.
- **Loads:** 3 inputs x 2 versions x 2 demands x 10 = 120, each a fresh
  process, version order alternated. Files probably still in the system's
  file cache (not flushed). Every load an exact hit: no analysis started, the
  cache folder unchanged.
- **Checked, all 120 passed:** every required plane byte-identical to its
  reference with the reference dimensions; exactly one load per input,
  version, demand and repetition (26 deliberate bad records or sets, all
  rejected).

Mean seconds over 10 loads (the standard deviation of every wall time below
is under 0.006 s; that bound is for wall time only, not for the CPU figures). "Mix only" is a
load that needs only the Mix; "joint" is one that also loads Left/Right, and
its Mix part is listed separately:

| Input | Load | 1.5.2.1 | 1.5.3 | 1.5.3 / 1.5.2.1 |
|---|---|---|---|---|
| H96 | Mix only | 0.136 | 0.130 | 0.96 |
| H96 | joint: Left/Right | 0.295 | 0.165 | 0.56 |
| H96 | joint: Mix + Left/Right | 0.430 | 0.297 | 0.69 |
| L44 | Mix only | 0.385 | 0.365 | 0.95 |
| L44 | joint: Left/Right | 0.837 | 0.466 | 0.56 |
| L44 | joint: Mix + Left/Right | 1.220 | 0.831 | 0.68 |
| DSD-derived | Mix only | 0.260 | 0.251 | 0.97 |
| DSD-derived | joint: Left/Right | 0.565 | 0.321 | 0.57 |
| DSD-derived | joint: Mix + Left/Right | 0.824 | 0.572 | 0.69 |

- **Loads are no slower than in 1.5.2.1**, although 1.5.3 checks a SHA-256
  of every analysis file it reads. The check now runs on a second thread
  while the data is decoded (the decoded result is used only if the check
  passes), the Left and Right channels are decoded at the same time, and a
  payload-sized copy was removed.
- **What it costs instead:** more processor time and, for Left/Right, more
  memory at once. Process CPU for a joint L44 load is 1.64 s against 1.23 s
  (about 1.3x); the peak working set of a joint load is higher by 56 to 160
  MiB (L44: 1116 against 957 MiB). Mix-only loads use the same memory as
  before (within 0.6 MiB).
- **Caveat on the Mix:** the cause of the Mix-only gain (3 to 5%) is not
  established. An intermediate build with the same Mix code measured slower
  Mix loads; a compiler code-layout difference is one possible explanation,
  not a proven one. The Left/Right gain comes from the parallel decode.
- The Mix loads when a track is opened; Left/Right load when a channel view
  first needs them, and can load again for the same track when what the view
  needs changes.
