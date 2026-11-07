# Exact NVIDIA GPU assist for the superlet pre-process (1.6.1, experimental)

Off by default: **Pre-process settings → GPU → "Exact NVIDIA assist (experimental)"**
(`spectrum.json`: `"exact_gpu": true`). With it on, part of a stereo pre-process runs on an
NVIDIA graphics card while the cores run the rest, and the analysis is the same to the last bit.

This implements the 2026-11-03 exact-ASLT research handoff
(`research/algorithm_20261023/handoff_161_20261103/CLAUDE_HANDOFF_161_ASLT_EXACT_GPU.md`,
Codex research worktree) on the private 1.6.1 line, with the 2026-11-04 review's corrections
(`review_161_20261104/REPLY_TO_CLAUDE_EXACT_GPU.md`).

## What the device computes

The superlet analysis (`src/spectrum/aslt.rs`) groups bars by their longest kernel and runs each
group size by size. In the 2^15 pass, a member's column is the overlap-save convolution of the
two channels with its kernel: per 32768-point block, the product of the block's spectrum with the
kernel's and a pruned inverse transform that computes only the hop-spaced outputs the block owns
(`InvWork::outputs`). On the owner's descriptor that work is about **47 % of the whole
analysis's CPU time** (measured: 296 of 630 CPU-seconds for 190 s at 48 kHz, 1024 bars).

The lane takes exactly that inverse for whole blocks, for 32 kernels of one tap length and both
channels at a time (64 pairs a block). Everything else stays where it was, in the same order:
kernel spectra (`Morlet::new` + forward transform), the direct edge frames, any block before or
after the device's run (in practice the partial last ones), the `2·(1/n)·y` scaling, the stereo mix
before magnitudes, the norms, and the per-bar member folds and final `exp`.

The shader evaluates rustfft's own AVX graph for this size,
`MixedRadix8xn(MixedRadix8xn(Butterfly512))`, with the product fused into the first pass and only
the requested leaf outputs, using that plan's stored coefficients (read back by
`rustfft::ExactGpuWords32768`, vendor patch) and the same operations in the same order: IEEE
round-to-nearest, no fused multiply-add, no flush-to-zero. Its outputs are the CPU's bits.

## Provenance of the shader

`src/spectrum/aslt_exact_gpu.ptx` is derived by `tools/aslt_exact_gpu/derive_ptx.py` from the
frozen research PTX `integratable_161_20261103/kernel.ptx`
(SHA-256 `8e25fd4302b058b4f595aa220fef70a171978db7badc861905344366a6238c9f`):

- kept verbatim: the module header and the four entries launched (`root8`, `inner8`,
  `whole_phase1_meta`, `whole_phase2_meta`); the legacy `whole_phase1/2` are dropped, which halves
  what the driver compiles on first use (883 KB → 453 KB);
- every floating-point instruction is unchanged, which the script asserts entry by entry;
- integer addressing only: the metadata offset and leaf-input offset the research fixed for 78
  outputs a block become placeholders filled from `aslt_exact_gpu.rs`'s layout constants (one
  source of truth, `MAX_M = 101`), and the hop between requested outputs, an immediate 267 (48 kHz
  at 180 fps), is read from the metadata instead.

Derived file SHA-256: `4c5281ed08b694669604e70f44c2a16f34b73f88c907d8c055a7088570684fbd`.

## Admission (everything else is the unchanged CPU route)

- Windows x86-64, a CPU with AVX and FMA (the plan the shader mirrors), `nvcuda.dll` present,
  device 0 of compute capability 7.5 exactly (Turing: the GeForce RTX 20 and GTX 16 series,
  among others). The results here were measured on one card, the owner's RTX 2060 SUPER; other
  7.5 devices are admitted by class, not because each was tested.
- A joint (stereo) analysis with at least two analysis workers. Mono never qualifies.
- The analysis rate and hop the analysis actually runs at, not the file's format. At 180 fps,
  rates above about 69 kHz never qualify (no 2^15 block owns 64 frames): today that is 96 kHz
  files and DSD at its default 88.2/96 kHz decimation. A DSD analysis decimated to 44.1 or
  48 kHz is judged like PCM at that rate.
- The 2^15 pass of a size-sequential group, its block sets built.
- Chunks of 32 bars whose members in the pass all take the shared-block route with the same
  sequence of tap counts, and whose interior blocks own 64–101 frames (every 2^15 kernel at
  44.1 and 48 kHz with the 180 fps hop). Remainders and other bars are the cores'.
- A bank's host memory within 256 MiB, checked before anything is allocated: per kernel its
  wavelet's three tables (`re`, `im`, `env`), its spectrum, its two complex columns and the three
  magnitude planes made from them, plus one inverse transform's buffer, scratch and pruned
  workspace. The estimate takes bounds, not the admitted values (a 2^15 block is only chosen for
  kernels of at most 16,384 taps; the workspace is sized for any admitted output count), in
  checked arithmetic: 43.7 MiB for a 190 s track. This budget alone would allow about 26 minutes
  at 180 fps, but that is only this formula's ceiling, not a track length the lane supports: the
  pass needs its 2^15 block sets first, and the group's shared-set budget (512 MiB, 32 bytes per
  signal sample for one size of a stereo analysis) admits none above about 349.5 s at 48 kHz or
  380.4 s at 44.1 kHz. In practice the lane can assist tracks up to about six minutes; longer
  ones take the CPU route.

## Scheduling

One pool thread hosts the lane and takes chunks from the front of a queue; the other workers run
every other bar and then take chunks from the back, so neither side waits while there is work and
the analysis never uses more threads than the chosen worker count. A chunk is one bank per member
position; each bar's members fold in their order and a bar is finished once. All CUDA state lives on
the lane's own thread and is retired when the pass ends.

## Failure, cancellation, retirement, calibration

- The first bank of every analysis is checked: its first device block is recomputed on the cores
  for all 64 pairs and compared bit for bit before anything is used.
- A bank that fails or disagrees is recomputed on the cores from its members (it was never folded);
  the lane is not used again for that analysis and the chunks it had not started go to the cores.
- Cancellation: while the lane's thread waits for the device it checks the analysis about every
  20 ms (a blocking wait with a timeout, not a spin), because once the cores have run out of work
  nothing else does. 20 ms is how often it looks, not how long a stop takes: once it sees the
  stop it asks the device to stop at its next block boundary, after the block in progress (its
  copies and kernels cannot be interrupted, about a millisecond on the owner's card); the host's
  own work between banks and thread scheduling add their latency. The reply is still awaited,
  and a bank that comes back after the stop is neither folded nor finished, so it never becomes
  progress. The pass returns nothing, as before.
- Retirement: when the pass ends the lane's thread synchronises, frees each buffer, unloads the
  module and destroys the context, attempting every step whatever the others return, and is
  joined. A failed step or a panic on that thread is reported in the log
  (`exact GPU lane retired with an error: …`), and a setup that fails reports its cleanup and
  thread outcome with its own error. Dropping a lane that was not retired (an unwinding analysis)
  does the same unchecked.
- Calibration: a group the lane was asked to start in is not recorded as a cores time in the
  legacy GPU calibration, whether the lane helped, failed at its first bank, or never came up:
  that time holds the lane's setup and a bank or its replay. Its routing decision is retired with
  the group, its exploration attempt charged. (The owner's default group, 2^15, is below the
  calibration's 2^17 floor; a longer window's 2^17 group with a 2^15 pass is the case this covers.)

Tests (need the device; `cargo test --release exact_gpu -- --ignored --test-threads=1`):
`a_bank_is_the_cpu_routes_members_bit_for_bit` (48 and 44.1 kHz, first and last member positions),
`an_analysis_with_the_lane_is_the_analysis_without_it` (whole joint analyses, every plane),
`two_workers_and_mono_analyses_are_unchanged`, `a_failed_bank_is_recomputed_on_the_cores`,
`a_bank_the_check_rejects_is_recomputed_on_the_cores`,
`a_cancelled_analysis_returns_nothing_and_the_next_is_whole`,
`a_cancel_after_the_cores_drain_stops_the_bank_in_flight` (a forced schedule: the device is held
inside its first bank until the cores have drained, the analysis is stopped, and the device must
stop at the held block), `a_group_the_lane_was_tried_in_is_not_a_cores_sample` (first-block failure,
check rejection and a setup panic, through the live calibration), and
`a_failed_cleanup_and_a_panicking_lane_are_reported`. `the_bank_estimate_bounds_every_admitted_kernel`
needs no device.

## Measured (owner's i7-9700F, RTX 2060 SUPER, 6 analysis workers)

The 190 s records below were taken on 2026-11-04 with the build at `b424014`/`cae5e02`, before
the review's corrections. The corrections change the lane's lifecycle, accounting and wording, not
the arithmetic; the scheduling changes only by the 20 ms poll in the lane's wait and one more
cancellation check per bank. They were not rerun; one 60 s production pair was, on the corrected
build (below).

Whole-track joint pre-process of generated 190 s stereo PCM at the owner's descriptor (n 9, q 2,
1024 bars, 180 fps, Blend 0.5, window 0.25 s), `r10_bench` (`MOOSIK_R10_EXACT`), census timers
off, legacy WGPU route off, the driver's compile cache warm. Outputs compared by hash of the
production u16 codes of every plane.

| | wall | process CPU | peak commit | planes |
|---|---:|---:|---:|---|
| 48 kHz, cores only (runs 1, 4) | 111.85 s, 108.32 s | 631.6 s, 611.2 s | 633 MB | `cd448171…` |
| 48 kHz, lane on (runs 2, 3) | 89.43 s, 90.77 s | 485.1 s, 484.7 s | 895 MB | identical |
| 44.1 kHz, cores only | 97.62 s | 554.9 s | 603 MB | `85b4ab09…` |
| 44.1 kHz, lane on | 83.30 s | 456.8 s | 866 MB | identical |

So **20.0 % and 16.2 % less wall time per 48 kHz pair** (A/B/B/A; 18.2 % on the combined times)
and **14.7 % at 44.1 kHz** (one pair), every output code the same. At 48 kHz the lane took 11 of
13 chunks (2112 of the pass's 3868 members), at 44.1 kHz 11 of 19 (the cores took the rest from
the back of the queue). The whole-analysis f32 planes are compared bit for bit by the 30 s tests
above.

The production path (`preprocess_file`: decode, analysis, cache publication), generated 60 s
48 kHz stereo WAV: 32.75 s cores only, 27.62 s with the lane (15.7 %), planes identical. The two
publications' files were compared byte for byte at the time: `.mix` and `.lr` differed in exactly
the 16 bytes at offsets 16–31 (the publication's random generation token), the `.head` only in
its token text. Those files were deleted with the run's temporary directory and cannot be rehashed
now; the comparison's command and output are kept with the review evidence. The bench's
`cache_bytes`/`sidecar_bytes` fields read 0 there: they are legacy fields that do not measure the
v2 cache files.

The same pair on the corrected build (`129703c`, 2026-11-05, one pair): 32.46 s cores only,
26.42 s with the lane, planes identical (and the same hashes as the 2026-11-04 pair). Both
publications are kept with the review evidence; with the token normalized (bytes 16–31 of `.mix`
and `.lr` zeroed, the `.head` token text zeroed) all three files hash the same.

The census behind the choice (`r10_bench` census arm and `r11_chunk_census`, 48 kHz, 1024 bars,
190 s, cores only): the 2^15 shared-block route is 295.7 of 630 CPU-seconds (47 %), 98.9 % of it
the inverse; six cap-derived tap counts (8251/9001/9751/10501/11251/12001) carry 240.8 s; the
research lane's single tap count (12001) only 40.7 s (6.5 %), which is why it had to be broadened.

Cost (`what_the_lane_holds`, process counters, the test alone in a fresh process; 2026-11-04 and
again on the corrected build): setup 407 ms the first time a driver compiles the shader, 60–100 ms
after (its cache); device buffers 42.7 MiB; while the lane is up the process's commit is about
+230 MiB and its working set about +104 MiB over its level before the lane, a bank +3 MiB on top.
After the lane and the test's inputs are dropped, the process stays about +139 MiB of commit and
+16 MiB of working set over that level: an observed delta, not attributed byte for byte (it may be
the driver's process state after `cuInit`, or allocator or planner retention). In a process that
had already run nine other lane tests the same reading was +220 / +90 MiB while up and +157 /
+24 MiB after: the deltas depend on what the process did before.

## Limits

- Only the 2^15 pass of stereo analyses; mono, other sizes, other devices and other compute
  capabilities take the CPU route.
- Bars whose tap sequence in the pass is shared by fewer than 32 bars stay on the cores (at 48 kHz,
  1024 bars, 64.5 % of the pass's members fall in full chunks; at 44.1 kHz, 94 %).
- The lane competes with anything else using the card (a music video, a local translation model);
  that contention, and playback and spectrum fluency in the app during an assisted analysis, are
  not measured.
- The process keeps the observed delta above after the first use (see Cost).
- Track length: about six minutes at most in practice (see Admission); 15–20 % is what the
  measured examples gained, not a promise for every track.
- The host side of a bank (kernel spectra, edges, the cores' blocks, norms) runs between device
  runs, not beside them; pipelining it is a measured opportunity (the cores took 8 of 19 chunks at
  44.1 kHz), not done here.
