# High-accuracy GPU for the superlet pre-process (1.6.2, experimental)

Off by default: **Pre-process settings → GPU → "High-accuracy GPU (experimental)"**.
With it on, the whole superlet analysis of a stereo track runs on an NVIDIA
graphics card instead of the cores. It is a separate route with its own
numerics, not a faster way to the same bits. For that, see the
[exact NVIDIA assist](ASLT_EXACT_GPU.md).

## What it buys

Measured on the owner's settings: 1024 bars, 180 fps, Superlet q 2.0, a
0.25 s window, 9 wavelets, spread 0.5, blend 0.5. The machine is an i7-9700F
with an RTX 2060 SUPER.

| Track | High-accuracy GPU | Standard on six cores |
|---|---|---|
| 3:36 at 48 kHz | about 12.5 s, analysed and saved | about two minutes |
| 3:47 at 192 kHz | about 41 s | about 14 minutes |

Most of that time is the card's own arithmetic. While the card works, the
cores prepare the next piece of work and fold in the last one, so they never
hold it up.

The first track of a session at each rate and setting also plans the work
before the card starts: about 1.2 s at 48 kHz and 5 s at 192 kHz. Later tracks
reuse that plan.

## How close it is

The reference is a higher-precision calculation of the same analysis, done
directly in f64. It was checked on five songs (44.1 to 192 kHz) and seven of
the settings this route takes:

- **High-accuracy GPU:** within one display step (0.0195 dB) of the reference
  everywhere, Diff and the edges included. Made-up stress signals have put it
  two steps off in Diff.
- **Standard:** can be several steps off at a handful of points.

The two never match bit for bit. High-accuracy GPU therefore keeps its own
cache, and a track already analysed with Standard stays Standard until you
press **Reanalyse with High-accuracy GPU**. The Standard cache is kept.

## What it needs

Anything outside the list below is analysed with Standard, and the reason is
shown under the switch.

- An NVIDIA card of compute capability 7.5 or later (Turing, GeForce RTX 20
  and GTX 16 series, and newer) and its driver. Only the driver's
  `nvcuda.dll` is used: no CUDA toolkit and no extra DLLs.
- Superlet bar mapping, 256 to 2048 bars, 100 to 480 fps, a window up to
  1.2 s.
- With two or more wavelets, a spread of 0.1 or more, or 0.3 or more above
  48 kHz. Below that, most of the work cannot use the card's fast path and
  the route would be slower than Standard.
- A stereo track at 44.1, 48, 88.2, 96, 176.4 or 192 kHz that is not too long:
  5:10 at 1024 bars and 180 fps, shorter with more bars or frames.

## What it costs

- About 0.6 GB of graphics memory while it runs; about 1.7 GB for a 3:47
  track at 192 kHz.
- More system memory than Standard.
- It has been tested on one card only, an RTX 2060 SUPER.

## When something goes wrong

- If the route fails on a track, that track is analysed once with Standard.
- After a graphics card error, the route stays off until Moosik restarts.
- Cancelling (another track, or Stop) ends the pass and frees the card's
  memory before anything else starts.

With the exact NVIDIA assist also on: a track analysed here does not use it,
and a track analysed with Standard still does.
