# rustfft 6.4.1 — Moosik patch (1.5.3)

The opening archive/addition counts below describe the original 1.5.3 patch,
not subsequent extensions.

Exact published archive (copied from the local cargo registry) plus additions only — no existing
line removed or changed (643 added lines under `src/`). Measured in round 10, in production
since round 11.

- `Fft::process_pruned_with_scratch(buffer, scratch, needed, out, work)`: one in-place transform
  of which only the outputs at `needed` are wanted; `out[i]` is the value `process_with_scratch`
  would have left at `buffer[needed[i]]`, bit for bit. Default: full transform, then gather.
- `Fft::process_pruned_product_with_scratch(s, h, …)` and
  `Fft::process_pruned_inplace_product_with_scratch(h, buffer, …)`: the same on the product
  `s[i]·h[i]` (num-complex `MulAssign` operations). Default: product pass, then the above.
- `Fft::prunes()` (default `false`) and `Fft::pruned_work_len(needed)` (default 0).
- `PrunedWork { idx, val }`: caller-owned bookkeeping, sized by `pruned_work_len`, so a pruned
  transform allocates nothing; each pruning level takes its share from the front.
- **One contract, checked before any processing by every implementation** (`check_pruned`,
  `check_product`): `buffer.len() == len`; `scratch.len() >= get_inplace_scratch_len()`;
  `out.len() == needed.len()`; `needed` strictly increasing and `< len`; `work` at least
  `pruned_work_len`; and for the product forms `s.len() == h.len() == len`. (The round-10
  Bluestein product fallback zipped its operands, so a short signal or kernel would have left
  part of the buffer unmultiplied. The analysis's own calls were always equal-length, power-of-two
  and sorted; this is API hardening, not a fix for an observed defect.)
- AVX mixed-radix family (`boilerplate_mixedradix!`): `perform_pruned`,
  `perform_pruned_product`, `perform_pruned_inplace_product`, `pruned_rows` — the unmodified
  column butterflies, then the inner FFT only on rows holding a wanted output, outputs read where
  the transpose would have fetched them. These structs exist only after rustfft's own AVX/FMA
  detection, so the pruned code never runs on a CPU without them.
- `avx_vector.rs`: `ProductBuf`, `InplaceProductBuf` (`AvxArrayMut` adapters whose loads form the
  product; f32 loads via `moveldup/movehdup`, `mul`, pair swap, `mul`, `addsub` — no FMA).
- `BluesteinsAvx`: non-pruning implementations under the same contract (full transform, gather).
- Dispatch in `boilerplate_avx_fft_commondata!`. Every other plan (scalar, SSE, Rader's, the
  butterflies) takes the trait defaults.

## Round 12-E (1.5.3): requested-output Butterfly256/512 leaves

- `Butterfly256Avx` / `Butterfly512Avx` (f32, AVX+FMA, both directions) override
  `process_pruned_with_scratch`: after `check_pruned`, for 1..=4 requested outputs, each
  output is computed by the original primitives (`AvxVector::add`/`sub`/`rotate90`/
  `mul_complex`, `apply_butterfly8_twiddle1/3`, `neg`) on the original operands and constant
  vectors, with the original operand order and FMA use; outputs no request reads, their twiddle
  products, the packed transposes (replaced by storing the live lanes where the transpose puts
  them) and the unread row-pass column sets are skipped. Empty sets, sets of 5+ and
  `Butterfly128Avx` take the full leaf and gather (without a second contract check).
- Equality: see the qualification on `Fft::process_pruned_with_scratch` (NaN payload/sign and
  FP status flags are not part of the contract; finite values, signed zeros, infinities and
  NaN-ness are).
- `moosik-diagnostics` feature: global leaf-call counters (`moosik_diagnostics`), for
  diagnostic test builds only; off by default and in every ordinary or timed build.
- Round 13 (K0): `Butterfly256Avx`/`Butterfly512Avx` report `prunes() == true` (they have the
  requested-output path), so callers that consult it — the analysis's `InvWork::outputs` —
  use it for a standalone 256/512 plan; `Butterfly128Avx` keeps `false`. Arithmetic unchanged.

## Round 13 lane A (1.5.3): live-row column passes

- In a pruned `MixedRadix4xnAvx`/`MixedRadix8xnAvx` transform (both precisions), the column pass
  computes only the rows that hold a requested output when the transform has at least 8192 points
  and at most half the rows do, each row by the
  operations `AvxVector::column_butterfly4/8` use for it, then the same `mul_complex(twiddle, value)`
  and store; every input is still loaded. More live rows, other radices, or a partial-vector remainder
  keep the dense pass. Values are unchanged (the rows not computed are never read by `pruned_rows`).
- `moosik-diagnostics` gains live/dense pass counters (diagnostic builds only).
- R13 closeout (diagnostic builds only): each pruned mixed-radix level can be recorded per thread
  (`level`, `levels_record`, `levels_take`: length, rows, live mask, branch, element size) for
  per-test geometry/branch evidence; recording is opt-in per thread, so analysis pool threads never
  accumulate it. `mixed_radix4` builds a radix-4 plan with the planner's own constructor for tests
  of a geometry the planner does not choose (f64 power-of-two plans are radix-8 chains).

## 1.5.5: requested-output f64 Butterfly256/512 leaves

The f64 scratch-butterfly macro now dispatches requested transforms to
`avx64_pruned.rs`. The 256/512 leaves evaluate the original full graph for
1..=4 requested outputs, using the original AVX/FMA primitives, constants,
operand order and packed-transpose addresses. Existing full functions are
unchanged; 64/128 and empty/5+ requests keep full execution and gather.
No planner or FFT-size selection changes. The same CPU admission and pre-write
argument checks apply. NaN payload/sign and floating-point status flags remain
outside the existing equality contract.

Tests live in `avx64_pruned_tests.rs`. An isolated offline gate at
`research/v155/leaf_gates` mirrors them through the public AVX planner because
the vendor's original dev-dependency wasm-bindgen-test is absent offline.
It includes singleton, mixed-row, product, contract and exhaustive-pair checks
against the unchanged full functions, both directions. This is local modified
source under the upstream MIT/Apache-2.0 terms, not an upstream release.
