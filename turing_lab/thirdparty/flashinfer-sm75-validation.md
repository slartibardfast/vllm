# FlashInfer sm_75 validation (plan/0006; updated 2026-09-05, v0.6.18)

- **Prefill: PASS** (0.6.17 and 0.6.18). single_prefill_with_kv_cache
  causal d=128: max_err 0.00094.
- **Decode, single request: PASS** (0.6.18). b=1, h=8, d=128, kv=256:
  max_err 0.00014 (re-confirmed 2026-09-05). The 0.6.17 JIT compile
  failure is GONE in 0.6.18 (the #3621 fix landed).
- **Decode, multi request: PASS** (0.6.18) — the earlier FAULT claim is
  RETRACTED (our harness bug, same class as the retracted NaN finding).
  The paged-cache contract is `(num_pages, 2, page_size, h, d)`; our
  earlier scripts passed `(b, 2, kvlen, h, d)` with page_size=1, so at
  b>=2 the kernel indexed past the page dimension and hit an illegal
  address (the compute-sanitizer 16B OOB reads were real, but caused by
  our malformed tensor). With the contract-correct layout, 2026-09-05:
  b=2/4/8 mixed lengths (1024..8) and b=16 x 2048 pass on BOTH the
  classic and use_tensor_cores variants, max_err 0.00006..0.00072.
  Nothing to file upstream; vLLM on sm_75 may use FlashInfer decode.
- venv note: flashinfer-cubin has no 0.6.18 wheel; version check
  bypassed via FLASHINFER_DISABLE_VERSION_CHECK=1 (harmless on sm_75,
  everything JITs from source).

Decode throughput (median of 50 event-timed forward calls, clocks
locked 1455 MHz, contract-correct layout, 2026-09-05): b=1:
kv=1024: 50.5 us -> 19.8 tok/s; kv=4096: 73.7 us -> 13.6 tok/s;
kv=16384: 156.7 us -> 6.4 tok/s (matches the earlier record within
noise, so those timings were unaffected by the layout bug). b=8
uniform: kv=1024: 100.7 us -> 79.5 tok/s aggregate; kv=4096:
266.9 us -> 30.0 tok/s aggregate. This is the attention step alone
(per-call wall time including wrapper overhead); a full decode step
adds the MLP GEMMs.

vLLM selection facts (verified in code, 2026-09-05): on capability
(7,5) the V1 selector auto-selects TRITON_ATTN — FLASH_ATTN is gated
`capability >= (8,0)` (vllm/v1/attention/backends/flash_attn.py) and
FLASHINFER likewise (citing flashinfer #3620); forcing FLASH_ATTN
would still import the vendored `vllm_flash_attn`, not the
quilt-patched upstream package. Engine traffic reaches the bridge
only through the fork's bridge attention backend (plan/0006 item in
flight); the FlashInfer multi-request pass above removes the decode
blocker on their side.

## d=256 baseline - plan/0007 step 0.1 (2026-09-06, flashinfer 0.6.18)

Harness: fi_d256_baseline.py (Qwen3.5/3.8-27B full-attention shape:
h_q=24, h_kv=4, d=256, fp16, GQA 6:1, page_size=1, NHD paged).

- Decode: CORRECT. max_err 0.00011-0.00049 vs torch reference across
  uniform, mixed and long kv-length batches. Baseline tg (b=1):
  16.2 tok/s at kv=1024, 11.8 at kv=4096, 5.2 at kv=16384; 66.2
  aggregate at b=8, kv=1024. Medians of 3, RTX 6000, 1455 MHz.
- Prefill (BatchPrefillWithPagedKVCacheWrapper): SILENTLY WRONG in every
  probed configuration - MHA and GQA, causal and non-causal, d=128 and
  d=256, ctx 8 to 2048. max_err up to 4.24 against a torch einsum
  reference and torch SDPA, which agree with each other. The wrapper
  launches and returns tensors (no launch failure), so this is the
  dangerous failure class: the research's "no SM75 CI machine" (issue
  #1648) is exactly what silent wrongness looks like. DISQUALIFIED by
  plan/0007 gate 0 (no silent wrongness) - not usable as a baseline at
  any head dim; a differential test against torch must gate any future
  FlashInfer-on-sm75 attempt (upstream PR #3621 remains
  unmerged and the fix era in 0.6.18 does not cover this path).

Reported into plan/0007 as the step 0 decode baseline and the step 0
prefill blocker.

## FA2-native same-hardware probe (plan/0007 step 0.2, 2026-09-06)

The installed quilt (flash_attn 2.8.3, editable) exposes FA2's own
kernels outside the bridge predicate: the raw binding
`flash_attn_2_cuda.fwd` runs FA2's in-tree sub-sm80 path on this RTX
6000 (the path every upstream Turing issue assumed, never run).

- d=256 fp16 causal: LAUNCH FAILURE - cudaErrorInvalidValue. Consistent
  with the smem analysis: the d=256 fwd template needs exactly 65536 B
  dynamic plus static smem, over Turing's 64 KiB block cap.
- d=128 fp16 causal (GQA 6:1, ctx 256): RUNS but SILENTLY WRONG -
  gate 0 max_err 2.20 vs agreeing torch einsum/SDPA references; wrong in
  every 64-row block (per-block err 2.20/1.27/0.85/0.59, decaying with
  row index - approximation-shaped), every head; FA2's own softmax_lse
  is wrong by 1.14; scale-convention hypotheses (no-scale, doubled,
  log2e-scaled) all ruled out. The defect is inside the sub-sm80 path's
  score computation itself. DISQUALIFIED by gate 0.

Route C verdict for plan/0007: no FA2 or FlashInfer prefill baseline
exists on sm_75 at any head dim; FlashInfer decode stands as the only
third-party same-hardware baseline. The bridge kernel remains the only
gate-passing tensor-core attention on this hardware at d=64/128 - which
is precisely why the sub-80 path was declared dormant in plan/0006 and
the bridge built.
