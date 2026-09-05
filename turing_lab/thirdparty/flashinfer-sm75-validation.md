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
