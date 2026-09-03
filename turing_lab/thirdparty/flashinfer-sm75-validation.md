# FlashInfer sm_75 validation (plan/0006; updated 2026-09-04, v0.6.18)

- **Prefill: PASS** (0.6.17 and 0.6.18). single_prefill_with_kv_cache
  causal d=128: max_err 0.00094.
- **Decode, single request: PASS** (0.6.18). b=1, h=8, d=128, kv=256:
  max_err 0.00011. The 0.6.17 JIT compile failure is GONE in 0.6.18
  (the #3621 fix landed).
- **Decode, multi request: FAULTS** (0.6.18, both the classic and the
  use_tensor_cores variants). b>=2 OOB-reads inside
  BatchDecodeWithPagedKVCacheKernel on compute_75 (compute-sanitizer:
  invalid __global__ 16B reads; single-request identical shapes pass,
  so the plan()/pool wiring is correct — this is an upstream kernel
  bug on sm_75, the #3620 family surviving #3621).
- venv note: flashinfer-cubin has no 0.6.18 wheel; version check
  bypassed via FLASHINFER_DISABLE_VERSION_CHECK=1 (harmless on sm_75,
  everything JITs from source).

Single-request decode throughput (median of 50 event-timed forward
calls, clocks locked 1455 MHz, 2026-09-04): kv=1024: 49.2 us ->
20.3 tok/s; kv=4096: 73.6 us -> 13.6 tok/s; kv=16384: 155.7 us ->
6.4 tok/s. This is the attention step alone (per-call wall time
including wrapper overhead); a full decode step adds the MLP GEMMs.

Consequence: vLLM on sm_75 must not use FlashInfer decode until
upstream fixes the multi-request kernel; our fork's backend selection
already prefers the FlashAttention-routed bridge path, so this is a
recorded upstream bug, not a local gap. File/link the upstream issue
from the quilt repo when convenient.
