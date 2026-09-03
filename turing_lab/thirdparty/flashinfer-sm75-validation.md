# FlashInfer sm_75 validation (plan/0006, 2026-09-03)

flashinfer 0.6.17 (venv pin; 0.6.18 is current upstream):

- **Prefill: PASS.** `single_prefill_with_kv_cache(q, k, v, causal=True)`
  at d=128, s=512: max_err 0.00094 vs the fp64 reference — the JIT
  built and ran for compute_75 (cache: ~/.cache/flashinfer/0.6.17/75).
- **Decode: JIT compile failure** on the batch-decode op for sm_75
  (nvcc error mid-build of batch_decode_jit_binding.cu) — consistent
  with the vLLM-tracked upstream breakage (#3620, fix PR #3621); our
  invocation may also mis-order plan() args. Open: retry after
  bumping to 0.6.18 (carries the fix) and verify plan() arg order.
- Consequence for vLLM V1 on sm_75: prefill-class FlashInfer APIs
  work; the default decode path rides the bridge-routed FlashAttention
  quilt or the Triton fallback until 0.6.18 is validated.
