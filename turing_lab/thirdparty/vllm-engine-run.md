# vLLM engine run on TU102 (plan/0006, 2026-09-04/05)

Single card (CUDA_VISIBLE_DEVICES=1), vLLM V1 (this fork), eager mode:

- model: /opt/models/Intel-Qwen3.6-27B-int4-AutoRound (W4A16 AutoRound)
- dtype float16, max_model_len 2048, gpu_memory_utilization 0.92
- warmup + 4 sequences x 64 tokens, temperature 0
- **VLLM_ENGINE_OK tokens=256 time=10.56s tok/s=24.2**

The run exercises the sm_75 stack end to end: the Turing W4A16 kernel
selection (plan/0002 backend) for the GEMMs and the V1 attention path.

## TP=2, both cards (2026-09-05)

`tensor_parallel_size=2` over the NVLink-bridged pair, eager mode, fp16,
max_model_len 4096, gpu_memory_utilization 0.90, clocks locked 1455 MHz.
Environment: CUDA_HOME=/opt/cuda, FLASHINFER_DISABLE_VERSION_CHECK=1 (the
flashinfer-cubin 0.6.17/0.6.18 wheel mismatch; harmless on sm_75).

Attention backend for this run (from the engine log): **TRITON_ATTN**
auto-selected — "Using TRITON_ATTN attention backend out of potential
backends: ['TRITON_ATTN', 'FLEX_ATTENTION']". On capability (7,5)
FLASH_ATTN and FLASHINFER are gated >= (8,0), so the quilt-routed bridge
kernel does not receive engine traffic yet (see the bridge-backend item
in plan/0006). Numbers below are the honest baseline for that A/B.

- **TP2_SHORT: 8 sequences x 64 tokens, 6.68 s -> 76.7 tok/s aggregate**
- TP2_CTX ctx=512, single sequence, 32 decoded tokens incl. prefill:
  3.24 s -> 9.9 tok/s end-to-end
- TP2_CTX ctx=2048: 5.03 s -> 6.4 tok/s end-to-end
- **VLLM_TP2_OK**

## Bridge attention backend: routing evidence + A/B (2026-09-05)

`BridgeAttentionBackend` (vllm/v1/attention/backends/bridge_attn.py) is
the plan's thin dispatch adapter: on capability (7,5), fp16, head_dim
{64,128}, causal decoder attention, the V1 selector auto-selects
BRIDGE_ATTN (registry + platform priorities; sm_80+ never resolves
there) and every layer forward calls the real upstream
`flash_attn.flash_attn_func` — whose quilt v3 sm_75 route executes the
bridge kernel, seqlen_q != seqlen_k bottom-right causal included
(chunked prefill with prefix, padded lengths, single-token decode).
The 27B model's head_dim 256 is outside the kernel's current scope:
the backend declines and the selector falls through to TRITON_ATTN
(the boundary works as designed; d=256 needs the 32-row KV tile +
Q-in-registers restructuring, recorded in the plan backlog).

Evidence (clocks locked 1455 MHz, eager, GPU 1):
- engine log: "Using BRIDGE_ATTN attention backend out of potential
  backends: ['BRIDGE_ATTN', 'TRITON_ATTN', 'FLEX_ATTENTION']"
- torch profiler trace committed as
  `vllm-bridge-routing-trace.json.gz` (Qwen2.5-1.5B-Instruct, 8 decode
  steps): `bridge_fwd_kernel<128>` fires 224 times = 28 layers x 8
  steps; the same trace's elementwise copy/index kernels are the
  per-layer page-gather cost of this evidence-first backend.

A/B, Qwen2.5-1.5B-Instruct (head_dim 128, GQA 2:1), same protocol
both arms (`vllm_bridge_ab.py`, `vllm_tp2_ab.py`):

| workload | BRIDGE | TRITON | ratio |
|---|---|---|---|
| 1 card, 4x64 tok short | 56.1 tok/s | 124.5 tok/s | 0.45 |
| 1 card, ctx 512 -> 32 tok | 30.1 tok/s | 42.6 tok/s | 0.71 |
| 1 card, ctx 2048 -> 32 tok | 28.0 tok/s | 28.0 tok/s | 1.00 |
| TP=2, 8x64 tok short | 24.2 tok/s | 212.4 tok/s | 0.11 |
| TP=2, ctx 512 -> 32 tok | 19.0 tok/s | 41.4 tok/s | 0.46 |
| TP=2, ctx 2048 -> 32 tok | 14.4 tok/s | 33.1 tok/s | 0.44 |

Honest reading: routing is proven and correctness is oracle-gated, but
the backend is evidence-first, not performance-first — per-(layer,
request) python dispatch plus the paged-KV gather copies dominate at
small context (they scale with batch x layers x ranks), and decode
pays a zero-padded 64-row Q tile. Parity at ctx 2048 single-card shows
the kernel itself is not the short-context bottleneck. Backlog (plan
README): varlen/paged bridge kernel consuming the block table
directly (removes gather + python loop), d=256 tile restructuring,
ldmatrix fragment loads.
