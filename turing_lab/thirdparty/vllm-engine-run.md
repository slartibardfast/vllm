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
