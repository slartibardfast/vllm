# vLLM engine run on TU102 (plan/0006, 2026-09-04)

Single card (CUDA_VISIBLE_DEVICES=1), vLLM V1 (this fork), eager mode:

- model: /opt/models/Intel-Qwen3.6-27B-int4-AutoRound (W4A16 AutoRound)
- dtype float16, max_model_len 2048, gpu_memory_utilization 0.92
- warmup + 4 sequences x 64 tokens, temperature 0
- **VLLM_ENGINE_OK tokens=256 time=10.56s tok/s=24.2**

The run exercises the sm_75 stack end to end: the Turing W4A16 kernel
selection (plan/0002 backend) for the GEMMs and the V1 attention path.
Remaining variant: tensor_parallel_size=2 across both cards (weights
~14 GB fit a single card, so TP=2 is a throughput variant, not a
necessity).
