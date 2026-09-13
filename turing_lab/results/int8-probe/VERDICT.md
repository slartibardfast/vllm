# INT8 probe verdict (2026-09-13, plan/0008): quantize YES, load NO (loader gap)

- question: what does an INT8 path buy on TU102 vs the W4A16 incumbent?
- W8A8 quantization (autoround, fixture grade iters 40 / nsamples 48 /
  g128, 4B fixture): SUCCEEDS — 248/347 layers quantized (visual towers
  and lm_head excluded by the recipe), 3167 s tuning, peak 17.6 GB RAM /
  19.5 GB VRAM. Checkpoint:
  /opt/models/Qwen3.5-4B-W8A8-AR-f16/Qwen3.5-4B-w8g128/.
- engine load: FAILS in the lane's loader —
  `AttributeError: MergedColumnParallelLinear has no attribute
  'orig_layer'` (qwen3_5.py load_weights path): the AutoRound int8
  method does not cover this model's merged parallel linears. This is
  a checkpoint-to-loader integration gap, not a compute-capability
  refusal; no INT8 kernel was ever reached.
- probe invocations (recorded in the log): default backend selection
  refused sm_75 outright (FA2 requires cc >= 8) — stock loads on this
  host must pin TRITON_ATTN; a stdin-heredoc driver breaks vLLM worker
  spawn (workers re-execute `<stdin>`) — load attempts need a real
  script file with a __main__ guard.
- disposition: INT8 is unreachable on this stack today without loader
  work (the merged-linear int8 integration); the quantized fixture is
  on disk and the quantization cost is banked, so a future window that
  closes the loader gap can measure the headroom question directly.
  The W4A16 incumbent (51-57 tok/s short / 41.3 champion-path bridge)
  stands.
