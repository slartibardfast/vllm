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

## Correction (2026-09-13, plan/0009 #int8-loader): the fixture was never int8

Re-examined the banked checkpoint against the export path. The
index holds only orig_layer.weight tensors for every language linear
and the safetensors headers read BF16 (sampled q_proj and gate_proj);
no weight_scale, no zero-point, nothing quantized exists anywhere in
the file. With activation quant, autoround 0.15.0's auto_round format
exports the triton-act wrapper IR (export/export_to_autoround/
export.py: act_bits <= 8 routes to pack_qact_layer, whose QuantLinear
comes from auto_round_extension, a runtime this stack does not have)
and what save_pretrained captured is the wrapper's bf16 original.
The tuning itself (248/347 layers) never left the process.

So the orig_layer AttributeError was wrapper naming reaching the
weight walker, not a qwen3_5 loader gap, and the "quantize YES /
load NO" split above was wrong at its root: there was nothing
quantized to load. Disposition updated: the fix is export-side.
requantize_llmcompressor.sh (beside this verdict) re-runs the
fixture-grade recipe with --format llm_compressor, which packs int8
weights, group scales and activation scales as compressed-tensors
that vLLM loads natively (no model-file patch). The headroom
measurement (plan/0009 #int8-headroom) runs against that export.
