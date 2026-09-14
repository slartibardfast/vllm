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

## Headroom verdict (2026-09-14, plan/0009 #int8-headroom): NO HEADROOM - W8A8 is 15-21 pct slower

The full chain closed end-to-end this window: the llm_compressor
re-export (requantize_llmcompressor.sh, adjusted to the format's
requirements: sym dynamic W8A8, per-channel weights - grouped g128
is rejected at the format check) produced a real compressed-tensors
checkpoint (int8 weights, fp16 per-channel scales, dynamic
token-level int8 activations) at
/opt/models/Qwen3.5-4B-W8A8-CT-f16; the ENGINE LOADS IT NATIVELY on
the champion configuration - the 0008 loader gap is closed, no model
patch required.

Measurement, 3 fresh restarts per arm, identical rows, champion
config (BRIDGE_ATTN, paged-split PPS2, graphs), zero preemptions:

- short_decode: W8A8 426.9 vs W4A16 538.1 (ratio 0.79)
- ctx512_decode: 80.2 vs 94.0 (0.85)
- ctx2048_decode: 75.9 vs 89.2 (0.85)

(headroom-summary.json; headroom-w8a8ct/ and headroom-w4a16/.)

VERDICT: the int8 tensor rate (203 TOPS, 2x fp16) buys nothing in
the decode regime because decode is weight-bandwidth-bound and W8
doubles the weight bytes; the narrower activation path does not pay
for them. INT8 W8A8 is now a MEASURED lane on this silicon: closed,
negative. The W4A16 incumbent stands.
