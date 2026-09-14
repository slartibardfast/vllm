#!/usr/bin/env bash
# plan/0009 #int8-loader: the W8A8 re-export (engine-native format).
#
# Why: the banked fixture (/opt/models/Qwen3.5-4B-W8A8-AR-f16) is NOT
# int8. With --format auto_round and activation quant, autoround
# 0.15.0 exports the triton-act wrapper IR: every language linear is
# saved as orig_layer.weight in BF16 (verified in the safetensors
# headers; no scale or zero-point tensors exist anywhere in the
# index). The engine-side orig_layer AttributeError was that wrapper
# naming reaching the weight walker, not a qwen3_5 loader gap, and
# the tuning result (248/347 layers) never left the process.
#
# The fix is export-side: --format llm_compressor packs int8 weights
# plus group scales plus activation scales as compressed-tensors
# (auto_round/export/export_to_llmcompressor/export.py,
# construct_ct_scheme: weights num_bits 8 type int, strategy group;
# input_activations num_bits 8), which vLLM loads natively with no
# model-file patch. Fixture-grade tuning as before (iters 40 /
# nsamples 48), g128 symmetric, seed 42.
set -euo pipefail
CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0} \
~/tools/autoround-venv/bin/auto-round quantize \
  --model_name /opt/models/Qwen3.5-4B \
  --bits 8 --data_type int --group_size -1 \
  --act_bits 8 --act_data_type int \
  --model_dtype float16 --scale_dtype fp16 \
  --format llm_compressor \
  --iters 40 --nsamples 48 --seqlen 2048 --seed 42 \
  --output_dir /opt/models/Qwen3.5-4B-W8A8-CT-f16 \
  --device_map cuda:0
