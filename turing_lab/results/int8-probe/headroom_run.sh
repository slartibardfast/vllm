#!/usr/bin/env bash
# plan/0009 #int8-headroom: the W8A8 compressed-tensors fixture vs the
# W4A16 incumbent on identical rows, champion config, medians across
# fresh restarts. Greedy texts compared against the incumbent arm for
# a sanity check (different quant = not expected to match; the rows
# are the measurement, the texts are recorded not gated).
# Usage: bash headroom_run.sh <model-dir> <tag> [restarts]
set -euo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"
cd "$HERE/../../.."
MODEL="${1:?model dir}"
TAG="${2:?tag}"
R="${3:-3}"
OUT="$HERE/headroom-$TAG"
mkdir -p "$OUT"
for i in $(seq 1 "$R"); do
  COMMITTED_NO_EAGER=1 \
  BRIDGE_PAGED_DECODE=split \
  BRIDGE_PPS=2 \
  VLLM_ATTENTION_BACKEND=BRIDGE_ATTN \
  FLASHINFER_DISABLE_VERSION_CHECK=1 \
  CUDA_HOME=/opt/cuda \
  .venv/bin/python turing_lab/thirdparty/vllm_macro_gate.py \
      --arm BRIDGE_ATTN 2 "$MODEL" > "$OUT/arm-r$i.log" 2>&1 || \
      { echo "restart $i FAILED"; tail -5 "$OUT/arm-r$i.log"; exit 1; }
  grep "SPLIT\[" "$OUT/arm-r$i.log" | head -3
  echo "restart $i done"
done
