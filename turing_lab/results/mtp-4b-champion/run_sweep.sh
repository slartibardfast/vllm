#!/usr/bin/env bash
# plan/0009 #mtp-4b: the 4B fixture on the CHAMPION path (decode
# paged-split PPS2, verify/prefill gather, graphs) - does the 4B's
# stock-path K2 compounding (2.12x, TP2) transfer to the champion
# configuration? 3 restarts of the ksweep driver (base + K1..4).
# Usage: bash run_sweep.sh
set -euo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"
cd "$HERE/../../.."   # sm75-marlin worktree root
MODEL=/opt/models/Qwen3.5-4B-W4A16-AR-f16/Qwen3.5-4B-w4g128

for r in 1 2 3; do
  KSWEEP_TP=2 \
  VLLM_ATTENTION_BACKEND=BRIDGE_ATTN \
  BRIDGE_PAGED_DECODE=split \
  BRIDGE_PPS=2 \
  FLASHINFER_DISABLE_VERSION_CHECK=1 \
  CUDA_HOME=/opt/cuda \
  .venv/bin/python turing_lab/thirdparty/vllm_ksweep.py "$MODEL" \
      > "$HERE/sweep-r$r.log" 2>&1
  for k in 0 1 2 3 4; do
    [ -f turing_lab/thirdparty/ksweep-k$k.json ] && \
      cp turing_lab/thirdparty/ksweep-k$k.json "$HERE/k${k}-r${r}.json"
  done
  cp turing_lab/thirdparty/ksweep-report.json "$HERE/report-r$r.json" 2>/dev/null || true
  echo "restart $r done"
done
