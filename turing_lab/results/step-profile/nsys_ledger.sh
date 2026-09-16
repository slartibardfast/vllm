#!/usr/bin/env bash
set -euo pipefail
W=/opt/repo/agentic-vllm/software/vllm/sm75-marlin
cd "$W"
BACKEND="$1"
OUT=turing_lab/results/step-profile/nsys-$BACKEND
mkdir -p "$OUT"
env COMMITTED_NO_EAGER=1 \
    BRIDGE_PAGED_DECODE=split BRIDGE_PPS=2 \
    VLLM_ATTENTION_BACKEND=$BACKEND \
    FLASHINFER_DISABLE_VERSION_CHECK=1 CUDA_HOME=/opt/cuda \
    nsys profile -o "$OUT/step" --force-overwrite true \
      --trace=cuda --sample=none --cpuctxsw=none \
    $W/.venv/bin/python $W/turing_lab/thirdparty/nsys_step_child.py
nsys stats --report cuda_gpu_kern_sum "$OUT/step.nsys-rep" > "$OUT/kern-sum.txt"
echo "ledger at $OUT"
