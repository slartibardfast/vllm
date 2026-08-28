#!/usr/bin/env bash
# sm_86 reference run for plan/0002 D8 (agentic-vllm Turing Marlin program).
# Self-gating: refuses to run while llama-server holds the GPU, and while
# the graphics clock lock is not at 1665 MHz. It never stops or starts
# llama-server; the operator does that.
set -euo pipefail

TAG=sm86
OUT="$HOME/marlin-bench-$TAG.json"
LOG="$HOME/marlin-bench-$TAG.log"
EXPECT_MHZ=1665

if pgrep -x llama-server >/dev/null; then
  echo "window NOT open: llama-server is running (pids: $(pgrep -x llama-server | tr '\n' ' '))" >&2
  echo "stop llama-server first; this script never touches it." >&2
  exit 1
fi

used=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | head -1)
if [ "$used" -gt 500 ]; then
  echo "window NOT open: GPU already holds ${used} MiB" >&2
  exit 1
fi

mhz=$(nvidia-smi --query-gpu=clocks.current.graphics --format=csv,noheader,nounits | head -1)
if [ "$mhz" -ne "$EXPECT_MHZ" ]; then
  echo "clock lock is ${mhz} MHz, expected ${EXPECT_MHZ}:" >&2
  echo "  sudo nvidia-smi -pm 1 && sudo nvidia-smi -lgc ${EXPECT_MHZ},${EXPECT_MHZ}" >&2
  exit 1
fi

echo "window open ($(date -Is)): GPU free, clocks at ${mhz} MHz. benchmarking..."
cd "$HOME"
~/vllm-ref/bin/python marlin_bench.py --tag "$TAG" 2>&1 | tee "$LOG"
echo "done: $OUT"
