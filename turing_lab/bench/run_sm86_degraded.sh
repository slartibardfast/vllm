#!/usr/bin/env bash
# Degraded-protocol sm_86 run for plan/0002 D8: the 4096x4096 shape only,
# executed alongside the resident llama-server because the clean window
# has not been opened. Timing validity is proven, not assumed:
#   - GPU utilization sampled every 250 ms; any nonzero sample invalidates
#   - llama-server metrics counters diffed before/after; any delta invalidates
#   - the 1665 MHz clock lock verified before and after
# The fp64 correctness reference runs on CPU (--cpu-ref) because the card
# has ~0.5 GiB of allocatable headroom while the server is resident.
# This script never stops, starts, or reconfigures llama-server. If the
# verdict is VALID the numbers are usable; the clean-window run via
# ~/run_sm86_bench.sh remains the protocol of record and supersedes this
# one if it is ever performed.
set -euo pipefail

TAG=sm86-contended
OUT="$HOME/marlin-bench-$TAG.json"
LOG="$HOME/marlin-bench-$TAG.log"
UTIL="$HOME/marlin-bench-$TAG-util.log"
EXPECT_MHZ=1665
METRIC_URL=http://127.0.0.1:8080/metrics

mhz() { nvidia-smi --query-gpu=clocks.current.graphics --format=csv,noheader,nounits | head -1; }
tokens() { curl -s -m3 "$METRIC_URL" | awk '/^llamacpp:prompt_tokens_total/{print $2}'; }

if pgrep -x llama-server >/dev/null; then
  echo "llama-server is resident (pids $(pgrep -x llama-server | tr '\n' ' ')); degraded protocol"
else
  echo "llama-server is NOT running: use ~/run_sm86_bench.sh for the clean protocol instead" >&2
  exit 1
fi

pre_mhz=$(mhz)
if [ "$pre_mhz" -ne "$EXPECT_MHZ" ]; then
  echo "clock lock is ${pre_mhz} MHz, expected ${EXPECT_MHZ}" >&2
  exit 1
fi
pre_tokens=$(tokens)
echo "pre: clocks ${pre_mhz} MHz, prompt_tokens_total ${pre_tokens}"

# utilization sampler, ~250 ms period, millisecond timestamps included
( while :; do
    printf '%s %s\n' "$(date +%s%3N)" \
      "$(nvidia-smi --query-gpu=utilization.gpu --format=csv,noheader,nounits | head -1)"
    sleep 0.25
  done ) >"$UTIL" &
MON_PID=$!
trap 'kill "$MON_PID" 2>/dev/null || true' EXIT

cd "$HOME"
set +e
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
  ~/vllm-ref/bin/python marlin_bench.py --shapes 4096x4096 --cpu-ref \
  --tag "$TAG" 2>&1 | tee "$LOG"
BENCH_RC=${PIPESTATUS[0]}
set -e
kill "$MON_PID" 2>/dev/null || true
wait "$MON_PID" 2>/dev/null || true
trap - EXIT

post_mhz=$(mhz)
post_tokens=$(tokens)
# utilization is informational only: our own benchmark kernels register in
# the counter, so it cannot discriminate. The authoritative idleness proof
# is the server's own served-token counter: llama-server only touches the
# GPU to serve a request, and the counter must not move.
max_util=$(grep -E '^[0-9]+ [0-9]+$' "$UTIL" | awk '{if($2+0>m)m=$2+0}END{if(NR)printf "%d",m; else print "n/a"}')
samples=$(grep -cE '^[0-9]+ [0-9]+$' "$UTIL" || true)
tok_delta=$((post_tokens - pre_tokens))

echo "post: clocks ${post_mhz} MHz, prompt_tokens_total ${post_tokens}"
echo "monitor: ${samples} samples, max utilization ${max_util}% (informational), served-token delta ${tok_delta}"

VERDICT=INVALID
if [ "$BENCH_RC" -eq 0 ] && [ "$tok_delta" -eq 0 ] \
   && [ "$post_mhz" -eq "$EXPECT_MHZ" ]; then
  VERDICT=VALID
fi
echo "verdict: $VERDICT"
printf 'bench_rc=%s samples=%s max_util=%s tok_delta=%s pre_mhz=%s post_mhz=%s verdict=%s\n' \
  "$BENCH_RC" "$samples" "$max_util" "$tok_delta" "$pre_mhz" "$post_mhz" "$VERDICT" \
  > "$HOME/marlin-bench-$TAG-verdict.txt"
[ "$VERDICT" = VALID ]
