#!/usr/bin/env bash
# plan/0009 #k1-inversion: the stock-backend K1 control at the
# standing protocol (the banked control was single-rep). The champion
# K1 arm is already banked at 3 reps (mtp-champion-sweep/k1-r{1,2,3});
# this runs the stock (TRITON) arm 3 times on the 27B, base + K1 only
# (the ksweep driver's run() is driven directly for the pair).
# Usage: bash stock_arm.sh
set -euo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"
cd "$HERE/../../.."   # sm75-marlin worktree root
MODEL=/opt/models/Intel-Qwen3.6-27B-int4-AutoRound

for r in 1 2 3; do
  KSWEEP_TP=2 \
  VLLM_ATTENTION_BACKEND=TRITON_ATTN \
  FLASHINFER_DISABLE_VERSION_CHECK=1 \
  CUDA_HOME=/opt/cuda \
  .venv/bin/python -c "
import sys
sys.path.insert(0, 'turing_lab/thirdparty')
import vllm_ksweep as ks
base = ks.run('$MODEL', 0, 'k0')
k1 = ks.run('$MODEL', 1, 'k1')
match = sum(a == b for a, b in zip(base['texts'], k1['texts']))
print('stock base', base['mixed'], 'k1', k1['mixed'],
      'speedup', {c: round(k1['mixed'][c] / base['mixed'][c], 2)
                  for c in base['mixed']},
      'greedy', f'{match}/{len(base[\"texts\"])}')
" > "$HERE/stock-r$r.log" 2>&1
  for k in 0 1; do
    [ -f turing_lab/thirdparty/ksweep-k$k.json ] && \
      cp turing_lab/thirdparty/ksweep-k$k.json "$HERE/stock-k${k}-r${r}.json"
  done
  echo "stock restart $r done"
done
