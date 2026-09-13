#!/usr/bin/env bash
# plan/0009 #short-row: the decisive engine A/B on the kernel commit.
# Today's fresh incumbent baseline (baseline-reRun.log) shows the
# short row at the 195 class again. This arm re-runs the committed
# protocol with the PRE-surgery kernel (8b26330508's walk) to decide
# whether the kernel commit is the cause: pre-arm short ~213 means
# yes (then profile in-engine); pre-arm ~195 means the recorded cost
# was cross-session drift, not the surgery.
# Usage: bash prekernel_ab.sh [restarts]
set -euo pipefail
cd "$(dirname "$0")/../../.."   # sm75-marlin worktree root
R="${1:-2}"
KERNEL=turing_lab/bridge/bridge_paged_decode_split.cu
EXTDIR=~/.cache/torch_extensions/py313_cu130/bridge_paged_decode_split

trap 'git checkout -- "$KERNEL"' EXIT

git show 8b26330508:"$KERNEL" > "$KERNEL"
rm -rf "$EXTDIR"   # force the JIT to rebuild the pre-surgery binary

COMMITTED_NO_EAGER=1 \
COMMITTED_ARM_ENV="BRIDGE_PAGED_DECODE=split,BRIDGE_PPS=2" \
.venv/bin/python turing_lab/thirdparty/vllm_committed_run.py \
    BRIDGE_ATTN "$R" 2>&1 | tee turing_lab/results/short-row-recovery/prekernel-arm.log

# the trap restores the incumbent kernel; the extension cache rebuilds
# on the next engine start (verified: no stale-binary path)
