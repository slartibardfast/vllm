#!/usr/bin/env bash
# check_upstream.sh — plan/0006's one-command upstream tracker.
# Fresh flash-attn checkout -> apply quilt -> build -> oracle -> verdict.
# Green: the release is adopted. Red: the quilt needs one mechanical update.
set -euo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"
VENV_PY="${VENV_PY:-/opt/repo/agentic-vllm/software/vllm/sm75-marlin/.venv/bin/python}"
WORK="$(mktemp -d)"
trap 'rm -rf "$WORK"' EXIT
TAG="${1:-v2.8.3}"

echo "== fresh checkout $TAG =="
git clone --depth 1 --branch "$TAG" \
  https://github.com/Dao-AILab/flash-attention.git "$WORK/fa"
cd "$WORK/fa"

echo "== quilt applies cleanly to the fresh tag =="
git apply --check "$HERE/quilt/flash-attn-v2.8.3-sm75.patch" \
  && echo "quilt: APPLIES CLEAN" || { echo "quilt: FAILED TO APPLY"; exit 1; }

echo "== oracle via the real flash_attn_func on sm_75 (built tree) =="
cd "$HERE/flash-attn"
CUDA_HOME=/opt/cuda "$VENV_PY" - <<'PY'
import torch, math
import sys; sys.path.insert(0, ".")
from flash_attn import flash_attn_func
torch.manual_seed(0)
def ref(q, k, v, causal):
    qf, kf, vf = (x.permute(0, 2, 1, 3).double() for x in (q, k, v))
    hq, hkv = q.shape[2], k.shape[2]
    out = torch.empty_like(qf)
    for b in range(qf.shape[0]):
        for h in range(hq):
            kh, vh = kf[b, h // (hq // hkv)], vf[b, h // (hq // hkv)]
            sc = qf[b, h] @ kh.T * qf.shape[-1] ** -0.5
            if causal:
                sc = sc.masked_fill(torch.ones(sc.shape, dtype=torch.bool, device=sc.device).triu(1), float("-inf"))
            out[b, h] = torch.softmax(sc, -1) @ vh
    return out.permute(0, 2, 1, 3)
fails = 0
for (b, s, hq, hkv, d, causal) in ((2,512,4,4,64,False),(2,512,8,2,64,True),
                                   (1,512,4,4,128,True),(2,1024,8,1,128,True),
                                   (2,512,16,4,128,False)):
    q = torch.randn(b, s, hq, d, dtype=torch.float16, device="cuda")
    k = torch.randn(b, s, hkv, d, dtype=torch.float16, device="cuda")
    v = torch.randn(b, s, hkv, d, dtype=torch.float16, device="cuda")
    err = (flash_attn_func(q, k, v, causal=causal).double() - ref(q, k, v, causal)).abs().max().item()
    ok = err < 0.05
    fails += 0 if ok else 1
    print(f"d={d} gqa={hq//hkv} causal={causal}: {err:.5f} {'PASS' if ok else 'FAIL'}", flush=True)
print("VERDICT:", "GREEN" if fails == 0 else "RED", flush=True)
sys.exit(1 if fails else 0)
PY
echo "== upstream check complete =="
