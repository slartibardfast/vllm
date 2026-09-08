#!/bin/bash
# Window preflight: run FIRST in every GPU window, before any batch.
# Each check exists because the 2026-09-07/08 window paid for its
# absence. Non-zero exit = fix before proceeding.
set -u
VENV=/opt/repo/agentic-vllm/software/vllm/sm75-marlin/.venv/bin/python
fail=0
ok()   { echo "PREFLIGHT OK   $1"; }
bad()  { echo "PREFLIGHT FAIL $1"; fail=1; }

# 1. GPUs actually free (llama-server stopped, no residual processes)
used=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits |
       sort -rn | head -1)
[ "$used" -lt 1000 ] && ok "GPUs free (max used ${used} MiB)" ||
    bad "GPUs occupied (max used ${used} MiB) - stop llama-server first"

# 2. Clocks locked at the house 1455 MHz
clocks=$(nvidia-smi --query-gpu=clocks.sm --format=csv,noheader,nounits |
         sort -u | tr '\n' ' ')
[ "$clocks" = "1455 1455 " ] && ok "clocks locked 1455" ||
    bad "clocks not locked (got: $clocks) - sudo nvidia-smi -lgc 1455 -i 0,1"

# 3. Toolchain root for worker JIT builds
[ "${CUDA_HOME:-/unset}" = "/opt/cuda" ] && ok "CUDA_HOME=/opt/cuda" ||
    bad "CUDA_HOME unset (export CUDA_HOME=/opt/cuda)"

# 4. flashinfer pairing: no 0.6.18 cubin exists upstream; the documented
# bypass is required and no probe arm executes flashinfer kernels
export FLASHINFER_DISABLE_VERSION_CHECK=1
ok "flashinfer version-check bypass set (reason in REPORT.md)"

# 5. One-shape kernel smoke on EACH model's d256 shape (the known-good
# control FIRST - the lesson of the retracted instantiation-gap
# diagnosis). Fails fast if the vendored tree or lane header regressed.
$VENV - << 'EOF' && ok "bridge d256 kernel smoke (both model shapes)" \
                 || bad "bridge d256 kernel smoke FAILED - check the FA2 tree vs HEAD and lane header"
import torch
from flash_attn import flash_attn_func
for h_q, h_kv, sq in ((24, 4, 512), (8, 2, 1)):
    q = torch.randn(1, sq, h_q, 256, dtype=torch.float16, device="cuda")
    k = torch.randn(1, 512, h_kv, 256, dtype=torch.float16, device="cuda")
    v = torch.randn(1, 512, h_kv, 256, dtype=torch.float16, device="cuda")
    o = flash_attn_func(q, k, v, softmax_scale=0.0390625, causal=True)
    torch.cuda.synchronize()
print("smoke ok")
EOF

# 6. The fixture and the 27B are on disk
for m in /opt/models/Qwen3.5-4B-W4A16-AR-f16/Qwen3.5-4B-w4g128 \
         /opt/models/Intel-Qwen3.6-27B-int4-AutoRound; do
    [ -f "$m/config.json" ] && ok "model present: $m" ||
        bad "model MISSING: $m"
done

exit $fail
