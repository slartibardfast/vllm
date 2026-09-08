# TP2 restart-variance probe on the Qwen3.5-4B same-dimension fixture (2026-09-07/08)

Fixture: Qwen3.5-4B W4A16 AutoRound (sym g128, fp16 emission, seed 42,
fixture-grade calibration iters 40 / nsamples 48 - engine-variance grade,
not quality-SOTA; first attempt at iters 200 / nsamples 128 paced ~564
s/layer and was stopped, the venv needed pillow+torchvision for the
multimodal processor). Same dimensions as the 27B target: d256
full-attention (16Q/4KV global), GDN 128/128, interval 4, 32 layers.
Protocol: the recorded macro-gate arm harness, fresh engine per restart,
median-of-5 in-process reps, clocks locked 1455 MHz, NCCL init/tuning
logging per restart. FLASHINFER_DISABLE_VERSION_CHECK=1 set with reason:
flashinfer-python 0.6.18 has no matching cubin release (latest cubin
0.6.17); no arm executes flashinfer kernels.

## Root-cause record: the bridge d256 route broke after the campaign (RESOLVED)

The first probe pass failed on EVERY BRIDGE restart (both ranks, both
models - the 27B reproduced it too) with CUDA error: invalid argument,
and an initial diagnosis suspected a head-shape instantiation gap. That
diagnosis was WRONG. The true cause, found by diffing the vendored
flash-attn tree against its campaign-final commits:

1. A half-done revert sat STAGED in the thirdparty/flash-attn tree after
   the campaign: the interface route set was cut from (64, 128, 256)
   back to (64, 128), the softcap plumbing stripped from the shim call,
   and the stale csrc header copies re-added. d256 traffic fell through
   to the 2026-08-29 prebuilt flash_attn_2_cuda binary, which cannot
   launch a D=256 kernel on sm_75 (smem over the 64 KiB optin ceiling)
   - hence invalid argument.
2. The lane's bisect-era edits to turing_lab/bridge/flash_fwd_sm75.cuh
   (the kLd compile-time split + d256 API) are LOAD-BEARING and were
   never committed: the FA2 HEAD shim includes the canonical lane
   header by relative path and expects that API. Commit
   9989333 exists precisely because of this coupling.
3. Fix: the FA2 tree restored to HEAD (interface d256 route + softcap
   shim + stale csrc copies deleted), the lane edits restored (they had
   been stashed during diagnosis - a mistake that itself demonstrated
   the coupling: the shim fails to compile without them). Direct kernel
   verification after the fix: 27B-shape prefill (24/4, d256) OK; 4B
   TP2-shape decode (8/2, d256) OK.

Standing hazard recorded: any checkout/revert in either tree
(the FA2 vendored tree or the lane's turing_lab/bridge) can silently
break the engine's d256 path. The lane kernel edits must be committed
(planned this window) so the green-gates state is fully in version
control on both sides.

## Probe findings

1. **TRITON_ATTN control arm: restart-STABLE on this fixture.** Two
   restarts: ctx512_decode 24.3 / 24.6 (band 1.2 percent), ctx2048_decode
   24.3 / 24.5 (band 0.8 percent), short_decode 173.0 / 177.7 (band 2.6
   percent). The 12-17 tok/s cross-restart swing recorded on the 1.5B
   gate model does NOT reproduce on the same-dimension hybrid under the
   control backend.
2. BRIDGE arm final state (window close 2026-09-08 ~00:20): after the
   root-cause fix, DIRECT kernel calls pass on both models' shapes
   (27B prefill 24/4 d256 OK; 4B TP2 decode 8/2 d256 OK), but the
   in-engine arm still failed its restarts with one residual:
   the engine worker's JIT build of the bridge extension needs
   CUDA_HOME ("CUDA_HOME environment variable is not set"). The fix is
   applied in the driver (env CUDA_HOME=/opt/cuda); the probe rerun is
   the FIRST ITEM of the next window. The window closed before that
   rerun per the operator.

## Next

- Commit the lane kernel edits + this record (the durable fix).
- The 27B variance verdict remains the acceptance question; the 4B
  bridge rows inform whether the swing reproduces on the hybrid at all.
