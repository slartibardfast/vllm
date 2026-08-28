#!/usr/bin/env python3
"""Cross-rig incumbent Marlin benchmark (W4A16 GPTQ, symmetric).

Runs the same packed-weight benchmark through vLLM's own helpers on any
rig, so numbers differ only by hardware. Run under locked clocks.

usage: marlin_bench.py [--reps 20] [--shapes 4096x4096,11008x4096,4096x11008]
                       [--ms 1,8,32,128,512] [--tag local]
                       [--cpu-ref]
"""
import argparse
import json
import subprocess
from statistics import median

import torch

from vllm import _custom_ops as ops
from vllm.model_executor.layers.quantization.utils.marlin_utils import (
    marlin_make_workspace_new,
    marlin_permute_scales,
)
from vllm.scalar_type import scalar_types


def gpu_meta():
    out = subprocess.run(
        ["nvidia-smi", "--query-gpu=name,driver_version,clocks.current.graphics",
         "--format=csv,noheader"], capture_output=True, text=True).stdout
    return out.strip().splitlines()[0] if out.strip() else "unknown"


def pack_q(w, G):
    """Quantize W (N, K) to GPTQ symmetric u4b8 and pack to (K/8, N) int32."""
    N, K = w.shape
    groups = K // G
    s_ng = (w.view(N, groups, G).abs().amax(dim=2) / 7.0).clamp(min=1e-8).half()
    s = s_ng.T.contiguous()  # (groups, N)
    q = torch.clamp(
        torch.round(w.view(N, groups, G).double()
                    / s_ng.double().unsqueeze(2)) + 8, 0, 15).to(torch.int32)
    q = q.view(N, K)
    packed = torch.zeros(K // 8, N, dtype=torch.int32, device=w.device)
    for j in range(8):
        packed |= (q[:, j::8] & 0xF).T << (4 * j)
    return packed, s, q, s_ng


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--reps", type=int, default=20)
    ap.add_argument("--shapes", default="4096x4096,11008x4096,4096x11008")
    ap.add_argument("--ms", default="1,8,32,128,512")
    ap.add_argument("--tag", default="run")
    ap.add_argument("--cpu-ref", action="store_true",
                    help="evaluate the fp64 reference on CPU (for"
                         " memory-constrained GPUs); identical IEEE fp64")
    args = ap.parse_args()
    shapes = [tuple(int(x) for x in s.split("x")) for s in args.shapes.split(",")]
    ms = [int(x) for x in args.ms.split(",")]

    dev = "cuda:0"
    wtype = scalar_types.uint4b8
    G = 128
    results = []

    for (N, K) in shapes:
        w = torch.randn(N, K, dtype=torch.float16, device=dev) * 0.05
        packed, s, q, sc = pack_q(w, G)
        empty = torch.empty(0, dtype=torch.int32, device=dev)
        qweight = ops.gptq_marlin_repack(
            packed, perm=empty, size_k=K, size_n=N, num_bits=4, is_a_8bit=False)
        w_s = marlin_permute_scales(s, size_k=K, size_n=N, group_size=G,
                                    is_a_8bit=False)
        workspace = marlin_make_workspace_new(dev)
        g_idx, zp = empty, empty

        # correctness once per shape against the float64 dequant reference
        A = torch.randn(64, K, dtype=torch.float16, device=dev)
        if args.cpu_ref:
            # reference on CPU keeps the GPU pool small; fp64 is fp64
            w_deq = (q.cpu().double() - 8.0) * \
                s.repeat_interleave(G, 0).cpu().double().T
            ref = A.cpu().double() @ w_deq.T
            out = ops.marlin_gemm(
                A, None, qweight, None, w_s, None, None, zp, g_idx,
                g_idx, workspace, wtype, size_m=64, size_n=N, size_k=K,
                is_k_full=True, use_atomic_add=False, use_fp32_reduce=True)
            err = (out.cpu().double() - ref).abs().max().item()
            del packed, q, w
        else:
            w_deq = (q.double() - 8.0) * s.repeat_interleave(G, 0).double().T
            A = torch.randn(64, K, dtype=torch.float16, device=dev)
            ref = A.double() @ w_deq.T
            out = ops.marlin_gemm(
                A, None, qweight, None, w_s, None, None, zp, g_idx,
                g_idx, workspace, wtype, size_m=64, size_n=N, size_k=K,
                is_k_full=True, use_atomic_add=False, use_fp32_reduce=True)
            err = (out.double() - ref).abs().max().item()
        assert err < 0.05 * max(ref.abs().max().item(), 1e-6), f"correctness {err}"

        for M in ms:
            x = torch.randn(M, K, dtype=torch.float16, device=dev)
            for _ in range(5):
                ops.marlin_gemm(
                    x, None, qweight, None, w_s, None, None, zp, g_idx,
                    g_idx, workspace, wtype, size_m=M, size_n=N, size_k=K,
                    is_k_full=True, use_atomic_add=False, use_fp32_reduce=True)
            torch.cuda.synchronize()
            ts = []
            for _ in range(args.reps):
                a, b = torch.cuda.Event(True), torch.cuda.Event(True)
                a.record()
                ops.marlin_gemm(
                    x, None, qweight, None, w_s, None, None, zp, g_idx,
                    g_idx, workspace, wtype, size_m=M, size_n=N, size_k=K,
                    is_k_full=True, use_atomic_add=False, use_fp32_reduce=True)
                b.record()
                torch.cuda.synchronize()
                ts.append(a.elapsed_time(b))
            med = median(ts)
            tf = 2 * M * N * K / (med * 1e-3) / 1e12
            row = {"N": N, "K": K, "M": M, "ms": round(med, 4),
                   "tflops": round(tf, 2), "max_err": round(err, 5)}
            results.append(row)
            print(f"rig_row {json.dumps(row)}")

    meta = gpu_meta()
    print(f"rig_meta {json.dumps({'gpu': meta, 'tag': args.tag})}")
    with open(f"marlin-bench-{args.tag}.json", "w") as f:
        json.dump({"gpu": meta, "results": results}, f, indent=2)


if __name__ == "__main__":
    main()
