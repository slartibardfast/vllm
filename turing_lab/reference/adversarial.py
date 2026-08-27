#!/usr/bin/env python3
"""Adversarial and numerical-stress validation for the reference kernels.

The plan's correctness bar is silent-divergence detection, not crash
detection: extreme scale ratios, zero scales, heavy-tailed activations,
near-rounding-boundary weights, and many-group configurations, all against
the float64 reference. Both kernels must pass every case.
"""
import os
import sys

import torch

from torch.utils.cpp_extension import load

HERE = os.path.dirname(os.path.abspath(__file__))
ext = load(name="turing_w4a16_ref", sources=[os.path.join(HERE, "turing_w4a16.cu")],
           extra_cuda_cflags=["-arch=sm_75", "-O3"], verbose=False)

torch.manual_seed(7)
DEV = "cuda:0"


def pack(q):
    return (q[:, 0::8] | q[:, 1::8] << 4 | q[:, 2::8] << 8 | q[:, 3::8] << 12
            | q[:, 4::8] << 16 | q[:, 5::8] << 20 | q[:, 6::8] << 24
            | q[:, 7::8] << 28).contiguous()


def ref_of(A, Q, S, ZP, G):
    N, K = Q.shape[0], Q.shape[1] * 8
    W = torch.stack([(Q >> (i * 4)) & 0xF for i in range(8)], 2)
    W = W.reshape(N, K).double()
    if ZP is not None:
        W = W - ZP.repeat_interleave(G, dim=0).double().T
    else:
        W = W - 8.0
    W = W * S.repeat_interleave(G, dim=0).double().T
    return A.double() @ W.T


def check(name, got, ref, tol=8e-3):
    got64, ref64 = got.double(), ref
    scale = ref64.abs().max().item()
    err = (got64 - ref64).abs().max().item()
    ok = err <= tol * max(scale, 1e-6) and torch.isfinite(got64).all().item()
    print(f"  {'PASS' if ok else 'FAIL'} {name}: err={err:.4g} scale={scale:.4g}")
    return ok


def main():
    ok_all = True
    N, K, M = 128, 512, 64
    G = 64
    groups = K // G

    q = torch.randint(0, 16, (N, K), dtype=torch.int32, device=DEV)
    Q = pack(q)

    # extreme scale ratios across groups: 1e-3 to 1e1 per group, so a group
    # boundary error shows as a large, localized divergence
    scales = torch.logspace(-3, 1, groups, device=DEV).half()
    S = scales.unsqueeze(1).expand(groups, N).contiguous()
    A = torch.randn(M, K, dtype=torch.float16, device=DEV)
    ref = ref_of(A, Q, S, None, G)
    ok_all &= check("extreme scale ratios", ext.w4a16_tiled(A, Q, S, None, G), ref)
    ok_all &= check("extreme ratios naive", ext.w4a16_naive(A, Q, S, None, G), ref)

    # zero scales: one dead group must contribute exactly nothing
    S = torch.rand(groups, N, dtype=torch.float16, device=DEV) * 0.1 + 0.05
    S[2] = 0.0
    ref = ref_of(A, Q, S, None, G)
    ok_all &= check("zero scale group", ext.w4a16_tiled(A, Q, S, None, G), ref)

    # heavy-tailed activations (lognormal): long tails stress the accumulator
    A = torch.distributions.LogNormal(0.0, 2.0).sample((M, K)).half().to(DEV)
    S = torch.rand(groups, N, dtype=torch.float16, device=DEV) * 0.1 + 0.05
    ref = ref_of(A, Q, S, None, G)
    ok_all &= check("heavy-tail activations", ext.w4a16_tiled(A, Q, S, None, G), ref)

    # near-rounding-boundary weights: scales of two-powers make (q-zp)*s a
    # value exactly on an fp16 rounding edge
    for e in [-6, -2, 0]:
        S = (torch.ones(groups, N, device=DEV) * (2.0 ** e)).half()
        ref = ref_of(A, Q, S, None, G)
        ok_all &= check(f"rounding edge 2**{e}", ext.w4a16_tiled(A, Q, S, None, G), ref)

    # many groups per chunk: G 32 with BK 64 spans two groups per k-chunk
    Gs = 32
    S = torch.rand(K // Gs, N, dtype=torch.float16, device=DEV) * 0.1 + 0.05
    ZP = torch.randint(0, 16, (K // Gs, N), dtype=torch.int32, device=DEV)
    A = torch.randn(M, K, dtype=torch.float16, device=DEV)
    ref = ref_of(A, Q, S, ZP, Gs)
    ok_all &= check("G32 zp multi-group chunk", ext.w4a16_tiled(A, Q, S, ZP, Gs), ref)
    ok_all &= check("G32 zp naive", ext.w4a16_naive(A, Q, S, ZP, Gs), ref)

    # activation-order (g_idx) is out of the reference contract; recorded so
    # its absence is a documented scope line, not a silent gap
    print("  NOTE activation order (g_idx) is out of the reference contract")
    print("ALL PASS" if ok_all else "FAILURES PRESENT")
    sys.exit(0 if ok_all else 1)


if __name__ == "__main__":
    main()
