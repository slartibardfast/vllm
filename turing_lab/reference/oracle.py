#!/usr/bin/env python3
"""Correctness oracle for the reference Turing W4A16 kernels.

Builds the extension, checks both kernels against a float64 reference over
shapes, group sizes, and zero-point modes, then times the tiled kernel for
the performance baseline.
"""
import os
import sys
import torch
from torch.utils.cpp_extension import load

HERE = os.path.dirname(os.path.abspath(__file__))
cuda_flags = ["-arch=sm_75", "-O3", "--use_fast_math"]
ext = load(name="turing_w4a16_ref", sources=[os.path.join(HERE, "turing_w4a16.cu")],
           extra_cuda_cflags=cuda_flags, verbose=False)

torch.manual_seed(0)
DEV = "cuda:0"


def make_case(M, N, K, G, sym):
    A = torch.randn(M, K, dtype=torch.float16, device=DEV)
    q = torch.randint(0, 16, (N, K), dtype=torch.int32, device=DEV)
    Q = (q[:, 0::8] | q[:, 1::8] << 4 | q[:, 2::8] << 8 | q[:, 3::8] << 12
         | q[:, 4::8] << 16 | q[:, 5::8] << 20 | q[:, 6::8] << 24
         | q[:, 7::8] << 28).contiguous()
    if G == -1:
        G = K
    n_groups = K // G
    S = (torch.rand(n_groups, N, dtype=torch.float16, device=DEV) * 0.1 + 0.05)
    ZP = None if sym else torch.randint(0, 16, (n_groups, N), dtype=torch.int32,
                                        device=DEV)
    return A, Q, S, ZP, G


def ref_dequant(A, Q, S, ZP, G):
    N, K = Q.shape[0], Q.shape[1] * 8
    nib = torch.stack([(Q >> (i * 4)) & 0xF for i in range(8)], dim=2)
    W = nib.reshape(N, K).to(torch.float64)
    Sg = S.repeat_interleave(G, dim=0).to(torch.float64).T  # (N, K)
    if ZP is not None:
        W = W - ZP.repeat_interleave(G, dim=0).to(torch.float64).T
    else:
        W = W - 8.0
    W = W * Sg
    return A.to(torch.float64) @ W.T


def check(name, got, ref, tol):
    got64 = got.to(torch.float64)
    err = (got64 - ref).abs().max().item()
    scale = ref.abs().max().item()
    ok = err <= tol * max(scale, 1e-6)
    print(f"  {'PASS' if ok else 'FAIL'} {name}: max_abs_err={err:.4g} "
          f"(tol {tol:.0e} x {scale:.3g})")
    return ok


def main():
    all_ok = True
    cases = [
        (16, 64, 64, 64, True),
        (64, 128, 256, 64, True),
        (1, 64, 128, 32, False),
        (7, 192, 384, 128, False),
        (100, 256, 256, 64, True),
        (64, 512, 512, 128, False),
        (33, 128, 768, 32, True),
    ]
    for (M, N, K, G, sym) in cases:
        A, Q, S, ZP, Gc = make_case(M, N, K, G, sym)
        ref = ref_dequant(A, Q, S, ZP, Gc)
        tag = f"M{M} N{N} K{K} G{Gc} {'sym' if sym else 'zp'}"
        print(tag)
        out = ext.w4a16_naive(A, Q, S, ZP, Gc)
        all_ok &= check("naive", out, ref, 2e-3)
        out = ext.w4a16_tiled(A, Q, S, ZP, Gc)
        all_ok &= check("tiled", out, ref, 5e-3)

    print("baseline timing (tiled, locked clocks, median of 20):")
    A, Q, S, ZP, Gc = make_case(512, 4096, 4096, 128, True)
    for M in [1, 8, 32, 128, 512]:
        A = torch.randn(M, 4096, dtype=torch.float16, device=DEV)
        for _ in range(5):
            ext.w4a16_tiled(A, Q, S, ZP, Gc)
        torch.cuda.synchronize()
        ts = []
        for _ in range(20):
            s, e = torch.cuda.Event(True), torch.cuda.Event(True)
            s.record(); ext.w4a16_tiled(A, Q, S, ZP, Gc); e.record()
            torch.cuda.synchronize()
            ts.append(s.elapsed_time(e))
        ts.sort()
        med = ts[len(ts) // 2]
        tflops = 2 * M * 4096 * 4096 / (med * 1e-3) / 1e12
        print(f"  M={M:4d} median={med:.3f} ms  {tflops:.2f} TFLOP/s")
    print("ALL PASS" if all_ok else "FAILURES PRESENT")
    sys.exit(0 if all_ok else 1)


if __name__ == "__main__":
    main()
