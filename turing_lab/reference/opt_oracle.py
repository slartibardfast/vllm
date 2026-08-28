#!/usr/bin/env python3
"""Oracle + benchmark for the optimized Turing W4A16 kernel (symmetric)."""
import os, sys, json
import torch
from torch.utils.cpp_extension import load

HERE = os.path.dirname(os.path.abspath(__file__))
ext = load(name="turing_w4a16_opt", sources=[os.path.join(HERE, "turing_w4a16_opt.cu")],
           extra_cuda_cflags=["-arch=sm_75", "-O3"], verbose=False)
torch.manual_seed(3)
DEV = "cuda:0"


def pack(q):
    # (N, K/8): low nibble first along k, row-major per output channel
    out = torch.zeros(q.shape[0], q.shape[1] // 8, dtype=torch.int32, device=DEV)
    for j in range(8):
        out |= (q[:, j::8] & 0xF) << (4 * j)
    return out.contiguous()


def ref_of(A, q, s, G):
    N, K = q.shape
    W = q.double() - 8.0
    W = W * s.repeat_interleave(G, dim=0).double().T
    return A.double() @ W.T


def case(M, N, K, G):
    A = torch.randn(M, K, dtype=torch.float16, device=DEV)
    q = torch.randint(0, 16, (N, K), dtype=torch.int32, device=DEV)
    s = (torch.rand(K // G, N, device=DEV) * 0.02 + 0.02).half()
    Q = pack(q)
    ref = ref_of(A, q, s, G)
    out = ext.w4a16_opt(A, Q, s, G)
    err = (out.double() - ref).abs().max().item()
    tol = 8e-3 * max(ref.abs().max().item(), 1e-6)
    ok = err <= tol
    print(f"  {'PASS' if ok else 'FAIL'} M{M} N{N} K{K} G{G}: err={err:.4g}")
    return ok


def bench(M, N, K, G, reps=20):
    A = torch.randn(M, K, dtype=torch.float16, device=DEV)
    q = torch.randint(0, 16, (N, K), dtype=torch.int32, device=DEV)
    s = (torch.rand(K // G, N, device=DEV) * 0.02 + 0.02).half()
    Q = pack(q)
    for _ in range(5):
        ext.w4a16_opt(A, Q, s, G)
    torch.cuda.synchronize()
    ts = []
    for _ in range(reps):
        a, b = torch.cuda.Event(True), torch.cuda.Event(True)
        a.record(); ext.w4a16_opt(A, Q, s, G); b.record()
        torch.cuda.synchronize()
        ts.append(a.elapsed_time(b))
    return min(ts), ts[len(ts)//2]


def main():
    ok = True
    for (M, N, K, G) in [(16,128,64,64),(64,256,256,64),(1,128,256,128),
                         (7,192,384,128),(100,256,256,64),(64,512,512,128)]:
        ok &= case(M, N, K, G)
    print("optimized baseline (min / median of 20, locked clocks):")
    for (N, K) in [(4096, 4096), (11008, 4096), (4096, 11008)]:
        for M in [1, 8, 32, 128, 512]:
            mn, med = bench(M, N, K, 128)
            tf = 2 * M * N * K / (med * 1e-3) / 1e12
            print(f"  M={M:4d} N={N} K={K}: median={med:.3f} ms  {tf:.2f} TFLOP/s")
    print("ALL PASS" if ok else "FAILURES PRESENT")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
