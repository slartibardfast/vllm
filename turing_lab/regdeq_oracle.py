#!/usr/bin/env python3
"""Oracle + benchmark for the register-dequant W4A16 kernel
(turing_w4a16_regdeq.cu, plan/0011).

B is packed in the INCUMBENT Marlin layout per the resolved contract
(plan/0002-turing-marlin-backend/marlin-contract.md, term-verified):
output word W[T,U,th,w] at linear address T*(2*N) + U*128 + th*4 + w,
nibble i -> (n,k) per the contract table with n0 = 64U + 16w + c,
t = th%4, c = th/4. Scales fp16 group-128 in the 8x8-permuted layout
(within each 64-column block: stored[p] = orig[perm[p]],
perm[8i+j] = i+8j).

Usage: regdeq_oracle.py [quick]
  default: 7-case battery + M sweep (median of 20)
  quick:   battery only
"""
import os
import statistics
import sys

import torch
from torch.utils.cpp_extension import load

HERE = os.path.dirname(os.path.abspath(__file__))
DEV = "cuda:0"

ext = load(
    name="turing_w4a16_regdeq",
    sources=[os.path.join(HERE, "turing_w4a16_regdeq.cu")],
    extra_cuda_cflags=["-arch=sm_75", "-O3"], verbose=False)


def pack_contract(q):
    """q: (N, K) int4 values 0..15 -> the incumbent packed layout
    ((K/16, 2*N) uint32), per the contract's closed-form index map.
    Runs on CPU tensors; the caller moves the result to the device."""
    N, K = q.shape
    assert K % 16 == 0 and N % 64 == 0
    T, U = K // 16, N // 64
    out = torch.zeros(T, 2 * N, dtype=torch.int32)
    qcpu = q.cpu()
    kt = torch.arange(T, dtype=torch.int64) * 16  # tile base k
    for th in range(32):
        t, c = th % 4, th // 4
        for w in range(4):
            n0 = 16 * w + c
            # nibble i -> (n-offset, k-in-tile) per the contract table
            mapping = ((n0, 2 * t), (n0, 8 + 2 * t),
                       (n0 + 8, 2 * t), (n0 + 8, 8 + 2 * t),
                       (n0, 2 * t + 1), (n0, 9 + 2 * t),
                       (n0 + 8, 2 * t + 1), (n0 + 8, 9 + 2 * t))
            for u in range(U):
                word = torch.zeros(T, dtype=torch.int32)
                for i, (dn, kk) in enumerate(mapping):
                    word |= (qcpu[64 * u + dn, kt + kk] & 0xF) << (4 * i)
                out[:, u * 128 + th * 4 + w] = word
    return out.contiguous()


def permute_scales(s):
    """s: (K/128, N) fp16 -> the incumbent 8x8-transposed layout."""
    G, N = s.shape
    assert N % 64 == 0
    out = s.clone()
    for b in range(N // 64):
        blk = s[:, b * 64:(b + 1) * 64]
        perm = torch.tensor([i + 8 * j for i in range(8)
                             for j in range(8)])
        out[:, b * 64:(b + 1) * 64] = blk[:, perm]
    return out.contiguous()


def ref_of(A, q, s, G):
    N, K = q.shape
    W = (q.double() - 8.0) * s.repeat_interleave(G, dim=0).double().T
    return A.double() @ W.T


def case(M, N, K, G):
    A = torch.randn(M, K, dtype=torch.float16, device=DEV)
    q = torch.randint(0, 16, (N, K), dtype=torch.int32, device=DEV)
    s = (torch.rand(K // G, N, device=DEV) * 0.02 + 0.02).half()
    ref = ref_of(A, q, s, G)
    out = ext.turing_w4a16_regdeq(A, pack_contract(q),
                                  permute_scales(s), M, N, K)
    err = (out.double() - ref).abs().max().item()
    tol = 8e-3 * max(ref.abs().max().item(), 1e-6)
    ok = err <= tol
    print(f"  {'PASS' if ok else 'FAIL'} M{M} N{N} K{K} G{G}: "
          f"err={err:.4g}", flush=True)
    return ok


def bench(M, N, K, G, reps=20):
    A = torch.randn(M, K, dtype=torch.float16, device=DEV)
    q = torch.randint(0, 16, (N, K), dtype=torch.int32, device=DEV)
    s = (torch.rand(K // G, N, device=DEV) * 0.02 + 0.02).half()
    B = pack_contract(q)
    S = permute_scales(s)
    for _ in range(5):
        ext.turing_w4a16_regdeq(A, B, S, M, N, K)
    torch.cuda.synchronize()
    ts = []
    for _ in range(reps):
        s0 = torch.cuda.Event(enable_timing=True)
        e0 = torch.cuda.Event(enable_timing=True)
        s0.record()
        ext.turing_w4a16_regdeq(A, B, S, M, N, K)
        e0.record()
        torch.cuda.synchronize()
        ts.append(s0.elapsed_time(e0))
    med = statistics.median(ts)
    tflops = 2.0 * M * N * K / (med * 1e-3) / 1e12
    print(f"  M{M:<5d}: {med * 1e3:9.1f} us  {tflops:7.2f} TFLOP/s",
          flush=True)
    return tflops


def main():
    quick = len(sys.argv) > 1 and sys.argv[1] == "quick"
    torch.manual_seed(3)
    print("oracle battery (fp64 reference):")
    fails = 0
    for M, N, K, G in ((1, 4096, 4096, 128), (8, 4096, 4096, 128),
                       (64, 4096, 4096, 128), (512, 4096, 4096, 128),
                       (512, 11008, 4096, 128), (16, 4096, 11008, 128),
                       (256, 4096, 4096, 64)):
        fails += 0 if case(M, N, K, G) else 1
    if fails:
        print(f"BATTERY: {fails} FAIL")
        sys.exit(1)
    print("BATTERY: all pass")
    if quick:
        return
    print("M sweep (N=K=4096, G=128, median of 20):")
    for M in (1, 2, 4, 8, 16, 32, 64, 128, 256, 512):
        bench(M, 4096, 4096, 128)


if __name__ == "__main__":
    main()
