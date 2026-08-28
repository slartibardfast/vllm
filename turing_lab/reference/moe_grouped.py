#!/usr/bin/env python3
"""MoE grouped dispatch over the validated packed-staging kernel.

Tokens are routed to their top-k experts; per expert, the routed tokens
run through the packed W4A16 GEMM and are scaled by the routing weight.
Validated against a float64 reference built from the same quantized
weights.
"""
import os
import sys
import torch
from torch.utils.cpp_extension import load

HERE = os.path.dirname(os.path.abspath(__file__))
ext = load(name="turing_w4a16_opt2_moe3", sources=[os.path.join(HERE, "turing_w4a16_opt.cu")],
           extra_cuda_cflags=["-arch=sm_75", "-O3"], verbose=False)
torch.manual_seed(0)
DEV = "cuda:0"


def build_expert(N, K, G):
    """Returns (q nibbles (N,K) int32, packed Q (N,K/8) int32, scales (K/G,N) fp16)."""
    q = torch.randint(0, 16, (N, K), dtype=torch.int32, device=DEV)
    Q = torch.zeros(N, K // 8, dtype=torch.int32, device=DEV)
    for j in range(8):
        Q |= (q[:, j::8] & 0xF) << (4 * j)
    s = (torch.rand(K // G, N, device=DEV) * 0.02 + 0.02).half()
    return q, Q, s


def run_expert(A, Q, s, G):
    return ext.w4a16_opt2(A, Q, s, G)


def main():
    E, T, N, K, G, top_k = 4, 32, 256, 64, 64, 2
    tokens = torch.randn(T, K, dtype=torch.float16, device=DEV)
    router = torch.randn(T, E, device=DEV)
    topv, topi = router.topk(top_k, dim=1)
    weights = torch.softmax(topv, dim=1)

    experts = [build_expert(N, K, G) for _ in range(E)]

    # float64 reference with the same routing
    ref = torch.zeros(T, N, dtype=torch.float64, device=DEV)
    for e in range(E):
        q_nk, Q_e, s_e = experts[e]
        t_idx, k_pos = (topi == e).nonzero(as_tuple=True)
        rw = weights[t_idx, k_pos].double().unsqueeze(1)
        w = (q_nk.double() - 8.0) * s_e.repeat_interleave(K // G, 0).double().T
        for i, t in enumerate(t_idx.tolist()):
            ref[t] += (tokens[t].double() @ w.T) * rw[i]

    # routed forward over the validated kernel
    out = torch.zeros(T, N, dtype=torch.float16, device=DEV)
    for e in range(E):
        q_nk, Q_e, s_e = experts[e]
        t_idx, k_pos = (topi == e).nonzero(as_tuple=True)
        if t_idx.numel() == 0:
            continue
        A_e = tokens[t_idx]
        out_e = run_expert(A_e, Q_e, s_e, G)
        rw = weights[t_idx, k_pos].double().unsqueeze(1)
        for i, t in enumerate(t_idx.tolist()):
            out[t] += (out_e[i].double() * rw[i]).half()

    err = (out.double() - ref).abs().max().item()
    tol = 8e-2 * max(ref.abs().max().item(), 1e-6)
    status = "PASS" if err <= tol else "FAIL"
    print(f"{status} MoE grouped forward: T={T} E={E} N={N} K={K} G={G} "
          f"top_k={top_k}: err={err:.4g} (tol {tol:.3g})")
    sys.exit(0 if status == "PASS" else 1)


if __name__ == "__main__":
    main()
