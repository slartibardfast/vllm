#!/usr/bin/env python3
"""Transport characterization and multi-GPU strategy measurements.

The operator's scope: single GPU, GPU-to-GPU by PCIe, GPU-to-GPU by
NVLink. This module measures what each transport actually delivers on
this host (the two Quadro RTX 6000s are NVLink-bridged, 2 links), then
times the two multi-GPU W4A16 strategies over each available path:

- nshard2: each GPU computes half the N columns; the output halves are
  gathered (one cross-GPU copy, no reduction);
- kshard2: each GPU reduces over half of K; the fp16 partials are
  summed on GPU0 (one cross-GPU copy plus an add).

Everything composes the single-GPU search variants, so the strategy
dimension of a future dispatch table is (variant, split-K, placement).
"""
import json
from statistics import median

import torch


def measure_transport(mgpu, size_mb: int = 256, iters: int = 20,
                      lat_iters: int = 200) -> dict:
    """Bandwidth and latency for P2P (NVLink when enabled) and for the
    staged-through-host path (PCIe x2)."""
    can01, can10 = mgpu.can_access_peer()
    enabled = mgpu.enable_peer_access()
    src = torch.randn(size_mb * 1024 * 1024 // 2, dtype=torch.float16,
                      device="cuda:0")
    dst = torch.empty_like(src, device="cuda:1")
    small = torch.randn(1024, dtype=torch.float16, device="cuda:0")
    dsmall = torch.empty_like(small, device="cuda:1")

    def time_copy(f):
        f()
        torch.cuda.synchronize()
        ts = []
        for _ in range(iters):
            s, e = torch.cuda.Event(True), torch.cuda.Event(True)
            s.record()
            f()
            e.record()
            torch.cuda.synchronize()
            ts.append(s.elapsed_time(e))
        return median(sorted(ts))

    # P2P: cross-device copy_ goes through cudaMemcpyPeerAsync, which uses
    # the NVLink bridge once peer access is enabled.
    p2p_ms = time_copy(lambda: dst.copy_(src, non_blocking=True))
    p2p_lat_ms = time_copy(lambda: dsmall.copy_(small, non_blocking=True))
    # Staged through host: PCIe up + PCIe down (D2H then H2D).
    host = torch.empty_like(src, device="cpu", pin_memory=True)
    host_small = torch.empty_like(small, device="cpu", pin_memory=True)
    staged_ms = time_copy(
        lambda: (host.copy_(src, non_blocking=True),
                 dst.copy_(host, non_blocking=True)))
    staged_lat_ms = time_copy(
        lambda: (host_small.copy_(small, non_blocking=True),
                 dsmall.copy_(host_small, non_blocking=True)))

    def gbps(ms):
        return size_mb / 1024 / (ms * 1e-3)
    return {
        "can_access_peer": [can01, can10],
        "peer_enabled": bool(enabled),
        "p2p": {"ms": p2p_ms, "gbps": gbps(p2p_ms),
                "latency_us": p2p_lat_ms * 1e3},
        "staged": {"ms": staged_ms, "gbps": gbps(staged_ms),
                   "latency_us": staged_lat_ms * 1e3},
    }


def nshard2(ext, vi: int, A, Q, S, empty, G: int = 128, nz: int = 1):
    """N-sharded across two GPUs: returns (Out on cuda:0, ms_components)."""
    M, K = A.shape
    N = Q.shape[0]
    N2 = N // 2
    A1 = A.to("cuda:1")
    Q0, Q1 = Q[:N2].contiguous(), Q[N2:].to("cuda:1").contiguous()
    S0, S1 = S[:, :N2].contiguous(), S[:, N2:].to("cuda:1").contiguous()
    s, e = torch.cuda.Event(True), torch.cuda.Event(True)
    s.record()
    out0, _ = ext.launch(vi, A, Q0, S0, empty, G, nz)
    out1, _ = ext.launch(vi, A1, Q1, S1, empty, G, nz)
    out1 = out1.to("cuda:0", non_blocking=True)
    e.record()
    torch.cuda.synchronize()
    return torch.cat([out0, out1], dim=1), s.elapsed_time(e)


def kshard2(ext, vi: int, A, Q, S, empty, G: int = 128):
    """K-sharded across two GPUs with fp32 partials: each GPU reduces its
    own K half (both A and Q are sharded — the driver derives K from
    A's width, so sharding only Q reads out of bounds) and the exact
    fp32 partials are summed on cuda:0. Returns (Out on cuda:0, ms)."""
    M, K = A.shape
    kw2 = (K // 8) // 2
    kh = K // 2
    A0, A1 = A[:, :kh].contiguous(), A[:, kh:].to("cuda:1").contiguous()
    Q0, Q1 = Q[:, :kw2].contiguous(), Q[:, kw2:].to("cuda:1").contiguous()
    S0, S1 = S[: K // (2 * G), :].contiguous(), S[K // (2 * G):, :].to(
        "cuda:1").contiguous()
    s, e = torch.cuda.Event(True), torch.cuda.Event(True)
    s.record()
    p0 = ext.launch_partial(vi, A0, Q0, S0, empty, G)
    p1 = ext.launch_partial(vi, A1, Q1, S1, empty, G)
    p1 = p1.to("cuda:0", non_blocking=True)
    e.record()
    torch.cuda.synchronize()
    return (p0 + p1).half(), s.elapsed_time(e)


def measure_strategies(ext, mgpu, shapes, msweeps, G: int = 128) -> dict:
    """End-to-end timing + oracle for single-GPU vs nshard2 vs kshard2."""
    from harness import make_problem
    mgpu.enable_peer_access()
    rows = []
    for (N, K) in shapes:
        for M in msweeps:
            A, Q, S, empty, ref_fn = make_problem(N, K, M, seed=3)  # noqa: B023
            ref = ref_fn()
            tol = 0.05 * max(ref.abs().max().item(), 1e-6)

            def timed(f, reps=10):
                f()
                torch.cuda.synchronize()
                ts = []
                for _ in range(reps):
                    st, en = torch.cuda.Event(True), torch.cuda.Event(True)
                    st.record()
                    f()
                    en.record()
                    torch.cuda.synchronize()
                    ts.append(st.elapsed_time(en))
                return median(sorted(ts))

            # single-GPU reference: best-of-first-gen config, nz=1; the
            # loop-variable bindings are deliberate defaults (B023)
            out_single, _ = ext.launch(7, A, Q, S, empty, G, 1)
            t_single = timed(
                lambda A=A, Q=Q, S=S, empty=empty: ext.launch(
                    7, A, Q, S, empty, G, 1))
            out_ns, t_ns = nshard2(ext, 7, A, Q, S, empty, G)
            out_ks, t_ks = kshard2(ext, 7, A, Q, S, empty, G)
            e_ns = (out_ns.double() - ref).abs().max().item()
            e_ks = (out_ks.double() - ref).abs().max().item()
            row = {
                "N": N, "K": K, "M": M,
                "single_ms": t_single,
                "nshard2_ms": t_ns, "nshard2_ok": e_ns < tol,
                "kshard2_ms": t_ks, "kshard2_ok": e_ks < tol,
                "nshard2_speedup": round(t_single / t_ns, 3),
                "kshard2_speedup": round(t_single / t_ks, 3),
            }
            rows.append(row)
            print(json.dumps(row))
    return rows
