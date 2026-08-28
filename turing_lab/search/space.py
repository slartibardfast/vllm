#!/usr/bin/env python3
"""The candidate space and the analytic pre-filter.

The program's doctrine (performance-model.md) is "model filters
candidates; only hardware timing selects them". This module makes the
model literal: every candidate from the compiled variant table is parsed
into (strategy, BM, BN, BK, warps), checked against the occupancy rule
(at least two resident blocks per SM on TU102), and given a roofline
prediction per M regime so the sweep only measures near-frontier
candidates.
"""

from dataclasses import dataclass

# TU102 measured rates (performance-model.md): FP16 tensor-core peak,
# HBM bandwidth, and a conservative single-launch overhead.
TC_PEAK_FLOPS = 101.3e12
BW_BYTES_S = 535e9
LAUNCH_S = 4e-6
# 64 KiB shared memory and 65536 registers per SM on Turing.
SMEM_PER_SM = 64 * 1024
REGS_PER_SM = 65536
OCCUPANCY_BLOCKS = 2  # the measured occupancy rule


@dataclass
class Candidate:
    index: int
    name: str
    strategy: str
    bm: int
    bn: int
    bk: int
    warps: int
    smem_bytes: int
    legal: bool
    reason: str


def strategy_smem(strategy: str, bm: int, bn: int, bk: int) -> int:
    """Static shared memory per block, matching turing_search.cu."""
    half = 2
    if strategy == "staged":
        return (bm * (bk + 8) + bn * (bk + 8)) * half
    if strategy == "regdeq":
        return bm * (bk + 8) * half + 2 * bn * (bk // 8) * 4 + bn * half
    if strategy == "pipe":
        return 2 * (bm * (bk + 8) + bn * (bk + 8)) * half
    raise ValueError(strategy)


def parse_name(name: str) -> tuple[str, int, int, int, int]:
    """staged_64_128_64_w4x2 -> (staged, 64, 128, 64, 8 total warps)."""
    parts = name.split("_")
    strategy = parts[0]
    bm, bn, bk = int(parts[1]), int(parts[2]), int(parts[3])
    warps = int(parts[4][1:].split("x")[0])
    return strategy, bm, bn, bk, warps


def build_space(ext) -> list[Candidate]:
    """Parse the compiled variant table and apply the legality gates."""
    out = []
    for i in range(ext.variant_count()):
        name = ext.variant_name(i)
        strategy, bm, bn, bk, warps = parse_name(name)
        smem = strategy_smem(strategy, bm, bn, bk)
        legal, reason = True, ""
        if smem * OCCUPANCY_BLOCKS > SMEM_PER_SM:
            legal, reason = False, f"smem {smem}B x{OCCUPANCY_BLOCKS} > 64KiB"
        elif bn % 8 or bm % 16:
            legal, reason = False, "tile not m16n8-compatible"
        out.append(Candidate(i, name, strategy, bm, bn, bk, warps, smem, legal, reason))
    return out


def roofline_seconds(m: int, n: int, k: int) -> float:
    """Lower bound on kernel time from the measured machine rates."""
    traffic = (m * k + (n * k) // 2 + m * n) * 2  # A fp16 + packed u4 + out
    return max(traffic / BW_BYTES_S, 2 * m * n * k / TC_PEAK_FLOPS, LAUNCH_S)


def frontier(
    cands: list[Candidate], m: int, n: int, k: int, factor: float = 1.5
) -> list[Candidate]:
    """Keep candidates whose roofline prediction is within `factor` of the
    best-predicted candidate for this M regime. Candidates whose tile
    cannot cover the shape are dropped regardless."""
    scored = []
    for c in cands:
        if not c.legal:
            continue
        if n % c.bn:
            continue
        blocks_m = (m + c.bm - 1) // c.bm
        blocks = blocks_m * (n // c.bn)
        # wave quantization: a config whose blocks do not fill the GPU's
        # SM count evenly pays a partial wave; model it as ceil overhead
        sms = 72  # TU102
        waves = -(-blocks // sms) if blocks else 1
        t = roofline_seconds(m, n, k) * max(1.0, waves * sms / max(blocks, 1))
        scored.append((t, c))
    if not scored:
        return []
    best = min(t for t, _ in scored)
    return [c for t, c in scored if t <= factor * best]
