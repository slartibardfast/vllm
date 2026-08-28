#!/usr/bin/env python3
"""Timing, noise floors, and statistical winner gates.

Measurement discipline, in one place:
- timing is CUDA-event, median of reps after warmup, synchronize per rep;
- the noise floor is estimated per (M, shape) by re-timing a fixed
  baseline candidate several times; sigma is the spread of medians;
- a candidate only counts as a winner if it beats the baseline by more
  than max(5%, 3 sigma) of the baseline median;
- A/B comparisons interleave the two candidates' reps to neutralize
  clock drift;
- every measurement session records a clock-lock attestation.
"""

import subprocess
from dataclasses import dataclass, field
from statistics import median

import torch


def clocks_mhz() -> int:
    out = subprocess.run(
        [
            "nvidia-smi",
            "--query-gpu=clocks.current.graphics",
            "--format=csv,noheader,nounits",
        ],
        capture_output=True,
        text=True,
    )
    return int(out.stdout.strip().splitlines()[0])


@dataclass
class Attestation:
    pre_mhz: int
    post_mhz: int
    expect_mhz: int
    ok: bool
    note: str = ""


@dataclass
class Timing:
    median_ms: float
    samples_ms: list = field(default_factory=list)


class Harness:
    def __init__(self, ext, expect_mhz: int = 1455):
        self.ext = ext
        self.expect_mhz = expect_mhz
        self.att = Attestation(clocks_mhz(), 0, expect_mhz, False)

    def __enter__(self):
        self.att.pre_mhz = clocks_mhz()
        return self

    def __exit__(self, *exc):
        self.att.post_mhz = clocks_mhz()
        self.att.ok = (
            self.att.pre_mhz == self.expect_mhz and self.att.post_mhz == self.expect_mhz
        )
        return False

    def time_variant(
        self,
        vi: int,
        A,
        Q,
        S,
        empty,
        G: int,
        nz: int = 1,
        reps: int = 10,
        warmup: int = 3,
    ) -> Timing:
        for _ in range(warmup):
            self.ext.launch(vi, A, Q, S, empty, G, nz)
        torch.cuda.synchronize()
        ts = []
        for _ in range(reps):
            s, e = torch.cuda.Event(True), torch.cuda.Event(True)
            s.record()
            self.ext.launch(vi, A, Q, S, empty, G, nz)
            e.record()
            torch.cuda.synchronize()
            ts.append(s.elapsed_time(e))
        return Timing(median(sorted(ts)), ts)

    def noise_floor(
        self,
        vi: int,
        A,
        Q,
        S,
        empty,
        G: int,
        nz: int,
        sessions: int = 5,
        reps: int = 10,
    ) -> float:
        """Relative sigma of the median across repeated sessions."""
        meds = [
            self.time_variant(vi, A, Q, S, empty, G, nz, reps=reps).median_ms
            for _ in range(sessions)
        ]
        mu = median(meds)
        if mu <= 0:
            return 1.0
        spread = (max(meds) - min(meds)) / 2
        return spread / mu

    def beats(
        self, t_winner: Timing, t_baseline: Timing, sigma_rel: float
    ) -> tuple[bool, str]:
        """Winner gate: faster by more than max(5%, 3 sigma) of baseline."""
        margin = max(0.05, 3 * sigma_rel) * t_baseline.median_ms
        delta = t_baseline.median_ms - t_winner.median_ms
        ok = delta > margin
        return ok, (
            f"delta {delta:.4f} ms vs required {margin:.4f} ms"
            f" (baseline {t_baseline.median_ms:.4f},"
            f" winner {t_winner.median_ms:.4f})"
        )

    def pair_ab(
        self, vi_a: int, vi_b: int, A, Q, S, empty, G: int, nz: int, reps: int = 10
    ) -> tuple[Timing, Timing]:
        """Interleaved A/B/A/B timing against clock drift."""
        ta, tb = [], []
        for _ in range(reps):
            for vi, acc in ((vi_a, ta), (vi_b, tb)):
                s, e = torch.cuda.Event(True), torch.cuda.Event(True)
                s.record()
                self.ext.launch(vi, A, Q, S, empty, G, nz)
                e.record()
                torch.cuda.synchronize()
                acc.append(s.elapsed_time(e))
        return Timing(median(sorted(ta)), ta), Timing(median(sorted(tb)), tb)


def make_problem(
    N: int, K: int, M: int, G: int = 128, seed: int = 0, dev: str = "cuda:0"
):
    """Packed W4A16 GPTQ-symmetric problem: A (M,K), Q (N, K/8) int32
    nibble-packed, S (K/G, N) fp16. Returns (A, Q, S, empty, ref_fn)."""
    torch.manual_seed(seed)
    w = torch.randn(N, K, dtype=torch.float16, device=dev) * 0.05
    s_ng = (w.view(N, K // G, G).abs().amax(dim=2) / 7.0).clamp(min=1e-8).half()
    S = s_ng.T.contiguous()
    q = (
        torch.clamp(
            torch.round(w.view(N, K // G, G).double() / s_ng.double().unsqueeze(2)) + 8,
            0,
            15,
        )
        .to(torch.int32)
        .view(N, K)
    )
    packed = torch.zeros(K // 8, N, dtype=torch.int32, device=dev)
    for j in range(8):
        packed |= (q[:, j::8] & 0xF).T << (4 * j)
    Q = packed.T.contiguous()
    A = torch.randn(M, K, dtype=torch.float16, device=dev)
    empty = torch.empty(0, dtype=torch.int32, device=dev)

    def ref_fn():
        w_deq = (q.double() - 8.0) * S.repeat_interleave(G, 0).double().T
        return A.double() @ w_deq.T

    return A, Q, S, empty, ref_fn


def oracle_ok(
    ext, vi: int, A, Q, S, empty, G: int, nz: int, ref_fn, tol_rel: float = 0.05
) -> tuple[bool, float]:
    out, _ = ext.launch(vi, A, Q, S, empty, G, nz)
    ref = ref_fn()
    err = (out.double() - ref).abs().max().item()
    return err < tol_rel * max(ref.abs().max().item(), 1e-6), err
