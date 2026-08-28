#!/usr/bin/env python3
"""The auto-research scheduler: Bayesian sampling over a fidelity ladder.

Layers, in the order the program's doctrine applies them:
1. the analytic pre-filter (space.py) removes illegal and off-frontier
   candidates before any timing;
2. a fidelity ladder bounds cost: R0 times two M points with 3 reps,
   R1 the full M sweep on one shape with 5 reps, R2 the full adversarial
   shape set with 10 reps; a candidate advances only if it stays in the
   top tier of its regime at each rung;
3. Optuna's TPE sampler is the Bayesian layer: every measured candidate
   is told back its rung score, so the sampler concentrates subsequent
   trials on productive (strategy, tile, split-K) regions;
4. the winner gate (harness.beats) decides promotion against the seeded
   baseline - the first-generation table winner - so the emitted table
   can never regress below today's;
5. regime boundaries are resolved by bisection between adjacent winners
   with a noise-scaled hysteresis band.

The emitted dispatch table maps every M to (variant, split-K) and carries
the attestation and gate evidence alongside.
"""

import json
import subprocess
from dataclasses import dataclass, field

import optuna
from harness import Harness, make_problem, oracle_ok
from space import build_space

optuna.logging.set_verbosity(optuna.logging.WARNING)


@dataclass
class Rung:
    shapes: list  # [(N, K)]
    msweeps: list  # [M]
    reps: int


@dataclass
class RegimeResult:
    m_lo: int
    m_hi: int
    winner_index: int
    winner_name: str
    nz: int
    gate: str
    rows: list = field(default_factory=list)


def git_sha() -> str:
    try:
        return subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"], capture_output=True, text=True
        ).stdout.strip()
    except Exception:
        return "unknown"


class Scheduler:
    """One Scheduler instance drives one full search; construct inside a
    Harness context (the attestation wraps every timing it performs)."""

    RUNGS = [
        Rung(shapes=[(4096, 4096)], msweeps=[1, 512], reps=3),
        Rung(shapes=[(4096, 4096)], msweeps=[1, 8, 32, 128, 512], reps=5),
        Rung(
            shapes=[(4096, 4096), (4160, 4096)], msweeps=[1, 8, 32, 128, 512], reps=10
        ),
    ]
    PROMOTE_FRACTION = 1 / 3
    NZ_CHOICES = [1, 2, 4, 8]

    def __init__(
        self,
        harness: Harness,
        ext,
        regimes: list[dict],
        seed: int = 0,
        strategies: list[str] | None = None,
        verbose: bool = True,
    ):
        self.h = harness
        self.ext = ext
        self.regimes = regimes  # [{"lo": 1, "hi": 8, "M": [1, 8]}]
        self.seed = seed
        self.verbose = verbose
        self.space = [
            c
            for c in build_space(ext)
            if strategies is None or c.strategy in strategies
        ]
        self.table_rows = []

    def _score(
        self, vi: int, nz: int, rung: Rung, m_lo: int, m_hi: int, problems: dict
    ) -> float:
        """Worst-case TFLOP/s across the rung's (shape, M) points for this
        regime's M values - the program's minimax convention."""
        worst = float("inf")
        for N, K in rung.shapes:
            for M in rung.msweeps:
                if not (m_lo <= M <= m_hi):
                    continue
                key = (N, K, M)
                if key not in problems:
                    problems[key] = make_problem(N, K, M, seed=0)
                A, Q, S, empty, ref_fn = problems[key]
                ok, _ = oracle_ok(
                    self.ext, vi, A, Q, S, empty, G=128, nz=nz, ref_fn=ref_fn
                )
                if not ok:
                    return 0.0  # oracle failure is an instant zero
                t = self.h.time_variant(
                    vi, A, Q, S, empty, 128, nz, reps=rung.reps, warmup=2
                )
                tf = 2 * M * N * K / (t.median_ms * 1e-3) / 1e12
                worst = min(worst, tf)
        return worst

    def search_regime(self, r: dict, budget: int = 24) -> RegimeResult:
        m_lo, m_hi = r["lo"], r["hi"]
        problems: dict = {}
        legal = [c for c in self.space if c.legal]
        # split-K slices must not straddle a group boundary (G=128):
        # K is a multiple of 128 on every shape we sweep, so nz <= 8 keeps
        # slices group-aligned automatically (K/nz multiple of 128).
        nz_choices = self.NZ_CHOICES
        sampler = optuna.samplers.TPESampler(seed=self.seed)
        study = optuna.create_study(sampler=sampler, direction="maximize")
        dists = {
            "vi": optuna.distributions.IntDistribution(0, len(legal) - 1),
            "nz": optuna.distributions.CategoricalDistribution(nz_choices),
        }
        best = (0.0, 0, 1, "?")  # (score, vi, nz, name)
        told = 0
        promoted: set = set()
        for rung_i, rung in enumerate(self.RUNGS):
            pool = []
            trials_this_rung = (
                budget if rung_i == 0 else max(4, int(budget * self.PROMOTE_FRACTION))
            )
            for _ in range(trials_this_rung):
                t = study.ask(dists)
                pos = t.params["vi"]  # position in the legal list
                cand = legal[pos]
                nz = t.params["nz"]
                if rung_i > 0 and promoted and (pos, nz) not in promoted:
                    study.tell(t, 0.0)  # culled: never measured
                    continue
                score = self._score(cand.index, nz, rung, m_lo, m_hi, problems)
                study.tell(t, score)
                told += 1
                pool.append((score, pos, nz, cand.name))
                if self.verbose:
                    print(
                        f"  [{m_lo}-{m_hi}] rung{rung_i} {cand.name}"
                        f" nz={nz}: {score:.2f} TF/s"
                    )
            if not pool or pool[0][0] <= 0.0:
                continue
            best = pool[0]
            if rung_i + 1 < len(self.RUNGS):
                keep = max(3, int(len(pool) * self.PROMOTE_FRACTION))
                promoted = {(pos, nz) for _, pos, nz, _ in pool[:keep]}
                for score, pos, nz, name in pool[:keep]:
                    study.enqueue_trial({"vi": pos, "nz": nz}, skip_if_exists=True)
        score, pos, nz, name = best
        return RegimeResult(
            m_lo, m_hi, legal[pos].index, name, nz, f"TPE ladder best {score:.2f} TF/s"
        )

    def winner_gate(
        self,
        res: RegimeResult,
        baseline_name: str,
        baseline_index: int,
        baseline_nz: int,
    ) -> RegimeResult:
        """The emitted winner must beat the first-generation baseline
        beyond max(5%, 3 sigma); otherwise the baseline stays."""
        rung = self.RUNGS[-1]
        worst_sigma = 0.0
        ok_all = True
        for N, K in rung.shapes:
            for M in rung.msweeps:
                if not (res.m_lo <= M <= res.m_hi):
                    continue
                A, Q, S, empty, _ = make_problem(N, K, M, seed=1)
                tw = self.h.time_variant(
                    res.winner_index, A, Q, S, empty, 128, res.nz, reps=10
                )
                tb = self.h.time_variant(
                    baseline_index, A, Q, S, empty, 128, baseline_nz, reps=10
                )
                sigma = self.h.noise_floor(
                    baseline_index, A, Q, S, empty, 128, baseline_nz, sessions=3
                )
                worst_sigma = max(worst_sigma, sigma)
                ok, why = self.h.beats(tw, tb, sigma)
                ok_all &= ok
                if self.verbose:
                    print(
                        f"  gate M={M}: winner {tw.median_ms:.4f} vs"
                        f" baseline {tb.median_ms:.4f} -> {ok}"
                    )
        if not ok_all:
            res.winner_index = baseline_index
            res.winner_name = baseline_name
            res.nz = baseline_nz
            res.gate = (
                f"baseline retained ({res.gate} lost the gate, sigma {worst_sigma:.4f})"
            )
        else:
            res.gate += f" | beat baseline, sigma {worst_sigma:.4f}"
        return res

    def resolve_boundaries(
        self, results: list[RegimeResult], granularity: int = 8
    ) -> list[dict]:
        """Bisect the crossover M between adjacent regime winners."""
        bounds = []
        for a, b in zip(results, results[1:]):
            if a.winner_index == b.winner_index and a.nz == b.nz:
                bounds.append({"m": b.m_lo, "hysteresis": 0})
                continue
            lo, hi = a.m_lo, b.m_lo
            while hi - lo > granularity:
                mid = (lo + hi) // 2 // granularity * granularity
                A, Q, S, empty, _ = make_problem(4096, 4096, mid, seed=2)
                ta, tb = self.h.pair_ab(
                    a.winner_index, b.winner_index, A, Q, S, empty, 128, a.nz, reps=5
                )
                if tb.median_ms < ta.median_ms:
                    hi = mid
                else:
                    lo = mid
            bounds.append({"m": hi, "hysteresis": granularity})
        return bounds


def emit_table(
    path: str, results: list[RegimeResult], bounds: list[dict], att, sha: str
) -> dict:
    table = {
        "version": 1,
        "git": sha,
        "attestation": {"pre_mhz": att.pre_mhz, "post_mhz": att.post_mhz, "ok": att.ok},
        "regimes": [
            {
                "m_lo": r.m_lo,
                "m_hi": r.m_hi,
                "variant": r.winner_name,
                "index": r.winner_index,
                "nz": r.nz,
                "gate": r.gate,
            }
            for r in results
        ],
        "boundaries": bounds,
    }
    with open(path, "w") as f:
        json.dump(table, f, indent=2)
    return table
