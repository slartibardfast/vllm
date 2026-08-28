#!/usr/bin/env python3
"""The kernel search: oracle-check, SASS-gate, time, and select.

Modern pipeline (default): the compiled variant table is parsed into a
candidate space (space.py), filtered by the analytic model, driven
through a Bayesian multi-fidelity ladder (schedule.py), gated against
the first-generation baseline (harness.py), and emitted as a versioned
dispatch table with boundaries, attestation, and gate evidence.

Legacy pipeline (--legacy): the original fixed 8-candidate sweep with
the original minimax selection, kept for provenance.

usage: search.py [--fidelity quick|full] [--strategies staged,regdeq,pipe]
                 [--budget 24] [--seed 0] [--legacy] [--tag auto]
"""

import argparse
import json
import os
import re
import subprocess
import sys
from datetime import datetime, timezone

import torch
from torch.utils.cpp_extension import load

HERE = os.path.dirname(os.path.abspath(__file__))
NVCC = os.environ.get("NVCC", "/opt/cuda/bin/nvcc")


def build():
    r = subprocess.run(
        [
            NVCC,
            "-arch=sm_75",
            "-O3",
            "-I",
            HERE,
            "-c",
            "turing_search.cu",
            "-o",
            "turing_search.o",
        ],
        capture_output=True,
        text=True,
    )
    rc, out = r.returncode, r.stderr
    if rc:
        sys.exit(f"compile failed:\n{out[-2000:]}")
    return True


def sass_gate():
    """Register and stack use per kernel, keyed by the mangled template
    ints so the harness can match kernels to candidates."""
    out = subprocess.run(
        ["/opt/cuda/bin/cuobjdump", "-res-usage", "turing_search.o"],
        capture_output=True,
        text=True,
    )
    gates = {}
    for m in re.finditer(
        r"Function\s+(\S*ILi(\d+)ELi(\d+)ELi(\d+)ELi(\d+)ELi(\d+)E\S*):"
        r"[\s\S]*?REG:\s*(\d+)\s+STACK:\s*(\d+)",
        out.stdout,
    ):
        gates[f"{m.group(2)}_{m.group(3)}_{m.group(6)}_{m.group(7)}"] = {
            "regs": int(m.group(6)),
            "stack": int(m.group(7)),
        }
    return gates


def legacy_main(ext):
    """The original fixed sweep: full matrix, minimax relative selection."""
    torch.manual_seed(0)
    DEV = "cuda:0"
    shapes = [(4096, 4096), (4160, 4096)]
    msweeps = [1, 8, 32, 128, 512]
    table = {}
    for vi in range(ext.variant_count()):
        name = ext.variant_name(vi)
        parts = name.split("_")
        bn = int(parts[2])
        ok_all, rows = True, []
        for N, K in shapes:
            if N % bn:
                continue
            q = torch.randint(0, 16, (N, K), dtype=torch.int32, device=DEV)
            Q = (
                q[:, 0::8]
                | q[:, 1::8] << 4
                | q[:, 2::8] << 8
                | q[:, 3::8] << 12
                | q[:, 4::8] << 16
                | q[:, 5::8] << 20
                | q[:, 6::8] << 24
                | q[:, 7::8] << 28
            ).contiguous()
            S = torch.rand(K // 128, N, dtype=torch.float16, device=DEV) * 0.1 + 0.05
            for M in msweeps:
                A = torch.randn(M, K, dtype=torch.float16, device=DEV)
                out, _ = ext.launch(vi, A, Q, S, None, 128, 1)
                torch.cuda.synchronize()
                ref = (
                    A.double()
                    @ (
                        (
                            torch.stack([(Q >> (i * 4)) & 0xF for i in range(8)], 2)
                            .reshape(N, K)
                            .double()
                            - 8
                        )
                        * S.repeat_interleave(128, 0).double()
                    ).T
                ).T
                err = (out.double() - ref).abs().max().item()
                ok = err < 0.05 * max(ref.abs().max().item(), 1e-6)
                ok_all &= ok
                for _ in range(3):
                    ext.launch(vi, A, Q, S, None, 128, 1)
                torch.cuda.synchronize()
                ts = []
                for _ in range(10):
                    s, e = torch.cuda.Event(True), torch.cuda.Event(True)
                    s.record()
                    ext.launch(vi, A, Q, S, None, 128, 1)
                    e.record()
                    torch.cuda.synchronize()
                    ts.append(s.elapsed_time(e))
                ts.sort()
                med = ts[len(ts) // 2]
                tf = 2 * M * N * K / (med * 1e-3) / 1e12
                rows.append(
                    {
                        "N": N,
                        "K": K,
                        "M": M,
                        "ms": med,
                        "tflops": round(tf, 2),
                        "err_ok": ok,
                    }
                )
        table[name] = {"oracle_ok": ok_all, "n_multiple": bn, "rows": rows}
    best_by_shape = {}
    for name, d in table.items():
        for r in d["rows"]:
            key = (r["N"], r["K"], r["M"])
            if key not in best_by_shape or r["tflops"] > best_by_shape[key][1]:
                best_by_shape[key] = (name, r["tflops"])
    result = {}
    for name, d in table.items():
        worsts = [
            min(
                (r["tflops"] / best_by_shape[(r["N"], r["K"], r["M"])][1])
                for r in d["rows"]
                if (r["N"], r["K"], r["M"]) in best_by_shape
            )
        ]
        d["worst_relative"] = worsts[0] if worsts else 0.0
    for M in msweeps:
        cands = []
        for name, d in table.items():
            rels = [
                r["tflops"] / best_by_shape[(r["N"], r["K"], r["M"])][1]
                for r in d["rows"]
                if r["M"] == M and (r["N"], r["K"], r["M"]) in best_by_shape
            ]
            if rels:
                cands.append((min(rels), name))
        if cands:
            cands.sort(reverse=True)
            result[M] = {"config": cands[0][1], "worst_relative": round(cands[0][0], 3)}
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    with open(f"config-table-{stamp}.json", "w") as f:
        json.dump({"candidates": table, "table": result}, f, indent=2)
    print(json.dumps(result, indent=2))
    print(f"legacy table -> config-table-{stamp}.json")


def modern_main(args):
    from harness import Harness
    from schedule import Scheduler, emit_table, git_sha
    from space import build_space

    build()
    ext = load(
        name="turing_search_drv",
        sources=[
            os.path.join(HERE, "driver.cu"),
            os.path.join(HERE, "turing_search.cu"),
        ],
        extra_cuda_cflags=["-arch=sm_75", "-O3"],
        verbose=False,
    )
    print(f"{ext.variant_count()} candidate(s)")

    space = build_space(ext)
    legal = [c for c in space if c.legal]
    print(f"legal after model filter: {len(legal)}/{len(space)}")
    for c in space:
        if not c.legal:
            print(f"  rejected {c.name}: {c.reason}")

    expect_mhz = int(os.environ.get("TURING_EXPECT_MHZ", "1455"))
    regimes = [
        {"lo": 1, "hi": 8, "M": [1, 8]},
        {"lo": 9, "hi": 64, "M": [32, 64]},
        {"lo": 65, "hi": 192, "M": [128, 192]},
        {"lo": 193, "hi": 4096, "M": [256, 512]},
    ]
    with Harness(ext, expect_mhz=expect_mhz) as h:
        sched = Scheduler(
            h,
            ext,
            regimes,
            seed=args.seed,
            strategies=(args.strategies.split(",") if args.strategies else None),
        )
        budget = args.budget if args.fidelity == "full" else max(6, args.budget // 4)
        results = []
        for r in regimes:
            print(f"regime M in [{r['lo']}, {r['hi']}]:")
            res = sched.search_regime(r, budget=budget)
            results.append(res)
            print(f"  winner: {res.winner_name} nz={res.nz} ({res.gate})")
        # winner gate against the first-generation seeded baseline: the
        # first-generation table's winners per M regime (bm64_bn64_w8 for
        # M <= 128, bm64_bn128_w8 for M = 512), resolved by name so
        # propose.py's renumbering cannot silently retarget the gate
        by_name = {ext.variant_name(i): i for i in range(ext.variant_count())}
        first_gen_names = ["staged_64_64_64_w4x2", "staged_64_64_64_w4x2",
                           "staged_64_64_64_w4x2", "staged_64_128_64_w4x2"]
        first_gen = [(by_name[n], 1) for n in first_gen_names]
        for res, (bvi, bnz) in zip(results, first_gen):
            res = sched.winner_gate(res, ext.variant_name(bvi), bvi, bnz)
            print(f"  gated: {res.winner_name} nz={res.nz} ({res.gate})")
        bounds = sched.resolve_boundaries(results)
        sha = git_sha()
        stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        path = os.path.join(HERE, f"dispatch-table-{stamp}.json")
        table = emit_table(path, results, bounds, h.att, sha)
        print(f"dispatch table -> {path}")
        print(json.dumps(table["regimes"], indent=2))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--fidelity", choices=["quick", "full"], default="quick")
    ap.add_argument("--strategies", default=None, help="comma list: staged,regdeq,pipe")
    ap.add_argument(
        "--budget", type=int, default=24, help="TPE trials at the top rung per regime"
    )
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--legacy", action="store_true")
    args = ap.parse_args()

    if args.legacy:
        build()
        ext = load(
            name="turing_search_drv",
            sources=[
                os.path.join(HERE, "driver.cu"),
                os.path.join(HERE, "turing_search.cu"),
            ],
            extra_cuda_cflags=["-arch=sm_75", "-O3"],
            verbose=False,
        )
        legacy_main(ext)
    else:
        modern_main(args)


if __name__ == "__main__":
    main()
