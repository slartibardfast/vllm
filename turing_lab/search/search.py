#!/usr/bin/env python3
"""The kernel search: oracle-check, SASS-gate, time, and select.

Pipeline per candidate: compile (the ptxas gate), one oracle spot check, the
SASS gate (register count and spills from cuobjdump -res-usage), then timing
over the adversarial M set against N=4096 K=4096 and N=4160 K=4032 (a
64-multiple with awkward aspect). Selection is minimax: the config whose
WORST case over the shape set is best wins its regime; the emitted table is
the generated Turing configuration table (D5).
"""
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
        [NVCC, "-arch=sm_75", "-O3", "-I", HERE, "-c", "turing_search.cu",
         "-o", "turing_search.o"], capture_output=True, text=True)
    rc, out = r.returncode, r.stderr
    if rc:
        sys.exit(f"compile failed:\n{out[-2000:]}")
    return True


def sass_gate():
    """Register and stack use per kernel, keyed by the mangled template ints
    (BM_BN_BK_WR_WN) so the harness can match kernels to candidates."""
    out = subprocess.run(["/opt/cuda/bin/cuobjdump", "-res-usage",
                          "turing_search.o"], capture_output=True, text=True).stdout
    gates = {}
    for m in re.finditer(
            r"Function\s+(\S*ILi(\d+)ELi(\d+)ELi(\d+)ELi(\d+)ELi(\d+)E\S*):"
            r"[\s\S]*?REG:\s*(\d+)\s+STACK:\s*(\d+)", out):
        bm, bn, bk, wr, wn = m.groups()[1:6]
        gates[f"{bm}_{bn}_{wr}_{wn}"] = {"regs": int(m.group(6)),
                                         "stack": int(m.group(7))}
    return gates


def main():
    build()
    # the compiled object carries the variants; bind through a second load of
    # the same source so the python side can drive launches
    search_ext = load(name="turing_search_drv",
                      sources=[os.path.join(HERE, "driver.cu"),
                               os.path.join(HERE, "turing_search.cu")],
                      extra_cuda_cflags=["-arch=sm_75", "-O3"], verbose=False)
    nv = search_ext.variant_count()
    print(f"{nv} candidate(s)")
    gates = sass_gate()

    torch.manual_seed(0)
    DEV = "cuda:0"
    shapes = [(4096, 4096), (4160, 4096)]
    msweeps = [1, 8, 32, 128, 512]

    table = {}
    for vi in range(nv):
        name = search_ext.variant_name(vi)
        parts = name.split("_")
        bm_i, bn_i, warps = int(parts[0][2:]), int(parts[1][2:]), int(parts[2][1:])
        wr = bm_i // 16
        wn = warps // wr
        g = gates.get(f"{bm_i}_{bn_i}_{wr}_{wn}", {})
        bn = bn_i
        ok_all, rows = True, []
        for (N, K) in shapes:
            if N % bn:
                # the config's own applicability constraint, the same
                # divisibility rule the incumbent's is_valid_config enforces
                continue
            q = torch.randint(0, 16, (N, K), dtype=torch.int32, device=DEV)
            Q = (q[:, 0::8] | q[:, 1::8] << 4 | q[:, 2::8] << 8 | q[:, 3::8] << 12
                 | q[:, 4::8] << 16 | q[:, 5::8] << 20 | q[:, 6::8] << 24
                 | q[:, 7::8] << 28).contiguous()
            S = torch.rand(K // 128, N, dtype=torch.float16, device=DEV) * 0.1 + 0.05
            for M in msweeps:
                A = torch.randn(M, K, dtype=torch.float16, device=DEV)
                out = search_ext.launch(vi, A, Q, S, None, 128)
                torch.cuda.synchronize()
                ref = (A.double() @ ((torch.stack([(Q >> (i * 4)) & 0xF for i in range(8)],
                      2).reshape(N, K).double() - 8) * S.repeat_interleave(128, 0).double().T).T)
                err = (out.double() - ref).abs().max().item()
                ok = err < 0.05 * max(ref.abs().max().item(), 1e-6)
                ok_all &= ok
                for _ in range(3):
                    search_ext.launch(vi, A, Q, S, None, 128)
                torch.cuda.synchronize()
                ts = []
                for _ in range(10):
                    s, e = torch.cuda.Event(True), torch.cuda.Event(True)
                    s.record(); search_ext.launch(vi, A, Q, S, None, 128); e.record()
                    torch.cuda.synchronize()
                    ts.append(s.elapsed_time(e))
                ts.sort()
                med = ts[len(ts) // 2]
                tf = 2 * M * N * K / (med * 1e-3) / 1e12
                rows.append({"N": N, "K": K, "M": M, "ms": med, "tflops": round(tf, 2),
                             "err_ok": ok})
        table[name] = {"oracle_ok": ok_all, "n_multiple": bn, "sass": g,
                       "rows": rows}

    # relative scoring pass (needs every candidate's rows): each candidate's
    # TFLOP/s over the best seen for the same shape, so the memory-bound
    # M=1 rows cannot dominate the minimax
    best_by_shape = {}
    for t in table.values():
        for r in t["rows"]:
            key = (r["N"], r["K"], r["M"])
            best_by_shape[key] = max(best_by_shape.get(key, 0.0), r["tflops"])
    for name, t in table.items():
        t["worst_tflops"] = round(min((r["tflops"] for r in t["rows"]), default=0.0), 2)
        rels = [r["tflops"] / best_by_shape[(r["N"], r["K"], r["M"])]
                for r in t["rows"] if t["oracle_ok"]]
        t["worst_relative"] = round(min(rels), 3) if rels else 0.0
        print(f"{name}: worst={t['worst_tflops']:.2f} TFLOP/s "
              f"oracle={'ok' if t['oracle_ok'] else 'FAIL'} "
              f"worst_rel={t['worst_relative']} sass={t['sass']}")

    # minimax selection per M regime: the config with the best worst-case
    # RELATIVE performance across the shape set it covers
    print("\ngenerated configuration table (minimax over the shape set):")
    result = {}
    for M in msweeps:
        best = None
        for name, t in table.items():
            if not t["oracle_ok"] or "worst_relative" not in t:
                continue
            rel = min(r["tflops"] / best_by_shape[(r["N"], r["K"], r["M"])]
                      for r in t["rows"] if r["M"] == M)
            if best is None or rel > best[1]:
                best = (name, rel)
        result[M] = {"config": best[0], "worst_relative": round(best[1], 3)}
        print(f"  M={M:4d} -> {best[0]} (worst-case relative {best[1]:.3f})")

    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    with open(os.path.join(HERE, f"config-table-{stamp}.json"), "w") as f:
        json.dump({"candidates": table, "table": result}, f, indent=2)
    print(f"written: config-table-{stamp}.json")


if __name__ == "__main__":
    main()
