#!/usr/bin/env python3
"""w4a16 M>=128 acceptance eval: oracle gate + regime timing (GPU-dedicated).

Builds the search extension from THIS task's turing_search.cu (the weco
mutable copy; the lane driver.cu is read-only), gates every variant on
the float64 oracle exactly as legacy_main does ((nibble - 8) * group
scale, 5 percent relative), then times M in {128, 256, 512} on the
(4096, 4096) shape: 5 warmup, median of 20 CUDA-event reps. Prints the
full per-variant table and a single summary metric: the best
oracle-clean variant's M512 TFLOP/s (the failing acceptance regime);
zero if every variant fails its oracle. Spill/SASS/occupancy legality is
NOT checked here - adoption goes through the full ladder gates per the
adopt-forked-outputs-only rule.
"""

import json
import os
import sys

import torch
from torch.utils.cpp_extension import load

HERE = os.path.dirname(os.path.abspath(__file__))
LANE = ("/opt/repo/agentic-vllm/software/vllm/sm75-marlin/"
        "turing_lab/search")
SRC = os.path.join(HERE, "turing_search.cu")
DRIVER = os.path.join(LANE, "driver.cu")

MS = (128, 256, 512)
NZS = (1, 2)
WARMUP, REPS = 5, 20


def main():
    ext = load(name="w4a16_eval",
               sources=[DRIVER, SRC],
               extra_cuda_cflags=["-arch=sm_75", "-O3"], verbose=False)
    torch.manual_seed(0)
    dev = "cuda"
    N, K, G = 4096, 4096, 128
    q = torch.randint(0, 16, (N, K), dtype=torch.int32, device=dev)
    Q = (q[:, 0::8] | q[:, 1::8] << 4 | q[:, 2::8] << 8 |
         q[:, 3::8] << 12 | q[:, 4::8] << 16 | q[:, 5::8] << 20 |
         q[:, 6::8] << 24 | q[:, 7::8] << 28).contiguous()
    S = torch.rand(K // G, N, dtype=torch.float16, device=dev) * 0.1 + 0.05
    W64 = (((torch.stack([(Q >> (i * 4)) & 0xF for i in range(8)], 2)
             .reshape(N, K).double() - 8)
            * S.repeat_interleave(G, 0).double().T))
    table = []

    def sweep(name, launcher):
        entry = {"variant": name, "ok": True, "rows": []}
        best = {}
        for M in MS:
            A = torch.randn(M, K, dtype=torch.float16, device=dev)
            ref = A.double() @ W64.T
            for nz in NZS:
                out = launcher(A, nz)
                torch.cuda.synchronize()
                err = (out.double() - ref).abs().max().item()
                ok = err < 0.05 * max(ref.abs().max().item(), 1e-6)
                entry["ok"] &= ok
                for _ in range(WARMUP):
                    launcher(A, nz)
                torch.cuda.synchronize()
                ts = []
                for _ in range(REPS):
                    s, e = (torch.cuda.Event(True),
                            torch.cuda.Event(True))
                    s.record()
                    launcher(A, nz)
                    e.record()
                    torch.cuda.synchronize()
                    ts.append(s.elapsed_time(e))
                ts.sort()
                med = ts[len(ts) // 2]
                tf = 2 * M * N * K / (med * 1e-3) / 1e12
                entry["rows"].append({"M": M, "nz": nz,
                                      "ms": round(med, 3),
                                      "tflops": round(tf, 2),
                                      "err_ok": ok})
                best[M] = max(best.get(M, 0.0), tf)
        table.append(entry)
        flag = "OK " if entry["ok"] else "BAD"
        print(f"{flag} {name:34s} " +
              " ".join(f"M{m}={best[m]:6.2f}" for m in MS) +
              " (best of nz 1/2)", flush=True)

    for vi in range(ext.variant_count()):
        sweep(ext.variant_name(vi),
              lambda A, nz, vi=vi: ext.launch(vi, A, Q, S, None, G, nz)[0])

    ext2 = load(name="w4a16_opt_eval",
                sources=[os.path.join(HERE, "turing_w4a16_opt.cu")],
                extra_cuda_cflags=["-arch=sm_75", "-O3"], verbose=False)
    def opt_splitk(A, nz):
        # atomics contract: single zeroed (M, N) fp32 plane, slices
        # accumulate, then the cast (k_to_half equivalent)
        WS = torch.zeros(A.size(0), N, dtype=torch.float32, device=dev)
        for i in range(nz):
            ext2.w4a16_opt_splitk(A, Q, S, WS, G, i, nz)
        return WS.half()

    for name, fn in (
            ("opt1", lambda A, nz: ext2.w4a16_opt(A, Q, S, G)),
            ("opt2", lambda A, nz: ext2.w4a16_opt2(A, Q, S, G)),
            ("opt2_splitk", opt_splitk)):
        sweep(name, fn)

    clean = [e for e in table if e["ok"]]
    metric = max((r["tflops"] for e in clean
                  for r in e["rows"] if r["M"] == 512), default=0.0)
    print(f"SUMMARY metric={metric:.2f} "
          f"(best oracle-clean M512 TFLOP/s)", flush=True)
    with open(os.path.join(HERE, "eval-table.json"), "w") as f:
        json.dump({"table": table, "metric": metric}, f, indent=1)


if __name__ == "__main__":
    sys.exit(main())
