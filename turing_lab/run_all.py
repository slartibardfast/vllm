#!/usr/bin/env python3
"""turing_lab runner: builds the microbenchmarks, probes the s4 MMA shape,
runs the sweep with warmup + repeats + median, and writes
results/<tag>-gpu<N>.json per card.

usage: run_all.py [--smoke] [--repeats N] [--gpus "0,1"] [--tag TAG]
--smoke uses small iteration counts; it answers "does it build and run", not
"what is the number".
"""
import argparse
import json
import os
import re
import subprocess
import sys
from datetime import datetime, timezone

HERE = os.path.dirname(os.path.abspath(__file__))
BUILD = os.path.join(HERE, "build")
RESULTS = os.path.join(HERE, "results")
NVCC = os.environ.get("NVCC", "/opt/cuda/bin/nvcc")
ARCH = os.environ.get("LAB_ARCH", "sm_75")


def sh(cmd, env=None):
    r = subprocess.run(cmd, shell=True, capture_output=True, text=True, env=env)
    return r.returncode, r.stdout.strip(), r.stderr.strip()


def build():
    os.makedirs(BUILD, exist_ok=True)
    common = ["-arch", ARCH, "-O3", "-I", HERE]
    for name in ["bench_mma", "bench_contention", "bench_memory"]:
        rc, _, err = sh(f"{NVCC} {' '.join(common)} {name}.cu -o {BUILD}/{name}")
        if rc:
            sys.exit(f"build failed for {name}:\n{err}")
    # the s4 probe is its own translation: ptxas rejecting the instruction is
    # the answer to the plan's unmeasured prerequisite
    rc, _, err = sh(f"{NVCC} {' '.join(common)} -DLAB_HAS_S4 bench_mma.cu "
                    f"-o {BUILD}/bench_mma_s4")
    return rc == 0, (err.splitlines() or [""])[-1]


def gpu_meta():
    rc, out, _ = sh("nvidia-smi --query-gpu=name,driver_version,clocks.current.graphics"
                    " --format=csv,noheader")
    return out.replace("\n", " | ") if rc == 0 else "nvidia-smi unavailable"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--repeats", type=int, default=5)
    ap.add_argument("--gpus", default="0,1")
    ap.add_argument("--tag", default=datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ"))
    args = ap.parse_args()
    mma_iters = ["2000"] if args.smoke else ["100000"]
    cont_iters = ["3000"] if args.smoke else ["30000"]
    mem_args = ["256", "20"] if args.smoke else ["1024", "100"]

    os.makedirs(RESULTS, exist_ok=True)
    s4_ok, s4_note = build()

    for gpu in [g.strip() for g in args.gpus.split(",") if g.strip()]:
        env = dict(os.environ, CUDA_VISIBLE_DEVICES=gpu)

        def meas(binary, *a):
            # one untimed warmup, then N timed runs; the median-ms run's full
            # output is the result. Short kernels and first-touch effects make
            # a single run fiction; the median of repeats is the smallest
            # honest unit.
            rc, out, err = sh(f"{binary} {' '.join(a)}", env=env)
            if rc != 0:
                return f"ERROR rc={rc}: {err.splitlines()[-1] if err else '?'}"
            rows = []
            for _ in range(args.repeats):
                rc, out, err = sh(f"{binary} {' '.join(a)}", env=env)
                if rc != 0:
                    return f"ERROR rc={rc}"
                mm = re.search(r"\bms=([0-9.]+)", out)
                rows.append((float(mm.group(1)) if mm else 1e9, out))
            rows.sort(key=lambda r: r[0])
            return rows[len(rows) // 2][1]

        m = {
            "f16_f32acc_m16n8k8": meas(f"{BUILD}/bench_mma", "f16f32", mma_iters[0]),
            "f16_f16acc_m16n8k8": meas(f"{BUILD}/bench_mma", "f16f16", mma_iters[0]),
            "int8_s8_m8n8k16": meas(f"{BUILD}/bench_mma", "s8", mma_iters[0]),
        }
        if s4_ok:
            m["int4_s4_m8n8k16"] = meas(f"{BUILD}/bench_mma_s4", "s4", mma_iters[0])
        for kind in ["hfma2", "ffma", "lop3"]:
            for k in ["0", "2", "4", "8", "16"]:
                m[f"contention_{kind}_x{k}"] = meas(f"{BUILD}/bench_contention", kind, k,
                                                    cont_iters[0])
        m["ldg_read_1024mib"] = meas(f"{BUILD}/bench_memory", *mem_args)
        m["sts_conflict_free"] = meas(f"{BUILD}/bench_shared", "sts", "10000")
        m["sts_conflicted"] = meas(f"{BUILD}/bench_shared", "sts_conflict", "10000")
        m["ldsm_x4"] = meas(f"{BUILD}/bench_shared", "ldsm", "10000")
        m["staging_chain"] = meas(f"{BUILD}/bench_shared", "staging", "10000")
        m["warp_scaling"] = meas(f"{BUILD}/bench_shared", "warp_scale", "40000")

        results = {
            "tag": args.tag, "gpu_index": gpu,
            "when": datetime.now(timezone.utc).isoformat(),
            "smoke": args.smoke, "repeats": args.repeats,
            "gpu": gpu_meta(), "arch": ARCH,
            "s4_mma_available": s4_ok,
            "s4_probe_note": s4_note if not s4_ok else "ptxas accepted m8n8k16.s4",
            "measurements": m,
        }
        jpath = os.path.join(RESULTS, f"{args.tag}-gpu{gpu}.json")
        with open(jpath, "w") as f:
            json.dump(results, f, indent=2)
        print(json.dumps(results, indent=2))
        print(f"\nwritten: {jpath}")


if __name__ == "__main__":
    main()
