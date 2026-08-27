#!/usr/bin/env python3
"""turing_lab runner: builds the microbenchmarks, probes the s4 MMA shape,
runs the sweep, and writes results/<tag>.json plus a readable table.

usage: run_all.py [--smoke] [--tag TAG]
--smoke uses small iteration counts; it answers "does it build and run", not
"what is the number".
"""
import argparse
import json
import os
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
    # bench_mma gates the s4 path on the flag, so a successful build means the
    # instruction exists for this arch; a rejection means it does not.
    return rc == 0, (err.splitlines() or [""])[-1]


def gpu_meta():
    rc, out, _ = sh("nvidia-smi --query-gpu=name,driver_version,clocks.current.graphics"
                    " --format=csv,noheader")
    return out.replace("\n", " | ") if rc == 0 else "nvidia-smi unavailable"


def run(binary, *args):
    rc, out, err = sh(f"{binary} {' '.join(args)}")
    return out if rc == 0 else f"ERROR rc={rc}: {err.splitlines()[-1] if err else '?'}"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--tag", default=datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ"))
    args = ap.parse_args()
    scale = ["2000"] if args.smoke else ["20000"]

    os.makedirs(RESULTS, exist_ok=True)
    s4_ok, s4_note = build()
    results = {
        "tag": args.tag,
        "when": datetime.now(timezone.utc).isoformat(),
        "smoke": args.smoke,
        "gpu": gpu_meta(),
        "arch": ARCH,
        "s4_mma_available": s4_ok,
        "s4_probe_note": s4_note if not s4_ok else "ptxas accepted m8n8k16.s4",
        "measurements": {},
    }
    m = results["measurements"]
    m["f16_f32acc_m16n8k16"] = run(f"{BUILD}/bench_mma", "f16f32", scale[0])
    m["f16_f16acc_m16n8k8"] = run(f"{BUILD}/bench_mma", "f16f16", scale[0])
    m["int8_s8_m8n8k16"] = run(f"{BUILD}/bench_mma", "s8", scale[0])
    if s4_ok:
        m["int4_s4_m8n8k16"] = run(f"{BUILD}/bench_mma_s4", "s4", scale[0])
    for kind in ["hfma2", "ffma", "lop3"]:
        for k in ["0", "4", "8", "16"]:
            m[f"contention_{kind}_x{k}"] = run(f"{BUILD}/bench_contention", kind, k,
                                               "3000" if args.smoke else "10000")
    m["ldg_read_256mib"] = run(f"{BUILD}/bench_memory", "256", "20" if args.smoke else "50")

    jpath = os.path.join(RESULTS, f"{args.tag}.json")
    with open(jpath, "w") as f:
        json.dump(results, f, indent=2)
    print(json.dumps(results, indent=2))
    print(f"\nwritten: {jpath}")


if __name__ == "__main__":
    main()
