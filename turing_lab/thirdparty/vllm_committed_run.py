#!/usr/bin/env python3
"""The committed both-card run (Phase 1c): the goal's decisive number.

3 fresh engine restarts of the 27B W4A16 target at TP2, the gate arm
protocol per restart (median-of-5 in-process reps), and the acceptance
assertions parsed from each child's engine log:
  - engaged attention backend (the run is labeled by what actually ran)
  - KV cache size + dtype at load (the #47549 silent-fallback class)
  - zero PREEMPTED events (V1 recompute-only: admission must hold)
Output: committed.json + a transcript with the floor comparison. The
floor is recorded as a derived hypothesis per the numbers rule
(checkpoint GiB per card over the measured 535 GB/s), never as a
measured fact.
"""

import json
import os
import re
import subprocess
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
VENV_PY = os.path.abspath(
    os.path.join(HERE, "..", "..", ".venv", "bin", "python"))
GATE = os.path.join(HERE, "vllm_macro_gate.py")
MODEL = "/opt/models/Intel-Qwen3.6-27B-int4-AutoRound"
OUTDIR = os.path.join(HERE, "..", "results",
                       "committed-both-card-noneager")
if os.environ.get("COMMITTED_NO_EAGER") != "1":
    OUTDIR = os.path.join(HERE, "..", "results",
                           "committed-both-card")
CHECKPOINT_GIB = 18.0   # du, 2026-09-08; per card = 9.0 under TP2
MEASURED_GB_S = 535.0   # TU102 paper, pure-read protocol

ROWS = ("short_decode", "ctx512_decode", "ctx2048_decode")


def parse_rows(text):
    return {m.group(2): float(m.group(3))
            for m in re.finditer(r"SPLIT\[([^\]]+)\] (\w+)=([0-9.]+)",
                                 text)}


def parse_assertions(text):
    a = {}
    m = re.search(r"Using AttentionBackendEnum\.(\w+) backend", text)
    a["backend"] = m.group(1) if m else None
    m = re.search(r"GPU KV cache size: ([0-9,]+) tokens", text)
    a["kv_tokens"] = m.group(1) if m else None
    for pat, key in ((r"kv cache dtype[: ]+(\S+)", "kv_dtype"),
                     (r"KV cache dtype[: ]+(\S+)", "kv_dtype")):
        m = re.search(pat, text, re.I)
        if m:
            a["kv_dtype"] = m.group(1)
            break
    a["preempted_events"] = len(re.findall(r"PREEMPTED", text))
    return a


def main():
    backend = sys.argv[1] if len(sys.argv) > 1 else "BRIDGE_ATTN"
    restarts = int(sys.argv[2]) if len(sys.argv) > 2 else 3
    os.makedirs(OUTDIR, exist_ok=True)
    floor_tps = CHECKPOINT_GIB / 2 * (1 << 30) / (MEASURED_GB_S * 1e9)
    summary = {"started": time.strftime("%Y-%m-%dT%H:%M:%S"),
               "model": MODEL, "backend": backend,
               "protocol": "fresh engine per restart; median-of-5 reps "
                           "in-process; clocks locked 1455; assertions "
                           "parsed from engine logs",
               "floor_hypothesis_tps_nomtp": round(1.0 / floor_tps, 1),
               "restarts": []}
    consec_fails = 0
    for i in range(1, restarts + 1):
        if consec_fails >= 2:
            print("ABORT: consecutive failures - environment", flush=True)
            break
        log_path = os.path.join(OUTDIR, f"{backend}_r{i}.log")
        env = dict(os.environ)
        env["NCCL_DEBUG"] = "INFO"
        env["NCCL_DEBUG_SUBSYS"] = "INIT,TUNING"
        env["FLASHINFER_DISABLE_VERSION_CHECK"] = "1"
        env["CUDA_HOME"] = "/opt/cuda"
        for kv in (os.environ.get("COMMITTED_ARM_ENV") or "").split(","):
            if "=" in kv:
                k, v = kv.split("=", 1)
                env[k.strip()] = v.strip()
        import os as _o
        if _o.environ.get("COMMITTED_NO_EAGER") == "1":
            env["COMMITTED_NO_EAGER"] = "1"
        t0 = time.time()
        with open(log_path, "w") as log:
            child = subprocess.Popen(
                [VENV_PY, GATE, "--arm", backend, "2", MODEL],
                stdout=log, stderr=subprocess.STDOUT, env=env, cwd=HERE)
            child.wait()
        dt = time.time() - t0
        text = open(log_path, errors="replace").read()
        rows = parse_rows(text)
        entry = {"restart": i, "exit": child.returncode,
                 "wall_s": round(dt, 1),
                 "assertions": parse_assertions(text),
                 "rows": {r: rows.get(r) for r in ROWS}}
        summary["restarts"].append(entry)
        consec_fails = 0 if child.returncode == 0 else consec_fails + 1
        print(f"[{backend}_r{i}] exit={child.returncode} "
              f"{entry['assertions']} "
              f"{ {r: rows.get(r) for r in ROWS} }", flush=True)

    ok = [e for e in summary["restarts"] if e["exit"] == 0]
    verdict = {}
    for row in ROWS:
        vals = sorted(e["rows"].get(row) for e in ok
                      if e["rows"].get(row) is not None)
        if vals:
            med = vals[len(vals) // 2]
            verdict[row] = {"median": med,
                            "band_pct": round(
                                (vals[-1] - vals[0]) / med * 100, 1),
                            "vals": vals}
    summary["verdict"] = verdict
    backends = {e["assertions"].get("backend") for e in ok}
    preempts = sum(e["assertions"]["preempted_events"] for e in ok)
    summary["labels"] = {
        "engaged_backend(s)": sorted(b for b in backends if b),
        "preempted_events_total": preempts}
    floor = summary["floor_hypothesis_tps_nomtp"]
    if "ctx512_decode" in verdict:
        med = verdict["ctx512_decode"]["median"]
        summary["floor_ratio_ctx512"] = round(med / floor, 3)
    print(f"\nVERDICT {json.dumps(verdict, indent=None)}", flush=True)
    print(f"LABELS {json.dumps(summary['labels'])} "
          f"floor={floor} tps (derived hypothesis)", flush=True)
    with open(os.path.join(
            OUTDIR, f"committed-{backend.lower()}.json"), "w") as f:
        json.dump(summary, f, indent=1)


if __name__ == "__main__":
    main()
