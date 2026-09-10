#!/usr/bin/env python3
"""The A/B harness (window 3): approaches versus champion.

Interleaved paired restarts (A,B,A,B,... - the bisect discipline that
neutralizes drift), fresh engine per restart, the gate arm protocol
per restart, labels asserted, winner gate max(5 percent, 3 sigma).
Arms may differ by attention backend AND by engine-path env overrides
(candidate approaches gate their code paths on env flags). Runs under
CUDA graphs by default (the campaign context; EAGER=1 env flips).

usage: vllm_ab_run.py --arm "label:BACKEND[:K=V,K=V]" (twice)
                     --model PATH [--pairs 3] [--outdir DIR]
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
LEDGER_MD = os.path.join(HERE, "..", "results", "ab-champion-ledger.md")
LEDGER_JSON = os.path.join(HERE, "..", "results", "ab-ledger.json")

ROWS = ("short_decode", "ctx512_decode", "ctx2048_decode")


def parse_arm(spec):
    parts = spec.split(":")
    label, backend = parts[0], parts[1]
    env = {}
    for kv in (parts[2].split(",") if len(parts) > 2 else []):
        if "=" in kv:
            k, v = kv.split("=", 1)
            env[k.strip()] = v.strip()
    return {"label": label, "backend": backend, "env": env}


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
    a["preempted"] = len(re.findall(r"PREEMPTED", text))
    return a


def run_restart(arm, model, outdir, i):
    tag = f"{arm['label']}_p{i}"
    log_path = os.path.join(outdir, f"{tag}.log")
    env = dict(os.environ)
    env.update({"NCCL_DEBUG": "INFO",
                "NCCL_DEBUG_SUBSYS": "INIT,TUNING",
                "FLASHINFER_DISABLE_VERSION_CHECK": "1",
                "CUDA_HOME": "/opt/cuda"})
    if os.environ.get("EAGER") != "1":
        env["COMMITTED_NO_EAGER"] = "1"
    env.update(arm["env"])
    t0 = time.time()
    with open(log_path, "w") as log:
        child = subprocess.Popen(
            [VENV_PY, GATE, "--arm", arm["backend"], "2", model],
            stdout=log, stderr=subprocess.STDOUT, env=env, cwd=HERE)
        child.wait()
    dt = time.time() - t0
    text = open(log_path, errors="replace").read()
    return {"label": arm["label"], "pair": i, "exit": child.returncode,
            "wall_s": round(dt, 1),
            "assertions": parse_assertions(text),
            "rows": {r: parse_rows(text).get(r) for r in ROWS}}


def verdict(entry):
    out = {}
    for row in ROWS:
        a = sorted(r["rows"][row] for r in entry["restarts"]
                   if r["label"] == entry["arm_a"]["label"]
                   and r["rows"][row] is not None)
        b = sorted(r["rows"][row] for r in entry["restarts"]
                   if r["label"] == entry["arm_b"]["label"]
                   and r["rows"][row] is not None)
        if not a or not b:
            continue
        ma, mb = a[len(a) // 2], b[len(b) // 2]
        spreads = [(max(v) - min(v)) / (sum(v) / len(v))
                   for v in (a, b) if len(v) > 1]
        sigma = max(spreads) if spreads else 0.0
        delta = (mb - ma) / ma * 100.0
        gate = max(5.0, 300.0 * sigma)
        out[row] = {"a_median": ma, "b_median": mb,
                    "delta_pct": round(delta, 1),
                    "gate_pct": round(gate, 1),
                    "verdict": ("B_WINS" if delta > gate else
                                "A_WINS" if delta < -gate else "NO_DIFF")}
    return out


def main():
    args = sys.argv[1:]
    specs = []
    model = None
    pairs = 3
    i = 0
    while i < len(args):
        if args[i] == "--arm":
            specs.append(args[i + 1])
            i += 2
        elif args[i] == "--model":
            model = args[i + 1]
            i += 2
        elif args[i] == "--pairs":
            pairs = int(args[i + 1])
            i += 2
        else:
            i += 1
    assert len(specs) == 2 and model, "need --arm x2 and --model"
    outdir = os.path.join(HERE, "..", "results", "ab-run-" +
                          time.strftime("%Y%m%dT%H%M%S"))
    os.makedirs(outdir, exist_ok=True)
    arm_a, arm_b = parse_arm(specs[0]), parse_arm(specs[1])
    entry = {"started": time.strftime("%Y-%m-%dT%H:%M:%S"),
             "model": model, "pairs": pairs,
             "protocol": "interleaved fresh restarts, gate arm protocol, "
                         "graphs on (EAGER env flips), winner gate "
                         "max(5 pct, 3 sigma)",
             "arm_a": arm_a, "arm_b": arm_b, "restarts": []}
    consec_fails = 0
    for i in range(1, pairs + 1):
        for arm in (arm_a, arm_b):
            if consec_fails >= 2:
                print("ABORT: consecutive failures", flush=True)
                break
            r = run_restart(arm, model, outdir, i)
            entry["restarts"].append(r)
            consec_fails = 0 if r["exit"] == 0 else consec_fails + 1
            print(f"[{r['label']}_p{i}] exit={r['exit']} "
                  f"{ {k: r['rows'][k] for k in ROWS} }", flush=True)
    v = verdict(entry)
    entry["verdict"] = v
    print(f"\nVERDICT {json.dumps(v)}", flush=True)
    js = json.load(open(LEDGER_JSON)) if os.path.exists(LEDGER_JSON) else []
    js.append(entry)
    with open(LEDGER_JSON, "w") as f:
        json.dump(js, f, indent=1)
    with open(LEDGER_MD, "a") as f:
        f.write(f"\n## {entry['started']} - {arm_a['label']} vs "
                f"{arm_b['label']} ({os.path.basename(model)}, "
                f"graphs={'off' if os.environ.get('EAGER') == '1' else 'on'})\n")
        for row, d in v.items():
            f.write(f"- {row}: {d['a_median']} vs {d['b_median']} "
                    f"({d['delta_pct']:+.1f} pct, gate {d['gate_pct']} pct)"
                    f" -> {d['verdict']}\n")
    print("ledger appended", flush=True)


if __name__ == "__main__":
    main()
