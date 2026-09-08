"""TP2 restart-variance probe (plan/0007 follow-up, lacunae item 1 step 1).

Rides the recorded macro-gate harness (vllm_macro_gate.py --arm, now with
an optional model argument): one fresh engine process per restart,
median-of-5 in-process reps, eager mode, clocks locked at 1455 MHz.

MODEL (operator NAK on the 1.5B gate model, 2026-09-07): the real target
- Intel-Qwen3.6-27B-int4-AutoRound, the plan/0007 integration model
(hybrid 48 GDN + 16 full-attention, d256), W4A16 on 2x TU102.

Arms: 6 restarts of BRIDGE_ATTN tp2 (the variance-flagged rows), then 2
control restarts of TRITON_ATTN tp2 (variance should be backend-
independent per the bisect record). Protocol notes: NCCL_DEBUG=INFO
SUBSYS=INIT,TUNING in the child env (init-time logging only); max_model_len
4096 per the gate harness.
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
MODEL = ("/opt/models/Qwen3.5-4B-W4A16-AR-f16/"
         "Qwen3.5-4B-w4g128")
OUTDIR = os.path.join(HERE, "..", "results", "restart-variance-20260907")

ROWS = ("short_decode", "ctx512_decode", "ctx2048_decode",
        "ctx512_mixed", "ctx2048_mixed")

ARMS = ([("BRIDGE_ATTN", i) for i in range(1, 7)] +
        [("TRITON_ATTN", i) for i in range(1, 3)])

# abort after this many consecutive failed restarts: an environment
# error must burn one restart, not the whole batch (the 2026-09-07/08
# window paid three times for the absence of this gate)
ABORT_AFTER_CONSEC_FAILS = 2


def parse_args():
    args = sys.argv[1:]
    out = {"model": MODEL, "outdir": OUTDIR, "bridge": 6, "control": 2}
    i = 0
    while i < len(args):
        if args[i] == "--model":
            out["model"] = args[i + 1]
            i += 2
        elif args[i] == "--outdir":
            out["outdir"] = args[i + 1]
            i += 2
        elif args[i] == "--bridge":
            out["bridge"] = int(args[i + 1])
            i += 2
        elif args[i] == "--control":
            out["control"] = int(args[i + 1])
            i += 2
        else:
            print(f"unknown arg {args[i]}", file=sys.stderr)
            sys.exit(2)
    return out


def parse_rows(text):
    rows = {}
    for m in re.finditer(r"SPLIT\[([^\]]+)\] (\w+)=([0-9.]+)", text):
        rows[m.group(2)] = float(m.group(3))
    return rows


def parse_nccl(text):
    facts = {}
    for pat, key in (
            (r"NCCL INFO (Using network|via|Channel [0-9]+/[0-9]+ :",
             "net"),):
        pass
    for line in text.splitlines():
        if "NCCL INFO" in line and any(
                k in line for k in ("via", "Using", "Channels", "Trees",
                                    "Algorithm", "Protocol", "connected")):
            facts.setdefault("nccl_lines", []).append(
                line.split("NCCL INFO", 1)[1].strip()[:160])
    nccl_ver = re.search(r"NCCL version ([0-9.]+)", text)
    if nccl_ver:
        facts["version"] = nccl_ver.group(1)
    return facts


def main():
    cfg = parse_args()
    global ARMS, OUTDIR
    OUTDIR = cfg["outdir"]
    ARMS = ([("BRIDGE_ATTN", i) for i in range(1, cfg["bridge"] + 1)] +
            [("TRITON_ATTN", i) for i in range(1, cfg["control"] + 1)])
    os.makedirs(OUTDIR, exist_ok=True)
    summary = {"started": time.strftime("%Y-%m-%dT%H:%M:%S"),
               "model": cfg["model"],
               "protocol": "fresh engine per restart; median-of-5 reps "
                           "in-process (gate harness); clocks 1455; "
                           "NCCL_DEBUG=INFO SUBSYS=INIT,TUNING",
               "restarts": []}
    consec_fails = 0
    for backend, i in ARMS:
        if consec_fails >= ABORT_AFTER_CONSEC_FAILS:
            print(f"ABORT: {consec_fails} consecutive failed restarts - "
                  f"environment error, not variance; fix and rerun",
                  flush=True)
            break
        tag = f"{backend}_r{i}"
        log_path = os.path.join(OUTDIR, f"{tag}.log")
        env = dict(os.environ)
        env["NCCL_DEBUG"] = "INFO"
        env["NCCL_DEBUG_SUBSYS"] = "INIT,TUNING"
        # flashinfer-python 0.6.18 has no matching cubin release (latest
        # cubin is 0.6.17); the runtime version check would kill engine
        # init. No arm here executes flashinfer kernels (BRIDGE/TRITON
        # only), so the documented bypass is set with this reason.
        env["FLASHINFER_DISABLE_VERSION_CHECK"] = "1"
        # engine workers JIT-build the bridge extension; they need the
        # toolchain root (the interface's setdefault runs in-process,
        # the worker subprocess path does not always carry it)
        env["CUDA_HOME"] = "/opt/cuda"
        t0 = time.time()
        with open(log_path, "w") as log:
            child = subprocess.Popen(
                [VENV_PY, GATE, "--arm", backend, "2", cfg["model"]],
                stdout=log, stderr=subprocess.STDOUT, env=env, cwd=HERE)
            child.wait()
        dt = time.time() - t0
        text = open(log_path, errors="replace").read()
        rows = parse_rows(text)
        entry = {"tag": tag, "backend": backend, "restart": i,
                 "exit": child.returncode, "wall_s": round(dt, 1),
                 "rows": {r: rows.get(r) for r in ROWS},
                 "nccl": parse_nccl(text)}
        summary["restarts"].append(entry)
        if child.returncode == 0:
            consec_fails = 0
        else:
            consec_fails += 1
        got = {r: rows.get(r) for r in ("ctx512_decode", "ctx2048_decode")}
        print(f"[{tag}] exit={child.returncode} wall={dt:.0f}s {got}",
              flush=True)

    print("\n=== cross-restart summary ===")
    for backend in ("BRIDGE_ATTN", "TRITON_ATTN"):
        entries = [e for e in summary["restarts"] if e["backend"] == backend
                   and e["exit"] == 0]
        for row in ("ctx512_decode", "ctx2048_decode", "short_decode"):
            vals = sorted(e["rows"].get(row) for e in entries
                          if e["rows"].get(row) is not None)
            if len(vals) >= 2:
                med = vals[len(vals) // 2]
                band = (vals[-1] - vals[0]) / med * 100.0
                print(f"{backend} {row}: n={len(vals)} vals={vals} "
                      f"median={med:.1f} band={band:.1f}%", flush=True)
    with open(os.path.join(OUTDIR, "summary.json"), "w") as f:
        json.dump(summary, f, indent=1)
    print("summary written", flush=True)


if __name__ == "__main__":
    main()
