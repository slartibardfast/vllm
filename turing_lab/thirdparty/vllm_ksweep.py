#!/usr/bin/env python3
"""MTP K-sweep (Track 2): where does acceptance beat draft cost under
CUDA graphs? TP1 on the 4B (single card, no allreduce confound), one
variable: K. Arms: base (no spec) + K in {1,2,3,4}; graphs on; greedy
texts compared vs base for losslessness at every K.
"""

import json
import os
import subprocess
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
VENV_PY = os.path.abspath(
    os.path.join(HERE, "..", "..", ".venv", "bin", "python"))

PROBES = [
    "Explain why the sky is blue at sunset but not at noon, in two sentences.",
    "Write a Python function that merges two sorted lists, then explain its edge cases.",
    "Summarize the tradeoffs between tensor parallelism and pipeline parallelism.",
]

CHILD = r'''
import json, sys, time
from vllm import LLM, SamplingParams
model, out_path, k = sys.argv[1], sys.argv[2], int(sys.argv[3])
kw = dict(model=model, dtype="float16", tensor_parallel_size=1,
          gpu_memory_utilization=0.90, max_model_len=4096,
          enforce_eager=False, enable_prefix_caching=False)
if k > 0:
    kw["speculative_config"] = {"method": "mtp",
                                "num_speculative_tokens": k}
llm = LLM(**kw)
sp = SamplingParams(max_tokens=128, temperature=0.0, ignore_eos=True)
llm.generate(["warmup"], SamplingParams(max_tokens=8, temperature=0.0,
                                        ignore_eos=True))
outs = llm.generate(%(probes)r, sp)
texts = [o.outputs[0].text for o in outs]
rows = {}
for ctx in (512, 2048):
    prompt = "word " * ctx
    t0 = time.perf_counter()
    o = llm.generate([prompt] * 4, SamplingParams(max_tokens=32,
                    temperature=0.0, ignore_eos=True))
    dt = time.perf_counter() - t0
    ntok = sum(len(x.outputs[0].token_ids) for x in o)
    rows[str(ctx)] = round(ntok / dt, 1)
json.dump({"texts": texts, "mixed": rows}, open(out_path, "w"))
''' % {"probes": PROBES}


def run(model, k, tag):
    out = os.path.join(HERE, f"ksweep-{tag}.json")
    env = dict(os.environ)
    env.update({"FLASHINFER_DISABLE_VERSION_CHECK": "1",
                "CUDA_HOME": "/opt/cuda", "CUDA_VISIBLE_DEVICES": "1"})
    with open(os.path.join(HERE, f"ksweep-{tag}.log"), "w") as lf:
        c = subprocess.Popen([VENV_PY, "-c", CHILD, model, out,
                              str(k)], stdout=lf, stderr=subprocess.STDOUT,
                             env=env)
        c.wait()
    if c.returncode != 0 or not os.path.exists(out):
        return None
    return json.load(open(out))


def main():
    model = sys.argv[1]
    base = run(model, 0, "k0")
    if base is None:
        print("BASE FAILED", flush=True)
        sys.exit(1)
    report = {"base_mixed": base["mixed"]}
    for k in (1, 2, 3, 4):
        r = run(model, k, f"k{k}")
        if r is None:
            report[f"k{k}"] = "FAILED"
            continue
        match = sum(a == b for a, b in zip(base["texts"], r["texts"]))
        report[f"k{k}"] = {
            "mixed": r["mixed"],
            "greedy_match": f"{match}/{len(PROBES)}",
            "speedup": {c: round(r["mixed"][c] / base["mixed"][c], 2)
                        for c in base["mixed"]}}
        print(f"K={k}: {report[f'k{k}']}", flush=True)
    print(json.dumps(report, indent=1), flush=True)
    with open(os.path.join(HERE, "ksweep-report.json"), "w") as f:
        json.dump(report, f, indent=1)


if __name__ == "__main__":
    main()
