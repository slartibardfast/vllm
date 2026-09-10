#!/usr/bin/env python3
"""Gate-0 for engine-path candidates: greedy equality under graphs.

The candidate's env versus the incumbent's env, same backend, same
prompts, CUDA graphs ON (the hazard class being tested: python-side
constants baked at capture must not corrupt replay). Outputs must match
exactly. Reports per-prompt match + token-level max diff.
"""

import json
import os
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
VENV_PY = os.path.abspath(
    os.path.join(HERE, "..", "..", ".venv", "bin", "python"))

PROBES = [
    "Explain why the sky is blue at sunset but not at noon, in two sentences.",
    "Write a Python function that merges two sorted lists, then explain its edge cases.",
    "Summarize the tradeoffs between tensor parallelism and pipeline parallelism.",
    "A lighthouse keeper has three lamps... continue the story for three sentences.",
    "What are the main differences between TCP and UDP, and when would you pick each?",
    "Describe the water cycle, briefly.",
    "Name three sorting algorithms and their complexities.",
    "Translate 'good morning, the ship departs at dawn' into French.",
]

CHILD = r'''
import json, sys
from vllm import LLM, SamplingParams
model, out_path, backend = sys.argv[1], sys.argv[2], sys.argv[3]
kw = dict(model=model, dtype="float16", tensor_parallel_size=2,
          gpu_memory_utilization=0.90, max_model_len=4096,
          enforce_eager=False, attention_backend=backend,
          enable_prefix_caching=False)
llm = LLM(**kw)
sp = SamplingParams(max_tokens=128, temperature=0.0, ignore_eos=True)
llm.generate(["warmup"], SamplingParams(max_tokens=8, temperature=0.0,
                                        ignore_eos=True))
outs = llm.generate(%(probes)r, sp)
texts = [o.outputs[0].text for o in outs]
ids = [list(o.outputs[0].token_ids) for o in outs]
json.dump({"texts": texts, "ids": ids}, open(out_path, "w"))
''' % {"probes": PROBES}


def run(tag, model, backend, paged):
    out = os.path.join(HERE, f"gate0-{tag}.json")
    log = os.path.join(HERE, f"gate0-{tag}.log")
    env = dict(os.environ)
    env.update({"FLASHINFER_DISABLE_VERSION_CHECK": "1",
                "CUDA_HOME": "/opt/cuda",
                "COMMITTED_NO_EAGER": "1",
                # the candidate flag IS the A: base=0, cand=1 (parent env
                # does not leak the toggle - the script owns it)
                "BRIDGE_BATCHED_GATHER": "0",
                "BRIDGE_PAGED_DECODE":
                    ("1" if paged else "0")})
    with open(log, "w") as lf:
        c = subprocess.Popen([VENV_PY, "-c", CHILD, model, out, backend],
                             stdout=lf, stderr=subprocess.STDOUT, env=env)
        c.wait()
    if c.returncode != 0 or not os.path.exists(out):
        return None, c.returncode
    return json.load(open(out)), c.returncode


def main():
    model = sys.argv[1]
    backend = sys.argv[2] if len(sys.argv) > 2 else "BRIDGE_ATTN"
    # triangulation mode: --cross BACKEND runs base=BACKEND (paged off)
    # vs cand=backend-with-paged-on when they differ
    cross = sys.argv[3] if len(sys.argv) > 3 else None
    if cross:
        base, rc0 = run("base", model, cross, False)
        cand, rc1 = run("cand", model, backend, True)
    else:
        base, rc0 = run("base", model, backend, False)
        cand, rc1 = run("cand", model, backend, True)
    report = {"model": model, "backend": backend,
              "base_rc": rc0, "cand_rc": rc1}
    if base and cand:
        exact = sum(a == b for a, b in zip(base["texts"],
                                           cand["texts"]))
        tok_diff = 0
        for ia, ib in zip(base["ids"], cand["ids"]):
            for j, (x, y) in enumerate(zip(ia, ib)):
                if x != y:
                    tok_diff += 1
            tok_diff += abs(len(ia) - len(ib))
        report["greedy_match"] = f"{exact}/{len(PROBES)}"
        report["token_diffs"] = tok_diff
    print(json.dumps(report, indent=1), flush=True)


if __name__ == "__main__":
    main()
