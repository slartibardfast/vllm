#!/usr/bin/env python3
"""MTP probe (Phase 1d): the goal's 'MTP multiplying past the floor'.

The fork's native method="mtp" (Qwen3_5MTP, 1-layer head carried in
both checkpoints) at K=3, versus the no-spec baseline, on the same
engine config. Gate-0 suite, in order:
  1. GREEN-GATE: greedy (temp 0) outputs with MTP must match the
     no-spec baseline outputs on every probe prompt - speculative
     decoding is lossless under greedy; a mismatch is a numerics
     failure, not a speed question.
  2. CANARY: per-output 4-gram repetition ratio (the #53180 class:
     compressed-KV x MTP silently-accepted degenerate loops). Real
     prompts only - synthetic 'word word' prompts cannot show it.
  3. THROUGHPUT: decode tok/s with vs without (the gate row protocol).
  4. ACCEPTANCE: parsed from the engine log if reported; else derived
     from generated-token counts vs forward counts when available.
Shakedown target: the 4B fixture first; the 27B is the record.
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

PROBES = [
    "Explain why the sky is blue at sunset but not at noon, in two sentences.",
    "Write a Python function that merges two sorted lists, then explain its edge cases.",
    "Summarize the tradeoffs between tensor parallelism and pipeline parallelism.",
    "A lighthouse keeper has three lamps... continue the story for three sentences.",
    "What are the main differences between TCP and UDP, and when would you pick each?",
]

CHILD = r'''
import json, sys, time
from vllm import LLM, SamplingParams
model, out_path, use_mtp = sys.argv[1], sys.argv[2], sys.argv[3] == "mtp"
kw = dict(model=model, dtype="float16", tensor_parallel_size=2,
          gpu_memory_utilization=0.90, max_model_len=4096,
          enforce_eager=True, enable_prefix_caching=False)
import os as _o
if _o.environ.get("BRIDGE_PAGED_DECODE"):
    kw["attention_backend"] = "BRIDGE_ATTN"
import os as _o
if _o.environ.get("COMMITTED_NO_EAGER") == "1":
    kw["enforce_eager"] = False
if use_mtp:
    kw["speculative_config"] = {"method": "mtp", "num_speculative_tokens": 3}
llm = LLM(**kw)
sp = SamplingParams(max_tokens=128, temperature=0.0, ignore_eos=True)
# warmup
llm.generate(["warmup"], SamplingParams(max_tokens=8, temperature=0.0,
                                        ignore_eos=True))
outs = llm.generate(%(probes)r, sp)
texts = [o.outputs[0].text for o in outs]
# throughput: the decode row (ctx512 prefill + 31-token decode tail)
import torch
for ctx in (512, 2048):
    prompt = "word " * ctx
    t0 = time.perf_counter()
    o = llm.generate([prompt] * 1, SamplingParams(max_tokens=32,
                    temperature=0.0, ignore_eos=True))[0]
    dt = time.perf_counter() - t0
    ntok = len(o.outputs[0].token_ids)
    print(f"MTPROBE ctx{ctx} mixed={ntok/dt:.1f}", flush=True)
json.dump(texts, open(out_path, "w"))
''' % {"probes": PROBES}


def rep4_ratio(text):
    words = text.split()
    if len(words) < 8:
        return 0.0
    grams = [tuple(words[i:i + 4]) for i in range(len(words) - 3)]
    return 1.0 - len(set(grams)) / len(grams)


def run_child(tag, model, use_mtp):
    out = os.path.join(HERE, f"mtp-{tag}.json")
    log = os.path.join(HERE, f"mtp-{tag}.log")
    env = dict(os.environ)
    env["FLASHINFER_DISABLE_VERSION_CHECK"] = "1"
    env["CUDA_HOME"] = "/opt/cuda"
    with open(log, "w") as lf:
        c = subprocess.Popen([VENV_PY, "-c", CHILD, model, out,
                              "mtp" if use_mtp else "base"],
                             stdout=lf, stderr=subprocess.STDOUT, env=env)
        c.wait()
    text = open(log, errors="replace").read()
    mixed = {int(m.group(1)): float(m.group(2))
             for m in re.finditer(r"MTPROBE ctx(\d+) mixed=([0-9.]+)",
                                  text)}
    acc = re.findall(r"acceptance[^0-9]*([0-9.]+)", text, re.I)
    return (json.load(open(out)) if c.returncode == 0 and
            os.path.exists(out) else None, mixed, acc[:3], c.returncode)


def main():
    model = sys.argv[1]
    base_texts, base_mixed, _, rc0 = run_child("base", model, False)
    mtp_texts, mtp_mixed, acc, rc1 = run_child("mtp", model, True)
    report = {"model": model, "base_rc": rc0, "mtp_rc": rc1}
    if base_texts and mtp_texts:
        matches = sum(a == b for a, b in zip(base_texts, mtp_texts))
        report["greedy_match"] = f"{matches}/{len(PROBES)}"
        report["canary_rep4_base"] = [round(rep4_ratio(t), 3)
                                      for t in base_texts]
        report["canary_rep4_mtp"] = [round(rep4_ratio(t), 3)
                                     for t in mtp_texts]
        report["mixed_tok_s_base"] = base_mixed
        report["mixed_tok_s_mtp"] = mtp_mixed
        report["speedup"] = {str(k): round(mtp_mixed[k] / base_mixed[k], 2)
                             for k in base_mixed if k in mtp_mixed}
        report["acceptance_logged"] = acc
    print(json.dumps(report, indent=1), flush=True)
    with open(os.path.join(HERE, f"mtp-report.json"), "w") as f:
        json.dump(report, f, indent=1)


if __name__ == "__main__":
    main()
