#!/usr/bin/env python3
"""plan/0009 #short-row mechanism hunt: per-kernel CUDA time of the
short-decode step, eager mode (kernels exposed outside graph replay),
champion env. Run once per kernel arm (pre-surgery / post-surgery);
diff the top rows to name where engine time moved.
Usage: step_profile.py <model> <out.json>
"""
import json
import sys
import time

from vllm import LLM, SamplingParams


def main():
    model, out = sys.argv[1], sys.argv[2]
    llm = LLM(model=model, dtype="float16", tensor_parallel_size=2,
              gpu_memory_utilization=0.90, max_model_len=4096,
              enforce_eager=True, enable_prefix_caching=False,
              attention_backend="BRIDGE_ATTN")
    llm.generate(["warmup"], SamplingParams(max_tokens=8,
                                            temperature=0.0, ignore_eos=True))
    prompt = "Explain flash attention in one paragraph."
    # a few unprofiled reps to seat clocks/caches, then one profiled
    for _ in range(3):
        llm.generate([prompt] * 8, SamplingParams(max_tokens=64,
                       temperature=0.0, ignore_eos=True))
    from torch.profiler import profile, ProfilerActivity
    with profile(activities=[ProfilerActivity.CPU,
                               ProfilerActivity.CUDA]) as prof:
        t0 = time.perf_counter()
        llm.generate([prompt] * 8, SamplingParams(max_tokens=64,
                       temperature=0.0, ignore_eos=True))
        dt = time.perf_counter() - t0
    rows = []
    for e in prof.key_averages():
        us = getattr(e, "self_device_time_total", 0) or getattr(e, "device_time_total", 0)
        if us and us > 0:
            rows.append({"name": e.key[:100], "count": e.count,
                         "cuda_us": round(us, 1)})
    rows.sort(key=lambda r: -r["cuda_us"])
    json.dump({"wall_s": round(dt, 2), "top_kernels": rows[:50]},
              open(out, "w"), indent=1)
    for r in rows[:12]:
        print(f"{r['cuda_us']:>10.1f} us  x{r['count']:<5d} {r['name']}")


if __name__ == "__main__":
    main()
