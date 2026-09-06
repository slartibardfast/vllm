import sys
import time

from vllm import LLM, SamplingParams

# Prefill/decode-split A/B for the sm_75 attention backends (2026-09-06,
# post-tuning kernel). Protocol matches the recorded vllm_bridge_ab /
# vllm_tp2_ab rows (same model, kwargs, prompts) with the timing split:
#   PREFILL-ONLY: max_tokens=1 -> ctx prompt tokens + 1 decode step
#   MIXED:        max_tokens=32 -> same prompt + 32 decode steps
#   DECODE-PHASE: 31 / (t_mixed - t_prefill)  (both contain 1 decode step,
#                 the difference is decode steps 2..32 at full context)
# Run under gpu-lease: --mode gpu0 for tp=1, --mode exclusive for tp=2.

MODEL = ("/home/dconnolly/.cache/huggingface/hub/models--Qwen--"
         "Qwen2.5-1.5B-Instruct/snapshots/"
         "989aa7980e4cf806f80c7fef2b1adb7bc71aa306")


def main():
    backend = sys.argv[1] if len(sys.argv) > 1 and sys.argv[1] != "-" else None
    tp = int(sys.argv[2]) if len(sys.argv) > 2 else 1
    tag = f"{backend or 'auto'}-tp{tp}"
    kw = dict(model=MODEL, dtype="float16", tensor_parallel_size=tp,
              gpu_memory_utilization=0.90, max_model_len=4096,
              enforce_eager=True)
    if backend:
        kw["attention_backend"] = backend
    llm = LLM(**kw)

    n_seq = 4 * tp
    sp = SamplingParams(max_tokens=64, temperature=0.0, ignore_eos=True)
    llm.generate(["warmup"], sp)

    t0 = time.perf_counter()
    out = llm.generate(["Explain flash attention in one paragraph."] * n_seq, sp)
    dt = time.perf_counter() - t0
    toks = sum(len(o.outputs[0].token_ids) for o in out)
    print(f"SPLIT[{tag}] SHORT-DECODE tokens={toks} time={dt:.2f}s "
          f"tok/s={toks/dt:.1f}", flush=True)

    for ctx in (512, 2048):
        prompt = ["word " * ctx]
        t0 = time.perf_counter()
        llm.generate(prompt, SamplingParams(max_tokens=1, temperature=0.0,
                                            ignore_eos=True))
        t_prefill = time.perf_counter() - t0

        t0 = time.perf_counter()
        out = llm.generate(prompt, SamplingParams(max_tokens=32, temperature=0.0,
                                                  ignore_eos=True))
        t_mixed = time.perf_counter() - t0
        toks = sum(len(o.outputs[0].token_ids) for o in out)
        t_decode = t_mixed - t_prefill
        print(f"SPLIT[{tag}] CTX={ctx} prefill={ctx/t_prefill:.1f} tok/s "
              f"(incl 1 decode step) | decode={31/t_decode:.1f} tok/s "
              f"(31 steps, derived) | mixed={toks/t_mixed:.1f} tok/s",
              flush=True)
    print(f"SPLIT[{tag}] DONE", flush=True)


if __name__ == "__main__":
    main()
