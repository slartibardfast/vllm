import sys
import time

from vllm import LLM, SamplingParams


def main():
    backend = sys.argv[1] if len(sys.argv) > 1 else None
    model = ("/home/dconnolly/.cache/huggingface/hub/models--Qwen--"
             "Qwen2.5-1.5B-Instruct/snapshots/"
             "989aa7980e4cf806f80c7fef2b1adb7bc71aa306")
    kw = dict(model=model, dtype="float16", tensor_parallel_size=2,
              gpu_memory_utilization=0.90, max_model_len=4096,
              enforce_eager=True)
    if backend:
        kw["attention_backend"] = backend
    tag = backend or "auto"
    llm = LLM(**kw)
    sp = SamplingParams(max_tokens=64, temperature=0.0, ignore_eos=True)
    llm.generate(["warmup"], sp)
    t0 = time.perf_counter()
    out = llm.generate(["Explain flash attention in one paragraph."] * 8, sp)
    dt = time.perf_counter() - t0
    toks = sum(len(o.outputs[0].token_ids) for o in out)
    print(f"TP2AB[{tag}] SHORT tokens={toks} time={dt:.2f}s "
          f"tok/s={toks/dt:.1f}", flush=True)
    for ctx in (512, 2048):
        sp2 = SamplingParams(max_tokens=32, temperature=0.0, ignore_eos=True)
        t0 = time.perf_counter()
        out = llm.generate(["word " * ctx], sp2)
        dt = time.perf_counter() - t0
        toks = sum(len(o.outputs[0].token_ids) for o in out)
        print(f"TP2AB[{tag}] CTX ctx={ctx} time={dt:.2f}s tok/s={toks/dt:.1f}",
              flush=True)
    print(f"TP2AB[{tag}] DONE", flush=True)


if __name__ == "__main__":
    main()
