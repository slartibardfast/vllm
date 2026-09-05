import sys
import time

from vllm import LLM, SamplingParams


def main():
    backend = sys.argv[1] if len(sys.argv) > 1 else None
    model = sys.argv[2] if len(sys.argv) > 2 else \
        "/opt/models/Intel-Qwen3.6-27B-int4-AutoRound"
    from vllm.config.profiler import ProfilerConfig
    kw = dict(model=model,
              dtype="float16", gpu_memory_utilization=0.92, max_model_len=4096,
              enforce_eager=True,
              profiler_config=ProfilerConfig(profiler="torch",
                                              torch_profiler_dir="/tmp/prof_out"))
    if backend:
        kw["attention_backend"] = backend
    tag = backend or "auto"
    llm = LLM(**kw)
    sp = SamplingParams(max_tokens=64, temperature=0.0, ignore_eos=True)
    llm.generate(["warmup"], sp)

    # timed short-context batch (matches the recorded baseline protocol)
    t0 = time.perf_counter()
    out = llm.generate(["Explain flash attention in one paragraph."] * 4, sp)
    dt = time.perf_counter() - t0
    toks = sum(len(o.outputs[0].token_ids) for o in out)
    print(f"AB[{tag}] SHORT tokens={toks} time={dt:.2f}s tok/s={toks/dt:.1f}",
          flush=True)

    # profiler window: one small generate (timing excluded from the number)
    llm.start_profile()
    llm.generate(["one two three four five"], SamplingParams(
        max_tokens=8, temperature=0.0, ignore_eos=True))
    llm.stop_profile()

    # context-length scaling
    for ctx in (512, 2048):
        sp2 = SamplingParams(max_tokens=32, temperature=0.0, ignore_eos=True)
        t0 = time.perf_counter()
        out = llm.generate(["word " * ctx], sp2)
        dt = time.perf_counter() - t0
        toks = sum(len(o.outputs[0].token_ids) for o in out)
        print(f"AB[{tag}] CTX ctx={ctx} time={dt:.2f}s tok/s={toks/dt:.1f}",
              flush=True)
    print(f"AB[{tag}] DONE", flush=True)


if __name__ == "__main__":
    main()
