#!/usr/bin/env python3
"""plan/0010 nsys ledger child: ctx512 decode window, champion config."""
def main():
    from vllm import LLM, SamplingParams
    llm = LLM(model="/opt/models/Intel-Qwen3.6-27B-int4-AutoRound",
              dtype="float16", tensor_parallel_size=2,
              gpu_memory_utilization=0.90, max_model_len=4096,
              enforce_eager=False, enable_prefix_caching=False)
    llm.generate(["warmup"], SamplingParams(max_tokens=8,
                 temperature=0.0, ignore_eos=True))
    p = "word " * 512
    for _ in range(2):
        llm.generate([p] * 8, SamplingParams(max_tokens=48,
                     temperature=0.0, ignore_eos=True))
if __name__ == "__main__":
    main()
