# vLLM end-to-end correctness gate (plan/0006 pause, 2026-09-05)

Numerics before timing: every engine run before this gate counted tokens
with ignore_eos; this pause validated that the stack's OUTPUTS are
correct. Driver: `vllm_correctness_gate.py` (arms, gates, and window
construction are in the script). Clocks locked 1455 MHz both cards,
eager mode, seeds fixed, temperature 0 throughout.

## Scope

- The loadable 27B is Intel-Qwen3.6-27B-int4 (Qwen3.8-27B exists locally
  only as 52 GB fp16, which does not fit 2x24 GB, or gguf builds for
  the rope-router track). The 27B gates the W4A16 marlin + 48
  linear-attention layers + 16 full-attention (head_dim 256, TRITON
  backend) + hybrid KV allocator + quantized MTP stack.
- The bridge's end-to-end gates ride the dense Qwen2.5 models: 1.5B
  (head_dim 128, GQA 6:1) and 0.5B (head_dim 64).
- wikitext-2 test raw, ~2k-token windows, PPL via prompt_logprobs.

## Finding: a real kernel bug, caught before any speed work

The first bridge arm FAILED: all logprobs non-finite, generation
degenerate ("locklocklock..."). Root cause (instrumented-kernel
bisection): the online-softmax exponent path packed the RAW (unscaled)
S and row-max m into fp16 BEFORE subtracting. Real activations produce
raw dots past fp16 range (observed S_raw = 110,575 for a head whose
scaled score is 9,774; fp16 max is 65,504) — the packed values saturate
to inf and inf - inf = NaN. The oracle's random +-1 inputs could never
reach that regime, which is why every prior gate was green.

Fix: compute (S - m) * scale * log2e in fp32 and convert to half only
for the vectorized ex2.approx.f16x2 (the quantity is <= 0, so a large
negative flushes to -inf and exp2 gives 0 — safe). Regression-locked in
`fwd_oracle.cu` with in_scale = 100 magnitude cases (raw dots ~1e5):
suite ALL PASS. Also caught and repaired in passing: a stray
`git checkout -- .` had silently reverted the lab kernel/oracle/bench
files (the quilt tree the engine actually uses was never affected).

## Results (all gates PASS unless stated)

| arm | stack | result |
|---|---|---|
| p15_bridge | 1.5B, BRIDGE_ATTN | PPL 9.5012; sweep 512..16384 coherent; 10/10 gates |
| p15_triton | 1.5B, TRITON_ATTN | PPL 9.5009; 10/10 gates |
| p15_chunk_unchunk | 1.5B bridge, 16k budget | PPL(4k windows) 8.8123 |
| p15_chunk_cache_off | 1.5B bridge, 512-token chunks | PPL 8.8123 (bit-identical) |
| p15_chunk_cache_on | + prefix caching | PPL 8.8123 (bit-identical) |
| p05_bridge | 0.5B (d=64), BRIDGE_ATTN | PPL 13.4709 |
| p05_triton | 0.5B, TRITON_ATTN | PPL 13.4700 |
| b27_tp2 | 27B int4 TP=2 | PPL 6.7763; sweep 512..32768 coherent; 512-token decode at ctx 8k stable; all gates |

Cross-checks:

- Bridge vs TRITON PPL: 1.5B 9.5012 vs 9.5009 (0.003%); 0.5B 13.4709
  vs 13.4700 (0.007%) — well inside the 0.5% gate.
- Chunked prefill and prefix caching are numerically transparent on the
  bridge: identical PPL to four decimals vs the unchunked run (this is
  the q0/bottom-right-causal workout: 8 chunks per 4k window).
- TP=1 vs TP=2 greedy identity (27B): exact 32-token matches at ctx 512,
  2048 and 8192 (3/3). Operational note: the ctx-8192 run initially
  wedged because the prompt exactly equaled the default
  max_num_batched_tokens (no room for the decode step, single card);
  chunking the arm at 4096 resolved it — a scheduler boundary
  observation worth an upstream look, not a numerics issue.
- 27B outputs are fluent wikitext continuation at every context up to
  32768 (excerpts in the gate logs), with healthy distinct-3gram ratios
  (0.60-0.98) and zero non-finite logprobs anywhere.
- Upstream checker on the fixed quilt: VERDICT GREEN.

## Verdict

The pause ends GREEN: the correctness of the marlin+linear-attention
27B stack and of the bridge-routed dense models is established
end-to-end, and the one real defect found (the fp16-saturation NaN) is
fixed and regression-locked at three levels (C++ oracle magnitude
cases, route-level checks, engine gate). Speed work (architectural
backend fix, then the auto-research ladder over the kernel schedule
space) starts from this validated base.
