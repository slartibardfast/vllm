# MTP record (2026-09-08): the goal's second clause delivered

Native method="mtp" (Qwen3_5MTP, the checkpoint's 1-layer MTP head,
K=3), TP2, eager, fresh engines per arm, identical configs.

| target | greedy gate | canary | ctx512 speedup | ctx2048 speedup |
|---|---|---|---|---|
| Qwen3.5-4B W4A16 (shakedown) | 5/5 match | clean, identical profiles | 1.81x (20.1 -> 36.3) | 1.54x (16.4 -> 25.3) |
| Qwen3.6-27B W4A16 (record) | 5/5 match | clean, identical profiles | 1.75x (9.9 -> 17.3) | 1.32x (7.3 -> 9.6) |

Lossless under greedy through the full hybrid depth (48 GDN + 16
full-attention layers): speculative decoding with the native head
changes no token on any probe prompt. Canary note: the two nonzero
4-gram ratios (0.044, 0.013) are identical in BOTH arms - model-
intrinsic repetition, not MTP-induced (the #53180 failure mode does
not appear; no compressed KV in this config).

Reading: the with-MTP projection against the committed TRITON baseline
is ~22 tok/s at ctx512-class decode (12.7 x 1.75), with the eager-mode
caveat carrying over. Acceptance is not directly logged by the engine
in this configuration; the speedup implies a high per-token acceptance
(consistent with the corpus's 0.75-0.90 upstream band). Next rungs:
non-eager (CUDA-graph) MTP run; direct acceptance instrumentation.
