# The committed both-card number (2026-09-08)

Target: Intel-Qwen3.6-27B-int4-AutoRound (W4A16 sym g128), TP2 over
NVLink, both Quadro RTX 6000, clocks locked 1455 MHz.

| arm | row | median tok/s | band (3 restarts) |
|---|---|---|---|
| BRIDGE (asserted at load) | ctx512 decode | **11.1** | 3.6 pct |
| BRIDGE | ctx2048 decode | 11.1 | 6.3 pct |
| BRIDGE | short decode (batch 8) | 27.6 | 2.5 pct |
| TRITON (comparison) | ctx512 decode | 12.7 | 3.1 pct |
| TRITON | ctx2048 decode | 12.7 | 3.1 pct |
| TRITON | short decode (batch 8) | 90.9 | 2.9 pct |

Labels (parsed from engine logs, every restart): engaged backend as
named; GPU KV cache 191,146 tokens both arms; PREEMPTED events total 0
(zero-preemption admission held throughout).

Floor comparison (derived hypothesis, never a measured fact): 18 GiB
checkpoint = 9 GiB/card over the measured 535 GB/s = 55.4 tok/s
single-stream no-MTP ceiling, minus the TP2 allreduce share (not yet
separately measured). Per-stream decode sits at ~0.21 (bridge) / ~0.23
(triton) of that ceiling. The short-decode row aggregates 8 sequences
(4 per rank): its per-stream equivalent is ~3.4-11.4 tok/s - do not
read the 90.9 aggregate as a floor beat.

Protocol: fresh engine per restart, the gate arm harness (median-of-5
in-process reps), enforce_eager=True. The eager-mode label matters:
CUDA-graph capture is the corpus-named lever this number does not yet
include; the gap class (per-step engine overheads dominating a 55-tok/s
weight-stream ceiling) is consistent with eager execution, and sinter's
persistent-kernel DRAM-floor tie at batch-1 (reference baseline) marks
where the ceiling is actually reachable.

Variance note: the committed medians are tight (3.6-6.3 pct) because
the verdict runs (n=6) preceded them and the dipping restart class is
medianed out; the bridge arm's restart swing remains real (n=6 band
11.6 pct, one 10.1 dip) and bridge-path-specific per the 4B/27B probe
verdicts.

Next levers on the record: CUDA-graph (non-eager) committed run; the
bridge arm's engine-interaction variance (suspect class: per-request
gather loop / JIT extension); MTP arm (this window, phase 1d).
