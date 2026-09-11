# Dual-lane mutation ledger (2026-09-08 window, local pattern)

Protocol: agent-driven mutation loops through the identical oracle-
gated evals (the local weco pattern - no cloud, no login), one surgical
edit per round, judged against seated baselines (opt 23.54 / search
18.39) under the winner gate max(5 percent, 3 sigma).

| lane | round | mutation | M512 result | verdict |
|---|---|---|---|---|
| search | 1 | staged_64_128_64 w4x2 -> w4x4 | 18.39 -> 18.88 (+2.7 pct) | REAL but sub-gate: three repeats 18.88/18.89/18.88 (sigma ~0.005) vs baseline family 18.32-18.45 - non-overlapping distributions, below the 5 pct adoption bar. KEPT as candidate in the task dir; not adopted. |
| search | 2 | regdeq_64_128_64 w4x2 -> w4x4 | 18.12 -> 12.87 (-29 pct) | FALSIFIED: the register wall is causally live - regdeq rides it at w4x2 and the wider unroll collapses occupancy exactly as the red-line predicts. Reverted. |
| search | 4 | staged_64_64_64 w4x2 -> w4x4 | 15.51 -> 13.31 (-14 pct) | FALSIFIED: the unroll lever is TILE-CONDITIONAL (+2.7 at BN128, -14 at BN64). Reverted. |
| opt | 1 | k_opt2 WN2 2 -> 4 (staging ILP) | 23.52 -> 23.54 (flat) | FALSIFIED: staging unroll is not the bottleneck - the double buffer already hides staging latency. Reverted. |
| opt | 2 | k_opt2 BN2 128 -> 64 (occupancy play) | 23.54 -> 13.10 (-44 pct) | FALSIFIED: narrower tiles starve the deep pipeline - the tensor-pipes-need-work-depth lesson, quantified. opt1 untouched (rides the outer constants); splitk flat (already small-tile). Reverted; lane verified restored at 23.55. |

Model evidence banked: the register wall (causal, -29 pct on violation),
pipeline work depth (causal, -44 pct on starvation), and unroll gains
(tile-conditional, bounded ~3 pct). Conclusion: the 23.5 -> 51-57 gap
is not reachable by config knobs at this granularity - the corpus's
named surgery (3+ stage pipeline, register-resident dequant with the
repack interlace) is the remaining path, and the knob space around the
current optimum is now mapped.

## Phase C addendum (2026-09-11, 72h window): the pipeline thesis falsified twice

k_opt3 (true 2-buffer software pipeline, authored clean, oracle-green):
concurrent-smem-write variant 16.66 (-29 pct vs opt2's 23.50);
post-compute-staging + L2-prefetch variant 15.13 (-36 pct). Three
structures triangulate the mechanism: the mma stream at this tile
shape (64x128x32) is SMEM-BANDWIDTH-BOUND, not latency-starved -
opt2's synchronous barrier separation IS the optimization; any
concurrent staging traffic or added barrier is pure loss. The corpus's
"deeper software pipeline" lever is falsified in this direction on
this silicon. The surviving path to 51-57: register-resident dequant
with the repack interlace (removes the weight stream from smem
entirely; days-grade surgery - host repack tooling + kernel rewrite),
scoped for a dedicated window. Phase C pivots to the bounded
alternates (INT8 probe, local loops); the code is retained in the
task dir for the record.
