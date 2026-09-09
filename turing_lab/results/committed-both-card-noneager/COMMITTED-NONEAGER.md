# The non-eager committed run (2026-09-09): CUDA graphs, the floor reached

Same protocol as the committed record (27B W4A16 TP2, median of 3 fresh
restarts, backend+KV asserted, zero-preemption, clocks locked) with one
variable: enforce_eager=False (CUDA graphs on), via the env-gated gate
arm (COMMITTED_NO_EAGER=1; the eager default is unchanged).

| arm | row | eager | non-eager | gain | band (3 restarts) |
|---|---|---|---|---|---|
| TRITON | ctx512 decode | 12.7 | **50.1** | 3.94x | 0.0 pct (50.1/50.1/50.1) |
| TRITON | ctx2048 decode | 12.7 | 47.8 | 3.76x | 1.0 pct |
| TRITON | short decode (batch 8) | 90.9 | 292.4 | 3.22x | 0.1 pct |
| BRIDGE | ctx512 decode | 11.1 | 27.3 | 2.46x | 9.2 pct |
| BRIDGE | ctx2048 decode | 11.1 | 17.3 | 1.56x | 27.2 pct |

FLOOR: TRITON ctx512 at 50.1 tok/s = 0.90 of the 55.4 tok/s derived
single-stream no-MTP ceiling (9 GiB/card over the measured 535 GB/s) -
and the ceiling does not yet subtract the TP2 allreduce share, so the
effective attainment is higher. The corpus prediction (per-step launch
overhead dominating the weight-stream ceiling; graphs reclaim it) is
confirmed quantitatively. The variance story sharpens: TRITON+graphs is
restart-DEAD-FLAT (0.0 percent); the bridge arm's engine cost is now
isolated as ~23 tok/s of ctx512 headroom (27.3 vs 50.1) plus the
persisting wide bands (9-27 percent) - the per-request gather loop,
next campaign's surgical target.

KV note: graph capture memory reduces the KV pool (161-175K tokens vs
191K eager) - recorded per restart in the logs; admission held (zero
PREEMPTED).

The floor-anchored acceptance is effectively MET on the TRITON backend.
The with-MTP measurement (graphs x native mtp K=3) follows this record;
the eager-measured 1.75x projects ~87 tok/s ctx512-class.

## MTP x graphs (the final measurement, 2026-09-09)

Native mtp K=3 under CUDA graphs on the 27B: greedy-lossless (5/5),
canaries clean, but NO COMPOUND - mixed-row speedup 0.93x (ctx512) /
1.11x (ctx2048) versus the graphs-only base. The eager-world 1.75x was
largely buying back the same launch latency the graphs already
reclaimed; the levers overlap, they do not stack. The 87 tok/s
projection is falsified for this configuration. Past-the-floor via MTP
needs a mechanism that survives graphs (deeper K, acceptance-aware
scheduling, or MTP reserved to prefill-heavy regimes) - recorded as the
follow-up, with both regimes' real numbers kept separate.
