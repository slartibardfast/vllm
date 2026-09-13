# Short-row investigation state (2026-09-13, plan/0009)

Facts banked so far:

1. The recorded cost REPRODUCES: fresh incumbent baseline today
   195.5 / 41.6 / 40.2 (medians-of-3, labels asserted, zero
   preemptions) - short exactly at the recorded 195.5.
2. The kernel commit is the CAUSE, same-day A/B: with only the walk
   kernel swapped to the pre-surgery version (8b26330508's), the
   committed protocol returns short 216.1 (211.1/216.1) - the
   pre-surgery class - while mid/long fall to 35.8/35.1 (the
   surgery's +20 pct given back). Both directions reproduce.
3. ISOLATED (oracle harness, single GPU, eager, (8, 24, 4) grid,
   uniform sl): the post-surgery kernel is 2.4-3x FASTER than pre at
   the short shapes (0.099 ms vs 0.270 ms at sl 74), bit-identical
   outputs. The engine inverts this. Mechanism OPEN: candidates
   ruled out are the combine kernel (untouched by the commit), the
   host wrapper and workspace (untouched), and build flags (engine
   and oracle compile identically; the engine's cached extension
   verified rebuilt per arm).
4. ncu of the incumbent at the ENGINE short shape (12Q/2KV, ragged
   sl, batch 8, PPS2): latency-bound on an empty GPU - 0.17 waves,
   L1 31 pct, DRAM 9 pct, SM 13 pct, 53 us duration. The wall is the
   per-token staging/sync/scoring chain, not bandwidth - which the
   half2 staging and KV pipelining surgeries (plan/0009) attack
   directly.
5. step-time arithmetic: the -8.4 pct short delta is ~3.6 ms per
   41 ms step - far larger than the 16 attention-kernel instances
   can account for directly (~1.6 ms total). The inversion therefore
   lives in step-level interaction (graph scheduling, NCCL overlap,
   or launch-count effects), not in kernel duration. The eager
   per-kernel profiler could not attribute device time in this
   kineto build (two attempts, zero kernel rows); a chrome-trace
   diff remains the follow-up if the bandwidth surgeries do not
   recover the row.

Disposition: proceed to the half2-staging gate train (its committed
A/B doubles as the short-row acceptance run: short recovers toward
the 213 class while mid/long hold their bands).

## Verdict (2026-09-13, plan/0009 #short-row): terminal, cost stands

The falsification chain is complete:

1. The trimmed-staging class cannot recover the row: the half2
   surgery (the named staging trim, bank-conflict-free, minus 16 pct
   isolated at the short shape) produced a committed A/B NO_DIFF
   (193.1 vs 195.5, bands overlapping) - see the ledger entry.
2. The sensitivity floor explains why: only multi-ms per-step kernel
   deltas move the engine rows (the register surgery: ~16 ms/step =
   +20 pct; half2: ~0.1 ms/step = invisible). The projected ceiling
   for any further staging/sync trim, including the pipelining
   surgery's one-sync-per-token, is the same sub-band class.
3. The regression itself is an engine-context inversion: the
   post-surgery kernel is 3x FASTER isolated at the short shape yet
   the engine step is ~3.6 ms slower - a step-level effect (graph
   scheduling, NCCL overlap, or launch interaction) that no
   kernel-internal surgery addresses. The eager per-kernel profiler
   could not attribute device time in this kineto build; the
   chrome-trace diff is the named follow-up for a future window.

Disposition: the -8.4 pct short-row cost of the champion stands,
recorded honestly against its +20 pct mid/long gains (the ledger's
inner-loop WIN entry). Recovery attempts closed: staging trim
(half2, this window). The single-row fast path was not authored:
the pre-kernel arm IS the single-row-class path and its engine cost
profile (mid/long -14 pct) is the wrong trade. Queue moves on.
