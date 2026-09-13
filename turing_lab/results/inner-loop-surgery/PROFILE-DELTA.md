# ncu mechanism delta: register-accumulator surgery (plan/0008)

walk kernel, champion shape (SPLIT=1, LONGCTX=1, PPS2), ncu --set full.
Pre-surgery: profile-20260913-023404 (archived as profile-pre-surgery.txt).
Post-surgery: profile-post (archived as profile-post-surgery.txt).

| metric | pre | post |
|---|---|---|
| L1/TEX throughput | 84.5 pct (the wall) | n/a — no longer the reported bound |
| DRAM throughput | 3.2 pct | 51-58 pct on the long-ctx launches |
| Duration (largest launch) | 607.7 us | 456.9 us (-25 pct; -58 pct on the mid launches) |
| Avg active threads/warp | 11.17 | 32.0 (28.5 not predicated off) |
| Warp cycles/issued instr | 13.35 | (barrier-stall OPT note gone) |
| Occupancy limiter | smem: 2 blocks/SM | registers: 4 blocks/SM (smem limit 32) |

Reading: the wall moved from an implementation artifact (shared-memory
accumulator traffic + lane-0 serialization + smem-capped occupancy) to
the DRAM path itself - the correct bound for attention. Committed
consequence: ctx512 +20.4 pct, ctx2048 +19.8 pct (band 0.0), short
-8.4 pct (fixed per-token overhead). The next lever, if ever needed,
is bandwidth-level (half2 staging against the 2-way smem bank split,
KV read pipelining), not control flow.
