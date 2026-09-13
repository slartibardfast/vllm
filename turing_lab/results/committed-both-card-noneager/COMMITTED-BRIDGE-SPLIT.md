# The bridge committed run, paged-split (2026-09-10, 72h window)

27B W4A16, TP2, CUDA graphs, median of 3 fresh restarts, all labels
asserted every restart: BRIDGE_ATTN engaged (paged-split path),
GPU KV 175,672 tokens, PREEMPTED total 0. The bridge arm's first
committed number on the owned kernel path.

| row | median | band (3 restarts) | gather-path history |
|---|---|---|---|
| ctx512 decode | 26.3 tok/s | 7.6 pct | 11.1 eager / 27.3 graphs, bands 9-27 pct |
| ctx2048 decode | 27.1 tok/s | 1.8 pct | collapse-era 4.1 (v1 paged) |
| short decode (batch 8) | 207.0 tok/s | 1.8 pct | 33.5 (gather) / 215 (v1 paged) |

THE VARIANCE IS ELIMINATED. The bridge arm's cross-restart swing - the
campaign's founding question (22-24 pct bands on the 4B, 9-27 pct on
the gather path) - is gone at 1.8-7.6 pct bands. Confirmed by
elimination: the variance lived in the per-request python gather
loop's engine interaction, not the kernel, not NCCL, not the
scheduler. The paged-split single-launch path is restart-stable.

Floor placement (honest): 0.47-0.49 of the derived 55.4 tok/s
single-stream ceiling. The doctrine acceptance (bridge >= the TRITON
baseline of 50.1/47.8 at mid/long ctx) remains OPEN - the Phase B
PAGES_PER_SPLIT sweep targets it. Single-stream mid-ctx is the gap;
batch short-decode (207 vs 292) and restart stability now lead the
gather path everywhere.

## Update (2026-09-11): the PPS2 champion committed run

Median of 3 fresh restarts, labels asserted every restart (BRIDGE_ATTN
engaged, KV 175,672, zero preemptions): ctx512 34.3 (band 9.0 pct),
ctx2048 34.4 (band 2.6 pct), short 213.5 (band 2.3 pct). Floor ratio
0.62 both rows (from 0.47-0.49); 0.68-0.72 of the TRITON baseline at
mid/long ctx, with the PPS1 alternative touching 0.81 at ctx512
(40.5/40.6). The bridge arc: gather 27.3 -> split 26.3 -> tuned 34.3,
restart-stable throughout. The doctrine acceptance (bridge >= baseline)
remains open at ~0.7; the named residual lever is the per-token inner
loop (4 barriers/token, warp-per-pair scoring).

## Update (2026-09-13): the register-accumulator surgery committed run — NEW STANDING CHAMPION

The plan/0008 inner-loop surgery (register-resident accumulators,
warp-distributed pairs, warp-uniform softmax, two staging syncs per
token, 1 KB smem) through the full gate train: standalone compile
clean, oracle 36/36 at PPS2 (max rel 2.54e-04, the pre-surgery class),
then medians of 3 fresh restarts, labels asserted every restart
(BRIDGE_ATTN engaged, KV 175,672, zero preemptions):

- ctx512 41.3 (band 9.9 pct) vs 34.3 = +20.4 pct
- ctx2048 41.2 (band 0.0 pct — 41.2/41.2/41.2) vs 34.4 = +19.8 pct
- short 195.5 (band 2.9 pct) vs 213.5 = -8.4 pct

Floor ratio 0.75 (from 0.62); 0.82 of the TRITON baseline at mid ctx.
The bridge arc: gather 27.3 -> split 26.3 -> tuned 34.3 -> register
accumulators 41.3. The short-row cost is real and recorded (fixed
per-token staging/broadcast overhead dominates at tiny contexts); the
mid/long rows are the deployment targets. Mechanism evidence: ncu
pre/post under .weco/c2-paged-decode/profile-20260913-* (L1/smem 84.5
pct, 11.2/32 lanes, 30.3 pct barrier stalls, smem-limited occupancy
before). Protocol transcript: results/inner-loop-surgery/committed-run.log.
