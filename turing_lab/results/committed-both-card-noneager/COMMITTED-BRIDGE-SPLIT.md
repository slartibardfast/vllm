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
