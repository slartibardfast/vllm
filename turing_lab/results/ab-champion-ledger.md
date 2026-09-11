
## 2026-09-09T23:23:08 - bridge-gather vs triton-baseline (Intel-Qwen3.6-27B-int4-AutoRound, graphs=on)
- short_decode: 35.0 vs 292.4 (+735.4 pct, gate 14.0 pct) -> B_WINS
- ctx512_decode: 28.5 vs 50.1 (+75.8 pct, gate 51.4 pct) -> B_WINS
- ctx2048_decode: 25.5 vs 48.1 (+88.6 pct, gate 130.3 pct) -> NO_DIFF

## 2026-09-10T00:22:57 - bridge-gather vs bridge-batched (Intel-Qwen3.6-27B-int4-AutoRound, graphs=on)
- short_decode: 35.4 vs 7.6 (-78.5 pct, gate 15.7 pct) -> A_WINS
- ctx512_decode: 24.4 vs 24.0 (-1.6 pct, gate 51.6 pct) -> NO_DIFF
- ctx2048_decode: 14.6 vs 16.8 (+15.1 pct, gate 56.8 pct) -> NO_DIFF

## 2026-09-10T03:51:58 - bridge-gather vs bridge-paged (Intel-Qwen3.6-27B-int4-AutoRound, graphs=on)
- short_decode: 33.3 vs 215.3 (+546.5 pct, gate 26.4 pct) -> B_WINS
- ctx512_decode: 25.9 vs 12.9 (-50.2 pct, gate 76.0 pct) -> NO_DIFF
- ctx2048_decode: 21.7 vs 4.1 (-81.1 pct, gate 21.2 pct) -> A_WINS

## 2026-09-10T21:54:48 - bridge-gather vs bridge-paged-split (Intel-Qwen3.6-27B-int4-AutoRound, graphs=on)
- short_decode: 34.2 vs 201.8 (+490.1 pct, gate 64.3 pct) -> B_WINS
- ctx512_decode: 25.5 vs 26.3 (+3.1 pct, gate 72.6 pct) -> NO_DIFF
- ctx2048_decode: 16.8 vs 27.7 (+64.9 pct, gate 54.7 pct) -> B_WINS

## MTP x paged-split (2026-09-10, 72h window) — RED at gate-0

4B, paged-split backend, native mtp K=3: greedy_match 0/5, degenerate
canaries (rep4 0.333/0.082 on two prompts; base arm all-zero clean).
The 1.86x/1.61x speedups are void - outputs are wrong. Corruption
requires MTP x paged jointly (each alone is green: window-4 gate-0
8/8; MTP-gather 5/5). Prime suspect: the verify step's q_len=4 path
through the paged arm (first engine exercise of the guard) - mask
convention vs the draft-KV semantics, or multi-row output mapping.
DISPOSITION: MTP serves on the gather path (proven); the combination
stays disabled; root-cause owned by Phase D (MTP economics). The
kernel's q_len<=4 math is oracle-clean (18/18 at ql=4) - the defect,
if in the kernel, is convention-level, not math-level.

## 2026-09-10T23:24:29 - split-pps8 vs split-pps4 (Intel-Qwen3.6-27B-int4-AutoRound, graphs=on)
- short_decode: 201.5 vs 201.9 (+0.2 pct, gate 5.0 pct) -> NO_DIFF
- ctx512_decode: 25.9 vs 30.3 (+17.0 pct, gate 28.0 pct) -> NO_DIFF
- ctx2048_decode: 27.7 vs 29.5 (+6.5 pct, gate 5.0 pct) -> B_WINS

## 2026-09-10T23:56:16 - split-pps8 vs split-pps16 (Intel-Qwen3.6-27B-int4-AutoRound, graphs=on)
- short_decode: 202.9 vs 207.0 (+2.0 pct, gate 10.8 pct) -> NO_DIFF
- ctx512_decode: 28.2 vs 20.2 (-28.4 pct, gate 5.0 pct) -> A_WINS
- ctx2048_decode: 28.1 vs 20.2 (-28.1 pct, gate 6.5 pct) -> A_WINS

## 2026-09-11T00:27:25 - split-pps4 vs split-pps2 (Intel-Qwen3.6-27B-int4-AutoRound, graphs=on)
- short_decode: 207.8 vs 216.0 (+3.9 pct, gate 16.2 pct) -> NO_DIFF
- ctx512_decode: 30.3 vs 33.4 (+10.2 pct, gate 28.0 pct) -> NO_DIFF
- ctx2048_decode: 30.0 vs 35.0 (+16.7 pct, gate 11.2 pct) -> B_WINS

## 2026-09-11T00:58:26 - split-pps2 vs split-pps1 (Intel-Qwen3.6-27B-int4-AutoRound, graphs=on)
- short_decode: 218.3 vs 221.2 (+1.3 pct, gate 5.0 pct) -> NO_DIFF
- ctx512_decode: 36.7 vs 40.6 (+10.6 pct, gate 29.1 pct) -> NO_DIFF
- ctx2048_decode: 36.0 vs 34.8 (-3.3 pct, gate 7.0 pct) -> NO_DIFF

## PAGES_PER_SPLIT sweep (2026-09-11, 72h window Phase B)

Gradient monotone finer-is-better to the knee: 16 loses everywhere
(-28 pct); 4 > 8 at long ctx (+6.5 gate-cleared); 2 > 4 (+16.7
gate-cleared); 1 vs 2 splits the verdict - PPS1 owns ctx512 (40.5/40.6
dead-consistent, 0.81 of the TRITON baseline 50.1, up from 0.52),
PPS2 owns ctx2048 (35.9/36.0 vs 34.0/34.8), nothing gate-cleared
between them. CHAMPION: PPS2 by the deciding long-ctx row; PPS1 the
mid-ctx alternative. The walk-parallelism lever is exhausted at
~40/36 vs the baseline's 50.1/47.8 - the residual ~20 pct lives in
the per-token inner loop (4 barriers/token, warp-per-pair scoring),
a deeper lever recorded for future surgery.

## MTP x paged root-cause trail (2026-09-11, Phase D)

Three discriminators, isolated traffic class: (1) qlen==1-only paged +
MTP = 4/5 clean canaries, real 1.52x speedup - the draft phase is
innocent; (2) K=1 (minimal verify, qlen=2) = 0/5 - depth is not the
variable; (3) metadata dump: ordinary prefills arrive as ONE large
chunk (qlen > 4, falls to loop) while MTP's chunks are the ONLY
multi-row-paged traffic (qlen 2-5). Conclusion: MULTI-ROW PAGED
(qlen 2-4) is red, period - and the oracle cannot see it (its
fully-written page model matches the kernel's assumptions; the
engine's real cache state during those steps does not - framework
interplay, likely cache-write ordering or slot state at verify).
DISPOSITION: MTP serves on gather (proven 5/5, 1.75x/2.12x); the
combination stays disabled; the root-cause re-opens with the
narrowed frame (multi-row, K-independent, oracle-blind) next window.
The convergence run proceeds on the proven arms.
