
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
