
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
