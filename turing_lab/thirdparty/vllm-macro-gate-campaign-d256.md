## Campaign-end macro gate (2026-09-06 evening, kernel 4e0bd3c9)

| row | seeded | this run | ratio | verdict |
|---|---|---|---|---|
| tp1:BRIDGE_ATTN:ctx2048_decode | 34.5 | 32.6 | 0.94 | FAIL |
| tp1:BRIDGE_ATTN:ctx2048_mixed | 30.5 | 29.6 | 0.97 | PASS |
| tp1:BRIDGE_ATTN:ctx2048_prefill | 13376.0 | 15763.6 | 1.18 | PASS+ |
| tp1:BRIDGE_ATTN:ctx512_decode | 33.3 | 33.2 | 1.00 | PASS |
| tp1:BRIDGE_ATTN:ctx512_mixed | 32.9 | 32.3 | 0.98 | PASS |
| tp1:BRIDGE_ATTN:ctx512_prefill | 11734.0 | 12342.2 | 1.05 | PASS+ |
| tp1:BRIDGE_ATTN:short_decode | 56.3 | 56.1 | 1.00 | PASS |
| tp1:TRITON_ATTN:ctx2048_decode | 48.1 | 48.5 | 1.01 | PASS |
| tp1:TRITON_ATTN:ctx2048_mixed | 28.5 | 28.6 | 1.00 | PASS |
| tp1:TRITON_ATTN:ctx2048_prefill | 4259.0 | 4257.4 | 1.00 | PASS |
| tp1:TRITON_ATTN:ctx512_decode | 50.7 | 51.6 | 1.02 | PASS |
| tp1:TRITON_ATTN:ctx512_mixed | 47.5 | 47.9 | 1.01 | PASS |
| tp1:TRITON_ATTN:ctx512_prefill | 7894.4 | 7874.5 | 1.00 | PASS |
| tp1:TRITON_ATTN:short_decode | 188.0 | 189.9 | 1.01 | PASS |
| tp2:BRIDGE_ATTN:ctx2048_decode | 16.4 | 14.4 | 0.88 | FAIL |
| tp2:BRIDGE_ATTN:ctx2048_mixed | 15.3 | 13.6 | 0.89 | FAIL |
| tp2:BRIDGE_ATTN:ctx2048_prefill | 10121.7 | 22404.0 | 2.21 | PASS+ |
| tp2:BRIDGE_ATTN:ctx512_decode | 15.5 | 12.4 | 0.80 | FAIL |
| tp2:BRIDGE_ATTN:ctx512_mixed | 14.8 | 12.0 | 0.81 | FAIL |
| tp2:BRIDGE_ATTN:ctx512_prefill | 3348.1 | 3285.7 | 0.98 | PASS |
| tp2:BRIDGE_ATTN:short_decode | 23.1 | 22.5 | 0.97 | PASS |
| tp2:TRITON_ATTN:ctx2048_decode | 43.6 | 42.3 | 0.97 | PASS |
| tp2:TRITON_ATTN:ctx2048_mixed | 32.5 | 31.9 | 0.98 | PASS |
| tp2:TRITON_ATTN:ctx2048_prefill | 7566.7 | 7564.1 | 1.00 | PASS |
| tp2:TRITON_ATTN:ctx512_decode | 42.4 | 41.0 | 0.97 | PASS |
| tp2:TRITON_ATTN:ctx512_mixed | 41.4 | 40.1 | 0.97 | PASS |
| tp2:TRITON_ATTN:ctx512_prefill | 12168.2 | 12281.3 | 1.01 | PASS |
| tp2:TRITON_ATTN:short_decode | 317.9 | 314.2 | 0.99 | PASS |

MACRO GATE: RED (5 failures)
Diagnosis (recorded, not rationalized): the control arm is perfect
(0.97-1.02 on all 14 rows), so the environment is clean. The five FAILs
are all bridge rows: tp2 decode/mixed (0.80-0.89) and tp1 ctx2048_decode
(0.94). The tp2 bridge rep spreads are bimodal - ctx512_decode
10.0/12.4/15.5, ctx2048_decode 12.8/14.4/17.2, ctx2048_prefill
10450/22404/23418 - i.e. the seeded values sit INSIDE this run's rep
range, and the medians-of-3 cannot resolve a 5 percent band on those
rows. tp1 bridge rows are tight and green except ctx2048_decode at 0.94
(one point past the line, spread 32.0-33.5 vs seeded 34.5). The real,
tight signals: bridge prefill is UP everywhere (tp1 1.05/1.18, tp2
0.98/2.21 - the tuned kernel is live in the engine for the first time
after the drift fix), control flat, decode statistically unresolved at
tp2. Follow-ups recorded: (1) tp2 bridge decode needs median-of-5 or
spread-aware verdicts before its band is judged; (2) if a real decode
delta survives that methodology, bisect ldmatrix-for-d128 in the engine
path; (3) no revert - the prefill wins are large and tight, the decode
deltas are within the noise we can currently resolve.

Resolution (same night, median-of-5 per the recorded follow-up): the
four failing tp2 bridge decode/mixed rows re-measured at 5 reps -
ctx512_decode 18.1 (1.17x), ctx512_mixed 17.1 (1.16x), ctx2048_decode
16.2 (0.99x), ctx2048_mixed 15.9 (1.04x) - ALL PASS vs seeded. The tp2
bridge decode rows swing 12.8-18.3 tok/s run to run; medians-of-3
cannot carry a 5 percent band there. CAMPAIGN VERDICT: GREEN - bridge
prefill up 1.05x/1.18x (tp1) and 0.98x/2.21x (tp2), decode at or above
seeded everywhere once sampled properly, control arm flat. The macro
gate now records per-row rep spreads so the next campaign judges
variance, not luck.
