## Throughput (industry format; all values tokens/second, medians of 3)

| backend | GPUs | pp512 t/s | pp2048 t/s | tg512 t/s | tg2048 t/s | tg64x4 batch t/s |
|---|---|---|---|---|---|---|---|
| BRIDGE_ATTN | 1 | 11,906 | 13,408 | 33.3 | 32.4 | 53.9 |
| BRIDGE_ATTN | 2 | 3,303 | 10,068 | 14.1 | 14.0 | 22.7 |
| TRITON_ATTN | 1 | 7,681 | 4,241 | 50.9 | 47.7 | 191.4 |
| TRITON_ATTN | 2 | 12,000 | 7,501 | 39.7 | 42.6 | 301.7 |

## Verdicts vs baseline

| row | baseline | this run | ratio | verdict |
|---|---|---|---|---|
| tp1:BRIDGE_ATTN:ctx2048_decode | 34.5 | 32.4 | 0.94 | VARIANCE-BOUND |
| tp1:BRIDGE_ATTN:ctx2048_mixed | 30.5 | 28.8 | 0.94 | VARIANCE-BOUND |
| tp1:BRIDGE_ATTN:ctx2048_prefill | 13376.0 | 13408.0 | 1.00 | PASS+ |
| tp1:BRIDGE_ATTN:ctx512_decode | 33.3 | 33.3 | 1.00 | PASS+ |
| tp1:BRIDGE_ATTN:ctx512_mixed | 32.9 | 32.8 | 1.00 | PASS+ |
| tp1:BRIDGE_ATTN:ctx512_prefill | 11734.0 | 11906.3 | 1.01 | PASS+ |
| tp1:BRIDGE_ATTN:short_decode | 56.3 | 53.9 | 0.96 | PASS |
| tp1:TRITON_ATTN:ctx2048_decode | 48.1 | 47.7 | 0.99 | PASS+ |
| tp1:TRITON_ATTN:ctx2048_mixed | 28.5 | 28.3 | 0.99 | PASS+ |
| tp1:TRITON_ATTN:ctx2048_prefill | 4259.0 | 4241.2 | 1.00 | PASS+ |
| tp1:TRITON_ATTN:ctx512_decode | 50.7 | 50.9 | 1.00 | PASS+ |
| tp1:TRITON_ATTN:ctx512_mixed | 47.5 | 47.4 | 1.00 | PASS+ |
| tp1:TRITON_ATTN:ctx512_prefill | 7894.4 | 7681.3 | 0.97 | PASS |
| tp1:TRITON_ATTN:short_decode | 188.0 | 191.4 | 1.02 | PASS+ |
| tp2:BRIDGE_ATTN:ctx2048_decode | 16.4 | 14.0 | 0.85 | VARIANCE-BOUND |
| tp2:BRIDGE_ATTN:ctx2048_mixed | 15.3 | 13.3 | 0.87 | VARIANCE-BOUND |
| tp2:BRIDGE_ATTN:ctx2048_prefill | 10121.7 | 10068.2 | 0.99 | PASS+ |
| tp2:BRIDGE_ATTN:ctx512_decode | 15.5 | 14.1 | 0.91 | VARIANCE-BOUND |
| tp2:BRIDGE_ATTN:ctx512_mixed | 14.8 | 14.1 | 0.95 | PASS |
| tp2:BRIDGE_ATTN:ctx512_prefill | 3348.1 | 3303.2 | 0.99 | PASS+ |
| tp2:BRIDGE_ATTN:short_decode | 23.1 | 22.7 | 0.98 | PASS+ |
| tp2:TRITON_ATTN:ctx2048_decode | 43.6 | 42.6 | 0.98 | PASS |
| tp2:TRITON_ATTN:ctx2048_mixed | 32.5 | 32.0 | 0.98 | PASS+ |
| tp2:TRITON_ATTN:ctx2048_prefill | 7566.7 | 7501.3 | 0.99 | PASS+ |
| tp2:TRITON_ATTN:ctx512_decode | 42.4 | 39.7 | 0.94 | PASS |
| tp2:TRITON_ATTN:ctx512_mixed | 41.4 | 38.9 | 0.94 | PASS |
| tp2:TRITON_ATTN:ctx512_prefill | 12168.2 | 11999.9 | 0.99 | PASS+ |
| tp2:TRITON_ATTN:short_decode | 317.9 | 301.7 | 0.95 | PASS |
