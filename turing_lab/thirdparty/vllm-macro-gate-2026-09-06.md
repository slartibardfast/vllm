| row | baseline | this run | ratio | verdict |
|---|---|---|---|---|
| tp1:BRIDGE_ATTN:ctx2048_decode | 34.5 | - | seeded |
| tp1:BRIDGE_ATTN:ctx2048_mixed | 28.0 | 30.5 | 1.09 | PASS+ |
| tp1:BRIDGE_ATTN:ctx2048_prefill | 13376.0 | - | seeded |
| tp1:BRIDGE_ATTN:ctx512_decode | 33.3 | - | seeded |
| tp1:BRIDGE_ATTN:ctx512_mixed | 30.1 | 32.9 | 1.09 | PASS+ |
| tp1:BRIDGE_ATTN:ctx512_prefill | 11734.0 | - | seeded |
| tp1:BRIDGE_ATTN:short_decode | 56.1 | 56.3 | 1.00 | PASS+ |
| tp1:TRITON_ATTN:ctx2048_decode | 48.1 | - | seeded |
| tp1:TRITON_ATTN:ctx2048_mixed | 28.0 | 28.5 | 1.02 | PASS+ |
| tp1:TRITON_ATTN:ctx2048_prefill | 4259.0 | - | seeded |
| tp1:TRITON_ATTN:ctx512_decode | 50.7 | - | seeded |
| tp1:TRITON_ATTN:ctx512_mixed | 42.6 | 47.5 | 1.12 | PASS+ |
| tp1:TRITON_ATTN:ctx512_prefill | 7894.4 | - | seeded |
| tp1:TRITON_ATTN:short_decode | 124.5 | 188.0 | 1.51 | INVALID-ENV |
| tp2:BRIDGE_ATTN:ctx2048_decode | 16.4 | - | seeded |
| tp2:BRIDGE_ATTN:ctx2048_mixed | 14.4 | 15.3 | 1.06 | PASS+ |
| tp2:BRIDGE_ATTN:ctx2048_prefill | 10121.7 | - | seeded |
| tp2:BRIDGE_ATTN:ctx512_decode | 15.5 | - | seeded |
| tp2:BRIDGE_ATTN:ctx512_mixed | 19.0 | 14.8 | 0.78 | FAIL |
| tp2:BRIDGE_ATTN:ctx512_prefill | 3348.1 | - | seeded |
| tp2:BRIDGE_ATTN:short_decode | 24.2 | 23.1 | 0.95 | PASS |
| tp2:TRITON_ATTN:ctx2048_decode | 43.6 | - | seeded |
| tp2:TRITON_ATTN:ctx2048_mixed | 33.1 | 32.5 | 0.98 | PASS+ |
| tp2:TRITON_ATTN:ctx2048_prefill | 7566.7 | - | seeded |
| tp2:TRITON_ATTN:ctx512_decode | 42.4 | - | seeded |
| tp2:TRITON_ATTN:ctx512_mixed | 41.4 | 41.4 | 1.00 | PASS+ |
| tp2:TRITON_ATTN:ctx512_prefill | 12168.2 | - | seeded |
| tp2:TRITON_ATTN:short_decode | 212.4 | 317.9 | 1.50 | INVALID-ENV |
