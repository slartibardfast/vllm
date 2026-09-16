# The nsys decode-step ledger (2026-09-16, plan/0010) - 16-step window, ctx512, dev0

Raw profile trees (nsys-*/: step.nsys-rep, step.sqlite, ~0.5-1.3 GB
each) are kept on disk, gitignored (GitHub 100 MB limit). This file
is the distilled ledger; the queries are reproducible from the
sqlite (CUPTI_ACTIVITY_KIND_KERNEL, filter to the last 16 decode
steps, deviceId=0).

Arm: BRIDGE_ATTN, champion env (paged-split PPS2, graphs), 27B TP2.
Step span 383.9 ms / 16 steps = 24.0 ms/step (matches the committed
41.3 tok/s row - the trace is honest). Kernel-busy 46 pct - 54 pct
of each step is inter-kernel gaps.

| item | per-step | note |
|---|---|---|
| bridge_paged_split_walk | 6.96 ms (x10.4, 666 us avg) | grid z=196: the block table is max_model_len-wide; at ctx512 ~180 of 196 splits own zero pages and each still emits ~6 KB of zero partials the combine never reads |
| lm-head GEMM (turing_fp16) | ~3.0 ms (x0.6, ~3 ms each) | 150B-vocab tensor GEMM at tiny M, ~5x over its byte time; SHARED with the TRITON arm - part of the floor gap, not the bridge gap |
| GDN fused_recurrent | 1.61 ms (x31, 52 us avg) | the 24 GDN layers' decode path |
| bridge_paged_split_combine | 0.29 ms (x10.4, 28 us avg) | correctly skips inactive splits |
| nccl allreduce | (RING_LL seen at prefill sizes; decode sizes ride vllm::cross_device_reduce_1stage - the custom one-shot IS dispatching) | |
| elementwise/index soup | ~1.5 ms | spread thin |

Findings: (1) the walk kernel's in-engine duration is 4x its isolated
duration purely from empty-split waste - fixed by the empty-split
guard (this tree, the new champion 42.3/40.2/202.6). (2) The floor
gap (champion 24.0 ms vs the 18.0-18.3 ms floor) is dominated by
items the TRITON arm shares: the LM-head GEMM and the 54 pct idle
gaps - candidates for a floor-focused follow-on, not bridge work.
(3) TRITON-arm ledger captured beside it (nsys-TRITON_ATTN/) for the
same 16-step slice.
