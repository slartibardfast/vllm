# Multi-row paged red: root cause closed (2026-09-13, plan/0008)

The MTP-x-paged red (ledger 2026-09-10: 0/5 greedy, degenerate
canaries, "corruption requires MTP x paged jointly") is a query-layout
bug in the bridge_attn.py glue, not in the kernel, the cache, or the
draft. CLOSED with a two-line fix and green gates.

## Mechanism

The paged kernels' contract is q of layout (req, head, q_row, d). The
glue handed them `query[:n*qlen].view(n, H, qlen, D)` - a flat
REINTERPRETATION of the engine's token-major (req, q_row, head, d)
rows. Every q_len > 1 step therefore attended scrambled query
vectors (the kernel itself is exact: the oracle validates it against
an independent fp64 reference, and the brute-force matrix on a dumped
verify step reproduces its output to fp16 noise under its own reading).
At q_len == 1 the view degenerates to the true layout, which is why
every committed decode number was green while every multi-row step was
red. K-independence, draft-exoneration, oracle-blindness and the clean
seq_lens / cache writes all follow.

Two auxiliary one-line defects were fixed in the same pass: the output
side flat-viewed the kernel's (req, head, q_row, d) result back into
(req, q_row, head, d) rows (same scramble class, output direction).

## Evidence trail

- 774-step dual log (BRIDGE_PAGED_DUAL): deterministic divergence on
  every verify step pre-fix; seq_lens device-vs-CPU identical in all
  steps (cache-state hypothesis dead); draft-tail K norms equal head
  norms (write-ordering hypothesis dead).
- One-shot raw dump (/tmp pattern, BRIDGE_PAGED_DUMP) + brute-force
  fp64 matrix: kernel == reference on all rows under its own input
  reading (maxerr 0.0075, fp16 noise); token-major reference diverges
  on rows >= 1 exactly as the engine did. Row 0's apparent exactness
  was a q-independent artifact (dumped step had sl == qlen, so row 0's
  span was a single position: output = that position's V).
- Post-fix discriminator (4B, MTP K=1, paged-split PPS=2, eager):
  greedy_match 5/5, zero first divergences, identical output lengths;
  774 dual-log steps, max worst diff 0.031 (accumulation-order
  tie-break class, the window-5 adjudicated benign family).

## Fix (lane, this commit)

- input side: `query[:n*qlen].view(n, qlen, H, D).transpose(1,2)
  .contiguous()` - repack to the kernel's contract, never flat-view.
- output side: `out_p.transpose(1, 2).reshape(n*qlen, H, D)` - regroup
  the kernel's rows to token order.
- instrumentation retained, env-gated off by default:
  BRIDGE_PAGED_DUAL / BRIDGE_PAGED_DUAL_LOG / BRIDGE_PAGED_DUMP
  (plan/0008 root-cause tooling).

Unblocks: MTP verification rides the paged-split champion path (the
plan/0008 K-sweep runs decode+verify on the champion configuration).
