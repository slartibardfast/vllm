
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

## MTP K-tree on the champion path (2026-09-13, plan/0008) — GREEN, non-compounding — supersedes the RED entry above

- question: with the multi-row paged red root-caused to the glue's
  q-layout view (see MULTIROW-PAGED-ROOT-CAUSE.md) and fixed, does MTP
  compound on the 27B champion path, and at which K?
- config: decode paged-split PPS2 + verify gather + CUDA graphs,
  Intel-Qwen3.6-27B, native mtp, 3 fresh restarts per K, medians of the
  probe's mixed rows (results/mtp-champion-sweep/, summary.json).
- K1: 1.06x ctx512 (24.8 vs 23.4) / 0.87x ctx2048 (11.5 vs 13.2)
- K2: 0.86x ctx512 (20.2 vs 23.6) / 0.93x ctx2048 (12.3 vs 13.2)
- K3 (banked, convergence run): 1.12x / 1.07x
- VERDICT: NO_COMPOUND on the 27B — every K lands within 0.86-1.12;
  the draft cost scales with the model and eats the acceptance gain.
  The 4B (2.12x at K2) remains the MTP-friendly shape. Disposition:
  MTP stays available and greedy-lossless; the 27B deployment question
  is dead, the 4B-family question stays open.
- greedy gate: 4/5 on ALL six reps, deterministic, K-independent;
  ADJUDICATED BENIGN via the third reference — the stock-backend K1
  control also scores 4/5 (adjudication-stock-k1.json), canaries are
  identical in every run, and the root-cause logit diffs are <= 0.031
  (fp16 tie-break class, the window-5 adjudication). The mismatch is
  MTP-vs-plain-decode arithmetic, not the bridge.
- side observation (single rep, no verdict): stock K1 speedups
  1.19/1.14 exceed champion K1's — the inverse of the K3 ordering;
  one look next window.
- provenance: lane 734cd84be4; intent: question=MTP-on-champion
  viability, mechanism=native mtp over the paged-split champion,
  disposition=available-not-compounding.

## Inner-loop surgery: register-resident accumulators (2026-09-13, plan/0008) — WIN, new champion

- question: the per-token inner loop is the named residual (champion
  34.3 vs the 50.1 TRITON baseline). ncu at the champion shape (PPS2,
  long ctx): L1/smem throughput 84.5 pct, DRAM 3.2 pct, FP32 ~1 pct of
  peak, 11.2 of 32 lanes active, 30.3 pct barrier stalls, occupancy
  smem-limited at 2 CTAs/SM — the sm_o shared-memory accumulation and
  lane-0-serialized softmax updates are the wall.
- surgery: register-resident accumulators (each warp owns its pairs,
  each lane owns d/32 dims), warp-uniform softmax with a lane-0
  broadcast, staging cut from five syncs per token to two, smem from
  ~100 KB to 1 KB (occupancy no longer smem-limited). Combine kernel
  and workspace layout unchanged.
- gates: standalone compile clean; oracle 36/36 at PPS2 (std +
  long-ctx, max rel 2.54e-04 — the pre-surgery class); committed-run
  medians-of-3 fresh restarts, graphs, labels asserted, zero
  preemptions.
- VERDICT: WIN. ctx512 41.3 (38.1/41.3/42.2, band 9.9) vs 34.3 =
  +20.4 pct; ctx2048 41.2 (41.2/41.2/41.2, band 0.0) vs 34.4 =
  +19.8 pct; short 195.5 vs 213.5 = -8.4 pct (real, band 2.9 —
  hypothesis: the unconditional two-row staging and the per-pair
  broadcast shfl raise the fixed per-token cost, which dominates at
  tiny contexts). Floor ratio 0.62 -> 0.75; TRITON-baseline ratio
  0.68 -> 0.82. NEW STANDING CHAMPION: 41.3 / 41.2 / 195.5 (mid /
  long / short ctx decode).
- provenance: lane 8b26330508 + this surgery commit; ncu profiles
  pre/post under .weco/c2-paged-decode/profile-*; intent:
  question=inner-loop residual, mechanism=register accumulators +
  warp-uniform softmax, disposition=champion.

## half2 staging surgery (2026-09-13, plan/0009) — NO_DIFF in-engine, champion unchanged

- question: PROFILE-DELTA's named bandwidth lever — does restaging
  K/V as __half2 on a lane-owns-pairs remap (removing the 2-way smem
  bank split of the scalar lane+32*ii pattern, halving the staging
  instruction count) move the champion rows?
- gates: nvcc clean; oracle 36/36 at PPS2 (std + long-ctx); gate-0
  triangulation vs TRITON cross-base 8/8 exact, zero token diffs;
  committed A/B medians-of-3 fresh restarts, graphs, labels asserted,
  zero preemptions.
- isolated (oracle harness, TP2 per-rank shapes): short-shape kernel
  80.6 -> 67.6 us (minus 16 pct vs incumbent); long-shape 670.4 ->
  609.7 us (minus 9 pct). Versus the pre-surgery kernel: 3.6x/4.4x.
- VERDICT: NO_DIFF in-engine. A/B 193.1/41.9/39.8 vs same-day
  incumbent baseline 195.5/41.6/40.2 — every row inside overlapping
  bands; short not recovered, mid/long held. The champion does NOT
  move (champion moves only on a WIN).
- mechanism: engine sensitivity floor. The register surgery moved
  the rows because it cut ~2 ms per walk launch (~16 ms/step at
  8 full-attn layers); half2 cuts ~13-60 us per launch (~0.1-0.5
  ms/step) — under the band. The kernel patch is preserved at
  results/short-row-recovery/half2-kernel.patch (strictly better in
  isolation, bank-conflict-free by construction), re-adoptable if a
  future lever reopens the smem path.
- provenance: working tree reverted to the incumbent champion after
  the A/B; extension cache cleared so subsequent engines rebuild the
  ledger champion. intent: question=bandwidth lever one (half2
  staging), mechanism=lane+32*j half2 remap, disposition=falsified-
  as-engine-mover (isolated win recorded).

## MTP K-tree on the 4B CHAMPION path (2026-09-13, plan/0009) — GREEN, deeply negative — the family question closed

- question: does the 4B's stock-path compounding (1.84/2.12/1.62/1.53
  at K1..4, TP2, graphs) transfer to the champion configuration
  (decode paged-split PPS2, verify/prefill gather)?
- config: Qwen3.5-4B W4A16 fixture, TP2, CUDA graphs, BRIDGE_ATTN
  with the paged-split decode arm, 3 fresh restarts, medians
  (results/mtp-4b-champion/summary.json).
- K1 0.53/0.74; K2 0.44/0.70; K3 0.59/0.73; K4 0.91/0.84 (ctx512 /
  ctx2048 speedup medians). Base mixed medians 176.9/73.9.
- greedy: 3/3 lossless on every rep at every K.
- VERDICT: NO COMPOUND, and worse - every depth is deeply NEGATIVE on
  the champion path, a stronger penalty than the 27B's (0.86-1.12).
  Mechanism (recorded hypothesis): the champion decode arm is much
  faster per step than stock on this fixture (base 176.9 vs the
  stock-path class), so the MTP draft-and-verify overhead (draft
  passes + gather-arm verify rows) has a far higher relative price;
  the stock path's compounding was partly buying back launch overhead
  the champion path no longer pays. The 4B deployment shape stays on
  its measured STOCK-path numbers; MTP x champion is dead for the
  family, not just the 27B.
- provenance: lane this commit; intent: question=4B-family MTP
  transfer, mechanism=champion-config K-tree on the fixture,
  disposition=closed-negative.

## K1 inversion settled (2026-09-13, plan/0009) — REAL: stock-K1 exceeds champion-K1

- question: the champion-path K-tree's single-rep side observation -
  the stock backend's K1 speedups exceeded the champion path's, the
  inverse of the deepest-K ordering. Replicate at the standing
  protocol.
- config: 27B, TP2, graphs; stock arm = TRITON backend, base + K1,
  3 fresh restarts (results/k1-inversion/summary.json); champion-K1
  medians banked from the plan/0008 K-tree (3 reps each).
- stock-K1 speedup medians: 1.32 ctx512 / 1.13 ctx2048 (vals
  1.27-1.42 / 1.08-1.21) vs champion-K1 banked 1.06 / 0.87. Greedy
  3/3 on every rep.
- VERDICT: the inversion is REAL and material (plus 25 pct at ctx512,
  plus 26 pct at ctx2048 over the champion path's K1). Recorded
  mechanism (hypothesis, consistent with the 4B K-tree the same
  window): MTP buys back launch/step overhead the champion path no
  longer pays, while the draft-and-verify machinery prices higher
  against the champion's faster decode steps. Net: spec decode
  favors the STOCK path on this engine at every measured scale -
  the champion deployment stays no-MTP.
- provenance: lane this commit; intent: question=K1-ordering
  inversion, mechanism=stock-K1 3-rep replication, disposition=
  confirmed-inversion.

## KV read pipelining surgery (2026-09-13, plan/0009) — NEGATIVE, falsified

- question: PROFILE-DELTA's second named bandwidth lever - explicit
  register prefetch of the next token's K/V row overlapped with the
  current token's compute, double-buffered staging, one sync per
  token (from two).
- gates: nvcc clean (after fixing the int-to-half init the torch
  build rejected); oracle 36/36 at PPS2 (std + long-ctx); gate-0
  6/8 with the dual-arm adjudication (see
  results/kv-pipelining/GATE0-ADJUDICATION.md: mean worst-row diff
  0.004, max 0.0625, the fp16 tie-break class - nvcc's FMA
  contraction moved with the new register schedule); committed A/B
  medians-of-3.
- VERDICT: NEGATIVE. A/B 186.7/36.8/39.5 vs the same-day incumbent
  baseline 195.5/41.6/40.2: ctx512 -11.5 pct at band 0.3 (decisive,
  bands do not overlap), short -4.5 pct (186.7 vs 195.5, just
  outside), ctx2048 -1.7 pct (overlapping). The prefetch registers
  and doubled staging buffers cost more at the mid-ctx shape than
  the latency hiding buys. The champion does NOT move; the working
  tree reverted; patch preserved at
  results/kv-pipelining/pipelining-kernel.patch.
- standing after both bandwidth levers: the split walk kernel's
  named lever list from PROFILE-DELTA is now exhausted (half2
  NO_DIFF, pipelining NEGATIVE) - the champion 41.3/41.2/195.5 class
  stands as this kernel generation's engine-measured optimum on
  this silicon.
- provenance: lane this commit; intent: question=bandwidth lever
  two (KV read pipelining), mechanism=register prefetch +
  double-buffer, disposition=falsified-negative.

## INT8 headroom measured (2026-09-14, plan/0009) — lane closed, negative

- question: what does a narrower activation path buy on this silicon
  versus the W4A16 incumbent (the deferred 0008 probe)?
- chain: llm_compressor export (per-channel sym dynamic W8A8
  compressed-tensors) -> engine loads natively on the champion
  config -> identical rows, 3 fresh restarts per arm.
- VERDICT: NO HEADROOM. short 426.9 vs 538.1 (0.79), ctx512 80.2 vs
  94.0 (0.85), ctx2048 75.9 vs 89.2 (0.85). Decode is
  weight-bandwidth-bound; W8 doubles the weight bytes and the int8
  tensor rate cannot buy them back at decode batch shapes. The
  formats doctrine's last unexplored lane on TU102 is measured and
  closed. Disposition: W4A16 stands everywhere.
- provenance: results/int8-probe/headroom-summary.json; intent:
  question=int8 headroom, mechanism=compressed-tensors W8A8 on the
  champion path, disposition=closed-negative.

## Dynamic split sizing (2026-09-15, plan/0010) — NO_DIFF, champion unchanged; kernel-structure path CLOSED for the gap

- question: the ncu stall capture at the ctx512 champion shape named
  60 pct of issued-instruction cycles in L1TEX-load latency (8.3 cyc)
  plus CTA-barrier waits (8.2 cyc) at 0.89 waves - more CTAs should
  hide them (PPS1 measured +7.9 pct paired at gate-1, but a flat
  switch cost the short row -15 pct). Does a piecewise dynamic split
  table (short: ~3 pages/split; ctx512-class: 1 page/split; long: the
  flat PPS2 form; env-gated BRIDGE_DYN_SPLITS, capture-safe since
  splits derive from the static max_pages dim) take the ctx512 win
  without the short cost?
- gates: oracle 72/72 (flat and dynamic paths, std + long-ctx);
  committed PAIRED A/B, both arms fresh same-day, 3 restarts each.
- VERDICT: NO_DIFF on every row. short 194.2 vs 194.1; ctx512 42.2
  vs 42.3; ctx2048 40.9 vs 40.9. The gate-1 ctx512 signal was a
  low-day artifact (the flat arm's own ctx512 band today was 19.4
  pct: 38.0-46.2). The champion does NOT move; the patch is preserved
  at results/step-profile/dyn-splits.patch (default-off, orphans
  nothing).
- STANDING after four structure levers (half2 NO_DIFF, pipelining
  NEGATIVE, dynamic splits NO_DIFF, plus the 0009 pair): the walk
  kernel's internal structure does not hold the committed ctx512
  gap. The 4.3 ms lives at step level outside the attention kernel
  duration - the nsys step ledger is the remaining instrument.
- provenance: lane this commit; intent: question=dynamic split
  sizing, mechanism=piecewise table on max_pages, disposition=
  falsified-no-diff.
