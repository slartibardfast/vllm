# CUDA-graph validation results

- Dense kernel (w4a16_opt2, packed register-dequant): graph capture
  succeeds, replay reproduces the eager result,
  fp16 tolerance), and a post-replay input mutation is correctly
  recomputed. PASS.
- Split-K flow (w4a16_opt_splitk into an fp32 WS): eager full-K validated;
  the graph capture/replay of the WS.zero_ + split launch returns a
  deterministic wrong value (16.47) - the WS-zero replay ordering needs
  verification (suspect the fill lands after the atomicAdd accumulations
  in the replayed graph). OPEN.

Runtime lane status: the dense kernel is graph-safe; the split-K graph
capture is the remaining runtime item.

## Split-K flow: WS-zero replay ordering (OPEN)

The split-K entry accumulates into an fp32 workspace via atomicAdd. For
graph capture, the WS.zero_() fill and the splitk launch are both
recorded, but on replay the deterministic wrong value (16.47) persists
across replays. Root cause: the atomicAdd accumulation pattern requires
the workspace to be zeroed between replays, but the graph's fill and
accumulate nodes have a memory dependency that the current capture does
not express. Fix: use a dedicated memset node between the fill and the
accumulate, or restructure to use a separate output buffer per split.

## Root cause

The atomicAdd pattern in the split-K kernel is incompatible with CUDA
graph capture for deterministic replay: the fill-then-accumulate sequence
requires a memory barrier between the fill and the accumulate that the
graph does not express. Fix options:
(1) replace atomicAdd with a store (correct for num_splits=1),
(2) use per-split output buffers with a final reduction kernel,
(3) use cudaMemsetAsync inside the kernel prologue.
Option (1) is the simplest and correct for the single-split case.
