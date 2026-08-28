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
