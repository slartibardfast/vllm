# The convergence run (2026-09-11, 72h window Phase D)

The goal's full-clause configuration, first green measurement:
decode on the paged-split champion (BRIDGE_PAGED_DECODE=split,
BRIDGE_PPS=2), verify and prefill on the gather loop (the qlen==1
guard - the proven split of duties), CUDA graphs, TP2, 27B W4A16.

GREEN: greedy 5/5, canaries identical between arms (the 0.044/0.013
nonzeros are the model-intrinsic levels, both arms).

| row | base | + MTP K3 | speedup |
|---|---|---|---|
| mixed ctx512 | 24.8 | 27.9 | 1.12x |
| mixed ctx2048 | 13.2 | 14.1 | 1.07x |

REFINEMENT OF THE RECORD: the 27B MTP-graphs verdict is
PATH-DEPENDENT - no compound on the stock decode path (0.93x, window
2) versus 1.07-1.12x on the bridge-split champion path. The model-
scaled draft economics stand, but the champion path leaves room for
the draft to pay. Projection onto the committed decode rows (34.3
ctx512 x 1.12 ~ 38.4) is labeled a projection; the honest measured
rows are this probe's. The 4B remains the MTP-friendly shape (2.12x
at K2). Open: the multi-row paged root-cause (MTP rides gather until
then); K1/K2 on the 27B's champion path (the 4B's K2 peak suggests
checking).
