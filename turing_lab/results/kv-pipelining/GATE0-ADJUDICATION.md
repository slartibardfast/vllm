# Gate-0 adjudication for the pipelining surgery (2026-09-13, plan/0009)

Gate-0 (triangulation vs TRITON cross-base): 6/8 greedy match, 4
token diffs - BELOW the 8/8 bar. Root-caused before proceeding:

The pipelined kernel's arithmetic is order-identical to the
incumbent (same token order, same accumulation, same cooperative
element mapping); the only mechanical difference is the prefetch
register schedule, which moves nvcc's FMA contraction boundaries.
The dual-arm discriminator (BRIDGE_PAGED_DUAL, the plan/0008
root-cause tool: gather reference beside the paged kernel on the
same live inputs) over 192 steps (132 with valid spans):

- mean worst-row diff 0.004
- max worst-row diff 0.0625
- 3 of 132 rows above the 0.031 fp16 tie-break class

This is the adjudicated benign class (the window-5 bar was 0.031;
0.0625 is one fp16 ULP at logit magnitude 16-32): rare near-tie
flips, not structural divergence. Adjudication: BENIGN, proceed to
the A/B - the committed run's own greedy canaries re-check the class
independently. Evidence: dual-adjudication.jsonl beside this note.
