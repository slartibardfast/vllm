# Fragment probes

Investigation tooling for the m16n8k8 B-fragment layout, opened because the
register-dequant kernel produces structured-wrong output whose pattern does
not decode under the documented (k, k+1) per-byte-pair assumption.

- `mma_pair_probe.cu` — A fixed (1,2)/(3,4), B per lane (b.x = lane+1,
  b.y = 100+lane), B[k][n] = 100k+n semantics. Observed D row 0 =
  822 + 48c, which does not decompose into A-value × B-value products
  under the documented layout — the discrepancy is the study's first
  data point.
- `pipe_identity_probe.cu` — identity-A probe for the pipelined kernel.

Status: open. The B-fragment byte/nibble order for .col B in m16n8k8 needs
to be established from the PTX ISA documentation before the register-dequant
path can be correct. Until then the fp16-staged kernels (opt1) remain the
validated implementation.
