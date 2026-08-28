# k-mapping probe findings (open)

`kmap_probe.cu`: for probe k in 0..3, A row 0 carries a single 1 at A-
fragment position k (a0.x, a0.y, a1.x, a1.y); B lane L carries
b.x = 1000(L+1), b.y = 1000(L+1)+7. D[0][0..3] per probe:

```
kpos 0 (a0.x): 10000  26000  42000  58000
kpos 1 (a0.y): 10030  26032  42032  58032
kpos 2 (a1.x):     0      0      0      0
kpos 3 (a1.y):     0      0      0      0
```

Decode: the a0.x products are the b.x values of LANES 9, 25, 41, 57 - a
16-lane stride. Lanes 41 and 57 do not exist in a 32-thread warp, so the
mma's internal B consumption does not follow the documented per-lane
fragment layout under any simple half-order variant (swapped halves give
identical sums; (k, k+4) pairings do not reproduce the values). Also:
a1 positions contribute nothing to row 0, confirming the a0/a1 row split
(g, g+8).

Conclusion: the m16n8k8 B-fragment byte order cannot be established from
hand-rolled probes alone; the PTX ISA fragment tables for the .col B
matrix must be consulted directly, and the staged-word layout of the
register-dequant kernel rebuilt to match. Until then the fp16-staged
kernels (opt1 23-25 TFLOP/s, pipe 13-14 TFLOP/s) remain the validated
implementations.

## CORRECTION (same day)

The 822 + 48c pattern and the 16-lane-stride reading are INVALID: the probe's
host D buffer was 16 floats while the kernel wrote 64 (out-of-bounds device
writes into the allocation pool, copied back as garbage). With the buffer
sized correctly the probe reproduces cleanly and matches the documented
fragment layout (verified via the power-of-2 B-half decode in
kmap_power2_probe.cu). The open item is therefore NOT the fragment layout:
it is the staging-to-fragment composition inside the committed
double-buffered k_opt2, to be bisected with a correctly-sized standalone
repro.

## RESOLUTION (per-lane-distinct-b probe, same day)

The per-lane-distinct-b probe (b.x = lane+1, b.y = 100+lane) with
A = e at a0.x confirms: the mma sums the four row-group lanes' b.x values
into D[0][0] (= 1+2+3+4 = 10), exactly as the documented fragment model
predicts (each of the 4 lanes holds 2 of the 8 k values for the column).
The fragment model is CORRECT. The earlier 822+48c reading was an
artifact of the undersized host D buffer (now fixed).

The register-dequant kernel's remaining issue is NOT the fragment layout:
it is the correctness of the even/odd nibble-plane staging under the full
M/N/K sweep, which the opt_oracle validates case by case.
