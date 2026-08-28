# m16n8k8 fragment model — CONFIRMED empirically (power-of-2 probe)

The power-of-2 probe (kmap_power2_probe.cu: B half0 = 2^lane, half1 = 0,
A = e at a0.x) decoded cleanly:

- D[0][0] = 1 + 2 + 4 + 8        (b.x of lanes 0..3, k = 0..3, column 0)
- D[0][1] = 16 + 32 + 64 + 128   (b.x of lanes 4..7, column 1)
- D[0][2], D[0][3]: lanes 8..15, 16..23 likewise.

So the mma fragment model is exactly as documented: lane L = (g, c) holds
b.x = B[2c][g], b.y = B[2c+1][g]; the A fragment a0 = A[g][2c, 2c+1],
a1 = A[g+8][2c, 2c+1]; D fragments row g / g+8, cols 2c / 2c+1.

Consequence for the register-dequant kernel (turing_w4a16_opt.cu): the
byte_perm extraction is verified correct in isolation (dq standalone), and
the fragment model is confirmed - so the remaining nan must come from the
staging-to-fragment composition. Next debugging step: dump the fragment
values for ALL (nt, kk) steps of the first chunk (not just the first mma)
and compare against the expected per-slot dequantized values; the suspect
is a later-step slot mismatch in the sQE/sQO layout or the part[] handling
in an unrolled iteration.
