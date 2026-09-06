// flash_fwd_sm75.cuh — bridge-native FlashAttention forward for Turing.
//
// The sm_75 execution path for flash-attn/FlashInfer/vLLM (plan/0006):
// fp16 attention built entirely on cu_sm80_on_sm75 primitives — no
// CUTLASS atoms, no cp.async, no upstream kernel code.
//
// v1 scope: head dim 64, fp16, causal or full attention, s multiple of
// 64, no dropout/GQA/varlen. m16n8k8 fragments; K/V row-major smem with
// B-fragment reads as two LDS.U16 (k-pair halves at fixed column).
//
// Fragment conventions (validated by the layoutsolve differential):
//   lane l: g = l>>2, t = l&3
//   A (m16n8k8, 16x8): reg0 = {A[g][2t], A[g][2t+1]},
//                      reg1 = {A[g+8][2t], A[g+8][2t+1]}
//   B (8x8):           b = {B[2t][n], B[2t+1][n]}
//   D (16x8):          d0={D[g][2t],D[g][2t+1]} d1={D[g][2t+2],D[g][2t+3]}
//                      d2={D[g+8][2t],D[g+8][2t+1]} d3={D[g+8][2t+2],D[g+8][2t+3]}
#pragma once
#include <cstdint>
#include <cuda_fp16.h>
#include "sm80_on_sm75.cuh"

namespace bridge_flash {

constexpr int kBlockRows = 64;   // query rows per CTA
constexpr int kBlockKV = 64;     // kv rows per tile
constexpr int kThreads = 128;    // 4 warps
constexpr float kLog2e = 1.4426950408889634f;

__device__ __forceinline__ uint32_t pack_half(float lo, float hi) {
  return (uint32_t)__half_as_ushort(__float2half(lo)) |
         ((uint32_t)__half_as_ushort(__float2half(hi)) << 16);
}

// The forward for one (b, h): q points at the (sq, d) query block, k/v at
// the (s, d) KV matrix. Query row 0 sits at absolute position q0
// (bottom-right causal, the FA2 seqlen_q != seqlen_k contract: q0 = s - sq
// for a prefill chunk with prefix; q0 = 0 and sq = s for the dense case).
// D in {64, 128}; d=64 double-buffers K/V, d=128 single-buffers (smem).
template <int D>
__device__ __forceinline__ void flash_fwd_one(
    const __half* __restrict__ q, const __half* __restrict__ k,
    const __half* __restrict__ v, __half* __restrict__ out, int sq, int s,
    int q0, bool causal, void* smem) {
  constexpr int kStride = D + 8;
  // occupancy experiment (step 4b): single-buffering K/V for d=64 halves
  // smem (46KB -> 27.6KB) so two CTAs fit per SM; the other CTA's compute
  // hides this CTA's synchronous LDG->STS staging
  constexpr bool kDBuf = false;
  __half* sQ = reinterpret_cast<__half*>(smem);        // 64 x kStride
  // double-buffered K/V: LDGs for tile nt+1 issue while compute runs
  // on tile nt (the register-staging pipeline; per-thread LDG->STS
  // ordering is scoreboard-enforced, commit/wait are free fences)
  constexpr int kKVBufs = kDBuf ? 2 : 1;
  __half* sK = sQ + kBlockRows * kStride;              // kKVBufs x 64 x kStride
  __half* sV = sK + kKVBufs * kBlockKV * kStride;      // kKVBufs x 64 x kStride

  const int tid = threadIdx.x;
  const int warp = tid >> 5, lane = tid & 31;
  const int g = lane >> 2, t = lane & 3;
  const int row_lo = warp * 16 + g;                    // query row (tile)
  const int row_hi = row_lo + 8;
  const long q_base = (long)blockIdx.y * kBlockRows;

  // ---- stage Q (fp16 rows, zero-padded past sq): 2048 u32 total ----
  for (int i = tid; i < kBlockRows * (D / 2); i += kThreads) {
    int r = i / (D / 2), c = (i % (D / 2)) * 2;
    __half* dst = sQ + r * kStride + c;
    if (q_base + r < sq) {
      const __half* p = q + (long)(q_base + r) * D + c;
      bridge::sts_u32(dst, bridge::ldg_cs_u32(p));
    } else {
      bridge::sts_u32(dst, 0);
    }
  }
  __syncthreads();

  // Q A-fragments: qf[ks][0] = {Q[row_lo][ks*8+2t], +1}; [1] = row hi.
  // d=64 -> 8 k8 steps.
  uint32_t qf[D / 8][2];
#pragma unroll
  for (int ks = 0; ks < D / 8; ks++) {
    qf[ks][0] = *reinterpret_cast<const uint32_t*>(&sQ[row_lo * kStride + ks * 8 + 2 * t]);
    qf[ks][1] = *reinterpret_cast<const uint32_t*>(&sQ[row_hi * kStride + ks * 8 + 2 * t]);
  }

  // ---- iterate K/V tiles ----
  float acc[D / 8][4];   // O accumulator: D/8 d-column groups x 4 rows
  memset(acc, 0, sizeof(acc));
  float m_lo = -INFINITY, m_hi = -INFINITY;
  float l_lo = 0.f, l_hi = 0.f;
  const float scale_log2e = (1.f / sqrtf((float)D)) * kLog2e;

  const int n_tiles = s / kBlockKV;
  // causal compares kv columns against ABSOLUTE query positions; the
  // output row index stays tile-relative (the chunk's own rows)
  const int q_abs_lo = q0 + (int)q_base + row_lo;
  const int q_abs_hi = q_abs_lo + 8;
  // tiles whose first column exceeds this CTA's highest absolute query
  // row are fully masked: every element of S is -inf, P is 0, and the
  // tile contributes nothing — stop at the diagonal (FA2 tile skipping).
  // q_abs_max <= s-1 for any real layout, so nt_end <= n_tiles.
  int nt_end = n_tiles;
  if (causal) {
    const int q_abs_max = q0 + (int)q_base + (kBlockRows - 1);
    const int last_needed = q_abs_max / kBlockKV + 1;
    if (last_needed < nt_end) nt_end = last_needed;
  }

  // per-thread staging registers for one K/V tile pair:
  // 64*64/128 halves... = 32 u32 per thread for K, 32 for V
  constexpr int kU32PerTile = kBlockKV * (D / 2) / kThreads;
  uint32_t rK[kDBuf ? kU32PerTile : 1];
  uint32_t rV[kDBuf ? kU32PerTile : 1];
  auto load_kv_regs = [&](int nt_) {
    const long k_base = (long)nt_ * kBlockKV;
    for (int i = 0; i < kBlockKV * (D / 2) / kThreads; i++) {
      int idx = tid + i * kThreads;
      int r = idx / (D / 2), c = (idx % (D / 2)) * 2;
      rK[i] = bridge::ldg_cs_u32(k + k_base * D + (long)r * D + c);
      rV[i] = bridge::ldg_cs_u32(v + k_base * D + (long)r * D + c);
    }
  };
  auto store_kv_smem = [&](int buf) {
    for (int i = 0; i < kBlockKV * (D / 2) / kThreads; i++) {
      int idx = tid + i * kThreads;
      int r = idx / (D / 2), c = (idx % (D / 2)) * 2;
      bridge::sts_u32(sK + buf * kBlockKV * kStride + r * kStride + c, rK[i]);
      bridge::sts_u32(sV + buf * kBlockKV * kStride + r * kStride + c, rV[i]);
    }
  };

  auto stage_now = [&](int nt_) {   // single-buffer path (d=128)
    const long k_base = (long)nt_ * kBlockKV;
    for (int i = 0; i < kU32PerTile; i++) {
      int idx = tid + i * kThreads;
      int r = idx / (D / 2), c = (idx % (D / 2)) * 2;
      bridge::sts_u32(sK + r * kStride + c,
                      bridge::ldg_cs_u32(k + k_base * D + (long)r * D + c));
      bridge::sts_u32(sV + r * kStride + c,
                      bridge::ldg_cs_u32(v + k_base * D + (long)r * D + c));
    }
  };
  if (kDBuf) load_kv_regs(0);
  for (int nt = 0; nt < nt_end; nt++) {
    if (kDBuf) {
      store_kv_smem(nt & 1);
      __syncthreads();
      if (nt + 1 < n_tiles) load_kv_regs(nt + 1);  // latency hides behind
      // the S/softmax/PV work below (scoreboard-ordered per thread)
    } else {
      stage_now(nt);
      __syncthreads();
    }

    // ---- S = Q K^T fragments: st[nt8][r4] ----
#ifndef FWD_BISECT
#define FWD_BISECT 4
#endif
    float st[8][4];
#pragma unroll
    for (int nt8 = 0; nt8 < 8; nt8++)
#pragma unroll
      for (int r4 = 0; r4 < 4; r4++) st[nt8][r4] = 0.f;

    uint32_t kb[2][4];
#pragma unroll
    for (int ks = 0; ks < D / 8; ks++) {
      // K B-fragments via ldmatrix: two x4 loads cover all 8 nt8 groups
      // for this d-octave. Matrix M of group G carries K rows
      // (4*G+M)*8..+7 over d = ks*8..+8; the x4 output distribution
      // (row = l/4, colpair = l%4) is exactly {K[n][ks*8+2t], +1},
      // n = nt8*8 + g.
#pragma unroll
      for (int grp = 0; grp < 2; grp++) {
        const __half* arow =
            sK + (kDBuf ? (nt & 1) * kBlockKV * kStride : 0) +
            ((4 * grp + (lane >> 3)) * 8 + (lane & 7)) * kStride + ks * 8;
        bridge::ldmatrix_x4(kb[grp][0], kb[grp][1], kb[grp][2], kb[grp][3],
                            arow);
      }
#pragma unroll
      for (int nt8 = 0; nt8 < 8; nt8++)
        bridge::mma_m16n8k8_f32(st[nt8][0], st[nt8][1], st[nt8][2],
                                st[nt8][3], qf[ks][0], qf[ks][1],
                                kb[nt8 >> 2][nt8 & 3]);
    }

    // ---- causal mask (kv col absolute > query row absolute -> -inf) ----
    if (causal && FWD_BISECT >= 3) {
#pragma unroll
      for (int nt8 = 0; nt8 < 8; nt8++) {
        const int c0 = nt * kBlockKV + nt8 * 8 + 2 * t;
        if (c0 > q_abs_lo) st[nt8][0] = -INFINITY;
        if (c0 + 1 > q_abs_lo) st[nt8][1] = -INFINITY;
        if (c0 > q_abs_hi) st[nt8][2] = -INFINITY;
        if (c0 + 1 > q_abs_hi) st[nt8][3] = -INFINITY;
      }
    }

    // ---- online softmax: rows lo/hi tracked separately ----
    float rmax_lo = -INFINITY, rmax_hi = -INFINITY;
#pragma unroll
    for (int nt8 = 0; nt8 < 8; nt8++) {
      rmax_lo = fmaxf(rmax_lo, fmaxf(st[nt8][0], st[nt8][1]));
      rmax_hi = fmaxf(rmax_hi, fmaxf(st[nt8][2], st[nt8][3]));
    }
#pragma unroll
    for (int off = 1; off <= 2; off <<= 1) {
      rmax_lo = fmaxf(rmax_lo, __shfl_xor_sync(0xffffffff, rmax_lo, off));
      rmax_hi = fmaxf(rmax_hi, __shfl_xor_sync(0xffffffff, rmax_hi, off));
    }
    float m_new_lo = fmaxf(m_lo, rmax_lo);
    float m_new_hi = fmaxf(m_hi, rmax_hi);
    // corr rescales old P values exp2((S-m_old)*scale) to the new max:
    // the exponent difference carries the SAME scale factor
    float corr_lo = (m_lo == -INFINITY) ? 0.f
                    : exp2f((m_lo - m_new_lo) * scale_log2e);
    float corr_hi = (m_hi == -INFINITY) ? 0.f
                    : exp2f((m_hi - m_new_hi) * scale_log2e);
    if (isinf(m_new_lo)) { m_new_lo = 0.f; corr_lo = 0.f; }
    if (isinf(m_new_hi)) { m_new_hi = 0.f; corr_hi = 0.f; }
    m_lo = m_new_lo;   // the running max must advance, or every tile's
    m_hi = m_new_hi;   // corr zeroes the accumulator

#if FWD_BISECT < 2
    return;
#endif
    // ---- O correction for the row maxima: rescale the OLD accumulator
    // before adding this tile's contribution ----
#pragma unroll
    for (int cc = 0; cc < D / 8; cc++) {
      acc[cc][0] *= corr_lo;
      acc[cc][1] *= corr_lo;
      acc[cc][2] *= corr_hi;
      acc[cc][3] *= corr_hi;
    }

    // ---- P = exp2((S - m_new) * scale * log2e), half2-vectorized
    // (native ex2.approx.f16x2 on sm_75). S and m are RAW dot products
    // (the scale folds into the exponent): with real activations they
    // exceed fp16 range (|S| > 65504), so the subtraction MUST run in
    // fp32 before the half conversion — packing raw S into half first
    // saturates to inf and inf - inf is NaN (found by the engine
    // correctness gate, 2026-09-05). (S - m) <= 0 always, so a large
    // negative product flushes to -inf in half and exp2 gives 0: safe.
    uint32_t pfa[8][2];
#pragma unroll
    for (int ks = 0; ks < 8; ks++) {
      float2 dla = make_float2((st[ks][0] - m_new_lo) * scale_log2e,
                               (st[ks][1] - m_new_lo) * scale_log2e);
      float2 dlb = make_float2((st[ks][2] - m_new_hi) * scale_log2e,
                               (st[ks][3] - m_new_hi) * scale_log2e);
      __half2 ea = h2exp2(__floats2half2_rn(dla.x, dla.y));
      __half2 eb = h2exp2(__floats2half2_rn(dlb.x, dlb.y));
      pfa[ks][0] = *reinterpret_cast<uint32_t*>(&ea);  // row lo, kv pair
      pfa[ks][1] = *reinterpret_cast<uint32_t*>(&eb);  // row hi
    }

    // ---- P V: O += P x V per k-step and output tile ----
#if FWD_BISECT < 3
    return;
#endif
#pragma unroll
    for (int ks = 0; ks < 8; ks++) {
      // V B-fragments via ldmatrix.trans: matrix M of group ccg carries V
      // rows ks*8..+7 over d = (4*ccg+M)*8..+8. The .trans distribution
      // hands lane l {V[ks*8+2t][(4*ccg+M)*8 + g], +1} — exactly
      // {V[k][n], V[k+1][n]} with k = ks*8+2t, n = cc*8+g.
      uint32_t vb[4];
#pragma unroll
      for (int ccg = 0; ccg < D / 32; ccg++) {
        const __half* vrow =
            sV + (kDBuf ? (nt & 1) * kBlockKV * kStride : 0) +
            (ks * 8 + (lane & 7)) * kStride + (4 * ccg + (lane >> 3)) * 8;
        bridge::ldmatrix_x4_trans(vb[0], vb[1], vb[2], vb[3], vrow);
#pragma unroll
        for (int M = 0; M < 4; M++)
          bridge::mma_m16n8k8_f32(acc[4 * ccg + M][0], acc[4 * ccg + M][1],
                                  acc[4 * ccg + M][2], acc[4 * ccg + M][3],
                                  pfa[ks][0], pfa[ks][1], vb[M]);
      }
    }

#if FWD_BISECT < 4
    return;
#endif
    // ---- row sums of P (rows lo/hi) via butterfly over t ----
    float psum_lo = 0.f, psum_hi = 0.f;
#pragma unroll
    for (int ks = 0; ks < 8; ks++) {
      float2 fl = __half22float2(*reinterpret_cast<__half2*>(&pfa[ks][0]));
      float2 fh = __half22float2(*reinterpret_cast<__half2*>(&pfa[ks][1]));
      psum_lo += fl.x + fl.y;
      psum_hi += fh.x + fh.y;
    }
#pragma unroll
    for (int off = 1; off <= 2; off <<= 1) {
      psum_lo += __shfl_xor_sync(0xffffffff, psum_lo, off);
      psum_hi += __shfl_xor_sync(0xffffffff, psum_hi, off);
    }
    l_lo = l_lo * corr_lo + psum_lo;
    l_hi = l_hi * corr_hi + psum_hi;

    // single-buffered V/K (d=128) needs the trailing barrier so the next
    // tile's stage_now cannot clobber smem while warps still read it;
    // the double-buffered path (d=64) writes the OTHER buffer next, so
    // the post-store barrier alone suffices.
    if (!kDBuf) __syncthreads();
  }

  // ---- epilogue: O / l, write rows < sq (tile-relative indices) ----
  const int out_lo = (int)q_base + row_lo;
  const int out_hi = out_lo + 8;
#pragma unroll
  for (int nt8 = 0; nt8 < D / 8; nt8++) {
    if (out_lo < sq) {
      out[(long)out_lo * D + nt8 * 8 + 2 * t] =
          __float2half(acc[nt8][0] / l_lo);
      out[(long)out_lo * D + nt8 * 8 + 2 * t + 1] =
          __float2half(acc[nt8][1] / l_lo);
    }
    if (out_hi < sq) {
      out[(long)out_hi * D + nt8 * 8 + 2 * t] =
          __float2half(acc[nt8][2] / l_hi);
      out[(long)out_hi * D + nt8 * 8 + 2 * t + 1] =
          __float2half(acc[nt8][3] / l_hi);
    }
  }
}

}  // namespace bridge_flash
