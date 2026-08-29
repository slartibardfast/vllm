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
constexpr int kStride = 72;      // smem row stride (64 + 4 pad), u32-even
constexpr float kLog2e = 1.4426950408889634f;

__device__ __forceinline__ uint32_t pack_half(float lo, float hi) {
  return (uint32_t)__half_as_ushort(__float2half(lo)) |
         ((uint32_t)__half_as_ushort(__float2half(hi)) << 16);
}

// B fragment: the two k-pair halves of one (kv row pair, column) from
// row-major shared memory, packed into one u32 for the mma.
__device__ __forceinline__ uint32_t load_b(const __half* smem, int kv_row,
                                           int col) {
  const __half* p = smem + kv_row * kStride + col;
  return (uint32_t)__half_as_ushort(p[0]) |
         ((uint32_t)__half_as_ushort(p[1]) << 16);
}

// The forward for one (b, h): q/k/v point at the (s, d) head matrix.
__device__ __forceinline__ void flash_fwd_one(
    const __half* __restrict__ q, const __half* __restrict__ k,
    const __half* __restrict__ v, __half* __restrict__ out, int s, bool causal,
    void* smem) {
  constexpr int D = 64;
  __half* sQ = reinterpret_cast<__half*>(smem);        // 64 x kStride
  __half* sK = sQ + kBlockRows * kStride;              // 64 x kStride
  __half* sV = sK + kBlockKV * kStride;                // 64 x kStride

  const int tid = threadIdx.x;
  const int warp = tid >> 5, lane = tid & 31;
  const int g = lane >> 2, t = lane & 3;
  const int row_lo = warp * 16 + g;                    // query row (tile)
  const int row_hi = row_lo + 8;
  const long q_base = (long)blockIdx.y * kBlockRows;

  // ---- stage Q (fp16 rows, zero-padded past s) ----
  for (int i = tid; i < kBlockRows * D / 8; i += kThreads) {
    int r = (i * 2) / D, c = (i * 2) % D;   // c even: 2 halves per u32
    __half* dst = sQ + r * kStride + c;
    if (q_base + r < s) {
      const __half* p = q + (long)(q_base + r) * D + c;
      bridge::sts_u32(dst, bridge::ldg_cs_u32(p));
    } else {
      bridge::sts_u32(dst, 0);
    }
  }
  __syncthreads();

  // Q A-fragments: qf[st][0] = k pair (2t, 2t+1) of row lo; [1] = row hi
  uint32_t qf[4][2];
#pragma unroll
  for (int st = 0; st < 4; st++) {
    qf[st][0] = *reinterpret_cast<const uint32_t*>(&sQ[row_lo * kStride + st * 8 + 2 * t]);
    qf[st][1] = *reinterpret_cast<const uint32_t*>(&sQ[row_hi * kStride + st * 8 + 2 * t]);
  }

  // ---- iterate K/V tiles ----
  float acc[8][4];   // O accumulator: 8 d-column groups x 4 rows
  memset(acc, 0, sizeof(acc));
  float m_lo = -INFINITY, m_hi = -INFINITY;
  float l_lo = 0.f, l_hi = 0.f;
  const float scale_log2e = (1.f / sqrtf((float)D)) * kLog2e;

  const int n_tiles = s / kBlockKV;
  const int q_abs_lo = (int)q_base + row_lo;
  const int q_abs_hi = q_abs_lo + 8;

  for (int nt = 0; nt < n_tiles; nt++) {
    const long k_base = (long)nt * kBlockKV;
    // ---- stage K, V tiles ----
    for (int i = tid; i < kBlockKV * D / 8; i += kThreads) {
      int r = (i * 2) / D, c = (i * 2) % D;   // c even
      const __half* pk = k + k_base * D + (long)r * D + c;
      const __half* pv = v + k_base * D + (long)r * D + c;
      bridge::sts_u32(sK + r * kStride + c, bridge::ldg_cs_u32(pk));
      bridge::sts_u32(sV + r * kStride + c, bridge::ldg_cs_u32(pv));
    }
    __syncthreads();

    // ---- S = Q K^T fragments: st[nt8][r4] ----
#ifndef FWD_BISECT
#define FWD_BISECT 4
#endif
    float st[8][4];
#pragma unroll
    for (int nt8 = 0; nt8 < 8; nt8++)
#pragma unroll
      for (int r4 = 0; r4 < 4; r4++) st[nt8][r4] = 0.f;

#pragma unroll
    for (int ks = 0; ks < 4; ks++) {
      const int kv_row = ks * 8 + 2 * t;
#pragma unroll
      for (int nt8 = 0; nt8 < 8; nt8++) {
        uint32_t b = load_b(sK, kv_row, nt8 * 8 + g);
        bridge::mma_m16n8k8_f32(st[nt8][0], st[nt8][1], st[nt8][2],
                                st[nt8][3], qf[ks][0], qf[ks][1], b);
      }
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
    float corr_lo = (m_lo == -INFINITY) ? 0.f : exp2f(m_lo - m_new_lo);
    float corr_hi = (m_hi == -INFINITY) ? 0.f : exp2f(m_hi - m_new_hi);
    if (isinf(m_new_lo)) { m_new_lo = 0.f; corr_lo = 0.f; }
    if (isinf(m_new_hi)) { m_new_hi = 0.f; corr_hi = 0.f; }

#if FWD_BISECT < 2
    return;
#endif
#if FWD_BISECT < 2
    return;
#endif
    // ---- P = exp2((S - m_new) * scale * log2e), packed fp16 ----
    uint32_t pf[4][8][2];
#pragma unroll
    for (int ks = 0; ks < 4; ks++) {
      const int kv_row = ks * 8 + 2 * t;
#pragma unroll
      for (int nt8 = 0; nt8 < 8; nt8++) {
        pf[ks][nt8][0] = pack_half(
            exp2f((st[nt8][0] - m_new_lo) * scale_log2e),
            exp2f((st[nt8][1] - m_new_lo) * scale_log2e));
        pf[ks][nt8][1] = pack_half(
            exp2f((st[nt8][2] - m_new_hi) * scale_log2e),
            exp2f((st[nt8][3] - m_new_hi) * scale_log2e));
      }
    }

    // ---- P V: O += P chunks x V, per k8 step ----
#if FWD_BISECT < 3
    return;
#endif
#pragma unroll
    for (int ks = 0; ks < 4; ks++) {
      const int kv_row = ks * 8 + 2 * t;
#pragma unroll
      for (int nt8 = 0; nt8 < 8; nt8++) {
        uint32_t b = load_b(sV, kv_row, nt8 * 8 + g);
        bridge::mma_m16n8k8_f32(acc[nt8][0], acc[nt8][1], acc[nt8][2],
                                acc[nt8][3], pf[ks][nt8][0], pf[ks][nt8][1],
                                b);
      }
    }

#if FWD_BISECT < 4
    return;
#endif
#if FWD_BISECT < 4
    return;
#endif
    // ---- row sums of P (rows lo/hi) via butterfly over t ----
    float psum_lo = 0.f, psum_hi = 0.f;
#pragma unroll
    for (int ks = 0; ks < 4; ks++) {
#pragma unroll
      for (int nt8 = 0; nt8 < 8; nt8++) {
        __half2 h2 = *reinterpret_cast<__half2*>(&pf[ks][nt8][0]);
        float2 f2 = __half22float2(h2);
        psum_lo += f2.x + f2.y;
        __half2 h3 = *reinterpret_cast<__half2*>(&pf[ks][nt8][1]);
        float2 f3 = __half22float2(h3);
        psum_hi += f3.x + f3.y;
      }
    }
#pragma unroll
    for (int off = 1; off <= 2; off <<= 1) {
      psum_lo += __shfl_xor_sync(0xffffffff, psum_lo, off);
      psum_hi += __shfl_xor_sync(0xffffffff, psum_hi, off);
    }
    l_lo = l_lo * corr_lo + psum_lo;
    l_hi = l_hi * corr_hi + psum_hi;

    // ---- O correction for the row maxima ----
#pragma unroll
    for (int nt8 = 0; nt8 < 8; nt8++) {
      acc[nt8][0] *= corr_lo;
      acc[nt8][1] *= corr_lo;
      acc[nt8][2] *= corr_hi;
      acc[nt8][3] *= corr_hi;
    }
    __syncthreads();
  }

  // ---- epilogue: O / l, write rows < s ----
#pragma unroll
  for (int nt8 = 0; nt8 < 8; nt8++) {
    if (q_abs_lo < s) {
      out[(long)q_abs_lo * D + nt8 * 8 + 2 * t] =
          __float2half(acc[nt8][0] / l_lo);
      out[(long)q_abs_lo * D + nt8 * 8 + 2 * t + 1] =
          __float2half(acc[nt8][1] / l_lo);
    }
    if (q_abs_hi < s) {
      out[(long)q_abs_hi * D + nt8 * 8 + 2 * t] =
          __float2half(acc[nt8][2] / l_hi);
      out[(long)q_abs_hi * D + nt8 * 8 + 2 * t + 1] =
          __float2half(acc[nt8][3] / l_hi);
    }
  }
}

}  // namespace bridge_flash
