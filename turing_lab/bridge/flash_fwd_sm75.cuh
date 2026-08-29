// flash_fwd_sm75.cuh — bridge-native FlashAttention forward for Turing.
//
// The sm_75 execution path for flash-attn/FlashInfer/vLLM (plan/0006):
// fp16 attention built entirely on cu_sm80_on_sm75 primitives — streaming
// loads, register-staged smem tiles, the m16n8k16 MMA adapter, online
// softmax with butterfly row-reduction. No CUTLASS atoms, no cp.async,
// no upstream kernel code.
//
// v1 scope: head dim 64, fp16, causal or full attention, s a multiple of
// 64, no dropout. d=128, general s, GQA follow the v1 oracle.
#pragma once
#include <cstdint>
#include <cuda_fp16.h>
#include "sm80_on_sm75.cuh"

namespace bridge_flash {

constexpr int kBlockRows = 64;   // query rows per CTA
constexpr int kBlockKV = 64;     // kv rows per tile
constexpr int kThreads = 128;    // 4 warps
constexpr float kLog2e = 1.4426950408889634f;

// Load one 64x64 fp16 tile (row-major, row stride = head dim) global->smem.
__device__ __forceinline__ void stage_tile(const __half* __restrict__ src,
                                           int src_stride,
                                           __half* __restrict__ dst,
                                           int dst_stride, int rows_valid,
                                           int cols_valid, int tid) {
  constexpr int kV4 = 64 * 64 * 2 / 16;  // 512 v4 ops
  for (int i = tid; i < kV4; i += kThreads) {
    int r = (i * 4) / 64, c = (i * 4) % 64;
    __half* d = dst + r * dst_stride + c;
    if (r < rows_valid && c + 4 <= cols_valid) {
      uint32_t r0, r1, r2, r3;
      bridge::ldg_cs_v4(r0, r1, r2, r3, src + (long)r * src_stride + c);
      bridge::sts_v4(d, r0, r1, r2, r3);
    } else {
      for (int j = 0; j < 4; j++)
        d[j] = __float2half((r < rows_valid && c + j < cols_valid)
                                ? __half2float(src[(long)r * src_stride + c + j])
                                : 0.f);
    }
  }
}

// The forward for one (b, h): q/k/v point at the (s, d) head matrix.
__device__ __forceinline__ void flash_fwd_one(
    const __half* __restrict__ q, const __half* __restrict__ k,
    const __half* __restrict__ v, __half* __restrict__ out, int s, int d,
    bool causal, bool q_aligned, void* smem) {
  // d == 64 in v1; q_block is 64-aligned (grid covers ceil(s/64) blocks
  // per h; tail rows are computed but only written when < s).
  __half* sQ = reinterpret_cast<__half*>(smem);            // 64 x 72
  __half* sK = sQ + kBlockRows * 72;                       // 64 x 72
  __half* sV = sK + kBlockKV * 72;                         // 64 x 72

  const int tid = threadIdx.x;
  const int warp = tid >> 5, lane = tid & 31;
  const int g = lane >> 2, t = lane & 3;
  const int row_lo = warp * 16 + g;                        // local row
  const int row_hi = row_lo + 8;
  const long q_base = (long)blockIdx.y * kBlockRows;

  // ---- stage Q once (d = 64: full 64x64 tile) ----
  stage_tile(q, d, sQ, 72, min((int)(s - q_base), 64), 64, tid);
  __syncthreads();

  // Q fragments: 4 k16 steps x 4 regs (A layout, rows row_lo/row_hi)
  uint32_t qf[4][4];
#pragma unroll
  for (int st = 0; st < 4; st++) {
    const __half* r0 = &sQ[row_lo * 72 + st * 16 + 2 * t];
    const __half* r1 = &sQ[row_hi * 72 + st * 16 + 2 * t];
    qf[st][0] = *reinterpret_cast<const uint32_t*>(r0);
    qf[st][1] = *reinterpret_cast<const uint32_t*>(r0 + 8);
    qf[st][2] = *reinterpret_cast<const uint32_t*>(r1);
    qf[st][3] = *reinterpret_cast<const uint32_t*>(r1 + 8);
  }

  // O accumulator (fp32), B-fragment layout over d: acc[cc][r4],
  // cc = d/8 column group, r4 = {lo,lo+1,hi,hi+1}
  float acc[8][4];
  memset(acc, 0, sizeof(acc));
  float m_lo = -INFINITY, m_hi = -INFINITY;
  float l_lo = 0.f, l_hi = 0.f;

  const float scale_log2e = (1.f / sqrtf((float)d)) * kLog2e;
  const int n_tiles = (s + kBlockKV - 1) / kBlockKV;
  const int q_abs = (int)q_base + row_lo;    // absolute query rows
  const int q_abs_hi = (int)q_base + row_hi;

  for (int nt = 0; nt < n_tiles; nt++) {
    const int k0 = nt * kBlockKV;
    if (causal && k0 > q_abs_hi) break;      // strictly future keys

    stage_tile(k + (long)nt * kBlockKV * d, d, sK, 72,
               min(s - k0, kBlockKV), 64, tid);
    stage_tile(v + (long)nt * kBlockKV * d, d, sV, 72,
               min(s - k0, kBlockKV), 64, tid);
    __syncthreads();

    // S fragments: st[nt8][r4], nt8 = kv column group (8), r4 rows
    float st[8][4];
    memset(st, 0, sizeof(st));
#pragma unroll
    for (int stp = 0; stp < 4; stp++) {
      const int kb = stp * 16 + 2 * t;
#pragma unroll
      for (int nt8 = 0; nt8 < 8; nt8++) {
        uint32_t b0 = *reinterpret_cast<const uint32_t*>(&sK[(kb)*72 + nt8 * 8 + g]);
        uint32_t b1 = *reinterpret_cast<const uint32_t*>(&sK[(kb + 8) * 72 + nt8 * 8 + g]);
        uint32_t a[4] = {qf[stp][0], qf[stp][1], qf[stp][2], qf[stp][3]};
        float dd[4];
        bridge::mma_m16n8k16_f32(dd, a, (const uint32_t*)&b0);
        // D layout: d0={r g, c 2t}, d1={r g, c 2t+1}, d2={r g+8, c 2t},
        // d3={r g+8, c 2t+1}
        st[nt8][0] += dd[0];
        st[nt8][1] += dd[2];
        st[nt8][2] += dd[1];
        st[nt8][3] += dd[3];
        (void)b1;
      }
    }

    // causal mask on the diagonal band (kv col absolute > query row abs)
    if (causal && k0 + kBlockKV > q_abs) {
#pragma unroll
      for (int nt8 = 0; nt8 < 8; nt8++) {
        const int c0 = k0 + nt8 * 8 + 2 * t;
        const int c1 = c0 + 1;
        if (c0 > q_abs) st[nt8][0] = -INFINITY;
        if (c1 > q_abs) st[nt8][1] = -INFINITY;
        if (c0 > q_abs_hi) st[nt8][2] = -INFINITY;
        if (c1 > q_abs_hi) st[nt8][3] = -INFINITY;
      }
    }

    // row maxima (rows lo/hi) within this tile, then butterfly over t
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

    // online update (separate rows lo/hi)
    float m_new_lo = fmaxf(m_lo, rmax_lo);
    float m_new_hi = fmaxf(m_hi, rmax_hi);
    float corr_lo = (m_lo == -INFINITY) ? 0.f : exp2f((m_lo - m_new_lo));
    float corr_hi = (m_hi == -INFINITY) ? 0.f : exp2f((m_hi - m_new_hi));
    if (isinf(m_new_lo)) corr_lo = 0.f;  // fully masked row
    if (isinf(m_new_hi)) corr_hi = 0.f;

    // P = exp2((S - m_new) * scale * log2e), fp32 in the fragments
#pragma unroll
    for (int nt8 = 0; nt8 < 8; nt8++) {
      st[nt8][0] = exp2f((st[nt8][0] - m_new_lo) * scale_log2e);
      st[nt8][1] = exp2f((st[nt8][1] - m_new_lo) * scale_log2e);
      st[nt8][2] = exp2f((st[nt8][2] - m_new_hi) * scale_log2e);
      st[nt8][3] = exp2f((st[nt8][3] - m_new_hi) * scale_log2e);
    }
    // masked lanes carry -inf -> exp2 gives 0; -inf - (-inf) = NaN guard:
    // rows with rmax = -inf only occur when s < row (tail), excluded by
    // the epilogue row guard.

    // row sums of P (rows lo/hi) via butterfly over t
    float psum_lo = st[0][0] + st[0][1] + st[1][0] + st[1][1] +
                    st[2][0] + st[2][1] + st[3][0] + st[3][1];
    float psum_hi = st[0][2] + st[0][3] + st[1][2] + st[1][3] +
                    st[2][2] + st[2][3] + st[3][2] + st[3][3];
#pragma unroll
    for (int off = 1; off <= 2; off <<= 1) {
      psum_lo += __shfl_xor_sync(0xffffffff, psum_lo, off);
      psum_hi += __shfl_xor_sync(0xffffffff, psum_hi, off);
    }
    l_lo = l_lo * corr_lo + psum_lo;
    l_hi = l_hi * corr_hi + psum_hi;

    // O correction for the row maxima
#pragma unroll
    for (int cc = 0; cc < 8; cc++) {
      acc[cc][0] *= corr_lo;
      acc[cc][1] *= corr_lo;
      acc[cc][2] *= corr_hi;
      acc[cc][3] *= corr_hi;
    }

    // P·V: for each 8-column d group cc and each 16-wide kv chunk
    // P A-fragment: a0 = {P[lo][c], P[lo][c+1]} = st[nt8][0] bits,
    //               a1 = {P[hi][c], P[hi][c+1]} = st[nt8][2],
    //               a2 = st[nt8][1], a3 = st[nt8][3]
    // (D->A fragment correspondence established by the oracle)
    const int kc = 0;  // kv chunk base inside the tile (64 = 4 chunks)
#pragma unroll
    for (int cc = 0; cc < 8; cc++) {
#pragma unroll
      for (int kc16 = 0; kc16 < 4; kc16++) {
        uint32_t a[4];
        a[0] = __half_as_ushort(__float2half(st[kc16 * 2][0])) |
               ((uint32_t)__half_as_ushort(__float2half(st[kc16 * 2 + 1][0])) << 16);
        a[1] = __half_as_ushort(__float2half(st[kc16 * 2][3])) |
               ((uint32_t)__half_as_ushort(__float2half(st[kc16 * 2 + 1][3])) << 16);
        a[2] = __half_as_ushort(__float2half(st[kc16 * 2][1])) |
               ((uint32_t)__half_as_ushort(__float2half(st[kc16 * 2 + 1][1])) << 16);
        a[3] = __half_as_ushort(__float2half(st[kc16 * 2][3 + 1])) |
               ((uint32_t)__half_as_ushort(__float2half(st[kc16 * 2 + 1][3 + 1])) << 16);
        // V B-fragment: rows k0+kc16*16+2t (+1), +8; col cc*8+g
        const __half* v0 = &sV[(kc16 * 16 + 2 * t) * 72 + cc * 8 + g];
        const __half* v1 = &sV[(kc16 * 16 + 16 + 2 * t) * 72 + cc * 8 + g];
        uint32_t b0 = *reinterpret_cast<const uint32_t*>(v0);
        uint32_t b1 = *reinterpret_cast<const uint32_t*>(v1);
        bridge::mma_m16n8k16_f32(acc[cc], a, (const uint32_t*)&b0);
        (void)b1;
      }
    }
    __syncthreads();
    (void)kc; (void)q_abs_hi;
  }

  // ---- epilogue: O / l, write the fragment values ----
  if (q_abs < s) {
    float inv_lo = 1.f / l_lo, inv_hi = 1.f / l_hi;
    out[(long)q_abs * d + cc_off(t) + g] = __float2half(acc[0][0] * inv_lo);
  }
  (void)q_abs_hi;
}

// helper placeholder — replaced in the epilogue rewrite
__device__ __forceinline__ int cc_off(int) { return 0; }
