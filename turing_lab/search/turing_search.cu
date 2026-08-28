// turing_search.cu — the kernel search space, three templated strategies.
//
// STRATEGY 0 "staged": weights dequantized to fp16 in shared memory at
// staging time (scale and zero point folded there; the only ZP-capable
// strategy — the reference contract).
// STRATEGY 1 "regdequant": packed words staged raw, split into even/odd
// nibble lanes; dequant happens in registers per fragment pair via
// byte_perm + the 0x6400 bias trick; scales applied in the accumulator
// domain. Symmetric-only (the 1032 bias bakes in zp=8).
// STRATEGY 2 "pipe": per-thread staging registers, double-buffered
// shared memory, dequant at store time. BK is fixed at 32 (two k8 fragment
// pairs per chunk), THREADS = max(BM, BN) * 2.
//
// Split-K is runtime: grid.z slices the K range, each slice writes fp32
// partials to a workspace, k_reduce sums them. nz == 1 writes fp16
// directly with no workspace. The SASS gate and the timing harness live
// in search.py / harness.py.
#include <cuda_fp16.h>
#include <cstdint>

#define DEVINL __device__ __forceinline__

DEVINL uint32_t lop3_or_and(uint32_t a, uint32_t mask, uint32_t ex) {
  uint32_t r;
  asm volatile("lop3.b32 %0, %1, %2, %3, 0xea;\n" : "=r"(r) : "r"(a), "r"(mask), "r"(ex));
  return r;
}

// ---------------- strategy 0: staged dequant in shared memory ----------------
template <int BM, int BN, int BK, int WARPS_R, int WARPS_N>
__device__ void staged_body(const __half* __restrict__ A,
                            const uint32_t* __restrict__ Q,
                            const __half* __restrict__ S,
                            const int32_t* __restrict__ ZP,
                            __half* __restrict__ Out, float* __restrict__ Part,
                            int M, int N, int K, int k_words, int G,
                            int k_begin, int k_end, int nz) {
  __shared__ __half sA[BM][BK + 8];
  __shared__ __half sBT[BN][BK + 8];

  int n0 = blockIdx.x * BN;
  int m_base = blockIdx.y * BM;
  int tid = threadIdx.x;
  int warp = tid >> 5, lane = tid & 31;
  int warp_r = warp / WARPS_N, warp_n = warp % WARPS_N;
  int r0 = warp_r * 16;
  int n_base = warp_n * (BN / WARPS_N);
  int gg = lane >> 2, cc = lane & 3;

  float acc[BN / WARPS_N / 8][4] = {};
  constexpr int STAGE_THREADS = WARPS_R * WARPS_N * 32;

  for (int k0 = k_begin; k0 < k_end; k0 += BK) {
    int kk_end = min(BK, k_end - k0);
    for (int i = tid; i < BM * kk_end; i += STAGE_THREADS) {
      int r = i / kk_end, c = i % kk_end;
      int gm = m_base + r, gk = k0 + c;
      __half v = __float2half(0.f);
      if (gm < M && gk < K) v = A[(long)gm * K + gk];
      sA[r][c] = v;
    }
    for (int i = tid; i < (kk_end * BN) / 8; i += STAGE_THREADS) {
      int nn = i / (kk_end / 8);
      int kk = (i % (kk_end / 8)) * 8;
      int g = (k0 + kk) / G;
      uint32_t word = Q[(long)(n0 + nn) * k_words + ((k0 + kk) >> 3)];
      int zp = ZP ? ZP[g * N + n0 + nn] : 8;
      float sc = __half2float(S[g * N + n0 + nn]);
#pragma unroll
      for (int j = 0; j < 8; j++) {
        int q = (int)((word >> (j * 4)) & 0xFu);
        sBT[nn][kk + j] = __float2half((float)(q - zp) * sc);
      }
    }
    __syncthreads();
    for (int kk = 0; kk < kk_end; kk += 8) {
#pragma unroll
      for (int nt = 0; nt < BN / WARPS_N / 8; nt++) {
        __half2 a0 = *(__half2*)&sA[r0 + gg][kk + cc * 2];
        __half2 a1 = *(__half2*)&sA[r0 + gg + 8][kk + cc * 2];
        __half2 b = *(__half2*)&sBT[n_base + nt * 8 + gg][kk + cc * 2];
        float* dd = acc[nt];
        asm volatile("mma.sync.aligned.m16n8k8.row.col.f32.f16.f16.f32 "
                     "{%0,%1,%2,%3}, {%4,%5}, {%6}, {%0,%1,%2,%3};\n"
                     : "+f"(dd[0]), "+f"(dd[1]), "+f"(dd[2]), "+f"(dd[3])
                     : "r"(*reinterpret_cast<uint32_t*>(&a0)),
                       "r"(*reinterpret_cast<uint32_t*>(&a1)),
                       "r"(*reinterpret_cast<uint32_t*>(&b)));
      }
    }
    __syncthreads();
  }

  int z = blockIdx.z;
#pragma unroll
  for (int nt = 0; nt < BN / WARPS_N / 8; nt++) {
#pragma unroll
    for (int half = 0; half < 2; half++) {
      int r = m_base + r0 + gg + half * 8;
      int c = n0 + n_base + nt * 8 + cc * 2;
      if (r < M) {
        if (nz == 1) {
          Out[(long)r * N + c] = __float2half(acc[nt][half * 2]);
          Out[(long)r * N + c + 1] = __float2half(acc[nt][half * 2 + 1]);
        } else {
          float* dst = Part + ((long)z * M + r) * N + c;
          dst[0] = acc[nt][half * 2];
          dst[1] = acc[nt][half * 2 + 1];
        }
      }
    }
  }
}

// ------------- strategy 1: register dequant (skip-flop bias trick) -------------
template <int BM, int BN, int BK, int WARPS_R, int WARPS_N>
__device__ void regdeq_body(const __half* __restrict__ A,
                            const uint32_t* __restrict__ Q,
                            const __half* __restrict__ S,
                            __half* __restrict__ Out, float* __restrict__ Part,
                            int M, int N, int K, int k_words, int G,
                            int k_begin, int k_end, int nz) {
  __shared__ __half sA[BM][BK + 8];
  // packed words split into even/odd nibble lanes, so a k,k+1 fragment pair
  // is one byte_perm away and both halves carry the same scale factor
  __shared__ uint32_t sQE[BN][BK / 8];
  __shared__ uint32_t sQO[BN][BK / 8];
  __shared__ __half sS[BN];

  int n0 = blockIdx.x * BN;
  int m_base = blockIdx.y * BM;
  int tid = threadIdx.x;
  int warp = tid >> 5, lane = tid & 31;
  int warp_r = warp / WARPS_N, warp_n = warp % WARPS_N;
  int r0 = warp_r * 16;
  int n_base = warp_n * (BN / WARPS_N);
  int gg = lane >> 2, cc = lane & 3;

  float acc[BN / WARPS_N / 8][4] = {};
  float part[BN / WARPS_N / 8][4];
  constexpr int STAGE_THREADS = WARPS_R * WARPS_N * 32;
  const __half2 bias1032 = __float2half2_rn(1032.0f);

  for (int k0 = k_begin; k0 < k_end; k0 += BK) {
    int kk_end = min(BK, k_end - k0);
    for (int i = tid; i < BM * kk_end; i += STAGE_THREADS) {
      int r = i / kk_end, c = i % kk_end;
      int gm = m_base + r, gk = k0 + c;
      __half v = __float2half(0.f);
      if (gm < M && gk < K) v = A[(long)gm * K + gk];
      sA[r][c] = v;
    }
    for (int i = tid; i < (kk_end * BN) / 8; i += STAGE_THREADS) {
      int nn = i / (kk_end / 8);
      int wcol = i % (kk_end / 8);
      uint32_t w = Q[(long)(n0 + nn) * k_words + ((k0 + wcol * 8) >> 3)];
      sQE[nn][wcol] = w & 0x0f0f0f0fu;
      sQO[nn][wcol] = (w >> 4) & 0x0f0f0f0fu;
    }
    // stage scales for this chunk's group
    int g = k0 / G;
    for (int i = tid; i < BN; i += STAGE_THREADS) {
      sS[i] = S[g * N + n0 + i];
    }
    __syncthreads();
#pragma unroll
    for (int nt = 0; nt < BN / WARPS_N / 8; nt++) {
#pragma unroll
      for (int e = 0; e < 4; e++) part[nt][e] = 0.f;
    }

    for (int kk = 0; kk < kk_end; kk += 8) {
#pragma unroll
      for (int nt = 0; nt < BN / WARPS_N / 8; nt++) {
        int col = n_base + nt * 8 + gg;
        uint32_t wE = sQE[col][kk / 8];
        uint32_t wO = sQO[col][kk / 8];
        // interleave the lane's k,k+1 nibble pair into half lanes, bias 1024
        uint32_t spread = __byte_perm(wE, wO, 0x0400 + (cc << 8) + cc);
        uint32_t hbits = lop3_or_and(spread, 0x000F000Fu, 0x64006400u);
        __half2 h2 = *reinterpret_cast<__half2*>(&hbits);
        __half2 w2 = __hsub2(h2, bias1032);
        __half2 a0 = *(__half2*)&sA[r0 + gg][kk + cc * 2];
        __half2 a1 = *(__half2*)&sA[r0 + gg + 8][kk + cc * 2];
        float* dd = part[nt];
        asm volatile("mma.sync.aligned.m16n8k8.row.col.f32.f16.f16.f32 "
                     "{%0,%1,%2,%3}, {%4,%5}, {%6}, {%0,%1,%2,%3};\n"
                     : "+f"(dd[0]), "+f"(dd[1]), "+f"(dd[2]), "+f"(dd[3])
                     : "r"(*reinterpret_cast<uint32_t*>(&a0)),
                       "r"(*reinterpret_cast<uint32_t*>(&a1)),
                       "r"(*reinterpret_cast<uint32_t*>(&w2)));
      }
    }
    // scales in the accumulator domain, applied to this chunk's partial
    // before it joins the running sum: lane covers rows gg / gg+8 and
    // columns cc*2, cc*2+1 of its 8-column tile
    __syncthreads();
#pragma unroll
    for (int nt = 0; nt < BN / WARPS_N / 8; nt++) {
      float s0 = __half2float(sS[n_base + nt * 8 + cc * 2]);
      float s1 = __half2float(sS[n_base + nt * 8 + cc * 2 + 1]);
      acc[nt][0] += part[nt][0] * s0;
      acc[nt][1] += part[nt][1] * s1;
      acc[nt][2] += part[nt][2] * s0;
      acc[nt][3] += part[nt][3] * s1;
    }
    __syncthreads();
  }

  int z = blockIdx.z;
#pragma unroll
  for (int nt = 0; nt < BN / WARPS_N / 8; nt++) {
#pragma unroll
    for (int half = 0; half < 2; half++) {
      int r = m_base + r0 + gg + half * 8;
      int c = n0 + n_base + nt * 8 + cc * 2;
      if (r < M) {
        if (nz == 1) {
          Out[(long)r * N + c] = __float2half(acc[nt][half * 2]);
          Out[(long)r * N + c + 1] = __float2half(acc[nt][half * 2 + 1]);
        } else {
          float* dst = Part + ((long)z * M + r) * N + c;
          dst[0] = acc[nt][half * 2];
          dst[1] = acc[nt][half * 2 + 1];
        }
      }
    }
  }
}

// ------------------- strategy 2: double-buffered pipeline -------------------
// Per-thread staging registers; A: rows [tid/2], halves [seg*16..+16);
// B: rows [tid/2], words [wpair*2..+2). Requires THREADS == max(BM, BN)*2
// and BK == 32 (16 halves per thread segment = 4 uint32 of A, 2 words of B).
template <int BM, int BN, int WARPS_R, int WARPS_N>
__device__ void pipe_body(const __half* __restrict__ A,
                          const uint32_t* __restrict__ Q,
                          const __half* __restrict__ S,
                          __half* __restrict__ Out, float* __restrict__ Part,
                          int M, int N, int K, int k_words, int G,
                          int k_begin, int k_end, int nz) {
  constexpr int BK = 32;
  __shared__ __half sA[2][BM][BK + 8];
  __shared__ __half sBT[2][BN][BK + 8];

  int n0 = blockIdx.x * BN;
  int m_base = blockIdx.y * BM;
  int tid = threadIdx.x;
  int warp = tid >> 5, lane = tid & 31;
  int r0 = (warp / WARPS_N) * 16;
  int n_base = (warp % WARPS_N) * (BN / WARPS_N);
  int gg = lane >> 2, cc = lane & 3;

  float acc[BN / WARPS_N / 8][4] = {};

  uint32_t rA[8], rW[2];
  __half rS;

  auto load_regs = [&](int k0) {
    int r = tid / 2, seg = tid % 2;
    if (r < BM) {
      int gm = m_base + r, gk = k0 + seg * 16;
      if (gm < M && gk + 15 < K) {
        const uint32_t* src = reinterpret_cast<const uint32_t*>(&A[(long)gm * K + gk]);
#pragma unroll
        for (int i = 0; i < 8; i++) rA[i] = src[i];
      } else {
#pragma unroll
        for (int i = 0; i < 8; i++) rA[i] = 0u;
      }
    }
    int nn = tid / 2;
    if (nn < BN) {
      int gn = n0 + nn, wpair = tid % 2;
      if (gn < N && k0 + wpair * 16 < K) {
        rW[0] = Q[(long)gn * k_words + (k0 >> 3) + wpair * 2];
        rW[1] = Q[(long)gn * k_words + (k0 >> 3) + wpair * 2 + 1];
      } else {
        rW[0] = rW[1] = 0u;
      }
      rS = S[(k0 / G) * N + gn];
    }
  };

  auto store_shared = [&](int buf) {
    int r = tid / 2, seg = tid % 2;
    if (r < BM) {
#pragma unroll
      for (int i = 0; i < 8; i++)
        *reinterpret_cast<uint32_t*>(&sA[buf][r][seg * 16 + i * 2]) = rA[i];
    }
    int nn = tid / 2;
    if (nn < BN) {
      float sc = __half2float(rS);
      __half* dst = &sBT[buf][nn][(tid % 2) * 16];
      uint32_t w0 = rW[0], w1 = rW[1];
#pragma unroll
      for (int j = 0; j < 8; j++) {
        dst[j] = __float2half((float)((int)((w0 >> (4 * j)) & 0xFu) - 8) * sc);
        dst[8 + j] = __float2half((float)((int)((w1 >> (4 * j)) & 0xFu) - 8) * sc);
      }
    }
  };

  auto compute = [&](int buf, int cols) {
    for (int kk = 0; kk < cols; kk += 8) {
#pragma unroll
      for (int nt = 0; nt < BN / WARPS_N / 8; nt++) {
        __half2 a0 = *reinterpret_cast<const __half2*>(&sA[buf][r0 + gg][kk + cc * 2]);
        __half2 a1 = *reinterpret_cast<const __half2*>(&sA[buf][r0 + gg + 8][kk + cc * 2]);
        __half2 b = *reinterpret_cast<const __half2*>(&sBT[buf][n_base + nt * 8 + gg][kk + cc * 2]);
        float* dd = acc[nt];
        asm volatile("mma.sync.aligned.m16n8k8.row.col.f32.f16.f16.f32 "
                     "{%0,%1,%2,%3}, {%4,%5}, {%6}, {%0,%1,%2,%3};\n"
                     : "+f"(dd[0]), "+f"(dd[1]), "+f"(dd[2]), "+f"(dd[3])
                     : "r"(*reinterpret_cast<uint32_t*>(&a0)),
                       "r"(*reinterpret_cast<uint32_t*>(&a1)),
                       "r"(*reinterpret_cast<uint32_t*>(&b)));
      }
    }
  };

  int slice = (k_end - k_begin + BK - 1) / BK;
  int n_chunks = slice > 0 ? slice : 0;

  if (n_chunks > 0) load_regs(k_begin);
  for (int c = 0; c < n_chunks; c++) {
    int k0 = k_begin + c * BK;
    int cols = min(BK, k_end - k0);
    store_shared(c & 1);
    __syncthreads();
    if (c + 1 < n_chunks) load_regs(k0 + BK);
    compute(c & 1, cols);
    __syncthreads();
  }

  int z = blockIdx.z;
#pragma unroll
  for (int nt = 0; nt < BN / WARPS_N / 8; nt++) {
#pragma unroll
    for (int half = 0; half < 2; half++) {
      int r = m_base + r0 + gg + half * 8;
      int c = n0 + n_base + nt * 8 + cc * 2;
      if (r < M) {
        if (nz == 1) {
          Out[(long)r * N + c] = __float2half(acc[nt][half * 2]);
          Out[(long)r * N + c + 1] = __float2half(acc[nt][half * 2 + 1]);
        } else {
          float* dst = Part + ((long)z * M + r) * N + c;
          dst[0] = acc[nt][half * 2];
          dst[1] = acc[nt][half * 2 + 1];
        }
      }
    }
  }
}

// ------------------------------- dispatcher -------------------------------
template <int S, int BM, int BN, int BK, int WR, int WN>
__global__ void k_search(const __half* __restrict__ A,
                         const uint32_t* __restrict__ Q,
                         const __half* __restrict__ S_,
                         const int32_t* __restrict__ ZP,
                         __half* __restrict__ Out, float* __restrict__ Part,
                         int M, int N, int K, int k_words, int G) {
  int slice = (K + gridDim.z - 1) / gridDim.z;
  int k_begin = blockIdx.z * slice;
  int k_end = min(K, k_begin + slice);
  if (S == 0)
    staged_body<BM, BN, BK, WR, WN>(A, Q, S_, ZP, Out, Part, M, N, K, k_words,
                                    G, k_begin, k_end, gridDim.z);
  else if (S == 1)
    regdeq_body<BM, BN, BK, WR, WN>(A, Q, S_, Out, Part, M, N, K, k_words, G,
                                    k_begin, k_end, gridDim.z);
  else
    pipe_body<BM, BN, WR, WN>(A, Q, S_, Out, Part, M, N, K, k_words, G,
                              k_begin, k_end, gridDim.z);
}

__global__ void k_reduce(const float* __restrict__ P,
                         __half* __restrict__ Out, long total, int nz) {
  long i = (long)blockIdx.x * blockDim.x + threadIdx.x;
  if (i < total) {
    float s = 0.f;
    for (int z = 0; z < nz; z++) s += P[z * total + i];
    Out[i] = __float2half(s);
  }
}

template <int S, int BM, int BN, int BK, int WR, int WN>
void launch_k(const __half* A, const uint32_t* Q, const __half* Sc,
              const int32_t* ZP, __half* Out, float* Part, int M, int N,
              int K, int k_words, int G, int nz) {
  dim3 grid(N / BN, (M + BM - 1) / BM, nz);
  k_search<S, BM, BN, BK, WR, WN>
      <<<grid, WR * WN * 32>>>(A, Q, Sc, ZP, Out, Part, M, N, K, k_words, G);
}

// the variant table: one X-macro row per candidate; names carry the full
// config so the python side can parse legality from them alone
#define FOR_EACH_VARIANT(V)                                                        \
  V(0, staged, 64, 64, 64, 4, 1)                                                   \
  V(1, staged, 64, 128, 64, 4, 1)                                                  \
  V(2, staged, 64, 128, 64, 4, 2)                                                  \
  V(3, staged, 128, 64, 64, 8, 1)                                                  \
  V(4, staged, 128, 128, 64, 8, 1)                                                 \
  V(5, staged, 32, 64, 64, 2, 1)                                                   \
  V(6, staged, 32, 128, 64, 2, 2)                                                  \
  V(7, staged, 64, 64, 64, 4, 2)                                                   \
  V(8, regdeq, 64, 64, 64, 4, 1)                                                   \
  V(9, regdeq, 64, 128, 64, 4, 1)                                                  \
  V(10, regdeq, 64, 128, 64, 4, 2)                                                 \
  V(11, regdeq, 128, 64, 64, 8, 1)                                                 \
  V(12, regdeq, 128, 128, 64, 8, 1)                                                \
  V(13, regdeq, 32, 64, 64, 2, 1)                                                  \
  V(14, regdeq, 32, 128, 64, 2, 2)                                                 \
  V(15, regdeq, 64, 64, 64, 4, 2)                                                  \
  V(16, pipe, 64, 64, 32, 4, 1)                                                    \
  V(17, pipe, 128, 64, 32, 8, 1)                                                   \
  V(18, pipe, 64, 128, 32, 4, 2)                                                   \
  V(19, pipe, 128, 128, 32, 8, 1)

extern "C" {

int variant_count() { return 20; }

#define VARIANT_STR(s, bm, bn, bk, wr, wn) #s "_" #bm "_" #bn "_" #bk "_w" #wr "x" #wn
#define VARIANT_NAME(i, s, bm, bn, bk, wr, wn) VARIANT_STR(s, bm, bn, bk, wr, wn),

const char* variant_name(int i) {
  static const char* names[] = {FOR_EACH_VARIANT(VARIANT_NAME)};
  return (i >= 0 && i < 20) ? names[i] : "?";
}

#define VARIANT_CASE(i, s, bm, bn, bk, wr, wn)                                    \
  case i:                                                                         \
    launch_k<i == 0 ? 0 : (i <= 7 ? 0 : (i <= 15 ? 1 : 2)), bm, bn, bk, wr, wn>(  \
        A, Q, S, ZP, Out, Part, M, N, K, k_words, G, nz);                         \
    *bm_out = bm;                                                                 \
    *bn_out = bn;                                                                 \
    *bk_out = bk;                                                                 \
    *thr_out = wr * wn * 32;                                                      \
    *strat_out = (i <= 7) ? 0 : (i <= 15 ? 1 : 2);                                \
    break;

void variant_launch(int i, const __half* A, const uint32_t* Q,
                    const __half* S, const int32_t* ZP, __half* Out,
                    float* Part, int M, int N, int K, int k_words, int G,
                    int nz, int* bm_out, int* bn_out, int* bk_out,
                    int* thr_out, int* strat_out) {
  switch (i) { FOR_EACH_VARIANT(VARIANT_CASE) default: *bm_out = *bn_out = *bk_out = *thr_out = *strat_out = 0; }
}

void reduce_launch(const float* P, __half* Out, long total, int nz) {
  long blocks = (total + 255) / 256;
  k_reduce<<<(unsigned)blocks, 256>>>(P, Out, total, nz);
}
}
