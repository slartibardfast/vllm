// turing_search.cu — the kernel search space, one templated kernel.
//
// Generalizes the reference tiled kernel over (BM, BN, WARPS_R, WARPS_N):
#include <cuda_fp16.h>
#include <cstdint>
// WARPS_R = BM / 16 warps tile the rows in disjoint 16-row slices, WARPS_N
// tile the 64-wide... no — the full BN columns; each warp covers
// BN / WARPS_N columns as BN/WARPS_N/8 m16n8 tiles. Weights are staged
// n-major with scale and zero point folded in (the reference contract).
// The SASS gate and the timing harness live in search.py.
template <int BM, int BN, int BK, int WARPS_R, int WARPS_N>
__global__ void k_search(const __half* __restrict__ A,
                         const uint32_t* __restrict__ Q,
                         const __half* __restrict__ S,
                         const int32_t* __restrict__ ZP,
                         __half* __restrict__ Out, int M, int N, int K,
                         int k_words, int G) {
  __shared__ __half sA[BM][BK + 8];
  __shared__ __half sBT[BN][BK + 8];

  int n0 = blockIdx.x * BN;
  int m_base = blockIdx.y * BM;
  int tid = threadIdx.x;
  int warp = tid >> 5, lane = tid & 31;
  int warp_r = warp / WARPS_N, warp_n = warp % WARPS_N;
  int r0 = warp_r * 16;
  int n_base = warp_n * (BN / WARPS_N);
  int n_tiles = (BN / WARPS_N) / 8;
  int gg = lane >> 2, cc = lane & 3;

  float acc[BN / WARPS_N / 8][4] = {};
  constexpr int STAGE_THREADS = WARPS_R * WARPS_N * 32;

  for (int k0 = 0; k0 < K; k0 += BK) {
    for (int i = tid; i < BM * BK; i += STAGE_THREADS) {
      int r = i / BK, c = i % BK;
      int gm = m_base + r, gk = k0 + c;
      __half v = __float2half(0.f);
      if (gm < M && gk < K) v = A[(long)gm * K + gk];
      sA[r][c] = v;
    }
    for (int i = tid; i < (BK * BN) / 8; i += STAGE_THREADS) {
      int nn = i / (BK / 8);
      int kk = (i % (BK / 8)) * 8;
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
    for (int kk = 0; kk < BK; kk += 8) {
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

#pragma unroll
  for (int nt = 0; nt < BN / WARPS_N / 8; nt++) {
#pragma unroll
    for (int half = 0; half < 2; half++) {
      int r = m_base + r0 + gg + half * 8;
      int c = n0 + n_base + nt * 8 + cc * 2;
      if (r < M) {
        Out[(long)r * N + c] = __float2half(acc[nt][half * 2]);
        Out[(long)r * N + c + 1] = __float2half(acc[nt][half * 2 + 1]);
      }
    }
  }
}

template <int BM, int BN, int BK, int WR, int WN>
void launch_k(const __half* A, const uint32_t* Q, const __half* S,
              const int32_t* ZP, __half* Out, int M, int N, int K,
              int k_words, int G) {
  dim3 grid(N / BN, (M + BM - 1) / BM);
  k_search<BM, BN, BK, WR, WN><<<grid, WR * WN * 32>>>(A, Q, S, ZP, Out, M, N,
                                                       K, k_words, G);
}

// The generated table's search space: 8 candidates.
extern "C" {

int variant_count() { return 8; }

const char* variant_name(int i) {
  static const char* names[8] = {
      "bm64_bn64_w4", "bm64_bn128_w4", "bm64_bn128_w8", "bm128_bn64_w8",
      "bm128_bn128_w8", "bm32_bn64_w2", "bm32_bn128_w4", "bm64_bn64_w8"};
  return names[i];
}

void variant_launch(int i, const __half* A, const uint32_t* Q,
                    const __half* S, const int32_t* ZP, __half* Out, int M,
                    int N, int K, int k_words, int G, int* bm_out,
                    int* bn_out, int* thr_out) {
  switch (i) {
    case 0: launch_k<64, 64, 64, 4, 1>(A, Q, S, ZP, Out, M, N, K, k_words, G); *bm_out = 64; *bn_out = 64; *thr_out = 128; break;
    case 1: launch_k<64, 128, 64, 4, 1>(A, Q, S, ZP, Out, M, N, K, k_words, G); *bm_out = 64; *bn_out = 128; *thr_out = 128; break;
    case 2: launch_k<64, 128, 64, 4, 2>(A, Q, S, ZP, Out, M, N, K, k_words, G); *bm_out = 64; *bn_out = 128; *thr_out = 256; break;
    case 3: launch_k<128, 64, 64, 8, 1>(A, Q, S, ZP, Out, M, N, K, k_words, G); *bm_out = 128; *bn_out = 64; *thr_out = 256; break;
    case 4: launch_k<128, 128, 64, 8, 1>(A, Q, S, ZP, Out, M, N, K, k_words, G); *bm_out = 128; *bn_out = 128; *thr_out = 256; break;
    case 5: launch_k<32, 64, 64, 2, 1>(A, Q, S, ZP, Out, M, N, K, k_words, G); *bm_out = 32; *bn_out = 64; *thr_out = 64; break;
    case 6: launch_k<32, 128, 64, 2, 2>(A, Q, S, ZP, Out, M, N, K, k_words, G); *bm_out = 32; *bn_out = 128; *thr_out = 128; break;
    case 7: launch_k<64, 64, 64, 4, 2>(A, Q, S, ZP, Out, M, N, K, k_words, G); *bm_out = 64; *bn_out = 64; *thr_out = 256; break;
    default: *bm_out = *bn_out = *thr_out = 0;
  }
}
}
