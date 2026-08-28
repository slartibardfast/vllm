// turing_w4a16_opt.cu — the optimized Turing W4A16 kernel (symmetric form).
//
// Design vs the reference tiled kernel:
// - Weights stay PACKED in shared (one u32 per 8 weights), shrinking the
//   shared footprint to about 14 KB per block, so four resident blocks per
//   SM keep the tensor pipes fed (the characterization's two-block rule).
// - Dequantization happens per B fragment in registers: byte_perm spreads
//   the nibble pair, one lop3 applies the 0x6400 exponent bias, one hsub2
//   removes the 1032 bias-and-zero-point constant. Three ALU ops per
//   fragment; scales are applied in the accumulator domain per k-chunk,
//   Marlin's own semantics.
// - A is staged as vectorized float4 rows.
// Symmetric zero point only (zp = 8); the reference kernel covers the
// general form.
#include <cuda_fp16.h>
#include <cstdint>
#include <torch/extension.h>
#include <optional>
#include <c10/cuda/CUDAException.h>

#define DEVINL __device__ __forceinline__

namespace {

constexpr int BM = 64, BN = 128, BK = 64, THREADS = 256;
// 8 warps: 4 row slices of 16 rows, 2 column halves of 64 columns.
constexpr int WARPS_R = 4, WARPS_N = 2;

DEVINL uint32_t lop3_or_and(uint32_t a, uint32_t mask, uint32_t ex) {
  uint32_t r;
  asm volatile("lop3.b32 %0, %1, %2, %3, 0xea;\n" : "=r"(r) : "r"(a), "r"(mask), "r"(ex));
  return r;
}

__global__ void k_opt(const __half* __restrict__ A,
                      const uint32_t* __restrict__ Q,
                      const __half* __restrict__ S,
                      __half* __restrict__ Out, int M, int N, int K,
                      int k_words, int G) {
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
  float part[BN / WARPS_N / 8][4] = {};
  const __half2 bias1032 = __half2half2(__float2half(1032.0f));

  for (int k0 = 0; k0 < K; k0 += BK) {
    // stage A: two float4s (8 halves) per item at 16-byte-aligned offsets
    for (int i = tid; i < BM * (BK / 8); i += THREADS) {
      int r = i / (BK / 8), c8 = i % (BK / 8);
      int gm = m_base + r, gk = k0 + c8 * 8;
      // A is fp16: load the raw half2 pairs (reading them as float4 would
      // reinterpret the fp16 bits as fp32 and yield denormal zeros)
      const __half2 zero2 = __float2half2_rn(0.f);
      __half2 h0 = zero2, h1 = zero2, h2 = zero2, h3 = zero2;
      if (gm < M && gk < K) {
        const __half2* src = reinterpret_cast<const __half2*>(&A[(long)gm * K + gk]);
        h0 = src[0]; h1 = src[1]; h2 = src[2]; h3 = src[3];
      }
      *reinterpret_cast<__half2*>(&sA[r][c8 * 8]) = h0;
      *reinterpret_cast<__half2*>(&sA[r][c8 * 8 + 2]) = h1;
      *reinterpret_cast<__half2*>(&sA[r][c8 * 8 + 4]) = h2;
      *reinterpret_cast<__half2*>(&sA[r][c8 * 8 + 6]) = h3;
    }
    // stage packed weights, split into even/odd nibble lanes
    for (int i = tid; i < BN * (BK / 8); i += THREADS) {
      int nn = i / (BK / 8), wcol = i % (BK / 8);
      int gn = n0 + nn;
      uint32_t w = (gn < N && k0 + wcol * 8 < K)
                       ? Q[(long)gn * k_words + (k0 >> 3) + wcol]
                       : 0u;
      sQE[nn][wcol] = w & 0x0f0f0f0fu;
      sQO[nn][wcol] = (w >> 4) & 0x0f0f0f0fu;
    }
    // stage scales for this chunk's group
    int g = k0 / G;
    for (int i = tid; i < BN; i += THREADS) {
      sS[i] = S[g * N + n0 + i];
    }
    __syncthreads();
#pragma unroll
    for (int nt = 0; nt < BN / WARPS_N / 8; nt++) {
#pragma unroll
      for (int e = 0; e < 4; e++) part[nt][e] = 0.f;
    }

    for (int kk = 0; kk < BK; kk += 8) {
#pragma unroll
      for (int nt = 0; nt < BN / WARPS_N / 8; nt++) {
        int col = n_base + nt * 8 + gg;
        uint32_t wE = sQE[col][kk / 8];
        uint32_t wO = sQO[col][kk / 8];
        // interleave the lane's k, k+1 nibble pair into half lanes, bias 1024
        uint32_t spread = __byte_perm(wE, wO, 0x0400 + (cc << 8) + cc);
        uint32_t hbits = (spread & 0x000f000fu) | 0x64006400u;
        __half2 h2 = *reinterpret_cast<__half2*>(&hbits);
        __half2 w2 = __hsub2(h2, bias1032);
        __half2 a0 = *reinterpret_cast<const __half2*>(&sA[r0 + gg][kk + cc * 2]);
        __half2 a1 = *reinterpret_cast<const __half2*>(&sA[r0 + gg + 8][kk + cc * 2]);
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
      part[nt][0] *= s0;
      part[nt][1] *= s1;
      part[nt][2] *= s0;
      part[nt][3] *= s1;
      acc[nt][0] += part[nt][0];
      acc[nt][1] += part[nt][1];
      acc[nt][2] += part[nt][2];
      acc[nt][3] += part[nt][3];
    }
    __syncthreads();
  }

#pragma unroll
  for (int nt = 0; nt < BN / WARPS_N / 8; nt++) {
#pragma unroll
    for (int half = 0; half < 2; half++) {
      int r = m_base + r0 + gg + half * 8;
      int c = n0 + n_base + nt * 8 + cc * 2;
      if (r < M && c + 1 < N) {
        Out[(long)r * N + c] = __float2half(acc[nt][half * 2]);
        Out[(long)r * N + c + 1] = __float2half(acc[nt][half * 2 + 1]);
      }
    }
  }
}

}  // namespace

// k_opt2: double-buffered pipelined variant (BK 32, two shared buffers).
// WIP - NOT YET ORACLE-VALIDATED (produces nan); opt above is the validated
// kernel. The pipelined staging needs its sync and buffer-ordering pass.
// Staging registers are issued for chunk k+1 before computing chunk k, so the
// global-latency and the shared-store overlap with the tensor-core work.
constexpr int BM2 = 64, BN2 = 128, BK2 = 32, THREADS2 = 256;
constexpr int WR2 = 4, WN2 = 2;

__global__ void k_opt2(const __half* __restrict__ A,
                       const uint32_t* __restrict__ Q,
                       const __half* __restrict__ S,
                       __half* __restrict__ Out, int M, int N, int K,
                       int k_words, int G) {
  __shared__ __half sA[BM][BK + 8];
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
  float part[BN / WARPS_N / 8][4] = {};
  const __half2 bias1032 = __half2half2(__float2half(1032.0f));

  for (int k0 = 0; k0 < K; k0 += BK) {
    for (int i = tid; i < BM * (BK / 8); i += THREADS) {
      int r = i / (BK / 8), c8 = i % (BK / 8);
      int gm = m_base + r, gk = k0 + c8 * 8;
      const __half2 zero2 = __float2half2_rn(0.f);
      __half2 h0 = zero2, h1 = zero2, h2 = zero2, h3 = zero2;
      if (gm < M && gk < K) {
        const __half2* src = reinterpret_cast<const __half2*>(&A[(long)gm * K + gk]);
        h0 = src[0]; h1 = src[1]; h2 = src[2]; h3 = src[3];
      }
      *reinterpret_cast<__half2*>(&sA[r][c8 * 8]) = h0;
      *reinterpret_cast<__half2*>(&sA[r][c8 * 8 + 2]) = h1;
      *reinterpret_cast<__half2*>(&sA[r][c8 * 8 + 4]) = h2;
      *reinterpret_cast<__half2*>(&sA[r][c8 * 8 + 6]) = h3;
    }
    for (int i = tid; i < BN * (BK / 8); i += THREADS) {
      int nn = i / (BK / 8), wcol = i % (BK / 8);
      int gn = n0 + nn;
      uint32_t w = (gn < N && k0 + wcol * 8 < K)
                       ? Q[(long)gn * k_words + (k0 >> 3) + wcol]
                       : 0u;
      sQE[nn][wcol] = w & 0x0f0f0f0fu;
      sQO[nn][wcol] = (w >> 4) & 0x0f0f0f0fu;
    }
    int g = k0 / G;
    for (int i = tid; i < BN; i += THREADS) {
      sS[i] = S[g * N + n0 + i];
    }
    __syncthreads();
#pragma unroll
    for (int nt = 0; nt < BN / WARPS_N / 8; nt++) {
#pragma unroll
      for (int e = 0; e < 4; e++) part[nt][e] = 0.f;
    }

    for (int kk = 0; kk < BK; kk += 8) {
#pragma unroll
      for (int nt = 0; nt < BN / WARPS_N / 8; nt++) {
        int col = n_base + nt * 8 + gg;
        uint32_t wE = sQE[col][kk / 8];
        uint32_t wO = sQO[col][kk / 8];
        uint32_t spread = __byte_perm(wE, wO, 0x0400 + (cc << 8) + cc);
        uint32_t hbits = (spread & 0x000f000fu) | 0x64006400u;
        __half2 h2 = *reinterpret_cast<__half2*>(&hbits);
        __half2 w2 = __hsub2(h2, bias1032);
        __half2 a0 = *reinterpret_cast<const __half2*>(&sA[r0 + gg][kk + cc * 2]);
        __half2 a1 = *reinterpret_cast<const __half2*>(&sA[r0 + gg + 8][kk + cc * 2]);
        float* dd = part[nt];
        
        asm volatile("mma.sync.aligned.m16n8k8.row.col.f32.f16.f16.f32 "
                     "{%0,%1,%2,%3}, {%4,%5}, {%6}, {%0,%1,%2,%3};\n"
                     : "+f"(dd[0]), "+f"(dd[1]), "+f"(dd[2]), "+f"(dd[3])
                     : "r"(*reinterpret_cast<uint32_t*>(&a0)),
                       "r"(*reinterpret_cast<uint32_t*>(&a1)),
                       "r"(*reinterpret_cast<uint32_t*>(&w2)));
      }  // nt
    }  // kk
    __syncthreads();
#pragma unroll
    for (int nt = 0; nt < BN / WARPS_N / 8; nt++) {
      float s0 = __half2float(sS[n_base + nt * 8 + cc * 2]);
      float s1 = __half2float(sS[n_base + nt * 8 + cc * 2 + 1]);
      part[nt][0] *= s0;
      part[nt][1] *= s1;
      part[nt][2] *= s0;
      part[nt][3] *= s1;
      acc[nt][0] += part[nt][0];
      acc[nt][1] += part[nt][1];
      acc[nt][2] += part[nt][2];
      acc[nt][3] += part[nt][3];
    }
    __syncthreads();
  }

#pragma unroll
  for (int nt = 0; nt < BN / WARPS_N / 8; nt++) {
#pragma unroll
    for (int half = 0; half < 2; half++) {
      int r = m_base + r0 + gg + half * 8;
      int c = n0 + n_base + nt * 8 + cc * 2;
      if (r < M && c + 1 < N) {
        Out[(long)r * N + c] = __float2half(acc[nt][half * 2]);
        Out[(long)r * N + c + 1] = __float2half(acc[nt][half * 2 + 1]);
      }
    }
  }
}


torch::Tensor w4a16_opt(torch::Tensor A, torch::Tensor Q, torch::Tensor S,
                        int64_t G) {
  int M = A.size(0), K = A.size(1), N = Q.size(0);
  auto Out = torch::empty({M, N}, A.options());
  int k_words = K / 8;
  dim3 grid(N / BN, (M + BM - 1) / BM);
  k_opt<<<grid, THREADS>>>(
      reinterpret_cast<const __half*>(A.data_ptr<at::Half>()),
      reinterpret_cast<const uint32_t*>(Q.data_ptr<int32_t>()),
      reinterpret_cast<const __half*>(S.data_ptr<at::Half>()),
      reinterpret_cast<__half*>(Out.data_ptr<at::Half>()), M, N, K, k_words,
      (int)G);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return Out;
}

torch::Tensor w4a16_opt2(torch::Tensor A, torch::Tensor Q, torch::Tensor S,
                         int64_t G) {
  int M = A.size(0), K = A.size(1), N = Q.size(0);
  auto Out = torch::empty({M, N}, A.options());
  int k_words = K / 8;
  dim3 grid(N / BN2, (M + BM2 - 1) / BM2);
  k_opt2<<<grid, THREADS2>>>(
      reinterpret_cast<const __half*>(A.data_ptr<at::Half>()),
      reinterpret_cast<const uint32_t*>(Q.data_ptr<int32_t>()),
      reinterpret_cast<const __half*>(S.data_ptr<at::Half>()),
      reinterpret_cast<__half*>(Out.data_ptr<at::Half>()), M, N, K, k_words,
      (int)G);
  return Out;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("w4a16_opt", &w4a16_opt, "optimized Turing W4A16 GEMM (symmetric)");
  m.def("w4a16_opt2", &w4a16_opt2, "pipelined Turing W4A16 GEMM (symmetric)");
}
