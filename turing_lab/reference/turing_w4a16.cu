// turing_w4a16.cu — the reference Turing W4A16 backend kernels.
//
// Weight contract (documented, simple, oracle-checked; Marlin-format
// consumption is the optimized kernel's job, not the reference's):
//   qweight[n, k/8] : uint32, nibble (k%8) holds the 4-bit value of w[n,k],
//                     low nibble first.
//   scales[g, n]    : fp16, one scale per group of G k-elements.
//   zps[g, n]       : int32 zero points, or nullptr for the symmetric form
//                     where the zero point is the constant 8.
//   w[n, k] = (q - zp) * scales[g(k), n]
//
// Two kernels share that contract:
//   w4a16_naive — one thread per output element, fp32 CUDA-core math. Slow
//                 by construction; it is the on-GPU ground truth.
//   w4a16_tiled — shared-staged A, dequantize-B-to-shared, m16n8k8 FP16 MMA
//                 with FP32 accumulate (the sm_75 shape), scales applied in
//                 the accumulator domain per k-chunk. This is the shape the
//                 optimized backend will grow from.

#include <cuda_fp16.h>
#include <cstdint>
#include <torch/extension.h>
#include <optional>

#define DEVINL __device__ __forceinline__

namespace {

DEVINL uint32_t nibble(const uint32_t* q, long n, long k, long k_words) {
  return (q[n * k_words + (k >> 3)] >> ((k & 7) * 4)) & 0xFu;
}

__global__ void k_naive(const __half* __restrict__ A,
                        const uint32_t* __restrict__ Q,
                        const __half* __restrict__ S,
                        const int32_t* __restrict__ ZP,
                        __half* __restrict__ Out, int M, int N, int K,
                        int k_words, int G) {
  long idx = (long)blockIdx.x * blockDim.x + threadIdx.x;
  if ((long)M * N <= idx) return;
  int m = (int)(idx / N), n = (int)(idx % N);
  float acc = 0.f;
  for (int k0 = 0; k0 < K; k0 += G) {
    float chunk = 0.f;
    int k_end = min(k0 + G, K);
    for (int k = k0; k < k_end; k++) {
      int q = (int)nibble(Q, n, k, k_words);
      int zp = ZP ? ZP[(k0 / G) * N + n] : 8;
      float w = (float)(q - zp);
      chunk += __half2float(A[(long)m * K + k]) * w;
    }
    acc += chunk * __half2float(S[(k0 / G) * N + n]);
  }
  Out[(long)m * N + n] = __float2half(acc);
}

// Tiled: BM=64 guarded on M, BN=64, BK=64. 256 threads.
// sA[64][65] fp16 (padded against conflicts), sB[64][64] fp16 dequantized
// weights (q - zp, scale applied later in the accumulator domain), sS[64]
// scales for this k-chunk, one chunk per block k-step.
constexpr int BM = 64, BN = 64, BK = 64, THREADS = 128;  // 4 warps,
// each owning a disjoint 16-row slice of the tile

__global__ void k_tiled(const __half* __restrict__ A,
                        const uint32_t* __restrict__ Q,
                        const __half* __restrict__ S,
                        const int32_t* __restrict__ ZP,
                        __half* __restrict__ Out, int M, int N, int K,
                        int k_words, int G) {
  __shared__ __half sA[BM][BK + 8];  // pad keeps half2 fragment loads aligned
  __shared__ __half sBT[BN][BK + 8];  // B staged n-major: the mma B
                                      // fragment pairs consecutive k. Weights
                                      // are staged WITH scale and zero point
                                      // folded in, so any group size works
                                      // (G < BK spans several groups per
                                      // chunk); the optimized kernel moves the
                                      // scale back to the accumulator domain,
                                      // which is Marlin's own semantics.

  int n0 = blockIdx.x * BN;
  int m_base = blockIdx.y * BM;
  int tid = threadIdx.x;
  int warp = tid >> 5, lane = tid & 31;
  int r0 = warp * 16;  // this warp's disjoint 16-row slice

  // Each warp owns rows warp*16 + {0..15} x all 64 cols: one m16n8k8
  // fragment walk covers a disjoint 16-row slice, so no duplicated work and
  // no cross-warp store collisions.
  float acc[8][4] = {};  // final accumulators

  for (int k0 = 0; k0 < K; k0 += BK) {
    // stage A: 64x64 halves = 4096; 256 threads x 16
    for (int i = tid; i < BM * BK; i += THREADS) {
      int r = i / BK, c = i % BK;
      int gm = m_base + r, gk = k0 + c;
      __half v = __float2half(0.f);
      if (gm < M && gk < K) v = A[(long)gm * K + gk];
      sA[r][c] = v;
    }
    // stage B: dequant 64x64 = 4096 values; 8 nibbles per word -> 512 words;
    // 256 threads x 2 words. Each word lies inside one group (G is a
    // multiple of 8), so the group index is per word.
    for (int i = tid; i < (BK * BN) / 8; i += THREADS) {
      int widx = i;
      int nn = widx / (BK / 8);          // output col within tile
      int kk = (widx % (BK / 8)) * 8;    // k within chunk
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

    // compute this k-chunk into fresh partials: scales differ per chunk, so
    // the chunk partial is scaled in the accumulator domain before it joins
    // the running sum. Fragment walk per PTX ISA for m16n8k8 .f16.f32:
    //   lane L: gg = L >> 2, cc = L & 3
    //   A frag (2 regs): rows gg and gg+8, k cols cc*2, cc*2+1
    //   B frag (1 reg): the k pair cc*2, cc*2+1 at col gg
    //   D frag (4 f32): rows gg / gg+8, cols cc*2 / cc*2+1
    int gg = lane >> 2, cc = lane & 3;
    float d[8][4] = {};
    for (int kk = 0; kk < BK; kk += 8) {
#pragma unroll
      for (int nt = 0; nt < 8; nt++) {
        __half2 a0 = *(__half2*)&sA[r0 + gg][kk + cc * 2];
        __half2 a1 = *(__half2*)&sA[r0 + gg + 8][kk + cc * 2];
        __half2 b = *(__half2*)&sBT[nt * 8 + gg][kk + cc * 2];
        float* dd = d[nt];
        asm volatile("mma.sync.aligned.m16n8k8.row.col.f32.f16.f16.f32 "
                     "{%0,%1,%2,%3}, {%4,%5}, {%6}, {%0,%1,%2,%3};\n"
                     : "+f"(dd[0]), "+f"(dd[1]), "+f"(dd[2]), "+f"(dd[3])
                     : "r"(*reinterpret_cast<uint32_t*>(&a0)),
                       "r"(*reinterpret_cast<uint32_t*>(&a1)),
                       "r"(*reinterpret_cast<uint32_t*>(&b)));
      }
    }
#pragma unroll
    for (int nt = 0; nt < 8; nt++) {
      acc[nt][0] += d[nt][0];
      acc[nt][1] += d[nt][1];
      acc[nt][2] += d[nt][2];
      acc[nt][3] += d[nt][3];
    }
    __syncthreads();  // sA/sBT/sS restaged next iteration
  }

  // store; D frag: lane covers rows (warp*8 + gg) and (+8),
  // cols nt*8 + cc*2 (+1); M guarded, N is a multiple of 64 by contract
  int gg2 = lane >> 2, cc2 = lane & 3;
#pragma unroll
  for (int nt = 0; nt < 8; nt++) {
#pragma unroll
    for (int half = 0; half < 2; half++) {
      int r = r0 + gg2 + half * 8;
      int c = nt * 8 + cc2 * 2;
      if (m_base + r < M) {
        Out[(long)(m_base + r) * N + n0 + c] = __float2half(acc[nt][half * 2]);
        Out[(long)(m_base + r) * N + n0 + c + 1] =
            __float2half(acc[nt][half * 2 + 1]);
      }
    }
  }
}

}  // namespace

torch::Tensor w4a16_naive(torch::Tensor A, torch::Tensor Q, torch::Tensor S,
                          c10::optional<torch::Tensor> ZP, int64_t G) {
  int M = A.size(0), K = A.size(1), N = Q.size(0);
  auto Out = torch::empty({M, N}, A.options());
  int k_words = K / 8;
  const int32_t* zp = ZP.has_value() ? ZP.value().data_ptr<int32_t>() : nullptr;
  long total = (long)M * N;
  int blocks = (int)((total + 255) / 256);
  k_naive<<<blocks, 256>>>(
      reinterpret_cast<const __half*>(A.data_ptr<at::Half>()),
      reinterpret_cast<const uint32_t*>(Q.data_ptr<int32_t>()),
      reinterpret_cast<const __half*>(S.data_ptr<at::Half>()), zp,
      reinterpret_cast<__half*>(Out.data_ptr<at::Half>()), M, N, K, k_words,
      (int)G);
  return Out;
}

torch::Tensor w4a16_tiled(torch::Tensor A, torch::Tensor Q, torch::Tensor S,
                          c10::optional<torch::Tensor> ZP, int64_t G) {
  int M = A.size(0), K = A.size(1), N = Q.size(0);
  auto Out = torch::empty({M, N}, A.options());
  int k_words = K / 8;
  const int32_t* zp = ZP.has_value() ? ZP.value().data_ptr<int32_t>() : nullptr;
  dim3 grid(N / BN, (M + BM - 1) / BM);
  k_tiled<<<grid, THREADS>>>(
      reinterpret_cast<const __half*>(A.data_ptr<at::Half>()),
      reinterpret_cast<const uint32_t*>(Q.data_ptr<int32_t>()),
      reinterpret_cast<const __half*>(S.data_ptr<at::Half>()), zp,
      reinterpret_cast<__half*>(Out.data_ptr<at::Half>()), M, N, K, k_words,
      (int)G);
  return Out;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("w4a16_naive", &w4a16_naive, "reference W4A16 GEMM, CUDA-core");
  m.def("w4a16_tiled", &w4a16_tiled, "reference W4A16 GEMM, mma m16n8k8");
}
