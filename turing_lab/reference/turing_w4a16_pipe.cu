// turing_w4a16_pipe.cu — pipelined double-buffered Turing W4A16 kernel.
//
// Fragment walk: the validated reference path — weights dequantized
// ((q - 8) * scale, folded at staging) into fp16 shared, m16n8k8 MMA with
// FP32 accumulate. Optimization over the reference: BK 32 chunks in two
// shared buffers, with the next chunk's global loads issued into registers
// before the current chunk's MMA stream so the global latency overlaps the
// tensor-core work; about 15 KB of shared per chunk pair keeps three
// resident blocks per SM (the characterization's occupancy rule).
//
// Contract: qweight (N, K/8) uint32, low nibble first along k; scales
// (K/G, N) fp16; symmetric zero point 8; G >= 32. Requires K % 32 == 0 and
// N % 64 == 0.

#include <cuda_fp16.h>
#include <cstdint>
#ifndef STANDALONE_MAIN
#include <torch/extension.h>
#include <optional>
#include <c10/cuda/CUDAException.h>
#endif

namespace {

constexpr int BM = 64, BN = 64, BK = 32, THREADS = 128;

__global__ void __launch_bounds__(THREADS)
k_pipe(const __half* __restrict__ A, const uint32_t* __restrict__ Q,
       const __half* __restrict__ S, __half* __restrict__ Out, int M, int N,
       int K, int k_words, int G) {
  __shared__ __half sA[2][BM][BK + 8];
  __shared__ __half sBT[2][BN][BK + 8];

  int n0 = blockIdx.x * BN;
  int m_base = blockIdx.y * BM;
  int tid = threadIdx.x;
  int warp = tid >> 5, lane = tid & 31;
  int r0 = warp * 16;
  int gg = lane >> 2, cc = lane & 3;

  float acc[8][4] = {};
  const __half2 zero2 = __float2half2_rn(0.f);

  // per-thread staging registers: 16 A halves (8 u32), 16 weight nibbles
  // (2 u32), one scale
  uint32_t rA[8], rW0, rW1;
  __half rS;

  auto load_regs = [&](int k0) {
    int r = tid / 2, seg = tid % 2;
    int gm = m_base + r, gk = k0 + seg * 16;
    if (gm < M && gk + 15 < K) {
      const uint32_t* src = reinterpret_cast<const uint32_t*>(&A[(long)gm * K + gk]);
      rA[0] = src[0]; rA[1] = src[1]; rA[2] = src[2]; rA[3] = src[3];
      rA[4] = src[4]; rA[5] = src[5]; rA[6] = src[6]; rA[7] = src[7];
    } else {
      rA[0] = rA[1] = rA[2] = rA[3] = 0u;
      rA[4] = rA[5] = rA[6] = rA[7] = 0u;
    }
    int nn = tid / 2, wpair = tid % 2;
    int gn = n0 + nn;
    if (gn < N && k0 + wpair * 16 < K) {
      rW0 = Q[(long)gn * k_words + (k0 >> 3) + wpair * 2];
      rW1 = Q[(long)gn * k_words + (k0 >> 3) + wpair * 2 + 1];
    } else {
      rW0 = rW1 = 0u;
    }
    int g = k0 / G;
    rS = S[g * N + n0 + nn];
  };

  auto store_shared = [&](int buf, int k0) {
    int r = tid / 2, seg = tid % 2;
#pragma unroll
    for (int i = 0; i < 8; i++) {
      *reinterpret_cast<uint32_t*>(&sA[buf][r][seg * 16 + i * 2]) = rA[i];
    }
    int nn = tid / 2, wpair = tid % 2;
    float sc = __half2float(rS);
    __half* dst = &sBT[buf][nn][wpair * 16];
#pragma unroll
    for (int w = 0; w < 2; w++) {
      uint32_t word = (w == 0) ? rW0 : rW1;
#pragma unroll
      for (int j = 0; j < 8; j++) {
        int qv = (int)((word >> (4 * j)) & 0xFu);
        dst[w * 8 + j] = __float2half((float)(qv - 8) * sc);
      }
    }
  };

  auto compute = [&](int buf) {
    for (int kk = 0; kk < BK; kk += 8) {
#pragma unroll
      for (int nt = 0; nt < 8; nt++) {
        __half2 a0 = *reinterpret_cast<const __half2*>(&sA[buf][r0 + gg][kk + cc * 2]);
        __half2 a1 = *reinterpret_cast<const __half2*>(&sA[buf][r0 + gg + 8][kk + cc * 2]);
        __half2 b = *reinterpret_cast<const __half2*>(&sBT[buf][nt * 8 + gg][kk + cc * 2]);
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

  int n_chunks = (K + BK - 1) / BK;
  load_regs(0);
  for (int c = 0; c < n_chunks; c++) {
    store_shared(c & 1, c * BK);
    __syncthreads();
    if (c + 1 < n_chunks) load_regs((c + 1) * BK);
    compute(c & 1);
    __syncthreads();
  }

#pragma unroll
  for (int nt = 0; nt < 8; nt++) {
#pragma unroll
    for (int half = 0; half < 2; half++) {
      int r = m_base + r0 + gg + half * 8;
      int c = n0 + nt * 8 + cc * 2;
      if (r < M && c + 1 < N) {
        Out[(long)r * N + c] = __float2half(acc[nt][half * 2]);
        Out[(long)r * N + c + 1] = __float2half(acc[nt][half * 2 + 1]);
      }
    }
  }
}

}  // namespace

#ifndef STANDALONE_MAIN
torch::Tensor w4a16_pipe(torch::Tensor A, torch::Tensor Q, torch::Tensor S,
                         int64_t G) {
  int M = A.size(0), K = A.size(1), N = Q.size(0);
  auto Out = torch::empty({M, N}, A.options());
  int k_words = K / 8;
  dim3 grid(N / BN, (M + BM - 1) / BM);
  k_pipe<<<grid, THREADS>>>(
      reinterpret_cast<const __half*>(A.data_ptr<at::Half>()),
      reinterpret_cast<const uint32_t*>(Q.data_ptr<int32_t>()),
      reinterpret_cast<const __half*>(S.data_ptr<at::Half>()),
      reinterpret_cast<__half*>(Out.data_ptr<at::Half>()), M, N, K, k_words,
      (int)G);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return Out;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("w4a16_pipe", &w4a16_pipe, "pipelined Turing W4A16 GEMM (symmetric)");
}
#endif

#ifdef STANDALONE_MAIN
#include <cstdio>
int main() {
  int M = 1, N = 128, K = 32, G = 32, k_words = K / 8;
  __half *A, *S, *Out; uint32_t *Q;
  cudaMalloc(&A, (size_t)M*K*2); cudaMalloc(&Q, (size_t)N*k_words*4);
  cudaMalloc(&S, (size_t)(K/G)*N*2); cudaMalloc(&Out, (size_t)M*N*2);
  __half *hA = (__half*)malloc(M*K*2);
  hA[0] = __float2half(1.0f);
  for (int i = 1; i < M*K; i++) hA[i] = __float2half(0.0f);
  cudaMemcpy(A, hA, M*K*2, cudaMemcpyHostToDevice);
  uint32_t *hQ = (uint32_t*)calloc(N*k_words, 4);
  for (int n = 0; n < N; n++) hQ[n*k_words] = ((n % 16) & 0xF);
  cudaMemcpy(Q, hQ, N*k_words*4, cudaMemcpyHostToDevice);
  __half *hS = (__half*)malloc((size_t)(K/G)*N*2);
  for (int i = 0; i < (K/G)*N; i++) hS[i] = __float2half(1.0f);
  cudaMemcpy(S, hS, (size_t)(K/G)*N*2, cudaMemcpyHostToDevice);
  dim3 grid(N/BN, (M+BM-1)/BM);
  k_pipe<<<grid, THREADS>>>(A, Q, S, Out, M, N, K, k_words, G);
  cudaError_t e = cudaDeviceSynchronize();
  printf("run: %s\n", cudaGetErrorString(e));
  __half *hOut = (__half*)malloc(M*N*2);
  cudaMemcpy(hOut, Out, M*N*2, cudaMemcpyDeviceToHost);
  for (int c = 0; c < 4; c++) printf("out[0,%d] = %f\n", c, __half2float(hOut[c]));
}
#endif
