// Execution-resource contention on sm_75: one HMMA stream interleaved with a
// varying number of same-warp ALU ops (HFMA2, FFMA, LOP3). The hypothesis
// under test: FP16 arithmetic shares an execution resource with HMMA on
// Turing, so apparently cheap dequant work steals tensor-core throughput.
//
// usage: bench_contention <hfma2|ffma|lop3> <alu_per_iter> [iters]
#include "common.h"

#include <cuda_fp16.h>
#include <cstdio>
#include <cstring>

__device__ __forceinline__ void mma_f16_f32(float* d, const uint32_t* a,
                                            const uint32_t* b) {
  asm volatile(
      "mma.sync.aligned.m16n8k8.row.col.f32.f16.f16.f32 "
      "{%0,%1,%2,%3}, {%4,%5}, {%6}, {%0,%1,%2,%3};\n"
      : "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3])
      : "r"(a[0]), "r"(a[1]), "r"(b[0]));
}

template <int K>
__global__ void k_hmma_alu(float* out, int iters, int kind) {
  uint32_t a[2], b[1];
  uint32_t s = 0x3f803f80u ^ (threadIdx.x * 0x9e3779b9u);
#pragma unroll
  for (int i = 0; i < 2; i++) a[i] = s += 0x9e3779b9u;
  b[0] = s += 0x85ebca6bu;
  float d[4] = {};
  uint32_t x = s, y = s + 0x2545f491u;
  float f = 1.0f, g = 0.5f;
  for (int i = 0; i < iters; i++) {
#pragma unroll
    for (int r = 0; r < 4; r++) mma_f16_f32(d, a, b);
#pragma unroll
    for (int k = 0; k < K; k++) {
      if (kind == 0) {  // HFMA2: fp16 packed arithmetic, the dequant-native op
        asm volatile("fma.rn.f16x2 %0, %0, %1, %2;\n" : "+r"(x) : "r"(y), "r"(b[0]));
      } else if (kind == 1) {  // FFMA: fp32 pipeline
        asm volatile("fma.rn.f32 %0, %0, %1, %2;\n" : "+f"(f) : "f"(g), "f"(d[0]));
      } else {  // LOP3: the logic op Marlin's dequant path is built on
        asm volatile("lop3.b32 %0, %0, %1, %2, 0xea;\n" : "+r"(x) : "r"(y), "r"(b[1]));
      }
    }
  }
  float acc = d[0] + d[1] + d[2] + d[3] + f + (float)(x & 1);
  if (acc == 12345.678f) out[threadIdx.x] = acc;
}

int main(int argc, char** argv) {
  int kind = (argc > 1) ? (!strcmp(argv[1], "ffma") ? 1 : (!strcmp(argv[1], "lop3") ? 2 : 0)) : 0;
  const char* names[3] = {"hfma2", "ffma", "lop3"};
  int k = (argc > 2) ? atoi(argv[2]) : 0;
  int iters = (argc > 3) ? atoi(argv[3]) : 10000;
  int blocks = sm_count() * kBlocksPerSm;
  float* out;
  CUDA_CHECK(cudaMalloc(&out, kThreads * sizeof(float)));

  Timer t;
  t.start();
  switch (k) {
    case 0: k_hmma_alu<0><<<blocks, kThreads>>>(out, iters, kind); break;
    case 1: k_hmma_alu<1><<<blocks, kThreads>>>(out, iters, kind); break;
    case 2: k_hmma_alu<2><<<blocks, kThreads>>>(out, iters, kind); break;
    case 4: k_hmma_alu<4><<<blocks, kThreads>>>(out, iters, kind); break;
    case 8: k_hmma_alu<8><<<blocks, kThreads>>>(out, iters, kind); break;
    case 16: k_hmma_alu<16><<<blocks, kThreads>>>(out, iters, kind); break;
    default: fprintf(stderr, "k must be one of 0 1 2 4 8 16\n"); return 2;
  }
  float ms = t.stop();
  CUDA_CHECK(cudaGetLastError());
  CUDA_CHECK(cudaDeviceSynchronize());
  const double mmas = (double)blocks * kWarpsPerBlock * iters * 4;
  printf("kind=%s alu_per_iter=%d mmas=%.0f ms=%.3f hmma_tflops=%.2f\n", names[kind],
         k, mmas, ms, mmas * 2048.0 / (ms * 1e-3) / 1e12);
  CUDA_CHECK(cudaFree(out));
  return 0;
}
