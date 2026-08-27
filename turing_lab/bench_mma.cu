// Tensor-core throughput on sm_75: FP16 HMMA (f32 and f16 accumulate),
// INT8 s8 m8n8k16, and the s4 investigation (compile with -DLAB_HAS_S4; if
// ptxas rejects the instruction the probe itself is the answer).
//
// usage: bench_mma <f16f32|f16f16|s8|s4> [iters]
#include "common.h"

#include <cuda_fp16.h>
#include <cstdio>
#include <cstring>

// m16n8k8, fp16 in, fp32 accumulate. This is the widest FP16-with-FP32-acc
// MMA sm_75 has: ptxas rejects the m16n8k16 f32-acc form below sm_80, which
// is the smaller-K fact the plan calls out. d = a*b + d (C == D).
__device__ __forceinline__ void mma_f16_f32(float* d, const uint32_t* a,
                                            const uint32_t* b) {
  asm volatile(
      "mma.sync.aligned.m16n8k8.row.col.f32.f16.f16.f32 "
      "{%0,%1,%2,%3}, {%4,%5}, {%6}, {%0,%1,%2,%3};\n"
      : "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3])
      : "r"(a[0]), "r"(a[1]), "r"(b[0]));
}

// m16n8k8, fp16 in, fp16 accumulate: the shape the incumbent Turing branch
// reaches for when use_fp16_accum is on. The A fragment is 2 registers at
// k8 (marlin splits its k16 tile into two of these).
__device__ __forceinline__ void mma_f16_f16(uint32_t* d, const uint32_t* a,
                                            const uint32_t* b) {
  asm volatile(
      "mma.sync.aligned.m16n8k8.row.col.f16.f16.f16.f16 "
      "{%0,%1}, {%2,%3}, {%4}, {%0,%1};\n"
      : "+r"(d[0]), "+r"(d[1])
      : "r"(a[0]), "r"(a[1]), "r"(b[0]));
}

// m8n8k16, s8 in, s32 accumulate.
__device__ __forceinline__ void mma_s8_s32(int* d, const uint32_t* a,
                                           const uint32_t* b) {
  asm volatile(
      "mma.sync.aligned.m8n8k16.row.col.s32.s8.s8.s32 "
      "{%0,%1}, {%2}, {%3}, {%0,%1};\n"
      : "+r"(d[0]), "+r"(d[1])
      : "r"(a[0]), "r"(b[0]));
}

// m8n8k16, s4 in, s32 accumulate: the unmeasured prerequisite. Guarded so a
// ptxas rejection isolates cleanly to the probe build.
#ifdef LAB_HAS_S4
__device__ __forceinline__ void mma_s4_s32(int* d, const uint32_t* a,
                                           const uint32_t* b) {
  asm volatile(
      "mma.sync.aligned.m8n8k16.row.col.s32.s4.s4.s32 "
      "{%0,%1}, {%2}, {%3}, {%0,%1};\n"
      : "+r"(d[0]), "+r"(d[1])
      : "r"(a[0]), "r"(b[0]));
}
#endif

// 8 independent accumulator sets per warp give the scheduler enough ILP to
// reach the pipe ceiling; the b fragment rotates so the loop cannot be
// constant-folded.
constexpr int kIlp = 8;

__global__ void k_f16_f32(float* out, int iters) {
  uint32_t a[2], b[kIlp];
  uint32_t s = 0x3f803f80u ^ (threadIdx.x * 0x9e3779b9u);
#pragma unroll
  for (int i = 0; i < 2; i++) a[i] = s += 0x9e3779b9u;
#pragma unroll
  for (int r = 0; r < kIlp; r++) b[r] = s += 0x85ebca6bu;
  float d[kIlp][4] = {};
  for (int i = 0; i < iters; i++) {
#pragma unroll
    for (int r = 0; r < kIlp; r++) mma_f16_f32(d[r], a, &b[r]);
  }
  float acc = 0.f;
#pragma unroll
  for (int r = 0; r < kIlp; r++)
#pragma unroll
    for (int e = 0; e < 4; e++) acc += d[r][e];
  if (acc == 12345.678f) out[threadIdx.x] = acc;  // never true; defeats DCE
}

__global__ void k_f16_f16(float* out, int iters) {
  uint32_t a[2], b[kIlp];
  uint32_t s = 0x3c003c00u ^ (threadIdx.x * 0x9e3779b9u);
#pragma unroll
  for (int i = 0; i < 2; i++) a[i] = s += 0x9e3779b9u;
#pragma unroll
  for (int r = 0; r < kIlp; r++) b[r] = s += 0x85ebca6bu;
  uint32_t d[kIlp][2] = {};
  for (int i = 0; i < iters; i++) {
#pragma unroll
    for (int r = 0; r < kIlp; r++) mma_f16_f16(d[r], a, &b[r]);
  }
  uint32_t acc = 0;
#pragma unroll
  for (int r = 0; r < kIlp; r++) acc += d[r][0] + d[r][1];
  if (acc == 0xdeadbeefu) out[threadIdx.x] = 1.f;
}

template <bool S4>
__global__ void k_int(int* out, int iters) {
  uint32_t a[kIlp], b[kIlp];
  uint32_t s = 0x03030303u ^ (threadIdx.x * 0x9e3779b9u);
#pragma unroll
  for (int r = 0; r < kIlp; r++) {
    a[r] = s += 0x0b0b0b0bu;  // small 4-bit-safe lanes
    b[r] = s += 0x07070707u;
  }
  int d[kIlp][2] = {};
  for (int i = 0; i < iters; i++) {
#pragma unroll
    for (int r = 0; r < kIlp; r++) {
      if (S4) {
#ifdef LAB_HAS_S4
        mma_s4_s32(d[r], a, b);  // shapes match; only the width differs
#endif
      } else {
        mma_s8_s32(d[r], a, b);
      }
    }
  }
  long long acc = 0;
#pragma unroll
  for (int r = 0; r < kIlp; r++) acc += d[r][0] + d[r][1];
  if (acc == 0xdeadbeefLL) out[threadIdx.x] = 1;
}

int main(int argc, char** argv) {
  const char* which = (argc > 1) ? argv[1] : "f16f32";
  int iters = iters_arg(argc, argv, 20000);
  int sms = sm_count();
  int blocks = sms * kBlocksPerSm;
  float ms = 0.f;

  Timer t;
  if (!strcmp(which, "f16f32") || !strcmp(which, "f16f16")) {
    float* out;
    CUDA_CHECK(cudaMalloc(&out, kThreads * sizeof(float)));
    if (which[4] == '3') {
      t.start();
      k_f16_f32<<<blocks, kThreads>>>(out, iters);
      ms = t.stop();
      const double mmas = (double)blocks * kWarpsPerBlock * iters * kIlp;
      printf("name=f16f32 mma=m16n8k8 flops_per_mma=2048 mmas=%.0f ms=%.3f "
             "tflops=%.2f\n",
             mmas, ms, mmas * 2048.0 / (ms * 1e-3) / 1e12);
    } else {
      t.start();
      k_f16_f16<<<blocks, kThreads>>>(out, iters);
      ms = t.stop();
      const double mmas = (double)blocks * kWarpsPerBlock * iters * kIlp;
      printf("name=f16f16 mma=m16n8k8 flops_per_mma=2048 mmas=%.0f ms=%.3f "
             "tflops=%.2f\n",
             mmas, ms, mmas * 2048.0 / (ms * 1e-3) / 1e12);
    }
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());
    CUDA_CHECK(cudaFree(out));
  } else if (!strcmp(which, "s8") || !strcmp(which, "s4")) {
    bool s4 = which[1] == '4';
    int* out;
    CUDA_CHECK(cudaMalloc(&out, kThreads * sizeof(int)));
    t.start();
    if (s4)
      k_int<true><<<blocks, kThreads>>>(out, iters);
    else
      k_int<false><<<blocks, kThreads>>>(out, iters);
    ms = t.stop();
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());
    const double mmas = (double)blocks * kWarpsPerBlock * iters * kIlp;
    printf("name=%s mma=m8n8k16 ops_per_mma=2048 mmas=%.0f ms=%.3f tops=%.2f\n",
           which, mmas, ms, mmas * 2048.0 / (ms * 1e-3) / 1e12);
    CUDA_CHECK(cudaFree(out));
  } else {
    fprintf(stderr, "unknown benchmark %s\n", which);
    return 2;
  }
  return 0;
}
