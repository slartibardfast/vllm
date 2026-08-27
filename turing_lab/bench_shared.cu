// Shared-memory path on sm_75: STS, LDSM (ldmatrix), the global->shared->
// register staging chain (Turing has no cp.async, so the chain is
// LDG -> STS -> barrier -> ldmatrix), bank-conflict mapping, and warp
// scaling of the HMMA stream.
//
// usage: bench_shared <sts|sts_conflict|ldsm|staging|warp_scale> [iters]
#include "common.h"

#include <cuda_fp16.h>
#include <cstdio>
#include <cstring>

__global__ void k_sts(float* out, int iters, int conflict) {
  __shared__ float4 s[32 * 64];
  int tid = threadIdx.x;
  float4 v = make_float4(1.f, 2.f, 3.f, 4.f);
  float acc = 0.f;
  for (int i = 0; i < iters; i++) {
    // conflict == 0: stride-1 across the warp; conflict == 1: 34-float
    // stride so 4 lanes hit one bank pair
    int idx = conflict ? ((tid * 34) & 31) + (tid & 32) + (i & 1) * 32 * 32
                       : tid + (i & 1) * 32 * 32;
#pragma unroll
    for (int r = 0; r < 8; r++) s[idx + r * 32] = v;
    __syncthreads();
    float4 r4 = s[(tid + i) % (32 * 64)];
    acc += r4.x + r4.y + r4.z + r4.w;
    __syncthreads();
  }
  if (acc == 12345.678f) out[tid] = acc;
}

// ldmatrix x4: one warp loads four 8x8 fp16 matrices from shared per issue.
__global__ void k_ldsm(float* out, int iters) {
  __shared__ half s[8][128];
  int tid = threadIdx.x;
  for (int i = 0; i < 128; i++) s[tid % 8][i] = __float2half((float)(tid + i));
  __syncthreads();
  // lane L addresses row L%8 of sub-matrix L/8; 16 sub-matrices fit across
  // the 128 columns, so the column base wraps within bounds for any warp id
  uint32_t addr = static_cast<uint32_t>(__cvta_generic_to_shared(
      &s[tid % 8][((tid / 8) % 16) * 8]));
  uint32_t r0, r1, r2, r3;
  float acc = 0.f;
  for (int i = 0; i < iters; i++) {
#pragma unroll
    for (int r = 0; r < 8; r++) {
      asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0,%1,%2,%3}, [%4];\n"
                   : "=r"(r0), "=r"(r1), "=r"(r2), "=r"(r3)
                   : "r"(addr));
      acc += (float)((r0 + r1 + r2 + r3) & 1);
    }
  }
  if (acc == 12345.678f) out[tid] = acc;
}

// The Turing staging chain: LDG.128 from global, STS.128 to shared,
// barrier, then consume through ldmatrix. No cp.async exists on sm_75, so
// the chain pays the register round trip the plan calls out.
__global__ void k_staging(const float4* __restrict__ g, float* out, int iters) {
  __shared__ float4 s[32 * 32];
  int tid = threadIdx.x;
  float acc = 0.f;
  for (int i = 0; i < iters; i++) {
    float4 v = __ldcg(&g[(size_t)blockIdx.x * blockDim.x + tid]);
    s[tid] = v;
    __syncthreads();
#pragma unroll
    for (int r = 0; r < 8; r++) {
      float4 q = s[(tid * 7 + r * 5) % (32 * 32)];
      acc += q.x + q.y + q.z + q.w;
    }
    __syncthreads();
  }
  if (acc == 12345.678f) out[tid] = acc;
}

// HMMA aggregate rate as occupancy scales: grid = sms * factor.
__global__ void k_scale(float* out, int iters) {
  uint32_t a[2], b;
  uint32_t s = 0x3f803f80u ^ (threadIdx.x * 0x9e3779b9u);
  a[0] = s += 0x9e3779b9u;
  a[1] = s += 0x85ebca6bu;
  b = s;
  float d[4] = {};
  for (int i = 0; i < iters; i++) {
#pragma unroll
    for (int r = 0; r < 4; r++) {
      asm volatile("mma.sync.aligned.m16n8k8.row.col.f32.f16.f16.f32 "
                   "{%0,%1,%2,%3}, {%4,%5}, {%6}, {%0,%1,%2,%3};\n"
                   : "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3])
                   : "r"(a[0]), "r"(a[1]), "r"(b));
    }
  }
  float acc = d[0] + d[1] + d[2] + d[3];
  if (acc == 12345.678f) out[threadIdx.x] = acc;
}

int main(int argc, char** argv) {
  const char* which = (argc > 1) ? argv[1] : "sts";
  int iters = iters_arg(argc, argv, 10000);
  int blocks = sm_count() * kBlocksPerSm;
  float* out;
  CUDA_CHECK(cudaMalloc(&out, kThreads * sizeof(float)));
  Timer t;

  if (!strcmp(which, "sts") || !strcmp(which, "sts_conflict")) {
    int conflict = !strcmp(which, "sts_conflict");
    t.start();
    k_sts<<<blocks, kThreads>>>(out, iters, conflict);
    float ms = t.stop();
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());
    const double bytes = (double)blocks * kThreads * 8 * 16.0 * iters * 2;
    printf("name=%s iters=%d ms=%.3f shared_gb_per_s=%.0f\n", which, iters, ms,
           bytes / (ms * 1e-3) / 1e9);
  } else if (!strcmp(which, "ldsm")) {
    t.start();
    k_ldsm<<<blocks, kThreads>>>(out, iters);
    float ms = t.stop();
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());
    const double mats = (double)blocks * kWarpsPerBlock * iters * 8.0 * 4;
    printf("name=ldsm iters=%d ms=%.3f matrices_per_s=%.3g\n", iters, ms, mats / (ms * 1e-3));
  } else if (!strcmp(which, "staging")) {
    const float4* g;
    CUDA_CHECK(cudaMalloc((void**)&g, (size_t)blocks * kThreads * sizeof(float4)));
    CUDA_CHECK(cudaMemset((void*)g, 1, (size_t)blocks * kThreads * sizeof(float4)));
    t.start();
    k_staging<<<blocks, kThreads>>>(g, out, iters);
    float ms = t.stop();
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());
    const double bytes = (double)blocks * kThreads * sizeof(float4) * iters;
    printf("name=staging iters=%d ms=%.3f chain_gb_per_s=%.0f\n", iters, ms,
           bytes / (ms * 1e-3) / 1e9);
    CUDA_CHECK(cudaFree((void*)g));
  } else if (!strcmp(which, "warp_scale")) {
    int sms = sm_count();
    for (int factor = 1; factor <= 16; factor *= 2) {
      int it = iters / 4;
      t.start();
      k_scale<<<sms * factor, kThreads>>>(out, it);
      float ms = t.stop();
      CUDA_CHECK(cudaGetLastError());
      CUDA_CHECK(cudaDeviceSynchronize());
      const double mmas = (double)sms * factor * kWarpsPerBlock * it * 4;
      printf("factor=%2d hmma_tflops=%.2f\n", factor,
             mmas * 2048.0 / (ms * 1e-3) / 1e12);
    }
  } else {
    fprintf(stderr, "unknown benchmark %s\n", which);
    return 2;
  }
  CUDA_CHECK(cudaFree(out));
  return 0;
}
