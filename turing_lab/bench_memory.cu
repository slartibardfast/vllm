// Global-read bandwidth (LDG.128): the weight-streaming side of the Marlin
// regime. Each thread strides through a float4 buffer with independent
// accumulators so loads stay in flight.
//
// usage: bench_memory [mib] [iters]
#include "common.h"

#include <cstdio>

__global__ void k_read(const float4* __restrict__ p, size_t n4, float* out) {
  float4 s = {0.f, 0.f, 0.f, 0.f};
  size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
  size_t stride = (size_t)gridDim.x * blockDim.x;
  for (; i < n4; i += stride) {
    float4 v = __ldcg(&p[i]);
    s.x += v.x; s.y += v.y; s.z += v.z; s.w += v.w;
  }
  if (s.x + s.y + s.z + s.w == 12345.678f) out[threadIdx.x] = s.x;
}

int main(int argc, char** argv) {
  size_t mib = (argc > 1) ? (size_t)atoi(argv[1]) : 256;
  int iters = (argc > 2) ? atoi(argv[2]) : 50;
  size_t n4 = mib * 1024 * 1024 / sizeof(float4);
  int blocks = sm_count() * 16;
  float4* p;
  CUDA_CHECK(cudaMalloc(&p, n4 * sizeof(float4)));
  CUDA_CHECK(cudaMemset(p, 1, n4 * sizeof(float4)));
  float* out;
  CUDA_CHECK(cudaMalloc(&out, kThreads * sizeof(float)));

  // one warm-up pass touches the pages
  k_read<<<blocks, kThreads>>>(p, n4, out);
  CUDA_CHECK(cudaGetLastError());
  CUDA_CHECK(cudaDeviceSynchronize());

  Timer t;
  t.start();
  for (int i = 0; i < iters; i++) k_read<<<blocks, kThreads>>>(p, n4, out);
  float ms = t.stop();
  CUDA_CHECK(cudaGetLastError());
  CUDA_CHECK(cudaDeviceSynchronize());
  const double bytes = (double)n4 * sizeof(float4) * iters;
  printf("mib=%zu iters=%d ms=%.3f gb_per_s=%.1f\n", mib, iters, ms,
         bytes / (ms * 1e-3) / 1e9);
  CUDA_CHECK(cudaFree(p));
  CUDA_CHECK(cudaFree(out));
  return 0;
}
