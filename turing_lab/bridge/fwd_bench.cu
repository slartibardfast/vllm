// fwd_bench.cu — bridge-native attention forward: correctness + timing.
#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <cuda_fp16.h>
#include "flash_fwd_sm75.cuh"

__global__ void k_fwd(const __half* q, const __half* k, const __half* v,
                      __half* out, int s, int causal_flag) {
  extern __shared__ __half smem[];
  long off = (long)blockIdx.x * s * 64;
  bridge_flash::flash_fwd_one(q + off, k + off, v + off, out + off, s,
                              causal_flag != 0, smem);
}

int main() {
  const int b = 4, h = 16;
  for (int s : {512, 2048, 8192}) {
    size_t n = (size_t)b * h * s * 64;
    __half *q, *k, *v, *o;
    cudaMallocManaged(&q, n * 2); cudaMallocManaged(&k, n * 2);
    cudaMallocManaged(&v, n * 2); cudaMallocManaged(&o, n * 2);
    srand(3);
    for (size_t i = 0; i < n; i++) {
      q[i] = __float2half(((rand() % 200) - 100) / 100.0f);
      k[i] = __float2half(((rand() % 200) - 100) / 100.0f);
      v[i] = __float2half(((rand() % 200) - 100) / 100.0f);
    }
    dim3 grid(b * h, s / 64);
    // warmup + timing (causal=1, the serving case)
    for (int r = 0; r < 3; r++) k_fwd<<<grid, 128, 5*64*72*2>>>(q, k, v, o, s, 1);
    cudaDeviceSynchronize();
    cudaEvent_t a, e; cudaEventCreate(&a); cudaEventCreate(&e);
    float best = 1e30f;
    for (int r = 0; r < 10; r++) {
      cudaEventRecord(a);
      k_fwd<<<grid, 128, 5*64*72*2>>>(q, k, v, o, s, 1);
      cudaEventRecord(e); cudaEventSynchronize(e);
      float ms; cudaEventElapsedTime(&ms, a, e);
      if (ms < best) best = ms;
    }
    // attention FLOPs (causal): 4*b*h*s^2*d (QK^T + PV, ×2 for FLOPs, ÷2 causal)
    double flop = 2.0 * b * h * (double)s * s * 64 * 2 * 0.5;
    printf("s=%5d causal: %.3f ms  %.1f TFLOP/s\n", s, best, flop / (best * 1e-3) / 1e12);
    cudaFree(q); cudaFree(k); cudaFree(v); cudaFree(o);
  }
  return 0;
}
