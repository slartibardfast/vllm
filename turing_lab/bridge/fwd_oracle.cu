// fwd_oracle.cu — the bridge-native attention forward vs the fp64 reference.
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
  const int b = 2, h = 4, s = 512, D = 64;
  size_t n = (size_t)b * h * s * D;
  __half *q, *k, *v, *o;
  cudaMallocManaged(&q, n * 2); cudaMallocManaged(&k, n * 2);
  cudaMallocManaged(&v, n * 2); cudaMallocManaged(&o, n * 2);
  srand(9);
  for (size_t i = 0; i < n; i++) {
    q[i] = __float2half(((rand() % 200) - 100) / 100.0f);
    k[i] = __float2half(((rand() % 200) - 100) / 100.0f);
    v[i] = __float2half(((rand() % 200) - 100) / 100.0f);
  }
  int failures = 0;
  for (int causal = 0; causal <= 1; causal++) {
    cudaMemset(o, 0, n * 2);
    k_fwd<<<dim3(b * h, s / 64), 128, 3 * 64 * 72 * 2>>>(q, k, v, o, s, causal);
    cudaError_t e = cudaDeviceSynchronize();
    if (e != cudaSuccess) {
      printf("causal=%d: %s\n", causal, cudaGetErrorString(e));
      failures++;
      continue;
    }
    double max_err = 0;
    for (int bh = 0; bh < b * h; bh++)
      for (int m = 0; m < s; m++) {
        double e_exp[512], mx = -1e30, denom = 0, out_ref[64];
        for (int kk = 0; kk < s; kk++) {
          double dot = 0;
          for (int dd = 0; dd < D; dd++)
            dot += (double)__half2float(q[((long)bh * s + m) * D + dd]) *
                   (double)__half2float(k[((long)bh * s + kk) * D + dd]);
          dot /= 8.0;
          if (causal && kk > m) dot = -1e30;
          e_exp[kk] = dot;
          if (dot > mx) mx = dot;
        }
        for (int kk = 0; kk < s; kk++) {
          e_exp[kk] = exp(e_exp[kk] - mx);
          denom += e_exp[kk];
        }
        for (int nn = 0; nn < D; nn++) {
          double acc = 0;
          for (int kk = 0; kk < s; kk++)
            acc += e_exp[kk] * (double)__half2float(v[((long)bh * s + kk) * D + nn]);
          out_ref[nn] = acc / denom;
        }
        for (int nn = 0; nn < D; nn++) {
          double got = (double)__half2float(o[((long)bh * s + m) * D + nn]);
          double d = fabs(got - out_ref[nn]);
          if (d > max_err) { max_err = d;
            printf("  worst@(bh=%d m=%d n=%d): got %.4f ref %.4f\n", bh, m, nn, got, out_ref[nn]); }
        }
      }
    printf("causal=%d: max_err %.5f %s\n", causal, max_err,
           max_err < 0.05 ? "PASS" : "FAIL");
    if (max_err >= 0.05) failures++;
  }
  return failures;
}
