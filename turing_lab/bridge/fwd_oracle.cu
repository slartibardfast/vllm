// fwd_oracle.cu — bridge-native attention forward vs the fp64 reference.
// Covers d=64 and d=128, MHA and GQA.
#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <cuda_fp16.h>
#include "flash_fwd_sm75.cuh"

template <int D>
__global__ void k_fwd(const __half* q, const __half* k, const __half* v,
                      __half* out, int s, int h_q, int h_kv, int causal_flag) {
  extern __shared__ __half smem[];
  int bh = blockIdx.x;                 // b * h_q
  int kv_head = (bh % h_q) / (h_q / h_kv);
  long qoff = (long)bh * s * D;
  long koff = ((long)(bh / h_q) * h_kv + kv_head) * s * D;
  bridge_flash::flash_fwd_one<D>(q + qoff, k + koff, v + koff, out + qoff, s,
                                 causal_flag != 0, smem);
}

template <int D>
int run_case(int b, int h_q, int h_kv, int s) {
  size_t nq = (size_t)b * h_q * s * D;
  size_t nkv = (size_t)b * h_kv * s * D;
  __half *q, *k, *v, *o;
  cudaMallocManaged(&q, nq * 2); cudaMallocManaged(&k, nkv * 2);
  cudaMallocManaged(&v, nkv * 2); cudaMallocManaged(&o, nq * 2);
  srand(9);
  for (size_t i = 0; i < nq; i++) q[i] = __float2half(((rand() % 200) - 100) / 100.0f);
  for (size_t i = 0; i < nkv; i++) {
    k[i] = __float2half(((rand() % 200) - 100) / 100.0f);
    v[i] = __float2half(((rand() % 200) - 100) / 100.0f);
  }
  constexpr int kStride = D + 8;
  constexpr bool kDBuf = (D <= 64);
  size_t smem = (size_t)(kDBuf ? 5 : 3) * 64 * kStride * 2;
  cudaFuncSetAttribute(k_fwd<D>, cudaFuncAttributeMaxDynamicSharedMemorySize,
                       (int)smem);
  int failures = 0;
  for (int causal = 0; causal <= 1; causal++) {
    cudaMemset(o, 0, nq * 2);
    k_fwd<D><<<dim3(b * h_q, s / 64), 128, smem>>>(q, k, v, o, s, h_q, h_kv,
                                                   causal);
    cudaError_t e = cudaDeviceSynchronize();
    if (e != cudaSuccess) { printf("d=%d gqa=%d causal=%d: %s\n", D, h_q / h_kv,
                                   causal, cudaGetErrorString(e)); failures++; continue; }
    double max_err = 0;
    for (int bh = 0; bh < b * h_q; bh++) {
      int kvh = (bh % h_q) / (h_q / h_kv);
      const __half* qb = q + (long)bh * s * D;
      const __half* kb = k + ((long)(bh / h_q) * h_kv + kvh) * s * D;
      const __half* vb = v + ((long)(bh / h_q) * h_kv + kvh) * s * D;
      for (int m = 0; m < s; m++) {
        double e_exp[512], mx = -1e30, denom = 0, out_ref[128];
        for (int kk = 0; kk < s; kk++) {
          double dot = 0;
          for (int dd = 0; dd < D; dd++)
            dot += (double)__half2float(qb[(long)m * D + dd]) *
                   (double)__half2float(kb[(long)kk * D + dd]);
          dot /= sqrt((double)D);
          if (causal && kk > m) dot = -1e30;
          e_exp[kk] = dot;
          if (dot > mx) mx = dot;
        }
        for (int kk = 0; kk < s; kk++) { e_exp[kk] = exp(e_exp[kk] - mx); denom += e_exp[kk]; }
        for (int nn = 0; nn < D; nn++) {
          double acc = 0;
          for (int kk = 0; kk < s; kk++)
            acc += e_exp[kk] * (double)__half2float(vb[(long)kk * D + nn]);
          out_ref[nn] = acc / denom;
        }
        for (int nn = 0; nn < D; nn++) {
          double d = fabs((double)__half2float(o[((long)bh * s + m) * D + nn]) - out_ref[nn]);
          if (d > max_err) max_err = d;
        }
      }
    }
    bool ok = max_err < 0.05;
    printf("d=%3d gqa=%d causal=%d: max_err %.5f %s\n", D, h_q / h_kv, causal,
           max_err, ok ? "PASS" : "FAIL");
    if (!ok) failures++;
  }
  cudaFree(q); cudaFree(k); cudaFree(v); cudaFree(o);
  return failures;
}

int main() {
  int failures = 0;
  failures += run_case<64>(2, 4, 4, 512);    // d=64 MHA
  failures += run_case<64>(2, 8, 2, 512);    // d=64 GQA 4:1
  failures += run_case<128>(2, 4, 4, 512);   // d=128 MHA
  failures += run_case<128>(1, 8, 1, 512);   // d=128 GQA 8:1
  printf(failures ? "\nSUITE: %d FAILURES\n" : "\nSUITE: ALL PASS\n", failures);
  return failures;
}
