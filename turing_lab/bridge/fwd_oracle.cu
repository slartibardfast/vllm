// fwd_oracle.cu — bridge-native attention forward vs the fp64 reference.
// Covers d=64 and d=128, MHA and GQA, dense (sq == s, q0 == 0) and
// chunked/bottom-right causal (sq < s, q0 = s - sq), and input magnitude:
// in_scale > 1 pushes RAW dot products past fp16 range (65504), which is
// the regression class found by the engine correctness gate on
// 2026-09-05 (half-saturated S made inf - inf = NaN in the softmax).
#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <cuda_fp16.h>
#include "flash_fwd_sm75.cuh"

template <int D, bool kLd>
__global__ void k_fwd(const __half* q, const __half* k, const __half* v,
                      __half* out, int sq, int s, int q0, int h_q, int h_kv,
                      int causal_flag, float softcap) {
  extern __shared__ __half smem[];
  int bh = blockIdx.x;                 // b * h_q
  int kv_head = (bh % h_q) / (h_q / h_kv);
  long qoff = (long)bh * sq * D;
  long koff = ((long)(bh / h_q) * h_kv + kv_head) * s * D;
  bridge_flash::flash_fwd_one<D, kLd>(q + qoff, k + koff, v + koff,
                                 out + qoff, sq, s, q0, causal_flag != 0,
                                 softcap, smem);
}

template <int D>
int run_case(int b, int h_q, int h_kv, int s, int sq, int q0, float in_scale,
             float softcap = 0.f) {
  size_t nq = (size_t)b * h_q * sq * D;
  size_t nkv = (size_t)b * h_kv * s * D;
  __half *q, *k, *v, *o;
  cudaMallocManaged(&q, nq * 2); cudaMallocManaged(&k, nkv * 2);
  cudaMallocManaged(&v, nkv * 2); cudaMallocManaged(&o, nq * 2);
  srand(9);
  for (size_t i = 0; i < nq; i++) q[i] = __float2half(((rand() % 200) - 100) / 100.0f * in_scale);
  for (size_t i = 0; i < nkv; i++) {
    k[i] = __float2half(((rand() % 200) - 100) / 100.0f * in_scale);
    v[i] = __float2half(((rand() % 200) - 100) / 100.0f * in_scale);
  }
  constexpr int kStride = D + 8;
  constexpr bool kDBuf = (D <= 64);
  size_t smem = (D == 256)
      ? size_t(2) * 32 * kStride * 2  // no sQ; 32-row K+V
      : size_t(3) * 64 * kStride * 2;
  int failures = 0;
  for (int causal = 0; causal <= 1; causal++) {
    cudaMemset(o, 0, nq * 2);
    // kLd mirrors the engine route: decode-shaped calls (one padded q
    // tile) use the pre-tuning loads; larger sq uses ldmatrix
    if (sq <= 64)
      cudaFuncSetAttribute(k_fwd<D, false>,
                           cudaFuncAttributeMaxDynamicSharedMemorySize,
                           (int)smem);
    else
      cudaFuncSetAttribute(k_fwd<D, true>,
                           cudaFuncAttributeMaxDynamicSharedMemorySize,
                           (int)smem);
    if (sq <= 64)
      k_fwd<D, false><<<dim3(b * h_q, sq / 64), 128, smem>>>(q, k, v, o, sq,
          s, q0, h_q, h_kv, causal, softcap);
    else
      k_fwd<D, true><<<dim3(b * h_q, sq / 64), 128, smem>>>(q, k, v, o, sq,
          s, q0, h_q, h_kv, causal, softcap);
    cudaError_t e = cudaDeviceSynchronize();
    if (e != cudaSuccess) { printf("d=%d gqa=%d causal=%d: %s\n", D, h_q / h_kv,
                                   causal, cudaGetErrorString(e)); failures++; continue; }
    double max_err = 0;
    for (int bh = 0; bh < b * h_q; bh++) {
      int kvh = (bh % h_q) / (h_q / h_kv);
      const __half* qb = q + (long)bh * sq * D;
      const __half* kb = k + ((long)(bh / h_q) * h_kv + kvh) * s * D;
      const __half* vb = v + ((long)(bh / h_q) * h_kv + kvh) * s * D;
      for (int m = 0; m < sq; m++) {
        double e_exp[2048], mx = -1e30, denom = 0, out_ref[256];
        const int q_abs = q0 + m;      // bottom-right causal row position
        for (int kk = 0; kk < s; kk++) {
          double dot = 0;
          for (int dd = 0; dd < D; dd++)
            dot += (double)__half2float(qb[(long)m * D + dd]) *
                   (double)__half2float(kb[(long)kk * D + dd]);
          dot /= sqrt((double)D);
          if (softcap > 0) dot = softcap * tanh(dot / softcap);
          if (causal && kk > q_abs) dot = -1e30;
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
          double d = fabs((double)__half2float(o[((long)bh * sq + m) * D + nn]) - out_ref[nn]);
          if (d > max_err) max_err = d;
        }
      }
    }
    bool ok = max_err < 0.05;
    printf("d=%3d gqa=%d s=%4d sq=%3d q0=%4d xs=%4.0f causal=%d: max_err %.5f %s\n", D,
           h_q / h_kv, s, sq, q0, in_scale, causal, max_err, ok ? "PASS" : "FAIL");
    if (!ok) failures++;
  }
  cudaFree(q); cudaFree(k); cudaFree(v); cudaFree(o);
  return failures;
}

int main() {
  int failures = 0;
  failures += run_case<64>(2, 4, 4, 512, 512, 0, 1);     // d=64 MHA dense
  failures += run_case<64>(2, 8, 2, 512, 512, 0, 1);     // d=64 GQA 4:1 dense
  failures += run_case<128>(2, 4, 4, 512, 512, 0, 1);    // d=128 MHA dense
  failures += run_case<128>(1, 8, 1, 512, 512, 0, 1);    // d=128 GQA 8:1 dense
  failures += run_case<64>(1, 4, 4, 384, 128, 256, 1);   // chunked: prefix 256 + 128
  failures += run_case<128>(1, 8, 2, 512, 64, 448, 1);   // chunked d=128 GQA
  failures += run_case<256>(1, 24, 4, 128, 64, 76, 1);   // d=256 decode leg (77 padded)
  failures += run_case<256>(1, 24, 4, 128, 128, 0, 1);   // d=256 prefill leg (77 padded)
  failures += run_case<256>(1, 24, 4, 320, 64, 256, 1);  // d=256 decode leg (301 padded)
  failures += run_case<256>(1, 4, 4, 512, 512, 0, 1);    // d=256 MHA dense
  failures += run_case<256>(2, 8, 2, 512, 512, 0, 1);    // d=256 GQA 4:1 dense
  failures += run_case<256>(1, 8, 2, 512, 64, 448, 1);   // chunked d=256 GQA
  failures += run_case<256>(1, 12, 2, 512, 512, 0, 100); // d=256 magnitude
  failures += run_case<256>(1, 4, 4, 512, 512, 0, 1, 50);   // d=256 softcap 50 (Gemma-style)
  failures += run_case<256>(1, 8, 2, 512, 64, 448, 1, 50);  // chunked d=256 softcap 50
  failures += run_case<128>(1, 4, 4, 512, 512, 0, 1, 50);   // d=128 softcap 50
  failures += run_case<128>(1, 4, 4, 64, 64, 0, 1, 50);     // single tile softcap 50
  failures += run_case<128>(1, 4, 4, 64, 64, 0, 1, 5);      // single tile softcap 5
  failures += run_case<128>(1, 4, 4, 64, 64, 0, 1, 1);      // single tile softcap 1
  failures += run_case<128>(4, 8, 8, 256, 64, 192, 1);   // decode shape
  // magnitude class: raw dots past fp16 range (the engine-gate finding)
  failures += run_case<128>(1, 12, 2, 512, 512, 0, 100); // d=128 GQA 6:1, |S_raw| ~ 1e5
  failures += run_case<64>(1, 12, 2, 512, 512, 0, 100);  // d=64 same
  failures += run_case<128>(1, 12, 2, 256, 64, 192, 100);// decode shape + magnitude
  printf(failures ? "\nSUITE: %d FAILURES\n" : "\nSUITE: ALL PASS\n", failures);
  return failures;
}
