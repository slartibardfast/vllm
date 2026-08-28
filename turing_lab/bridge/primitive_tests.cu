// primitive_tests.cu — the cu_sm80_on_sm75 conformance suite.
//
// Design: differential testing. Every bridged primitive is compared
// bit-exactly against an independent implementation of the documented
// sm_80 semantics on exactly-representable test values, so any layout
// or ordering mistake produces a large diff, never a tolerance pass.
//
// Red lines (build fails on regression):
// - staged-copy bridge sustains >= 90% of the plain-copy bandwidth on
//   identical data volumes;
// - all oracles are exact.

#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <cstring>
#include <cstdint>
#include <cuda_fp16.h>
#include "sm80_on_sm75.cuh"
#include <device_launch_parameters.h>

#define CHECK_CUDA(call)                                                     \
  do {                                                                       \
    cudaError_t e_ = (call);                                                 \
    if (e_ != cudaSuccess) {                                                 \
      printf("CUDA error %s at %d\n", cudaGetErrorString(e_), __LINE__);     \
      exit(1);                                                               \
    }                                                                        \
  } while (0)

static int g_failures = 0;
#define EXPECT(cond, name)                                                   \
  do {                                                                       \
    if (cond) { printf("PASS %s\n", name); }                                 \
    else { printf("FAIL %s\n", name); g_failures++; }                        \
  } while (0)

static unsigned short f16_bits(float f) {
  return __half_as_ushort(__float2half(f));
}

// ------------------------------------------------------------ MMA k16 ---
// 16x16 A times 16x8 B, fragments per PTX ISA 9.7.15.5.14: lane l has
// g = l>>2, t = l&3;
//   A: a0={A[g][2t],A[g][2t+1]} a1={A[g+8][2t],A[g+8][2t+1]}
//      a2={A[g][2t+8],A[g][2t+9]} a3={A[g+8][2t+8],A[g+8][2t+9]}
//   B: b0={B[2t][c],B[2t+1][c]} b1={B[2t+8][c],B[2t+9][c]}, c = g
//   D: d0={D[g][2t],D[g][2t+1]} d1={D[g][2t+2],D[g][2t+3]}
//      d2={D[g+8][2t],D[g+8][2t+1]} d3={D[g+8][2t+2],D[g+8][2t+3]}

__global__ void k_mma_adapter(const uint32_t* fragA, const uint32_t* fragB,
                              float* out) {
  int lane = threadIdx.x & 31;
  const uint32_t* a = fragA + lane * 4;
  const uint32_t* b = fragB + lane * 2;
  float d[4] = {0.f, 0.f, 0.f, 0.f};
  bridge::mma_m16n8k16_f32(d, a, b);
  out[lane * 4 + 0] = d[0];
  out[lane * 4 + 1] = d[1];
  out[lane * 4 + 2] = d[2];
  out[lane * 4 + 3] = d[3];
}

// naive reference over the composed tiles (fp16 inputs, fp32 math)
__global__ void k_mma_naive(const __half* A, const __half* B, float* D) {
  for (int m = threadIdx.x; m < 16; m += blockDim.x)
    for (int n = 0; n < 8; n++) {
      float acc = 0.f;
      for (int k = 0; k < 16; k++)
        acc += __half2float(A[m * 16 + k]) * __half2float(B[k * 8 + n]);
      D[m * 8 + n] = acc;
    }
}

void test_mma_adapter() {
  __half hA[256], hB[128];  // A 16x16, B 16x8 (k-major)
  srand(77);
  auto rnd = []() { return float((rand() % 5) - 2); };  // exact ints
  for (auto& v : hA) v = __float2half(rnd());
  for (auto& v : hB) v = __float2half(rnd());

  // pack the fragments exactly as the documented layout loads them
  uint32_t fragA[128], fragB[64];
  for (int lane = 0; lane < 32; lane++) {
    int g = lane >> 2, t = lane & 3;
    fragA[lane * 4 + 0] =
        (uint32_t)f16_bits(hA[g * 16 + 2 * t]) |
        ((uint32_t)f16_bits(hA[g * 16 + 2 * t + 1]) << 16);
    fragA[lane * 4 + 1] =
        (uint32_t)f16_bits(hA[(g + 8) * 16 + 2 * t]) |
        ((uint32_t)f16_bits(hA[(g + 8) * 16 + 2 * t + 1]) << 16);
    fragA[lane * 4 + 2] =
        (uint32_t)f16_bits(hA[g * 16 + 2 * t + 8]) |
        ((uint32_t)f16_bits(hA[g * 16 + 2 * t + 9]) << 16);
    fragA[lane * 4 + 3] =
        (uint32_t)f16_bits(hA[(g + 8) * 16 + 2 * t + 8]) |
        ((uint32_t)f16_bits(hA[(g + 8) * 16 + 2 * t + 9]) << 16);
    fragB[lane * 2 + 0] = (uint32_t)f16_bits(hB[(2 * t) * 8 + g]) |
                          ((uint32_t)f16_bits(hB[(2 * t + 1) * 8 + g]) << 16);
    fragB[lane * 2 + 1] = (uint32_t)f16_bits(hB[(2 * t + 8) * 8 + g]) |
                          ((uint32_t)f16_bits(hB[(2 * t + 9) * 8 + g]) << 16);
  }

  __half *dA, *dB;
  float *dD;
  uint32_t *dFA, *dFB;
  CHECK_CUDA(cudaMalloc(&dA, sizeof(hA)));
  CHECK_CUDA(cudaMalloc(&dB, sizeof(hB)));
  CHECK_CUDA(cudaMalloc(&dD, 128 * sizeof(float)));
  CHECK_CUDA(cudaMalloc(&dFA, sizeof(fragA)));
  CHECK_CUDA(cudaMalloc(&dFB, sizeof(fragB)));
  CHECK_CUDA(cudaMemcpy(dA, hA, sizeof(hA), cudaMemcpyHostToDevice));
  CHECK_CUDA(cudaMemcpy(dB, hB, sizeof(hB), cudaMemcpyHostToDevice));
  CHECK_CUDA(cudaMemcpy(dFA, fragA, sizeof(fragA), cudaMemcpyHostToDevice));
  CHECK_CUDA(cudaMemcpy(dFB, fragB, sizeof(fragB), cudaMemcpyHostToDevice));

  // reference: naive over the composed tiles
  float* dRef;
  CHECK_CUDA(cudaMalloc(&dRef, 128 * sizeof(float)));
  k_mma_naive<<<1, 64>>>(dA, dB, dRef);
  CHECK_CUDA(cudaDeviceSynchronize());

  // bridge adapter, outputs written back through the documented D layout
  __global__ void k_mma_adapter(const uint32_t*, const uint32_t*, float*);
  float hD[128];
  {
    // launch adapter and scatter per-lane outputs into the tile
    float* dOut;
    CHECK_CUDA(cudaMalloc(&dOut, 128 * sizeof(float)));
    k_mma_adapter<<<1, 32>>>(dFA, dFB, dOut);
    CHECK_CUDA(cudaDeviceSynchronize());
    float hOut[128];
    CHECK_CUDA(cudaMemcpy(hOut, dOut, sizeof(hOut), cudaMemcpyDeviceToHost));
    for (int lane = 0; lane < 32; lane++) {
      int g = lane >> 2, t = lane & 3;
      hD[g * 8 + 2 * t] = hOut[lane * 4 + 0];
      hD[g * 8 + 2 * t + 1] = hOut[lane * 4 + 1];
      hD[(g + 8) * 8 + 2 * t] = hOut[lane * 4 + 2];
      hD[(g + 8) * 8 + 2 * t + 1] = hOut[lane * 4 + 3];
    }
    float hRef[128];
    CHECK_CUDA(cudaMemcpy(hRef, dRef, sizeof(hRef), cudaMemcpyDeviceToHost));
    bool exact = true;
    for (int i = 0; i < 128; i++) {
      unsigned ub = *reinterpret_cast<unsigned*>(&hD[i]);
      unsigned ur = *reinterpret_cast<unsigned*>(&hRef[i]);
      if (ub != ur && !(hD[i] == hRef[i])) exact = false;
      if (hD[i] != hRef[i]) exact = exact && (fabsf(hD[i] - hRef[i]) < 1e-4f);
    }
    EXPECT(exact, "mma m16n8k16 adapter bit-matches naive reference");
    cudaFree(dOut);
  }
  cudaFree(dA); cudaFree(dB); cudaFree(dD); cudaFree(dFA); cudaFree(dFB);
  cudaFree(dRef);
}

// ---------------------------------------------------- staged copy red line
constexpr int kThreads = 256;

__global__ void k_staged(const uint32_t* __restrict__ g,
                         uint32_t* __restrict__ out, long block_v4) {
  __shared__ uint32_t buf[kThreads * 4];
  long base = (long)blockIdx.x * block_v4;
  for (long i = threadIdx.x; i < block_v4; i += blockDim.x) {
    uint32_t r0, r1, r2, r3;
    bridge::ldg_cs_v4(r0, r1, r2, r3, g + (base + i) * 4);
    bridge::commit_group();
    bridge::wait_group<0>();
    bridge::sts_v4(&buf[threadIdx.x * 4], r0, r1, r2, r3);
    __syncthreads();
    uint32_t o0, o1, o2, o3;
    bridge::lds_v4(o0, o1, o2, o3, &buf[threadIdx.x * 4]);
    long dst = (base + i) * 4;
    out[dst] = o0; out[dst + 1] = o1; out[dst + 2] = o2; out[dst + 3] = o3;
    __syncthreads();
  }
}

__global__ void k_plain(const uint32_t* __restrict__ g,
                        uint32_t* __restrict__ out, long block_v4) {
  long base = (long)blockIdx.x * block_v4;
  for (long i = threadIdx.x; i < block_v4; i += blockDim.x) {
    uint32_t r0, r1, r2, r3;
    asm volatile("ld.global.cs.v4.u32 {%0,%1,%2,%3}, [%4];\n"
                 : "=r"(r0), "=r"(r1), "=r"(r2), "=r"(r3)
                 : "l"(g + (base + i) * 4));
    long dst = (base + i) * 4;
    out[dst] = r0; out[dst + 1] = r1; out[dst + 2] = r2; out[dst + 3] = r3;
  }
}

void test_staged_redline() {
  const long words = 64L * 1024 * 1024;  // 256 MiB
  uint32_t *src, *dst;
  CHECK_CUDA(cudaMalloc(&src, words * 4));
  CHECK_CUDA(cudaMalloc(&dst, words * 4));
  CHECK_CUDA(cudaMemset(src, 0x5A, words * 4));

  long v4_total = words / 4;
  int blocks = (int)(v4_total / (kThreads * 4));
  long block_v4 = v4_total / blocks;

  // byte-exactness of the staged path
  k_staged<<<blocks, kThreads>>>(src, dst, block_v4);
  CHECK_CUDA(cudaDeviceSynchronize());
  int mismatches = 0;
  uint32_t* h = (uint32_t*)malloc(1 << 20);
  for (int chunk = 0; chunk < 4 && mismatches == 0; chunk++) {
    long off = (long)chunk * (words / 4 / 4);
    CHECK_CUDA(cudaMemcpy(h, dst + off * 4, 1 << 20, cudaMemcpyDeviceToHost));
    for (long i = 0; i < (1L << 18); i++)
      if (h[i] != 0x5A5A5A5Au) mismatches++;
  }
  free(h);
  EXPECT(mismatches == 0, "staged copy byte-exact");

  // bandwidth: staged vs plain, interleaved, best-of
  float best_staged = 1e30f, best_plain = 1e30f;
  cudaEvent_t ea, eb;
  CHECK_CUDA(cudaEventCreate(&ea));
  CHECK_CUDA(cudaEventCreate(&eb));
  for (int r = 0; r < 10; r++) {
    CHECK_CUDA(cudaEventRecord(ea));
    k_staged<<<blocks, kThreads>>>(src, dst, block_v4);
    CHECK_CUDA(cudaEventRecord(eb));
    CHECK_CUDA(cudaEventSynchronize(eb));
    float ms;
    CHECK_CUDA(cudaEventElapsedTime(&ms, ea, eb));
    if (ms < best_staged) best_staged = ms;

    CHECK_CUDA(cudaEventRecord(ea));
    k_plain<<<blocks * 4, kThreads>>>(src, dst, block_v4 / 4);
    CHECK_CUDA(cudaEventRecord(eb));
    CHECK_CUDA(cudaEventSynchronize(eb));
    CHECK_CUDA(cudaEventElapsedTime(&ms, ea, eb));
    if (ms < best_plain) best_plain = ms;
  }
  double bytes = (double)words * 4;
  double gbps_staged = bytes / (best_staged * 1e-3) / 1e9;
  double gbps_plain = bytes / (best_plain * 1e-3) / 1e9;
  double ratio = gbps_staged / gbps_plain;
  printf("staged %.1f GB/s | plain %.1f GB/s | ratio %.3f\n", gbps_staged,
         gbps_plain, ratio);
  EXPECT(ratio >= 0.90, "red line: staged copy >= 90% of plain streaming");
  cudaFree(src); cudaFree(dst);
}

// --------------------------------------------------------- mbarrier ------
__global__ void k_mbarrier(int* ok, int warps_per_block) {
  __shared__ bridge::SharedMbarrier bar;
  __shared__ int data[128];
  if (threadIdx.x == 0) {
    bar.init(blockDim.x);
    data[0] = 0;
  }
  __syncthreads();
  // every thread writes its slot, then arrives; waiter sees all writes
  data[threadIdx.x] = threadIdx.x + 1;
  bar.arrive();
  if (threadIdx.x == 0) {
    bar.wait();
    int sum = 0;
    for (int i = 0; i < blockDim.x; i++) sum += data[i];
    ok[blockIdx.x] = (sum == blockDim.x * (blockDim.x + 1) / 2);
  }
}

void test_mbarrier() {
  int blocks = 8, threads = 128;
  int* d_ok;
  CHECK_CUDA(cudaMalloc(&d_ok, blocks * sizeof(int)));
  k_mbarrier<<<blocks, threads>>>(d_ok, threads / 32);
  CHECK_CUDA(cudaDeviceSynchronize());
  int h[8];
  CHECK_CUDA(cudaMemcpy(h, d_ok, sizeof(h), cudaMemcpyDeviceToHost));
  bool all = true;
  for (int i = 0; i < blocks; i++) all &= (h[i] != 0);
  EXPECT(all, "mbarrier emulation: arrive/wait orders cross-thread writes");
  cudaFree(d_ok);
}

// --------------------------------------------------------- redux ---------
__global__ void k_redux(const int* in, int* sum, int* mx) {
  int v = in[threadIdx.x];
  int s = bridge::redux_add_u32(v);
  int m = bridge::redux_max_u32(v);
  if (threadIdx.x == 0) { *sum = s; *mx = m; }
}

void test_redux() {
  int h[32], sref = 0, mref = -1000;
  for (int i = 0; i < 32; i++) { h[i] = (i * 7) % 50 - 20; sref += h[i]; mref = mref > h[i] ? mref : h[i]; }
  int *d_in, *d_sum, *d_mx;
  CHECK_CUDA(cudaMalloc(&d_in, 128)); CHECK_CUDA(cudaMalloc(&d_sum, 4)); CHECK_CUDA(cudaMalloc(&d_mx, 4));
  CHECK_CUDA(cudaMemcpy(d_in, h, 128, cudaMemcpyHostToDevice));
  k_redux<<<1, 32>>>(d_in, d_sum, d_mx);
  int s = 0, m = 0;
  CHECK_CUDA(cudaMemcpy(&s, d_sum, 4, cudaMemcpyDeviceToHost));
  CHECK_CUDA(cudaMemcpy(&m, d_mx, 4, cudaMemcpyDeviceToHost));
  EXPECT(s == sref && m == mref, "redux butterfly exact");
  cudaFree(d_in); cudaFree(d_sum); cudaFree(d_mx);
}

// ------------------------------------------------- fp8 e4m3 -> f16 -----
// Exhaustive over all 256 E4M3 codes against the spec reference:
// value = sign * (exp==0 ? man*2^-9 : 2^(exp-7) * (1 + man/8)),
// codes 0x7F/0xFF are NaN. E4M3 is a subset of FP16: the bridge output
// must be bit-exact against the host reference.

__global__ void k_fp8_cvt(const uint8_t* in, unsigned short* out) {
  int i = threadIdx.x;
  out[i] = bridge::e4m3_to_f16(in[i]);
}

void test_fp8_cvt() {
  uint8_t in[256];
  unsigned short ref[256], dev[256];
  for (int c = 0; c < 256; c++) {
    in[c] = (uint8_t)c;
    int sign = (c & 0x80) ? -1 : 1;
    int exp = (c >> 3) & 0xF;
    int man = c & 0x7;
    float v;
    if (exp == 15 && man == 7)
      v = nanf("");                       // E4M3 NaN
    else if (exp == 0)
      v = (float)sign * (float)man * 0.001953125f;  // man * 2^-9 (sign kept)
    else
      v = (float)sign * ldexpf(1.0f + man / 8.0f, exp - 7);
    unsigned short bits;
    if (isnan(v)) {
      bits = (unsigned short)(0x7E00 | ((in[c] & 0x80) << 8));  // keep NaN sign
    } else if (v == 0.0f) {
      bits = (unsigned short)f16_bits(v);  // keep -0.0 sign
    } else {
      bits = f16_bits(v);
    }
    ref[c] = bits;
  }
  uint8_t* d_in;
  unsigned short* d_out;
  CHECK_CUDA(cudaMalloc(&d_in, 256));
  CHECK_CUDA(cudaMalloc(&d_out, 512));
  CHECK_CUDA(cudaMemcpy(d_in, in, 256, cudaMemcpyHostToDevice));
  k_fp8_cvt<<<1, 256>>>(d_in, (unsigned short*)d_out);
  CHECK_CUDA(cudaDeviceSynchronize());
  CHECK_CUDA(cudaMemcpy(dev, d_out, 512, cudaMemcpyDeviceToHost));
  int bad = 0;
  for (int c = 0; c < 256; c++) {
    if (ref[c] != dev[c]) {
      if (bad < 6)
        printf("  fp8 mismatch code 0x%02x: dev 0x%04x ref 0x%04x\n", c,
               dev[c], ref[c]);
      bad++;
    }
  }
  EXPECT(bad == 0, "fp8 e4m3 -> f16 exhaustive 256/256 exact");
  cudaFree(d_in); cudaFree(d_out);
}

// --------------------------------------------------------- bf16 RNE ----
void test_bf16_rne() {
  // exact cases: bf16 values pass through; a value needing round-up
  // increments; the RNE tie (round bit set, sticky clear, lsb clear)
  // stays put.
  struct Case { float f; uint32_t expect; };
  Case cases[] = {
      {1.0f, 0x3F80u},
      // exact tie between 1.0 and 1+2^-8: RNE keeps the even side
      {1.00390625f, 0x3F80u},
      // round bit set with sticky: rounds up to the next bf16
      {1.005859375f, 0x3F81u},
      // 65504 sits between bf16 65280 and 65536: RNE -> 65536
      {65504.0f, 0x4780u},
      {0.0f, 0x0u},
  };
  bool all = true;
  for (const auto& c : cases) {
    uint32_t got = bridge::f32_to_bf16_rne_bits(c.f);
    if (got != c.expect) {
      all = false;
      printf("  bf16 rne mismatch: f=%f got 0x%04x want 0x%04x\n", c.f, got,
             c.expect);
    }
  }
  EXPECT(all, "bf16 rne bit tricks");
}

int main() {
  printf("cu_sm80_on_sm75 conformance suite\n");
  test_mma_adapter();
  test_staged_redline();
  test_mbarrier();
  test_redux();
  test_fp8_cvt();
  test_bf16_rne();
  printf(g_failures ? "\nSUITE: %d FAILURES\n" : "\nSUITE: ALL PASS\n",
         g_failures);
  return g_failures ? 1 : 0;
}
