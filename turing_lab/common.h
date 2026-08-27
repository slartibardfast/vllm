// turing_lab: TU102 (sm_75) microbenchmarks for the resources Marlin exercises.
// Run under locked clocks. Every kernel keeps its operands in registers so the
// measurement isolates the execution pipes, not memory traffic, unless the
// benchmark's name says memory.
#pragma once

#include <cstdio>
#include <cstdint>
#include <cstdlib>
#include <cuda_runtime.h>

#define CUDA_CHECK(x)                                                        \
  do {                                                                       \
    cudaError_t e_ = (x);                                                    \
    if (e_ != cudaSuccess) {                                                 \
      fprintf(stderr, "CUDA error %s at %s:%d\n", cudaGetErrorString(e_),    \
              __FILE__, __LINE__);                                           \
      exit(1);                                                               \
    }                                                                        \
  } while (0)

struct Timer {
  cudaEvent_t a, b;
  Timer() { CUDA_CHECK(cudaEventCreate(&a)); CUDA_CHECK(cudaEventCreate(&b)); }
  void start() { CUDA_CHECK(cudaEventRecord(a)); }
  // returns elapsed milliseconds
  float stop() {
    CUDA_CHECK(cudaEventRecord(b));
    CUDA_CHECK(cudaEventSynchronize(b));
    float ms = 0.f;
    CUDA_CHECK(cudaEventElapsedTime(&ms, a, b));
    return ms;
  }
};

// Grid sized to oversubscribe all SMs; 8 warps per block keeps the scheduler
// fed while each warp's loop stays in registers.
constexpr int kBlocksPerSm = 8;
constexpr int kThreads = 256;
constexpr int kWarpsPerBlock = kThreads / 32;

inline int sm_count() {
  int dev = 0, sms = 0;
  CUDA_CHECK(cudaGetDevice(&dev));
  CUDA_CHECK(cudaDeviceGetAttribute(&sms, cudaDevAttrMultiProcessorCount, dev));
  return sms;
}

inline int iters_arg(int argc, char** argv, int default_iters) {
  return (argc > 2) ? atoi(argv[2]) : default_iters;
}
