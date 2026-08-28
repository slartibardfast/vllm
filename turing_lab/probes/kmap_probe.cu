#include <cuda_fp16.h>
#include <cstdint>
#include <cstdio>

// For probe k: A row 0 has a single 1 at fragment position k (positions
// 0..3 = the four A-fragment halves of row 0); B[k][n] = 1000*(lane+1) with
// the pair halves (1000*(lane+1) [b.x], 1000*(lane+1)+7 [b.y]).
// D[0][n] then reveals which lane-half holds that k for column n.
__global__ void probe(float* D, int kpos) {
  int lane = threadIdx.x & 31;
  int g = lane >> 2, c = lane & 3;
  float v0 = (kpos == 0) ? 1.f : 0.f;
  float v1 = (kpos == 1) ? 1.f : 0.f;
  float v2 = (kpos == 2) ? 1.f : 0.f;
  float v3 = (kpos == 3) ? 1.f : 0.f;
  __half2 a0 = __halves2half2(__float2half(v0), __float2half(v1));
  __half2 a1 = __halves2half2(__float2half(v2), __float2half(v3));
  __half2 b = __halves2half2(__float2half((float)(1000 * (lane + 1))),
                             __float2half((float)(1000 * (lane + 1) + 7)));
  float d[4] = {0,0,0,0};
  asm volatile("mma.sync.aligned.m16n8k8.row.col.f32.f16.f16.f32 "
               "{%0,%1,%2,%3}, {%4,%5}, {%6}, {%0,%1,%2,%3};\n"
               : "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3])
               : "r"(*reinterpret_cast<uint32_t*>(&a0)),
                 "r"(*reinterpret_cast<uint32_t*>(&a1)),
                 "r"(*reinterpret_cast<uint32_t*>(&b)));
  // row 0, cols 0..7
  D[2*c] = d[0];
  D[2*c+1] = d[1];
}

int main() {
  float* D;
  cudaMalloc(&D, 16*4);
  for (int kpos = 0; kpos < 4; kpos++) {
    cudaMemset(D, 0, 16*4);
    probe<<<1,32>>>(D, kpos);
    cudaDeviceSynchronize();
    float h[16];
    cudaMemcpy(h, D, 8*4, cudaMemcpyDeviceToHost);
    printf("kpos %d: D[0][:4] =", kpos);
    for (int c = 0; c < 4; c++) printf(" %11.1f", h[c]);
    printf("\n");
  }
  return 0;
}
