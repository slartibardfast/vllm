#include <cuda_fp16.h>
#include <cstdint>
#include <cstdio>

// B[k][n] = 100*k + n ; A row g: A[g][2c]=1, A[g][2c+1]=2, A[g+8][*]=3,4
// then D[g][2c]   = 1*B[2c][g] + 2*B[2c+1][g] = 600c + 200 + 3g
//      D[g][2c+1] = 1*B[2c+1][g] + 2*B[2c+2][g] = 600c + 500 + 3g
__global__ void probe(float* D) {
  int lane = threadIdx.x & 31;
  int g = lane >> 2, c = lane & 3;
  __half2 a0 = __halves2half2(__float2half(1.0f), __float2half(2.0f));
  __half2 a1 = __halves2half2(__float2half(3.0f), __float2half(4.0f));
  // B fragment halves: B[k=2c][n=g] and B[k=2c+1][n=g]
  __half2 b = __halves2half2(__float2half((float)(100 * (2*c) + g)),
                             __float2half((float)(100 * (2*c+1) + g)));
  float d[4] = {0,0,0,0};
  asm volatile("mma.sync.aligned.m16n8k8.row.col.f32.f16.f16.f32 "
               "{%0,%1,%2,%3}, {%4,%5}, {%6}, {%0,%1,%2,%3};\n"
               : "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3])
               : "r"(*reinterpret_cast<uint32_t*>(&a0)),
                 "r"(*reinterpret_cast<uint32_t*>(&a1)),
                 "r"(*reinterpret_cast<uint32_t*>(&b)));
  D[(g)*8 + 2*c] = d[0];
  D[(g)*8 + 2*c+1] = d[1];
  D[(g+8)*16 + 2*c] = d[2];
  D[(g+8)*16 + 2*c+1] = d[3];
}

int main() {
  float* D;
  cudaMalloc(&D, 16*16*4);
  cudaMemset(D, 0, 16*16*4);
  probe<<<1,32>>>(D);
  cudaDeviceSynchronize();
  float h[256];
  cudaMemcpy(h, D, 16*16*4, cudaMemcpyDeviceToHost);
  printf("row 0:"); for (int c = 0; c < 8; c++) printf(" %9.1f", h[c]);
  printf("\\nrow 8:"); for (int c = 0; c < 8; c++) printf(" %9.1f", h[128+c]);
  printf("\\n");
  return 0;
}
