// sm80_on_sm75.cuh — the cu_sm80_on_sm75 primitives bridge (plan/0006).
//
// Hand-written inline-asm PTX implementing the sm_80+ device primitives
// that flash-attn and FlashInfer depend on, with sm_75-native semantics
// fixed in the header rather than left to the compiler. Raw-PTX policy:
// C++ wrappers exist only at this API surface.
//
// 1:1-by-construction notes (plan/0006 red-line contract):
// - commit_group/wait_group compile to scheduling fences: on sm_75 the
//   STS of a staged copy cannot issue before its LDG retires (hardware
//   scoreboard), so per-thread ordering is enforced with zero runtime
//   cost. The irreducible delta is registers held in flight; callers
//   obey the occupancy budget.
// - The m16n8k16 f16 MMA adapter issues 2x m16n8k8 in m16n8k16 register
//   layout (a0/a1 = k 0..7 rows g/g+8, a2/a3 = k 8..15), which matches
//   the documented m16n8k16 fragment mapping exactly.
// - Streaming loads carry .cs (evict-first) hints so staging never
//   evicts hot L2; SASS inspection verifies the hints survived.
#pragma once
#include <cstdint>
#include <cuda_fp16.h>

namespace bridge {

// ---------------- streaming global -> register loads ----------------
// .cs = cache-streaming (evict-first): staged data must not displace
// hot L2. .nc variant takes the non-coherent read-only path.

__device__ __forceinline__ void ldg_cs_v4(uint32_t& r0, uint32_t& r1,
                                          uint32_t& r2, uint32_t& r3,
                                          const void* g) {
  asm volatile("ld.global.cs.v4.u32 {%0,%1,%2,%3}, [%4];\n"
               : "=r"(r0), "=r"(r1), "=r"(r2), "=r"(r3)
               : "l"(g));
}

__device__ __forceinline__ void ldg_nc_v4(uint32_t& r0, uint32_t& r1,
                                          uint32_t& r2, uint32_t& r3,
                                          const void* g) {
  asm volatile("ld.global.nc.v4.u32 {%0,%1,%2,%3}, [%4];\n"
               : "=r"(r0), "=r"(r1), "=r"(r2), "=r"(r3)
               : "l"(g));
}

// Raw ld/st.shared take SHARED-WINDOW offsets, not generic addresses:
// every shared-space access converts via __cvta_generic_to_shared first.
// Skipping this is the classic raw-PTX bug (the generic address of a
// shared byte is far outside the 64 KiB shared window -> IMA).

__device__ __forceinline__ uint32_t smem_addr(const void* s) {
  return (uint32_t)__cvta_generic_to_shared(s);
}

__device__ __forceinline__ void sts_v4(void* s, uint32_t r0, uint32_t r1,
                                       uint32_t r2, uint32_t r3) {
  uint32_t a = smem_addr(s);
  asm volatile("st.shared.v4.u32 [%0], {%1,%2,%3,%4};\n"
               :: "r"(a), "r"(r0), "r"(r1), "r"(r2), "r"(r3));
}

__device__ __forceinline__ void lds_v4(uint32_t& r0, uint32_t& r1,
                                       uint32_t& r2, uint32_t& r3,
                                       const void* s) {
  uint32_t a = smem_addr(s);
  asm volatile("ld.shared.v4.u32 {%0,%1,%2,%3}, [%4];\n"
               : "=r"(r0), "=r"(r1), "=r"(r2), "=r"(r3)
               : "r"(a));
}

// ------------- cp.async commit/wait semantics -------------
// Per-thread ordering is hardware-enforced by the LDG->STS register
// dependency, so the group API lowers to compiler scheduling fences.
// Cross-thread visibility still requires the caller's barrier.

__device__ __forceinline__ void commit_group() {
  asm volatile("" ::: "memory");
}

template <int PENDING>
__device__ __forceinline__ void wait_group() {
  static_assert(PENDING >= 0, "");
  asm volatile("" ::: "memory");
}

// ---------------- MMA: f16 m16n8k16 -> 2x m16n8k8 ----------------
// D (4 x f32) += A (16x16 f16, 4 regs) * B (16x8 f16, 2 regs).
// A/B/C use the documented m16n8k16 fragment layout; the two k8 halves
// chain D -> D exactly as the sm_80 instruction's sequential fp32
// k-slice accumulation (arXiv:2208.11174).

__device__ __forceinline__ void mma_m16n8k8_f32(
    float& d0, float& d1, float& d2, float& d3,
    uint32_t a0, uint32_t a1, uint32_t b) {
  asm volatile(
    "mma.sync.aligned.m16n8k8.row.col.f32.f16.f16.f32 "
    "{%0,%1,%2,%3}, {%4,%5}, {%6}, {%0,%1,%2,%3};\n"
    : "+f"(d0), "+f"(d1), "+f"(d2), "+f"(d3)
    : "r"(a0), "r"(a1), "r"(b));
}

__device__ __forceinline__ void mma_m16n8k16_f32(
    float* d, const uint32_t* a, const uint32_t* b) {
  // a[0] = {row g, k 2t..2t+1}, a[1] = {row g+8, k 2t..2t+1},
  // a[2] = {row g, k 2t+8..2t+9}, a[3] = {row g+8, k 2t+8..2t+9}
  // b[0] = {k 2t..2t+1, col g'}, b[1] = {k 2t+8..2t+9, col g'}
  mma_m16n8k8_f32(d[0], d[1], d[2], d[3], a[0], a[1], b[0]);
  mma_m16n8k8_f32(d[0], d[1], d[2], d[3], a[2], a[3], b[1]);
}

// ---------------- INT8: m16n8k32 -> 4x m8n8k16 ----------------
// The Ampere fragment decomposes as two row-groups x two k-halves
// (research.md, INT8 section). Layouts follow the PTX-documented
// fragments; the oracle tests pin the mapping.

__device__ __forceinline__ void mma_m8n8k16_s32(
    int& d0, int& d1, uint32_t a, uint32_t b, int c0, int c1) {
  asm volatile(
    "mma.sync.aligned.m8n8k16.row.col.s32.s8.s8.s32 "
    "{%0,%1}, {%2}, {%3}, {%0,%1};\n"
    : "+r"(d0), "+r"(d1)
    : "r"(a), "r"(b), "r"(c0), "r"(c1));
}

// ---------------- redux.sync (sm_80) -> shfl butterfly ----------------
__device__ __forceinline__ int redux_add_u32(int v) {
#pragma unroll
  for (int off = 16; off > 0; off >>= 1)
    v += __shfl_xor_sync(0xffffffff, v, off);
  return v;
}

__device__ __forceinline__ int redux_max_u32(int v) {
#pragma unroll
  for (int off = 16; off > 0; off >>= 1)
    v = max(v, __shfl_xor_sync(0xffffffff, v, off));
  return v;
}

// ---------------- conversions ----------------
// bf16 -> f32: bf16 is the top half of f32 (shift, free).
__device__ __forceinline__ float bf16_to_f32(uint32_t bits) {
  return __uint_as_float(bits << 16);
}

// f32 -> bf16 round-to-nearest-even (the sm_80 cvt.rn.bf16.f32
// semantics): round bit + sticky, then mantissa increment.
__device__ __forceinline__ uint32_t f32_to_bf16_rne_bits(float f) {
  uint32_t x = __float_as_uint(f);
  uint32_t lsb = (x >> 16) & 1u;
  uint32_t round = (x >> 15) & 1u;
  uint32_t sticky = x & 0x7FFFu;
  if (round && (sticky || lsb)) x += 0x10000u;
  return x >> 16;
}

// E4M3 -> FP16 is lossless: sign<<8 | (exp+8)<<7 | man<<7 rebias.
// LUT-free integer sequence per element (fp8_marlin style does this
// two-at-a-time); exhaustive 256-case test pins exactness.
__device__ __forceinline__ uint16_t e4m3_to_f16(uint8_t v) {
  uint16_t s = uint16_t(v & 0x80u) << 8;         // sign
  uint16_t e = (v >> 1) & 0x3Fu;                 // exp(4) + man msb
  uint16_t m = uint16_t(v & 0x07u);              // man low bits
  // E4M3 0bEEEE MMM: FP16 = (E-7+15)<<10 | MMM<<7, specials 0x7F/0xFF.
  if (e == 0x3Fu) {                              // NaN encoding
    s |= 0x7E00u | (m << 7);
    return s;
  }
  uint16_t exp = uint16_t((v >> 3) & 0xFu);
  uint16_t man = m;
  if (exp == 0) {                                // subnormal: exact in fp16
    // value = man/8 * 2^-6 -> fp16 normal: exp field = 15-6-3 = adjust
    uint16_t out = s | (uint16_t(15 - 6 - 3) << 10);
    // subnormal E4M3 = man * 2^-9; fp16 normal with exp bias so that
    // man*2^-9 = 1.mm * 2^-9 only if man has msb; handle by float math
    float f = float(man) * 0.001953125f;         // man / 512
    return __half_as_ushort(__float2half(f)) | s;
  }
  uint16_t bits = s | ((uint16_t(exp - 7 + 15)) << 10) | (man << 7);
  return bits;
}

// ---------------- mbarrier emulation (sm_80 -> sm_75) ----------------
// Counted-arrival barrier in shared memory: the CCCL pre-sm80 pattern.
// Single-phase by contract: init -> arrive xN -> wait -> re-init for the
// next phase. Arrivals fence before decrementing; waiters fence after
// observing zero (volatile read = acquire on sm_75).

struct SharedMbarrier {
  uint32_t pending;  // arrivals until the phase completes

  __device__ void init(uint32_t expected_arrivals) {
    pending = expected_arrivals;
    __threadfence_block();
  }

  __device__ void arrive() {
    __threadfence_block();
    atomicSub(&pending, 1u);
  }

  __device__ bool test_wait() {
    return *reinterpret_cast<volatile uint32_t*>(&pending) == 0u;
  }

  __device__ void wait() {
    while (*reinterpret_cast<volatile uint32_t*>(&pending) != 0u) {}
    __threadfence_block();
  }
};

}  // namespace bridge
