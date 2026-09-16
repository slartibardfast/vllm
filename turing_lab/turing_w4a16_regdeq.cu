// turing_w4a16_regdeq.cu — TU102 (sm_75) W4A16 GEMM consuming the INCUMBENT
// Marlin repacked weight layout directly (checkpoint-compatible: the engine's
// existing gptq_marlin_repack output feeds it unchanged), with the
// dequantization done entirely in registers via the two-lops3 LUT idiom.
//
// What this kernel is an evolution of
// -----------------------------------
// turing_lab/reference/turing_w4a16_opt.cu k_opt (the fork's validated
// 23.50 TFLOP/s oracle-passing kernel): same CTA/warp organization (256
// threads, 8 warps each owning a disjoint output column block, no
// cross-warp reduction, direct fork-style epilogue) and the same A-tile
// staging (plain half2 global reads into a padded row-major smem tile,
// conflict-free fragment reads). It is NOT an evolution of k_opt2 (that
// pipelined variant is marked NOT oracle-validated in the reference file).
//
// What changes vs k_opt
// ---------------------
// - B is consumed in the incumbent repacked layout (plan/0002
//   marlin-contract.md, "The repack interleave contract, RESOLVED"): the
//   int4 addressing of marlin_template.h:549-616 applies unchanged. No
//   self-defined repack, no sQE/sQO even/odd planes.
// - ZERO staging-time ALU on B: k_opt paid two AND/shift ops per word at
//   staging time (the sQE/sQO split) plus a byte_perm at consume time. Here
//   staging is a raw int4 memcpy and dequantization happens per B fragment
//   in registers with two lop3 ops (dequant.h:121-142 constants).
// - Scales are consumed in the incumbent's 8x8-permuted layout directly
//   (marlin_permute_scales output; contract "Scales" section), not the
//   fork's plain row-major S[g*N + n].
//
// Dequant choice (documented per mission spec)
// --------------------------------------------
// Two lop3 per word (dequant.h:121-142, GPTQ kU4B8 non-skip-flop path),
// then the incumbent's hsub2/hfma2 pair:
//   lo = lop3(q, 0x000f000f, 0x64006400)  -> fp16 {1024+q(k=2t), 1024+q(k=2t+1)}
//   hi = lop3(q, 0x00f000f0, 0x64006400)  -> fp16 {1024+16q(k=8+2t), ...}
//   frag_b[0] = hsub2(lo, 0x64086408)                    -> q - 8
//   frag_b[1] = hfma2(hi, 0x2c00 (2^-4), 0xd480 (-72))   -> q - 8
// followed by the scale multiply (marlin_template.h:107-116 scale()).
// The contract's alternative — folding the scale in as ONE hfma2 with a
// precomputed addend, w = (1024+q)*s - 1032*s — was REJECTED on the
// incumbent source's own guidance: dequant.h:39-47 states the
// scale(bias)-cached form "may have accuracy issue, so we should not use
// this in most cases" (fp16 cancellation between (1024+q)*s and 1032*s).
// Per the mission discipline (contract vs source conflict -> follow the
// source), the hsub2/hfma2 pair is used. Cost: 2 lop3 + 1 hsub2 + 1 hfma2
// + 1 hmul2 (scale) per word, zero ALU at staging.
//
// Index map, every expression traced
// ----------------------------------
// Packed word layout (contract table): word W[T,U,th,w] at linear int4
// address T*(N/2) + U*32 + th, word w within the int4; nibble i holds
// weight (n0, k) with n0 = 64U + 16w + c, t = th%4, c = th/4, k relative
// to the 16-k tile per the contract's nibble table. BN = 64 = exactly one
// U tile, so U = 0 and the n16 slab index equals the word index w.
// - Warp w owns output columns n_tile + 8w .. +8 (one m16n8 C fragment).
//   Even warps consume dequant(word[w>>1]); odd warps consume
//   dequant(word[w>>1] >> 8) -> column +8 (marlin_template.h:1235-1236,
//   b_quant_1 = b_quant_0 >> 8; contract "Fragment wiring").
//   Column of lane (t,c): 8w + c in both cases (16*(w>>1) + c = 8w + c).
// - Consuming lane (t,c) reads word th = 4c+t = lane (contract: the
//   lane's hardware k-pair index equals the repack t; NO shuffling).
//   Its int4 is smem slot T*32 + lane: the repack puts th = 4c+t
//   consecutive, so a warp's 32 int4 reads are one contiguous 512B span.
// - Scales: orig column q lives at stored half p = 8*(q%8) + q/8 within
//   each 64-block (marlin_utils.py:460-475, verified against contract).
//   Lane (t,c) needs column 8w+c -> stored half 8c + w, read as one half
//   and broadcast (marlin_template.h:664-665 s_sh_rd and :988 consume the
//   same permuted int4 in the incumbent's own warp split).
// - MMA: the exact incumbent Turing double issue (marlin_mma.h:22-35),
//   two mma.sync.aligned.m16n8k8.row.col.f32.f16.f16.f32 per k16, same
//   fp32 accumulator. A fragment = {A[g][2t], A[g+8][2t], A[g][8+2t],
//   A[g+8][8+2t]} (m16n8k16 A layout, read directly from smem as the
//   validated fork kernel does; no ldmatrix).
//
// Pipeline
// --------
// k-chunks of BK = 128 = one scale group (group_blocks = 8, contract
// "Scales"). Double-buffered smem for B int4s + A tile + scale row, the
// fork's two-sync discipline: per chunk, (1) issue global->register loads
// for chunk k+1, (2) compute chunk k from buffer k%2, __syncthreads(),
// (3) store the staged registers into buffer (k+1)%2, __syncthreads().
// Buffer (k+1)%2 was last read by compute(k-1), which completed before
// iteration k-1's barriers, so the stores in (3) race nothing; buffer k%2
// is never written while being read. (k_opt's own double-buffer attempt
// k_opt2 produces nan; this ordering is the hazard-argued replacement.)
//
// v1 scope cuts (OPEN items): single-stage full-K, one config, no
// split-K/workspace/locks, no act-order (g_idx) handling, no MoE, no
// bias, fp16 output only, N % 64 == 0 required. The epilogue writes
// fp16 directly from fp32 accumulators (fork k_opt style), so no
// use_fp32_reduce machinery either.
//
// No GPU execution in this task: this file is compile-verified only.

#include <cuda_fp16.h>
#include <cstdint>

#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAException.h>

#ifndef TORCH_EXTENSION_NAME
#define TORCH_EXTENSION_NAME turing_w4a16_regdeq
#endif

#define DEVINL __device__ __forceinline__

namespace {

// ---- config: 256 threads, tile m16 (thread_m_blocks = 1) x n64, k-chunk 128
constexpr int THREADS = 256;
constexpr int BM = 16;   // m tile: thread_m_blocks = 1
constexpr int BN = 64;   // n tile: exactly one 64-n repack U tile
constexpr int BK = 128;  // k chunk == group size (group_blocks = 8)
constexpr int K16S = BK / 16;        // k16 steps per chunk
constexpr int B_ROW_INT4S = 32;      // int4s per k16 row of the tile: the
                                     // repack's 128 words per (16k x 64n) tile
constexpr int A_ROW_HALVES = BK + 8; // +8 pad: fork k_opt's bank-conflict-free
                                     // fragment read pattern

// Lookup-table 3-input logic op, verbatim idiom of dequant.h:74-81.
template <int lut>
DEVINL int lop3(int a, int b, int c) {
  int res;
  asm volatile("lop3.b32 %0, %1, %2, %3, %4;\n"
               : "=r"(res)
               : "r"(a), "r"(b), "r"(c), "n"(lut));
  return res;
}

// GPTQ kU4B8 register dequant of one packed word into one column's B
// fragment {k-pair (2t,2t+1), k-pair (8+2t,9+2t)}; constants verbatim from
// dequant.h:121-142 (see header comment for the rejected addend variant).
DEVINL void dequant_u4b8(int q, __half2* frag_b) {
  const int LO = 0x000f000f;
  const int HI = 0x00f000f0;
  const int EX = 0x64006400;
  // Guarantee that the `(a & b) | c` operations are LOP3s.
  int lo = lop3<(0xf0 & 0xcc) | 0xaa>(q, LO, EX);
  int hi = lop3<(0xf0 & 0xcc) | 0xaa>(q, HI, EX);
  const int SUB = 0x64086408;
  const int MUL = 0x2c002c00;
  const int ADD = 0xd480d480;
  frag_b[0] = __hsub2(*reinterpret_cast<half2*>(&lo),
                      *reinterpret_cast<const half2*>(&SUB));
  frag_b[1] = __hfma2(*reinterpret_cast<half2*>(&hi),
                      *reinterpret_cast<const half2*>(&MUL),
                      *reinterpret_cast<const half2*>(&ADD));
}

// Multiply the dequantized fragment by its column scale; the incumbent's
// scale() idiom (marlin_template.h:107-116): broadcast one half, two hmul2.
DEVINL void scale_frag(__half2* frag_b, __half s) {
  const __half2 s2 = __half2half2(s);
  frag_b[0] = __hmul2(frag_b[0], s2);
  frag_b[1] = __hmul2(frag_b[1], s2);
}

// The exact incumbent Turing m16n8k8 double issue per k16 (marlin_mma.h
// :22-35, f32-accumulate path): {a0,a1}x{b[0]} lower k8, then {a2,a3}x
// {b[1]} upper k8, same accumulator. Constraint/clobber style copied 1:1.
DEVINL void mma_1688_f32(const uint32_t* a, const uint32_t* b, float* c) {
  asm volatile(
      "mma.sync.aligned.m16n8k8.row.col.f32.f16.f16.f32 "
      "{%0,%1,%2,%3}, {%4,%5}, {%6}, {%7,%8,%9,%10};\n"
      : "=f"(c[0]), "=f"(c[1]), "=f"(c[2]), "=f"(c[3])
      : "r"(a[0]), "r"(a[1]), "r"(b[0]), "f"(c[0]), "f"(c[1]), "f"(c[2]),
        "f"(c[3]));
  asm volatile(
      "mma.sync.aligned.m16n8k8.row.col.f32.f16.f16.f32 "
      "{%0,%1,%2,%3}, {%4,%5}, {%6}, {%7,%8,%9,%10};\n"
      : "=f"(c[0]), "=f"(c[1]), "=f"(c[2]), "=f"(c[3])
      : "r"(a[2]), "r"(a[3]), "r"(b[1]), "f"(c[0]), "f"(c[1]), "f"(c[2]),
        "f"(c[3]));
}

__global__ void __launch_bounds__(THREADS) turing_w4a16_regdeq_kernel(
    const __half* __restrict__ A,  // [M, K] fp16 activations
    const int4* __restrict__ Bp,   // marlin-repacked weights: {K/16, N/2}
                                   // int4s (contract word map)
    const int4* __restrict__ Sp,   // PERMUTED fp16 scales (marlin_permute_
                                   // scales output): {ceil(K/128), N/8} int4s
    __half* __restrict__ C,        // [M, N] fp16 output
    int M, int N, int K) {
  // Double-buffered staging: B int4s (8 KB), A tile (8.5 KB), scale row
  // (256 B) per buffer; ~17 KB total, so 3 CTAs fit per TU102 SM.
  __shared__ int4 sB[2][K16S * B_ROW_INT4S];
  __shared__ __half sA[2][BM][A_ROW_HALVES];
  __shared__ __half sS[2][BN];

  const int tid = threadIdx.x;
  const int warp = tid >> 5;
  const int lane = tid & 31;
  const int lane_t = lane & 3;   // hardware k-pair index == repack t
                                 // (contract "Fragment wiring")
  const int lane_c = lane >> 2;  // hardware groupID == repack c
  const bool odd = warp & 1;     // odd warps take the +8 column half
  const int word_sel = warp >> 1;  // n16 slab == word index within the int4
                                   // (BN = 64 -> U = 0; contract: word w
                                   // covers n0 = 64U + 16w + c)

  const int n_tile = blockIdx.x * BN;
  const int m_base = blockIdx.y * BM;

  // B rows are k16 units; row stride N/2 int4s (contract: 2*size_n words
  // per k16 row). Scale row stride N/8 int4s (incumbent s_gl_stride,
  // marlin_template.h:562).
  const int kB16 = K >> 4;
  const int b_row_stride = N >> 1;
  const int s_row_stride = N >> 3;
  const int nchunks = (K + BK - 1) / BK;

  float acc[4] = {0.f, 0.f, 0.f, 0.f};  // one m16n8 fp32 C fragment / lane

  // Per-thread staging registers for one chunk: 1 B int4, 1 A 16B item,
  // threads 0..7 hold the scale row.
  int4 b_reg;
  uint32_t a_reg[4];
  int4 s_reg;

  // Global -> registers for chunk `ch`. No shared memory is touched, so
  // these loads issue before compute(k) and overlap with it. The zero
  // fillers make out-of-range contributions exact zeros after dequant:
  // A rows pad to 0; the B sentinel word is all-nibble-8, which the
  // dequant maps to q - 8 = 0 (lo: 8|0x6400 - 0x6408 = 0; hi: hfma2
  // (1024+128, 2^-4, -72) = 0), so padding cannot perturb any term.
  auto load_regs = [&](int ch) {
    const int k0 = ch * BK;
    {  // B: one int4 per thread at (T = tid/32, th = tid%32)
      const int T = tid >> 5;
      const int th = tid & 31;
      const int brow = (k0 >> 4) + T;
      const int zero_word = static_cast<int>(0x88888888u);
      b_reg = (brow < kB16)
                  ? Bp[(long)brow * b_row_stride + (n_tile >> 1) + th]
                  : make_int4(zero_word, zero_word, zero_word, zero_word);
    }
    {  // A: one 16B item per thread (r = tid/16, c8 = tid%16), fork pattern
      const int r = tid >> 4;
      const int c8 = tid & 15;
      const int gm = m_base + r;
      const int gk = k0 + (c8 << 3);
      const __half2 zero2 = __float2half2_rn(0.f);
      __half2 h0 = zero2, h1 = zero2, h2 = zero2, h3 = zero2;
      if (gm < M && gk < K) {
        const __half2* src =
            reinterpret_cast<const __half2*>(&A[(long)gm * K + gk]);
        h0 = src[0]; h1 = src[1]; h2 = src[2]; h3 = src[3];
      }
      a_reg[0] = *reinterpret_cast<uint32_t*>(&h0);
      a_reg[1] = *reinterpret_cast<uint32_t*>(&h1);
      a_reg[2] = *reinterpret_cast<uint32_t*>(&h2);
      a_reg[3] = *reinterpret_cast<uint32_t*>(&h3);
    }
    if (tid < 8) {  // scale row g = k0/128, tile's 8 int4s
      const int g = k0 / 128;
      s_reg = Sp[(long)g * s_row_stride + (n_tile >> 3) + tid];
    }
  };

  // Registers -> the buffer just vacated by the barrier.
  auto store_smem = [&](int buf) {
    // Consumption reads sB[buf][T*32 + lane] with lane == th = 4c+t, so the
    // linear slot is exactly tid = T*32 + th (contract th map; no shuffle).
    sB[buf][tid] = b_reg;
    const int r = tid >> 4;
    const int c8 = tid & 15;
    *reinterpret_cast<uint32_t*>(&sA[buf][r][c8 * 8 + 0]) = a_reg[0];
    *reinterpret_cast<uint32_t*>(&sA[buf][r][c8 * 8 + 2]) = a_reg[1];
    *reinterpret_cast<uint32_t*>(&sA[buf][r][c8 * 8 + 4]) = a_reg[2];
    *reinterpret_cast<uint32_t*>(&sA[buf][r][c8 * 8 + 6]) = a_reg[3];
    if (tid < 8) {
      *reinterpret_cast<int4*>(&sS[buf][tid * 8]) = s_reg;
    }
  };

  // Consume one staged chunk: 8 k16 steps, 2 m16n8k8 issues each, all into
  // the same fp32 accumulators (scales applied per k16 in the fp16 weight
  // domain, incumbent matmul semantics marlin_template.h:1275-1277; within
  // a 128-group the scale is constant, so this matches the incumbent's
  // redundant-but-exact per-k16 application).
  auto compute = [&](int buf) {
    // Lane (t,c) of warp w needs the scale of column 8w + c, which the
    // 8x8 permute puts at stored half 8c + w (marlin_utils.py:460-475).
    const __half s_col = sS[buf][(lane_c << 3) + warp];
#pragma unroll
    for (int T = 0; T < K16S; ++T) {
      // B fragment: the lane's word th = 4c+t = lane of k16 row T.
      const int4 bq = sB[buf][T * B_ROW_INT4S + lane];
      int word = reinterpret_cast<const int*>(&bq)[word_sel];
      if (odd) word >>= 8;  // column +8 half (marlin_template.h:1235-1236)
      __half2 fb[2];
      dequant_u4b8(word, fb);   // 2 lop3 + hsub2/hfma2 (dequant.h:121-142)
      scale_frag(fb, s_col);    // broadcast half, two hmul2 (marlin_template.h:107-116)
      // A fragment, k16: {A[g][2t], A[g+8][2t], A[g][8+2t], A[g+8][8+2t]}
      // relative to the k16 row — the m16n8k16 A layout the double issue
      // consumes (fork k_opt reads the same halves for the k8 case).
      const uint32_t a0 =
          *reinterpret_cast<const uint32_t*>(&sA[buf][lane_c][T * 16 + 2 * lane_t]);
      const uint32_t a1 =
          *reinterpret_cast<const uint32_t*>(&sA[buf][lane_c + 8][T * 16 + 2 * lane_t]);
      const uint32_t a2 =
          *reinterpret_cast<const uint32_t*>(&sA[buf][lane_c][T * 16 + 8 + 2 * lane_t]);
      const uint32_t a3 =
          *reinterpret_cast<const uint32_t*>(&sA[buf][lane_c + 8][T * 16 + 8 + 2 * lane_t]);
      const uint32_t a[4] = {a0, a1, a2, a3};
      mma_1688_f32(a, reinterpret_cast<const uint32_t*>(fb), acc);
    }
  };

  // ---- two-sync double-buffer pipeline ----
  load_regs(0);
  store_smem(0);
  __syncthreads();
  for (int ch = 0; ch < nchunks; ++ch) {
    if (ch + 1 < nchunks) load_regs(ch + 1);  // overlap global latency
    compute(ch & 1);
    // All warps done with chunk ch; buffer (ch+1)%2 was last read by
    // compute(ch-1), which finished before the previous barriers, so it is
    // free to overwrite; buffer ch%2 is never written while in use.
    __syncthreads();
    if (ch + 1 < nchunks) store_smem((ch + 1) & 1);
    __syncthreads();
  }

  // ---- epilogue: direct fp32 -> fp16 global write, fork k_opt style ----
  // m16n8 C fragment: c0,c1 = D[groupID][2t],[2t+1]; c2,c3 = row +8
  // (marlin_mma.h m16n8k8 D layout). Warp w's columns: n_tile + 8w + 2t.
  const int row0 = m_base + lane_c;
  const int col = n_tile + (warp << 3) + (lane_t << 1);
  if (row0 < M) {
    C[(long)row0 * N + col] = __float2half(acc[0]);
    C[(long)row0 * N + col + 1] = __float2half(acc[1]);
  }
  if (row0 + 8 < M) {
    C[(long)(row0 + 8) * N + col] = __float2half(acc[2]);
    C[(long)(row0 + 8) * N + col + 1] = __float2half(acc[3]);
  }
}

}  // namespace

void turing_w4a16_regdeq(torch::Tensor A, torch::Tensor B_packed,
                         torch::Tensor scales_perm, torch::Tensor C_out,
                         int M, int N, int K) {
  TORCH_CHECK(A.is_cuda() && B_packed.is_cuda() && scales_perm.is_cuda() &&
                  C_out.is_cuda(),
              "all tensors must be CUDA");
  TORCH_CHECK(A.scalar_type() == torch::kHalf, "A must be fp16");
  TORCH_CHECK(B_packed.scalar_type() == torch::kInt,
              "B_packed must be int32 (marlin repacked words)");
  TORCH_CHECK(scales_perm.scalar_type() == torch::kHalf,
              "scales_perm must be fp16 (marlin_permute_scales output)");
  TORCH_CHECK(C_out.scalar_type() == torch::kHalf, "C_out must be fp16");
  TORCH_CHECK(A.is_contiguous() && B_packed.is_contiguous() &&
                  scales_perm.is_contiguous() && C_out.is_contiguous(),
              "all tensors must be contiguous");
  TORCH_CHECK(N % 64 == 0, "N must be a multiple of 64, got ", N);
  TORCH_CHECK(K % 16 == 0, "K must be a multiple of 16, got ", K);
  TORCH_CHECK(B_packed.numel() == (long)K * N / 8,
              "B_packed must hold K*N/8 int32 words (K/16 x 2N), got ",
              B_packed.numel());
  TORCH_CHECK(scales_perm.numel() == (long)((K + 127) / 128) * N,
              "scales_perm must hold ceil(K/128)*N fp16 scales (group 128)");
  TORCH_CHECK(C_out.size(0) == M && C_out.size(1) == N,
              "C_out must be [M, N]");

  const __half* A_p =
      reinterpret_cast<const __half*>(A.data_ptr<at::Half>());
  const int4* Bp = reinterpret_cast<const int4*>(B_packed.data_ptr<int>());
  const int4* Sp =
      reinterpret_cast<const int4*>(scales_perm.data_ptr<at::Half>());
  __half* C_p = reinterpret_cast<__half*>(C_out.data_ptr<at::Half>());

  dim3 grid(N / BN, (M + BM - 1) / BM);
  turing_w4a16_regdeq_kernel<<<grid, THREADS, 0,
                               at::cuda::getCurrentCUDAStream()>>>(
      A_p, Bp, Sp, C_p, M, N, K);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("turing_w4a16_regdeq", &turing_w4a16_regdeq,
        "TU102 W4A16 GEMM, incumbent Marlin repacked layout, register lop3 "
        "dequant (checkpoint-compatible)");
}
