// turing_w4a16_regdeq.cu — TU102 (sm_75) W4A16 GEMM consuming the INCUMBENT
// Marlin repacked weight layout directly (checkpoint-compatible: the engine's
// existing gptq_marlin_repack output feeds it unchanged), with the
// dequantization done entirely in registers via the two-lops3 LUT idiom.
//
// v2 (this revision)
// ------------------
// Two levers over v1, nothing else; numerics are byte-for-byte v1's (same
// LUT 0xea, hsub2 0x64086408, hfma2 0x2c00/0xd480, permuted scale halves,
// fp32 accumulate, partial-K sentinel, M<16 guards). The oracle battery is
// the gate.
//
// Lever 1 — multi-stage pipeline (incumbent-style ring, no cp.async):
//   v1 staged B/A/scales through a 2-buffer double buffer. v2 generalizes to
//   a STAGES-deep smem ring (compile-time constant TW4A16_STAGES, default 3,
//   range 2..4) with the incumbent's prefetch distance: the global loads for
//   chunk ch + STAGES - 1 are issued while computing chunk ch (the incumbent
//   issues fetch_to_shared stages-1 slots ahead; marlin_template.h:1796-1808
//   with wait_for_stage's cp_async_wait<stages - 2> + __syncthreads at
//   :924-931 as its stage-release discipline). Turing has no usable
//   cp.async in this design, so the ring is staged through registers:
//   STAGES - 1 per-thread register sets hold the in-flight chunks (chunk c
//   is loaded into set c % (STAGES - 1) at iteration c - (STAGES - 1) and
//   stored into smem slot c % STAGES at iteration c - 1; at most STAGES - 1
//   sets are ever live, so the set being filled is always the set just
//   vacated by the store two lines earlier in the same iteration). The
//   per-iteration sync pair is v1's, unchanged in role: the first barrier
//   ends compute(ch), the second publishes store(chunk ch+1) before
//   compute(ch+1) reads it. Overwrite safety: slot (ch+1) % STAGES was last
//   READ by compute(ch + 1 - STAGES), and at least 2*(STAGES-1) barriers
//   separate that read from the overwrite (v1's hazard argument, one ring
//   further around). TW4A16_STAGES=2 with TW4A16_BN=64 reduces line-for-line
//   to v1's pipeline.
//
// Lever 2 — wider N tile (BN=128, 8 warps x n16 each):
//   v1's BN=64 gave each of the 8 warps an 8-column slab (one m16n8 C
//   fragment per k16, the word split across warp pairs by the odd-warp >>8).
//   BN=128 gives each warp a full 16-column slab: warp w covers repack
//   U tile U = w>>2 and word w' = w&3 (n0 = 64U + 16w' + c, contract word
//   map), and consumes BOTH halves of its word per k16 — dequant(word) for
//   columns n0..n0+7 and dequant(word >> 8) for columns n0+8..n0+15 (the
//   incumbent's own b_quant_0/b_quant_1 = b_quant_0 >> 8 split,
//   marlin_template.h:1235-1236) — two m16n8k8 double issues per k16 into a
//   second fp32 accumulator quad. Choice documented per mission spec: the
//   alternative "incumbent thread_n_blocks=8-style column mapping" (its
//   small-batch {thread_k 128, thread_n 128, 256 threads} config) splits K
//   across warp rows and reduces partials through shared memory
//   (thread_block_reduce, marlin_template.h:1406-1461), which would change
//   v1's per-warp full-K fp32 accumulation order — rejected under the
//   numerics-freeze constraint. The chosen mapping keeps the fork's
//   no-cross-warp-reduction epilogue and v1's exact accumulation chain per
//   output element; each A fragment read now feeds two B fragments, halving
//   A-side LDS traffic per FLOP.
//   Shape selection is compile-time: TW4A16_BN in {64, 128} (default 128).
//   Rebuild with -DTW4A16_BN=64 to measure the narrow shape; both shapes are
//   compiled from this one file by the same template. N must be a multiple
//   of the selected BN (host check).
//
// What this kernel is an evolution of
// -----------------------------------
// turing_lab/reference/turing_w4a16_opt.cu k_opt (the fork's validated
// 23.50 TFLOP/s oracle-passing kernel): same CTA/warp organization (256
// threads, warps each owning a disjoint output column block, no
// cross-warp reduction, direct fork-style epilogue) and the same A-tile
// staging (plain half2 global reads into a padded row-major smem tile,
// conflict-free fragment reads). It is NOT an evolution of k_opt2 (that
// pipelined variant is marked NOT oracle-validated in the reference file).
// B is consumed in the incumbent repacked layout (plan/0002
// marlin-contract.md, "The repack interleave contract, RESOLVED"); scales
// in the incumbent's 8x8-permuted layout (contract "Scales").
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
// source), the hsub2/hfma2 pair is used.
//
// Index map, every expression traced
// ----------------------------------
// Packed word layout (contract table): word W[T,U,th,w] at linear int4
// address T*(N/2) + U*32 + th, word w within the int4; nibble i holds
// weight (n0, k) with n0 = 64U + 16w + c, t = th%4, c = th/4, k relative
// to the 16-k tile per the contract's nibble table.
// - BN=64: the tile is exactly one U tile (U = 0) and the n16 slab index
//   equals the word index w. Warp w owns columns n_tile + 8w .. +8; even
//   warps consume dequant(word[w>>1]), odd warps dequant(word[w>>1] >> 8).
//   Column of lane (t,c): 8w + c in both cases. UNCHANGED from v1.
// - BN=128: warp w owns U = w>>2, word w' = w&3, columns 64U + 16w' + c
//   and +8 (both half-words). Its scale pair: orig column q = 64U + 16w'+c
//   lives at stored half 8c + 2w' within the 64-block, and q + 8 at
//   8c + 2w' + 1 (marlin_utils.py:460-475 stored[p] = orig[perm[p]],
//   perm[8i+j] = i+8j) — ADJACENT halves, so the warp reads ONE aligned
//   half2 at (lane_c<<3) + 2w and broadcasts .x/.y to the two fragments.
//   Consuming lane (t,c) reads word th = 4c+t = lane of k16 row T at smem
//   slot T*(BN/2) + U*32 + lane — a warp's 32 int4 reads stay one
//   contiguous 512B span (contract: the lane's hardware k-pair index
//   equals the repack t; NO shuffling).
// - Both shapes: lane (t,c) dequantizes into fragments {k-pair (2t,2t+1),
//   k-pair (8+2t,9+2t)}; MMA is the exact incumbent Turing double issue
//   (marlin_mma.h:22-35), two mma.sync.aligned.m16n8k8.row.col.f32.f16.
//   f16.f32 per k16 per fragment, same fp32 accumulator. A fragment =
//   {A[g][2t], A[g+8][2t], A[g][8+2t], A[g+8][8+2t]} read directly from
//   smem as the validated fork kernel does (no ldmatrix).
//
// Shared-memory budget (drives the STAGES/BN defaults)
// ----------------------------------------------------
// Per stage: sB = (BK/16) * (BN/2) int4s; sA = 16 * 136 halves (the +8
// pad spreads row starts across all 32 banks, see A_ROW_HALVES); sS = BN
// halves.
//   BN=64,  S=2:  4096 + 4352 + 128 = 17,152 B  -> 3 CTAs/SM (v1's point)
//   BN=64,  S=3:  3 x 8,576  = 25,728 B          -> 2 CTAs/SM (51.4 KB of 64)
//   BN=64,  S=4:  4 x 8,576  = 34,304 B          -> 1 CTA/SM
//   BN=128, S=3:  3 x 12,800 = 38,400 B          -> 1 CTA/SM (37.5 KB < 48 KB
//                                                   static cap)
//   BN=128, S=4:  4 x 12,800 = 51,200 B          -> refused: over the 48 KB
//                                                   static smem cap (needs an
//                                                   extern-shared + attribute
//                                                   variant; OPEN)
// TU102: 64 KB shared per SM, 1024 threads (32 warps) per SM. Defaults
// BN=128/S=3 run 1 CTA of 8 warps per SM — the same occupancy class as the
// incumbent's small-batch lead config {thread_k 128, thread_n 128,
// 256 threads} at its stage count; the deep ring, not thread count, is
// what hides global latency there. The new limiter vs v1 is shared memory;
// measured parity is the first GPU run's job.
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

// v2 compile-time shape knobs (see header: shared-memory budget).
#ifndef TW4A16_BN
#define TW4A16_BN 64  // the measured-best shape (17.04 M512); BN128/S3
// falsified 2026-09-17: scale-pairing bug (missing +8U term) AND
// 1 CTA/SM occupancy; BN64/S3 ring also falsified (-49 pct, 2 CTAs/SM
// cost more than prefetch depth bought at this class on sm_75)
#endif
#ifndef TW4A16_STAGES
#define TW4A16_STAGES 2  // S3 falsified at BN64 (-49 pct M512)
#endif

#define DEVINL __device__ __forceinline__

namespace {

// ---- config: 256 threads, tile m16 (thread_m_blocks = 1), k-chunk 128;
// the n tile is the template parameter BN_V in {64, 128} (macro
// TW4A16_BN): 64 = exactly one 64-n repack U tile.
constexpr int THREADS = 256;
constexpr int BM = 16;   // m tile: thread_m_blocks = 1
constexpr int BK = 128;  // k chunk == group size (group_blocks = 8)
constexpr int K16S = BK / 16;        // k16 steps per chunk
constexpr int A_ROW_HALVES = BK + 8; // +8: stride 68 w = 4 mod 32
// spreads 8 row starts across all 32 banks (conflict-free reads);
// the +4 trial (v1.1 falsification, -26 pct M512) bought store
// conflicts at the read side's expense - measured, reverted
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

// Kernel template over the two v2 levers: BN_V in {64, 128} (n tile) and
// STAGES_V in [2, 4] (smem ring depth). STAGES_V = 2, BN_V = 64 is v1.
template <int BN_V, int STAGES_V>
__global__ void __launch_bounds__(THREADS) turing_w4a16_regdeq_kernel(
    const __half* __restrict__ A,  // [M, K] fp16 activations
    const int4* __restrict__ Bp,   // marlin-repacked weights: {K/16, N/2}
                                   // int4s (contract word map)
    const int4* __restrict__ Sp,   // PERMUTED fp16 scales (marlin_permute_
                                   // scales output): {ceil(K/128), N/8} int4s
    __half* __restrict__ C,        // [M, N] fp16 output
    int M, int N, int K) {
  static_assert(BN_V == 64 || BN_V == 128,
                "BN_V must be 64 or 128 (one or two repack U tiles)");
  static_assert(STAGES_V >= 2 && STAGES_V <= 4,
                "STAGES_V must be 2..4 (ring depth; 2 == v1)");
  // Static smem cap: 48 KB per block on every current arch.
  static_assert((long)STAGES_V *
                        (K16S * (BN_V / 2) * 16 + BM * A_ROW_HALVES * 2 +
                         BN_V * 2) <=
                    48 * 1024,
                "BN=128 with STAGES=4 needs 51.2 KB static smem (48 KB cap); "
                "use STAGES=3 or an extern-shared variant");

  // BN-derived geometry: U_CTAS 64-n repack tiles per CTA (1 or 2); the B
  // tile rows are k16 units of B_ROW_INT4S int4s each (the repack's 128
  // words per (16k x 64n) tile, times U_CTAS); RSETS staging register sets
  // carry the ring (chunk c -> set c % RSETS).
  constexpr int U_CTAS = BN_V / 64;
  constexpr int B_ROW_INT4S = 32 * U_CTAS;
  constexpr int RSETS = STAGES_V - 1;

  // STAGES-deep ring: B int4s, A tile, scale row per stage slot.
  __shared__ int4 sB[STAGES_V][K16S * B_ROW_INT4S];
  __shared__ __half sA[STAGES_V][BM][A_ROW_HALVES];
  __shared__ __half sS[STAGES_V][BN_V];

  const int tid = threadIdx.x;
  const int warp = tid >> 5;
  const int lane = tid & 31;
  const int lane_t = lane & 3;   // hardware k-pair index == repack t
                                 // (contract "Fragment wiring")
  const int lane_c = lane >> 2;  // hardware groupID == repack c

  const int n_tile = blockIdx.x * BN_V;
  const int m_base = blockIdx.y * BM;

  // B rows are k16 units; row stride N/2 int4s (contract: 2*size_n words
  // per k16 row). Scale row stride N/8 int4s (incumbent s_gl_stride,
  // marlin_template.h:562).
  const int kB16 = K >> 4;
  const int b_row_stride = N >> 1;
  const int s_row_stride = N >> 3;
  const int nchunks = (K + BK - 1) / BK;

  float acc[4 * U_CTAS] = {};  // U_CTAS m16n8 fp32 C fragments per lane;
                               // fragment 1 (BN=128) is the +8 column half

  // Per-thread staging registers, RSETS sets (one per in-flight chunk):
  // U_CTAS B int4s, one A 16B item; threads 0..BN_V/8-1 hold the scale row.
  int4 b_reg[RSETS][U_CTAS];
  uint32_t a_reg[RSETS][4];
  int4 s_reg[RSETS];

  // Global -> registers for chunk `ch` into set `set`. No shared memory is
  // touched, so these loads issue before compute(k) and overlap with it.
  // The zero fillers make out-of-range contributions exact zeros after
  // dequant: A rows pad to 0; the B sentinel word is all-nibble-8, which
  // the dequant maps to q - 8 = 0 (lo: 8|0x6400 - 0x6408 = 0; hi: hfma2
  // (1024+128, 2^-4, -72) = 0), so padding cannot perturb any term.
  auto load_regs = [&](int set, int ch) {
    const int k0 = ch * BK;
    const int zero_word = static_cast<int>(0x88888888u);
    if constexpr (BN_V == 64) {
      // B: one int4 per thread at (T = tid/32, th = tid%32).
      const int T = tid >> 5;
      const int th = tid & 31;
      const int brow = (k0 >> 4) + T;
      b_reg[set][0] = (brow < kB16)
                          ? Bp[(long)brow * b_row_stride + (n_tile >> 1) + th]
                          : make_int4(zero_word, zero_word, zero_word,
                                      zero_word);
    } else {
      // B: two int4s per thread at (T = tid/64, th = tid%64), rows T and
      // T + 4 (256 threads x 2 = 512 int4s = the full 8 x 64 stage).
      const int T = tid >> 6;
      const int th = tid & 63;
      const int col = (n_tile >> 1) + th;
#pragma unroll
      for (int j = 0; j < 2; ++j) {
        const int brow = (k0 >> 4) + T + 4 * j;
        b_reg[set][j] =
            (brow < kB16)
                ? Bp[(long)brow * b_row_stride + col]
                : make_int4(zero_word, zero_word, zero_word, zero_word);
      }
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
      a_reg[set][0] = *reinterpret_cast<uint32_t*>(&h0);
      a_reg[set][1] = *reinterpret_cast<uint32_t*>(&h1);
      a_reg[set][2] = *reinterpret_cast<uint32_t*>(&h2);
      a_reg[set][3] = *reinterpret_cast<uint32_t*>(&h3);
    }
    if (tid < BN_V / 8) {  // scale row g = k0/128, tile's BN_V/8 int4s
      const int g = k0 / 128;
      s_reg[set] = Sp[(long)g * s_row_stride + (n_tile >> 3) + tid];
    }
  };

  // Registers -> the stage slot just vacated by the barrier.
  auto store_smem = [&](int set, int slot) {
    if constexpr (BN_V == 64) {
      // Consumption reads sB[slot][T*32 + lane] with lane == th = 4c+t, so
      // the linear slot is exactly tid = T*32 + th (contract th map).
      sB[slot][tid] = b_reg[set][0];
    } else {
      // Consumption reads sB[slot][T*64 + U*32 + lane]: thread (T = tid/64,
      // th = tid%64) fills row T, then row T + 4; a warp's 32 stores stay
      // one contiguous 512B span (conflict-free LDS.128).
      sB[slot][(tid >> 6) * B_ROW_INT4S + (tid & 63)] = b_reg[set][0];
      sB[slot][((tid >> 6) + 4) * B_ROW_INT4S + (tid & 63)] = b_reg[set][1];
    }
    const int r = tid >> 4;
    const int c8 = tid & 15;
    *reinterpret_cast<uint32_t*>(&sA[slot][r][c8 * 8 + 0]) = a_reg[set][0];
    *reinterpret_cast<uint32_t*>(&sA[slot][r][c8 * 8 + 2]) = a_reg[set][1];
    *reinterpret_cast<uint32_t*>(&sA[slot][r][c8 * 8 + 4]) = a_reg[set][2];
    *reinterpret_cast<uint32_t*>(&sA[slot][r][c8 * 8 + 6]) = a_reg[set][3];
    if (tid < BN_V / 8) {
      *reinterpret_cast<int4*>(&sS[slot][tid * 8]) = s_reg[set];
    }
  };

  // Consume one staged chunk: 8 k16 steps, 2 m16n8k8 issues per B fragment,
  // all into the same fp32 accumulators (scales applied per k16 in the
  // fp16 weight domain, incumbent matmul semantics marlin_template.h:
  // 1275-1277; within a 128-group the scale is constant, so this matches
  // the incumbent's redundant-but-exact per-k16 application).
  auto compute = [&](int slot) {
    if constexpr (BN_V == 64) {
      // Warp w owns columns n_tile + 8w .. +8: word_sel = w>>1, odd warps
      // take the +8 half (marlin_template.h:1235-1236).
      const __half s_col = sS[slot][(lane_c << 3) + warp];
      const int word_sel = warp >> 1;
      const bool odd = warp & 1;
#pragma unroll
      for (int T = 0; T < K16S; ++T) {
        // B fragment: the lane's word th = 4c+t = lane of k16 row T.
        const int4 bq = sB[slot][T * B_ROW_INT4S + lane];
        int word = reinterpret_cast<const int*>(&bq)[word_sel];
        if (odd) word >>= 8;  // column +8 half
        __half2 fb[2];
        dequant_u4b8(word, fb);   // 2 lop3 + hsub2/hfma2 (dequant.h:121-142)
        scale_frag(fb, s_col);    // broadcast half, two hmul2
        // A fragment, k16: {A[g][2t], A[g+8][2t], A[g][8+2t], A[g+8][8+2t]}
        const uint32_t a0 =
            *reinterpret_cast<const uint32_t*>(&sA[slot][lane_c][T * 16 + 2 * lane_t]);
        const uint32_t a1 =
            *reinterpret_cast<const uint32_t*>(&sA[slot][lane_c + 8][T * 16 + 2 * lane_t]);
        const uint32_t a2 =
            *reinterpret_cast<const uint32_t*>(&sA[slot][lane_c][T * 16 + 8 + 2 * lane_t]);
        const uint32_t a3 =
            *reinterpret_cast<const uint32_t*>(&sA[slot][lane_c + 8][T * 16 + 8 + 2 * lane_t]);
        const uint32_t a[4] = {a0, a1, a2, a3};
        mma_1688_f32(a, reinterpret_cast<const uint32_t*>(fb), acc);
      }
    } else {
      // Warp w owns 16 columns (U = w>>2, word w' = w&3) and consumes BOTH
      // half-words: dequant(word) -> columns 16w'+c, dequant(word >> 8) ->
      // columns 16w'+8+c. Their permuted scale halves are adjacent
      // (8c+2w', 8c+2w'+1): one aligned half2 read, .x/.y per fragment.
      // v3 fix: the permute is per-64-block, so the absolute stored
      // half is 64U + 8c + 2w' - the v2 index dropped the 64U block
      // base (and used w where w' = w&3), reading the wrong 64-block's
      // scales for every U > 0.
      const __half2 s_pair = *reinterpret_cast<const __half2*>(
          &sS[slot][((warp >> 2) << 6) + (lane_c << 3)
                    + ((warp & 3) << 1)]);
      const __half s_lo = reinterpret_cast<const __half*>(&s_pair)[0];
      const __half s_hi = reinterpret_cast<const __half*>(&s_pair)[1];
      const int u_blk = warp >> 2;
      const int word_sel = warp & 3;
#pragma unroll
      for (int T = 0; T < K16S; ++T) {
        const int4 bq = sB[slot][T * B_ROW_INT4S + u_blk * 32 + lane];
        const int word = reinterpret_cast<const int*>(&bq)[word_sel];
        __half2 fb0[2];
        __half2 fb1[2];
        dequant_u4b8(word, fb0);
        dequant_u4b8(word >> 8, fb1);  // column +8 half (same k pairs)
        scale_frag(fb0, s_lo);
        scale_frag(fb1, s_hi);
        const uint32_t a0 =
            *reinterpret_cast<const uint32_t*>(&sA[slot][lane_c][T * 16 + 2 * lane_t]);
        const uint32_t a1 =
            *reinterpret_cast<const uint32_t*>(&sA[slot][lane_c + 8][T * 16 + 2 * lane_t]);
        const uint32_t a2 =
            *reinterpret_cast<const uint32_t*>(&sA[slot][lane_c][T * 16 + 8 + 2 * lane_t]);
        const uint32_t a3 =
            *reinterpret_cast<const uint32_t*>(&sA[slot][lane_c + 8][T * 16 + 8 + 2 * lane_t]);
        const uint32_t a[4] = {a0, a1, a2, a3};
        mma_1688_f32(a, reinterpret_cast<const uint32_t*>(fb0), acc);
        mma_1688_f32(a, reinterpret_cast<const uint32_t*>(fb1), acc + 4);
      }
    }
  };

  // ---- STAGES-deep ring pipeline (v1's two-sync discipline, widened) ----
  // Prologue: fill the STAGES - 1 staging register sets back to back (the
  // global loads all issue before any store waits on them), then spill
  // them into slots 0..STAGES-2.
#pragma unroll
  for (int c = 0; c < RSETS; ++c) {
    if (c < nchunks) load_regs(c, c);
  }
#pragma unroll
  for (int c = 0; c < RSETS; ++c) {
    if (c < nchunks) store_smem(c, c);
  }
  __syncthreads();
  for (int ch = 0; ch < nchunks; ++ch) {
    // Issue global->register loads for chunk ch + STAGES - 1 (the ring's
    // prefetch distance, incumbent-style) into the register set the
    // store below is about to vacate: set(ch + STAGES - 1) = set(ch)
    // ... mod RSETS, and chunk ch's store already happened at iteration
    // ch - 1, so the set is free exactly when this load fires.
    if (ch + RSETS < nchunks)
      load_regs((ch + RSETS) % RSETS, ch + RSETS);
    compute(ch % STAGES_V);
    // All warps done with chunk ch. Slot (ch+1) % STAGES was last read by
    // compute(ch + 1 - STAGES), which finished before that iteration's
    // barriers — at least 2*(STAGES-1) barriers back — so it is free to
    // overwrite; slot ch % STAGES is never written while in use.
    __syncthreads();
    // Chunks 0..STAGES-2 were stored by the prologue; the loop stores
    // chunk ch+1 from the register set loaded for it at iteration
    // ch + 1 - RSETS.
    if (ch + 1 < nchunks && ch + 1 >= RSETS)
      store_smem((ch + 1) % RSETS, (ch + 1) % STAGES_V);
    __syncthreads();
  }

  // ---- epilogue: direct fp32 -> fp16 global write, fork k_opt style ----
  // m16n8 C fragment: c0,c1 = D[groupID][2t],[2t+1]; c2,c3 = row +8
  // (marlin_mma.h m16n8k8 D layout).
  const int row0 = m_base + lane_c;
  if constexpr (BN_V == 64) {
    // Warp w's columns: n_tile + 8w + 2t.
    const int col = n_tile + (warp << 3) + (lane_t << 1);
    if (row0 < M) {
      C[(long)row0 * N + col] = __float2half(acc[0]);
      C[(long)row0 * N + col + 1] = __float2half(acc[1]);
    }
    if (row0 + 8 < M) {
      C[(long)(row0 + 8) * N + col] = __float2half(acc[2]);
      C[(long)(row0 + 8) * N + col + 1] = __float2half(acc[3]);
    }
  } else {
    // Warp w's columns: n_tile + 64*(w>>2) + 16*(w&3) + 2t, and +8 for the
    // second fragment (dequant(word >> 8) -> column n0 + 8, same rows).
    const int col0 = n_tile + ((warp >> 2) << 6) + ((warp & 3) << 4) +
                     (lane_t << 1);
    if (row0 < M) {
      C[(long)row0 * N + col0] = __float2half(acc[0]);
      C[(long)row0 * N + col0 + 1] = __float2half(acc[1]);
      C[(long)row0 * N + col0 + 8] = __float2half(acc[4]);
      C[(long)row0 * N + col0 + 9] = __float2half(acc[5]);
    }
    if (row0 + 8 < M) {
      C[(long)(row0 + 8) * N + col0] = __float2half(acc[2]);
      C[(long)(row0 + 8) * N + col0 + 1] = __float2half(acc[3]);
      C[(long)(row0 + 8) * N + col0 + 8] = __float2half(acc[6]);
      C[(long)(row0 + 8) * N + col0 + 9] = __float2half(acc[7]);
    }
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
  TORCH_CHECK(N % TW4A16_BN == 0, "N must be a multiple of TW4A16_BN (",
              TW4A16_BN, "), got ", N);
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

  dim3 grid(N / TW4A16_BN, (M + BM - 1) / BM);
#if TW4A16_BN == 128
  auto kfn = turing_w4a16_regdeq_kernel<128, TW4A16_STAGES>;
#else
  auto kfn = turing_w4a16_regdeq_kernel<64, TW4A16_STAGES>;
#endif
  kfn<<<grid, THREADS, 0, at::cuda::getCurrentCUDAStream()>>>(A_p, Bp, Sp,
                                                              C_p, M, N, K);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("turing_w4a16_regdeq", &turing_w4a16_regdeq,
        "TU102 W4A16 GEMM, incumbent Marlin repacked layout, register lop3 "
        "dequant (checkpoint-compatible); v2: STAGES ring + BN=128 shape "
        "(compile-time TW4A16_BN / TW4A16_STAGES)");
}
