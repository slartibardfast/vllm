// bridge_paged_decode.cu — C2 (window 3): paged decode attention, sm_75.
//
// Reads paged KV through the block table directly - no gather, no
// contiguous staging (the bridge arm's engine cost is the gather loop).
// Decode is bandwidth-bound, so the structure is token-centric: each KV
// token's K and V rows are staged to shared memory ONCE (one DRAM read
// per element), then all (q_row, head) pairs score against smem and
// accumulate PV into smem accumulators. FMA math by design; no tensor
// cores (our TRITON-at-the-floor result proves they are not needed).
//
// Graph-safety by construction: block_table/seq_lens are device tensors;
// nothing host-baked at capture (the C1 lesson).
//
// Layout (bridge_attn.py contract): kv_cache (blocks, h_kv, block_size,
// 2*d) fp16, K at [..., :d], V at [..., d:]; block_table (reqs,
// max_pages) int32; seq_lens (reqs,) int32; q (reqs, h_q, q_len, d)
// contiguous; out mirrors q. q_len in 1..QLMAX (K+1 for MTP); q_len>1
// uses bottom-right causal within the chunk.

#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <cuda_fp16.h>
#include <cstdint>

#define QLMAX 4
#define NTHREADS 256
#define NWARP (NTHREADS / 32)

__global__ void bridge_paged_decode_kernel(
    const __half* __restrict__ q, const __half* __restrict__ kv_cache,
    const int* __restrict__ block_table, const int* __restrict__ seq_lens,
    __half* __restrict__ out, int num_reqs, int h_q, int h_kv, int q_len,
    int d, int max_pages, int block_size, long sb, long sh, long st,
    float scale, float softcap) {
  const int req = blockIdx.x;
  const int kvh = blockIdx.y;
  if (req >= num_reqs) return;
  const int g = h_q / h_kv;           // q heads per kv head
  const int npair = q_len * g;        // (q row, local head) pairs
  const int seq_len = seq_lens[req];
  const int n_pages = (seq_len + block_size - 1) / block_size;
  const int tid = threadIdx.x;
  const int warp = tid >> 5, lane = tid & 31;

  extern __shared__ float smem[];
  __half* sK = reinterpret_cast<__half*>(smem);        // d halves
  __half* sV = sK + d;                                 // d halves
  float* sm_o = reinterpret_cast<float*>(sV + d);      // QLMAX*g*d
  float* sm_m = sm_o + QLMAX * g * d;                  // QLMAX*g
  float* sm_l = sm_m + QLMAX * g;                      // QLMAX*g
  float* sm_p = sm_l + QLMAX * g;                      // QLMAX*g (probs)
  int* sm_alive = reinterpret_cast<int*>(sm_p + QLMAX * g);  // CTA flag

  const __half* qbase = q + ((long)req * h_q + (long)kvh * g) * q_len * d;

  // init
  for (int i = tid; i < QLMAX * g; i += NTHREADS) {
    sm_m[i] = -INFINITY;
    sm_l[i] = 0.f;
  }
  for (int i = tid; i < QLMAX * g * d; i += NTHREADS) sm_o[i] = 0.f;
  __syncthreads();

  for (int p = 0; p < n_pages; p++) {
    const int b_id = block_table[req * max_pages + p];
    const __half* page = kv_cache + (long)b_id * sb + (long)kvh * sh;
    const int tok0 = p * block_size;
    for (int t = 0; t < block_size; t++) {
      const int pos = tok0 + t;
      if (pos >= seq_len) break;
      const __half* krow = page + (long)t * st;
      if (tid == 0) sm_alive[0] = 0;  // CTA-collective liveness
      // 1) stage K (and V lazily only if any pair survives the mask)
      for (int e = tid; e < d; e += NTHREADS) sK[e] = krow[e];
      __syncthreads();
      // 2) scores: one warp per (qr, local-head) pair, iterated
      bool any_alive = false;
      for (int pair = warp; pair < npair; pair += NWARP) {
        const int qr = pair / g, hl = pair % g;
        const int allowed = (pos <= seq_len - q_len + qr);
        if (!allowed) {
          if (lane == 0) sm_p[pair] = -INFINITY;  // sentinel: no weight
          continue;
        }
        any_alive = true;
        if (lane == 0) atomicOr(sm_alive, 1);
        const __half* qrow = qbase + ((long)hl * q_len + qr) * d;
        float acc = 0.f;
        // 32 lanes cover d elements (d=256 -> 8 per lane)
        for (int e = lane; e < d; e += 32)
          acc += __half2float(qrow[e]) * __half2float(sK[e]);
#pragma unroll
        for (int off = 16; off > 0; off >>= 1)
          acc += __shfl_down_sync(0xffffffffu, acc, off);
        if (lane == 0) {
          if (softcap > 0.f)
            acc = fminf(fmaxf(acc, -softcap), softcap);
          sm_p[pair] = acc * scale;
        }
      }
      __syncthreads();
      if (sm_alive[0] == 0) continue;  // masked for EVERY pair (CTA-wide)
      // 3) online softmax update per pair (one warp each, lane 0)
      for (int pair = warp; pair < npair; pair += NWARP) {
        if (lane != 0) continue;
        float sc = sm_p[pair];
        if (sc == -INFINITY) { sm_p[pair] = 0.f; continue; }
        float m_old = sm_m[pair];
        float m_new = fmaxf(m_old, sc);
        float alpha = (m_old == -INFINITY) ? 0.f : __expf(m_old - m_new);
        sm_m[pair] = m_new;
        sm_l[pair] = sm_l[pair] * alpha + __expf(sc - m_new);
        sm_p[pair] = __expf(sc - m_new);  // posterior weight for PV
        // rescale this pair's O slice (all d dims)
        float* orow = sm_o + pair * d;
        for (int e = 0; e < d; e++) orow[e] *= alpha;
      }
      __syncthreads();
      // 4) stage V and accumulate PV: thread-per-dim, all pairs
      const __half* vrow = krow + d;
      for (int e = tid; e < d; e += NTHREADS) sV[e] = vrow[e];
      __syncthreads();
      for (int e = tid; e < d; e += NTHREADS) {
        float v = __half2float(sV[e]);
        for (int pair = 0; pair < npair; pair++)
          sm_o[pair * d + e] += sm_p[pair] * v;
      }
      __syncthreads();
    }
  }
  // 5) normalize + write out (thread-per-dim per pair)
  for (int i = 0; i < npair; i++) {
    const int qr = i / g, hl = i % g;
    float l = sm_l[i];
    __half* orow = out + ((long)req * h_q + (long)kvh * g + hl) * q_len * d
                   + (long)qr * d;
    for (int e = tid; e < d; e += NTHREADS)
      orow[e] = __float2half(l > 0.f ? sm_o[i * d + e] / l : 0.f);
  }
}

torch::Tensor bridge_paged_decode(
    torch::Tensor q, torch::Tensor kv_cache, torch::Tensor block_table,
    torch::Tensor seq_lens, double scale, double softcap) {
  const int num_reqs = q.size(0), h_q = q.size(1), q_len = q.size(2),
            d = q.size(3);
  const int h_kv = kv_cache.size(1), block_size = kv_cache.size(2),
            max_pages = block_table.size(1);
  const int g = h_q / h_kv;
  TORCH_CHECK(d == 256, "paged decode v1: d must be 256");
  TORCH_CHECK(q_len >= 1 && q_len <= QLMAX, "q_len 1..4");
  TORCH_CHECK(h_q % h_kv == 0 && g * q_len <= 32 * NWARP,
              "pairs must fit the warp iterations");
  TORCH_CHECK((int)kv_cache.size(3) == 2 * d, "kv layout 2*d");
  TORCH_CHECK(kv_cache.scalar_type() == at::kHalf &&
              q.scalar_type() == at::kHalf, "fp16 only");
  auto out = torch::empty_like(q);
  dim3 grid(num_reqs, h_kv);
  size_t smem = (size_t)d * 2 * sizeof(__half) +
                (size_t)QLMAX * g * d * sizeof(float) +
                (size_t)3 * QLMAX * g * sizeof(float) +
                sizeof(int) * 4;  // sm_alive (+ pad)
  TORCH_CHECK(smem <= 48 * 1024, "smem over the static budget");
  auto stream = at::cuda::getCurrentCUDAStream();
  bridge_paged_decode_kernel<<<grid, NTHREADS, smem, stream>>>(
      reinterpret_cast<const __half*>(q.data_ptr<at::Half>()),
      reinterpret_cast<const __half*>(kv_cache.data_ptr<at::Half>()),
      block_table.data_ptr<int>(), seq_lens.data_ptr<int>(),
      reinterpret_cast<__half*>(out.data_ptr<at::Half>()), num_reqs, h_q,
      h_kv, q_len, d, max_pages, block_size, kv_cache.stride(0),
      kv_cache.stride(1), kv_cache.stride(2), (float)scale,
      (float)softcap);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return out;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("bridge_paged_decode", &bridge_paged_decode,
        "paged decode attention, sm_75 (window 3 C2 v1)");
}
