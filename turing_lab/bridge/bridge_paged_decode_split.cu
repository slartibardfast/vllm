// bridge_paged_decode_split.cu — split-KV surgery (window 5, CPU-authored).
//
// The v1 paged kernel (bridge_paged_decode.cu) walks all KV pages
// serially per CTA: at ctx2048 that is 128 pages x 4 barriers per
// token and it collapses (4.1 vs 21.7 tok/s, window 4 A/B). This
// variant chunks the page walk across grid.z blocks; each block
// computes partial online-softmax state (m, l, o) over its chunk and
// writes it to a workspace; a combine kernel merges the partials
// (m* = max m_s; o* = sum o_s exp(m_s - m*); l* likewise) and writes
// the final output. The per-token inner structure, masking, softcap,
// GQA mapping and graph-safety contract are identical to v1.
//
// Host policy: num_splits = ceil(n_pages / PAGES_PER_SPLIT), capped at
// MAX_SPLITS; one extra workspace tensor (fp32) is allocated per call
// by the host (the v1 kernel stays allocation-free; this variant
// accepts the allocation because the engine path frees per step —
// under CUDA graphs the allocation happens at capture only, which is
// sound because the workspace is scratch, fully overwritten per
// replay before combine reads it: combine reads exactly the splits
// the walk wrote in the same replay).

#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <cuda_fp16.h>
#include <cstdint>

#define QLMAX 4
#define NTHREADS 256
#define NWARP (NTHREADS / 32)
#define MAX_SPLITS 16
#define PAGES_PER_SPLIT 8

// phase 1: partial walk. grid = (reqs, h_kv, splits)
__global__ void bridge_paged_split_walk_kernel(
    const __half* __restrict__ q, const __half* __restrict__ kv_cache,
    const int* __restrict__ block_table, const int* __restrict__ seq_lens,
    float* __restrict__ ws_m, float* __restrict__ ws_l,
    float* __restrict__ ws_o, int num_reqs, int h_q, int h_kv, int q_len,
    int d, int max_pages, int block_size, long sb, long sh, long st,
    int splits, int pages_per_split, float scale, float softcap) {
  const int req = blockIdx.x;
  const int kvh = blockIdx.y;
  const int split = blockIdx.z;
  if (req >= num_reqs) return;
  const int g = h_q / h_kv;
  const int npair = q_len * g;
  const int seq_len = seq_lens[req];
  const int n_pages = (seq_len + block_size - 1) / block_size;
  const int p_begin = split * pages_per_split;
  const int p_end = min(n_pages, p_begin + pages_per_split);
  const int tid = threadIdx.x;
  const int warp = tid >> 5, lane = tid & 31;

  extern __shared__ float smem[];
  __half* sK = reinterpret_cast<__half*>(smem);
  __half* sV = sK + d;
  float* sm_o = reinterpret_cast<float*>(sV + d);
  float* sm_m = sm_o + QLMAX * g * d;
  float* sm_l = sm_m + QLMAX * g;
  float* sm_p = sm_l + QLMAX * g;
  int* sm_alive = reinterpret_cast<int*>(sm_p + QLMAX * g);

  const __half* qbase = q + ((long)req * h_q + (long)kvh * g) * q_len * d;

  for (int i = tid; i < QLMAX * g; i += NTHREADS) {
    sm_m[i] = -INFINITY;
    sm_l[i] = 0.f;
  }
  for (int i = tid; i < QLMAX * g * d; i += NTHREADS) sm_o[i] = 0.f;
  __syncthreads();

  for (int p = p_begin; p < p_end; p++) {
    const int b_id = block_table[req * max_pages + p];
    const __half* page = kv_cache + (long)b_id * sb + (long)kvh * sh;
    const int tok0 = p * block_size;
    for (int t = 0; t < block_size; t++) {
      const int pos = tok0 + t;
      if (pos >= seq_len) break;
      const __half* krow = page + (long)t * st;
      if (tid == 0) sm_alive[0] = 0;
      for (int e = tid; e < d; e += NTHREADS) sK[e] = krow[e];
      __syncthreads();
      for (int pair = warp; pair < npair; pair += NWARP) {
        const int qr = pair / g, hl = pair % g;
        const int allowed = (pos <= seq_len - q_len + qr);
        if (!allowed) {
          if (lane == 0) sm_p[pair] = -INFINITY;
          continue;
        }
        if (lane == 0) atomicOr(sm_alive, 1);
        const __half* qrow = qbase + ((long)hl * q_len + qr) * d;
        float acc = 0.f;
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
      if (sm_alive[0] == 0) continue;
      for (int pair = warp; pair < npair; pair += NWARP) {
        if (lane != 0) continue;
        float sc = sm_p[pair];
        if (sc == -INFINITY) { sm_p[pair] = 0.f; continue; }
        float m_old = sm_m[pair];
        float m_new = fmaxf(m_old, sc);
        float alpha = (m_old == -INFINITY) ? 0.f : __expf(m_old - m_new);
        sm_m[pair] = m_new;
        sm_l[pair] = sm_l[pair] * alpha + __expf(sc - m_new);
        sm_p[pair] = __expf(sc - m_new);
        float* orow = sm_o + pair * d;
        for (int e = 0; e < d; e++) orow[e] *= alpha;
      }
      __syncthreads();
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
  // emit partials to the workspace: [req][kvh][split][pair(, d)]
  const long pair_stride = (long)q_len * g;
  float* wm = ws_m + (((long)req * h_kv + kvh) * splits + split) * pair_stride;
  float* wl = ws_l + (((long)req * h_kv + kvh) * splits + split) * pair_stride;
  float* wo = ws_o +
              ((((long)req * h_kv + kvh) * splits + split) * pair_stride) * d;
  for (int i = 0; i < npair; i++) {
    wm[i] = sm_m[i];
    wl[i] = sm_l[i];
  }
  for (int i = tid; i < npair * d; i += NTHREADS) wo[i] = sm_o[i];
}

// phase 2: combine. grid = (reqs, h_kv); threads cover (pair, dim)
__global__ void bridge_paged_split_combine_kernel(
    const float* __restrict__ ws_m, const float* __restrict__ ws_l,
    const float* __restrict__ ws_o, const int* __restrict__ seq_lens,
    __half* __restrict__ out, int num_reqs, int h_q, int h_kv, int q_len,
    int d, int splits, int max_pages, int block_size, int pages_per_split) {
  const int req = blockIdx.x;
  const int kvh = blockIdx.y;
  if (req >= num_reqs) return;
  const int g = h_q / h_kv;
  const int npair = q_len * g;
  const int tid = threadIdx.x;
  const int seq_len = seq_lens[req];
  const int n_pages = (seq_len + block_size - 1) / block_size;
  const int active = min(splits,
                         (n_pages + pages_per_split - 1) / pages_per_split);

  const long pair_stride = (long)q_len * g;
  const float* wm = ws_m + (((long)req * h_kv + kvh) * splits) * pair_stride;
  const float* wl = ws_l + (((long)req * h_kv + kvh) * splits) * pair_stride;
  const float* wo = ws_o +
              (((long)req * h_kv + kvh) * splits) * pair_stride * d;

  extern __shared__ float csm[];  // m*, l*, per-split weights
  float* c_m = csm;               // QLMAX*g... sized q_len*g
  float* c_l = c_m + pair_stride;
  float* c_w = c_l + splits * 0 + pair_stride;  // per-split weight

  if (tid == 0) {
    for (int i = 0; i < npair; i++) {
      float mstar = -INFINITY;
      for (int s = 0; s < active; s++)
        mstar = fmaxf(mstar, wm[s * pair_stride + i]);
      c_m[i] = mstar;
      float lsum = 0.f;
      for (int s = 0; s < active; s++) {
        float w = (wm[s * pair_stride + i] == -INFINITY)
                      ? 0.f
                      : __expf(wm[s * pair_stride + i] - mstar);
        c_w[s * pair_stride + i] = w;
        lsum += wl[s * pair_stride + i] * w;
      }
      c_l[i] = lsum;
    }
  }
  __syncthreads();
  for (int i = 0; i < npair; i++) {
    const int qr = i / g, hl = i % g;
    __half* orow = out + ((long)req * h_q + (long)kvh * g + hl) * q_len * d
                   + (long)qr * d;
    for (int e = tid; e < d; e += NTHREADS) {
      float acc = 0.f;
      for (int s = 0; s < active; s++)
        acc += wo[(s * pair_stride + i) * d + e] *
               c_w[s * pair_stride + i];
      orow[e] = __float2half(c_l[i] > 0.f ? acc / c_l[i] : 0.f);
    }
  }
}

torch::Tensor bridge_paged_decode_split(
    torch::Tensor q, torch::Tensor kv_cache, torch::Tensor block_table,
    torch::Tensor seq_lens, double scale, double softcap) {
  const int num_reqs = q.size(0), h_q = q.size(1), q_len = q.size(2),
            d = q.size(3);
  const int h_kv = kv_cache.size(1), block_size = kv_cache.size(2),
            max_pages = block_table.size(1);
  const int g = h_q / h_kv;
  TORCH_CHECK(d == 256, "split v1: d must be 256");
  TORCH_CHECK(q_len >= 1 && q_len <= QLMAX, "q_len 1..4");
  TORCH_CHECK(h_q % h_kv == 0 && g * q_len <= 32 * NWARP,
              "pairs must fit the warp iterations");
  TORCH_CHECK((int)kv_cache.size(3) == 2 * d, "kv layout 2*d");
  const int pair_stride = q_len * g;
  // splits sized for the WORST request (shape-static under graphs)
  int splits = (max_pages + PAGES_PER_SPLIT - 1) / PAGES_PER_SPLIT;
  splits = std::min(splits, MAX_SPLITS);
  auto opts = q.options();
  auto ws_m = torch::empty({num_reqs, h_kv, splits, pair_stride},
                           opts.dtype(torch::kFloat));
  auto ws_l = torch::empty_like(ws_m);
  auto ws_o = torch::empty({num_reqs, h_kv, splits, pair_stride, d},
                           opts.dtype(torch::kFloat));
  auto out = torch::empty_like(q);
  dim3 grid(num_reqs, h_kv, splits);
  size_t smem = (size_t)d * 2 * sizeof(__half) +
                (size_t)QLMAX * g * d * sizeof(float) +
                (size_t)3 * QLMAX * g * sizeof(float) + sizeof(int) * 4;
  TORCH_CHECK(smem <= 48 * 1024, "smem over the static budget");
  auto stream = at::cuda::getCurrentCUDAStream();
  bridge_paged_split_walk_kernel<<<grid, NTHREADS, smem, stream>>>(
      reinterpret_cast<const __half*>(q.data_ptr<at::Half>()),
      reinterpret_cast<const __half*>(kv_cache.data_ptr<at::Half>()),
      block_table.data_ptr<int>(), seq_lens.data_ptr<int>(),
      ws_m.data_ptr<float>(), ws_l.data_ptr<float>(),
      ws_o.data_ptr<float>(), num_reqs, h_q, h_kv, q_len, d, max_pages,
      block_size, kv_cache.stride(0), kv_cache.stride(1),
      kv_cache.stride(2), splits, PAGES_PER_SPLIT, (float)scale,
      (float)softcap);
  size_t csmem = (size_t)(2 * pair_stride + splits * pair_stride) *
                 sizeof(float);
  dim3 cgrid(num_reqs, h_kv);
  bridge_paged_split_combine_kernel<<<cgrid, NTHREADS, csmem, stream>>>(
      ws_m.data_ptr<float>(), ws_l.data_ptr<float>(),
      ws_o.data_ptr<float>(), seq_lens.data_ptr<int>(),
      reinterpret_cast<__half*>(out.data_ptr<at::Half>()), num_reqs, h_q,
      h_kv, q_len, d, splits, max_pages, block_size, PAGES_PER_SPLIT);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return out;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("bridge_paged_decode_split", &bridge_paged_decode_split,
        "split-KV paged decode attention, sm_75 (window 5)");
}
