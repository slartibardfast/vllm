// driver.cu — the python-facing launcher for the search variants.
// Split-K (nz > 1) allocates the fp32 partial workspace here and runs the
// reduce pass; the kernel itself stays allocation-free.
#include <torch/extension.h>
#include <cuda_fp16.h>
#include <cstdint>
#include <c10/cuda/CUDAGuard.h>

extern "C" {
int variant_count();
const char* variant_name(int i);
void variant_launch(int i, const __half* A, const uint32_t* Q,
                    const __half* S, const int32_t* ZP, __half* Out,
                    float* Part, int M, int N, int K, int k_words, int G,
                    int nz, int fp32out, int* bm_out, int* bn_out,
                    int* bk_out, int* thr_out, int* strat_out);
void reduce_launch(const float* P, __half* Out, long total, int nz);
}

int variant_count_py() { return variant_count(); }

std::string variant_name_py(int i) { return variant_name(i); }

// Returns (Out, bm, bn, bk, threads, strategy) so the harness can check
// legality and record the winning config without re-parsing names.
std::vector<torch::Tensor> launch(int64_t i, torch::Tensor A, torch::Tensor Q,
                                  torch::Tensor S,
                                  c10::optional<torch::Tensor> ZP, int64_t G,
                                  int64_t nz) {
  int M = A.size(0), K = A.size(1), N = Q.size(0);
  auto Out = torch::empty({M, N}, A.options());
  int k_words = K / 8;
  // a zero-element ZP tensor is treated as absent: data_ptr() of an empty
  // tensor is not guaranteed null, and the kernel gates on pointer truth
  const int32_t* zp =
      (ZP.has_value() && ZP.value().numel() > 0)
          ? ZP.value().data_ptr<int32_t>()
          : nullptr;
  int bm, bn, bk, thr, strat;
  if (nz <= 1) {
    variant_launch((int)i,
                   reinterpret_cast<const __half*>(A.data_ptr<at::Half>()),
                   reinterpret_cast<const uint32_t*>(Q.data_ptr<int32_t>()),
                   reinterpret_cast<const __half*>(S.data_ptr<at::Half>()), zp,
                   reinterpret_cast<__half*>(Out.data_ptr<at::Half>()), nullptr,
                   M, N, K, k_words, (int)G, 1, 0, &bm, &bn, &bk, &thr,
                   &strat);
  } else {
    // empty is safe: every in-range cell of every slice is written, and
    // empty slices contribute exact zeros
    auto Part = torch::empty({(long)nz, (long)M, (long)N},
                             A.options().dtype(torch::kFloat));
    variant_launch((int)i,
                   reinterpret_cast<const __half*>(A.data_ptr<at::Half>()),
                   reinterpret_cast<const uint32_t*>(Q.data_ptr<int32_t>()),
                   reinterpret_cast<const __half*>(S.data_ptr<at::Half>()), zp,
                   reinterpret_cast<__half*>(Out.data_ptr<at::Half>()),
                   Part.data_ptr<float>(), M, N, K, k_words, (int)G, (int)nz,
                   0, &bm, &bn, &bk, &thr, &strat);
    reduce_launch(Part.data_ptr<float>(),
                  reinterpret_cast<__half*>(Out.data_ptr<at::Half>()),
                  (long)M * N, (int)nz);
  }
  auto opts = A.options().dtype(torch::kInt);
  return {Out,
          torch::tensor({bm, bn, bk, thr, strat}, opts)};
}

// Full-K pass writing the fp32 accumulator directly (no fp16 rounding):
// the building block for cross-GPU K-sharding with exact partials.
torch::Tensor launch_partial(int64_t i, torch::Tensor A, torch::Tensor Q,
                             torch::Tensor S,
                             c10::optional<torch::Tensor> ZP, int64_t G) {
  const c10::cuda::OptionalCUDAGuard guard(A.device());
  int M = A.size(0), K = A.size(1), N = Q.size(0);
  auto Part = torch::empty({M, N}, A.options().dtype(torch::kFloat));
  int k_words = K / 8;
  const int32_t* zp = ZP.has_value() ? ZP.value().data_ptr<int32_t>() : nullptr;
  int bm, bn, bk, thr, strat;
  variant_launch((int)i,
                 reinterpret_cast<const __half*>(A.data_ptr<at::Half>()),
                 reinterpret_cast<const uint32_t*>(Q.data_ptr<int32_t>()),
                 reinterpret_cast<const __half*>(S.data_ptr<at::Half>()), zp,
                 nullptr, Part.data_ptr<float>(), M, N, K, k_words, (int)G, 1,
                 1, &bm, &bn, &bk, &thr, &strat);
  return Part;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("variant_count", &variant_count_py);
  m.def("variant_name", &variant_name_py);
  m.def("launch", &launch,
        "launch search variant i (nz = split-K slice count)");
  m.def("launch_partial", &launch_partial,
        "full-K fp32 accumulation (no fp16 rounding)");
}
