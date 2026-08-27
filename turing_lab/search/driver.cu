// driver.cu — the python-facing launcher for the search variants.
#include <torch/extension.h>
#include <cuda_fp16.h>
#include <cstdint>

extern "C" {
int variant_count();
const char* variant_name(int i);
void variant_launch(int i, const __half* A, const uint32_t* Q,
                    const __half* S, const int32_t* ZP, __half* Out, int M,
                    int N, int K, int k_words, int G, int* bm_out,
                    int* bn_out, int* thr_out);
}

int variant_count_py() { return variant_count(); }

std::string variant_name_py(int i) { return variant_name(i); }

torch::Tensor launch(int64_t i, torch::Tensor A, torch::Tensor Q,
                     torch::Tensor S, c10::optional<torch::Tensor> ZP,
                     int64_t G) {
  int M = A.size(0), K = A.size(1), N = Q.size(0);
  auto Out = torch::empty({M, N}, A.options());
  int k_words = K / 8;
  const int32_t* zp = ZP.has_value() ? ZP.value().data_ptr<int32_t>() : nullptr;
  int bm, bn, thr;
  variant_launch((int)i,
                 reinterpret_cast<const __half*>(A.data_ptr<at::Half>()),
                 reinterpret_cast<const uint32_t*>(Q.data_ptr<int32_t>()),
                 reinterpret_cast<const __half*>(S.data_ptr<at::Half>()), zp,
                 reinterpret_cast<__half*>(Out.data_ptr<at::Half>()), M, N, K,
                 k_words, (int)G, &bm, &bn, &thr);
  return Out;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("variant_count", &variant_count_py);
  m.def("variant_name", &variant_name_py);
  m.def("launch", &launch, "launch search variant i");
}
