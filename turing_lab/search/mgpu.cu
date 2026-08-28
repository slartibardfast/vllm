// mgpu.cu — peer-access control for the two-GPU transport experiments.
// P2P over the NVLink bridge requires cudaDeviceEnablePeerAccess on both
// sides; without it cudaMemcpyPeerAsync silently stages through the host
// (the PCIe x2 path). The python layer measures both paths explicitly.
#include <torch/extension.h>
#include <cstdint>

static bool g_enabled = false;

// Returns [can01, can10] from cudaDeviceCanAccessPeer (hardware/driver
// capability, independent of enabling).
std::vector<bool> can_access_peer() {
  int a = 0, b = 0;
  cudaDeviceCanAccessPeer(&a, 0, 1);
  cudaDeviceCanAccessPeer(&b, 1, 0);
  return {(bool)a, (bool)b};
}

// Idempotent: enabling twice returns cudaErrorPeerAccessAlreadyEnabled,
// which is success for our purposes.
bool enable_peer_access() {
  if (g_enabled) return true;
  cudaSetDevice(0);
  cudaError_t e0 = cudaDeviceEnablePeerAccess(1, 0);
  cudaSetDevice(1);
  cudaError_t e1 = cudaDeviceEnablePeerAccess(0, 0);
  g_enabled = (e0 == cudaSuccess || e0 == cudaErrorPeerAccessAlreadyEnabled)
           && (e1 == cudaSuccess || e1 == cudaErrorPeerAccessAlreadyEnabled);
  return g_enabled;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("can_access_peer", &can_access_peer);
  m.def("enable_peer_access", &enable_peer_access);
}
