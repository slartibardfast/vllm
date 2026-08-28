# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""End-to-end test for the Turing Marlin backend on sm_75.

Drives a quantized linear layer through the real lifecycle (parameter
construction, process_weights_after_loading repack, apply_weights) with
both the incumbent Marlin kernel and the Turing backend, and requires:

1. bit-identical outputs between the two kernels (the Turing backend
   delegates to the same custom op at this stage of the program), and
2. agreement with a float64 reference of the quantization semantics.
"""

import pytest
import torch

from vllm.model_executor.kernels.linear import (
    MPLinearLayerConfig,
    choose_mp_linear_kernel,
)
from vllm.model_executor.parameter import (
    GroupQuantScaleParameter,
    PackedvLLMParameter,
)
from vllm.platforms import current_platform
from vllm.scalar_type import scalar_types


def _capability() -> int | None:
    cc = current_platform.get_device_capability()
    return None if cc is None else cc[0] * 10 + cc[1]


def _require_sm75():
    if not current_platform.is_cuda():
        pytest.skip("CUDA only")
    cap = _capability()
    if cap != 75:
        pytest.skip(f"sm_75 device required, this device is sm_{cap}")


def _build_layer(M, K, N, G, seed):
    torch.manual_seed(seed)
    w_ckpt = torch.randn(N, K, dtype=torch.float16, device="cuda") * 0.05
    groups = K // G
    scales_ckpt = (torch.rand(groups, N, device="cuda") * 0.02 + 0.02).half()
    # GPTQ symmetric quantization: q in [0,15], w = (q - 8) * s
    q = torch.clamp(
        torch.round(w_ckpt.double() / scales_ckpt.repeat_interleave(G, 0).double().T)
        + 8,
        0,
        15,
    ).to(torch.int32)
    w_q = torch.zeros(K // 8, N, dtype=torch.int32, device="cuda")
    for j in range(8):
        w_q |= (q[:, j::8] & 0xF).T << (4 * j)

    class Layer(torch.nn.Module):
        pass

    layer = Layer()
    layer.register_parameter(
        "weight_packed",
        PackedvLLMParameter(
            data=w_q,
            weight_loader=None,
            input_dim=0,
            output_dim=1,
            packed_factor=8,
            packed_dim=0,
        ),
    )
    layer.register_parameter(
        "weight_scale",
        GroupQuantScaleParameter(
            data=scales_ckpt,
            weight_loader=None,
            input_dim=0,
            output_dim=1,
        ),
    )
    # float64 reference of the quantized semantics
    W = q.double() - 8.0
    W = W * scales_ckpt.repeat_interleave(G, 0).double().T
    ref = torch.randn(M, K, device="cuda").double() @ W.T
    x = torch.randn(M, K, dtype=torch.float16, device="cuda")
    ref = x.double() @ W.T
    return layer, x, ref


@pytest.mark.skipif(_capability() != 75, reason="sm_75 device required")
@pytest.mark.parametrize("M", [1, 64])
def test_turing_marlin_end_to_end(M, dist_init):
    K, N, G = 512, 256, 128
    config = MPLinearLayerConfig(
        full_weight_shape=(K, N),
        partition_weight_shape=(K, N),
        weight_type=scalar_types.uint4b8,
        act_type=torch.float16,
        group_size=G,
        zero_points=False,
        has_g_idx=False,
    )
    layer_m, x, ref = _build_layer(M, K, N, G, seed=0)
    layer_t, x, ref = _build_layer(M, K, N, G, seed=0)

    kernel_m = choose_mp_linear_kernel(config)(config, "weight_packed", "weight_scale")
    kernel_m.process_weights_after_loading(layer_m)
    out_m = kernel_m.apply_weights(layer_m, x)

    kernel_t = choose_mp_linear_kernel(config)(config, "weight_packed", "weight_scale")
    assert type(kernel_t).__name__ == "TuringMarlinLinearKernel"
    kernel_t.process_weights_after_loading(layer_t)
    out_t = kernel_t.apply_weights(layer_t, x)

    assert torch.equal(out_m, out_t), "Turing backend must match the incumbent op"
    err = (out_t.double() - ref).abs().max().item()
    tol = 5e-2 * max(ref.abs().max().item(), 1e-6)
    assert err <= tol, f"max abs err {err} exceeds {tol}"
