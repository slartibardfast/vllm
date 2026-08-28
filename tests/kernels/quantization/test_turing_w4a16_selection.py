# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Selection tests for the Turing Marlin backend (sm_75).

On a TU102 (capability 7.5) device the Turing backend must win W4A16
selection; on any other capability it must fall through to the generic
Marlin kernel, leaving sm_80+ dispatch untouched.
"""

import pytest
import torch

from vllm.model_executor.kernels.linear import (
    MPLinearLayerConfig,
    choose_mp_linear_kernel,
)
from vllm.platforms import current_platform
from vllm.scalar_type import scalar_types


def _w4a16_config() -> MPLinearLayerConfig:
    K, N = 4096, 4096
    return MPLinearLayerConfig(
        full_weight_shape=(K, N),
        partition_weight_shape=(K, N),
        weight_type=scalar_types.uint4b8,
        act_type=torch.float16,
        group_size=128,
        zero_points=False,
        has_g_idx=False,
    )


def _capability() -> int | None:
    cc = current_platform.get_device_capability()
    return None if cc is None else cc[0] * 10 + cc[1]


def _require_sm75():
    if not current_platform.is_cuda():
        pytest.skip("CUDA only")
    cap = _capability()
    if cap is None:
        pytest.skip("device capability unavailable")
    if cap != 75:
        pytest.skip(f"sm_75 device required, this device is sm_{cap}")


def test_turing_backend_wins_on_sm75():
    _require_sm75()
    kernel = choose_mp_linear_kernel(_w4a16_config(), compute_capability=75)
    assert kernel.__name__ == "TuringMarlinLinearKernel"


def test_turing_backend_selected_by_default_on_this_device():
    _require_sm75()
    kernel = choose_mp_linear_kernel(_w4a16_config())
    assert kernel.__name__ == "TuringMarlinLinearKernel"


def _capability_of(platform_capability):
    """The backend's gate under a simulated device capability.

    Fall-through is asserted at the gate, not through
    choose_mp_linear_kernel(compute_capability=...): that argument only
    feeds the min-capability filter, while every kernel's can_implement
    queries the live platform, so a simulated capability cannot steer
    selection on a single-GPU host.
    """
    from vllm.model_executor.kernels.linear.mixed_precision.turing_marlin import (
        TuringMarlinLinearKernel,
    )
    from vllm.platforms import current_platform

    original = current_platform.get_device_capability
    current_platform.get_device_capability = lambda: platform_capability
    try:
        ok, reason = TuringMarlinLinearKernel.can_implement(_w4a16_config())
    finally:
        current_platform.get_device_capability = original
    return ok, reason


@pytest.mark.skipif(_capability() != 75, reason="exercises the gate on sm_75")
def test_turing_backend_gate_rejects_sm80_and_sm86():
    for cc in [(8, 0), (8, 6), (9, 0)]:
        ok, reason = _capability_of(cc)
        assert not ok, f"must reject sm_{cc[0]}{cc[1]}"
        assert "sm_75" in reason


@pytest.mark.skipif(_capability() != 75, reason="exercises the gate on sm_75")
def test_turing_backend_gate_rejects_unknown_capability():
    ok, reason = _capability_of(None)
    assert not ok
    assert "sm_75" in reason
