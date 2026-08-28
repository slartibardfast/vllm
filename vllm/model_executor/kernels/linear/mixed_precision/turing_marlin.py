# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""The Turing (sm_75) Marlin backend: the integration point of the
agentic-vllm Turing Marlin program (plan/0002 on the governing host).

Selection contract: wins kernel selection on sm_75 exactly, and falls
through to :class:`MarlinLinearKernel` everywhere else, so sm_80+ dispatch
is untouched. Weight processing and compute currently delegate to the
incumbent implementation; the optimized Turing kernel replaces the custom
op behind this same class as the program lands it (the weight-repack swap
point is `process_weights_after_loading`).

See the governing host repository (agentic-vllm, plan/0002) for the
program's characterization, model, reference kernel, and search results.
"""

from vllm.model_executor.kernels.linear.mixed_precision.marlin import (
    MarlinLinearKernel,
)
from vllm.model_executor.kernels.linear.mixed_precision.MPLinearKernel import (
    MPLinearLayerConfig,
)
from vllm.model_executor.layers.quantization.utils.marlin_utils import (
    query_marlin_supported_quant_types,
)
from vllm.platforms import current_platform
from vllm.scalar_type import ScalarType


class TuringMarlinLinearKernel(MarlinLinearKernel):
    @classmethod
    def get_min_capability(cls) -> int:
        return 75

    @classmethod
    def can_implement(cls, c: MPLinearLayerConfig) -> tuple[bool, str | None]:
        # The backend is gated to its own architecture: any other device
        # falls through to the generic Marlin kernel.
        if not current_platform.is_cuda():
            return False, "Turing Marlin only supported on CUDA"
        cc = current_platform.get_device_capability()
        if cc != (7, 5):
            return (
                False,
                f"Turing Marlin requires sm_75, device is sm_{cc[0]}{cc[1]}"
                if cc
                else "Turing Marlin requires sm_75, device capability unknown",
            )

        quant_types = query_marlin_supported_quant_types(c.zero_points)
        weight_type: ScalarType = c.weight_type
        if weight_type not in quant_types:
            return (
                False,
                (
                    f"Quant type ({weight_type}) not supported by"
                    f"  Turing Marlin, supported types are: {quant_types}"
                ),
            )
        return True, None
