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


class DispatchTable:
    """The emitted auto-research artifact (turing_lab/search/schedule.py:
    emit_table): maps a runtime M to a lab-kernel variant and split-K
    factor, with boundaries resolved to the measured crossover and a
    fallback config for out-of-range or non-coverable shapes.

    Consumed only when VLLM_TURING_DISPATCH_TABLE points at a table
    file; absent the env var, the incumbent delegation is untouched.
    The lab kernel's weight layout (linear nibble-packed Q) is produced
    by the lab repack, which lands in process_weights_after_loading as
    the program's next step; until then the table rides along as the
    selection contract the swap will consume.
    """

    def __init__(self, data: dict):
        self.version = data.get("version")
        self.git = data.get("git")
        self.regimes = sorted(data.get("regimes", []), key=lambda r: r["m_lo"])
        self.fallback = data.get("fallback", {"index": 7, "nz": 1})
        if not self.regimes:
            raise ValueError("dispatch table has no regimes")

    @classmethod
    def load(cls, path: str) -> "DispatchTable":
        import json
        with open(path) as f:
            return cls(json.load(f))

    def select(self, m: int) -> tuple[int, int, str]:
        """Returns (variant index, split-K, variant name) for runtime M."""
        chosen = None
        for r in self.regimes:
            if r["m_lo"] <= m <= r["m_hi"]:
                chosen = r
                break
        if chosen is None:
            chosen = min(self.regimes, key=lambda r: abs(r["m_lo"] - m))
        return chosen["index"], chosen.get("nz", 1), chosen["variant"]

    def covers_shape(self, n: int, k: int, G: int = 128) -> bool:
        """The lab kernels require N a multiple of their BN tile; the
        emitted table records the constraint via the fallback. Shapes the
        lab kernels cannot cover fall back to the incumbent."""
        return n % 64 == 0 and k % 64 == 0 and G == 128
