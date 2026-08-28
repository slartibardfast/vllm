# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""DispatchTable selection tests for the Turing backend's auto-research
artifact (plan/0003)."""

import pytest

from vllm.model_executor.kernels.linear.mixed_precision.turing_marlin import (
    DispatchTable,
)

TABLE = {
    "version": 1,
    "git": "test",
    "regimes": [
        {"m_lo": 1, "m_hi": 8, "variant": "staged_64_64_64_w4x2",
         "index": 7, "nz": 1, "gate": "test"},
        {"m_lo": 9, "m_hi": 64, "variant": "regdeq_64_64_64_w4x1",
         "index": 8, "nz": 1, "gate": "test"},
        {"m_lo": 65, "m_hi": 192, "variant": "pipe_64_128_32_w4x2",
         "index": 18, "nz": 2, "gate": "test"},
        {"m_lo": 193, "m_hi": 4096, "variant": "staged_64_128_64_w4x2",
         "index": 2, "nz": 1, "gate": "test"},
    ],
    "boundaries": [{"m": 16, "hysteresis": 8}, {"m": 128, "hysteresis": 8}],
    "fallback": {"index": 7, "nz": 1},
}


def test_select_maps_runtime_m_to_regime_winner():
    dt = DispatchTable(TABLE)
    assert dt.select(1) == (7, 1, "staged_64_64_64_w4x2")
    assert dt.select(8) == (7, 1, "staged_64_64_64_w4x2")
    assert dt.select(32) == (8, 1, "regdeq_64_64_64_w4x1")
    assert dt.select(128) == (18, 2, "pipe_64_128_32_w4x2")
    assert dt.select(512) == (2, 1, "staged_64_128_64_w4x2")


def test_select_out_of_range_falls_to_nearest_regime():
    dt = DispatchTable(TABLE)
    idx, nz, name = dt.select(99999)
    assert (idx, nz, name) == (2, 1, "staged_64_128_64_w4x2")


def test_empty_table_rejected():
    with pytest.raises(ValueError, match="no regimes"):
        DispatchTable({"version": 1, "regimes": []})


def test_covers_shape_gate():
    dt = DispatchTable(TABLE)
    assert dt.covers_shape(4096, 4096)
    assert not dt.covers_shape(4100, 4096)
    assert not dt.covers_shape(4096, 4096, G=64)
