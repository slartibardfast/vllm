# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Bridge attention backend: sm_75 executes the cu_sm80_on_sm75 primitives.

plan/0006's thin dispatch adapter (design rule 3): vLLM is consumer #1 of
the bridge, and the quilt-routed upstream flash-attn supplies only its
python API. Every layer forward gathers the paged KV cache into
per-sequence contiguous tensors and calls the real ``flash_attn_func``
(the patched sm_75 route inside it executes the bridge-native kernel,
seqlen_q != seqlen_k bottom-right causal included). The KV-cache write
is the framework's standard ``do_kv_cache_update`` slot scatter.

Scope guard: capability (7,5) only, fp16, head_dim {64, 128}, causal
decoder attention, fp16/auto KV cache. Anything else declines here so
the selector falls through to the next backend (TRITON_ATTN)."""

from dataclasses import dataclass
from typing import ClassVar

import torch

from vllm.config import VllmConfig
from vllm.logger import init_logger
from vllm.v1.attention.backend import (
    AttentionBackend,
    AttentionImpl,
    AttentionLayer,
    AttentionMetadataBuilder,
    AttentionType,
    CommonAttentionMetadata,
    MultipleOf,
)
from vllm.v1.kv_cache_interface import AttentionSpec

logger = init_logger(__name__)
_route_logged = False


@dataclass
class BridgeAttentionMetadata:
    """Per-step metadata: the CPU-side lengths are materialized once in
    build() (one device sync per scheduler step, not per layer)."""

    num_reqs: int
    num_actual_tokens: int
    causal: bool | torch.Tensor
    block_table: torch.Tensor
    query_start_loc_cpu: list[int]
    seq_lens_cpu: list[int]


class BridgeAttentionMetadataBuilder(
    AttentionMetadataBuilder[BridgeAttentionMetadata]
):
    def __init__(
        self,
        kv_cache_spec: AttentionSpec,
        layer_names: list[str],
        vllm_config: VllmConfig,
        device: torch.device,
    ):
        super().__init__(kv_cache_spec, layer_names, vllm_config, device)

    def build_for_cudagraph_capture(
        self, common_attn_metadata: CommonAttentionMetadata
    ) -> BridgeAttentionMetadata:
        return self.build(0, common_attn_metadata)

    def build(
        self,
        common_prefix_len: int,
        common_attn_metadata: CommonAttentionMetadata,
        fast_build: bool = False,
    ) -> BridgeAttentionMetadata:
        return BridgeAttentionMetadata(
            num_reqs=common_attn_metadata.num_reqs,
            num_actual_tokens=common_attn_metadata.num_actual_tokens,
            causal=common_attn_metadata.causal,
            block_table=common_attn_metadata.block_table_tensor,
            query_start_loc_cpu=common_attn_metadata.query_start_loc_cpu.tolist(),
            seq_lens_cpu=common_attn_metadata.seq_lens.cpu().tolist(),
        )


class BridgeAttentionBackend(AttentionBackend):
    supported_dtypes: ClassVar[list[torch.dtype]] = [torch.float16]
    supported_kv_cache_dtypes: ClassVar[list[str]] = ["auto", "float16"]

    forward_includes_kv_cache_update: bool = False

    @staticmethod
    def get_name() -> str:
        return "BRIDGE_ATTN"

    @staticmethod
    def get_impl_cls() -> type["BridgeAttentionImpl"]:
        return BridgeAttentionImpl

    @staticmethod
    def get_builder_cls() -> type["BridgeAttentionMetadataBuilder"]:
        return BridgeAttentionMetadataBuilder

    @staticmethod
    def get_supported_kernel_block_sizes() -> list[int | MultipleOf]:
        return [16]

    @classmethod
    def supports_head_size(cls, head_size: int) -> bool:
        return head_size in (64, 128)

    @classmethod
    def supports_compute_capability(cls, capability) -> bool:
        return capability.major == 7 and capability.minor == 5

    @staticmethod
    def use_cascade_attention(*args, **kwargs) -> bool:
        return False


class BridgeAttentionImpl(AttentionImpl[BridgeAttentionMetadata]):
    def __init__(
        self,
        num_heads: int,
        head_size: int,
        scale: float,
        num_kv_heads: int | None = None,
        alibi_slopes: list[float] | None = None,
        sliding_window: int | None = None,
        kv_cache_dtype: str = "auto",
        logits_soft_cap: float | None = None,
        attn_type: AttentionType = AttentionType.DECODER,
        kv_sharing_target_layer_name: str | None = None,
        **kwargs,
    ) -> None:
        if alibi_slopes is not None:
            raise NotImplementedError("bridge_attn: alibi not supported")
        if sliding_window is not None:
            raise NotImplementedError("bridge_attn: sliding window not supported")
        if logits_soft_cap is not None:
            raise NotImplementedError("bridge_attn: softcap not supported")
        if attn_type != AttentionType.DECODER:
            raise NotImplementedError("bridge_attn: decoder attention only")
        if kv_cache_dtype not in ("auto", "float16"):
            raise NotImplementedError("bridge_attn: fp16 KV cache only")
        self.num_heads = num_heads
        self.head_size = head_size
        self.scale = float(scale)
        self.num_kv_heads = num_kv_heads or num_heads
        self.kv_cache_dtype = kv_cache_dtype
        self.kv_sharing_target_layer_name = kv_sharing_target_layer_name
        if num_heads % self.num_kv_heads:
            raise ValueError("bridge_attn: GQA ratio must divide")
        # the patched upstream package (flash-attn 2.8.3 + quilt); its
        # sm_75 route JIT-loads the bridge shim on first use
        from flash_attn import flash_attn_func as _fa

        self._flash_attn_func = _fa
        self._logged = False

    def _gather_kv(self, kv_cache: torch.Tensor, pages: torch.Tensor):
        """Pages -> K/V row views, (tokens, num_kv_heads, head_size)."""
        nkv = kv_cache.shape[1]
        hs = self.head_size
        # page layout is (heads, tokens, [K|V]) per page; attention wants
        # token-major rows
        toks = kv_cache[pages].permute(0, 2, 1, 3).reshape(-1, nkv, 2 * hs)
        return toks[:, :, :hs], toks[:, :, hs:]

    def do_kv_cache_update(
        self,
        layer: AttentionLayer,
        key: torch.Tensor,
        value: torch.Tensor,
        kv_cache: torch.Tensor,
        slot_mapping: torch.Tensor,
    ):
        # logical (blocks, kv_heads, block_size, [K|V]) view; slots are
        # block_id * block_size + offset on the logical block axis
        bs = kv_cache.shape[2]
        hs = self.head_size
        b_idx = torch.div(slot_mapping, bs, rounding_mode="floor")
        o_idx = slot_mapping % bs
        kv_cache[b_idx, :, o_idx, :hs] = key
        kv_cache[b_idx, :, o_idx, hs:] = value

    def forward(
        self,
        layer: AttentionLayer,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        kv_cache: torch.Tensor,
        attn_metadata: BridgeAttentionMetadata,
        output: torch.Tensor,
        output_scale: torch.Tensor | None = None,
        output_block_scale: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if attn_metadata is None:
            # Profiling run.
            return output.fill_(0)
        if output_block_scale is not None:
            raise NotImplementedError(
                "bridge_attn: fused block-scale output not supported"
            )
        global _route_logged
        if not _route_logged:
            _route_logged = True
            if attn_metadata.causal is not True:
                raise NotImplementedError("bridge_attn: causal attention only")
            logger.info(
                "bridge_attn: attention executing on the cu_sm80_on_sm75 "
                "bridge kernel via flash-attn's sm_75 route"
            )
        fa = self._flash_attn_func
        qsl = attn_metadata.query_start_loc_cpu
        seq_lens = attn_metadata.seq_lens_cpu
        block_table = attn_metadata.block_table
        block_size = kv_cache.shape[2]
        nat = attn_metadata.num_actual_tokens
        for i in range(attn_metadata.num_reqs):
            p0, p1 = qsl[i], qsl[i + 1]
            if p1 <= p0 or p0 >= nat:
                continue
            p1 = min(p1, nat)
            seq_len = seq_lens[i]
            n_pages = (seq_len + block_size - 1) // block_size
            pages = block_table[i, :n_pages]
            k_rows, v_rows = self._gather_kv(kv_cache, pages)
            # bottom-right causal: query block is the chunk's own rows,
            # FA2's seqlen_q != seqlen_k contract covers prefix + chunk
            # and the single-token decode alike
            out = fa(
                query[p0:p1].unsqueeze(0), k_rows[:seq_len].unsqueeze(0),
                v_rows[:seq_len].unsqueeze(0), softmax_scale=self.scale,
                causal=True,
            )
            output[p0:p1].copy_(out.squeeze(0))
        return output
