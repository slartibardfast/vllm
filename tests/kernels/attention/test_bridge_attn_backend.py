# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""BridgeAttentionBackend unit tests (plan/0006).

The impl is exercised directly against an fp64 bottom-right-causal
reference: paged-cache slot scatter (do_kv_cache_update), per-sequence
gather, and the flash-attn sm_75 route (padding, chunked prefill with
prefix, single-token decode), across head dims and GQA ratios. The
capability predicate keeps the backend sm_75-only."""

import pytest
import torch

from vllm.platforms.interface import DeviceCapability
from vllm.v1.attention.backend import AttentionType
from vllm.v1.attention.backends.bridge_attn import (
    BridgeAttentionBackend,
    BridgeAttentionImpl,
    BridgeAttentionMetadata,
)

BS = 16  # page size


def ref_attention(q, k, v, scale):
    """fp64 bottom-right causal reference. q: (P, H, D), k/v: (L, HKV, D)."""
    sq, hq, d = q.shape
    seq_len, hkv, _ = k.shape
    qf = q.double()
    kf = k.double().repeat_interleave(hq // hkv, dim=1)
    vf = v.double().repeat_interleave(hq // hkv, dim=1)
    scores = torch.einsum('phd,lhd->hpl', qf, kf) * scale
    qpos = torch.arange(sq, device=q.device).view(-1, 1) + (seq_len - sq)
    kpos = torch.arange(seq_len, device=q.device).view(1, -1)
    scores = scores.masked_fill(kpos > qpos, float('-inf'))
    return torch.einsum('hpl,lhd->phd', scores.softmax(-1), vf)


def make_impl(h_q, h_kv, d, scale=None):
    return BridgeAttentionImpl(
        num_heads=h_q, head_size=d, scale=scale or d ** -0.5,
        num_kv_heads=h_kv, kv_cache_dtype="auto",
        attn_type=AttentionType.DECODER)


def make_cache(num_blocks, h_kv, d, device):
    return torch.zeros(num_blocks, h_kv, BS, 2 * d,
                       dtype=torch.float16, device=device)


def alloc_pages(seq_len):
    return list(range((seq_len + BS - 1) // BS))


@pytest.mark.parametrize("d,h_q,h_kv", [(64, 4, 4), (64, 8, 2), (128, 8, 8),
                                                       (256, 24, 4),
                                        (128, 8, 1)])
@pytest.mark.parametrize("prompt_len", [77, 128, 301])
def test_prefill_then_decode(d, h_q, h_kv, prompt_len):
    if not torch.cuda.is_available():
        pytest.skip("cuda required")
    assert torch.cuda.get_device_capability() == (7, 5), "sm_75 bridge host"
    torch.manual_seed(0)
    dev = "cuda"
    impl = make_impl(h_q, h_kv, d)
    kv_cache = make_cache(64, h_kv, d, dev)
    block_table = torch.tensor([alloc_pages(prompt_len + 3)], device=dev)
    # slots for the whole prompt: page-major token order
    slots = torch.tensor([p * BS + o for p in block_table[0].tolist()
                          for o in range(BS)][:prompt_len], device=dev)

    key = torch.randn(prompt_len, h_kv, d, dtype=torch.float16, device=dev)
    value = torch.randn(prompt_len, h_kv, d, dtype=torch.float16, device=dev)
    impl.do_kv_cache_update(None, key, value, kv_cache, slots)

    # ---- prefill: the full prompt as one chunk ----
    meta = BridgeAttentionMetadata(
        num_reqs=1, num_actual_tokens=prompt_len, causal=True,
        block_table=block_table, query_start_loc_cpu=[0, prompt_len],
        seq_lens_cpu=[prompt_len])
    q = torch.randn(prompt_len, h_q, d, dtype=torch.float16, device=dev)
    out = torch.zeros(prompt_len, h_q, d, dtype=torch.float16, device=dev)
    impl.forward(None, q, None, None, kv_cache, meta, out)
    k_rows, v_rows = impl._gather_kv(kv_cache, block_table[0])
    err = (out.double()
           - ref_attention(q, k_rows[:prompt_len], v_rows[:prompt_len],
                           impl.scale)).abs().max().item()
    assert err < 0.02, f"prefill d={d} gqa={h_q // h_kv} len={prompt_len}: {err}"

    # ---- decode steps: one token appended per step ----
    for step in range(3):
        pos = prompt_len + step
        kt = torch.randn(1, h_kv, d, dtype=torch.float16, device=dev)
        vt = torch.randn(1, h_kv, d, dtype=torch.float16, device=dev)
        slot = torch.tensor([block_table[0, pos // BS].item() * BS
                             + pos % BS], device=dev)
        impl.do_kv_cache_update(None, kt, vt, kv_cache, slot)
        meta = BridgeAttentionMetadata(
            num_reqs=1, num_actual_tokens=1, causal=True,
            block_table=block_table, query_start_loc_cpu=[0, 1],
            seq_lens_cpu=[pos + 1])
        qt = torch.randn(1, h_q, d, dtype=torch.float16, device=dev)
        ot = torch.zeros(1, h_q, d, dtype=torch.float16, device=dev)
        impl.forward(None, qt, None, None, kv_cache, meta, ot)
        k_rows, v_rows = impl._gather_kv(kv_cache, block_table[0])
        err = (ot.double()
               - ref_attention(qt, k_rows[:pos + 1], v_rows[:pos + 1],
                               impl.scale)).abs().max().item()
        assert err < 0.02, f"decode d={d} gqa={h_q // h_kv} pos={pos}: {err}"


def test_chunked_prefill_with_prefix():
    """A 64-row chunk on top of a 100-token prefix: bottom-right causal
    with seqlen_q != seqlen_k."""
    if not torch.cuda.is_available():
        pytest.skip("cuda required")
    assert torch.cuda.get_device_capability() == (7, 5), "sm_75 bridge host"
    torch.manual_seed(1)
    d, h_q, h_kv, prefix, chunk = 64, 4, 2, 100, 64
    dev = "cuda"
    impl = make_impl(h_q, h_kv, d)
    total = prefix + chunk
    kv_cache = make_cache(32, h_kv, d, dev)
    block_table = torch.tensor([alloc_pages(total)], device=dev)
    slots = torch.tensor([p * BS + o for p in block_table[0].tolist()
                          for o in range(BS)][:total], device=dev)
    key = torch.randn(total, h_kv, d, dtype=torch.float16, device=dev)
    value = torch.randn(total, h_kv, d, dtype=torch.float16, device=dev)
    impl.do_kv_cache_update(None, key, value, kv_cache, slots)
    meta = BridgeAttentionMetadata(
        num_reqs=1, num_actual_tokens=chunk, causal=True,
        block_table=block_table, query_start_loc_cpu=[0, chunk],
        seq_lens_cpu=[total])
    q = torch.randn(chunk, h_q, d, dtype=torch.float16, device=dev)
    out = torch.zeros(chunk, h_q, d, dtype=torch.float16, device=dev)
    impl.forward(None, q, None, None, kv_cache, meta, out)
    err = (out.double() - ref_attention(q, key, value, impl.scale)
           ).abs().max().item()
    assert err < 0.02, f"chunked prefill: {err}"


def test_multi_request_batch():
    if not torch.cuda.is_available():
        pytest.skip("cuda required")
    assert torch.cuda.get_device_capability() == (7, 5), "sm_75 bridge host"
    torch.manual_seed(2)
    d, h_q, h_kv = 128, 8, 8
    lens = [77, 1, 130, 64]
    dev = "cuda"
    impl = make_impl(h_q, h_kv, d)
    kv_cache = make_cache(64, h_kv, d, dev)
    max_pages = max((l + BS - 1) // BS for l in lens)
    block_table = torch.tensor(
        [[p + i * 16 for p in alloc_pages(l)]
         + [0] * (max_pages - (l + BS - 1) // BS)
         for i, l in enumerate(lens)], device=dev)
    q_rows, out = [], torch.zeros(sum(lens), h_q, d,
                                  dtype=torch.float16, device=dev)
    slot_rows, key_rows, value_rows = [], [], []
    for i, l in enumerate(lens):
        q_rows.append(torch.randn(l, h_q, d, dtype=torch.float16, device=dev))
        key_rows.append(torch.randn(l, h_kv, d, dtype=torch.float16, device=dev))
        value_rows.append(torch.randn(l, h_kv, d, dtype=torch.float16,
                                      device=dev))
        for pos in range(l):
            slot_rows.append(block_table[i, pos // BS].item() * BS + pos % BS)
    impl.do_kv_cache_update(
        None, torch.cat(key_rows), torch.cat(value_rows), kv_cache,
        torch.tensor(slot_rows, device=dev))
    qsl = [0]
    for l in lens:
        qsl.append(qsl[-1] + l)
    meta = BridgeAttentionMetadata(
        num_reqs=len(lens), num_actual_tokens=sum(lens), causal=True,
        block_table=block_table, query_start_loc_cpu=qsl, seq_lens_cpu=lens)
    impl.forward(None, torch.cat(q_rows), None, None, kv_cache, meta, out)
    for i, l in enumerate(lens):
        k_rows, v_rows = impl._gather_kv(kv_cache, block_table[i])
        err = (out[qsl[i]:qsl[i + 1]].double()
               - ref_attention(q_rows[i], k_rows[:l], v_rows[:l],
                               impl.scale)).abs().max().item()
        assert err < 0.02, f"req {i} len {l}: {err}"


def test_capability_predicate():
    assert BridgeAttentionBackend.supports_compute_capability(
        DeviceCapability(7, 5))
    assert not BridgeAttentionBackend.supports_compute_capability(
        DeviceCapability(8, 0))
    assert not BridgeAttentionBackend.supports_compute_capability(
        DeviceCapability(7, 0))
    assert BridgeAttentionBackend.get_name() == "BRIDGE_ATTN"
