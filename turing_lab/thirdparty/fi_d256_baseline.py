"""FlashInfer d=256 baseline on sm_75 for plan/0007 step 0.1.

Decode AND prefill at the Qwen3.5/3.6/3.8-27B full-attention shape:
h_q=24, h_kv=4, head_dim 256, fp16, GQA 6:1, page_size=1. Correctness
against a torch reference first (assert), then event-timed throughput,
median of 3 repetitions. Every section is failure-tolerant: a launch
failure on this sm_75 stack is a recorded finding, not a crash.

Run under gpu-lease gpu0:
  gpu-lease --mode gpu0 --wait 600 -- env CUDA_HOME=/opt/cuda \
    FLASHINFER_DISABLE_VERSION_CHECK=1 \
    .venv/bin/python turing_lab/thirdparty/fi_d256_baseline.py
"""
import traceback

import torch
import flashinfer
from statistics import median

h_q, h_kv, d, page_size = 24, 4, 256, 1
ws = torch.empty(512 * 1024 * 1024, dtype=torch.uint8, device="cuda")


def ref_decode(q, kk):
    """q (b,h_q,d); kk list of (l,2,h_kv,d) -> reference per request."""
    outs = []
    g = h_q // h_kv
    for i in range(q.shape[0]):
        k, v = kk[i][:, 0].float(), kk[i][:, 1].float()
        kg = k.repeat_interleave(g, dim=1)
        vg = v.repeat_interleave(g, dim=1)
        s = torch.einsum('hd,lhd->hl', q[i].float(), kg) / d ** 0.5
        o = torch.einsum('hl,lhd->hd', s.softmax(-1), vg)
        outs.append(o)
    return torch.stack(outs)


def try_section(label, fn):
    try:
        fn()
    except Exception as e:
        print(f"FI256[{label}] FAILED: {type(e).__name__}: {e}", flush=True)
        traceback.print_exc(limit=3)


def decode_correctness(batch_kv):
    b = len(batch_kv)
    q = torch.randn(b, h_q, d, dtype=torch.float16, device="cuda")
    kk = [torch.randn(l, 2, h_kv, d, dtype=torch.float16, device="cuda")
          for l in batch_kv]
    kv = torch.cat(kk).unsqueeze(2).contiguous()
    indptr = torch.tensor([0] + list(torch.tensor(batch_kv).cumsum(0)),
                          dtype=torch.int32, device="cuda")
    indices = torch.arange(kv.shape[0], dtype=torch.int32, device="cuda")
    last = torch.ones(b, dtype=torch.int32, device="cuda")
    w = flashinfer.BatchDecodeWithPagedKVCacheWrapper(ws, "NHD")
    w.plan(indptr, indices, last, h_q, h_kv, d, page_size)
    out = w.forward(q, kv)
    torch.cuda.synchronize()
    err = 0.0
    ref = ref_decode(q, kk)
    for i in range(b):
        err = max(err, (out[i].float() - ref[i]).abs().max().item())
    print(f"FI256[decode-correct] b={b} kv={batch_kv}: max_err={err:.5f}",
          flush=True)
    assert err < 5e-3, "decode d256 oracle regression"


def decode_throughput(b, kvlen):
    q = torch.randn(b, h_q, d, dtype=torch.float16, device="cuda")
    kv = torch.randn(b * kvlen, 2, 1, h_kv, d, dtype=torch.float16,
                     device="cuda")
    indptr = torch.arange(0, b * kvlen + 1, kvlen, dtype=torch.int32,
                          device="cuda")
    indices = torch.arange(b * kvlen, dtype=torch.int32, device="cuda")
    last = torch.ones(b, dtype=torch.int32, device="cuda")
    w = flashinfer.BatchDecodeWithPagedKVCacheWrapper(ws, "NHD")
    w.plan(indptr, indices, last, h_q, h_kv, d, 1)
    meds = []
    for _ in range(3):
        for _ in range(10):
            w.forward(q, kv)
        torch.cuda.synchronize()
        ts = []
        for _ in range(30):
            a, e = torch.cuda.Event(True), torch.cuda.Event(True)
            a.record(); w.forward(q, kv); e.record()
            torch.cuda.synchronize()
            ts.append(a.elapsed_time(e))
        meds.append(median(sorted(ts)))
    m = median(meds)
    print(f"FI256[decode] b={b} kv={kvlen}: tg={b * 1000.0 / (m * 1000):.1f} "
          f"tok/s aggregate (median of 3, {m * 1000:.0f} us/step)",
          flush=True)


def prefill_correctness(ctx):
    q = torch.randn(ctx, h_q, d, dtype=torch.float16, device="cuda")
    kk = torch.randn(ctx, 2, h_kv, d, dtype=torch.float16, device="cuda")
    kv = kk.unsqueeze(2).contiguous()
    qo_indptr = torch.tensor([0, ctx], dtype=torch.int32, device="cuda")
    kv_indptr = torch.tensor([0, ctx], dtype=torch.int32, device="cuda")
    indices = torch.arange(ctx, dtype=torch.int32, device="cuda")
    last = torch.full((1,), ctx, dtype=torch.int32, device="cuda")
    w = flashinfer.BatchPrefillWithPagedKVCacheWrapper(ws, "NHD")
    w.plan(qo_indptr, kv_indptr, indices, last, h_q, h_kv, d, page_size,
           causal=True, q_data_type=torch.float16)
    out = w.forward(q, kv)
    torch.cuda.synchronize()
    k, v = kk[:, 0].float(), kk[:, 1].float()
    kg = k.repeat_interleave(h_q // h_kv, dim=1)
    vg = v.repeat_interleave(h_q // h_kv, dim=1)
    s = torch.einsum('qhd,khd->hqk', q.float(), kg) / d ** 0.5
    mask = torch.triu(torch.ones(ctx, ctx, dtype=torch.bool,
                                 device="cuda"), 1)
    s.masked_fill_(mask, float("-inf"))
    o = torch.einsum('hqk,khd->qhd', s.softmax(-1), vg)
    err = (out.float() - o).abs().max().item()
    print(f"FI256[prefill-correct] ctx={ctx}: max_err={err:.5f}", flush=True)
    assert err < 5e-3, "prefill d256 oracle regression"


def prefill_throughput(ctx):
    q = torch.randn(ctx, h_q, d, dtype=torch.float16, device="cuda")
    kv = torch.randn(ctx, 2, 1, h_kv, d, dtype=torch.float16, device="cuda")
    qo_indptr = torch.tensor([0, ctx], dtype=torch.int32, device="cuda")
    kv_indptr = torch.tensor([0, ctx], dtype=torch.int32, device="cuda")
    indices = torch.arange(ctx, dtype=torch.int32, device="cuda")
    last = torch.full((1,), ctx, dtype=torch.int32, device="cuda")
    w = flashinfer.BatchPrefillWithPagedKVCacheWrapper(ws, "NHD")
    w.plan(qo_indptr, kv_indptr, indices, last, h_q, h_kv, d, 1,
           causal=True, q_data_type=torch.float16)
    meds = []
    for _ in range(3):
        for _ in range(3):
            w.forward(q, kv)
        torch.cuda.synchronize()
        ts = []
        for _ in range(20):
            a, e = torch.cuda.Event(True), torch.cuda.Event(True)
            a.record(); w.forward(q, kv); e.record()
            torch.cuda.synchronize()
            ts.append(a.elapsed_time(e))
        meds.append(median(sorted(ts)))
    m = median(meds)
    print(f"FI256[prefill] ctx={ctx}: pp={ctx / (m / 1000):,.0f} tok/s "
          f"(median of 3, {m:.2f} ms)", flush=True)


if __name__ == "__main__":
    print("flashinfer", flashinfer.__version__, flush=True)
    torch.manual_seed(0)
    for kv in ([256, 256], [1024, 512, 256, 128, 64, 32, 16, 8], [2048] * 4):
        try_section("decode-correct", lambda kv=kv: decode_correctness(kv))
    for b, kvlen in ((1, 1024), (1, 4096), (1, 16384), (8, 1024)):
        try_section("decode-throughput",
                    lambda b=b, kvlen=kvlen: decode_throughput(b, kvlen))
    for ctx in (512, 2048):
        try_section("prefill-correct",
                    lambda ctx=ctx: prefill_correctness(ctx))
        try_section("prefill-throughput",
                    lambda ctx=ctx: prefill_throughput(ctx))
    print("FI256[DONE]", flush=True)
