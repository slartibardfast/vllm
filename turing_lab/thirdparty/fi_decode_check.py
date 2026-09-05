"""FlashInfer batch-decode correctness + throughput on sm_75 (plan/0006).

The paged-cache contract is (num_pages, 2, page_size, h, d); page_size=1
here, so one page per token. Correctness sweeps uniform and mixed kv
lengths on both kernel variants; throughput is median of 50 event-timed
calls (b=1 and b=8 series). Run on this host:

  CUDA_VISIBLE_DEVICES=1 CUDA_HOME=/opt/cuda \
  FLASHINFER_DISABLE_VERSION_CHECK=1 \
  .venv/bin/python turing_lab/thirdparty/fi_decode_check.py [b]
"""
import sys
import torch
import flashinfer
from statistics import median

h, d, page_size = 8, 128, 1
ws = torch.empty(256 * 1024 * 1024, dtype=torch.uint8, device="cuda")


def run_correctness(batch_kv, use_tensor_cores):
    b = len(batch_kv)
    q = torch.randn(b, h, d, dtype=torch.float16, device="cuda")
    kk = [torch.randn(l, 2, h, d, dtype=torch.float16, device="cuda")
          for l in batch_kv]
    kv = torch.cat(kk).unsqueeze(2).contiguous()  # (sum_l, 2, 1, h, d)
    indptr = torch.tensor([0] + list(torch.tensor(batch_kv).cumsum(0)),
                          dtype=torch.int32, device="cuda")
    indices = torch.arange(len(kv), dtype=torch.int32, device="cuda")
    last = torch.ones(b, dtype=torch.int32, device="cuda")
    w = flashinfer.BatchDecodeWithPagedKVCacheWrapper(
        ws, "NHD", use_tensor_cores=use_tensor_cores)
    w.plan(indptr, indices, last, h, h, d, page_size)
    out = w.forward(q, kv)
    torch.cuda.synchronize()
    err = 0.0
    for i, l in enumerate(batch_kv):
        s = torch.einsum('hd,lhd->hl', q[i].float(), kk[i][:, 0].float()) / d ** 0.5
        o = torch.einsum('hl,lhd->hd', s.softmax(-1), kk[i][:, 1].float())
        err = max(err, (out[i].float() - o).abs().max().item())
    print(f"b={b} kv={batch_kv} tensor_cores={use_tensor_cores}: "
          f"max_err={err:.5f}", flush=True)
    assert err < 2e-3, "decode oracle regression"


def run_throughput(b, kvlen):
    q = torch.randn(b, h, d, dtype=torch.float16, device="cuda")
    kv = torch.randn(b * kvlen, 2, 1, h, d, dtype=torch.float16, device="cuda")
    indptr = torch.arange(0, b * kvlen + 1, kvlen, dtype=torch.int32, device="cuda")
    indices = torch.arange(b * kvlen, dtype=torch.int32, device="cuda")
    last = torch.ones(b, dtype=torch.int32, device="cuda")
    w = flashinfer.BatchDecodeWithPagedKVCacheWrapper(ws, "NHD")
    w.plan(indptr, indices, last, h, h, d, 1)
    for _ in range(10):
        w.forward(q, kv)
    torch.cuda.synchronize()
    ts = []
    for _ in range(50):
        a, e = torch.cuda.Event(True), torch.cuda.Event(True)
        a.record(); w.forward(q, kv); e.record()
        torch.cuda.synchronize()
        ts.append(a.elapsed_time(e))
    med = median(sorted(ts))
    print(f"decode b={b} kv={kvlen}: median {med*1000:.1f} us"
          f" -> {b*1000.0/(med*1000):.1f} tok/s aggregate", flush=True)


if __name__ == "__main__":
    print("flashinfer", flashinfer.__version__, flush=True)
    torch.manual_seed(0)
    if len(sys.argv) > 1:  # minimal repro mode: python fi_decode_check.py 2
        b = int(sys.argv[1])
        run_correctness([256] * b, False)
    else:
        for tc in (False, True):
            run_correctness([256, 256], tc)
            run_correctness([256, 256, 256, 256], tc)
            run_correctness([1024, 512, 256, 128, 64, 32, 16, 8], tc)
            run_correctness([2048] * 16, tc)
        for b, kvlen in ((1, 1024), (1, 4096), (1, 16384), (8, 1024), (8, 4096)):
            run_throughput(b, kvlen)
