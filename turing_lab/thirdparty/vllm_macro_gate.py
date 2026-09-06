"""vllm_macro_gate.py — the formal macro benchmark GATE for sm_75 A/B campaigns.

Run this at the END of every A/B campaign (kernel, backend, or engine
change). It executes the fixed protocol, compares against the recorded
baseline, and writes a versioned record. Exit 0 = gate green.

Protocol (frozen 2026-09-06; matches the recorded 2026-09-05 rows so
mixed numbers stay comparable):
  per backend in {BRIDGE_ATTN, TRITON_ATTN} x tp in {1, 2}:
    SHORT-DECODE   n_seq=4*tp seqs x 64 tokens (decode-dominated)
    per ctx in {512, 2048}:
      PREFILL-ONLY max_tokens=1  (ctx prompt tokens + 1 decode step)
      MIXED        max_tokens=32 (prompt + 32 decode steps)
      DECODE-PHASE 31 / (t_mixed - t_prefill), derived

Discipline: every arm runs under gpu-lease (tp1 -> gpu0 partition, tp2 ->
exclusive, NVLink/TP2 serializes); clocks attested 1455 MHz; temperature 0,
ignore_eos, enforce_eager; warmup excluded; prefix caching DISABLED (the
2026-09-06 ad-hoc run showed V1's default caching contaminating the
mixed/prefill split: triton "decode" read 120 tok/s at 4x context); every
row is the MEDIAN OF 3 REPEATS (single runs on this host have shown +-50%
same-protocol spread — the recorded 2026-09-05 baselines were single runs).

Presentation: records carry the industry layout. ppN = prefill
(prompt-processing) throughput in tokens/second at N context tokens;
tgN = decode (generation) throughput, batch 1, at N context tokens;
tg64x4 = batched-decode aggregate (4 sequences x 64 tokens, tp-scaled).
All values are tokens/second.

Gate rules:
  1. CONTROL (TRITON_ATTN touches neither the bridge kernel nor its
     headers): every row with a baseline must stay within +-20% of the
     baseline. A miss means the environment or baseline drifted -> the
     campaign is INVALID-ENV: fix the environment or re-seed the
     baseline, never credit the kernel.
  2. BRIDGE arm: no row with a baseline may regress by more than 5%
     (median vs baseline).
  3. Rows without a baseline (first run of a new row kind) are seeded,
     not judged; the next campaign is judged against them.

Usage:
  python vllm_macro_gate.py                 # parent: run everything, gate
  python vllm_macro_gate.py --arm BRIDGE_ATTN 1   # child: one arm
"""
import json
import re
import subprocess
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
LEASE = Path("/opt/repo/agentic-vllm/gpu-lease")
BASELINE = HERE / "vllm-macro-baseline.json"
CLOCK_MHZ = 1455

# Recorded 2026-09-05 (kernel 9dae9748): mixed rows only; prefill/decode
# split rows are seeded by the first formal run.
SEED_BASELINE = {
    "recorded": "2026-09-05",
    "kernel": "9dae9748",
    "rows": {
        "tp1:BRIDGE_ATTN:short_decode": 56.1,
        "tp1:BRIDGE_ATTN:ctx512_mixed": 30.1,
        "tp1:BRIDGE_ATTN:ctx2048_mixed": 28.0,
        "tp1:TRITON_ATTN:short_decode": 124.5,
        "tp1:TRITON_ATTN:ctx512_mixed": 42.6,
        "tp1:TRITON_ATTN:ctx2048_mixed": 28.0,
        "tp2:BRIDGE_ATTN:short_decode": 24.2,
        "tp2:BRIDGE_ATTN:ctx512_mixed": 19.0,
        "tp2:BRIDGE_ATTN:ctx2048_mixed": 14.4,
        "tp2:TRITON_ATTN:short_decode": 212.4,
        "tp2:TRITON_ATTN:ctx512_mixed": 41.4,
        "tp2:TRITON_ATTN:ctx2048_mixed": 33.1,
    },
}


def clocks_ok():
    out = subprocess.check_output(
        ["nvidia-smi", "--query-gpu=clocks.sm", "--format=csv,noheader"],
        text=True)
    mhz = [int(x.strip().split()[0]) for x in out.strip().splitlines()]
    ok = all(m == CLOCK_MHZ for m in mhz)
    print(f"clocks: {mhz} ({'OK' if ok else 'NOT LOCKED'})")
    return ok


def run_arm(backend, tp):
    from vllm import LLM, SamplingParams
    model = ("/home/dconnolly/.cache/huggingface/hub/models--Qwen--"
             "Qwen2.5-1.5B-Instruct/snapshots/"
             "989aa7980e4cf806f80c7fef2b1adb7bc71aa306")
    kw = dict(model=model, dtype="float16", tensor_parallel_size=tp,
              gpu_memory_utilization=0.90, max_model_len=4096,
              enforce_eager=True, attention_backend=backend,
              enable_prefix_caching=False)
    llm = LLM(**kw)
    llm.generate(["warmup"], SamplingParams(max_tokens=8, temperature=0.0,
                                            ignore_eos=True))

    def timed(prompt, max_tokens, n_seq=1):
        t0 = time.perf_counter()
        out = llm.generate([prompt] * n_seq,
                           SamplingParams(max_tokens=max_tokens,
                                          temperature=0.0, ignore_eos=True))
        return time.perf_counter() - t0, out

    tag = f"{backend}-tp{tp}"
    reps = {r: [] for r in
            ("short_decode", "ctx512_prefill", "ctx512_decode", "ctx512_mixed",
             "ctx2048_prefill", "ctx2048_decode", "ctx2048_mixed")}
    for _ in range(3):
        dt, out = timed("Explain flash attention in one paragraph.", 64, 4 * tp)
        toks = sum(len(o.outputs[0].token_ids) for o in out)
        reps["short_decode"].append(toks / dt)
        for ctx in (512, 2048):
            t_prefill, _ = timed("word " * ctx, 1)
            t_mixed, out = timed("word " * ctx, 32)
            toks = sum(len(o.outputs[0].token_ids) for o in out)
            reps[f"ctx{ctx}_prefill"].append(ctx / t_prefill)
            reps[f"ctx{ctx}_decode"].append(31.0 / (t_mixed - t_prefill))
            reps[f"ctx{ctx}_mixed"].append(toks / t_mixed)
    for row, vals in reps.items():
        vals.sort()
        print(f"SPLIT[{tag}] {row}={vals[1]:.1f} "
              f"(reps {vals[0]:.1f}/{vals[1]:.1f}/{vals[2]:.1f})", flush=True)
    print(f"SPLIT[{tag}] done", flush=True)


def parse_child(text, backend, tp):
    rows = {}
    for m in re.finditer(r"SPLIT\[([^\]]+)\] (\w+)=([0-9.]+)\s*"
                         r"\(reps ([0-9.]+)/([0-9.]+)/([0-9.]+)\)", text):
        rows[f"tp{tp}:{backend}:{m.group(2)}"] = float(m.group(3))
        rows[f"tp{tp}:{backend}:{m.group(2)}:spread"] = (
            float(m.group(4)), float(m.group(6)))
    return rows


def main():
    if len(sys.argv) > 2 and sys.argv[1] == "--arm":
        run_arm(sys.argv[2], int(sys.argv[3]))
        return

    assert clocks_ok(), "clocks are not locked; refusing to gate"

    baseline = json.loads(BASELINE.read_text()) if BASELINE.exists() \
        else SEED_BASELINE
    base_rows = baseline["rows"]

    results, child_texts = {}, {}
    for tp, mode in ((1, "gpu0"), (2, "exclusive")):
        for backend in ("BRIDGE_ATTN", "TRITON_ATTN"):
            cmd = [str(LEASE), "--mode", mode, "--wait", "1800", "--",
                   sys.executable, __file__, "--arm", backend, str(tp)]
            env_pass = {"CUDA_HOME": "/opt/cuda",
                        "FLASHINFER_DISABLE_VERSION_CHECK": "1",
                        "PATH": "/opt/cuda/bin:" + __import__("os").environ["PATH"]}
            print(f"=== arm {backend} tp{tp} (lease {mode}) ===", flush=True)
            proc = subprocess.run(cmd, capture_output=True, text=True,
                                  env={**__import__("os").environ, **env_pass})
            child_texts[f"{backend}-tp{tp}"] = proc.stdout[-4000:]
            results.update(parse_child(proc.stdout, backend, tp))
            if proc.returncode != 0:
                print(proc.stdout[-2000:], proc.stderr[-2000:])
                sys.exit(f"arm {backend} tp{tp} failed (rc={proc.returncode})")

    verdicts, md_rows = [], []
    failures = 0
    spreads = {k: v for k, v in results.items() if k.endswith(":spread")}
    results = {k: v for k, v in results.items() if not k.endswith(":spread")}
    for key, val in sorted(results.items()):
        b = base_rows.get(key)
        if b is None:
            verdicts.append((key, val, None, "SEEDED"))
            md_rows.append(f"| {key} | {val:.1f} | - | seeded |")
            continue
        ratio = val / b
        is_control = "TRITON" in key
        if is_control and not (0.80 <= ratio <= 1.20):
            v = "INVALID-ENV"
            failures += 1
        elif not is_control and ratio < 0.95:
            v = "FAIL"
            failures += 1
        else:
            v = "PASS" if ratio < 0.98 else "PASS+"
        verdicts.append((key, val, ratio, v))
        md_rows.append(f"| {key} | {b:.1f} | {val:.1f} | {ratio:.2f} | {v} |")

    stamp = time.strftime("%Y-%m-%d")
    json.dump({"date": stamp, "kernel": "69e91020",
               "rows": {k: v for k, v, _, _ in verdicts},
               "spreads": spreads,
               "baseline_used": baseline.get("recorded"),
               "verdicts": [(k, v) for k, _, _, v in verdicts]},
              open(HERE / f"vllm-macro-gate-{stamp}.json", "w"), indent=1)

    def r(backend, tp, row):
        return results.get(f"tp{tp}:{backend}:{row}")

    md = ["## Throughput (industry format; all values tokens/second, "
          "medians of 3)", "",
          "| backend | GPUs | pp512 t/s | pp2048 t/s | tg512 t/s | tg2048 t/s"
          " | tg64x4 batch t/s |", "|---|---|---|---|---|---|---|---|"]
    for backend in ("BRIDGE_ATTN", "TRITON_ATTN"):
        for tp in (1, 2):
            md.append(
                f"| {backend} | {tp} | {r(backend, tp, 'ctx512_prefill'):,.0f} "
                f"| {r(backend, tp, 'ctx2048_prefill'):,.0f} "
                f"| {r(backend, tp, 'ctx512_decode'):.1f} "
                f"| {r(backend, tp, 'ctx2048_decode'):.1f} "
                f"| {r(backend, tp, 'short_decode'):.1f} |")
    md += ["", "## Verdicts vs baseline", "",
           "| row | baseline | this run | ratio | verdict |",
           "|---|---|---|---|---|"] + md_rows
    (HERE / f"vllm-macro-gate-{stamp}.md").write_text("\n".join(md) + "\n")
    print("\n".join(md))
    print(f"\nMACRO GATE: {'GREEN' if failures == 0 else 'RED'} "
          f"({failures} failures, {sum(1 for *_, v in verdicts if v == 'SEEDED')} seeded)")
    sys.exit(0 if failures == 0 else 1)


if __name__ == "__main__":
    main()
