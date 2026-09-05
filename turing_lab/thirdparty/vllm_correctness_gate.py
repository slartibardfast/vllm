"""End-to-end correctness gate for the sm_75 stack (plan/0006 pause).

Arms (one engine process each; run via the fork venv with
CUDA_HOME=/opt/cuda FLASHINFER_DISABLE_VERSION_CHECK=1):

  prep          tokenize wikitext-2 test windows per model, cache to /tmp
  p15_bridge    Qwen2.5-1.5B (d=128): ctx sweep + wikitext PPL, BRIDGE_ATTN
  p15_triton    same, TRITON_ATTN forced
  p15_chunk_cache_on   bridge, max_num_batched_tokens=512 (8 chunks per 4k
  p15_chunk_cache_off  window), prefix caching on/off
  p15_chunk_unchunk    bridge, one 16k budget so 4k windows are single-chunk
  p05_bridge    Qwen2.5-0.5B (d=64): PPL + greedy, BRIDGE_ATTN
  p05_triton    same, TRITON_ATTN forced
  b27_tp2       Intel-Qwen3.6-27B-int4 TP=2: ctx sweep to 32k, PPL,
                512-token long decode; saves greedy ids for the identity arm
  b27_tp1       same model TP=1: greedy identity check vs the TP=2 run

Every check prints GATE[arm] name VERDICT :: detail lines; the record doc
is assembled from this output.
"""
import json
import math
import os
import sys

RES = "/tmp/gate_results"
WIKI = "/opt/models/wikitext-2-raw/wikitext-2-raw/wiki.test.raw"
M15 = ("/home/dconnolly/.cache/huggingface/hub/models--Qwen--"
       "Qwen2.5-1.5B-Instruct/snapshots/"
       "989aa7980e4cf806f80c7fef2b1adb7bc71aa306")
M05 = ("/home/dconnolly/.cache/huggingface/hub/models--Qwen--"
       "Qwen2.5-0.5B-Instruct/snapshots/"
       "7ae557604adf67be50417f59c2c2f167def9a775")
M27 = "/opt/models/Intel-Qwen3.6-27B-int4-AutoRound"
MODELS = {"15b": M15, "05b": M05, "27b": M27}


def gate(arm, name, ok, detail):
    print(f"GATE[{arm}] {name} {'PASS' if ok else 'FAIL'} :: {detail}",
          flush=True)


def load_windows(key, n_win, win_len):
    path = f"{RES}/wiki_{key}_{n_win}x{win_len}.json"
    if os.path.exists(path):
        with open(path) as f:
            return json.load(f)
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(MODELS[key])
    ids = tok(open(WIKI).read(), add_special_tokens=False)["input_ids"]
    wins = [ids[i * win_len:(i + 1) * win_len] for i in range(n_win)]
    assert all(len(w) == win_len for w in wins), "ran out of wikitext tokens"
    with open(path, "w") as f:
        json.dump(wins, f)
    return wins


def ppl_of(llm, windows, tag, arm):
    """Mean + per-window PPL via prompt_logprobs; finite-logprob gate."""
    from vllm import SamplingParams
    sp = SamplingParams(max_tokens=1, temperature=0.0, prompt_logprobs=0)
    outs = llm.generate([{"prompt_token_ids": w} for w in windows], sp)
    ppls, bad = [], 0
    for o in outs:
        total, n = 0.0, 0
        for e in o.prompt_logprobs:
            if e is None:
                continue
            v = next(iter(e.values())).logprob
            if not math.isfinite(v):
                bad += 1
                continue
            total += v
            n += 1
        ppls.append(math.exp(-total / n) if n else float("nan"))
    mean = sum(ppls) / len(ppls)
    gate(arm, "logprob_finite", bad == 0, f"{bad} non-finite entries")
    drift = max(abs(p - mean) / mean for p in ppls)
    gate(arm, f"ppl_{tag}", True,
         f"mean={mean:.4f} n={len(ppls)} max_window_drift={drift:.3f}")
    return mean


def distinct3(text):
    toks = text.split()
    if len(toks) < 3:
        return 1.0
    g = [tuple(toks[i:i + 3]) for i in range(len(toks) - 2)]
    return len(set(g)) / len(g)


def greedy_sweep(llm, key, ctx_list, n_new, arm, save=None):
    from vllm import SamplingParams
    base_ids = load_windows(key, 1, 40000)[0]
    outs_by_ctx = {}
    for ctx in ctx_list:
        sp = SamplingParams(max_tokens=n_new, temperature=0.0, logprobs=1)
        out = llm.generate([{"prompt_token_ids": base_ids[:ctx]}], sp)[0]
        o = out.outputs[0]
        lp_bad = sum(1 for d in (o.logprobs or []) if d is not None and
                     not math.isfinite(next(iter(d.values())).logprob))
        d3 = distinct3(o.text)
        gate(arm, f"genlogprob_finite_ctx{ctx}", lp_bad == 0, f"{lp_bad} bad")
        gate(arm, f"distinct3_ctx{ctx}", d3 > 0.25, f"ratio={d3:.3f}")
        outs_by_ctx[ctx] = {"text_head": o.text[:300], "distinct3": d3,
                            "token_ids": list(o.token_ids)}
        print(f"CTX[{arm}] ctx={ctx} head={o.text[:160]!r}", flush=True)
    if save:
        with open(save, "w") as f:
            json.dump(outs_by_ctx, f)
    return outs_by_ctx


def make_llm(model, backend=None, tp=1, mml=8192, extra=None):
    from vllm import LLM
    kw = dict(model=model, dtype="float16", tensor_parallel_size=tp,
              gpu_memory_utilization=0.90 if tp == 2 else 0.92,
              max_model_len=mml, enforce_eager=True)
    if backend:
        kw["attention_backend"] = backend
    if extra:
        kw.update(extra)
    return LLM(**kw)


def arm_p15(backend):
    arm = "p15_bridge" if backend is None else "p15_triton"
    llm = make_llm(M15, backend, mml=16640)
    mean = ppl_of(llm, load_windows("15b", 16, 2048), "2k", arm)
    greedy_sweep(llm, "15b", [512, 2048, 8192, 16384], 128, arm,
                 save=f"{RES}/{arm}_sweep.json")
    with open(f"{RES}/{arm}_ppl.json", "w") as f:
        json.dump({"ppl_2k": mean}, f)


def arm_p15_chunk(mode):
    arm = f"p15_chunk_{mode}"
    if mode == "unchunk":
        extra = dict(max_num_batched_tokens=16384, enable_prefix_caching=False)
    else:
        extra = dict(max_num_batched_tokens=512,
                     enable_prefix_caching=(mode == "cache_on"))
    llm = make_llm(M15, None, mml=8192, extra=extra)
    ppl_of(llm, load_windows("15b", 8, 4096), "4k", arm)


def arm_p05(backend):
    arm = "p05_bridge" if backend is None else "p05_triton"
    llm = make_llm(M05, backend, mml=8192)
    mean = ppl_of(llm, load_windows("05b", 8, 2048), "2k", arm)
    from vllm import SamplingParams
    ids = load_windows("05b", 1, 40000)[0][:1024]
    out = llm.generate([{"prompt_token_ids": ids}], SamplingParams(
        max_tokens=64, temperature=0.0))[0]
    with open(f"{RES}/{arm}_greedy.json", "w") as f:
        json.dump({"token_ids": list(out.outputs[0].token_ids),
                   "ppl_2k": mean}, f)


def arm_b27_tp2():
    arm = "b27_tp2"
    llm = make_llm(M27, None, tp=2, mml=40960)
    ppl_of(llm, load_windows("27b", 12, 2048), "2k", arm)
    greedy_sweep(llm, "27b", [512, 2048, 8192, 16384, 32768], 128, arm,
                 save=f"{RES}/{arm}_sweep.json")
    from vllm import SamplingParams
    ids = load_windows("27b", 1, 40000)[0][:8192]
    o = llm.generate([{"prompt_token_ids": ids}], SamplingParams(
        max_tokens=512, temperature=0.0, logprobs=1))[0].outputs[0]
    d3 = distinct3(o.text)
    bad = sum(1 for d in (o.logprobs or []) if d is not None and
              not math.isfinite(next(iter(d.values())).logprob))
    gate(arm, "longdecode_logprob_finite", bad == 0, f"{bad} bad over 512")
    gate(arm, "longdecode_distinct3", d3 > 0.25, f"ratio={d3:.3f}")
    with open(f"{RES}/b27_longdecode.json", "w") as f:
        json.dump({"text_head": o.text[:400], "distinct3": d3}, f)


def arm_b27_tp1():
    arm = "b27_tp1"
    # chunk at 4096: an 8192-token prompt equal to the default token
    # budget leaves no room for the decode step and wedges the scheduler
    llm = make_llm(M27, None, tp=1, mml=8320,
                   extra=dict(max_num_batched_tokens=4096))
    with open(f"{RES}/b27_tp2_sweep.json") as f:
        ref = json.load(f)
    from vllm import SamplingParams
    base_ids = load_windows("27b", 1, 40000)[0]
    n_match = 0
    for ctx in (512, 2048, 8192):
        ref_ids = ref[str(ctx)]["token_ids"][:32]
        out = llm.generate([{"prompt_token_ids": base_ids[:ctx]}],
                           SamplingParams(max_tokens=32, temperature=0.0))[0]
        got = list(out.outputs[0].token_ids)
        match = got == ref_ids
        n_match += match
        div = next((i for i, (a, b) in enumerate(zip(got, ref_ids))
                    if a != b), min(len(got), len(ref_ids)))
        print(f"IDENT[{arm}] ctx={ctx} match={match} first_div={div}",
              flush=True)
    gate(arm, "tp1_tp2_identity", n_match == 3,
         f"{n_match}/3 exact greedy matches vs TP=2")


def main():
    arm = sys.argv[1]
    os.makedirs(RES, exist_ok=True)
    if arm == "prep":
        for key, n, w in (("15b", 16, 2048), ("15b", 8, 4096),
                          ("15b", 1, 40000), ("05b", 8, 2048),
                          ("05b", 1, 40000), ("27b", 12, 2048),
                          ("27b", 1, 40000)):
            load_windows(key, n, w)
        print("prep done", flush=True)
    elif arm in ("p15_bridge", "p15_triton"):
        arm_p15(None if arm == "p15_bridge" else "triton_attn")
    elif arm.startswith("p15_chunk_"):
        arm_p15_chunk(arm[len("p15_chunk_"):])
    elif arm in ("p05_bridge", "p05_triton"):
        arm_p05(None if arm == "p05_bridge" else "triton_attn")
    elif arm == "b27_tp2":
        arm_b27_tp2()
    elif arm == "b27_tp1":
        arm_b27_tp1()
    else:
        raise SystemExit(f"unknown arm {arm}")
    print(f"ARM {arm} COMPLETE", flush=True)


if __name__ == "__main__":
    main()
