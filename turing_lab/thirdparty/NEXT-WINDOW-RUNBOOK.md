# Next-window runbook (prepared 2026-09-08, post-window recovery)

Everything below is turnkey. Order matters. Commands are exact.

# Next-window runbook (refreshed 2026-09-10, post window 4)

NO-WINDOW MODE IS IN FORCE (operator): CPU-advanceable work only.
GPU-BOUND items queue below; nothing runs them until a window opens.

## CPU-advanceable NOW (no GPU required)

1. Split-KV grid surgery (the headline item): extend
   turing_lab/bridge/bridge_paged_decode.cu - chunk the KV page walk
   across grid.z blocks (pages_per_block ~ 8-16), each block emits
   partial (m, l, o) to a workspace, a combine pass merges. Validate
   the DESIGN against the oracle contract (q_len 1..4, softcap, GQA);
   author the code; STANDALONE COMPILE via nvcc (CPU-only:
   nvcc -arch=sm_75 -c with the venv torch headers - window 4 proved
   this works without a GPU).
2. Paged-MTP q_len guard: extend the decode_shaped guard in
   bridge_attn.py's paged arm from q_len==1 to uniform q_len<=4 (the
   kernel already supports it; MTP verification steps carry K+1 rows).
   Code + py_compile only; gate-0 needs GPUs.

## GPU queue for the next window (in order)

0. Preflight (unchanged: turing_lab/thirdparty/preflight_window.sh).
1. Split-KV oracle battery (the 18-case suite in
   .weco/c2-paged-decode/oracle.py, extended with long-ctx cases
   >= 2048 exercising the split path).
2. Gate-0: paged-split vs triton, 8/8 bar (the triangulation pattern).
3. Champion A/B: bridge-paged-split vs bridge-gather, 3 pairs, THE
   ctx2048 ROW IS THE DECIDING ROW (the 4.1-vs-21.7 collapse).
4. MTP x paged: enable the q_len guard, gate-0, then the K2 arm on
   the 4B (the proven-compounding shape).
5. Close-out: ledger, records, clocks, llama-server, pushes.

## Superseded history

The window-2 era content below is superseded (probe/verdict/committed
runs all delivered; the committed numbers live in
turing_lab/results/committed-both-card*/). Kept for provenance.

## Open the window

    sudo -n systemctl stop llama-server
    sudo -n nvidia-smi -lgc 1455 -i 0
    sudo -n nvidia-smi -lgc 1455 -i 1
    bash turing_lab/thirdparty/preflight_window.sh

Preflight must print all OK before anything else. Its checks encode
every environment failure the last window paid for: GPUs free, clocks
locked, CUDA_HOME, the flashinfer-check bypass (no 0.6.18 cubin exists
upstream; no probe arm executes flashinfer kernels), the bridge d256
kernel smoke on BOTH model shapes (known-good control first - the
lesson of the retracted instantiation-gap diagnosis), fixtures present.

## Probe 1: the 4B same-dimension fixture

    cd turing_lab/thirdparty
    ../..//.venv/bin/python vllm_restart_variance.py

Defaults: 6 BRIDGE + 2 TRITON restarts on the 4B fixture, output to
turing_lab/results/restart-variance-20260907/. The driver now aborts
after 2 consecutive failed restarts (environment errors burn one
restart, never a batch). Context carried in the driver source and
REPORT.md: NCCL init/tuning logging per restart, the flashinfer bypass
reason, CUDA_HOME for worker JIT.

## Probe 2: the 27B verdict

    .venv/bin/python turing_lab/thirdparty/vllm_restart_variance.py \
        --model /opt/models/Intel-Qwen3.6-27B-int4-AutoRound \
        --outdir turing_lab/results/restart-variance-27b-$(date +%Y%m%d)

6 bridge restarts is the floor for the verdict (the recorded methodology:
medians across at least three; six gives the band). This is the number
that gates the committed run.

## The floor-anchored both-card committed run (the goal's deliverable)

Bar, derived from measured quantities only (the numbers rule):
- checkpoint on disk: 18 GiB (du, Intel-Qwen3.6-27B-int4-AutoRound);
  per card under TP2: 9 GiB
- measured decode bandwidth ceiling on this silicon: 535 GB/s (TU102
  paper, pure-read protocol)
- no-MTP floor: 9 GiB / 535 GB/s = 17.4 ms/token -> ~57 tok/s ceiling,
  minus the TP2 per-layer allreduce latency; record the measured
  allreduce share and publish the achieved/floor ratio with the number
- with-MTP band: floor x the measured acceptance band (upstream 27B
  MTP3 acceptance 0.75-0.90; weicj's Turing datapoint 1.39x e2e at K=3)
Protocol: median across >= 3 engine restarts, clocks locked, gates
green, PREEMPTED = failure (V1 recompute-only; zero-preemption
admission control per the KV-envelope finding). Commit the number with
the protocol transcript in this results tree and a plan/0005 README
amendment.

## Parallel track: weco ladder (GPU1 whenever a card is free)

    weco login   # operator, browser - THE ONLY outstanding blocker
    # then, from repo root:
    weco run \
      --sources .weco/w4a16-mregime/turing_search.cu \
                .weco/w4a16-mregime/turing_w4a16_opt.cu \
      --eval-command "bash .weco/w4a16-mregime/evaluate.sh" \
      --metric "SUMMARY metric" --goal maximize --steps 5 --output plain

Baseline seated: 23.50 TFLOP/s (opt2, M512, best-of-nz, 23/23 entries
oracle-clean). Adopt forked outputs only: any winner re-enters through
the ladder's full gates before landing in the lane. The committed
baseline table: turing_lab/results/w4a16-mregime-baseline/.

## Fixture reproduction recipe (only if the 4B needs re-making)

    uv venv ~/tools/autoround-venv --python 3.12
    uv pip install --python ~/tools/autoround-venv/bin/python \
        auto-round pillow torchvision
    CUDA_VISIBLE_DEVICES=0 ~/tools/autoround-venv/bin/auto-round quantize \
      --model_name /opt/models/Qwen3.5-4B --bits 4 --group_size 128 \
      --model_dtype float16 --scale_dtype fp16 --format auto_round \
      --iters 40 --nsamples 48 --seqlen 2048 --seed 42 \
      --output_dir /opt/models/Qwen3.5-4B-W4A16-AR-f16 --device_map cuda:0

Notes: the multimodal processor needs pillow+torchvision; full-grade
(iters 200 / nsamples 128) paced ~564 s/layer = 5 h and was stopped -
fixture grade is deliberate for engine-variance work. W4 sym g128 per
Intel's own 27B-int4 pattern; fp16 emission is the stage-10 recast
baked in.

## Close the window

    sudo -n nvidia-smi -rgc -i 0; sudo -n nvidia-smi -rgc -i 1
    sudo -n systemctl start llama-server
    curl -s http://127.0.0.1:8080/health   # {"status":"ok"}
Commit lane + outer, MEMORY entry, update this runbook's status.

## Provenance (recovery)

The vendored flash-attn tree is a local-only nested repo (lane
gitignores it; its origin is upstream Dao-AILab, unwritable). Its
campaign commits are archived as patches at
turing_lab/results/flash-attn-provenance/ - committed and pushed with
the lane. To rebuild the tree from scratch: clone upstream at v2.8.3
(060c918) and `git am` the patches in order.

## Pre-staged (2026-09-10, no-window authoring COMPLETE)

- `turing_lab/bridge/bridge_paged_decode_split.cu` — the split-KV
  kernel, AUTHORED and standalone-compiled clean (grid.z page chunks,
  partial m/l/o workspace, combine kernel; workspace is scratch, safe
  under graphs because combine reads only the splits the same replay's
  walk wrote).
- Oracle battery extended (`.weco/c2-paged-decode/oracle.py`):
  `SPLIT=1` routes to the split kernel; `LONGCTX=1` swaps in long
  sequences (64/4096 tokens, 300-page tables) that exercise the
  split path. Neither has RUN (no GPU) - both are first-window items.
- `bridge_attn.py` paged arm now accepts uniform q_len 1..4 (the MTP
  K+1 verification shape); py-compiled only - gate-0 is the first
  window item that exercises it.
- Window command list, in order: preflight -> `SPLIT=1` oracle (18
  cases) -> `SPLIT=1 LONGCTX=1` oracle (split path) -> gate-0
  (paged-split vs triton via a BRIDGE_PAGED_SPLIT env arm, to be
  added to the backend) -> champion A/B (ctx2048 row decides) ->
  MTP x paged (K2, 4B).
