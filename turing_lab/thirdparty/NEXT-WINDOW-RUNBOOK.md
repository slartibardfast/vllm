# Next-window runbook (prepared 2026-09-08, post-window recovery)

Everything below is turnkey. Order matters. Commands are exact.

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
