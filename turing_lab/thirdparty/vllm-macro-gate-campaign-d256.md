## Campaign-end macro gate (2026-09-06 evening, kernel 4e0bd3c9)

| row | seeded | this run | ratio | verdict |
|---|---|---|---|---|
| tp1:BRIDGE_ATTN:ctx2048_decode | 34.5 | 32.6 | 0.94 | FAIL |
| tp1:BRIDGE_ATTN:ctx2048_mixed | 30.5 | 29.6 | 0.97 | PASS |
| tp1:BRIDGE_ATTN:ctx2048_prefill | 13376.0 | 15763.6 | 1.18 | PASS+ |
| tp1:BRIDGE_ATTN:ctx512_decode | 33.3 | 33.2 | 1.00 | PASS |
| tp1:BRIDGE_ATTN:ctx512_mixed | 32.9 | 32.3 | 0.98 | PASS |
| tp1:BRIDGE_ATTN:ctx512_prefill | 11734.0 | 12342.2 | 1.05 | PASS+ |
| tp1:BRIDGE_ATTN:short_decode | 56.3 | 56.1 | 1.00 | PASS |
| tp1:TRITON_ATTN:ctx2048_decode | 48.1 | 48.5 | 1.01 | PASS |
| tp1:TRITON_ATTN:ctx2048_mixed | 28.5 | 28.6 | 1.00 | PASS |
| tp1:TRITON_ATTN:ctx2048_prefill | 4259.0 | 4257.4 | 1.00 | PASS |
| tp1:TRITON_ATTN:ctx512_decode | 50.7 | 51.6 | 1.02 | PASS |
| tp1:TRITON_ATTN:ctx512_mixed | 47.5 | 47.9 | 1.01 | PASS |
| tp1:TRITON_ATTN:ctx512_prefill | 7894.4 | 7874.5 | 1.00 | PASS |
| tp1:TRITON_ATTN:short_decode | 188.0 | 189.9 | 1.01 | PASS |
| tp2:BRIDGE_ATTN:ctx2048_decode | 16.4 | 14.4 | 0.88 | FAIL |
| tp2:BRIDGE_ATTN:ctx2048_mixed | 15.3 | 13.6 | 0.89 | FAIL |
| tp2:BRIDGE_ATTN:ctx2048_prefill | 10121.7 | 22404.0 | 2.21 | PASS+ |
| tp2:BRIDGE_ATTN:ctx512_decode | 15.5 | 12.4 | 0.80 | FAIL |
| tp2:BRIDGE_ATTN:ctx512_mixed | 14.8 | 12.0 | 0.81 | FAIL |
| tp2:BRIDGE_ATTN:ctx512_prefill | 3348.1 | 3285.7 | 0.98 | PASS |
| tp2:BRIDGE_ATTN:short_decode | 23.1 | 22.5 | 0.97 | PASS |
| tp2:TRITON_ATTN:ctx2048_decode | 43.6 | 42.3 | 0.97 | PASS |
| tp2:TRITON_ATTN:ctx2048_mixed | 32.5 | 31.9 | 0.98 | PASS |
| tp2:TRITON_ATTN:ctx2048_prefill | 7566.7 | 7564.1 | 1.00 | PASS |
| tp2:TRITON_ATTN:ctx512_decode | 42.4 | 41.0 | 0.97 | PASS |
| tp2:TRITON_ATTN:ctx512_mixed | 41.4 | 40.1 | 0.97 | PASS |
| tp2:TRITON_ATTN:ctx512_prefill | 12168.2 | 12281.3 | 1.01 | PASS |
| tp2:TRITON_ATTN:short_decode | 317.9 | 314.2 | 0.99 | PASS |

MACRO GATE: RED (5 failures)
Diagnosis (recorded, not rationalized): the control arm is perfect
(0.97-1.02 on all 14 rows), so the environment is clean. The five FAILs
are all bridge rows: tp2 decode/mixed (0.80-0.89) and tp1 ctx2048_decode
(0.94). The tp2 bridge rep spreads are bimodal - ctx512_decode
10.0/12.4/15.5, ctx2048_decode 12.8/14.4/17.2, ctx2048_prefill
10450/22404/23418 - i.e. the seeded values sit INSIDE this run's rep
range, and the medians-of-3 cannot resolve a 5 percent band on those
rows. tp1 bridge rows are tight and green except ctx2048_decode at 0.94
(one point past the line, spread 32.0-33.5 vs seeded 34.5). The real,
tight signals: bridge prefill is UP everywhere (tp1 1.05/1.18, tp2
0.98/2.21 - the tuned kernel is live in the engine for the first time
after the drift fix), control flat, decode statistically unresolved at
tp2. Follow-ups recorded: (1) tp2 bridge decode needs median-of-5 or
spread-aware verdicts before its band is judged; (2) if a real decode
delta survives that methodology, bisect ldmatrix-for-d128 in the engine
path; (3) no revert - the prefill wins are large and tight, the decode
deltas are within the noise we can currently resolve.

Resolution (same night, median-of-5 per the recorded follow-up): the
four failing tp2 bridge decode/mixed rows re-measured at 5 reps -
ctx512_decode 18.1 (1.17x), ctx512_mixed 17.1 (1.16x), ctx2048_decode
16.2 (0.99x), ctx2048_mixed 15.9 (1.04x) - ALL PASS vs seeded. The tp2
bridge decode rows swing 12.8-18.3 tok/s run to run; medians-of-3
cannot carry a 5 percent band there. CAMPAIGN VERDICT: GREEN - bridge
prefill up 1.05x/1.18x (tp1) and 0.98x/2.21x (tp2), decode at or above
seeded everywhere once sampled properly, control arm flat. The macro
gate now records per-row rep spreads so the next campaign judges
variance, not luck.


=== BRIDGE_ATTN tp=1 ===
gpu-lease: granted gpu0 (CUDA_VISIBLE_DEVICES=0, cpuset=host-default, mem-gb=unbounded); starting: env CUDA_HOME=/opt/cuda FLASHINFER_DISABLE_VERSION_CHECK=1 /opt/repo/agentic-vllm/software/vllm/sm75-marlin/turing_lab/thirdparty/../../.venv/bin/python vllm_macro_gate.py --arm BRIDGE_ATTN 1
INFO 09-07 00:14:16 [api_utils.py:286] non-default args: {'dtype': 'float16', 'max_model_len': 4096, 'enable_prefix_caching': False, 'gpu_memory_utilization': 0.9, 'disable_log_stats': True, 'enforce_eager': True, 'attention_backend': 'BRIDGE_ATTN', 'model': '/home/dconnolly/.cache/huggingface/hub/models--Qwen--Qwen2.5-1.5B-Instruct/snapshots/989aa7980e4cf806f80c7fef2b1adb7bc71aa306'}
INFO 09-07 00:14:16 [model.py:684] Resolved architecture: Qwen2ForCausalLM
WARNING 09-07 00:14:16 [model.py:2355] Casting torch.bfloat16 to torch.float16.
INFO 09-07 00:14:16 [model.py:2021] Using max model len 4096
INFO 09-07 00:14:16 [scheduler.py:242] Chunked prefill is enabled with max_num_batched_tokens=8192.
WARNING 09-07 00:14:16 [vllm.py:1322] Enforce eager set, disabling torch.compile and CUDAGraphs. This is equivalent to setting -cc.mode=none -cc.cudagraph_mode=none
WARNING 09-07 00:14:16 [vllm.py:1357] Inductor compilation was disabled by user settings, optimizations settings that are only active during inductor compilation will be ignored.
INFO 09-07 00:14:16 [kernel.py:365] Final IR op priority after setting platform defaults: IrOpPriorityConfig(rms_norm=['vllm_c', 'native'], fused_add_rms_norm=['vllm_c', 'native'])
INFO 09-07 00:14:16 [vllm.py:1536] Cudagraph is disabled under eager mode
INFO 09-07 00:14:16 [compilation.py:329] Enabled custom fusions: norm_quant, act_quant
(EngineCore pid=120706) INFO 09-07 00:14:19 [core.py:123] Initializing a V1 LLM engine (v0.28.1rc1.dev50+gcac8a75a7.d20260828) with config: model='/home/dconnolly/.cache/huggingface/hub/models--Qwen--Qwen2.5-1.5B-Instruct/snapshots/989aa7980e4cf806f80c7fef2b1adb7bc71aa306', speculative_config=None, tokenizer='/home/dconnolly/.cache/huggingface/hub/models--Qwen--Qwen2.5-1.5B-Instruct/snapshots/989aa7980e4cf806f80c7fef2b1adb7bc71aa306', skip_tokenizer_init=False, tokenizer_mode=auto, revision=None, tokenizer_revision=None, trust_remote_code=False, dtype=torch.float16, max_seq_len=4096, download_dir=None, load_format=auto, tensor_parallel_size=1, pipeline_parallel_size=1, data_parallel_size=1, decode_context_parallel_size=1, dcp_comm_backend=ag_rs, disable_custom_all_reduce=False, quantization=None, quantization_config=None, enforce_eager=True, enable_return_routed_experts=False, kv_cache_dtype=auto, device_config=cuda, structured_outputs_config=StructuredOutputsConfig(backend='auto', disable_any_whitespace=False, disable_additional_properties=False, reasoning_parser='', reasoning_parser_plugin='', enable_in_reasoning=False), observability_config=ObservabilityConfig(show_hidden_metrics_for_version=None, otlp_traces_endpoint=None, collect_detailed_traces=None, per_request_spec_decode_metrics='none', kv_cache_metrics=False, kv_cache_metrics_sample=0.01, cudagraph_metrics=False, enable_layerwise_nvtx_tracing=False, enable_mfu_metrics=False, enable_mm_processor_stats=False, enable_logging_iteration_details=False, jit_monitor_mode='warn', jit_monitor_verbose=False), seed=0, served_model_name=/home/dconnolly/.cache/huggingface/hub/models--Qwen--Qwen2.5-1.5B-Instruct/snapshots/989aa7980e4cf806f80c7fef2b1adb7bc71aa306, enable_prefix_caching=False, enable_chunked_prefill=True, pooler_config=None, compilation_config={'mode': <CompilationMode.NONE: 0>, 'debug_dump_path': None, 'cache_dir': '', 'compile_cache_save_format': 'binary', 'backend': 'inductor', 'custom_ops': ['all'], 'ir_enable_torch_wrap': False, 'splitting_ops': [], 'compile_mm_encoder': False, 'cudagraph_mm_encoder': False, 'encoder_cudagraph_token_budgets': [], 'encoder_cudagraph_max_vision_items_per_batch': 0, 'encoder_cudagraph_max_frames_per_batch': None, 'compile_sizes': [], 'compile_ranges_endpoints': [8192], 'inductor_compile_config': {'enable_auto_functionalized_v2': False, 'combo_kernels': True, 'benchmark_combo_kernel': True}, 'inductor_passes': {}, 'cudagraph_mode': <CUDAGraphMode.NONE: 0>, 'cudagraph_num_of_warmups': 0, 'cudagraph_capture_sizes': [], 'cudagraph_copy_inputs': False, 'cudagraph_specialize_lora': True, 'use_inductor_graph_partition': False, 'pass_config': {'fuse_norm_quant': True, 'fuse_act_quant': True, 'fuse_attn_quant': False, 'enable_sp': False, 'fuse_gemm_comms': False, 'fuse_allreduce_rms': False, 'enable_qk_norm_rope_fusion': False, 'fuse_rope_kvcache_cat_mla': False, 'fuse_act_padding': False, 'fuse_qk_norm_rope_kvcache': False}, 'max_cudagraph_capture_size': 0, 'dynamic_shapes_config': {'type': <DynamicShapesType.BACKED: 'backed'>, 'evaluate_guards': False, 'assume_32_bit_indexing': False}, 'local_cache_dir': None, 'fast_moe_cold_start': False, 'static_all_moe_layers': []}, kernel_config=KernelConfig(ir_op_priority=IrOpPriorityConfig(rms_norm=['vllm_c', 'native'], fused_add_rms_norm=['vllm_c', 'native']), enable_flashinfer_autotune=True, enable_cutedsl_warmup=True, enable_jit_warmup=True, enable_bf16x3_router_gemm=False, moe_backend='auto', linear_backend='auto')
(EngineCore pid=120706) INFO 09-07 00:14:19 [parallel_state.py:1638] world_size=1 rank=0 local_rank=0 distributed_init_method=file:///tmp/vllm_dist_e49c83ec15eb40568990451baa7b44ae backend=nccl
(EngineCore pid=120706) INFO 09-07 00:14:20 [parallel_state.py:1982] rank 0 in world size 1 is assigned as DP rank 0, PP rank 0, PCP rank 0, TP rank 0, EP rank N/A, EPLB rank N/A
(EngineCore pid=120706) INFO 09-07 00:14:20 [gpu_worker.py:399] Using V2 Model Runner
(EngineCore pid=120706) INFO 09-07 00:14:21 [model_runner.py:375] Loading model from scratch...
(EngineCore pid=120706) INFO 09-07 00:14:21 [cuda.py:436] Using AttentionBackendEnum.BRIDGE_ATTN backend.
(EngineCore pid=120706) INFO 09-07 00:14:21 [weight_utils.py:858] Filesystem type for checkpoints: XFS. Checkpoint size: 2.88 GiB. Available RAM: 114.00 GiB.
(EngineCore pid=120706) INFO 09-07 00:14:21 [weight_utils.py:881] Auto-prefetch is disabled because the filesystem (XFS) is not a recognized network FS (NFS/Lustre). If you want to force prefetching, start vLLM with --safetensors-load-strategy=prefetch.
(EngineCore pid=120706) 
Loading safetensors checkpoint shards:   0% Completed | 0/1 [00:00<?, ?it/s]
(EngineCore pid=120706) 
Loading safetensors checkpoint shards: 100% Completed | 1/1 [00:00<00:00,  1.28it/s]
(EngineCore pid=120706) 
Loading safetensors checkpoint shards: 100% Completed | 1/1 [00:00<00:00,  1.28it/s]
(EngineCore pid=120706) 
(EngineCore pid=120706) INFO 09-07 00:14:22 [default_loader.py:430] Loading weights took 0.79 seconds
(EngineCore pid=120706) INFO 09-07 00:14:22 [model_runner.py:397] Model loading took 2.98 GiB memory and 1.988948 seconds
(EngineCore pid=120706) WARNING 09-07 00:14:23 [topk_topp_sampler.py:69] FlashInfer top-p/top-k sampling unavailable: unsupported compute capability 7.5; falling back. Set VLLM_USE_FLASHINFER_SAMPLER=0 to silence.
(EngineCore pid=120706) INFO 09-07 00:14:23 [utils.py:306] Using LBNHC KV cache layout.
(EngineCore pid=120706) INFO 09-07 00:14:25 [gpu_worker.py:586] Available KV cache memory: 17.36 GiB
(EngineCore pid=120706) INFO 09-07 00:14:25 [kv_cache_utils.py:1879] GPU KV cache size: 650,208 tokens, Maximum concurrency for 4,096 tokens per request: 158.74x
(EngineCore pid=120706) INFO 09-07 00:14:25 [kernel_warmup.py:124] JIT kernel warmup starting.
(EngineCore pid=120706) INFO 09-07 00:14:25 [kernel_warmup.py:134] JIT kernel warmup finished in 0.00s.
(EngineCore pid=120706) INFO 09-07 00:14:26 [gpu_worker.py:817] Free memory on device (23.3/23.47 GiB) on startup. Desired GPU memory utilization is (0.9, 21.12 GiB). Actual usage is 3.25 GiB for consumed memory (weights + non-torch), 0.5 GiB for peak activation, and 0.0 GiB for CUDAGraph memory. Replace gpu_memory_utilization config with `--kv-cache-memory=18485812532` (17.22 GiB) to fit into requested memory, or `--kv-cache-memory=20830611968` (19.4 GiB) to fully utilize gpu memory. Current kv cache memory in use is 17.36 GiB.
(EngineCore pid=120706) INFO 09-07 00:14:26 [bridge_attn.py:210] bridge_attn: attention executing on the cu_sm80_on_sm75 bridge kernel via flash-attn's sm_75 route
(EngineCore pid=120706) INFO 09-07 00:15:25 [jit_monitor.py:85] Kernel JIT monitor activated; monitored JIT compilations during inference will use mode=warn.
(EngineCore pid=120706) INFO 09-07 00:15:26 [torch_utils.py:277] Reducing Torch threads from 8 to 1 for serving. Set OMP_NUM_THREADS in the external environment to override.
(EngineCore pid=120706) INFO 09-07 00:15:26 [core.py:368] init engine (profile, create kv cache, warmup model) took 62.76 s
(EngineCore pid=120706) INFO 09-07 00:15:26 [kernel.py:365] Final IR op priority after setting platform defaults: IrOpPriorityConfig(rms_norm=['vllm_c', 'native'], fused_add_rms_norm=['vllm_c', 'native'])
INFO 09-07 00:15:26 [hf.py:547] Detected the chat template content format to be 'string'. You can set `--chat-template-content-format` to override this.

Rendering prompts:   0%|          | 0/1 [00:00<?, ?it/s]
Rendering prompts: 100%|██████████| 1/1 [00:00<00:00, 87.84it/s]

Processed prompts:   0%|          | 0/1 [00:00<?, ?it/s, est. speed input: 0.00 toks/s, output: 0.00 toks/s]
Processed prompts: 100%|██████████| 1/1 [00:00<00:00,  3.80it/s, est. speed input: 7.61 toks/s, output: 30.42 toks/s]
Processed prompts: 100%|██████████| 1/1 [00:00<00:00,  3.80it/s, est. speed input: 7.61 toks/s, output: 30.42 toks/s]
Processed prompts: 100%|██████████| 1/1 [00:00<00:00,  3.79it/s, est. speed input: 7.61 toks/s, output: 30.42 toks/s]

Rendering prompts:   0%|          | 0/4 [00:00<?, ?it/s]
Rendering prompts: 100%|██████████| 4/4 [00:00<00:00, 1499.84it/s]

Processed prompts:   0%|          | 0/4 [00:00<?, ?it/s, est. speed input: 0.00 toks/s, output: 0.00 toks/s]
Processed prompts:  25%|██▌       | 1/4 [00:04<00:13,  4.63s/it, est. speed input: 1.73 toks/s, output: 13.83 toks/s]
Processed prompts: 100%|██████████| 4/4 [00:04<00:00,  4.63s/it, est. speed input: 6.91 toks/s, output: 55.31 toks/s]
Processed prompts: 100%|██████████| 4/4 [00:04<00:00,  1.16s/it, est. speed input: 6.91 toks/s, output: 55.31 toks/s]

Rendering prompts:   0%|          | 0/1 [00:00<?, ?it/s]
Rendering prompts: 100%|██████████| 1/1 [00:00<00:00, 248.26it/s]

Processed prompts:   0%|          | 0/1 [00:00<?, ?it/s, est. speed input: 0.00 toks/s, output: 0.00 toks/s]
Processed prompts: 100%|██████████| 1/1 [00:00<00:00, 26.39it/s, est. speed input: 13562.33 toks/s, output: 26.42 toks/s]
Processed prompts: 100%|██████████| 1/1 [00:00<00:00, 26.19it/s, est. speed input: 13562.33 toks/s, output: 26.42 toks/s]

Rendering prompts:   0%|          | 0/1 [00:00<?, ?it/s]
Rendering prompts: 100%|██████████| 1/1 [00:00<00:00, 254.35it/s]

Processed prompts:   0%|          | 0/1 [00:00<?, ?it/s, est. speed input: 0.00 toks/s, output: 0.00 toks/s]
Processed prompts: 100%|██████████| 1/1 [00:00<00:00,  1.04it/s, est. speed input: 533.75 toks/s, output: 33.29 toks/s]
Processed prompts: 100%|██████████| 1/1 [00:00<00:00,  1.04it/s, est. speed input: 533.75 toks/s, output: 33.29 toks/s]
Processed prompts: 100%|██████████| 1/1 [00:00<00:00,  1.04it/s, est. speed input: 533.75 toks/s, output: 33.29 toks/s]

Rendering prompts:   0%|          | 0/1 [00:00<?, ?it/s]
Rendering prompts: 100%|██████████| 1/1 [00:00<00:00, 79.34it/s]

Processed prompts:   0%|          | 0/1 [00:00<?, ?it/s, est. speed input: 0.00 toks/s, output: 0.00 toks/s]
Processed prompts: 100%|██████████| 1/1 [00:00<00:00,  8.26it/s, est. speed input: 16936.45 toks/s, output: 8.26 toks/s]
Processed prompts: 100%|██████████| 1/1 [00:00<00:00,  8.26it/s, est. speed input: 16936.45 toks/s, output: 8.26 toks/s]
Processed prompts: 100%|██████████| 1/1 [00:00<00:00,  8.23it/s, est. speed input: 16936.45 toks/s, output: 8.26 toks/s]

Rendering prompts:   0%|          | 0/1 [00:00<?, ?it/s]
Rendering prompts: 100%|██████████| 1/1 [00:00<00:00, 89.29it/s]

Processed prompts:   0%|          | 0/1 [00:00<?, ?it/s, est. speed input: 0.00 toks/s, output: 0.00 toks/s]
Processed prompts: 100%|██████████| 1/1 [00:01<00:00,  1.08s/it, est. speed input: 1906.09 toks/s, output: 29.77 toks/s]
Processed prompts: 100%|██████████| 1/1 [00:01<00:00,  1.08s/it, est. speed input: 1906.09 toks/s, output: 29.77 toks/s]
Processed prompts: 100%|██████████| 1/1 [00:01<00:00,  1.08s/it, est. speed input: 1906.09 toks/s, output: 29.77 toks/s]

Rendering prompts:   0%|          | 0/4 [00:00<?, ?it/s]
Rendering prompts: 100%|██████████| 4/4 [00:00<00:00, 1234.62it/s]

Processed prompts:   0%|          | 0/4 [00:00<?, ?it/s, est. speed input: 0.00 toks/s, output: 0.00 toks/s]
Processed prompts:  25%|██▌       | 1/4 [00:04<00:13,  4.57s/it, est. speed input: 1.75 toks/s, output: 14.01 toks/s]
Processed prompts: 100%|██████████| 4/4 [00:04<00:00,  4.57s/it, est. speed input: 7.00 toks/s, output: 56.03 toks/s]
Processed prompts: 100%|██████████| 4/4 [00:04<00:00,  1.14s/it, est. speed input: 7.00 toks/s, output: 56.03 toks/s]

Rendering prompts:   0%|          | 0/1 [00:00<?, ?it/s]
Rendering prompts: 100%|██████████| 1/1 [00:00<00:00, 260.42it/s]

Processed prompts:   0%|          | 0/1 [00:00<?, ?it/s, est. speed input: 0.00 toks/s, output: 0.00 toks/s]
Processed prompts: 100%|██████████| 1/1 [00:00<00:00, 28.36it/s, est. speed input: 14572.14 toks/s, output: 28.39 toks/s]
Processed prompts: 100%|██████████| 1/1 [00:00<00:00, 28.13it/s, est. speed input: 14572.14 toks/s, output: 28.39 toks/s]

Rendering prompts:   0%|          | 0/1 [00:00<?, ?it/s]
Rendering prompts: 100%|██████████| 1/1 [00:00<00:00, 266.69it/s]

Processed prompts:   0%|          | 0/1 [00:00<?, ?it/s, est. speed input: 0.00 toks/s, output: 0.00 toks/s]
Processed prompts: 100%|██████████| 1/1 [00:00<00:00,  1.02it/s, est. speed input: 523.72 toks/s, output: 32.67 toks/s]
Processed prompts: 100%|██████████| 1/1 [00:00<00:00,  1.02it/s, est. speed input: 523.72 toks/s, output: 32.67 toks/s]
Processed prompts: 100%|██████████| 1/1 [00:00<00:00,  1.02it/s, est. speed input: 523.72 toks/s, output: 32.67 toks/s]

Rendering prompts:   0%|          | 0/1 [00:00<?, ?it/s]
Rendering prompts: 100%|██████████| 1/1 [00:00<00:00, 81.60it/s]

Processed prompts:   0%|          | 0/1 [00:00<?, ?it/s, est. speed input: 0.00 toks/s, output: 0.00 toks/s]
Processed prompts: 100%|██████████| 1/1 [00:00<00:00,  8.52it/s, est. speed input: 17463.30 toks/s, output: 8.52 toks/s]
Processed prompts: 100%|██████████| 1/1 [00:00<00:00,  8.52it/s, est. speed input: 17463.30 toks/s, output: 8.52 toks/s]
Processed prompts: 100%|██████████| 1/1 [00:00<00:00,  8.49it/s, est. speed input: 17463.30 toks/s, output: 8.52 toks/s]

Rendering prompts:   0%|          | 0/1 [00:00<?, ?it/s]
Rendering prompts: 100%|██████████| 1/1 [00:00<00:00, 82.23it/s]

Processed prompts:   0%|          | 0/1 [00:00<?, ?it/s, est. speed input: 0.00 toks/s, output: 0.00 toks/s]
Processed prompts: 100%|██████████| 1/1 [00:01<00:00,  1.13s/it, est. speed input: 1817.40 toks/s, output: 28.38 toks/s]
Processed prompts: 100%|██████████| 1/1 [00:01<00:00,  1.13s/it, est. speed input: 1817.40 toks/s, output: 28.38 toks/s]
Processed prompts: 100%|██████████| 1/1 [00:01<00:00,  1.13s/it, est. speed input: 1817.40 toks/s, output: 28.38 toks/s]

Rendering prompts:   0%|          | 0/4 [00:00<?, ?it/s]
Rendering prompts: 100%|██████████| 4/4 [00:00<00:00, 1043.16it/s]

Processed prompts:   0%|          | 0/4 [00:00<?, ?it/s, est. speed input: 0.00 toks/s, output: 0.00 toks/s]
Processed prompts:  25%|██▌       | 1/4 [00:04<00:13,  4.66s/it, est. speed input: 1.72 toks/s, output: 13.74 toks/s]
Processed prompts: 100%|██████████| 4/4 [00:04<00:00,  4.66s/it, est. speed input: 6.87 toks/s, output: 54.96 toks/s]
Processed prompts: 100%|██████████| 4/4 [00:04<00:00,  1.16s/it, est. speed input: 6.87 toks/s, output: 54.96 toks/s]

Rendering prompts:   0%|          | 0/1 [00:00<?, ?it/s]
Rendering prompts: 100%|██████████| 1/1 [00:00<00:00, 269.30it/s]

Processed prompts:   0%|          | 0/1 [00:00<?, ?it/s, est. speed input: 0.00 toks/s, output: 0.00 toks/s]
Processed prompts: 100%|██████████| 1/1 [00:00<00:00, 27.51it/s, est. speed input: 14135.78 toks/s, output: 27.54 toks/s]
Processed prompts: 100%|██████████| 1/1 [00:00<00:00, 27.29it/s, est. speed input: 14135.78 toks/s, output: 27.54 toks/s]

Rendering prompts:   0%|          | 0/1 [00:00<?, ?it/s]
Rendering prompts: 100%|██████████| 1/1 [00:00<00:00, 267.60it/s]

Processed prompts:   0%|          | 0/1 [00:00<?, ?it/s, est. speed input: 0.00 toks/s, output: 0.00 toks/s]
Processed prompts: 100%|██████████| 1/1 [00:00<00:00,  1.06it/s, est. speed input: 545.51 toks/s, output: 34.03 toks/s]
Processed prompts: 100%|██████████| 1/1 [00:00<00:00,  1.06it/s, est. speed input: 545.51 toks/s, output: 34.03 toks/s]
Processed prompts: 100%|██████████| 1/1 [00:00<00:00,  1.06it/s, est. speed input: 545.51 toks/s, output: 34.03 toks/s]

Rendering prompts:   0%|          | 0/1 [00:00<?, ?it/s]
Rendering prompts: 100%|██████████| 1/1 [00:00<00:00, 84.08it/s]

Processed prompts:   0%|          | 0/1 [00:00<?, ?it/s, est. speed input: 0.00 toks/s, output: 0.00 toks/s]
Processed prompts: 100%|██████████| 1/1 [00:00<00:00,  8.51it/s, est. speed input: 17438.96 toks/s, output: 8.51 toks/s]
Processed prompts: 100%|██████████| 1/1 [00:00<00:00,  8.51it/s, est. speed input: 17438.96 toks/s, output: 8.51 toks/s]
Processed prompts: 100%|██████████| 1/1 [00:00<00:00,  8.48it/s, est. speed input: 17438.96 toks/s, output: 8.51 toks/s]

Rendering prompts:   0%|          | 0/1 [00:00<?, ?it/s]
Rendering prompts: 100%|██████████| 1/1 [00:00<00:00, 80.36it/s]

Processed prompts:   0%|          | 0/1 [00:00<?, ?it/s, est. speed input: 0.00 toks/s, output: 0.00 toks/s]
Processed prompts: 100%|██████████| 1/1 [00:01<00:00,  1.03s/it, est. speed input: 1986.49 toks/s, output: 31.02 toks/s]
Processed prompts: 100%|██████████| 1/1 [00:01<00:00,  1.03s/it, est. speed input: 1986.49 toks/s, output: 31.02 toks/s]
Processed prompts: 100%|██████████| 1/1 [00:01<00:00,  1.03s/it, est. speed input: 1986.49 toks/s, output: 31.02 toks/s]

Rendering prompts:   0%|          | 0/4 [00:00<?, ?it/s]
Rendering prompts: 100%|██████████| 4/4 [00:00<00:00, 1243.77it/s]

Processed prompts:   0%|          | 0/4 [00:00<?, ?it/s, est. speed input: 0.00 toks/s, output: 0.00 toks/s]
Processed prompts:  25%|██▌       | 1/4 [00:04<00:14,  4.70s/it, est. speed input: 1.70 toks/s, output: 13.62 toks/s]
Processed prompts: 100%|██████████| 4/4 [00:04<00:00,  4.70s/it, est. speed input: 6.81 toks/s, output: 54.46 toks/s]
Processed prompts: 100%|██████████| 4/4 [00:04<00:00,  1.18s/it, est. speed input: 6.81 toks/s, output: 54.46 toks/s]

Rendering prompts:   0%|          | 0/1 [00:00<?, ?it/s]
Rendering prompts: 100%|██████████| 1/1 [00:00<00:00, 212.06it/s]

Processed prompts:   0%|          | 0/1 [00:00<?, ?it/s, est. speed input: 0.00 toks/s, output: 0.00 toks/s]
Processed prompts: 100%|██████████| 1/1 [00:00<00:00, 26.78it/s, est. speed input: 13769.86 toks/s, output: 26.82 toks/s]
Processed prompts: 100%|██████████| 1/1 [00:00<00:00, 26.49it/s, est. speed input: 13769.86 toks/s, output: 26.82 toks/s]

Rendering prompts:   0%|          | 0/1 [00:00<?, ?it/s]
Rendering prompts: 100%|██████████| 1/1 [00:00<00:00, 214.61it/s]

Processed prompts:   0%|          | 0/1 [00:00<?, ?it/s, est. speed input: 0.00 toks/s, output: 0.00 toks/s]
Processed prompts: 100%|██████████| 1/1 [00:01<00:00,  1.03s/it, est. speed input: 495.72 toks/s, output: 30.92 toks/s]
Processed prompts: 100%|██████████| 1/1 [00:01<00:00,  1.03s/it, est. speed input: 495.72 toks/s, output: 30.92 toks/s]
Processed prompts: 100%|██████████| 1/1 [00:01<00:00,  1.04s/it, est. speed input: 495.72 toks/s, output: 30.92 toks/s]

Rendering prompts:   0%|          | 0/1 [00:00<?, ?it/s]
Rendering prompts: 100%|██████████| 1/1 [00:00<00:00, 88.63it/s]

Processed prompts:   0%|          | 0/1 [00:00<?, ?it/s, est. speed input: 0.00 toks/s, output: 0.00 toks/s]
Processed prompts: 100%|██████████| 1/1 [00:00<00:00,  8.54it/s, est. speed input: 17494.98 toks/s, output: 8.54 toks/s]
Processed prompts: 100%|██████████| 1/1 [00:00<00:00,  8.54it/s, est. speed input: 17494.98 toks/s, output: 8.54 toks/s]
Processed prompts: 100%|██████████| 1/1 [00:00<00:00,  8.50it/s, est. speed input: 17494.98 toks/s, output: 8.54 toks/s]

Rendering prompts:   0%|          | 0/1 [00:00<?, ?it/s]
Rendering prompts: 100%|██████████| 1/1 [00:00<00:00, 85.41it/s]

Processed prompts:   0%|          | 0/1 [00:00<?, ?it/s, est. speed input: 0.00 toks/s, output: 0.00 toks/s]
Processed prompts: 100%|██████████| 1/1 [00:01<00:00,  1.12s/it, est. speed input: 1831.16 toks/s, output: 28.60 toks/s]
Processed prompts: 100%|██████████| 1/1 [00:01<00:00,  1.12s/it, est. speed input: 1831.16 toks/s, output: 28.60 toks/s]
Processed prompts: 100%|██████████| 1/1 [00:01<00:00,  1.12s/it, est. speed input: 1831.16 toks/s, output: 28.60 toks/s]

Rendering prompts:   0%|          | 0/4 [00:00<?, ?it/s]
Rendering prompts: 100%|██████████| 4/4 [00:00<00:00, 1227.75it/s]

Processed prompts:   0%|          | 0/4 [00:00<?, ?it/s, est. speed input: 0.00 toks/s, output: 0.00 toks/s]
Processed prompts:  25%|██▌       | 1/4 [00:04<00:14,  4.78s/it, est. speed input: 1.67 toks/s, output: 13.39 toks/s]
Processed prompts: 100%|██████████| 4/4 [00:04<00:00,  4.78s/it, est. speed input: 6.70 toks/s, output: 53.57 toks/s]
Processed prompts: 100%|██████████| 4/4 [00:04<00:00,  1.19s/it, est. speed input: 6.70 toks/s, output: 53.57 toks/s]

Rendering prompts:   0%|          | 0/1 [00:00<?, ?it/s]
Rendering prompts: 100%|██████████| 1/1 [00:00<00:00, 265.43it/s]

Processed prompts:   0%|          | 0/1 [00:00<?, ?it/s, est. speed input: 0.00 toks/s, output: 0.00 toks/s]
Processed prompts: 100%|██████████| 1/1 [00:00<00:00, 27.80it/s, est. speed input: 14292.97 toks/s, output: 27.82 toks/s]
Processed prompts: 100%|██████████| 1/1 [00:00<00:00, 27.60it/s, est. speed input: 14292.97 toks/s, output: 27.82 toks/s]

Rendering prompts:   0%|          | 0/1 [00:00<?, ?it/s]
Rendering prompts: 100%|██████████| 1/1 [00:00<00:00, 268.93it/s]

Processed prompts:   0%|          | 0/1 [00:00<?, ?it/s, est. speed input: 0.00 toks/s, output: 0.00 toks/s]
Processed prompts: 100%|██████████| 1/1 [00:00<00:00,  1.03it/s, est. speed input: 529.41 toks/s, output: 33.02 toks/s]
Processed prompts: 100%|██████████| 1/1 [00:00<00:00,  1.03it/s, est. speed input: 529.41 toks/s, output: 33.02 toks/s]
Processed prompts: 100%|██████████| 1/1 [00:00<00:00,  1.03it/s, est. speed input: 529.41 toks/s, output: 33.02 toks/s]

Rendering prompts:   0%|          | 0/1 [00:00<?, ?it/s]
Rendering prompts: 100%|██████████| 1/1 [00:00<00:00, 79.76it/s]

Processed prompts:   0%|          | 0/1 [00:00<?, ?it/s, est. speed input: 0.00 toks/s, output: 0.00 toks/s]
Processed prompts: 100%|██████████| 1/1 [00:00<00:00,  8.52it/s, est. speed input: 17469.91 toks/s, output: 8.52 toks/s]
Processed prompts: 100%|██████████| 1/1 [00:00<00:00,  8.52it/s, est. speed input: 17469.91 toks/s, output: 8.52 toks/s]
Processed prompts: 100%|██████████| 1/1 [00:00<00:00,  8.49it/s, est. speed input: 17469.91 toks/s, output: 8.52 toks/s]

Rendering prompts:   0%|          | 0/1 [00:00<?, ?it/s]
Rendering prompts: 100%|██████████| 1/1 [00:00<00:00, 80.51it/s]

Processed prompts:   0%|          | 0/1 [00:00<?, ?it/s, est. speed input: 0.00 toks/s, output: 0.00 toks/s]
Processed prompts: 100%|██████████| 1/1 [00:01<00:00,  1.07s/it, est. speed input: 1910.31 toks/s, output: 29.83 toks/s]
Processed prompts: 100%|██████████| 1/1 [00:01<00:00,  1.07s/it, est. speed input: 1910.31 toks/s, output: 29.83 toks/s]
Processed prompts: 100%|██████████| 1/1 [00:01<00:00,  1.07s/it, est. speed input: 1910.31 toks/s, output: 29.83 toks/s]
SPLIT[BRIDGE_ATTN-tp1] short_decode=54.4 (reps 53.5/54.4/54.9)
SPLIT[BRIDGE_ATTN-tp1] ctx512_prefill=11863.9 (reps 11696.9/11863.9/12403.2)
SPLIT[BRIDGE_ATTN-tp1] ctx512_decode=32.8 (reps 31.1/32.8/33.2)
SPLIT[BRIDGE_ATTN-tp1] ctx512_mixed=32.5 (reps 30.7/32.5/32.9)
SPLIT[BRIDGE_ATTN-tp1] ctx2048_prefill=15599.4 (reps 15159.5/15599.4/15633.3)
SPLIT[BRIDGE_ATTN-tp1] ctx2048_decode=30.9 (reps 30.7/30.9/32.4)
SPLIT[BRIDGE_ATTN-tp1] ctx2048_mixed=28.3 (reps 28.0/28.3/29.4)
SPLIT[BRIDGE_ATTN-tp1] done
INFO 09-07 00:16:01 [utils.py:620] [shutdown] Process manager: send sigterm to process EngineCore
(EngineCore pid=120706) INFO 09-07 00:16:01 [core.py:1350] [shutdown] EngineCore: trigger received signal=SIGTERM
(EngineCore pid=120706) INFO 09-07 00:16:01 [core.py:1501] [shutdown] EngineCore: start mode=abort timeout=0s
(EngineCore pid=120706) INFO 09-07 00:16:01 [core.py:1532] [shutdown] EngineCore: request processing complete; starting resource teardown
(EngineCore pid=120706) INFO 09-07 00:16:01 [core.py:1363] [shutdown] EngineCore: exiting busy loop
rc=0 BRIDGE_ATTN tp1
=== TRITON_ATTN tp=1 ===
gpu-lease: granted gpu0 (CUDA_VISIBLE_DEVICES=0, cpuset=host-default, mem-gb=unbounded); starting: env CUDA_HOME=/opt/cuda FLASHINFER_DISABLE_VERSION_CHECK=1 /opt/repo/agentic-vllm/software/vllm/sm75-marlin/turing_lab/thirdparty/../../.venv/bin/python vllm_macro_gate.py --arm TRITON_ATTN 1
INFO 09-07 00:16:11 [api_utils.py:286] non-default args: {'dtype': 'float16', 'max_model_len': 4096, 'enable_prefix_caching': False, 'gpu_memory_utilization': 0.9, 'disable_log_stats': True, 'enforce_eager': True, 'attention_backend': 'TRITON_ATTN', 'model': '/home/dconnolly/.cache/huggingface/hub/models--Qwen--Qwen2.5-1.5B-Instruct/snapshots/989aa7980e4cf806f80c7fef2b1adb7bc71aa306'}
INFO 09-07 00:16:11 [model.py:684] Resolved architecture: Qwen2ForCausalLM
WARNING 09-07 00:16:11 [model.py:2355] Casting torch.bfloat16 to torch.float16.
INFO 09-07 00:16:11 [model.py:2021] Using max model len 4096
INFO 09-07 00:16:11 [scheduler.py:242] Chunked prefill is enabled with max_num_batched_tokens=8192.
WARNING 09-07 00:16:11 [vllm.py:1322] Enforce eager set, disabling torch.compile and CUDAGraphs. This is equivalent to setting -cc.mode=none -cc.cudagraph_mode=none
WARNING 09-07 00:16:11 [vllm.py:1357] Inductor compilation was disabled by user settings, optimizations settings that are only active during inductor compilation will be ignored.
INFO 09-07 00:16:11 [kernel.py:365] Final IR op priority after setting platform defaults: IrOpPriorityConfig(rms_norm=['vllm_c', 'native'], fused_add_rms_norm=['vllm_c', 'native'])
INFO 09-07 00:16:11 [vllm.py:1536] Cudagraph is disabled under eager mode
INFO 09-07 00:16:11 [compilation.py:329] Enabled custom fusions: norm_quant, act_quant
(EngineCore pid=120971) INFO 09-07 00:16:13 [core.py:123] Initializing a V1 LLM engine (v0.28.1rc1.dev50+gcac8a75a7.d20260828) with config: model='/home/dconnolly/.cache/huggingface/hub/models--Qwen--Qwen2.5-1.5B-Instruct/snapshots/989aa7980e4cf806f80c7fef2b1adb7bc71aa306', speculative_config=None, tokenizer='/home/dconnolly/.cache/huggingface/hub/models--Qwen--Qwen2.5-1.5B-Instruct/snapshots/989aa7980e4cf806f80c7fef2b1adb7bc71aa306', skip_tokenizer_init=False, tokenizer_mode=auto, revision=None, tokenizer_revision=None, trust_remote_code=False, dtype=torch.float16, max_seq_len=4096, download_dir=None, load_format=auto, tensor_parallel_size=1, pipeline_parallel_size=1, data_parallel_size=1, decode_context_parallel_size=1, dcp_comm_backend=ag_rs, disable_custom_all_reduce=False, quantization=None, quantization_config=None, enforce_eager=True, enable_return_routed_experts=False, kv_cache_dtype=auto, device_config=cuda, structured_outputs_config=StructuredOutputsConfig(backend='auto', disable_any_whitespace=False, disable_additional_properties=False, reasoning_parser='', reasoning_parser_plugin='', enable_in_reasoning=False), observability_config=ObservabilityConfig(show_hidden_metrics_for_version=None, otlp_traces_endpoint=None, collect_detailed_traces=None, per_request_spec_decode_metrics='none', kv_cache_metrics=False, kv_cache_metrics_sample=0.01, cudagraph_metrics=False, enable_layerwise_nvtx_tracing=False, enable_mfu_metrics=False, enable_mm_processor_stats=False, enable_logging_iteration_details=False, jit_monitor_mode='warn', jit_monitor_verbose=False), seed=0, served_model_name=/home/dconnolly/.cache/huggingface/hub/models--Qwen--Qwen2.5-1.5B-Instruct/snapshots/989aa7980e4cf806f80c7fef2b1adb7bc71aa306, enable_prefix_caching=False, enable_chunked_prefill=True, pooler_config=None, compilation_config={'mode': <CompilationMode.NONE: 0>, 'debug_dump_path': None, 'cache_dir': '', 'compile_cache_save_format': 'binary', 'backend': 'inductor', 'custom_ops': ['all'], 'ir_enable_torch_wrap': False, 'splitting_ops': [], 'compile_mm_encoder': False, 'cudagraph_mm_encoder': False, 'encoder_cudagraph_token_budgets': [], 'encoder_cudagraph_max_vision_items_per_batch': 0, 'encoder_cudagraph_max_frames_per_batch': None, 'compile_sizes': [], 'compile_ranges_endpoints': [8192], 'inductor_compile_config': {'enable_auto_functionalized_v2': False, 'combo_kernels': True, 'benchmark_combo_kernel': True}, 'inductor_passes': {}, 'cudagraph_mode': <CUDAGraphMode.NONE: 0>, 'cudagraph_num_of_warmups': 0, 'cudagraph_capture_sizes': [], 'cudagraph_copy_inputs': False, 'cudagraph_specialize_lora': True, 'use_inductor_graph_partition': False, 'pass_config': {'fuse_norm_quant': True, 'fuse_act_quant': True, 'fuse_attn_quant': False, 'enable_sp': False, 'fuse_gemm_comms': False, 'fuse_allreduce_rms': False, 'enable_qk_norm_rope_fusion': False, 'fuse_rope_kvcache_cat_mla': False, 'fuse_act_padding': False, 'fuse_qk_norm_rope_kvcache': False}, 'max_cudagraph_capture_size': 0, 'dynamic_shapes_config': {'type': <DynamicShapesType.BACKED: 'backed'>, 'evaluate_guards': False, 'assume_32_bit_indexing': False}, 'local_cache_dir': None, 'fast_moe_cold_start': False, 'static_all_moe_layers': []}, kernel_config=KernelConfig(ir_op_priority=IrOpPriorityConfig(rms_norm=['vllm_c', 'native'], fused_add_rms_norm=['vllm_c', 'native']), enable_flashinfer_autotune=True, enable_cutedsl_warmup=True, enable_jit_warmup=True, enable_bf16x3_router_gemm=False, moe_backend='auto', linear_backend='auto')
(EngineCore pid=120971) INFO 09-07 00:16:14 [parallel_state.py:1638] world_size=1 rank=0 local_rank=0 distributed_init_method=file:///tmp/vllm_dist_844e7739bd6746e3aa11697305c42a33 backend=nccl
(EngineCore pid=120971) INFO 09-07 00:16:14 [parallel_state.py:1982] rank 0 in world size 1 is assigned as DP rank 0, PP rank 0, PCP rank 0, TP rank 0, EP rank N/A, EPLB rank N/A
(EngineCore pid=120971) INFO 09-07 00:16:14 [gpu_worker.py:399] Using V2 Model Runner
(EngineCore pid=120971) INFO 09-07 00:16:15 [model_runner.py:375] Loading model from scratch...
(EngineCore pid=120971) INFO 09-07 00:16:15 [cuda.py:436] Using AttentionBackendEnum.TRITON_ATTN backend.
(EngineCore pid=120971) INFO 09-07 00:16:15 [weight_utils.py:858] Filesystem type for checkpoints: XFS. Checkpoint size: 2.88 GiB. Available RAM: 114.22 GiB.
(EngineCore pid=120971) INFO 09-07 00:16:15 [weight_utils.py:881] Auto-prefetch is disabled because the filesystem (XFS) is not a recognized network FS (NFS/Lustre). If you want to force prefetching, start vLLM with --safetensors-load-strategy=prefetch.
(EngineCore pid=120971) 
Loading safetensors checkpoint shards:   0% Completed | 0/1 [00:00<?, ?it/s]
(EngineCore pid=120971) 
Loading safetensors checkpoint shards: 100% Completed | 1/1 [00:00<00:00,  1.31it/s]
(EngineCore pid=120971) 
Loading safetensors checkpoint shards: 100% Completed | 1/1 [00:00<00:00,  1.31it/s]
(EngineCore pid=120971) 
(EngineCore pid=120971) INFO 09-07 00:16:16 [default_loader.py:430] Loading weights took 0.77 seconds
(EngineCore pid=120971) INFO 09-07 00:16:17 [model_runner.py:397] Model loading took 2.98 GiB memory and 1.980700 seconds
(EngineCore pid=120971) WARNING 09-07 00:16:17 [topk_topp_sampler.py:69] FlashInfer top-p/top-k sampling unavailable: unsupported compute capability 7.5; falling back. Set VLLM_USE_FLASHINFER_SAMPLER=0 to silence.
(EngineCore pid=120971) INFO 09-07 00:16:17 [utils.py:306] Using LBNHC KV cache layout.
(EngineCore pid=120971) INFO 09-07 00:16:20 [gpu_worker.py:586] Available KV cache memory: 17.36 GiB
(EngineCore pid=120971) INFO 09-07 00:16:20 [kv_cache_utils.py:1879] GPU KV cache size: 650,208 tokens, Maximum concurrency for 4,096 tokens per request: 158.74x
(EngineCore pid=120971) INFO 09-07 00:16:20 [kernel_warmup.py:124] JIT kernel warmup starting.
(EngineCore pid=120971) INFO 09-07 00:16:20 [kernel_warmup.py:134] JIT kernel warmup finished in 0.00s.
(EngineCore pid=120971) INFO 09-07 00:16:20 [gpu_worker.py:817] Free memory on device (23.3/23.47 GiB) on startup. Desired GPU memory utilization is (0.9, 21.12 GiB). Actual usage is 3.25 GiB for consumed memory (weights + non-torch), 0.5 GiB for peak activation, and 0.0 GiB for CUDAGraph memory. Replace gpu_memory_utilization config with `--kv-cache-memory=18485812532` (17.22 GiB) to fit into requested memory, or `--kv-cache-memory=20830611968` (19.4 GiB) to fully utilize gpu memory. Current kv cache memory in use is 17.36 GiB.
(EngineCore pid=120971) INFO 09-07 00:16:21 [jit_monitor.py:85] Kernel JIT monitor activated; monitored JIT compilations during inference will use mode=warn.
(EngineCore pid=120971) INFO 09-07 00:16:21 [torch_utils.py:277] Reducing Torch threads from 8 to 1 for serving. Set OMP_NUM_THREADS in the external environment to override.
(EngineCore pid=120971) INFO 09-07 00:16:21 [core.py:368] init engine (profile, create kv cache, warmup model) took 4.05 s
(EngineCore pid=120971) INFO 09-07 00:16:22 [kernel.py:365] Final IR op priority after setting platform defaults: IrOpPriorityConfig(rms_norm=['vllm_c', 'native'], fused_add_rms_norm=['vllm_c', 'native'])
INFO 09-07 00:16:22 [hf.py:547] Detected the chat template content format to be 'string'. You can set `--chat-template-content-format` to override this.

Rendering prompts:   0%|          | 0/1 [00:00<?, ?it/s]
Rendering prompts: 100%|██████████| 1/1 [00:00<00:00, 85.72it/s]

Processed prompts:   0%|          | 0/1 [00:00<?, ?it/s, est. speed input: 0.00 toks/s, output: 0.00 toks/s](EngineCore pid=120971) WARNING 09-07 00:16:22 [jit_monitor.py:141] Triton kernel JIT compilation during inference: kernel_unified_attention. This causes a latency spike; consider extending warmup to cover this shape/config.

Processed prompts: 100%|██████████| 1/1 [00:00<00:00,  5.54it/s, est. speed input: 11.09 toks/s, output: 44.33 toks/s]
Processed prompts: 100%|██████████| 1/1 [00:00<00:00,  5.54it/s, est. speed input: 11.09 toks/s, output: 44.33 toks/s]
Processed prompts: 100%|██████████| 1/1 [00:00<00:00,  5.53it/s, est. speed input: 11.09 toks/s, output: 44.33 toks/s]

Rendering prompts:   0%|          | 0/4 [00:00<?, ?it/s]
Rendering prompts: 100%|██████████| 4/4 [00:00<00:00, 1095.12it/s]

Processed prompts:   0%|          | 0/4 [00:00<?, ?it/s, est. speed input: 0.00 toks/s, output: 0.00 toks/s]
Processed prompts:  25%|██▌       | 1/4 [00:01<00:04,  1.33s/it, est. speed input: 6.00 toks/s, output: 47.98 toks/s]
Processed prompts: 100%|██████████| 4/4 [00:01<00:00,  1.33s/it, est. speed input: 23.97 toks/s, output: 191.79 toks/s]
Processed prompts: 100%|██████████| 4/4 [00:01<00:00,  3.00it/s, est. speed input: 23.97 toks/s, output: 191.79 toks/s]

Rendering prompts:   0%|          | 0/1 [00:00<?, ?it/s]
Rendering prompts: 100%|██████████| 1/1 [00:00<00:00, 205.06it/s]

Processed prompts:   0%|          | 0/1 [00:00<?, ?it/s, est. speed input: 0.00 toks/s, output: 0.00 toks/s]
Processed prompts: 100%|██████████| 1/1 [00:00<00:00, 16.43it/s, est. speed input: 8443.75 toks/s, output: 16.45 toks/s]
Processed prompts: 100%|██████████| 1/1 [00:00<00:00, 16.34it/s, est. speed input: 8443.75 toks/s, output: 16.45 toks/s]

Rendering prompts:   0%|          | 0/1 [00:00<?, ?it/s]
Rendering prompts: 100%|██████████| 1/1 [00:00<00:00, 122.08it/s]

Processed prompts:   0%|          | 0/1 [00:00<?, ?it/s, est. speed input: 0.00 toks/s, output: 0.00 toks/s]
Processed prompts: 100%|██████████| 1/1 [00:00<00:00,  1.50it/s, est. speed input: 769.45 toks/s, output: 48.00 toks/s]
Processed prompts: 100%|██████████| 1/1 [00:00<00:00,  1.50it/s, est. speed input: 769.45 toks/s, output: 48.00 toks/s]
Processed prompts: 100%|██████████| 1/1 [00:00<00:00,  1.50it/s, est. speed input: 769.45 toks/s, output: 48.00 toks/s]

Rendering prompts:   0%|          | 0/1 [00:00<?, ?it/s]
Rendering prompts: 100%|██████████| 1/1 [00:00<00:00, 60.77it/s]

Processed prompts:   0%|          | 0/1 [00:00<?, ?it/s, est. speed input: 0.00 toks/s, output: 0.00 toks/s]
Processed prompts: 100%|██████████| 1/1 [00:00<00:00,  2.14it/s, est. speed input: 4376.27 toks/s, output: 2.14 toks/s]
Processed prompts: 100%|██████████| 1/1 [00:00<00:00,  2.14it/s, est. speed input: 4376.27 toks/s, output: 2.14 toks/s]
Processed prompts: 100%|██████████| 1/1 [00:00<00:00,  2.13it/s, est. speed input: 4376.27 toks/s, output: 2.14 toks/s]

Rendering prompts:   0%|          | 0/1 [00:00<?, ?it/s]
Rendering prompts: 100%|██████████| 1/1 [00:00<00:00, 75.53it/s]

Processed prompts:   0%|          | 0/1 [00:00<?, ?it/s, est. speed input: 0.00 toks/s, output: 0.00 toks/s]
Processed prompts: 100%|██████████| 1/1 [00:01<00:00,  1.11s/it, est. speed input: 1848.25 toks/s, output: 28.86 toks/s]
Processed prompts: 100%|██████████| 1/1 [00:01<00:00,  1.11s/it, est. speed input: 1848.25 toks/s, output: 28.86 toks/s]
Processed prompts: 100%|██████████| 1/1 [00:01<00:00,  1.11s/it, est. speed input: 1848.25 toks/s, output: 28.86 toks/s]

Rendering prompts:   0%|          | 0/4 [00:00<?, ?it/s]
Rendering prompts: 100%|██████████| 4/4 [00:00<00:00, 1236.71it/s]

Processed prompts:   0%|          | 0/4 [00:00<?, ?it/s, est. speed input: 0.00 toks/s, output: 0.00 toks/s]
Processed prompts:  25%|██▌       | 1/4 [00:01<00:04,  1.39s/it, est. speed input: 5.77 toks/s, output: 46.19 toks/s]
Processed prompts: 100%|██████████| 4/4 [00:01<00:00,  1.39s/it, est. speed input: 23.08 toks/s, output: 184.67 toks/s]
Processed prompts: 100%|██████████| 4/4 [00:01<00:00,  2.88it/s, est. speed input: 23.08 toks/s, output: 184.67 toks/s]

Rendering prompts:   0%|          | 0/1 [00:00<?, ?it/s]
Rendering prompts: 100%|██████████| 1/1 [00:00<00:00, 266.12it/s]

Processed prompts:   0%|          | 0/1 [00:00<?, ?it/s, est. speed input: 0.00 toks/s, output: 0.00 toks/s]
Processed prompts: 100%|██████████| 1/1 [00:00<00:00, 16.75it/s, est. speed input: 8602.89 toks/s, output: 16.76 toks/s]
Processed prompts: 100%|██████████| 1/1 [00:00<00:00, 16.68it/s, est. speed input: 8602.89 toks/s, output: 16.76 toks/s]

Rendering prompts:   0%|          | 0/1 [00:00<?, ?it/s]
Rendering prompts: 100%|██████████| 1/1 [00:00<00:00, 265.65it/s]

Processed prompts:   0%|          | 0/1 [00:00<?, ?it/s, est. speed input: 0.00 toks/s, output: 0.00 toks/s]
Processed prompts: 100%|██████████| 1/1 [00:00<00:00,  1.44it/s, est. speed input: 740.17 toks/s, output: 46.17 toks/s]
Processed prompts: 100%|██████████| 1/1 [00:00<00:00,  1.44it/s, est. speed input: 740.17 toks/s, output: 46.17 toks/s]
Processed prompts: 100%|██████████| 1/1 [00:00<00:00,  1.44it/s, est. speed input: 740.17 toks/s, output: 46.17 toks/s]

Rendering prompts:   0%|          | 0/1 [00:00<?, ?it/s]
Rendering prompts: 100%|██████████| 1/1 [00:00<00:00, 61.25it/s]

Processed prompts:   0%|          | 0/1 [00:00<?, ?it/s, est. speed input: 0.00 toks/s, output: 0.00 toks/s]
Processed prompts: 100%|██████████| 1/1 [00:00<00:00,  2.14it/s, est. speed input: 4391.57 toks/s, output: 2.14 toks/s]
Processed prompts: 100%|██████████| 1/1 [00:00<00:00,  2.14it/s, est. speed input: 4391.57 toks/s, output: 2.14 toks/s]
Processed prompts: 100%|██████████| 1/1 [00:00<00:00,  2.14it/s, est. speed input: 4391.57 toks/s, output: 2.14 toks/s]

Rendering prompts:   0%|          | 0/1 [00:00<?, ?it/s]
Rendering prompts: 100%|██████████| 1/1 [00:00<00:00, 91.93it/s]

Processed prompts:   0%|          | 0/1 [00:00<?, ?it/s, est. speed input: 0.00 toks/s, output: 0.00 toks/s]
Processed prompts: 100%|██████████| 1/1 [00:01<00:00,  1.11s/it, est. speed input: 1843.97 toks/s, output: 28.80 toks/s]
Processed prompts: 100%|██████████| 1/1 [00:01<00:00,  1.11s/it, est. speed input: 1843.97 toks/s, output: 28.80 toks/s]
Processed prompts: 100%|██████████| 1/1 [00:01<00:00,  1.11s/it, est. speed input: 1843.97 toks/s, output: 28.80 toks/s]

Rendering prompts:   0%|          | 0/4 [00:00<?, ?it/s]
Rendering prompts: 100%|██████████| 4/4 [00:00<00:00, 2559.06it/s]

Processed prompts:   0%|          | 0/4 [00:00<?, ?it/s, est. speed input: 0.00 toks/s, output: 0.00 toks/s]
Processed prompts:  25%|██▌       | 1/4 [00:01<00:04,  1.36s/it, est. speed input: 5.89 toks/s, output: 47.11 toks/s]
Processed prompts: 100%|██████████| 4/4 [00:01<00:00,  1.36s/it, est. speed input: 23.55 toks/s, output: 188.40 toks/s]
Processed prompts: 100%|██████████| 4/4 [00:01<00:00,  2.94it/s, est. speed input: 23.55 toks/s, output: 188.40 toks/s]

Rendering prompts:   0%|          | 0/1 [00:00<?, ?it/s]
Rendering prompts: 100%|██████████| 1/1 [00:00<00:00, 589.00it/s]

Processed prompts:   0%|          | 0/1 [00:00<?, ?it/s, est. speed input: 0.00 toks/s, output: 0.00 toks/s]
Processed prompts: 100%|██████████| 1/1 [00:00<00:00, 16.57it/s, est. speed input: 8510.00 toks/s, output: 16.58 toks/s]
Processed prompts: 100%|██████████| 1/1 [00:00<00:00, 16.46it/s, est. speed input: 8510.00 toks/s, output: 16.58 toks/s]

Rendering prompts:   0%|          | 0/1 [00:00<?, ?it/s]
Rendering prompts: 100%|██████████| 1/1 [00:00<00:00, 262.03it/s]

Processed prompts:   0%|          | 0/1 [00:00<?, ?it/s, est. speed input: 0.00 toks/s, output: 0.00 toks/s]
Processed prompts: 100%|██████████| 1/1 [00:00<00:00,  1.47it/s, est. speed input: 754.11 toks/s, output: 47.04 toks/s]
Processed prompts: 100%|██████████| 1/1 [00:00<00:00,  1.47it/s, est. speed input: 754.11 toks/s, output: 47.04 toks/s]
Processed prompts: 100%|██████████| 1/1 [00:00<00:00,  1.47it/s, est. speed input: 754.11 toks/s, output: 47.04 toks/s]

Rendering prompts:   0%|          | 0/1 [00:00<?, ?it/s]
Rendering prompts: 100%|██████████| 1/1 [00:00<00:00, 208.75it/s]

Processed prompts:   0%|          | 0/1 [00:00<?, ?it/s, est. speed input: 0.00 toks/s, output: 0.00 toks/s]
Processed prompts: 100%|██████████| 1/1 [00:00<00:00,  2.14it/s, est. speed input: 4392.24 toks/s, output: 2.14 toks/s]
Processed prompts: 100%|██████████| 1/1 [00:00<00:00,  2.14it/s, est. speed input: 4392.24 toks/s, output: 2.14 toks/s]
Processed prompts: 100%|██████████| 1/1 [00:00<00:00,  2.14it/s, est. speed input: 4392.24 toks/s, output: 2.14 toks/s]

Rendering prompts:   0%|          | 0/1 [00:00<?, ?it/s]
Rendering prompts: 100%|██████████| 1/1 [00:00<00:00, 82.31it/s]

Processed prompts:   0%|          | 0/1 [00:00<?, ?it/s, est. speed input: 0.00 toks/s, output: 0.00 toks/s]
Processed prompts: 100%|██████████| 1/1 [00:01<00:00,  1.11s/it, est. speed input: 1850.17 toks/s, output: 28.89 toks/s]
Processed prompts: 100%|██████████| 1/1 [00:01<00:00,  1.11s/it, est. speed input: 1850.17 toks/s, output: 28.89 toks/s]
Processed prompts: 100%|██████████| 1/1 [00:01<00:00,  1.11s/it, est. speed input: 1850.17 toks/s, output: 28.89 toks/s]

Rendering prompts:   0%|          | 0/4 [00:00<?, ?it/s]
Rendering prompts: 100%|██████████| 4/4 [00:00<00:00, 1223.45it/s]

Processed prompts:   0%|          | 0/4 [00:00<?, ?it/s, est. speed input: 0.00 toks/s, output: 0.00 toks/s]
Processed prompts:  25%|██▌       | 1/4 [00:01<00:04,  1.34s/it, est. speed input: 5.98 toks/s, output: 47.83 toks/s]
Processed prompts: 100%|██████████| 4/4 [00:01<00:00,  1.34s/it, est. speed input: 23.91 toks/s, output: 191.25 toks/s]
Processed prompts: 100%|██████████| 4/4 [00:01<00:00,  2.99it/s, est. speed input: 23.91 toks/s, output: 191.25 toks/s]

Rendering prompts:   0%|          | 0/1 [00:00<?, ?it/s]
Rendering prompts: 100%|██████████| 1/1 [00:00<00:00, 251.56it/s]

Processed prompts:   0%|          | 0/1 [00:00<?, ?it/s, est. speed input: 0.00 toks/s, output: 0.00 toks/s]
Processed prompts: 100%|██████████| 1/1 [00:00<00:00, 16.82it/s, est. speed input: 8639.06 toks/s, output: 16.83 toks/s]
Processed prompts: 100%|██████████| 1/1 [00:00<00:00, 16.74it/s, est. speed input: 8639.06 toks/s, output: 16.83 toks/s]

Rendering prompts:   0%|          | 0/1 [00:00<?, ?it/s]
Rendering prompts: 100%|██████████| 1/1 [00:00<00:00, 265.63it/s]

Processed prompts:   0%|          | 0/1 [00:00<?, ?it/s, est. speed input: 0.00 toks/s, output: 0.00 toks/s]
Processed prompts: 100%|██████████| 1/1 [00:00<00:00,  1.49it/s, est. speed input: 764.80 toks/s, output: 47.70 toks/s]
Processed prompts: 100%|██████████| 1/1 [00:00<00:00,  1.49it/s, est. speed input: 764.80 toks/s, output: 47.70 toks/s]
Processed prompts: 100%|██████████| 1/1 [00:00<00:00,  1.49it/s, est. speed input: 764.80 toks/s, output: 47.70 toks/s]

Rendering prompts:   0%|          | 0/1 [00:00<?, ?it/s]
Rendering prompts: 100%|██████████| 1/1 [00:00<00:00, 86.94it/s]

Processed prompts:   0%|          | 0/1 [00:00<?, ?it/s, est. speed input: 0.00 toks/s, output: 0.00 toks/s]
Processed prompts: 100%|██████████| 1/1 [00:00<00:00,  2.14it/s, est. speed input: 4393.30 toks/s, output: 2.14 toks/s]
Processed prompts: 100%|██████████| 1/1 [00:00<00:00,  2.14it/s, est. speed input: 4393.30 toks/s, output: 2.14 toks/s]
Processed prompts: 100%|██████████| 1/1 [00:00<00:00,  2.14it/s, est. speed input: 4393.30 toks/s, output: 2.14 toks/s]

Rendering prompts:   0%|          | 0/1 [00:00<?, ?it/s]
Rendering prompts: 100%|██████████| 1/1 [00:00<00:00, 73.51it/s]

Processed prompts:   0%|          | 0/1 [00:00<?, ?it/s, est. speed input: 0.00 toks/s, output: 0.00 toks/s]
Processed prompts: 100%|██████████| 1/1 [00:01<00:00,  1.13s/it, est. speed input: 1810.29 toks/s, output: 28.27 toks/s]
Processed prompts: 100%|██████████| 1/1 [00:01<00:00,  1.13s/it, est. speed input: 1810.29 toks/s, output: 28.27 toks/s]
Processed prompts: 100%|██████████| 1/1 [00:01<00:00,  1.13s/it, est. speed input: 1810.29 toks/s, output: 28.27 toks/s]

Rendering prompts:   0%|          | 0/4 [00:00<?, ?it/s]
Rendering prompts: 100%|██████████| 4/4 [00:00<00:00, 1229.91it/s]

Processed prompts:   0%|          | 0/4 [00:00<?, ?it/s, est. speed input: 0.00 toks/s, output: 0.00 toks/s]
Processed prompts:  25%|██▌       | 1/4 [00:01<00:04,  1.46s/it, est. speed input: 5.50 toks/s, output: 43.98 toks/s]
Processed prompts: 100%|██████████| 4/4 [00:01<00:00,  1.46s/it, est. speed input: 21.98 toks/s, output: 175.85 toks/s]
Processed prompts: 100%|██████████| 4/4 [00:01<00:00,  2.75it/s, est. speed input: 21.98 toks/s, output: 175.85 toks/s]

Rendering prompts:   0%|          | 0/1 [00:00<?, ?it/s]
Rendering prompts: 100%|██████████| 1/1 [00:00<00:00, 258.65it/s]

Processed prompts:   0%|          | 0/1 [00:00<?, ?it/s, est. speed input: 0.00 toks/s, output: 0.00 toks/s]
Processed prompts: 100%|██████████| 1/1 [00:00<00:00, 16.79it/s, est. speed input: 8623.30 toks/s, output: 16.80 toks/s]
Processed prompts: 100%|██████████| 1/1 [00:00<00:00, 16.67it/s, est. speed input: 8623.30 toks/s, output: 16.80 toks/s]

Rendering prompts:   0%|          | 0/1 [00:00<?, ?it/s]
Rendering prompts: 100%|██████████| 1/1 [00:00<00:00, 262.82it/s]

Processed prompts:   0%|          | 0/1 [00:00<?, ?it/s, est. speed input: 0.00 toks/s, output: 0.00 toks/s]
Processed prompts: 100%|██████████| 1/1 [00:00<00:00,  1.49it/s, est. speed input: 762.90 toks/s, output: 47.59 toks/s]
Processed prompts: 100%|██████████| 1/1 [00:00<00:00,  1.49it/s, est. speed input: 762.90 toks/s, output: 47.59 toks/s]
Processed prompts: 100%|██████████| 1/1 [00:00<00:00,  1.49it/s, est. speed input: 762.90 toks/s, output: 47.59 toks/s]

Rendering prompts:   0%|          | 0/1 [00:00<?, ?it/s]
Rendering prompts: 100%|██████████| 1/1 [00:00<00:00, 84.87it/s]

Processed prompts:   0%|          | 0/1 [00:00<?, ?it/s, est. speed input: 0.00 toks/s, output: 0.00 toks/s]
Processed prompts: 100%|██████████| 1/1 [00:00<00:00,  2.14it/s, est. speed input: 4391.10 toks/s, output: 2.14 toks/s]
Processed prompts: 100%|██████████| 1/1 [00:00<00:00,  2.14it/s, est. speed input: 4391.10 toks/s, output: 2.14 toks/s]
Processed prompts: 100%|██████████| 1/1 [00:00<00:00,  2.14it/s, est. speed input: 4391.10 toks/s, output: 2.14 toks/s]

Rendering prompts:   0%|          | 0/1 [00:00<?, ?it/s]
Rendering prompts: 100%|██████████| 1/1 [00:00<00:00, 76.13it/s]

Processed prompts:   0%|          | 0/1 [00:00<?, ?it/s, est. speed input: 0.00 toks/s, output: 0.00 toks/s]
Processed prompts: 100%|██████████| 1/1 [00:01<00:00,  1.10s/it, est. speed input: 1863.66 toks/s, output: 29.10 toks/s]
Processed prompts: 100%|██████████| 1/1 [00:01<00:00,  1.10s/it, est. speed input: 1863.66 toks/s, output: 29.10 toks/s]
Processed prompts: 100%|██████████| 1/1 [00:01<00:00,  1.10s/it, est. speed input: 1863.66 toks/s, output: 29.10 toks/s]
SPLIT[TRITON_ATTN-tp1] short_decode=184.1 (reps 175.3/184.1/188.1)
SPLIT[TRITON_ATTN-tp1] ctx512_prefill=7889.0 (reps 7620.9/7889.0/7904.3)
SPLIT[TRITON_ATTN-tp1] ctx512_decode=49.8 (reps 48.9/49.8/50.6)
SPLIT[TRITON_ATTN-tp1] ctx512_mixed=46.7 (reps 45.8/46.7/47.2)
SPLIT[TRITON_ATTN-tp1] ctx2048_prefill=4229.3 (reps 4212.2/4229.3/4268.9)
SPLIT[TRITON_ATTN-tp1] ctx2048_decode=47.8 (reps 46.4/47.8/48.5)
SPLIT[TRITON_ATTN-tp1] ctx2048_mixed=28.5 (reps 27.9/28.5/28.5)
SPLIT[TRITON_ATTN-tp1] done
INFO 09-07 00:16:41 [utils.py:620] [shutdown] Process manager: send sigterm to process EngineCore
(EngineCore pid=120971) INFO 09-07 00:16:41 [core.py:1350] [shutdown] EngineCore: trigger received signal=SIGTERM
(EngineCore pid=120971) INFO 09-07 00:16:41 [core.py:1501] [shutdown] EngineCore: start mode=abort timeout=0s
(EngineCore pid=120971) INFO 09-07 00:16:41 [core.py:1532] [shutdown] EngineCore: request processing complete; starting resource teardown
(EngineCore pid=120971) INFO 09-07 00:16:41 [core.py:1363] [shutdown] EngineCore: exiting busy loop
rc=0 TRITON_ATTN tp1
=== BRIDGE_ATTN tp=2 ===
gpu-lease: granted exclusive; starting: env CUDA_HOME=/opt/cuda FLASHINFER_DISABLE_VERSION_CHECK=1 /opt/repo/agentic-vllm/software/vllm/sm75-marlin/turing_lab/thirdparty/../../.venv/bin/python vllm_macro_gate.py --arm BRIDGE_ATTN 2
INFO 09-07 00:16:51 [api_utils.py:286] non-default args: {'dtype': 'float16', 'max_model_len': 4096, 'tensor_parallel_size': 2, 'enable_prefix_caching': False, 'gpu_memory_utilization': 0.9, 'disable_log_stats': True, 'enforce_eager': True, 'attention_backend': 'BRIDGE_ATTN', 'model': '/home/dconnolly/.cache/huggingface/hub/models--Qwen--Qwen2.5-1.5B-Instruct/snapshots/989aa7980e4cf806f80c7fef2b1adb7bc71aa306'}
INFO 09-07 00:16:51 [model.py:684] Resolved architecture: Qwen2ForCausalLM
WARNING 09-07 00:16:51 [model.py:2355] Casting torch.bfloat16 to torch.float16.
INFO 09-07 00:16:51 [model.py:2021] Using max model len 4096
INFO 09-07 00:16:51 [scheduler.py:242] Chunked prefill is enabled with max_num_batched_tokens=8192.
WARNING 09-07 00:16:51 [vllm.py:1322] Enforce eager set, disabling torch.compile and CUDAGraphs. This is equivalent to setting -cc.mode=none -cc.cudagraph_mode=none
WARNING 09-07 00:16:51 [vllm.py:1357] Inductor compilation was disabled by user settings, optimizations settings that are only active during inductor compilation will be ignored.
INFO 09-07 00:16:51 [kernel.py:365] Final IR op priority after setting platform defaults: IrOpPriorityConfig(rms_norm=['vllm_c', 'native'], fused_add_rms_norm=['vllm_c', 'native'])
INFO 09-07 00:16:51 [vllm.py:1536] Cudagraph is disabled under eager mode
INFO 09-07 00:16:51 [compilation.py:329] Enabled custom fusions: norm_quant, act_quant
(EngineCore pid=121171) INFO 09-07 00:16:53 [core.py:123] Initializing a V1 LLM engine (v0.28.1rc1.dev50+gcac8a75a7.d20260828) with config: model='/home/dconnolly/.cache/huggingface/hub/models--Qwen--Qwen2.5-1.5B-Instruct/snapshots/989aa7980e4cf806f80c7fef2b1adb7bc71aa306', speculative_config=None, tokenizer='/home/dconnolly/.cache/huggingface/hub/models--Qwen--Qwen2.5-1.5B-Instruct/snapshots/989aa7980e4cf806f80c7fef2b1adb7bc71aa306', skip_tokenizer_init=False, tokenizer_mode=auto, revision=None, tokenizer_revision=None, trust_remote_code=False, dtype=torch.float16, max_seq_len=4096, download_dir=None, load_format=auto, tensor_parallel_size=2, pipeline_parallel_size=1, data_parallel_size=1, decode_context_parallel_size=1, dcp_comm_backend=ag_rs, disable_custom_all_reduce=False, quantization=None, quantization_config=None, enforce_eager=True, enable_return_routed_experts=False, kv_cache_dtype=auto, device_config=cuda, structured_outputs_config=StructuredOutputsConfig(backend='auto', disable_any_whitespace=False, disable_additional_properties=False, reasoning_parser='', reasoning_parser_plugin='', enable_in_reasoning=False), observability_config=ObservabilityConfig(show_hidden_metrics_for_version=None, otlp_traces_endpoint=None, collect_detailed_traces=None, per_request_spec_decode_metrics='none', kv_cache_metrics=False, kv_cache_metrics_sample=0.01, cudagraph_metrics=False, enable_layerwise_nvtx_tracing=False, enable_mfu_metrics=False, enable_mm_processor_stats=False, enable_logging_iteration_details=False, jit_monitor_mode='warn', jit_monitor_verbose=False), seed=0, served_model_name=/home/dconnolly/.cache/huggingface/hub/models--Qwen--Qwen2.5-1.5B-Instruct/snapshots/989aa7980e4cf806f80c7fef2b1adb7bc71aa306, enable_prefix_caching=False, enable_chunked_prefill=True, pooler_config=None, compilation_config={'mode': <CompilationMode.NONE: 0>, 'debug_dump_path': None, 'cache_dir': '', 'compile_cache_save_format': 'binary', 'backend': 'inductor', 'custom_ops': ['all'], 'ir_enable_torch_wrap': False, 'splitting_ops': [], 'compile_mm_encoder': False, 'cudagraph_mm_encoder': False, 'encoder_cudagraph_token_budgets': [], 'encoder_cudagraph_max_vision_items_per_batch': 0, 'encoder_cudagraph_max_frames_per_batch': None, 'compile_sizes': [], 'compile_ranges_endpoints': [8192], 'inductor_compile_config': {'enable_auto_functionalized_v2': False, 'combo_kernels': True, 'benchmark_combo_kernel': True}, 'inductor_passes': {}, 'cudagraph_mode': <CUDAGraphMode.NONE: 0>, 'cudagraph_num_of_warmups': 0, 'cudagraph_capture_sizes': [], 'cudagraph_copy_inputs': False, 'cudagraph_specialize_lora': True, 'use_inductor_graph_partition': False, 'pass_config': {'fuse_norm_quant': True, 'fuse_act_quant': True, 'fuse_attn_quant': False, 'enable_sp': False, 'fuse_gemm_comms': False, 'fuse_allreduce_rms': False, 'enable_qk_norm_rope_fusion': False, 'fuse_rope_kvcache_cat_mla': False, 'fuse_act_padding': False, 'fuse_qk_norm_rope_kvcache': False}, 'max_cudagraph_capture_size': 0, 'dynamic_shapes_config': {'type': <DynamicShapesType.BACKED: 'backed'>, 'evaluate_guards': False, 'assume_32_bit_indexing': False}, 'local_cache_dir': None, 'fast_moe_cold_start': False, 'static_all_moe_layers': []}, kernel_config=KernelConfig(ir_op_priority=IrOpPriorityConfig(rms_norm=['vllm_c', 'native'], fused_add_rms_norm=['vllm_c', 'native']), enable_flashinfer_autotune=True, enable_cutedsl_warmup=True, enable_jit_warmup=True, enable_bf16x3_router_gemm=False, moe_backend='auto', linear_backend='auto')
(EngineCore pid=121171) INFO 09-07 00:16:53 [multiproc_executor.py:153] DP group leader: node_rank=0, node_rank_within_dp=0, master_addr=127.0.0.1, mq_connect_ip=192.168.21.44 (local), world_size=2, local_world_size=2
(Worker pid=121184) INFO 09-07 00:16:54 [parallel_state.py:1638] world_size=2 rank=0 local_rank=0 distributed_init_method=file:///tmp/vllm_dist_91107ef94f4647e28b744f56bbfa4aea backend=nccl
(Worker pid=121185) INFO 09-07 00:16:54 [parallel_state.py:1638] world_size=2 rank=1 local_rank=1 distributed_init_method=file:///tmp/vllm_dist_91107ef94f4647e28b744f56bbfa4aea backend=nccl
(Worker pid=121184) INFO 09-07 00:16:55 [pynccl.py:113] vLLM is using nccl==2.29.7
(Worker pid=121184) WARNING 09-07 00:16:56 [symm_mem.py:67] SymmMemCommunicator: Device capability 7.5 not supported, communicator is not available.
(Worker pid=121185) WARNING 09-07 00:16:56 [symm_mem.py:67] SymmMemCommunicator: Device capability 7.5 not supported, communicator is not available.
(Worker pid=121185) WARNING 09-07 00:16:56 [flashinfer_all_reduce.py:383] FlashInfer All Reduce is disabled because it is not supported for world_size=2.
(Worker pid=121184) WARNING 09-07 00:16:56 [flashinfer_all_reduce.py:383] FlashInfer All Reduce is disabled because it is not supported for world_size=2.
(Worker pid=121184) INFO 09-07 00:16:56 [cuda_communicator.py:269] Using ['CUSTOM', 'PYNCCL'] all-reduce backends (in dispatch order) for group 'tp:0' out of potential backends: ['FLASHINFER', 'NCCL_SYMM_MEM', 'QUICK_REDUCE', 'AITER_CUSTOM', 'CUSTOM', 'SYMM_MEM', 'PYNCCL'].
(Worker pid=121184) INFO 09-07 00:16:56 [parallel_state.py:1982] rank 0 in world size 2 is assigned as DP rank 0, PP rank 0, PCP rank 0, TP rank 0, EP rank N/A, EPLB rank N/A
(Worker pid=121184) INFO 09-07 00:16:56 [gpu_worker.py:399] Using V2 Model Runner
(Worker_TP0 pid=121184) INFO 09-07 00:16:57 [model_runner.py:375] Loading model from scratch...
(Worker_TP0 pid=121184) INFO 09-07 00:16:57 [cuda.py:436] Using AttentionBackendEnum.BRIDGE_ATTN backend.
(Worker_TP1 pid=121185) INFO 09-07 00:16:57 [cuda.py:436] Using AttentionBackendEnum.BRIDGE_ATTN backend.
(Worker_TP0 pid=121184) INFO 09-07 00:16:57 [weight_utils.py:858] Filesystem type for checkpoints: XFS. Checkpoint size: 2.88 GiB. Available RAM: 113.82 GiB.
(Worker_TP0 pid=121184) INFO 09-07 00:16:57 [weight_utils.py:881] Auto-prefetch is disabled because the filesystem (XFS) is not a recognized network FS (NFS/Lustre). If you want to force prefetching, start vLLM with --safetensors-load-strategy=prefetch.
(Worker_TP0 pid=121184) 
Loading safetensors checkpoint shards:   0% Completed | 0/1 [00:00<?, ?it/s]
(Worker_TP0 pid=121184) 
Loading safetensors checkpoint shards: 100% Completed | 1/1 [00:00<00:00,  1.34it/s]
(Worker_TP0 pid=121184) 
Loading safetensors checkpoint shards: 100% Completed | 1/1 [00:00<00:00,  1.34it/s]
(Worker_TP0 pid=121184) 
(Worker_TP0 pid=121184) INFO 09-07 00:16:58 [default_loader.py:430] Loading weights took 0.75 seconds
(Worker_TP0 pid=121184) INFO 09-07 00:16:59 [model_runner.py:397] Model loading took 1.5 GiB memory and 2.133028 seconds
(Worker_TP1 pid=121185) INFO 09-07 00:16:59 [model_runner.py:397] Model loading took 1.5 GiB memory and 2.159379 seconds
(Worker_TP0 pid=121184) WARNING 09-07 00:16:59 [topk_topp_sampler.py:69] FlashInfer top-p/top-k sampling unavailable: unsupported compute capability 7.5; falling back. Set VLLM_USE_FLASHINFER_SAMPLER=0 to silence.
(EngineCore pid=121171) INFO 09-07 00:16:59 [torch_utils.py:277] Reducing Torch threads from 8 to 1 for serving. Set OMP_NUM_THREADS in the external environment to override.
(EngineCore pid=121171) INFO 09-07 00:16:59 [utils.py:306] Using LBNHC KV cache layout.
(Worker_TP0 pid=121184) INFO 09-07 00:17:02 [gpu_worker.py:586] Available KV cache memory: 19.09 GiB
(EngineCore pid=121171) INFO 09-07 00:17:02 [kv_cache_utils.py:1879] GPU KV cache size: 1,428,896 tokens, Maximum concurrency for 4,096 tokens per request: 348.85x
(Worker_TP0 pid=121184) INFO 09-07 00:17:02 [kernel_warmup.py:124] JIT kernel warmup starting.
(Worker_TP0 pid=121184) INFO 09-07 00:17:02 [kernel_warmup.py:134] JIT kernel warmup finished in 0.00s.
(Worker_TP1 pid=121185) INFO 09-07 00:17:02 [kernel_warmup.py:124] JIT kernel warmup starting.
(Worker_TP1 pid=121185) INFO 09-07 00:17:02 [kernel_warmup.py:134] JIT kernel warmup finished in 0.00s.
(Worker_TP0 pid=121184) INFO 09-07 00:17:02 [gpu_worker.py:817] Free memory on device (23.09/23.47 GiB) on startup. Desired GPU memory utilization is (0.9, 21.12 GiB). Actual usage is 1.71 GiB for consumed memory (weights + non-torch), 0.32 GiB for peak activation, and 0.0 GiB for CUDAGraph memory. Replace gpu_memory_utilization config with `--kv-cache-memory=20335500596` (18.94 GiB) to fit into requested memory, or `--kv-cache-memory=22453807616` (20.91 GiB) to fully utilize gpu memory. Current kv cache memory in use is 19.09 GiB.
(Worker_TP1 pid=121185) INFO 09-07 00:17:02 [gpu_worker.py:817] Free memory on device (23.09/23.46 GiB) on startup. Desired GPU memory utilization is (0.9, 21.11 GiB). Actual usage is 1.71 GiB for consumed memory (weights + non-torch), 0.32 GiB for peak activation, and 0.0 GiB for CUDAGraph memory. Replace gpu_memory_utilization config with `--kv-cache-memory=20327420007` (18.93 GiB) to fit into requested memory, or `--kv-cache-memory=22444829184` (20.9 GiB) to fully utilize gpu memory. Current kv cache memory in use is 19.08 GiB.
(Worker_TP1 pid=121185) INFO 09-07 00:17:02 [bridge_attn.py:210] bridge_attn: attention executing on the cu_sm80_on_sm75 bridge kernel via flash-attn's sm_75 route
(Worker_TP0 pid=121184) INFO 09-07 00:17:02 [bridge_attn.py:210] bridge_attn: attention executing on the cu_sm80_on_sm75 bridge kernel via flash-attn's sm_75 route
(Worker_TP0 pid=121184) INFO 09-07 00:17:16 [jit_monitor.py:85] Kernel JIT monitor activated; monitored JIT compilations during inference will use mode=warn.
(Worker_TP1 pid=121185) INFO 09-07 00:17:16 [jit_monitor.py:85] Kernel JIT monitor activated; monitored JIT compilations during inference will use mode=warn.
(Worker_TP0 pid=121184) INFO 09-07 00:17:17 [torch_utils.py:277] Reducing Torch threads from 8 to 1 for serving. Set OMP_NUM_THREADS in the external environment to override.
(EngineCore pid=121171) INFO 09-07 00:17:17 [core.py:368] init engine (profile, create kv cache, warmup model) took 18.41 s
(EngineCore pid=121171) INFO 09-07 00:17:18 [kernel.py:365] Final IR op priority after setting platform defaults: IrOpPriorityConfig(rms_norm=['vllm_c', 'native'], fused_add_rms_norm=['vllm_c', 'native'])
INFO 09-07 00:17:19 [hf.py:547] Detected the chat template content format to be 'string'. You can set `--chat-template-content-format` to override this.

Rendering prompts:   0%|          | 0/1 [00:00<?, ?it/s]
Rendering prompts: 100%|██████████| 1/1 [00:00<00:00, 84.11it/s]

Processed prompts:   0%|          | 0/1 [00:00<?, ?it/s, est. speed input: 0.00 toks/s, output: 0.00 toks/s]
Processed prompts: 100%|██████████| 1/1 [00:00<00:00,  2.09it/s, est. speed input: 4.17 toks/s, output: 16.68 toks/s]
Processed prompts: 100%|██████████| 1/1 [00:00<00:00,  2.09it/s, est. speed input: 4.17 toks/s, output: 16.68 toks/s]
Processed prompts: 100%|██████████| 1/1 [00:00<00:00,  2.08it/s, est. speed input: 4.17 toks/s, output: 16.68 toks/s]

Rendering prompts:   0%|          | 0/8 [00:00<?, ?it/s]
Rendering prompts: 100%|██████████| 8/8 [00:00<00:00, 1196.71it/s]

Processed prompts:   0%|          | 0/8 [00:00<?, ?it/s, est. speed input: 0.00 toks/s, output: 0.00 toks/s]
Processed prompts:  12%|█▎        | 1/8 [00:21<02:31, 21.64s/it, est. speed input: 0.37 toks/s, output: 2.96 toks/s]
Processed prompts:  25%|██▌       | 2/8 [00:22<00:55,  9.30s/it, est. speed input: 0.72 toks/s, output: 5.74 toks/s]
Processed prompts: 100%|██████████| 8/8 [00:22<00:00,  9.30s/it, est. speed input: 2.87 toks/s, output: 22.97 toks/s]
Processed prompts: 100%|██████████| 8/8 [00:22<00:00,  2.79s/it, est. speed input: 2.87 toks/s, output: 22.97 toks/s]

Rendering prompts:   0%|          | 0/1 [00:00<?, ?it/s]
Rendering prompts: 100%|██████████| 1/1 [00:00<00:00, 238.19it/s]

Processed prompts:   0%|          | 0/1 [00:00<?, ?it/s, est. speed input: 0.00 toks/s, output: 0.00 toks/s]
Processed prompts: 100%|██████████| 1/1 [00:00<00:00,  6.69it/s, est. speed input: 3433.04 toks/s, output: 6.69 toks/s]
Processed prompts: 100%|██████████| 1/1 [00:00<00:00,  6.69it/s, est. speed input: 3433.04 toks/s, output: 6.69 toks/s]
Processed prompts: 100%|██████████| 1/1 [00:00<00:00,  6.67it/s, est. speed input: 3433.04 toks/s, output: 6.69 toks/s]

Rendering prompts:   0%|          | 0/1 [00:00<?, ?it/s]
Rendering prompts: 100%|██████████| 1/1 [00:00<00:00, 265.18it/s]

Processed prompts:   0%|          | 0/1 [00:00<?, ?it/s, est. speed input: 0.00 toks/s, output: 0.00 toks/s]
Processed prompts: 100%|██████████| 1/1 [00:02<00:00,  2.24s/it, est. speed input: 228.84 toks/s, output: 14.27 toks/s]
Processed prompts: 100%|██████████| 1/1 [00:02<00:00,  2.24s/it, est. speed input: 228.84 toks/s, output: 14.27 toks/s]
Processed prompts: 100%|██████████| 1/1 [00:02<00:00,  2.24s/it, est. speed input: 228.84 toks/s, output: 14.27 toks/s]

Rendering prompts:   0%|          | 0/1 [00:00<?, ?it/s]
Rendering prompts: 100%|██████████| 1/1 [00:00<00:00, 76.11it/s]

Processed prompts:   0%|          | 0/1 [00:00<?, ?it/s, est. speed input: 0.00 toks/s, output: 0.00 toks/s]
Processed prompts: 100%|██████████| 1/1 [00:00<00:00,  5.56it/s, est. speed input: 11391.02 toks/s, output: 5.56 toks/s]
Processed prompts: 100%|██████████| 1/1 [00:00<00:00,  5.56it/s, est. speed input: 11391.02 toks/s, output: 5.56 toks/s]
Processed prompts: 100%|██████████| 1/1 [00:00<00:00,  5.54it/s, est. speed input: 11391.02 toks/s, output: 5.56 toks/s]

Rendering prompts:   0%|          | 0/1 [00:00<?, ?it/s]
Rendering prompts: 100%|██████████| 1/1 [00:00<00:00, 81.71it/s]

Processed prompts:   0%|          | 0/1 [00:00<?, ?it/s, est. speed input: 0.00 toks/s, output: 0.00 toks/s]
Processed prompts: 100%|██████████| 1/1 [00:02<00:00,  2.37s/it, est. speed input: 866.16 toks/s, output: 13.53 toks/s]
Processed prompts: 100%|██████████| 1/1 [00:02<00:00,  2.37s/it, est. speed input: 866.16 toks/s, output: 13.53 toks/s]
Processed prompts: 100%|██████████| 1/1 [00:02<00:00,  2.37s/it, est. speed input: 866.16 toks/s, output: 13.53 toks/s]

Rendering prompts:   0%|          | 0/8 [00:00<?, ?it/s]
Rendering prompts: 100%|██████████| 8/8 [00:00<00:00, 1572.45it/s]

Processed prompts:   0%|          | 0/8 [00:00<?, ?it/s, est. speed input: 0.00 toks/s, output: 0.00 toks/s]
Processed prompts:  12%|█▎        | 1/8 [00:21<02:30, 21.50s/it, est. speed input: 0.37 toks/s, output: 2.98 toks/s]
Processed prompts:  25%|██▌       | 2/8 [00:22<00:54,  9.15s/it, est. speed input: 0.73 toks/s, output: 5.81 toks/s]
Processed prompts: 100%|██████████| 8/8 [00:22<00:00,  9.15s/it, est. speed input: 2.91 toks/s, output: 23.26 toks/s]
Processed prompts: 100%|██████████| 8/8 [00:22<00:00,  2.75s/it, est. speed input: 2.91 toks/s, output: 23.26 toks/s]

Rendering prompts:   0%|          | 0/1 [00:00<?, ?it/s]
Rendering prompts: 100%|██████████| 1/1 [00:00<00:00, 264.32it/s]

Processed prompts:   0%|          | 0/1 [00:00<?, ?it/s, est. speed input: 0.00 toks/s, output: 0.00 toks/s]
Processed prompts: 100%|██████████| 1/1 [00:00<00:00,  6.81it/s, est. speed input: 3496.83 toks/s, output: 6.82 toks/s]
Processed prompts: 100%|██████████| 1/1 [00:00<00:00,  6.81it/s, est. speed input: 3496.83 toks/s, output: 6.82 toks/s]
Processed prompts: 100%|██████████| 1/1 [00:00<00:00,  6.79it/s, est. speed input: 3496.83 toks/s, output: 6.82 toks/s]

Rendering prompts:   0%|          | 0/1 [00:00<?, ?it/s]
Rendering prompts: 100%|██████████| 1/1 [00:00<00:00, 262.05it/s]

Processed prompts:   0%|          | 0/1 [00:00<?, ?it/s, est. speed input: 0.00 toks/s, output: 0.00 toks/s]
Processed prompts: 100%|██████████| 1/1 [00:02<00:00,  2.07s/it, est. speed input: 248.27 toks/s, output: 15.49 toks/s]
Processed prompts: 100%|██████████| 1/1 [00:02<00:00,  2.07s/it, est. speed input: 248.27 toks/s, output: 15.49 toks/s]
Processed prompts: 100%|██████████| 1/1 [00:02<00:00,  2.07s/it, est. speed input: 248.27 toks/s, output: 15.49 toks/s]

Rendering prompts:   0%|          | 0/1 [00:00<?, ?it/s]
Rendering prompts: 100%|██████████| 1/1 [00:00<00:00, 81.25it/s]

Processed prompts:   0%|          | 0/1 [00:00<?, ?it/s, est. speed input: 0.00 toks/s, output: 0.00 toks/s]
Processed prompts: 100%|██████████| 1/1 [00:00<00:00,  5.57it/s, est. speed input: 11407.57 toks/s, output: 5.57 toks/s]
Processed prompts: 100%|██████████| 1/1 [00:00<00:00,  5.57it/s, est. speed input: 11407.57 toks/s, output: 5.57 toks/s]
Processed prompts: 100%|██████████| 1/1 [00:00<00:00,  5.55it/s, est. speed input: 11407.57 toks/s, output: 5.57 toks/s]

Rendering prompts:   0%|          | 0/1 [00:00<?, ?it/s]
Rendering prompts: 100%|██████████| 1/1 [00:00<00:00, 80.94it/s]

Processed prompts:   0%|          | 0/1 [00:00<?, ?it/s, est. speed input: 0.00 toks/s, output: 0.00 toks/s]
Processed prompts: 100%|██████████| 1/1 [00:02<00:00,  2.38s/it, est. speed input: 861.56 toks/s, output: 13.46 toks/s]
Processed prompts: 100%|██████████| 1/1 [00:02<00:00,  2.38s/it, est. speed input: 861.56 toks/s, output: 13.46 toks/s]
Processed prompts: 100%|██████████| 1/1 [00:02<00:00,  2.38s/it, est. speed input: 861.56 toks/s, output: 13.46 toks/s]

Rendering prompts:   0%|          | 0/8 [00:00<?, ?it/s]
Rendering prompts: 100%|██████████| 8/8 [00:00<00:00, 1797.24it/s]

Processed prompts:   0%|          | 0/8 [00:00<?, ?it/s, est. speed input: 0.00 toks/s, output: 0.00 toks/s]
Processed prompts:  12%|█▎        | 1/8 [00:20<02:26, 20.98s/it, est. speed input: 0.38 toks/s, output: 3.05 toks/s]
Processed prompts:  25%|██▌       | 2/8 [00:21<00:54,  9.03s/it, est. speed input: 0.74 toks/s, output: 5.91 toks/s]
Processed prompts: 100%|██████████| 8/8 [00:21<00:00,  9.03s/it, est. speed input: 2.96 toks/s, output: 23.65 toks/s]
Processed prompts: 100%|██████████| 8/8 [00:21<00:00,  2.71s/it, est. speed input: 2.96 toks/s, output: 23.65 toks/s]

Rendering prompts:   0%|          | 0/1 [00:00<?, ?it/s]
Rendering prompts: 100%|██████████| 1/1 [00:00<00:00, 232.73it/s]

Processed prompts:   0%|          | 0/1 [00:00<?, ?it/s, est. speed input: 0.00 toks/s, output: 0.00 toks/s]
Processed prompts: 100%|██████████| 1/1 [00:00<00:00,  6.70it/s, est. speed input: 3437.17 toks/s, output: 6.70 toks/s]
Processed prompts: 100%|██████████| 1/1 [00:00<00:00,  6.70it/s, est. speed input: 3437.17 toks/s, output: 6.70 toks/s]
Processed prompts: 100%|██████████| 1/1 [00:00<00:00,  6.68it/s, est. speed input: 3437.17 toks/s, output: 6.70 toks/s]

Rendering prompts:   0%|          | 0/1 [00:00<?, ?it/s]
Rendering prompts: 100%|██████████| 1/1 [00:00<00:00, 297.05it/s]

Processed prompts:   0%|          | 0/1 [00:00<?, ?it/s, est. speed input: 0.00 toks/s, output: 0.00 toks/s]
Processed prompts: 100%|██████████| 1/1 [00:02<00:00,  2.42s/it, est. speed input: 212.09 toks/s, output: 13.23 toks/s]
Processed prompts: 100%|██████████| 1/1 [00:02<00:00,  2.42s/it, est. speed input: 212.09 toks/s, output: 13.23 toks/s]
Processed prompts: 100%|██████████| 1/1 [00:02<00:00,  2.42s/it, est. speed input: 212.09 toks/s, output: 13.23 toks/s]

Rendering prompts:   0%|          | 0/1 [00:00<?, ?it/s]
Rendering prompts: 100%|██████████| 1/1 [00:00<00:00, 81.76it/s]

Processed prompts:   0%|          | 0/1 [00:00<?, ?it/s, est. speed input: 0.00 toks/s, output: 0.00 toks/s]
Processed prompts: 100%|██████████| 1/1 [00:00<00:00, 12.91it/s, est. speed input: 26470.76 toks/s, output: 12.91 toks/s]
Processed prompts: 100%|██████████| 1/1 [00:00<00:00, 12.86it/s, est. speed input: 26470.76 toks/s, output: 12.91 toks/s]

Rendering prompts:   0%|          | 0/1 [00:00<?, ?it/s]
Rendering prompts: 100%|██████████| 1/1 [00:00<00:00, 82.18it/s]

Processed prompts:   0%|          | 0/1 [00:00<?, ?it/s, est. speed input: 0.00 toks/s, output: 0.00 toks/s]
Processed prompts: 100%|██████████| 1/1 [00:02<00:00,  2.18s/it, est. speed input: 938.87 toks/s, output: 14.66 toks/s]
Processed prompts: 100%|██████████| 1/1 [00:02<00:00,  2.18s/it, est. speed input: 938.87 toks/s, output: 14.66 toks/s]
Processed prompts: 100%|██████████| 1/1 [00:02<00:00,  2.18s/it, est. speed input: 938.87 toks/s, output: 14.66 toks/s]

Rendering prompts:   0%|          | 0/8 [00:00<?, ?it/s]
Rendering prompts: 100%|██████████| 8/8 [00:00<00:00, 2793.87it/s]

Processed prompts:   0%|          | 0/8 [00:00<?, ?it/s, est. speed input: 0.00 toks/s, output: 0.00 toks/s]
Processed prompts:  12%|█▎        | 1/8 [00:21<02:28, 21.17s/it, est. speed input: 0.38 toks/s, output: 3.02 toks/s]
Processed prompts:  25%|██▌       | 2/8 [00:21<00:53,  8.89s/it, est. speed input: 0.75 toks/s, output: 5.96 toks/s]
Processed prompts: 100%|██████████| 8/8 [00:21<00:00,  8.89s/it, est. speed input: 2.97 toks/s, output: 23.75 toks/s]
Processed prompts: 100%|██████████| 8/8 [00:21<00:00,  2.69s/it, est. speed input: 2.97 toks/s, output: 23.75 toks/s]

Rendering prompts:   0%|          | 0/1 [00:00<?, ?it/s]
Rendering prompts: 100%|██████████| 1/1 [00:00<00:00, 592.50it/s]

Processed prompts:   0%|          | 0/1 [00:00<?, ?it/s, est. speed input: 0.00 toks/s, output: 0.00 toks/s]
Processed prompts: 100%|██████████| 1/1 [00:00<00:00,  6.66it/s, est. speed input: 3415.99 toks/s, output: 6.66 toks/s]
Processed prompts: 100%|██████████| 1/1 [00:00<00:00,  6.66it/s, est. speed input: 3415.99 toks/s, output: 6.66 toks/s]
Processed prompts: 100%|██████████| 1/1 [00:00<00:00,  6.65it/s, est. speed input: 3415.99 toks/s, output: 6.66 toks/s]

Rendering prompts:   0%|          | 0/1 [00:00<?, ?it/s]
Rendering prompts: 100%|██████████| 1/1 [00:00<00:00, 463.51it/s]

Processed prompts:   0%|          | 0/1 [00:00<?, ?it/s, est. speed input: 0.00 toks/s, output: 0.00 toks/s]
Processed prompts: 100%|██████████| 1/1 [00:02<00:00,  2.19s/it, est. speed input: 234.22 toks/s, output: 14.61 toks/s]
Processed prompts: 100%|██████████| 1/1 [00:02<00:00,  2.19s/it, est. speed input: 234.22 toks/s, output: 14.61 toks/s]
Processed prompts: 100%|██████████| 1/1 [00:02<00:00,  2.19s/it, est. speed input: 234.22 toks/s, output: 14.61 toks/s]

Rendering prompts:   0%|          | 0/1 [00:00<?, ?it/s]
Rendering prompts: 100%|██████████| 1/1 [00:00<00:00, 82.33it/s]

Processed prompts:   0%|          | 0/1 [00:00<?, ?it/s, est. speed input: 0.00 toks/s, output: 0.00 toks/s]
Processed prompts: 100%|██████████| 1/1 [00:00<00:00,  5.51it/s, est. speed input: 11291.25 toks/s, output: 5.51 toks/s]
Processed prompts: 100%|██████████| 1/1 [00:00<00:00,  5.51it/s, est. speed input: 11291.25 toks/s, output: 5.51 toks/s]
Processed prompts: 100%|██████████| 1/1 [00:00<00:00,  5.49it/s, est. speed input: 11291.25 toks/s, output: 5.51 toks/s]

Rendering prompts:   0%|          | 0/1 [00:00<?, ?it/s]
Rendering prompts: 100%|██████████| 1/1 [00:00<00:00, 80.09it/s]

Processed prompts:   0%|          | 0/1 [00:00<?, ?it/s, est. speed input: 0.00 toks/s, output: 0.00 toks/s]
Processed prompts: 100%|██████████| 1/1 [00:01<00:00,  1.85s/it, est. speed input: 1104.76 toks/s, output: 17.25 toks/s]
Processed prompts: 100%|██████████| 1/1 [00:01<00:00,  1.85s/it, est. speed input: 1104.76 toks/s, output: 17.25 toks/s]
Processed prompts: 100%|██████████| 1/1 [00:01<00:00,  1.86s/it, est. speed input: 1104.76 toks/s, output: 17.25 toks/s]

Rendering prompts:   0%|          | 0/8 [00:00<?, ?it/s]
Rendering prompts: 100%|██████████| 8/8 [00:00<00:00, 1153.03it/s]

Processed prompts:   0%|          | 0/8 [00:00<?, ?it/s, est. speed input: 0.00 toks/s, output: 0.00 toks/s]
Processed prompts:  12%|█▎        | 1/8 [00:22<02:36, 22.29s/it, est. speed input: 0.36 toks/s, output: 2.87 toks/s]
Processed prompts:  25%|██▌       | 2/8 [00:22<00:57,  9.56s/it, est. speed input: 0.70 toks/s, output: 5.58 toks/s]
Processed prompts: 100%|██████████| 8/8 [00:22<00:00,  9.56s/it, est. speed input: 2.79 toks/s, output: 22.32 toks/s]
Processed prompts: 100%|██████████| 8/8 [00:22<00:00,  2.87s/it, est. speed input: 2.79 toks/s, output: 22.32 toks/s]

Rendering prompts:   0%|          | 0/1 [00:00<?, ?it/s]
Rendering prompts: 100%|██████████| 1/1 [00:00<00:00, 267.51it/s]

Processed prompts:   0%|          | 0/1 [00:00<?, ?it/s, est. speed input: 0.00 toks/s, output: 0.00 toks/s]
Processed prompts: 100%|██████████| 1/1 [00:00<00:00,  6.58it/s, est. speed input: 3376.25 toks/s, output: 6.58 toks/s]
Processed prompts: 100%|██████████| 1/1 [00:00<00:00,  6.58it/s, est. speed input: 3376.25 toks/s, output: 6.58 toks/s]
Processed prompts: 100%|██████████| 1/1 [00:00<00:00,  6.56it/s, est. speed input: 3376.25 toks/s, output: 6.58 toks/s]

Rendering prompts:   0%|          | 0/1 [00:00<?, ?it/s]
Rendering prompts: 100%|██████████| 1/1 [00:00<00:00, 261.20it/s]

Processed prompts:   0%|          | 0/1 [00:00<?, ?it/s, est. speed input: 0.00 toks/s, output: 0.00 toks/s]
Processed prompts: 100%|██████████| 1/1 [00:02<00:00,  2.10s/it, est. speed input: 244.24 toks/s, output: 15.23 toks/s]
Processed prompts: 100%|██████████| 1/1 [00:02<00:00,  2.10s/it, est. speed input: 244.24 toks/s, output: 15.23 toks/s]
Processed prompts: 100%|██████████| 1/1 [00:02<00:00,  2.10s/it, est. speed input: 244.24 toks/s, output: 15.23 toks/s]

Rendering prompts:   0%|          | 0/1 [00:00<?, ?it/s]
Rendering prompts: 100%|██████████| 1/1 [00:00<00:00, 110.77it/s]

Processed prompts:   0%|          | 0/1 [00:00<?, ?it/s, est. speed input: 0.00 toks/s, output: 0.00 toks/s]
Processed prompts: 100%|██████████| 1/1 [00:00<00:00,  5.59it/s, est. speed input: 11460.14 toks/s, output: 5.59 toks/s]
Processed prompts: 100%|██████████| 1/1 [00:00<00:00,  5.59it/s, est. speed input: 11460.14 toks/s, output: 5.59 toks/s]
Processed prompts: 100%|██████████| 1/1 [00:00<00:00,  5.58it/s, est. speed input: 11460.14 toks/s, output: 5.59 toks/s]

Rendering prompts:   0%|          | 0/1 [00:00<?, ?it/s]
Rendering prompts: 100%|██████████| 1/1 [00:00<00:00, 175.32it/s]

Processed prompts:   0%|          | 0/1 [00:00<?, ?it/s, est. speed input: 0.00 toks/s, output: 0.00 toks/s]
Processed prompts: 100%|██████████| 1/1 [00:02<00:00,  2.31s/it, est. speed input: 888.85 toks/s, output: 13.88 toks/s]
Processed prompts: 100%|██████████| 1/1 [00:02<00:00,  2.31s/it, est. speed input: 888.85 toks/s, output: 13.88 toks/s]
Processed prompts: 100%|██████████| 1/1 [00:02<00:00,  2.31s/it, est. speed input: 888.85 toks/s, output: 13.88 toks/s]
SPLIT[BRIDGE_ATTN-tp2] short_decode=23.0 (reps 22.3/23.0/23.3)
SPLIT[BRIDGE_ATTN-tp2] ctx512_prefill=3296.9 (reps 3258.3/3296.9/3297.5)
SPLIT[BRIDGE_ATTN-tp2] ctx512_decode=14.8 (reps 13.7/14.8/15.2)
SPLIT[BRIDGE_ATTN-tp2] ctx512_mixed=14.2 (reps 13.2/14.2/14.6)
SPLIT[BRIDGE_ATTN-tp2] ctx2048_prefill=10529.3 (reps 10484.9/10529.3/10587.8)
SPLIT[BRIDGE_ATTN-tp2] ctx2048_decode=14.2 (reps 14.1/14.2/14.6)
SPLIT[BRIDGE_ATTN-tp2] ctx2048_mixed=13.4 (reps 13.4/13.4/13.8)
SPLIT[BRIDGE_ATTN-tp2] done
INFO 09-07 00:19:33 [utils.py:620] [shutdown] Process manager: send sigterm to process EngineCore
(EngineCore pid=121171) INFO 09-07 00:19:33 [core.py:1350] [shutdown] EngineCore: trigger received signal=SIGTERM
(EngineCore pid=121171) INFO 09-07 00:19:33 [core.py:1501] [shutdown] EngineCore: start mode=abort timeout=0s
(EngineCore pid=121171) INFO 09-07 00:19:33 [core.py:1532] [shutdown] EngineCore: request processing complete; starting resource teardown
(EngineCore pid=121171) INFO 09-07 00:19:33 [core.py:1363] [shutdown] EngineCore: exiting busy loop
(EngineCore pid=121171) INFO 09-07 00:19:33 [multiproc_executor.py:472] [shutdown] Executor: waiting for worker exit count=2
(Worker_TP0 pid=121184) INFO 09-07 00:19:33 [multiproc_executor.py:836] Parent process exited, terminating worker queues
(EngineCore pid=121171) INFO 09-07 00:19:35 [multiproc_executor.py:479] [shutdown] Executor: all workers exited gracefully
rc=0 BRIDGE_ATTN tp2
=== TRITON_ATTN tp=2 ===
gpu-lease: granted exclusive; starting: env CUDA_HOME=/opt/cuda FLASHINFER_DISABLE_VERSION_CHECK=1 /opt/repo/agentic-vllm/software/vllm/sm75-marlin/turing_lab/thirdparty/../../.venv/bin/python vllm_macro_gate.py --arm TRITON_ATTN 2
INFO 09-07 00:19:44 [api_utils.py:286] non-default args: {'dtype': 'float16', 'max_model_len': 4096, 'tensor_parallel_size': 2, 'enable_prefix_caching': False, 'gpu_memory_utilization': 0.9, 'disable_log_stats': True, 'enforce_eager': True, 'attention_backend': 'TRITON_ATTN', 'model': '/home/dconnolly/.cache/huggingface/hub/models--Qwen--Qwen2.5-1.5B-Instruct/snapshots/989aa7980e4cf806f80c7fef2b1adb7bc71aa306'}
INFO 09-07 00:19:44 [model.py:684] Resolved architecture: Qwen2ForCausalLM
WARNING 09-07 00:19:44 [model.py:2355] Casting torch.bfloat16 to torch.float16.
INFO 09-07 00:19:44 [model.py:2021] Using max model len 4096
INFO 09-07 00:19:44 [scheduler.py:242] Chunked prefill is enabled with max_num_batched_tokens=8192.
WARNING 09-07 00:19:44 [vllm.py:1322] Enforce eager set, disabling torch.compile and CUDAGraphs. This is equivalent to setting -cc.mode=none -cc.cudagraph_mode=none
WARNING 09-07 00:19:44 [vllm.py:1357] Inductor compilation was disabled by user settings, optimizations settings that are only active during inductor compilation will be ignored.
INFO 09-07 00:19:44 [kernel.py:365] Final IR op priority after setting platform defaults: IrOpPriorityConfig(rms_norm=['vllm_c', 'native'], fused_add_rms_norm=['vllm_c', 'native'])
INFO 09-07 00:19:44 [vllm.py:1536] Cudagraph is disabled under eager mode
INFO 09-07 00:19:44 [compilation.py:329] Enabled custom fusions: norm_quant, act_quant
(EngineCore pid=121513) INFO 09-07 00:19:47 [core.py:123] Initializing a V1 LLM engine (v0.28.1rc1.dev50+gcac8a75a7.d20260828) with config: model='/home/dconnolly/.cache/huggingface/hub/models--Qwen--Qwen2.5-1.5B-Instruct/snapshots/989aa7980e4cf806f80c7fef2b1adb7bc71aa306', speculative_config=None, tokenizer='/home/dconnolly/.cache/huggingface/hub/models--Qwen--Qwen2.5-1.5B-Instruct/snapshots/989aa7980e4cf806f80c7fef2b1adb7bc71aa306', skip_tokenizer_init=False, tokenizer_mode=auto, revision=None, tokenizer_revision=None, trust_remote_code=False, dtype=torch.float16, max_seq_len=4096, download_dir=None, load_format=auto, tensor_parallel_size=2, pipeline_parallel_size=1, data_parallel_size=1, decode_context_parallel_size=1, dcp_comm_backend=ag_rs, disable_custom_all_reduce=False, quantization=None, quantization_config=None, enforce_eager=True, enable_return_routed_experts=False, kv_cache_dtype=auto, device_config=cuda, structured_outputs_config=StructuredOutputsConfig(backend='auto', disable_any_whitespace=False, disable_additional_properties=False, reasoning_parser='', reasoning_parser_plugin='', enable_in_reasoning=False), observability_config=ObservabilityConfig(show_hidden_metrics_for_version=None, otlp_traces_endpoint=None, collect_detailed_traces=None, per_request_spec_decode_metrics='none', kv_cache_metrics=False, kv_cache_metrics_sample=0.01, cudagraph_metrics=False, enable_layerwise_nvtx_tracing=False, enable_mfu_metrics=False, enable_mm_processor_stats=False, enable_logging_iteration_details=False, jit_monitor_mode='warn', jit_monitor_verbose=False), seed=0, served_model_name=/home/dconnolly/.cache/huggingface/hub/models--Qwen--Qwen2.5-1.5B-Instruct/snapshots/989aa7980e4cf806f80c7fef2b1adb7bc71aa306, enable_prefix_caching=False, enable_chunked_prefill=True, pooler_config=None, compilation_config={'mode': <CompilationMode.NONE: 0>, 'debug_dump_path': None, 'cache_dir': '', 'compile_cache_save_format': 'binary', 'backend': 'inductor', 'custom_ops': ['all'], 'ir_enable_torch_wrap': False, 'splitting_ops': [], 'compile_mm_encoder': False, 'cudagraph_mm_encoder': False, 'encoder_cudagraph_token_budgets': [], 'encoder_cudagraph_max_vision_items_per_batch': 0, 'encoder_cudagraph_max_frames_per_batch': None, 'compile_sizes': [], 'compile_ranges_endpoints': [8192], 'inductor_compile_config': {'enable_auto_functionalized_v2': False, 'combo_kernels': True, 'benchmark_combo_kernel': True}, 'inductor_passes': {}, 'cudagraph_mode': <CUDAGraphMode.NONE: 0>, 'cudagraph_num_of_warmups': 0, 'cudagraph_capture_sizes': [], 'cudagraph_copy_inputs': False, 'cudagraph_specialize_lora': True, 'use_inductor_graph_partition': False, 'pass_config': {'fuse_norm_quant': True, 'fuse_act_quant': True, 'fuse_attn_quant': False, 'enable_sp': False, 'fuse_gemm_comms': False, 'fuse_allreduce_rms': False, 'enable_qk_norm_rope_fusion': False, 'fuse_rope_kvcache_cat_mla': False, 'fuse_act_padding': False, 'fuse_qk_norm_rope_kvcache': False}, 'max_cudagraph_capture_size': 0, 'dynamic_shapes_config': {'type': <DynamicShapesType.BACKED: 'backed'>, 'evaluate_guards': False, 'assume_32_bit_indexing': False}, 'local_cache_dir': None, 'fast_moe_cold_start': False, 'static_all_moe_layers': []}, kernel_config=KernelConfig(ir_op_priority=IrOpPriorityConfig(rms_norm=['vllm_c', 'native'], fused_add_rms_norm=['vllm_c', 'native']), enable_flashinfer_autotune=True, enable_cutedsl_warmup=True, enable_jit_warmup=True, enable_bf16x3_router_gemm=False, moe_backend='auto', linear_backend='auto')
(EngineCore pid=121513) INFO 09-07 00:19:47 [multiproc_executor.py:153] DP group leader: node_rank=0, node_rank_within_dp=0, master_addr=127.0.0.1, mq_connect_ip=192.168.21.44 (local), world_size=2, local_world_size=2
(Worker pid=121526) INFO 09-07 00:19:47 [parallel_state.py:1638] world_size=2 rank=0 local_rank=0 distributed_init_method=file:///tmp/vllm_dist_7137983507d44190a505be3c2c423811 backend=nccl
(Worker pid=121527) INFO 09-07 00:19:47 [parallel_state.py:1638] world_size=2 rank=1 local_rank=1 distributed_init_method=file:///tmp/vllm_dist_7137983507d44190a505be3c2c423811 backend=nccl
(Worker pid=121526) INFO 09-07 00:19:49 [pynccl.py:113] vLLM is using nccl==2.29.7
(Worker pid=121526) WARNING 09-07 00:19:49 [symm_mem.py:67] SymmMemCommunicator: Device capability 7.5 not supported, communicator is not available.
(Worker pid=121527) WARNING 09-07 00:19:49 [symm_mem.py:67] SymmMemCommunicator: Device capability 7.5 not supported, communicator is not available.
(Worker pid=121527) WARNING 09-07 00:19:49 [flashinfer_all_reduce.py:383] FlashInfer All Reduce is disabled because it is not supported for world_size=2.
(Worker pid=121526) WARNING 09-07 00:19:49 [flashinfer_all_reduce.py:383] FlashInfer All Reduce is disabled because it is not supported for world_size=2.
(Worker pid=121526) INFO 09-07 00:19:49 [cuda_communicator.py:269] Using ['CUSTOM', 'PYNCCL'] all-reduce backends (in dispatch order) for group 'tp:0' out of potential backends: ['FLASHINFER', 'NCCL_SYMM_MEM', 'QUICK_REDUCE', 'AITER_CUSTOM', 'CUSTOM', 'SYMM_MEM', 'PYNCCL'].
(Worker pid=121526) INFO 09-07 00:19:49 [parallel_state.py:1982] rank 0 in world size 2 is assigned as DP rank 0, PP rank 0, PCP rank 0, TP rank 0, EP rank N/A, EPLB rank N/A
(Worker pid=121526) INFO 09-07 00:19:49 [gpu_worker.py:399] Using V2 Model Runner
(Worker_TP0 pid=121526) INFO 09-07 00:19:50 [model_runner.py:375] Loading model from scratch...
(Worker_TP1 pid=121527) INFO 09-07 00:19:50 [cuda.py:436] Using AttentionBackendEnum.TRITON_ATTN backend.
(Worker_TP0 pid=121526) INFO 09-07 00:19:50 [cuda.py:436] Using AttentionBackendEnum.TRITON_ATTN backend.
(Worker_TP0 pid=121526) INFO 09-07 00:19:50 [weight_utils.py:858] Filesystem type for checkpoints: XFS. Checkpoint size: 2.88 GiB. Available RAM: 113.22 GiB.
(Worker_TP0 pid=121526) INFO 09-07 00:19:50 [weight_utils.py:881] Auto-prefetch is disabled because the filesystem (XFS) is not a recognized network FS (NFS/Lustre). If you want to force prefetching, start vLLM with --safetensors-load-strategy=prefetch.
(Worker_TP0 pid=121526) 
Loading safetensors checkpoint shards:   0% Completed | 0/1 [00:00<?, ?it/s]
(Worker_TP0 pid=121526) 
Loading safetensors checkpoint shards: 100% Completed | 1/1 [00:00<00:00,  1.91it/s]
(Worker_TP0 pid=121526) 
Loading safetensors checkpoint shards: 100% Completed | 1/1 [00:00<00:00,  1.91it/s]
(Worker_TP0 pid=121526) 
(Worker_TP0 pid=121526) INFO 09-07 00:19:51 [default_loader.py:430] Loading weights took 0.53 seconds
(Worker_TP0 pid=121526) INFO 09-07 00:19:52 [model_runner.py:397] Model loading took 1.5 GiB memory and 1.982419 seconds
(Worker_TP0 pid=121526) WARNING 09-07 00:19:52 [topk_topp_sampler.py:69] FlashInfer top-p/top-k sampling unavailable: unsupported compute capability 7.5; falling back. Set VLLM_USE_FLASHINFER_SAMPLER=0 to silence.
(Worker_TP1 pid=121527) INFO 09-07 00:19:52 [model_runner.py:397] Model loading took 1.5 GiB memory and 2.091061 seconds
(EngineCore pid=121513) INFO 09-07 00:19:52 [torch_utils.py:277] Reducing Torch threads from 8 to 1 for serving. Set OMP_NUM_THREADS in the external environment to override.
(EngineCore pid=121513) INFO 09-07 00:19:52 [utils.py:306] Using LBNHC KV cache layout.
(Worker_TP0 pid=121526) INFO 09-07 00:19:55 [gpu_worker.py:586] Available KV cache memory: 19.09 GiB
(EngineCore pid=121513) INFO 09-07 00:19:55 [kv_cache_utils.py:1879] GPU KV cache size: 1,428,896 tokens, Maximum concurrency for 4,096 tokens per request: 348.85x
(Worker_TP1 pid=121527) INFO 09-07 00:19:55 [kernel_warmup.py:124] JIT kernel warmup starting.
(Worker_TP1 pid=121527) INFO 09-07 00:19:55 [kernel_warmup.py:134] JIT kernel warmup finished in 0.00s.
(Worker_TP0 pid=121526) INFO 09-07 00:19:55 [kernel_warmup.py:124] JIT kernel warmup starting.
(Worker_TP0 pid=121526) INFO 09-07 00:19:55 [kernel_warmup.py:134] JIT kernel warmup finished in 0.00s.
(Worker_TP0 pid=121526) INFO 09-07 00:19:55 [gpu_worker.py:817] Free memory on device (23.09/23.47 GiB) on startup. Desired GPU memory utilization is (0.9, 21.12 GiB). Actual usage is 1.71 GiB for consumed memory (weights + non-torch), 0.32 GiB for peak activation, and 0.0 GiB for CUDAGraph memory. Replace gpu_memory_utilization config with `--kv-cache-memory=20335500596` (18.94 GiB) to fit into requested memory, or `--kv-cache-memory=22453807616` (20.91 GiB) to fully utilize gpu memory. Current kv cache memory in use is 19.09 GiB.
(Worker_TP1 pid=121527) INFO 09-07 00:19:55 [gpu_worker.py:817] Free memory on device (23.09/23.46 GiB) on startup. Desired GPU memory utilization is (0.9, 21.11 GiB). Actual usage is 1.71 GiB for consumed memory (weights + non-torch), 0.32 GiB for peak activation, and 0.0 GiB for CUDAGraph memory. Replace gpu_memory_utilization config with `--kv-cache-memory=20327420007` (18.93 GiB) to fit into requested memory, or `--kv-cache-memory=22444829184` (20.9 GiB) to fully utilize gpu memory. Current kv cache memory in use is 19.08 GiB.
(Worker_TP0 pid=121526) INFO 09-07 00:19:56 [jit_monitor.py:85] Kernel JIT monitor activated; monitored JIT compilations during inference will use mode=warn.
(Worker_TP1 pid=121527) INFO 09-07 00:19:56 [jit_monitor.py:85] Kernel JIT monitor activated; monitored JIT compilations during inference will use mode=warn.
(Worker_TP0 pid=121526) INFO 09-07 00:19:56 [torch_utils.py:277] Reducing Torch threads from 8 to 1 for serving. Set OMP_NUM_THREADS in the external environment to override.
(EngineCore pid=121513) INFO 09-07 00:19:56 [core.py:368] init engine (profile, create kv cache, warmup model) took 4.69 s
(EngineCore pid=121513) INFO 09-07 00:19:58 [kernel.py:365] Final IR op priority after setting platform defaults: IrOpPriorityConfig(rms_norm=['vllm_c', 'native'], fused_add_rms_norm=['vllm_c', 'native'])
INFO 09-07 00:19:58 [hf.py:547] Detected the chat template content format to be 'string'. You can set `--chat-template-content-format` to override this.

Rendering prompts:   0%|          | 0/1 [00:00<?, ?it/s]
Rendering prompts: 100%|██████████| 1/1 [00:00<00:00, 44.63it/s]

Processed prompts:   0%|          | 0/1 [00:00<?, ?it/s, est. speed input: 0.00 toks/s, output: 0.00 toks/s](Worker_TP0 pid=121526) WARNING 09-07 00:19:58 [jit_monitor.py:141] Triton kernel JIT compilation during inference: kernel_unified_attention. This causes a latency spike; consider extending warmup to cover this shape/config.

Processed prompts: 100%|██████████| 1/1 [00:00<00:00,  3.15it/s, est. speed input: 6.30 toks/s, output: 25.21 toks/s]
Processed prompts: 100%|██████████| 1/1 [00:00<00:00,  3.15it/s, est. speed input: 6.30 toks/s, output: 25.21 toks/s]
Processed prompts: 100%|██████████| 1/1 [00:00<00:00,  3.15it/s, est. speed input: 6.30 toks/s, output: 25.21 toks/s]

Rendering prompts:   0%|          | 0/8 [00:00<?, ?it/s]
Rendering prompts: 100%|██████████| 8/8 [00:00<00:00, 1240.51it/s]

Processed prompts:   0%|          | 0/8 [00:00<?, ?it/s, est. speed input: 0.00 toks/s, output: 0.00 toks/s]
Processed prompts:  12%|█▎        | 1/8 [00:01<00:11,  1.61s/it, est. speed input: 4.98 toks/s, output: 39.87 toks/s]
Processed prompts: 100%|██████████| 8/8 [00:01<00:00,  1.61s/it, est. speed input: 38.59 toks/s, output: 308.75 toks/s]
Processed prompts: 100%|██████████| 8/8 [00:01<00:00,  4.82it/s, est. speed input: 38.59 toks/s, output: 308.75 toks/s]

Rendering prompts:   0%|          | 0/1 [00:00<?, ?it/s]
Rendering prompts: 100%|██████████| 1/1 [00:00<00:00, 535.81it/s]

Processed prompts:   0%|          | 0/1 [00:00<?, ?it/s, est. speed input: 0.00 toks/s, output: 0.00 toks/s]
Processed prompts: 100%|██████████| 1/1 [00:00<00:00, 26.10it/s, est. speed input: 13402.75 toks/s, output: 26.12 toks/s]
Processed prompts: 100%|██████████| 1/1 [00:00<00:00, 25.99it/s, est. speed input: 13402.75 toks/s, output: 26.12 toks/s]

Rendering prompts:   0%|          | 0/1 [00:00<?, ?it/s]
Rendering prompts: 100%|██████████| 1/1 [00:00<00:00, 481.27it/s]

Processed prompts:   0%|          | 0/1 [00:00<?, ?it/s, est. speed input: 0.00 toks/s, output: 0.00 toks/s]
Processed prompts: 100%|██████████| 1/1 [00:00<00:00,  1.18it/s, est. speed input: 602.92 toks/s, output: 37.61 toks/s]
Processed prompts: 100%|██████████| 1/1 [00:00<00:00,  1.18it/s, est. speed input: 602.92 toks/s, output: 37.61 toks/s]
Processed prompts: 100%|██████████| 1/1 [00:00<00:00,  1.17it/s, est. speed input: 602.92 toks/s, output: 37.61 toks/s]

Rendering prompts:   0%|          | 0/1 [00:00<?, ?it/s]
Rendering prompts: 100%|██████████| 1/1 [00:00<00:00, 73.00it/s]

Processed prompts:   0%|          | 0/1 [00:00<?, ?it/s, est. speed input: 0.00 toks/s, output: 0.00 toks/s]
Processed prompts: 100%|██████████| 1/1 [00:00<00:00,  3.87it/s, est. speed input: 7933.02 toks/s, output: 3.87 toks/s]
Processed prompts: 100%|██████████| 1/1 [00:00<00:00,  3.87it/s, est. speed input: 7933.02 toks/s, output: 3.87 toks/s]
Processed prompts: 100%|██████████| 1/1 [00:00<00:00,  3.86it/s, est. speed input: 7933.02 toks/s, output: 3.87 toks/s]

Rendering prompts:   0%|          | 0/1 [00:00<?, ?it/s]
Rendering prompts: 100%|██████████| 1/1 [00:00<00:00, 86.93it/s]

Processed prompts:   0%|          | 0/1 [00:00<?, ?it/s, est. speed input: 0.00 toks/s, output: 0.00 toks/s]
Processed prompts: 100%|██████████| 1/1 [00:00<00:00,  1.03it/s, est. speed input: 2102.00 toks/s, output: 32.83 toks/s]
Processed prompts: 100%|██████████| 1/1 [00:00<00:00,  1.03it/s, est. speed input: 2102.00 toks/s, output: 32.83 toks/s]
Processed prompts: 100%|██████████| 1/1 [00:00<00:00,  1.03it/s, est. speed input: 2102.00 toks/s, output: 32.83 toks/s]

Rendering prompts:   0%|          | 0/8 [00:00<?, ?it/s]
Rendering prompts: 100%|██████████| 8/8 [00:00<00:00, 1218.92it/s]

Processed prompts:   0%|          | 0/8 [00:00<?, ?it/s, est. speed input: 0.00 toks/s, output: 0.00 toks/s]
Processed prompts:  12%|█▎        | 1/8 [00:01<00:12,  1.73s/it, est. speed input: 4.64 toks/s, output: 37.09 toks/s]
Processed prompts: 100%|██████████| 8/8 [00:01<00:00,  1.73s/it, est. speed input: 36.08 toks/s, output: 288.62 toks/s]
Processed prompts: 100%|██████████| 8/8 [00:01<00:00,  4.51it/s, est. speed input: 36.08 toks/s, output: 288.62 toks/s]

Rendering prompts:   0%|          | 0/1 [00:00<?, ?it/s]
Rendering prompts: 100%|██████████| 1/1 [00:00<00:00, 264.04it/s]

Processed prompts:   0%|          | 0/1 [00:00<?, ?it/s, est. speed input: 0.00 toks/s, output: 0.00 toks/s]
Processed prompts: 100%|██████████| 1/1 [00:00<00:00, 27.10it/s, est. speed input: 13927.89 toks/s, output: 27.13 toks/s]
Processed prompts: 100%|██████████| 1/1 [00:00<00:00, 26.88it/s, est. speed input: 13927.89 toks/s, output: 27.13 toks/s]

Rendering prompts:   0%|          | 0/1 [00:00<?, ?it/s]
Rendering prompts: 100%|██████████| 1/1 [00:00<00:00, 253.31it/s]

Processed prompts:   0%|          | 0/1 [00:00<?, ?it/s, est. speed input: 0.00 toks/s, output: 0.00 toks/s]
Processed prompts: 100%|██████████| 1/1 [00:00<00:00,  1.29it/s, est. speed input: 662.40 toks/s, output: 41.32 toks/s]
Processed prompts: 100%|██████████| 1/1 [00:00<00:00,  1.29it/s, est. speed input: 662.40 toks/s, output: 41.32 toks/s]
Processed prompts: 100%|██████████| 1/1 [00:00<00:00,  1.29it/s, est. speed input: 662.40 toks/s, output: 41.32 toks/s]

Rendering prompts:   0%|          | 0/1 [00:00<?, ?it/s]
Rendering prompts: 100%|██████████| 1/1 [00:00<00:00, 82.61it/s]

Processed prompts:   0%|          | 0/1 [00:00<?, ?it/s, est. speed input: 0.00 toks/s, output: 0.00 toks/s]
Processed prompts: 100%|██████████| 1/1 [00:00<00:00,  3.87it/s, est. speed input: 7925.03 toks/s, output: 3.87 toks/s]
Processed prompts: 100%|██████████| 1/1 [00:00<00:00,  3.87it/s, est. speed input: 7925.03 toks/s, output: 3.87 toks/s]
Processed prompts: 100%|██████████| 1/1 [00:00<00:00,  3.86it/s, est. speed input: 7925.03 toks/s, output: 3.87 toks/s]

Rendering prompts:   0%|          | 0/1 [00:00<?, ?it/s]
Rendering prompts: 100%|██████████| 1/1 [00:00<00:00, 80.94it/s]

Processed prompts:   0%|          | 0/1 [00:00<?, ?it/s, est. speed input: 0.00 toks/s, output: 0.00 toks/s]
Processed prompts: 100%|██████████| 1/1 [00:00<00:00,  1.02it/s, est. speed input: 2081.13 toks/s, output: 32.50 toks/s]
Processed prompts: 100%|██████████| 1/1 [00:00<00:00,  1.02it/s, est. speed input: 2081.13 toks/s, output: 32.50 toks/s]
Processed prompts: 100%|██████████| 1/1 [00:00<00:00,  1.02it/s, est. speed input: 2081.13 toks/s, output: 32.50 toks/s]

Rendering prompts:   0%|          | 0/8 [00:00<?, ?it/s]
Rendering prompts: 100%|██████████| 8/8 [00:00<00:00, 1299.00it/s]

Processed prompts:   0%|          | 0/8 [00:00<?, ?it/s, est. speed input: 0.00 toks/s, output: 0.00 toks/s]
Processed prompts:  12%|█▎        | 1/8 [00:01<00:10,  1.57s/it, est. speed input: 5.09 toks/s, output: 40.75 toks/s]
Processed prompts: 100%|██████████| 8/8 [00:01<00:00,  1.57s/it, est. speed input: 39.51 toks/s, output: 316.08 toks/s]
Processed prompts: 100%|██████████| 8/8 [00:01<00:00,  4.94it/s, est. speed input: 39.51 toks/s, output: 316.08 toks/s]

Rendering prompts:   0%|          | 0/1 [00:00<?, ?it/s]
Rendering prompts: 100%|██████████| 1/1 [00:00<00:00, 253.82it/s]

Processed prompts:   0%|          | 0/1 [00:00<?, ?it/s, est. speed input: 0.00 toks/s, output: 0.00 toks/s]
Processed prompts: 100%|██████████| 1/1 [00:00<00:00, 27.21it/s, est. speed input: 13980.92 toks/s, output: 27.24 toks/s]
Processed prompts: 100%|██████████| 1/1 [00:00<00:00, 26.94it/s, est. speed input: 13980.92 toks/s, output: 27.24 toks/s]

Rendering prompts:   0%|          | 0/1 [00:00<?, ?it/s]
Rendering prompts: 100%|██████████| 1/1 [00:00<00:00, 204.85it/s]

Processed prompts:   0%|          | 0/1 [00:00<?, ?it/s, est. speed input: 0.00 toks/s, output: 0.00 toks/s]
Processed prompts: 100%|██████████| 1/1 [00:00<00:00,  1.23it/s, est. speed input: 629.18 toks/s, output: 39.25 toks/s]
Processed prompts: 100%|██████████| 1/1 [00:00<00:00,  1.23it/s, est. speed input: 629.18 toks/s, output: 39.25 toks/s]
Processed prompts: 100%|██████████| 1/1 [00:00<00:00,  1.23it/s, est. speed input: 629.18 toks/s, output: 39.25 toks/s]

Rendering prompts:   0%|          | 0/1 [00:00<?, ?it/s]
Rendering prompts: 100%|██████████| 1/1 [00:00<00:00, 83.42it/s]

Processed prompts:   0%|          | 0/1 [00:00<?, ?it/s, est. speed input: 0.00 toks/s, output: 0.00 toks/s]
Processed prompts: 100%|██████████| 1/1 [00:00<00:00,  3.87it/s, est. speed input: 7938.23 toks/s, output: 3.87 toks/s]
Processed prompts: 100%|██████████| 1/1 [00:00<00:00,  3.87it/s, est. speed input: 7938.23 toks/s, output: 3.87 toks/s]
Processed prompts: 100%|██████████| 1/1 [00:00<00:00,  3.86it/s, est. speed input: 7938.23 toks/s, output: 3.87 toks/s]

Rendering prompts:   0%|          | 0/1 [00:00<?, ?it/s]
Rendering prompts: 100%|██████████| 1/1 [00:00<00:00, 89.54it/s]

Processed prompts:   0%|          | 0/1 [00:00<?, ?it/s, est. speed input: 0.00 toks/s, output: 0.00 toks/s]
Processed prompts: 100%|██████████| 1/1 [00:01<00:00,  1.00s/it, est. speed input: 2044.25 toks/s, output: 31.93 toks/s]
Processed prompts: 100%|██████████| 1/1 [00:01<00:00,  1.00s/it, est. speed input: 2044.25 toks/s, output: 31.93 toks/s]
Processed prompts: 100%|██████████| 1/1 [00:01<00:00,  1.00s/it, est. speed input: 2044.25 toks/s, output: 31.93 toks/s]

Rendering prompts:   0%|          | 0/8 [00:00<?, ?it/s]
Rendering prompts: 100%|██████████| 8/8 [00:00<00:00, 1244.46it/s]

Processed prompts:   0%|          | 0/8 [00:00<?, ?it/s, est. speed input: 0.00 toks/s, output: 0.00 toks/s]
Processed prompts:  12%|█▎        | 1/8 [00:01<00:10,  1.55s/it, est. speed input: 5.17 toks/s, output: 41.37 toks/s]
Processed prompts: 100%|██████████| 8/8 [00:01<00:00,  1.55s/it, est. speed input: 40.14 toks/s, output: 321.09 toks/s]
Processed prompts: 100%|██████████| 8/8 [00:01<00:00,  5.02it/s, est. speed input: 40.14 toks/s, output: 321.09 toks/s]

Rendering prompts:   0%|          | 0/1 [00:00<?, ?it/s]
Rendering prompts: 100%|██████████| 1/1 [00:00<00:00, 237.52it/s]

Processed prompts:   0%|          | 0/1 [00:00<?, ?it/s, est. speed input: 0.00 toks/s, output: 0.00 toks/s]
Processed prompts: 100%|██████████| 1/1 [00:00<00:00, 26.80it/s, est. speed input: 13769.77 toks/s, output: 26.82 toks/s]
Processed prompts: 100%|██████████| 1/1 [00:00<00:00, 26.60it/s, est. speed input: 13769.77 toks/s, output: 26.82 toks/s]

Rendering prompts:   0%|          | 0/1 [00:00<?, ?it/s]
Rendering prompts: 100%|██████████| 1/1 [00:00<00:00, 227.96it/s]

Processed prompts:   0%|          | 0/1 [00:00<?, ?it/s, est. speed input: 0.00 toks/s, output: 0.00 toks/s]
Processed prompts: 100%|██████████| 1/1 [00:00<00:00,  1.31it/s, est. speed input: 673.18 toks/s, output: 41.99 toks/s]
Processed prompts: 100%|██████████| 1/1 [00:00<00:00,  1.31it/s, est. speed input: 673.18 toks/s, output: 41.99 toks/s]
Processed prompts: 100%|██████████| 1/1 [00:00<00:00,  1.31it/s, est. speed input: 673.18 toks/s, output: 41.99 toks/s]

Rendering prompts:   0%|          | 0/1 [00:00<?, ?it/s]
Rendering prompts: 100%|██████████| 1/1 [00:00<00:00, 197.32it/s]

Processed prompts:   0%|          | 0/1 [00:00<?, ?it/s, est. speed input: 0.00 toks/s, output: 0.00 toks/s]
Processed prompts: 100%|██████████| 1/1 [00:00<00:00,  3.90it/s, est. speed input: 7984.37 toks/s, output: 3.90 toks/s]
Processed prompts: 100%|██████████| 1/1 [00:00<00:00,  3.90it/s, est. speed input: 7984.37 toks/s, output: 3.90 toks/s]
Processed prompts: 100%|██████████| 1/1 [00:00<00:00,  3.89it/s, est. speed input: 7984.37 toks/s, output: 3.90 toks/s]

Rendering prompts:   0%|          | 0/1 [00:00<?, ?it/s]
Rendering prompts: 100%|██████████| 1/1 [00:00<00:00, 227.64it/s]

Processed prompts:   0%|          | 0/1 [00:00<?, ?it/s, est. speed input: 0.00 toks/s, output: 0.00 toks/s]
Processed prompts: 100%|██████████| 1/1 [00:00<00:00,  1.03it/s, est. speed input: 2117.73 toks/s, output: 33.07 toks/s]
Processed prompts: 100%|██████████| 1/1 [00:00<00:00,  1.03it/s, est. speed input: 2117.73 toks/s, output: 33.07 toks/s]
Processed prompts: 100%|██████████| 1/1 [00:00<00:00,  1.03it/s, est. speed input: 2117.73 toks/s, output: 33.07 toks/s]

Rendering prompts:   0%|          | 0/8 [00:00<?, ?it/s]
Rendering prompts: 100%|██████████| 8/8 [00:00<00:00, 2564.34it/s]

Processed prompts:   0%|          | 0/8 [00:00<?, ?it/s, est. speed input: 0.00 toks/s, output: 0.00 toks/s]
Processed prompts:  12%|█▎        | 1/8 [00:01<00:10,  1.55s/it, est. speed input: 5.17 toks/s, output: 41.39 toks/s]
Processed prompts: 100%|██████████| 8/8 [00:01<00:00,  1.55s/it, est. speed input: 40.13 toks/s, output: 321.02 toks/s]
Processed prompts: 100%|██████████| 8/8 [00:01<00:00,  5.01it/s, est. speed input: 40.13 toks/s, output: 321.02 toks/s]

Rendering prompts:   0%|          | 0/1 [00:00<?, ?it/s]
Rendering prompts: 100%|██████████| 1/1 [00:00<00:00, 252.88it/s]

Processed prompts:   0%|          | 0/1 [00:00<?, ?it/s, est. speed input: 0.00 toks/s, output: 0.00 toks/s]
Processed prompts: 100%|██████████| 1/1 [00:00<00:00, 26.98it/s, est. speed input: 13859.89 toks/s, output: 27.00 toks/s]
Processed prompts: 100%|██████████| 1/1 [00:00<00:00, 26.75it/s, est. speed input: 13859.89 toks/s, output: 27.00 toks/s]

Rendering prompts:   0%|          | 0/1 [00:00<?, ?it/s]
Rendering prompts: 100%|██████████| 1/1 [00:00<00:00, 258.14it/s]

Processed prompts:   0%|          | 0/1 [00:00<?, ?it/s, est. speed input: 0.00 toks/s, output: 0.00 toks/s]
Processed prompts: 100%|██████████| 1/1 [00:00<00:00,  1.20it/s, est. speed input: 614.71 toks/s, output: 38.34 toks/s]
Processed prompts: 100%|██████████| 1/1 [00:00<00:00,  1.20it/s, est. speed input: 614.71 toks/s, output: 38.34 toks/s]
Processed prompts: 100%|██████████| 1/1 [00:00<00:00,  1.20it/s, est. speed input: 614.71 toks/s, output: 38.34 toks/s]

Rendering prompts:   0%|          | 0/1 [00:00<?, ?it/s]
Rendering prompts: 100%|██████████| 1/1 [00:00<00:00, 86.38it/s]

Processed prompts:   0%|          | 0/1 [00:00<?, ?it/s, est. speed input: 0.00 toks/s, output: 0.00 toks/s]
Processed prompts: 100%|██████████| 1/1 [00:00<00:00,  3.88it/s, est. speed input: 7952.03 toks/s, output: 3.88 toks/s]
Processed prompts: 100%|██████████| 1/1 [00:00<00:00,  3.88it/s, est. speed input: 7952.03 toks/s, output: 3.88 toks/s]
Processed prompts: 100%|██████████| 1/1 [00:00<00:00,  3.87it/s, est. speed input: 7952.03 toks/s, output: 3.88 toks/s]

Rendering prompts:   0%|          | 0/1 [00:00<?, ?it/s]
Rendering prompts: 100%|██████████| 1/1 [00:00<00:00, 89.25it/s]

Processed prompts:   0%|          | 0/1 [00:00<?, ?it/s, est. speed input: 0.00 toks/s, output: 0.00 toks/s]
Processed prompts: 100%|██████████| 1/1 [00:00<00:00,  1.02it/s, est. speed input: 2081.92 toks/s, output: 32.51 toks/s]
Processed prompts: 100%|██████████| 1/1 [00:00<00:00,  1.02it/s, est. speed input: 2081.92 toks/s, output: 32.51 toks/s]
Processed prompts: 100%|██████████| 1/1 [00:00<00:00,  1.02it/s, est. speed input: 2081.92 toks/s, output: 32.51 toks/s]
SPLIT[TRITON_ATTN-tp2] short_decode=307.3 (reps 287.4/307.3/314.7)
SPLIT[TRITON_ATTN-tp2] ctx512_prefill=12094.6 (reps 11956.3/12094.6/12200.6)
SPLIT[TRITON_ATTN-tp2] ctx512_decode=38.9 (reps 38.1/38.9/39.7)
SPLIT[TRITON_ATTN-tp2] ctx512_mixed=38.1 (reps 37.5/38.1/38.9)
SPLIT[TRITON_ATTN-tp2] ctx2048_prefill=7522.4 (reps 7492.1/7522.4/7538.2)
SPLIT[TRITON_ATTN-tp2] ctx2048_decode=42.7 (reps 41.7/42.7/42.7)
SPLIT[TRITON_ATTN-tp2] ctx2048_mixed=32.1 (reps 31.5/32.1/32.1)
SPLIT[TRITON_ATTN-tp2] done
INFO 09-07 00:20:17 [utils.py:620] [shutdown] Process manager: send sigterm to process EngineCore
(EngineCore pid=121513) INFO 09-07 00:20:17 [core.py:1350] [shutdown] EngineCore: trigger received signal=SIGTERM
(EngineCore pid=121513) INFO 09-07 00:20:17 [core.py:1501] [shutdown] EngineCore: start mode=abort timeout=0s
(EngineCore pid=121513) INFO 09-07 00:20:17 [core.py:1532] [shutdown] EngineCore: request processing complete; starting resource teardown
(EngineCore pid=121513) INFO 09-07 00:20:17 [core.py:1363] [shutdown] EngineCore: exiting busy loop
(EngineCore pid=121513) INFO 09-07 00:20:17 [multiproc_executor.py:472] [shutdown] Executor: waiting for worker exit count=2
(Worker_TP0 pid=121526) INFO 09-07 00:20:17 [multiproc_executor.py:836] Parent process exited, terminating worker queues
(EngineCore pid=121513) INFO 09-07 00:20:19 [multiproc_executor.py:479] [shutdown] Executor: all workers exited gracefully
rc=0 TRITON_ATTN tp2
ALL-ARMS-DONE

