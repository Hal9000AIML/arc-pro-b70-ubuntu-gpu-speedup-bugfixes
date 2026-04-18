# Benchmarks

All numbers are single-stream generation of the Fibonacci prompt:

> Write an iterative Python function that computes the nth Fibonacci number.

with `max_tokens=300`, `temperature=0.1`, `/no_think` for Qwen3. Values are `timings.predicted_per_second` from llama.cpp's `/v1/chat/completions` response.

## 2026-04-18 reference run (4× B70 box, 5 concurrent servers)

| Port | Model | GPU | Backend | pp tok/s | tg tok/s |
|---|---|---|---|---|---|
| 8000 | gemma-4-26B-A4B Q8_0 | 1 | SYCL | 68 | 26.4 |
| 8001 | Qwen3-Coder-30B-A3B Q5_K_M | 3 | SYCL (DISABLE_OPT) | 50 | 57.7 |
| 8002 | Qwen3-4B-Instruct Q6_K | 3 | Vulkan | 90 | 33.0 |
| 8003 | Qwen3.6-35B-A3B Q6_K_XL + 0.6B draft | 0 | Vulkan (spec) | 81 | 25.0 |
| 8004 | Qwen3-Next-80B-A3B IQ3_XXS | 2 | SYCL (DISABLE_OPT) | 36 | 21.2 |

GPU3 is shared between :8001 (code) and :8002 (fast). Both are benchmarked with the other server idle (warm but not generating). Concurrent generation on both degrades each by ~30%.

## Regression guardrails

If your numbers are dramatically below these, check in order:

1. **Mesa version** — Vulkan perf regresses hard without the `kisak/kisak-mesa` PPA. `apt-cache policy mesa-vulkan-drivers` should show 26.0.5+.
2. **SYCL cache poisoning** — if a SYCL tier is 5–10× slower than expected right after a restart, `rm -rf ~/.cache/libsycl_cache ~/.cache/neo_compiler_cache` and restart. Most common after kernel or driver update.
3. **Env vars** — confirm `GGML_SYCL_DISABLE_OPT=1` set (otherwise MoE SEGVs or dense runs reorder-buggy kernels).
4. **Two servers on one GPU, both SYCL** — 10× penalty. One of them must be Vulkan. See backend-selection.md Rule 4.
5. **Flash attention on SYCL MoE** — try `-fa 0`. FA + SYCL + MoE is a known crash path on B70.

## Head-to-head vs vLLM TP=1 (same card, same model)

2026-04-18, GPU3 evacuated (:8001 + :8002 both stopped), Qwen3-Coder-30B-A3B on single B70.

**vLLM setup** (container `intel/vllm:0.17.0-xpu`, vllm `0.1.dev14456+gde3f7fe65`, pytorch `2.10.0+xpu`):

```bash
docker run -d --name vllm-b70-017 \
  --device=/dev/dri \
  -v /mnt/models:/llm/models:ro \
  --ipc=host --network host \
  --entrypoint sleep intel/vllm:0.17.0-xpu infinity

docker exec -d vllm-b70-017 bash -c '
export ZE_AFFINITY_MASK=3
export VLLM_USE_V1=1
export VLLM_WORKER_MULTIPROC_METHOD=spawn
export VLLM_ALLOW_LONG_MAX_MODEL_LEN=1
export PYTORCH_ALLOC_CONF=expandable_segments:True
vllm serve /llm/models/Qwen3-Coder-30B-A3B-GPTQ \
  --served-model-name Qwen3-Coder-30B-vllm \
  --host 0.0.0.0 --port 8101 \
  --tensor-parallel-size 1 --max-model-len 16384 \
  --gpu-memory-utilization 0.80 --max-num-seqs 4 \
  --enforce-eager --trust-remote-code \
  --block-size 64 --dtype float16
'
```

Model loads in 93s. Application startup complete. 2 warmup calls, then 3× bench runs of the standard Fibonacci prompt with `max_tokens=300, temperature=0.1`.

**llama.cpp setup:** start_coder_sycl.sh from scripts/ (see Rule 2 in backend-selection.md).

| Engine | Run 1 | Run 2 | Run 3 | Avg tok/s |
|---|---|---|---|---|
| llama.cpp SYCL + cherry-picks, Q5_K_M | 59.90 | 59.91 | 59.06 | **59.6** |
| vLLM 0.17.0-xpu TP=1, GPTQ-Int4 | 13.85 | 13.84 | 13.85 | **13.85** |

llama.cpp is **4.3× faster** than vLLM TP=1 on this hardware for this model.

Why: vLLM XPU requires `--enforce-eager` (XPU Graph not supported in PyTorch 2.10), emits `WARNING: Currently, the 4-bit gptq_gemm kernel for GPTQ is buggy. Please switch to gptq_marlin.` (Marlin is CUDA-only), and lacks the MoE MMVQ fusion and K-quant native-subgroup-size patches we cherry-pick. vLLM's TP=4 sharding path beats this kit on a single big model (540 tok/s documented on Qwen3.5-27B TP=4) — TP=1 is vLLM's weakest case on B70.

## Solo-GPU numbers (no co-tenant)

For reference, these are the same tiers when they own their GPU:

| Model | Backend | Solo tg | Co-tenant tg | Penalty |
|---|---|---|---|---|
| Qwen3-Coder-30B Q5_K_M | SYCL | 60 | 57.7 | 4% (if pair is Vulkan 4B) |
| Qwen3-Coder-30B Q5_K_M | SYCL | 60 | 7 | 88% (if pair is SYCL anything) |
| gemma-4-26B Q8_0 | SYCL | 27 | 26.4 | 2% |
| Qwen3-Next-80B IQ3_XXS | SYCL | 22 | 21.2 | 4% |

The lesson: SYCL + SYCL on one card is catastrophic. SYCL + Vulkan on one card is fine if the Vulkan one is small.
