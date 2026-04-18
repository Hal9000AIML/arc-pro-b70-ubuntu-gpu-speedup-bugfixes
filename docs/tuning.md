# B70 tuning reference

Flags, env vars, and what they actually do. Everything here is measured — if the commentary disagrees with your bench, trust your bench and open an issue.

## Env vars

| Var | Value | Applies to | Why |
|---|---|---|---|
| `GGML_SYCL_DISABLE_OPT` | `1` | SYCL | Disables fused reorder-MMVQ path. Fixes MoE slot-init SEGV and rare dense reorder crashes. Costs ~5% on dense, negligible on MoE. |
| `UR_L0_ENABLE_RELAXED_ALLOCATION_LIMITS` | `1` | SYCL (Level Zero) | Lets llama-server allocate KV buffers >4GB in a single slab. Without this, long-context loads fail on 32GB B70. |
| `ZES_ENABLE_SYSMAN` | `1` | SYCL | Enables sysman telemetry (GPU power/freq visible to xpu-smi). Harmless. |
| `SYCL_CACHE_PERSISTENT` | **unset** | SYCL | Do NOT set to 1. Poisons cross-restart kernel cache on B70; causes SEGV on next boot. Let JIT recompile each time. |
| `VK_DRIVER_FILES` | *(auto)* | Vulkan | Leave unset; Mesa picks the right ICD. Only set if you're forcing a specific driver. |

## llama-server flags that matter on B70

| Flag | Recommended | Why |
|---|---|---|
| `-ngl 999` | always | Offload all layers to GPU. 32GB is plenty for 30B Q5 + KV. |
| `-c` | `16384` – `32768` | B70's 32GB fits generous context. Don't go higher than you need; KV eats VRAM. |
| `--batch-size 2048 --ubatch-size 512` | yes | Measured sweet spot on B70 for pp throughput. Higher batch helps ingestion, ubatch bounded by compute-engine width. |
| `--parallel` | `1` | Multiple parallel slots split VRAM and hurt TG for single-stream workloads. Bump to 2 only if you truly serve concurrent users. |
| `-fa 0` / `--flash-attn on` | see backend-selection.md | FA is stable on Vulkan, fragile on SYCL MoE. |
| `--defrag-thold 0.1` | yes | Aggressive KV defrag. On long-lived servers without this, VRAM fragments and inference stalls after a few hundred requests. |
| `--no-warmup` | yes | Warmup runs a throwaway inference which sometimes JITs the wrong kernel. Warm via first real request instead. |
| `-t 1` | yes | GPU path uses 1 host thread. More threads hurt — they fight for GPU submission queue. |
| `--jinja --reasoning off` | per-model | Pick correct chat template. `reasoning off` for Qwen3 non-thinking variants. |

## KV cache quantization

| KV quant | VRAM vs f16 | Quality loss | When to use |
|---|---|---|---|
| `f16` (default) | 100% | none | default |
| `q8_0` | 50% | imperceptible | long context, or co-tenant GPU |
| `q4_0` | 25% | measurable | only for >64k context on small cards |

Use `--cache-type-k q8_0 --cache-type-v q8_0` for the agentic tier example in `scripts/start_agentic_36.sh`.

## Flash attention vs `-fa 0` — measure before trusting

Intuition says `--flash-attn on` always helps. On B70 SYCL, **it can regress token-generation** even for dense models where it compiles cleanly.

Measured 2026-04-18, gemma-4-26B-A4B Q8_0 on SYCL1, `GGML_SYCL_DISABLE_OPT=1`, 4× B70 live deployment, 3 warmed runs of identical 20-tok prompt / 300 max tokens:

| Flag | tg tok/s | pp tok/s |
|---|---|---|
| `-fa 0` (baseline) | **26.47** | 74.2 |
| `--flash-attn on` | 25.98 (-1.8%) | 81.6 (+10%) |

FA helps prompt processing and hurts token generation on this specific stack. If your workload is bulk prompt ingestion (RAG, summarization of large documents), FA is a win. If it's interactive chat (users waiting on streaming tokens), `-fa 0` wins. We default to `-fa 0` on dense SYCL for this reason.

MoE SYCL: don't even try FA — it crashes at slot init, documented separately in `backend-selection.md` Rule 5.

Vulkan: FA is generally fine and positive on B70 (Mesa 26+ has solid coopmat support). Our agentic tier uses `--flash-attn on` on Vulkan.

**Rule of thumb:** always bench your specific model + quant + prompt shape. FA has more variance than any other single flag on B70.

## Speculative decoding

Tested pattern: Qwen3-0.6B-Q8 as draft for Qwen3.6-35B-A3B target on Vulkan. Gives 1.5–2.5× tg on acceptance, neutral on reject.

Draft flags:
```
--model-draft /mnt/models/Qwen3-0.6B-Q8_0.gguf
--device-draft Vulkan0             # same GPU as target on Vulkan is fine
-ngld 999
--draft-max 16 --draft-min 1 --draft-p-min 0.5
```

Don't use SYCL for spec decode on B70 — kernel-cache contention between the two models kills the server.

## Thermals

B70 cooler handles 225W TDP fine in a well-ventilated case. Expect:
- idle: 40–50°C
- chat tier (dense, bursty): 60–70°C
- reasoning tier (sustained MoE on 80B): 75–85°C — warm but safe

If a card is consistently >90°C, check airflow between cards. Two adjacent B70s with no gap can hotbox.
