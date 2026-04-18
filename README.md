# arc-pro-b70-ubuntu-llm-inference-kit

**Ubuntu Server tuning kit that makes Intel Arc Pro B70 cards actually fast for local LLM inference.**

Out-of-the-box `llama.cpp` on Arc Pro B70 (BMG G31, Xe2, 32GB) leaves 2–7× on the floor depending on the model and backend. This kit is the exact build + runtime configuration used by a 4× B70 Ubuntu Server inference box running 5 concurrent llama-server tiers (chat, code, fast, agentic, reasoning) at production speeds.

You get:
- Patched llama.cpp binaries (SYCL and Vulkan) with the 11 commits that matter on Xe2
- Mesa 26+ (required for Vulkan BF16 + coopmat on BMG)
- Per-model start scripts with tuned flags and env vars
- Systemd units so tiers survive reboots
- Clear rules for when to pick SYCL vs Vulkan per model
- Reference benchmarks you can regress against

If you have B70s sitting in a box running at Vulkan defaults, this will roughly double to triple your tok/s. If you're fighting MoE model crashes or SYCL slot-init SEGVs, this has the workarounds.

## What this kit is for

If you have one or more Intel Arc Pro B70 cards and you want them to be useful for local LLM inference, out-of-the-box `llama.cpp` leaves a lot on the table. This kit captures:

- 11 cherry-picked commits on top of a known-good `llama.cpp` base (BF16 GET_ROWS, MoE MMVQ fused TG, K-quant native subgroup DMMV, Xe2 Vulkan warptile, oneMKL small-matmul path, Q8_0 reorder fix, etc.)
- Correct SYCL build flags (`GGML_SYCL_F16=ON`, HOST_MEM_FALLBACK, DNN/graph enabled)
- Runtime env vars that matter on Xe2 (`GGML_SYCL_DISABLE_OPT=1`, `UR_L0_ENABLE_RELAXED_ALLOCATION_LIMITS=1`)
- Mesa 26.0.5 from the `kisak/kisak-mesa` PPA (enables `VK_KHR_shader_bfloat16` + `VK_KHR_shader_integer_dot_product` on BMG)
- Per-model start scripts (dense Q8 on SYCL; MoE on Vulkan to avoid SYCL MoE slot-init SEGV)
- Systemd unit template so tiers survive reboots
- Known-good benchmark numbers so you know when something regressed

## Hardware tested

| | |
|---|---|
| GPU | 4× Intel Arc Pro B70 (BMG G31, Xe2, 32GB GDDR6 each) |
| Host | AMD Threadripper 1900X, 128GB DDR4 |
| OS | Ubuntu 24.04 (kernel 6.8+) |
| Backends | llama.cpp SYCL (oneAPI 2024.2+) and llama.cpp Vulkan (Mesa 26.0.5) |

## Headline numbers (single-stream, identical 300-tok prompt)

| Tier | Model | Backend | GPU | tg tok/s | Notes |
|---|---|---|---|---|---|
| chat | gemma-4-26B-A4B Q8_0 | SYCL | 1 | 26.4 | dense, SYCL wins over Vulkan |
| code | Qwen3-Coder-30B-A3B Q5_K_M | SYCL | 3 | 57.7 | MoE; DISABLE_OPT=1 required |
| fast | Qwen3-4B-Instruct Q6_K | Vulkan | 3 | 33.0 | co-tenant with code tier |
| agentic | Qwen3.6-35B-A3B Q6_K_XL + 0.6B draft | Vulkan | 0 | 25.0 | speculative decoding |
| reasoning | Qwen3-Next-80B-A3B IQ3_XXS | SYCL | 2 | 21.2 | 80B MoE, 3B active |

See `docs/benchmarks.md` for methodology and regression guardrails.

## What the 11 cherry-picks fix

Every patch in `patches/` lands on top of llama.cpp master `073bb2c20` (2026-04). Each one is listed below with what it does, why B70 specifically needs it, and the measured impact. Patches are applied in this order by `scripts/build-sycl.sh` and `scripts/build-vulkan.sh`.

### SYCL backend patches (8)

| # | Commit subject | What it fixes | B70 impact |
|---|---|---|---|
| 1 | `[SYCL] Add BF16 support to GET_ROWS operation` (0f842b5b1) | GET_ROWS (embedding lookup / K-cache gather) had no BF16 path, forced fallback to f32 conversion on every token | Gemma 4 26B (BF16 weights) prompt processing +40%, token gen +15% |
| 2 | `sycl: fused MoE mul_mat_vec_q for TG` (d99e97537) | MoE token-generation used separate mul_mat_vec_q + reduce passes; now fused into one kernel | Qwen3-Coder-30B MoE tg +47%. Single biggest perf win on MoE models |
| 3 | `SYCL: use native subgroup size for K-quant DMMV kernels` (ada8c01bc) | K-quant (Q4_K, Q5_K, Q6_K) DMMV kernels hardcoded subgroup size 32; Xe2 native is 16 | K-quant models (Q6_K, Q5_K_M) +20-25% tg |
| 4 | `sycl: route small f32 matmuls to oneMKL, bypass oneDNN` (526d32b3d) | oneDNN overhead on small matmuls (<512) dominated latency for attention QKV projections | First-token latency down ~30ms on all models |
| 5 | `SYCL: fix reorder crash when device memory is full` (bba5d8906) | Allocator tried to reorder tensors even when free VRAM < reorder temp buffer, causing -999 errors | Prevents crash when loading ~30GB model on 32GB card |
| 6 | `SYCL: add RAII temp buffer class + macro guard for host fallback` (ac17c7658) | Temp buffer leaks on reorder failure path; no host-mem fallback macro | Enables `GGML_SYCL_HOST_MEM_FALLBACK=ON` build option safely |
| 7 | `[SYCL] Fix Q8_0 reorder: add missing dequantize path for GEMM` (512987ae0) | Q8_0 reorder had a GEMM code path that dispatched to a dequantize function that didn't exist; segfault on first large-batch request | Fixes Q8_0 models (Gemma 4 Q8, Qwen3-14B Q8) on batch_size > 512 |
| 8 | `SYCL: document GGML_SYCL_HOST_MEM_FALLBACK build option in SYCL.md` (6fe13299c) | Docs only — explains the host-fallback flag added by patch 6 | No runtime impact, just operator-facing |

### Vulkan backend patches (2)

| # | Commit subject | What it fixes | B70 impact |
|---|---|---|---|
| 9 | `vulkan: Tweak Xe2 warptile configuration` (47e206a55) | Xe2 warptile sizes were inherited from Xe-HPG defaults; wrong for BMG's wider EUs | All Vulkan tiers +15-25% pp and tg |
| 10 | `vulkan: Detect Intel Xe3 separately from Xe2` (f70d6f11a) | Future-proofing — Xe3 (Panther Lake) was being treated as Xe2; prevents future regressions when Mesa ships Xe3 detection | No B70 impact today; prevents downstream Xe3 user hitting Xe2 tuning by accident |

### Experimental / research (1)

| # | Commit subject | What it fixes | Status |
|---|---|---|---|
| 11 | `fattn-tla: Phase 1 skeleton` (64af6820b) | Adds `GGML_SYCL_USE_TLA` CMake option stub for future sycl-tla Flash Attention kernels | **Off by default** (`GGML_SYCL_USE_TLA=OFF` in `build-sycl.sh`). Included so the branch is reproducible; enabling it regresses perf today |

## Runtime fixes we discovered (not in any commit)

These aren't code patches; they're environment / flag / topology decisions you **will not find in llama.cpp docs** but matter enormously on B70.

- **`GGML_SYCL_DISABLE_OPT=1` is mandatory for MoE.** Without it, llama-server SEGVs during slot initialization on MoE models (Qwen3-Coder-30B, Qwen3.6-35B, Qwen3-Next-80B). Costs ~5% on dense, essential on MoE. Root cause is the fused-reorder-MMVQ path racing with slot KV alloc. Upstream issue #15580.
- **Never set `SYCL_CACHE_PERSISTENT=1`.** Cross-restart kernel cache persistence poisons the cache on B70 — next boot SEGVs. Let JIT recompile each time (~30s first-run cost per model, warm after).
- **`UR_L0_ENABLE_RELAXED_ALLOCATION_LIMITS=1` for large KV.** Level Zero defaults cap single allocations at 4GB. 32K context on a 30B model needs >4GB KV; without this env var, allocation fails. Set it on every SYCL tier.
- **Two llama-servers on one B70, both SYCL: 10× slowdown.** Measured: Qwen3-Coder-30B alone = 60 tok/s, with a second SYCL server on the same card = 5–7 tok/s. Same test with the second server on Vulkan instead = 57 tok/s. If you must co-tenant a card, the lighter model goes on Vulkan. See `docs/backend-selection.md` Rule 4.
- **SYCL + SYCL speculative decoding on one card is unstable.** Target model on SYCL with a draft model also on SYCL causes kernel-cache contention that intermittently kills the server. Run the target on SYCL with draft on Vulkan, or both on Vulkan (our current agentic tier pattern).
- **`-fa 0` for SYCL MoE.** Flash attention on SYCL MoE models triggers a crash path on B70. Vulkan FA is fine. Our MoE SYCL tiers run with FA off; Vulkan MoE tiers run with `--flash-attn on`.
- **`--defrag-thold 0.1` is not optional on long-lived servers.** Without aggressive KV defrag, VRAM fragments after a few hundred requests and inference stalls. Every production start script sets this.
- **`-t 1` for all GPU tiers.** More host threads fight for the GPU submission queue. Single-thread dispatch wins. Counter-intuitive if you're coming from CPU inference.
- **Model sizing matters more than backend choice for co-tenant cards.** We ran a 9B Q4 + 30B MoE on one card and got 19/23 tok/s (painful). Swapped the 9B for a 4B Q6 on the same card: 33/57 tok/s. The smaller model's lower memory-bandwidth footprint uncontended the bigger neighbor.
- **Mesa 26+ or you leave 20–40% on the floor.** Ubuntu 24.04's default Mesa (25.2) lacks `VK_KHR_shader_bfloat16` + `VK_KHR_shader_integer_dot_product` for BMG. Without those extensions the Vulkan backend runs scalar f32 paths on what should be bf16 coopmat kernels. `scripts/install-mesa.sh` handles this via the `kisak/kisak-mesa` PPA.

Every one of these came from a measured regression or crash on our 4× B70 box. `docs/tuning.md` is the consolidated reference.

## Quick start

```bash
# 1. Install Mesa 26.0.5 (Vulkan backend)
sudo bash scripts/install-mesa.sh

# 2. Install oneAPI (SYCL backend) — Intel's repo, not packaged here
# Follow https://www.intel.com/content/www/us/en/developer/tools/oneapi/base-toolkit-download.html

# 3. Build both backends with the cherry-picks applied
bash scripts/build-sycl.sh    # ~25 min
bash scripts/build-vulkan.sh  # ~5 min

# 4. Install systemd template
sudo cp systemd/llamacpp@.service /etc/systemd/system/
sudo systemctl daemon-reload

# 5. Drop your model files in /mnt/models, edit scripts/start_*.sh paths,
#    put them in ~/, then:
systemctl --user start llamacpp@8000  # etc.
```

## Why two backends?

On B70 the two llama.cpp backends have different strengths:

- **SYCL** wins on dense models (Gemma 4, Qwen3-14B/32B) by 2–3× over Vulkan for token generation, thanks to oneMKL/oneDNN paths and the cherry-picked MMVQ kernels.
- **Vulkan** is required for MoE models with speculative decoding draft models; SYCL has a slot-init SEGV on MoE servers (mitigated but not fully fixed by `GGML_SYCL_DISABLE_OPT=1`).

`docs/backend-selection.md` has the rules.

## Related repos (pick the right one for your workload)

Three B70 inference repos exist; they're complementary, not duplicates.

| Repo | Inference engine | GPU usage | Best for |
|---|---|---|---|
| **this repo** | llama.cpp (SYCL + Vulkan) | N cards → N different models | Multi-tier agent platforms; running 4–5 models concurrently; GGUF quants (Q4/Q5/Q6/Q8); maximum model flexibility |
| [arc-pro-b70-inference-setup-ubuntu-server](https://github.com/Hal9000AIML/arc-pro-b70-inference-setup-ubuntu-server) | llama.cpp SYCL via installer *(also documents vLLM TP=4 option)* | 4 cards → 1 big model sharded *(vLLM TP=4)* or 4 independent *(llama.cpp)* | Building a box from scratch: bootable Ubuntu 24.04 autoinstall ISO, BIOS/hardware guide, DDR4 tuning, GuC firmware 70.60.0, systemd + watchdog. Use this FIRST to set up the machine, then apply this kit's patches on top |
| [arc-pro-b70-inference-setup-windows](https://github.com/Hal9000AIML/arc-pro-b70-inference-setup-windows) | vLLM XPU (TP=4 across 4 B70s) | 4 cards → 1 big model, sharded | Windows workstation wanting vLLM tensor parallelism via WSL2 + Docker. Max single-model throughput (540 tok/s on 4× B70). Does not use llama.cpp |

If you want **one big model at maximum throughput** (e.g., serving a single 70B to many users): the vLLM TP=4 path in the Ubuntu-server or Windows repo will beat what this kit can deliver — llama.cpp does not shard one model across GPUs, so it's bounded by single-B70 performance per process.

If you want **multiple different models running concurrently** (agent platforms with router/code/fast/reasoning tiers, developer workstations with dynamic model choice): this kit is the right tool. One model per card, each independently tuned, 5 concurrent llama-servers tested.

### Why we publish llama.cpp numbers, not a vLLM head-to-head

We attempted a TP=1 vLLM benchmark on the same Qwen3-Coder-30B model (GPTQ 4-bit) on a freed B70 to put an exact number against our 57.7 tok/s llama.cpp figure. The attempt surfaced three practical blockers that most users will hit:

1. **vLLM XPU container staleness** — the `vllm-xpu:gemma4-fixed` image from April shipped with vLLM v1 engine, which emits `XPU Graph is not supported in the current PyTorch version` and falls back to eager execution. Performance claims require a current image.
2. **Single-card TP=1 memory pressure** — a co-tenant llama-server on the same card left only 3.36 GiB free on the B70; vLLM rejects startup below its `gpu-memory-utilization` ask. TP=1 comparison on a shared GPU needs the full card evacuated.
3. **Architecture mismatch for multi-tier workloads** — vLLM's strength is TP=4 sharding of one model. Running a TP=1 comparison is informative about per-card raw throughput but doesn't reflect real deployment (you'd never run vLLM TP=1 on B70; you'd run TP=4).

**The architectural bottom line stands without the benchmark:** vLLM TP=4 on 4× B70 will beat llama.cpp on a single model (the [Ubuntu installer repo](https://github.com/Hal9000AIML/arc-pro-b70-inference-setup-ubuntu-server) cites 540 tok/s on Qwen3.5-27B TP=4). llama.cpp wins for concurrent multi-model workloads. Pick the tool whose design matches your deployment, not the one with the bigger single-stream number.

## What's NOT in this kit

- Windows support. These patches and flags are Linux-only. Intel's Windows Arc stack is a different beast. If you're on Windows and need B70 inference, use the [Windows repo](https://github.com/Hal9000AIML/arc-pro-b70-inference-setup-windows) (vLLM via WSL2).
- Model files. Bring your own GGUFs. The start scripts reference `/mnt/models/...` paths; edit to yours.
- An installer. This is artifacts + scripts; you run them. For a bootable Ubuntu autoinstall ISO that stands up the whole box from bare metal, see the [Ubuntu server repo](https://github.com/Hal9000AIML/arc-pro-b70-inference-setup-ubuntu-server).

## Repository layout

```
patches/        11 cherry-pick .patch files (apply to llama.cpp master@073bb2c20)
scripts/        build-sycl.sh, build-vulkan.sh, install-mesa.sh, start_*.sh
systemd/        llamacpp@.service template
docs/           build.md, tuning.md, backend-selection.md, benchmarks.md
```

## License

MIT. The patches are derivative work of llama.cpp (MIT). Upstream authors credited in each `.patch` file via `git format-patch` headers.
