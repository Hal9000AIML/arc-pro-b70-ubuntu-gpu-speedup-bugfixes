#!/usr/bin/env python3
"""b70-plan — generate B70-tuned llama-server start scripts from a declarative layout.

Reads a YAML config describing which model runs on which GPU card(s) and the intent
(chat, code, fast, agentic, reasoning, etc.), then applies the rules in
docs/backend-selection.md to pick SYCL vs Vulkan per tier, sets the correct env
vars and flags, and emits ready-to-run start_<port>.sh scripts.

Auto-selection logic (per tier):
  1. Speculative decoding (draft_model set)    -> Vulkan (SYCL spec decode on one card is unstable)
  2. Multi-card split (cards = [N, M, ...])    -> SYCL (better multi-device support)
  3. Co-tenant card (another tier on same card)
       lighter model                           -> Vulkan
       heavier model                           -> SYCL (+ DISABLE_OPT=1 if MoE)
  4. Solo card, MoE architecture               -> SYCL + GGML_SYCL_DISABLE_OPT=1
  5. Solo card, dense architecture             -> SYCL

Models are declared by filename relative to MODELS_DIR; architecture (MoE vs dense)
is detected by filename heuristic (A3B/MoE/Mixtral/-MoE-) then optionally by reading
GGUF metadata (`*.expert_count` > 0).

Usage:
    python3 b70-plan.py --config layout.yaml --out ~/         # writes ~/start_<port>.sh
    python3 b70-plan.py --scan /mnt/models > suggested.yaml   # interactive template
    python3 b70-plan.py --config layout.yaml --dry-run        # print decisions only
"""
from __future__ import annotations

import argparse
import dataclasses
import pathlib
import struct
import sys
import textwrap
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Any

try:
    import yaml
except ImportError:
    sys.exit("b70-plan needs PyYAML: pip install pyyaml  (or: sudo apt install python3-yaml)")


# -----------------------------------------------------------------------------
# Model architecture detection
# -----------------------------------------------------------------------------

MOE_FILENAME_MARKERS = ("a3b", "moe", "mixtral", "-a3b-", "-moe-", "qwen3-next-80b")


def is_moe_by_filename(filename: str) -> bool:
    """Heuristic MoE detection from GGUF filename. Accurate for Qwen3/Mixtral/DeepSeek family."""
    lower = filename.lower()
    return any(marker in lower for marker in MOE_FILENAME_MARKERS)


def read_gguf_expert_count(path: pathlib.Path) -> int | None:
    """Read expert_count from GGUF metadata. Returns None if unreadable (not fatal)."""
    try:
        with path.open("rb") as f:
            magic = f.read(4)
            if magic != b"GGUF":
                return None
            version = struct.unpack("<I", f.read(4))[0]
            if version < 2:
                return None
            tensor_count, kv_count = struct.unpack("<QQ", f.read(16))
            # Scan KV for *.expert_count
            for _ in range(min(kv_count, 500)):  # bound the scan
                key_len = struct.unpack("<Q", f.read(8))[0]
                if key_len > 256:
                    return None  # corrupt or unexpected; bail
                key = f.read(key_len).decode("utf-8", errors="replace")
                vtype = struct.unpack("<I", f.read(4))[0]
                value = _read_gguf_value(f, vtype)
                if key.endswith("expert_count") and isinstance(value, int):
                    return value
    except Exception:
        return None
    return 0


def _read_gguf_value(f, vtype: int) -> Any:
    """Minimal GGUF value reader — only needs to walk past values to find expert_count."""
    fmt = {
        0: "<B", 1: "<b", 2: "<H", 3: "<h", 4: "<I", 5: "<i",
        6: "<f", 7: "<?", 10: "<Q", 11: "<q", 12: "<d",
    }
    if vtype in fmt:
        size = struct.calcsize(fmt[vtype])
        return struct.unpack(fmt[vtype], f.read(size))[0]
    if vtype == 8:  # string
        n = struct.unpack("<Q", f.read(8))[0]
        return f.read(n).decode("utf-8", errors="replace")
    if vtype == 9:  # array
        atype = struct.unpack("<I", f.read(4))[0]
        n = struct.unpack("<Q", f.read(8))[0]
        for _ in range(n):
            _read_gguf_value(f, atype)
        return None
    raise ValueError(f"unknown gguf vtype {vtype}")


# -----------------------------------------------------------------------------
# Tier config + decision engine
# -----------------------------------------------------------------------------

@dataclass
class Tier:
    port: int
    alias: str
    model: str                              # relative to models_dir
    cards: list[int]                        # e.g. [3] or [2, 3]
    ctx: int = 16384
    parallel: int = 1
    batch_size: int = 2048
    ubatch_size: int = 512
    draft_model: str | None = None          # spec decode
    flash_attn: bool | None = None          # None = auto
    kv_quant: str | None = None             # e.g. "q8_0"
    chat_template: str | None = None
    reasoning: str = "off"
    extra_args: list[str] = field(default_factory=list)


@dataclass
class Plan:
    tier: Tier
    backend: str                            # "sycl" or "vulkan"
    is_moe: bool
    co_tenant_with: list[int]               # ports of other tiers sharing any of our cards
    env: dict[str, str]
    reasons: list[str]


def detect_moe(tier: Tier, models_dir: pathlib.Path, probe_gguf: bool) -> bool:
    """Decide whether this tier's model is MoE."""
    if is_moe_by_filename(tier.model):
        return True
    if probe_gguf:
        path = models_dir / tier.model
        if path.is_file():
            n = read_gguf_expert_count(path)
            if n is not None and n > 0:
                return True
    return False


def plan_tier(tier: Tier, all_tiers: list[Tier], models_dir: pathlib.Path, probe_gguf: bool) -> Plan:
    reasons: list[str] = []
    is_moe = detect_moe(tier, models_dir, probe_gguf)

    # Find co-tenants (other tiers sharing at least one card with us)
    my_cards = set(tier.cards)
    co_tenants = [
        t for t in all_tiers
        if t.port != tier.port and my_cards & set(t.cards)
    ]
    co_tenant_ports = [t.port for t in co_tenants]

    # Decision tree
    if tier.draft_model:
        backend = "vulkan"
        reasons.append("speculative decoding -> Vulkan (SYCL spec decode is unstable on one card)")
    elif len(tier.cards) > 1:
        backend = "sycl"
        reasons.append(f"multi-card split across {tier.cards} -> SYCL (better multi-device support on B70)")
    elif co_tenants:
        # Co-tenant card — lighter model goes Vulkan, heavier stays SYCL
        # "Lighter" = our model file is smaller than any co-tenant's
        my_size = _model_size(models_dir / tier.model)
        their_min = min((_model_size(models_dir / t.model) for t in co_tenants), default=0)
        if my_size and their_min and my_size < their_min:
            backend = "vulkan"
            reasons.append(f"co-tenant of port(s) {co_tenant_ports}; we're the smaller model -> Vulkan (cedes compute to co-tenant)")
        else:
            backend = "sycl"
            reasons.append(f"co-tenant of port(s) {co_tenant_ports}; we're the heavier model -> SYCL")
            if is_moe:
                reasons.append("MoE architecture on SYCL requires GGML_SYCL_DISABLE_OPT=1")
    elif is_moe:
        backend = "sycl"
        reasons.append("MoE on solo card -> SYCL + GGML_SYCL_DISABLE_OPT=1")
    else:
        backend = "sycl"
        reasons.append("dense on solo card -> SYCL")

    env = _env_for(backend, is_moe)
    return Plan(tier=tier, backend=backend, is_moe=is_moe,
                co_tenant_with=co_tenant_ports, env=env, reasons=reasons)


def _model_size(path: pathlib.Path) -> int:
    try:
        return path.stat().st_size
    except OSError:
        return 0


def _env_for(backend: str, is_moe: bool) -> dict[str, str]:
    if backend == "sycl":
        env = {
            "UR_L0_ENABLE_RELAXED_ALLOCATION_LIMITS": "1",
            "ZES_ENABLE_SYSMAN": "1",
        }
        # DISABLE_OPT is on by default for SYCL on B70 (5% perf cost vs rare crashes)
        env["GGML_SYCL_DISABLE_OPT"] = "1"
        return env
    return {}  # Vulkan needs nothing special at env level


# -----------------------------------------------------------------------------
# Start-script emission
# -----------------------------------------------------------------------------

SYCL_BINARY = "/opt/llama.cpp/llama-sycl-src/build-f16/bin/llama-server"
VULKAN_BINARY = "/opt/llama.cpp/llama-vulkan-build/bin/llama-server"


def device_flag(backend: str, cards: list[int]) -> str:
    prefix = "SYCL" if backend == "sycl" else "Vulkan"
    return ",".join(f"{prefix}{c}" for c in cards)


def emit_script(plan: Plan, models_dir: pathlib.Path) -> str:
    t = plan.tier
    # Flash attention default: off for SYCL MoE, on for Vulkan, off for SYCL dense-safe-by-default
    fa = t.flash_attn
    if fa is None:
        fa = (plan.backend == "vulkan")
    fa_flag = "--flash-attn on" if fa else "-fa 0"

    binary = SYCL_BINARY if plan.backend == "sycl" else VULKAN_BINARY
    device = device_flag(plan.backend, t.cards)

    lines: list[str] = ["#!/usr/bin/env bash"]
    lines.append(f"# Generated by b70-plan — {plan.backend.upper()} backend")
    lines.append(f"# Rationale:")
    for r in plan.reasons:
        lines.append(f"#   - {r}")
    lines.append("set -euo pipefail")
    lines.append("")

    if plan.backend == "sycl":
        lines.append("source /opt/intel/oneapi/setvars.sh --force 2>/dev/null")
    for k, v in plan.env.items():
        lines.append(f"export {k}={v}")

    lines.append("")
    lines.append(f"exec {binary} \\")
    lines.append(f"  --model {models_dir}/{t.model} \\")
    if t.draft_model:
        lines.append(f"  --model-draft {models_dir}/{t.draft_model} \\")
        lines.append(f"  --device-draft {device} -ngld 999 \\")
        lines.append(f"  --draft-max 16 --draft-min 1 --draft-p-min 0.5 \\")
    lines.append(f"  --device {device} \\")
    lines.append(f"  -ngl 999 -c {t.ctx} \\")
    lines.append(f"  --parallel {t.parallel} \\")
    lines.append(f"  --batch-size {t.batch_size} --ubatch-size {t.ubatch_size} \\")
    lines.append(f"  --defrag-thold 0.1 \\")
    lines.append(f"  {fa_flag} \\")
    if t.kv_quant:
        lines.append(f"  --cache-type-k {t.kv_quant} --cache-type-v {t.kv_quant} \\")
    if t.chat_template:
        lines.append(f"  --chat-template-file {t.chat_template} \\")
    lines.append(f"  --jinja --reasoning {t.reasoning} \\")
    lines.append(f"  --host 0.0.0.0 --port {t.port} \\")
    lines.append(f'  --alias "{t.alias}" \\')
    lines.append(f"  -t 1 --no-warmup --log-file /tmp/llama-{t.port}.log")
    for a in t.extra_args:
        lines.append(f"  {a} \\")
    return "\n".join(lines) + "\n"


# -----------------------------------------------------------------------------
# CLI
# -----------------------------------------------------------------------------

def load_config(path: pathlib.Path) -> tuple[pathlib.Path, list[Tier], bool]:
    data = yaml.safe_load(path.read_text())
    models_dir = pathlib.Path(data.get("models_dir", "/mnt/models"))
    probe_gguf = bool(data.get("probe_gguf", True))
    tiers: list[Tier] = []
    for entry in data.get("tiers", []):
        cards = entry.get("cards")
        if cards is None:
            card = entry.get("card")
            if card is None:
                raise SystemExit(f"tier {entry.get('port')} missing 'card' or 'cards'")
            cards = [card]
        elif isinstance(cards, int):
            cards = [cards]
        tiers.append(Tier(
            port=entry["port"],
            alias=entry["alias"],
            model=entry["model"],
            cards=list(cards),
            ctx=entry.get("ctx", 16384),
            parallel=entry.get("parallel", 1),
            batch_size=entry.get("batch_size", 2048),
            ubatch_size=entry.get("ubatch_size", 512),
            draft_model=entry.get("draft_model"),
            flash_attn=entry.get("flash_attn"),
            kv_quant=entry.get("kv_quant"),
            chat_template=entry.get("chat_template"),
            reasoning=entry.get("reasoning", "off"),
            extra_args=entry.get("extra_args", []),
        ))
    return models_dir, tiers, probe_gguf


def scan_models_dir(models_dir: pathlib.Path) -> str:
    """Emit a starter YAML from a scan of models_dir."""
    out = [
        f"models_dir: {models_dir}",
        "probe_gguf: true",
        "# Edit to assign models to cards. B70 has 4 cards: indices 0-3.",
        "# Co-tenant (two tiers on one card) is supported; b70-plan auto-selects backend.",
        "tiers:",
    ]
    base_port = 8000
    for i, p in enumerate(sorted(models_dir.glob("*.gguf"))[:5]):
        rel = p.relative_to(models_dir)
        out.append(f"  - port: {base_port + i}")
        out.append(f"    alias: {p.stem[:40]}")
        out.append(f"    model: {rel}")
        out.append(f"    card: {i % 4}")
        out.append(f"    ctx: 16384")
    return "\n".join(out) + "\n"


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", type=pathlib.Path, help="layout YAML")
    ap.add_argument("--out", type=pathlib.Path, default=pathlib.Path.home(),
                    help="where to write start_<port>.sh (default: ~/)")
    ap.add_argument("--scan", type=pathlib.Path, help="scan models_dir and emit a starter YAML to stdout")
    ap.add_argument("--dry-run", action="store_true", help="print decisions, don't write files")
    args = ap.parse_args()

    if args.scan:
        sys.stdout.write(scan_models_dir(args.scan))
        return 0

    if not args.config:
        ap.error("--config required (or --scan to generate a starter)")

    models_dir, tiers, probe_gguf = load_config(args.config)

    # Plan each tier (all tiers must be known so co-tenant detection works)
    plans = [plan_tier(t, tiers, models_dir, probe_gguf) for t in tiers]

    # Print decisions
    card_assignments: dict[int, list[int]] = defaultdict(list)
    for p in plans:
        for c in p.tier.cards:
            card_assignments[c].append(p.tier.port)
    print("=" * 72)
    print("B70 layout plan")
    print("=" * 72)
    print(f"models_dir: {models_dir}")
    print()
    for card in sorted(card_assignments):
        ports = card_assignments[card]
        marker = " (SHARED)" if len(ports) > 1 else ""
        print(f"  GPU{card}{marker}: ports {ports}")
    print()
    for p in plans:
        t = p.tier
        tag = f":{t.port} {t.alias}"
        print(f"  {tag:40s} cards={t.cards} backend={p.backend.upper():7s} moe={'yes' if p.is_moe else 'no'}")
        for r in p.reasons:
            print(f"    └── {r}")
    print()

    if args.dry_run:
        return 0

    # Write scripts
    args.out.mkdir(parents=True, exist_ok=True)
    for p in plans:
        dst = args.out / f"start_{p.tier.port}.sh"
        dst.write_text(emit_script(p, models_dir))
        dst.chmod(0o755)
        print(f"  wrote {dst}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
