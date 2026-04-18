#!/usr/bin/env bash
# Idempotent installer for arc-pro-b70-ubuntu-llm-inference-kit.
# Installs Mesa 26, builds llama.cpp SYCL + Vulkan with B70 cherry-picks,
# drops systemd unit, stages start_*.sh in $HOME.
#
# Re-runnable: skips steps already completed (Mesa already 26+, builds already present).
# Destructive only in /opt/llama.cpp/* (fresh clones/builds) and ~/.cache/libsycl_cache.
#
# Usage:   sudo -E bash install.sh
# Env:     MODELS_DIR (default /mnt/models) — where your GGUFs live
#          SKIP_MESA=1 / SKIP_SYCL=1 / SKIP_VULKAN=1 to skip phases
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
MODELS_DIR="${MODELS_DIR:-/mnt/models}"

# If run via sudo, figure out the invoking user so we put start scripts in their HOME
TARGET_USER="${SUDO_USER:-$USER}"
TARGET_HOME="$(getent passwd "$TARGET_USER" | cut -d: -f6)"

log()  { printf '\033[1;34m[b70-install]\033[0m %s\n' "$*"; }
warn() { printf '\033[1;33m[b70-install WARN]\033[0m %s\n' "$*"; }
die()  { printf '\033[1;31m[b70-install ERROR]\033[0m %s\n' "$*" >&2; exit 1; }

[[ $EUID -eq 0 ]] || die "Run as root (sudo -E bash install.sh). Needs apt + /etc/systemd write."
[[ -d "$HERE/patches" ]] || die "Run from the repo root (patches/ not found)."

log "Target user: $TARGET_USER  (home: $TARGET_HOME)"
log "Models dir:  $MODELS_DIR"

# ---------- 1. Build prerequisites ----------
log "Installing build prerequisites (git, cmake, ninja, glslc, libvulkan-dev)"
apt-get update -qq
apt-get install -y --no-install-recommends \
  git cmake ninja-build build-essential pkg-config \
  libvulkan-dev glslang-tools \
  curl ca-certificates gnupg lsb-release

# ---------- 2. Mesa 26+ (Vulkan) ----------
MESA_VER="$(dpkg-query -W -f='${Version}' mesa-vulkan-drivers 2>/dev/null || echo 0)"
if [[ "${SKIP_MESA:-}" != "1" ]] && ! [[ "$MESA_VER" == 26.* ]] && ! [[ "$MESA_VER" == 27.* ]]; then
  log "Mesa current: $MESA_VER — upgrading to 26.x from kisak/kisak-mesa PPA"
  bash "$HERE/scripts/install-mesa.sh"
else
  log "Mesa already $MESA_VER — skipping PPA install"
fi

# ---------- 3. oneAPI (SYCL) check ----------
if [[ "${SKIP_SYCL:-}" != "1" ]]; then
  if [[ ! -f /opt/intel/oneapi/setvars.sh ]]; then
    warn "Intel oneAPI not found at /opt/intel/oneapi. SYCL backend will be skipped."
    warn "To install: https://www.intel.com/content/www/us/en/developer/tools/oneapi/base-toolkit-download.html"
    warn "Then re-run: sudo -E bash install.sh"
    SKIP_SYCL=1
  fi
fi

# ---------- 4. Build SYCL backend ----------
if [[ "${SKIP_SYCL:-}" != "1" ]]; then
  log "Building llama.cpp SYCL backend (this takes ~15-25 min)"
  sudo -u "$TARGET_USER" -H bash "$HERE/scripts/build-sycl.sh"
else
  warn "Skipping SYCL build."
fi

# ---------- 5. Build Vulkan backend ----------
if [[ "${SKIP_VULKAN:-}" != "1" ]]; then
  log "Building llama.cpp Vulkan backend (~5 min)"
  sudo -u "$TARGET_USER" -H bash "$HERE/scripts/build-vulkan.sh"
else
  warn "Skipping Vulkan build."
fi

# ---------- 6. systemd unit ----------
log "Installing systemd template: /etc/systemd/system/llamacpp@.service"
install -m 0644 "$HERE/systemd/llamacpp@.service" /etc/systemd/system/llamacpp@.service
systemctl daemon-reload

# ---------- 7. Stage start scripts ----------
log "Staging start_*.sh in $TARGET_HOME (paths rewritten to MODELS_DIR=$MODELS_DIR)"
for s in "$HERE"/scripts/start_*.sh; do
  name="$(basename "$s")"
  dst="$TARGET_HOME/$name"
  if [[ -f "$dst" ]] && ! cmp -s "$s" "$dst"; then
    backup="$dst.pre-b70-install.$(date +%s)"
    log "  existing $name differs — saving backup at $backup"
    mv "$dst" "$backup"
  fi
  sed "s|/mnt/models|$MODELS_DIR|g" "$s" > "$dst"
  chmod +x "$dst"
  chown "$TARGET_USER:$TARGET_USER" "$dst"
done

# ---------- 8. Smoke test binaries ----------
log "Smoke-testing binaries"
if [[ -x /opt/llama.cpp/llama-sycl-src/build-f16/bin/llama-server ]]; then
  /opt/llama.cpp/llama-sycl-src/build-f16/bin/llama-server --version 2>&1 | head -2 || true
fi
if [[ -x /opt/llama.cpp/llama-vulkan-build/bin/llama-server ]]; then
  /opt/llama.cpp/llama-vulkan-build/bin/llama-server --version 2>&1 | head -2 || true
fi

# ---------- 9. Next-step hints ----------
cat <<EOF

\033[1;32m[b70-install] Done.\033[0m

Next:
  1. Put your GGUF files in $MODELS_DIR (or set MODELS_DIR= and re-run to re-stage scripts).
  2. Edit which card each script uses (--device SYCL<N> or Vulkan<N>) in $TARGET_HOME/start_*.sh
     — OR use 'b70-plan' to generate tuned start scripts from a declarative config:
       python3 $HERE/scripts/b70-plan.py --help
  3. Launch a tier:
       sudo -u $TARGET_USER systemctl --user start llamacpp@8000
     or manually:
       bash $TARGET_HOME/start_gemma_sycl.sh

Docs: $HERE/docs/backend-selection.md, tuning.md, benchmarks.md
EOF
