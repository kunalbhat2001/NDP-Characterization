#!/usr/bin/env bash
# Build Pin tools under pintools/. Each tool lives in pintools/<name>/ with
# <name>.cpp, makefile and makefile.rules; output goes to
# pintools/<name>/obj-intel64/<name>.so.
#
# Usage: scripts/build_pintools.sh [--clean] [tool ...]
#   tool      tool name(s) to build (default: every dir in pintools/)
#   --clean   remove build outputs instead of building
#
# Env overrides:
#   PIN_ROOT  Pin kit to build against (default: the kit get_pin.sh installs)
#   JOBS      parallel make jobs (default: nproc)
set -euo pipefail

PIN_NAME="pin-3.30-98830-g1d7b601b3-gcc-linux"
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PIN_ROOT="${PIN_ROOT:-$REPO_ROOT/$PIN_NAME}"
JOBS="${JOBS:-$(nproc)}"

CLEAN=0
TOOLS=()
for arg in "$@"; do
    case "$arg" in
        --clean) CLEAN=1 ;;
        -h|--help) sed -n '2,13p' "$0"; exit 0 ;;
        -*) echo "Unknown option: $arg" >&2; exit 1 ;;
        *) TOOLS+=("$arg") ;;
    esac
done

if [[ ${#TOOLS[@]} -eq 0 ]]; then
    for d in "$REPO_ROOT"/pintools/*/; do
        [[ -f "$d/makefile" ]] && TOOLS+=("$(basename "$d")")
    done
fi

if [[ $CLEAN -eq 1 ]]; then
    for t in "${TOOLS[@]}"; do
        echo "[build_pintools] Cleaning $t"
        rm -rf "$REPO_ROOT/pintools/$t/obj-intel64"
    done
    exit 0
fi

if [[ ! -f "$PIN_ROOT/source/tools/Config/makefile.config" ]]; then
    echo "[build_pintools] Pin kit not found at $PIN_ROOT" >&2
    echo "                 Run scripts/get_pin.sh or set PIN_ROOT." >&2
    exit 1
fi

FAILED=()
for t in "${TOOLS[@]}"; do
    dir="$REPO_ROOT/pintools/$t"
    if [[ ! -f "$dir/makefile" || ! -f "$dir/$t.cpp" ]]; then
        echo "[build_pintools] Skipping $t: expected $dir/makefile and $dir/$t.cpp" >&2
        FAILED+=("$t")
        continue
    fi
    echo "[build_pintools] Building $t"
    if make -C "$dir" -j"$JOBS" PIN_ROOT="$PIN_ROOT" "obj-intel64/$t.so"; then
        echo "[build_pintools] -> $dir/obj-intel64/$t.so"
    else
        FAILED+=("$t")
    fi
done

if [[ ${#FAILED[@]} -gt 0 ]]; then
    echo "[build_pintools] FAILED: ${FAILED[*]}" >&2
    exit 1
fi
echo "[build_pintools] All tools built."
