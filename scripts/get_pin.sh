#!/usr/bin/env bash
# Download and extract the Intel Pin kit into the repo root.
#
# Usage: scripts/get_pin.sh [--force]
#   --force   re-download and replace an existing kit
#
# Env overrides:
#   PIN_DEST  directory to extract into (default: repo root)
set -euo pipefail

PIN_NAME="pin-3.30-98830-g1d7b601b3-gcc-linux"
PIN_URL="https://software.intel.com/sites/landingpage/pintool/downloads/${PIN_NAME}.tar.gz"
PIN_SHA256="be4f1130445c3fc4d83b7afad85c421d418f60013c33e8ee457bc7c9c194de1b"

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
DEST="${PIN_DEST:-$REPO_ROOT}"
PIN_DIR="$DEST/$PIN_NAME"

FORCE=0
for arg in "$@"; do
    case "$arg" in
        --force) FORCE=1 ;;
        -h|--help) sed -n '2,9p' "$0"; exit 0 ;;
        *) echo "Unknown argument: $arg" >&2; exit 1 ;;
    esac
done

if [[ -x "$PIN_DIR/pin" && $FORCE -eq 0 ]]; then
    echo "[get_pin] Pin already present at $PIN_DIR (use --force to reinstall)"
    exit 0
fi

TMP="$(mktemp -d)"
trap 'rm -rf "$TMP"' EXIT

echo "[get_pin] Downloading $PIN_URL"
curl -fSL --retry 3 -o "$TMP/pin.tar.gz" "$PIN_URL"

echo "[get_pin] Verifying checksum"
echo "$PIN_SHA256  $TMP/pin.tar.gz" | sha256sum -c --quiet -

echo "[get_pin] Extracting to $DEST"
mkdir -p "$DEST"
tar xzf "$TMP/pin.tar.gz" -C "$TMP"
rm -rf "$PIN_DIR"
mv "$TMP/$PIN_NAME" "$PIN_DIR"

echo "[get_pin] Done: $PIN_DIR/pin"
