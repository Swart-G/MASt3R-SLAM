#!/usr/bin/env bash
set -euo pipefail

project_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
output="${1:-$project_root/releases/mast3r-slam-linux-x86_64}"
uv_binary="${UV_BINARY:-$(command -v uv || true)}"

if [[ -z "$uv_binary" || ! -x "$uv_binary" ]]; then
  echo "A standalone uv binary is required to build the Linux release." >&2
  echo "Install uv or set UV_BINARY=/absolute/path/to/uv." >&2
  exit 2
fi

python3 "$project_root/scripts/build_linux_release.py" \
  --uv "$uv_binary" \
  --output "$output"
