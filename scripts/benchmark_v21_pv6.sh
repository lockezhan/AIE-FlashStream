#!/usr/bin/env bash
set -euo pipefail

ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
XCLBIN="${1:-$ROOT/build.hw_v21_pv6/llama3_attention_v21_pv6.xclbin}"
RESULT_DIR="${2:-$ROOT/results/v21_pv6}"
HOST="$ROOT/host/llama3_attention_host.exe"

[[ -f "$XCLBIN" ]] || { echo "missing XCLBIN: $XCLBIN" >&2; exit 2; }
mkdir -p "$RESULT_DIR"
make -C "$ROOT" all

sha256sum "$XCLBIN" | tee "$RESULT_DIR/xclbin_sha256.txt"
timeout 20s "$HOST" --xclbin "$XCLBIN" --batch 1 --warmup 0 --runs 1 \
  --verify --output-json "$RESULT_DIR/single_verified.json" \
  --output-csv "$RESULT_DIR/single_verified.csv" \
  | tee "$RESULT_DIR/single_verified.txt"

# Sustained execution is a separate gate. V21 currently completes run 1 and
# stalls on run 2; preserve the timeout status as a reproducible artifact.
set +e
timeout 20s stdbuf -oL "$HOST" --xclbin "$XCLBIN" --batch 1 --warmup 0 \
  --runs 10 --no-verify --profile \
  | tee "$RESULT_DIR/repeat10_diagnostic.txt"
repeat_status=${PIPESTATUS[0]}
set -e
printf 'repeat10_exit_status=%d\n' "$repeat_status" \
  | tee "$RESULT_DIR/repeat10_status.txt"
if [[ "$repeat_status" -ne 0 && "$repeat_status" -ne 124 ]]; then
  exit "$repeat_status"
fi

echo "V21 PV6 benchmark artifacts: $RESULT_DIR"
