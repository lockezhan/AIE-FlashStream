#!/usr/bin/env bash
set -euo pipefail

ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
SOURCE_ROOT="${PURE_PL_ROOT:?set PURE_PL_ROOT to the Pure-PL baseline source/build tree}"
XCLBIN="${1:-$SOURCE_ROOT/build.hw_v1_baseline/llama3_attention_v1.xclbin}"
HOST="${PURE_PL_HOST:-$SOURCE_ROOT/build.hw_v1_baseline/llama3_attention_host_v1_baseline.exe}"
RESULT_DIR="${2:-$ROOT/results/matched_b1_pure_pl_v1_vs_v21_pv6_20260830/pure_pl_v1}"
SAMPLES="${3:-10}"
XRT_ROOT="${XRT_ROOT:-/opt/xilinx/xrt}"
XBUTIL="$XRT_ROOT/bin/xbutil"
BDF="${VCK5000_BDF:-0000:af:00.1}"
EXPECTED_XCLBIN_SHA="15b01daedc8d6ba497eaeb2644b299d9bc46c6069d06580db8025fe01cce9b2c"
EXPECTED_HOST_SHA="47b2b63547794a422865b12f5961cd8bc87cd97978b59f28510e1f009cc6f12e"

[[ -f "$XCLBIN" ]] || { echo "missing XCLBIN: $XCLBIN" >&2; exit 2; }
[[ -x "$HOST" ]] || { echo "missing Host: $HOST" >&2; exit 2; }
[[ -x "$XBUTIL" ]] || { echo "missing xbutil: $XBUTIL" >&2; exit 2; }
[[ "$SAMPLES" =~ ^[1-9][0-9]*$ ]] || { echo "invalid sample count: $SAMPLES" >&2; exit 2; }

mkdir -p "$RESULT_DIR"
xclbin_sha=$(sha256sum "$XCLBIN" | awk '{print $1}')
host_sha=$(sha256sum "$HOST" | awk '{print $1}')
[[ "$xclbin_sha" == "$EXPECTED_XCLBIN_SHA" ]] || {
  echo "refusing unexpected pure-PL XCLBIN: $xclbin_sha" >&2; exit 3;
}
[[ "$host_sha" == "$EXPECTED_HOST_SHA" ]] || {
  echo "refusing unexpected pure-PL Host: $host_sha" >&2; exit 3;
}

{
  printf 'git_sha=%s\n' "$(git -C "$ROOT" rev-parse HEAD)"
  printf 'host=%s\nhost_sha256=%s\n' "$HOST" "$host_sha"
  printf 'xclbin=%s\nxclbin_sha256=%s\n' "$XCLBIN" "$xclbin_sha"
  printf 'device_bdf=%s\n' "$BDF"
  printf 'workload=B1,S32,HQ32,HKV8,D128\n'
  printf 'protocol=reset_then_one_process_then_one_kernel_launch\n'
  printf 'warmup=0\nmeasured_per_process=1\nsamples=%s\nseed=7\n' "$SAMPLES"
} > "$RESULT_DIR/experiment.env"

for sample in $(seq 1 "$SAMPLES"); do
  tag=$(printf 'sample_%02d' "$sample")
  if [[ -s "$RESULT_DIR/${tag}.json" && -s "$RESULT_DIR/${tag}.csv" ]]; then
    echo "[$tag] preserving completed sample"
    continue
  fi
  echo "[$tag] resetting $BDF"
  "$XBUTIL" reset --device "$BDF" > "$RESULT_DIR/${tag}_reset.log" 2>&1
  ready=0
  for attempt in $(seq 1 30); do
    if "$XBUTIL" examine 2>/dev/null | grep -F "$BDF" | grep -q 'Yes'; then
      ready=1
      break
    fi
    sleep 1
  done
  [[ "$ready" -eq 1 ]] || { echo "[$tag] device did not become ready" >&2; exit 4; }
  sleep 2
  echo "[$tag] running one independent pure-PL B1 launch"
  timeout 120s "$HOST" --xclbin "$XCLBIN" --batch 1 --warmup 0 --runs 1 \
    --seed 7 --verify --profile \
    --output-json "$RESULT_DIR/${tag}.json" \
    --output-csv "$RESULT_DIR/${tag}.csv" \
    > "$RESULT_DIR/${tag}.log" 2>&1
  grep -E 'run=1|PASS|FAIL|p50|Mean abs|Max abs' "$RESULT_DIR/${tag}.log" || true
done

echo "Matched pure-PL B1 evidence: $RESULT_DIR"
