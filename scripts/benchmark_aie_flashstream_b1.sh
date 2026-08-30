#!/usr/bin/env bash
set -euo pipefail

ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
XCLBIN="${1:-$ROOT/prebuilt/vck5000/llama3_attention.xclbin}"
RESULT_DIR="${2:-$ROOT/results/matched_b1/aie_flashstream}"
SAMPLES="${3:-10}"
HOST="$ROOT/host/llama3_attention_host.exe"
XRT_ROOT="${XRT_ROOT:-/opt/xilinx/xrt}"
XBUTIL="$XRT_ROOT/bin/xbutil"
BDF="${VCK5000_BDF:-0000:af:00.1}"
EXPECTED_XCLBIN_SHA="bff430b2b827c72469e5b147027786de1cf98a5f5ce587910f04590547d08b58"

[[ -f "$XCLBIN" ]] || { echo "missing XCLBIN: $XCLBIN" >&2; exit 2; }
[[ -x "$HOST" ]] || { echo "missing Host: $HOST" >&2; exit 2; }
[[ -x "$XBUTIL" ]] || { echo "missing xbutil: $XBUTIL" >&2; exit 2; }
[[ "$SAMPLES" =~ ^[1-9][0-9]*$ ]] || { echo "invalid sample count: $SAMPLES" >&2; exit 2; }

mkdir -p "$RESULT_DIR"
xclbin_sha=$(sha256sum "$XCLBIN" | awk '{print $1}')
host_sha=$(sha256sum "$HOST" | awk '{print $1}')
[[ "$xclbin_sha" == "$EXPECTED_XCLBIN_SHA" ]] || {
  echo "refusing unexpected XCLBIN: $xclbin_sha" >&2; exit 3;
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
  echo "[$tag] running one independent B1 launch"
  timeout 120s "$HOST" --xclbin "$XCLBIN" --batch 1 --warmup 0 --runs 1 \
    --seed 7 --verify --profile \
    --output-json "$RESULT_DIR/${tag}.json" \
    --output-csv "$RESULT_DIR/${tag}.csv" \
    > "$RESULT_DIR/${tag}.log" 2>&1
  grep -E 'run=1|PASS|FAIL|p50|Mean abs|Max abs' "$RESULT_DIR/${tag}.log" || true
done

(cd "$RESULT_DIR" && sha256sum experiment.env sample_*.csv sample_*.json sample_*.log sample_*_reset.log > SHA256SUMS)
echo "AIE-FlashStream matched B1 campaign completed in $RESULT_DIR"
