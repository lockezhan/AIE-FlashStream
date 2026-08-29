#!/usr/bin/env bash
set -euo pipefail

ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
BUILD_DIR="$ROOT/build.check_v20_single_score_2phase"
PLATFORM=/opt/xilinx/platforms/xilinx_vck5000_gen4x8_qdma_2_202220_1/xilinx_vck5000_gen4x8_qdma_2_202220_1.xpfm
VITIS_SETTINGS=/s3/Xilinx_Vitis_2022.2/Vitis/2022.2/settings64.sh

if [[ "$BUILD_DIR" != "$ROOT/build.check_v20_single_score_2phase" ]]; then
  echo "refusing unsafe build directory: $BUILD_DIR" >&2
  exit 2
fi

rm -rf -- "$BUILD_DIR"
mkdir -p "$BUILD_DIR"

# shellcheck source=/dev/null
source "$VITIS_SETTINGS"
export XILINXD_LICENSE_FILE=/home/psdz/.Xilinx/Xilinx.lic
export XILINX_VITIS_AIETOOLS="${XILINX_VITIS_AIETOOLS:-/s3/Xilinx_Vitis_2022.2/Vitis/2022.2/aietools}"
export PATH="$XILINX_VITIS_AIETOOLS/bin:$XILINX_VITIS_AIETOOLS/tps/lnx64/target/bin/LNa64bin:$PATH"

echo "[1/4] AIE hardware compile"
(
  cd "$ROOT/aie"
  aiecompiler --target=hw \
    --platform="$PLATFORM" \
    --include="$ROOT/aie" \
    --include="$XILINX_VITIS_AIETOOLS/include" \
    --workdir="$BUILD_DIR/work_aie" \
    --output-archive="$BUILD_DIR/libadf_v20_single_score_2phase.a" \
    --pl-freq=300 \
    llama3_gqa_8group_packet.cpp
) 2>&1 | tee "$BUILD_DIR/aiecompiler_hw.log"

echo "[2/4] Exact AIE compute-tile gate"
python3 -c 'import json,sys; p=json.load(open(sys.argv[1])); n=len(p["ActiveCores"]); print(f"AIE compute tiles: {n}"); sys.exit(0 if n == 56 else 1)' \
  "$BUILD_DIR/work_aie/aie/active_cores.json" | tee "$BUILD_DIR/tile_count.txt"

echo "[3/4] PL v++ compile (no link)"
(
  cd "$ROOT"
  v++ -c -t hw \
    --platform "$PLATFORM" \
    --kernel_frequency 300 \
    --hls.jobs 16 \
    -k llama3_attention_aie8_packet \
    --temp_dir "$BUILD_DIR/_x_pl" \
    --report_dir "$BUILD_DIR/reports_pl" \
    -o "$BUILD_DIR/llama3_aie8_v20_single_score_2phase.xo" \
    pl/llama3_attention_aie_8group_packet.cpp
) 2>&1 | tee "$BUILD_DIR/vpp_compile_pl.log"

echo "[4/4] Warning and Fmax gates"
python3 -c '
import re, sys
forbidden = re.compile(r"dataflow.*deadlock|stream.*(count|token).*mismatch|multi[- ]?(reader|writer)|fifo.*(warning|deadlock)|unsupported.*read_nb|read_nb.*unsupported|packet.*(count|token).*inconsisten", re.IGNORECASE)
failed = False
for path in sys.argv[1:]:
    with open(path) as f:
        for i, line in enumerate(f, 1):
            if "Executing Cmd:" in line:
                continue
            m = forbidden.search(line)
            if m:
                print(f"FORBIDDEN WARNING in {path}:{i}: {line.strip()}", file=sys.stderr)
                failed = True
if failed:
    sys.exit(1)
' "$BUILD_DIR/aiecompiler_hw.log" "$BUILD_DIR/vpp_compile_pl.log"

grep "Estimated Fmax:" "$BUILD_DIR/vpp_compile_pl.log" | tee "$BUILD_DIR/hls_estimated_fmax.txt"

echo "V20 single-Score/two-phase AIE+PL compile gate PASS"
