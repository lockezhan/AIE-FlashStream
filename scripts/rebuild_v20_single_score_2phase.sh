#!/usr/bin/env bash
set -euo pipefail

ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
BUILD_DIR="$ROOT/build.hw_v20_single_score_2phase"
PLATFORM=/opt/xilinx/platforms/xilinx_vck5000_gen4x8_qdma_2_202220_1/xilinx_vck5000_gen4x8_qdma_2_202220_1.xpfm
VITIS_SETTINGS=/s3/Xilinx_Vitis_2022.2/Vitis/2022.2/settings64.sh

if [[ "$BUILD_DIR" != "$ROOT/build.hw_v20_single_score_2phase" ]]; then
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

echo "[1/5] Fresh AIE hardware compile"
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

python3 -c 'import json,sys; p=json.load(open(sys.argv[1])); n=len(p["ActiveCores"]); print(f"AIE compute tiles: {n}"); sys.exit(0 if n == 56 else 1)' \
  "$BUILD_DIR/work_aie/aie/active_cores.json" | tee "$BUILD_DIR/tile_count.txt"

echo "[2/5] Fresh PL compile"
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

echo "[3/5] Fresh AIE+PL link"
(
  cd "$ROOT"
  v++ -l -t hw \
    --platform "$PLATFORM" \
    --optimize 2 \
    --clock.defaultFreqHz 300000000 \
    --save-temps \
    --temp_dir "$BUILD_DIR/_x_link" \
    --report_dir "$BUILD_DIR/reports_link" \
    --config link/eight_groups_packet.cfg \
    -o "$BUILD_DIR/llama3_attention_v20_single_score_2phase.xsa" \
    "$BUILD_DIR/llama3_aie8_v20_single_score_2phase.xo" \
    "$BUILD_DIR/libadf_v20_single_score_2phase.a"
) 2>&1 | tee "$BUILD_DIR/vpp_link.log"

echo "[4/5] Fresh package"
v++ --package -t hw \
  --platform "$PLATFORM" \
  --temp_dir "$BUILD_DIR/_x_package" \
  --report_dir "$BUILD_DIR/reports_package" \
  --package.boot_mode=ospi \
  -o "$BUILD_DIR/llama3_attention_v20_single_score_2phase.xclbin" \
  "$BUILD_DIR/llama3_attention_v20_single_score_2phase.xsa" \
  "$BUILD_DIR/libadf_v20_single_score_2phase.a" \
  2>&1 | tee "$BUILD_DIR/vpp_package.log"

echo "[5/5] Artifact SHA256"
sha256sum \
  "$BUILD_DIR/libadf_v20_single_score_2phase.a" \
  "$BUILD_DIR/llama3_aie8_v20_single_score_2phase.xo" \
  "$BUILD_DIR/llama3_attention_v20_single_score_2phase.xsa" \
  "$BUILD_DIR/llama3_attention_v20_single_score_2phase.xclbin" \
  | tee "$BUILD_DIR/artifact_sha256.txt"

echo "V20 clean hardware build PASS: $BUILD_DIR"
