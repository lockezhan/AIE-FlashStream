#!/usr/bin/env bash
set -euo pipefail

ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
BUILD_DIR="$ROOT/build.check_v21_pv6"
PLATFORM=/opt/xilinx/platforms/xilinx_vck5000_gen4x8_qdma_2_202220_1/xilinx_vck5000_gen4x8_qdma_2_202220_1.xpfm
VITIS_SETTINGS=/s3/Xilinx_Vitis_2022.2/Vitis/2022.2/settings64.sh

[[ "$BUILD_DIR" == "$ROOT/build.check_v21_pv6" ]] || exit 2
rm -rf -- "$BUILD_DIR"
mkdir -p "$BUILD_DIR"
# shellcheck source=/dev/null
source "$VITIS_SETTINGS"
export XILINXD_LICENSE_FILE=/home/psdz/.Xilinx/Xilinx.lic
export XILINX_VITIS_AIETOOLS="${XILINX_VITIS_AIETOOLS:-/s3/Xilinx_Vitis_2022.2/Vitis/2022.2/aietools}"
export PATH="$XILINX_VITIS_AIETOOLS/bin:$XILINX_VITIS_AIETOOLS/tps/lnx64/target/bin/LNa64bin:$PATH"

echo "[1/5] Reciprocal LUT and PL transport"
python3 "$ROOT/tests/check_reciprocal_lut.py"
g++ -std=c++17 -O2 -I/s3/Xilinx_Vitis_2022.2/Vitis_HLS/2022.2/include \
  -I"$ROOT/pl" "$ROOT/tests/pl_transport_tb.cpp" -o "$BUILD_DIR/pl_transport_tb"
"$BUILD_DIR/pl_transport_tb"

echo "[2/5] Generate, compile and run production-topology x86sim"
python3 "$ROOT/tests/v20_single_score_2phase_sim.py" generate --dir "$ROOT/aie"
(cd "$ROOT/aie" && aiecompiler --target=x86sim --platform="$PLATFORM" \
  --include="$ROOT/aie" --include="$XILINX_VITIS_AIETOOLS/include" \
  --workdir="$ROOT/aie/build.x86sim_v21_pv6" --output-archive="$BUILD_DIR/libadf_x86.a" \
  llama3_gqa_8group_packet_sim.cpp) 2>&1 | tee "$BUILD_DIR/aiecompiler_x86.log"
(cd "$ROOT/aie" && x86simulator --pkg-dir="$ROOT/aie/build.x86sim_v21_pv6") 2>&1 | tee "$BUILD_DIR/x86sim.log"
python3 "$ROOT/tests/v20_single_score_2phase_sim.py" verify --dir "$ROOT/aie"
[[ $(grep -c 'V20_SCORE_CACHE.*action=load' "$BUILD_DIR/x86sim.log") -eq 8 ]]
[[ $(grep -c 'V20_SCORE_CACHE.*action=reuse-no-read' "$BUILD_DIR/x86sim.log") -eq 8 ]]
[[ $(grep -c 'V20_VALUE_CACHE.*action=load' "$BUILD_DIR/x86sim.log") -eq 48 ]]
[[ $(grep -c 'V20_VALUE_CACHE.*action=reuse-no-read' "$BUILD_DIR/x86sim.log") -eq 48 ]]

echo "[3/5] AIE hardware compile"
(cd "$ROOT/aie" && aiecompiler --target=hw --platform="$PLATFORM" \
  --include="$ROOT/aie" --include="$XILINX_VITIS_AIETOOLS/include" \
  --workdir="$BUILD_DIR/work_aie" --output-archive="$BUILD_DIR/libadf_v21_pv6.a" \
  --pl-freq=300 llama3_gqa_8group_packet.cpp) 2>&1 | tee "$BUILD_DIR/aiecompiler_hw.log"

echo "[4/5] Exact 64-compute-tile gate"
python3 -c 'import json,sys; p=json.load(open(sys.argv[1])); n=len(p["ActiveCores"]); print(f"AIE compute tiles: {n}"); sys.exit(n != 64)' \
  "$BUILD_DIR/work_aie/aie/active_cores.json" | tee "$BUILD_DIR/tile_count.txt"

echo "[5/5] PL compile only (no link)"
(cd "$ROOT" && v++ -c -t hw --platform "$PLATFORM" --kernel_frequency 300 \
  --hls.jobs 16 -k llama3_attention_aie8_packet --temp_dir "$BUILD_DIR/_x_pl" \
  --report_dir "$BUILD_DIR/reports_pl" -o "$BUILD_DIR/llama3_aie8_v21_pv6.xo" \
  pl/llama3_attention_aie_8group_packet.cpp) 2>&1 | tee "$BUILD_DIR/vpp_compile_pl.log"
grep "Estimated Fmax:" "$BUILD_DIR/vpp_compile_pl.log" | tee "$BUILD_DIR/hls_estimated_fmax.txt"
echo "V21 PV6 cheap compile gate PASS"
