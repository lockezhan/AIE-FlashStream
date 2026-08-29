#!/bin/bash
set -euo pipefail

echo "================================================================================"
echo "          STARTING FRESH FULL-8 HARDWARE BUILD (TARGET=HW)"
echo "================================================================================"
echo "Timestamp: $(date -u)"
echo "Git Hash: $(git rev-parse --short HEAD)"

source /s3/Xilinx_Vitis_2022.2/Vitis/2022.2/settings64.sh
export XILINXD_LICENSE_FILE=/home/psdz/.Xilinx/Xilinx.lic
export XILINX_VITIS_AIETOOLS=/s3/Xilinx_Vitis_2022.2/Vitis/2022.2/aietools
export PATH=$XILINX_VITIS_AIETOOLS/bin:$XILINX_VITIS_AIETOOLS/tps/lnx64/target/bin/LNa64bin:$PATH

ROOT=/s3/zhangyann/AIE-FlashStream
BUILD_DIR=$ROOT/build.hw_v20_full8_fresh
PLATFORM=/opt/xilinx/platforms/xilinx_vck5000_gen4x8_qdma_2_202220_1/xilinx_vck5000_gen4x8_qdma_2_202220_1.xpfm

# Ensure completely fresh build directory
rm -rf "$BUILD_DIR"
mkdir -p "$BUILD_DIR"

cd "$ROOT/aie"
echo "[V20 FRESH FULL-8 BUILD] Step 1/4: Compiling full-8 AIE graph..."
aiecompiler --target=hw \
  --platform="$PLATFORM" \
  --include="$ROOT/aie" --include="$XILINX_VITIS_AIETOOLS/include" \
  --workdir="$BUILD_DIR/work_aie" \
  --output-archive="$BUILD_DIR/libadf_v20.a" \
  --pl-freq=300 \
  llama3_gqa_8group_packet.cpp > "$BUILD_DIR/aiecompiler_hw.log" 2>&1

echo "[V20 FRESH FULL-8 BUILD] Step 1 Complete: AIE Archive at $BUILD_DIR/libadf_v20.a"

cd "$ROOT"
echo "[V20 FRESH FULL-8 BUILD] Step 2/4: Compiling full-8 PL kernel (v++ -c)..."
v++ -c -t hw \
  --platform "$PLATFORM" \
  --kernel_frequency 300 --hls.jobs 16 \
  -k llama3_attention_aie8_packet \
  -o "$BUILD_DIR/llama3_aie8_v20.xo" \
  pl/llama3_attention_aie_8group_packet.cpp > "$BUILD_DIR/vpp_compile_pl.log" 2>&1

echo "[V20 FRESH FULL-8 BUILD] Step 2 Complete: PL XO at $BUILD_DIR/llama3_aie8_v20.xo"

echo "[V20 FRESH FULL-8 BUILD] Step 3/4: Linking AIE + PL (v++ -l)..."
v++ -l -t hw \
  --platform "$PLATFORM" \
  --optimize 2 --clock.defaultFreqHz 300000000 --save-temps \
  --temp_dir "$BUILD_DIR/_x_temp" --report_dir "$BUILD_DIR/reports" \
  --config link/eight_groups_packet.cfg -o "$BUILD_DIR/llama3_attention_v20.xsa" \
  "$BUILD_DIR/llama3_aie8_v20.xo" "$BUILD_DIR/libadf_v20.a" > "$BUILD_DIR/vpp_link.log" 2>&1

echo "[V20 FRESH FULL-8 BUILD] Step 3 Complete: XSA generated at $BUILD_DIR/llama3_attention_v20.xsa"

echo "[V20 FRESH FULL-8 BUILD] Step 4/4: Packaging (v++ --package)..."
v++ --package -t hw \
  --platform "$PLATFORM" \
  --temp_dir "$BUILD_DIR/_x_temp" --report_dir "$BUILD_DIR/reports" \
  --package.boot_mode=ospi -o "$BUILD_DIR/llama3_attention_v20.xclbin" \
  "$BUILD_DIR/llama3_attention_v20.xsa" "$BUILD_DIR/libadf_v20.a" > "$BUILD_DIR/vpp_package.log" 2>&1

echo "================================================================================"
echo "          FRESH FULL-8 HARDWARE BUILD COMPLETED SUCCESSFULLY!"
echo "          Output xclbin: $BUILD_DIR/llama3_attention_v20.xclbin"
echo "================================================================================"
