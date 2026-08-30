#!/bin/bash
set -euo pipefail

source /s3/Xilinx_Vitis_2022.2/Vitis/2022.2/settings64.sh
export XILINXD_LICENSE_FILE=/home/psdz/.Xilinx/Xilinx.lic
export XILINX_VITIS_AIETOOLS=/s3/Xilinx_Vitis_2022.2/Vitis/2022.2/aietools
export PATH=$XILINX_VITIS_AIETOOLS/bin:$XILINX_VITIS_AIETOOLS/tps/lnx64/target/bin/LNa64bin:$PATH

ROOT=/s3/zhangyann/AIE-FlashStream
BUILD_DIR=$ROOT/build.hw_v20_1group_debug
PLATFORM=/opt/xilinx/platforms/xilinx_vck5000_gen4x8_qdma_2_202220_1/xilinx_vck5000_gen4x8_qdma_2_202220_1.xpfm

mkdir -p "$BUILD_DIR"
cd "$ROOT"

{
  printf 'timestamp_utc='
  date -u '+%Y-%m-%dT%H:%M:%SZ'
  printf 'commit='
  git rev-parse HEAD
  printf '\\nstatus --short\\n'
  git status --short
  printf '\\ndiff\\n'
  git diff --no-ext-diff --binary
} > "$BUILD_DIR/source_manifest.txt"

cd "$ROOT/aie"
echo "[V20-1GROUP] Compiling the true one-group AIE graph..."
aiecompiler --target=hw \
  --platform="$PLATFORM" \
  --include="$ROOT/aie" --include="$XILINX_VITIS_AIETOOLS/include" \
  --workdir="$BUILD_DIR/work_aie" \
  --output-archive="$BUILD_DIR/libadf_v20_1group_debug.a" \
  --pl-freq=300 \
  llama3_gqa_1group_packet_hw_debug.cpp \
  > "$BUILD_DIR/aiecompiler_hw.log" 2>&1

cd "$ROOT"
echo "[V20-1GROUP] Compiling the matching one-group PL kernel..."
v++ -c -t hw \
  --platform "$PLATFORM" \
  --kernel_frequency 300 --hls.jobs 16 \
  -k llama3_attention_aie8_packet \
  -o "$BUILD_DIR/llama3_aie1_v20_debug.xo" \
  pl/llama3_attention_aie_1group_packet_debug.cpp \
  > "$BUILD_DIR/vpp_compile_pl.log" 2>&1

echo "[V20-1GROUP] Linking fresh AIE and PL objects..."
v++ -l -t hw \
  --platform "$PLATFORM" \
  --optimize 2 --clock.defaultFreqHz 300000000 --save-temps \
  --temp_dir "$BUILD_DIR/_x_temp" --report_dir "$BUILD_DIR/reports" \
  --config link/one_group_packet_v20_debug.cfg \
  -o "$BUILD_DIR/llama3_attention_v20_1group_debug.xsa" \
  "$BUILD_DIR/llama3_aie1_v20_debug.xo" \
  "$BUILD_DIR/libadf_v20_1group_debug.a" \
  > "$BUILD_DIR/vpp_link.log" 2>&1

echo "[V20-1GROUP] Packaging..."
v++ --package -t hw \
  --platform "$PLATFORM" \
  --temp_dir "$BUILD_DIR/_x_temp" --report_dir "$BUILD_DIR/reports" \
  --package.boot_mode=ospi \
  -o "$BUILD_DIR/llama3_attention_v20_1group_debug.xclbin" \
  "$BUILD_DIR/llama3_attention_v20_1group_debug.xsa" \
  "$BUILD_DIR/libadf_v20_1group_debug.a" \
  > "$BUILD_DIR/vpp_package.log" 2>&1

{
  printf 'timestamp_utc='
  date -u '+%Y-%m-%dT%H:%M:%SZ'
  printf 'commit='
  git rev-parse HEAD
  printf '\\nartifacts\\n'
  for artifact in \
      "$BUILD_DIR/libadf_v20_1group_debug.a" \
      "$BUILD_DIR/llama3_aie1_v20_debug.xo" \
      "$BUILD_DIR/llama3_attention_v20_1group_debug.xsa" \
      "$BUILD_DIR/llama3_attention_v20_1group_debug.xclbin"; do
    sha256sum "$artifact"
    stat -c 'mtime=%y size=%s path=%n' "$artifact"
  done
} > "$BUILD_DIR/artifact_manifest.txt"

echo "[V20-1GROUP] Build complete."
