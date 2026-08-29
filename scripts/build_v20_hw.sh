#!/bin/bash
set -e

echo "================================================================================"
echo "          STARTING V20 FULL HARDWARE BUILD (TARGET=HW)"
echo "================================================================================"
echo "Timestamp: $(date -u)"
echo "Snapshot Tag: V20_4Q_FUSED_VERIFIED"
echo "Git Hash: $(git rev-parse --short HEAD)"

source /s3/Xilinx_Vitis_2022.2/Vitis/2022.2/settings64.sh
export XILINXD_LICENSE_FILE=/home/psdz/.Xilinx/Xilinx.lic
export XILINX_VITIS_AIETOOLS=/s3/Xilinx_Vitis_2022.2/Vitis/2022.2/aietools
export PATH=/s3/Xilinx_Vitis_2022.2/Vitis/2022.2/aietools/bin:/s3/Xilinx_Vitis_2022.2/Vitis/2022.2/aietools/tps/lnx64/target/bin/LNa64bin:$PATH

BUILD_DIR=/s3/zhangyann/AIE-FlashStream/build.hw_v20
mkdir -p $BUILD_DIR

rm -f /s3/zhangyann/AIE-FlashStream/aie/llama3_v21_config.h

echo "\n--- STEP 1/3: Running aiecompiler for 8-group V20 graph (target=hw) ---"
cd /s3/zhangyann/AIE-FlashStream/aie
aiecompiler --target=hw \
  --platform=/opt/xilinx/platforms/xilinx_vck5000_gen4x8_qdma_2_202220_1/xilinx_vck5000_gen4x8_qdma_2_202220_1.xpfm \
  --include=/s3/zhangyann/AIE-FlashStream/aie --include=/s3/Xilinx_Vitis_2022.2/Vitis/2022.2/aietools/include \
  --workdir=$BUILD_DIR/work_aie \
  --output-archive=$BUILD_DIR/libadf_v20.a \
  --pl-freq=300 \
  /s3/zhangyann/AIE-FlashStream/aie/llama3_gqa_8group_packet.cpp > $BUILD_DIR/aiecompiler_hw.log 2>&1

echo "[V20 HW BUILD] Step 1 Complete: AIE Archive generated at $BUILD_DIR/libadf_v20.a"

echo "\n--- STEP 2/3: Running v++ HLS compilation for PL shell (target=hw) ---"
cd /s3/zhangyann/AIE-FlashStream
v++ -c -t hw \
  --platform /opt/xilinx/platforms/xilinx_vck5000_gen4x8_qdma_2_202220_1/xilinx_vck5000_gen4x8_qdma_2_202220_1.xpfm \
  --kernel_frequency 300 --hls.jobs 16 \
  -k llama3_attention_aie8_packet -o $BUILD_DIR/llama3_aie8_v20.xo \
  pl/llama3_attention_aie_8group_packet.cpp > $BUILD_DIR/vpp_compile_pl.log 2>&1

echo "[V20 HW BUILD] Step 2 Complete: PL XO generated at $BUILD_DIR/llama3_aie8_v20.xo"

echo "\n--- STEP 3/3: Running v++ Link & Package (target=hw) ---"
v++ -l -t hw \
  --platform /opt/xilinx/platforms/xilinx_vck5000_gen4x8_qdma_2_202220_1/xilinx_vck5000_gen4x8_qdma_2_202220_1.xpfm \
  --optimize 2 --clock.defaultFreqHz 300000000 --save-temps \
  --temp_dir $BUILD_DIR/_x_temp --report_dir $BUILD_DIR/reports \
  --config link/eight_groups_packet.cfg -o $BUILD_DIR/llama3_attention_v20.xsa \
  $BUILD_DIR/llama3_aie8_v20.xo $BUILD_DIR/libadf_v20.a > $BUILD_DIR/vpp_link.log 2>&1

v++ --package -t hw \
  --platform /opt/xilinx/platforms/xilinx_vck5000_gen4x8_qdma_2_202220_1/xilinx_vck5000_gen4x8_qdma_2_202220_1.xpfm \
  --temp_dir $BUILD_DIR/_x_temp --report_dir $BUILD_DIR/reports \
  --package.boot_mode=ospi -o $BUILD_DIR/llama3_attention_v20.xclbin \
  $BUILD_DIR/llama3_attention_v20.xsa $BUILD_DIR/libadf_v20.a > $BUILD_DIR/vpp_package.log 2>&1

echo "================================================================================"
echo "          V20 FULL HARDWARE BUILD COMPLETED SUCCESSFULLY!"
echo "          Output xclbin: $BUILD_DIR/llama3_attention_v20.xclbin"
echo "================================================================================"
