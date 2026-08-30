# Building AIE-FlashStream

The hardware build scripts are the executable source of truth; this document
defines their environment and intended use.

## Environment

The tested setup is Vitis/Vivado 2022.2 with XRT 2.15.225 and platform
`xilinx_vck5000_gen4x8_qdma_2_202220_1`. Override installation paths when the
tools are installed elsewhere:

```bash
export VITIS_ROOT=/s3/Xilinx_Vitis_2022.2/Vitis/2022.2
export VITIS_SETTINGS="$VITIS_ROOT/settings64.sh"
export VCK5000_PLATFORM=/opt/xilinx/platforms/xilinx_vck5000_gen4x8_qdma_2_202220_1/xilinx_vck5000_gen4x8_qdma_2_202220_1.xpfm
export XILINX_XRT=/opt/xilinx/xrt
export VITIS_HLS_INCLUDE=/s3/Xilinx_Vitis_2022.2/Vitis_HLS/2022.2/include
```

Set `XILINXD_LICENSE_FILE` according to the local license server or file; no
user-specific license path is embedded in the final scripts.

## Host-only build

```bash
make
```

This creates `host/llama3_attention_host.exe` using C++17 and XRT.

## Cheap compile gate

```bash
scripts/check_compile.sh
```

The gate checks the reciprocal LUT, PL transport, production-topology x86sim,
the exact 64-tile AIE mapping and a PL compile without full link.
Generated files are placed under `build.check/`.

## Complete hardware rebuild

```bash
scripts/rebuild_hardware.sh
```

This cleanly rebuilds the AIE archive, PL XO, linked XSA and packaged xclbin
under `build.hw/`. A full hardware link and route takes approximately
2 hours; do not start it merely to reproduce the published measurements because
the SHA-pinned tested binary is available under `prebuilt/vck5000/`.

The streaming connectivity configuration is in `link/eight_groups_packet.cfg`.
