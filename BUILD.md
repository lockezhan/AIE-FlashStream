# Building AIE-FlashStream

The final hardware architecture is V21 PV6. The scripts are the executable
source of truth; this document defines their environment and intended use.

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
scripts/check_v21_pv6_compile.sh
```

The gate checks the reciprocal LUT, PL transport, production-topology x86sim,
the exact 64-tile AIE mapping and a PL compile without full link.
Generated files are placed under `build.check_v21_pv6/` and
`aie/build.x86sim_v21_pv6/`.

## Complete V21 hardware build

```bash
scripts/rebuild_v21_pv6.sh
```

This cleanly rebuilds the AIE archive, PL XO, linked XSA and packaged xclbin
under `build.hw_v21_pv6/`. A full hardware link and route can take significant
time; do not start it merely to reproduce the published measurements because
the SHA-pinned tested image is available under `prebuilt/vck5000/v21_pv6/`.

The final connectivity file is `link/eight_groups_packet_pv6.cfg`. Historical
build scripts are retained only under `scripts/development/` and are not final
release entry points.
