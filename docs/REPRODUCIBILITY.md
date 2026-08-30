# Reproducibility

## Tested environment

| Item | Tested value |
|---|---|
| Board | AMD Versal VCK5000 |
| Platform | `xilinx_vck5000_gen4x8_qdma_2_202220_1` |
| Vitis/Vivado | 2022.2 |
| XRT | 2.15.225 |
| PL clock | 300 MHz |
| Device BDF | `0000:af:00.1` |

Configure paths for the local installation:

```bash
export VITIS_ROOT=/s3/Xilinx_Vitis_2022.2/Vitis/2022.2
export VITIS_SETTINGS="$VITIS_ROOT/settings64.sh"
export VCK5000_PLATFORM=/opt/xilinx/platforms/xilinx_vck5000_gen4x8_qdma_2_202220_1/xilinx_vck5000_gen4x8_qdma_2_202220_1.xpfm
export XILINX_XRT=/opt/xilinx/xrt
export XRT_ROOT="$XILINX_XRT"
export VITIS_HLS_INCLUDE=/s3/Xilinx_Vitis_2022.2/Vitis_HLS/2022.2/include
```

Set `XILINXD_LICENSE_FILE` for the local license service. The scripts accept
the environment variables above and do not depend on a maintainer's home
directory.

## Host build

```bash
make
./host/llama3_attention_host.exe --version
```

## Cheap compile gate

```bash
scripts/check_v21_pv6_compile.sh
```

This runs the reciprocal-LUT test, PL transport C++ test, production-topology
x86sim, 64-AIE-tile hardware compile gate and PL compile-only gate. Generated
directories are `build.check_v21_pv6/` and `aie/build.x86sim_v21_pv6/`.

## Full hardware rebuild

```bash
scripts/rebuild_v21_pv6.sh
```

The clean build is written to `build.hw_v21_pv6/`. It produces the AIE
archive, PL XO, linked XSA, packaged xclbin and artifact hashes. Full link and
route can take significant time. The public prebuilt image is copied only
after its SHA and board evidence are accepted.

## Board reset, execution and verification

```bash
cd prebuilt/vck5000
sha256sum -c SHA256SUMS
cd ../..

"$XRT_ROOT/bin/xbutil" reset --device 0000:af:00.1
./host/llama3_attention_host.exe \
  --xclbin prebuilt/vck5000/llama3_attention.xclbin \
  --batch 1 --warmup 0 --runs 1 --seed 7 --verify --profile
```

Verification covers every output element and reports mean, p99, p99.9 and
maximum absolute error. Functional pass uses the declared 0.05 maximum-error
bound; the 0.003 internal target is an additional warning gate.

## Benchmark outputs

- `scripts/benchmark_aie_flashstream_b1.sh`: ten independent AIE-FlashStream B1 samples;
- `scripts/benchmark_pure_pl_b1.sh`: ten independent Pure-PL B1
  samples using the separately archived baseline build; set
  `PURE_PL_ROOT=/path/to/llama3-attention` before running it.

Matched B1 scripts reset the card, poll until the device is ready, allow a
short enumeration-settle interval, reload the pinned xclbin and execute one
kernel launch per process. Raw JSON/CSV/log evidence is under
`results/matched_b1/`; paper-level summaries are
under `results/final/`.

The accelerator artifact is designed for single-request execution: a second invocation
without reset/reload stalls. Do not use the independent-sample results to
claim steady-state requests/s or tail latency.
