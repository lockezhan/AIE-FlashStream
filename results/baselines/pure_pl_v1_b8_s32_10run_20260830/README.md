# Pure-PL V1 baseline evidence

This directory is the version-controlled evidence package for the 10-run
VCK5000 pure-PL baseline campaign used by the paper comparison.

## Provenance

- Source project: `/s3/zhangyann/AIE-FPT26/llama3-attention`
- Source build: `build.hw_v1_baseline`
- Board experiment time: 2026-08-30 13:06:03 (Asia/Shanghai)
- Device: VCK5000, BDF `0000:af:00.1`
- XRT: 2.15.225
- Workload: B=8, S=32, Hq/Hkv=32/8, D=128
- Warmup/measured/seed: 5/10/7
- Pure-PL XCLBIN SHA-256:
  `15b01daedc8d6ba497eaeb2644b299d9bc46c6069d06580db8025fe01cce9b2c`
- Pure-PL Host SHA-256:
  `47b2b63547794a422865b12f5961cd8bc87cd97978b59f28510e1f009cc6f12e`
- Same-day V19 XCLBIN SHA-256:
  `1022984139d8e79aa99a5c4ebafd3d0d5eeae4e341d89552b3f1b2adcf1b522f`
- Original accidental standalone commit: `61e5ed0` (superseded by the
  AIE-FlashStream repository commit containing this directory)

## Contents

- `run.log`: pure-PL command, board/XRT metadata and complete output
- `v1_pl_baseline_b8_s32_w5_r10_seed7.json`: pure-PL aggregate metrics
- `v1_pl_baseline_b8_s32_w5_r10_seed7.csv`: pure-PL raw samples
- `v19_peer_run.log`: same-day V19 peer command and output
- `v19_proposed_same_day_b8_s32_w5_r10_seed7.json`: V19 aggregate metrics
- `v19_proposed_same_day_b8_s32_w5_r10_seed7.csv`: V19 raw samples
- `SHA256SUMS`: immutable hashes of the six raw evidence files

Run `sha256sum -c SHA256SUMS` from this directory to verify the package.
The paper-facing interpretation is in
`docs/baselines/pure_pl_v1_b8_s32_10run_20260830.md`.
