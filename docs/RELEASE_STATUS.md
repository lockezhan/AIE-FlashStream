# Public-release status

## Source of truth

- Architecture: Eight groups, one Score + one Softmax + six PV per
  group, 64 AIE compute tiles
- Connectivity: `link/eight_groups_packet.cfg`
- Prebuilt xclbin:
  `prebuilt/vck5000/llama3_attention.xclbin`
- Prebuilt xclbin SHA-256:
  `bff430b2b827c72469e5b147027786de1cf98a5f5ce587910f04590547d08b58`

## Evidence status

- Matched B1 comparison: complete, ten independent samples per design
- Pure-PL baseline functional and 0.003 gates: pass
- AIE-FlashStream functional 0.05 gate: pass
- AIE-FlashStream internal 0.003 target: fail, maximum error 0.004983485
- Repeated invocation without reset/reload: fail
- B=8 single launch: fail, 120-second timeout
- Sustained-throughput claim: not supported

## Release metadata status

1. **License**: Apache-2.0 license full text is included in `LICENSE`.
2. **Citation**: Paper publication and DOI metadata will be added upon conference proceedings release.

The liveness and strict-error limitations do not prevent source publication if
they remain disclosed, but they prevent sustained-throughput and strict-0.003
claims.
