# Public-release status

## Frozen source of truth

- Architecture: V21 PV6, eight groups, one Score + one Softmax + six PV per
  group, 64 AIE compute tiles
- Final connectivity: `link/eight_groups_packet_pv6.cfg`
- Final prebuilt xclbin:
  `prebuilt/vck5000/v21_pv6/llama3_attention_v21_pv6.xclbin`
- Final xclbin SHA-256:
  `bff430b2b827c72469e5b147027786de1cf98a5f5ce587910f04590547d08b58`

The former root `llama3_attention.xclbin` had SHA-256
`69b14fa6f221583a537cd9e97901f9b2504c8d82e7a40038fdadd65eb3cc06e5`
and did not match V21. It was removed from the release layout rather than
silently relabeled.

## Evidence status

- Matched B1 comparison: complete, ten independent samples per design
- Pure-PL functional and 0.003 gates: pass
- V21 functional 0.05 gate: pass
- V21 internal 0.003 target: fail, maximum error 0.004983485
- V21 repeated invocation without reset/reload: fail
- V21 B8 single launch: fail, 120-second timeout
- Sustained-throughput claim: not supported

## Remaining public-release blockers

1. **LICENSE is missing.** The maintainer must select a license before public
   open-source release. Apache-2.0 or MIT are reasonable choices, but no choice
   has been made in this repository.
2. **Citation metadata is incomplete.** `CITATION.cff.template` still requires
   the final title, authors, proceedings title and DOI when assigned.

The liveness and strict-error limitations do not prevent source publication if
they remain disclosed, but they prevent sustained-throughput and strict-0.003
claims.
