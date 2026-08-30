# V21 PV6 validation summary

| Gate | Result |
|---|---|
| Reciprocal LUT test | PASS |
| PL transport C++ test | PASS |
| Production-topology x86sim | PASS |
| Score load/reuse events | 8 / 8 |
| PV load/reuse events | 48 / 48 |
| AIE compute tiles | 64 / 64 |
| Physical PLIO | 16 |
| PL compile and loop-II constraints | PASS |
| Routed 300 MHz timing | PASS |
| Independent B1 board samples | 10 / 10 completed |
| Full-output functional bound 0.05 | PASS |
| Internal maximum-error target 0.003 | FAIL (max 0.004983485) |
| Repeated launch without reset/reload | FAIL (second invocation stalls) |
| B8 single launch | FAIL (120-second timeout) |

The supported paper protocol is ten independent B1 launches, each preceded by
a board reset and xclbin reload. The results establish isolated-request
latency and functional correctness, not sustained throughput.

Sources: `docs/v21_pv6_hardware_performance_report.md`,
`docs/PERFORMANCE.md`, and `results/final/comparison_summary.json`.
