# Validation Summary

This report is a sanitized transcription of completed tool and board gates.

| Gate | Result |
|---|---|
| Transport/math Python model | PASS |
| Value-tail ASan/UBSan | PASS |
| PL C simulation | PASS |
| x86 simulation (`graph.run(2)`) | PASS |
| AIE simulation (`graph.run(2)`) | PASS |
| AIE compiler warnings/critical/errors | 0 / 0 / 0 |
| Active AIE compute tiles | 64 / 64 |
| AIE router errors | 0 |
| VPL check errors | 0 |
| POST-VPL check errors | 0 |
| Board functional verification, batch 1 | PASS |
| Board functional verification, batch 8 | PASS |

The one-group graph analysis explicitly emitted a single Softmax source with
five broadcast destinations. AIE simulation profiling showed all five PV paths
within a narrow cycle range; the fifth path did not serialize or lag the others.

The 28/28/24/24/24 output is reconstructed into four sequential 32-D DDR planes.
HLS reports achieved II=1 for all five packet drains, all four random-index
collection loops, and all four sequential DDR write loops. The pair splitter is
II=2 while emitting two rows per iteration, i.e. one output row per cycle, and
is faster than its six/seven-word upstream producer.
