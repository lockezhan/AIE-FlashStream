# V21 PV6 token ledger

The V20 Q and K contracts are unchanged: two physical Q streams use split4,
two physical K streams use split4, and each graph invocation processes two
queries. Phase 0 loads K/V and phase 1 reuses the resident caches.

| PV slice | Offset | Width | V raw 128-bit words/group | Output payload words/packet |
|---:|---:|---:|---:|---:|
| 0 | 0 | 24 | 96 | 192 |
| 1 | 24 | 24 | 96 | 192 |
| 2 | 48 | 16 | 64 | 128 |
| 3 | 64 | 24 | 96 | 192 |
| 4 | 88 | 24 | 96 | 192 |
| 5 | 112 | 16 | 64 | 128 |

V totals 512 raw 128-bit words per group, identical to V20. Each physical V
slice carries eight packets. Each output slice carries 16 packets (two per
group), 512 row-pairs, and 1024 rows. Every dataflow stream has one producer
and one consumer; only slice1 and slice4 rows are explicitly duplicated, once,
for reconstruction of the four contiguous host planes.
