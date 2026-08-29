# V20 single-Score / two-phase token ledger

This ledger is for one `batch` processed by one PL kernel dataflow
invocation.  The fixed model dimensions are sequence length 32, 32 query
heads, eight KV heads, GQA 4:1, and head dimension 128.  A Q or K payload
contains eight 128-bit scale words followed by 256 128-bit INT8 words, for
264 payload words per packet.

The physical Q/K lane mapping is:

- lane 0: G0, G1, G2, G3 (packet IDs 0, 1, 2, 3)
- lane 1: G4, G5, G6, G7 (local packet IDs 0, 1, 2, 3)

Q is scheduled wave-major on both lanes: Q0 for all four lane-local groups,
then Q1 for all groups, then Q2, then Q3.  Q0/Q1 feed AIE invocation 0 and
Q2/Q3 feed invocation 1.  Each K and V packet is emitted only once, before
the phase-0 consumer caches it.

## Internal stream accounting

| Stream | Producer tokens | Consumer tokens | Balance and interpretation |
|---|---:|---:|---|
| `q_ddr[0]` | 4 groups x 4 Q/group x 32 rows x 4 DDR words = 2048 `DdrWord` | 2048 | balanced; G0..G3 |
| `q_ddr[1]` | 2048 `DdrWord` | 2048 | balanced; G4..G7 |
| `q_scales[0]` | 16 Q x 32 rows x 2 scales = 1024 `ap_uint<16>` | 16 packets x 8 words x 8 scales = 1024 | balanced |
| `q_scales[1]` | 1024 | 1024 | balanced |
| `q_quantized[0]` | 16 Q x 256 = 4096 `ap_uint<128>` | 4096 | balanced |
| `q_quantized[1]` | 4096 | 4096 | balanced |
| `q_payload[0]` | 16 packets x 264 = 4224 `ap_uint<128>` | 4224 | balanced; 16 Q packets |
| `q_payload[1]` | 4224 | 4224 | balanced; 16 Q packets |
| `k_ddr` | 8 KV x 32 rows x 4 DDR words = 1024 `DdrWord` | 1024 | balanced; every KV head read once |
| `k_scales[0]` | 4 K x 32 rows x 2 scales = 256 `ap_uint<16>` | 4 packets x 8 words x 8 scales = 256 | balanced; G0..G3 only |
| `k_scales[1]` | 256 | 256 | balanced; G4..G7 only |
| `k_quantized[0]` | 4 K x 256 = 1024 `ap_uint<128>` | 1024 | balanced; no duplicate K |
| `k_quantized[1]` | 1024 | 1024 | balanced; no duplicate K |
| `k_payload[0]` | 4 packets x 264 = 1056 `ap_uint<128>` | 1056 | balanced; four K packets |
| `k_payload[1]` | 1056 | 1056 | balanced; four K packets |
| `v_raw[0]` | 8 groups x 112 = 896 `ap_uint<128>` | 896 | balanced; ValueD=28 |
| `v_raw[1]` | 896 | 896 | balanced; ValueD=28 |
| `v_raw[2]` | 8 groups x 96 = 768 `ap_uint<128>` | 768 | balanced; ValueD=24 |
| `v_raw[3]` | 768 | 768 | balanced; ValueD=24 |
| `v_raw[4]` | 768 | 768 | balanced; ValueD=24 |
| `output_pairs28[0]` | 8 groups x 2 packets x 32 pairs = 512 `SlicePair<28>` | 512 | balanced |
| `output_pairs28[1]` | 512 | 512 | balanced |
| `output_pairs24[0]` | 512 `SlicePair<24>` | 512 | balanced |
| `output_pairs24[1]` | 512 | 512 | balanced |
| `output_pairs24[2]` | 512 | 512 | balanced |
| `slice_rows[0]` | 512 pairs x 2 = 1024 `SliceRow` | 1024 by output-plane 0 | balanced |
| `slice_rows[1]` | 1024 | 1024 by fanout 0 | balanced |
| `slice_rows[2]` | 1024 | 1024 by fanout 1 | balanced |
| `slice_rows[3]` | 1024 | 1024 by fanout 2 | balanced |
| `slice_rows[4]` | 1024 | 1024 by output-plane 3 | balanced |
| `cross_left[0]` | 1024 | 1024 by output-plane 0 | balanced |
| `cross_right[0]` | 1024 | 1024 by output-plane 1 | balanced |
| `cross_left[1]` | 1024 | 1024 by output-plane 1 | balanced |
| `cross_right[1]` | 1024 | 1024 by output-plane 2 | balanced |
| `cross_left[2]` | 1024 | 1024 by output-plane 2 | balanced |
| `cross_right[2]` | 1024 | 1024 by output-plane 3 | balanced |

## Physical packet accounting

The packet adapter adds one 128-bit AXIS beat per packet to carry the packet
header and trailing payload bits.  Packet counts, rather than adapter beats,
define the AIE transaction contract:

| Physical stream | Packets per batch | Payload contract |
|---|---:|---|
| `packet_q0` | 16 | four waves x four groups, 264 payload words/packet |
| `packet_q1` | 16 | four waves x four groups, 264 payload words/packet |
| `packet_k0` | 4 | one K packet for each of G0..G3 |
| `packet_k1` | 4 | one K packet for each of G4..G7 |
| each `packet_v[0..4]` | 8 | one V slice packet per group |
| each `packet_o[0..4]` | 16 | two 2Q packets per group |

For every output slice, 16 packets x 64 rows = 1024 rows, or 16 packets x
32 row pairs = 512 pairs.  Output phase is reconstructed with
`packets_seen[group]`; global merge arrival order is never used as phase.

## AIE invocation balance

Per group, invocation 0 consumes Q0 and Q1 plus one K packet and one packet
from each V slice, then emits one 2Q window per output slice.  Invocation 1
consumes Q2 and Q3, consumes zero K/V packets, reuses tile-local K/V caches,
and emits the second 2Q window.  Thus each group consumes four Q packets,
one K packet, five V packets, and produces ten output packets across the five
slices.  The graph is run for exactly two iterations per complete 4Q group.
