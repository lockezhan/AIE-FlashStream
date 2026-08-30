# Final V21 PV6 architecture

This document describes only the frozen release architecture.

## 1. Workload definition

The accelerator implements one dense causal-GQA attention workload from
Llama3-8B with sequence length 32, 32 query heads, eight KV heads and head
dimension 128. Four query heads share one K/V head, producing eight independent
GQA groups. Inputs and outputs are BF16.

## 2. PL/AIE partition

The PL shell reads Q/K/V from DDR, converts BF16 Q and K to INT8, packetizes
the physical streams, schedules two execution phases and reconstructs the six
AIE output slices into four contiguous 32-dimensional BF16 DDR planes.

The AIE array performs INT8 Score with INT32 accumulation, FP32 Softmax and
FP32 PV accumulation before BF16 output conversion. K/V state is retained in
tile-local memory between the two phases.

## 3. GQA-aware resource allocation

Each of the eight groups contains exactly:

- one Score tile;
- one Softmax tile;
- six PV tiles.

Thus `8 × (1 + 1 + 6) = 64` AIE compute tiles. K/V sharing is expressed in the
hardware topology rather than by duplicating Score computation for every query
head.

## 4. Stateful two-phase KV reuse

One graph invocation covers the four query heads associated with a KV head:

- phase 0: process Q0 and Q1 while loading K and V into tile-local storage;
- phase 1: process Q2 and Q3 while reusing the resident K and V.

The second phase consumes no duplicate K/V payload. Score and all six PV tiles
use the same load/reuse protocol.

## 5. Six-way PV decomposition

The 128 output dimensions are statically divided as follows:

| PV slice | Offset | Width | V words/group | Output words/packet |
|---:|---:|---:|---:|---:|
| 0 | 0 | 24 | 96 | 192 |
| 1 | 24 | 24 | 96 | 192 |
| 2 | 48 | 16 | 64 | 128 |
| 3 | 64 | 24 | 96 | 192 |
| 4 | 88 | 24 | 96 | 192 |
| 5 | 112 | 16 | 64 | 128 |

The widths sum to 128, and each group receives 512 raw 128-bit V words.

## 6. AIE tile layout

Group `g` occupies columns `4g..4g+3` on rows 1 and 2:

```text
row 1:  Score | PV5 | PV4 | Softmax
row 2:  PV3   | PV2 | PV1 | PV0
```

Eight groups occupy 32 columns and 64 compute tiles. This placement is encoded
in `aie/llama3_gqa_8group_packet.cpp`.

## 7. PL/AIE stream topology

The 16 physical PLIO streams are two Q, two K, six V and six O. Q/K streams
use packet splitters to address four groups per physical lane; each V stream
uses an eight-way splitter and each O stream an eight-way merger.

```text
BF16 Q/K/V
    |
PL quantize + packetize + two-phase scheduler
    |  2Q + 2K + 6V
    v
8 GQA groups in parallel
    |  each group: Score -> Softmax -> {PV0..PV5}
    |              phase 0 load K/V; phase 1 reuse K/V
    v  6O
PL depacketize + six-slice collector
    |
four contiguous 32-D BF16 output planes
```

## 8. Output reconstruction

The six output streams carry slices `24/24/16/24/24/16`. PL collects the
packetized rows and forms four sequential 32-dimensional planes for the Host.
Only the rows required at the plane boundaries are duplicated; the external
tensor remains `[B, HQ, S, D]` with `D=128`.
