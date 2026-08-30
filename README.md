# AIE-FlashStream

AIE-FlashStream is a GQA-aware Llama3-8B causal-attention accelerator for the
AMD Versal VCK5000. “FlashStream” names the streaming hardware dataflow; this
project does **not** implement or claim the FlashAttention algorithm.

![AIE-FlashStream Final Architecture](figures/AIE-FlashStream_Final_Architecture.png)

## Final architecture

The frozen workload is `S=32`, `HQ=32`, `HKV=8`, `D=128`, with four query
heads sharing each KV head. The eight GQA groups each contain one Score tile,
one Softmax tile and six PV tiles:

```text
8 groups × (1 Score + 1 Softmax + 6 PV) = 64 AIE compute tiles
```

The six PV slices cover dimensions `24/24/16/24/24/16` at offsets
`0/24/48/64/88/112`. Physical PLIO comprises two Q, two K, six V and six O
streams (16 total).

The PL shell accepts BF16 Q/K/V, quantizes Q/K to INT8, packetizes and
schedules the streams, and reconstructs BF16 output into four contiguous
32-dimensional planes. AIE tiles execute INT8 Score, FP32 Softmax and FP32 PV.
Each graph invocation uses two phases: phase 0 processes Q0/Q1 and loads K/V;
phase 1 processes Q2/Q3 and reuses the tile-local K/V state.

## Tested platform

- Board: AMD Versal VCK5000
- Platform: `xilinx_vck5000_gen4x8_qdma_2_202220_1`
- Vitis/Vivado: 2022.2
- XRT: 2.15.225
- PL clock: 300 MHz

## Headline performance (Matched B=1 vs Pure-PL Baseline)

| Metric | Pure-PL V1 Baseline | AIE-FlashStream | Speedup / Improvement |
|---|---:|---:|---:|
| **Kernel Latency (median)** | 63.910 ms | **0.470 ms** | **135.87×** (99.26% reduction) |
| **End-to-End Latency (median)** | 64.373 ms | **0.994 ms** | **64.74×** (98.46% reduction) |
| **Numerical Accuracy (MAE)** | — | **3.73e-4** | INT8 quantization error |

## Repository structure

- `aie/`: final AIE graph, kernels and tile placement
- `pl/`: HLS quantization, packet transport and output reconstruction
- `link/eight_groups_packet_pv6.cfg`: final PL/AIE connectivity
- `host/`: XRT host, verification and benchmark reporting
- `scripts/`: final compile, rebuild and board benchmark entry points
- `tests/`: numerical oracle and transport/compile gates
- `prebuilt/vck5000/v21_pv6/`: SHA-pinned board-tested V21 xclbin
- `docs/`: architecture, reproducibility and performance sources of truth
- `reports/`: compact final implementation summaries
- `results/`: immutable raw evidence and paper-level summaries

## Quick start

```bash
export XILINX_XRT=/opt/xilinx/xrt
make

cd prebuilt/vck5000/v21_pv6
sha256sum -c SHA256SUMS
cd ../../..

/opt/xilinx/xrt/bin/xbutil reset --device 0000:af:00.1
./host/llama3_attention_host.exe \
  --xclbin prebuilt/vck5000/v21_pv6/llama3_attention_v21_pv6.xclbin \
  --batch 1 --warmup 0 --runs 1 --seed 7 --verify --profile
```

See [architecture](docs/ARCHITECTURE.md),
[reproducibility](docs/REPRODUCIBILITY.md),
[performance](docs/PERFORMANCE.md), [build](BUILD.md), and [run](RUN.md).

The current xclbin completes isolated requests but does not support repeated
launches without reset/reload. This limitation is part of the published
performance protocol and must not be hidden in throughput claims.
