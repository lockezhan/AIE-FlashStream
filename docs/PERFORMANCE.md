# Performance source of truth

## Scope and fairness

This page compares one dense Llama3-8B causal-GQA attention workload, not a
full Llama3 model. Both designs use `B=1`, `S=32`, `HQ=32`, `HKV=8`, `D=128`,
seed 7 and BF16 input/output. Each data point is one Host process and one
kernel launch after a VCK5000 reset/reload; ten independent samples are used
per design. Timing scope is H2D + kernel + D2H.

The Pure-PL design computes BF16 Score with FP32 Softmax/PV. AIE-FlashStream uses
INT8 Score with FP32 Softmax/PV. This is workload- and protocol-matched, but
not arithmetic-precision-identical; numerical error is reported with latency.

## Paper-ready matched B1 comparison

| Metric | Pure-PL baseline | AIE-FlashStream | Speedup / Improvement |
|---|---:|---:|---:|
| Independent samples | 10 | 10 | — |
| Kernel median (ms) | 63.910412 | 0.470387 | 135.87×; 99.264% lower |
| Kernel mean (ms) | 63.907616 | 0.474839 | 134.59× |
| Kernel min/max (ms) | 63.864952 / 63.931048 | 0.414302 / 0.519036 | — |
| E2E median (ms) | 64.373416 | 0.994343 | 64.74×; 98.455% lower |
| E2E mean (ms) | 64.363983 | 1.002077 | 64.23× |
| E2E min/max (ms) | 64.282666 / 64.407459 | 0.918540 / 1.067933 | — |
| Effective kernel GFLOP/s | 0.135357 | 18.390712 | 135.87× |
| Effective E2E GFLOP/s | 0.134384 | 8.699968 | 64.74× |

Effective GFLOP/s uses the Host's fixed count of 8,650,752 attention
operations per request. It is an application-effective metric, not AIE peak
throughput.

## Numerical accuracy

The same seeded input is verified against the FP32 CPU oracle in every sample.

| Metric | Pure-PL baseline | AIE-FlashStream |
|---|---:|---:|
| Mean absolute error | 0.000185180 | 0.000373284 |
| P99 absolute error | 0.000990033 | 0.001737416 |
| P99.9 absolute error | 0.001863956 | 0.002876103 |
| Maximum absolute error | 0.001953006 | 0.004983485 |
| Functional bound 0.05 | PASS | PASS |
| Internal target 0.003 | PASS | FAIL |

The valid statement is “functionally correct with a numerical warning” for
AIE-FlashStream. Equal-precision, bitwise-equivalence and strict-0.003 claims are invalid.

## Routed implementation data

| Metric | Pure-PL baseline | AIE-FlashStream |
|---|---:|---:|
| AIE compute tiles | 0 | 64 |
| Physical PLIO | 0 | 16 |
| PL registers | 43,710 | 99,131 |
| CLB LUT | 34,815 | 62,685 |
| BRAM tiles | 45 | 186.5 |
| DSP58 | 19 | 24 |
| URAM | 131 | 0 |
| PL frequency | 300 MHz | 300 MHz |
| Routed WNS | 0.000 ns | +0.119 ns |
| Routed TNS | 0.000 ns | 0.000 ns |

AIE-FlashStream HLS estimated Fmax is 411 MHz; routed hold slack is WHS `+0.010 ns`, THS
`0.000 ns`.

## Known limits

- B=8 as one single kernel launch timed out at 120 seconds; it is not a valid
  matched B8 data point.
- A second invocation without reset/reload stalls. The independent B1
  campaign is not a sustained-throughput distribution.
- With ten samples, nearest-rank p95 is the maximum sample and is not used as a
  headline tail-latency claim.
- QKV projections, MLP, layer normalization, KV-cache management and token
  sampling are outside this measured kernel.

Machine-readable values and hashes are in `results/final/`. Raw matched B1
evidence is in `results/matched_b1/`.
