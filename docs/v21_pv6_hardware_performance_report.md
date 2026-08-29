# V21 PV6 hardware performance report

## Executive result

V21 implements GQA-aware resource rebalancing on AMD Versal VCK5000. Each of
eight GQA groups contains one Score kernel, one Softmax kernel, and six PV
kernels. The resulting 8 Score + 8 Softmax + 48 PV mapping occupies all 64 AIE
compute tiles. The PV partition is `24/24/16/24/24/16` at offsets
`0/24/48/64/88/112`; Q/K scheduling and the two-query, two-phase V20 protocol
are unchanged.

The routed design meets 300 MHz timing and a cold single request completes in
approximately 0.482 ms kernel time. Full output verification is functionally
correct. However, a second invocation without reprogramming the board stalls.
Consequently the present artifact supports a paper claim about implementation,
timing closure, single-request latency, and numerical behavior, but **does not
yet support a sustained-throughput or warmup-10/runs-100 latency claim**.

## Experimental setup

| Item | Configuration |
|---|---|
| Board | AMD Versal VCK5000, BDF `0000:af:00.1` |
| Platform | `xilinx_vck5000_gen4x8_qdma_2_202220_1` |
| Vitis/Vivado | 2022.2 |
| XRT | 2.15.225 |
| PL clock | 300 MHz |
| Workload | B=1, S=32, 32 Q heads, 8 KV heads, D=128 |
| Numeric formats | BF16 I/O, INT8 Score, FP32 Softmax/PV |
| Seed | 7 |
| XCLBIN SHA256 | `bff430b2b827c72469e5b147027786de1cf98a5f5ce587910f04590547d08b58` |

## Architecture and transport

| Metric | V21 value |
|---|---:|
| AIE compute tiles | 64 |
| Score / Softmax / PV tiles | 8 / 8 / 48 |
| Physical PLIO | 16 (2 Q, 2 K, 6 V, 6 O) |
| Maximum PV width | 24 dimensions (6 four-D quads) |
| V payload per group | 512 x 128-bit words |
| Host output | four contiguous 32-D DDR planes |

The AIE compiler reported exactly 64 active cores and 16 AIE IP ports. x86sim
verified 8 Score load + 8 reuse events and 48 PV load + 48 reuse events, with
all 131,072 output values present, no NaN/Inf, and no simulation deadlock.

## Implementation results

| Metric | Result |
|---|---:|
| HLS estimated Fmax | 411 MHz |
| Routed WNS | 0.119 ns |
| Routed TNS | 0.000 ns |
| Routed WHS | 0.010 ns |
| Routed THS | 0.000 ns |
| PL registers (kernel synthesis) | 99,131 (5.51%) |
| PL CLB LUTs (kernel synthesis) | 62,685 (6.97%) |
| Block RAM tiles | 186.5 (19.29%) |
| DSP58 | 24 (1.22%) |
| URAM | 0 |

All user timing constraints are met. The HLS compile satisfied all loop II
constraints. Vitis emitted its established AXIS clock-association warnings for
the generated AIE interfaces; these did not prevent route or a successful
single hardware execution.

## Measured single-request performance

Three independently observed cold single-request kernel times were 0.479731,
0.481711, and 0.483979 ms. Their mean is 0.481807 ms, median 0.481711 ms, and
range 0.479731--0.483979 ms. At the median, the host's attention operation
count corresponds to 17.958 GFLOP/s. These samples are reported as cold
single-request observations, not as a steady-state distribution.

## Numerical accuracy

| Metric | Result |
|---|---:|
| Verified values | 131,072 |
| NaN / Inf | 0 / 0 |
| Mean absolute error | 0.000373 |
| P99 absolute error | 0.001737 |
| P99.9 absolute error | 0.002876 |
| Maximum absolute error | 0.004983 |
| Worst coordinate | B0, head 1, row 3, dim 38 |

The host functional threshold of 0.05 passes. The stricter internal target of
0.003 fails at the single worst value; therefore the correct description is
“functionally correct with a numerical warning,” not strict 0.003 compliance.

## Sustained-run limitation

The required warmup=10/runs=100 experiment timed out after 120 seconds. A
line-buffered diagnostic (`warmup=0`, `runs=10`) printed run 1 at 0.483979 ms
and then timed out while waiting for run 2. This is a repeat-invocation
liveness defect. Until fixed, throughput, requests/s, tokens/s, p95, and p99
must not be presented as steady-state V21 measurements.

## Paper-ready wording

> V21 reallocates the eight Score tiles removed by the GQA-aware V20 design to
> one additional PV engine per group, producing a 64-tile organization with 8
> Score, 8 Softmax, and 48 PV kernels. The 24/24/16/24/24/16 decomposition
> reduces the maximum per-PV dimension work from seven to six four-dimensional
> quads. On VCK5000, the design routes at 300 MHz with 0.119 ns setup slack and
> completes an isolated B1-S32 request in approximately 0.482 ms. Full-model
> verification is functionally correct (mean absolute error 3.73e-4), although
> the current artifact requires further work to support repeated invocations.

## Reproduction

```bash
# Cheap functional and compile gates (no full link)
scripts/check_v21_pv6_compile.sh

# Full clean hardware rebuild
scripts/rebuild_v21_pv6.sh

# Board single-request verification and repeat-run diagnostic
/opt/xilinx/xrt/bin/xbutil reset --device 0000:af:00.1
scripts/benchmark_v21_pv6.sh
```

Raw summary: `results/v21_pv6/single_run_summary.txt`. The host now identifies
the design as `Llama3-AIE-FlashStream V21 PV6`, host version `21.0.0`, and
reports `64 / 64` AIE tiles, fixing the stale V20 `56 / 64` display.
