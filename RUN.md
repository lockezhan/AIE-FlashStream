# Running the final V21 PV6 design

## Verify the binary

The root-level binary from earlier development is not the final V21 image and
is intentionally absent from the release layout. Verify the prebuilt artifact:

```bash
cd prebuilt/vck5000/v21_pv6
sha256sum -c SHA256SUMS
cd ../../..
```

Expected SHA-256:

```text
bff430b2b827c72469e5b147027786de1cf98a5f5ce587910f04590547d08b58
```

## One verified request

```bash
make
/opt/xilinx/xrt/bin/xbutil reset --device 0000:af:00.1
./host/llama3_attention_host.exe \
  --xclbin prebuilt/vck5000/v21_pv6/llama3_attention_v21_pv6.xclbin \
  --batch 1 --warmup 0 --runs 1 --seed 7 --verify --profile
```

The host measures H2D + kernel + D2H and verifies all 131,072 BF16 outputs
against the CPU oracle. Functional acceptance is maximum absolute error below
0.05; the additional 0.003 target is reported separately.

## Reproducible board campaigns

```bash
# Single verified request plus known repeat-invocation diagnostic
scripts/benchmark_v21_pv6.sh \
  prebuilt/vck5000/v21_pv6/llama3_attention_v21_pv6.xclbin \
  results/v21_pv6

# Ten independent B1 samples, reset/reload before every launch
scripts/benchmark_v21_pv6_matched_b1.sh \
  prebuilt/vck5000/v21_pv6/llama3_attention_v21_pv6.xclbin
```

V21 currently stalls on a second launch without reset/reload, and a B8 single
launch timed out. Therefore the paper comparison uses ten independent B1
samples for both Pure-PL and V21; it does not claim sustained throughput.
See `docs/PERFORMANCE.md` and `results/final/`.
