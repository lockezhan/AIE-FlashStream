# Running AIE-FlashStream

## Verify the binary

Verify the prebuilt hardware artifact:

```bash
cd prebuilt/vck5000
sha256sum -c SHA256SUMS
cd ../..
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
  --xclbin prebuilt/vck5000/llama3_attention.xclbin \
  --batch 1 --warmup 0 --runs 1 --seed 7 --verify --profile
```

The host measures H2D + kernel + D2H and verifies all 131,072 BF16 outputs
against the CPU oracle. Functional acceptance is maximum absolute error below
0.05; the additional 0.003 target is reported separately.

## Reproducible board campaigns

```bash
# Ten independent B1 samples, reset/reload before every launch
scripts/benchmark_aie_flashstream_b1.sh \
  prebuilt/vck5000/llama3_attention.xclbin
```

The paper comparison uses ten independent B1 samples for both the Pure-PL baseline
and AIE-FlashStream. See `docs/PERFORMANCE.md` and `results/final/`.
