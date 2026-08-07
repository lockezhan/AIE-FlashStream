# Running Benchmarks

## Basic Run (Batch = 1)
```bash
./host/llama3_attention_host.exe --xclbin llama3_attention.xclbin
```

## Batch Run (Batch = 8)
```bash
./host/llama3_attention_host.exe --xclbin llama3_attention.xclbin --batch 8
```

## Save Results to JSON / CSV
```bash
./host/llama3_attention_host.exe --xclbin llama3_attention.xclbin --batch 8 \
  --output-json results.json --output-csv results.csv
```
