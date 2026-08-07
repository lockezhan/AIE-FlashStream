# Building AIE-FlashStream

## Prerequisites
- Xilinx XRT 2022.2+ installed at `/opt/xilinx/xrt` (or set `XILINX_XRT`).
- C++17 compiler (`g++`).

## Compiling Host Executable

```bash
make
```

This compiles `host/llama3_attention_host.cpp` to `host/llama3_attention_host.exe`.
