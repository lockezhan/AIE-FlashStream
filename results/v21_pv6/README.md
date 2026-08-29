# V21 PV6 hardware results

This directory holds the small, version-controlled summary of the V21 board
experiment. Machine-generated JSON/CSV files are ignored by Git and can be
regenerated with:

```bash
/opt/xilinx/xrt/bin/xbutil reset --device 0000:af:00.1
scripts/benchmark_v21_pv6.sh
```

The tested XCLBIN SHA256 is
`bff430b2b827c72469e5b147027786de1cf98a5f5ce587910f04590547d08b58`.
Single requests complete and pass the functional error limit. Repeated use of
the same loaded design currently stalls on the second invocation, so no
warmup-10/runs-100 throughput claim is made.
