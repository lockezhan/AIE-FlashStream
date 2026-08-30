#!/usr/bin/env python3
"""Verify every reciprocal mantissa LUT entry against its definition."""

import re
from pathlib import Path


source = (Path(__file__).parents[1] / "pl/llama3_attention_pl_common.hpp").read_text()
match = re.search(r"kReciprocalMantissaQ15\[128\]\s*=\s*\{(.*?)\};", source, re.S)
if match is None:
    raise SystemExit("reciprocal LUT declaration not found")
actual = [int(token) for token in match.group(1).replace("\n", " ").split(",") if token.strip()]
expected = [round(127 * 32768 / (128 + index)) for index in range(128)]
if len(actual) != 128:
    raise SystemExit(f"expected 128 LUT entries, found {len(actual)}")
for index, (got, want) in enumerate(zip(actual, expected)):
    if got != want:
        raise SystemExit(f"LUT[{index}] is {got}, expected {want}")
if 17340 not in actual:
    raise SystemExit("required LUT value 17340 is missing")
print("reciprocal LUT PASS: all 128 entries match round(127*32768/(128+i))")
