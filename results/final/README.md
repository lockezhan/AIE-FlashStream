# Final paper-level results

This directory contains only compact, version-controlled summaries. Raw
measurements remain in their experiment directories.

- `comparison_summary.json`: complete machine-readable provenance, metrics,
  accuracy, resources, timing and limitations
- `comparison_table.csv`: flat paper-table data
- `SHA256SUMS`: hashes for both summary files

The baseline is Pure-PL. The proposed design is AIE-FlashStream.
The comparison is matched at B1 and uses ten independent reset/reload
samples per design. It covers one causal-GQA attention workload, not a complete
Llama3 model.
