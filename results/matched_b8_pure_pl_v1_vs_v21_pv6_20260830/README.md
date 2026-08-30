# V21 matched-B8 feasibility audit

The final V21 xclbin was tested with B8/S32/HQ32/HKV8/D128 and seed 7 as one
kernel launch after a board reset/reload. The first sample timed out after 120
seconds without producing a measured-run record. No valid V21 B8 latency was
obtained, and no remaining samples were attempted.

The failure evidence is retained in `experiment.env`, `sample_01_reset.log`
and `sample_01.log`. The fallback required by the fairness audit was completed:
ten independent B1 samples were collected for both Pure-PL and V21 in
`results/matched_b1_pure_pl_v1_vs_v21_pv6_20260830/`.

This failed experiment must not be used as a performance sample.
