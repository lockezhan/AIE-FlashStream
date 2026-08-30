# V21 PV6 validation

- [x] 1 Score/group and 1 Softmax/group
- [x] 6 PV/group; 8 groups give exactly 64 compute kernels
- [x] `kQueriesPerPhase == 2`; phase-0 K/V load and phase-1 reuse unchanged
- [x] widths `24,24,16,24,24,16`; offsets `0,24,48,64,88,112`
- [x] four D24 windows are 3072 bytes; two D16 windows are 2048 bytes
- [x] Q and K remain 2 x split4; V is 6 x split8; O is 6 x merge8
- [x] physical PLIO count is 16 (2 Q + 2 K + 6 V + 6 O)
- [x] host output remains four contiguous 32-D planes
- [x] reciprocal LUT unit test
- [x] PL transport C simulation
- [x] x86sim functional/numerical and 8/8 Score, 48/48 PV load/reuse
- [x] AIE hardware compile and exact active-tile/placement audit
- [x] PL `v++ -c`, structural warning, resource, and Fmax audit
- [x] full link timing and single-request board liveness/correctness
- [x] 10/10 independent reset/reload B1 board samples
- [ ] repeated board invocation: run 2 currently stalls (120-second gate)
- [ ] B8 single launch: timed out at 120 seconds
