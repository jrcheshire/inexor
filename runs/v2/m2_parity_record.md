# M-v2-2 exit gate: the promoted force IS the probe, bitwise, at the operating point

**Result, and it is a pass.** Vista job 897904 (`gh`, GH200, jax 0.10.2, branch
`jc/v2-force-promote` @ `5aa86a4`), two legs, **1 min 44 s wall**. Cards
`m2_parity_cdev8.json`, `m2_parity_cgh64.json`. Nothing here amends D-v2-16; it
discharges clause 7.

## The gate

| leg | particles | geometry | elements compared | differing | max abs delta |
|---|---|---|---|---|---|
| cdev8 | 2,097,152 | T=64 b=32 P=128, 64 tiles | 6,291,456 | **0** | **0.0** |
| **cgh64** | **134,217,728** | **T=256 b=32 P=320, 64 tiles** | **402,653,184** | **0** | **0.0** |

cgh64 at T=256/b=32 is the configuration every D-v2-10, D-v2-11 and D-v2-12
number was measured at. With the two CPU geometries already in
`tests/test_two_level_force.py` (the tile identity, and a clustered
padding-dominated field at pad_frac > 0.75), **D-v2-16 clause 7's three
geometries are satisfied and the promotion is faithful.**

Supporting equalities at both legs: `partition_ok` true on both arms,
`n_overhang_total` 0 on both, `padded_P` agreeing, and the oracle non-vacuous
(peak |g| 2.35 and 3.12, every one of 402 M elements nonzero at cgh64).

## Why this is readable, which is most of the work

**Leg 0 demonstrates determinism instead of trusting a flag.** `tile_paint_f64`
accumulates through `.at[].add` on an f64 mesh, and on CUDA that is an atomic
scatter with a non-reproducible order -- M2 S4 measured `paint_f32` differing in
156 of 4096 elements across 8 identical calls. Comparing two arms under that is a
coin flip that sometimes passes. `XLA_FLAGS=--xla_gpu_deterministic_ops=true`
removes it, but an env var is a self-report and self-reported knobs have lied in
this project before, so the script paints one tile twice and requires bit
equality before anything else runs: **0 differing cells, on `gpu`, at both legs.**
The sbatch separately refuses to run on a CPU backend, because a CPU fallback
would pass this gate deterministically and meaninglessly.

**Both arms are driven from ONE membership** -- the probe's own bucketing. The
layout's `tile_members` returns the same member SET as the probe's but not
necessarily the same ORDER, and order changes the f64 scatter-add sequence and
therefore the bits. Passing membership in is what makes this a test of the FORCE
rather than of two bucketings, and it is why `force_short_tiled` takes
`member_fn` and `cap` rather than computing them.

**Anti-vacuity is asserted, not assumed.** `r_s=None` zeroes the short kernel
entirely, which is exactly how the first V4 parity check passed on all-zero
arrays; the card carries `oracle_peak` and `oracle_nonzero` and reports VACUOUS
rather than PASS if either collapses.

## Incidental numbers, not claims

The promoted arm ran 14.2 s against the probe's 20.3 s at cgh64. **Do not read
that as a speedup.** The two do different amounts of host work -- the probe
computes its own bucketing inside the timed region while the promoted arm is
handed membership -- so the comparison is not axis-matched and was never intended
to be. A performance comparison needs its own instrument.

cgh64 geometry as realized: brick 32, cap 4,335,616, pad_frac 0.058.

## Cost

One Vista `gh` job, 1 min 44 s against a 3 h request. The generous wall was
deliberate (billed for actual use, and the last three runtime estimates on this
project were all low); this time it overshot in the other direction.
