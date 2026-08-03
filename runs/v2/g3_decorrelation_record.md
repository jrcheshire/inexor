# G3 -- the decorrelation map, the cdev ensemble, and the box ladder

What configuration of the tiled arm still shares phases with the monolithic one,
and at what price. Producer `scripts/v2_g3_decorrelation_map.py`; readout
`scripts/v2_g3_paired_readout.py` (reproduces every number below).

`k_usable` = the largest k at which `r(k)` is still above a threshold, by linear
interpolation on the first crossing. `vol_ratio` = `(P/T)^3` = the cost of the
tiled run in monolithic evolves.

| rung | box | L | n_fine | job | seeds | card |
|---|---|---|---|---|---|---|
| 1 | cdev8 | 64 | 256 | deneb 296 | 0 | `g3_decorrelation_cdev8.json` |
| 2 | cdev | 128 | 512 | deneb 297 | 0 | `g3_decorrelation_cdev.json` |
| 2 | cdev | 128 | 512 | Vista 883777_[1-8] | 1-8 | `g3_decorrelation_cdev_seed{1..8}.json` |
| 3 | cgh64 | 256 | 1024 | Vista 883778 | 0 | `g3_decorrelation_cgh64.json` |

Same fine cell (0.25 Mpc/h) throughout, so each rung is 8x the volume of the last.

## Why this needed an ensemble and a ladder

**The confound is exact.** At fixed cost, `cost = (P/T)^3` fixes `b/T`, so
`P/n_fine = T(1 + 2b/T)/n_fine` is EXACTLY proportional to `T`. The 3.375x triple
reads `P/n_fine` = 0.094 / 0.188 / 0.375 at cdev, a clean doubling. Tile size and
box fraction are ONE variable inside a single box; no cleverer single-box design
separates them, and only the box ladder can.

**Single-seed rankings are not rankings.** Job 297 showed an apparent
non-monotonicity in `T` at fixed absolute buffer, ~20% against a 29% effect, which
no mechanism explains.

Both are now answered.

## 1. The ranking is real: 16/16 paired cells

`k(r=0.5)|T=128 / k(r=0.5)|T=64`, ranked WITHIN each seed, then aggregated in log:

| cost | per-seed range | geometric mean | 1sd | sem | T=128 ahead |
|---|---|---|---|---|---|
| 3.375x | 1.086 - 1.561 | **1.297** | [1.153, 1.459] | [1.244, 1.352] | **8/8** |
| 8.00x | 1.662 - 2.219 | **1.829** | [1.662, 2.012] | [1.768, 1.891] | **8/8** |

The per-seed ratios never bracket 1.0, and the full ordering
`T=128 > T=64 > T=32` holds in all 16 seed x cost-triple cells. Job 297 was not
an outlier: its 1.287 and 1.740 sit essentially on the ensemble means.

**The pairing is what makes 8 seeds enough.** The ABSOLUTE `k(r=0.5)` scatters
12.0 / 11.9 / 15.7% (3.375x) and 13.9 / 16.5 / 21.8% (8.00x) seed to seed, while
the paired ratio scatters 12.5% and never crosses 1. A ratio of independent means
would have thrown away the shared-IC cancellation that produces that.

## 2. The advantage depends on which r the gate wants

Same ratio at the 3.375x triple, by threshold:

| level | geometric mean | 1sd | n > 1 |
|---|---|---|---|
| `k(r=0.9)` | **1.034** | [0.953, 1.122] | 7/8 |
| `k(r=0.5)` | 1.297 | [1.153, 1.459] | 8/8 |
| `k(r=0.2)` | 1.274 | [1.123, 1.444] | 8/8 |

At the cheap triple T=128 buys essentially NOTHING at r=0.9 -- the 1sd band
straddles 1.0. At the 8.00x triple it buys 1.906 there. So a Stage 5 gate written
on a high correlation threshold and one written on r=0.5 do not select the same
geometry, and the 8.00x row is the operative one if the gate wants r=0.9.

## 3. THE FINDING: absolute buffer sets the physics, tile size sets the price

At fixed ABSOLUTE buffer `b = 8 Mpc/h`, over the 8 seeds:

| arm | P | tiles | vol_ratio | mean k(r=0.5) | mean wall (Vista) |
|---|---|---|---|---|---|
| T=64, b=32 fine | 128 | 512 | 8.000 | 0.4519 | 1276 s |
| T=128, b=32 fine | 192 | 64 | 3.375 | 0.4534 | 472 s |

Paired ratio T64/T128 = **0.993**, sem [0.957, 1.031], 4/8 either way. A NULL.

The two tile sizes deliver the same usable k to 0.3% at the same absolute buffer,
while T=128 costs 2.37x less padded volume and 2.7x less wall. Job 297's
non-monotonicity was realization scatter, and what replaces it is stronger than
the original claim: **`b` in Mpc/h governs `k_usable`; `T` governs the price.**
This is consistent end to end with Stage 1, where strict containment needed
`b = 16 Mpc/h` at EVERY core size -- an absolute physical scale.

Corollary for the cost-matched triples: they are iso-cost, not iso-buffer, so
walking up `T` along one walks the absolute buffer up with it (2 -> 4 -> 8 Mpc/h
at 3.375x). The apparent "T advantage" at fixed cost IS the buffer advantage,
bought with the volume that a larger tile frees up.

## 4. The box ladder, at cost 3.375x, k(r=0.5)

| box | T=32 | T=64 | T=128 | 128/64 | excess | seeds |
|---|---|---|---|---|---|---|
| cdev8 | 0.5200 | 0.5171 | 0.9508 | 1.839 | 0.839 | 1 |
| cdev (seed 0) | 0.3548 | 0.3670 | 0.4722 | 1.287 | 0.287 | 1 |
| cdev (ensemble) | | | | **1.297** | 0.297 | 8 |
| cgh64 | 0.2574 | 0.2738 | 0.3304 | 1.206 | 0.206 | 1 |

The excess decays with volume but is DECELERATING: 0.839 -> 0.297 is a factor
0.354 per 8x, 0.297 -> 0.206 is 0.694. The two-point extrapolation made from 297
alone predicted ~0.10 at cgh64; the measured 0.206 is twice that, so the
advantage is dying more slowly than that trend implied.

**The last rung is not resolved.** cgh64's 1.206 lies INSIDE the cdev ensemble's
own 1sd spread [1.153, 1.459], so a single realization cannot distinguish
"decayed further" from "same as cdev, low draw". Against 1.0 it is 1.4 sigma if
cgh64 scatters like cdev (12.5%) and 3.9 sigma if scatter falls as
sqrt(volume) -- the latter is an expectation, not a measurement. What IS
established at cgh64 is the direction and the ordering
(`k(r=0.5)` = 0.2574 / 0.2738 / 0.3304, monotone in T, none flagged degenerate,
`p_frac` only 0.047 / 0.094 / 0.188 so the box-fraction confound is weak there).

## 5. Wall time

At matched `vol_ratio` the arms are FLOP-matched but nowhere near wall-matched,
because tile count changes by 64x across a triple while padded volume does not:

| host, box | T=32 | T=64 | T=128 | T32/T128 |
|---|---|---|---|---|
| Vista, cdev (8-seed mean) | 3492 s | 773 s | 472 s | 7.4x |
| Vista, cgh64 | 27758 s | 6143 s | 3857 s | 7.2x |
| deneb, cdev8 | 131 s | 59 s | 53 s | 2.5x |
| deneb, cdev (job 297) | 1759 s | 568 s | 442 s | 4.0x |

**Compare wall only within a host.** The same box and configs read 7.4x on Vista
and 4.0x on deneb, so the ratio is machine-dependent and the four rows are not a
box-scaling trend. Within Vista it is flat at ~7.2-7.4x. Both axes -- usable k
and wall -- point the same way toward large tiles.

## What this licenses for Stage 5

- The tiled-arm geometry should be chosen by the ABSOLUTE buffer the gate needs,
  then the largest T that fits. It should NOT be chosen by walking an iso-cost
  line, which confounds the two.
- The plan's assumed Stage 5 geometry (T=64, b=16 fine = 4 Mpc/h at cdev) is
  dominated: at cgh64, T=128/b=32 fine gives more usable k for the same padded
  volume and 1.6x less wall than T=64/b=16.
- The r=0.9 result says the geometry choice is not separable from the threshold
  choice. Pin the estimand's r threshold before pinning the geometry.

## Not licensed / owed

- **The cgh64 rung has no error bar.** An ensemble there is the only way to
  resolve the last decay step; it costs ~10.5 h/seed on a Vista gg node at the
  3.375x triple (27758 + 6143 + 3857 s), so 8 seeds is ~84 node-hours. Whether
  Stage 5 needs that resolved, or only needs the ordering (which one seed plus
  the cdev ensemble already gives), is a Stage 5 scoping call.
- cdev8's 1.839 is also one seed and is the point most likely to be overstated,
  being the smallest box.
- Nothing here bears on the SEAM statistic itself. `k_usable` is a correlation
  diagnostic; `R_Q`'s non-monotonicity in brokenness (see
  `g3_ladder_record.md`) is unaffected and still governs the gate's construction.
- The equilateral control's Floor B remains unmeasured at cdev.
