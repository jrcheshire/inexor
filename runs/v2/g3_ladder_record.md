# G3 Stage 4 -- the identity ladder, and what it revealed about the gate

Config `cdev8` (`n_part=128`, `n_fine=256`, `L=64` Mpc/h), single seed, T=64
fine cells (16 Mpc/h core), pivot b=16 (P=96, `P/n_fine` = 0.375, 64 tiles).
Gate triangles `(m k_f, 12 k_f, 12 k_f)` for m = 2,3,4,6 plus an equilateral,
`k_short = 1.178 h/Mpc`. Producer `scripts/v2_g3_ladder.py`, deneb jobs 283/284/290,
card `runs/v2/g3_ladder_cdev8_stage4.json`.

## The machinery is correct

All seven hard-fail rungs pass, at 1e-13 against a 1e-12 tolerance taken from
Stage 3's measured instrument floor:

| rung | value | catches |
|---|---|---|
| `A0_frame_identity` | 1.19e-13 | the COLA algebra incl. the half-drift signs |
| `frame_zero` | 1.01e-13 | frame plumbing, unconfounded with COLA truncation |
| `A1_one_tile` | 1.19e-13 | tile bookkeeping, ownership, reassembly, frame add-back |
| `partition` | 64 tiles, every particle owned once | the dispatch trap |
| `residual_zero` | 0 exactly | restriction + reassembly lossless |
| `quasi_linear` | 0 exactly | a constant pipeline offset |
| `buffer_to_box` | 1.28e-13 | the buffer family terminates at the exact answer |

`buffer_to_box` is the strong one: 64 separate tiles, each solving a padded
region the size of the whole mesh, reassembling to the monolithic answer.

## THE FINDING: R_Q is not monotone in brokenness

| b | P/n_fine | max\|R_Q\| | r(k_long) | r(k_short) |
|---|---|---|---|---|
| 0 | 0.250 | 0.2514 | 0.8616 | 0.0138 |
| 8 | 0.312 | 0.4213 | 0.8876 | 0.0226 |
| 16 | 0.375 | 0.3982 | 0.9199 | 0.0284 |
| 32 | 0.500 | 0.2997 | 0.9574 | 0.3582 |
| 64 | 0.750 | 0.1096 | 0.9689 | 0.6679 |

Both correlation measures rise monotonically with the buffer across the whole
scan. `max|R_Q|` does not -- and the single row that breaks its ordering is
`b = 0`, the most decorrelated point. From `b = 8` onward `R_Q` IS monotone
(0.421, 0.398, 0.300, 0.110).

**Once two fields decorrelate, the ratio of their bispectra tends to a bounded
value rather than growing.** So a maximally broken arm can read a LOWER `R_Q`
than a partially broken one, and the plan's `kill_control` requirement
(`|R(b=0)|` exceeding `|R(pivot)|` by >= 5 sigma) is unsatisfiable BY
CONSTRUCTION whenever both arms sit in the decorrelated regime. It is not a
threshold that went unmet; it is a criterion this statistic cannot satisfy.
`r(k)` is monotone in brokenness and is what a kill control has to be built on.

The inverted ordering was observed three times (jobs 283, 284, 290) before the
cause was isolated, including once at a wrong `k_short` that produced the right
symptom for the wrong reason.

## Why the gate triangles are in that regime

At the pivot (T=64, b=16) the per-shell correlation runs

    k (h/Mpc)   0.196  0.295  0.393  0.589  1.178
    r(pivot)    0.920  0.817  0.632  0.343  0.028

Every gate triangle carries `k_short = 1.178`, where `r = 0.028`. `R_Q` there is
the ratio of two fields with essentially no phase relation.

The mechanism is consistent with the Stage 1 pricing result. The tile's missing
force is dominated by a component that is nearly UNIFORM across the tile
(`uniform_frac` measured 0.76-0.96). A uniform force error displaces the tile
bodily; tiles drift differently from one another; differential drift decorrelates
small-scale phases. A drift of ~2.7 Mpc/h fully decorrelates k = 1.18 h/Mpc.

## Two failure modes, and R_Q cannot tell them apart

    arm            max|R_Q|   r(k_short)
    pivot (b=16)     0.398      0.028
    kill  (b=0)      0.251      0.014
    2LPT             0.722      0.409

Pure 2LPT is FIFTEEN TIMES better correlated at `k_short` than the tiled run,
while having roughly twice the `R_Q`. 2LPT keeps phases and gets amplitudes
wrong; aggressive tiling scrambles phases. A gate on `R_Q` alone scores 2LPT as
the worse arm, which is the opposite of what the correlation says. `span_check`
needs a correlation criterion alongside the amplitude one.

## The tension this exposes for A3

Usable correlation at `k_short = 1.18` needs `b >= 32` at T=64, i.e. `P >= 128`
and `vol_ratio >= 8`. The configurations cheap enough to make A3 attractive
(`vol_ratio ~ 3`) decorrelate at the scales the gate is about. Either `k_short`
comes down to where `r` is appreciable (<= ~0.4 h/Mpc at the pivot), or the
tiling is made less aggressive at a compute cost, or the gate statistic changes.
That is a genuine A3 result rather than a measurement problem, and it is the
substance of the Stage 5 design decision.

## Caveats

Single seed, one config, one T. The `r` monotonicity is clean (5 points, both
measures) and unlikely to be noise, but the quantitative thresholds want
confirmation. The equilateral control's Floor B is still unmeasured at cdev
(parked, see the Stage 3 record).
