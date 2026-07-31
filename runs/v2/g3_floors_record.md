# G3 Stage 3 -- the floors, measured on monolithic pairs only

Written before any tiled number exists, which is the point: D-v2-7's 15% bar is
kept, but the estimand is pinned at the checkpoint below with no tiled arm in
existence, so the choice cannot leak the answer.

Config `cdev8` (`n_part=128`, `n_fine=256`, `L=64` Mpc/h, fine cell 0.25 Mpc/h),
20 BullFrog steps, `a = 0.1 -> 1.0`, f64 throughout, `paint="int"` (the
deterministic primal path -- a reproducible paint is a precondition for the
bit-repro rung meaning anything).

Producer: `scripts/v2_g3_floors.py`. Cards: `runs/v2/g3_floors_cdev8.json`,
`runs/v2/g3_floors_cdev8_bc.json`.

Triangles: squeezed `(m k_f, 12 k_f, 12 k_f)` for `m = 2, 3, 4, 6`, plus an
equilateral control at the same `k_short = 12 k_f`. The equilateral is there so
a generic error cannot be read as a squeezed coupling failure: if a perturbation
hurts both equally, the squeezed framing is wrong.

## The statistics, and what each one actually does

| statistic | definition | response to a deterministic window |
|---|---|---|
| `R_B` | `B_t/B_m - 1` | `T1 T2 T3 - 1` -- responds fully |
| `R_Q` | `Q_t/Q_m - 1`, `Q = B/(P1P2+P2P3+P3P1)` | `1/T - 1` for uniform `T` -- **does not cancel** |
| `rho` | `(B_t/B_m)/(T1 T2 T3) - 1`, `T` measured per shell | flat to 1e-4 -- window-invariant by construction |
| `W` | `B[eps, m, m]/B[m,m,m]`, `eps = delta_t - T(k) delta_m` | 0 to 1e-4 |

`W` places the residual on the LONG leg. `T` is estimated per SHELL rather than
per mode; a per-mode estimate would make `eps` identically zero by construction
and `W` vacuous. The price is that a window varying inside a shell leaves a
residual `W` sees, which is `W`'s floor and is measured below rather than
assumed.

## D -- the discriminators against known answers

Injected window `T(k) = 1 + 0.05 exp(-(k/k0)^2)`, `k0 = 6 k_f`, applied at the
field level to the evolved monolithic field.

| tri | `R_B` | expected | resid | `R_Q` | `rho` | `W` |
|---|---|---|---|---|---|---|
| sq2 | 0.045549 | 0.045469 | 8.1e-05 | -0.037017 | 3.8e-05 | 1.7e-05 |
| sq3 | 0.040117 | 0.040028 | 8.9e-05 | -0.030309 | 3.1e-05 | 1.9e-05 |
| sq4 | 0.033330 | 0.033484 | -1.5e-04 | -0.022600 | 3.7e-05 | 2.9e-05 |
| sq6 | 0.019787 | 0.019561 | 2.3e-04 | -0.011075 | 1.0e-04 | 9.9e-05 |
| equi | 0.002759 | 0.002739 | 2.0e-05 | -0.000896 | 1.7e-05 | 5.6e-06 |

`R_B` tracks the shell-averaged window product to ~1e-4, and the residual is
within-shell variation of `T`. That same variation is `W`'s floor:

    max |W| under a deterministic window = 9.9e-05
    max |rho| under a deterministic window = 1.0e-04

### `R_Q` does NOT cancel a deterministic window

This contradicts the stated rationale for gating on `R_Q` (that it removes the
P(k) contamination `R_B` carries, which D-v2-9 already gates separately).
Measured `max |R_Q| = 0.0370` under a pure 5% window.

The reason is algebraic, not numerical. Under `delta_t = T delta_m`,
`B -> T^3 B` and `P -> T^2 P`, so `Q = B/(P1P2+P2P3+P3P1) -> T^3/T^4 = 1/T`.
It is the same algebra that makes tree-level `Q ~ 1/b` under linear bias.
Confirmed against the closed form on the equilateral control, where all three
legs share one `T = 1.000915782`:

    predicted R_Q = 1/T - 1 = -0.00091494    measured -0.000896
    predicted R_B = T^3 - 1 =  0.00274986    measured  0.002759

So `R_Q` reduces a P(k) contamination relative to `R_B` by roughly 3x and flips
its sign, but does not remove it: a 2% P(k) error still lands ~1% in `R_Q`,
against a 15% bar.

`rho` divides the measured transfer out explicitly and is window-flat to 1e-4,
about 400x better than `R_Q`. Its cost is conditioning: it divides by a measured
`T`, so it diverges when the arms decorrelate (see Floor A). That is what the
estimand's "conditioning cut" has to govern.

## A -- the instrument floor

| null | max `\|R_B\|` | max `\|R_Q\|` | max `\|rho\|` | max `\|W\|` |
|---|---|---|---|---|
| bit-repro pair | 0.0 | 0.0 | 2.2e-16 | 2.1e-16 |
| both arms translated | 0.0 | 0.0 | 0.0 | 0.0 |
| translated vs original | 3.3e-16 | 4.4e-16 | 9.1e+05 | 1.1e-01 |
| f32 vs f64 EVOLUTION (ICs fixed at f64) | 1.4e-06 | 1.1e-06 | 1.2e-06 | 3.7e-07 |

Three things to read off this.

**Translation invariance holds to 3e-16 in `R_B` and `R_Q`.** B is
translation-invariant analytically, so this is a real check on the estimator and
it passes at machine precision.

**`W` and `rho` are NOT translation-invariant.** A rigid translation is a
direction-dependent phase ramp, and no shell-constant real transfer absorbs it:
the measured per-shell `T` collapses toward zero, so `rho` diverges and `W`
reads 0.106. Harmless if the two arms are never relatively translated, which
for tiled-vs-monolithic they are not, but it bounds what either statistic may
be read to mean and it is why `rho` needs a conditioning cut rather than a
tolerance.

**The f32 rung was wrong the first time, and the corrected value is 1e-6.**
Passing `fdtype=float32` into `ic.linear_density` changes the `jax.random`
STREAM, not merely its precision, so the first version of this rung compared two
DIFFERENT REALIZATIONS and read `R_B` 0.375 / `R_Q` 0.0695 -- which looks like a
catastrophic precision floor at half the bar. The tell was the correlation:
`1-r ~ 1` at every k INCLUDING the box fundamental, with an rms difference 1.39x
the field itself. A merely less accurate simulation still preserves large-scale
phases. With the ICs built at f64 and cast only for the evolution, the arms
correlate to `1-r = 1.3e-13` at `k_f` and the true precision floor is ~1e-6 in
`R_Q`. Recorded because the failure mode is generic: any "precision" arm built
by threading a dtype through an RNG measures realization scatter, not precision.

## B / C -- resolving power and cosmic variance

24 seeds, arm B = the final field times the deterministic window above.

| tri | sigma_B(`R_Q`) | sigma_B(`R_B`) | sigma_C | cancel | sigma_G/\|B\| | C/G | n_tri |
|---|---|---|---|---|---|---|---|
| sq2 | 8.53e-04 | 3.04e-04 | 0.555 | 651 | 0.0389 | 14.3 | 25800 |
| sq3 | 1.02e-03 | 1.67e-04 | 0.595 | 586 | 0.0410 | 14.5 | 28104 |
| sq4 | 8.62e-04 | 2.54e-04 | 0.619 | 718 | 0.0300 | 20.6 | 47088 |
| sq6 | 4.25e-04 | 1.34e-04 | 0.631 | 1486 | 0.0273 | 23.1 | 66000 |
| equi | 1.38e-05 | 1.92e-05 | 0.673 | 48771 | 0.0345 | 19.5 | 138000 |

`r(k_long)` across seeds: mean 0.99999940, min 0.99999914. The arms genuinely
share long modes, so the cancellation factor means what it says: 586-1486x on
squeezed configurations, against G6's 100-500x for P(k).

### The Gaussian-sigma diagnostic does not discriminate here

The plan predicted measured and analytic Gaussian sigma would differ a lot for
squeezed and AGREE for equilateral, the point being that agreement on squeezed
would prove the estimator blind to squeezed physics. Measured `C/G` is 14-23 for
squeezed AND 19.5 for equilateral, so nothing agrees and the test has no power
at this configuration.

The cause is the configuration, not the estimator. At `cdev8`,
`k_short = 12 k_f = 1.18 h/Mpc`, far past `k_nl ~ 0.2-0.3`, so non-Gaussian
covariance dominates every triangle. Even the longest legs
(`k_long = 0.196-0.589 h/Mpc`) sit at or past nonlinearity: an `L = 64` Mpc/h box
has no linear long mode to squeeze against. This is a reason to run the
diagnostic at `cdev` (`L = 128`) before reading anything into it, and it is the
first instance of the plan's own "volume at fixed cell" lever.

## E -- the DYNAMICAL null, which is the floor that actually binds

Section B/C builds arm B by windowing the FINAL field. That construction cannot
chaotically amplify, so its scatter is small by design and its resolvability
verdict would be optimistic if the dynamics were chaotic. A tiled arm differs
during EVOLUTION. So: perturb the initial linear density by a relative `eps` and
evolve both arms in full.

rms `R_Q`, 8 seeds:

| eps | sq2 | sq3 | sq4 | sq6 | equi |
|---|---|---|---|---|---|
| 1e-10 | 4.4e-09 | 3.9e-09 | 4.1e-09 | 4.2e-09 | 5.5e-09 |
| 1e-08 | 3.4e-08 | 3.3e-08 | 4.6e-08 | 4.7e-08 | 5.0e-08 |
| 1e-06 | 2.5e-07 | 2.0e-07 | 3.2e-07 | 2.3e-07 | 3.4e-07 |

**There is no chaotic amplification.** `rms R_Q` scales as about `eps^0.44` with
no saturation across four decades: a 1e-10 IC perturbation stays at 1e-9. Twenty
BullFrog steps from `a = 0.1` on a 0.25 Mpc/h mesh do not drive the divergence
the concern assumed. The dynamical floor is therefore BELOW the window-based
Floor B, not above it, and Floor B's 8.5e-4 stands as the binding number.

## Where the floors land

| floor | `R_Q` |
|---|---|
| bit-repro | 0 exactly |
| translation | 4.4e-16 |
| f32 vs f64 evolution | 1.1e-06 |
| dynamical null, eps = 1e-6 | 2.5e-07 |
| **B, resolving power (24 seeds)** | **8.5e-04 to 1.0e-03** |
| D-v2-7 bar / 3 | 5.0e-02 |

**Resolvability: G3 is a MEASUREMENT at `cdev8`, with ~50x margin** on every
gate-eligible triangle, by the criterion `sigma_B <= bar/3`. Nothing here is
close to making the 15% bar a bound.

## Checkpoint questions for JC

1. **The estimand.** `R_Q` was chosen to remove the P(k) contamination that
   `R_B` carries, and measurably does not: it responds as `1/T`, leaving ~1% of
   a 2% P(k) error in a 15% bar. `rho` is window-flat to 1e-4 but needs a
   conditioning cut on the measured transfer. Options: keep `R_Q` and accept a
   known ~1% window leak; move the gate to `rho` and pin a conditioning cut; or
   gate on `R_Q` and REQUIRE `rho` to agree, treating a divergence between them
   as a window-contamination flag.
2. **Triangle set and reduction.** Currently 4 squeezed plus 1 equilateral
   control; max or median over them; and whether the equilateral is a gated
   companion or context only.
3. **`W`'s second validation cannot be run as written.** "Scramble long-mode
   phases and confirm `W` returns the injected size" does not work for any `W`
   that is genuinely zero under a deterministic window: a phase ROTATION is
   still mode-diagonal and cancels exactly as a window does, and a full phase
   SCRAMBLE decorrelates the arms so the cross-bispectrum of an independent
   field with the reference is zero in expectation too. `W` responds to new mode
   COUPLING, which is arguably the property wanted. Proposed replacement: inject
   a known quadratic long-short coupling and check `W` recovers its amplitude.
4. **Configuration.** `k_short = 1.18 h/Mpc` at `cdev8` is deep in the nonlinear
   regime and the box has no linear long mode. Re-run the floors at `cdev`
   (`L = 128`) before pinning, or pin at `cdev8` and treat `cdev` as
   confirmation?
