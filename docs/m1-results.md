# M1 results — forward PM validation

Working record for milestone M1 (plan: tender-stargazing-map). Sections fill
in as S5-S7 land; the parity gate numbers are ratified with JC only AFTER the
floors below (floor-first protocol; see CLAUDE.md tolerances convention).

Machine key: all S5 numbers measured on the laptop (M4 Max, "andromeda") —
mbody is MLX/Metal-only so the harness lives here; deneb reserved for the S7
CUDA legs (over-subscribed as of 2026-07-13).

## S5 — harness + measured repro floors (2026-07-13)

Harness: `scripts/_m1_common.py` (numpy-only neutral CIC + P(k)/r(k)
estimator, npz exchange schema) + `m1_export_ics.py` / `m1_run_mbody.py` /
`m1_run_disco.py` / `m1_parity.py`, one pixi env per code, cross-process npz
(xcheck.py pattern). All matched configs: L = 256 Mpc/h, a: 0.1 -> 1.0, log
spacing, BullFrog, 2LPT ICs, seed 0, shared a_steps array (np.geomspace —
bitwise the inexor grid; every code integrates on the SAME array).

Sanity identities checked before any run:
- DISCO-DJ q-ordering vs the inexor/mbody Lagrangian grid: EXACT (max diff 0.0).
- a_grid: _m1_common == inexor.integrate.a_grid bitwise (asserted at export).
- CAMB tables: parity-env camb 1.6.5 vs mbody's internal backend agree to
  max |dP/P| = 6.3e-4 (interpolation/k-grid class; either source works, both
  written: `runs/m1/pk_camb.txt`, `runs/m1/pk_camb_mbody.txt`).
- CAMB vs EH98 (context for the S6 deficit axis): max |dP/P| = 3.8%
  (BAO wiggles + small-scale).

### Repro floors (identical-input repeats, x5, worst pair vs run 0)

| code | config | rms dx [cells] | max dP/P | max 1-r |
|---|---|---|---|---|
| mbody (MLX f32, Metal) | 64^3 K=10 | 5.0e-7 | 5.8e-8 | 5.3e-13 |
| mbody (MLX f32, Metal) | 128^3 K=40 | 1.0e-6 | 6.0e-8 | 7.2e-13 |
| DISCO-DJ (JAX f64 CPU) | 64^3 K=10 | 0 (bit-identical) | 0 | 3.3e-16 |
| DISCO-DJ (JAX f64 CPU, fresh-trace) | 64^3 K=10 | 0 (bit-identical) | 0 | 3.3e-16 |
| DISCO-DJ (JAX f64 CPU) | 128^3 K=40 | 0 (bit-identical) | 0 | 2.2e-16 |
| DISCO-DJ (JAX f64 CPU, fresh-trace) | 128^3 K=40 | 0 (bit-identical) | 0 | 2.2e-16 |

(The 1-r "3.3e-16" on bit-identical inputs is the estimator's own float64
epsilon, not a field difference. mbody's floor is the expected Metal
CIC-scatter class from its own docs, ~1e-6 cells.)

### First parity numbers (64^3 K=10, matched-IC injection; NOT gates yet)

| pair | rms dx [cells] | max dP/P | max 1-r |
|---|---|---|---|
| inexor_float (f64) vs mbody | 4.2e-6 | 1.1e-6 | 4.3e-11 |
| inexor_float (f64) vs DISCO-DJ | 3.6e-3 | 1.7e-2 | 1.3e-4 |
| inexor_float (f64) vs inexor int16 (Tier B) | 1.8e-3 | 1.5e-4 | 7.7e-6 |

Readings (preliminary, S6 will formalize):
- **inexor vs mbody sits at mbody's own f32 floor** — the physics port is
  faithful to the source code at its repeatability limit.
- **inexor vs DISCO-DJ deviation is k-INCREASING**: |dP/P| 1.1e-3 in the four
  lowest-k bins -> 4.6e-3 mid -> 1.7e-2 approaching Nyquist; 1-r low-k 2.7e-7.
  The shape says force-kernel / window-implementation class, not growth
  coefficients (those would be k-flat). Low-k (the f_NL regime) dynamics
  agree at the 1e-3 level. Attribution -> S6 matrix.
- **Tier B (int16 vs f64 float)** is k-flat at the few-1e-5 level with a
  1.5e-4 worst bin at 64^3 K=10 — measured against the D-010 1e-4-bar
  convention this needs the proper S6 quantization-gate config (K-vs-2K +
  mesh arm) before any verdict; recorded, not relaxed, not passed.

### 128^3 K=40 parity (gate-grade config)

| pair | rms dx [cells] | max dP/P | max 1-r |
|---|---|---|---|
| inexor_float (f64) vs mbody | 1.6e-5 | 1.4e-6 | 2.3e-10 |
| inexor_float (f64) vs DISCO-DJ | 1.2e-2 | 1.0e-1 | 1.6e-3 |
| inexor_float (f64) vs inexor int16 (Tier B) | 8.3e-3 | 2.8e-4 | 5.6e-5 |

k-band structure:

| pair | low-k (4 bins) | k < 0.4 | mid | near Nyquist (k~1.5) |
|---|---|---|---|---|
| vs mbody, dP/P | 7.3e-7 | 1.2e-6 | 1.4e-6 | 5.7e-7 |
| vs DISCO-DJ, dP/P | 1.8e-3 | 1.6e-2 | 4.8e-2 | 1.0e-1 |
| vs int16, dP/P | 1.3e-4 | 2.0e-4 | 1.9e-4 | 9.0e-5 |

Readings:
- **mbody agreement is k-FLAT at ~1e-6 dP/P over all 40 steps** — at/near
  mbody's own measured floor (6e-8 per repeat, f32 accumulation class).
  Tier-A shape for the ported physics: as good as the reference can resolve.
- **DISCO-DJ deviation grows with k AND with (K, resolution)**: at fixed
  k ~ 0.4-0.8 it is 4.8e-2 here vs 1.7e-2 in the 64^3 K=10 run. Low-k stays
  1.8e-3 / r = 1 - 7e-7. Consistent with a per-step force difference at high
  k (kernel/window handling), compounding over steps — S6 attribution axes:
  deconvolve, worder, grad_kernel_order 0 vs 2, K, res, + growth-table diff.
- **int16 quantization is k-flat at 1-2e-4 dP/P at this config** (worst bin
  2.8e-4) — sits ABOVE the D-010 1e-4 R2-class bar as a raw reading; the
  formal Tier-B verdict belongs to the S6 quantization gate (K-vs-2K + mesh
  arm at the ratified config), where the bar is re-ratified with JC. Recorded
  as measured; nothing relaxed.

## S6 — parity gates + deficit attribution

TBD (gates ratified with JC against the floors above; m1_deficit.py matrix:
{ZA, 2LPT} x K x integrator x resolution x deconvolve x {EH98, CAMB}).

## S7 — deneb CUDA legs + 512^3 smoke

TBD.
