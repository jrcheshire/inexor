# M2 results — the exact-replay adjoint (paper core)

Working record for milestone M2 (plan: buzzing-seeking-patterson). Sections fill
in as S1-S6 land. The adjoint is a genuine `jax.custom_vjp` around `evolve`
(D-003 discretise-then-optimise; NOT a continuous backsolve): fwd residual =
final integer state only (O(1) in steps), reverse pass = bit-exact `step_rev`
replay + STE-twin (`step_float`) VJP in lattice units, paint per D-006 (int
primal / f32 twin). Gates ratified with JC floor-first (CLAUDE.md convention).

Machine key: S1-S3 measured on the laptop (M4 Max, CPU x64). Gate-grade 128^3
and the CUDA arms confirm on deneb (S4); the 1024^3 flagship money plots on
Vista GH200 (S5).

## S1-S2 — adjoint + IC wrappers (2026-07-13)

- `src/inexor/adjoint.py`: `evolve_grad` (custom_vjp, both integrator families),
  `adjoint_grad_fnl` / `adjoint_grad_ic` (mbody parity, thin eager wrappers over
  the custom_vjp -- jax.grad composition reproduces mbody's `_reversible_ic_grad`
  5-stage skeleton automatically). `src/inexor/losses.py`: differentiable jnp
  `band_power_loss` / `field_l2_loss` (paint=f32 VJP path), `power_spectrum`,
  `fundamental_k_edges`.
- Correctness (16^3, K=6-8): adjoint vs jax.grad(evolve_float) = norm-ratio
  1.0000, corr 0.99999, median-rel ~1e-4 (both integrators, both drivers);
  scan and perstep give bit-identical grads. d/d[f_NL, amplitude] vs float-path
  jax.grad AND central FD all agree (f_NL needs an f_NL-scale FD eps -- the
  f_NL*phi_G^2 term is ~1e-10, so a small eps starves FD by f64 cancellation).
- Boundary (shared with the forward `evolve`): `evolve_grad` composes with outer
  `jax.grad` -- including nested in a larger differentiation (the upgrade over
  mbody's eager-only manual adjoints) -- but an outer `jax.jit` is unsupported
  (the s_w0 policy needs a concrete `float(max|v0|)`; the jitted units are the
  internal drivers). Pinned by a test.

## S3 — gradient-fidelity gate (floor-first; 2026-07-13)

`scripts/m2_grad_gate.py`, x64, 64^3 K=10, both integrators, both losses. The
promoted adjoint = int-paint primal replay + f32-paint STE-twin VJP (D-006).

### The reference's OWN floors (measured first)

The per-particle gradient of a band-power / field-L2 loss is intrinsically
noise-dominated (many near-zero components; f32 paint scatter):

| reference floor (band_power / field_l2) | norm-ratio | corr | median-rel |
|---|---|---|---|
| float-twin f32 vs f64 (input grad x) | 0.63 / 0.80 | 0.06 / 0.001 | ~1.0 |
| central-FD vs float-twin (f64, input grad x) | 0.60 / 0.54 | 0.34 / 0.71 | 0.42-0.91 |

Consequence: per-component max-rel and FD are the WRONG gate reference. FD of the
QUANTIZED loss is additionally invalid below the lattice step (the R4 staircase:
sub-lattice eps rounds to the same integer -> FD == 0). The gate is on GLOBAL
metrics.

### The gate quantity — adjoint vs float-path gradient (D-010 R4 restatement)

| config | loss | input-grad median-rel (x / v) | corr | norm-ratio | IC d/d[f_NL,amp] rel |
|---|---|---|---|---|---|
| bullfrog 64^3 K=10 | band_power | 1.96e-4 / 1.68e-4 | 0.99999 | 1.0001 | 6.0e-5 / 6.8e-5 |
| bullfrog 64^3 K=10 | field_l2 | 3.11e-3 / 2.87e-3 | 0.99984 | 1.0007 | 9.5e-4 / 4.5e-4 |
| fastpm 64^3 K=10 | band_power | 1.86e-4 / 1.56e-4 | 0.99999 | 1.0000 | 1.5e-4 / 1.8e-5 |
| fastpm 64^3 K=10 | field_l2 | 3.33e-3 / 3.02e-3 | 0.99965 | 0.9995 | 1.3e-4 / 2.4e-4 |

The adjoint reproduces the f64 float-path gradient far below the reference's own
per-component noise floor. **R4 restatement:** the promoted-adjoint gradient
error (worst median-rel 3.3e-3) is well inside the D-010 R4 gate (7.5e-2).

### Ratified gate (D-015)

Per config x loss: input-grad median-rel <= 1e-2, corr >= 0.999, |norm-ratio-1|
<= 1%; IC-param rel <= 5e-3. **GATE PASS at 64^3** (both integrators). Artifact:
runs/m2/grad_gate.json.

Code note: `paint_f32` materializes an f32 mesh, so the VJP twin computes the
paint in f32 (correct per D-006). The reverse sweep aligns the vjp input
cotangent to the twin's output dtype and restores the carry dtype (scan requires
input==output types); `losses.density_f32` casts the f32 paint back to the input
dtype. So the adjoint is dtype-clean under x64.

### Remaining S3 (deneb arm)

Gate-grade **128^3 K=40** + a **CUDA** run (the gpu-env pytest already carries
`test_adjoint.py`; reversibility tier-0 rides the existing arm) confirm the gate
at the money-plot scale and on the authoritative backend. Runs alongside S4 on
deneb.

## S4-S6 — TBD
