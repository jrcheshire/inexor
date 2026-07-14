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

## S4 — Vista env, GPU nondeterminism, memory profile (2026-07-14)

### The Vista linux-aarch64 gpu env (deferred from M1) — DONE

The `gpu` feature's specs were already right for Vista; only `platforms` was
x86-only, so it was extended to `["linux-64", "linux-aarch64"]` rather than
duplicated into a parallel feature. One spec set now covers deneb (RTX 3050) and
Vista (GH200); `cuda = "12"` is a driver FLOOR, satisfied by deneb's 13.3 and
Vista's 12.x alike. conda-forge does ship an aarch64 CUDA jaxlib, so this is a
committed-lock install with no native solve on Vista: `gpu/linux-aarch64`
resolves to 196 packages (jaxlib 0.10.2 cuda129, cudart 12.9, cublas 12.9,
python 3.14). Confirmed working on Vista job 831091: jax 0.10.2, `CudaDevice(id=0)`,
GH200, 97/101 tests passing on first contact with aarch64.

**The blocker was not the platform list but `dynamic = ["version"]`.** pixi must
read a package's metadata to solve an env for a platform it cannot run, and a
dynamic version forces it to execute hatchling to compute that metadata — so the
solve died with "no compatible Python interpreter for 'osx-arm64'". Verified as
the cause by making the version static and re-solving (sub-second success); xcat,
whose aarch64 lock has always worked, pins a static version for the same reason.
`[project] version` in pyproject.toml is now the single source of truth and
`__init__.py` reads it back via `importlib.metadata`. **Do not reintroduce a
dynamic version.**

### GPU gradient nondeterminism (deneb jobs 14/15) — mechanism CLOSED

Job 831091 failed 4/101, all asserting exact/near-exact float equality; all pass
on CPU. It was the first time `test_adjoint.py` had ever run on a GPU (M2 S1
code; deneb's M1 job 12 predates it).

The first hypothesis was wrong and is recorded here so it is not re-proposed:
*"S1 promoted M1's cross-driver `bitwise agreement is a CPU regression canary,
not a design gate` into a gate; the drivers disagree on CUDA."* **Job 14 refuted
it** — the cross-driver residual is bitwise IDENTICAL on CUDA (0/12288 in both x
and w, all three integrators; scales identical). The drivers agree. The test's
premise holds.

What actually fails is **determinism**: the same gradient, same driver, same
inputs, computed twice on CUDA, differs. The failing tests would fail
scan-vs-scan. Job 15 localized it bottom-up (8 repeats, identical inputs, deneb
3050):

| layer | result |
|---|---|
| `paint_int` (int scatter-add) | deterministic, 0/4096 |
| `paint_f32` (f32 scatter-add) | **NONDET**, 156/4096, max_rel 2.34e-7 |
| `force_int` (PRIMAL path) | deterministic, 0/12288 |
| `force_f32` (twin primal) | **NONDET**, 9792/12288 |
| `force_f32` VJP | **NONDET**, 11418/12288 |
| full bwd (`jax.grad`) | **NONDET**, 18384/24576 |

**The f32 CIC scatter-add is the source** — its GPU atomic accumulation order is
not reproducible. It seeds at f32 roundoff and amplifies through the FFT solve
and gather. This is R3's "f32 nondeterministic everywhere", now measured in the
adjoint rather than argued by analogy.

**The primal path is untouched** (`paint_int`/`force_int` bit-stable), so
bit-exact replay and reversibility — the paper's core claim — are UNAFFECTED.
D-006 requires the STE twin to paint in f32, so this is designed-in, not a
regression. Against D-015's global metrics the run-to-run spread is pure f32
roundoff (median-rel ~1e-7, corr 1.000000000, ratio 1.000000000): **no ratified
gate is threatened.** What is dead is the assumption that a GPU gradient is
bit-reproducible.

Note for anyone reading the raw json: the diagnostic's `max_rel` of 1.42e+23 at
the VJP layer is an artifact of dividing by a ~1e-30 denominator floor on
near-zero components (the same noise-domination S3 documented), not a real error.

### Decision (JC, 2026-07-14): deterministic ops for the TESTS, not production

`XLA_FLAGS=--xla_gpu_deterministic_ops=true` removes the nondeterminism at every
layer. Options weighed: (A) relax the assertions to D-015-style tolerances, (B)
keep them and run under the flag, (C) adopt the flag in production.

**B adopted.** Relaxing to tolerances would have cost the canary — the
bit-equality assertion is precisely what catches the drivers ACTUALLY diverging,
and job 14 proved they do not. XLA_FLAGS is process-wide and read at backend
init, so marked tests need their own process: the `detflag` marker + `test-det`
task, with `test` SKIPPING them visibly on GPU (an omission you can see beats one
you cannot) and running them on CPU, where the backend is deterministic and the
flag is a no-op. Validated on deneb job 18: `test` → 4 skipped with reason,
`test-det` → **4 passed**. All four survive, including the f_NL-vs-IC test that
compares two DIFFERENT compiled programs — deterministic ops did oblige them to
agree. **No tolerance was relaxed anywhere.**

**C rejected**, priced on deneb job 18 (`scripts/m2_detflag_cost.py`):

| n_mesh | forward | full adjoint | bwd (derived) |
|---|---|---|---|
| 64 | 2.05x | 2.95x | **3.17x** |
| 128 | 1.63x | 3.36x | **3.83x** |
| 256 | — | **OOM** | **OOM** |

Three findings. The bwd costs 3.2-3.8x, well above M0 R3's 1.37-1.78x — but that
figure was the paint in isolation and was never the right comparison. The forward
slowed 1.6-2.05x, where the model predicted ~1.0x since it rides the already
deterministic `paint_int`: **the flag does something not modelled here**, so
these numbers are directional, not precise. And the detflag arm **OOM'd at 256^3
where the default arm completed in 5.66s** (XLA: "can't reduce memory use below
4.55GiB by rematerialization") — consistent with deterministic ops replacing
atomics with multi-pass reductions needing extra scratch, though that is a single
unisolated observation. The memory cost is the decisive one: the flagship claim
IS a memory claim, and the flag would eat the same 80 GB budget. Production stays
on the fast path; gradients are reproducible only under the flag, and that must
be stated rather than implied. **The R6 profiler deliberately does not set it —
production is the config worth profiling.**

Caveat: 64^3/128^3 are small enough that launch overhead is a real fraction of
the wall, so the ratios need not hold at 512^3. Pricing C properly would need a
Vista run at scale; not bought, given the direction.

### R6 peak-memory profile — PENDING

`scripts/m2_mem_profile.py` is written and CPU-smoked but its GPU legs are UNRUN:
job 831091 aborted at pytest (`set -e`) before reaching the 512^3 smoke and the
profile. Re-run of `m2_vista.sbatch` (now passing its test leg) carries both.

Two notes already banked from building it. **One config per subprocess**, because
XLA's `peak_bytes_in_use` is monotonic within a process with no reset API — a
second config in the same process inherits the first's high-water mark, so
in-process sequencing would have silently reported the running maximum rather
than each peak. And **ceiling-1 cannot be settled by an n-sweep**: the carry and
the transients both scale as n_part, so the slope is identical under both
hypotheses; the script instead prints the measurement against both predictions at
the same n (~8.5 vs ~13 GiB at 512^3 on Sec. 9's own transient estimate).

**Unit trap in the Sec. 9 budget**, found while testing the report path: an
"80 GB" GPU is 80e9 B = **74.5 GiB**, so the budget's own upper end (75 GiB =
80.5 GB) does NOT fit. The margin is thinner than the prose reads. Verdicts
compare bytes; output prints both units. Whether Sec. 9's range wants restating
is a call for once there is a measurement.

### Operational lesson: `-n auto` is hostile on a single-GPU node

Deneb job 16 is VOID. Its GPU test legs ran the `test`/`test-det` pixi tasks,
which hardcode `pytest -n auto`; `auto` = CPU count, and `--exclusive` handed
over all 64 cores, so 64 xdist workers each initialized JAX and preallocated
`XLA_PYTHON_CLIENT_MEM_FRACTION` of the SAME 6 GB card. One took 5050 MiB, the
rest got 526/114/86, the GPU sat at 0%, and the tests reported failures that were
OOM — and that read exactly like the genuine nondeterminism failures under
investigation. Caught only because JC noticed the job pulling 1 CPU thread and 0%
GPU.

`m1_deneb.sbatch` and the original `m2_vista.sbatch` call pytest DIRECTLY; that
was not incidental. `-n auto` is right for laptop/CI CPU runs and harmful on one
GPU, so the fix lives at the call site, not in the task:
**`export PYTEST_XDIST_AUTO_NUM_WORKERS=1` in every GPU sbatch** (xdist honours
it to resolve `auto`; verified "created: 1/1 worker"). Invisible locally — the
laptop resolves `auto` to 16 and runs those tests on CPU anyway. This guard is a
sharp edge that every future GPU sbatch must remember; if it bites again, make
the tasks GPU-aware instead of relying on a comment.

## S5-S6 — TBD
