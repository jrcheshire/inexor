# inexor architecture

Status: bootstrap draft (2026-07-10), pre-M0. Every quantitative claim below is
either (a) verified against a primary source in the 2026-07-10 research session
(citations at bottom), (b) measured in mbody (file references given), or
(c) design analysis pending M0 validation — marked **[M0: Rn]** with the risk
experiment that tests it. Nothing here is folklore.

## 1. Thesis and non-goals

Large-scale-structure N-body is compute-light and memory-bound (TianNu used 73%
of TianHe-2's memory but 13% of its compute [CUBE18]; HACC: "memory-limited on
even the largest machines" [HACC-CACM]; PKDGRAV3 sized the 2-trillion-particle
Euclid Flagship to Piz Daint's RAM, not to any time budget [PKD17]). For
*differentiable* PM codes the wall is sharper: naive reverse-mode AD stores the
trajectory (O(N_steps) memory; pmwd demonstrably OOMs at 2x steps or 2^3x
resolution of a 128^3/15-step baseline [PMWD]), and the standard escape —
adjoint with reverse-time replay — replays in floating point, drifting by up to
~5e-2 of the field std in single precision (pmwd Table 2 [PMWD]; mbody's
reversible adjoint has the same class of wall at the float32 CIC scatter-add
floor, ~1e-4 cells, `mbody/tests/test_integrate.py`).

**inexor's thesis: put the phase space on a fixed-point integer lattice.** One
design decision buys three properties:

1. **Compression** — 12 B/particle (int16) or 6 B/particle (int8) persistent
   state, vs 24+ B float (CUBE's demonstrated design point [CUBE18]).
2. **Bit-exact reversibility** — integer state updated by rounded float
   increments is exactly invertible by modular subtraction (JANUS pattern
   [JANUS]); the adjoint's reverse replay returns the *same bits*, so gradient
   memory is O(1) in steps with no replay-drift error and no f64 tax.
3. **Periodic boundaries for free** — modular integer wrap IS the periodic box,
   and keeps every update bijective on Z/2^16.

The claim to defend in the paper: **exact discrete gradients of the simulation
actually run, at f32 speed and 80 GB-class memory for a 1024^3 full adjoint.**
This matters most on consumer GPUs, where f64 runs at 1/64 rate.

**Non-goals** (recorded so scope creep is a deliberate act): hydro; tree/P3M
short-range forces; multi-node distribution (sharded psums would reopen the
determinism question — post-M4 at the earliest); production SPHEREx mocks
(disco-mocks/DISCO-DJ own that); MLX/Metal backend (jax-metal is dead; mbody
remains the Apple-Silicon code).

## 2. Positioning vs prior art

| code | differentiable | adjoint memory | replay fidelity | state size | notes |
|---|---|---|---|---|---|
| pmwd | yes | O(1) in steps | float replay; f32 RMSD up to 5.2e-2, f64 1e-11..1e-13 [PMWD] | f32 | discretise-then-optimise-ish; reversal drift is the documented error source |
| DISCO-DJ | yes | O(1) in steps | float replay via Diffrax continuous backsolve (`nbody/steppers/leapfrog_adjoint.py`) | f32/f64 | optimise-then-discretise: gradients of the *continuous* flow, not of the discrete sim run |
| mbody | yes (MLX) | O(grid), step-independent | float32; scatter-add floor ~1e-4 cells (measured) | f32 | eager-only adjoint entry points; the direct ancestor |
| CUBE / CUBE2 | no | — | — | 6 bpp (asymptotic; 9.84 measured config), 12 bpp full-accuracy [CUBE18, CUBE2] | compression validated below the PM error floor at k <~ 0.2 k_Nyq |
| JANUS | no | — | bit-exact (integer phase space) [JANUS] | 64-bit ints | few-body direct summation; proves the reversibility mechanism |
| **inexor** | **yes** | **O(1) in steps** | **bit-exact** | **12 / 6 bpp** | the CUBE x JANUS x autodiff intersection — unoccupied |

inexor is discretise-then-optimise: `custom_vjp` gradients are the exact
transpose of the discrete forward map actually executed (up to the STE
treatment of rounding, Sec. 8), which is the correct object for optimizing
through *this* simulator.

**Citation guard** (verified corrections from the research session — do not
reintroduce): HACC has NO published bytes-per-particle figure (the "36 B/p"
floating around is PKDGRAV3's 2LPT IC-generation footnote); "CUBE2 ~20 B/p vs
Gadget-2 ~80 B/p" is a search-engine conflation (the 80 is Springel 2005's own
number; CUBE2 reports 6/12 bpp + 10-30% overhead, no Gadget comparison);
AbacusSummit kept sparse full time slices (don't say "lightcones instead of
snapshots" unqualified); pmwd's benchmark GPU is an H100 PCIe 80 GB, not A100.

## 3. State representation

### Positions: global box fixed-point, uint16[3, N]

Decode: `x_f = s_x * x_int`, with `s_x = L / 2^16` (lattice ~15.3 kpc/h for
L = 1000 Mpc/h). Key identity: with one particle per Lagrangian cell
(mbody's `n_particles == n_mesh` convention) and `n_mesh | 2^16`, the
Lagrangian site q_i is an exact lattice multiple, so global fixed-point
position and int16 displacement-from-site are the *same bits* up to an integer
offset known from the particle index. Store global; reinterpret as displacement
when convenient (int8 mode, diagnostics).

- **Integer wrap = periodic BC, exactly.** Modular arithmetic makes every
  drift a bijection on Z/2^16 regardless of overflow — no wrap logic, and
  reverse-subtraction is always exact.
- **Rejected: CUBE's cell-relative storage** (int offsets within a coarse cell,
  per-cell particle lists). Strictly higher precision, but per-cell sorting
  with dynamic occupancy is hostile to JAX (static shapes, vmap/scan, AD
  through permutations) and destroys the static index<->site map the adjoint
  and the int8 velocity reference rely on.
- **Honest cost:** at n_mesh = 1024 the global lattice gives 64 sub-cell
  levels (~6 fractional CIC bits); at 2048, 32 levels. This is the main
  physics price of the design. Quantization noise sigma ~ 4.4 kpc/h at
  L = 1000 — far below nonlinear displacement scales; expected signature is a
  ~k^2 sigma^2 white-ish P(k) floor. **[M0: R5]**

### Velocities: int16[3, N], BullFrog D-time, linear quantization

- Time variable is growth factor D with v = dx/dD. For a pure linear mode
  v = Psi_1(q) = const in D-time — all velocity dynamics is nonlinear growth.
  (This is why the D-time frame is the right home for a quantized velocity.)
- **Linear** quantization `v_f = s_v * w_int`. CUBE's CDF-shaped bins are
  rejected for the evolving state: nonlinear bins break additive-increment
  invertibility (Sec. 4) and make the straight-through gradient a biased
  bin-density mess. (CDF bins remain an option for *output* compression.)
- **Wrap, never clamp** — clamping is non-injective and destroys reversibility
  structurally; wrap gives an overflowing particle wrong physics but keeps the
  run exactly reversible and the adjoint exact. Monitor max|w_int| per step;
  fail loudly above ~0.9 * 32767. Physics-error vs correctness-error
  separation: overflow can bias science, it can never corrupt gradients
  silently. The no-clamp rule is a grep-able invariant (decisions.md D-007).

### int8 mode (6 B/p) — "small boxes / demos" tier

- Velocities: int8 residual vs a **static** coarse Zel'dovich Psi_1 reference
  gathered at the particle's Lagrangian site by static index (e.g. 128^3 x 3
  f32 = 24 MB). Because the ZA velocity in D-time is a *constant of the run*,
  the reference is deterministic, off the AD carry, and available in both sweep
  directions — dodging CUBE's evolving-reference chicken-and-egg problem in
  reverse.
- Positions: int8 cannot span the box; displacement-limited representation
  (+-R, e.g. +-32 Mpc/h -> 0.25 Mpc/h lattice), whose wrap is NOT periodic —
  a > R displacement wraps unphysically. Loud range monitoring; never the
  default.

Layout: SoA, `x: uint16[3, N]`, `w: int16[3, N]`, N = n^3 flattened.

## 4. The step: reversible integer BullFrog

BullFrog [BF, via mbody `integrate.py`] is a DKD with an affine kick:

```
x_mid = x + (dD/2) v
v'    = alpha v + (beta / D_mid) g(x_mid)
x'    = x_mid + (dD/2) v'
```

Coefficients port from mbody (`_bullfrog_weights`, `bullfrog_coeffs`; pinned by
the EdS closed-form test at < 1e-12).

### The alpha problem (real; no in-place fix exists)

JANUS works because leapfrog is shear-structured: every update is
`register += round(f(other_register))`, invertible by modular subtraction of
the identically recomputed increment. BullFrog's kick is NOT of this form:

1. `v' = rint(alpha v) + rint(kick)` is not injective for BullFrog's actual
   alphas (EdS: -5/7, 0.29, 0.53, ... -> 1 - 3/(2n)); any |alpha| < 1
   contraction collides lattice points (pigeonhole), and rint(v'/alpha) cannot
   recover v.
2. The tempting `v' = v + rint((alpha - 1) v + beta g)` fails too: the
   increment depends on the register being updated (self-referential), which
   the reverse sweep does not have yet.

Deeper: the kick's Jacobian is alpha^(3N) — it genuinely contracts velocity
phase-space volume, and no rounding scheme makes a contraction bijective on a
fixed lattice. The escape is a change of units, not a cleverer rounding.

### The fix: the w-frame scale ladder

Define P_0 = 1, P_{k+1} = alpha_k P_k, and store `w = v / (s_w0 P_k)`. Then
the affine kick becomes **purely additive** in w:

```
w' = w + [ beta_k / (D_mid,k P_{k+1} s_w0) ] g      ( := w + kappa_k g )
```

and the half-drifts pick up ladder factors:

```
c1_k = (dD_k/2) s_w0 P_k     / s_x        (pre-kick w)
c2_k = (dD_k/2) s_w0 P_{k+1} / s_x        (post-kick w')
```

All {c1, c2, kappa} are constants of the step, computed host-side in float64
(numpy — mbody's "precision island" pattern; `jax_enable_x64` stays off on
device) and cast to f32 scalars. Decoding v' = s_w0 P_{k+1} w' reproduces the
affine kick exactly. Negative alpha (from-cold first step) just sign-flips the
ladder; alpha != 0 is all that is required.

**Cost — dynamic range:** |P_K| = prod|alpha_k| shrinks roughly like K^(-3/2);
for a 10-step EdS-like schedule |P_10| ~= 0.02, i.e. the ladder consumes
~5.6 bits of the int16 budget. Consequences (all enforced by setup-time
float64 range assertions that refuse unfit schedules):

- int16 + BullFrog is comfortable for **K <~ 8-12 steps — exactly BullFrog's
  design regime** (DISCO-DJ II: 6 steps -> percent-level P(k) at k ~ 0.2 h/Mpc
  [DDJ2]). **[M0: R2]**
- K >~ 20 or int8 velocities: use **FastPM** (kick natively additive,
  alpha == 1, no ladder — port mbody's `fastpm_kick_factor`/`drift_factor`).
  This is the principled reason inexor ships both integrators.
- Fallback if range tightens: a **remainder ledger** — at chosen seams,
  re-quantize w onto a fresh scale and store the rounding remainders (int8,
  3 B/p per seam) so the seam stays bijective. O(#seams) memory. Recorded,
  not built.
- (Lemma worth a test, unused by BullFrog: `w' = rint(m w)` IS exactly
  invertible via `w = rint(w'/m)` when |m| > 1.)

### Step pseudocode

State `(x: uint16[3,N], w: int16[3,N])`; per-step constants `c = (c1, c2,
kappa)` f32. `rint_i(z) = jnp.rint(z).astype(int32)` — **route through int32**:
XLA float->int conversion overflow semantics are backend-defined, while
int32 -> int16/uint16 narrowing is guaranteed modular. Adds are
`(state.astype(int32) + inc32).astype(state.dtype)` == modular lattice add.

```python
def step_fwd(x, w, c):
    x1 = iadd_u16(x, rint_i(c.c1 * w.astype(f32)))     # half-drift 1
    g  = force_f32(dequant_x(x1))                      # Sec. 6; deterministic
    w1 = iadd_i16(w, rint_i(c.kappa * g))              # additive kick (w-frame)
    x2 = iadd_u16(x1, rint_i(c.c2 * w1.astype(f32)))   # half-drift 2
    return x2, w1

def step_rev(x2, w1, c):                               # exact inverse
    x1 = isub_u16(x2, rint_i(c.c2 * w1.astype(f32)))
    g  = force_f32(dequant_x(x1))                      # same bits in -> same bits out
    w  = isub_i16(w1, rint_i(c.kappa * g))
    x  = isub_u16(x1, rint_i(c.c1 * w.astype(f32)))
    return x, w
```

Three roundings per step, all of the JANUS form
`register +-= rint(f(other_register, consts))`; the kick's force argument x1
is reconstructed in reverse *before* it is needed. `jnp.rint` is IEEE
round-half-even (deterministic); int16 -> f32 is exact; a single f32 multiply
is correctly rounded on every backend.

**Bit-exactness scope (honest):** same process, same device, same XLA
version/flags. CPU vs GPU, or different cuFFT versions, give different f32
forces; reversibility and adjoint replay hold *within* a run — which is all
the adjoint needs.

## 5. Determinism of the force path

Bit-exact replay requires `force_f32(x)` to be a deterministic function of the
bits of x. The only nondeterministic op in the PM chain is the CIC
**scatter-add** (f32 atomics in nondeterministic order = non-associative).
Gather, elementwise ops, and cuFFT-for-a-fixed-plan-on-a-fixed-device are
deterministic.

**Primary design: integer-accumulation paint.** Quantize the 8 CIC corner
weights to fixed point (~2^11-12 fraction bits, int32) and scatter-add into an
**int32 mesh**. Integer addition is associative and commutative, so the result
is bit-identical regardless of atomic ordering — determinism by algebra, not
serialization, at full atomic speed. Headroom: max mass/cell ~1e4 particles x
2^12 << 2^31; keep fraction bits low enough that cell sums stay < 2^24 for
exact int32 -> f32 conversion. **[M0: R3]**

Crucial scoping simplification: determinism is required **only on the primal
force path** (trajectory + replay). The backward per-step VJP (Sec. 8) uses the
ordinary differentiable f32 CIC paint — atomics nondeterminism there perturbs
gradients at the ulp level only, never the trajectory. The int paint therefore
needs no VJP rule at all.

Rejected alternatives: `--xla_gpu_deterministic_ops` (process-global
pessimization; bring-up crutch only); sort + segment_sum (a 2^30-key sort per
force solve + a 4 GiB permutation; fallback only); 8-color corner partitioning
(a hand-written-kernel trick that does not compose in XLA).

**Residual hazard:** forward and reverse sweeps are different XLA programs; in
principle fusion could differ in the force's elementwise epilogue, flipping
last-ulp bits that matter only on exact rint half-ties (~1e-7 of particles,
but nonzero). Mitigations, in order: share the identical `force_f32` function
object/jaxpr in both step functions; if that ever fails, per-step `jax.jit`
dispatch so one compiled executable serves both sweeps (independently
motivated by memory, Sec. 9). Tier-0 CI test = forward K then reverse K equals
initial bits, on GPU, many seeds. **[M0: R1]**

Related environment facts: JAX on macOS-arm64 CPU has known run-to-run
nondeterminism quirks (memory note: backend-resolved defaults pattern) — R1's
authoritative runs are CUDA; pad shapes to powers of two to avoid the
per-shape recompile tax.

## 6. Force solve (port of mbody, dtype-disciplined)

- Geometric split (mbody `forces.py`): solve `lap phi = delta`, return
  `g = -grad phi`; ALL cosmology prefactors live in the integrator
  coefficients. Force kernel `ik/k^2` is literally the Zel'dovich displacement
  kernel — shared `_k_components(box)` module and a machine-precision
  force == ZA cross-check test (mbody convention).
- rfftn half-grid (N, N, N//2+1); k = 0 force vanishes; spectral ik kernel
  (finite-difference kernels + CIC deconvolution are recorded future options,
  as in mbody).
- CIC (mbody `painting.py`): 8-corner trilinear; `stop_gradient` on integer
  cell indices, gradients through fractional weights only; `cic_read_vector`
  reads all 3 force components through one shared stencil (~40% gradient
  memory saving, measured in mbody).
- **Sequential per-component solves** — never materialize 3 force meshes at
  once (memory, Sec. 9).
- Two paint implementations, one interface: `paint_int` (deterministic primal,
  Sec. 5) and `paint_f32` (differentiable twin for the VJP path).

## 7. Initial conditions

Port mbody `ic.py` / `lpt.py`: Gaussian phi_G from a seeded spectrum, local
f_NL in the potential (`phi = phi_G + f_NL (phi_G^2 - <phi_G^2>)`, then
`delta = M phi` with the Poisson/transfer factor M(k, z)), Zel'dovich + 2LPT
displacements, differentiable in (f_NL, amplitude). The analytic tree-level
local-bispectrum template comes along as a test oracle. IC encode is the STE
boundary: `(x0_f, v0_f) -> (x_int, w_int)` via `ste_round`.

## 8. Autodiff design

Public API takes and returns floats; integers are internal representation.

**The STE (straight-through estimator) idiom:**

```python
def ste_round(z):          # primal == rint(z); Jacobian == identity
    return z + lax.stop_gradient(jnp.rint(z) - z)
```

**The float twin.** `step_float(xf, wf, c)` mirrors `step_fwd` but in f32 with
`ste_round` at the three rounding sites — so its *primal outputs bit-match the
true integer trajectory* while its Jacobian treats rounding as identity. Each
per-step VJP is then linearized exactly on the trajectory the forward pass
actually took. Gradient error from ignoring rounding is O(quantization step);
whether that stays benign over ~10 steps is THE scientific risk. **[M0: R4]**

**custom_vjp structure:**

```python
@partial(jax.custom_vjp, nondiff_argnums=(0,))
def evolve(cfg, x0_f, v0_f, coeffs):
    x, w = encode(cfg, x0_f, v0_f)                    # ste_round at the boundary
    (x, w), _ = lax.scan(lambda s, c: (step_fwd(*s, c), None), (x, w), coeffs)
    return decode(cfg, x, w)

def evolve_fwd(cfg, x0_f, v0_f, coeffs):
    out = evolve(cfg, x0_f, v0_f, coeffs)
    return out, (encode_final(out), coeffs)           # residuals: FINAL int state only

def evolve_bwd(cfg, res, cot):
    def rev_body(carry, c):
        x, w, xbar, wbar = carry
        x_prev, w_prev = step_rev(x, w, c)            # bit-exact replay
        xf, wf = dequant_x(x_prev), dequant_w(w_prev, c)
        _, vjp = jax.vjp(lambda a, b: step_float(a, b, c), xf, wf)
        xbar, wbar = vjp((xbar, wbar))
        return (x_prev, w_prev, xbar, wbar), None
    ... lax.scan(rev_body, ..., coeffs, reverse=True)
    return xbar_ic, wbar_ic, None                     # through encode's STE
```

- Residuals are O(1) in steps (final int state only) — reversibility IS the
  checkpointing; no `jax.checkpoint` across steps.
- Within a step: `jax.checkpoint` around the CIC paint/read (mbody's
  `recompute_cic`, measured ~18% peak-VJP saving) — the FFT solve is linear
  and needs no saved activations.
- Mixed-dtype scan carry `(uint16, int16, f32, f32)` is scan-native. Per-step
  coefficients (including the ladder scale, so dequant units stay consistent)
  ride as stacked `xs`; cotangents are kept in w-lattice units and converted
  once at the ends.
- Because `evolve` is a genuine `custom_vjp`, it composes with outer
  `jax.grad` / `jit` / optimizers — a concrete upgrade over mbody's eager-only
  `adjoint_grad_*` entry points (thin parity wrappers retained).

## 9. Memory budget (flagship: 1024^3 particles + 1024^3 mesh, 80 GB GPU)

Persistent through the backward sweep:

| item | dtype | size |
|---|---|---|
| x_int | uint16 x3 | 6 GiB |
| w_int | int16 x3 | 6 GiB |
| xbar | f32 x3 | 12 GiB |
| wbar | f32 x3 | 12 GiB |
| **carry** | | **36 GiB** |

Per-reverse-step transients (f32 mesh = 4 GiB at 1024^3): dequantized (xf, wf)
at the VJP point; checkpointed CIC fraction weights; delta mesh; rfft spectrum
(~4 GiB complex64); **sequential per-component force solves** (one mesh live at
a time); cuFFT workspace ~4-8 GiB; force-at-particles 12 GiB. Realistic peak
**~65-75 GiB — fits 80 GB with no slack to waste.** The int state is exactly
what makes it fit; a float-state design pays +24 GiB and busts the budget.
That asymmetry is the paper's memory claim. **[M0: R6-class profiling at 512^3
before believing the extrapolation]**

Two identified ceilings:

1. **scan-carry double-buffering.** If XLA fails to alias the 36 GiB carry in
   place, the sweep wants 72 GiB up front. Production driver is therefore a
   Python loop over per-step `jax.jit(step, donate_argnums=...)` — K ~ 10
   dispatches is negligible, donation gives true in-place semantics, and it
   independently pins forward/reverse to one compiled executable (Sec. 5).
   `lax.scan` remains the small-problem/test driver.
2. **The mesh at 2x resolution.** A 2048^3 mesh does not fit 80 GB even
   forward-only (delta + spectrum + one component + workspace > 100 GiB). The
   flagship target stands at 1024^3/1024^3; a CUBE2-style two-level
   (coarse-global + fine-local) mesh is the post-M4 growth path, recorded not
   designed.

## 10. Validation architecture

mbody's conventions carry over wholesale:

- **Analytic oracles:** BullFrog weights vs EdS closed form (< 1e-12); force
  == Zel'dovich identity; tree-level local-bispectrum template; FastPM/
  BullFrog reduce-to-exact small-step tests.
- **Cross-code:** forward parity vs mbody (same physics, different backend)
  and vs DISCO-DJ at matched config. **Oracle-parity-floor rule:** measure the
  reference's own reproducibility floor FIRST (memory note
  reference_oracle_parity_floor), then set gates; tolerances are never relaxed
  without discussion with JC.
- **Reversibility tier-0:** forward K + reverse K == initial bits (exact
  integer equality, not a tolerance), CPU and CUDA, many seeds.
- **Gradient gates:** custom_vjp vs `jax.grad` of the float twin vs central
  finite differences; adjoint-vs-replay parity (mbody pattern).
- **Quantization physics:** CUBE-style check that int16 error < the PM
  algorithm's own error for k <~ 0.2 k_Nyq under few-step BullFrog stepping.
- Optional-import CCL cross-check layer for cosmology (mbody
  `test_external_ccl.py` pattern).
- Measured-floor tolerances documented with provenance comments (which probe
  script measured them), per mbody/sfbfs house style.

## Module layout

```
src/inexor/
  config.py      # frozen hashable dataclasses (Box/Time/Quant configs)
  cosmology.py   # growth D(a), E(a); host float64 island (numpy/scipy)
  codec.py       # NOVEL CORE 1: lattice scales, encode/decode, ste_round,
                 # w-frame ladder + setup-time range assertions,
                 # int32-routed rint_i / modular iadd/isub, overflow monitors
  painting.py    # paint_int (deterministic primal) + paint_f32 (diff twin)
                 # + cic_read / cic_read_vector
  forces.py      # geometric Poisson solve, shared _k_components,
                 # sequential per-component solves; force_f32 entry
  lpt.py         # ZA/2LPT displacements (shared ik/k^2 kernel)
  ic.py          # linear density, P(k) input, f_NL-in-the-potential
  integrate.py   # bullfrog/fastpm coefficient tables (float64 host),
                 # step_fwd / step_rev / step_float, scan + per-step-jit drivers
  adjoint.py     # NOVEL CORE 2: custom_vjp evolve, reverse sweep,
                 # IC-parameter convenience wrappers
  diagnostics.py # reversibility check, wrap counters, P(k) estimator
```

Everything except `codec.py` and `adjoint.py` is a port-with-dtype-discipline
from mbody (`mbody/integrate.py`, `forces.py`, `painting.py`, `lpt.py` are the
reference files).

## Sources

Verified in the 2026-07-10 research session (primary PDFs):
[CUBE18] Yu, Pen & Wang 2018, ApJS 237:24 (arXiv:1712.06121) — 6 bpp, x1v1
error below PM error at k <~ 0.2 k_Nyq, TianNu 73%/13%.
[CUBE2] arXiv:2512.12629 — 6/12 bpp modes, IOS-as-linked-list, multi-level PM.
[JANUS] Rein & Tamayo 2017, MNRAS 473:3351 (arXiv:1704.07715) — bit-wise
reversible integer phase space; 1000-particle collapse recovered exactly.
[PMWD] Li et al., ApJS (arXiv:2211.09815) — adjoint + reverse replay, O(1)
memory; f32 reversal RMSD to 5.2e-2; AD OOM baselines; H100 benchmark.
[DDJ2] DISCO-DJ II (arXiv:2510.05206) — adjoint memory independent of steps;
BullFrog 6-step accuracy; Diffrax backsolve implementation (site-packages
`discodj/nbody/steppers/leapfrog_adjoint.py`).
[PKD17] Potter, Stadel & Teyssier 2017 (arXiv:1609.08621) — 62 B/p budget
(Table 1), memory-limited framing, on-the-fly analysis, 2LPT-in-36-B footnote.
[HACC-CACM] Habib et al., CACM — "memory-limited on even the largest
machines"; particle overloading ~10%.
[BF] Rampf, List & Hahn 2024 (arXiv:2409.19049), via mbody's implementation
and docs/bullfrog.md derivation.
[Abacus] Garrison et al. 2021, MNRAS 508:575 — 32 B/p state; cell-offset
mantissa trick (the float cousin of our fixed-point move).
