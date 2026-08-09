# inexor decision log (ADR-style)

Format: Status / Context / Decision / Consequences. Newest at bottom. A
decision here is locked until explicitly re-litigated with JC.

---

## D-001 — int16 default, int8 opt-in
- **Status:** accepted (JC, 2026-07-10 bootstrap).
- **Context:** CUBE validated 1-byte phase space below the PM error floor, but
  at cell-relative precision we deliberately trade away (D-004); gradient
  fidelity through quantization is untested (M0 R4).
- **Decision:** int16 (12 B/p, CUBE's "full accuracy" tier) is the default and
  the paper's primary configuration; int8 (6 B/p) is an opt-in tier built in
  M3, presented as a demonstrated mode rather than the headline.
- **Consequences:** de-risks M0-M2; the memory claim is still decisive
  (12 vs 24+ B/p is what makes the 1024^3 adjoint fit 80 GB).

## D-002 — M0 is a hard go/no-go gate
- **Status:** accepted (JC, 2026-07-10).
- **Context:** the deepest risk (STE gradient fidelity, R4) is cheap to test
  and existential.
- **Decision:** M0's five probes run before real implementation; R4 failure at
  int16 pivots to an exact-reversible UNCOMPRESSED design (still novel) or
  stops the project. Verdicts recorded here.
- **Consequences:** protects the investment; the pivot path is pre-agreed so a
  negative result is a cheap outcome, not a sunk cost.

## D-003 — discrete exact-replay adjoint (discretise-then-optimise)
- **Status:** accepted (design pass, 2026-07-10).
- **Context:** DISCO-DJ's adjoint is a Diffrax continuous backsolve
  (optimise-then-discretise: gradients of the continuous flow); pmwd replays
  the discrete map but in floating point (f32 drift to 5.2e-2 RMSD).
- **Decision:** inexor computes the exact transpose of the discrete forward
  map actually executed, via custom_vjp + bit-exact integer replay.
- **Consequences:** gradients are exactly consistent with the simulator (the
  right object for optimization through it); the differentiation vs prior art
  is architectural, not incremental.

## D-004 — global uint16 fixed-point positions; wrap = periodic BC
- **Status:** accepted (design pass, 2026-07-10).
- **Context:** CUBE stores cell-relative offsets (higher precision, needs
  per-cell particle sorting); JAX wants static shapes and static index maps.
- **Decision:** positions are global box fixed-point uint16 (== displacement
  from the Lagrangian site up to a static integer offset); modular wrap is the
  periodic box; cell-relative storage rejected.
- **Consequences:** 64 sub-cell levels at 1024 mesh (~6 fractional CIC bits) —
  the design's main physics cost, validated in R5; every update is bijective
  on Z/2^16 by construction.

## D-005 — the w-frame scale ladder for BullFrog's contracting kick
- **Status:** accepted (design pass, 2026-07-10); range budget validated in R2.
- **Context:** BullFrog's affine kick (|alpha| < 1) contracts velocity phase
  space; no rounding scheme is bijective for a contraction on a fixed lattice;
  the self-referential increment reformulation also fails.
- **Decision:** store w = v / (s_w0 * P_k) with P_{k+1} = alpha_k * P_k; the
  kick becomes purely additive (JANUS-legal); drift/kick constants are host
  float64 per step.
- **Consequences:** costs ~5-6 bits of int16 over ~10 steps -> int16+BullFrog
  is a K <~ 8-12 design (BullFrog's regime); FastPM (alpha == 1) is the
  many-step/int8 integrator; remainder-ledger fallback recorded unbuilt;
  setup-time float64 range assertions refuse unfit schedules.

## D-006 — int32-mesh CIC paint: determinism by associativity
- **Status:** accepted (R3 confirmed on CUDA, 2026-07-13 — see D-010).
- **Context:** f32 scatter-add atomics are order-nondeterministic; bit-exact
  replay needs a deterministic primal force.
- **Decision:** quantize CIC corner weights to fixed point, accumulate in an
  int32 mesh (integer addition is associative -> order-independent), convert
  to f32 delta afterwards. Determinism needed on the primal path only; the
  backward VJP uses the ordinary f32 differentiable paint (no VJP rule for the
  int paint).
- **Consequences:** full atomic speed expected; rejected alternatives
  (--xla_gpu_deterministic_ops, sort+segment_sum, 8-color) recorded with
  reasons in architecture.md Sec. 5.

## D-007 — wrap-never-clamp state invariant
- **Status:** accepted (design pass, 2026-07-10).
- **Context:** clamping/saturation is non-injective and silently destroys
  reversibility; wrap gives wrong physics but exact replay.
- **Decision:** no saturating operation may ever touch the integer state;
  overflow is monitored loudly (per-step max|w| counters) and is a physics
  error, never a correctness error. Lint/grep-testable invariant.
- **Consequences:** a run can be scientifically wrong but its gradients are
  never silently corrupted; monitoring is a first-class diagnostic.

## D-008 — ruff (not black/flake8)
- **Status:** accepted (2026-07-10).
- **Context:** sfbfs/jht lineage uses ruff, line-length 100; mbody's
  black/flake8 is the older convention.
- **Decision:** ruff check + ruff format, line-length 100. ASCII-only in .py
  files (house style).

## D-009 — no MLX backend
- **Status:** accepted (2026-07-10).
- **Context:** jax-metal is dead (Apple abandoned it for MLX); xcat descoped
  its MLX seam; mbody is the MLX code.
- **Decision:** inexor is JAX-only (CPU + CUDA). Apple-Silicon GPU work stays
  in mbody.
- **Consequences:** local dev runs JAX-CPU; authoritative
  determinism/performance results come from CUDA devices (Vista/albireo).

## D-010 — M0 gate review: verdicts and GO
- **Status:** accepted (JC, 2026-07-13 gate review).
- **Context:** all five probes ran to verdict (CPU legs 2026-07-12; R1/R3
  CUDA-authoritative on albireo/deneb's RTX 3050 [6 GB variant], 2026-07-13).
  Thresholds were measure-then-negotiate per D-002. Probe outputs in
  `runs/m0/` (gitignored); full logs beside them.
- **Decision:**
  - **R4 gate ratified at 7.5e-2 relative gradient error** at production
    int16. Measured 9e-4..3e-2 across all six loss x param combos (within
    0.03-0.26 sigma of the staircase-regression FD reference), anchor O(q)
    slopes +1.0..+2.2, no K growth 3->12: **PASS**.
  - **R2 noise bar ratified as <= 1e-4 relative P(k)** (the strict
    below-stepping-floor criterion rejected as the governing bar — BullFrog
    is near-exact at low k where that floor is ~1e-6). Measured pass with a
    >= 3-bit s_w0 window: **PASS**.
  - **R1 PASS** (authoritative CUDA): 100/100 seeds x all 5 drivers, exact
    integer equality, wrap-adversarial exact; int-paint primal force. The
    forbidden f32-paint configuration fails densely on the same hardware
    (archived, `runs/m0/r1_gpu_f32paint/`) — the design's premise shown both
    ways.
  - **R3 PASS** (CUDA, 2^26/384^3 fallback; 512^3 uniform corroboration
    archived): int paint 10/10 bit-identical incl. re-trace and the
    clustered contention worst case; 0.79-0.85x f32 runtime (FASTER, vs the
    >2x trigger); f32 + --xla_gpu_deterministic_ops costs 1.37-1.78x f32.
  - **R5 strict PASS** (10-100x below the resolution floor at
    flagship-equivalent 64 levels; error ladder discriminates).
  - Single-seed R2/R5 matrices accepted (no second seed).
  - **GO to M1.**
- **Consequences:** both load-bearing bets validated; M1 opens with its own
  milestone plan. s_w0 policy tightened (D-011); schedule feasibility
  documented as a paper limitation (D-012). R4's gradients were measured
  through the f32-paint force (the intended VJP twin, D-006) while the
  production primal is the int paint — restate when M2 promotes the adjoint.

## D-011 — s_w0 policy: c_growth 4.0 -> 2.5
- **Status:** accepted (JC, 2026-07-13 gate review).
- **Context:** the pre-R2 placeholder c_growth = 4.0 was ~1 bit conservative;
  R2 measured the velocity growth factor at 1.94 (128^3, exact LCDM, K = 5-15
  log schedules).
- **Decision:** default c_growth = 2.5 (~0.3 bits of headroom over the
  measured 1.94); margin 0.9 unchanged.
- **Consequences:** ~0.6 bits returned to the velocity range budget; the
  policy stays a default, not an assertion — D-005's setup-time float64
  range checks still refuse unfit configurations.

## D-012 — schedule feasibility is a documented constraint (K >= 3)
- **Status:** accepted (JC, 2026-07-13 gate review).
- **Context:** the w-frame ladder (D-005) inherits a feasibility condition
  from BullFrog's alphas: a rung's bit cost diverges as |alpha_k| -> 0, and
  the alphas depend only on the schedule (growth factors at step endpoints).
  Measured: K = 2 from a_i = 0.1 has alpha_1 = 0.018 — below the guard, on
  the zero-crossing — so two steps cannot run at int16+BullFrog; lin-0.04
  crosses zero near K = 11. Float BullFrog has no such constraint (it just
  integrates K = 2 inaccurately).
- **Decision:** keep the |alpha_k| >= 0.05 setup-time guard; document
  "int16+BullFrog needs K >= 3 from a_i = 0.1, and any schedule must keep
  every |alpha_k| >= 0.05" as a paper limitation (paper/outline.md), not a
  bug to engineer away.
- **Consequences:** a reviewer sweeping K downward hits a loud, documented
  refusal instead of silent garbage; the remainder-ledger fallback (D-005)
  remains the unbuilt escape hatch if a near-zero-alpha schedule is ever
  genuinely needed.

## D-013 — M1 Tier-A parity gates (mbody + DISCO-DJ)
- **Status:** accepted (JC, 2026-07-13; "green light" after the attribution
  review).
- **Context:** floor-first protocol (plan tender-stargazing-map S5/S6).
  Measured floors: mbody repeats at ~1e-6 cells rms (Metal f32 CIC scatter);
  DISCO-DJ repeats bit-identical (JAX f64 CPU, fresh-trace included). The raw
  inexor <-> DISCO-DJ gap was ATTRIBUTED to three named convention choices
  (docs/m1-results.md S6): (1) Nyquist plane zeroed in DISCO-DJ's order-0 ik
  gradient kernel — the ONLY force-operator difference (7e-8 residual);
  (2) BullFrog alpha from tabulated true-LCDM D2 vs mbody's EdS relation;
  (3) explicit-a-array midpoint = arithmetic in a with unequal D-drift halves
  vs mbody's D-midpoint equal halves. The convention-adapted replay
  reproduces disco_final to rms 2.2e-8 cells / |dP/P| 1.9e-8 (f64 roundoff).
- **Decision:** two permanent Tier-A parity gates, run by the harness
  (scripts/m1_*), gated against the shared physics:
  - **mbody arm** (injected matched ICs, stock conventions): rms <= 1e-4
    cells, max |dP/P| <= 1e-5. (Measured at 128^3 K=40: 1.6e-5 / 1.4e-6.)
  - **DISCO-DJ arm** (convention-adapted replay, m1_force_probe --side
    coeffs/replay): rms <= 1e-6 cells, max |dP/P| <= 1e-6, max 1-r <= 1e-12.
    (Measured: 2.2e-8 / 1.9e-8 / 2.1e-14.)
  The raw stock-vs-DISCO-DJ comparison is documented in m1-results.md as a
  characterized convention difference, NOT gated: its size is set by
  DISCO-DJ's kernel/coefficient choices, outside inexor's control.
- **Consequences:** inexor's forward physics is pinned to BOTH references at
  their respective information limits (mbody at its f32 floor, DISCO-DJ at
  f64 roundoff); any future regression in force kernel, LPT, growth, or
  stepping trips a gate with ~50x headroom rather than hiding in convention
  noise. The convention ledger doubles as the suspect list for the M0 ~4%
  low-k deficit (S6 matrix).

## D-014 — M1 Tier-B quantization gate (5e-4 end-to-end)
- **Status:** accepted (JC, 2026-07-13).
- **Context:** the end-to-end int16 quantization error at the production
  config (128^3, K=40, BullFrog, int paint) is k-flat at 1-3e-4 in |dP/P|
  (max 2.8e-4), is PURE phase-space quantization (the f32-kernel share is
  4e-7), and sits below the PM method's own floors in every band — mesh
  floor 9x at low k, ~2000x mid; stepping floor except at low k where
  BullFrog's linear exactness makes that floor artificially tiny
  (docs/m1-results.md Tier-B table; scripts/m1_quant_gate.py).
- **Decision:** Tier-B gate = **max |dP/P| (int16 vs evolve_float f64,
  matched schedule, production config) <= 5e-4**, PLUS the strict invariant
  that the quantization error stays below the PM mesh floor in every k band.
  D-010's 1e-4 bar REMAINS IN FORCE for the like-for-like ladder-budget
  statistic it was ratified on (M0-R2); it is a different measurement and
  is not relaxed by this ADR.
- **Consequences:** the paper's "compression is free" claim (CUBE-style)
  is backed by a measured floor comparison, not an analogy; frac_bits=12 /
  c_growth=2.5 stand as production defaults with ~2x gate headroom. If a
  future config trips the gate, the escalation path is frac_bits 13 or
  c_growth tuning, costed against the ladder budget first.

## D-015 — M2 gradient-fidelity gate (global-metric, floor-first)
- **Status:** accepted (JC, 2026-07-13).
- **Context:** M2's promoted adjoint = int-paint primal trajectory (bit-exact
  replay) + f32-paint STE-twin VJP (D-006). The D-010 R4 gate (7.5e-2 relative
  gradient error) was measured on the M0 twin probe and explicitly deferred for
  restatement to M2. Floor-first measurement (scripts/m2_grad_gate.py, x64,
  64^3, both integrators): the per-particle gradient of a band-power / field-L2
  loss is intrinsically NOISE-DOMINATED -- the reference's own floors correlate
  only ~0-0.7 per component (float-twin f32-vs-f64 corr ~0-0.06; central-FD-vs-
  float corr 0.34-0.71), and FD of the QUANTIZED loss is invalid below the
  lattice step (the R4 staircase). Against that, the adjoint reproduces the f64
  float-path gradient GLOBALLY to median-rel 1.9e-4 (band_power) / 3.3e-3
  (field_L2), corr > 0.9996, norm-ratio ~1.000; IC-param d/d[f_NL, amplitude]
  to rel 6e-5..9e-4.
- **Decision:** gate on GLOBAL metrics, NOT per-component max-rel or FD. Per
  config x loss: **adjoint-vs-float input-grad median-rel (top-decile |ref|)
  <= 1e-2 AND Pearson corr >= 0.999 AND |norm-ratio - 1| <= 1%**, and
  **IC-param adjoint-vs-float rel <= 5e-3**. The R4 restatement (D-010) is the
  headline: promoted-adjoint gradient error (worst median-rel) 3.3e-3 << 7.5e-2.
  GATE PASS at 64^3 (both integrators). Gate-grade 128^3 K=40 + the CUDA arm
  confirm on deneb (S4).
- **Consequences:** the paper's "f32-speed gradients with f64-grade fidelity"
  claim (core claim 3) is backed by a measured triangulation, not an assertion;
  the f32-paint VJP twin (D-006) is validated as gradient-faithful despite the
  int-paint primal. The reference-floor characterization also fixes the correct
  gate REFERENCE for the money plots (P2): the float-replay drift is compared
  globally, never per-component.

## D-v2-8 — v2 requirements chain (seed V0 pin)
- **Status:** accepted (JC, 2026-07-15, seed V0).
- **Context:** v2 is a GENERAL memory-floor PM N-body engine — "N-body that
  takes less memory," full stop. It is deliberately NOT a SPHEREx main-line
  mock producer (disco-mocks + DISCO-DJ own that role) and its goals, framing,
  and scale regimes must not be assumed to mirror the SPHEREx-focused
  projects (JC caution, 2026-07-15). Requirements below are pinned in
  engine-intrinsic terms; the survey-specific inputs examined at V0 (v28 bin
  table, abundance matching, per-bin Fisher) were demoted to non-normative
  session context and live in the V0 worklog, not here.
- **Decision (the chain — every item JC-ratified at V0):**
  1. **k_sci = 2.0 h/Mpc.** Statistics must be numerically clean (below the
     PM error floor, D-v2-1's k <= 0.2 k_Nyq criterion) through k = 2.0,
     giving **fine-mesh cell <= 0.2*pi/k_sci = 0.31 Mpc/h**. Motivation:
     bias-model studies need divergence in k in [1, 2] to be attributable to
     the model, not the mesh. (The disco-mocks Quijote-config validation that
     prompted this ran cell = 1.95 Mpc/h, k_Nyq = 1.61 h/Mpc: its k = 1-2
     band lay at 0.6-1.25x Nyquist, numerically invalid by this criterion.)
  2. **Bias route = hybrid.** Halo catalogs (FoF or proxy finder) at
     CALIBRATION configs to fit/validate bias models; field-level Lagrangian
     bias for PRODUCTION runs. The halo finder is in scope but off the
     production critical path.
  3. **n_p = 100** particles per halo at the mass floor (the standard bar for
     halo-clustering/bias robustness).
  4. **M_min is DERIVED, not required.** spacing = (mesh ratio) * cell with
     mesh:particle ratio in [1, 2] (G5 measures where the split lands) ->
     spacing 0.31-0.63 Mpc/h -> m_p ~= 2.6e9-2.2e10 Msun/h -> M_min(n_p=100)
     ~= 2.6e11-2.2e12 Msun/h. Each config REPORTS its M_min; no tomographic
     or survey-sample frame enters the requirement. If a consumer later needs
     a lower M_min, that moves the config table, not the architecture.
  5. **Velocity/RSD bar = multipole-grade.** Codec + numerics error in P0 AND
     P2 below the fine-mesh PM floor for k <= 0.2 k_Nyq — D-v2-1 applied to
     redshift-space multipoles. This is the G2c velocity-codec gate bar.
  6. **Tolerances.** P(k): D-v2-1 (CUBE-grade vs the fine mesh) — **the bar
     half of this is SUPERSEDED by D-v2-9: absolute |dP/P| <= 3e-2 in-band.**
     Squeezed bispectrum: <= 15% (D-v2-7). Calibration configs: HMF within 5%
     of a calibrated reference over M > M_min; halo b1 within 2% at k <= 0.25.
  7. **Realizations.** Nominal production batch = 100; O(5) at calibration
     configs (bias fitting is not a covariance job). Covariance-grade
     ensembles (500-1000) are OUT OF SCOPE — that is disco-mocks' role.
  8. **Cost-of-memory instrument ratified as specced** (plan-plan Sec 3:
     every gate reports peak B/p all-in, wall/step, SU per
     realization-equivalent; Pareto rule at V4; >5x SU for 2x memory needs
     explicit JC sign-off).
- **Consequences:** the requirement chain runs off k_sci alone — no survey
  dependency to rot. The v2 config table (plan-plan Sec 2) frames capacity by
  HARDWARE CLASS (deneb dev / single GH200 / S3 H100 node), not by survey
  samples. Gate week (V1-V3) measures against these bars; V4 re-derives the
  capacity numbers with gate-measured B/p. Tolerances are never relaxed
  without JC (standing convention).

## D-v2-9 — the P(k) fidelity bar becomes ABSOLUTE (D-v2-1's floor half retired)
- **Status:** accepted (JC, 2026-07-15, seed V2a). Supersedes the BAR half of
  D-v2-1; D-v2-1's BAND half is untouched and still pins D-v2-8 clause 1.
- **Context:** D-v2-1 bundled two separable things: a BAND (k <= 0.2 k_Nyq of
  the fine mesh) and a BAR ("error below the PM error floor"). The band is
  sound. The bar is not measurable as written.
  Job 39 tried to measure that floor by refining the MESH at FIXED particles
  (256^3 particles seen by a 256 / 512 / 1024 mesh). Its 1024 rung therefore ran
  mesh:particle = 4 — four cells inside one interparticle gap — which resolves
  two-body scattering the 512 run smooths over. Its (kc)^2 validity check failed
  (1.35 force / 0.76 evolved against ~4) and failed decisively: e(256 vs 512) =
  0.050 was SMALLER than e(512 vs 1024) = 0.066, i.e. the discrepancy GREW under
  refinement, which mesh truncation error cannot do. The rung measured the onset
  of discreteness, not convergence. **The split error itself was never in
  question** — it is measured against mono at the SAME mesh with the SAME
  particles, so the discretization is common-mode and cancels exactly. Only the
  bar was broken.
  Job 40 then measured C-dev's error on a ladder at FIXED mesh:particle = 2
  (128^3/256, 256^3/512 = C-dev, 512^3/1024 = reference), matched phase (proven:
  degenerate limit 5.9e-16, shared modes 6-7e-16). Result: **6.14e-2 in-band max**
  (5.90e-2 Poisson-subtracted). The reference is itself unconverged, so this is a
  **LOWER BOUND**, and it is not "the mesh transfer" narrowly — it bundles force
  resolution, IC bandwidth and particle load. Richardson is unavailable: the
  k-dependence is textbook (k^1.95 over k = 0.4-1.4) but the amplitude scales as
  cell^1.27, not cell^2, because joint refinement moves IC bandwidth and particle
  load too and only the force follows the mesh law.
- **Decision:**
  1. **Band: unchanged.** k <= 0.2 k_Nyq of the fine mesh (D-v2-1's surviving
     half). D-v2-8 clause 1 (k_sci = 2.0 -> fine cell <= 0.31) is unaffected.
  2. **Bar: |dP/P| <= 3e-2, ABSOLUTE, in-band**, for the two-level split against
     monolithic at the same mesh. Replaces "below the PM error floor".
  3. **Mandatory reported diagnostic (not gated): the split-to-discretization
     ratio vs k.** A single max-over-band number provably hides an
     order-of-magnitude low-k excess (below); the bar may not be read without
     its shape.
- **Rationale:** ABSOLUTE because D-v2-8 pins fine cell <= 0.31 for every config
  in the table — C-dev / C-gh / C-hero differ in VOLUME, not resolution — so the
  discretization error at fixed k is common to all three and the bar transfers
  unchanged. A floor-relative bar would need re-measuring per config and would
  re-open this contamination every time. 3e-2 is ~half the measured lower bound
  (6.14e-2) on the config's own discretization error, so the split can never
  dominate the band's high-k end where the bias-model purpose lives (k = 1-2).
  The bar is deliberately NOT grounded on the low-k regime where the split does
  dominate: no D-v2-8 science bar demands better there, and inventing one would
  be unsupported.
- **Consequences:**
  - **The split is the DOMINANT P(k) error for k < 1.53** — 3.6x C-dev's own
    discretization error at k = 0.5, 7.4x at the fundamental. The curves have
    different shapes (discretization ~ k^2 throughout; the split rises then
    PLATEAUS at ~2.5e-2 above k ~ 1.4) and both maxima land at the band edge,
    the one place the ordering has already reversed. "2.6e-2 vs 6.1e-2,
    subdominant" is therefore an artifact of comparing maxima of
    differently-shaped curves — the same trap as job 39's "headroom 2.5x".
    Clause 3 exists to keep that visible.
  - In ABSOLUTE terms the low-k split error is small (<= 1.1% at k <= 0.5, 0.36%
    at k <= 0.25) and clears every science bar in D-v2-8 (halo b1 within 2% at
    k <= 0.25 implies ~4% on P(k) there). The real consequence is narrower: the
    architecture leaves a ~1%-at-k=0.5 coherent suppression that the mesh error
    does NOT dominate, so a measured-transfer correction (the omsoc pattern)
    cannot later be used to claim sub-percent large-scale accuracy. That floor
    is the architecture's, not the mesh's.
  - Margin is thin and must be read as such: the V2a config (gauss + TSC +
    matching, tile 128 / buf 32) measures 2.61e-2 = **0.87 of the bar**. The
    lower-bound character of 6.14e-2 probably means the true margin is better,
    but that is not measured. The knob is tile size (gauss tiles as ~2.0/P;
    error falls with BIGGER tiles, which costs memory) — a V4 Pareto call.
  - The G5 verdict is NOT decided here. This ADR supplies the bar it was blocked
    on; the verdict stays JC's.
  - **OPEN, deliberately not decided here: D-v2-8 clause 5 (the multipole-grade
    RSD bar) invokes the SAME retired language** — "P0 and P2 below the
    fine-mesh PM floor". It inherits this ADR's problem verbatim. It is left
    alone because JC ratified a P(k) split bar, not an RSD one, and G2c's
    verdict does not hinge on it: t9 passed by ~3 orders (1.3e-4 vs 1.3e-1), so
    no plausible correction to that floor reopens the tier choice. If a later
    gate needs the RSD bar to be tight, it needs a D-v2-9-style absolute number
    first.
  - Record: `runs/v2/g5b_abs_transfer.md`; figures `runs/v2/g5b_cdev_tolerance.png`
    (the bar) and `runs/v2/g5_cdev_band.png` (why the old floor failed). Probe:
    `scripts/v2_g5b_abs_transfer.py` (deneb job 40).

## D-v2-10 — G5 verdict: PASS (A2's spine ratified); perf evaluation moves to the config-table homes
- **Status:** accepted (JC, 2026-07-16, seed V2a close). Resolves the verdict
  D-v2-9 left open; retires the G5 kill-line branch (A1 fallback / 4096^3 to
  A4-only) unexercised. **One clause superseded by D-v2-11** (2026-07-29): the
  low-k bullet's "a measured-transfer correction cannot currently be claimed to
  remove it" is answered, and the assessment it defers is now done. The rest of
  this entry, including the verdict, stands as written.
- **Decision:** the two-level split (gauss + TSC + matching) passes the D-v2-9
  bar — 2.61e-2 vs 3e-2 in-band at the V2a probe config (tile 128 / buf 32,
  P = 192, C-dev) — and is ratified as A2's spine. The clause-3 shape diagnostic
  was read, not just the max: the split dominates C-dev's own discretization
  error below k = 1.53 while clearing every D-v2-8 science bar.
- **The 0.87-of-bar margin is a GATE-CONFIG ARTIFACT, not a product property.**
  Three measured scalings, all from job 39, establish this:
  1. **Error ∝ 1/P** (P = the padded tile period, in fine cells). Verified two
     ways: error x P flat at 0.5-0.7 across the (T, b) scan, and the two P = 192
     partitions (T128/b32 vs T64/b64) agree to ~10%. How P splits between tile
     and buffer is error-irrelevant; the erfc buffer mechanism is absent
     (kernel study, three signatures).
  2. **Peak memory is an ABSOLUTE working set ∝ P^3, box-independent**, vs
     monolithic's O(N^3): tiled 415 MB at P = 192 (C-dev) vs 57 MB at P = 96
     (the cdev8 leg) = the P^3 law. Mono: 377 B/p where it fits (cdev8) and
     OOM at C-dev on the 6 GB card — the memory ratio only improves with box
     size, so C-dev understates the win everywhere else in the table.
     (NB the "27.3 vs 377 B/p (~14x)" pair previously quoted for C-dev is
     actually the cdev8 leg — job 39 log lines 158-161; at C-dev proper the
     comparison is "tiled 24.7 B/p vs mono does-not-fit".)
  3. **Wall overhead = the padded-volume ratio (1 + 2b/T)^3 = 3.375 at both
     measured points**: 3.54/1.03 = 3.44x at cdev8 (T64/b16) and
     32.2/9.9 = 3.23x at C-dev (T128/b32). It SHRINKS for bigger tiles; the
     efficient frontier is big tile + minimal buffer.
  Consequence: at the config-table homes a P = 384-512 tile costs 3-8 GB in
  flight (trivial on a GH200/H100 node) and buys ~2-3x more margin. Nothing
  will ever run at the gate's operating point.
- **Framing (binding on future gates): performance/Pareto evaluation happens at
  the config-table TACC homes (C-gh = one GH200, C-hero = one S3 H100 node).**
  deneb is the correctness/dev ground and a free CUDA-path check — its 6 GB
  card must not shape architecture decisions or gate configs again. Probe
  fallbacks that existed only to fit deneb (the G2c cdev8 fallback; "G5
  unblocks the G2c rerun") are contortions, deleted: the full C-dev G2c rerun
  (where t9/t12 actually separate at 2^17 levels) runs MONOLITHICALLY on a
  GH200 with no new machinery.
- **The operating (T, b) per config is V4's Pareto call, NOT fixed here** —
  informed by G5c (Vista GH200, launched at this close): cross-hardware
  replication of the C-dev split number, a box ladder at fixed cell/spacing
  (cgh64 = 512^3 / L 256) for the 1/P COEFFICIENT under more long-wavelength
  power (it moved 3-4x between the kernel-study config and C-dev — the one
  unknown that could erode the at-scale margin story), and real (B/p, wall)
  capacity triples on the C-gh hardware class.
- **Accepted with eyes open: the low-k architectural floor.** Below k ~ 1.5 the
  split is the sim's dominant P(k) error (a coherent suppression, ~1% at
  k = 0.5, plateauing ~2.5e-2 above k ~ 1.4); it shrinks as 1/P but its
  structure is the architecture's, and a measured-transfer correction cannot
  currently be claimed to remove it. Whether it is realization-stable enough
  to correct as a transfer is DELIBERATELY a separate assessment, after G5c —
  not folded into this verdict. D-v2-9 clause 3 keeps it visible per config.
- **Record:** `runs/v2/g5_kernel_findings.md` (mechanism),
  `runs/v2/g5b_abs_transfer.md` (bar), `runs/v2/cost_of_memory.md` (triples),
  `runs/v2/g5_results_cdev.json` (job 39 data incl. the tile scan).

## D-v2-11 — the low-k split error IS a correctable transfer, and it calibrates off-box
- **Status:** accepted (JC, 2026-07-29, G6 close). Supersedes ONE clause of
  D-v2-10: "a measured-transfer correction cannot currently be claimed to
  remove it". D-v2-10 is otherwise untouched and its verdict stands. This is
  the assessment D-v2-10 deliberately deferred until after G5c.
- **Context:** D-v2-10 accepted the low-k floor with eyes open and left two
  questions for a separate assessment. (a) Is the split's low-k error a
  coherent, removable WINDOW, or is it decorrelation? (b) Can the transfer
  T-bar be calibrated ONCE on a small box and applied at a production box?
  (b) is the compute question, and it decides whether the memory saving is
  real: the monolithic reference is the only memory-expensive object in the
  scheme, and it is needed only during calibration. If T-bar drifts with
  volume, calibration inherits the production box's cost and the saving is
  notional.
  Measured on a fixed-cell box ladder (cdev8 / cdev / cgh64 = 1x / 8x / 64x
  volume at fine cell 0.25, with coarse mesh and gate band held), so comparing
  their ensemble-mean transfers is pure box transport at fixed resolution:
  deneb job 188 (cdev, 16 seeds), deneb job 192 (cdev8, 16 seeds plus the 8x
  transport leg), Vista job 866415 (cgh64, 6 seeds plus both remaining legs).
  Every config's degenerate limits passed ~8 orders under tolerance (a
  full-period tile-origin shift reproducing its unshifted twin, a translated
  monolithic run reproducing the untranslated one), so the ratio machinery is
  not manufacturing the signal it is being used to measure.
- **Decision:**
  1. **The low-k split error is ratified as a CORRECTABLE TRANSFER, not an
     irreducible architectural floor.** It is a near-pure window: max(1 - r)
     at k <= 0.5 is 3e-5, i.e. phase-perfect. A leave-one-out transfer at
     C-dev cuts k <= 0.5 from 8.92e-3 to 9.90e-4 (factor 9.0) and the gate
     band from 1.97e-2 to 4.25e-3 (factor 4.6).
  2. **T-bar may be calibrated OFF-BOX, on a small ensemble.** On D-v2-9's
     gate band the three transport legs leave 1.889e-3 (1x -> 8x), 4.208e-4
     (8x -> 64x) and 2.227e-3 (1x -> 64x) against the 3e-2 bar: 1/16, 1/71
     and 1/13 of it. The low-k power-law indices match on a common window
     (deltas 0.093 / 0.081 / 0.117 against the script's 0.15 MATCH criterion),
     which is what licenses extrapolating T-bar below a small box's
     fundamental. Production therefore calibrates once, stores T-bar, and
     applies it per mock: no ensemble and no monolithic reference at the
     production box.
  3. **Re-calibration triggers.** Tile size and tile origin are measured
     stable and are NOT triggers: the low-k transfer is P-independent (cgh64
     evolved |dP/P| = 1.896e-2 for T128 = T256 = T512 alike, job 861849), and
     origin randomization costs rms 2.6e-5 in band against a 2.4e-3
     seed-to-seed scatter. Fine cell, cosmology and redshift are UNMEASURED
     and are treated as triggers until measured, with redshift bounded by
     calibrating at the output z.
  4. **The correction is reported, not gated.** D-v2-9's bar continues to be
     read on the UNCORRECTED split and clause 3 (the split-to-discretization
     ratio vs k) is unchanged. Whether a corrected P(k) ever becomes the
     gated quantity is a V4 call, not decided here.
- **Consequences:**
  - D-v2-10's low-k bullet stands except for the superseded sentence. Its 1/P
    shrinkage is the band-EDGE plateau, a different feature from the low-k
    end, which no tile size reaches: only a transfer touches it.
  - **A single-box floor survives the correction.** The per-realization
    scatter (sigma/|mean| at k <= 0.5 of 0.20 / 0.18 / 0.16 across the three
    rungs) is physical, not sampling noise: the ratio is taken between two
    runs sharing initial conditions, so cosmic variance cancels and the
    measured scatter sits 100-500x BELOW the sqrt(2/N_modes) floor. The
    ensemble MEAN is therefore cheap to pin (determined to ~3% at cdev), but
    a perfect T-bar still leaves ~1e-3 at k <= 0.5 on any single box and no
    number of calibration seeds removes it. That is the accuracy a corrected
    production mock inherits.
  - **Three statistics answer three different questions**, and reading any one
    alone gives the wrong answer: the absolute residual against the bar (the
    gate question, answered above); sigma_calib, i.e. is a transport bias
    RESOLVED at all (1.4 sigma at 8x, consistent with none; 4.7 sigma at 64x
    from a 1x box, so the drift is real, just far too small to matter); and
    the ratio to the big box's own leave-one-out, which is NOT a "cost",
    because a per-realization prediction error and a mean-to-mean bias are
    different estimands.
  - **Not licensed, deliberately.** The RESOLUTION axis is untested (all of
    this holds the fine cell at 0.25). Cosmology and redshift are unmeasured
    as triggers. Correctability was measured within ONE kernel and split
    configuration (gauss + TSC + matching). cgh64 ran 6 seeds against 16 at
    the other two rungs, so its own leave-one-out is the least well
    determined number in the record.
  - **Record:** `runs/v2/g6_split_stability.md` (every number, the degenerate
    limits, and two card-reading traps that each inverted a verdict before
    being fixed). Verdict cards `runs/v2/g6b_transport_cdev8_cdev.json`,
    `g6b_transport_cdev_cgh64.json`, `g6b_transport_cdev8_cgh64.json`
    (tracked). Probes `scripts/v2_g6_split_stability.py`,
    `scripts/v2_g6b_calib_transport.py`.

## D-v2-12 — G3 verdict: A3 fails the squeezed-B bar; A2's buffer is sized at 4 r_s
- **Status:** accepted (JC, 2026-08-06, G3 close). Discharges BOTH
  halves of the V2 exit criterion ("squeezed-B error number -> A3 verdict + A2
  buffer sizing"): clauses 1-4 kill A3, clause 5 sizes A2's buffer. Fires the
  pre-registered V2 kill line in `docs/plan-plan-v2.md` ("G3 squeezed-B > 15%
  at sane buffers -> A3 dies (A2 unaffected); publish the negative (P-C)
  regardless"). Does not touch D-v2-10 or D-v2-11: A2's spine stands, and
  clause 5 is a measurement ON that spine, not a change to it.
- **Context:** G3 asked whether sCOLA-style sequential independent tiles
  reproduce the monolithic squeezed bispectrum inside D-v2-7's 15% bar.
  Measured at cdev (L=128, n_fine=512), seed 0, deneb jobs 360/362, arms
  T=128 at b = 4/8/16 Mpc/h plus T=64/b=8, costing 1.95x / 3.38x / 8.00x /
  8.00x monolithic evolves.
  **Two of the three ratified G3 statistics were found broken during the
  readout and this verdict does not rest on either** (see clause 3). The
  numbers below are `rho_auto`, built afterwards and validated against an
  exact translation null, and the auto amplitude ratio, which involves no
  ratio between realizations at all.
- **Decision:**
  1. **A3 FAILS the 15% bar at cdev, by 2.7-6.1x.** At the well-conditioned
     cells (`k_short` = 6 k_f, where `r` = 0.70-0.92 and nothing is
     degenerate), max |`rho_auto`| over the estimand triangles is 0.656 /
     0.720 / 0.398 / 0.924 for b = 4 / 8 / 16 Mpc/h and T=64/b=8. The best
     arm is 2.7x the bar and costs 8x monolithic. **At A3's own budgeted
     compute cost of 3.2x** (plan-plan Sec. 5, the premise of the whole
     route) the nearest arm is b = 8 Mpc/h at 3.38x, which reads 0.720 --
     **4.8x the bar.**
  2. **The failure is realization-INDEPENDENT and mechanistically
     understood.** The auto amplitude ratio `A(k) = sqrt(P_t/P_m)` involves
     no cross-realization comparison. At the box fundamental it is 0.441 /
     0.424 / 0.633 / 0.394 across the four arms, i.e. **the tiled field
     carries 15-40% of the monolithic LONG-MODE POWER**. The squeezed
     bispectrum is by construction the coupling of small-scale power to that
     long mode, so no estimand and no amount of ensemble averaging repairs a
     systematic deficit of this size. Dividing the deficit out explicitly is
     what `rho_auto` does, and 40-92% error survives it.
  3. **`R_Q` and `rho` are WITHDRAWN as G3 statistics.** `R_Q` normalizes by
     each arm's OWN power, so the power deficit enters `Q`'s denominator with
     the opposite sign to the bispectrum deficit and `R_Q` flips positive on
     the very triangles the tile handles worst (17 of 19 positives at
     `x <= 1`); on the smoke it also fails to rank the kill control at EVERY
     `k_short`. `rho` divides by the CROSS transfer `T = r * A`, hence by the
     correlation, and was measured to track `1/r^2 - 1` across sixteen cells
     within 10-20%, reaching 3.6e4 where `r` crosses zero. Any further G3
     reading uses `rho_auto`.
  4. **The realization-matched estimand is retired as the default for
     tiled-vs-monolithic comparison** (JC, 2026-08-06: it was adopted for
     interpretability, not statistical necessity). In the decorrelated
     regime -- which is the tiling's normal operating regime -- a shared-IC
     ratio compares two different small-scale realizations and stops being
     the quantity of interest. Future comparisons use ensemble/statistical
     agreement. This clause does NOT weaken clause 1, which clause 2 makes
     independent of the framing.
  5. **A2's BUFFER IS SIZED AT 4 r_s, and G3's second deliverable is
     discharged.** The V2 exit criterion reads "squeezed-B error number ->
     A3 verdict + A2 buffer sizing"; clauses 1-4 answer the first half and
     this answers the second. Measured at cdev, seed 0, deneb job 364, on the
     synchronous two-level force (global coarse long range + `force_short_tiled`
     on fine tiles, i.e. `force_two_level`, the D-v2-10 spine) with the fine
     level TILED at T=128:
     - `max|rho_auto|` = **0.0051 against the 0.15 bar, a 29x margin**, at
       both b = 4 `r_s` and b = 8 `r_s`. The two are identical to four
       decimals, so **4 `r_s` is ample and 8 buys nothing.**
     - Auto amplitude 1.000 -> 0.994 over k = 1..12 k_f and **r = 1.000 at
       every shell**, against the failing arm's 0.441 and 0.368.
     - Cost 22.6 s/step against the monolithic 9.9 s/step recorded in
       `runs/v2/cost_of_memory.md` = **2.3x**, inside A3's 3.2x premise and
       below the 3.23x the ratified two-level config was measured at.
     **Why the realization-matched reading is admissible HERE** even though
     clause 4 retires it as the default: it fails in the DECORRELATED regime,
     and these arms are correlated at r = 1.000 at every shell. There is no
     decorrelation for the shared-IC ratio to misreport. The same margin read
     on `R_B` is 0.0102, i.e. 15x under the bar, so the conclusion does not
     hinge on the choice of statistic either.
     **Scope:** one seed at cdev. The margin is 29x against a 12-22% seed
     scatter on related quantities, so the VERDICT is safe at one seed; no
     individual digit here is an ensemble statement, and this is not a cgh64
     number.
- **Consequences:**
  - **A3 dies as a squeezed-B mock route. A2 is unaffected** and D-v2-10 /
    D-v2-11 stand untouched. The negative is publishable (P-C) and is
    strengthened, not weakened, by the mechanism in clause 2.
  - **THE ANOMALY IS RESOLVED, and it is not a bug** (2026-08-06, probe
    `scripts/v2_g3_tile_force_scale.py`). Pure 2LPT -- the background the
    tiles sit on -- has `A(k_f)` = 0.980 while the tiled arm has 0.394, which
    looked like the tiling destroying large-scale power its own background
    had right. The cause is structural. A tile's force comes from
    `force_global` on a PERIODIC PADDED BOX, which supports no mode longer
    than that box, while the analytic frame terms in `cola_step_bullfrog`
    (`c_k1*psi1 + c_k2*psi2`) keep their GLOBAL long-wavelength content. COLA
    works because those two very nearly cancel at large scales; here the force
    side of the cancellation is simply absent, so the residual grows a
    spurious large-scale piece that fights the background displacement.
    Measured directly on the force, one evaluation per arm and no evolution:
    the tile force loses amplitude specifically BELOW the padded tile's own
    fundamental `k_P = 2*pi/(P*cell)` and recovers above it, with the knee
    tracking `k_P` as it moves (smoke, `k_P` = 2.00 vs 1.33 k_f). The
    production-box density shows the same signature across all four arms at
    three different `k_P`, from data already on disk: amplitude below `k_P` of
    0.441 / 0.424 / 0.633 / 0.394 rising to 0.920 / 0.903 / 0.929 / 0.839 at
    the first shell above it, for `k_P` = 3.20 / 2.67 / 2.00 / 4.00 k_f.
  - **Consequence of that resolution: the controlling variable is the PADDED
    TILE SIZE relative to the box, not the buffer as such.** Getting
    box-scale modes right requires the padded tile to approach the box, i.e.
    not to tile. This is why the buffer ladder never converges toward 1 and
    why the literature's larger buffer would not fix it either -- more buffer
    moves `k_P` down slowly and at cubic cost.
  - **The structural fix is identified but UNTESTED, and it is the one thing
    worth trying before A3 is closed for good.** The missing piece is exactly
    the long-range force, and this project already has validated machinery
    that supplies it: the two-level split ratified in D-v2-10/D-v2-11, a
    global coarse solve plus a local fine solve. A tiled arm that took its
    long-wavelength force from the global coarse mesh instead of from the
    tile's own periodization would have no reason to show this deficit. That
    is a HYPOTHESIS with a clear mechanism, not a measured result; it is not
    the configuration tested here, and clause 1 stands for the configuration
    that was.
  - **The buffer the literature specifies was never tested, deliberately.**
    Leclercq (2003.04925) gives buffer >= 25 Mpc/h; every arm here is at or
    below 16. At cdev geometry b = 25 Mpc/h costs 16.8x and b = 32 costs
    27.0x, against A3's 3.2x budget -- so the buffer that might rescue the
    physics removes the reason to want A3 at all. That is what makes "at sane
    buffers" in the kill line satisfied. It is a cost argument, not a physics
    one, and clause 1 should be read that way.
  - **Cancelled as verdict inputs:** the cgh64 bispectrum ensemble (~84 Vista
    node-hours) and Stage 7 A3 capacity work. cgh64 is worse on the
    controlling variable (`k(r=0.5)` = 0.330 at 8 Mpc/h vs cdev's 0.472), so
    it would refine a number already several times over the bar in the
    direction of failing harder.
  - **Not licensed.** ONE SEED at ONE box. The margin is 2.7-6.1x against a
    12-22% seed scatter on related quantities, so the VERDICT is safe at one
    seed, but no individual number here is an ensemble statement. Nothing
    here bears on tiled mocks for P(k)-only use, on non-squeezed statistics,
    or on any resolution other than the 0.25 Mpc/h fine cell.
  - **Record:** `runs/v2/g3_stage5_record.md` secs. 1-10 (sec. 8 the
    diagnosis, sec. 9 the `rho_auto` measurement, sec. 10 the bracket-scan
    pre-registration). Cards `g3_stage5_cdev.json`,
    `g3_floors_cdev_flatw.json`. Probes `scripts/v2_g3_stage5.py`,
    `v2_g3_stage5_readout.py`, `v2_g3_card_repro.py`,
    `tests/test_auto_transfer.py`.

## D-v2-13 — V3 verdict: host-resident state streams; the GH200 ceiling is ~116 GB; `staged` is the production path

- **Status:** accepted (JC, 2026-08-06, V3/G4 close). Clauses 1, 2, 4, 5
  ratified as drafted; clause 3 was challenged, revised on re-examined
  evidence, and re-ratified. Discharges both halves of the V3 exit criterion
  ("the A2-on-GH claim gets its measured footing" + "cost-of-memory gains the
  GH points"). Does not touch D-v2-8 through D-v2-12: the state tier, the
  fidelity bar, A2's spine, the transfer correction and G3's verdict all
  stand. Next gate is V4 (architecture freeze).
- **Context:** C-gh is 2048^3 particles, and at the D-v2-8 T9 tier that is
  77.3 GB of state alone — 81% of the GH200's 96 GB HBM before a mesh
  exists. So C-gh is real only if host-resident state is reachable at useful
  bandwidth. Measured on Vista `gh` (jobs 894036, 894118, 894166) and, for
  the fabric comparison, Stampede3 `h100` (jobs 3380722, 3380888), with one
  estimator and matched rungs. Full record `runs/v2/g4_record.md`;
  triples in `runs/v2/cost_of_memory.md` §V3.
- **Decision:**
  1. **V3 PASSES, with the exit criterion amended to name the regime
     actually measured.** The seed asked "coherent (ATS) vs staged vs
     infeasible". That trichotomy is **not expressible on this stack**: jax
     0.10.2 + CUDA aarch64 exposes only `['device', 'pinned_host']`, so
     pageable LPDDR is not addressable through JAX memory kinds at all. The
     measured answer is a fourth regime — **state larger than HBM streams out
     of page-locked host memory at 359-367 GB/s with 4.00 GiB of device
     residency, independent of working-set size**, and C-gh's paint runs on
     one GH200 (2048^3, 13.29 s, 6.46e8 particles/s).
  2. **The GH200 host ceiling is ~116 GB (108 GiB) and is a HARD CLIFF that
     binds config selection.** Full rate at 116.0 GB, ~100x collapse at
     120.3 GB — no graceful region. The mechanism is physical LPDDR capacity,
     established two ways: a bracket that straddles it to within 4.3 GB, and
     an S3 h100 node running the identical 128 GiB rung at full rate on a
     1007 GiB host. A C-gh operating point must therefore be specified CLEAR
     of the ceiling, not near it — the same failure mode as D-v2-10's
     0.87-of-bar margin. C-gh's T9 state is 77.3 GB, ~1.5x under.
  3. **`staged` (explicit `device_put` per chunk) is the production path, on
     measured performance at the config-table home per D-v2-10.** On the REAL
     C-gh paint on the GH200, `staged` is **2.78x faster** than the
     XLA-managed arm (13.29 s vs 36.94 s). **The ladder's ordering INVERTS**:
     on a bandwidth-bound streaming reduction the XLA-managed arm is 14-20%
     faster, and that advantage neither transfers nor survives sign on real
     work. The XLA-managed path is recorded not as a rejected alternative but
     as a **memory-vs-wall knob** — it uses half the device memory (16.00 vs
     31.33 GiB) for 2.78x the wall, worth taking deliberately if device
     residency ever binds. **The first draft of this clause argued
     portability and conceded the 14-20%; both were wrong** (JC: production
     goes to Vista and ultimately Horizon, both C2C-class, with S3 h100 a
     fallback, so a portability trade optimizes for the wrong machine). Do
     not reintroduce either argument.
  4. **The word "coherent" must not be used for this capability.** The seed
     and the design study use it to mean ATS access to pageable memory.
     Nothing here tests that. Say "host-resident streaming via `pinned_host`";
     record ATS as untested and not expressible through jax memory kinds on
     jax 0.10.2.
  5. **C-hero is viable on memory grounds, and the fabric ratio must not be
     used to size it.** One S3 h100 streams 4096^3's full T9 state — 618.5 GB
     = 576 GiB — at full rate, flat across a 9x span of working set, on a
     1007 GiB host. It costs **1.80x** the GH200's wall on the real C-gh
     paint. The C2C-vs-PCIe FABRIC ratio is 6.71-6.87x and **does not
     propagate**: paint is compute- and latency-bound, running 46x below its
     own machine's ladder rate. **1.80x is the transferable figure; 6.9x is
     not a wall ratio.**
- **Consequences and limits:**
  - **The step-level number is unmeasured at every config-table home.** The
    tiled two-level force has only ever run at `cgh64` (512^3 particles =
    C-gh at 1/64 volume). C-gh is 64x that and C-hero 512x, with tile counts
    going 512 -> 32,768 -> 262,144. Nothing here licenses a wall/step claim
    at C-gh or C-hero, and V4's node ladder needs one.
  - **No monolithic reference exists at hero scale**, so a hero-scale run can
    measure COST only; accuracy has to come from D-v2-11's off-box transport.
    That is a scoping decision V4 must take explicitly.
  - **Not licensed:** anything about multi-GPU (inexor has none — `evolve` is
    an eager single-device orchestrator), any operating (T, b) for C-gh
    (reserved to V4 by D-v2-10), and any claim that the GB200 route is
    hardware-risky. GB200 fp64 is **not** a differentiator — it is
    essentially Hopper GPU-for-GPU with ~2x the per-card HBM (JC, from run
    history); the earlier "gb is slower for fp64" note came from a single
    uncontrolled workload row and is retracted.
  - **Provenance note:** job 894010 is VOID and is recorded as provenance,
    not data — it measured JAX's environment defaults
    (`XLA_PJRT_GPU_HOST_MEMORY_LIMIT_GB` = 64 GB, and a 0.75 device fraction)
    rather than the hardware. The probe now reads both caps back off the
    device and declares the capacity witness void if they do not clear the
    ladder.
  - **Record:** `runs/v2/g4_record.md` (§1-4 pre-registration written before
    any rung existed; §5.1-5.8 results; §6 the ratified clauses).
    `runs/v2/cost_of_memory.md` §V3. Probes `scripts/v2_g4_gh_memory.py`,
    `scripts/v2_g4b_fp64_capability.py`. Cards `g4_gh_memory{,_smoke,_bracket,
    _s3,_s3real,_s3hero}.json`, `g4b_capability{_gh,_s3}.json`.

---

# V4 architecture freeze -- RATIFIED 2026-08-08

D-v2-14 through D-v2-18 below were ACCEPTED by JC on 2026-08-08, unamended.
They rest on `runs/v2/v4_architecture_record.md` (jobs 896159, 896160, 896408)
and `runs/v2/v4_pricing_record.md`. Every clause either cites a measured number
or labels itself an estimate; the estimates stay labelled after ratification,
and D-v2-14 clause 2's slack term is an M-v2-1 exit condition precisely because
ratifying it did not measure it.

## D-v2-14 -- State architecture: T9 at the implementable quantum, on a brick-sorted layout

- **Status:** accepted (JC, 2026-08-08, V4 freeze; drafted 2026-08-07). Amends
  D-v2-8's state-tier clause; does not touch its requirements chain.
  **Clause 3 SUPERSEDED by D-v2-19 (JC, 2026-08-08)** -- both of its
  load-bearing assertions measured false. Clauses 1, 2, 4 and 5 stand; clause
  2's slack figure is re-derived there. **Clause 2's INDEX term (uint16,
  0.25 B/p) superseded by D-v2-20 (JC, 2026-08-08)**: the index is uint32 at
  0.50 B/p and the all-in figure is ~10.54. Left byte-frozen otherwise, per the
  D-v2-10 -> D-v2-11 precedent.
- **Context:** D-v2-8 ratified T9 = 9 B/p on G2c's `t9` arm. That gate says in
  its own docstring (`v2_g2c_accum_gate.py:35`) that it "measures
  REPRESENTATION error only (storage layout is a build decision)". V4 takes
  that decision, and the arithmetic does not work as assumed: the gated arm
  quantizes at `fine_cell/256`, which an int8 carries only if its bucket is one
  fine cell, and at C-gh a per-fine-cell index costs ~69 GB against the 77 GB
  of state it indexes. CUBE's sorted-by-cell layout works because CUBE runs one
  particle per cell; our fine mesh is 2x the particle grid per side.
- **Decision:**
  1. **The position bucket is 1.0 Mpc/h (two particle cells), quantum
     `fine_cell/64`.** Measured at cdev, K=40, job 896160: max |dP/P| 4.123e-4
     and worst-case dP2/P0 2.035e-3, against D-v2-9's absolute 3e-2 -- a 15x
     margin, and ~2.5 orders under both the step and mesh floors. The `t9`
     control reproduced its ratified 1.3e-4 exactly.
     **Chosen on margin plus a 2.15 GB index, NOT on a measured ordering:** the
     ladder is non-monotonic at these levels (`t9c1` beats `t9` on dP/P,
     `t9c4` beats `t9c2` on dP2/P0), so all four arms passing is what is
     established, not that c=2 is the best of them. One seed, one config.
  2. **The all-in state cost is 10.15 B/p, not 9.** Payload 9.00, bucket index
     0.25, brick CSR 0.004, slack and arena 0.90. At C-gh that is 87.2 GB,
     1.33x under the ~116 GB cliff rather than 1.50x. D-v2-8's capacity column
     is re-derived with this figure, which is what D-v2-8 said V4 would do.
     **The 0.90 slack term is an ESTIMATE**, from a hand argument about
     migration rates; it is 7.7 GB of the 87.2 and its measurement is an exit
     condition of the codec milestone.
  3. **The canonical layout is brick-sorted**, so a tile's members are
     contiguous spans. `cap` and the slack absorbing inter-brick migration are
     the same buffer, and capacity is per-bucket rather than a global max over
     tiles. A full re-sort is not affordable (a second 77 GB scatter target is
     over the ceiling); the operation is eject-and-reinsert. Bucket overflow
     escalates slack -> arena -> loud refusal and **may never clamp** (D-007).
  4. **The layout is admissible only because the paint is order-independent**
     (D-006). Brick-sorting reorders particles every step, so any
     order-dependent accumulation on the primal path silently becomes
     irreproducible. This makes clause 2 of D-v2-16 a precondition, not a
     nicety.
  5. **Particle IDs are an opt-in tier**, int32 at +4 B/p, available where it
     fits (n_side <= 1290) and refused above. Production C-gh carries none.
     Expressed as a seventh column on a struct-of-arrays state so physics
     kernels never see it and only layout kernels vary.
- **Record:** `runs/v2/v4_architecture_record.md` section 4; job 896160;
  arms `t9c1`/`t9c2`/`t9c4` in `scripts/v2_g2c_accum_gate.py`.

## D-v2-15 -- IC architecture: disk staging, a 1D transfer table, and an out-of-core FFT

- **Status:** accepted (JC, 2026-08-08, V4 freeze; drafted 2026-08-07).
- **Context:** the only IC memory number on record (76 B/p, v1 R6) is device
  only. Nothing had ever measured the host side, and IC generation was
  sidestepped by every gate loading from an offline npz.
- **Decision:**
  1. **The IC host term is real, ~90 B/p, and FLAT in N** (126.5 / 95.4 / 89.6
     at n = 256 / 512 / 1024, job 896159). It extrapolates to **~773 GB at
     2048^3 against a 116 GB ceiling** and is the binding IC constraint. My
     pre-registration predicted growth with N and was wrong about the shape;
     a fixed number of half-grid arrays gives constant B/p.
  2. **The colour and transfer evaluation moves to a 1D log-spaced |k| table
     with interpolation.** P(k) is a function of |k| only. Both halves already
     exist in the tree (`cosmology.py:207` uses a 4000-point log grid;
     `linear_power(backend="table")` already log-log interpolates). This
     removes essentially all of clause 1's term.
  3. **Disk is an explicit staging tier for the IC stage only.** The generator
     emits T9-encoded slabs; `evolve` streams them back into `pinned_host`
     chunks. The evolve loop does NOT spill: state fits host at C-gh
     (87.2/116 GB) and C-hero (~697/1007 GiB).
  4. **An out-of-core FFT is REQUIRED at C-gh and cannot defer to C-hero.**
     The largest monolithic `jnp.fft.rfftn` on a GH200 is between 1024^3 and
     1536^3 (job 896159 leg 7: 1024^3 peaks at 28.04 GB for a 4.00 GB field,
     1536^3 OOMs on a 27.04 GiB allocation). **The workspace ratio is 7x the
     field** and is the number the slab design must budget against.
  5. **Reproducibility is defined against a decomposition, not across one.**
     `jax.random.normal`'s stream is shape-dependent, so a slab-decomposed
     white-noise field is not bit-identical to a monolithic one. The canonical
     noise unit is one XY plane keyed by `fold_in`, which makes the field
     invariant to slab thickness. It is NOT resolution-independent; fixed-phase
     cross-resolution comparison still requires equal N.
- **Record:** `runs/v2/v4_architecture_record.md` sections 5 and 6.

## D-v2-16 -- Force architecture freeze

- **Status:** accepted (JC, 2026-08-08, V4 freeze; drafted 2026-08-07).
  Discharges D-v2-10's reservation of the operating (T, b) to V4.
- **Decision:**
  1. **The force is never materialized globally.** Its only consumer is the
     integrator's elementwise kick and ownership is a partition, so the kick
     applies tile-locally on device and velocities are written back through the
     layout. This deletes both O(N) f64 host arrays -- the 2 x 206 GB that have
     kept C-gh unrunnable. A global `sink="accumulate"` path is retained for
     tests only, with a size refusal.
  2. **`paint_tsc_int` is a required deliverable.** `paint_tsc_f64` accumulates
     through order-dependent f64 `.at[].add`, and `--assign-long` defaults to
     `tsc`, so **the coarse arm ratified in D-v2-10 violates D-006 today**. No
     existing document names this. It is also a precondition for D-v2-14
     clause 3.
  3. **The long-range force is staged as a per-tile coarse sub-block**, not
     held resident. Resident is 12.9 GB at C-gh but 103 GB at C-hero against
     96 GB of HBM. Sub-block staging makes the whole force path O(tile) in
     device memory and removes the C-hero cliff by construction. (Derived, not
     measured.)
  4. **Geometry: T=256, b=32 for C-gh**, chosen on `cap` per
     `v4_pricing_record.md` section 7 and now supported by a second,
     independent argument -- gather saturates near 18 GB/s only at >= 256 KB
     runs, which would need b ~ 64, costing 1.7x padded volume for 1.5x gather.
     Roughly a wash. **This remains PROVISIONAL until measured at C-gh**; the
     criterion is frozen, the number is not.
  5. **`cap` is confirmed as the cost variable, with nothing co-varying**
     (job 896159 legs 3-4): `stage` is linear in `cap` (1.487 at x1.5, 1.973 at
     x2.0) and device goes as ~`cap^0.5`. The fixed-`n_brick` isolation owed by
     `v4_pricing_record.md` is discharged by a cleaner instrument and is struck
     from the owed list.
  6. **The streaming constraint is the HOST GATHER, not the transfer.** Pinned
     H2D is 176-221 GB/s and flat in run length; the scattered-run gather at a
     37 KB brick span is 12.2 GB/s (job 896408). Worth 6.9x on `stage` at f64
     and 18x at T9 -- **contingent on gathering directly into
     `cudaHostAlloc`-backed memory, which jax cannot express and which is
     therefore an ffi-level build item.**
  7. **Promotion out of `scripts/v2_g5_core.py` is gated on bitwise parity**
     against the retained probe at three geometries, with the probe kept
     unmodified as the reference oracle. D-v2-10, D-v2-11 and D-v2-12 are all
     measurements OF that code; if the promoted engine is not numerically
     identical, three ratified records quietly stop describing the shipped
     artifact.
- **Record:** `runs/v2/v4_architecture_record.md` sections 1, 2, 3.

## D-v2-17 -- Gating scope: the uncorrected split stays gated; C-hero is capacity-only

- **Status:** accepted (JC, 2026-08-08, V4 freeze; drafted 2026-08-07).
  Discharges D-v2-11 clause 4 and D-v2-13's hero-scoping item.
- **Decision:**
  1. **D-v2-9's bar continues to be read on the UNCORRECTED split.**
     D-v2-11's transfer stays reported and is applied in production, but
     gating the corrected quantity would let a calibration absorb an
     architecture error, and the transfer's own irreducible residual (~1e-3 at
     low k on any single box) would move inside the gate rather than beside it.
  2. **C-hero is a capacity demonstration only.** No monolithic reference can
     exist at hero scale, so hero runs measure cost, memory and completion;
     accuracy is inherited via D-v2-11's off-box transport, which is what that
     transport was ratified for. Any hero data product ships with that caveat
     attached.

## D-v2-18 -- Build roadmap

- **Status:** accepted (JC, 2026-08-08, V4 freeze; drafted 2026-08-07). Replaces
  the M-v2-1..4 placeholders in `docs/plan-plan-v2.md` §5, applied 2026-08-08.
- **Decision:** the codec moves first, because the layout is defined in terms
  of it and the IC stage emits it.

  | id | scope | exit gate |
  |---|---|---|
  | M-v2-1 | codec + layout: T9 pack/unpack, brick-sorted state, per-bucket capacity, incremental exchange, opt-in ID tier | exact round-trip; wrap-never-clamp asserted; **measured** migrant distribution and slack at cgh64 (D-v2-14 clause 2) |
  | M-v2-2 | harden and promote the two-level force; `paint_tsc_int`; the ffi pinned gather | bitwise parity vs the probe at 3 geometries; runtime invariants become tests; regression tests for the three measured bugs |
  | M-v2-3 | engine core on T9 state | correctness vs the v1 parity arms where configs overlap |
  | M-v2-4 | f32 force mesh | thread `fdtype`, re-run `v2_g3_floors.py` unchanged, read against the MESH FLOOR not zero; own gate, cannot ride on D-v2-9's or G6's |
  | M-v2-5 | streamed ICs + out-of-core FFT | tile-IC identity vs monolithic at f64; the transfer table's error < 1e-4 |
  | M-v2-6 | capacity | **a complete 2048^3 mock on one Vista gh node** |
  | M-v2-7 | output stage | HMF within 5%, halo b1 within 2% at k <= 0.25; squeezed B <= 15%; disco-mocks read-back |

  **Crosswalk, mandatory.** The charter's IDs are referenced elsewhere and must
  not be silently renumbered: old M-v2-1 engine core -> new M-v2-3; old M-v2-2
  tiles+ICs -> new M-v2-5; old M-v2-3 codec+capacity -> split across new
  M-v2-1 and M-v2-6; old M-v2-4 output -> new M-v2-7. In particular
  `runs/v2/cost_of_memory.md:77` defers the Pallas `atomic_add` decision to
  "M-v2-1", which under this ladder is the codec; that pointer must be re-aimed
  at M-v2-2.

  **Not scheduled, deliberately:** PP-in-tiles (own gate, never a default);
  reviving A3 via the frozen-background arm (shelved); an absolute RSD bar,
  which D-v2-8 clause 5 still needs before anything leans on it.

## D-v2-19 -- State layout: spare pooled per brick, and the re-sort is affordable

- **Status:** accepted (JC, 2026-08-08). Supersedes D-v2-14 clause 3 and
  re-derives its clause 2 slack figure. D-v2-14's other clauses stand: the
  1.0 Mpc/h bucket, the `fine_cell/64` quantum, the opt-in int32 id tier, and
  the requirement that the paint be order-independent are all untouched.
  **Clause 5's index term (0.250) and the uint16 deliverable in "what this does
  not establish" are SUPERSEDED by D-v2-20 (JC, 2026-08-08)**: the index is
  uint32 at 0.500 and the all-in figure is ~10.54, 1.28x under the cliff. Every
  other term of clause 5 stands.
- **Context:** clause 3 ratified per-bucket capacity with eject-and-reinsert,
  and ruled out periodic re-sorting because "a second 77 GB scatter target is
  over the ceiling". Building it measured both halves false, and measured a
  term the accounting had missed. Records: `runs/v2/m1_layout_record.md`,
  `runs/v2/m1_brick_packed_record.md`; probe `scripts/v2_m1_migration.py`;
  cdev8 under the ratified two-level force, whose ICs and force were checked
  bitwise against the G5 driver before anything was concluded from them.
- **Decision:**
  1. **Spare is pooled per BRICK; buckets are packed tight inside it.** A
     bucket cannot be given a fraction of a slot, so per-bucket spare costs one
     whole slot per occupied bucket -- **12.5% of payload at ~8 particles per
     bucket, whatever the setting**, since `ceil()` already returns >= 1. (The
     `min_spare` floor that appeared to cause this is INERT; the cause is
     granularity.) Pooled over a brick's ~4096 particles, 10% means 10%:
     measured slots hold at exactly 1.100x N.
  2. **Bucket slot boundaries are DERIVED, not stored** -- a prefix sum of
     `occupancy` within a brick. The stored int64 boundary per bucket was
     8.6 GB at C-gh, **a full 1.00 B/p that D-v2-14 clause 2's table never
     counted**. 1.00 -> 0.002 B/p.
  3. **Capacity is redistributed by a periodic in-place repack.** Frozen
     capacity fails at EVERY granularity -- per-bucket it ran away to 62% of
     particles in the arena and climbing; per-brick a collapsing halo outgrew
     even 50% spare by step 6. But a repack **is not a sort**: `migrate` keeps
     every particle in the right bucket and bucket order is a fixed spatial
     ordering, so restoring the layout is a MONOTONE rearrangement -- two
     in-place passes with **O(chunk) scratch, measured at 0.13-0.52 MB
     independent of N**. Clause 3's "second 77 GB scatter target" does not
     apply. Cost 27 ms/step against a 4 s force step.
  4. **A small arena is still required, and may never clamp (D-007).** With
     repack every step a fixed brick fraction still overflows, by MORE as the
     fraction rises (37 particles at 10%, 248 at 15%, 461 at 20%) because more
     spare reaches heavier clustering before failing. 461 of 2.1e6 is 0.02%: a
     rare-event problem, absorbed by a 1-2% arena for ~0.05 B/p. **Peak use
     measured 0.57%**, against 22% for the per-bucket design. An arena particle
     still BELONGS to its brick and `brick_members` returns it -- omitting it
     deleted it from the force with nothing raising, measured at 98.4% loss on
     a stress fixture and 0.57% at the operating point.
  5. **The all-in figure, re-derived:** payload 9.000 + index 0.250 + brick
     boundaries 0.002 + slack 0.901 + arena 0.051 = **10.204 B/p**, against
     clause 2's 10.15 (+0.5%). At C-gh **87.7 GB against the ~116 GB cliff,
     1.32x under** -- the headroom clause 2 claimed. The per-bucket layout's
     best measured setting is ~13.4 B/p, 1.09x under, once its boundary array
     is counted. Layout overhead 6.0% of the force.
- **What this does NOT establish.** One config at **1/4096 of C-gh's volume**,
  one seed, CPU, state resident. The uint16 per-bucket index has **11.0x
  headroom** at cdev8's peak bucket population of 5943 and larger volumes hold
  rarer, denser peaks -- whether C-gh stays under 65535 is unmeasured and is a
  named deliverable of the capacity run. Whether the 6.0% overhead survives a
  streaming state is untested. `repack` sorts by key when the arena is
  non-empty, O(N log N), which wants a merge at C-gh since only the few arena
  residents are out of order.
- **Consequence for D-v2-14 clause 2:** its capacity column is re-derived at
  10.204 B/p. The 0.90 slack estimate it labelled as an estimate turns out to
  be very nearly right FOR THE POOLED DESIGN (0.901) and unreachable for the
  per-bucket one it was written about, where granularity forces >= 1.125.

## D-v2-20 -- The per-bucket index is uint32: the ceiling is removed, not measured

- **Status:** accepted (JC, 2026-08-08). Supersedes the **index dtype and its
  0.25 B/p cost** in D-v2-14 clause 2 and D-v2-19 clause 5, and **discharges
  the uint16 deliverable** D-v2-19's "what this does not establish" assigned to
  the capacity run. Every other term of both clauses stands. Ratified on JC's
  "moving the ratified number is OK", in the session that built it.
- **Context:** D-v2-19 left open whether a uint16 index (ceiling 65535) survives
  at C-gh, and named a cgh64 re-run with tail counting (~1.7 node-hours) as the
  way to settle it. It cannot be settled that way at acceptable cost. Bucket
  occupancy is volume-INVARIANT in its body (p99 121/123/122, p99.9 704/688/706
  across 64x) while only the peak grows (5943/7581/13774), so the question is
  entirely about the far tail -- and the one attempt to extrapolate that tail
  **failed its own validation by 10x**, predicting cgh64's peak at ~135700
  against 13774 measured, because the fit is dominated by well-populated low
  thresholds while the real tail falls far faster. Three points were never going
  to establish a bound two rungs away.
- **Decision:**
  1. **The per-bucket occupancy index is uint32, 0.50 B/p** (4.29 GB at C-gh
     against uint16's 2.15). The ceiling stops being a proposition anything has
     to establish.
  2. **The all-in figure is ~10.54 B/p**, from D-v2-19 clause 5's 10.204 at
     cdev8 and 10.292 at cgh64. At C-gh that is ~91 GB against the ~116 GB
     cliff, **1.28x under** rather than 1.32x. This is what a run costs; the
     0.25 buys removal of an unbounded risk out of a margin that has it.
  3. **uint16 remains constructible**, via `BrickPackedLayout.build(
     index_dtype=)` and the probe's `--index-dtype`, so the ratified 0.25 B/p
     figure stays reproducible rather than deleted.
  4. **Every write to the index is guarded, and the guard lives at the cast.**
     `_to_index` is the single narrowing path for `build`, `migrate` and
     `repack`. This fixes a defect as much as it implements a decision: before
     it, only `build` was guarded, and `migrate` and `repack` -- the two that
     run every step -- narrowed bare. numpy narrows modularly, so an occupancy
     of 65536 stored as 0 would not merely misreport one bucket; occupancy IS
     the derived bucket-boundary prefix sum, so it relocates the span of every
     later bucket in that brick, and `check()` samples three bricks.
     **`migrate` and `repack` do not see the same number** -- `migrate` counts
     the brick runs net of arena spills, `repack` pulls the arena back in and
     counts everything -- so a bucket can pass one and fail the other, and
     guarding `migrate` alone would not have covered it.
- **The int32 `key` ceiling is REFUSED, deliberately not widened.** The same
  silent-wrap class sits in `key` (the brick-major bucket ordinal) one config
  rung away: C-hero's 2048^3 = 8.59e9 buckets against int32's 2.15e9 would
  narrow the high buckets to negative ordinals. Widening is the wrong fix,
  because `key` cannot be resident at production scale in EITHER width: 34.4 GB
  at C-gh as int32, 68.7 as int64, against ~91 GB of state on a ~116 GB host --
  and the same is true of `particle_to_slot` (68.7 GB) and `slot_to_particle`
  (75.6 GB). All three are scaffolding for a probe that keeps positions in their
  original order and indexes into them; the streamed engine stores state IN slot
  order, where a particle's bucket is implied by where it sits and none of the
  three exists. A build-time refusal makes the ceiling loud today; removing it
  belongs to M-v2-3/M-v2-6.
- **~21 B/p of scaffolding is now REPORTED, beside the all-in total and not
  inside it.** Those three arrays are twice the state budget and appeared in no
  accounting -- the same shape as the 1.00 B/p `bucket_start` term D-v2-19
  clause 2 found uncounted. Believed correct to exclude, for the reason above;
  reported so the exclusion is visible rather than inferred.
- **What this does NOT establish.** Nothing about the occupancy distribution:
  the peak at C-gh is still unmeasured and this decision is precisely a
  statement that it need not be measured. uint32's own ceiling (4.29e9) is not
  argued to be unreachable from data either -- it is 3.1e5 times the largest
  peak observed, on a quantity whose body does not grow with volume at all.
- **Record:** `src/inexor/layout.py` (`_to_index`, `_refuse_key_overflow`),
  `tests/test_brick_packed.py` (one test per write path), probe
  `scripts/v2_m1_migration.py --index-dtype`; tail measurements in
  `runs/v2/m1_brick_packed_record.md`.
