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
