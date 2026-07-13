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
