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
- **Status:** proposed (design pass, 2026-07-10); confirm via R3 measurements.
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
