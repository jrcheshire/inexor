# inexor roadmap (master plan)

Convention: this is the master plan; **each milestone gets its own detailed
plan drawn up in plan mode at the session that opens it** (sfbfs/xphot
pattern). This file sets scope, entry criteria, validation gates, and exit
artifacts; milestone plans set file-level steps. Tolerances marked "measure
first" are set with JC after the floor is measured — never assumed, never
relaxed silently.

Current status: **M1 DONE** (2026-07-13; forward PM validated, results in
docs/m1-results.md, gates D-013/D-014 in decisions.md). M0 PASSED — GO (gate
review 2026-07-13; D-010..D-012). Next: M2 (the adjoint), opening with its own
detailed milestone plan.

---

## M0 — kill-or-confirm (HARD GATE)

Purpose: validate or kill the two load-bearing bets — bit-exact replay under
XLA, and useful STE gradients through quantized dynamics — before any real
implementation investment. Everything is a standalone probe script
(`scripts/m0_r*.py`), hours-to-a-day each, throwaway-quality code allowed.

Entry: this bootstrap. Compute: R1/R3 need a CUDA device (Vista, or albireo
once its GPU is installed); CPU-local covers logic but not atomics/fusion.

| # | experiment | question | kill/pivot trigger |
|---|---|---|---|
| R1 | 64^3 CUDA, jit fwd K=10 + rev K=10, assert exact int equality with initial state; 100 seeds; scan driver AND per-step-jit driver | does bit-exact replay survive XLA fusion? | fails in both drivers -> the design premise dies (unlikely — mechanism is JANUS-proven; failure would be an XLA bug to isolate). Fails scan-only -> per-step-jit becomes the production driver (also decides Sec. 9 ceiling (1)) |
| R2 | float64 reference BullFrog at 128^3; simulate exact w-frame quantization; final P(k) + positions vs K = 5..15 | does the alpha ladder's ~5-6-bit budget leave acceptable velocity resolution? | unacceptable noise -> FastPM-additive integrator promoted, or remainder ledger built |
| R3 | paint 1e8 random particles twice on CUDA via int32-mesh scatter-add; bit-compare; bench vs f32 atomics and vs --xla_gpu_deterministic_ops | is the int-accumulation paint deterministic AND fast in XLA? | nondeterministic (would mean XLA lowers int scatter non-atomically — investigate) or >2x slower -> sort+segment_sum fallback, cost re-budgeted |
| R4 | 64^3 twin implementations (int-state vs pure-float): custom_vjp gradient of a band-power loss vs jax.grad of the float twin vs central FD; error vs K and vs quantization width | **do STE gradients through ~6-fractional-bit CIC weights stay useful?** THE deepest risk | gradient error not O(quantization) / grows uncontrolled with K at int16 -> **pivot: exact-reversible UNCOMPRESSED (f32-state, int-free) code — still novel vs pmwd/DISCO-DJ — or stop** |
| R5 | 256^3 float sim vs same sim with positions re-quantized to the int16 lattice each step; P(k) ratio + force cross-check | is the 64-sub-cell-level position lattice below the PM error floor? | P(k) deviation > PM error at k <= k_Nyq/2 -> larger n_mesh:lattice ratio or int16-positions-only redesign |

Gate review with JC at close: verdicts recorded in decisions.md; ROADMAP
ticked; go/pivot/stop decided explicitly. The M0 gradient-fidelity threshold
is measured-then-negotiated (open question #1 of the bootstrap plan).

Exit artifacts: `scripts/m0_r{1..5}_*.py`, probe outputs under `runs/m0/`
(gitignored, summarized in decisions.md), verdict ADR entries.

## M1 — forward PM  ✅ DONE (2026-07-13)

Verdict: forward PM validated. mbody parity at mbody's own repro floor (Tier A,
k-flat ~1e-6 dP/P); DISCO-DJ gap attributed to 3 named conventions and closed
at f64 roundoff (2e-8 cells adapted replay) -> gate D-013; M0 "4% deficit"
decomposed (binning artifact + seed scatter + a real -2.6% code-independent
mode-coupling suppression); Tier-B int16 = pure quantization below the PM mesh
floor -> gate D-014; exact reversibility confirmed at scale on CUDA (256^3
n_diff=0). 512^3 headline run deferred to a larger-GPU env (Vista aarch64) at
M2. Full record: docs/m1-results.md. Detailed plan: tender-stargazing-map.

Scope: real package code. `codec.py` (int16 first), `painting.py`,
`forces.py`, `lpt.py`, `ic.py`, `integrate.py` forward path (BullFrog +
exact-KDK fallback + FastPM), P(k) diagnostic. No adjoint yet.

Gates: mbody forward parity at matched config (measure mbody's own repro floor
first); DISCO-DJ forward parity at matched config (same rule); force == ZA
identity at machine precision; BullFrog EdS closed-form pin; CUBE-style
quantization-below-PM-error check under few-step stepping; reversibility
tier-0 test promoted from R1 probe to CI.

Exit: forward sim validated at 256^3-512^3 on CPU + one CUDA smoke run;
docs/m1-results.md.

## M2 — the adjoint (paper core)

Scope: `adjoint.py` custom_vjp exact-replay adjoint; per-step-jit production
driver (or scan, per R1); deterministic paint decision finalized with measured
cost; IC-parameter wrappers (d/df_NL, d/damplitude — mbody parity).

Gates: adjoint == float-twin jax.grad == FD at measured floors; gradient of
band-power and field-level losses; bit-exact replay in CI on CUDA.

**Money plots (the paper's evidence, produced here):**
1. max differentiable N per 80 GB GPU: inexor vs pmwd vs DISCO-DJ (the OOM
   curve).
2. gradient fidelity vs float-replay adjoint at f32 (replay drift comparison).
3. wall-clock overhead of exact adjoint vs naive AD vs float replay.

Compute placement: Vista GH200 (gg queue) for the 80 GB-class runs; state
machine/queue up front per convention.

Exit: docs/m2-results.md + draft money plots; paper writing can start.

## M3 — int8 mode + the memory frontier

Scope: int8 codec tier (static ZA velocity reference; displacement-limited
positions), 6 B/p benchmark, consumer-GPU demo (albireo, once its GPU lands):
f32-speed exact gradients where f64 is 1/64 rate.

Gates: int8 error ladder vs int16 vs float (extends R2/R5); range monitors
loud; no-clamp invariant lint test.

Exit: docs/m3-results.md; the 6 B/p headline number, honestly scoped.

## M4 — science demo (standalone; NOT omsoc-coupled)

Scope: field-level f_NL toy — gradient-based IC/parameter inference on an
inexor forward model (optimizer or NUTS via the likelihood seam); Lagrangian
bias field ported from mbody if the demo needs a tracer.

Gates: recovery of injected f_NL / IC modes at forecast precision on the toy;
end-to-end grad checks.

Exit: docs/m4-results.md; paper assembly (outline -> draft).

## Post-M4 (recorded only; each needs its own scoping)

- Two-level tiled mesh (CUBE2/sCOLA pattern) -> 2048^3-class meshes /
  beyond-HBM; re-opens determinism questions (sharded psums) — quarantined
  from the core design.
- Vista linux-aarch64 pixi feature (xcat pattern) when GH200 runs become
  routine.
- Public flip + PyPI: check `inexor` name availability BEFORE any release;
  author-field release-blocker checklist (pyproject authors/maintainers,
  CITATION.cff, __author__, built METADATA) — "James Cheshire" everywhere.
- CDF-shaped bins for *output* snapshot compression (not evolving state).
- sCOLA-style spatial tiling (the far bigger memory lever, per the research
  session) — a separate project decision, not an inexor milestone yet.
