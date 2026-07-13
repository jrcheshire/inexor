# CLAUDE.md -- inexor

Exactly reversible, compressed-state differentiable N-body in JAX: fixed-point
integer phase space (int16 default / int8 opt-in) makes evolution bit-exactly
reversible, so the adjoint is an exact replay with O(1)-in-steps memory at
12 (or 6) bytes/particle. Methods-paper track (OJAp); James Cheshire's
personal project — NOT SPHEREx-pipeline critical path (disco-mocks/DISCO-DJ
own production mocks; mbody owns MLX/Apple-Silicon).

## Current milestone

**M0 probes COMPLETE, gate review pending** (opened 2026-07-12; CUDA legs
2026-07-13; plan `~/.claude/plans/let-s-put-the-rubber-composed-penguin.md`).
All five verdicts in (`runs/m0/`): R1 authoritative PASS on CUDA (deneb RTX
3050) -- 100/100 seeds x all 5 drivers + wrap-adversarial exact, with the
force on paint="int" (the f32-paint default fails densely on GPU exactly as
Sec. 5 predicts; failure run archived in runs/m0/r1_gpu_f32paint/). R3 PASS
at 2^26/384^3 (the 3050 is the 6 GB variant; 512^3 uniform corroboration
archived) -- int paint 10/10 bit-identical incl. retrace + clustered
worst-case, and FASTER than f32 (0.79-0.85x; f32+deterministic-ops flag
costs 1.37-1.78x f32). R2 range PASS / noise-bar for gate review; R4 (THE
gate) PASS-shaped -- g_STE within the FD reference's own SE at production
int16, anchor O(q) slopes +1.0..+2.2, no K growth; R5 strict PASS at
flagship-equivalent 64 levels. NEXT: the M0 gate review with JC (verdict
ADRs -> decisions.md, R4 threshold + R2 noise bar negotiated, go/pivot/stop,
roadmap ticked). Note for R4 talking points: its CPU gradients ran through
the f32-paint force (the intended VJP twin), while production primal is int
paint -- state this at threshold negotiation.

## Doc map (read before proposing anything)

- `docs/architecture.md` — THE design document. Every design decision with
  rationale + [M0: Rn] risk tags. The w-frame ladder (Sec. 4), the
  deterministic int paint (Sec. 5), and the custom_vjp adjoint (Sec. 8) are
  the load-bearing novel pieces.
- `docs/roadmap.md` — master plan M0-M4; M0 is a HARD go/no-go gate; each
  milestone gets a fresh detailed plan at its opening session.
- `docs/decisions.md` — ADR log (D-001..D-009). Locked until re-litigated
  with JC.
- `paper/outline.md` — claims, money plots, verified citation list + the
  citation guard (refuted claims never to reintroduce).

## Conventions

- pixi: `pixi run test` / `test-fast` / `lint` / `format`. Envs: `default`
  (CPU JAX, osx-arm64 dev) and `gpu` (linux-64 CUDA 12). Co-commit `pixi.lock`
  on dependency changes.
- ruff, line-length 100. ASCII-only in .py files (`->`, `x`, spelled-out
  Greek).
- Library code never toggles `jax_enable_x64`; callers opt in (jht/sfbfs
  convention). Host-side coefficient tables are numpy float64 (precision
  island).
- **wrap-never-clamp invariant** (D-007): no saturating op may touch integer
  state, ever.
- No notebooks; scripts + ipython `%run`. Probe scripts in `scripts/`,
  outputs in `runs/` (gitignored).
- Tolerances: measure the floor first, set gates with JC, never relax
  silently. Reversibility tests assert EXACT integer equality, not tolerances.
- Author = "James Cheshire" <cheshire@caltech.edu> in everything shipped
  (release blocker; never "Jamie").
- Commit as you go; do NOT `git push` unless told (JC manages the remote —
  private GitHub until further notice).
- Ask before resource-heavy local runs (shared laptop). CUDA runs: Vista or
  albireo (once its GPU lands); state machine/queue up front.

## Reference code (read-only from here)

- `~/spherex/mbody/mbody/{integrate,forces,painting,lpt}.py` — the port
  sources (BullFrog weights, geometric force split, CIC patterns, reversible
  adjoint skeleton).
- DISCO-DJ installed in `~/spherex/disco-mocks/.pixi/.../site-packages/discodj`
  (steppers + Diffrax backsolve adjoint) — comparison target; measure its
  repro floor before setting parity gates.

## Umbrella

Threads: `~/notes/threads/spherex/inexor.md`. Umbrella conventions
(`~/spherex/CLAUDE.md`, `~/.claude/CLAUDE.md`) win on conflict.
