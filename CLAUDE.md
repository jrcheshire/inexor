# CLAUDE.md -- inexor

Exactly reversible, compressed-state differentiable N-body in JAX: fixed-point
integer phase space (int16 default / int8 opt-in) makes evolution bit-exactly
reversible, so the adjoint is an exact replay with O(1)-in-steps memory at
12 (or 6) bytes/particle. Methods-paper track (OJAp); James Cheshire's
personal project — NOT SPHEREx-pipeline critical path (disco-mocks/DISCO-DJ
own production mocks; mbody owns MLX/Apple-Silicon).

## Current milestone

**M0 PASSED — GO** (gate review with JC, 2026-07-13). All five probes PASS
under the ratified thresholds: R4 gate 7.5e-2 relative gradient error
(measured 9e-4..3e-2); R2 bar = relative P(k) <= 1e-4; R1 100/100 x 5
drivers CUDA-exact; R3 int paint deterministic AND faster than f32; R5
strict. Verdict ADRs D-010..D-012 in `docs/decisions.md` (incl. c_growth
4.0 -> 2.5 and the K >= 3 schedule-feasibility paper limitation); probe
outputs in `runs/m0/` (both laptop and deneb copies). NEXT: **M1 (forward
PM)** — open it with a fresh detailed milestone plan (master-plan
convention); real package code into `src/inexor/` (codec, painting, forces,
lpt, ic, integrate), parity gates vs mbody + DISCO-DJ after measuring their
own repro floors. Deneb git relay: bare repo `deneb:~/git/inexor.git`
(remote `deneb`), working clone `deneb:~/spherex/inexor` — push here AND to
origin; deneb has no GitHub key. Deneb's RTX 3050 is the 6 GB variant.

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
- Ask before resource-heavy local runs (shared laptop). CUDA runs: deneb (via
  Slurm; RTX 3050 6 GB) or Vista; state machine/queue up front.

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
