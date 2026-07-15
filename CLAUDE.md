# CLAUDE.md -- inexor

**v2 (2026-07-14): memory-floor PM mock engine** — maximize (volume x
halo-grade resolution) per single GPU/node; compute freely traded for memory.
The v1 thesis (exactly reversible, compressed-state differentiable N-body)
was HALTED 2026-07-14 — its premise measured false; read
`docs/retrospective.md` before touching anything v1-motivated. v1's validated
components (int16 codec, deterministic int paint, integrators, harness suite)
carry forward. Methods track; James Cheshire's personal project — NOT
SPHEREx-pipeline critical path (disco-mocks/DISCO-DJ own production mocks;
mbody owns MLX/Apple-Silicon).

## Current milestone

**v2 OPENED** (2026-07-14). Master strategy + session seeds:
`docs/plan-plan-v2.md` — START THERE. Near-term sequence: seed V0
(requirements pin + tradeoff frame, with JC) -> V1/V2/V3 (the gate week:
G1 kernel floor, G2c accumulated codec gate, G5 two-level force split,
G3 tile seams incl. squeezed bispectrum at the 15% bar, G4 GH200 coherent
path) -> V4 (architecture freeze + build roadmap). Design-space study:
`docs/design-study-2026-07-14.html` (artifact:
https://claude.ai/code/artifact/370ba948-b9ea-4e42-9c15-92dce10fdf55).
New probes: `scripts/v2_g2_residual_range.py`, `scripts/v2_g2b_codec_ladder.py`
(results quoted in the plan-plan; outputs `runs/v2/`, gitignored).

v1 record (closed): M0 PASSED — GO (D-010..D-012, `runs/m0/`); M1 forward PM
DONE 2026-07-13 (mbody parity at its floor; DISCO-DJ gap attributed to 3
conventions, gate D-013; Tier-B int16 below the mesh floor, gate D-014;
512^3 exact roundtrip n_diff=0 on a GH200); **M2 (the adjoint) HALTED
2026-07-14 — the premise measured false** (`docs/retrospective.md` is the
read-first record; D-015 passed before the halt). v1 M3/M4 are cancelled;
`docs/roadmap.md` stands as the v1 historical record.

Deneb git relay: bare repo `deneb:~/git/inexor.git` (remote `deneb`),
working clone `deneb:~/spherex/inexor` — push here AND to origin. GitHub
access is HTTPS + PAT (deneb has `credential.helper store`); never SSH.
Deneb's RTX 3050 is the 6 GB variant; jobs REQUIRE explicit `--mem` (see the
umbrella albireo memory).

## Doc map (read before proposing anything)

- `docs/plan-plan-v2.md` — **v2 master strategy + session seeds** (locked
  decisions D-v2-1..7, gate week, seed prompts). The v2 entry point.
- `docs/design-study-2026-07-14.html` — the v2 design-space study (measured
  memory anatomy, approach catalog, synergy matrix, architectures A1-A5).
- `docs/retrospective.md` — why v1 halted; methodology + trap list; read
  before re-proposing anything v1-flavored.
- `docs/architecture.md` — the v1 design document (header warning applies:
  its motivation and Sec. 9 budget are the design-time record, NOT current
  fact). Every design decision with
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
