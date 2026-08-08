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

**Building M-v2-1 (codec + layout).** The gate week (V1, V2a, V2b, V3) and the
V4 architecture freeze are CLOSED: D-v2-14..18 ratified 2026-08-08 and the
build ladder is D-v2-18's seven rungs, M-v2-1 through M-v2-7. Master strategy +
session seeds: `docs/plan-plan-v2.md` — START THERE; the freeze's measurements
are `runs/v2/v4_architecture_record.md` and `runs/v2/v4_pricing_record.md`.
Design-space study: `docs/design-study-2026-07-14.html` (artifact:
https://claude.ai/code/artifact/370ba948-b9ea-4e42-9c15-92dce10fdf55).

**The probes in `scripts/v2_*.py` are the ratified oracles** — D-v2-10,
D-v2-11 and D-v2-12 are measurements OF `scripts/v2_g5_core.py`, and D-v2-16
clause 7 gates promotion on bitwise parity against it kept unmodified. They
import `inexor.{config,cosmology,ic,lpt}`, `integrate.{a_grid,bullfrog_table,
bullfrog_float_coeffs,float_step_bullfrog}`, `forces.make_force_fn`, and some
PRIVATE painting/diagnostics symbols (`_cic_pieces`, `_corner_flat_weight`,
`_CORNERS`, `_k_grid`, `_bin_edges`, `_shell_mask`). Those names carry a
stability contract in practice: changing them silently moves an oracle.

v1 record (closed): M0 PASSED — GO (D-010..D-012, `runs/m0/`); M1 forward PM
DONE 2026-07-13 (mbody parity at its floor; DISCO-DJ gap attributed to 3
conventions, gate D-013; Tier-B int16 below the mesh floor, gate D-014;
512^3 exact roundtrip n_diff=0 on a GH200); **M2 (the adjoint) HALTED
2026-07-14 — the premise measured false** (`docs/retrospective.md` is the
read-first record; D-015 passed before the halt). v1 M3/M4 are cancelled;
`docs/roadmap.md` stands as the v1 historical record.

Deneb working clone: `deneb:~/src/inexor` (fresh clone 2026-07-15, JC's
`~/src` layout convention; origin = GitHub HTTPS + PAT via
`credential.helper store`, never SSH — pulls from GitHub directly). The old
`~/spherex/inexor` checkout is DELETED; sbatch scripts from v1/M-era assume
the old path and are kept as historical record only — new sbatch scripts
`cd ~/src/inexor`. The M-era bare relay `deneb:~/git/inexor.git` still
exists but is unused. Deneb's RTX 3050 is the 6 GB variant; jobs REQUIRE
explicit `--mem` (see the umbrella albireo memory).

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
- `docs/decisions.md` — ADR log (D-001..D-015, D-v2-8..D-v2-19; D-v2-1..7 live
  in the plan-plan table). Locked until re-litigated with JC.
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
