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

**M-v2-1 through M-v2-5 are CLOSED; M-v2-6 (capacity: a complete 2048^3 mock
on one Vista gh node) is next.**

**M-v2-5 (streamed ICs + out-of-core FFT) closed 2026-08-10, ratified as
D-v2-23** (record `runs/v2/m5_ic_record.md`). The IC stage is rebuilt IN
PLACE on the plane-keyed `m5-foldin-1` noise stream (one axis-0 plane keyed
by `fold_in`, CPU-drawn; a seed now denotes a DIFFERENT realization than
before -- cards carry `ic_stream` and readouts refuse to pool across it), a
universal-node 1D P+T table (error 2.571e-7 vs the 1e-4 bar; nodes are
FIXED [1e-4, 1e2] because per-grid nodes broke G5b's shared-modes identity),
and a host-resident out-of-core FFT whose compute unit is ONE PLANE by
measurement (pocketfft moves bits with batch size). Headlines: **the
streamed generator reads 8.6 B/p where the old path reads 95.7** (fitted
cubic 8.79 -> 75.3 GB at 2048^3 against the 116 GB GH200 host); **a 2048^3
transform the device cannot fit ran correctly in 40.4 GB of host** (Vista
902182: roundtrip 2.38e-6, Parseval 1.69e-7, P(k) vs the BIN-AVERAGED
oracle max|z| 2.90); and **the loaded T9 state is bitwise the monolithic
`SlotState.build`** at every scale tested (n up to 512, f_NL 0 and 10).
f_NL differentiability is RETIRED with v1 (`ic.colour_white` is the seam a
gated jnp twin would be built behind). `runs/m1` stored references remain
valid (loaded, never regenerated); `m1_export_ics.py` is historical.
Ops trap for any gg sbatch: `JAX_PLATFORMS=cpu` is LOAD-BEARING -- jax
0.10's CUDA plugin hard-raises on cuInit on a GPU-less node, and
`CONDA_OVERRIDE_CUDA` alone no longer suffices.

**M-v2-4 (f32 coarse force mesh) closed 2026-08-10, D-v2-22** (record
`runs/v2/m4_f32_mesh_record.md`): coarse mesh f32, fine unchanged; 1.830x
peak host at n_coarse=1024; tier-2 accuracy miss recorded as a miss with
the f64 reference's own 3.590e-2 mesh floor as context; the harness's
1.834 asymptote is the 8.0 B/cell int64 paint accumulator (engine nets to
1.9945).

M-v2-3's exit gate passed all three parts (Vista 898169/898242, record
`runs/v2/m3_engine_record.md`, ratified as **D-v2-21**): the engine's force is
bitwise the ratified path's at cgh64 -- 0 of 402,653,184 elements, driven from a
DIFFERENT membership order than the reference, which is only possible because
both paints became integer -- and the T9 codec costs 7.125e-4 at C-dev K=40
against D-v2-9's 3e-2, a 42x margin, measured for the first time through the
real two-level force and the real storage layout. State now lives in slot order,
so the ~21 B/p of scaffolding and the int32 `key` ceiling are gone rather than
widened. D-v2-18 is explicit that M-v2-4 gets its OWN gate and cannot ride on
D-v2-9's or G6's. M-v2-2's gate is `runs/v2/m2_parity_record.md` (Vista 897904).

Two things M-v2-3 found that no document had named, both now in D-v2-21.
(1) **The tiled SHORT-range paint was order-dependent with no integer twin**
(`forces.tile_paint_f64`), so D-v2-14 clause 4's premise — the layout is
admissible only because the paint is order-independent — was FALSE on the arm
carrying most of the force; D-v2-16 clause 2 named only the coarse
`paint_tsc_int`. Both arms are integer now, and that is what makes the bitwise
gate above possible from two DIFFERENT membership orders.
(2) **M-v2-3's written exit gate could not be read literally**: no v1 parity
config overlaps a v2 one (v1 is single-level at 2-4 Mpc/h, mesh:particle 1; v2
is 0.25 Mpc/h at mesh:particle 2) and the v1 quantized arm's generator was
deleted at the retirement. Replaced by the three-part gate D-v2-21 clause 6
records.

**Owed out of M-v2-3** (record's "Owed" section): `SlotState.repack` allocates O(N) where D-v2-19 clause 3 establishes a monotone
in-place form; and the parity instrument compares an O(N) array, so **cgh64 is
its ceiling** — hero-scale parity needs a per-tile statistical form.

The gate week (V1, V2a, V2b, V3) and the V4 architecture freeze are CLOSED:
D-v2-14..18 ratified 2026-08-08 and the build ladder is D-v2-18's seven rungs,
M-v2-1 through M-v2-7. Two ADRs landed on top of the freeze the same day:
**D-v2-19** (spare pools per brick; the re-sort is a monotone rearrangement,
not a sort) and **D-v2-20** (the per-bucket index is uint32, 0.50 B/p, all-in
~10.54 — the uint16 ceiling is removed rather than measured, and the cgh64 tail
re-run is no longer owed). Master strategy +
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
**`runs/m1/` is ~1.3 GB, gitignored and NOT force-added** (`git ls-files runs/m1`
is empty) — it exists on this laptop only. It holds the stored `mbody_final_*`,
`disco_final_*` and `disco_coeffs_*` reference states that the surviving D-013
arms compare against, so M-v2-3's regression leg depends on a working copy that
no other machine can reconstruct. Do not move or clear it.

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
- `docs/decisions.md` — ADR log (D-001..D-015, D-v2-8..D-v2-25; D-v2-1..7 live
  in the plan-plan table). Locked until re-litigated with JC. Note two ADRs
  supersede parts of D-v2-14: D-v2-19 (clause 3, layout) and D-v2-20 (clause
  2's index term) — read those before quoting a B/p figure.
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

- DISCO-DJ installed in `~/spherex/disco-mocks/.pixi/.../site-packages/discodj`
  (steppers + Diffrax backsolve adjoint) — the comparison target for the PM
  step and integrator; measure its repro floor before setting parity gates.
  `scripts/v2_disco_crosscheck.py` runs it on our ICs and scores it with our card.
- The BullFrog integrator is defined by Rampf, List & Hahn 2024
  (arXiv:2409.19049); its weights need the true LCDM growth pair (Sec. 4.4).
- `~/spherex/mbody/mbody/{integrate,forces,painting,lpt}.py` — where much of
  the early code was ported from. mbody was a toy: it is provenance, not a
  physics reference. Its EdS BullFrog weights were the cause of the engine's
  ~13% high-k deficit (D-v2-25); judge inherited conventions against the
  physics or DISCO-DJ/emulators, never by parity with mbody.

## Umbrella

Threads: `~/notes/threads/spherex/inexor.md`. Umbrella conventions
(`~/spherex/CLAUDE.md`, `~/.claude/CLAUDE.md`) win on conflict.
