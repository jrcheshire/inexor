# inexor v2 -- plan-plan (master strategy + session seeds)

Written 2026-07-14, immediately after the v1 halt (read `retrospective.md`
first) and the design-space study. This is the MASTER document for v2: the
locked decisions, the strategy, and a set of SEEDS. Each seed becomes a full
detailed plan in the session that opens it (master-plan convention: this file
sets scope and gates; milestone plans set file-level steps). Seeds carry a
paste-able prompt and their references so those sessions start warm.

Supersedes the forward-looking part of `roadmap.md` (v1 M3/M4 are cancelled;
the M0-M2 record there stands). Companion: the design-space study, committed
as `docs/design-study-2026-07-14.html` (artifact:
https://claude.ai/code/artifact/370ba948-b9ea-4e42-9c15-92dce10fdf55).

---

## 0. Locked decisions (JC, 2026-07-14)

| # | decision |
|---|---|
| D-v2-1 | **Fidelity bar = CUBE-grade with a fine mesh** (error below the PM error floor for k <= 0.2 k_Nyq of the FINE mesh; the mesh is made fine enough that every science k sits inside that range). D-014-grade is not required for v2 mocks. |
| D-v2-2 | **Science target = halo-grade**: resolution sufficient to fit the bias (halos small enough to host the tracer population). Exact numbers pinned at seed V0. |
| D-v2-3 | **Differentiability deferred**: not in the initial build; acceptable as a later SECONDARY, SLOWER mode. Design must not preclude it (twin-kernel rule, Sec. 4). |
| D-v2-4 | **Compute is generally worth less memory**, but the tradeoff gets MAPPED, not assumed: every gate reports (peak B/p, wall/step, SU) so a cost-of-memory curve accumulates (Sec. 3). |
| D-v2-5 | **Standalone project.** No collaboration angle for now (COCA hook dropped). |
| D-v2-6 | **Same repo, v2.** v1's validated components carry; v1's thesis does not. |
| D-v2-7 | **Squeezed-bispectrum tolerance = 15%** for tiled-vs-monolithic agreement (the G3 kill line). |

## 1. Thesis

v2 is a **memory-floor PM mock engine**: maximize (volume x halo-grade
resolution) per single GPU / single node, trading compute for memory.

The measured enemy ordering (v1's R6 profile + the design study) is
**transients >> mesh > state**: JAX forward peaked at 138 B/particle against a
12 B/particle state, while CUBE runs production PM at 9.84 B/particle all-in.
So v2 attacks, in order:

1. **Transients**: chunked custom paint/gather kernels + preallocated buffers
   (Pallas/ffi first, CUDA core only if the G1 gate forces it).
2. **Mesh**: two-level force (global coarse mesh + fine mesh in streamed
   tiles; PMFAST/CUBE2 lineage) so fine-mesh residency is O(tile) and
   halo-grade fine meshes become affordable.
3. **State**: CUBE-class codec (int8 cell-relative positions + int8 velocity
   residual vs the local coarse-grid mean flow, CDF-binned), 6 B/particle,
   with int16-global (validated, 12 B/p) as the fallback tier.

Plus: deterministic int32-mesh paint everywhere (bit-stable AND faster,
measured); on-demand tile ICs from counter-based noise (kills the measured
76 B/p IC peak); out-of-core tiers (GH200 LPDDR, S3 host/NVMe) for hero runs.

Capacity targets that fall out (design study Sec. 6, to re-derive at V4 with
gate-measured numbers): ~10-16 B/p all-in -> 2048^3 on one Vista GH node;
4096^3 host-resident on one Stampede3 H100 node.

**Non-goals** (deliberate): multi-node MPI (post-v2 at the earliest); hydro;
general-purpose tree/FMM; any differentiability work before a thread names it
(D-v2-3); any capability claim before its premise gate.

## 2. Requirements chain (numbers to PIN at V0)

The two locked decisions D-v2-1 + D-v2-2 combine into a resolution chain:

- **Mass resolution.** m_p = 2.775e11 * Omega_m * (L/N_p)^3 h^-1 Msun
  (~8.6e10 * (L/N_p in Mpc/h)^3 at Omega_m = 0.31). If the bias fit needs
  halos of mass M_min with >= n_p particles, then L/N_p <=
  (M_min / (n_p * 8.6e10))^(1/3) Mpc/h. Example: M_min = 1e12, n_p = 100
  -> L/N_p <= 0.49 Mpc/h -> 2048^3 in a 1 Gpc/h box, or 4096^3 in 2 Gpc/h.
  **V0 pins M_min, n_p, and the bias-fitting route** (FoF-style catalogs vs
  field-level Lagrangian bias a la BACCO -- the latter relaxes halo-finding
  but not resolution).
- **Force / mesh resolution.** CUBE-grade validity needs k_sci <= 0.2 k_Nyq
  of the fine mesh, i.e. cell <= 0.2 * pi / k_sci. k_sci = 0.5 h/Mpc ->
  cell <= 1.26 Mpc/h; k_sci = 1.0 -> cell <= 0.63. With particle spacing
  ~0.5 Mpc/h this puts the fine mesh at 1-2x the particle grid; at 4096^3
  particles a global fine mesh (up to 8192^3, 2.2 TB in f32) is impossible
  monolithically -- **the two-level tiled mesh is load-bearing, not an
  optimization**.
- **Velocity fidelity.** Halo-grade mocks may carry RSD; the velocity codec
  bar (what error in what velocity statistic) is currently UNSET. V0 pins it
  (SPHEREx photo-z smearing likely makes it loose; do not assume).
- **Statistics + tolerances.** P(k) per band; squeezed bispectrum <= 15%
  (D-v2-7); halo mass function / bias tolerance (V0); number of realizations
  per config (drives the SU side of the tradeoff map).

## 3. The compute<->memory tradeoff map (D-v2-4)

Instrument, not afterthought. Every gate run reports the triple
**(peak B/p all-in, wall/step, SU per realization-equivalent)** using the v1
mem-profiler discipline (one config per subprocess, `peak_bytes_in_use`,
sweep the discriminating axis) extended with timing + SU accounting.

Running record: `runs/v2/cost_of_memory.md` (a table + one figure), updated
at V1/V2/V3. Expected points, roughly left (cheap-memory) to right:
A3 sequential tiles (x3.2 compute), A4 NVMe-streamed, A2 PCIe-staged,
A2 C2C-streamed (GH), A2 all-HBM, A1 monolithic. Decision rule at V4: stay on
the Pareto front, B/p primary, but a point that buys 2x memory for >5x SU
needs an explicit JC sign-off.

## 4. What carries from v1

| v1 asset | v2 role |
|---|---|
| `config/cosmology/lpt/ic/forces/painting` | carry nearly as-is (painting gains the tile/chunk interface) |
| `codec.py` (int16 + ladder + wrap) | fallback state tier S1; the wrap-never-clamp invariant (D-007) binds every new int codec |
| `integrate.py` (BullFrog/FastPM/KDK) | carry; COLA-style frame optional later |
| deterministic int32-mesh paint (D-006 lineage) | mandatory everywhere (bit-stable + faster, measured) |
| `adjoint.py` + STE twin machinery | PARKED for the deferred differentiable mode; the **twin-kernel rule**: every new fast kernel ships either a `custom_vjp` stub or a pure-JAX twin, so the slow differentiable mode stays buildable without redesign |
| instruments: `m1_quant_gate`, parity harness (mbody/DISCO arms), `m2_mem_profile` | the gate-week instruments; quant-gate gains codec plug-ins, mem-profiler gains the SU/wall triple |
| gates D-010..D-015, floor-first protocol | conventions carry; v2 gates get their own ADR numbers (D-v2-*) |
| citation guard (retrospective + design study Sec. 1) | binds all v2 docs; additions: "HACC ~10% overload memory" is folklore; sCOLA 16-Gpc projection unverified |
| ops: deneb Slurm (explicit --mem, PYTEST_XDIST_AUTO_NUM_WORKERS=1), Vista aarch64 gpu env, pixi lock co-commit | carry unchanged |

New probes added with this doc (run this session on stored M1 data):
`scripts/v2_g2_residual_range.py` (residual-frame dynamic range; ZA beats
2LPT as reference; velocity tails 10-14 sigma) and
`scripts/v2_g2b_codec_ladder.py` (one-shot codec ladder; int8 cell-relative
= 3.1e-5 max dP/P, int16-grade at half the bytes; pure residual coding fails
on tails). Outputs land in `runs/v2/` (gitignored).

## 5. Seeds

Sequence: V0 -> V1 -> V2 -> V3 -> V4 (freeze) -> V5+ (build) -> VD (deferred).
V1-V3 are the "gate week": every premise priced before the architecture
freezes. Each seed's prompt is meant to be pasted at that session's start
(plan mode; floors before gates; state compute placement up front).

### V0 -- requirements pin + tradeoff frame  [with JC; no compute]

- **Goal:** pin the Sec. 2 numbers (M_min, n_p, bias route, k_sci, velocity
  bar, statistics/tolerances, realization counts) and ratify the Sec. 3
  tradeoff instrument. Produce the v2 config table (the 2-4 named
  (L, N_p, mesh) configs everything else measures against).
- **Exit:** an ADR (D-v2-8) recording the requirement chain; config table in
  this file.
- **Kill:** none (definition session).
- **Prompt:**
  ```
  Open inexor v2 seed V0 (requirements pin). Read docs/plan-plan-v2.md
  Secs 0-3, docs/retrospective.md Sec 7, and the design study
  (docs/design-study-2026-07-14.html) Secs 3 and 8. With me, pin: halo mass
  floor M_min + particles-per-halo n_p + bias-fitting route (halo catalogs
  vs field-level Lagrangian bias); k_sci and the implied fine-mesh cell;
  velocity-statistic bar for RSD; per-statistic tolerances (squeezed-B is
  locked at 15%); realizations per config. Derive the v2 config table
  (L, N_p, n_mesh_fine, m_p, per-node home). Then ratify the cost-of-memory
  instrument (triple = peak B/p, wall/step, SU). Write ADR D-v2-8 and update
  plan-plan Sec 2. No builds. Check SPHEREx-side inputs (chimera/mbody
  CLAUDE.md, disco-mocks v28 mock specs) before proposing numbers.
  ```
- **References:** design study Sec 2-3 tables; `~/spherex/mbody` +
  `~/spherex/disco-mocks` (mock conventions); chimera CLAUDE.md for tracer
  populations; BACCO hybrid bias (arXiv:2307.09134, 2407.07949) for the
  field-level route.

### V1 -- gate pair G1 (kernel floor) + G2c (codec, accumulated)  [deneb + laptop]

- **Goal:** (G1) measure the custom-kernel forward floor: chunked atomic-add
  CIC paint + gather (Pallas first; note the documented Pallas GPU `scatter`
  primitive gap -- a hand atomic-add kernel is the route; jax.ffi CUDA
  fallback), peak B/p + wall vs the XLA-native path at 128^3-512^3 on deneb.
  (G2c) plug the v2 codecs into `m1_quant_gate`: int8 cell-relative
  positions; int8 CDF velocity residual vs coarse-grid mean flow (CUBE
  Sec 2.2 mechanism); int16-global control. Accumulated (quantize-every-step)
  error vs the FINE-mesh floor at the D-v2-1 bar, on a V0 config.
- **Exit:** G1 number (B/p floor + wall ratio) -> framework verdict input;
  G2c verdict per codec tier (6 vs 9 vs 12 B/p at the v2 bar). Both land in
  the cost-of-memory table.
- **Kill:** G1 > 2x the hand-managed estimate (~30 B/p monolithic class) ->
  X2 (JAX+kernels) dies; escalate X3 (CUDA core) vs A3 (tiles-in-plain-JAX)
  to JC. G2c: 6 B/p tier fails the bar -> fall to 9 B/p (int16 velocity);
  9 fails -> S1 (12 B/p, validated) and the capacity table shifts one notch.
- **Prompt:**
  ```
  Open inexor v2 seed V1 (gate week 1: G1 kernel floor + G2c codec gate).
  Read docs/plan-plan-v2.md Secs 1-5 (V1), the design study Secs 3-4
  (measured anatomy + S2/S3 cards), scripts/v2_g2b_codec_ladder.py and its
  results, and retrospective.md Sec 5 (measurement methodology). Plan G1:
  a Pallas atomic-add CIC paint+gather microbench (chunked, preallocated),
  deneb via Slurm (explicit --mem; PYTEST_XDIST_AUTO_NUM_WORKERS=1), peak
  B/p + wall vs the XLA path at 128^3-512^3, one config per subprocess.
  Plan G2c: extend scripts/m1_quant_gate.py with pluggable codecs (int8
  cell-relative pos; int8 CDF vel residual vs coarse-grid mean flow, CUBE
  1712.06121 Sec 2.2; int16 control), accumulated error at the V0 config
  against the fine-mesh floor, velocity statistic per D-v2-8. Floors first;
  kill lines per plan-plan V1. Record the (B/p, wall, SU) triples.
  ```
- **References:** `scripts/m1_quant_gate.py`, `scripts/m2_mem_profile.py`,
  `scripts/v2_g2b_codec_ladder.py` + `runs/v2/g2b_results.json`; CUBE
  arXiv:1712.06121 (Table 1, Sec 2.2); Pallas scatter gap jax#31876;
  measured anchors: one-shot int8-cellrel 3.1e-5, one-shot->accumulated
  multiplier ~20x (D-014 calibration).

### V2 -- gate pair G5 (two-level force) + G3 (tile seams, squeezed-B)  [deneb]

- **Goal:** (G5) PMFAST-pattern two-level force (global coarse 4x + fine
  tile + polynomial blend) vs monolithic fine mesh at 256^3: force-error
  ladder + P(k), D-013-class metrics, buffer-width scan. (G3) sCOLA-style
  sequential independent tiles vs monolithic at 256^3-512^3: P(k), cross-r,
  and the **squeezed bispectrum** (the statistic the tiling literature never
  measured), against the 15% bar (D-v2-7).
- **Exit:** G5: split error quantified vs the fine-mesh floor -> M3 verdict
  (A2's spine). G3: squeezed-B error number -> A3 verdict + A2 buffer sizing.
  Both -> cost-of-memory points (A3's x3.2 compute cost becomes a measured
  SU number).
- **Kill:** G5 split error above the D-v2-1 band after kernel matching ->
  A2 falls back to monolithic-per-node (A1 shape) and the 4096^3 target
  moves to A4-only. G3 squeezed-B > 15% at sane buffers -> A3 dies (A2
  unaffected); publish the negative (P-C) regardless.
- **Prompt:**
  ```
  Open inexor v2 seed V2 (gate week 2: G5 two-level force + G3 tile seams).
  Read docs/plan-plan-v2.md V2, design study Secs 4.3-4.4 (M3, E4 cards) and
  6, and the sCOLA numbers (Leclercq 2003.04925: buffer >= 25 Mpc/h, tile >=
  50, r ~ 3.2; error is seam-localized; no published bispectrum test).
  Plan G5: implement the PMFAST-style coarse+fine split as a probe (not
  package code) on 256^3; force-error ladder vs monolithic (D-013 metrics),
  buffer-width scan, kernel matching per Hockney-Eastwood. Plan G3: evolve
  the same ICs monolithic vs sequential 2LPT-frame tiles; measure P(k),
  cross-r, and the squeezed bispectrum (k_long at the box fundamental few-x,
  k_short in the science band); bar = 15% (D-v2-7). deneb via Slurm.
  Floors first (estimator floor on the monolithic pair before judging
  tiles). Record (B/p, wall, SU) for the tiled arm.
  ```
- **References:** PMFAST astro-ph/0402443 (verified: 4x coarse, tiles,
  polynomial matching, 6 floats + 1 int per particle); CUBE2 2512.12629
  Sec 2.2-2.3 (3-level + buffer overhead 10-30%); sCOLA 1502.07751 +
  2003.04925; existing `_m1_common` estimator + a bispectrum estimator to
  add (verify formula vs a primary source before use).

### V3 -- G4 GH200 path + first streaming tradeoff points  [1-2 Vista jobs]

- **Goal:** does JAX/XLA on a Vista GH node touch LPDDR-resident arrays at
  C2C speeds (ATS/coherent path), or only via explicit `pinned_host`
  staging? Bench a real paint step at a >96 GB working set both ways;
  measure GB/s and the (B/p, wall) triple for the streamed-state pattern.
- **Exit:** the A2-on-GH claim gets its measured footing (coherent vs
  staged vs infeasible); cost-of-memory gains the GH points.
- **Kill:** nothing dies; a negative result downgrades 2048^3-on-one-GH to
  explicit staging (slower, still viable) or reroutes to S3 nodes.
- **Prompt:**
  ```
  Open inexor v2 seed V3 (G4: GH200 coherent-memory reality check). Read
  docs/plan-plan-v2.md V3 and the design study E2 card (measured GH200
  numbers: HBM 3.4 TB/s, LPDDR 486 GB/s, C2C 375/297 GB/s; JAX host-offload
  = pinned_host only, coherent path undocumented). Plan one Vista GH job:
  (a) allocate >96 GB of state, attempt jnp ops on host-backed arrays (ATS
  path) vs explicit device_put(pinned_host) staging vs HBM-resident control;
  (b) run the G1 paint kernel over streamed chunks; measure sustained GB/s
  and wall/step. gpu env exists (pixi, linux-aarch64). Generous walltime;
  state compute placement up front. Record the triples; update
  runs/v2/cost_of_memory.md.
  ```
- **References:** Schieffer+ 2407.07850, Fusco+ 2408.11556 (GH200 measured
  bandwidths); JAX host-offloading doc (pinned_host); Vista ops in
  CLAUDE.md; the G1 kernel from V1.

### V4 -- architecture freeze + v2 roadmap  [with JC]

- **Goal:** with G1/G2c/G5/G3/G4 numbers on the table: pick the architecture
  (expected: A2 Mock Factory shape -- state tier per G2c, framework per G1,
  mesh split per G5, node ladder per G4; A3 as a companion product if G3
  passed), freeze it in ADRs, and write the v2 build roadmap (M-v2-1..4,
  each opening with its own plan session).
- **Exit:** ADRs D-v2-9+ (architecture); roadmap section appended to this
  file; V5 seeds instantiated with real numbers; updated capacity table.
- **Kill:** if the gate week gutted every lane (unlikely -- S1 + A1 remain
  validated fallbacks), reconvene on whether v2 proceeds at all.
- **Prompt:**
  ```
  Open inexor v2 seed V4 (architecture freeze). Read docs/plan-plan-v2.md
  (all), runs/v2/cost_of_memory.md, and the V1-V3 verdicts. Present the
  gate-week numbers against the design study's projections (Secs 4, 6);
  propose the frozen architecture (state tier, mesh split, framework,
  node ladder, codec bar) as ADRs; draft the v2 build milestones
  (M-v2-1 engine core; M-v2-2 tiles + on-demand ICs; M-v2-3 codec + capacity
  runs; M-v2-4 halo/bias output stage + mock validation), each with entry/
  exit gates in the roadmap convention. Decision session: plans and ADRs
  only, no code.
  ```

### V5+ -- build milestones  [instantiated at V4]

Placeholders; V4 writes their real seeds with gate-informed numbers:

- **M-v2-1 engine core:** two-level PM forward on the chosen framework,
  V0-config correctness vs the v1 parity arms (mbody/DISCO where configs
  overlap), deterministic paint, per-step-jit/donation discipline.
- **M-v2-2 tiles + ICs:** tile streaming, buffer machinery, counter-based
  on-demand tile ICs (tiling-invariant by keying on absolute q-cell/mode --
  verify identity vs monolithic at f64 first; the sCOLA master-field
  precedent slices, it does not regenerate: this part is new).
- **M-v2-3 codec + capacity:** the G2c-winning codec live; capacity ladder
  runs (deneb -> GH -> S3 hero) with the cost-of-memory table finalized.
- **M-v2-4 output stage:** on-the-fly P(k)/bispectrum + halo-proxy or
  field-level-bias emission per D-v2-8; snapshot packing tier (pack9-class);
  v28-style mock hand-off format (disco-mocks compatibility check).

### VD -- differentiable slow mode  [deferred; opens only when named]

- **Trigger:** a concrete thread (bayes-lss field-level f_NL at scale, an
  xcat-chain coupling, or similar) states the N and the gradient it needs.
  Per the retrospective rule: no named workflow, no build.
- **Shape when it opens:** D2 (f32 backsolve replay -- drift measured 4e-5,
  flat in K) through the twin kernels; tile-chunked reverse sweep (adjoint
  through spatial tiles is unpublished); target adjoint <= 2x forward.
  P-A (the "where the bytes go in differentiable PM" benchmark note) can be
  written from EXISTING v1 measurements at any time, independent of VD.
- **Prompt (when triggered):**
  ```
  Open inexor v2 seed VD (differentiable slow mode). Precondition: name the
  workflow (project, N, loss, gradient consumer) in the first message. Read
  docs/plan-plan-v2.md VD, design study Sec 4.6, retrospective.md Secs 2+7,
  and docs/m2-results.md (the 437-490 B/p adjoint measurements). Plan G6
  first: a custom-VJP paint microbench (tile-chunked reverse) on deneb --
  does the ~430 B/p transient actually collapse to O(tile)? Only if G6
  passes, plan the mode: f32 backsolve replay via the twin kernels, adjoint
  <= 2x forward gate, D-015-style global-metric fidelity gates.
  ```

## 6. Risk register

| risk | hits | mitigation |
|---|---|---|
| Pallas atomic-add path immature on our stack | V1/G1 | jax.ffi CUDA kernel fallback is in the same gate; X3 escalation path defined |
| velocity codec fails the (unpinned) RSD bar | V1/G2c | 9 B/p and 12 B/p tiers are one-line fallbacks; V0 pins the bar before the gate runs |
| two-level split error above the band | V2/G5 | A1 monolithic shape remains; capacity targets shift, project survives |
| tile seams fail 15% squeezed-B | V2/G3 | A3 dies only; negative result is publishable (P-C); A2 unaffected |
| JAX cannot see GH LPDDR coherently | V3/G4 | explicit staging path measured in the same job; S3 nodes as alternative |
| halo-grade quietly drags in PP/tree scope | V0, V4 | V0 pins the bias route first; PP-in-tiles is a V4 decision with its own gate, never a default |
| framework fork (X3) doubles engineering | V4 | only reachable via a failed, measured G1; JC sign-off required |
| scratchpad-era measurements rot | now | G2/G2b scripts + results committed into scripts/ + runs/v2/ with this doc |

## 7. Conventions carried (binding)

Floor-first gates, tolerances never relaxed silently; premise gate before any
build (every seed has a kill line); name-the-workflow for capabilities;
wrap-never-clamp (D-007) on any integer state; deterministic int paint on
every primal path; ASCII-only in .py; ruff, line-length 100; pixi with lock
co-commit; commit-as-you-go, never push (JC pushes); deneb via Slurm with
explicit --mem; generous walltimes; state compute placement up front;
"James Cheshire" in anything shipped (release blocker); citation guard per
retrospective + design study Sec 1/9.
