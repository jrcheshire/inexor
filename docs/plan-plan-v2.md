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
| D-v2-1 | **Fidelity bar = CUBE-grade with a fine mesh** (error below the PM error floor for k <= 0.2 k_Nyq of the FINE mesh; the mesh is made fine enough that every science k sits inside that range). D-014-grade is not required for v2 mocks. **AMENDED by D-v2-9: the BAND half stands; the "below the PM error floor" BAR is retired — that floor is not measurable at fixed particles (job 39) — and is replaced by an absolute |dP/P| <= 3e-2 in-band.** |
| D-v2-2 | **Science target = halo-grade**: resolution sufficient to fit the bias (halos small enough to host the tracer population). Exact numbers pinned at seed V0. |
| D-v2-3 | **Differentiability deferred**: not in the initial build; acceptable as a later SECONDARY, SLOWER mode. Design must not preclude it (twin-kernel rule, Sec. 4). |
| D-v2-4 | **Compute is generally worth less memory**, but the tradeoff gets MAPPED, not assumed: every gate reports (peak B/p, wall/step, SU) so a cost-of-memory curve accumulates (Sec. 3). |
| D-v2-5 | **Standalone project.** No collaboration angle for now (COCA hook dropped). |
| D-v2-6 | **Same repo, v2.** v1's validated components carry; v1's thesis does not. |
| D-v2-7 | **Squeezed-bispectrum tolerance = 15%** for tiled-vs-monolithic agreement (the G3 kill line). |
| D-v2-8 | **Requirements chain pinned at seed V0** (JC, 2026-07-15; full ADR in `decisions.md`): general engine, NOT SPHEREx-anchored (disco-mocks/DISCO-DJ own main-line mocks); k_sci = 2.0 h/Mpc -> fine cell <= 0.31 Mpc/h; hybrid bias route (halo catalogs at calibration, field-level in production); n_p = 100; M_min DERIVED from the chain, reported per config; multipole-grade RSD bar (P0+P2 under D-v2-1); HMF 5% / halo-b1 2% at calibration; realizations 100 prod / O(5) calib; Sec 3 instrument ratified. |
| D-v2-9 | **P(k) fidelity bar is now ABSOLUTE** (JC, 2026-07-15, seed V2a; full ADR in `decisions.md`). D-v2-1's BAND stands (k <= 0.2 k_Nyq fine, so D-v2-8 clause 1 is unaffected); its "below the PM error floor" BAR is retired as unmeasurable at fixed particles and replaced by **\|dP/P\| <= 3e-2 in-band**, plus a **mandatory reported split-to-discretization ratio vs k**. Absolute because D-v2-8 pins the same fine cell across the whole config table, so it transfers unchanged. C-dev's own discretization error measured at 6.14e-2 in-band (job 40, a LOWER bound). NB **the split DOMINATES that error for k < 1.53** (7.4x at the fundamental) — max-over-band comparisons hide this, which is why the ratio clause is mandatory. |
| D-v2-10 | **G5 verdict: PASS — A2's spine ratified** (JC, 2026-07-16; full ADR in `decisions.md`). 2.61e-2 vs the 3e-2 bar at the probe config (tile 128 / buf 32, C-dev). **The 0.87 margin is a gate-config artifact**: error ∝ 1/P, peak memory ∝ P^3 as an ABSOLUTE box-independent working set, wall overhead = the padded-volume ratio — so at the config-table homes tile size is a cheap accuracy dial. **Binding framing: performance/Pareto evaluation happens at the config-table TACC homes; deneb is the correctness/dev ground** — its 6 GB never shapes a gate config again (deneb-fit contortions deleted; the full C-dev G2c rerun runs monolithically on a GH200). Operating (T, b) per config = V4's Pareto call, informed by G5c (Vista: cross-hardware anchor + box-ladder 1/P coefficient + real capacity triples). The low-k architectural floor is accepted and assessed SEPARATELY (realization stability -> correctable-as-transfer question), after G5c. **One clause superseded by D-v2-11.** |
| D-v2-11 | **The low-k split error IS a correctable transfer, and it calibrates OFF-BOX** (JC, 2026-07-29, G6 close; full ADR in `decisions.md`). Supersedes D-v2-10's "a measured-transfer correction cannot currently be claimed"; the G5 verdict is untouched. Near-pure window (max(1-r) = 3e-5 at k <= 0.5, phase-perfect), so a leave-one-out transfer removes a factor 9.0 at k <= 0.5 within C-dev. On a fixed-cell 1x/8x/64x box ladder the transported residual on the gate band is 1/16, 1/71 and 1/13 of the 3e-2 bar, so **T-bar is calibrated once on a small ensemble and applied per mock: no monolithic reference and no ensemble at the production box**, which is what makes the memory saving real rather than notional. Tile size and origin are measured non-triggers; **fine cell, cosmology and redshift are UNMEASURED and treated as triggers**. The correction is reported, not gated: D-v2-9's bar is still read on the UNCORRECTED split. Residual floor: ~12-20% per-realization scatter is physical (shared-IC ratio cancels cosmic variance), leaving ~1e-3 at k <= 0.5 on any single box regardless of calibration seeds. |
| D-v2-12 | **G3 verdict: A3 dies; A2's buffer is sized at 4 r_s** (JC, 2026-08-06, G3 close; full ADR in `decisions.md`). Fires the pre-registered V2 kill line. A3 (sCOLA-style sequential independent tiles) misses D-v2-7's 15% bar by 2.7-6.1x at cdev, and the failure is realization-INDEPENDENT: each tile solves the FULL kernel on its own periodic padded box, so it carries only 15-40% of the monolithic long-mode POWER and the amplitude knee tracks the padded-tile fundamental across four arms. A2 unaffected; D-v2-10/D-v2-11 untouched; the negative is publishable (P-C). **Two of the three ratified G3 statistics were withdrawn** -- R_Q carries the power deficit into Q's denominator with the opposite sign and flips positive at x <= 1, and rho divides by the CROSS transfer T = r*A and so tracks 1/r^2 - 1 (measured 16/16 cells within 10-20%, reaching 3.6e4 where r crosses zero). Readings now use rho_auto, built on the AUTO transfer. The realization-matched estimand is retired as the default for tiled-vs-monolithic comparison. **Clause 5 discharges G3's other deliverable**: on the two-level spine with the fine level TILED, max|rho_auto| = 0.0051 vs the 0.15 bar (29x) at BOTH 4 and 8 r_s, identical to four decimals, so **4 r_s is ample and 8 buys nothing**, at 2.3x monolithic compute. |
| D-v2-13 | **V3 verdict: host-resident state streams; the GH200 ceiling is ~116 GB; `staged` is the production path** (JC, 2026-08-06, V3/G4 close; full ADR in `decisions.md`). Discharges both halves of V3's exit. **The seed's coherent/staged/infeasible trichotomy is not expressible** -- jax 0.10.2 + CUDA aarch64 exposes only `['device','pinned_host']`, so ATS on pageable LPDDR is untested and the word "coherent" must not enter the record. Measured instead: state larger than HBM streams at 359-367 GB/s with **4.00 GiB device residency independent of working-set size** (C-gh paint 3.91 B/p vs G5's tiled 24.7 / mono 377), and C-gh's 2048^3 paint runs on one GH200. **The host ceiling is ~116 GB and is a HARD CLIFF** (full rate at 116.0 GB, ~100x collapse at 120.3 GB; physical LPDDR, confirmed by a 4.3 GB bracket and by an S3 h100 running the same rung flat on a 1007 GiB host), so a C-gh operating point must sit CLEAR of it -- C-gh's T9 state is 77.3 GB, ~1.5x under. **`staged` beats the XLA-managed arm 2.78x on the REAL paint while LOSING 14-20% on the ladder** -- the ordering inverts, so the microbenchmark predicts neither ratio nor sign; the XLA-managed path is kept as a memory-vs-wall knob (half the device memory for 2.78x the wall). **C-hero is viable on memory**: one h100 streams 4096^3's full 618.5 GB T9 state flat across a 9x span, at **1.80x** the GH200's wall on real work -- and the 6.71-6.87x FABRIC ratio does NOT propagate. Step-level cost is unmeasured everywhere: the two-level force has only run at cgh64 = C-gh at 1/64 volume. |
| D-v2-14 | **State architecture: T9 at the implementable quantum, on a brick-sorted layout** (JC, 2026-08-08, V4 freeze; full ADR in `decisions.md`). The gated `t9` arm quantized at `fine_cell/256`, which an int8 carries only if its bucket is ONE FINE CELL — a per-cell index costs ~69 GB at C-gh against the 77 GB of state it indexes, so the ratified tier was gated at a resolution no affordable layout can deliver. **Bucket becomes 1.0 Mpc/h, quantum `fine_cell/64`**, 15x margin on D-v2-9's bar with a 2.15 GB index; chosen on margin, NOT on a measured ordering (the ladder is non-monotonic and all four arms pass). **All-in is 10.15 B/p, not 9.** Layout is brick-sorted with per-bucket capacity and eject-and-reinsert exchange; overflow escalates slack -> arena -> refusal and **may never clamp**. Admissible only because the paint is order-independent, which makes D-v2-16 clause 2 a precondition. IDs are an opt-in int32 tier, refused above n_side 1290. |
| D-v2-15 | **IC architecture: disk staging, a 1D transfer table, an out-of-core FFT** (JC, 2026-08-08, V4 freeze; full ADR in `decisions.md`). The IC host term is real at **~90 B/p and FLAT in N** (~773 GB at 2048^3 against a 116 GB ceiling) — never measured before, and the shape prediction was wrong. The fix is cheap and both halves are already in the tree (1D log-k table + `linear_power(backend="table")`). Disk stages the IC stage ONLY; evolve does not spill. **A 2048^3 `rfftn` does not fit a GH200** (ceiling between 1024^3 and 1536^3, workspace 7x the field), so the out-of-core layer is REQUIRED at C-gh and cannot defer to C-hero. Reproducibility is defined against a decomposition, not across one: the canonical noise unit is one XY plane keyed by `fold_in`. |
| D-v2-16 | **Force architecture freeze** (JC, 2026-08-08, V4 freeze; full ADR in `decisions.md`). **The force is never materialized globally** — ownership is a partition so the kick applies tile-locally, deleting the 2 x 206 GB of host arrays that kept C-gh unrunnable. **`paint_tsc_int` is a required deliverable**: `paint_tsc_f64` uses order-dependent f64 `.at[].add` and `--assign-long` defaults to tsc, so the coarse arm ratified in D-v2-10 violates D-006 today. The long-range force is staged as a per-tile coarse sub-block (resident is 12.9 GB at C-gh but 103 GB at C-hero against 96 GB HBM). Geometry T=256/b=32, **PROVISIONAL until measured at C-gh** — the criterion is frozen, the number is not. `cap` is the cost variable with nothing co-varying. The streaming constraint is the HOST GATHER, not the transfer, and the 18x is contingent on an ffi-level pinned gather jax cannot express. **Promotion out of `scripts/v2_g5_core.py` is gated on bitwise parity against the retained probe at three geometries.** |
| D-v2-17 | **Gating scope** (JC, 2026-08-08, V4 freeze; full ADR in `decisions.md`). D-v2-9's bar continues to be read on the **UNCORRECTED** split: gating the corrected quantity would let a calibration absorb an architecture error, and the transfer's own ~1e-3 irreducible residual would move inside the gate rather than beside it. **C-hero is a capacity demonstration only** — no monolithic reference can exist at hero scale, so hero runs measure cost, memory and completion, and accuracy is inherited via D-v2-11's off-box transport with that caveat attached to any data product. |
| D-v2-18 | **Build roadmap** (JC, 2026-08-08, V4 freeze; full ADR in `decisions.md`). Seven rungs M-v2-1..7, codec first because the layout is defined in terms of it and the IC stage emits it. Table + mandatory crosswalk in Sec. 5. |

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

## 2. Requirements chain (PINNED at V0 — D-v2-8, JC 2026-07-15)

Framing caution first (JC): v2 is a GENERAL memory-floor engine. Do not
import goals, framing, or scale regimes from the SPHEREx-focused projects;
disco-mocks + DISCO-DJ (incl. Alex Krolewski's work) own SPHEREx main-line
mocks. The chain below is engine-intrinsic and runs off k_sci alone; the
survey-side inputs examined at V0 are non-normative worklog context.

- **Force / mesh resolution (the root pin).** k_sci = 2.0 h/Mpc: statistics
  numerically clean (D-v2-1: k <= 0.2 k_Nyq of the fine mesh) through the
  k = 1-2 band where bias-model divergence must be attributable to the model,
  not the mesh -> **fine cell <= 0.31 Mpc/h**. With mesh:particle ratio 1-2x
  (G5 measures where the split lands), particle spacing lands at 0.31-0.63
  Mpc/h; a global fine mesh at the hero scale (8192^3, 2.2 TB f32) is
  impossible monolithically -- **the two-level tiled mesh is load-bearing,
  not an optimization**.
- **Mass resolution (DERIVED, not required).** m_p = 2.775e11 * Omega_m *
  (L/N_p)^3 h^-1 Msun (~8.75e10 * (L/N_p in Mpc/h)^3 at Omega_m = 0.3153).
  With n_p = 100 at the floor, the chain gives M_min ~= 2.6e11-2.2e12 Msun/h
  depending on the mesh ratio. Each config REPORTS its M_min; no survey
  frame enters the requirement (D-v2-8 item 4).
- **Bias route = hybrid.** Halo catalogs (FoF/proxy) at calibration configs
  fit/validate the bias model; production runs paint field-level Lagrangian
  bias. Halo finder in scope, off the production critical path.
- **Velocity fidelity = multipole-grade.** Codec + numerics error in P0 AND
  P2 below the fine-mesh PM floor for k <= 0.2 k_Nyq (D-v2-1 applied to
  redshift-space multipoles). This is G2c's velocity gate bar.
- **Statistics + tolerances.** P(k): D-v2-1's band + **D-v2-9's absolute bar
  (|dP/P| <= 3e-2 in-band)**, the floor half having been retired. Squeezed
  bispectrum <= 15%
  (D-v2-7). Calibration configs: HMF within 5% of a calibrated reference,
  halo b1 within 2% at k <= 0.25. Realizations: 100 per production config,
  O(5) at calibration; covariance-grade ensembles out of scope.

### v2 config table (V0; capacity numbers re-derived at V4 with gate-measured B/p)

Named by hardware class. Mesh ratio provisionally 2x (cell = spacing/2);
G5 may tighten it. Coarse mesh = fine/4 (PMFAST pattern). m_p = 1.1e10
Msun/h and M_min(n_p=100) ~= 1.1e12 Msun/h at all three primary configs
(spacing 0.5 Mpc/h, cell 0.25 <= 0.31 Mpc/h).

| config | home | N_p | L [Mpc/h] | fine mesh | coarse |
|---|---|---|---|---|---|
| C-dev | deneb RTX 3050 6 GB / laptop CPU | 256^3 | 128 | 512^3 | 128^3 |
| C-gh | 1x Vista GH200 (96 HBM + 116 LPDDR) | 2048^3 | 1024 | 4096^3 (tiled) | 1024^3 |
| C-hero | 1x S3 H100 node (4x96 HBM + 1 TB + NVMe) | 4096^3 | 2048 | 8192^3 (tiled) | 2048^3 |
| C-vol (optional) | as C-hero | 4096^3 | 2580 | 8192^3 (tiled) | 2048^3 |

C-vol trades mass floor for volume at the bar's edge (spacing 0.63, cell
0.315 ~ the 0.31 limit; M_min 2.2e12); it exists to make the volume<->M_min
slider explicit, not as a commitment. Gate week runs on C-dev; C-gh is V3's
target; C-hero/C-vol are V4 capacity checks.

**Capacity column, re-derived at V4 (D-v2-14 clause 2).** The ratified state
tier costs **10.15 B/p all-in, not 9**: T9 payload 9.00 (int8 x3 positions +
int16 x3 velocities), bucket index 0.25, brick CSR 0.004, slack and arena 0.90.
The bucket is 1.0 Mpc/h (two particle cells), so the index is (L/1.0)^3 uint16
entries.

| config | N_p | T9 payload | bucket index | all-in state | host | headroom |
|---|---|---|---|---|---|---|
| C-dev | 256^3 | 0.15 GB | 0.004 GB | 0.17 GB | — | not binding |
| C-gh | 2048^3 | 77.3 GB | 2.15 GB | **87.2 GB** | ~116 GB LPDDR (a HARD cliff, D-v2-13) | **1.33x under**, not the 1.50x the 9 B/p figure implied |
| C-hero | 4096^3 | 618.5 GB | 17.2 GB | ~697 GB | ~1 TB | ~1.4x under |
| C-vol | 4096^3 | 618.5 GB | 34.4 GB | ~715 GB | ~1 TB | ~1.4x under |

**The 0.90 B/p slack term is an ESTIMATE** from a hand argument about migration
rates -- 7.7 GB of C-gh's 87.2. Measuring it at `cgh64` is an exit condition of
M-v2-1, and a materially larger number moves this table and amends D-v2-14.

## 3. The compute<->memory tradeoff map (D-v2-4)

Instrument, not afterthought. Every gate run reports the triple
**(peak B/p all-in, wall/step, SU per realization-equivalent)** using the v1
mem-profiler discipline (one config per subprocess, `peak_bytes_in_use`,
sweep the discriminating axis) extended with timing + SU accounting.

Running record: `runs/v2/cost_of_memory.md` (a table + one figure), updated
at V1/V2/V3. **Where it is measured (D-v2-10): performance/Pareto points come
from the config-table TACC homes; deneb rows are dev-ground context, never
architecture-deciding.** Expected points, roughly left (cheap-memory) to right:
A3 sequential tiles (x3.2 compute), A4 NVMe-streamed, A2 PCIe-staged,
A2 C2C-streamed (GH), A2 all-HBM, A1 monolithic. Decision rule at V4: stay on
the Pareto front, B/p primary, but a point that buys 2x memory for >5x SU
needs an explicit JC sign-off.

### V4 candidate: f32 for the force MESH and FFT workspace (JC, 2026-07-31)

**Not the state.** D-v2-8 ratified T9 at 9 B/p (int8 cell-relative positions +
int16 velocity); f32 positions and velocities are 24 B/p, so the codec already
beats f32 by 2.7x and there is nothing to win on the state side.

**The mesh is a different term, and it is the one that dominates.** The measured
cgh64 CPU working set is 125 GB at `n_fine = 1024`, against ~48 GB of device
state at 360 B/p. An f64 `1024^3` mesh alone is 8.6 GB and the force needs
several of those plus complex FFT workspace. Halving that term is worth roughly
2x volume at fixed memory, or ~26% more cells per side -- squarely the v2 thesis
(maximize volume x resolution per node, compute freely traded for memory).

**Nothing measured to date bears on it.** `force_global` paints through
`paint_f32(pos, n_mesh, box_size, fdtype=jnp.float64)` and returns f64
regardless of the dtype of the positions handed in (confirmed: f32 positions
in, f64 force out). G3 Stage 3's f32 rung therefore carried the STATE only, and
its 1.1e-6 in `R_Q` licenses nothing about an f32 force solve.

**What it would take.** Thread an `fdtype` through `density_f64` / `force_global`
(the package paint already accepts one), then re-run
`scripts/v2_g3_floors.py` unchanged and read the result against the MESH FLOOR
rather than against zero -- the same discipline D-v2-8's codec ladder used. It
is a change to the running version that D-v2-9's P(k) bar and G6's transfer
calibration were both established at, so it needs its own gate and cannot ride
on theirs.

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
V1-V3 are the "gate week" -- **all CLOSED: V1/V2a/V2b/V3 as of 2026-08-06
(D-v2-8..D-v2-13), and V4 as of 2026-08-08 (D-v2-14..18). The build ladder
M-v2-1..7 is open at Sec. 5; M-v2-1 is in flight**: every premise priced before
the architecture froze. Each seed's prompt is meant to be pasted at that session's start
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

**STATUS (2026-07-16): split into V2a (G5) + V2b (G3). V2a is CLOSED — G5
PASSED, D-v2-10 (records: `runs/v2/g5_kernel_findings.md`,
`runs/v2/g5b_abs_transfer.md`; bar per D-v2-9). The kill branch below was
retired unexercised. G5c (Vista GH200, one job) extends the close-out:
cross-hardware anchor at cdev, box-ladder 1/P coefficient at cgh64, real
(B/p, wall) capacity triples, plus the un-gated G2c full-C-dev rerun —
monolithic on the GH200, NOT via split-force machinery (that dependency was
a deneb-fit contortion, deleted per D-v2-10). V2b (G3) opens next; note the
estimator is a PORT from mbody (`fields.py:174` + `ic.py:151`), not a build.**

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
- **Kill:** G5 split error above the D-v2-9 bar after kernel matching ->
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

**STATUS (2026-08-08): CLOSED. D-v2-14..18 ratified unamended; the exit is
discharged** -- ADRs written, the roadmap section below replaced with the
seven-rung ladder, the capacity table re-derived at 10.15 B/p, and the
crosswalk applied. Record `runs/v2/v4_architecture_record.md` (jobs 896159,
896160, 896408) and `runs/v2/v4_pricing_record.md` (895315/895316/895439).
The seed's own framing survived contact with one exception worth keeping: A3
did NOT pass, so there is no companion product, and the freeze had to reopen
the ratified state tier because G2c had measured representation error while
explicitly deferring storage layout.

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

### V5+ -- build milestones  [instantiated at V4; ratified as D-v2-18, 2026-08-08]

The ladder below REPLACES the four placeholders this section used to carry.
**The codec moves first**, because the layout is defined in terms of it and the
IC stage emits it. Each rung opens with its own detailed plan session.

| id | scope | exit gate |
|---|---|---|
| M-v2-1 | codec + layout: T9 pack/unpack, brick-sorted state, per-bucket capacity, incremental exchange, opt-in ID tier | exact round-trip; wrap-never-clamp asserted; **measured** migrant distribution and slack at cgh64 (D-v2-14 clause 2) |
| M-v2-2 | harden and promote the two-level force; `paint_tsc_int`; the ffi pinned gather | bitwise parity vs the probe at 3 geometries; runtime invariants become tests; regression tests for the three measured bugs |
| M-v2-3 | engine core on T9 state | correctness vs the v1 parity arms where configs overlap |
| M-v2-4 | f32 force mesh | thread `fdtype`, re-run `v2_g3_floors.py` unchanged, read against the MESH FLOOR not zero; own gate, cannot ride on D-v2-9's or G6's |
| M-v2-5 | streamed ICs + out-of-core FFT | tile-IC identity vs monolithic at f64; the transfer table's error < 1e-4 |
| M-v2-6 | capacity | **a complete 2048^3 mock on one Vista gh node** |
| M-v2-7 | output stage | HMF within 5%, halo b1 within 2% at k <= 0.25; squeezed B <= 15%; disco-mocks read-back |

**Crosswalk from the old numbering** (the charter's IDs are referenced
elsewhere and must not read as silently renumbered): old M-v2-1 engine core ->
new M-v2-3; old M-v2-2 tiles+ICs -> new M-v2-5; old M-v2-3 codec+capacity ->
split across new M-v2-1 and M-v2-6; old M-v2-4 output -> new M-v2-7. Applied
2026-08-08 in `runs/v2/cost_of_memory.md`, whose Pallas `atomic_add` deferral
pointed at "M-v2-1" and now points at M-v2-2. NB the gitignored
`g5_results_*.json` cards carry an `M-v2-2` in their `host_bucketing` note under
the OLD meaning, which is new M-v2-1; they are frozen records, left as written.

**Not scheduled, deliberately:** PP-in-tiles (own gate, never a default);
reviving A3 via the frozen-background arm (shelved, D-v2-12); an absolute RSD
bar, which D-v2-8 clause 5 still needs before anything leans on it.

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
| two-level split error above the band | V2/G5 | RESOLVED 2026-07-16: G5 passed (D-v2-10); A1 fallback retired unexercised |
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
