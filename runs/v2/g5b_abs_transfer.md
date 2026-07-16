# G5b -- C-dev's absolute discretization error, and the tolerance it grounds

Running record for the V2a bar. Companion to `g5_kernel_findings.md` (the kernel
study) and `cost_of_memory.md` (the V1-V3 instrument). Deneb job 40, main @
1f37ef1. Figures: `g5_cdev_band.png` (job 39's broken floor),
`g5b_cdev_tolerance.png` (this).

## Why the bar needed replacing

D-v2-1 has two separable halves:

  (a) the BAND -- k <= 0.2 k_Nyq of the fine mesh. INTACT, and independent; it
      is what pins k_sci = 2.0 -> fine cell <= 0.31 (D-v2-8 clause 1).
  (b) the BAR -- "error below the PM error floor". BROKEN: job 39 could not
      measure that floor.

Job 39's floor refined the MESH at FIXED particles (256^3 particles seen by
256/512/1024). Its 1024 rung therefore ran mesh:particle = 4 -- four cells
inside one interparticle gap -- which resolves two-body scattering the 512 run
smooths over. Its (kc)^2 check failed 1.35 (force) / 0.76 (evolved) against ~4,
and decisively: e(256 vs 512) = 0.050 was SMALLER than e(512 vs 1024) = 0.066.
The discrepancy GREW under refinement. Mesh truncation error cannot do that.

**The split error itself was never in question** -- it is measured against mono
at the SAME mesh with the SAME particles, so the discretization is common-mode
and cancels exactly. Only the bar was broken.

## What job 40 measured

A ladder at FIXED mesh:particle = 2 (C-dev's own ratio), matched phase, one
common paint mesh, G5's schedule:

    r128   128^3 / 256   cell 0.500
    r256   256^3 / 512   cell 0.250   <- C-dev
    r512   512^3 / 1024  cell 0.125   <- reference

Matched phase is PROVEN, not assumed: the degenerate limit reproduces
`ic.gaussian_delta` to 5.9e-16 and shared modes agree across rungs to 6-7e-16.
(The shared-mode check earned its keep instantly: it first failed at 0.984/0.875
= exactly |(N_lo/N_hi)^3 - 1|, an unnormalized rfftn in the CHECK. The degenerate
limit passing while shared modes failed is what pinned it to the comparison
rather than the construction.)

### The number

| quantity | in-band max abs dP/P |
|---|---|
| C-dev vs reference, raw | **6.14e-2** |
| C-dev vs reference, Poisson-subtracted | 5.90e-2 |
| one rung coarser (r128 vs C-dev) | 6.62e-2 |
| two-level split (job 39, gauss+TSC+match, tile 128 buf 32) | 2.61e-2 |

The shot bracket is narrow (6.14 vs 5.90e-2), so different particle loads are
NOT driving this.

### Richardson is NOT available -- and why

The k-DEPENDENCE is textbook: C-dev vs reference rises as **k^1.95** over
k = 0.4-1.4. That is the (kc)^2 law, cleanly.

The AMPLITUDE scaling is not: halving the cell drops the error by **2.41x**, not
4x, i.e. as cell^1.27. Joint refinement changes three things at once -- force
resolution, IC bandwidth, particle load -- and only the first follows the mesh
law. Each finer rung also carries an octave of new small-scale IC modes whose
mode-coupling contribution into the band does not shrink as (kc)^2.

So `6.14e-2` is **C-dev's total discretization error vs a 2x-better-resolved
sim**, and since the reference is itself unconverged (cell 0.125), it is a
**LOWER BOUND** on C-dev's error against truth. It is NOT "the mesh transfer"
in the narrow sense. No extrapolation to the converged limit is claimed.

## The finding the max-over-band statistic hides

**The two-level split is the DOMINANT P(k) error for k < 1.53.**

| k | split | C-dev disc. | ratio |
|---|---|---|---|
| 0.05 | 2.69e-4 | 3.66e-5 | **7.37** |
| 0.25 | 3.61e-3 | 7.47e-4 | **4.83** |
| 0.49 | 1.06e-2 | 2.96e-3 | **3.59** |
| 0.98 | 2.19e-2 | 1.02e-2 | 2.15 |
| 1.52 | 2.47e-2 | 2.45e-2 | 1.01 |
| 2.01 | 2.14e-2 | 4.23e-2 | 0.51 |
| 2.50 | 1.75e-2 | 6.14e-2 | 0.28 |

The two curves have different shapes: discretization rises as k^2 all the way,
while the split rises then PLATEAUS at ~2.5e-2 above k ~ 1.4. Both maxima land
at the band edge, where the ordering has already reversed -- so "2.6e-2 vs
6.1e-2, subdominant" is an artifact of the statistic, exactly as job 39's
"headroom 2.5x" was. Comparing maxima of differently-shaped curves is the trap.

In ABSOLUTE terms the low-k split error is small (<= 1.1% for k <= 0.5, 0.36% at
k <= 0.25) and clears every science bar D-v2-8 states -- b1 within 2% at
k <= 0.25 implies ~4% on P(k) there, against the split's 0.36%. The consequence
is narrower than "it fails": the architecture puts a ~1%-at-k=0.5 coherent
suppression under the mock that the mesh error does NOT dominate, so a measured
transfer correction (the omsoc pattern) could not later be used to claim
sub-percent large-scale accuracy. That floor is the architecture's, not the
mesh's.

## Proposed tolerance (D-v2-9) -- JC ratifies; nothing self-ratified

- **Band: unchanged.** k <= 0.2 k_Nyq(fine). D-v2-1's surviving half.
- **Bar: |dP/P| <= 3e-2, ABSOLUTE, in-band**, for the two-level split against
  monolithic at the same mesh. Replaces "below the PM error floor".
- **Mandatory reported diagnostic** (not gated): the split-to-discretization
  ratio vs k. A single max-over-band number provably hides an order-of-magnitude
  low-k excess; the bar must not be readable without its shape.

Rationale for 3e-2:

- **Absolute, so it cannot rot.** It does not re-open the floor question, and it
  transfers across the whole config table unchanged -- D-v2-8 pins fine cell
  <= 0.31 for every config, so C-dev / C-gh / C-hero differ in VOLUME, not
  resolution, and the discretization error at fixed k is the same for all three.
  A floor-relative bar would need re-measuring per config and would re-open this
  contamination each time.
- **Set at ~half the measured lower bound** (6.14e-2) on the config's own
  discretization error, so the split can never dominate the band's high-k end,
  where the mock's bias-model purpose lives (k = 1-2).
- It is NOT grounded on the low-k regime, where the split does dominate: no
  science bar in D-v2-8 demands better there, and inventing one would be
  unsupported.

**Margin is thin and stated as such:** the current config measures 2.61e-2 =
0.87 of the bar. Since 6.14e-2 is a LOWER bound, the true margin is probably
better -- but it is not measured, and the bar should not be read as comfortable.
The knob is tile size (gauss tiles as ~2.0/P; error falls with BIGGER tiles,
which costs memory) -- the V4 Pareto call, not a gate pass/fail.

## Ops notes

- Job 40: CPU-only, deneb, ~22 min wall, --mem=54G. IC gen was cheap (512^3
  2LPT in 17.7s); the evolve legs dominate (r256 114.8s, r512 1096.6s).
- **XLA-CPU thread pool sized from `nproc` (64) inside a 32-core cgroup** ->
  ~2:1 oversubscription of spin-waiting threads; the r256 leg ran 114.8s against
  job 39's 20.6s for the same solve in f64. Correctness unaffected, ~5x wall tax.
  Next run: `-c 64`, or pin the pool (umbrella memory
  `reference_xla_cpu_thread_pool`).
- `PYTHONUNBUFFERED=1` in the sbatch closed the block-buffered-log gap carried
  over from the G5 orchestrator.
