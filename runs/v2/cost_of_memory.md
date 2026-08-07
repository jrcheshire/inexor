# v2 cost-of-memory record (D-v2-4 instrument)

Running record of the compute<->memory tradeoff triples, updated at V1/V2/V3
(plan-plan Sec 3). Every gate run reports **(peak B/p all-in, wall/step, SU
per realization-equivalent)**; deneb rows carry wall only (no SU accounting
on our own cluster -- SU becomes real at V3 on Vista). Decision rule at V4:
stay on the Pareto front, B/p primary; a point buying 2x memory for >5x SU
needs explicit JC sign-off.

Committed by exception from the gitignored `runs/` (git add -f): this is the
V1-V3 running RECORD, not run output -- it must survive machines.

## V1 -- G1 kernel floor (deneb RTX 3050 6 GB)

Peak B/p is all-in over n_part^3 (positions + mesh + transients);
hand-managed = analytic inputs+mesh+output floor (kill line = pallas > 2x).

XLA baselines (job 26; the transient the custom kernels attack):

| op | impl | shape (np:nm) | peak B/p | hand B/p | x hand | wall ms |
|---|---|---|---|---|---|---|
| paint_int | xla | 128:256 | 184.0 | 44.0 | 4.18 | 72.7 |
| paint_f32 | xla | 128:256 | 179.0 | 44.0 | 4.07 | 77.1 |
| gather | xla | 128:256 | 292.0 | 120.0 | 2.43 | 163.6 |
| composed | xla | 128:256 | 285.0 | 88.0 | 3.24 | 163.6 |
| paint_int | xla | 256:512 | 168.0 | 44.0 | 3.82 | 584.2 |
| paint_f32 | xla | 256:512 | 164.0 | 44.0 | 3.73 | 617.5 |
| gather | xla | 256:512 | 260.0 | 120.0 | 2.17 | 1301.6 |
| composed | xla | 256:512 | 263.6 | 88.0 | 3.00 | 1301.6 |
| all ops | xla | 512:512 | OOM (6 GB) | 44-120 | - | - |

Pallas arms (job 35, gpu-pallas env, after the round-2..4 stack fixes --
see "toolchain findings" below). Correctness on CUDA: **pallas paint_int
BIT-IDENTICAL to XLA paint_int (n_diff = 0)** at both shapes, run-to-run 0;
gather/composed EXACTLY equal (max reldiff 0.0); paint_f32 2-5e-7 (add
order). f32-exact-integer guard OK at both shapes.

| op | impl | shape | peak B/p | x hand | wall ms (chunk 4096) | wall ms (chunk 1024) |
|---|---|---|---|---|---|---|
| paint_int | pallas | 128:256 | 148.0 | 3.36 | 508.9 | - |
| paint_f32 | pallas | 128:256 | 132.0 | 3.00 | 357.6 | - |
| gather | pallas | 128:256 | 224.0 | 1.87 | 1037.5 | - |
| composed | pallas | 128:256 | 224.0 | 2.55 | 1037.9 | - |
| paint_int | pallas | 256:512 | 140.0 | 3.18 | 696.3 | **373.7** |
| paint_f32 | pallas | 256:512 | 124.0 | 2.82 | 550.4 | **337.8** |
| gather | pallas | 256:512 | 212.0 | 1.77 | 1259.4 | **634.0** |
| composed | pallas | 256:512 | 212.0 | 2.41 | 1262.4 | **634.4** |

Readings (verdict is JC's):
- Memory: pallas beats XLA on every op/shape (paint 140-148 vs 164-180;
  gather/composed 212-224 vs 257-296 B/p). Wall at chunk 1024: pallas is
  FASTER than XLA (paint 374 vs 584 ms; gather 634 vs 1302). chunk 16384
  fails to launch (register pressure); 4096 is 1.9x slower than 1024.
- Kill-line accounting: raw ratios (2.4-3.4x hand-managed) sit above the
  2x line, BUT the measured peak carries removable bench artifacts: the
  position-component copies (+12 B/p; a production engine stores components
  natively) and the f32->int32 mesh conversion (+32 B/p at 256:512; an
  artifact of the integer-atomic workaround below). Subtracting only those
  two, composed lands ~168 B/p -> ~1.9x. The intrinsic kernel transient
  above the analytic floor is ~20 B/p.
- 512:512 OOMs BOTH impls in this bench (the harness holds correctness
  reference arrays); not a clean capacity statement.

**Toolchain findings (Pallas/Triton on jax 0.10.2, sm_86) -- candidate
upstream reports, JC's call:**
1. conda-forge CUDA jaxlib ships no Pallas GPU lowering at all (jobs 26-29);
   Google pypi jax[cuda12] wheels required (isolated `gpu-pallas` env).
2. jax 0.10 defaults pallas-GPU to Mosaic GPU; where that backend cannot
   import (pre-Hopper), compiler_params=None -> misleading "install jaxlib
   GPU" error. Fix: explicit pltriton.CompilerParams().
3. **plt.atomic_add into int32/uint32 refs is a SILENT NO-OP** (f32 works;
   i64/f64 unsupported; minimal repro 2026-07-15, jobs 33-34's all-zeros
   mesh). Workaround: exact-integer-valued f32 accumulation + a 2^24 cell-sum
   guard -> lossless int32 conversion, order-independent, bit-stable. NB the
   guard binds at production frac_bits=12 x 1e4-particle cells (4.1e7 >
   2^24): production options = fewer frac_bits / CAS loop / jax.ffi kernel /
   upstream fix -- an M-v2-1 decision.
4. lax round_p (jnp.rint) has no Triton lowering; boolean mask algebra in
   one rint reformulation ALSO mis-lowered silently before the all-zeros
   finding superseded it (unconfirmed whether real; the shipped kernel is
   boolean-free regardless).
5. Pallas interpret-mode atomic_add is last-write-wins on duplicate indices
   within a call -- CPU correctness checks need duplicate-free sets.

## V1 -- G2c accumulated codec (deneb RTX 3050 6 GB, job 32)

**Gate config = C-dev/8** (128^3 particles, L = 64, fine 256^3, coarse 64^3
-- same cell/spacing/quanta as C-dev, pre-agreed fallback): C-dev's 512^3
MONOLITHIC-FFT probe force does not fit 6 GB (16/17 workers failed at
CUBIN-load; the monolithic force is precisely what the two-level design
removes, so this is a probe limitation, not a codec verdict). Kaiser
estimator validation: OK (1.4e-3 / 4.5e-3).

Config degeneracy note: at fine mesh 256, int8 cell-relative = 256 x 256 =
2^16 levels = EXACTLY the int16-global lattice, so t9 == t12 identically
here; they differ only at C-dev scale (2^17 levels) and above.

Floors at K=40 (gate band k <= 2.51 h/Mpc): step (40 vs 80) dP/P 4.0e-2,
dP0/P0 2.0e-2, dP2/P0 4.9e-2; **mesh (256 vs 128) dP/P 1.3e-1, dP0/P0
1.1e-1, dP2/P0 1.25e-1 (= the D-v2-1 bar)**.

| tier | state B/p | K | peak B/p | wall/step s | dP/P | dP0/P0 | dP2/P0 | vs mesh floor |
|---|---|---|---|---|---|---|---|---|
| ref | - | 40 | 319 | 0.051 | - | - | - | - |
| t6lin | 6 | 10 | 319 | 0.173 | 4.8e-2 | 3.0e-2 | 6.5e-2 | below (marginal) |
| t6lin | 6 | 20 | 319 | 0.163 | 1.8e-1 | 9.0e-2 | 2.1e-1 | ABOVE |
| t6lin | 6 | 40 | 319 | 0.182 | 4.3e-1 | 2.3e-1 | 5.6e-1 | ABOVE (grows ~K) |
| t6cdf | 6 | 40 | 319 | 0.227 | 1.2e+0 | 1.1e+0 | 1.6e+0 | ABOVE (catastrophic) |
| t9 | 9 | 40 | 319 | 0.052 | 1.3e-4 | 1.4e-4 | 4.9e-4 | below, ~3 orders |
| t12 | 12 | 40 | 319 | 0.052 | 1.3e-4 | 1.4e-4 | 4.9e-4 | below (== t9 here) |

Readings:
- **6 B/p tier fails the accumulated gate** (both variants; error grows ~K;
  per-step re-clipping of the int8 velocity residual compounds -- outlier
  frac 1.8-3.8e-3/step). Kill-line ladder -> fall to 9 B/p.
- **9 B/p tier passes with ~3 orders of margin** at every K, in P0 AND P2,
  and its roundtrip is FREE in wall (0.052 vs 0.051 s/step ref).
- **VERDICT RATIFIED (JC, 2026-07-15): the v2 state tier = T9, 9 B/p**
  (int8 cell-relative positions + int16 velocity), per the D-v2-8 kill
  ladder as planned.
- Evolution peak 319 B/p for every arm = the XLA monolithic force transient
  (G1's quarry); the codec never moves the peak.
- One-shot -> accumulated: t9-class one-shot was 3.1e-5 (G2b int8-cellrel at
  int16-grade); accumulated K=40 = 1.3e-4, x4 -- much kinder than the ~20x
  D-014 anchor suggested. t6's velocity residual is where accumulation bites.

## V2a -- G5 two-level split (deneb job 39; verdict D-v2-10)

Where these numbers may be USED (D-v2-10): performance/Pareto decisions are
made from config-table-home measurements (G5c on Vista and later); the deneb
rows below are the dev-ground record and the scaling-law evidence, not an
operating point. Data: `runs/v2/g5_results_cdev.json` + the job 39 log.

Capacity (GPU f32 force legs, gauss T128/b32 unless noted):

| config | arm | peak B/p | peak MB | note |
|---|---|---|---|---|
| C-dev | mono | - | - | OOM (6 GB) = the capacity result |
| C-dev | tiled (P=192) | 24.7 | 415.2 | absolute working set, box-independent |
| cdev8 | mono | 377.0 | 790.6 | mono fits at 1/8 volume |
| cdev8 | tiled (T64/b16, P=96) | 27.3 | 57.2 | 415/57 ~= (192/96)^3: the P^3 law |

Evolution wall (CPU f64, 20 steps; wall ratio = the padded-volume ratio
(1+2b/T)^3 = 3.375 at both points):

| config | mono s/step | two-level s/step | ratio |
|---|---|---|---|
| C-dev (T128/b32) | 9.9 | 32.2 | 3.23x |
| cdev8 (T64/b16) | 1.03 | 3.54 | 3.44x |

Readings (ratified as D-v2-10, JC 2026-07-16):
- Split evolved error 2.61e-2 vs the 3e-2 D-v2-9 bar at the probe config;
  the split-to-discretization shape (clause 3) read alongside, not just the
  max (`runs/v2/g5b_abs_transfer.md`).
- Error is set by P ALONE (error x P flat at 0.5-0.7 across the (T,b) scan;
  the two P=192 partitions agree to ~10%), memory by P^3 as an absolute
  working set, wall overhead by the buffer fraction. Efficient frontier =
  big tile + minimal buffer; the gate config (0.87 of bar) is a deneb-fit
  artifact, and no production config will run there.
- The 3.4x wall for a >=14x memory ratio is the accepted philosophy trade
  (JC, verbatim in the 07-15 worklog): these sims are memory-expensive, not
  time-expensive; subverting that is the point of v2.

## V3 -- G4 memory path (Vista GH200 job 894036/894118, Stampede3 h100 job 3380722/3380888)

Full record + reading rules: `runs/v2/g4_record.md`. Jobs 894005 (smoke) and
894010 (VOID -- it measured JAX's env defaults, not the hardware) are
provenance, not data.

**The headline for this record: streaming DECOUPLES device residency from
state size.** Device peak is set by the chunk and the mesh, not by N, so the
B/p column stops scaling with the particle count -- which is the entire point
of v2 and the first time it has been demonstrated above the config table's
dev rung.

Memory-path bandwidth, one estimator, matched rungs (2 GiB chunks):

| working set | GH200 staged | GH200 coherent | S3 H100 staged | S3 H100 coherent |
|---|---|---|---|---|
| 64 GiB | 367.0 | 418.0 | 53.4 | 51.2 |
| 88 GiB | 360.0 | 418.5 | 53.4 | 51.2 |
| 104 GiB | 359.1 | 429.2 | 53.5 | 51.2 |
| 128 GiB | **2.4** | **1.6** | 53.4 | 51.2 |
| 256 / 448 / 576 GiB | n/a (over LPDDR) | n/a | 53.8 / 53.8 / 53.8 | 51.3 / 51.2 / 51.2 |

Device residency (the B/p story), constant in working set on BOTH machines:

| arm | device peak | note |
|---|---|---|
| staged / coherent, any rung to 576 GiB | **4.00 GiB (2.0 chunks)** | independent of N |
| hbm control | tracks the set | OOMs past the cap, as designed |
| C-gh paint, 2048^3, either machine | 31.3 GiB = **3.91 B/p** | vs G5 tiled 24.7, mono 377 |

Capacity ceilings, measured:

| machine | HBM (detected) | host ceiling for streaming | mechanism |
|---|---|---|---|
| Vista GH200 | 95.6 GiB | **~116 GB (108 GiB), HARD CLIFF** | physical LPDDR; ~100x drop in one 4 GB step |
| S3 h100 | 93.6 GiB | >= 618.5 GB, no cliff seen | 1006.9 GiB host; 618.5 GB = 61% of it |

Real-workload triple (C-gh paint, 2048^3 into 1024^3, staged):

| machine | wall | particles/s | device peak | fabric |
|---|---|---|---|---|
| Vista GH200 | 13.29 s | 6.46e8 | 31.3 GiB | C2C |
| S3 h100 | 23.90 s | 3.59e8 | 31.3 GiB | PCIe |

Readings (verdict is JC's, sec. 6 of the G4 record):
- **Host-resident state larger than HBM streams**, at 359-367 GB/s (C2C) or
  53-54 GB/s (PCIe), with 4.00 GiB device residency at every rung to 576 GiB.
- **The fabric ratio is 6.71-6.87x but real work is 1.80x.** Paint is compute-
  and latency-bound; on the GH200 it runs 46x below that machine's own ladder
  rate. Do NOT size a config from the bandwidth ratio.
- **The GH200's host ceiling is a hard cliff at physical LPDDR**, not a
  roll-off, so an operating point must sit clear of it. C-gh's T9 state is
  77.3 GB, ~1.5x under.
- **C-hero's full 4096^3 T9 state (618.5 GB) streams at full rate** on one
  H100 with 1 TB host -- flat across a 9x span of working set.
- `coherent` (XLA-managed) beats `staged` (explicit) by 14-20% on C2C and
  loses by 4% on PCIe, so it is a Grace-Hopper statement, not a general one.
- SU rates for the Pareto arithmetic: Vista gh 1 SU/GPU-hr (1 GPU/node, so
  the GPU is the charging unit); S3 h100 4 SU/node-hr across 4 GPUs, so a
  single-GPU job there idles 75% of the billed hardware.

## Older anchors (v1 record, for scale)

- XLA-native forward evolution peaked ~126-138 B/p (v1 R6, m2-results).
- Full adjoint ~440-490 B/p for inexor AND DISCO-DJ (v1; VD-only concern).
- CUBE production all-in: 9.84 B/p (existence proof, design study Sec 3).
