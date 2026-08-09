"""M-v2-4 exit gate: the f32 coarse force mesh.

Probe code, NOT package code. Built in `v2_m3_engine_gate.py`'s mould and
importing its `CONFIGS`, `_geom`, `make_ics`, `_pk` and `demonstrate_determinism`
UNMODIFIED -- reusing its estimator is load-bearing, because it makes E directly
comparable to M-v2-3's codec figure of 7.125e-4 rather than merely similar to it.

WHY NOT THE INSTRUMENT THE CHARTER NAMES. D-v2-18's ladder row says "re-run
`v2_g3_floors.py` unchanged". That cannot be done and would not answer the
question if it could. `v2_g3_floors.py:257` imports `force_global` from
`v2_g5_core` -- the FROZEN PROBE -- so threading a dtype through the package
never reaches it; and it drives that force as `which="mono"`, a single-level
monolithic solve, so it never touches the two-level coarse arm this milestone is
about. Its existing `fdtype` arm casts only the initial `x, v`, and
`float_step_bullfrog` promotes back to f64 on the first operation, so the 1.1e-6
on record measures the f32 rounding of one array and nothing else. Same class as
M-v2-3's gate, whose written exit criterion also could not be read literally.
The bispectrum floors it reports are still used here -- as the pre-registration
anchor, which is the one thing they can honestly do for this milestone.

THE ESTIMAND, stated so it can be checked mechanically:

  E(cfg, K, seed) = max over in-band bins of |P_f32(k) / P_f64(k) - 1|

  - P is `v2_m3_engine_gate._pk`, imported unmodified: integer CIC paint on the
    FINE mesh, |rfftn|^2 * L^3/n^6, binned on `diagnostics._bin_edges`, bins
    with count > 0.
  - band: bin CENTRES with k <= 0.2 * k_Nyq(fine), D-v2-9's band.
  - statistic: MAX over in-band bins. Median and the full dpp_vs_k curve are
    reported beside it, per D-v2-9 clause 3 -- `g5b_abs_transfer.md` found a max
    over a band hiding an order-of-magnitude low-k excess, so the shape is
    mandatory and ungated.
  - denominator: the f64 arm.
  - both arms are `engine.run` on a SlotState built once from f64 ICs and
    deep-copied, identical in every parameter except `coarse_dtype`, with
    `fine_dtype="float64"` pinned on both.

THE BAR IS TWO-TIER, and saying so is the point (JC, 2026-08-09).

  Tier 1 -- the BAR: 0 < E <= 3.0e-3. A budget-share number, not a floor: it is
  10% of D-v2-9's 3e-2 total architecture-error budget, of which the split
  already spends 2.61e-2 and the codec 7.125e-4. Capping one implementation
  choice at a tenth keeps the ordering split >> codec >= f32 intact, and it
  transfers across the config table for the same reason D-v2-9 does (the configs
  differ in volume, not resolution). This is what a pass is reported against.

  Tier 2 -- the pre-registered EXPECTATION, which is what actually
  discriminates: a few x 1e-6, at most 1e-5. Above 1e-4 is a FINDING requiring
  investigation, not an automatic fail.

  Both tiers are stated because the bar alone is nearly unfailable at a 300-3000x
  margin -- which is the same objection that disqualified gating on the 1.3e-1
  mesh floor, and it would have been hidden by quoting one threshold. Setting the
  bar after leg 1 reads was rejected: G3 pinned its estimand before any tiled
  number existed precisely so the choice could not leak the answer.

PRE-REGISTERED, before the numbers exist.

  Leg 1 at cdev K=40 is expected at a few x 1e-6 and at most 1e-5. This is a
  strictly SMALLER perturbation than the only f32 evolution on record --
  `v2_g3_floors.py`'s floor-A rung ran the whole monolithic evolution at f32 and
  read R_Q = 1.1e-6 at cdev8 / 7.7e-7 at cdev (`g3_floors_record.md:206,238`), a
  cubic statistic and so ~3x more sensitive than P(k) -- and comparable to G1's
  f32 paint scatter of 2-5e-7 (`cost_of_memory.md:35`). The dominant remaining
  term is the f32 FFT itself, ~sqrt(log2 N) * eps ~ 2e-7 relative at 1024^3.
  M-v2-4's S3 measured the decode fork to be empty at every config in the table,
  so cancellation in `counts/mean - 1` is NOT expected to contribute.

  Leg 1b is expected to grow no faster than sqrt(K) over K = 20/40/80. Growth
  linear in K or faster means a systematic bias rather than roundoff, and that
  blocks adoption regardless of the bar.

  Leg 1 at cgh64 vs cdev8 matched at K=10 is expected within 3x (the mode-count
  prediction, sqrt(8) = 2.8) rather than 16x (the dynamic-range prediction: k_f
  is 2x smaller so the 1/k^2 kernel is 4x larger at the fundamental). A
  16x-class reading falsifies the transfer to C-gh and makes this a cdev-only
  license.

  Leg 2 is a CENSUS and decides nothing. M-v2-4's S3 settled the decode by
  arithmetic: without the `- 1` the two decode orders are bitwise identical
  always (mean is exactly 8.0 at every config and the frac_bits scale is a power
  of two, so every step is an exact rescaling), and with it they differ only for
  cells needing more than 24 significand bits, i.e. >= 512x the mean, at 6e-8
  relative. What remains worth measuring is the occupancy extreme-value law over
  volume, as input to C-gh and C-hero. Counting is by ROUND-TRIP failure, not by
  a 2^24 magnitude threshold: `5000 * 2^12 = 625 * 2^15` is exact at 2.05e7, so
  a magnitude threshold overstates the loss.

  Leg 3 is expected at 1.9-2.0x on the coarse force meshes, NOT 2.0x overall:
  the int64 accumulator, the int32 paint and the f64 kernel BUILD do not change
  dtype, and the build alone is 12.0 GiB at C-gh. Its three arms separate the
  dtype saving from S6's slabbed decode, which is a memory improvement in the
  f64 arm too and must not be credited to f32.

Legs, and where each runs:

  0  dtype ledger + determinism        smoke        precondition, exits non-zero
  1  accuracy, paired ICs              cdev (gated) Vista; cdev8/cgh64 anchors
  1b K-ladder 20/40/80                 cdev8        deneb
  2  decode census                     rides leg 1  reported
  3  coarse memory ladder, THREE arms  synthetic    reported
  4  floors: step and coarse mesh      cdev8        reported, ladder context
  5  fine-arm reading                  cdev8        measured, NOT gated
  6  f64 regression                    cdev8        `v2_m3_engine_gate.py`
                                                    verbatim, in the sbatch
"""

import argparse
import copy
import json
import os
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(HERE)
OUT_DIR = os.path.join(REPO, "runs", "v2")

sys.path.insert(0, HERE)
import v2_m3_engine_gate as m3  # noqa: E402

CONFIGS = m3.CONFIGS
A_INIT, A_FINAL, SPACING, SEED, ALPHA = m3.A_INIT, m3.A_FINAL, m3.SPACING, m3.SEED, m3.ALPHA

BAR = 3.0e-3                 # tier 1, budget share
EXPECT_MAX = 1.0e-5          # tier 2, pre-registered expectation
FINDING_ABOVE = 1.0e-4       # tier 2, the level that demands investigation
MIN_BAND_BINS = 8


def _engine_config(g, coarse_dtype, fine_dtype="float64", slack=0.10):
    from inexor import engine

    return engine.EngineConfig(
        box_size=g["L"], n_part=g["n_part"], n_fine=g["n_fine"], n_coarse=g["n_coarse"],
        n_tile=g["tile"], b_fine=g["buf"], alpha=ALPHA, brick_slack=slack,
        coarse_dtype=coarse_dtype, fine_dtype=fine_dtype,
    )


# ===========================================================================
# leg 0 -- the precondition
# ===========================================================================


def leg_dtype_ledger(g, slack=0.10, arena_frac=0.02):
    """Every coarse-arm intermediate's REALIZED dtype, at both settings.

    This is the precondition, and it exists because the milestone's entire
    failure mode is a knob that reports success without applying. Three separate
    dtype parameters in this package turned out not to apply while M-v2-4 was
    being built: `kernel_grids`' died at `split_kernels`, the gathers'
    `dtype=gx.dtype` was defeated by f64 weights promoting the accumulation, and
    `make_tile_force_fn` dropped `tile_delta_from_int`'s by calling it
    positionally. A dtype parameter on this codebase is not evidence of a dtype
    path, so the arm is read rather than trusted.
    """
    import jax.numpy as jnp

    from inexor import engine, forces, state
    from inexor.codec import T9Layout

    x, v, _ = m3.make_ics(g)
    t9 = T9Layout(box_size=g["L"], n_part=g["n_part"], bucket_cells=2)
    out = {}
    for name in ("float64", "float32"):
        ec = _engine_config(g, name, slack=slack)
        ec.validate()
        st = state.SlotState.build(
            x, v, t9, g["n_fine"] // ec.n_brick, brick_slack=slack, arena_frac=arena_frac
        )
        t0 = time.perf_counter()
        delta = engine.coarse_delta_streamed(st, ec)
        meshes = forces.coarse_force_meshes(
            jnp.asarray(delta), ec.n_coarse, ec.box_size, "long", r_s=ec.r_s,
            match=(ec.coarse_cell, ec.fine_cell), fdtype=ec.np_coarse_dtype,
        )
        origin, extent = forces.coarse_subblock_origin_extent(
            (0, 0, 0), ec.n_tile, ec.n_coarse, ec.n_fine
        )
        sub = [forces.stage_coarse_subblock(np.asarray(m), origin, extent) for m in meshes]
        wall = time.perf_counter() - t0
        out[name] = dict(
            delta=np.dtype(delta.dtype).name,
            force_mesh=np.dtype(np.asarray(meshes[0]).dtype).name,
            staged_subblock=np.dtype(sub[0].dtype).name,
            solve_wall_s=wall,
        )
    want = {"float64": "float64", "float32": "float32"}
    ok = all(
        out[n][k] == want[n]
        for n in want for k in ("delta", "force_mesh", "staged_subblock")
    )
    return dict(
        ledger=out, ok=ok,
        wall_ratio_f64_over_f32=float(
            out["float64"]["solve_wall_s"] / max(out["float32"]["solve_wall_s"], 1e-12)
        ),
    )


# ===========================================================================
# leg 1 -- accuracy, and leg 2 riding on it
# ===========================================================================


def _run_arm(g, x, v, coeffs, coarse_dtype, fine_dtype, slack, arena_frac, census=False):
    from inexor import engine, state
    from inexor.codec import T9Layout

    ec = _engine_config(g, coarse_dtype, fine_dtype, slack=slack)
    ec.validate()
    t9 = T9Layout(box_size=g["L"], n_part=g["n_part"], bucket_cells=2)
    st = state.SlotState.build(
        x, v, t9, g["n_fine"] // ec.n_brick, brick_slack=slack, arena_frac=arena_frac
    )
    seen = []
    t0 = time.perf_counter()
    out = engine.run(st, ec, coeffs, collect=seen.append, census=census)
    st.check()
    wall = time.perf_counter() - t0
    xs = np.concatenate([st.decode_brick(b)[1] for b in range(st.n_bricks)])
    return xs, wall, out, seen


def leg_accuracy(cfg, g, k_steps, seed=SEED, slack=0.10, arena_frac=0.02,
                 fine_arm=False, census=True):
    """f32 coarse vs f64 coarse, same ICs, everything else identical.

    `fine_arm=True` moves the FINE knob instead, holding coarse at f64. That is
    leg 5: measured, not gated, because the fine mesh is P^3 and box-independent
    so it carries no capacity argument -- what it tells M-v2-6 is whether the
    per-tile working set can halve too.
    """
    from inexor.integrate import a_grid, bullfrog_float_coeffs, bullfrog_table

    x, v, cosmo = m3.make_ics(g, seed=seed)
    a_steps = a_grid(A_INIT, A_FINAL, k_steps, SPACING)
    co = bullfrog_float_coeffs(bullfrog_table(a_steps, cosmo))

    if fine_arm:
        arms = (("float64", "float64"), ("float64", "float32"))
    else:
        arms = (("float64", "float64"), ("float32", "float64"))

    # the reference arm first, and the SAME x, v handed to both -- deep-copied
    # so neither run can mutate what the other starts from. Threading a dtype
    # into IC generation would change the jax.random STREAM and give the two
    # arms different realizations; that is a recorded catastrophe
    # (`g3_floors_record.md:101-111`), which is why ICs are f64 here always.
    xr, wall_r, _, seen_r = _run_arm(
        g, copy.deepcopy(x), copy.deepcopy(v), co, arms[0][0], arms[0][1],
        slack, arena_frac, census=census,
    )
    xt, wall_t, _, _ = _run_arm(
        g, copy.deepcopy(x), copy.deepcopy(v), co, arms[1][0], arms[1][1],
        slack, arena_frac, census=False,
    )

    k, p_r = m3._pk(xr, g["n_fine"], g["L"], g["n_part"] ** 3)
    _, p_t = m3._pk(xt, g["n_fine"], g["L"], g["n_part"] ** 3)
    k_nyq = np.pi * g["n_fine"] / g["L"]
    band = k <= 0.2 * k_nyq
    dpp = np.abs(p_t[band] / p_r[band] - 1.0)
    e_max = float(dpp.max())
    n_bins = int(band.sum())

    # leg 2 rides the reference arm's per-step stats
    census_out = None
    if census and seen_r:
        peaks = [int(s["coarse_peak_int"]) for s in seen_r if "coarse_peak_int" in s]
        inex = [int(s.get("coarse_cells_inexact_f32", 0)) for s in seen_r]
        census_out = dict(
            peak_int_max=max(peaks) if peaks else None,
            peak_int_final=peaks[-1] if peaks else None,
            implied_max_cell_particles=(max(peaks) / 2.0**12) if peaks else None,
            cells_inexact_f32_max=max(inex) if inex else None,
            n_coarse_cells=g["n_coarse"] ** 3,
            note="counted by f32 round-trip failure, NOT by a 2^24 magnitude "
                 "threshold; see the module docstring",
        )

    # NAME the condition that fired. A bare "vacuous" sends the reader to the
    # source; and the conditions mean very different things -- an identical
    # final state means the knob did nothing, while too few bins means the
    # CONFIG cannot host the measurement (which is `smoke`, always, by design).
    why = []
    if e_max == 0.0:
        why.append("E is exactly zero: the two arms produced identical spectra")
    if np.array_equal(xr, xt):
        why.append("the two arms' final positions are bitwise equal: the knob did nothing")
    if n_bins < MIN_BAND_BINS:
        why.append(f"only {n_bins} in-band bins against the {MIN_BAND_BINS} minimum: this "
                   "config's band is too narrow to read a spectrum from")
    vacuous = bool(why)
    return dict(
        config=cfg, k=k_steps, seed=seed,
        arms=dict(reference=dict(coarse=arms[0][0], fine=arms[0][1]),
                  test=dict(coarse=arms[1][0], fine=arms[1][1])),
        estimand="max over in-band bins of |P_test(k)/P_reference(k) - 1|; band is bin "
                 "centres with k <= 0.2 k_Nyq(fine); P from v2_m3_engine_gate._pk",
        k_gate=float(0.2 * k_nyq),
        n_band_bins=n_bins,
        E_max=e_max,
        E_median=float(np.median(dpp)),
        dpp_vs_k=[[float(a), float(b)] for a, b in zip(k[band], dpp)],
        bar=BAR, expect_max=EXPECT_MAX, finding_above=FINDING_ABOVE,
        margin_vs_bar=float(BAR / max(e_max, 1e-30)),
        within_expectation=bool(e_max <= EXPECT_MAX),
        is_finding=bool(e_max > FINDING_ABOVE),
        vacuous=vacuous, vacuous_why=why,
        ok=bool(0.0 < e_max <= BAR and not vacuous),
        wall_reference_s=wall_r, wall_test_s=wall_t,
        census=census_out,
        gated=not fine_arm,
    )


# ===========================================================================
# leg 3 -- the coarse memory ladder, three arms
# ===========================================================================


def _decode_whole_array(mesh, frac_bits, mean, fdtype):
    """The PRE-M-v2-4 decode, reconstructed so its cost can be measured.

    S6 replaced this with a slabbed form. Both are elementwise and
    `tests/test_force_dtypes.py` asserts they are bitwise equal, so this is a
    fair cost comparison rather than a different computation. It is here and not
    in the package because the package should not carry a worse version of
    something for the sake of benchmarking it.
    """
    import jax.numpy as jnp

    from inexor.painting import counts_from_int

    counts = counts_from_int(mesh.astype(np.int32), frac_bits, fdtype=jnp.float64)
    return (np.asarray(counts) / mean - 1.0).astype(fdtype)


ARMS = ("f64_whole", "f64_slab", "f32_slab")


def _rss_bytes():
    import resource

    r = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return r if sys.platform == "darwin" else r * 1024


def ladder_worker(n, arm, frac_bits=12):
    """ONE measurement, in a fresh process. Prints a JSON line on stdout.

    A fresh process per measurement is not fastidiousness: `ru_maxrss` is a
    HIGH-WATER mark that never resets, so several arms measured in one process
    report a monotonically rising number and the arm that ran LAST reads
    highest whatever it actually cost. Built that way first here, and the f32
    arm duly came out "worse" than f64 at every rung. Same shape as the
    `peak_bytes_in_use` trap already on record for jax.
    """
    import jax.numpy as jnp

    from inexor import forces

    mean = 8.0
    rng = np.random.default_rng(0)
    mesh = (rng.poisson(mean, size=(n, n, n)) * 2**frac_bits).astype(np.int64)
    dt = np.float32 if arm.startswith("f32") else np.float64
    base = _rss_bytes()
    if arm == "f64_whole":
        delta = _decode_whole_array(mesh, frac_bits, mean, dt)
    else:
        out = np.empty((n, n, n), dtype=dt)
        slab = max(1, min(n, 32))
        for i0 in range(0, n, slab):
            s = mesh[i0 : i0 + slab]
            out[i0 : i0 + slab] = s.astype(np.float64) * 2.0**-frac_bits / mean - 1.0
        delta = out
    after_decode = _rss_bytes()
    t0 = time.perf_counter()
    g = forces.coarse_force_meshes(
        jnp.asarray(delta), n, 1.0 * n, "long", r_s=1.0, fdtype=np.dtype(dt)
    )
    g = [np.asarray(x) for x in g]
    wall = time.perf_counter() - t0
    return dict(
        n_coarse=n, arm=arm, dtype=np.dtype(dt).name,
        force_mesh_dtype=np.dtype(g[0].dtype).name,
        rss_after_mesh_bytes=base, rss_after_decode_bytes=after_decode,
        peak_rss_bytes=_rss_bytes(), solve_wall_s=wall,
    )


def leg_mesh_ladder(rungs, frac_bits=12):
    """Coarse-arm peak host memory over n_coarse, at three arms, one process each.

    THREE arms, not two, and this is the correction that keeps the reading
    honest: S6's slabbed decode improves the f64 arm's OWN peak, so an
    f64-before vs f32-after comparison would credit the dtype with a change that
    has nothing to do with dtype.

      f64_whole   the pre-M-v2-4 decode at f64   (what the engine used to pay)
      f64_slab    the shipped decode at f64      (what it pays today)
      f32_slab    the shipped decode at f32      (what the milestone buys)

    Synthetic delta rather than a particle run, deliberately: the quantity
    scales as n_coarse^3 and not as N, and at cdev the coarse mesh is 16 MB per
    array against a ~330 MB interpreter baseline -- unreadable. This drives the
    real `coarse_force_meshes` on a mesh of the right shape, which is the term
    under test.
    """
    import subprocess

    rows = []
    for n in rungs:
        for arm in ARMS:
            cmd = [sys.executable, os.path.abspath(__file__), "--leg", "ladder-worker",
                   "--worker-n", str(n), "--worker-arm", arm]
            p = subprocess.run(cmd, capture_output=True, text=True)
            line = [ln for ln in p.stdout.splitlines() if ln.startswith("{")]
            if p.returncode != 0 or not line:
                rows.append(dict(n_coarse=n, arm=arm, failed=True,
                                 returncode=p.returncode, stderr=p.stderr[-2000:]))
                print(f"    n={n:5d} {arm:10s} FAILED rc={p.returncode} "
                      f"(an OOM here is a datum, not a defect)", flush=True)
                continue
            row = json.loads(line[-1])
            rows.append(row)
            print(f"    n={n:5d} {arm:10s} peak {row['peak_rss_bytes'] / 1024**3:6.2f} GiB"
                  f"  solve {row['solve_wall_s']:6.2f}s", flush=True)
    # the readings that matter are RATIOS within a rung, across arms
    ratios = {}
    for n in rungs:
        got = {r["arm"]: r for r in rows if r.get("n_coarse") == n and not r.get("failed")}
        if {"f64_slab", "f32_slab"} <= set(got):
            ratios[str(n)] = dict(
                f64_slab_over_f32_slab=float(
                    got["f64_slab"]["peak_rss_bytes"] / got["f32_slab"]["peak_rss_bytes"]
                ),
                f64_whole_over_f64_slab=(
                    float(got["f64_whole"]["peak_rss_bytes"]
                          / got["f64_slab"]["peak_rss_bytes"])
                    if "f64_whole" in got else None
                ),
            )
    return dict(
        rows=rows, ratios=ratios,
        note="one SUBPROCESS per (rung, arm): ru_maxrss is a high-water mark and never "
             "resets, so arms measured in one process report a monotone rise and the last "
             "arm always reads highest. Read the ratios, not the absolute peaks -- those "
             "carry a ~0.25 GiB interpreter+jax baseline.",
        reading="`f64_whole_over_f64_slab` is the arm that keeps the dtype reading honest, "
                "and it is expected to be ~1.0: the slabbed decode shrinks the DECODE "
                "phase but the process high-water mark is set later, by the solve (kernel "
                "build + kernels + dk + three force meshes), and the decode transient is "
                "freed before any of that is allocated. So slabbing is not a peak "
                "reduction and must not be reported as one -- it would become one only if "
                "the solve's own peak came down, e.g. by caching the coarse kernels. "
                "`f64_slab_over_f32_slab` is the milestone's number and should approach "
                "2.0 from below as the fixed baseline dilutes.",
    )


# ===========================================================================
# leg 4 -- the floors ladder
# ===========================================================================


def leg_floors(cfg, g, k_steps, slack=0.10, arena_frac=0.02):
    """The context rungs, all at f64, so leg 1 is read against something.

    The step floor (K vs 2K) is `v2_g2c_accum_gate`'s discipline. The COARSE
    MESH floor (n_coarse vs n_coarse/2) does not exist anywhere on record --
    G2c's 1.3e-1 is the FINE mesh floor -- and it is the only "mesh floor" that
    is about the arm this milestone changes. Measuring it is what makes the
    charter's "read against the MESH FLOOR" instruction mean something.
    """
    from inexor.integrate import a_grid, bullfrog_float_coeffs, bullfrog_table

    x, v, cosmo = m3.make_ics(g)
    k_nyq = np.pi * g["n_fine"] / g["L"]

    def _evolve(gg, k):
        a_steps = a_grid(A_INIT, A_FINAL, k, SPACING)
        co = bullfrog_float_coeffs(bullfrog_table(a_steps, cosmo))
        xs, _, _, _ = _run_arm(gg, copy.deepcopy(x), copy.deepcopy(v), co,
                               "float64", "float64", slack, arena_frac)
        return m3._pk(xs, gg["n_fine"], gg["L"], gg["n_part"] ** 3)

    def _dpp(a, b):
        (ka, pa), (_, pb) = a, b
        band = ka <= 0.2 * k_nyq
        return float(np.abs(pa[band] / pb[band] - 1.0).max())

    ref = _evolve(g, k_steps)
    step = _dpp(ref, _evolve(g, 2 * k_steps))
    g_half = dict(g)
    g_half["n_coarse"] = g["n_coarse"] // 2
    coarse = _dpp(ref, _evolve(g_half, k_steps))
    return dict(config=cfg, k=k_steps,
                step_floor_K_vs_2K=step,
                coarse_mesh_floor=coarse,
                n_coarse=g["n_coarse"], n_coarse_half=g_half["n_coarse"],
                note="both at f64; context for leg 1, never a bar")


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--config", default="smoke", choices=sorted(CONFIGS))
    ap.add_argument("--leg", default="ledger",
                    choices=("ledger", "accuracy", "fine", "ladder", "floors",
                             "ladder-worker"))
    ap.add_argument("--worker-n", type=int, default=None, help=argparse.SUPPRESS)
    ap.add_argument("--worker-arm", default=None, help=argparse.SUPPRESS)
    ap.add_argument("--k", type=int, default=40)
    ap.add_argument("--seed", type=int, default=SEED)
    ap.add_argument("--tile", type=int, default=None)
    ap.add_argument("--buf", type=int, default=32)
    ap.add_argument("--slack", type=float, default=0.10)
    ap.add_argument("--arena-frac", type=float, default=0.02)
    ap.add_argument("--rungs", type=int, nargs="+", default=[128, 256, 512])
    ap.add_argument("--no-census", action="store_true")
    ap.add_argument("--out-suffix", default="")
    args = ap.parse_args()

    import jax

    jax.config.update("jax_enable_x64", True)

    # the ladder's per-measurement worker: one measurement, one process, JSON on
    # stdout, and NO determinism precondition (it paints nothing)
    if args.leg == "ladder-worker":
        print(json.dumps(ladder_worker(args.worker_n, args.worker_arm)), flush=True)
        return 0

    g = m3._geom(args.config, args.tile, args.buf)
    print(f"[m4-gate] {args.config} leg={args.leg} tile={g['tile']} buf={g['buf']} "
          f"backend={jax.devices()[0].platform}", flush=True)

    # --- precondition, before every leg
    det = m3.demonstrate_determinism(g)
    print(f"  determinism: {det['n_diff']} differing cells over {det['occupied_cells']} "
          f"occupied -> {'OK' if det['ok'] else 'FAIL'}", flush=True)
    if not det["ok"]:
        print("FATAL: the paint is not order-independent on this build.", flush=True)
        return 1

    t0 = time.perf_counter()
    if args.leg == "ledger":
        res = leg_dtype_ledger(g, args.slack, args.arena_frac)
        for name, row in res["ledger"].items():
            print(f"  {name}: delta={row['delta']} mesh={row['force_mesh']} "
                  f"sub={row['staged_subblock']} solve {row['solve_wall_s']:.2f}s", flush=True)
        print(f"  wall f64/f32 = {res['wall_ratio_f64_over_f32']:.2f}x "
              f"-> {'OK' if res['ok'] else 'FAIL'}", flush=True)
        if not res["ok"]:
            print("FATAL: an arm did not carry the dtype it was configured with. Nothing "
                  "below would be readable -- an f32 arm silently running f64 reads as a "
                  "spectacular pass.", flush=True)
            res["determinism"] = det
            _write(res, args)
            return 1
    elif args.leg in ("accuracy", "fine"):
        res = leg_accuracy(args.config, g, args.k, seed=args.seed, slack=args.slack,
                           arena_frac=args.arena_frac, fine_arm=(args.leg == "fine"),
                           census=not args.no_census)
        verdict = "PASS" if res["ok"] else ("VACUOUS" if res["vacuous"] else "FAIL")
        print(f"  E = {res['E_max']:.3e} (median {res['E_median']:.3e}) over "
              f"{res['n_band_bins']} in-band bins", flush=True)
        print(f"  tier 1 bar {BAR:.1e}: {res['margin_vs_bar']:.0f}x margin -> {verdict}",
              flush=True)
        for w in res["vacuous_why"]:
            print(f"    vacuous: {w}", flush=True)
        print(f"  tier 2 expectation <= {EXPECT_MAX:.1e}: "
              f"{'within' if res['within_expectation'] else 'OUTSIDE'}"
              f"{'  ** FINDING **' if res['is_finding'] else ''}", flush=True)
        if res["census"]:
            c = res["census"]
            print(f"  census: peak cell sum {c['peak_int_max']} "
                  f"(~{c['implied_max_cell_particles']:.0f} particles), "
                  f"{c['cells_inexact_f32_max']} of {c['n_coarse_cells']} cells inexact "
                  f"in f32", flush=True)
    elif args.leg == "ladder":
        res = leg_mesh_ladder(args.rungs)
    else:
        res = leg_floors(args.config, g, args.k, args.slack, args.arena_frac)
        print(f"  step floor (K vs 2K)  {res['step_floor_K_vs_2K']:.3e}", flush=True)
        print(f"  coarse mesh floor     {res['coarse_mesh_floor']:.3e}", flush=True)

    res["determinism"] = det
    res["wall_s"] = time.perf_counter() - t0
    _write(res, args)
    return 0


def _write(res, args):
    import subprocess

    try:
        res["commit"] = subprocess.check_output(
            ["git", "-C", REPO, "rev-parse", "HEAD"], text=True
        ).strip()
    except Exception:
        res["commit"] = None
    res["slurm_job_id"] = os.environ.get("SLURM_JOB_ID")
    os.makedirs(OUT_DIR, exist_ok=True)
    path = os.path.join(OUT_DIR, f"m4_gate_{args.config}_{args.leg}{args.out_suffix}.json")
    with open(path, "w") as fh:
        json.dump(res, fh, indent=1)
    print(f"  card -> {path}", flush=True)


if __name__ == "__main__":
    sys.exit(main())
