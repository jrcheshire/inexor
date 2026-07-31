"""G3 Stage 4 -- the identity ladder and the two bracket controls.

WHAT A LADDER IS FOR. Each rung differs from the one above by EXACTLY ONE
mechanism, and each has an answer known exactly rather than approximately. If
rung N passes and rung N+1 fails, the single mechanism that changed between them
is the culprit -- you get localization instead of a disappointing number with no
address. G5's discipline, applied to a whole pipeline.

EVERY HARD-FAIL RUNG REFUSES THE RUN. A check that did not run reads exactly
like one that passed, so a rung that cannot be evaluated raises rather than
printing a warning and continuing.

TOLERANCES COME FROM STAGE 3, NOT FROM TASTE. Algebraic rungs (things that are
exact algebra) take the measured f64 instrument floor: bit-repro read 0 exactly
and translation 6.7e-16 in R_Q at cdev, so ALGEBRAIC_TOL is set at 1e-12 in
position terms, ~4 orders above the measured floor and ~10 orders below the bar.

THE TWO BRACKET CONTROLS ARE ABOUT THE GATE, NOT THE CODE.
  kill_control  tiled at b = 0, the maximally broken tiling. If THAT also
                passes 15%, the statistic cannot see tiling failure and the gate
                is void.
  span_check    a pure-2LPT arm with no residual force at all. If the crudest
                possible approximation already lands inside 15%, every sCOLA
                configuration passes trivially and the bar tests nothing.
Together they establish that the bar sits where the gate has both POWER (broken
things fail) and DYNAMIC RANGE (not everything passes). The inequality direction
is DERIVED from the measured |R| values here rather than asserted in advance.

Usage:
    pixi run python scripts/v2_g3_ladder.py --config smoke
    pixi run python scripts/v2_g3_ladder.py --config cdev8 --tile 64 --buf 16
"""

import argparse
import json
import os
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

from v2_g5_two_level_force import A_CONTROL, A_PIVOT, N_STEPS, geometry  # noqa: E402

OUT_DIR = os.path.join(os.path.dirname(HERE), "runs", "v2")

# Algebraic rungs: exact up to f64 association. Stage 3 measured the instrument
# floor at 0 (bit-repro) and 6.7e-16 (translation) in R_Q; this is the position
# -space analogue, set well above roundoff and far below anything physical.
ALGEBRAIC_TOL = 1e-12


class RungFailure(AssertionError):
    pass


def _rms(a):
    return float(np.sqrt((np.asarray(a, np.float64) ** 2).sum(axis=1).mean()))


def _rel_pos(x_a, x_b, box_size):
    """Minimum-imaged rms position difference, relative to the rms displacement."""
    import v2_g3_core as g3

    d = g3.min_image(np.asarray(x_a) - np.asarray(x_b), box_size)
    return _rms(d)


def build(cfg, seed):
    """ICs, frame, coefficients, and the monolithic reference arm."""
    import jax
    import jax.numpy as jnp

    from inexor import ic, lpt
    from inexor.config import Cosmology
    from inexor.cosmology import growth_factor_a
    from inexor.integrate import a_grid, bullfrog_float_coeffs, bullfrog_table, float_step_bullfrog
    from v2_g5_core import force_global

    import v2_g3_core as g3

    g = geometry(cfg)
    ell, n_fine, n_part = g["L"], g["n_fine"], g["n_part"]
    cosmo = Cosmology()
    d0 = ic.linear_density(jax.random.PRNGKey(seed), n_part, ell, cosmo, f_NL=0.0,
                           fdtype=jnp.float64)
    x_ic, v_ic = lpt.lpt_ics(d0, ell, A_CONTROL, cosmo, order=2, fdtype=jnp.float64)
    q, psi1, psi2 = g3.lpt_frame(d0, ell)
    d_init = growth_factor_a(A_CONTROL, cosmo)
    d_final = growth_factor_a(A_PIVOT, cosmo)
    table = bullfrog_table(a_grid(A_CONTROL, A_PIVOT, N_STEPS, "log"), cosmo)

    # THE INITIAL CONDITION IS ASSERTED, NOT ASSUMED. Every tiled number
    # downstream is invalid if y0 or u0 is not identically zero, and assuming it
    # would make that invalidity silent.
    y0, u0, max_y0, max_u0 = g3.frame_residual_at_init(x_ic, v_ic, q, psi1, psi2, d_init, ell)
    if max_y0 != 0.0 or max_u0 != 0.0:
        raise RungFailure(f"frame != ICs at a_init: max|y0| {max_y0:.3e}, max|u0| {max_u0:.3e}")

    def force(pos):
        out, _ = force_global(pos, n_fine, ell, n_part**3, "mono", assign="cic")
        return jnp.asarray(out)

    x, v = jnp.asarray(x_ic, jnp.float64), jnp.asarray(v_ic, jnp.float64)
    for c in bullfrog_float_coeffs(table):
        x, v = float_step_bullfrog(x, v, tuple(np.asarray(c, np.float64)), force, ell)
    x_mono = np.asarray(x, np.float64)

    qi = g3.lagrangian_index(q, n_part, ell)
    coeffs = g3.bullfrog_cola_coeffs(table)
    return dict(g=g, q=q, psi1=psi1, psi2=psi2, qi=qi, coeffs=coeffs, table=table,
                x_mono=x_mono, x_ic=x_ic, v_ic=v_ic, d_init=d_init, d_final=d_final,
                force_mono=force, cosmo=cosmo, max_y0=max_y0, max_u0=max_u0)


# ===========================================================================
# rungs
# ===========================================================================


def rung_A0(B, res):
    """cola_mono vs mono: the COLA algebra, including the half-drift signs."""
    import v2_g3_core as g3

    g = B["g"]
    x_c, _, _ = g3.evolve_cola(
        np.zeros_like(B["q"]), np.zeros_like(B["q"]), B["q"], B["psi1"], B["psi2"],
        B["coeffs"], B["force_mono"], g["L"], B["d_final"],
    )
    rel = _rel_pos(x_c, B["x_mono"], g["L"])
    res["A0_frame_identity"] = dict(rel=rel, tol=ALGEBRAIC_TOL, ok=rel < ALGEBRAIC_TOL)
    return res["A0_frame_identity"]


def rung_frame_zero(B, res):
    """mono-sCOLA with the frame identically zero.

    With Psi1 = Psi2 = 0 the residual stepper reduces ALGEBRAICALLY to
    float_step_bullfrog on (x = y + q, v = u), so this isolates frame plumbing
    and sign conventions from COLA truncation -- a failure here is the frame
    wiring, a failure in A0 with this passing is the COLA algebra.
    """
    import v2_g3_core as g3

    g = B["g"]
    zero = np.zeros_like(B["psi1"])
    y0 = g3.min_image(np.asarray(B["x_ic"]) - B["q"], g["L"])
    u0 = np.asarray(B["v_ic"], np.float64)
    x_c, _, _ = g3.evolve_cola(y0, u0, B["q"], zero, zero, B["coeffs"],
                               B["force_mono"], g["L"], 0.0)
    rel = _rel_pos(x_c, B["x_mono"], g["L"])
    res["frame_zero"] = dict(rel=rel, tol=ALGEBRAIC_TOL, ok=rel < ALGEBRAIC_TOL)
    return res["frame_zero"]


def rung_A1_one_tile(B, res):
    """T = n_fine, b = 0: the tile IS the box, so it must reproduce cola_mono.

    Catches tile bookkeeping, ownership, reassembly and frame add-back in one
    shot, with the physics held identical by construction.
    """
    import v2_g3_core as g3

    g = B["g"]
    x_t, _, d = g3.evolve_scola(
        B["q"], B["psi1"], B["psi2"], B["qi"], B["coeffs"], g["L"], g["n_fine"],
        g["n_part"], g["n_fine"], 0, B["d_final"],
    )
    rel = _rel_pos(x_t, B["x_mono"], g["L"])
    res["A1_one_tile"] = dict(rel=rel, tol=ALGEBRAIC_TOL, ok=rel < ALGEBRAIC_TOL,
                              n_tiles=d["n_tiles"], n_out_core=d["n_out_core_total"])
    return res["A1_one_tile"]


def rung_residual_zero(B, res, n_tile, b_fine):
    """Tiled run with the residual force forced to 0, vs the same frame monolithic.

    No force is computed on either side, so the two arms follow the identical
    analytic frame and any difference is restriction + reassembly alone.
    """
    import v2_g3_core as g3

    g = B["g"]
    x_t, _, _ = g3.evolve_scola(
        B["q"], B["psi1"], B["psi2"], B["qi"], B["coeffs"], g["L"], g["n_fine"],
        g["n_part"], n_tile, b_fine, B["d_final"], zero_residual=True,
    )
    zero_force = lambda p: np.zeros_like(np.asarray(p))  # noqa: E731
    x_m, _, _ = g3.evolve_cola(
        np.zeros_like(B["q"]), np.zeros_like(B["q"]), B["q"], B["psi1"], B["psi2"],
        B["coeffs"], zero_force, g["L"], B["d_final"],
    )
    rel = _rel_pos(x_t, x_m, g["L"])
    res["residual_zero"] = dict(rel=rel, tol=ALGEBRAIC_TOL, ok=rel < ALGEBRAIC_TOL)
    return res["residual_zero"]


def rung_partition(B, res, n_tile, b_fine):
    """Ownership: every particle in exactly one core, and n_tiles > 1.

    n_tiles == 1 in a gate arm is a FAIL, not a pass: a "tiled" run that is
    secretly monolithic reproduces the reference perfectly and certifies nothing.
    """
    import v2_g3_core as g3
    from v2_g5_core import padded_size

    g = B["g"]
    _, b_real = padded_size(n_tile, b_fine, n_fine=g["n_fine"])
    tiles, t_lag, b_lag = g3.lagrangian_tiles(B["qi"], g["n_part"], g["n_fine"], n_tile, b_real)
    n_expect = (g["n_fine"] // n_tile) ** 3
    owner = np.zeros(g["n_part"] ** 3, np.int64)
    for _, idx, cim in tiles:
        owner[idx[np.asarray(cim)]] += 1
    ok = bool(owner.min() == 1 and owner.max() == 1 and len(tiles) == n_expect and len(tiles) > 1)
    res["partition"] = dict(n_tiles=len(tiles), n_expect=n_expect, owner_min=int(owner.min()),
                            owner_max=int(owner.max()), sum_core=int(owner.sum()),
                            n_part_total=int(g["n_part"] ** 3), t_lag=t_lag, b_lag=b_lag, ok=ok)
    return res["partition"]


def rung_buffer_to_box(B, res, n_tile):  # noqa: D401
    """Grow b until P = n_fine: the buffer scan must terminate at the exact answer.

    Establishes that the b-scan is a ONE-PARAMETER FAMILY whose endpoint is the
    monolithic solve, so a nonzero R at finite b is the buffer and nothing else.
    """
    import v2_g3_core as g3
    from v2_g5_core import padded_size

    g = B["g"]
    # COST NOTE. This rung's cost is n_tiles * (P/n_fine)^3 = (P/T)^3, which at
    # b_max is (n_fine/T)^3 -- 64x a monolithic evolve for T = n_fine/4, but 512x
    # for T = n_fine/8. At cdev with T=64 that is 30 h. What the rung certifies
    # (the buffer family terminates at the exact monolithic answer) is a property
    # of the tiling MACHINERY and not of the gate's T, so it is run at its own,
    # larger tile size where the box is big. The gate's T is exercised by
    # buffer_monotonicity and the brackets instead.
    b_max = (g["n_fine"] - n_tile) // 2
    p_side, _ = padded_size(n_tile, b_max, n_fine=g["n_fine"])
    if p_side != g["n_fine"]:
        res["buffer_to_box"] = dict(ok=False, reason=f"P={p_side} != n_fine={g['n_fine']}")
        return res["buffer_to_box"]
    x_t, _, d = g3.evolve_scola(
        B["q"], B["psi1"], B["psi2"], B["qi"], B["coeffs"], g["L"], g["n_fine"],
        g["n_part"], n_tile, b_max, B["d_final"],
    )
    rel = _rel_pos(x_t, B["x_mono"], g["L"])
    res["buffer_to_box"] = dict(rel=rel, tol=ALGEBRAIC_TOL, ok=rel < ALGEBRAIC_TOL,
                                b_max=int(b_max), p_side=int(p_side), n_tiles=d["n_tiles"])
    return res["buffer_to_box"]


def rung_quasi_linear(B, res, n_tile, b_fine):
    """Both arms only to a = A_CONTROL: a constant pipeline offset would survive.

    At the initial time the residual has not grown, so tiled and monolithic must
    agree to roundoff whatever the tiling. A nonzero answer here is an offset in
    the pipeline masquerading as physics at late times.
    """
    import v2_g3_core as g3

    g = B["g"]
    x_t, _, _ = g3.evolve_scola(
        B["q"], B["psi1"], B["psi2"], B["qi"], B["coeffs"][:0], g["L"], g["n_fine"],
        g["n_part"], n_tile, b_fine, B["d_init"],
    )
    rel = _rel_pos(x_t, B["x_ic"], g["L"])
    res["quasi_linear"] = dict(rel=rel, tol=ALGEBRAIC_TOL, ok=rel < ALGEBRAIC_TOL)
    return res["quasi_linear"]


def _stats_vs_mono(x_arm, B, tris, dk=None):
    """R_B / R_Q / rho / W of an arm against the monolithic reference."""
    from inexor import painting
    import v2_g3_floors as fl

    g = B["g"]
    npart, nf, ell = g["n_part"], g["n_fine"], g["L"]

    def dens(x):
        return np.asarray(
            painting.density_contrast(x, nf, ell, npart**3, paint="int"), np.float64
        )

    return fl.stats(dens(x_arm), dens(B["x_mono"]), ell, tris, dk=dk)


def bracket_controls(B, res, n_tile, b_fine, tris, names):
    """kill_control and span_check -- the gate's power and its dynamic range."""
    import v2_g3_core as g3

    g = B["g"]
    out = {}

    # kill_control: tiled at b = 0, the maximally broken tiling.
    #
    # GUARD: b = 0 REQUESTED is not b = 0 REALIZED. padded_size rounds up to the
    # FFT_FRIENDLY ladder, whose smallest entry is 32, so any T below that gets a
    # buffer it never asked for -- measured at T=16, where b = 0, 4 and 8 all
    # collapse to P=32, b_realized=8 and the kill control became bit-identical to
    # the pivot. It then "passed" the power check by comparing a configuration to
    # itself. A kill control that is not maximally broken voids the bracket, so
    # this raises rather than reporting.
    from v2_g5_core import padded_size as _psz

    _, b_kill = _psz(n_tile, 0, n_fine=g["n_fine"])
    if b_kill != 0:
        raise RungFailure(
            f"kill_control is not a kill control: T={n_tile} with b=0 requested rounds to "
            f"b_realized={b_kill} (FFT_FRIENDLY starts at 32), so the 'maximally broken' "
            f"arm carries a real buffer. Use T >= 32 or the bracket is void."
        )
    x_kill, _, _ = g3.evolve_scola(
        B["q"], B["psi1"], B["psi2"], B["qi"], B["coeffs"], g["L"], g["n_fine"],
        g["n_part"], n_tile, 0, B["d_final"],
    )
    s_kill = _stats_vs_mono(x_kill, B, tris)

    # span_check: pure 2LPT, no residual force at all.
    x_2lpt = np.mod(g3.x_lpt(B["q"], B["psi1"], B["psi2"], B["d_final"]), g["L"])
    s_2lpt = _stats_vs_mono(x_2lpt, B, tris)

    # the pivot: the configuration actually under test.
    x_piv, _, _ = g3.evolve_scola(
        B["q"], B["psi1"], B["psi2"], B["qi"], B["coeffs"], g["L"], g["n_fine"],
        g["n_part"], n_tile, b_fine, B["d_final"],
    )
    s_piv = _stats_vs_mono(x_piv, B, tris)

    # r(k) PER SHELL, for every arm. rho divides by the measured transfer, so it
    # is only interpretable where the arms are still correlated -- and a tiled
    # arm CAN decorrelate outright at small scales. Without this column an
    # exploding rho looks like a bug in rho rather than what it is: the
    # conditioning cut doing its job. Stage 3 predicted exactly this and the
    # first cdev8 ladder run hit it (T_prod ~ 1.2e-5 at k_short = 2.36 h/Mpc).
    import v2_g3_floors as fl
    from inexor import painting

    gg = B["g"]
    centers = sorted({float(k) for t in tris for k in t})
    kf_ = 2.0 * np.pi / gg["L"]

    def _d(x):
        return np.asarray(
            painting.density_contrast(x, gg["n_fine"], gg["L"], gg["n_part"] ** 3, paint="int"),
            np.float64,
        )

    d_mono = _d(B["x_mono"])
    rk = {}
    for arm_name, xa in (("pivot", x_piv), ("kill_control", x_kill), ("span_check", x_2lpt)):
        t_sh, r_sh = fl.shell_transfer(_d(xa), d_mono, gg["L"], centers, kf_)
        rk[arm_name] = dict(centers=[float(c) for c in centers],
                            T=[float(v) for v in t_sh], r=[float(v) for v in r_sh])
    out["shell_r"] = rk

    out["kill_control"] = {k: list(s_kill[k]) for k in ("R_B", "R_Q", "rho", "W")}
    out["span_check"] = {k: list(s_2lpt[k]) for k in ("R_B", "R_Q", "rho", "W")}
    out["pivot"] = {k: list(s_piv[k]) for k in ("R_B", "R_Q", "rho", "W")}
    out["names"] = names
    res["brackets"] = out
    return out


def rung_buffer_monotonicity(B, res, n_tile, tris, names, b_list=None):
    """R_Q vs b at fixed T. A REPORT rung, but the one that diagnoses the bracket.

    The kill_control requirement is that a maximally broken tiling (b = 0) be
    WORSE than the configuration under test. If it is not, there are only two
    possibilities and they need separating: either something other than the
    buffer is varying, or the configuration is degenerate and the buffer was
    never the controlling variable. A monotone scan says the buffer is in charge;
    a non-monotone one says read the padded fraction P/n_fine before anything
    else, because a tile that covers most of the box is not a tile.
    """
    import v2_g3_core as g3
    from v2_g5_core import padded_size

    g = B["g"]
    if b_list is None:
        b_max = (g["n_fine"] - n_tile) // 2
        b_list = sorted({0, n_tile // 8, n_tile // 4, n_tile // 2, min(n_tile, b_max)})
        b_list = [b for b in b_list if b <= b_max]
    rows = []
    for b in b_list:
        try:
            p_side, b_real = padded_size(n_tile, b, n_fine=g["n_fine"])
        except ValueError:
            continue
        x_t, _, _ = g3.evolve_scola(
            B["q"], B["psi1"], B["psi2"], B["qi"], B["coeffs"], g["L"], g["n_fine"],
            g["n_part"], n_tile, b, B["d_final"],
        )
        s = _stats_vs_mono(x_t, B, tris)
        # r(k) ALONGSIDE R_Q, because the two can disagree about whether the
        # buffer is helping. R_Q evaluated where the arms have decorrelated is
        # not a bias measurement -- it is the difference of two fields with no
        # phase relation, and its ordering in b is then noise. r is the clean
        # monotone diagnostic; if r rises with b while max|R_Q| wanders, the
        # non-monotonicity is a property of the STATISTIC, not of the physics.
        import v2_g3_floors as fl
        from inexor import painting

        centers = sorted({float(k) for t in tris for k in t})
        d_t = np.asarray(painting.density_contrast(
            x_t, g["n_fine"], g["L"], g["n_part"] ** 3, paint="int"), np.float64)
        d_m = np.asarray(painting.density_contrast(
            B["x_mono"], g["n_fine"], g["L"], g["n_part"] ** 3, paint="int"), np.float64)
        _, r_sh = fl.shell_transfer(d_t, d_m, g["L"], centers, 2.0 * np.pi / g["L"])
        rows.append(dict(b=int(b), b_realized=int(b_real), p_side=int(p_side),
                         p_frac=float(p_side) / float(g["n_fine"]),
                         max_abs_R_Q=float(np.nanmax(np.abs(s["R_Q"]))),
                         r=[float(v) for v in r_sh],
                         r_kshort=float(r_sh[-1]), r_klong=float(r_sh[0]),
                         R_Q=list(s["R_Q"])))
    res["buffer_monotonicity"] = dict(n_tile=int(n_tile), names=names, rows=rows)
    return res["buffer_monotonicity"]


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", default="smoke", choices=("smoke", "cdev8", "cdev", "cgh64"))
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--tile", type=int, default=None, help="T in FINE cells")
    ap.add_argument("--buf", type=int, default=None, help="b in FINE cells")
    ap.add_argument("--long-mults", type=int, nargs="+", default=None)
    ap.add_argument("--k-short-mult", type=int, default=None)
    ap.add_argument("--skip-brackets", action="store_true")
    ap.add_argument("--bufbox-tile", type=int, default=None,
                    help="T for buffer_to_box only; its cost is (n_fine/T)^3 mono-evolves")
    ap.add_argument("--b-scan", type=int, nargs="+", default=None,
                    help="explicit b list for buffer_monotonicity (cost ~ (P/T)^3 each)")
    ap.add_argument("--out-suffix", default="")
    args = ap.parse_args()

    import jax

    jax.config.update("jax_enable_x64", True)
    import v2_g3_floors as fl

    t0 = time.perf_counter()
    print(f"=== G3 Stage 4 identity ladder: {args.config} ===")
    B = build(args.config, args.seed)
    g = B["g"]
    n_tile = args.tile if args.tile else g["n_fine"] // 4
    b_fine = args.buf if args.buf else max(1, n_tile // 4)
    print(f"  L={g['L']} n_part={g['n_part']} n_fine={g['n_fine']}  T={n_tile} b={b_fine}")
    print(f"  frame == ICs at a_init: max|y0| {B['max_y0']:.1e}  max|u0| {B['max_u0']:.1e}")
    print(f"  build {time.perf_counter() - t0:.1f}s")

    res = {}
    hard = [
        ("A0_frame_identity", lambda: rung_A0(B, res)),
        ("frame_zero", lambda: rung_frame_zero(B, res)),
        ("A1_one_tile", lambda: rung_A1_one_tile(B, res)),
        ("partition", lambda: rung_partition(B, res, n_tile, b_fine)),
        ("residual_zero", lambda: rung_residual_zero(B, res, n_tile, b_fine)),
        ("quasi_linear", lambda: rung_quasi_linear(B, res, n_tile, b_fine)),
        ("buffer_to_box", lambda: rung_buffer_to_box(B, res, args.bufbox_tile or n_tile)),
    ]
    print(f"\n  {'rung':>20s} {'value':>13s} {'tol':>10s}  verdict")
    failures = []
    for name, fn in hard:
        t1 = time.perf_counter()
        r = fn()
        val = r.get("rel", float("nan"))
        ok = r["ok"]
        extra = "" if "rel" in r else f"  n_tiles={r.get('n_tiles')}"
        print(f"  {name:>20s} {val:13.3e} {r.get('tol', float('nan')):10.0e}  "
              f"{'PASS' if ok else 'FAIL'}{extra}  [{time.perf_counter() - t1:.1f}s]")
        if not ok:
            failures.append(name)

    if failures:
        raise RungFailure(
            f"hard-fail rungs did not pass: {failures}. Refusing to report bracket "
            "controls or any tiled number -- a ladder rung that fails invalidates "
            "everything below it."
        )

    if not args.skip_brackets:
        kf = 2.0 * np.pi / g["L"]
        tris, names = fl._triangles(kf, args.long_mults, args.k_short_mult)
        print(f"\n  === bracket controls (T={n_tile}) ===")
        br = bracket_controls(B, res, n_tile, b_fine, tris, names)
        print(f"  {'tri':>6s} {'R_Q pivot':>12s} {'R_Q b=0':>12s} {'R_Q 2LPT':>12s}")
        for i, nm in enumerate(names):
            print(f"  {nm:>6s} {br['pivot']['R_Q'][i]:12.4e} "
                  f"{br['kill_control']['R_Q'][i]:12.4e} {br['span_check']['R_Q'][i]:12.4e}")
        mp = float(np.nanmax(np.abs(br["pivot"]["R_Q"])))
        mk = float(np.nanmax(np.abs(br["kill_control"]["R_Q"])))
        m2 = float(np.nanmax(np.abs(br["span_check"]["R_Q"])))
        print(f"\n  {'k':>8s} " + " ".join(f"{'r[' + a[:4] + ']':>11s}"
              for a in ("pivot", "kill", "span")))
        rr = br["shell_r"]
        for i, c in enumerate(rr["pivot"]["centers"]):
            print(f"  {c:8.4f} " + " ".join(f"{rr[a]['r'][i]:11.6f}"
                  for a in ("pivot", "kill_control", "span_check")))
        print("  (rho is only interpretable where r is not near 0; a decorrelated")
        print("   arm makes the measured transfer tiny and rho explodes by construction)")
        print(f"\n  max|R_Q|: pivot {mp:.4f}   kill(b=0) {mk:.4f}   span(2LPT) {m2:.4f}")
        print(f"  gate has POWER      (kill > bar=0.15):  {'YES' if mk > 0.15 else 'NO'}")
        print(f"  gate has RANGE      (2LPT > bar=0.15):  {'YES' if m2 > 0.15 else 'NO'}")
        print(f"  kill exceeds pivot  (|R_kill| > |R_piv|): {'YES' if mk > mp else 'NO'}")
        res["bracket_summary"] = dict(max_pivot=mp, max_kill=mk, max_2lpt=m2,
                                      power=bool(mk > 0.15), rng=bool(m2 > 0.15),
                                      kill_gt_pivot=bool(mk > mp))

        print("\n  === buffer_monotonicity (report rung; diagnoses the bracket) ===")
        bm = rung_buffer_monotonicity(B, res, n_tile, tris, names, b_list=args.b_scan)
        print(f"  {'b':>5s} {'b_real':>7s} {'P':>5s} {'P/n_fine':>9s} {'max|R_Q|':>11s} "
              f"{'r(k_long)':>10s} {'r(k_short)':>11s}")
        for r in bm["rows"]:
            print(f"  {r['b']:5d} {r['b_realized']:7d} {r['p_side']:5d} "
                  f"{r['p_frac']:9.3f} {r['max_abs_R_Q']:11.4e} "
                  f"{r['r_klong']:10.6f} {r['r_kshort']:11.6f}")
        vals = [r["max_abs_R_Q"] for r in bm["rows"]]
        mono = all(vals[i] >= vals[i + 1] for i in range(len(vals) - 1))
        rl = [r["r_klong"] for r in bm["rows"]]
        rs = [r["r_kshort"] for r in bm["rows"]]
        mono_rl = all(rl[i] <= rl[i + 1] for i in range(len(rl) - 1))
        mono_rs = all(rs[i] <= rs[i + 1] for i in range(len(rs) - 1))
        print(f"  max|R_Q| monotone decreasing in b: {'YES' if mono else 'NO'}")
        print(f"  r(k_long)  monotone increasing in b: {'YES' if mono_rl else 'NO'}")
        print(f"  r(k_short) monotone increasing in b: {'YES' if mono_rs else 'NO'}")
        if (mono_rl or mono_rs) and not mono:
            print("  => the buffer DOES help monotonically (r), while max|R_Q| wanders.")
            print("     R_Q evaluated where r ~ 0 is not a bias measurement, so its")
            print("     ordering in b is noise. Read r before reading R_Q.")
        res["bracket_summary"]["r_klong_monotone"] = bool(mono_rl)
        res["bracket_summary"]["r_kshort_monotone"] = bool(mono_rs)
        if not mono:
            print("  NOT monotone -> read P/n_fine first: a tile covering most of the")
            print("  box is not a tile, and the buffer was never the controlling variable.")
        res["bracket_summary"]["buffer_monotone"] = bool(mono)

    os.makedirs(OUT_DIR, exist_ok=True)
    path = os.path.join(OUT_DIR, f"g3_ladder_{args.config}{args.out_suffix}.json")
    res["config"] = dict(cfg=args.config, seed=args.seed, n_tile=n_tile, b_fine=b_fine,
                         **{k: float(v) if isinstance(v, (int, float, np.generic)) else v
                            for k, v in g.items()})
    with open(path, "w") as f:
        json.dump(res, f, indent=2, default=float)
    print(f"\nwrote {path}   [{time.perf_counter() - t0:.1f}s total]")


if __name__ == "__main__":
    main()
