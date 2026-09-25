"""The two-level force against Newton: radial profile of one particle's pull.

Every force test before this one is a parity or an identity -- bitwise the
ratified probe, or long + short == the single-level fine-mesh PM. None of them
compares the force with 1/r^2, so an error the fine-mesh PM shares, or one sitting
at the split scale r_s, is invisible to all of them. So is it to every accuracy
ladder on record, because each one held r_s fixed at 1.0 Mpc/h by design.

    pixi run python scripts/v2_force_profile.py --config cdev8 -o figures/force_profile.png

WHAT IS MEASURED. A source particle at a random point, test particles at
log-spaced separations in random directions. The force is linear in the density
and the integer paint rounds per particle, so

    g_source(at tests) = g(source + tests) - g(tests alone)

holds EXACTLY, and the test particles' own mass cancels. Each arm is the call the
M-v2-3 gate uses as its ratified path (`v2_m3_engine_gate.force_fn`), which the
engine's force is bitwise equal to (D-v2-21):

    long   force_global(which="long", assign="tsc", paint="int",
                        match=(coarse_cell, fine_cell))
    short  force_short_tiled(paint="int")

plus `which="mono"` on the fine mesh, the single-level PM the split was gated
against.

THE REFERENCE IS EWALD, and it splits EXACTLY into the two arms' targets. The
split S(k) = exp(-k^2 r_s^2) is Ewald's with a = 1 / (2 r_s): the reciprocal sum
at that a is the long arm's continuum force and the real-space (erfc) sum is the
short arm's. So each arm is scored against its own target, and a defect is
located in the arm that carries it. The total is a second Ewald sum at a
parameter chosen for convergence, and its independence of that parameter is
checked, as is its approach to Newton at small r.

CONVENTIONS (`forces.py`): div g = -delta, delta = counts / mean - 1 with
mean = n_total / n_mesh^3, so one particle is a point mass m = L^3 / n_total and
g = -m r_vec / (4 pi r^3) near it. `F_r` below is the ATTRACTIVE radial
component, positive toward the source.
"""

import argparse
import itertools
import json
import os
import sys

import numpy as np
from scipy.special import erfc

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(REPO, "src"))
sys.path.insert(0, os.path.join(REPO, "scripts"))


# --- the continuum reference ------------------------------------------------


def _min_image(d, L):
    return d - L * np.rint(d / L)


def ewald_real(d, L, a, n_img=1):
    """Real-space (erfc) part of the periodic force of a unit point mass, per
    unit m, at displacements d (n,3) from the source. Attractive: -r_vec/..."""
    g = np.zeros_like(d)
    for n in itertools.product(range(-n_img, n_img + 1), repeat=3):
        rv = d + L * np.asarray(n, float)
        r = np.linalg.norm(rv, axis=1)
        f = (erfc(a * r) + 2.0 * a * r / np.sqrt(np.pi) * np.exp(-(a * r) ** 2)) / r**3
        g -= rv * f[:, None]
    return g / (4.0 * np.pi)


def ewald_recip(d, L, a, n_max):
    """Reciprocal part, per unit m, with the uniform background removed (k=0
    dropped). g_k = i k delta_k / k^2 with delta_k = e^{-k^2/4a^2} / L^3."""
    g = np.zeros_like(d)
    ns = np.arange(-n_max, n_max + 1)
    for nx in ns:
        k = 2.0 * np.pi / L * np.stack(np.meshgrid([nx], ns, ns, indexing="ij"), -1).reshape(-1, 3)
        k2 = (k**2).sum(1)
        keep = k2 > 0
        k, k2 = k[keep], k2[keep]
        w = np.exp(-k2 / (4.0 * a * a)) / k2
        ph = d @ k.T                                   # (n, nk)
        # g(r) = sum_k i k w e^{-i k.r} / L^3 -> real part: sum_k k w sin(k.r) / L^3,
        # with the sign fixed by requiring attraction (checked against Newton)
        g -= (np.sin(ph) * w) @ k
    return g / L**3


def ewald_total(d, L, a=None, n_max=None):
    a = 5.0 / L if a is None else a
    n_max = int(np.ceil(2.0 * a * 6.5 * L / (2.0 * np.pi))) if n_max is None else n_max
    return ewald_real(d, L, a, n_img=2) + ewald_recip(d, L, a, n_max)


# --- the engine's arms -------------------------------------------------------


def engine_arms(pos, g, r_s, coarse_cell, fine_cell, n_total):
    """(long, short, mono) forces at every row of `pos`, the gate's ratified calls."""
    import jax.numpy as jnp
    import v2_g5_core as probe
    from inexor import forces

    x = np.asarray(pos, dtype=np.float64)
    g_long, _ = forces.force_global(
        x, g["n_coarse"], g["L"], n_total, "long", r_s=r_s,
        match=(coarse_cell, fine_cell), assign="tsc", paint="int",
    )
    _, b_real = forces.padded_size(g["tile"], g["buf"], n_fine=g["n_fine"])
    n_brick = probe.choose_brick(g["tile"], b_real, g["n_fine"])
    order, starts, nbk = probe.brick_buckets(x, g["n_fine"], n_brick, fine_cell)
    n_side = g["n_fine"] // g["tile"]
    tiles = list(itertools.product(range(n_side), repeat=3))
    cap, _ = probe.tile_capacity(order, starts, nbk, tiles, g["tile"], b_real, n_brick)
    g_short, _ = forces.force_short_tiled(
        x, g["n_fine"], g["L"], n_total, g["tile"], g["buf"],
        lambda t: probe.tile_members(order, starts, nbk, t, g["tile"], b_real, n_brick),
        cap, r_s=r_s, paint="int",
    )
    g_mono, _ = forces.force_global(jnp.asarray(x), g["n_fine"], g["L"], n_total, "mono")
    return np.asarray(g_long), np.asarray(g_short), np.asarray(g_mono)


def _radial(gv, d):
    """Attractive radial component and the tangential magnitude."""
    r = np.linalg.norm(d, axis=1)
    rhat = d / r[:, None]
    fr = -(gv * rhat).sum(1)
    ft = np.linalg.norm(gv + fr[:, None] * rhat, axis=1)
    return fr, ft


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--config", default="cdev8")
    ap.add_argument("--n-sources", type=int, default=4)
    ap.add_argument("--n-tests", type=int, default=3000)
    ap.add_argument("--r-min", type=float, default=0.05)
    ap.add_argument("--r-max", type=float, default=None, help="default L/4")
    ap.add_argument("--n-bins", type=int, default=24)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("-o", "--out", default=None, help="figure path")
    ap.add_argument("--json", default=None, help="write the binned profile here")
    ap.add_argument("--match-order-solve", type=int, default=None,
                    help="EXPERIMENT ARM: build the coarse match factor with this "
                         "assignment order on the solve (coarse) side. The engine "
                         "passes the default, 2 (CIC), while painting and gathering "
                         "the coarse arm with TSC (order 3). Patched into this "
                         "process only; the library is untouched")
    args = ap.parse_args()

    import jax

    jax.config.update("jax_enable_x64", True)
    from inexor.plan import PRESETS, RATIFIED

    if args.match_order_solve is not None:
        import functools

        from inexor import forces

        forces.cic_match_factor = functools.partial(
            forces.cic_match_factor, order_solve=int(args.match_order_solve))
        print(f"  EXPERIMENT ARM: coarse match factor order_solve={args.match_order_solve}")

    p = PRESETS[args.config]
    g = dict(L=float(p["box"]), n_fine=p["n_fine"], n_coarse=p["n_coarse"],
             tile=p["tile"], buf=p["buf"])
    L = g["L"]
    fine_cell, coarse_cell = L / g["n_fine"], L / g["n_coarse"]
    r_s = RATIFIED["alpha"] * coarse_cell
    a_split = 1.0 / (2.0 * r_s)
    n_total = int(p["n_part"]) ** 3
    m = L**3 / n_total
    r_max = L / 4.0 if args.r_max is None else args.r_max
    print(f"  {args.config}: L={L:g} fine cell {fine_cell:g} coarse cell {coarse_cell:g} "
          f"r_s={r_s:g} Mpc/h | tile {g['tile']} buf {g['buf']} | m = L^3/n_total = {m:.3e}")

    # --- the reference checks itself before it scores anything
    rng = np.random.default_rng(args.seed + 1000)
    dchk = rng.normal(size=(64, 3))
    dchk *= (np.geomspace(0.05, r_max, 64) / np.linalg.norm(dchk, axis=1))[:, None]
    e1, e2 = ewald_total(dchk, L, a=4.0 / L), ewald_total(dchk, L, a=6.0 / L)
    a_dep = float(np.max(np.abs(e1 - e2)) / np.max(np.abs(e2)))
    newt = -dchk / (4.0 * np.pi * np.linalg.norm(dchk, axis=1)[:, None] ** 3)
    small = np.linalg.norm(dchk, axis=1) < 0.5
    near = float(np.max(np.abs(e2[small] / newt[small] - 1.0)))
    print(f"  Ewald self-checks: a-dependence {a_dep:.2e} (max rel), "
          f"vs Newton at r < 0.5: {near:.2e}")
    if a_dep > 1e-8 or near > 1e-5:
        raise SystemExit("FATAL: the Ewald reference does not check out; nothing is scored")

    rows = []
    for s in range(args.n_sources):
        rng = np.random.default_rng(args.seed + s)
        src = rng.uniform(0.0, L, size=3)
        u = rng.normal(size=(args.n_tests, 3))
        u /= np.linalg.norm(u, axis=1)[:, None]
        r = np.exp(rng.uniform(np.log(args.r_min), np.log(r_max), size=args.n_tests))
        tests = np.mod(src + u * r[:, None], L)
        d = _min_image(tests - src, L)

        with_src = np.vstack([tests, src[None, :]])
        la, sa, ma = engine_arms(with_src, g, r_s, coarse_cell, fine_cell, n_total)
        lt, st, mt = engine_arms(tests, g, r_s, coarse_cell, fine_cell, n_total)
        eng = {"long": la[:-1] - lt, "short": sa[:-1] - st, "mono": ma[:-1] - mt}
        eng["total"] = eng["long"] + eng["short"]

        ref_total = m * ewald_total(d, L)
        ref_short = m * ewald_real(d, L, a_split, n_img=1)
        ref = {"total": ref_total, "short": ref_short, "long": ref_total - ref_short,
               "mono": ref_total}
        for i in range(args.n_tests):
            rows.append((np.linalg.norm(d[i]),
                         *[_radial(eng[a][i:i + 1], d[i:i + 1])[j][0]
                           for a in ("total", "long", "short", "mono") for j in (0, 1)],
                         *[_radial(ref[a][i:i + 1], d[i:i + 1])[0][0]
                           for a in ("total", "long", "short")]))
        print(f"  source {s + 1}/{args.n_sources} done")

    R = np.array(rows)
    r = R[:, 0]
    cols = dict(total=(1, 2), long=(3, 4), short=(5, 6), mono=(7, 8))
    refc = dict(total=9, long=10, short=11, mono=9)
    edges = np.geomspace(args.r_min, r_max, args.n_bins + 1)
    idx = np.digitize(r, edges) - 1
    newton = m / (4.0 * np.pi * r**2)
    out = dict(config=args.config, r_s=r_s, fine_cell=fine_cell, coarse_cell=coarse_cell,
               r_mid=[], n=[])
    for a in cols:
        out[f"{a}_over_ref"] = []
        out[f"{a}_over_newton"] = []
        out[f"{a}_tangential"] = []
        out[f"{a}_scatter"] = []
    print("\n   r [Mpc/h]   n   total/ref  long/refL-frac  short/refS-frac  mono/ref   "
          "(each arm's error as a fraction of the TOTAL reference)   tangential total")
    for b in range(args.n_bins):
        sel = idx == b
        if not sel.any():
            continue
        out["r_mid"].append(float(np.sqrt(edges[b] * edges[b + 1])))
        out["n"].append(int(sel.sum()))
        line = []
        for a, (cr, ct) in cols.items():
            rt = R[sel, refc["total"]]
            # every arm's residual is normalized by the TOTAL reference, so the
            # columns add: (long - refL) + (short - refS) = total - ref
            err = (R[sel, cr] - R[sel, refc[a]]) / rt
            ratio = R[sel, cr] / R[sel, refc[a]]
            out[f"{a}_over_ref"].append(float(np.mean(ratio)))
            out[f"{a}_over_newton"].append(float(np.mean(R[sel, cr] / newton[sel])))
            out[f"{a}_tangential"].append(float(np.mean(R[sel, ct] / rt)))
            out[f"{a}_scatter"].append(float(np.std(ratio)))
            line.append(float(np.mean(err)))
        print(f"   {out['r_mid'][-1]:8.3f} {out['n'][-1]:5d}   "
              f"{1 + line[0]:.4f}      {line[1]:+.4f}         {line[2]:+.4f}        "
              f"{1 + line[3]:.4f}          {out['total_tangential'][-1]:.4f}")
    if args.json:
        with open(args.json, "w") as fh:
            json.dump(out, fh, indent=1)
        print(f"  profile -> {args.json}")

    if args.out:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        rm = np.asarray(out["r_mid"])
        fig, (ax, bx) = plt.subplots(1, 2, figsize=(11.0, 4.2))
        for a, c in (("total", "k"), ("mono", "C2")):
            ax.plot(rm, out[f"{a}_over_ref"], color=c, marker="o", ms=3, label=a)
        ax.axhline(1.0, color="0.5", lw=1)
        for x, lab in ((fine_cell, "fine cell"), (coarse_cell, "coarse cell"), (r_s, None)):
            ax.axvline(x, color="0.7", ls=":", lw=1)
        ax.set_xscale("log")
        ax.set_xlabel(r"$r\ [h^{-1}{\rm Mpc}]$")
        ax.set_ylabel(r"$F_r / F_r^{\rm Ewald}$")
        ax.legend(frameon=False)
        for a, c in (("long", "C0"), ("short", "C1")):
            bx.plot(rm, out[f"{a}_over_ref"], color=c, marker="o", ms=3,
                    label=f"{a} / its Ewald part")
        bx.axhline(1.0, color="0.5", lw=1)
        bx.axvline(r_s, color="0.7", ls=":", lw=1)
        bx.set_xscale("log")
        bx.set_xlabel(r"$r\ [h^{-1}{\rm Mpc}]$")
        bx.legend(frameon=False)
        fig.tight_layout()
        fig.savefig(args.out, dpi=180)
        print(f"  figure -> {args.out}")


if __name__ == "__main__":
    main()
