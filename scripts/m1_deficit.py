"""M1 S6: attribute the M0 ~4% low-k absolute P(k) deficit.

M0 observation (worklog 2026-07-12): evolved P(k) at 64^3 sits ~4% below the
THEORY linear P at k = 0.025 (the fundamental bin), single seed, ZA ICs, EH98.
The fundamental bin holds a handful of modes, so the absolute-vs-theory metric
carries tens of percent of sample variance per seed -- this matrix therefore
leads with a variance-cancelling metric and carries the M0-style one alongside:

  growth transfer  T(k) = P_final(k) / P_init(k) vs (D_f/D_i)^2
      (both P's measured on the PARTICLES with the same neutral estimator, so
      realization noise, CIC window, and discreteness cancel; deviations are
      DYNAMICS: stepping + transient + mode coupling)
  absolute-vs-theory  A(k) = P_final(k) / (D_f^2 P_theory(k)) - 1
      (the M0 metric; realization noise does NOT cancel)
  IC check  I(k) = P_init(k) / (D_i^2 P_theory(k)) - 1
      (pure realization + IC-discreteness term; A ~ I + growth deficit)

Axes: lpt {1,2} x K {5,10,20,40} x integrator {bullfrog,fastpm,exact} at 64^3
(+ 128^3 and CAMB-table and multi-seed spot checks), deconvolve on/off as a
post-hoc column. All arms evolve_float f64 (quantization plays no role here).

    pixi run python scripts/m1_deficit.py            # full matrix -> json
    pixi run python scripts/m1_deficit.py --quick    # headline cells only
"""

import argparse
import itertools
import json
import os
import sys

os.environ.setdefault("JAX_ENABLE_X64", "1")

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import _m1_common as M  # noqa: E402

REPO = os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
KLOW = 0.05  # h/Mpc: the "low-k" verdict band (M0 quoted k = 0.025)


def run_cell(n, K, integ, lpt_order, seed=0, pk_source="eh98", spacing="log"):
    import jax

    jax.config.update("jax_enable_x64", True)
    import jax.numpy as jnp

    from inexor import config, integrate, lpt
    from inexor.cosmology import growth_factor_a, linear_power
    from inexor.ic import gaussian_delta

    cosmo = config.Cosmology(**M.COSMO)
    box = config.BoxConfig(n_mesh=n, box_size=M.BOX_SIZE)
    time = config.TimeConfig(
        a_init=M.A_INIT, a_final=M.A_FINAL, n_steps=K, spacing=spacing, integrator=integ
    )
    table = None
    if pk_source == "camb":
        d = np.loadtxt(os.path.join(M.RUNS, "pk_camb.txt"))
        table = (d[:, 0], d[:, 1])

    key = jax.random.PRNGKey(seed)
    delta0 = gaussian_delta(
        key,
        n,
        M.BOX_SIZE,
        cosmo,
        fdtype=jnp.float64,
        backend="table" if table else "eh98",
        table=table,
    )
    x0, v0 = lpt.lpt_ics(delta0, M.BOX_SIZE, M.A_INIT, cosmo, order=lpt_order, fdtype=jnp.float64)
    xf, _ = integrate.evolve_float(box, time, cosmo, x0, v0, fdtype=jnp.float64)

    L = M.BOX_SIZE
    D_i = growth_factor_a(M.A_INIT, cosmo)
    D_f = growth_factor_a(M.A_FINAL, cosmo)
    di = M.cic_paint(np.mod(np.asarray(x0), L), n, L)
    df = M.cic_paint(np.mod(np.asarray(xf), L), n, L)
    kc, P_i = M.pk(di, n, L)
    _, P_f = M.pk(df, n, L)
    _, P_i_dec = M.pk_deconvolved(di, n, L)
    _, P_f_dec = M.pk_deconvolved(df, n, L)
    P_th = linear_power(kc, cosmo, backend="table" if table else "eh98", table=table)

    low = kc <= KLOW
    growth_dev = P_f / P_i / (D_f / D_i) ** 2 - 1.0
    growth_dev_dec = P_f_dec / P_i_dec / (D_f / D_i) ** 2 - 1.0
    abs_dev = P_f / (D_f**2 * P_th) - 1.0
    abs_dev_dec = P_f_dec / (D_f**2 * P_th) - 1.0
    ic_dev = P_i / (D_i**2 * P_th) - 1.0

    return dict(
        n=n,
        K=K,
        integrator=integ,
        lpt_order=lpt_order,
        seed=seed,
        pk_source=pk_source,
        spacing=spacing,
        k=kc.tolist(),
        growth_dev_lowk=float(np.mean(growth_dev[low])),
        growth_dev_dec_lowk=float(np.mean(growth_dev_dec[low])),
        abs_dev_lowk=float(np.mean(abs_dev[low])),
        abs_dev_dec_lowk=float(np.mean(abs_dev_dec[low])),
        ic_dev_lowk=float(np.mean(ic_dev[low])),
        growth_dev=growth_dev.tolist(),
        abs_dev=abs_dev.tolist(),
        ic_dev=ic_dev.tolist(),
    )


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--quick", action="store_true")
    ap.add_argument(
        "--seeds",
        type=int,
        default=0,
        help="ALSO sweep this many seeds at the headline cell (64^3 K=10 "
        "bullfrog 2LPT) -- the mean pins the deficit against its "
        "large per-realization scatter",
    )
    ap.add_argument("--out", default=os.path.join(M.RUNS, "deficit_matrix.json"))
    args = ap.parse_args()

    cells = []
    if args.seeds:
        gds, ads = [], []
        for seed in range(args.seeds):
            c = run_cell(64, 10, "bullfrog", 2, seed=seed)
            cells.append(c)
            gds.append(c["growth_dev_lowk"])
            ads.append(c["abs_dev_lowk"])
            print(
                f"seed={seed:2d}: growth_dev {c['growth_dev_lowk']:+.4f}  "
                f"abs_dev {c['abs_dev_lowk']:+.4f}  ic_dev {c['ic_dev_lowk']:+.4f}"
            )
        gds, ads = np.array(gds), np.array(ads)
        print(
            f"SEED SWEEP (n={args.seeds}): growth_dev mean {gds.mean():+.4f} "
            f"+- {gds.std(ddof=1) / np.sqrt(len(gds)):.4f} (scatter {gds.std(ddof=1):.4f}); "
            f"abs_dev mean {ads.mean():+.4f} (scatter {ads.std(ddof=1):.4f})"
        )
        out = dict(
            meta=M.make_meta("inexor", "deficit-seeds", dict(klow=KLOW, n_seeds=args.seeds), REPO),
            cells=cells,
        )
        with open(os.path.join(M.RUNS, "deficit_seeds.json"), "w") as f:
            json.dump(out, f, indent=1)
        return

    if args.quick:
        combos = [(64, 10, "bullfrog", 1), (64, 10, "bullfrog", 2)]
    else:
        combos = list(
            itertools.product([64], [5, 10, 20, 40], ["bullfrog", "fastpm", "exact"], [1, 2])
        )
        combos += [(128, 10, "bullfrog", 1), (128, 10, "bullfrog", 2), (128, 40, "bullfrog", 2)]

    for n, K, integ, lpt_order in combos:
        c = run_cell(n, K, integ, lpt_order)
        cells.append(c)
        print(
            f"n={n:4d} K={K:3d} {integ:8s} lpt{lpt_order}: "
            f"growth_dev(lowk) {c['growth_dev_lowk']:+.4f}  "
            f"abs_dev(lowk) {c['abs_dev_lowk']:+.4f}  "
            f"ic_dev(lowk) {c['ic_dev_lowk']:+.4f}"
        )

    if not args.quick:
        # axis spot checks at the headline cell (64^3 K=10 bullfrog 2LPT)
        for seed in (1, 2, 3):
            c = run_cell(64, 10, "bullfrog", 2, seed=seed)
            cells.append(c)
            print(
                f"seed={seed}: growth_dev {c['growth_dev_lowk']:+.4f}  "
                f"abs_dev {c['abs_dev_lowk']:+.4f}  ic_dev {c['ic_dev_lowk']:+.4f}"
            )
        c = run_cell(64, 10, "bullfrog", 2, pk_source="camb")
        cells.append(c)
        print(
            f"camb table: growth_dev {c['growth_dev_lowk']:+.4f}  "
            f"abs_dev {c['abs_dev_lowk']:+.4f}  ic_dev {c['ic_dev_lowk']:+.4f}"
        )
        c = run_cell(64, 10, "bullfrog", 2, spacing="linear")
        cells.append(c)
        print(
            f"linear spacing: growth_dev {c['growth_dev_lowk']:+.4f}  "
            f"abs_dev {c['abs_dev_lowk']:+.4f}"
        )

    out = dict(
        meta=M.make_meta(
            "inexor",
            "deficit-matrix",
            dict(klow=KLOW, box_size=M.BOX_SIZE, a_init=M.A_INIT, a_final=M.A_FINAL),
            REPO,
        ),
        cells=cells,
    )
    with open(args.out, "w") as f:
        json.dump(out, f, indent=1)
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
