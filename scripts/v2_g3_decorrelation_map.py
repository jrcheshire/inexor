"""G3 -- where does a tiled arm still share phases with the monolithic one?

THE DECISION THIS EXISTS FOR. Stage 4 found that R_Q is only interpretable where
the two arms remain correlated, and that at cdev8 with T=64/b=16 the correlation
at k_short = 1.18 h/Mpc is r = 0.028 -- no phase relation at all. That leaves a
choice for Stage 5: bring k_short down to where r is appreciable, or keep it and
pay for a less aggressive tiling. The two give different A3 compute factors, so
the choice is worth measuring rather than arguing.

WHAT IS MEASURED. r(k) for a grid of (T, b) at cdev8, with the cost of each
configuration known exactly: a tiled run costs n_tiles * (P/n_fine)^3 = (P/T)^3
monolithic evolves, i.e. the vol_ratio itself. From r(k) comes k_usable, the k at
which r falls through a threshold -- the largest scale the gate can trust for
that configuration.

THE GRID IS COST-CONTROLLED, which is the point. Three (T, b) pairs share each
vol_ratio:

    vol_ratio 1.95:  T=64/b=8    T=128/b=16
    vol_ratio 3.4:   T=32/b=8    T=64/b=16   T=128/b=32
    vol_ratio 8:     T=32/b=16   T=64/b=32   T=128/b=64
    vol_ratio 27:    T=32/b=32   T=64/b=64

So the sweep answers a question a one-dimensional b-scan cannot: AT FIXED COST,
does tile size matter? If the k_usable points collapse onto a single curve in
vol_ratio, cost is the only variable and the tiling geometry is free to choose;
if they separate, there is a preferred T at every price.

Usage:
    pixi run python scripts/v2_g3_decorrelation_map.py --config cdev8
    pixi run python scripts/v2_g3_decorrelation_map.py --plot-only
"""

import argparse
import json
import os
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

from v2_g5_two_level_force import geometry  # noqa: E402

OUT_DIR = os.path.join(os.path.dirname(HERE), "runs", "v2")
FIG_DIR = os.path.join(os.path.dirname(HERE), "runs", "v2", "figs")

# (T, b) in FINE cells. Cost-controlled: see the module docstring.
GRID = [
    (32, 8), (32, 16), (32, 32),
    (64, 8), (64, 16), (64, 32), (64, 64),
    (128, 16), (128, 32), (128, 64),
]
R_LEVELS = (0.9, 0.5, 0.2)


def k_at_r(k, r, level):
    """Largest k with r >= level, by linear interpolation on the first crossing.

    r(k) is monotone decreasing in practice but not guaranteed to be, so this
    takes the FIRST downward crossing rather than the last -- the gate cares
    about the scale beyond which it can no longer trust the arm, and a later
    recovery above the level does not restore that trust.
    """
    k = np.asarray(k, float)
    r = np.asarray(r, float)
    below = np.where(r < level)[0]
    if len(below) == 0:
        return float(k[-1])
    i = below[0]
    if i == 0:
        return float(k[0])
    r0, r1, k0, k1 = r[i - 1], r[i], k[i - 1], k[i]
    if r0 == r1:
        return float(k0)
    return float(k0 + (level - r0) * (k1 - k0) / (r1 - r0))


def measure(cfg, seed):
    import jax

    jax.config.update("jax_enable_x64", True)
    from inexor import painting
    from inexor.diagnostics import cross_r
    from v2_g5_core import padded_size

    import v2_g3_core as g3
    import v2_g3_ladder as lad

    B = lad.build(cfg, seed)
    g = B["g"]
    ell, n_fine, n_part = g["L"], g["n_fine"], g["n_part"]

    def dens(x):
        return np.asarray(
            painting.density_contrast(x, n_fine, ell, n_part**3, paint="int"), np.float64
        )

    d_mono = dens(B["x_mono"])
    rows = []
    t0 = time.perf_counter()
    for n_tile, b_fine in GRID:
        try:
            p_side, b_real = padded_size(n_tile, b_fine, n_fine=n_fine)
        except ValueError:
            continue
        vol_ratio = (p_side / n_tile) ** 3
        t1 = time.perf_counter()
        x_t, _, d = g3.evolve_scola(
            B["q"], B["psi1"], B["psi2"], B["qi"], B["coeffs"], ell, n_fine,
            n_part, n_tile, b_fine, B["d_final"],
        )
        k, r, nmodes = cross_r(dens(x_t), d_mono, ell)
        row = dict(
            n_tile=int(n_tile), b_fine=int(b_fine), b_realized=int(b_real),
            p_side=int(p_side), vol_ratio=float(vol_ratio),
            p_frac=float(p_side) / float(n_fine), n_tiles=int(d["n_tiles"]),
            b_mpc=float(b_real) * ell / n_fine, t_mpc=float(n_tile) * ell / n_fine,
            # P == n_fine means the tile IS the box: r == 1 by construction and
            # k_usable is the mesh Nyquist. Kept as an anchor (it must read 1)
            # but excluded from the cost/usable-k panel, where it would be a
            # free point that no real tiling can reach.
            degenerate=bool(p_side >= n_fine),
            k=[float(v) for v in k], r=[float(v) for v in r],
            wall=time.perf_counter() - t1,
        )
        for lv in R_LEVELS:
            row[f"k_r{lv}"] = k_at_r(k, r, lv)
        rows.append(row)
        print(f"  T={n_tile:4d} b={b_fine:3d} P={p_side:4d} vol={vol_ratio:6.2f}x  "
              f"k(r=0.5)={row['k_r0.5']:7.4f}  r(k=1.18)={np.interp(1.178, k, r):7.4f}  "
              f"[{row['wall']:.0f}s]", flush=True)
    print(f"  total {time.perf_counter() - t0:.0f}s")
    return dict(config=cfg, seed=seed, geometry={k: float(v) for k, v in g.items()},
                r_levels=list(R_LEVELS), rows=rows)


def plot(card, path):
    """Two panels: r(k) per configuration, and usable-k against cost.

    Colour encodes vol_ratio, which is a MAGNITUDE, so it is a single-hue
    sequential ramp (viridis: perceptually uniform and CVD-safe by construction)
    rather than a categorical cycle. Marker shape carries T, so configurations
    at matched cost are separable without relying on colour alone.
    """
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D

    rows = card["rows"]
    vols = np.array([r["vol_ratio"] for r in rows])
    norm = matplotlib.colors.LogNorm(vmin=vols.min(), vmax=vols.max())
    cmap = matplotlib.colormaps["viridis"]
    marks = {32: "o", 64: "s", 128: "^"}

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11.5, 4.4))

    for r in rows:
        ax1.semilogx(r["k"], r["r"], lw=1.6, color=cmap(norm(r["vol_ratio"])),
                     ls=":" if r.get("degenerate") else "-", solid_capstyle="round")
    ax1.axhline(0.5, color="0.35", lw=1.0, ls="--", zorder=1)
    ax1.text(ax1.get_xlim()[0] * 1.15, 0.52, "conditioning cut", fontsize=8, color="0.35")
    ax1.set_xlabel(r"$k$  [$h\,\mathrm{Mpc}^{-1}$]")
    ax1.set_ylabel(r"$r(k)$   tiled vs monolithic")
    ax1.set_ylim(-0.05, 1.03)
    ax1.grid(alpha=0.18, lw=0.6)
    ax1.set_axisbelow(True)
    for sp in ("top", "right"):
        ax1.spines[sp].set_visible(False)

    for r in [z for z in rows if not z.get("degenerate")]:
        ax2.scatter(r["vol_ratio"], r["k_r0.5"], s=64, marker=marks[r["n_tile"]],
                    color=cmap(norm(r["vol_ratio"])), edgecolor="0.25", linewidth=0.7,
                    zorder=3)
    ax2.set_xscale("log")
    ax2.set_yscale("log")
    ax2.set_xlabel(r"cost  [monolithic evolves, $=(P/T)^3$]")
    ax2.set_ylabel(r"$k$ at $r=0.5$   [$h\,\mathrm{Mpc}^{-1}$]")
    ax2.grid(alpha=0.18, lw=0.6)
    ax2.set_axisbelow(True)
    for sp in ("top", "right"):
        ax2.spines[sp].set_visible(False)

    handles = [Line2D([], [], marker=m, ls="", color="0.35", markersize=7,
                      label=f"$T$ = {t} cells ({t * card['geometry']['L'] / card['geometry']['n_fine']:.0f} Mpc/$h$)")
               for t, m in marks.items()]
    ax2.legend(handles=handles, frameon=False, fontsize=8, loc="lower right")

    sm = matplotlib.cm.ScalarMappable(norm=norm, cmap=cmap)
    cb = fig.colorbar(sm, ax=(ax1, ax2), fraction=0.028, pad=0.015)
    cb.set_label("cost  [monolithic evolves]", fontsize=9)

    os.makedirs(os.path.dirname(path), exist_ok=True)
    fig.savefig(path, dpi=160, bbox_inches="tight")
    print(f"wrote {path}")


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", default="cdev8", choices=("smoke", "cdev8", "cdev"))
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--plot-only", action="store_true")
    ap.add_argument("--out-suffix", default="")
    args = ap.parse_args()

    path = os.path.join(OUT_DIR, f"g3_decorrelation_{args.config}{args.out_suffix}.json")
    if args.plot_only:
        card = json.load(open(path))
    else:
        print(f"=== G3 decorrelation map: {args.config} ===")
        g = geometry(args.config)
        print(f"  L={g['L']} n_part={g['n_part']} n_fine={g['n_fine']}  "
              f"k_f={2 * np.pi / g['L']:.4f} h/Mpc")
        card = measure(args.config, args.seed)
        os.makedirs(OUT_DIR, exist_ok=True)
        with open(path, "w") as f:
            json.dump(card, f, indent=2, default=float)
        print(f"wrote {path}")

    print("\n  cost-controlled comparison (does T matter at fixed cost?)")
    by_vol = {}
    for r in card["rows"]:
        by_vol.setdefault(round(r["vol_ratio"], 2), []).append(r)
    print(f"  {'cost':>7s} {'T':>5s} {'b':>4s} {'b Mpc/h':>8s} {'k(r=0.9)':>9s} "
          f"{'k(r=0.5)':>9s} {'k(r=0.2)':>9s}")
    for v in sorted(by_vol):
        for r in sorted(by_vol[v], key=lambda z: z["n_tile"]):
            flag = "  DEGENERATE (tile == box)" if r.get("degenerate") else ""
            print(f"  {v:7.2f} {r['n_tile']:5d} {r['b_fine']:4d} {r['b_mpc']:8.1f} "
                  f"{r['k_r0.9']:9.4f} {r['k_r0.5']:9.4f} {r['k_r0.2']:9.4f}{flag}")

    plot(card, os.path.join(FIG_DIR, f"g3_decorrelation_{args.config}{args.out_suffix}.png"))


if __name__ == "__main__":
    main()
