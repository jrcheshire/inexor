"""G3 Stage 5 -- the mechanism ensemble: how the squeezed-bispectrum error
depends on the ABSOLUTE buffer and on where the small-scale leg sits.

WHY THIS IS NOT THE STAGE 5 THE PLAN DESCRIBED. Two measurements moved it:

(1) THE PINNED GATE TRIANGLES ARE DECORRELATED. The estimand's small-scale leg
    is k_short = 1.178 h/Mpc, and the tiled arm holds r >= 0.5 against the
    monolithic one only out to 0.47 h/Mpc at cdev with an 8 Mpc/h buffer, 0.90
    with 16 Mpc/h (runs/v2/g3_decorrelation_record.md). Under the ratified
    R_CONDITION_CUT = 0.5 that leaves NO gradeable triangle at any cdev
    geometry. R_Q cannot be read there at all: once two fields decorrelate the
    ratio of their bispectra saturates at a bounded value and stops being
    monotone in brokenness (runs/v2/g3_ladder_record.md).
(2) THE PLAN'S GEOMETRY IS DOMINATED. The absolute buffer in Mpc/h sets usable
    k; tile size sets only the price (paired ratio 0.993 at fixed 8 Mpc/h buffer
    across a 2.37x difference in padded volume). Geometry is therefore chosen by
    the buffer the gate needs and then the largest tile that fits -- never by
    walking an iso-cost line, which confounds the two exactly.

So this script scans a BUFFER LADDER IN ABSOLUTE UNITS against a LADDER OF
SMALL-SCALE LEGS, and the eligibility boundary between them is a measured
output rather than an assumption. The verdict is read off the resulting curve.

WHAT IS RATIFIED AND NOT RE-OPENED HERE: gate on R_Q with rho required to agree;
estimand pinned at cdev; reduce by MAX over gate-eligible triangles; the R vs x
curve is mandatory and ungated (D-v2-9 clause 3); the conditioning cut is
r(k_short) >= 0.5, reported below it and never gated.

THE NON-VACUITY RULE. A cell with no gradeable triangle reads NOT GRADEABLE and
never PASS. A configuration broken enough to decorrelate everywhere would
otherwise pass the gate by having nothing left to fail -- an absent check reads
exactly like a passed one, which is the failure this project has hit before.

BINNING MATCHES THE FLOORS RUN, deliberately: dk = k_f on every leg, the default
v2_g3_floors used for the cdev pin (runs/v2/g3_floors_cdev_pin.json). The plan's
dk_short = 4 k_f would buy triangle count at the price of making every sigma_B
in that card inapplicable to these numbers, which is the wrong trade for a gate
whose resolvability argument rests on them.

Usage:
    pixi run python scripts/v2_g3_stage5.py --config smoke
    pixi run python scripts/v2_g3_stage5.py --config cdev --seed 0
"""

import argparse
import json
import os
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

from v2_g5_two_level_force import _host_rss_bytes  # noqa: E402

OUT_DIR = os.path.join(os.path.dirname(HERE), "runs", "v2")
FIG_DIR = os.path.join(OUT_DIR, "figs")

# The gate's conditioning cut (JC, 2026-07-31). Gradeable means the arms still
# share phases at the small-scale leg. The second level is REPORTED alongside so
# the checkpoint can see how the verdict moves under a stricter reading -- the
# geometry ranking is threshold-dependent (1.297 at r=0.5 vs 1.034 at r=0.9),
# so the two are not separable and both belong in the card.
R_GATE = 0.5
R_REPORT = (0.9, 0.5, 0.2)

# Arms are (tile, buffer) in FINE CELLS, never in Mpc/h at the API boundary --
# but they are CHOSEN in Mpc/h. At cdev (0.25 Mpc/h cells): 16/32/64 fine =
# 4/8/16 Mpc/h. Arm D repeats arm B's absolute buffer at half the tile size, so
# the "absolute buffer sets the physics" null gets tested on the SEAM statistic
# and not only on the correlation, where it was measured.
DEFAULT_ARMS = {
    "cdev": ((128, 16), (128, 32), (128, 64), (64, 32)),
    "cdev8": ((64, 16), (64, 32), (64, 64), (32, 32)),
    "cgh64": ((256, 32), (256, 64), (256, 128), (128, 64)),
    "smoke": ((16, 8), (16, 16), (32, 8)),
}
# A triangle enters the ESTIMAND only if it is actually squeezed. With long legs
# running to m = 7, a shallow k_short (m_short = 6) would otherwise fold a
# near-equilateral configuration into the max of a SQUEEZED-bispectrum gate. The
# full set is still reported and plotted -- the R-vs-x curve is mandatory and
# ungated -- but the number the bar is compared against is restricted here.
SQUEEZE_RATIO = 2.0
# k_short in units of THIS BOX's k_f. Physical match across the ladder is the
# whole point of the multiples differing per config: 24 at cdev == 12 at cdev8
# == 48 at cgh64 == 1.178 h/Mpc. Getting this wrong once voided every bracket
# number in deneb job 283.
DEFAULT_K_SHORT = {
    "cdev": (6, 12, 18, 24),
    "cdev8": (3, 6, 9, 12),
    "cgh64": (12, 24, 36, 48),
    "smoke": (4, 6, 8),
}
DEFAULT_LONG_MULTS = (1, 2, 3, 4, 5, 6, 7)
# Sub-volume counts for the position-dependent P(k). Both are powers of two
# because an equal-cube split needs n_sub | N and N is a power of two; the
# lattice is kept off the tile walls by the half-sub-volume OFFSET, not by
# coprimality, which is unavailable here (see diagnostics._straddle_fraction).
DEFAULT_N_SUB = (4, 8)

ARM_KINDS = ("scola", "mono", "kill_control", "span_check")


def arm_tag(n_tile, b_fine):
    return f"scola_T{int(n_tile)}_b{int(b_fine)}"


def parse_arms(spec):
    out = []
    for part in spec.split(","):
        t, b = part.split(":")
        out.append((int(t), int(b)))
    return tuple(out)


def k_at_r(k, r, level):
    """Largest k with r still above `level`, by the FIRST downward crossing.

    Imported behaviour, not re-derived: identical to
    v2_g3_decorrelation_map.k_at_r, which the tracked cards were built with.
    """
    import v2_g3_decorrelation_map as dm

    return dm.k_at_r(k, r, level)


def check_anchor(cfg, seed, rows, rtol):
    """Cross-check k(r=0.5) against the tracked decorrelation cards.

    THE TOLERANCE IS DELIBERATELY LOOSE (1% by default), and that is not
    sloppiness. The cdev seed cards were produced on Vista (linux-aarch64) and
    this script runs on deneb (linux-64), so an f64 N-body run through two
    different XLA-CPU backends is not bitwise comparable and a 1e-6 bound would
    fire on the machine, not on a regression. 1% sits ~12x below the seed-to-
    seed scatter of the quantity (12-22%) and ~30x below the effect being
    ranked, so it still catches any real change in the tile machinery.

    Returns a list of per-arm comparison dicts; raises on a mismatch.
    """
    names = [f"g3_decorrelation_{cfg}_seed{seed}.json"]
    if seed == 0:
        names.append(f"g3_decorrelation_{cfg}.json")
    card = None
    for nm in names:
        p = os.path.join(OUT_DIR, nm)
        if os.path.exists(p):
            with open(p) as f:
                card = json.load(f)
            card_name = nm
            break
    if card is None:
        print(f"  [anchor] no decorrelation card for {cfg} seed {seed} -- skipped")
        return []

    ref = {(int(r["n_tile"]), int(r["b_fine"])): r for r in card["rows"]}
    out, bad = [], []
    for row in rows:
        key = (row["n_tile"], row["b_fine"])
        if key not in ref:
            continue
        want = float(ref[key]["k_r0.5"])
        got = float(row["k_usable"]["0.5"])
        rel = abs(got - want) / want if want else np.nan
        out.append(dict(arm=row["arm"], k_r05=got, reference=want, rel=float(rel),
                        source=card_name))
        print(f"  [anchor] {row['arm']:16s} k(r=0.5) {got:.4f} vs {want:.4f}  rel {rel:.2e}")
        if not np.isfinite(rel) or rel > rtol:
            bad.append((row["arm"], got, want, rel))
    if bad:
        raise AssertionError(
            f"k(r=0.5) does not reproduce {card_name} within rtol={rtol}: {bad}. "
            "The tile machinery changed under a frozen measurement -- do not read any "
            "number from this run until that is explained."
        )
    return out


def response_pair(d_arm, d_mono, box_size, n_sub, k_centers, tiles_per_side, dk=None):
    """Position-dependent P(k) response of both arms, and the tiled/mono ratio.

    Skips centers the sub-volume grid cannot resolve rather than failing the
    run: which k_short a given n_sub can reach is a property of the geometry,
    and the card records what was used.

    The filter is an EXACT mode count on the sub-grid, not the fundamental-based
    test the estimator raises on. Those are not the same condition -- a band can
    clear 0.5 * k_f_sub and still contain no mode (measured at smoke, n_sub = 8:
    the 1.178 h/Mpc shell sits below that grid's smallest nonzero |k| = 1.571).
    """
    from inexor.diagnostics import _k_grid, _shell_mask, subvolume_response

    n_mesh = d_mono.shape[0]
    s = n_mesh // n_sub
    sub_box = box_size / n_sub
    w = (2.0 * np.pi / box_size) if dk is None else float(dk)
    _, _, k_mag_sub = _k_grid(s, sub_box)
    usable = [float(c) for c in k_centers
              if _shell_mask(k_mag_sub, float(c) - 0.5 * w, float(c) + 0.5 * w).any()]
    if not usable:
        return None
    kw = dict(dk=w, tiles_per_side=tiles_per_side)
    r_m = subvolume_response(d_mono, box_size, n_sub, usable, **kw)
    r_t = subvolume_response(d_arm, box_size, n_sub, usable, **kw)
    ratio = r_t["slope"] / r_m["slope"] - 1.0
    # Paired error: the two arms share ICs, so their delta_bar are nearly
    # identical and the errors are strongly correlated. Adding them in
    # quadrature OVERSTATES the uncertainty on the ratio; it is reported as an
    # upper bound and labelled as one rather than quietly used as a sigma.
    err_bound = np.abs(ratio + 1.0) * np.sqrt(
        (r_t["slope_err"] / r_t["slope"]) ** 2 + (r_m["slope_err"] / r_m["slope"]) ** 2
    )
    return dict(
        n_sub=int(n_sub), k_centers=usable, straddle_frac=r_m["straddle_frac"],
        sub_box=r_m["sub_box"], n_blocks=r_m["n_blocks"],
        slope_mono=[float(v) for v in r_m["slope"]],
        slope_tiled=[float(v) for v in r_t["slope"]],
        slope_err_mono=[float(v) for v in r_m["slope_err"]],
        slope_err_tiled=[float(v) for v in r_t["slope_err"]],
        ratio=[float(v) for v in ratio],
        ratio_err_upper=[float(v) for v in err_bound],
    )


def measure(cfg, seed, args):
    import jax

    jax.config.update("jax_enable_x64", True)
    from inexor import painting
    from inexor.diagnostics import cross_r
    from v2_g5_core import padded_size

    import v2_g3_core as g3
    import v2_g3_floors as fl
    import v2_g3_ladder as lad

    t_all = time.perf_counter()
    B = lad.build(cfg, seed)
    g = B["g"]
    ell, n_fine, n_part = g["L"], g["n_fine"], g["n_part"]
    kf = 2.0 * np.pi / ell
    fine_cell = ell / n_fine

    def dens(x):
        return np.asarray(
            painting.density_contrast(x, n_fine, ell, n_part**3, paint="int"), np.float64
        )

    d_mono = dens(B["x_mono"])
    k_shorts = tuple(args.k_short_mults)
    long_mults = tuple(args.long_mults)
    # Every shell either arm will need, so the transfer/correlation curve is
    # measured once per arm on exactly the gate's own shells.
    all_centers = sorted({float(m * kf) for m in long_mults}
                         | {float(ks * kf) for ks in k_shorts})

    print(f"=== G3 Stage 5: {cfg} seed {seed} ===")
    print(f"  L={ell:g} n_fine={n_fine:g} n_part={n_part:g} cell={fine_cell:.3f} Mpc/h "
          f"k_f={kf:.4f}")
    print("  k_short (k_f mult -> h/Mpc): "
          + ", ".join(f"{ks}->{ks * kf:.3f}" for ks in k_shorts))
    print(f"  long mults: {long_mults}   gate cut r >= {R_GATE}")

    rows = []
    for n_tile, b_fine in args.arms:
        p_side, b_real = padded_size(n_tile, b_fine, n_fine=n_fine)
        vol_ratio = (p_side / n_tile) ** 3
        tag = arm_tag(n_tile, b_fine)
        t0 = time.perf_counter()
        x_t, _, diag = g3.evolve_scola(
            B["q"], B["psi1"], B["psi2"], B["qi"], B["coeffs"], ell, n_fine,
            n_part, n_tile, b_fine, B["d_final"],
        )
        wall = time.perf_counter() - t0
        d_t = dens(x_t)

        k_curve, r_curve, _ = cross_r(d_t, d_mono, ell)
        k_usable = {str(lv): k_at_r(k_curve, r_curve, lv) for lv in R_REPORT}
        t_shell, r_shell, a_shell = fl.shell_transfer(d_t, d_mono, ell, all_centers, kf)
        r_at = dict(zip(all_centers, (float(v) for v in r_shell)))

        cells = {}
        for ks in k_shorts:
            tris, names = fl._triangles(kf, long_mults, ks)
            s = fl.stats(d_t, d_mono, ell, tris)
            sq = [i for i, nm in enumerate(names)
                  if nm != "equi" and ks >= SQUEEZE_RATIO * long_mults[i]]
            r_ks = r_at[float(ks * kf)]
            # BOTH conditions, and the second is the vacuity one: a cell with no
            # squeezed triangle left has nothing to take a max over, so it is
            # not gradeable even where the arms are perfectly correlated.
            gradeable = bool(r_ks >= R_GATE) and len(sq) > 0
            # x = k_long / k_P, the tile's own fundamental -- the abscissa the
            # mandatory ungated curve is drawn against.
            x_vals = [float(m * p_side) / n_fine for m in long_mults]
            cell = dict(
                k_short_mult=int(ks), k_short=float(ks * kf), names=names,
                x=x_vals, r_k_short=float(r_ks), gradeable=gradeable,
                estimand_names=[names[i] for i in sq],
                R_Q=[float(v) for v in s["R_Q"]], R_B=[float(v) for v in s["R_B"]],
                rho=[float(v) for v in s["rho"]], W=[float(v) for v in s["W"]],
                rho_auto=[float(v) for v in s["rho_auto"]],
                T_prod=[float(v) for v in s["T_prod"]],
                A_prod=[float(v) for v in s["A_prod"]],
                n_tri=[float(v) for v in s["n_tri"]],
            )
            # THE ESTIMAND: max |R_Q| over gate-eligible SQUEEZED triangles. The
            # equilateral is a reported control and never enters it. Emitted
            # only when the cell is gradeable -- see the non-vacuity rule.
            if gradeable:
                rq = np.abs([s["R_Q"][i] for i in sq])
                rh = np.abs([s["rho"][i] for i in sq])
                cell["max_abs_R_Q"] = float(np.nanmax(rq))
                cell["max_abs_rho"] = float(np.nanmax(rh))
                cell["argmax_name"] = names[sq[int(np.nanargmax(rq))]]
                # rho must AGREE with R_Q; a divergence flags window
                # contamination rather than a tiling failure (JC, 2026-07-31).
                # PER TRIANGLE, not max-vs-max: two maxima can land on different
                # triangles and agree numerically while the curves disagree
                # everywhere, which is the same max-over-band blindness D-v2-9
                # clause 3 exists to prevent.
                diff = np.abs(np.array([s["R_Q"][i] - s["rho"][i] for i in sq]))
                scale = np.maximum(np.abs([s["R_Q"][i] for i in sq]), 1e-12)
                cell["max_rho_deviation"] = float(np.nanmax(diff / scale))
                cell["rho_agrees"] = bool(cell["max_rho_deviation"] <= args.rho_tol)
                # rho_auto: REPORTED, NOT GATED. The ratified precondition is
                # rho, and swapping the gate statistic is a checkpoint decision,
                # not something a producer may do silently. Emitted alongside so
                # the checkpoint can be argued on measured behaviour -- the pilot
                # showed rho tracks 1/r^2 rather than the tiling error, and the
                # auto transfer carries no r (runs/v2/g3_stage5_record.md sec. 8).
                ra = np.abs([s["rho_auto"][i] for i in sq])
                cell["max_abs_rho_auto"] = float(np.nanmax(ra))
                d_a = np.abs(np.array([s["R_Q"][i] - s["rho_auto"][i] for i in sq]))
                cell["max_rho_auto_deviation"] = float(np.nanmax(d_a / scale))
                cell["rho_auto_agrees"] = bool(
                    cell["max_rho_auto_deviation"] <= args.rho_tol)
            else:
                cell["verdict"] = "NOT GRADEABLE"
                cell["why"] = ("decorrelated" if r_ks < R_GATE
                               else "no squeezed triangle at this k_short")
            cells[str(ks)] = cell

        resp = []
        if not args.skip_response:
            tps = int(n_fine // n_tile)
            for n_sub in args.n_sub:
                if n_fine % n_sub:
                    continue
                pr = response_pair(d_t, d_mono, ell, n_sub,
                                   [ks * kf for ks in k_shorts], tps)
                if pr is not None:
                    resp.append(pr)

        rows.append(dict(
            arm=tag, kind="scola", n_tile=int(n_tile), b_fine=int(b_fine),
            b_realized=int(b_real), b_mpc=float(b_real) * fine_cell,
            t_mpc=float(n_tile) * fine_cell, p_side=int(p_side),
            p_frac=float(p_side) / float(n_fine), vol_ratio=float(vol_ratio),
            n_tiles=int(diag["n_tiles"]), degenerate=bool(p_side >= n_fine),
            wall=float(wall), k=[float(v) for v in k_curve],
            r=[float(v) for v in r_curve], k_usable=k_usable,
            shell_centers=all_centers, shell_T=[float(v) for v in t_shell],
            shell_r=[float(v) for v in r_shell],
            # shell_A closes the gap that blocked the pilot's own re-analysis:
            # with T and r alone, any statistic built on the AUTO transfer needs
            # a fresh run. With A on the card, (1 + R_B)/A_prod - 1 is
            # recoverable per triangle from the card, no evolve required.
            shell_A=[float(v) for v in a_shell],
            cells=cells, response=resp,
        ))
        print(f"  {tag:16s} b={b_real * fine_cell:5.1f} Mpc/h P={p_side:4d} "
              f"vol={vol_ratio:6.2f}x tiles={diag['n_tiles']:5d} "
              f"k(r=0.5)={k_usable['0.5']:.4f} [{wall:.0f}s]", flush=True)

    out = dict(
        config=cfg, seed=int(seed), kinds=list(ARM_KINDS),
        geometry={k: float(v) for k, v in g.items()},
        gate=dict(bar=0.15, r_gate=R_GATE, r_report=list(R_REPORT),
                  long_mults=list(long_mults), k_short_mults=list(k_shorts),
                  dk="k_f on every leg (matches g3_floors_cdev_pin)"),
        rows=rows,
    )
    out["anchor"] = check_anchor(cfg, seed, rows, args.anchor_rtol)

    if not args.skip_brackets:
        # The kill control is a tiled arm at b = 0, and padded_size rounds up to
        # FFT_FRIENDLY (smallest entry 32), so a tile below 32 fine cells gets a
        # buffer it never asked for and the bracket is void -- bracket_controls
        # raises on exactly that. Pick the arm by that property instead of by
        # position, so a legal default exists at every config.
        legal = [i for i, (t, _) in enumerate(args.arms)
                 if padded_size(t, 0, n_fine=n_fine)[1] == 0]
        idx = args.bracket_arm if args.bracket_arm is not None else (legal[-1] if legal else None)
        if idx is None:
            raise SystemExit("no arm can host a b=0 kill control (every tile < 32 fine "
                             "cells); pass --skip-brackets or add a larger tile")
        n_tile, b_fine = args.arms[idx]
        tris, names = fl._triangles(kf, long_mults, args.bracket_k_short or k_shorts[0])
        print(f"  brackets at {arm_tag(n_tile, b_fine)} "
              f"(k_short mult {args.bracket_k_short or k_shorts[0]}) ...", flush=True)
        res = {}
        lad.bracket_controls(B, res, n_tile, b_fine, tris, names)
        out["brackets"] = res["brackets"]

    out["peak_host_rss_bytes"] = int(_host_rss_bytes())
    out["wall_total"] = float(time.perf_counter() - t_all)
    print(f"  total {out['wall_total']:.0f}s, peak host RSS "
          f"{out['peak_host_rss_bytes'] / 1e9:.1f} GB")
    return out


def eligibility_map(card):
    """The FIRST thing printed and the first section of the record.

    Which (buffer, k_short) cells the gate can speak to at all. A verdict read
    without this table in front of it is not interpretable, because an empty row
    looks exactly like a passing one in any table of R_Q alone.
    """
    ks = card["gate"]["k_short_mults"]
    kf = 2.0 * np.pi / card["geometry"]["L"]
    print("\n  ELIGIBILITY (r at k_short; * = gradeable at r >= "
          f"{card['gate']['r_gate']})")
    head = "    arm               b[Mpc/h]  cost  " + "".join(
        f"{k * kf:>9.3f}" for k in ks
    )
    print(head)
    n_ok = 0
    for row in card["rows"]:
        cells = "".join(
            f"{row['cells'][str(k)]['r_k_short']:>8.3f}"
            + ("*" if row["cells"][str(k)]["gradeable"] else " ")
            for k in ks
        )
        n_ok += sum(1 for k in ks if row["cells"][str(k)]["gradeable"])
        print(f"    {row['arm']:16s} {row['b_mpc']:7.1f} {row['vol_ratio']:6.2f}x{cells}")
    print(f"    gradeable cells: {n_ok} of {len(card['rows']) * len(ks)}")
    return n_ok


def verdict_table(card):
    ks = card["gate"]["k_short_mults"]
    print("\n  MAX |R_Q| over gate-eligible squeezed triangles (bar 0.15)")
    for row in card["rows"]:
        parts = []
        for k in ks:
            c = row["cells"][str(k)]
            if c["gradeable"]:
                flag = "" if c.get("rho_agrees", False) else "!rho"
                parts.append(f"{c['max_abs_R_Q']:>8.4f}{flag:<4s}")
            else:
                parts.append(f"{'--':>8s}    ")
        print(f"    {row['arm']:16s} " + "".join(parts))
    print("    -- = not gradeable (arms decorrelated at that k_short); "
          "never read as a pass")


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", default="smoke", choices=("smoke", "cdev8", "cdev", "cgh64"))
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--arms", default=None, help="T:b,T:b,... in FINE CELLS")
    ap.add_argument("--k-short-mults", type=int, nargs="+", default=None,
                    help="k_short shell centres in units of THIS box's k_f")
    ap.add_argument("--long-mults", type=int, nargs="+", default=list(DEFAULT_LONG_MULTS))
    ap.add_argument("--n-sub", type=int, nargs="+", default=list(DEFAULT_N_SUB))
    ap.add_argument("--rho-tol", type=float, default=0.25,
                    help="relative agreement required between max|R_Q| and max|rho|")
    ap.add_argument("--anchor-rtol", type=float, default=0.01)
    ap.add_argument("--bracket-arm", type=int, default=None,
                    help="index into --arms (default: largest tile that can host b=0)")
    ap.add_argument("--bracket-k-short", type=int, default=None)
    ap.add_argument("--skip-brackets", action="store_true")
    ap.add_argument("--skip-response", action="store_true")
    ap.add_argument("--out-suffix", default="")
    args = ap.parse_args()

    args.arms = parse_arms(args.arms) if args.arms else DEFAULT_ARMS[args.config]
    if args.k_short_mults is None:
        args.k_short_mults = list(DEFAULT_K_SHORT[args.config])

    card = measure(args.config, args.seed, args)
    n_ok = eligibility_map(card)
    card["gradeable_cells"] = int(n_ok)
    if n_ok == 0:
        # Not an error: it is the A3 result at this box, and it is stated as one
        # rather than dressed up as a pass.
        print("\n  NOT GRADEABLE ANYWHERE: no (buffer, k_short) cell keeps the arms "
              "correlated at the gate cut.\n  That is a measurement about A3 at this "
              "configuration, NOT a pass.")
    else:
        verdict_table(card)

    os.makedirs(OUT_DIR, exist_ok=True)
    path = os.path.join(OUT_DIR, f"g3_stage5_{args.config}{args.out_suffix}.json")
    with open(path, "w") as f:
        json.dump(card, f, indent=2, default=float)
    print(f"\nwrote {path}")


if __name__ == "__main__":
    main()
