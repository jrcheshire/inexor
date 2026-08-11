"""M-v2-5 exit gate B: the transfer table's error, measured not trusted.

THE ESTIMAND (JC 2026-08-10): e_max(cfg) = max over k in [k_f, sqrt(3)*k_Nyq]
of |P_table(k) / P_eh98(k) - 1|, per config row of the production table --
C-dev (n=256, L=128), C-gh (2048, 1024), C-hero (4096, 2048). The field-level
effect is reported as CONTEXT, not gated (a relative P error eps moves the
colour, hence the field, by eps/2 in amplitude).

HOW A MAX OVER ~n^3/2 REALIZED |k| IS TAKEN WITHOUT A 3D GRID. Every realized
|k| lies in the continuous gate range, so a bound on the continuous max bounds
the realized max. e(k) is smooth INSIDE each table interval (difference of two
smooth 1D functions; kinks only at nodes) but NOT monotone -- the BAO band
makes it oscillatory -- so the instrument is a per-interval dense sweep:
33 Chebyshev-in-ln(k) points per table interval overlapping the range, max over
all. Two honesty checks ride every sweep: (1) a 65-point re-sweep must move the
max < 1% (else 33 under-resolves and the sweep densifies -- a pre-registered
falsifier, not a knob); (2) at C-dev ONLY, the exact realized-|k| multiset
(8.5e6 values -- affordable at dev scale precisely because production must
never materialize it) must read <= the sweep max.

THE BAR IS TWO-TIER.

  Tier 1 -- the BAR: e_max < 1e-4 per config, at the production table density.
  The charter number (D-v2-18's ladder row), estimand pinned by JC 2026-08-10.

  Tier 2 -- the pre-registered EXPECTATION: log-log linear interpolation error
  goes as (h^2/8) * max|d^2 lnP / d lnk^2| with h the node spacing in ln k.
  BAO curvature dominates: |f''| ~ 5-9 around k ~ 0.1-0.2. At 4000 points over
  the production range that derives to ~1e-5 (at most 3e-5; above 5e-5 is a
  FINDING). At the 16384-point default, h^2 scaling predicts ~16x less, ~6e-7.
  The h^2-scaling sub-leg across {800, 2048, 4000, 16384} is the falsifier: if
  e_max does not fall ~quadratically in density, interpolation is NOT the
  dominant term and the table build is wrong somewhere else.

Reported beside the max, ungated: argmax k and a per-decade error profile
(g5b_abs_transfer lesson -- a max over a band can hide the shape).
"""

import argparse
import hashlib
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(HERE)
OUT_DIR = os.path.join(REPO, "runs", "v2")

# (n_mesh, box_size) of the IC (particle) grid per config-table row.
CONFIGS = {
    "cdev": (256, 128.0),
    "cgh": (2048, 1024.0),
    "chero": (4096, 2048.0),
}

# NB nodes are UNIVERSAL (cosmology.K_TABLE_MIN..MAX, ~6 decades) rather than
# per-grid (~3.5 decades) since the G5b shared-modes finding of 2026-08-10;
# the production density is sized for the wider span.
DENSITY_LADDER = (800, 2048, 4000, 16384, 32768)
N_POINTS_PRODUCTION = 32768


def _sweep_e_max(tab, cosmo, k_lo, k_hi, pts_per_interval):
    """Max |P_tab/P_eh98 - 1| over Chebyshev-in-ln(k) points in every table
    interval overlapping [k_lo, k_hi]. Returns (e_max, k_argmax, e_of_k_pts)."""
    from inexor.cosmology import linear_power

    lnk = np.log(tab.k)
    lo = np.searchsorted(tab.k, k_lo, side="right") - 1
    hi = np.searchsorted(tab.k, k_hi, side="left")
    lo = max(lo, 0)
    a = lnk[lo:hi]  # interval left edges
    b = lnk[lo + 1 : hi + 1]  # interval right edges
    mid = 0.5 * (a + b)
    half = 0.5 * (b - a)
    # Chebyshev nodes in each interval (interior; endpoints are table nodes
    # where the error is ~0 by construction)
    j = np.arange(pts_per_interval)
    theta = np.pi * (2 * j + 1) / (2 * pts_per_interval)
    ln_pts = mid[:, None] + half[:, None] * np.cos(theta)[None, :]
    k_pts = np.exp(ln_pts.ravel())
    k_pts = k_pts[(k_pts >= k_lo) & (k_pts <= k_hi)]
    e = np.abs(tab.P_of_k(k_pts) / linear_power(k_pts, cosmo) - 1.0)
    i = int(np.argmax(e))
    return float(e[i]), float(k_pts[i]), k_pts, e


def _per_decade_profile(k_pts, e):
    edges = 10.0 ** np.arange(np.floor(np.log10(k_pts.min())), np.ceil(np.log10(k_pts.max())) + 1)
    prof = []
    for d_lo, d_hi in zip(edges[:-1], edges[1:]):
        m = (k_pts >= d_lo) & (k_pts < d_hi)
        if m.any():
            prof.append(dict(k_lo=float(d_lo), k_hi=float(d_hi), e_max=float(e[m].max())))
    return prof


def run_config(name, n_mesh, box_size, cosmo):
    from inexor.cosmology import ic_k_table, linear_power

    k_f = 2.0 * np.pi / box_size
    k_hi = np.sqrt(3.0) * np.pi * n_mesh / box_size
    res = dict(config=name, n_mesh=n_mesh, box_size=box_size, k_f=k_f, k_hi=float(k_hi))
    print(f"config {name}: n={n_mesh} L={box_size} gate range [{k_f:.4e}, {k_hi:.4e}]", flush=True)

    tab = ic_k_table(cosmo, n_mesh, box_size, n_points=N_POINTS_PRODUCTION)
    res["table_spec"] = dict(
        n_points=N_POINTS_PRODUCTION,
        k_min=float(tab.k[0]),
        k_max=float(tab.k[-1]),
        backend="eh98",
        sha256_kPT=hashlib.sha256(
            tab.k.tobytes() + tab.P.tobytes() + tab.T.tobytes()
        ).hexdigest(),
    )

    e33, k33, k_pts, e = _sweep_e_max(tab, cosmo, k_f, k_hi, 33)
    e65, k65, _, _ = _sweep_e_max(tab, cosmo, k_f, k_hi, 65)
    drift = abs(e65 - e33) / e33 if e33 > 0 else 0.0
    res["sweep"] = dict(e_max_33=e33, e_max_65=e65, argmax_k=k33, resweep_drift=float(drift))
    res["per_decade"] = _per_decade_profile(k_pts, e)
    res["honesty_resweep_ok"] = bool(drift < 0.01)
    print(f"  e_max 33/interval {e33:.3e} at k={k33:.4f}; 65/interval {e65:.3e} "
          f"(drift {drift:.2%})", flush=True)

    # honesty check 2, C-dev only: exact realized multiset <= the sweep max
    if name == "cdev":
        kx = 2.0 * np.pi * np.fft.fftfreq(n_mesh, d=box_size / n_mesh)
        kz = 2.0 * np.pi * np.fft.rfftfreq(n_mesh, d=box_size / n_mesh)
        kk = np.sqrt(
            kx.reshape(-1, 1, 1) ** 2 + kx.reshape(1, -1, 1) ** 2 + kz.reshape(1, 1, -1) ** 2
        ).ravel()
        kk = kk[kk > 0]
        e_grid = float(
            np.max(np.abs(tab.P_of_k(kk) / linear_power(kk, cosmo) - 1.0))
        )
        res["multiset_e_max"] = e_grid
        res["honesty_multiset_ok"] = bool(e_grid <= max(e33, e65) * (1 + 1e-12))
        print(f"  exact multiset e_max {e_grid:.3e} (<= sweep: {res['honesty_multiset_ok']})",
              flush=True)

    res["tier1_pass"] = bool(max(e33, e65) < 1e-4)
    res["tier2_finding"] = bool(max(e33, e65) > 5e-5)
    return res


def run_density_ladder(cosmo):
    """The h^2 falsifier: e_max across table densities at the C-dev row."""
    from inexor.cosmology import ic_k_table

    n_mesh, box_size = CONFIGS["cdev"]
    k_f = 2.0 * np.pi / box_size
    k_hi = np.sqrt(3.0) * np.pi * n_mesh / box_size
    rungs = []
    for npts in DENSITY_LADDER:
        tab = ic_k_table(cosmo, n_mesh, box_size, n_points=npts)
        h = float(np.log(tab.k[-1] / tab.k[0]) / (npts - 1))
        e_max, k_arg, _, _ = _sweep_e_max(tab, cosmo, k_f, k_hi, 33)
        rungs.append(dict(n_points=npts, h=h, e_max=e_max, argmax_k=k_arg))
        print(f"  density {npts}: h={h:.3e} e_max={e_max:.3e} at k={k_arg:.4f}", flush=True)
    lg_h = np.log([r["h"] for r in rungs])
    lg_e = np.log([r["e_max"] for r in rungs])
    slope = float(np.polyfit(lg_h, lg_e, 1)[0])
    print(f"  scaling slope d ln(e_max)/d ln(h) = {slope:.3f} (h^2 predicts 2)", flush=True)
    return dict(rungs=rungs, slope=slope, h2_ok=bool(1.6 < slope < 2.4))


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out-suffix", default="")
    args = ap.parse_args()

    from inexor.config import Cosmology

    cosmo = Cosmology()
    res = dict(configs=[run_config(name, n, L, cosmo) for name, (n, L) in CONFIGS.items()])
    print("density ladder (h^2 falsifier):", flush=True)
    res["density_ladder"] = run_density_ladder(cosmo)

    ok = (
        all(c["tier1_pass"] for c in res["configs"])
        and all(c["honesty_resweep_ok"] for c in res["configs"])
        and all(c.get("honesty_multiset_ok", True) for c in res["configs"])
        and res["density_ladder"]["h2_ok"]
    )
    res["ok"] = bool(ok)

    sys.path.insert(0, HERE)
    import v2_m5_ic_gate as m5

    class _A:
        leg = "table-bar"
        out_suffix = args.out_suffix

    m5.OUT_DIR = OUT_DIR
    m5._write(res, _A())
    findings = [c["config"] for c in res["configs"] if c["tier2_finding"]]
    if findings:
        print(f"TIER-2 FINDING (> 5e-5) at: {findings}", flush=True)
    print("PASS" if ok else "FAIL", flush=True)
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
