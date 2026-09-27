"""Check a P(k) card's nonlinear boost against CAMB halofit / HMcode and EuclidEmulator2.

    pixi exec --spec camb --spec matplotlib --spec numpy -- \
        python scripts/compare/pk_boost_reference.py RUN/realization_pk.json \
        --cosmology RUN/export.json -o figures/hero_pk_boost.png

Run through `pixi exec`, not the project env: camb is a one-off reference, not an
engine dependency, and adding it would move `pixi.lock`. EuclidEmulator2 is used if
`euclidemu2` imports (see `ee2_ratio_figure.py` for its install), else skipped.

The compared quantity is the boost B(k) = P_nl(k) / P_lin(k), each side against its
own linear theory: the card's oracle is EH98 (`cosmology.py`, sigma8-normalized) and
CAMB's is a Boltzmann solve. They differ by a few percent in shape and wiggles; the
ratio of ratios cancels most of that, where P vs P would read it as a nonlinear-
modeling discrepancy.

Scope: HMcode2020 is quoted at a few percent for LCDM here, so this is a sanity
check at the several-percent level (is a ~5x boost at k = 1 the right size), not a
validation. At the top of the band the card is on the coarse mesh, where PM force
resolution and mass assignment suppress power while nonlinearity raises it; the
comparison is clean for roughly k < 0.5 h/Mpc. The cosmology must come from the card
or `--cosmology`; there is no default.
"""

import argparse
import json

import numpy as np

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402


def camb_boost(cos, z, kmax, version):
    """B(k) = P_nl/P_lin from CAMB at `cos`, normalized to its sigma8."""
    import camb

    h = cos["h"]
    pars = camb.set_params(
        H0=100.0 * h,
        ombh2=cos["Omega_b"] * h * h,
        omch2=(cos["Omega_m"] - cos["Omega_b"]) * h * h,
        ns=cos["n_s"],
        TCMB=cos.get("T_cmb_K", 2.7255),
        # the engine's Cosmology has massless neutrinos; CAMB defaults to 0.06 eV
        mnu=0.0, num_massive_neutrinos=0, omk=0.0,
        halofit_version=version,
    )
    pars.set_matter_power(redshifts=[z], kmax=max(kmax * 2.0, 10.0))

    # sigma8 is an output of the solve: rescale As to hit it (sigma8 ~ sqrt(As))
    pars.NonLinear = camb.model.NonLinear_none
    s8 = camb.get_results(pars).get_sigma8_0()
    pars.InitPower.As = pars.InitPower.As * (cos["sigma8"] / s8) ** 2
    lin = camb.get_results(pars)
    kh, _, plin = lin.get_matter_power_spectrum(minkh=1e-3, maxkh=kmax, npoints=400)

    pars.NonLinear = camb.model.NonLinear_both
    nl = camb.get_results(pars)
    kh2, _, pnl = nl.get_matter_power_spectrum(minkh=1e-3, maxkh=kmax, npoints=400)
    assert np.allclose(kh, kh2)
    return kh, plin[0], pnl[0], float(nl.get_sigma8_0()), float(pars.InitPower.As)


def ee2_boost(cos, z, k, A_s):
    """EuclidEmulator2's boost B(k) at `cos` and redshift `z`, evaluated at `k`.

    EE2 emulates B(k) directly, so nothing is divided here. `A_s` comes from a CAMB
    solve: the engine is sigma8-normalized and EE2 is parameterized by A_s. Refuses a
    cosmology outside EE2's training range (an emulator off-range returns silently).
    """
    import euclidemu2

    e = euclidemu2.PyEuclidEmulator()
    par = dict(Omega_b=cos["Omega_b"], Omega_m=cos["Omega_m"], m_ncdm=0.0,
               n_s=cos["n_s"], h=cos["h"], w0_fld=-1.0, wa_fld=0.0, A_s=A_s)
    for name, (lo, hi) in e.bounds.items():
        if not (lo <= par[name] <= hi):
            raise SystemExit(
                f"EE2 is not valid at this cosmology: {name}={par[name]:g} is "
                f"outside its training range [{lo:g}, {hi:g}]. An emulator "
                f"evaluated off its range returns a number and no warning."
            )
    kv, bz = e.get_boost(par, [float(z)], custom_kvec=np.asarray(k, float))
    return np.asarray(bz[0], float)


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("card")
    ap.add_argument("--cosmology", default=None,
                    help="a JSON file carrying a `cosmology` block (the export "
                         "header has one). Required unless the card does -- there "
                         "is no default, because a reference built at the wrong "
                         "parameters looks exactly like a discrepancy")
    ap.add_argument("-o", "--out", default="figures/hero_pk_boost.png")
    ap.add_argument("--no-ee2", action="store_true",
                    help="skip EuclidEmulator2 even if it imports")
    ap.add_argument("--dpi", type=int, default=200)
    args = ap.parse_args()

    with open(args.card) as fh:
        card = json.load(fh)
    s = card["summary"]
    k = np.asarray(s["k_mean"], float)
    boost = np.asarray(s["p"], float) / np.asarray(s["p_oracle"], float)
    a_out = float(card["a_out"])
    z = 1.0 / a_out - 1.0
    cos = card.get("cosmology") or s.get("cosmology")
    if cos is None and args.cosmology:
        with open(args.cosmology) as fh:
            cos = json.load(fh).get("cosmology")
    if cos is None:
        raise SystemExit(
            "no cosmology: this card carries none (cards written before this "
            "check existed do not), so pass --cosmology pointing at a JSON with "
            "a `cosmology` block -- the export header has one. There is no "
            "default on purpose."
        )
    print(f"  cosmology {cos}")

    fig, (ax, bx) = plt.subplots(1, 2, figsize=(11.0, 4.3))
    ax.plot(k, boost, ls="-", lw=1.9, color="C0", marker="o", ms=2.8,
            label=f"inexor {round(s['n_particles'] ** (1 / 3))}$^3$, $z={z:g}$")

    styles = {"mead2020": ("C1", "--", "HMcode2020"),
              "takahashi": ("C2", ":", "halofit (Takahashi)")}
    refs = {}
    A_s = None
    for version, (c, ls, lab) in styles.items():
        kh, plin, pnl, s8, a_s = camb_boost(cos, z, float(k.max()) * 1.2, version)
        A_s = a_s
        b = pnl / plin
        refs[version] = np.interp(k, kh, b)
        ax.plot(kh, b, ls=ls, lw=1.6, color=c, label=f"{lab}, $\\sigma_8={s8:.3f}$")
        print(f"  {lab:22s} sigma8={s8:.4f}")
    print(f"  A_s for sigma8={cos['sigma8']:g}: {A_s:.4e}")

    if not args.no_ee2:
        try:
            refs["ee2"] = ee2_boost(cos, z, k, A_s)
            styles["ee2"] = ("C3", "-.", "EuclidEmulator2")
            ax.plot(k, refs["ee2"], ls="-.", lw=1.6, color="C3",
                    label="EuclidEmulator2")
            print("  EuclidEmulator2        boost emulated natively")
        except ImportError:
            print("  EuclidEmulator2        not installed; skipped")

    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlim(k.min() * 0.9, k.max() * 1.05)
    ax.set_xlabel(r"$k\ \ [h\,{\rm Mpc}^{-1}]$")
    ax.set_ylabel(r"$B(k)=P_{\rm nl}/P_{\rm lin}$")
    ax.legend(frameon=False, fontsize=9)

    for version, (c, ls, lab) in styles.items():
        bx.plot(k, boost / refs[version], ls=ls, lw=1.7, color=c, marker="o", ms=2.8,
                label=lab)
    bx.axhline(1.0, lw=1.0, color="0.45", ls="-")
    bx.fill_between([k.min() * 0.9, k.max() * 1.05], 0.975, 1.025, color="0.6",
                    alpha=0.18, lw=0, label="HMcode2020 quoted accuracy")
    bx.set_xscale("log")
    bx.set_xlim(k.min() * 0.9, k.max() * 1.05)
    bx.set_xlabel(r"$k\ \ [h\,{\rm Mpc}^{-1}]$")
    bx.set_ylabel(r"$B_{\rm inexor}\,/\,B_{\rm reference}$")
    bx.legend(frameon=False, fontsize=9, loc="lower left")

    fig.tight_layout()
    fig.savefig(args.out, dpi=args.dpi, bbox_inches="tight")
    print(f"-> {args.out}")

    names = [n for n in ("ee2", "mead2020", "takahashi") if n in refs]
    print("\n  " + f"{'k':>8} {'B_sim':>8} " +
          " ".join(f"{styles[n][2][:9]:>10}" for n in names))
    for i in range(len(k)):
        if i % 4 and k[i] > 0.15:
            continue
        print(f"  {k[i]:8.4f} {boost[i]:8.3f} " +
              " ".join(f"{boost[i] / refs[n][i]:10.4f}" for n in names))


if __name__ == "__main__":
    main()
