"""Check a P(k) card's nonlinear boost against CAMB halofit / HMcode.

    pixi exec --spec camb --spec matplotlib --spec numpy -- \
        python scripts/v2_pk_boost_reference.py runs/v2/d7f_1013309_hero_pk.json \
        -o figures/hero_pk_boost.png

Run through `pixi exec`, not the project env: camb is a one-off reference here,
not a dependency of the engine, and adding it would move `pixi.lock`.

THE COMPARED QUANTITY IS THE BOOST, B(k) = P_nl(k) / P_lin(k), each side
against ITS OWN linear theory. The card's oracle is full EH98 (`cosmology.py`,
Eq. 16, sigma8-normalized) and CAMB's is a Boltzmann solve; they differ by a
few percent in shape and in the wiggles, and that difference is not what this
is measuring. Taking a ratio of ratios cancels most of it. Comparing P to P
directly would fold the transfer-function difference into the answer and read
as a nonlinear-modeling discrepancy.

WHAT THIS CAN AND CANNOT SETTLE. HMcode2020 is quoted at a few percent for
LCDM over the k and z here, so this is a sanity check at the several-percent
level -- enough to say whether a boost of ~5x at k = 1 is the right size, not
enough to validate one. And at the top of the band the card is measured on the
coarse mesh, where PM force resolution and mass assignment both suppress power
while real nonlinearity raises it; agreement there mixes the two. The band this
argument is clean over is roughly k < 0.5 h/Mpc.
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
        # the engine's Cosmology carries no neutrino mass, so match it rather
        # than inherit CAMB's default 0.06 eV
        mnu=0.0, num_massive_neutrinos=0, omk=0.0,
        halofit_version=version,
    )
    pars.set_matter_power(redshifts=[z], kmax=max(kmax * 2.0, 10.0))

    # sigma8 is an OUTPUT of a Boltzmann solve, so hit the target by rescaling
    # the primordial amplitude: P ~ As, sigma8 ~ sqrt(As).
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
    """EuclidEmulator2's boost at `cos`. Emitted natively, so nothing is
    divided here -- EE2 emulates B(k) directly, which is why it needs no
    Boltzmann solve and why it is the closest match to what the card reports.

    `A_s` comes from the CAMB solve rather than being asked for: the engine's
    Cosmology is normalized by sigma8 and EE2 is parameterized by A_s, so the
    two have to be tied together by an actual calculation.
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
            label=f"inexor 4096$^3$, $z={z:g}$")

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
