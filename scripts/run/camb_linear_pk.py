"""A z = 0 linear P(k) table from CAMB, for `--pk-table` (format inexor-linear-pk-1).

    pixi exec --spec python=3.12 --spec camb --spec numpy -- \
        python scripts/run/camb_linear_pk.py -o data/linear_pk_camb.json

Run through `pixi exec`: camb is a one-off input maker, not an engine dependency. The
cosmology is the engine's default (`inexor.config.Cosmology`), or the `cosmology` block of
`--cosmology FILE` (an export header has one). CAMB is set to match the engine: flat LCDM,
massless neutrinos, T_CMB from the cosmology, and A_s solved so sigma8 is the cosmology's.
k is log-spaced over [1e-4, 1e2] h/Mpc (`cosmology.K_TABLE_MIN/MAX`, which the loader
requires) with a margin either side. The file is `LinearPkTable.record()`'s format; the
loader (`cosmology.load_linear_pk`) re-checks the cosmology, sigma8 and k range.
"""

import argparse
import dataclasses
import importlib.util
import json
import os
import sys

import numpy as np

FORMAT = "inexor-linear-pk-1"
K_LO, K_HI = 1e-4 / 1.05, 1e2 * 1.05


def default_cosmology():
    """The engine's default cosmology, read from src/inexor/config.py without importing the
    package (this script runs outside the project env)."""
    path = os.path.join(os.path.dirname(os.path.abspath(__file__)), os.pardir, os.pardir,
                        "src", "inexor", "config.py")
    spec = importlib.util.spec_from_file_location("inexor_config", path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod  # dataclasses resolves the module by name
    spec.loader.exec_module(mod)
    return dataclasses.asdict(mod.Cosmology())


def camb_linear(cos, k_hmpc):
    """(P(k) at z = 0 on `k_hmpc`, its sigma8, A_s, camb version) for the cosmology dict."""
    import camb

    h = cos["h"]
    pars = camb.set_params(
        H0=100.0 * h,
        ombh2=cos["Omega_b"] * h * h,
        omch2=(cos["Omega_m"] - cos["Omega_b"]) * h * h,
        ns=cos["n_s"],
        TCMB=cos["T_cmb_K"],
        mnu=0.0, num_massive_neutrinos=0, omk=0.0,
    )
    pars.set_matter_power(redshifts=[0.0], kmax=float(k_hmpc[-1]) * h * 1.2)
    pars.NonLinear = camb.model.NonLinear_none
    # sigma8 is an output of the solve and scales as sqrt(A_s) in linear theory
    s8 = camb.get_results(pars).get_sigma8_0()
    pars.InitPower.As = pars.InitPower.As * (cos["sigma8"] / s8) ** 2
    res = camb.get_results(pars)
    interp = res.get_matter_power_interpolator(nonlinear=False, hubble_units=True,
                                               k_hunit=True, extrap_kmax=False)
    P = interp.P(0.0, k_hmpc)
    return np.asarray(P, dtype=np.float64).ravel(), float(res.get_sigma8_0()), \
        float(pars.InitPower.As), camb.__version__


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("-o", "--out", required=True, help="the table (JSON)")
    ap.add_argument("--cosmology", default=None,
                    help="a JSON file with a `cosmology` block (default: the engine's)")
    ap.add_argument("--n-points", type=int, default=4000,
                    help="log-spaced k nodes (default 4000: ~60 per BAO period at k = 0.2)")
    args = ap.parse_args(argv)

    cos = default_cosmology()
    if args.cosmology:
        with open(args.cosmology) as fh:
            cos = json.load(fh)["cosmology"]
    k = np.geomspace(K_LO, K_HI, int(args.n_points))
    P, s8, A_s, version = camb_linear(cos, k)
    rec = dict(format=FORMAT, source="camb", z=0.0, cosmology=cos,
               generator=dict(code="camb", version=version, A_s=A_s, sigma8=s8,
                              massless_neutrinos=True),
               k=k.tolist(), P=P.tolist())
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    with open(args.out, "w") as fh:
        json.dump(rec, fh)
    print(f"  {args.out}: camb {version}, {k.size} points over k = [{k[0]:.3e}, {k[-1]:.3e}] "
          f"h/Mpc, sigma8 {s8:.6f} (target {cos['sigma8']}), A_s {A_s:.4e}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
