"""M1 S5: export matched ICs + linear P(k) tables for the cross-code parity harness.

HISTORICAL, OLD-STREAM GENERATOR (M-v2-5, 2026-08-10): `inexor.ic` was
replaced in place (D-v2-15 clause 5 -- a seed now denotes a DIFFERENT
realization), so re-running this at HEAD writes ICs the stored
`runs/m1/mbody_final_*` / `disco_final_*` references were never run from,
severing the cross-code correspondence under the same tag scheme. The
stored `runs/m1/ics_*.npz` remain valid (the D-013 arms LOAD them, never
regenerate); do not re-export over them.

Runs in the INEXOR env (this repo, default env):
    JAX_ENABLE_X64 is set below -- exports are float64 end to end.

    pixi run python scripts/m1_export_ics.py --n 64 --steps 10
    pixi run -e parity python scripts/m1_export_ics.py --dump-camb-table

Writes to runs/m1/:
    ics_<tag>.npz      -- delta0 + (x, v_d) at a_init + shared a_steps (schema
                          in _m1_common; one file per lpt_order in --lpt)
    pk_eh98.txt        -- two-column (k [h/Mpc], P [(Mpc/h)^3]) z=0 table from
                          inexor's EH98 (DISCO-DJ from_file format, verified)
    pk_camb.txt        -- same, from CAMB (--dump-camb-table; parity env only).
                          Normalized to COSMO's sigma8 like mbody's backend.

The matched-IC injection decouples dynamics parity from linear theory: every
code starts from the SAME (x, v_d), so Tier-A differences are the integrator/
force port, not the spectrum. The P(k) tables serve DISCO-DJ's builder and the
S6 {EH98, CAMB} deficit axis.
"""

import argparse
import os
import sys

os.environ.setdefault("JAX_ENABLE_X64", "1")

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import _m1_common as M  # noqa: E402

REPO = os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))


def export_ics(args):
    import jax

    jax.config.update("jax_enable_x64", True)
    import jax.numpy as jnp

    from inexor import config, integrate, lpt
    from inexor.ic import gaussian_delta

    cosmo = config.Cosmology(**M.COSMO)
    a_steps = M.a_grid(args.a_init, args.a_final, args.steps, args.spacing)
    a_inexor = integrate.a_grid(args.a_init, args.a_final, args.steps, args.spacing)
    assert np.array_equal(a_steps, a_inexor), "a_grid mismatch between _m1_common and inexor"

    table = None
    if args.pk_table:
        d = np.loadtxt(args.pk_table)
        table = (d[:, 0], d[:, 1])
    key = jax.random.PRNGKey(args.seed)
    delta0 = gaussian_delta(
        key,
        args.n,
        M.BOX_SIZE,
        cosmo,
        fdtype=jnp.float64,
        backend="table" if table else "eh98",
        table=table,
    )

    g_i = integrate._G_f(args.a_init, cosmo)
    g_f = integrate._G_f(args.a_final, cosmo)
    for order in args.lpt:
        x, v_d = lpt.lpt_ics(  # v1 references: EdS D2 (D-013)
            delta0, M.BOX_SIZE, args.a_init, cosmo, order=order, fdtype=jnp.float64,
            growth2="eds",
        )
        tag = M.run_tag(args.n, args.steps, args.spacing, args.integrator, order, args.seed)
        cfg = dict(
            n_mesh=args.n,
            box_size=M.BOX_SIZE,
            a_init=args.a_init,
            a_final=args.a_final,
            n_steps=args.steps,
            spacing=args.spacing,
            integrator=args.integrator,
            lpt_order=order,
            seed=args.seed,
        )
        meta = M.make_meta(
            "inexor",
            "export-ics",
            cfg,
            REPO,
            pk_source=os.path.basename(args.pk_table) if args.pk_table else "eh98",
            G_f_a_init=g_i,
            G_f_a_final=g_f,
        )
        path = os.path.join(M.RUNS, f"ics_{tag}.npz")
        M.save_state(
            path, np.asarray(x), np.asarray(v_d), meta, delta0=np.asarray(delta0), a_steps=a_steps
        )
        print(f"wrote {path}  (x {x.shape} {x.dtype}, max|v_d| {float(np.abs(v_d).max()):.4f})")


def dump_pk_eh98():
    from inexor import config
    from inexor.cosmology import linear_power

    cosmo = config.Cosmology(**M.COSMO)
    k = np.geomspace(1e-4, 1e2, 800)
    P = linear_power(k, cosmo, z=0.0, backend="eh98")
    path = os.path.join(M.RUNS, "pk_eh98.txt")
    os.makedirs(M.RUNS, exist_ok=True)
    np.savetxt(
        path,
        np.column_stack([k, P]),
        fmt="%.18e",
        header="k [h/Mpc]   P(k) [(Mpc/h)^3]  z=0  inexor EH98 sigma8-normalized",
    )
    print(f"wrote {path}")


def dump_pk_camb():
    """CAMB z=0 P(k) at COSMO, sigma8-rescaled (mirrors mbody._camb_pk0 verbatim
    so this table and mbody's internal backend share one convention)."""
    import camb
    from camb import model

    c = M.COSMO
    h = c["h"]
    pars = camb.set_params(
        H0=100.0 * h,
        ombh2=c["Omega_b"] * h * h,
        omch2=(c["Omega_m"] - c["Omega_b"]) * h * h,
        ns=c["n_s"],
        As=2.0e-9,
        mnu=0.0,
        omk=0.0,
        TCMB=2.7255,
    )
    pars.set_matter_power(redshifts=[0.0], kmax=100.0 * h * 1.2)
    pars.NonLinear = model.NonLinear_none
    results = camb.get_results(pars)
    kh, _, pk0 = results.get_matter_power_spectrum(minkh=1e-4, maxkh=1e2, npoints=800)
    try:
        sigma8_camb = float(results.get_sigma8_0())
    except Exception:
        sigma8_camb = float(results.get_sigma8()[-1])
    P = pk0[0] * (c["sigma8"] / sigma8_camb) ** 2
    path = os.path.join(M.RUNS, "pk_camb.txt")
    os.makedirs(M.RUNS, exist_ok=True)
    np.savetxt(
        path,
        np.column_stack([kh, P]),
        fmt="%.18e",
        header=f"k [h/Mpc]   P(k) [(Mpc/h)^3]  z=0  CAMB {camb.__version__} "
        f"sigma8-rescaled from {sigma8_camb:.6f} to {c['sigma8']}",
    )
    print(f"wrote {path}  (sigma8_camb before rescale: {sigma8_camb:.6f})")


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--n", type=int, default=64)
    ap.add_argument("--steps", type=int, default=10)
    ap.add_argument("--spacing", default="log", choices=["log", "linear"])
    ap.add_argument("--integrator", default="bullfrog", choices=["bullfrog", "fastpm", "exact"])
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--a-init", type=float, default=M.A_INIT)
    ap.add_argument("--a-final", type=float, default=M.A_FINAL)
    ap.add_argument("--lpt", type=int, nargs="+", default=[1, 2], choices=[1, 2])
    ap.add_argument(
        "--pk-table", default=None, help="color delta0 from this (k, P) table instead of EH98"
    )
    ap.add_argument(
        "--dump-camb-table",
        action="store_true",
        help="only write pk_camb.txt (requires the parity env)",
    )
    ap.add_argument("--skip-ics", action="store_true", help="only write the P(k) tables")
    args = ap.parse_args()

    if args.dump_camb_table:
        dump_pk_camb()
        return
    dump_pk_eh98()
    if not args.skip_ics:
        export_ics(args)


if __name__ == "__main__":
    main()
