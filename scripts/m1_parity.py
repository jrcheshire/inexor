"""M1 S5: inexor arms + the cross-code parity comparator (runs in the INEXOR env).

    pixi run python scripts/m1_parity.py run --tag n64k10log_bullfrog_lpt2_s0
    pixi run python scripts/m1_parity.py compare --tag n64k10log_bullfrog_lpt2_s0 \
        --a inexor_float --b disco_final

`run` computes the two inexor arms from the injected ICs:
    inexor_float_<tag>.npz -- evolve_float in f64 (never quantized): the
        Tier-A arm, isolating the physics port from quantization.
    inexor_int16_<tag>.npz -- the production quantized path (int16 phase
        space, int paint, f32 kernels): Tier B = int16 vs inexor_float
        against the D-010 relative-P(k) <= 1e-4 bar.

`compare` measures any two final-state npz files with the SAME neutral
numpy estimator (_m1_common), writes parity_<a>_vs_<b>_<tag>.json and a
two-panel figure (P ratio + r(k)) under runs/m1/.
"""

import argparse
import json
import os
import sys

os.environ.setdefault("JAX_ENABLE_X64", "1")

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import _m1_common as M  # noqa: E402

REPO = os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
ARMS = ("inexor_float", "inexor_int16", "mbody_final", "mbody_own", "disco_final", "disco_own")


def run_inexor(tag):
    import jax

    jax.config.update("jax_enable_x64", True)
    import jax.numpy as jnp

    from inexor import config, integrate

    ics = M.load_state(os.path.join(M.RUNS, f"ics_{tag}.npz"))
    cfg = ics["meta"]["config"]
    box = config.BoxConfig(n_mesh=cfg["n_mesh"], box_size=cfg["box_size"])
    time = config.TimeConfig(
        a_init=cfg["a_init"],
        a_final=cfg["a_final"],
        n_steps=cfg["n_steps"],
        spacing=cfg["spacing"],
        integrator=cfg["integrator"],
    )
    quant = config.QuantConfig()
    cosmo = config.Cosmology(**M.COSMO)
    x0 = jnp.asarray(ics["x"])
    v0 = jnp.asarray(ics["v_d"])

    xf, vf = integrate.evolve_float(
        box, time, cosmo, x0.astype(jnp.float64), v0.astype(jnp.float64), fdtype=jnp.float64
    )
    meta = M.make_meta("inexor", "float64", cfg, REPO, ic_file=f"ics_{tag}.npz")
    p = M.save_state(
        os.path.join(M.RUNS, f"inexor_float_{tag}.npz"),
        np.mod(np.asarray(xf), box.box_size),
        np.asarray(vf),
        meta,
        a_steps=ics["a_steps"],
    )
    print(f"wrote {p}")

    xq, vq = integrate.evolve(
        box,
        time,
        quant,
        cosmo,
        x0.astype(jnp.float32),
        v0.astype(jnp.float32),
        driver="scan",
        paint="int",
        fdtype=jnp.float32,
    )
    meta = M.make_meta(
        "inexor",
        "int16",
        cfg,
        REPO,
        ic_file=f"ics_{tag}.npz",
        quant=dict(
            x_bits=quant.x_bits,
            frac_bits=quant.frac_bits,
            c_growth=quant.c_growth,
            margin=quant.margin,
        ),
    )
    p = M.save_state(
        os.path.join(M.RUNS, f"inexor_int16_{tag}.npz"),
        np.mod(np.asarray(xq), box.box_size),
        np.asarray(vq),
        meta,
        a_steps=ics["a_steps"],
    )
    print(f"wrote {p}")


def compare(tag, name_a, name_b, fig=True):
    sa = M.load_state(os.path.join(M.RUNS, f"{name_a}_{tag}.npz"))
    sb = M.load_state(os.path.join(M.RUNS, f"{name_b}_{tag}.npz"))
    cfg = sa["meta"]["config"]
    out = M.compare_states(sa["x"], sb["x"], cfg["n_mesh"], cfg["box_size"])
    out["a"], out["b"], out["tag"] = name_a, name_b, tag
    out["meta_a"], out["meta_b"] = sa["meta"], sb["meta"]
    jpath = os.path.join(M.RUNS, f"parity_{name_a}_vs_{name_b}_{tag}.json")
    with open(jpath, "w") as f:
        json.dump(out, f, indent=1)
    print(f"wrote {jpath}")
    print(
        f"  {name_a} vs {name_b}: rms {out['rms_cells']:.3e} cells, "
        f"max |dP/P| {out['ratio_max_absdev']:.3e}, max 1-r {out['one_minus_r_max']:.3e}"
    )

    if fig:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        k = np.array(out["k"])
        fig_, (ax1, ax2) = plt.subplots(2, 1, figsize=(6.5, 6.5), sharex=True)
        ax1.semilogx(k, np.array(out["p_ratio"]) - 1.0, "o-", ms=3)
        ax1.axhline(0.0, color="k", lw=0.6)
        ax1.set_ylabel(f"P_{name_b} / P_{name_a} - 1")
        ax2.semilogx(k, 1.0 - np.array(out["r"]), "o-", ms=3)
        ax2.set_yscale("symlog", linthresh=1e-12)
        ax2.axhline(0.0, color="k", lw=0.6)
        ax2.set_xlabel("k [h/Mpc]")
        ax2.set_ylabel("1 - r(k)")
        fig_.suptitle(f"{name_a} vs {name_b}  ({tag})", fontsize=10)
        fig_.tight_layout()
        fpath = os.path.join(M.RUNS, f"parity_{name_a}_vs_{name_b}_{tag}.png")
        fig_.savefig(fpath, dpi=150)
        print(f"wrote {fpath}")
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    sub = ap.add_subparsers(dest="cmd", required=True)
    ap_run = sub.add_parser("run", help="compute the inexor arms from injected ICs")
    ap_run.add_argument("--tag", required=True)
    ap_cmp = sub.add_parser("compare", help="compare two final-state npz files")
    ap_cmp.add_argument("--tag", required=True)
    ap_cmp.add_argument("--a", required=True, choices=ARMS)
    ap_cmp.add_argument("--b", required=True, choices=ARMS)
    ap_cmp.add_argument("--no-fig", action="store_true")
    args = ap.parse_args()

    if args.cmd == "run":
        run_inexor(args.tag)
    else:
        compare(args.tag, args.a, args.b, fig=not args.no_fig)


if __name__ == "__main__":
    main()
