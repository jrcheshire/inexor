"""M1 S5: inexor arms + the cross-code parity comparator (runs in the INEXOR env).

    pixi run python scripts/m1_parity.py run --tag n64k10log_bullfrog_lpt2_s0
    pixi run python scripts/m1_parity.py compare --tag n64k10log_bullfrog_lpt2_s0 \
        --a inexor_float --b disco_final

`run` recomputes the never-quantized f64 arm from the injected ICs, into
`inexor_replumb_<tag>.npz` -- BESIDE the stored `inexor_float_<tag>.npz` rather
than over it, so the stored v1 reference stays intact and comparing the two is a
real regression test on the shared force/paint stack.

    inexor_float_<tag>.npz    -- the STORED Tier-A reference (v1, D-013).
    inexor_replumb_<tag>.npz  -- the same thing recomputed today.
    inexor_int16_<tag>.npz    -- STORED ONLY. Its generator was deleted at the
        v1 retirement and cannot be re-plumbed: T9 is a different tier in kind,
        so a T9 number is a different quantity at a different quantum rather
        than a re-read of D-014's ratified 5e-4.

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
ARMS = (
    "inexor_float", "inexor_replumb", "inexor_replumb32", "inexor_int16",
    "mbody_final", "mbody_own", "disco_final", "disco_own",
)


def run_inexor(tag, kernel_dtype="f64"):
    """Regenerate the never-quantized f64 arm. RE-PLUMBED at M-v2-3.

    This called `integrate.evolve_float` and `integrate.evolve`, both deleted at
    the v1 retirement, so the whole `run` path had been dead since 2026-08-08
    while `compare` stayed alive on the stored npz. The float arm re-plumbs in
    six lines from the pieces that survived -- `make_force_fn` +
    `bullfrog_float_coeffs` + `float_step_bullfrog` -- which is the same loop
    `m1_kernel_probe.py:58-77` already writes.

    **The int16 arm is NOT re-plumbed and cannot be.** Its codec was the global
    uint16 lattice and the w-frame ladder, deleted with the thesis they served.
    T9 is a different tier in kind -- bucket-relative uint8 offsets, plain
    max-range int16 velocity, no ladder -- so a T9 measurement would be a
    DIFFERENT quantity at a different quantum, not a re-read of D-014's ratified
    5e-4. That gate stands on its stored card (`runs/m1/quant_gate_*.json`) and
    on `inexor_int16_*.npz`, and this function no longer pretends otherwise.
    """
    import jax

    jax.config.update("jax_enable_x64", True)
    import jax.numpy as jnp

    from inexor import config
    from inexor.forces import make_force_fn
    from inexor.integrate import a_grid, bullfrog_float_coeffs, bullfrog_table
    from inexor.integrate import float_step_bullfrog

    ics = M.load_state(os.path.join(M.RUNS, f"ics_{tag}.npz"))
    cfg = ics["meta"]["config"]
    box = config.BoxConfig(n_mesh=cfg["n_mesh"], box_size=cfg["box_size"])
    cosmo = config.Cosmology(**M.COSMO)
    x = jnp.asarray(ics["x"], jnp.float64)
    v = jnp.asarray(ics["v_d"], jnp.float64)

    if cfg["integrator"] != "bullfrog":
        raise ValueError(
            f"only the bullfrog arm is re-plumbed; {cfg['integrator']!r} would need its own "
            "KDK loop and no stored reference uses one"
        )
    a_steps = a_grid(cfg["a_init"], cfg["a_final"], cfg["n_steps"], cfg["spacing"])
    if not np.allclose(a_steps, ics["a_steps"], rtol=0, atol=0):
        raise ValueError("the regenerated schedule is not the stored one; the arms would differ")
    # The KERNEL dtype is a knob because it is not recoverable from the stored
    # metadata: `evolve_float` took an fdtype for the STATE and built its force
    # internally, and `make_force_fn` defaults to f32. Which one the stored
    # reference used is therefore a question to measure, not to assume.
    kd = jnp.float64 if kernel_dtype == "f64" else jnp.float32
    force_fn = make_force_fn(box, fdtype=kd, paint="int")
    for c in bullfrog_float_coeffs(bullfrog_table(a_steps, cosmo)):
        x, v = float_step_bullfrog(x, v, tuple(np.asarray(c, np.float64)), force_fn, box.box_size)

    # A SEPARATE name, deliberately. `inexor_float_{tag}.npz` is a stored v1
    # reference that exists on one laptop and nowhere else (runs/m1 is gitignored
    # and was never force-added), so regenerating it in place would destroy the
    # artifact this arm is supposed to be checked against. Writing beside it makes
    # "does the re-plumbed driver reproduce the stored reference" a real test
    # rather than a tautology.
    meta = M.make_meta("inexor", f"float64/kernel_{kernel_dtype}", cfg, REPO,
                       ic_file=f"ics_{tag}.npz")
    sfx = "" if kernel_dtype == "f64" else "32"
    p = M.save_state(
        os.path.join(M.RUNS, f"inexor_replumb{sfx}_{tag}.npz"),
        np.mod(np.asarray(x), box.box_size),
        np.asarray(v),
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
    ap_run.add_argument("--kernel-dtype", default="f64", choices=("f64", "f32"))
    ap_cmp = sub.add_parser("compare", help="compare two final-state npz files")
    ap_cmp.add_argument("--tag", required=True)
    ap_cmp.add_argument("--a", required=True, choices=ARMS)
    ap_cmp.add_argument("--b", required=True, choices=ARMS)
    ap_cmp.add_argument("--no-fig", action="store_true")
    args = ap.parse_args()

    if args.cmd == "run":
        run_inexor(args.tag, args.kernel_dtype)
    else:
        compare(args.tag, args.a, args.b, fig=not args.no_fig)


if __name__ == "__main__":
    main()
