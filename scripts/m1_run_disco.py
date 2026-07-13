"""M1 S5: DISCO-DJ arm of the parity harness (runs in DISCO-MOCKS' pixi env, JAX f64).

Invoke from the inexor repo root:

    pixi run --manifest-path ~/spherex/disco-mocks/pixi.toml \
        python scripts/m1_run_disco.py --tag n64k10log_bullfrog_lpt2_s0 --mode repro

Modes:
    repro    -- N_REPEATS identical-input evolutions (+ jax.clear_caches()
                between runs when --fresh-trace, so recompilation is part of
                the measured floor); pairwise spread = DISCO-DJ's floor.
    inject   -- one evolution of the injected (x, v_d) -> disco_final_<tag>.npz
    own-lpt  -- DISCO-DJ's OWN 2LPT from the injected delta0 (their LPT, our
                field) -> disco_own_<tag>.npz (isolates the LPT port at S6).

Conventions (verified against discodj 0.0.2 source, plan tender-stargazing-map):
    with_external_ics takes MOMENTUM vel and divides by cosmo.Fplus(a_ini)
    internally (Fplus = a^3 E D' == mbody/inexor G_f), so vel = v_d * Fplus.
    run_nbody: time_var accepts the explicit shared a_steps array;
    grad_kernel_order DEFAULTS TO 4 (finite difference) -- 0 (ik) is passed
    explicitly; the linear P(k) always comes from_file (its built-in EH98 has
    Tcmb=2.72548 vs our 2.7255). A q-ordering identity check runs before any
    evolution.
"""

import argparse
import json
import os
import sys

os.environ.setdefault("JAX_ENABLE_X64", "1")

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import _m1_common as M  # noqa: E402

DISCO_REPO = os.path.expanduser("~/spherex/disco-mocks")
N_REPEATS = 5


def _fresh_dj(cfg, pk_file, precision="double"):
    from discodj import DiscoDJ

    dj = DiscoDJ(
        dim=3,
        res=cfg["n_mesh"],
        boxsize=cfg["box_size"],
        cosmo=dict(M.COSMO_DISCO),
        precision=precision,
    )
    dj = dj.with_timetables()
    dj = dj.with_linear_ps(transfer_function="from_file", filename=pk_file, fix_sigma8=False)
    return dj


def _check_q_ordering(dj, cfg):
    """dj.q (C-order flat) must equal the inexor/mbody Lagrangian grid exactly."""
    n, L = cfg["n_mesh"], cfg["box_size"]
    q = np.asarray(dj.ensure_flat_shape(dj.q))
    d = L / n
    coords = np.arange(n, dtype=np.float64) * d
    qx, qy, qz = np.meshgrid(coords, coords, coords, indexing="ij")
    expect = np.stack([qx.ravel(), qy.ravel(), qz.ravel()], axis=1)
    err = np.max(np.abs(q - expect))
    if err > 1e-12 * L:
        raise RuntimeError(f"q-ordering mismatch: max |dj.q - lagrangian_grid| = {err:.3e}")
    print(f"q-ordering identity check PASSED (max diff {err:.3e})")


def _evolve_injected(dj, ics, cfg):
    a_steps = np.asarray(ics["a_steps"], dtype=np.float64)
    a_i, a_f = float(a_steps[0]), float(a_steps[-1])
    fplus_i = float(dj.cosmo.Fplus(a_i))
    fplus_f = float(dj.cosmo.Fplus(a_f))
    dj = dj.with_external_ics(pos=ics["x"], vel=ics["v_d"] * fplus_i)
    X, P, _ = dj.run_nbody(
        a_i,
        a_f,
        len(a_steps) - 1,
        time_var=a_steps,
        stepper="bullfrog",
        method="pm",
        res_pm=cfg["n_mesh"],
        worder=2,
        antialias=0,
        grad_kernel_order=0,
        laplace_kernel_order=0,
        deconvolve=False,
        convert_to_numpy=True,
    )
    X = np.asarray(X, dtype=np.float64).reshape(-1, 3)
    V = np.asarray(P, dtype=np.float64).reshape(-1, 3) / fplus_f
    L = cfg["box_size"]
    return np.mod(X, L), V


def _evolve_own_lpt(dj, ics, cfg):
    """DISCO-DJ's own 2LPT ICs from the injected delta0 field, then BullFrog."""
    a_steps = np.asarray(ics["a_steps"], dtype=np.float64)
    a_i, a_f = float(a_steps[0]), float(a_steps[-1])
    fplus_f = float(dj.cosmo.Fplus(a_f))
    dj = dj.with_external_ics(delta=ics["delta0"])
    dj = dj.with_lpt(n_order=cfg["lpt_order"])
    X, P, _ = dj.run_nbody(
        a_i,
        a_f,
        len(a_steps) - 1,
        time_var=a_steps,
        stepper="bullfrog",
        method="pm",
        res_pm=cfg["n_mesh"],
        worder=2,
        antialias=0,
        grad_kernel_order=0,
        laplace_kernel_order=0,
        deconvolve=False,
        convert_to_numpy=True,
    )
    X = np.asarray(X, dtype=np.float64).reshape(-1, 3)
    V = np.asarray(P, dtype=np.float64).reshape(-1, 3) / fplus_f
    L = cfg["box_size"]
    return np.mod(X, L), V


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--tag", required=True)
    ap.add_argument("--mode", default="repro", choices=["repro", "inject", "own-lpt"])
    ap.add_argument("--repeats", type=int, default=N_REPEATS)
    ap.add_argument(
        "--fresh-trace",
        action="store_true",
        help="clear jax caches + rebuild DiscoDJ between repro repeats",
    )
    ap.add_argument("--pk", default=None, help="P(k) table (default runs/m1/pk_eh98.txt)")
    args = ap.parse_args()

    import jax

    jax.config.update("jax_enable_x64", True)

    ics = M.load_state(os.path.join(M.RUNS, f"ics_{args.tag}.npz"))
    cfg = ics["meta"]["config"]
    pk_file = args.pk or os.path.join(M.RUNS, "pk_eh98.txt")

    dj = _fresh_dj(cfg, pk_file)
    _check_q_ordering(dj, cfg)

    if args.mode == "own-lpt":
        x1, v1 = _evolve_own_lpt(dj, ics, cfg)
        meta = M.make_meta(
            "discodj",
            "own-lpt",
            cfg,
            DISCO_REPO,
            pk_source=os.path.basename(pk_file),
            ic_file=f"ics_{args.tag}.npz",
        )
        path = M.save_state(
            os.path.join(M.RUNS, f"disco_own_{args.tag}.npz"), x1, v1, meta, a_steps=ics["a_steps"]
        )
        print(f"wrote {path}")
        return

    if args.mode == "inject":
        x1, v1 = _evolve_injected(dj, ics, cfg)
        meta = M.make_meta(
            "discodj",
            "inject",
            cfg,
            DISCO_REPO,
            pk_source=os.path.basename(pk_file),
            ic_file=f"ics_{args.tag}.npz",
        )
        path = M.save_state(
            os.path.join(M.RUNS, f"disco_final_{args.tag}.npz"),
            x1,
            v1,
            meta,
            a_steps=ics["a_steps"],
        )
        print(f"wrote {path}")
        return

    # repro floor
    finals, v_run0 = [], None
    for i in range(args.repeats):
        if args.fresh_trace and i > 0:
            jax.clear_caches()
            dj = _fresh_dj(cfg, pk_file)
        x1, v1 = _evolve_injected(dj, ics, cfg)
        finals.append(x1)
        if v_run0 is None:
            v_run0 = v1
        print(f"repro run {i + 1}/{args.repeats} done")
    floor = M.repro_floor(finals, cfg["n_mesh"], cfg["box_size"])
    floor["meta"] = M.make_meta(
        "discodj",
        "repro",
        cfg,
        DISCO_REPO,
        fresh_trace=args.fresh_trace,
        pk_source=os.path.basename(pk_file),
        ic_file=f"ics_{args.tag}.npz",
    )
    suffix = "_fresh" if args.fresh_trace else ""
    jpath = os.path.join(M.RUNS, f"disco_repro_{args.tag}{suffix}.json")
    with open(jpath, "w") as f:
        json.dump(floor, f, indent=1)
    meta = M.make_meta(
        "discodj",
        "inject",
        cfg,
        DISCO_REPO,
        pk_source=os.path.basename(pk_file),
        ic_file=f"ics_{args.tag}.npz",
    )
    M.save_state(
        os.path.join(M.RUNS, f"disco_final_{args.tag}.npz"),
        finals[0],
        v_run0,
        meta,
        a_steps=ics["a_steps"],
    )
    print(f"wrote {jpath}")
    print(
        f"  floor: worst rms {floor['worst_rms_cells']:.3e} cells, "
        f"worst |dP/P| {floor['worst_ratio_absdev']:.3e}, "
        f"worst 1-r {floor['worst_one_minus_r']:.3e}"
    )


if __name__ == "__main__":
    main()
