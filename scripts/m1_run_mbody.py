"""M1 S5: mbody arm of the parity harness (runs in MBODY's pixi env, MLX/Metal f32).

Invoke from the inexor repo root:

    pixi run --manifest-path ~/spherex/mbody/pixi.toml \
        python scripts/m1_run_mbody.py --tag n64k10log_bullfrog_lpt2_s0 --mode repro

Modes:
    repro     -- N_REPEATS identical-input evolutions of the injected ICs; the
                 pairwise spread is mbody's MEASURED repeatability floor
                 (floor-first protocol: floors land before any gate number).
                 Writes mbody_repro_<tag>.json + the run-0 final state.
    inject    -- one evolution of the injected ICs -> mbody_final_<tag>.npz
                 (the Tier-A referee arm).
    own-ics   -- mbody end-to-end (its own 2LPT + CAMB ICs at the matched
                 config/seed) -> mbody_own_<tag>.npz (S6 deficit referee).
    dump-camb -- write runs/m1/pk_camb_mbody.txt from mbody's CAMB backend
                 (the alternative CAMB-table source; provenance recorded).

Conventions (verified against mbody/integrate.py):
    mbody momentum p = G_f(a) * v_d with G_f = a^3 E D'  (H0 = 1 units), so the
    injected npz v_d converts via mbody's own _G_f. mbody is float32 MLX --
    the cast is part of its floor, not ours.
"""

import argparse
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import _m1_common as M  # noqa: E402

MBODY_REPO = os.path.expanduser("~/spherex/mbody")
N_REPEATS = 5


def _build(cfg):
    from mbody.config import BoxConfig, Cosmology

    box = BoxConfig(box_size=cfg["box_size"], n_mesh=cfg["n_mesh"], n_particles=cfg["n_mesh"])
    cosmo = Cosmology(**M.COSMO)
    return box, cosmo


def _evolve_injected(ics, cfg):
    """One mbody evolution from the injected (x, v_d); returns final (x, v_d) numpy."""
    import mlx.core as mx

    from mbody import forces as FO
    from mbody import integrate as I

    box, cosmo = _build(cfg)
    a_steps = np.asarray(ics["a_steps"], dtype=np.float64)
    g_i = I._G_f(float(a_steps[0]), cosmo)
    g_f = I._G_f(float(a_steps[-1]), cosmo)
    x0 = mx.array(ics["x"].astype(np.float32))
    p0 = mx.array((ics["v_d"] * g_i).astype(np.float32))
    force_fn = FO.make_force_fn(box)
    x1, p1 = I.evolve_state(
        x0, p0, box, cosmo, a_steps, force_fn=force_fn, integrator=cfg["integrator"]
    )
    mx.eval(x1, p1)
    return np.array(x1, dtype=np.float64), np.array(p1, dtype=np.float64) / g_f


def _evolve_own(cfg):
    """mbody end-to-end at the matched config (its own 2LPT + CAMB ICs)."""
    import mlx.core as mx

    from mbody import forces as FO
    from mbody import integrate as I
    from mbody.config import TimeStepping

    box, cosmo = _build(cfg)
    time = TimeStepping(
        z_init=1.0 / cfg["a_init"] - 1.0, z_final=1.0 / cfg["a_final"] - 1.0, n_steps=cfg["n_steps"]
    )
    x0, p0 = I.initial_state(
        box, cosmo, time, seed=cfg["seed"], backend="camb", lpt_order=cfg["lpt_order"]
    )
    # the SHARED grid (np.geomspace), not mbody's exp-of-linspace variant
    a_steps = M.a_grid(cfg["a_init"], cfg["a_final"], cfg["n_steps"], cfg["spacing"])
    g_f = I._G_f(float(a_steps[-1]), cosmo)
    force_fn = FO.make_force_fn(box)
    x1, p1 = I.evolve_state(
        x0, p0, box, cosmo, a_steps, force_fn=force_fn, integrator=cfg["integrator"]
    )
    mx.eval(x1, p1)
    return np.array(x1, dtype=np.float64), np.array(p1, dtype=np.float64) / g_f, a_steps


def dump_camb():
    from mbody.config import Cosmology
    from mbody.cosmology import _camb_pk0

    cosmo = Cosmology(**M.COSMO)
    kh, pk0 = _camb_pk0(cosmo)
    path = os.path.join(M.RUNS, "pk_camb_mbody.txt")
    os.makedirs(M.RUNS, exist_ok=True)
    np.savetxt(
        path,
        np.column_stack([kh, pk0]),
        fmt="%.18e",
        header="k [h/Mpc]   P(k) [(Mpc/h)^3]  z=0  mbody CAMB backend (sigma8-rescaled)",
    )
    print(f"wrote {path}")


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--tag", help="ics_<tag>.npz to run against")
    ap.add_argument("--mode", default="repro", choices=["repro", "inject", "own-ics", "dump-camb"])
    ap.add_argument("--repeats", type=int, default=N_REPEATS)
    args = ap.parse_args()

    if args.mode == "dump-camb":
        dump_camb()
        return

    assert args.tag, "--tag is required for run modes"
    ics = M.load_state(os.path.join(M.RUNS, f"ics_{args.tag}.npz"))
    cfg = ics["meta"]["config"]

    if args.mode == "own-ics":
        x1, v1, a_steps = _evolve_own(cfg)
        meta = M.make_meta("mbody", "own-ics", cfg, MBODY_REPO, pk_source="camb-mbody")
        path = M.save_state(
            os.path.join(M.RUNS, f"mbody_own_{args.tag}.npz"), x1, v1, meta, a_steps=a_steps
        )
        print(f"wrote {path}")
        return

    if args.mode == "inject":
        x1, v1 = _evolve_injected(ics, cfg)
        meta = M.make_meta("mbody", "inject", cfg, MBODY_REPO, ic_file=f"ics_{args.tag}.npz")
        path = M.save_state(
            os.path.join(M.RUNS, f"mbody_final_{args.tag}.npz"),
            x1,
            v1,
            meta,
            a_steps=ics["a_steps"],
        )
        print(f"wrote {path}")
        return

    # repro floor: identical-input repeats, pairwise spread vs run 0
    finals, v_run0 = [], None
    for i in range(args.repeats):
        x1, v1 = _evolve_injected(ics, cfg)
        finals.append(x1)
        if v_run0 is None:
            v_run0 = v1
        print(f"repro run {i + 1}/{args.repeats} done")
    floor = M.repro_floor(finals, cfg["n_mesh"], cfg["box_size"])
    floor["meta"] = M.make_meta("mbody", "repro", cfg, MBODY_REPO, ic_file=f"ics_{args.tag}.npz")
    jpath = os.path.join(M.RUNS, f"mbody_repro_{args.tag}.json")
    with open(jpath, "w") as f:
        json.dump(floor, f, indent=1)
    meta = M.make_meta("mbody", "inject", cfg, MBODY_REPO, ic_file=f"ics_{args.tag}.npz")
    M.save_state(
        os.path.join(M.RUNS, f"mbody_final_{args.tag}.npz"),
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
