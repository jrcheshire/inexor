"""DISCO-DJ on our ICs, scored with our card: does a single-mesh PM share the deficit?

Three phases, each in the env it needs; they exchange one npz each.

    export  (inexor env)       IC slabs -> (x, v_d) + the shared a-grid + EH98 table
    evolve  (disco-mocks env)  DISCO-DJ PM on the injected (x, v_d) -> final (x, v_d)
    evolve-mono (inexor env)   our single-mesh PM on the same (x, v_d), same output
    card    (inexor env)       final particles -> the realization script's P(k) card

    pixi run python scripts/compare/disco_crosscheck.py export --config cgh64 \
        --ic-dir IC --k-steps 120 --out W/disco_in.npz
    pixi run --manifest-path ~/src/disco-mocks/pixi.toml -e gpu \
        python scripts/compare/disco_crosscheck.py evolve --in W/disco_in.npz \
        --n-mesh 1024 --out W/disco_out.npz
    pixi run python scripts/compare/disco_crosscheck.py card --in W/disco_out.npz \
        -- --config cgh64 --k-steps 120 --workdir W --coarse-match-order 3

`evolve` uses the M1 parity settings (`m1_run_disco._evolve_injected`): CIC
(worder 2), ik gradient and Laplacian, no deconvolution, no antialiasing,
BullFrog on the explicit a-grid, momentum = v_d * Fplus(a). Particle order is
irrelevant to the PM: DISCO-DJ stores X - q periodically wrapped and adds q back
before every paint (`nbody/acc.py`).

`card` takes everything after `--` as realization-script arguments, so the bins,
the paint, the deconvolution and the layout are the ones the engine's cards use.
"""

import argparse
import json
import os
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(HERE), "run"))


def _save(path, **arrays):
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    meta = arrays.pop("meta")
    np.savez(path, meta=np.array(json.dumps(meta)), **arrays)
    print(f"  wrote {path} ({os.path.getsize(path) / 1e9:.2f} GB)", flush=True)


def _load(path):
    with np.load(path) as f:
        out = {k: f[k] for k in f.files if k != "meta"}
        out["meta"] = json.loads(str(f["meta"]))
    return out


def _disco_cosmo(c):
    """DISCO-DJ's cosmo dict for the same universe; it is flat by construction, so
    Omega_de is derived from closure and must not be passed."""
    return dict(Omega_c=c.Omega_m - c.Omega_b, Omega_b=c.Omega_b, h=c.h, sigma8=c.sigma8,
                n_s=c.n_s, Omega_k=0.0, w0=-1.0, wa=0.0)


def cmd_export(args):
    import realization as R
    from inexor import icgen
    from inexor.cosmology import linear_power

    g = R._geom(args.config)
    cosmo = R._cosmo()
    _, a_steps = R._coeffs(cosmo, args.k_steps, args.a_init, args.growth2)
    R._require_ic_epoch(args.ic_dir, args.a_init)
    R._require_ic_growth2(args.ic_dir, args.growth2)
    st = icgen.load_slot_state(args.ic_dir)
    xs, vs = [], []
    for b in range(st.n_bricks):
        _, xb, vb = st.decode_brick(b)
        xs.append(np.asarray(xb, np.float64))
        vs.append(np.asarray(vb, np.float64))
    x, v = np.concatenate(xs), np.concatenate(vs)
    if x.shape[0] != g["n_part"] ** 3:
        raise SystemExit(f"decoded {x.shape[0]:,} particles, expected {g['n_part'] ** 3:,}")
    print(f"== export {args.config}: {x.shape[0]:,} particles from {args.ic_dir}, "
          f"{args.k_steps} steps a={a_steps[0]:.4f}->{a_steps[-1]:.4f}, "
          f"rms|v_d| {np.sqrt(np.mean(np.sum(v**2, 1))):.3f}", flush=True)

    # the same z = 0 EH98 table M1's parity arm fed DISCO-DJ (m1_export_ics.dump_pk_eh98)
    k = np.geomspace(1e-4, 1e2, 800)
    P = linear_power(k, cosmo, z=0.0, backend="eh98")
    out_dir = os.path.dirname(os.path.abspath(args.out))
    os.makedirs(out_dir, exist_ok=True)
    pk_path = os.path.join(out_dir, "pk_eh98.txt")
    np.savetxt(pk_path, np.column_stack([k, P]), fmt="%.18e",
               header="k [h/Mpc]   P(k) [(Mpc/h)^3]  z=0  inexor EH98 sigma8-normalized")
    _save(args.out, x=x, v_d=v, a_steps=np.asarray(a_steps, np.float64),
          meta=dict(config=args.config, n_part=g["n_part"], box_size=g["L"],
                    ic_dir=args.ic_dir, k_steps=args.k_steps, a_init=float(a_steps[0]),
                    growth2=args.growth2, pk_file=pk_path))
    return 0


def cmd_evolve(args):
    os.environ.setdefault("JAX_ENABLE_X64", "1")
    import jax

    jax.config.update("jax_enable_x64", True)
    from discodj import DiscoDJ

    from inexor.config import Cosmology

    d = _load(args.inp)
    meta = d["meta"]
    n_part, L = int(meta["n_part"]), float(meta["box_size"])
    a_steps = d["a_steps"]
    a_i, a_f = float(a_steps[0]), float(a_steps[-1])
    print(f"== evolve DISCO-DJ: {n_part}^3 particles, L={L}, PM mesh {args.n_mesh}^3, "
          f"{len(a_steps) - 1} BullFrog steps a={a_i:.4f}->{a_f:.4f}, "
          f"precision {args.precision}, backend {jax.devices()[0].platform}", flush=True)

    dj = DiscoDJ(dim=3, res=n_part, boxsize=L, cosmo=_disco_cosmo(Cosmology()),
                 precision=args.precision)
    dj = dj.with_timetables()
    dj = dj.with_linear_ps(transfer_function="from_file", filename=meta["pk_file"],
                           fix_sigma8=False)
    fplus_i, fplus_f = float(dj.cosmo.Fplus(a_i)), float(dj.cosmo.Fplus(a_f))
    dj = dj.with_external_ics(pos=d["x"], vel=d["v_d"] * fplus_i)
    del d
    t0 = time.perf_counter()
    X, P, _ = dj.run_nbody(
        a_i, a_f, len(a_steps) - 1, time_var=a_steps, stepper="bullfrog", method="pm",
        res_pm=args.n_mesh, worder=2, antialias=0, grad_kernel_order=0,
        laplace_kernel_order=0, deconvolve=False, convert_to_numpy=True,
    )
    wall = time.perf_counter() - t0
    X = np.mod(np.asarray(X, np.float64).reshape(-1, 3), L)
    V = np.asarray(P, np.float64).reshape(-1, 3) / fplus_f
    peak = None
    try:
        peak = jax.devices()[0].memory_stats().get("peak_bytes_in_use")
    except Exception:
        pass
    print(f"  evolve wall {wall:.1f} s (compile included), device peak "
          f"{'n/a' if peak is None else f'{peak / 1e9:.1f} GB'}", flush=True)
    meta.update(n_mesh=args.n_mesh, precision=args.precision, evolve_wall_s=wall,
                device_peak_bytes=peak,
                code="discodj", settings="M1 parity: worder=2, ik grad/laplace, "
                "no deconvolution, no antialias, bullfrog")
    _save(args.out, x=X, v_d=V, a_steps=a_steps, meta=meta)
    return 0


def cmd_evolve_mono(args):
    """Our own single-mesh PM on the same injected (x, v_d): the arm between the two.

    `force_global(which="mono", assign="cic")` -- the kernel the two-level split
    was gated against (floor F0) and `make_force_fn`'s -- on one n_mesh^3 mesh, f64,
    driven by the engine's own BullFrog coefficients through the float reference
    driver. Against DISCO-DJ it differs only in operator and stepper conventions
    (D-013's ledger); against the engine only in the two-level machinery.
    """
    os.environ.setdefault("JAX_ENABLE_X64", "1")
    import jax

    jax.config.update("jax_enable_x64", True)
    import jax.numpy as jnp

    import realization as R
    from inexor import engine, forces

    R._require_cpu()
    d = _load(args.inp)
    meta = d["meta"]
    n_part, L = int(meta["n_part"]), float(meta["box_size"])
    # exports from before the flag were made with the EdS weights and ICs
    growth2 = meta.get("growth2", "eds")
    co, a_steps = R._coeffs(R._cosmo(), int(meta["k_steps"]), float(meta["a_init"]), growth2)
    # an export from another platform carries the same grid to roundoff, not bitwise
    da = float(np.max(np.abs(np.asarray(a_steps, np.float64) - d["a_steps"])))
    if da > 1e-12:
        raise SystemExit(f"the export's a-grid is not the one these coefficients were built "
                         f"on (max |da| {da:.3e})")
    n_tot = n_part**3
    print(f"== evolve inexor mono: {n_part}^3 particles, L={L}, "
          f"PM mesh {args.n_mesh}^3 (cic, f64), "
          f"{len(co)} BullFrog steps a={a_steps[0]:.4f}->{a_steps[-1]:.4f}, "
          f"growth2 {growth2}",
          flush=True)

    t_last = [time.perf_counter()]

    def force_fn(x):
        g, _ = forces.force_global(np.asarray(x, np.float64), args.n_mesh, L, n_tot,
                                   "mono", assign="cic", paint="f64")
        now = time.perf_counter()
        print(f"    force {now - t_last[0]:6.2f}s", flush=True)
        t_last[0] = now
        return jnp.asarray(g)

    t0 = time.perf_counter()
    X, V = engine.float_run_bullfrog_sync(jnp.asarray(d["x"]), jnp.asarray(d["v_d"]), co,
                                          force_fn, L)
    wall = time.perf_counter() - t0
    print(f"  evolve wall {wall:.1f} s", flush=True)
    meta.update(n_mesh=args.n_mesh, precision="double", evolve_wall_s=wall,
                code="inexor-mono",
                settings="force_global mono, cic, f64 paint, float_run_bullfrog_sync")
    _save(args.out, x=np.mod(np.asarray(X, np.float64), L), v_d=np.asarray(V, np.float64),
          a_steps=d["a_steps"], meta=meta)
    return 0


def cmd_card(args, rest):
    import realization as R
    from inexor import summary
    from inexor.codec import T9Layout
    from inexor.state import SlotState

    R._require_cpu()
    rargs = R.build_parser().parse_args(["card"] + rest)
    g = R._geom(rargs.config, rargs.n_fine, rargs.buf, rargs.n_coarse, rargs.n_part)
    ec = R._engine_config(g, rargs, None)
    cosmo = R._cosmo()
    d = _load(args.inp)
    if int(d["meta"]["n_part"]) != g["n_part"] or float(d["meta"]["box_size"]) != g["L"]:
        raise SystemExit(f"particles are {d['meta']['n_part']}^3 in {d['meta']['box_size']}, "
                         f"the card config is {g['n_part']}^3 in {g['L']}")
    t9 = T9Layout(box_size=g["L"], n_part=g["n_part"], bucket_cells=2)
    st = SlotState.build(d["x"], d["v_d"], t9, g["n_fine"] // ec.n_brick,
                         brick_slack=rargs.slack, alloc_margin=rargs.alloc_margin,
                         arena_frac=rargs.arena_frac)
    st.check()
    t0 = time.perf_counter()
    card = summary.pk_summary_card(st, ec, cosmo, 1.0, slab=rargs.slab,
                                   min_weight=rargs.min_weight, edges=None,
                                   progress=R._heartbeat(rargs), pool=None)
    wall = time.perf_counter() - t0
    out = os.path.join(rargs.workdir, "disco_pk.json")
    os.makedirs(rargs.workdir, exist_ok=True)
    with open(out, "w") as fh:
        json.dump(dict(card="inexor-disco-crosscheck-pk-1", source=d["meta"],
                       wall_s=wall, a_out=1.0, summary=card), fh, indent=1)
    print(f"== card of {args.inp}: {card['n_bins']} bins, wall {wall:.1f} s -> {out}",
          flush=True)
    return 0


def main():
    argv = sys.argv[1:]
    rest = []
    if "--" in argv:
        i = argv.index("--")
        argv, rest = argv[:i], argv[i + 1:]
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = ap.add_subparsers(dest="phase", required=True)
    e = sub.add_parser("export")
    e.add_argument("--config", required=True)
    e.add_argument("--ic-dir", required=True)
    e.add_argument("--k-steps", type=int, required=True)
    e.add_argument("--a-init", type=float, default=0.1)
    e.add_argument("--growth2", default="lcdm", choices=("lcdm", "eds"))
    e.add_argument("--out", required=True)
    v = sub.add_parser("evolve")
    v.add_argument("--in", dest="inp", required=True)
    v.add_argument("--n-mesh", type=int, required=True)
    v.add_argument("--precision", default="double", choices=("double", "single"),
                   help="double is the M1 parity setting")
    v.add_argument("--out", required=True)
    m = sub.add_parser("evolve-mono")
    m.add_argument("--in", dest="inp", required=True)
    m.add_argument("--n-mesh", type=int, required=True, help="the single mesh")
    m.add_argument("--out", required=True)
    c = sub.add_parser("card")
    c.add_argument("--in", dest="inp", required=True)
    args = ap.parse_args(argv)
    if args.phase == "export":
        return cmd_export(args)
    if args.phase == "evolve":
        return cmd_evolve(args)
    if args.phase == "evolve-mono":
        return cmd_evolve_mono(args)
    return cmd_card(args, rest)


if __name__ == "__main__":
    sys.exit(main())
