"""D2e on a GB200: the device tile's GPU-vs-CPU floor, and what one tile costs a card at the 4096^3 tile shape.

Every arm is a fresh subprocess, so each device high-water mark belongs to one
arm and nothing earlier in the job can set it.

  xback-cpu   JAX_PLATFORMS=cpu, cross-backend geometry (cdev by default, arena
              residents present). Per tile: the eager device tile must equal
              `engine.tile_task` exactly (check A, on this machine); then the
              short-force inputs (u, live, owned) and `one_tile`'s output, and
              the JITTED tile's forces, codes and scales, are saved to --workdir.
  xback-gpu   the default backend, same state and coarse meshes. `one_tile` on the
              CPU arm's exact inputs, and the jitted tile, compared against the
              CPU arm: layout (owned slots, written bricks, counts) must be
              exact; forces are reported in eps of their dtype x the tile's rms
              force, codes as max |dw| and count. NO bar is applied: this arm
              measures the floor a bar would be set from.
  short       one_tile alone at a tile geometry, on host-decoded inputs: warm
              call, then timed reps each ended by a readback, and the device
              peak. Like-for-like with the gb probe's per-tile device force.
  tile        the jitted device tile at a tile geometry: warm tile (compile),
              timed tiles each ended by the result readback, host prep (decode
              plan + coarse staging) timed separately, device peak, and the
              bytes of state arrays this path uploads per call (it decodes
              against the whole off/w; that is not the 4096^3 design).

The tile geometry defaults to cgh64 with tile 512 / buffer 32: P=576 and ~23.9M
rows per tile, the 4096^3 preset's tile shape and row count.

Device arms record their platform and refuse `cpu` unless `--allow-cpu`.
Output streams line by line and the card is rewritten after every arm.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time

import numpy as np

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(REPO, "src"))
sys.path.insert(0, os.path.join(REPO, "scripts"))

ALPHA_K, BCOEF = 0.87, 1.31


def _say(msg):
    print(msg, flush=True)


def _geometry(preset, tile=None, buf=None):
    from inexor.plan import PRESETS

    g = dict(PRESETS[preset])
    if tile:
        g["tile"] = int(tile)
    if buf:
        g["buf"] = int(buf)
    return g


def _platform():
    import jax

    return str(jax.devices()[0].platform)


def _mem(key):
    import jax

    stats = jax.devices()[0].memory_stats()
    return None if not stats else int(stats.get(key, 0))


def _require_device(allow_cpu):
    p = _platform()
    if p == "cpu" and not allow_cpu:
        raise RuntimeError("device arm ran on the CPU backend; refusing to report it "
                           "as a GPU reading (pass --allow-cpu for a laptop smoke)")
    return p


def _setup(g, arena, seed=7, fine=None, coarse=None):
    """State, engine config, header, coarse meshes, one_tile, members, shapes."""
    from inexor import forces
    from inexor.device import tile as dtile
    from inexor.plan import engine_config
    from v2_d2d_device_paint import _build_state

    over = {}
    if fine:
        over["fine_dtype"] = fine
    if coarse:
        over["coarse_dtype"] = coarse
    ec = engine_config(g, **over)
    t0 = time.perf_counter()
    st = _build_state(g, ec, seed=seed, arena=arena)
    build_s = time.perf_counter() - t0
    members = {t: st.tile_bricks(t, ec.n_tile, ec._b_realized, ec.n_brick, ec.n_fine)
               for t in ec.tiles}
    counts = [sum(st.brick_member_count(b) for b in members[t]) for t in ec.tiles]
    cap = forces.capacity_shape(forces.tile_capacity(counts), rungs=ec.cap_rungs)
    one_tile, geom = forces.make_tile_force_fn(
        ec.n_fine, ec.box_size, ec.n_total, ec.n_tile, ec.b_fine, r_s=ec.r_s,
        paint=ec.paint_short, frac_bits=ec.frac_bits, fdtype=ec.np_fine_dtype)
    C = dict(cap=int(cap), n_tile=ec.n_tile, n_brick=ec.n_brick, n_fine=ec.n_fine,
             n_coarse=ec.n_coarse, box=ec.box_size, coarse_cell=ec.coarse_cell,
             cell=geom["cell"], b_real=int(ec._b_realized), alpha_k=ALPHA_K, bcoef=BCOEF)
    rng = np.random.default_rng(11)
    g_coarse = [rng.normal(scale=0.3, size=(ec.n_coarse,) * 3).astype(ec.np_coarse_dtype)
                for _ in range(3)]
    return dict(ec=ec, st=st, members=members, cap=int(cap), C=C, g_coarse=g_coarse,
                one_tile=one_tile, P=int(geom["P"]), build_s=build_s,
                shapes=dtile.tile_step_shapes(st), max_rows=int(max(counts)))


def _short_inputs(st, C, t, bricks):
    """`engine.tile_task`'s short-force inputs, from the host decode (numpy)."""
    from inexor.forces import owned_mask_from_bricks, tile_origin_extent

    slots, x, _v = st.decode_bricks(bricks)
    bor = np.repeat(np.asarray(bricks, dtype=np.int64),
                    [st.brick_member_count(b) for b in bricks])
    m, cap = len(slots), C["cap"]
    idx = np.resize(np.arange(m), cap)
    live = np.zeros(cap, dtype=bool)
    live[:m] = True
    origin, _ = tile_origin_extent(t, C["n_tile"], C["b_real"], C["cell"])
    u = np.mod(x[idx] - origin, C["box"])
    own = np.zeros(cap, dtype=bool)
    own[:m] = owned_mask_from_bricks(bor, t, C["n_tile"], C["n_brick"],
                                     C["n_fine"] // C["n_brick"])
    return u, live, own & live


def _floor(a, b):
    """max |a - b| in eps of a's dtype x rms(a), over float arrays of one dtype."""
    a64, b64 = np.asarray(a, dtype=np.float64), np.asarray(b, dtype=np.float64)
    rms = float(np.sqrt(np.mean(a64**2)))
    eps = float(np.finfo(np.asarray(a).dtype).eps)
    d = np.abs(a64 - b64)
    return dict(n_diff=int((np.asarray(a) != np.asarray(b)).sum()), n=int(np.asarray(a).size),
                eps_x_rms=(float(d.max()) / (eps * rms)) if rms > 0 else 0.0,
                rel_rms=(float(np.sqrt(np.mean(d**2))) / rms) if rms > 0 else 0.0)


# ------------------------------------------------------------------ the arms


def arm_xback_cpu(args):
    import jax

    jax.config.update("jax_enable_x64", True)
    import jax.numpy as jnp

    from inexor import engine
    from inexor.device import tile as dtile

    g = _geometry(args.xback_preset)
    s = _setup(g, arena=True, fine=args.fine, coarse=args.coarse)
    st, C = s["st"], s["C"]
    if st.arena_used == 0:
        raise RuntimeError("VACUOUS: the arena state has no residents")
    os.makedirs(args.workdir, exist_ok=True)
    rec = dict(arm="xback-cpu", platform=_platform(), preset=args.xback_preset,
               fine=s["ec"].fine_dtype, coarse=s["ec"].coarse_dtype, P=s["P"],
               cap=s["cap"], arena_used=int(st.arena_used), build_s=s["build_s"],
               tiles=[])
    rc = 0
    for i, t in enumerate(s["ec"].tiles):
        b = s["members"][t]
        host = engine.tile_task(st, s["one_tile"], C, s["g_coarse"], t, b)
        eager = dtile.tile_task_device(st, s["one_tile"], C, s["g_coarse"], t, b)
        exact = all(np.array_equal(host[k], eager[k])
                    for k in ("slots_o", "w_codes", "run_bricks", "run_scales"))
        u, live, own = _short_inputs(st, C, t, b)
        gs = np.asarray(s["one_tile"](jnp.asarray(u), jnp.asarray(live), jnp.asarray(own))[0])
        t0 = time.perf_counter()
        j = dtile.tile_task_device(st, s["one_tile"], C, s["g_coarse"], t, b, jit=True,
                                   shapes=s["shapes"], with_forces=True)
        jit_s = time.perf_counter() - t0
        np.savez(os.path.join(args.workdir, f"tile{i}.npz"), u=u, live=live, own=own, gs=gs,
                 slots_o=j["slots_o"], run_bricks=j["run_bricks"], w_codes=j["w_codes"],
                 run_scales=j["run_scales"], n_owned=j["n_owned"], n_out=j["n_out"],
                 **{f"f_{k}": v for k, v in j["forces"].items()})
        rec["tiles"].append(dict(t=list(t), eager_exact_vs_host=exact, jit_s=jit_s,
                                 n_owned=j["n_owned"]))
        _say(f"[xback-cpu] tile {t}: eager device == host tile_task: {exact}; "
             f"jit {jit_s:.2f}s; {j['n_owned']} owned")
        if not exact:
            rc = 3
    return rec, rc


def arm_xback_gpu(args):
    import jax

    jax.config.update("jax_enable_x64", True)
    import jax.numpy as jnp

    from inexor.device import tile as dtile

    platform = _require_device(args.allow_cpu)
    g = _geometry(args.xback_preset)
    s = _setup(g, arena=True, fine=args.fine, coarse=args.coarse)
    st, C = s["st"], s["C"]
    rec = dict(arm="xback-gpu", platform=platform, preset=args.xback_preset,
               fine=s["ec"].fine_dtype, coarse=s["ec"].coarse_dtype, P=s["P"],
               cap=s["cap"], arena_used=int(st.arena_used), tiles=[])
    rc = 0
    worst = dict(short_alone=0.0, g_short=0.0, g_long=0.0, v_new=0.0)
    codes_diff = codes_n = codes_max = 0
    for i, t in enumerate(s["ec"].tiles):
        cpu = np.load(os.path.join(args.workdir, f"tile{i}.npz"))
        own = cpu["own"]
        gs = np.asarray(s["one_tile"](jnp.asarray(cpu["u"]), jnp.asarray(cpu["live"]),
                                      jnp.asarray(own))[0])
        short_alone = _floor(cpu["gs"][own], gs[own])
        j = dtile.tile_task_device(st, s["one_tile"], C, s["g_coarse"], t, s["members"][t],
                                   jit=True, shapes=s["shapes"], with_forces=True)
        layout = (np.array_equal(j["slots_o"], cpu["slots_o"])
                  and np.array_equal(j["run_bricks"], cpu["run_bricks"])
                  and j["n_owned"] == int(cpu["n_owned"]) and j["n_out"] == int(cpu["n_out"]))
        fl = {k: _floor(cpu[f"f_{k}"], j["forces"][k]) for k in ("g_short", "g_long", "v_new")}
        dw = np.abs(j["w_codes"].astype(np.int32) - cpu["w_codes"].astype(np.int32))
        ds = np.abs(j["run_scales"] - cpu["run_scales"]) / cpu["run_scales"]
        codes_diff += int((dw > 0).sum())
        codes_n += dw.size
        codes_max = max(codes_max, int(dw.max()))
        worst["short_alone"] = max(worst["short_alone"], short_alone["eps_x_rms"])
        for k in fl:
            worst[k] = max(worst[k], fl[k]["eps_x_rms"])
        rec["tiles"].append(dict(t=list(t), layout_exact=layout, short_alone=short_alone,
                                 **fl, codes_diff=int((dw > 0).sum()), codes_max=int(dw.max()),
                                 scale_rel_max=float(ds.max())))
        _say(f"[xback-gpu] tile {t}: layout exact {layout}; one_tile alone "
             f"{short_alone['eps_x_rms']:.1f} eps x rms ({short_alone['n_diff']} of "
             f"{short_alone['n']} differ); jitted tile g_short {fl['g_short']['eps_x_rms']:.1f}"
             f" g_long {fl['g_long']['eps_x_rms']:.1f} v_new {fl['v_new']['eps_x_rms']:.1f}"
             f" eps x rms; codes {int((dw > 0).sum())} differ (max {int(dw.max())}); "
             f"scales rel max {float(ds.max()):.2e}")
        if not layout:
            rc = 3
    rec.update(worst_eps_x_rms=worst, codes_diff=codes_diff, codes_n=codes_n,
               codes_max=codes_max)
    _say(f"[xback-gpu] WORST over tiles, eps x rms: {worst}; codes {codes_diff} of "
         f"{codes_n} differ, max |dw| {codes_max}")
    return rec, rc


def arm_short(args):
    import jax

    jax.config.update("jax_enable_x64", True)
    import jax.numpy as jnp

    platform = _require_device(args.allow_cpu)
    g = _geometry(args.tile_preset, args.tile, args.buf)
    s = _setup(g, arena=False)
    st, C = s["st"], s["C"]
    tiles = s["ec"].tiles[: int(args.reps) + 1]
    _say(f"[short] {args.tile_preset} tile {g['tile']} buf {g['buf']}: P={s['P']} "
         f"cap={s['cap']:,} (max tile rows {s['max_rows']:,}); built {s['build_s']:.1f}s "
         f"on {platform}")
    in_use0 = _mem("bytes_in_use")
    times, phases = [], []
    for rep, t in enumerate(tiles):
        u, live, own = _short_inputs(st, C, t, s["members"][t])
        t0 = time.perf_counter()
        ua = jax.block_until_ready(jnp.asarray(u))
        la = jax.block_until_ready(jnp.asarray(live))
        oa = jax.block_until_ready(jnp.asarray(own))
        t1 = time.perf_counter()
        out = s["one_tile"](ua, la, oa)
        jax.block_until_ready(out)
        t2 = time.perf_counter()
        np.asarray(out[0])
        t3 = time.perf_counter()
        ph = dict(h2d=t1 - t0, compute=t2 - t1, d2h=t3 - t2)
        phases.append(ph)
        times.append(t3 - t1)
        _say(f"[short]   {'warm' if rep == 0 else f'rep {rep}'} tile {t}: h2d "
             f"{ph['h2d']:.4f}s compute {ph['compute']:.4f}s d2h {ph['d2h']:.4f}s  "
             f"device peak {(_mem('peak_bytes_in_use') or 0) / 1e9:.2f} GB")
        del out, ua, la, oa
    peak = _mem("peak_bytes_in_use")
    rec = dict(arm="short", platform=platform, preset=args.tile_preset, tile=g["tile"],
               buf=g["buf"], P=s["P"], cap=s["cap"], max_rows=s["max_rows"],
               build_s=s["build_s"], warm_s=times[0], rep_s=times[1:], phases=phases,
               bytes_in_use_before=in_use0, peak_bytes_in_use=peak,
               readback_bytes=int(s["cap"]) * 3 * np.dtype(s["ec"].np_fine_dtype).itemsize)
    if peak is not None and in_use0 is not None:
        rec["peak_minus_baseline"] = peak - in_use0
    reps = phases[1:] or phases
    med = {k: float(np.median([p[k] for p in reps])) for k in reps[0]}
    rec["median_phases"] = med
    _say(f"[short]   median ms/tile: compute {med['compute'] * 1e3:.1f}, readback of "
         f"{rec['readback_bytes'] / 1e6:.0f} MB {med['d2h'] * 1e3:.1f}, input upload "
         f"{med['h2d'] * 1e3:.1f}")
    return rec, 0


def arm_tile(args):
    import jax

    jax.config.update("jax_enable_x64", True)

    from inexor.device import tile as dtile

    platform = _require_device(args.allow_cpu)
    g = _geometry(args.tile_preset, args.tile, args.buf)
    s = _setup(g, arena=False)
    st, C = s["st"], s["C"]
    upload = int(sum(getattr(st, k).nbytes for k in dtile.STATE_FIELDS))
    tiles = s["ec"].tiles[: int(args.reps) + 1]
    mode = "staged" if args.staged else "plain"
    _say(f"[tile:{mode}] {args.tile_preset} tile {g['tile']} buf {g['buf']}: P={s['P']} "
         f"cap={s['cap']:,}; state arrays {upload / 1e9:.2f} GB "
         f"{'placed on the device once' if args.staged else 'uploaded every call'}; "
         f"built {s['build_s']:.1f}s on {platform}")
    in_use0 = _mem("bytes_in_use")
    ds, stage_s = None, None
    if args.staged:
        t0 = time.perf_counter()
        ds = dtile.stage_state_on_device(st)
        stage_s = time.perf_counter() - t0
        _say(f"[tile:{mode}]   state placed in {stage_s:.3f}s")
    # the untimed call first: its wall is the production call's, with no syncs
    # inserted; the timed calls then attribute it
    walls, phases, owned = [], [], []
    for rep, t in enumerate(tiles):
        b = s["members"][t]
        t0 = time.perf_counter()
        res = dtile.tile_task_device(st, s["one_tile"], C, s["g_coarse"], t, b, jit=True,
                                     shapes=s["shapes"], device_state=ds)
        walls.append(time.perf_counter() - t0)
        tm = {}
        t0 = time.perf_counter()
        dtile.tile_task_device(st, s["one_tile"], C, s["g_coarse"], t, b, jit=True,
                               shapes=s["shapes"], device_state=ds, timings=tm)
        tm["total"] = time.perf_counter() - t0
        phases.append(tm)
        owned.append(int(res["n_owned"]))
        _say(f"[tile:{mode}]   {'warm' if rep == 0 else f'rep {rep}'} tile {t}: untimed "
             f"{walls[-1]:.3f}s | timed {tm['total']:.3f}s = "
             + " ".join(f"{k} {tm[k]:.3f}" for k in ("plan", "stage", "h2d_tile",
                                                     "h2d_state", "compute", "d2h",
                                                     "result"))
             + f"; device peak {(_mem('peak_bytes_in_use') or 0) / 1e9:.2f} GB")
    peak = _mem("peak_bytes_in_use")
    rec = dict(arm="tile", mode=mode, platform=platform, preset=args.tile_preset,
               tile=g["tile"], buf=g["buf"], P=s["P"], cap=s["cap"], max_rows=s["max_rows"],
               build_s=s["build_s"], state_place_s=stage_s, warm_s=walls[0],
               rep_s=walls[1:], phases=phases, n_owned=owned, upload_bytes=upload,
               bytes_in_use_before=in_use0, peak_bytes_in_use=peak,
               traces=dtile._TRACES[0])
    reps = phases[1:] or phases
    med = {k: float(np.median([p[k] for p in reps])) for k in reps[0]}
    rec["median_phases"] = med
    if peak is not None and in_use0 is not None:
        rec["peak_minus_baseline"] = peak - in_use0
        rec["peak_minus_upload"] = peak - in_use0 - upload
        _say(f"[tile:{mode}]   peak over baseline {(peak - in_use0) / 1e9:.2f} GB, of "
             f"which state arrays {upload / 1e9:.2f} GB; per padded row excluding them "
             f"{(peak - in_use0 - upload) / s['cap']:.0f} B")
    _say(f"[tile:{mode}]   median untimed {np.median(walls[1:] or walls):.3f}s/tile; "
         "median timed phases: " + " ".join(f"{k} {v:.3f}" for k, v in med.items())
         + f"; traces {dtile._TRACES[0]}")
    return rec, 0


def arm_loop(args):
    """`tile_loop_device` over whole steps: results written on the device, the
    state copied back once per step. Warm step (compile), untimed step (the
    reference wall), synced timed step (phases accumulated over its tiles)."""
    import jax

    jax.config.update("jax_enable_x64", True)
    from inexor.device import tile as dtile

    platform = _require_device(args.allow_cpu)
    g = _geometry(args.tile_preset, args.tile, args.buf)
    s = _setup(g, arena=False)
    st, C = s["st"], s["C"]
    state_bytes = int(sum(getattr(st, k).nbytes for k in dtile.STATE_FIELDS))
    back_bytes = int(st.w.nbytes + st.vel_scale.nbytes)
    _say(f"[loop] {args.tile_preset} tile {g['tile']} buf {g['buf']}: P={s['P']} "
         f"cap={s['cap']:,}, {len(s['members'])} tiles; state {state_bytes / 1e9:.2f} GB "
         f"placed once, {back_bytes / 1e9:.2f} GB copied back per step; built "
         f"{s['build_s']:.1f}s on {platform}")
    in_use0 = _mem("bytes_in_use")
    t0 = time.perf_counter()
    ds = dtile.stage_state_on_device(st)
    place_s = time.perf_counter() - t0
    steps = []
    for tag in ("warm", "untimed", "timed"):
        tm = {} if tag == "timed" else None
        t0 = time.perf_counter()
        out = dtile.tile_loop_device(st, s["one_tile"], C, s["g_coarse"], s["members"],
                                     s["shapes"], device_state=ds, timings=tm)
        wall = time.perf_counter() - t0
        n = out["tiles_run"]
        steps.append(dict(tag=tag, wall_s=wall, tiles=n, n_owned=out["n_owned"],
                          n_particles=int(st.n_particles), timings=tm))
        line = f"[loop]   {tag} step: {wall:.3f}s over {n} tiles = {wall / n:.3f}s/tile"
        if tm:
            per_tile = " ".join(f"{k} {tm[k] / n * 1e3:.1f}" for k in
                                ("plan", "stage", "h2d_tile", "compute") if k in tm)
            line += (f" | timed ms/tile: {per_tile}; copy-back once "
                     f"{tm.get('d2h_state', 0.0) * 1e3:.1f} ms")
        _say(line + f"; owned {out['n_owned']:,} of {int(st.n_particles):,}; device peak "
             f"{(_mem('peak_bytes_in_use') or 0) / 1e9:.2f} GB")
    peak = _mem("peak_bytes_in_use")
    rec = dict(arm="loop", platform=platform, preset=args.tile_preset, tile=g["tile"],
               buf=g["buf"], P=s["P"], cap=s["cap"], max_rows=s["max_rows"],
               build_s=s["build_s"], state_bytes=state_bytes, copy_back_bytes=back_bytes,
               place_s=place_s, steps=steps, bytes_in_use_before=in_use0,
               peak_bytes_in_use=peak, traces=dtile._TRACES[0])
    if peak is not None and in_use0 is not None:
        rec["peak_minus_baseline"] = peak - in_use0
        _say(f"[loop]   peak over baseline {(peak - in_use0) / 1e9:.2f} GB (state "
             f"{state_bytes / 1e9:.2f} GB of it); traces {dtile._TRACES[0]}")
    return rec, 0


ARMS = {"xback-cpu": arm_xback_cpu, "xback-gpu": arm_xback_gpu, "short": arm_short,
        "tile": arm_tile, "loop": arm_loop}


# ------------------------------------------------------------ the orchestrator


def _run_worker(argv, env_extra, tag):
    env = dict(os.environ, PYTHONUNBUFFERED="1", **env_extra)
    cmd = [sys.executable, os.path.abspath(__file__), "--worker", *argv]
    _say(f"\n--- arm {tag}: {' '.join(argv)} {env_extra or ''}")
    p = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True,
                         env=env)
    rec = None
    for line in p.stdout:
        if line.startswith("WORKER_JSON "):
            rec = json.loads(line[len("WORKER_JSON "):])
        else:
            print(line, end="", flush=True)
    rc = p.wait()
    if rec is None:
        _say(f"--- arm {tag}: rc={rc}, NO RECORD CAME BACK (not a reading)")
        return dict(arm=tag, rc=rc, record=None), max(rc, 1)
    _say(f"--- arm {tag}: rc={rc}")
    return dict(arm=tag, rc=rc, record=rec), rc


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--worker", action="store_true", help=argparse.SUPPRESS)
    ap.add_argument("--arm", choices=sorted(ARMS), help=argparse.SUPPRESS)
    ap.add_argument("--xback-preset", default="cdev")
    ap.add_argument("--fine", default=None, help="fine dtype override (default: ratified)")
    ap.add_argument("--coarse", default=None, help="coarse dtype override (default: ratified)")
    ap.add_argument("--tile-preset", default="cgh64")
    ap.add_argument("--tile", type=int, default=512)
    ap.add_argument("--buf", type=int, default=32)
    ap.add_argument("--reps", type=int, default=3)
    ap.add_argument("--workdir", default=os.path.join(REPO, "runs", "v2", "_d2e_xback"))
    ap.add_argument("--arms", default="xback,short,tile",
                    help="comma list of xback, short, tile, tile-staged, loop")
    ap.add_argument("--staged", action="store_true", help=argparse.SUPPRESS)
    ap.add_argument("--allow-cpu", action="store_true")
    ap.add_argument("--smoke", action="store_true",
                    help="smoke geometry everywhere, CPU allowed: exercises the apparatus only")
    ap.add_argument("--out-suffix", default="")
    args = ap.parse_args(argv)

    if args.worker:
        rec, rc = ARMS[args.arm](args)
        print("WORKER_JSON " + json.dumps(rec), flush=True)
        return rc

    if args.smoke:
        args.xback_preset, args.tile_preset, args.tile, args.buf, args.reps = (
            "smoke", "smoke", 16, 8, 1)
        args.allow_cpu = True
    common = ["--reps", str(args.reps), "--workdir", args.workdir,
              "--xback-preset", args.xback_preset, "--tile-preset", args.tile_preset,
              "--tile", str(args.tile), "--buf", str(args.buf)]
    common += ["--allow-cpu"] if args.allow_cpu else []
    common += ["--fine", args.fine] if args.fine else []
    common += ["--coarse", args.coarse] if args.coarse else []
    out = os.path.join(REPO, "runs", "v2", f"d2e_device_tile{args.out_suffix}.json")
    os.makedirs(os.path.dirname(out), exist_ok=True)
    card = dict(argv=sys.argv, started=time.strftime("%Y-%m-%dT%H:%M:%S"),
                job=os.environ.get("SLURM_JOB_ID"), node=os.uname().nodename,
                commit=subprocess.run(["git", "-C", REPO, "rev-parse", "--short", "HEAD"],
                                      capture_output=True, text=True).stdout.strip(),
                arms=[])

    def write():
        with open(out, "w") as f:
            json.dump(card, f, indent=1)

    worst = 0
    arms = args.arms.split(",")
    if "xback" in arms:
        res, rc = _run_worker(["--arm", "xback-cpu", *common], {"JAX_PLATFORMS": "cpu"},
                              "xback-cpu")
        card["arms"].append(res)
        write()
        worst = max(worst, rc)
        if rc == 0:
            res, rc = _run_worker(["--arm", "xback-gpu", *common], {}, "xback-gpu")
            card["arms"].append(res)
            write()
            worst = max(worst, rc)
        else:
            _say("\nthe CPU arm failed or did not record; the GPU comparison is not run")
    for name in ("short", "tile", "tile-staged", "loop"):
        if name in arms:
            argv = ["--arm", "tile", "--staged"] if name == "tile-staged" else ["--arm", name]
            res, rc = _run_worker([*argv, *common], {}, name)
            card["arms"].append(res)
            write()
            worst = max(worst, rc)
    card["finished"] = time.strftime("%Y-%m-%dT%H:%M:%S")
    write()
    _say(f"\ncard: {out}\nworst rc {worst}")
    return 0 if worst == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
