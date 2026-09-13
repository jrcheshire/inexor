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


def _xback_tiles(s, args):
    """The cross-backend tiles: all of them, or the first `--xback-tiles`."""
    tiles = list(s["ec"].tiles)
    return tiles[: int(args.xback_tiles)] if args.xback_tiles else tiles


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

    g = _geometry(args.xback_preset, args.xback_tile, args.xback_buf)
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
    for i, t in enumerate(_xback_tiles(s, args)):
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
    g = _geometry(args.xback_preset, args.xback_tile, args.xback_buf)
    s = _setup(g, arena=True, fine=args.fine, coarse=args.coarse)
    st, C = s["st"], s["C"]
    rec = dict(arm="xback-gpu", platform=platform, preset=args.xback_preset,
               fine=s["ec"].fine_dtype, coarse=s["ec"].coarse_dtype, P=s["P"],
               cap=s["cap"], arena_used=int(st.arena_used), tiles=[])
    rc = 0
    worst = dict(short_alone=0.0, g_short=0.0, g_long=0.0, v_new=0.0)
    codes_diff = codes_n = codes_max = 0
    for i, t in enumerate(_xback_tiles(s, args)):
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
    shard = None
    if args.coarse_device:
        from inexor.device import coarse as dcoarse

        shard = dcoarse.whole_mesh_shard(s["g_coarse"])
    _say(f"[loop] coarse on {'device' if shard else 'host'}; "
         f"{args.tile_preset} tile {g['tile']} buf {g['buf']}: P={s['P']} "
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
                                     s["shapes"], device_state=ds, timings=tm,
                                     coarse_shard=shard)
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
    rec = dict(arm="loop", coarse="device" if shard else "host", platform=platform,
               preset=args.tile_preset, tile=g["tile"],
               buf=g["buf"], P=s["P"], cap=s["cap"], max_rows=s["max_rows"],
               build_s=s["build_s"], state_bytes=state_bytes, copy_back_bytes=back_bytes,
               place_s=place_s, steps=steps, bytes_in_use_before=in_use0,
               peak_bytes_in_use=peak, traces=dtile._TRACES[0])
    if peak is not None and in_use0 is not None:
        rec["peak_minus_baseline"] = peak - in_use0
        _say(f"[loop]   peak over baseline {(peak - in_use0) / 1e9:.2f} GB (state "
             f"{state_bytes / 1e9:.2f} GB of it); traces {dtile._TRACES[0]}")
    return rec, 0


def _placement_receipt():
    """What this process can actually run on: the proof that pinning and device
    isolation applied, not the flags that asked for them."""
    import jax

    rec = dict(cuda_visible_devices=os.environ.get("CUDA_VISIBLE_DEVICES"),
               n_devices=len(jax.devices()), platform=_platform())
    if hasattr(os, "sched_getaffinity"):
        cpus = sorted(os.sched_getaffinity(0))
        rec.update(n_cpus=len(cpus), cpu_min=cpus[0], cpu_max=cpus[-1])
    try:
        with open("/proc/self/status") as f:
            for line in f:
                if line.startswith("Mems_allowed_list"):
                    rec["mems_allowed"] = line.split(":", 1)[1].strip()
    except OSError:
        pass
    return rec


def _part(items, spec):
    """Contiguous part i of k of a list ("i/k")."""
    i, k = (int(x) for x in spec.split("/"))
    return [items[j] for j in np.array_split(np.arange(len(items)), k)[i]]


def arm_split(args):
    """One process's share of a concurrent tile loop: its part of the tiles on
    its one visible device, results written on the device, no copy-back. Warm
    step, then waits at `--barrier` so every process's timed steps start
    together."""
    import jax

    jax.config.update("jax_enable_x64", True)
    from inexor.device import tile as dtile

    platform = _require_device(args.allow_cpu)
    receipt = _placement_receipt()
    if receipt["n_devices"] != 1:
        raise RuntimeError(f"split worker sees {receipt['n_devices']} devices, wants 1: "
                           "CUDA_VISIBLE_DEVICES did not isolate it")
    g = _geometry(args.tile_preset, args.tile, args.buf)
    s = _setup(g, arena=False)
    st, C = s["st"], s["C"]
    tiles = _part(list(s["ec"].tiles), args.part)
    ds = dtile.stage_state_on_device(st)
    t0 = time.perf_counter()
    dtile.tile_loop_device(st, s["one_tile"], C, s["g_coarse"], s["members"], s["shapes"],
                           tiles=tiles, device_state=ds, write_host=False)
    warm_s = time.perf_counter() - t0
    tag = f"[split {args.label} part {args.part}]"
    _say(f"{tag} {len(tiles)} tiles, warm step {warm_s:.1f}s, receipt {receipt}")
    if args.ready:
        open(args.ready, "w").close()
    while args.barrier and not os.path.exists(args.barrier):
        time.sleep(0.005)
    walls = []
    t_all = time.perf_counter()
    for _ in range(int(args.steps)):
        t0 = time.perf_counter()
        dtile.tile_loop_device(st, s["one_tile"], C, s["g_coarse"], s["members"],
                               s["shapes"], tiles=tiles, device_state=ds, write_host=False)
        walls.append(time.perf_counter() - t0)
    total = time.perf_counter() - t_all
    n = int(args.steps) * len(tiles)
    _say(f"{tag} {int(args.steps)} steps x {len(tiles)} tiles in {total:.3f}s = "
         f"{n / total:.2f} tiles/s ({total / n * 1e3:.1f} ms/tile); median step "
         f"{np.median(walls):.3f}s")
    return dict(arm="split", label=args.label, part=args.part, platform=platform,
                receipt=receipt, P=s["P"], cap=s["cap"], tiles=[list(t) for t in tiles],
                steps=int(args.steps), warm_s=warm_s, step_walls=walls, total_s=total,
                tiles_per_s=n / total), 0


def arm_stage(args):
    """Host staging of the coarse sub-blocks from real-size coarse meshes: three
    n^3 f32 meshes, every page written, staged at 4096^3-geometry tile origins
    (16 tiles per side, extent n/16 + 4) in two passes; the same staging from a
    256^3 mesh (993837's size) in the same process as the reference."""
    from inexor.forces import coarse_subblock_origin_extent, stage_coarse_subblock

    if args.ready:  # run as a group of one; nothing to synchronize with
        open(args.ready, "w").close()
    n = int(args.stage_n)
    need = 3 * n**3 * 4
    avail = None
    try:
        with open("/proc/meminfo") as f:
            for line in f:
                if line.startswith("MemAvailable"):
                    avail = int(line.split()[1]) * 1024
    except OSError:
        pass
    receipt = dict(n_cpus=len(os.sched_getaffinity(0)) if hasattr(os, "sched_getaffinity")
                   else None, mem_available=avail)
    if avail is not None and avail < 1.2 * need:
        raise RuntimeError(f"{need / 1e9:.1f} GB of meshes against {avail / 1e9:.1f} GB "
                           "available; refusing rather than swapping")
    n_tile, n_fine = n // 4, 4 * n  # c-hero's ratios: 16 tiles/side, coarse = fine/4
    t0 = time.perf_counter()
    meshes = []
    for c in range(3):
        m = np.empty((n, n, n), dtype=np.float32)
        for i in range(n):
            m[i] = np.float32(c + i * 1e-3)
        meshes.append(m)
    fill_s = time.perf_counter() - t0
    side = n_fine // n_tile
    rng = np.random.default_rng(3)
    tiles = [(0, 0, 0), (side - 1, side - 1, side - 1), (0, side - 1, side // 2)]
    tiles += [tuple(int(v) for v in rng.integers(0, side, 3)) for _ in range(int(args.stage_tiles) - 3)]
    o_ext = [coarse_subblock_origin_extent(t, n_tile, n, n_fine) for t in tiles]
    extent = int(o_ext[0][1])

    def per_tile(ms, origins):
        out = []
        for o, _e in origins:
            t0 = time.perf_counter()
            for g in ms:
                stage_coarse_subblock(g, o, extent)
            out.append(time.perf_counter() - t0)
        return out

    pass1 = per_tile(meshes, o_ext)
    pass2 = per_tile(meshes, o_ext)
    del meshes
    n_ref = 256 if n >= 512 else n
    ref = [np.full((n_ref,) * 3, np.float32(c), dtype=np.float32) for c in range(3)]
    ref_o = [(np.mod(np.asarray(o), n_ref), e) for o, e in o_ext]
    per_tile(ref, ref_o)  # warm
    ref_t = per_tile(ref, ref_o)
    rec = dict(arm="stage", n=n, mesh_bytes=need, fill_s=fill_s, extent=extent,
               tiles=[list(t) for t in tiles], pass1_s=pass1, pass2_s=pass2, n_ref=n_ref,
               ref_s=ref_t, receipt=receipt)
    for k in ("pass1_s", "pass2_s", "ref_s"):
        rec[k.replace("_s", "_median_ms")] = float(np.median(rec[k]) * 1e3)
        rec[k.replace("_s", "_p90_ms")] = float(np.percentile(rec[k], 90) * 1e3)
    _say(f"[stage] three {n}^3 f32 meshes ({need / 1e9:.1f} GB) filled in {fill_s:.1f}s; "
         f"extent {extent}, {len(tiles)} tiles x 3 meshes: pass 1 median "
         f"{rec['pass1_median_ms']:.1f} ms (p90 {rec['pass1_p90_ms']:.1f}), pass 2 "
         f"{rec['pass2_median_ms']:.1f} ms (p90 {rec['pass2_p90_ms']:.1f}); from a "
         f"{n_ref}^3 mesh {rec['ref_median_ms']:.1f} ms; receipt {receipt}")
    return rec, 0


def arm_dgather(args):
    """One card's x-shard of the three coarse meshes at 4096^3 geometry, built
    on the device from a formula of global cell coordinates; each tile's three
    sub-blocks gathered by the compiled `device.coarse.subblock_device`, synced,
    no readback, and every block checked against the formula. The host
    placement's other cost -- uploading host-staged blocks -- timed beside it."""
    import jax

    jax.config.update("jax_enable_x64", True)
    import jax.numpy as jnp

    from inexor.device.coarse import check_covers, subblock_device
    from inexor.forces import COARSE_HALO, coarse_subblock_origin_extent

    if args.ready:
        open(args.ready, "w").close()
    platform = _require_device(args.allow_cpu)
    n = int(args.stage_n)
    n_tile, n_fine = n // 4, 4 * n
    side = n_fine // n_tile
    x0, nx = -COARSE_HALO, n // 4 + 2 * COARSE_HALO  # a 4-way x split, halo planes
    shard = dict(x0=x0, nx=nx, n=n)
    in_use0 = _mem("bytes_in_use")

    def build(c):
        xg = (x0 + jnp.arange(nx, dtype=jnp.int32)) % n
        ax = jnp.arange(n, dtype=jnp.int32)
        yz = ax[:, None] * n + ax[None, :]
        return (yz[None, :, :] + (c + 1) * xg[:, None, None]).astype(jnp.float32)

    t0 = time.perf_counter()
    meshes = tuple(jax.block_until_ready(jax.jit(build, static_argnums=0)(c)) for c in range(3))
    build_s = time.perf_counter() - t0
    rng = np.random.default_rng(5)
    tiles = [(0, 0, 0), (3, side - 1, side - 1), (1, 0, side - 1)]
    tiles += [(int(rng.integers(0, 4)), int(rng.integers(0, side)), int(rng.integers(0, side)))
              for _ in range(int(args.stage_tiles) - 3)]
    extent = int(coarse_subblock_origin_extent(tiles[0], n_tile, n, n_fine)[1])
    fn = jax.jit(lambda ms, xa, o: subblock_device(ms, xa, n, o, extent))
    x0a = jnp.asarray(x0, dtype=jnp.int64)

    def expected(o, c):
        r = np.arange(extent, dtype=np.int64)
        xg, yi, zi = (o[0] + r) % n, (o[1] + r) % n, (o[2] + r) % n
        return ((yi[:, None] * n + zi[None, :])[None, :, :]
                + (c + 1) * xg[:, None, None]).astype(np.float32)

    o_first = np.asarray(coarse_subblock_origin_extent(tiles[0], n_tile, n, n_fine)[0])
    t0 = time.perf_counter()
    jax.block_until_ready(fn(meshes, x0a, jnp.asarray(o_first, dtype=jnp.int64)))
    compile_s = time.perf_counter() - t0
    wrong = np.asarray(fn(meshes, jnp.asarray(x0 + 1, dtype=jnp.int64),
                          jnp.asarray(o_first, dtype=jnp.int64))[0])
    anti_vacuity = not np.array_equal(wrong, expected(o_first, 0))
    gather_s, h2d_s, exact = [], [], True
    for t in tiles:
        o, e = coarse_subblock_origin_extent(t, n_tile, n, n_fine)
        check_covers(shard, o, e)
        oa = jax.block_until_ready(jnp.asarray(np.asarray(o), dtype=jnp.int64))
        t0 = time.perf_counter()
        blocks = jax.block_until_ready(fn(meshes, x0a, oa))
        gather_s.append(time.perf_counter() - t0)
        host_blocks = [expected(o, c) for c in range(3)]
        exact &= all(np.array_equal(np.asarray(b), h) for b, h in zip(blocks, host_blocks))
        t0 = time.perf_counter()
        up = [jax.block_until_ready(jnp.asarray(h)) for h in host_blocks]
        h2d_s.append(time.perf_counter() - t0)
        del blocks, up
    peak = _mem("peak_bytes_in_use")
    shard_bytes = 3 * nx * n * n * 4
    rec = dict(arm="dgather", platform=platform, n=n, shard_x0=x0, shard_nx=nx,
               shard_bytes=shard_bytes, extent=extent, tiles=[list(t) for t in tiles],
               build_s=build_s, compile_s=compile_s, gather_s=gather_s, h2d_s=h2d_s,
               bitwise_vs_formula=bool(exact), anti_vacuity=bool(anti_vacuity),
               bytes_in_use_before=in_use0, peak_bytes_in_use=peak)
    for k in ("gather_s", "h2d_s"):
        rec[k.replace("_s", "_median_ms")] = float(np.median(rec[k]) * 1e3)
        rec[k.replace("_s", "_p90_ms")] = float(np.percentile(rec[k], 90) * 1e3)
    if peak is not None and in_use0 is not None:
        rec["peak_minus_baseline"] = peak - in_use0
    _say(f"[dgather] shard {nx}x{n}x{n} x3 ({shard_bytes / 1e9:.1f} GB) built on {platform} "
         f"in {build_s:.1f}s; compile {compile_s:.2f}s; {len(tiles)} tiles x 3 blocks of "
         f"{extent}^3: device gather median {rec['gather_median_ms']:.2f} ms (p90 "
         f"{rec['gather_p90_ms']:.2f}); host-staged upload {rec['h2d_median_ms']:.2f} ms (p90 "
         f"{rec['h2d_p90_ms']:.2f}); every block == formula: {exact}; wrong offset "
         f"differs: {anti_vacuity}; device peak over baseline "
         f"{(rec.get('peak_minus_baseline') or 0) / 1e9:.1f} GB")
    return rec, 0 if (exact and anti_vacuity) else 3


def arm_numa(args):
    """Where a touched 2 GB allocation's pages actually land, and under which
    memory policy, read from /proc/self/numa_maps: the receipt that a
    `numactl --membind` prefix applied (Mems_allowed shows the cpuset only)."""
    if args.ready:
        open(args.ready, "w").close()
    a = np.ones(2 * 1024**3 // 8)
    rec = dict(arm="numa", label=args.label, touched_bytes=int(a.nbytes))
    try:
        best = None
        with open("/proc/self/numa_maps") as f:
            for line in f:
                parts = line.split()
                pages = {p.split("=")[0]: int(p.split("=")[1]) for p in parts[2:]
                         if p[0] == "N" and "=" in p and p[1:].split("=")[0].isdigit()}
                total = sum(pages.values())
                if best is None or total > best[0]:
                    best = (total, parts[1], pages)
        rec.update(largest_mapping_pages=best[0], largest_mapping_policy=best[1],
                   largest_mapping_nodes=best[2])
    except (OSError, TypeError) as exc:
        rec["numa_maps"] = f"unavailable: {exc}"
    _say(f"[numa {args.label}] {rec}")
    return rec, 0


ARMS = {"xback-cpu": arm_xback_cpu, "xback-gpu": arm_xback_gpu, "short": arm_short,
        "tile": arm_tile, "loop": arm_loop, "split": arm_split, "stage": arm_stage,
        "dgather": arm_dgather, "numa": arm_numa}


def _pin_prefix(socket, socket_cores):
    """(command prefix, method) binding a process to one socket, or ([], 'none')."""
    import shutil

    if shutil.which("numactl"):
        return ["numactl", f"--cpunodebind={socket}", f"--membind={socket}"], "numactl"
    if shutil.which("taskset"):
        return ["taskset", "-c", socket_cores[socket]], "taskset"
    return [], "none"


def _run_group(label, workers, workdir):
    """Run workers concurrently: each gets `--ready` / `--barrier` files, the
    barrier is released once every worker is ready, and output streams with a
    per-worker prefix. A worker that dies before ready kills the group."""
    import threading

    gdir = os.path.join(workdir, f"group_{label}")
    os.makedirs(gdir, exist_ok=True)
    barrier = os.path.join(gdir, "go")
    if os.path.exists(barrier):
        os.remove(barrier)
    procs, recs, lock = [], {}, threading.Lock()
    _say(f"\n--- group {label}: {len(workers)} workers")
    for i, (argv, env_extra, prefix) in enumerate(workers):
        ready = os.path.join(gdir, f"ready{i}")
        if os.path.exists(ready):
            os.remove(ready)
        env = dict(os.environ, PYTHONUNBUFFERED="1", **env_extra)
        cmd = [*prefix, sys.executable, os.path.abspath(__file__), "--worker", *argv,
               "--ready", ready, "--barrier", barrier]
        p = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                             text=True, env=env)
        procs.append((p, ready))

        def pump(p=p, i=i):
            for line in p.stdout:
                if line.startswith("WORKER_JSON "):
                    with lock:
                        recs[i] = json.loads(line[len("WORKER_JSON "):])
                else:
                    print(f"  [{label}:{i}] {line}", end="", flush=True)

        threading.Thread(target=pump, daemon=True).start()
    while not all(os.path.exists(r) for _p, r in procs):
        dead = [i for i, (p, r) in enumerate(procs) if p.poll() is not None and not os.path.exists(r)]
        if dead:
            for p, _r in procs:
                if p.poll() is None:
                    p.kill()
            _say(f"--- group {label}: worker(s) {dead} died before ready; group killed")
            return dict(group=label, rc=1, records={}), 1
        time.sleep(0.05)
    open(barrier, "w").close()
    rcs = [p.wait() for p, _r in procs]
    time.sleep(0.2)
    worst = max(rcs + [0 if len(recs) == len(procs) else 1])
    _say(f"--- group {label}: rcs {rcs}")
    return dict(group=label, rc=worst, rcs=rcs, records=[recs.get(i) for i in range(len(procs))]), worst


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
                    help="comma list of xback, xback576, short, tile, tile-staged, loop, "
                         "loop-shard, dgather, numa, split, stage")
    ap.add_argument("--staged", action="store_true", help=argparse.SUPPRESS)
    ap.add_argument("--part", default="0/1", help=argparse.SUPPRESS)
    ap.add_argument("--label", default="", help=argparse.SUPPRESS)
    ap.add_argument("--ready", default=None, help=argparse.SUPPRESS)
    ap.add_argument("--barrier", default=None, help=argparse.SUPPRESS)
    ap.add_argument("--steps", type=int, default=30, help="timed steps per split worker")
    ap.add_argument("--gpus", type=int, default=4, help="devices in the split arm")
    ap.add_argument("--gpu-sockets", default="0,0,1,1",
                    help="socket of each GPU (gb: GPUs 0-1 on socket 0, 2-3 on 1)")
    ap.add_argument("--socket-cores", default="0-71,72-143",
                    help="core list per socket, for taskset if numactl is absent")
    ap.add_argument("--xback-tile", type=int, default=None)
    ap.add_argument("--xback-buf", type=int, default=None)
    ap.add_argument("--xback-tiles", type=int, default=None,
                    help="compare only the first N tiles across backends")
    ap.add_argument("--coarse-device", action="store_true", help=argparse.SUPPRESS)
    ap.add_argument("--stage-n", type=int, default=2048, help="coarse mesh side, stage arm")
    ap.add_argument("--stage-tiles", type=int, default=32)
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
        args.allow_cpu, args.steps, args.stage_n, args.stage_tiles = True, 2, 64, 5
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
    if "xback" in arms or "xback576" in arms:
        xcommon = list(common)
        if "xback576" in arms and not args.smoke:
            # the 4096^3 tile shape, and only the first few tiles: a CPU tile at
            # P=576 is the expensive leg
            xcommon += ["--xback-preset", "cgh64", "--xback-tile", "512", "--xback-buf", "32",
                        "--xback-tiles", str(args.xback_tiles or 2)]
        elif args.xback_tiles:
            xcommon += ["--xback-tiles", str(args.xback_tiles)]
        res, rc = _run_worker(["--arm", "xback-cpu", *xcommon], {"JAX_PLATFORMS": "cpu"},
                              "xback-cpu")
        card["arms"].append(res)
        write()
        worst = max(worst, rc)
        if rc == 0:
            res, rc = _run_worker(["--arm", "xback-gpu", *xcommon], {}, "xback-gpu")
            card["arms"].append(res)
            write()
            worst = max(worst, rc)
        else:
            _say("\nthe CPU arm failed or did not record; the GPU comparison is not run")
    named = {"tile-staged": ["--arm", "tile", "--staged"],
             "loop-shard": ["--arm", "loop", "--coarse-device"],
             "dgather": ["--arm", "dgather", "--stage-n", str(args.stage_n),
                         "--stage-tiles", str(args.stage_tiles)]}
    for name in ("short", "tile", "tile-staged", "loop", "loop-shard", "dgather"):
        if name in arms:
            argv = named.get(name, ["--arm", name])
            res, rc = _run_worker([*argv, *common], {}, name)
            card["arms"].append(res)
            write()
            worst = max(worst, rc)
    if "split" in arms:
        sockets = [int(x) for x in args.gpu_sockets.split(",")]
        cores = args.socket_cores.split(",")
        split_common = [*common, "--steps", str(args.steps)]
        for pinned in (True, False):
            mode = "pinned" if pinned else "unpinned"
            prefix0, method = _pin_prefix(sockets[0], cores) if pinned else ([], "none")
            card.setdefault("pinning_method", {})[mode] = method
            ref = [(["--arm", "split", "--label", f"ref-{mode}", "--part", "0/1",
                     *split_common], {"CUDA_VISIBLE_DEVICES": "0"}, prefix0)]
            res, rc = _run_group(f"ref-{mode}", ref, args.workdir)
            card["arms"].append(res)
            write()
            worst = max(worst, rc)
            four = []
            for i in range(int(args.gpus)):
                prefix = _pin_prefix(sockets[i], cores)[0] if pinned else []
                four.append((["--arm", "split", "--label", f"four-{mode}", "--part",
                              f"{i}/{args.gpus}", *split_common],
                             {"CUDA_VISIBLE_DEVICES": str(i)}, prefix))
            res, rc = _run_group(f"four-{mode}", four, args.workdir)
            card["arms"].append(res)
            if rc == 0 and card["arms"][-2]["rc"] == 0:
                r1 = card["arms"][-2]["records"][0]["tiles_per_s"]
                r4 = sum(r["tiles_per_s"] for r in res["records"])
                res["split_efficiency"] = r4 / r1
                _say(f"--- {mode}: four devices {r4:.2f} tiles/s against one {r1:.2f} "
                     f"= {r4 / r1:.2f}x (method {method})")
            write()
            worst = max(worst, rc)
    if "numa" in arms:
        cores = args.socket_cores.split(",")
        prefix1, method = _pin_prefix(1, cores)
        card.setdefault("pinning_method", {})["numa"] = method
        for label, prefix in (("bound-socket1", prefix1), ("default", [])):
            res, rc = _run_group(f"numa-{label}", [(["--arm", "numa", "--label", label,
                                                     *common], {"JAX_PLATFORMS": "cpu"},
                                                    prefix)], args.workdir)
            card["arms"].append(res)
            write()
            worst = max(worst, rc)
    if "stage2048" in arms or "stage" in arms:
        socket0, method = _pin_prefix(0, args.socket_cores.split(","))
        card.setdefault("pinning_method", {})["stage"] = method
        res, rc = _run_group("stage", [(["--arm", "stage", "--stage-n", str(args.stage_n),
                                         "--stage-tiles", str(args.stage_tiles), *common],
                                        {"JAX_PLATFORMS": "cpu"}, socket0)], args.workdir)
        card["arms"].append(res)
        write()
        worst = max(worst, rc)
    card["finished"] = time.strftime("%Y-%m-%dT%H:%M:%S")
    write()
    _say(f"\ncard: {out}\nworst rc {worst}")
    return 0 if worst == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
