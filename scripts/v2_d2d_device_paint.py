"""D2d on a GB200: is the device coarse paint bitwise across backends, and what does a chunk cost a card?

Every arm is a fresh subprocess, so each device high-water mark belongs to one
arm and nothing earlier in the job can set it.

  xback-host  JAX_PLATFORMS=cpu. The HOST engine's streamed density
              (`engine.coarse_delta_streamed`) for two states at one geometry --
              plain, and with arena residents -- hashed.
              It also hashes the JITTED device paint on the CPU backend.
  xback-dev   the default backend. The DEVICE paint's density
              (`device.paint.coarse_delta_device`) for the same two states,
              eager AND jitted, must hash-equal the CPU arm's host density. This
              is the cross-backend claim: the laptop suite can only show
              device == host on ONE backend.
  chunk       one chunk of L bricks at production brick geometry (4096 rows
              and 512 buckets per brick), so its rows are a 4096^3 chunk's:
              L=4096 at 512^3 is 1/16 of a 4096^3 x-slab, L=16384 at 512^3 a
              quarter, L=65536 at 1024^3 a whole x-slab. One warm call, then
              timed reps each ended by a host readback; the device's
              `peak_bytes_in_use`; then the block compared bitwise against host
              `decode_bricks` fed through the same kernel.

Device arms record the platform they ran on and refuse `cpu` unless
`--allow-cpu` (the laptop smoke), so a leg that silently fell back to the CPU
cannot pass as a GPU reading.

Output streams line by line and the card is rewritten after every arm, so a
killed job keeps what finished.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
import time

import numpy as np

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(REPO, "src"))

# The preset family (`plan.PRESETS`): box = N/2, n_fine = 2N, n_coarse = N/2,
# tile 256 / buf 32 from cdev up. 1024^3 is not a preset and follows the family.
EXTRA_GEOMETRY = {1024: dict(n_part=1024, box=512.0, n_fine=2048, n_coarse=512,
                             tile=256, buf=32)}
N_4096 = 4096**3
NB_4096 = 256  # bricks per side at 4096^3


def _geometry(n_part):
    from inexor.plan import PRESETS

    for g in PRESETS.values():
        if g["n_part"] == n_part:
            return dict(g)
    return dict(EXTRA_GEOMETRY[n_part])


def _engine_config(g):
    from inexor.plan import engine_config

    return engine_config(g)


def _build_state(g, ec, seed, arena):
    """A perturbed lattice, laid out at the engine's own brick count.

    `arena=True`: no spare in any brick, then one serial numpy migrate, so every
    brick crossing lands in the arena. The kernel is named rather than defaulted
    so both backends' arms build the identical state.
    """
    from inexor import state
    from inexor.codec import T9Layout

    n, L = g["n_part"], g["box"]
    sp = L / n
    ax = (np.arange(n) + 0.5) * sp
    x = np.empty((n**3, 3), dtype=np.float64)
    x[:, 0] = np.repeat(ax, n * n)
    x[:, 1] = np.tile(np.repeat(ax, n), n)
    x[:, 2] = np.tile(ax, n * n)
    rng = np.random.default_rng(seed)
    for c in range(3):
        x[:, c] += rng.normal(scale=0.25 * sp, size=n**3)
        np.mod(x[:, c], L, out=x[:, c])
    if arena:
        v = np.random.default_rng(seed + 1).normal(scale=0.5, size=x.shape)
    else:
        v = np.zeros_like(x)
    nb = g["n_fine"] // ec.n_brick
    t9 = T9Layout(box_size=L, n_part=n, bucket_cells=2)
    if not arena:
        return state.SlotState.build(x, v, t9, nb, arena_frac=0.05)
    st = state.SlotState.build(x, v, t9, nb, brick_slack=0.0, arena_frac=0.30)
    del x, v
    state.drift_and_migrate(st, 2.0, kernel="numpy")
    return st


def _sha(a):
    a = np.ascontiguousarray(a)
    return f"{a.dtype.name}{list(a.shape)}:{hashlib.sha256(a.tobytes()).hexdigest()}"


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


def _say(msg):
    print(msg, flush=True)


# ------------------------------------------------------------------ the arms


def arm_xback_host(args):
    import jax

    jax.config.update("jax_enable_x64", True)
    from inexor import engine
    from inexor.device import paint as dpaint

    g = _geometry(args.n_part)
    ec = _engine_config(g)
    rec = dict(arm="xback-host", n_part=g["n_part"], platform=_platform())
    rc = 0
    for tag, arena in (("plain", False), ("arena", True)):
        t0 = time.perf_counter()
        st = _build_state(g, ec, seed=7, arena=arena)
        t1 = time.perf_counter()
        d = engine.coarse_delta_streamed(st, ec)
        t2 = time.perf_counter()
        # the jitted device paint on THIS backend, against the host engine here
        dj = dpaint.coarse_delta_device(st, ec, jit=True)
        jit_equal = _sha(dj) == _sha(d)
        rec[tag] = dict(sha=_sha(d), arena_used=int(st.arena_used),
                        abs_max=float(np.abs(d).max()),
                        build_s=t1 - t0, paint_s=t2 - t1, jit_sha=_sha(dj),
                        jit_equal_to_host_here=jit_equal)
        _say(f"[xback-host] {tag}: arena_used={st.arena_used} |delta|max="
             f"{rec[tag]['abs_max']:.3f} build {t1 - t0:.1f}s paint {t2 - t1:.1f}s; "
             f"JIT on this backend BITWISE = {jit_equal}")
        if not jit_equal:
            rc = 3
    return rec, rc


def arm_xback_dev(args):
    import jax

    jax.config.update("jax_enable_x64", True)
    from inexor.device import paint as dpaint

    host = json.load(open(args.host_hashes))
    g = _geometry(args.n_part)
    ec = _engine_config(g)
    rec = dict(arm="xback-dev", n_part=g["n_part"],
               platform=_require_device(args.allow_cpu))
    rc = 0
    for tag, arena in (("plain", False), ("arena", True)):
        st = _build_state(g, ec, seed=7, arena=arena)
        if arena and st.arena_used == 0:
            raise RuntimeError("VACUOUS: the arena state has no residents")
        rec[tag] = dict(arena_used=int(st.arena_used))
        for mode in ("eager", "jit"):
            s = {}
            t0 = time.perf_counter()
            d = dpaint.coarse_delta_device(st, ec, stats=s, jit=(mode == "jit"))
            t1 = time.perf_counter()
            sha = _sha(d)
            equal = sha == host[tag]["sha"]
            rec[tag][mode] = dict(sha=sha, equal_to_cpu_host=equal,
                                  device_chunks=s["coarse_device_chunks"],
                                  chunk_bricks=s["coarse_chunk_bricks"],
                                  jit_traces=s.get("coarse_jit_traces"),
                                  paint_s=t1 - t0)
            _say(f"[xback-dev] {tag} {mode}: BITWISE vs CPU host = {equal}  chunks="
                 f"{s['coarse_device_chunks']} ({s['coarse_chunk_bricks']} bricks) "
                 f"traces={s.get('coarse_jit_traces')} arena_used={st.arena_used} "
                 f"{t1 - t0:.1f}s")
            if not equal:
                rc = 3
    return rec, rc


def arm_chunk(args):
    import jax

    jax.config.update("jax_enable_x64", True)
    import jax.numpy as jnp

    from inexor.device import paint as dpaint
    from inexor.forces import capacity_shape
    from inexor.painting import paint_tsc_int_subblock
    from inexor.plan import PAINT_CHUNK_B_PER_ROW, PAINT_WINDOW_B_PER_ROW

    platform = _require_device(args.allow_cpu)
    g = _geometry(args.n_part)
    ec = _engine_config(g)
    L = int(args.chunk_bricks)
    t0 = time.perf_counter()
    st = _build_state(g, ec, seed=0, arena=False)
    build_s = time.perf_counter() - t0
    nb = st.bricks_per_side
    rows = int(dpaint.chunk_rows(st, L)[0])
    pad = int(capacity_shape(rows, rungs=ec.cap_rungs))
    bricks = np.arange(0, L, dtype=np.int64)
    frac = rows / (N_4096 / NB_4096)
    jit = bool(args.jit)
    shapes = dpaint.step_shapes(st, L, pad) if jit else None
    _say(f"[chunk] {'JIT' if jit else 'eager'} n_part={g['n_part']} L={L} bricks "
         f"rows={rows:,} pad={pad:,} "
         f"= {frac:.4f} of a 4096^3 x-slab; built in {build_s:.1f}s on {platform}")

    in_use0, peak0 = _mem("bytes_in_use"), _mem("peak_bytes_in_use")
    traces0 = dpaint._TRACES[0]
    times, s = [], None
    for rep in range(int(args.reps) + 1):
        guard = []
        t0 = time.perf_counter()
        sub, origin, extent = dpaint.paint_chunk(st, bricks, 0, L, ec, pad, guard,
                                                 jit=jit, shapes=shapes)
        dpaint.check_containment(guard)
        s = np.asarray(sub)
        dt = time.perf_counter() - t0
        times.append(dt)
        _say(f"[chunk]   {'warm' if rep == 0 else f'rep {rep}'}: {dt:.3f}s  "
             f"device peak {(_mem('peak_bytes_in_use') or 0) / 1e9:.2f} GB")
        del sub
    peak = _mem("peak_bytes_in_use")
    traces = dpaint._TRACES[0] - traces0
    t0 = time.perf_counter()
    if jit:
        dpaint.slab_window_fixed(st, bricks, shapes)
    else:
        dpaint.slab_window(st, bricks)
    window_s = time.perf_counter() - t0

    _, xh, _ = st.decode_bricks(list(bricks))
    xp = np.zeros((pad, 3), dtype=np.float64)
    xp[: len(xh)] = xh
    lv = np.zeros(pad, dtype=bool)
    lv[: len(xh)] = True
    ref = np.asarray(paint_tsc_int_subblock(
        jnp.asarray(xp), tuple(int(o) for o in origin), tuple(int(e) for e in extent),
        ec.n_coarse, ec.box_size, ec.frac_bits, live=lv))
    equal = bool(np.array_equal(ref, s)) and int(np.abs(ref).sum()) > 0

    rec = dict(arm="chunk", platform=platform, jit=jit, jit_traces=traces,
               jit_shapes=shapes, n_part=g["n_part"], chunk_bricks=L,
               bricks_per_side=nb, rows=rows, pad=pad, frac_of_4096_xslab=frac,
               build_s=build_s, warm_s=times[0], rep_s=times[1:], window_s=window_s,
               bitwise_vs_host_decode=equal, bytes_in_use_before=in_use0,
               peak_before=peak0, peak_bytes_in_use=peak)
    if peak is not None and in_use0 is not None:
        per_row = (peak - in_use0) / pad
        rows_4096 = frac * N_4096 / NB_4096
        pad_4096 = int(capacity_shape(int(rows_4096), rungs=ec.cap_rungs))
        rec.update(peak_minus_baseline=peak - in_use0, b_per_padded_row=per_row,
                   planner_b_per_row=PAINT_CHUNK_B_PER_ROW + PAINT_WINDOW_B_PER_ROW,
                   projected_4096_gb=per_row * pad_4096 / 1e9,
                   planner_4096_gb=(PAINT_CHUNK_B_PER_ROW + PAINT_WINDOW_B_PER_ROW)
                   * pad_4096 * 1.0 / 1e9)
        _say(f"[chunk]   peak over baseline {(peak - in_use0) / 1e9:.2f} GB = "
             f"{per_row:.1f} B/padded row (planner charges "
             f"{PAINT_CHUNK_B_PER_ROW + PAINT_WINDOW_B_PER_ROW}); at 4096^3 this chunk "
             f"is {rec['projected_4096_gb']:.1f} GB against a charge of "
             f"{rec['planner_4096_gb']:.1f}")
    reps = times[1:] or times
    _say(f"[chunk]   median {np.median(reps):.3f}s/chunk (host window prep "
         f"{window_s:.3f}s of it); traces {traces}; BITWISE vs host decode = {equal}")
    return rec, 0 if equal else 3


ARMS = {"xback-host": arm_xback_host, "xback-dev": arm_xback_dev, "chunk": arm_chunk}


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
    ap.add_argument("--n-part", type=int, default=None, help=argparse.SUPPRESS)
    ap.add_argument("--chunk-bricks", type=int, default=None, help=argparse.SUPPRESS)
    ap.add_argument("--jit", action="store_true", help=argparse.SUPPRESS)
    ap.add_argument("--host-hashes", default=None, help=argparse.SUPPRESS)
    ap.add_argument("--xback-n", type=int, default=256,
                    help="particles per side for the cross-backend arms (cdev: 256)")
    ap.add_argument("--chunks", default="512:4096,512:16384,1024:65536",
                    help="n_part:chunk_bricks[:eager|jit] list for the chunk arms, "
                         "smallest first (mode defaults to eager)")
    ap.add_argument("--reps", type=int, default=3)
    ap.add_argument("--allow-cpu", action="store_true")
    ap.add_argument("--smoke", action="store_true",
                    help="32^3 everything, CPU allowed: exercises the apparatus only")
    ap.add_argument("--skip-xback", action="store_true")
    ap.add_argument("--out-suffix", default="")
    args = ap.parse_args(argv)

    if args.worker:
        rec, rc = ARMS[args.arm](args)
        print("WORKER_JSON " + json.dumps(rec), flush=True)
        return rc

    if args.smoke:
        args.xback_n, args.chunks, args.reps = 32, "32:64:eager,32:64:jit", 1
    common = ["--reps", str(args.reps)] + (["--allow-cpu"] if args.allow_cpu or args.smoke
                                           else [])
    out = os.path.join(REPO, "runs", "v2", f"d2d_device_paint{args.out_suffix}.json")
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
    if not args.skip_xback:
        hashes = out.replace(".json", "_hosthashes.json")
        res, rc = _run_worker(["--arm", "xback-host", "--n-part", str(args.xback_n)],
                              {"JAX_PLATFORMS": "cpu"}, "xback-host")
        card["arms"].append(res)
        write()
        worst = max(worst, rc)
        if res["record"] is not None:
            with open(hashes, "w") as f:
                json.dump(res["record"], f)
            res, rc = _run_worker(["--arm", "xback-dev", "--n-part", str(args.xback_n),
                                   "--host-hashes", hashes, *common], {}, "xback-dev")
            card["arms"].append(res)
            write()
            worst = max(worst, rc)
        if worst:
            _say("\nFATAL: the cross-backend identity did not hold or did not run; "
                 "no chunk timing or memory is read as the device paint's")
            card["verdict"] = "cross-backend identity FAILED or missing"
            write()
            return 1
    for spec in args.chunks.split(","):
        parts = spec.split(":")
        n_part, L = int(parts[0]), int(parts[1])
        mode = parts[2] if len(parts) > 2 else "eager"
        if mode not in ("eager", "jit"):
            raise ValueError(f"chunk spec {spec!r}: mode must be eager or jit")
        res, rc = _run_worker(["--arm", "chunk", "--n-part", str(n_part),
                               "--chunk-bricks", str(L), *common,
                               *(["--jit"] if mode == "jit" else [])], {},
                              f"chunk {spec}")
        card["arms"].append(res)
        write()
        worst = max(worst, rc)
    card["finished"] = time.strftime("%Y-%m-%dT%H:%M:%S")
    write()
    _say(f"\ncard: {out}\nworst rc {worst}")
    return 0 if worst == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
