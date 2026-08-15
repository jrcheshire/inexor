"""C1 of the wall plan: do threads scale the per-tile body AT ALL?

The wall plan's W2 (parallel tile loop) has two candidate execution models:
threads in one process, or persistent worker processes over shared memory. The
design review predicts threads are dead -- `tile_long` must stay eager
(D-v2-21's gather refusal), and eager per-op dispatch is a long sequence of
GIL-serialized calls; decode and quantize are GIL-bound numpy -- but the model
is killed by THIS measurement, not by argument.

What it runs: the realistic per-tile body -- jitted `one_tile` (int paint ->
3x FFT -> gather), `np.asarray` D2H, the eager `gather_coarse_subblock` on a
staged sub-block, and the per-brick-run quantize loop -- over 16 task
executions (8 unique synthetic tiles, cycled), serially and from a
ThreadPoolExecutor at N = 2/4/8. Inputs are pre-staged and read-only; each
execution is pure compute returning fresh arrays, exactly like the engine's
loop body.

Readouts:
  speedup(N) = wall(serial) / wall(N)   -- the decisive number
  digest mismatches vs the serial reference -- MUST be 0. Same executable,
    same inputs; a nonzero means concurrent execution itself is unsafe in this
    process and kills ALL in-process concurrency, not just the pool shape.

KILL CRITERION (pre-registered in the plan): speedup(8) < 2x kills the thread
model and W2 proceeds with worker processes. Expected outcome: killed.

The laptop verdict is indicative (macOS-arm64); the Linux confirmation rides
C2's sbatch. Synthetic inputs are fine here because the question is about the
DISPATCH PATH, not the physics: shapes, dtypes and op mix are the engine's.

Usage (laptop, ~3 min):
  pixi run python scripts/v2_m6_c1_threads.py
  pixi run python scripts/v2_m6_c1_threads.py --preset p192   # 5x cheaper
"""

import os

os.environ.setdefault("JAX_PLATFORMS", "cpu")

import argparse  # noqa: E402
import hashlib  # noqa: E402
import itertools  # noqa: E402
import json  # noqa: E402
import platform  # noqa: E402
import subprocess  # noqa: E402
import sys  # noqa: E402
import time  # noqa: E402
from concurrent.futures import ThreadPoolExecutor  # noqa: E402

import numpy as np  # noqa: E402

import jax  # noqa: E402

jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp  # noqa: E402

from inexor.codec import INT16_MAX  # noqa: E402
from inexor.forces import (  # noqa: E402
    coarse_subblock_origin_extent,
    gather_coarse_subblock,
    make_tile_force_fn,
    padded_size,
    stage_coarse_subblock,
    tile_origin_extent,
)

# Geometry presets mirror the operating configs (m3.CONFIGS + T/b), with `cap`
# at the measured cdev value for p320 -- compile cost and buffer sizes are
# shape-driven, so the cap must be the real one, not a convenient one.
PRESETS = {
    "p320": dict(n_fine=512, box=128.0, n_part_side=256, n_tile=256, b_fine=32,
                 n_coarse=128, cap=5_284_492, m=2_700_000),
    "p192": dict(n_fine=512, box=128.0, n_part_side=256, n_tile=128, b_fine=32,
                 n_coarse=128, cap=1_284_096, m=340_000),
}

QUANT_RUNS = 64  # synthetic per-brick runs inside the quantize loop


def build_tasks(p, n_unique, seed=20260814):
    """Pre-stage read-only inputs for `n_unique` synthetic tiles."""
    cell = p["box"] / p["n_fine"]
    P, b_real = padded_size(p["n_tile"], p["b_fine"], n_fine=p["n_fine"])
    n_side = p["n_fine"] // p["n_tile"]
    tiles = list(itertools.product(range(n_side), repeat=3))
    rng = np.random.default_rng(seed)
    g_coarse = [rng.standard_normal((p["n_coarse"],) * 3) for _ in range(3)]
    coarse_cell = p["box"] / p["n_coarse"]

    tasks = []
    for i in range(n_unique):
        t = tiles[i % len(tiles)]
        m, cap = p["m"], p["cap"]
        # tile-local positions, padded to `cap` by tiling rows exactly as the
        # engine does (engine.py:708-712) -- one_tile's shapes are cap-keyed
        x_loc = rng.uniform(0.0, P * cell, size=(m, 3))
        x_pad = x_loc[np.resize(np.arange(m), cap)]
        live = np.zeros(cap, dtype=bool)
        live[:m] = True
        core_lo, core_hi = b_real * cell, (b_real + p["n_tile"]) * cell
        own = np.zeros(cap, dtype=bool)
        own[:m] = np.all((x_loc >= core_lo) & (x_loc < core_hi), axis=1)
        own &= live
        n_own = int(own.sum())
        # global positions inside this tile's core, for the long-arm gather
        origin, _ = tile_origin_extent(t, p["n_tile"], b_real, cell)
        xo = np.zeros((cap, 3), dtype=np.float64)
        xo[:n_own] = np.mod(
            np.asarray(origin) + core_lo + rng.uniform(0.0, p["n_tile"] * cell, (n_own, 3)),
            p["box"],
        )
        lv = np.zeros(cap, dtype=bool)
        lv[:n_own] = True
        o_cells, extent = coarse_subblock_origin_extent(
            t, p["n_tile"], p["n_coarse"], p["n_fine"]
        )
        subs = [stage_coarse_subblock(g, o_cells, extent) for g in g_coarse]
        v = rng.standard_normal((n_own, 3))
        tasks.append(dict(x_pad=x_pad, live=live, own=own, n_own=n_own, xo=xo, lv=lv,
                          o_cells=o_cells, subs=subs, v=v, m=m))
    return tasks, dict(P=int(P), b_real=int(b_real), cell=cell, coarse_cell=coarse_cell)


def make_body(one_tile, geom, n_coarse):
    """The engine's per-tile phase mix: jit call, D2H, eager gather, quantize."""

    def body(task):
        u = jnp.asarray(task["x_pad"])  # np->device, per tile, like the engine
        g_short, _, _ = one_tile(u, jnp.asarray(task["live"]), jnp.asarray(task["own"]))
        g_short = np.asarray(g_short)[: task["m"]]
        g_long = np.asarray(
            gather_coarse_subblock(
                *task["subs"], jnp.asarray(task["xo"]), task["o_cells"],
                geom["coarse_cell"], n_coarse, assign="tsc", live=task["lv"],
            )
        )[: task["n_own"]]
        g_tot = g_short[task["own"][: task["m"]]] + g_long  # fancy-index, like g_short[owned]
        v_new = 0.9 * task["v"] + 0.1 * g_tot
        # per-brick-run quantize, GIL-bound numpy exactly like engine.py:778-786
        edges = np.linspace(0, task["n_own"], QUANT_RUNS + 1).astype(np.int64)
        codes = np.empty_like(v_new, dtype=np.int16)
        for lo, hi in zip(edges[:-1], edges[1:]):
            vb = v_new[lo:hi]
            s = float(np.max(np.abs(vb))) / INT16_MAX if hi > lo else 1.0
            s = s if s > 0.0 else 1.0
            codes[lo:hi] = np.rint(vb / s).astype(np.int16)
        h = hashlib.sha256()
        h.update(g_short.tobytes())
        h.update(g_long.tobytes())
        h.update(codes.tobytes())
        return h.hexdigest()

    return body


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--preset", default="p320", choices=sorted(PRESETS))
    ap.add_argument("--unique", type=int, default=8)
    ap.add_argument("--execs", type=int, default=16)
    ap.add_argument("--workers", type=int, nargs="+", default=[2, 4, 8])
    ap.add_argument("--repeats", type=int, default=2)
    ap.add_argument("--out", default=os.path.join("runs", "v2", "m6_c1_threads.json"))
    a = ap.parse_args()

    assert jax.devices()[0].platform == "cpu", "C1 is a CPU-dispatch question; refuse elsewhere"
    p = PRESETS[a.preset]
    print(f"C1 threads: preset {a.preset}, {a.unique} unique tiles x {a.execs} executions")

    one_tile, geom_f = make_tile_force_fn(
        p["n_fine"], p["box"], p["n_part_side"] ** 3, p["n_tile"], p["b_fine"],
        paint="int",
    )
    tasks, geom = build_tasks(p, a.unique)
    geom["coarse_cell"] = p["box"] / p["n_coarse"]
    body = make_body(one_tile, geom, p["n_coarse"])
    stream = [tasks[i % a.unique] for i in range(a.execs)]

    t0 = time.perf_counter()
    ref0 = body(stream[0])  # warmup: compile one_tile + trace the gather
    print(f"  warmup (compile) {time.perf_counter() - t0:.2f} s")

    # serial reference + digests
    walls = {}
    mismatches = 0
    t0 = time.perf_counter()
    ref = [body(t) for t in stream]
    walls["1"] = [time.perf_counter() - t0]
    for _ in range(a.repeats - 1):
        t0 = time.perf_counter()
        again = [body(t) for t in stream]
        walls["1"].append(time.perf_counter() - t0)
        mismatches += sum(x != y for x, y in zip(again, ref))
    assert ref[0] == ref0

    for n in a.workers:
        walls[str(n)] = []
        for _ in range(a.repeats):
            with ThreadPoolExecutor(max_workers=n) as ex:
                t0 = time.perf_counter()
                got = list(ex.map(body, stream))
                walls[str(n)].append(time.perf_counter() - t0)
            mismatches += sum(x != y for x, y in zip(got, ref))

    base = min(walls["1"])
    speed = {n: base / min(w) for n, w in walls.items()}
    print(f"  serial {base:.2f} s for {a.execs} executions")
    for n in sorted(speed, key=int):
        print(f"  N={n}: wall {min(walls[n]):.2f} s  speedup {speed[n]:.2f}x")
    print(f"  digest mismatches vs serial: {mismatches} (MUST be 0)")
    top = speed[str(max(int(n) for n in speed))]
    verdict = "threads KILLED (speedup(max N) < 2x)" if top < 2.0 else "threads SURVIVE so far"
    print(f"  VERDICT: {verdict}")

    commit = subprocess.run(["git", "rev-parse", "--short", "HEAD"],
                            capture_output=True, text=True).stdout.strip()
    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    with open(a.out, "w") as fh:
        json.dump(dict(preset=a.preset, unique=a.unique, execs=a.execs,
                       walls_s=walls, speedup=speed, digest_mismatches=mismatches,
                       verdict=verdict, commit=commit, machine=platform.machine(),
                       system=platform.system(), jax=jax.__version__,
                       argv=sys.argv[1:]), fh, indent=1)
    print(f"  card -> {a.out}")
    return 0 if mismatches == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
