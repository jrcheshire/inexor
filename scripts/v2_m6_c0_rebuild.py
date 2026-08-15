"""C0 of the wall plan: what does rebuilding `make_tile_force_fn` every step cost?

`engine.py:680` rebuilds the tile force -- a fresh closure, so jax re-traces,
plus `split_kernels` rebuilding 3 complex128 half-grids (791 MB at P=320) --
once PER STEP. Workers in the W2 design hoist it by construction (built once
per run); this canary prices what the serial baseline pays today, so parallel
speedups get quoted against an honest denominator, and answers empirically
whether the per-step rebuild re-COMPILES (XLA executable rebuilt) or only
re-traces (cache hit on identical HLO).

Arms, same shapes throughout (cap fixed, so no ladder effects):
  rebuild : N pseudo-steps of [make_tile_force_fn -> one call, blocked]
  hoisted : build once, N calls (first call excluded as compile warmup)

Readout: per-step rebuild tax = mean(build_s + call_s) - steady call_s.

Usage (laptop, ~2 min):
  pixi run python scripts/v2_m6_c0_rebuild.py
"""

import os

os.environ.setdefault("JAX_PLATFORMS", "cpu")

import argparse  # noqa: E402
import json  # noqa: E402
import platform  # noqa: E402
import subprocess  # noqa: E402
import sys  # noqa: E402
import time  # noqa: E402

import numpy as np  # noqa: E402

import jax  # noqa: E402

jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp  # noqa: E402

from inexor.forces import make_tile_force_fn, padded_size  # noqa: E402

CDEV = dict(n_fine=512, box=128.0, n_part_side=256, n_tile=256, b_fine=32,
            cap=5_284_492, m=2_700_000)


def build():
    return make_tile_force_fn(
        CDEV["n_fine"], CDEV["box"], CDEV["n_part_side"] ** 3, CDEV["n_tile"],
        CDEV["b_fine"], paint="int",
    )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", type=int, default=5)
    ap.add_argument("--out", default=os.path.join("runs", "v2", "m6_c0_rebuild.json"))
    a = ap.parse_args()
    assert jax.devices()[0].platform == "cpu", "C0 prices the CPU serial baseline"

    P, _ = padded_size(CDEV["n_tile"], CDEV["b_fine"], n_fine=CDEV["n_fine"])
    cell = CDEV["box"] / CDEV["n_fine"]
    rng = np.random.default_rng(20260814)
    m, cap = CDEV["m"], CDEV["cap"]
    x = rng.uniform(0.0, P * cell, size=(m, 3))
    u = jnp.asarray(x[np.resize(np.arange(m), cap)])
    live = np.zeros(cap, dtype=bool)
    live[:m] = True
    own = jnp.asarray(live & (rng.uniform(size=cap) < 0.7))
    live = jnp.asarray(live)

    def one_call(fn):
        t0 = time.perf_counter()
        g, o, n = fn(u, live, own)
        jax.block_until_ready((g, o, n))
        return time.perf_counter() - t0

    # hoisted arm first: its warmup also seeds any process-wide compile cache,
    # which BIASES the rebuild arm DOWNWARD -- i.e. the tax reported here is a
    # lower bound on what a cold serial run pays per step
    fn0, _ = build()
    warm = one_call(fn0)
    steady = [one_call(fn0) for _ in range(a.steps)]

    builds, firsts = [], []
    for _ in range(a.steps):
        t0 = time.perf_counter()
        fn, _ = build()
        builds.append(time.perf_counter() - t0)
        firsts.append(one_call(fn))

    steady_s = float(np.median(steady))
    tax = float(np.mean(builds) + np.mean(firsts) - steady_s)
    # Recompile discriminator: the rebuilt closure's first-call EXCESS over a
    # steady call, read against the measured compile cost (warm - steady). Near
    # 1.0 = every rebuild pays the whole compile; near 0.0 = trace-only.
    excess_frac = float((np.mean(firsts) - steady_s) / max(warm - steady_s, 1e-9))
    recompiles = excess_frac > 0.5
    print(f"C0 rebuild tax at P={P}, cap={cap}:")
    print(f"  hoisted: warmup(compile) {warm:.2f} s, steady call {steady_s:.3f} s")
    print(f"  rebuild: kernel build {np.mean(builds):.3f} s, first call {np.mean(firsts):.3f} s")
    print(f"  per-step tax = {tax:.3f} s, first-call excess = {excess_frac:.2f} of compile "
          f"({'RE-COMPILES each rebuild' if recompiles else 'trace only; compile cache hits'})")
    print("  NB lower bound: the warm arm ran first, so any process-wide compile "
          "cache was already seeded when the rebuild arm ran.")

    commit = subprocess.run(["git", "rev-parse", "--short", "HEAD"],
                            capture_output=True, text=True).stdout.strip()
    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    with open(a.out, "w") as fh:
        json.dump(dict(P=P, cap=cap, steps=a.steps, warmup_s=warm, steady_s=steady,
                       build_s=builds, first_call_s=firsts, per_step_tax_s=tax,
                       recompiles_each_rebuild=recompiles, first_call_excess_frac=excess_frac,
                       commit=commit,
                       machine=platform.machine(), system=platform.system(),
                       jax=jax.__version__, argv=sys.argv[1:]), fh, indent=1)
    print(f"  card -> {a.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
