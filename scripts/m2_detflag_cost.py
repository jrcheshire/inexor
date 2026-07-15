"""M2 S4: what does XLA deterministic-ops cost? (input to JC's production call)

Jobs 14/15 established: the f32 CIC scatter-add is nondeterministic on CUDA, so
the STE twin's VJP and hence the gradient are not reproducible run to run, while
the PRIMAL int path is bit-stable. `--xla_gpu_deterministic_ops=true` removes
the nondeterminism at every layer. JC has adopted the flag for the bit-equality
TESTS (option B). Whether to adopt it in PRODUCTION is a separate call and wants
a cost number -- M0 R3 priced "detflag-f32" at 1.37-1.78x, but that was the
paint in isolation, not the adjoint.

Measured here, separately, because the flag should NOT cost the same everywhere:
  - forward-only: rides paint_int, which is already deterministic => expect
    ~no cost. If the forward slows materially, the flag is doing something we
    have not understood and the number needs explaining before it is trusted.
  - full adjoint: adds the bwd's f32 twin VJP => this is where the flag bites.
  - bwd (derived): full - forward.
A single blended number would hide exactly the structure that makes this
decision, since production spends its time in both.

Run twice, once per XLA_FLAGS arm, then --compare (see m2_detflag_cost.sbatch).
Sizes are powers of two: BoxConfig enforces 2^16 % n_mesh == 0, so 384 is not a
valid mesh (the M1 S7 lesson). 256^3 is the deneb ceiling; 512^3 OOMs its 6 GB.

Timing: JIT compile is excluded by a warm-up call; medians over repeats, not
means, so one scheduler hiccup cannot move the answer. block_until_ready
everywhere -- JAX dispatch is async and an unblocked timer measures nothing.

Compute: deneb (free).  sbatch scripts/m2_detflag_cost.sbatch
Outputs runs/m2/detflag_cost_{default,detflag}.json. Sets no policy.
"""

import argparse
import json
import os
import statistics
import time

import jax
import jax.numpy as jnp

from inexor import PLANCK, BoxConfig, QuantConfig, TimeConfig
from inexor.adjoint import evolve_grad
from inexor.ic import gaussian_delta
from inexor.lpt import lpt_ics

REPO = os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
RUNS = os.path.join(REPO, "runs", "m2")
COSMO = PLANCK
QUANT = QuantConfig()


def _median_time(fn, repeats):
    jax.block_until_ready(fn())  # warm-up: compile is not the measurement
    ts = []
    for _ in range(repeats):
        t0 = time.perf_counter()
        jax.block_until_ready(fn())
        ts.append(time.perf_counter() - t0)
    return statistics.median(ts), min(ts), max(ts)


def run_size(n_mesh, K, repeats):
    box = BoxConfig(n_mesh=n_mesh, box_size=256.0)
    time_cfg = TimeConfig(a_init=0.1, a_final=1.0, n_steps=K, integrator="bullfrog")
    d0 = gaussian_delta(jax.random.PRNGKey(0), n_mesh, box.box_size, COSMO, fdtype=jnp.float32)
    x0, v0 = lpt_ics(d0, box.box_size, 0.1, COSMO, order=2, fdtype=jnp.float32)

    def loss(a, b):
        xf, vf = evolve_grad(box, time_cfg, QUANT, COSMO, "perstep", jnp.float32, a, b)
        return jnp.sum(xf**2) + 0.5 * jnp.sum(vf**2)

    fwd_med, fwd_lo, fwd_hi = _median_time(lambda: loss(x0, v0), repeats)
    grad = jax.grad(loss, argnums=(0, 1))
    adj_med, adj_lo, adj_hi = _median_time(lambda: grad(x0, v0), repeats)
    return {
        "n_mesh": n_mesh,
        "K": K,
        "repeats": repeats,
        "forward_s": fwd_med,
        "forward_min_s": fwd_lo,
        "forward_max_s": fwd_hi,
        "adjoint_s": adj_med,
        "adjoint_min_s": adj_lo,
        "adjoint_max_s": adj_hi,
        "bwd_derived_s": adj_med - fwd_med,
    }


def _compare():
    paths = {t: os.path.join(RUNS, f"detflag_cost_{t}.json") for t in ("default", "detflag")}
    missing = [p for p in paths.values() if not os.path.exists(p)]
    if missing:
        print(f"compare: missing {missing} -- run both arms first")
        return
    d = {t: json.load(open(p)) for t, p in paths.items()}
    print("\n===== XLA deterministic-ops COST (detflag / default) =====")
    print(f"  default arm: {d['default']['device']}   detflag arm: {d['detflag']['device']}")
    print("\n  n_mesh      forward             full adjoint          bwd (derived)")
    print("            def    det  ratio    def    det  ratio     def    det  ratio")
    for a, b in zip(d["default"]["sizes"], d["detflag"]["sizes"]):
        assert a["n_mesh"] == b["n_mesh"]

        def r(x, y):
            return y / x if x > 0 else float("nan")

        print(
            f"  {a['n_mesh']:>5}  {a['forward_s']:6.3f} {b['forward_s']:6.3f} "
            f"{r(a['forward_s'], b['forward_s']):5.2f}x  "
            f"{a['adjoint_s']:6.3f} {b['adjoint_s']:6.3f} "
            f"{r(a['adjoint_s'], b['adjoint_s']):5.2f}x   "
            f"{a['bwd_derived_s']:6.3f} {b['bwd_derived_s']:6.3f} "
            f"{r(a['bwd_derived_s'], b['bwd_derived_s']):5.2f}x"
        )
    print("\n  M0 R3 priced detflag-f32 (the paint alone) at 1.37-1.78x.")
    print("  Forward rides paint_int (already deterministic) -> expect ~1.0x there;")
    print("  a forward ratio far from 1.0 means the flag does something unmodelled.")
    print("\n  Production adoption is JC's call; this sets no policy.")


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--n", default="64,128,256", help="comma-separated n_mesh (powers of 2)")
    ap.add_argument("--steps", type=int, default=10)
    ap.add_argument("--repeats", type=int, default=5)
    ap.add_argument("--compare", action="store_true", help="read both arms' json and tabulate")
    args = ap.parse_args()

    if args.compare:
        _compare()
        return

    dev = jax.devices()[0]
    xla_flags = os.environ.get("XLA_FLAGS", "")
    tag = "detflag" if "deterministic" in xla_flags else "default"
    print(f"=== m2_detflag_cost [{tag}]: {dev.platform} / {dev} ===")
    print(f"    XLA_FLAGS={xla_flags!r}")

    sizes = []
    for n in [int(x) for x in args.n.split(",") if x.strip()]:
        rec = run_size(n, args.steps, args.repeats)
        sizes.append(rec)
        print(
            f"  n={n:<5} forward {rec['forward_s']:.4f}s   adjoint {rec['adjoint_s']:.4f}s   "
            f"bwd(derived) {rec['bwd_derived_s']:.4f}s"
        )

    os.makedirs(RUNS, exist_ok=True)
    path = os.path.join(RUNS, f"detflag_cost_{tag}.json")
    with open(path, "w") as f:
        json.dump(
            {"platform": dev.platform, "device": str(dev), "xla_flags": xla_flags, "sizes": sizes},
            f,
            indent=1,
        )
    print(f"wrote {path}")


if __name__ == "__main__":
    main()
