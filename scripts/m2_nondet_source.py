"""M2 S4 diagnostic 2: WHERE is the reverse sweep's CUDA nondeterminism?

m2_driver_diag (deneb job 14) refuted the tidy story: the cross-driver residual
is bitwise IDENTICAL (0/12288, all three integrators), so the drivers agree and
test_scan_perstep_grads_identical's premise holds. What actually fails is
determinism -- the SAME gradient, same driver, same inputs, computed twice,
differs on CUDA. So the failing tests would fail comparing scan to scan.

Leading suspicion: D-006 requires the VJP twin to paint in f32, and reverse-mode
through CIC means scatter-adds, whose GPU atomic ORDER is not reproducible.
That matches R3 ("f32 nondeterministic everywhere") and matches why M1 deleted
test_m0_bridge at close -- its only GPU failure was an f32-paint bit-equality
assert. But the previous diagnostic's tidy story was also well-motivated and
still wrong, so this one localizes the effect by measurement, bottom-up, instead
of arguing from R3 by analogy.

Each layer is called R times on identical inputs and every result compared
bitwise to the first. Layers, innermost first:
    1. paint_int      -- integer scatter-add. Integer addition IS associative,
                         so this should be deterministic (R3 says so; verify).
    2. paint_f32      -- f32 scatter-add. The prime suspect.
    3. force_int      -- paint_int -> FFT solve -> gather (the PRIMAL path).
    4. force_f32      -- paint_f32 -> FFT solve -> gather (the twin's primal).
    5. force_f32 VJP  -- adds the transpose of the gather, itself a scatter-add.
    6. full bwd       -- jax.grad through evolve_grad; the observed failure.
A clean split (1,3 deterministic; 2,4,5,6 not) pins it on the f32 scatter and
tells us the primal/replay path is untouched -- which is the claim that matters.

ARM 2 (the actionable half): does XLA's deterministic-ops flag fix it, and is
this the "detflag-f32" arm M0 R3 priced at 1.37-1.78x? If it works, JC has a
real choice: accept f32-roundoff nondeterminism and gate on D-015's global
metrics, or buy bitwise reproducibility for ~1.4-1.8x. The sbatch runs this
script twice, with and without the flag. The flag name is not asserted here --
if XLA rejects it, that shows up as a startup error and is itself the answer.

Compute: deneb (free, RTX 3050); n_mesh=16.  sbatch scripts/m2_nondet_source.sbatch
Outputs runs/m2/nondet_source.json. Sets NO tolerance.
"""

import json
import os

import jax
import jax.numpy as jnp
import numpy as np

from inexor import PLANCK, BoxConfig, QuantConfig, TimeConfig
from inexor.adjoint import evolve_grad
from inexor.forces import make_force_fn
from inexor.ic import gaussian_delta
from inexor.lpt import lpt_ics
from inexor.painting import paint_f32, paint_int

REPO = os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
RUNS = os.path.join(REPO, "runs", "m2")

BOX = BoxConfig(n_mesh=16, box_size=100.0)
QUANT = QuantConfig()
COSMO = PLANCK
REPEATS = 8


def _flat(x):
    return np.concatenate([np.asarray(a).ravel() for a in jax.tree.leaves(x)])


def _determinism(name, fn, R=REPEATS):
    """Call fn() R times on identical inputs; compare every result to the first."""
    ref = _flat(jax.block_until_ready(fn()))
    worst_n, worst_rel = 0, 0.0
    for _ in range(R - 1):
        out = _flat(jax.block_until_ready(fn()))
        d = out != ref
        n = int(np.count_nonzero(d))
        if n:
            denom = np.maximum(np.abs(ref), 1e-30)
            worst_rel = max(worst_rel, float(np.max(np.abs(out - ref) / denom)))
        worst_n = max(worst_n, n)
    return {
        "layer": name,
        "deterministic": worst_n == 0,
        "max_n_differing": worst_n,
        "n_total": int(ref.size),
        "max_rel": worst_rel,
        "repeats": R,
    }


def main():
    dev = jax.devices()[0]
    xla_flags = os.environ.get("XLA_FLAGS", "")
    print(f"=== m2_nondet_source: {dev.platform} / {dev} / jax {jax.__version__} ===")
    print(f"    XLA_FLAGS={xla_flags!r}")
    if dev.platform == "cpu":
        print(
            "NOTE: CPU backend -- the effect is CUDA-only; expect all layers "
            "deterministic here (plumbing check only)."
        )

    key = jax.random.PRNGKey(0)
    d0 = gaussian_delta(key, BOX.n_mesh, BOX.box_size, COSMO, fdtype=jnp.float32)
    x0, v0 = lpt_ics(d0, BOX.box_size, 0.1, COSMO, order=2, fdtype=jnp.float32)
    x0, v0 = x0.astype(jnp.float32), v0.astype(jnp.float32)

    force_int = make_force_fn(BOX, paint="int", frac_bits=QUANT.frac_bits)
    force_f32 = make_force_fn(BOX, paint="f32", frac_bits=QUANT.frac_bits)
    cot = jnp.ones_like(x0)

    def force_f32_vjp():
        _, vjp = jax.vjp(force_f32, x0)
        return vjp(cot)

    def full_bwd():
        time = TimeConfig(a_init=0.1, a_final=1.0, n_steps=6, integrator="bullfrog")

        def loss(a, b):
            xf, vf = evolve_grad(BOX, time, QUANT, COSMO, "scan", jnp.float32, a, b)
            return jnp.sum(xf**2) + 0.5 * jnp.sum(vf**2)

        return jax.grad(loss, argnums=(0, 1))(x0, v0)

    layers = [
        (
            "1. paint_int   (int scatter-add)",
            lambda: paint_int(x0, BOX.n_mesh, BOX.box_size, frac_bits=QUANT.frac_bits),
        ),
        ("2. paint_f32   (f32 scatter-add)", lambda: paint_f32(x0, BOX.n_mesh, BOX.box_size)),
        ("3. force_int   (PRIMAL path)", lambda: force_int(x0)),
        ("4. force_f32   (twin primal)", lambda: force_f32(x0)),
        ("5. force_f32 VJP (+gather transpose)", force_f32_vjp),
        ("6. full bwd    (jax.grad, the failure)", full_bwd),
    ]

    recs = []
    print(f"\n--- determinism by layer ({REPEATS} repeats, identical inputs) ---")
    for name, fn in layers:
        r = _determinism(name, fn)
        recs.append(r)
        flag = "OK  " if r["deterministic"] else "NONDET"
        print(
            f"  {flag}  {name:<40} {r['max_n_differing']:>6}/{r['n_total']:<6} differ"
            f"   max_rel {r['max_rel']:.2e}"
        )

    print("\n--- reading ---")
    by = {r["layer"][0]: r["deterministic"] for r in recs}
    primal_ok = by.get("1", False) and by.get("3", False)
    f32_bad = not by.get("2", True) or not by.get("4", True)
    bwd_bad = not by.get("6", True)
    if not bwd_bad:
        print("  The bwd is deterministic here, so there is nothing to localize. Expected")
        print("  on CPU. On a CUDA device this would mean job 14's nondeterminism did not")
        print("  reproduce -- suspect the repeat count or a warm-up effect before")
        print("  concluding it is gone.")
    elif primal_ok and f32_bad:
        print("  int paint + PRIMAL force deterministic, f32 paint NOT => the f32")
        print("  scatter-add is the source. The bit-exact replay/reversibility claim")
        print("  is UNAFFECTED (it rides the int path); what is not reproducible is")
        print("  the STE twin's VJP, which D-006 requires to paint in f32.")
    elif primal_ok and not f32_bad:
        print("  f32 paint/force are deterministic yet the bwd is not => the source is")
        print("  DOWNSTREAM of the paint (gather transpose / FFT / scan). Not the")
        print("  suspected mechanism -- do not write up the f32-scatter story.")
    elif not primal_ok:
        print("  The INT paint or the PRIMAL force is nondeterministic. That would")
        print("  contradict R3 and threaten bit-exact replay itself. Escalate: this is")
        print("  far more serious than the test failures that started this.")
    print("\n  Tolerances are NOT set here. Numbers go to JC.")

    os.makedirs(RUNS, exist_ok=True)
    tag = "detflag" if "deterministic" in xla_flags else "default"
    path = os.path.join(RUNS, f"nondet_source_{tag}.json")
    with open(path, "w") as f:
        json.dump(
            {
                "platform": dev.platform,
                "device": str(dev),
                "jax": jax.__version__,
                "xla_flags": xla_flags,
                "layers": recs,
            },
            f,
            indent=1,
        )
    print(f"\nwrote {path}")


if __name__ == "__main__":
    main()
