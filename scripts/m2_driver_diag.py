"""M2 S4 diagnostic: why test_adjoint's exactness assertions fail on CUDA.

Vista job 831091 (the first time test_adjoint.py ever ran on a GPU -- it is M2
S1 code, and deneb's M1 job 12 predates it) failed 4 of 101:
    test_scan_perstep_grads_identical[bullfrog|fastpm|exact]  (jnp.array_equal)
    test_adjoint_grad_fnl_equals_ic_component                 (rel=1e-6)
All four assert exact/near-exact FLOAT equality across DIFFERENTLY-COMPILED
programs. All pass on macOS-arm64 CPU.

The suspicion is that they encode a false premise. test_scan_perstep_grads_identical
says in its own docstring "Both drivers produce a bit-identical integer
trajectory (M1), so the residual -- and therefore the gradient -- is identical",
but M1's own test_integrate.test_scan_and_perstep_drivers_agree_bitwise carries
the opposite note: "differently-compiled programs may tie-flip ... Bitwise
agreement here is a CPU regression canary, not a design gate." Architecture
Sec. 5/9 agrees: run_perstep exists partly to pin fwd/rev to ONE compiled
executable precisely because differently-compiled programs need not agree bit
for bit. So S1 promoted a canary into a gate.

That is the CONVENIENT conclusion, so this script tries to falsify it instead of
assuming it. The chain under test is:
    residual (final int state) identical  =>  bwd is the same program on the
    same inputs  =>  gradient bit-identical.
If the gradients differ, exactly one link must be broken. This measures WHICH:

  A. cross-driver residual: is the final INT state bitwise equal (scan vs
     perstep)? If NO -> the premise is false at this config and the ~1e-6
     gradient spread is the downstream consequence of a few rint tie-flips.
     If YES -> the premise holds and the bug is real, living in the bwd.
  B. bwd determinism: same driver, same inputs, run twice -> bit-identical?
     A NO here would mean CUDA nondeterminism inside the reverse sweep, which
     would be a genuine defect (and would contradict the "TACC deterministic"
     finding in the umbrella jax-macos-cpu-nondeterminism memory).
  C. the size of the disagreement, in the global metrics D-015 already uses, so
     any tolerance proposal rests on measured numbers rather than on whatever
     value happens to make the suite green.

Config is copied EXACTLY from tests/test_adjoint.py (n_mesh=16, box 100, K=6,
gaussian_delta seed 0, 2LPT, f32) -- a diagnostic that reproduces at a different
config proves nothing about the failure.

Compute: any CUDA device; n_mesh=16 is tiny. deneb (free, RTX 3050) is the point
-- do not spend Vista SUs on this.
    sbatch scripts/m2_driver_diag.sbatch          # deneb
Outputs runs/m2/driver_diag.json + a printed table. Sets NO tolerance: the
numbers go to JC, per the tolerances convention.
"""

import json
import os

import jax
import jax.numpy as jnp
import numpy as np

from inexor import PLANCK, BoxConfig, QuantConfig, TimeConfig
from inexor.adjoint import _forward, adjoint_grad_fnl, adjoint_grad_ic, evolve_grad
from inexor.ic import gaussian_delta
from inexor.losses import band_power_loss, fundamental_k_edges
from inexor.lpt import lpt_ics

REPO = os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
RUNS = os.path.join(REPO, "runs", "m2")

# EXACTLY tests/test_adjoint.py's config.
BOX = BoxConfig(n_mesh=16, box_size=100.0)
QUANT = QuantConfig()
COSMO = PLANCK
INTEGRATORS = ("bullfrog", "fastpm", "exact")


def _ics(seed=0, fdtype=jnp.float32):
    key = jax.random.PRNGKey(seed)
    d0 = gaussian_delta(key, BOX.n_mesh, BOX.box_size, COSMO, fdtype=fdtype)
    x0, v0 = lpt_ics(d0, BOX.box_size, 0.1, COSMO, order=2, fdtype=fdtype)
    return x0.astype(fdtype), v0.astype(fdtype)


def _loss(x_f, v_f):
    return jnp.sum(x_f**2) + 0.5 * jnp.sum(v_f**2)


def _time(integrator, K=6):
    return TimeConfig(a_init=0.1, a_final=1.0, n_steps=K, integrator=integrator)


def _int_cmp(a, b):
    """Bitwise comparison of two integer state arrays."""
    a, b = np.asarray(a), np.asarray(b)
    d = a.astype(np.int64) - b.astype(np.int64)
    nz = int(np.count_nonzero(d))
    return {
        "bitwise_equal": nz == 0,
        "n_differing": nz,
        "n_total": int(d.size),
        "frac_differing": nz / d.size,
        "max_abs_diff": int(np.max(np.abs(d))) if d.size else 0,
    }


def _grad_metrics(a, b):
    a, b = np.asarray(a).ravel(), np.asarray(b).ravel()
    same = bool(np.array_equal(a, b))
    denom = np.maximum(np.abs(b), 1e-30)
    rel = np.abs(a - b) / denom
    return {
        "bitwise_equal": same,
        "n_differing": int(np.count_nonzero(a != b)),
        "n_total": int(a.size),
        "max_rel": float(np.max(rel)),
        "median_rel": float(np.median(rel)),
        "norm_ratio": float(np.linalg.norm(a) / np.linalg.norm(b)),
        "corr": float(np.corrcoef(a, b)[0, 1]),
    }


def _grad_with(driver, time, x0, v0):
    def loss(x0_, v0_):
        return _loss(*evolve_grad(BOX, time, QUANT, COSMO, driver, jnp.float32, x0_, v0_))

    return jax.grad(loss, argnums=(0, 1))(x0, v0)


def run_integrator(integrator):
    time = _time(integrator)
    x0, v0 = _ics()
    rec = {"integrator": integrator}

    # ---- A: is the cross-driver residual (final int state) bitwise equal? ----
    (_, res_s) = _forward(BOX, time, QUANT, COSMO, x0, v0, "scan", jnp.float32)
    (_, res_p) = _forward(BOX, time, QUANT, COSMO, x0, v0, "perstep", jnp.float32)
    xs, ws = res_s[0], res_s[1]
    xp, wp = res_p[0], res_p[1]
    rec["residual_x"] = _int_cmp(xs, xp)
    rec["residual_w"] = _int_cmp(ws, wp)
    rec["residual_identical"] = (
        rec["residual_x"]["bitwise_equal"] and rec["residual_w"]["bitwise_equal"]
    )
    # The scales are host-side floats and driver-independent by construction;
    # assert that rather than trust it, so "residual differs" can only mean the
    # integer state.
    rec["scales_identical"] = bool(res_s[3] == res_p[3] and res_s[4] == res_p[4])

    # ---- B: is the bwd deterministic? same driver, same inputs, twice ----
    g1x, g1v = _grad_with("scan", time, x0, v0)
    g2x, g2v = _grad_with("scan", time, x0, v0)
    rec["bwd_repeat_deterministic"] = bool(
        np.array_equal(np.asarray(g1x), np.asarray(g2x))
        and np.array_equal(np.asarray(g1v), np.asarray(g2v))
    )

    # ---- C: how big is the cross-driver gradient disagreement? ----
    gpx, gpv = _grad_with("perstep", time, x0, v0)
    rec["grad_x_scan_vs_perstep"] = _grad_metrics(g1x, gpx)
    rec["grad_v_scan_vs_perstep"] = _grad_metrics(g1v, gpv)
    return rec


def run_fnl_vs_ic():
    """Failure 4: adjoint_grad_fnl vs the f_NL component of adjoint_grad_ic.
    Two different jax.grad compositions => two different compiled programs."""
    time = _time("bullfrog")
    k_edges = fundamental_k_edges(BOX.n_mesh, BOX.box_size, n_bins=4)
    zeros_pk = np.zeros(len(k_edges) - 1, dtype=np.float32)

    def loss_field(x_f, v_f):
        return band_power_loss(x_f, BOX, k_edges, jnp.asarray(zeros_pk, dtype=x_f.dtype))

    g_fnl = float(adjoint_grad_fnl(loss_field, BOX, time, QUANT, COSMO, f_NL=5.0))
    g_ic = adjoint_grad_ic(loss_field, BOX, time, QUANT, COSMO, theta=(5.0, 1.0))
    g_ic0 = float(g_ic[0])
    rel = abs(g_fnl - g_ic0) / max(abs(g_ic0), 1e-30)
    return {
        "g_fnl": g_fnl,
        "g_ic0": g_ic0,
        "abs_diff": abs(g_fnl - g_ic0),
        "rel_diff": rel,
        "test_gate_rel": 1e-6,
        "passes_current_gate": rel <= 1e-6,
    }


def main():
    dev = jax.devices()[0]
    print(f"=== m2_driver_diag: {dev.platform} / {dev} / jax {jax.__version__} ===")
    if dev.platform == "cpu":
        print(
            "NOTE: CPU backend -- the failures are CUDA-only, so this run is a "
            "plumbing check and is EXPECTED to show everything identical."
        )

    recs = [run_integrator(i) for i in INTEGRATORS]
    fnl = run_fnl_vs_ic()

    print("\n--- A: cross-driver residual (final INT state), scan vs perstep ---")
    print("    (the premise of test_scan_perstep_grads_identical)")
    for r in recs:
        rx, rw = r["residual_x"], r["residual_w"]
        print(
            f"  {r['integrator']:<9} identical={str(r['residual_identical']):<5} "
            f"x: {rx['n_differing']:>5}/{rx['n_total']} differ (max |d| {rx['max_abs_diff']})  "
            f"w: {rw['n_differing']:>5}/{rw['n_total']} differ (max |d| {rw['max_abs_diff']})  "
            f"scales_identical={r['scales_identical']}"
        )

    print("\n--- B: bwd determinism (same driver, same inputs, twice) ---")
    for r in recs:
        print(f"  {r['integrator']:<9} bit-identical on repeat: {r['bwd_repeat_deterministic']}")

    print("\n--- C: cross-driver gradient disagreement (global metrics, D-015 style) ---")
    for r in recs:
        gx = r["grad_x_scan_vs_perstep"]
        gv = r["grad_v_scan_vs_perstep"]
        print(
            f"  {r['integrator']:<9} dx: {gx['n_differing']:>5}/{gx['n_total']} differ  "
            f"max_rel {gx['max_rel']:.2e}  med_rel {gx['median_rel']:.2e}  "
            f"corr {gx['corr']:.9f}  ratio {gx['norm_ratio']:.9f}"
        )
        print(
            f"  {'':<9} dv: {gv['n_differing']:>5}/{gv['n_total']} differ  "
            f"max_rel {gv['max_rel']:.2e}  med_rel {gv['median_rel']:.2e}  "
            f"corr {gv['corr']:.9f}  ratio {gv['norm_ratio']:.9f}"
        )

    print("\n--- D: failure 4, adjoint_grad_fnl vs adjoint_grad_ic[0] ---")
    print(
        f"  g_fnl {fnl['g_fnl']:.9g}  g_ic[0] {fnl['g_ic0']:.9g}  "
        f"rel {fnl['rel_diff']:.2e}  (test asserts rel <= {fnl['test_gate_rel']:.0e}: "
        f"{'PASS' if fnl['passes_current_gate'] else 'FAIL'})"
    )

    print("\n--- reading ---")
    any_resid_differs = any(not r["residual_identical"] for r in recs)
    all_bwd_det = all(r["bwd_repeat_deterministic"] for r in recs)
    grads_differ = any(
        not (
            r["grad_x_scan_vs_perstep"]["bitwise_equal"]
            and r["grad_v_scan_vs_perstep"]["bitwise_equal"]
        )
        for r in recs
    )
    if not grads_differ and fnl["passes_current_gate"]:
        print("  Everything bitwise identical and the f_NL/IC gate passes => the Vista")
        print("  failures do NOT reproduce here. Expected on CPU (they are CUDA-only);")
        print("  on a CUDA device this would instead mean the failure is not config-")
        print("  determined and this diagnostic has not captured it.")
    elif any_resid_differs and all_bwd_det:
        print("  Residual DIFFERS across drivers + bwd is deterministic => the test's")
        print("  premise is false at this config; the gradient spread is downstream of")
        print("  rint tie-flips between two differently-compiled forward programs.")
        print("  This is M1's documented canary, not an adjoint defect.")
    elif not any_resid_differs and not all_bwd_det:
        print("  Residual IDENTICAL but bwd is nondeterministic => a REAL defect in the")
        print("  reverse sweep. Do not touch the tests; debug the bwd.")
    elif not any_resid_differs and all_bwd_det:
        print("  Residual identical AND bwd deterministic, yet grads differ => neither")
        print("  hypothesis holds. Something is wrong that this script does not model.")
    else:
        print("  Residual differs AND bwd is nondeterministic => two effects at once;")
        print("  fix the nondeterminism before reasoning about the drivers.")
    print("\n  Tolerances are NOT set here. Numbers go to JC.")

    os.makedirs(RUNS, exist_ok=True)
    path = os.path.join(RUNS, "driver_diag.json")
    with open(path, "w") as f:
        json.dump(
            {
                "platform": dev.platform,
                "device": str(dev),
                "jax": jax.__version__,
                "integrators": recs,
                "fnl_vs_ic": fnl,
            },
            f,
            indent=1,
        )
    print(f"\nwrote {path}")


if __name__ == "__main__":
    main()
