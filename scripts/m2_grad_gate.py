"""M2 S3: gradient-fidelity gate, FLOOR-FIRST.

The promoted adjoint = int-paint primal trajectory (bit-exact replay) + f32-paint
STE-twin VJP (D-006). We must show its gradients are faithful. Floor-first
(CLAUDE.md tolerances convention; reference_oracle_parity_floor): measure the
REFERENCE's own floors before quoting any adjoint gate number.

Three references, in order:
  1. float-precision floor: f32 vs f64 float-twin jax.grad (the autodiff
     reference's own numerical spread).
  2. FD floor: central FD of the f64 FLOAT loss vs f64 float-twin jax.grad
     (validates the smooth reference; FD of the QUANTIZED loss is invalid below
     the lattice step -- R4 staircase).
  3. the quantity to GATE: adjoint (quantized) vs float-twin jax.grad, for
     input grads (d/dx0, d/dv0) AND IC-param grads (d/df_NL, d/damplitude), both
     losses (band-power + field-L2). This IS the D-010 R4 restatement: report
     against the ratified 7.5e-2 R4 gate.

Metrics are GLOBAL (norm ratio, correlation, median-rel on the top-decile |ref|)
-- single-component rel is noise-dominated where the reference grad ~ 0.

Run (laptop/deneb, x64 for clean FD):
    pixi run python scripts/m2_grad_gate.py [--quick]
Outputs runs/m2/grad_gate.json + a printed table. Gate numbers are ratified
with JC AFTER reading the floors -- this script sets none.
"""

# ruff: noqa: E402  (x64 env var must precede any array-creating jax import)
import argparse
import json
import os

os.environ.setdefault("JAX_ENABLE_X64", "1")  # x64 before jax init (caller opts in)

import jax

jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
import numpy as np

from inexor import PLANCK, BoxConfig, QuantConfig, TimeConfig
from inexor.adjoint import adjoint_grad_ic, evolve_grad
from inexor.ic import linear_density
from inexor.integrate import evolve_float
from inexor.losses import band_power_loss, density_f32, field_l2_loss, fundamental_k_edges
from inexor.lpt import lpt_ics

COSMO = PLANCK
QUANT = QuantConfig()
REPO = os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
RUNS = os.path.join(REPO, "runs", "m2")

# Ratified gate (JC, M2 S3) -- global metrics, NOT per-component/FD (the loss
# gradient is per-particle noise-dominated; FD of the quantized loss is invalid
# below the lattice -- R4 staircase). ADR D-015.
GATE_MEDREL = 1e-2   # adjoint-vs-float input-grad median-rel (top-decile |ref|)
GATE_CORR = 0.999    # Pearson correlation
GATE_RATIO = 0.01    # |norm-ratio - 1|
GATE_IC_REL = 5e-3   # IC-param d/d[f_NL,amp] adjoint-vs-float rel
R4_GATE = 7.5e-2     # D-010 R4 bar (the restatement headline)


def _global_metrics(a, b):
    """norm ratio, Pearson corr, median-rel on the top-decile |b|."""
    a, b = np.asarray(a).ravel(), np.asarray(b).ravel()
    ratio = float(np.linalg.norm(a) / np.linalg.norm(b))
    corr = float(np.corrcoef(a, b)[0, 1])
    thr = np.quantile(np.abs(b), 0.9)
    m = np.abs(b) >= thr
    medrel = float(np.median(np.abs(a[m] - b[m]) / np.abs(b[m])))
    maxrel = float(np.max(np.abs(a[m] - b[m]) / np.abs(b[m])))
    return dict(ratio=ratio, corr=corr, medrel=medrel, maxrel=maxrel)


def _make_ic(n_mesh, box_size, f_NL, amplitude, fdtype, seed=0):
    key = jax.random.PRNGKey(seed)
    d0 = amplitude * linear_density(key, n_mesh, box_size, COSMO, f_NL=f_NL, fdtype=fdtype)
    return lpt_ics(d0, box_size, 0.1, COSMO, order=2, fdtype=fdtype)


def _loss_builders(box, f_NL0):
    """Return {name: loss_field(x_f, v_f)} for the two loss classes."""
    k_edges = fundamental_k_edges(box.n_mesh, box.box_size, n_bins=8)
    # field-L2 target = the f_NL=0 evolved-IC density (a fixed reference field)
    x_t, _ = _make_ic(box.n_mesh, box.box_size, 0.0, 1.0, jnp.float64)
    target_delta = np.asarray(density_f32(x_t, box))
    zeros_pk = np.zeros(len(k_edges) - 1)

    def band(x_f, v_f):
        return band_power_loss(x_f, box, k_edges, zeros_pk.astype(x_f.dtype))

    def fieldl2(x_f, v_f):
        return field_l2_loss(x_f, box, target_delta.astype(x_f.dtype))

    return {"band_power": band, "field_l2": fieldl2}


def run_config(n_mesh, K, integrator, f_NL0=5.0, fd_samples=24):
    box = BoxConfig(n_mesh=n_mesh, box_size=256.0)
    time = TimeConfig(a_init=0.1, a_final=1.0, n_steps=K, integrator=integrator)
    losses = _loss_builders(box, f_NL0)
    out = {"n_mesh": n_mesh, "K": K, "integrator": integrator, "losses": {}}

    for lname, loss_field in losses.items():
        rec = {}

        # ---- input-grad references at the fixed (f_NL0, amp=1) ICs ----
        for fdtype, tag in ((jnp.float64, "f64"), (jnp.float32, "f32")):
            x0, v0 = _make_ic(n_mesh, box.box_size, f_NL0, 1.0, fdtype)
            x0, v0 = x0.astype(fdtype), v0.astype(fdtype)

            def lq(x0_, v0_, _fd=fdtype):
                return loss_field(*evolve_grad(box, time, QUANT, COSMO, "scan", _fd, x0_, v0_))

            def lf(x0_, v0_, _fd=fdtype):
                return loss_field(*evolve_float(box, time, COSMO, x0_, v0_, paint="f32", fdtype=_fd))

            gq = jax.grad(lq, argnums=(0, 1))(x0, v0)  # adjoint (quantized)
            gf = jax.grad(lf, argnums=(0, 1))(x0, v0)  # float-twin
            rec[f"adjoint_vs_floattwin_x_{tag}"] = _global_metrics(gq[0], gf[0])
            rec[f"adjoint_vs_floattwin_v_{tag}"] = _global_metrics(gq[1], gf[1])
            if tag == "f64":
                gf64 = gf
                lf64 = lf
                x064, v064 = x0, v0
            else:
                # (1) float-precision floor: f32 vs f64 float-twin grad
                rec["floor_floattwin_f32_vs_f64_x"] = _global_metrics(gf[0], gf64[0])
                rec["floor_floattwin_f32_vs_f64_v"] = _global_metrics(gf[1], gf64[1])

        # (2) FD floor: central FD of the f64 FLOAT loss vs f64 float-twin grad
        key = jax.random.PRNGKey(7)
        rows = np.asarray(jax.random.randint(key, (fd_samples,), 0, box.n_total))
        eps = 1e-4
        fdx, adx = [], []
        for i in rows:
            for c in range(3):
                d = jnp.zeros_like(x064).at[i, c].set(eps)
                fdx.append(float((lf64(x064 + d, v064) - lf64(x064 - d, v064)) / (2 * eps)))
                adx.append(float(gf64[0][i, c]))
        rec["floor_FD_vs_floattwin_x"] = _global_metrics(np.array(adx), np.array(fdx))

        # ---- IC-param grads (d/df_NL, d/damplitude): adjoint vs float vs FD ----
        theta = (f_NL0, 1.0)
        g_adj = adjoint_grad_ic(loss_field, box, time, QUANT, COSMO, theta=theta, fdtype=jnp.float64)

        def lf_ic(th):
            x0, v0 = _make_ic(n_mesh, box.box_size, th[0], th[1], jnp.float64)
            return loss_field(*evolve_float(box, time, COSMO, x0, v0, paint="f32", fdtype=jnp.float64))

        g_flt = jax.grad(lf_ic)(jnp.asarray(theta, jnp.float64))
        # FD with per-parameter matched eps (f_NL is a large lever on a tiny term)
        eps_ic = np.array([50.0, 1e-4])
        fd_ic = []
        for i in range(2):
            d = jnp.zeros(2, jnp.float64).at[i].set(eps_ic[i])
            fd_ic.append(float((lf_ic(jnp.asarray(theta) + d) - lf_ic(jnp.asarray(theta) - d)) / (2 * eps_ic[i])))
        rec["ic_grad"] = {
            "param": ["f_NL", "amplitude"],
            "adjoint": [float(x) for x in g_adj],
            "float_twin": [float(x) for x in g_flt],
            "FD": fd_ic,
            "adjoint_vs_float_rel": [
                abs(float(g_adj[i]) - float(g_flt[i])) / (abs(float(g_flt[i])) + 1e-30) for i in range(2)
            ],
            "float_vs_FD_rel": [
                abs(float(g_flt[i]) - fd_ic[i]) / (abs(fd_ic[i]) + 1e-30) for i in range(2)
            ],
        }
        out["losses"][lname] = rec
    return out


def _fmt(m):
    return f"ratio {m['ratio']:.5f} corr {m['corr']:.6f} medrel {m['medrel']:.2e} maxrel {m['maxrel']:.2e}"


def _check_gate(results):
    """Apply the ratified gate to every config x loss. Returns (all_pass,
    worst_medrel, failures[])."""
    failures = []
    worst = 0.0
    for r in results:
        tag = f"{r['integrator']} {r['n_mesh']}^3 K={r['K']}"
        for lname, rec in r["losses"].items():
            for comp in ("x", "v"):
                m = rec[f"adjoint_vs_floattwin_{comp}_f64"]
                worst = max(worst, m["medrel"])
                ok = (m["medrel"] <= GATE_MEDREL and m["corr"] >= GATE_CORR
                      and abs(m["ratio"] - 1.0) <= GATE_RATIO)
                if not ok:
                    failures.append(f"{tag} [{lname}] grad-{comp}: {_fmt(m)}")
            for i, p in enumerate(rec["ic_grad"]["param"]):
                rel = rec["ic_grad"]["adjoint_vs_float_rel"][i]
                if rel > GATE_IC_REL:
                    failures.append(f"{tag} [{lname}] d/d{p} rel {rel:.2e} > {GATE_IC_REL}")
    return len(failures) == 0, worst, failures


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--quick", action="store_true", help="64^3 only")
    ap.add_argument("--k-sweep", default=None,
                    help="P2 drift arm: comma-separated K at 64^3 bullfrog, replacing the "
                         "gate configs. Float replay drift ACCUMULATES over steps while exact "
                         "replay does not, so error-vs-K is the comparison that decides whether "
                         "exact replay buys anything; a single K cannot show it.")
    args = ap.parse_args()
    if args.k_sweep:
        configs = [(64, int(k), "bullfrog") for k in args.k_sweep.split(",") if k.strip()]
    else:
        configs = [(64, 10, "bullfrog"), (64, 10, "fastpm")]
        if not args.quick:
            configs += [(64, 40, "bullfrog"), (128, 40, "bullfrog")]

    results = []
    for n, K, integ in configs:
        print(f"\n===== {integ} {n}^3 K={K} =====")
        r = run_config(n, K, integ)
        results.append(r)
        for lname, rec in r["losses"].items():
            print(f"  [{lname}]")
            print(f"    FLOOR float-twin f32-vs-f64: x {_fmt(rec['floor_floattwin_f32_vs_f64_x'])}")
            print(f"    FLOOR FD-vs-float-twin (f64): x {_fmt(rec['floor_FD_vs_floattwin_x'])}")
            print(f"    GATE  adjoint-vs-float-twin (f64): x {_fmt(rec['adjoint_vs_floattwin_x_f64'])}")
            print(f"    GATE  adjoint-vs-float-twin (f64): v {_fmt(rec['adjoint_vs_floattwin_v_f64'])}")
            ic = rec["ic_grad"]
            print(f"    IC d/d[f_NL,amp] adjoint-vs-float rel {[f'{x:.2e}' for x in ic['adjoint_vs_float_rel']]}"
                  f"  float-vs-FD rel {[f'{x:.2e}' for x in ic['float_vs_FD_rel']]}")

    all_pass, worst, failures = _check_gate(results)
    print("\n===== GATE (ratified D-015) =====")
    print(f"  median-rel <= {GATE_MEDREL}, corr >= {GATE_CORR}, |ratio-1| <= {GATE_RATIO}, "
          f"IC-rel <= {GATE_IC_REL}")
    print(f"  worst adjoint-vs-float median-rel = {worst:.2e}  (R4 restatement vs {R4_GATE}: "
          f"{'PASS' if worst <= R4_GATE else 'FAIL'})")
    print(f"  GATE: {'PASS' if all_pass else 'FAIL'}")
    for fl in failures:
        print(f"    FAIL {fl}")

    os.makedirs(RUNS, exist_ok=True)
    path = os.path.join(RUNS, "grad_gate.json")
    with open(path, "w") as f:
        json.dump({
            "gate": {"medrel": GATE_MEDREL, "corr": GATE_CORR, "ratio": GATE_RATIO,
                     "ic_rel": GATE_IC_REL, "R4_reference": R4_GATE},
            "gate_pass": all_pass, "worst_medrel": worst, "configs": results,
        }, f, indent=1)
    print(f"\nwrote {path}")


if __name__ == "__main__":
    main()
