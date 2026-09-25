"""M1 S6 probe: attribute the inexor <-> DISCO-DJ high-k parity gap.

Hypothesis (from reading discodj 0.0.2 nbody/acc.py + core/kernels.py):
DISCO-DJ's order-0 gradient kernel ZEROES THE NYQUIST PLANE in the gradient
direction; the mbody lineage keeps -i k_nyq there. Everything else in the
pure-PM force path (CIC scatter, DC-zeroed density, -1/k^2 with the same DC
guard, CIC gather) is structurally identical.

This probe runs the inexor f64 float BullFrog path twice from the injected
ICs -- stock kernel vs Nyquist-zeroed kernel (a LOCAL variant; the library
kernel is untouched) -- and compares both against the saved disco_final.
If the zeroed arm collapses the gap, the attribution is named and the
residual is the growth-table / coefficient floor the parity gate can sit on.

    pixi run python scripts/m1_kernel_probe.py --tag n64k10log_bullfrog_lpt2_s0
"""

import argparse
import os
import sys

os.environ.setdefault("JAX_ENABLE_X64", "1")

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import _m1_common as M  # noqa: E402


def _force_fn(n_mesh, box_size, n_total, zero_nyquist):
    """f64 CIC PM force, optionally with DISCO-DJ's Nyquist-zeroed ik kernel."""
    import jax.numpy as jnp

    from inexor.forces import k_components
    from inexor.painting import cic_read_vector, density_contrast

    N, L = n_mesh, box_size
    ikx, iky, ikz, inv_k2 = k_components(N, L, np.float64)
    if zero_nyquist:
        ikx = np.asarray(ikx).copy()
        iky = np.asarray(iky).copy()
        ikz = np.asarray(ikz).copy()
        ikx[N // 2, :, :] = 0.0
        iky[:, N // 2, :] = 0.0
        ikz[:, :, -1] = 0.0  # rfft axis: Nyquist sits at the end
        ikx, iky, ikz = jnp.asarray(ikx), jnp.asarray(iky), jnp.asarray(ikz)

    def force(pos):
        delta = density_contrast(pos, N, L, n_total, paint="f32").astype(jnp.float64)
        dk = jnp.fft.rfftn(delta)
        gx = jnp.fft.irfftn(dk * ikx * inv_k2, s=(N, N, N))
        gy = jnp.fft.irfftn(dk * iky * inv_k2, s=(N, N, N))
        gz = jnp.fft.irfftn(dk * ikz * inv_k2, s=(N, N, N))
        return cic_read_vector(gx, gy, gz, pos, N, L).astype(jnp.float64)

    return force


def _evolve(ics, cfg, zero_nyquist):
    import jax

    jax.config.update("jax_enable_x64", True)
    import jax.numpy as jnp

    from inexor import config, integrate

    cosmo = config.Cosmology(**M.COSMO)
    a_steps = np.asarray(ics["a_steps"], dtype=np.float64)
    force = _force_fn(cfg["n_mesh"], cfg["box_size"], cfg["n_mesh"] ** 3, zero_nyquist)
    # v1 stored-reference conventions: EdS weights (D-013)
    coeffs = integrate.bullfrog_float_coeffs(integrate.bullfrog_table(a_steps, cosmo, growth2="eds"))
    x = jnp.asarray(ics["x"], dtype=jnp.float64)
    v = jnp.asarray(ics["v_d"], dtype=jnp.float64)
    for c in coeffs:
        x, v = integrate.float_step_bullfrog(
            x, v, tuple(np.asarray(c, np.float64)), force, cfg["box_size"]
        )
    return np.mod(np.asarray(x), cfg["box_size"])


def _band_table(label, out, k):
    n = len(k)
    ratio = np.abs(np.array(out["p_ratio"]) - 1.0)
    print(f"  {label}")
    for lo, hi, lab in [
        (0, 4, "low-k"),
        (4, 16, "k<0.4*"),
        (16, n - 8, "mid"),
        (n - 8, n, "Nyquist"),
    ]:
        if hi > lo:
            print(f"    {lab:8s} |dP/P| max {ratio[lo:hi].max():.3e}")
    print(f"    rms {out['rms_cells']:.3e} cells, max 1-r {out['one_minus_r_max']:.3e}")


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--tag", required=True)
    args = ap.parse_args()

    ics = M.load_state(os.path.join(M.RUNS, f"ics_{args.tag}.npz"))
    cfg = ics["meta"]["config"]
    disco = M.load_state(os.path.join(M.RUNS, f"disco_final_{args.tag}.npz"))
    n, L = cfg["n_mesh"], cfg["box_size"]

    x_stock = _evolve(ics, cfg, zero_nyquist=False)
    x_zeroed = _evolve(ics, cfg, zero_nyquist=True)

    k = None
    print(f"tag {args.tag}: inexor f64 float arms vs saved disco_final")
    for label, xa in [
        ("stock ik kernel   vs disco", x_stock),
        ("Nyquist-zeroed ik vs disco", x_zeroed),
    ]:
        out = M.compare_states(xa, disco["x"], n, L)
        if k is None:
            k = np.array(out["k"])
        _band_table(label, out, k)
    out = M.compare_states(x_stock, x_zeroed, n, L)
    _band_table("stock vs zeroed (the kernel term itself)", out, k)


if __name__ == "__main__":
    main()
