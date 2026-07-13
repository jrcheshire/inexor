"""M1 S6 probe: compare the raw PM force operators on the SAME configuration.

The K-scan showed the inexor <-> DISCO-DJ gap converges (K=10 vs K=80 nearly
identical profiles) to a k-increasing plateau -- a force-OPERATOR difference,
not stepping. This probe evaluates both codes' accelerations for the same
injected 2LPT particle configuration and compares per particle.

Two sides (each in its own env):

    pixi run --manifest-path ~/spherex/disco-mocks/pixi.toml \
        python scripts/m1_force_probe.py --tag <tag> --side disco
    pixi run python scripts/m1_force_probe.py --tag <tag> --side inexor

The disco side writes runs/m1/force_disco_<tag>.npz; the inexor side computes
its own force (stock and Nyquist-zeroed ik variants) and prints the
comparison (rms/max per-particle force diff + P(k) of the CIC-painted
difference-magnitude proxy via displacement-free direct stats).
"""

import argparse
import os
import sys

os.environ.setdefault("JAX_ENABLE_X64", "1")

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import _m1_common as M  # noqa: E402


def side_disco(tag):
    import jax

    jax.config.update("jax_enable_x64", True)
    from discodj.core.grids import get_fourier_grid
    from discodj.nbody.acc import calc_acc_PM

    ics = M.load_state(os.path.join(M.RUNS, f"ics_{tag}.npz"))
    cfg = ics["meta"]["config"]
    n, L = cfg["n_mesh"], cfg["box_size"]

    # psi = x - q with the same q the codes share (verified identity)
    d = L / n
    coords = np.arange(n, dtype=np.float64) * d
    qx, qy, qz = np.meshgrid(coords, coords, coords, indexing="ij")
    q = np.stack([qx.ravel(), qy.ravel(), qz.ravel()], axis=1)
    psi = ics["x"] - q
    psi -= L * np.round(psi / L)  # periodic min-image displacement

    kg = get_fourier_grid((n,) * 3, L, sparse_k_vecs=True, dtype_num=64, with_jax=False)
    k_vecs = kg["k_vecs"]
    g = calc_acc_PM(
        psi=np.asarray(psi),
        dim=3,
        res_pm=n,
        n_part=n,
        k=None,
        k_vecs=k_vecs,
        boxsize=L,
        antialias=0,
        grad_order=0,
        lap_order=0,
        dtype_num=64,
        worder=2,
        deconvolve=False,
        with_jax=True,
    )
    g = np.asarray(g, dtype=np.float64).reshape(-1, 3)
    path = os.path.join(M.RUNS, f"force_disco_{tag}.npz")
    np.savez_compressed(path, g=g)
    print(f"wrote {path}  (|g| rms {np.sqrt((g**2).mean()):.6e})")


def side_coeffs(tag):
    """Dump DISCO-DJ's exact runtime BullFrog coefficients for our a_steps.

    Replicates DKDPiIntegrator.bullfrog() (steppers/dkd_pi_integrator.py) with
    the explicit-array time_var convention: internal time = STEP INDEX, so
    a_mid is the ARITHMETIC midpoint of the a boundaries (disco_stepper.py
    internal_to_a linear interp), and the two drift halves are unequal in D.
    alpha uses DISCO-DJ's tabulated true-LCDM D2plus (not the EdS relation).
    """
    import jax

    jax.config.update("jax_enable_x64", True)
    from discodj import DiscoDJ

    ics = M.load_state(os.path.join(M.RUNS, f"ics_{tag}.npz"))
    cfg = ics["meta"]["config"]
    a = np.asarray(ics["a_steps"], dtype=np.float64)
    dj = DiscoDJ(
        dim=3,
        res=cfg["n_mesh"],
        boxsize=cfg["box_size"],
        cosmo=dict(M.COSMO_DISCO),
        precision="double",
    ).with_timetables()
    c = dj.cosmo

    D1u = float(np.asarray(c._timetables["Dplus_unnormed_at_1"]))

    def Dp(x):
        return np.asarray(c.Dplus(x), dtype=np.float64)

    def Dpda(x):
        return np.asarray(c.Dplusda(x), dtype=np.float64)

    def D2p(x):
        return np.asarray(
            c.get_interpolated_property(x, key_from="a", key_to="D2plus"), dtype=np.float64
        )

    def D2pda(x):
        return np.asarray(
            c.get_interpolated_property(x, key_from="a", key_to="D2plusda"), dtype=np.float64
        )

    a_mid = 0.5 * (a[:-1] + a[1:])  # internal = step index -> arithmetic a midpoint
    all_D = Dp(a) * D1u
    all_Dda = Dpda(a) * D1u
    all_D2 = D2p(a) * D1u**2
    all_D2da = D2pda(a) * D1u**2
    D_begin, D_end = all_D[:-1], all_D[1:]
    dD = D_end - D_begin
    xi = D_begin / dD
    bracket = (all_D2[:-1] + all_D2da[:-1] / all_Dda[:-1] * dD / 2.0) / ((xi + 0.5) * dD) - (
        xi + 0.5
    ) * dD
    alphas = (all_D2da[1:] / all_Dda[1:] - bracket) / (all_D2da[:-1] / all_Dda[:-1] - bracket)
    D_mid_norm = Dp(a_mid)
    betas = (1.0 - alphas) / D_mid_norm
    dd1 = D_mid_norm - D_begin / D1u
    dd2 = D_end / D1u - D_mid_norm

    path = os.path.join(M.RUNS, f"disco_coeffs_{tag}.npz")
    np.savez(path, alphas=alphas, betas=betas, dd1=dd1, dd2=dd2)
    print(f"wrote {path}")
    print(f"  alpha[0] {alphas[0]:.10f}, dd1[0]/dd2[0] {dd1[0] / dd2[0]:.6f}")


def side_inexor(tag):
    import jax

    jax.config.update("jax_enable_x64", True)
    import jax.numpy as jnp

    from inexor import config
    from inexor.forces import k_components, make_force_fn
    from inexor.painting import cic_read_vector, density_contrast

    ics = M.load_state(os.path.join(M.RUNS, f"ics_{tag}.npz"))
    cfg = ics["meta"]["config"]
    n, L = cfg["n_mesh"], cfg["box_size"]
    box = config.BoxConfig(n_mesh=n, box_size=L)
    x = jnp.asarray(ics["x"], dtype=jnp.float64)

    g_stock = np.asarray(make_force_fn(box, fdtype=jnp.float64, paint="f32")(x))

    # Nyquist-zeroed ik variant (DISCO-DJ's gradient-kernel convention)
    ikx, iky, ikz, inv_k2 = k_components(n, L, np.float64)
    ikx = np.asarray(ikx).copy()
    iky = np.asarray(iky).copy()
    ikz = np.asarray(ikz).copy()
    ikx[n // 2, :, :] = 0.0
    iky[:, n // 2, :] = 0.0
    ikz[:, :, -1] = 0.0
    delta = density_contrast(x, n, L, n**3, paint="f32").astype(jnp.float64)
    dk = jnp.fft.rfftn(delta)
    gx = jnp.fft.irfftn(dk * jnp.asarray(ikx) * inv_k2, s=(n, n, n))
    gy = jnp.fft.irfftn(dk * jnp.asarray(iky) * inv_k2, s=(n, n, n))
    gz = jnp.fft.irfftn(dk * jnp.asarray(ikz) * inv_k2, s=(n, n, n))
    g_zeroed = np.asarray(cic_read_vector(gx, gy, gz, x, n, L))

    with np.load(os.path.join(M.RUNS, f"force_disco_{tag}.npz")) as f:
        g_disco = f["g"]

    g_rms = np.sqrt((g_disco**2).mean())
    print(f"|g_disco| rms {g_rms:.6e};  |g_stock| rms {np.sqrt((g_stock**2).mean()):.6e}")
    for label, ga in [("stock ", g_stock), ("zeroed", g_zeroed)]:
        dgi = ga - g_disco
        print(
            f"  inexor {label} vs disco: rms |dg|/rms|g| {np.sqrt((dgi**2).mean()) / g_rms:.3e}, "
            f"max |dg|/rms|g| {np.abs(dgi).max() / g_rms:.3e}"
        )
    dgi = g_stock - g_zeroed
    print(f"  stock vs zeroed (kernel term): rms {np.sqrt((dgi**2).mean()) / g_rms:.3e}")


def side_replay(tag):
    """Replay inexor's float path with DISCO-DJ's exact recipe (zeroed-Nyquist
    kernel + dumped coefficients + their unequal drift halves). Agreement with
    disco_final at the f64 level closes the attribution."""
    import jax

    jax.config.update("jax_enable_x64", True)
    import jax.numpy as jnp

    from inexor.forces import k_components
    from inexor.painting import cic_read_vector, density_contrast

    ics = M.load_state(os.path.join(M.RUNS, f"ics_{tag}.npz"))
    cfg = ics["meta"]["config"]
    n, L = cfg["n_mesh"], cfg["box_size"]
    with np.load(os.path.join(M.RUNS, f"disco_coeffs_{tag}.npz")) as f:
        alphas, betas, dd1, dd2 = f["alphas"], f["betas"], f["dd1"], f["dd2"]

    ikx, iky, ikz, inv_k2 = k_components(n, L, np.float64)
    ikx = np.asarray(ikx).copy()
    iky = np.asarray(iky).copy()
    ikz = np.asarray(ikz).copy()
    ikx[n // 2, :, :] = 0.0
    iky[:, n // 2, :] = 0.0
    ikz[:, :, -1] = 0.0
    ikx, iky, ikz = jnp.asarray(ikx), jnp.asarray(iky), jnp.asarray(ikz)

    def force(pos):
        delta = density_contrast(pos, n, L, n**3, paint="f32").astype(jnp.float64)
        dk = jnp.fft.rfftn(delta)
        gx = jnp.fft.irfftn(dk * ikx * inv_k2, s=(n, n, n))
        gy = jnp.fft.irfftn(dk * iky * inv_k2, s=(n, n, n))
        gz = jnp.fft.irfftn(dk * ikz * inv_k2, s=(n, n, n))
        return cic_read_vector(gx, gy, gz, pos, n, L)

    x = jnp.asarray(ics["x"], dtype=jnp.float64)
    v = jnp.asarray(ics["v_d"], dtype=jnp.float64)
    for i in range(len(alphas)):
        x = jnp.mod(x + dd1[i] * v, L)
        g = force(x)
        v = alphas[i] * v + betas[i] * g
        x = jnp.mod(x + dd2[i] * v, L)

    disco = M.load_state(os.path.join(M.RUNS, f"disco_final_{tag}.npz"))
    out = M.compare_states(np.asarray(x), disco["x"], n, L)
    print(f"disco-recipe replay vs disco_final ({tag}):")
    print(
        f"  rms {out['rms_cells']:.3e} cells, max |dP/P| {out['ratio_max_absdev']:.3e}, "
        f"max 1-r {out['one_minus_r_max']:.3e}"
    )


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--tag", required=True)
    ap.add_argument("--side", required=True, choices=["disco", "inexor", "coeffs", "replay"])
    args = ap.parse_args()
    if args.side == "disco":
        side_disco(args.tag)
    elif args.side == "coeffs":
        side_coeffs(args.tag)
    elif args.side == "replay":
        side_replay(args.tag)
    else:
        side_inexor(args.tag)


if __name__ == "__main__":
    main()
