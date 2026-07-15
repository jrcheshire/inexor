"""v2 probe G2b: one-shot codec ladder on a real evolved state (stored M1 data).

Run 2026-07-14; numbers quoted in docs/plan-plan-v2.md Sec 4 and the design
study Sec 3. Findings: int8 cell-relative positions measure 3.1e-5 max |dP/P|
one-shot (int16-global-grade at half the bytes, no 2^16 mesh ceiling); pure
LPT-residual position coding FAILS on saturated heavy tails (0.8% outliers
dominate; more bits do not help); every 8-bit velocity codec has tail
problems; CDF companding is worse than linear on the tails vs a static LPT
reference (CUBE's live scheme uses the local coarse-grid mean flow instead).

Calibration: D-014 measured the ACCUMULATED int16 error at this config as
max|dP/P| = 2.8e-4 vs 1.4e-5 one-shot here -> ~20x one-shot->accumulated
multiplier. The accumulated version of this ladder is gate G2c (seed V1).

Run from the repo root:  JAX_ENABLE_X64=1 pixi run python scripts/v2_g2b_codec_ladder.py
"""

import json
import os

import jax.numpy as jnp
import numpy as np
from scipy.special import erf, erfinv

from inexor.config import Cosmology
from inexor.cosmology import growth_factor_a
from inexor.diagnostics import cross_r, pk_estimator
from inexor.lpt import lagrangian_grid, zeldovich_displacement
from inexor.painting import density_contrast

RUNS = "runs/m1"
OUT_DIR = "runs/v2"
TAG, N = "n128k40log_bullfrog_lpt2_s0", 128


def wrap_min_image(d, L):
    return (d + 0.5 * L) % L - 0.5 * L


def quantize_linear(vals, quantum, lo, hi):
    """Uniform quantizer with SATURATION at [lo, hi]; returns (deq, outlier_frac)."""
    out = np.mean((vals < lo) | (vals > hi))
    q = np.clip(vals, lo, hi)
    return np.rint(q / quantum) * quantum, float(out)


def quantize_cdf_gauss(vals, sigma, levels=256, c=6.0):
    """Gaussian-companded quantizer: uniform bins in Phi(r/sigma)."""
    u = 0.5 * (1.0 + erf(vals / (np.sqrt(2.0) * sigma)))
    edge = 0.5 / levels
    u_q = (np.floor(u * levels) + 0.5) / levels
    u_q = np.clip(u_q, edge, 1.0 - edge)
    deq = np.sqrt(2.0) * sigma * erfinv(2.0 * u_q - 1.0)
    out = np.mean(np.abs(vals) > c * sigma)
    return deq, float(out)


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    ics = np.load(f"{RUNS}/ics_{TAG}.npz", allow_pickle=True)
    fin = np.load(f"{RUNS}/inexor_float_{TAG}.npz", allow_pickle=True)
    meta = json.loads(str(ics["meta"]))
    cfg = meta["config"]
    cosmo = Cosmology(**meta["cosmo"])
    L = float(cfg["box_size"])
    a_f = float(cfg["a_final"])

    delta0 = jnp.asarray(ics["delta0"], dtype=jnp.float64)
    q = np.asarray(lagrangian_grid(N, L, fdtype=jnp.float64))
    psi1 = np.asarray(zeldovich_displacement(delta0, L, fdtype=jnp.float64))
    D1f = growth_factor_a(a_f, cosmo)
    x_za = q + D1f * psi1
    v_za = psi1

    x_f = np.asarray(fin["x"])
    v_f = np.asarray(fin["v_d"])
    n_tot = x_f.shape[0]

    def delta_of(x):
        pos = jnp.asarray(np.mod(x, L))
        return density_contrast(pos, N, L, n_tot, paint="int")

    d_ref = delta_of(x_f)
    k_ref, p_ref, _ = pk_estimator(d_ref, L)
    nk = len(np.asarray(k_ref))
    lowk = slice(0, max(4, nk // 8))

    r_pos = wrap_min_image(x_f - x_za, L)
    sig_pos = float(np.sqrt(np.mean(r_pos**2)) / np.sqrt(3))

    results = {"config": TAG, "sigma_pos_resid": sig_pos}
    print(f"[{TAG}] sigma_pos_resid/comp = {sig_pos:.4f} Mpc/h; ref P(k) bins {nk}")

    # ---- position codecs ----
    pos_codecs = {}
    q16 = L / 2**16
    pos_codecs["int16_global"] = (np.rint(x_f / q16) * q16, 0.0)
    q12 = L / 2**12
    pos_codecs["int12_global"] = (np.rint(x_f / q12) * q12, 0.0)
    qcell8 = (L / N) / 256.0
    pos_codecs["int8_cellrel_CUBE"] = (np.rint(x_f / qcell8) * qcell8, 0.0)
    for c, bits, name in [(6.0, 8, "int8_residZA_c6"), (8.0, 8, "int8_residZA_c8"),
                          (6.0, 10, "int10_residZA_c6"), (6.0, 12, "int12_residZA_c6")]:
        quantum = 2 * c * sig_pos / (2**bits)
        rq, out = quantize_linear(r_pos, quantum, -c * sig_pos, c * sig_pos)
        pos_codecs[name] = (x_za + rq, out)

    for name, (xq, outfrac) in pos_codecs.items():
        d_q = delta_of(xq)
        _, p_q, _ = pk_estimator(d_q, L)
        _, r_x, _ = cross_r(d_q, d_ref, L)
        dpp = np.abs(np.asarray(p_q) / np.asarray(p_ref) - 1.0)
        one_minus_r = np.abs(1.0 - np.asarray(r_x))
        entry = {
            "max_dP_P": float(np.nanmax(dpp)),
            "lowk_dP_P": float(np.nanmean(dpp[lowk])),
            "max_1mr": float(np.nanmax(one_minus_r)),
            "outlier_frac": outfrac,
        }
        results[name] = entry
        print(f"  POS {name:22s} max|dP/P| {entry['max_dP_P']:.3e}  "
              f"lowk {entry['lowk_dP_P']:.3e}  max|1-r| {entry['max_1mr']:.3e}  "
              f"outliers {outfrac:.2e}")

    # ---- velocity codecs (representation error only; positions carry P(k)) ----
    r_vel = v_f - v_za
    sig_vel = float(np.sqrt(np.mean(r_vel**2)) / np.sqrt(3))
    sig_vfull = float(np.sqrt(np.mean(v_f**2)) / np.sqrt(3))
    results["sigma_vel_resid"] = sig_vel
    results["sigma_vel_full"] = sig_vfull
    print(f"  sigma_vel_resid {sig_vel:.4f}, sigma_vel_full {sig_vfull:.4f} (v_d units)")

    vel_codecs = {}
    qv16 = 2 * 6.0 * sig_vfull / 2**16
    vel_codecs["v_int16_full_c6"] = quantize_linear(v_f, qv16, -6 * sig_vfull, 6 * sig_vfull)
    qv8 = 2 * 6.0 * sig_vel / 256
    dq8, out8 = quantize_linear(r_vel, qv8, -6 * sig_vel, 6 * sig_vel)
    vel_codecs["v_int8_residZA_lin_c6"] = (v_za + dq8, out8)
    deq, out = quantize_cdf_gauss(r_vel, sig_vel, levels=256)
    vel_codecs["v_int8_residZA_cdf"] = (v_za + deq, out)
    qv12 = 2 * 6.0 * sig_vel / 4096
    dq12, out12 = quantize_linear(r_vel, qv12, -6 * sig_vel, 6 * sig_vel)
    vel_codecs["v_int12_residZA_lin_c6"] = (v_za + dq12, out12)

    for name, (vq, outfrac) in vel_codecs.items():
        err = vq - v_f
        entry = {
            "rms_err_over_sigvfull": float(np.sqrt(np.mean(err**2)) / (np.sqrt(3) * sig_vfull)),
            "p999_err": float(np.percentile(np.abs(err), 99.9)),
            "outlier_frac": float(outfrac),
        }
        results[name] = entry
        print(f"  VEL {name:22s} rms_err/sig_v {entry['rms_err_over_sigvfull']:.3e}  "
              f"p99.9 err {entry['p999_err']:.3e}  outliers {entry['outlier_frac']:.2e}")

    with open(f"{OUT_DIR}/g2b_results.json", "w") as fh:
        json.dump(results, fh, indent=2)
    print(f"wrote {OUT_DIR}/g2b_results.json")


if __name__ == "__main__":
    main()
