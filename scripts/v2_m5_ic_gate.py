"""M-v2-5 exit gate: streamed ICs + the out-of-core FFT. Probe code, NOT package code.

Built in `v2_m3_engine_gate.py`'s mould. Legs land stage by stage; the leg table
below records which exist at any given commit.

WHY NOT THE INSTRUMENT THE CHARTER NAMES. D-v2-18's ladder row says "tile-IC
identity vs monolithic at f64". It cannot be read literally: D-v2-15 clause 5
(ratified at the V4 freeze) establishes that `jax.random.normal`'s stream is
shape-dependent, so a slab-decomposed white-noise field is NOT bit-identical to
a monolithic one -- there is no old-monolithic field the new tiles could be
identical TO. Third milestone running whose written exit criterion could not be
read literally (M-v2-3, M-v2-4 precedent). The re-scope, JC-ratified 2026-08-10:

  (i)  DECOMPOSITION INVARIANCE of the new per-plane fold_in construction --
       the same field, bitwise at f64, whatever slab thickness or tiling
       produced it (clause 5's own definition of reproducibility);
  (ii) END-TO-END IDENTITY -- the streamed generator through T9 slabs and the
       loader is bitwise the monolithic SlotState build of the same (new) field
       at dev scale;
  (iii) old-vs-new agreement is STATISTICAL only (identical ensembles by
       construction; the comparison is plumbing, not physics).

THE MIRROR AND ITS LICENSE (leg `mirror-license`, this commit). The old colour
path is reproduced probe-locally (`_old_gaussian_delta`, `_old_linear_density`,
`_old_poisson_factor`) so the memory ladder and the statistical arm keep an
old-stream reference after `inexor.ic` is replaced in place. The license is a
bitwise degenerate-identity check against `inexor.ic` run ON THE PRE-REPLACEMENT
TREE -- the same license `v2_g5b_abs_transfer.py`'s `_colour` carries, and like
it, the check is only re-runnable at commits where the old code still exists.
After the replacement this leg MUST fail its identity arms (the new stream is a
different realization at every seed; D-v2-15 clause 5 says so), and the card
written here is the frozen record that the mirror was exact when it could be
checked.

Legs, and where each runs (stage in parentheses = not yet built):

  mirror-license  bitwise mirror vs inexor.ic     laptop CPU   this commit
  (0   construction ledger                        laptop       S7)
  (I   decomposition invariance                   laptop/deneb S7)
  (II  transfer-table bar                         laptop       S7; probe at S1)
  (III end-to-end SlotState identity              laptop/deneb S7)
  (IV  old-vs-new statistical, 32 seeds N=128     laptop       S7)
  (V   memory ladder, 3 arms x 3 rungs            antares      S8)
  (VI  2048^3 out-of-core FFT                     Vista gh     S8, own script)
"""

import argparse
import hashlib
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(HERE)
OUT_DIR = os.path.join(REPO, "runs", "v2")

SEED = 0


# ---------------------------------------------------------------------------
# The old-stream mirror: verbatim reimplementation of the PRE-M-v2-5 ic.py
# colour path. `cosmology.linear_power(backend="eh98")`, `transfer_eh98` and
# `growth_factor_md` are NOT being replaced (they are the reference the new
# table is validated against), so importing them here is legitimate; what is
# mirrored is exactly what M-v2-5 replaces -- the monolithic
# `jax.random.normal` stream and the full-3D-half-grid colour evaluation.
# ---------------------------------------------------------------------------

C_OVER_H0 = 299792.458 / 100.0


def _old_gaussian_delta(key, n_mesh, box_size, cosmo, fdtype=None):
    """Verbatim mirror of pre-M-v2-5 ic.gaussian_delta (eh98 backend, amplitude 1)."""
    import jax
    import jax.numpy as jnp

    from inexor.cosmology import linear_power

    if fdtype is None:
        fdtype = jnp.float32
    N, L = n_mesh, box_size
    white = jax.random.normal(key, (N, N, N), dtype=fdtype)
    dk = jnp.fft.rfftn(white)
    kx = 2.0 * np.pi * np.fft.fftfreq(N, d=L / N)
    kz = 2.0 * np.pi * np.fft.rfftfreq(N, d=L / N)
    kk = np.sqrt(kx.reshape(N, 1, 1) ** 2 + kx.reshape(1, N, 1) ** 2 + kz.reshape(1, 1, -1) ** 2)
    kk_safe = kk.copy()
    kk_safe[0, 0, 0] = kk.flat[1]
    colour = np.sqrt(linear_power(kk_safe.ravel(), cosmo).reshape(kk.shape) * N**3 / L**3)
    colour[0, 0, 0] = 0.0
    npdt = np.float64 if fdtype == jnp.float64 else np.float32
    dk = dk * jnp.asarray(colour.astype(npdt))
    return jnp.fft.irfftn(dk, s=(N, N, N))


def _old_poisson_factor(n_mesh, box_size, cosmo, z=0.0):
    """Verbatim mirror of pre-M-v2-5 ic.poisson_factor (full 3D half-grid M(k))."""
    from inexor.cosmology import growth_factor_md, transfer_eh98

    N, L = n_mesh, box_size
    kx = 2.0 * np.pi * np.fft.fftfreq(N, d=L / N)
    kz = 2.0 * np.pi * np.fft.rfftfreq(N, d=L / N)
    k_mag = np.sqrt(
        kx.reshape(N, 1, 1) ** 2 + kx.reshape(1, N, 1) ** 2 + kz.reshape(1, 1, -1) ** 2
    )
    k_safe = np.where(k_mag > 0, k_mag, 1.0)
    T = transfer_eh98(k_safe.ravel(), cosmo).reshape(k_mag.shape)
    a = 1.0 / (1.0 + z)
    D_md = growth_factor_md(a, cosmo)
    M = (2.0 / 3.0) * C_OVER_H0**2 * k_mag**2 * T * D_md / cosmo.Omega_m
    return np.where(k_mag > 0, M, 1.0)


def _old_linear_density(key, n_mesh, box_size, cosmo, f_NL=0.0, fdtype=None):
    """Verbatim mirror of pre-M-v2-5 ic.linear_density."""
    import jax.numpy as jnp

    N = n_mesh
    if fdtype is None:
        fdtype = jnp.float32
    delta_G = _old_gaussian_delta(key, N, box_size, cosmo, fdtype)
    npdt = np.float64 if fdtype == jnp.float64 else np.float32
    M = jnp.asarray(_old_poisson_factor(N, box_size, cosmo).astype(npdt))
    phi_G = jnp.fft.irfftn(jnp.fft.rfftn(delta_G) / M, s=(N, N, N))
    mean_phi2 = jnp.mean(phi_G**2)
    phi_NG = phi_G + f_NL * (phi_G**2 - mean_phi2)
    return jnp.fft.irfftn(jnp.fft.rfftn(phi_NG) * M, s=(N, N, N))


# ---------------------------------------------------------------------------
# leg: mirror-license
# ---------------------------------------------------------------------------


def leg_mirror_license():
    """The mirror against inexor.ic, with expectations set by the tree's era.

    PRE-REPLACEMENT (ic.IC_STREAM absent): bitwise identity on every arm --
    the license, banked as the frozen card m5_gate_mirror-license.json at the
    S0 commit. POST-REPLACEMENT (IC_STREAM present): the field arms MUST
    DIFFER at every seed (D-v2-15 clause 5 -- identity here would mean the
    fold_in stream was never wired), while poisson_factor, whose analytic body
    did not move, MUST still match. Same instrument, two eras, both fail-able.

    Anti-vacuity either way: the fields have real dynamic range (rms > 0.1),
    and the different-seed control does NOT match.
    """
    import jax
    import jax.numpy as jnp

    from inexor import ic
    from inexor.config import Cosmology

    new_stream = hasattr(ic, "IC_STREAM")
    fields_equal = not new_stream
    print(f"  era: {'post-replacement (fields must differ)' if new_stream else 'pre-replacement (license: fields must match)'}",
          flush=True)

    cosmo = Cosmology()
    arms = {}
    ok = True

    def _cmp(name, a, b, expect_equal=True):
        nonlocal ok
        a = np.asarray(a)
        b = np.asarray(b)
        n_diff = int(np.sum(a != b))
        rms = float(np.sqrt(np.mean(a.astype(np.float64) ** 2)))
        passed = (n_diff == 0) == expect_equal
        ok = ok and passed
        arms[name] = dict(n_diff=n_diff, rms=rms, expect_equal=expect_equal, ok=passed)
        tag = "ok" if passed else "FAIL"
        print(f"  {name}: n_diff={n_diff} rms={rms:.3e} [{tag}]", flush=True)

    for N in (64, 128):
        for fdt, fname in ((jnp.float64, "f64"), (jnp.float32, "f32")):
            key = jax.random.PRNGKey(SEED)
            mine = _old_gaussian_delta(key, N, 128.0, cosmo, fdt)
            theirs = ic.gaussian_delta(key, N, 128.0, cosmo, fdtype=fdt)
            _cmp(f"gaussian_delta_n{N}_{fname}", mine, theirs, expect_equal=fields_equal)

    key = jax.random.PRNGKey(SEED)
    for f_NL in (0.0, 10.0):
        mine = _old_linear_density(key, 64, 128.0, cosmo, f_NL=f_NL, fdtype=jnp.float64)
        theirs = ic.linear_density(key, 64, 128.0, cosmo, f_NL=f_NL, fdtype=jnp.float64)
        _cmp(f"linear_density_n64_f64_fnl{int(f_NL)}", mine, theirs, expect_equal=fields_equal)

    _cmp(
        "poisson_factor_n64",
        _old_poisson_factor(64, 128.0, cosmo),
        ic.poisson_factor(64, 128.0, cosmo),
    )

    # anti-vacuity: rms must be real, and a different seed must NOT match
    rms_all = [a["rms"] for a in arms.values()]
    if min(rms_all) < 0.1:
        ok = False
        print(f"  ANTI-VACUITY FAIL: min rms {min(rms_all):.3e} < 0.1", flush=True)
    other = _old_gaussian_delta(jax.random.PRNGKey(SEED + 1), 64, 128.0, cosmo, jnp.float64)
    same = ic.gaussian_delta(jax.random.PRNGKey(SEED), 64, 128.0, cosmo, fdtype=jnp.float64)
    _cmp("different_seed_differs", other, same, expect_equal=False)

    fp = hashlib.sha256(
        np.asarray(
            _old_gaussian_delta(jax.random.PRNGKey(SEED), 64, 128.0, cosmo, jnp.float64)
        ).tobytes()
    ).hexdigest()
    return dict(ok=ok, arms=arms, old_stream_fingerprint_n64_f64_seed0=fp)


# ---------------------------------------------------------------------------
# provenance + card
# ---------------------------------------------------------------------------


def _provenance(args):
    """Where this card was produced and what was asked of it (m4 pattern).

    New load-bearing field this milestone: `ic_stream`. After M-v2-5 a seed
    under `inexor.ic` denotes a different realization than the same seed before
    it; cards carry the stream identity so a readout can REFUSE to pool across
    it. Cards written before the field exists read as UNKNOWN, never backfilled.
    `threefry_partitionable` is recorded because `fold_in`'s derived keys depend
    on it (True on the locked jax 0.10.2; the generator asserts rather than
    sets it).
    """
    import jax

    from inexor import ic

    dev = jax.devices()[0]
    try:
        x64 = bool(jax.config.jax_enable_x64)
    except Exception:
        x64 = None
    return dict(
        backend=dev.platform,
        device_kind=dev.device_kind,
        n_devices=jax.device_count(),
        host_cores=os.cpu_count(),
        x64=x64,
        jax_version=jax.__version__,
        threefry_partitionable=bool(jax.config.jax_threefry_partitionable),
        ic_stream=getattr(ic, "IC_STREAM", "pre-m5-normal"),
        xla_flags=os.environ.get("XLA_FLAGS"),
        omp_num_threads=os.environ.get("OMP_NUM_THREADS"),
        argv=sys.argv[1:],
    )


def _write(res, args):
    import subprocess

    try:
        res["commit"] = subprocess.check_output(
            ["git", "-C", REPO, "rev-parse", "HEAD"], text=True
        ).strip()
    except Exception:
        res["commit"] = None
    res["slurm_job_id"] = os.environ.get("SLURM_JOB_ID")
    res["provenance"] = _provenance(args)
    os.makedirs(OUT_DIR, exist_ok=True)
    path = os.path.join(OUT_DIR, f"m5_gate_{args.leg}{args.out_suffix}.json")
    with open(path, "w") as fh:
        json.dump(res, fh, indent=1)
    print(f"  card -> {path}", flush=True)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--leg", default="mirror-license", choices=["mirror-license"])
    ap.add_argument("--out-suffix", default="")
    args = ap.parse_args()

    import jax

    jax.config.update("jax_enable_x64", True)

    if args.leg == "mirror-license":
        print("leg mirror-license: the old-stream mirror vs inexor.ic, bitwise", flush=True)
        res = leg_mirror_license()
    else:  # pragma: no cover
        raise SystemExit(f"unknown leg {args.leg!r}")

    _write(res, args)
    if not res["ok"]:
        print("LEG FAILED", flush=True)
        return 1
    print("leg passed", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
