"""Amplitude of the local-f_NL squeezed bispectrum in inexor's ICs, with its uncertainty
(paper App. A.3).

The setup of tests/test_bispectrum.py::test_squeezed_matches_binned_template: linear density on
an n^3 grid in an L box (z = 0, seed s), triangles (m k_f, 8 k_f, 8 k_f) for m = 2, 3, 4; the
signal [B(+f_NL) - B(-f_NL)] / 2 at matched phases cancels the Gaussian term; the oracle is the
bin-averaged tree-level template (diagnostics.local_bispectrum_binned). The amplitude is the
least-squares scale of the template per seed; its mean and standard error over seeds (0 to
seeds - 1) are reported, with the per-bin ratios. CPU, float64.

    pixi run --frozen python scripts/validation/fnl_amplitude.py --f-nl 2000 --seeds 48
    pixi run --frozen python scripts/validation/fnl_amplitude.py --f-nl 100 --seeds 400
"""

import argparse
import os
import time

import numpy as np
from _common import setup_jax, write_json


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--n", type=int, default=32, help="mesh per side")
    ap.add_argument("--box", type=float, default=256.0, help="box side, Mpc/h")
    ap.add_argument("--f-nl", type=float, default=2000.0)
    ap.add_argument("--seeds", type=int, default=48, help="matched +-f_NL seed pairs")
    ap.add_argument("--out", default=None,
                    help="output JSON (default runs/validation/icval/fnl_f<f_NL>_s<seeds>.json)")
    args = ap.parse_args()
    out = args.out or os.path.join("runs", "validation", "icval",
                                   f"fnl_f{args.f_nl:g}_s{args.seeds}.json")

    jax = setup_jax(x64=True)
    import jax.numpy as jnp

    from inexor import ic
    from inexor.config import Cosmology
    from inexor.diagnostics import bispectrum, local_bispectrum_binned

    n, ell, f = args.n, args.box, args.f_nl
    cosmo = Cosmology()
    kf = 2.0 * np.pi / ell
    tris = [(m * kf, 8 * kf, 8 * kf) for m in (2, 3, 4)]

    def measure(seed, fnl):
        d = ic.linear_density(jax.random.PRNGKey(seed), n, ell, cosmo, f_NL=fnl,
                              fdtype=jnp.float64)
        return np.asarray(bispectrum(np.asarray(d, np.float64), ell, tris)[0])

    t0 = time.perf_counter()
    sig = np.array([0.5 * (measure(s, f) - measure(s, -f)) for s in range(args.seeds)])
    tmpl = np.asarray(local_bispectrum_binned(n, ell, cosmo, tris, f))
    per_seed = sig @ tmpl / np.sum(tmpl**2)
    amp, sem = float(per_seed.mean()), float(per_seed.std(ddof=1) / np.sqrt(len(per_seed)))
    ratio = sig / tmpl
    rec = dict(n=n, box=ell, f_nl=f, seeds=args.seeds, k_long_over_kf=[2, 3, 4],
               k_short_over_kf=8, amplitude=amp, amplitude_sem=sem,
               per_bin_ratio=ratio.mean(0).tolist(),
               per_bin_sem=(ratio.std(0, ddof=1) / np.sqrt(len(ratio))).tolist(),
               wall_s=time.perf_counter() - t0)
    write_json(out, rec)
    print(f"f_NL={f:g}, {args.seeds} seeds: amplitude {amp:.4f} +- {sem:.4f} "
          f"(per bin {np.round(rec['per_bin_ratio'], 3)} +- {np.round(rec['per_bin_sem'], 3)}), "
          f"{rec['wall_s']:.0f} s", flush=True)


if __name__ == "__main__":
    main()
