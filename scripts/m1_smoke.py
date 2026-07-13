"""M1 S7: forward smoke at scale on CUDA (deneb RTX 3050, 6 GB).

Attempts the M1-exit 512^3 forward run (2LPT ICs -> int16 BullFrog evolve,
perstep driver with donation + max|w| monitor); on GPU OOM falls back to the
PRE-AGREED 384^3 (documented, never silent -- plan tender-stargazing-map S7).
--roundtrip additionally runs the K-forward + K-reverse exact replay at the
smoke size (the product claim, at scale).

    XLA_PYTHON_CLIENT_MEM_FRACTION=0.95 pixi run -e gpu \
        python scripts/m1_smoke.py --roundtrip
"""

import argparse
import json
import os
import sys
import time as _time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import _m1_common as M  # noqa: E402

REPO = os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))


def attempt(n, K, lpt_order, do_roundtrip):
    import jax
    import jax.numpy as jnp

    from inexor import config, integrate

    cosmo = config.Cosmology(**M.COSMO)
    box = config.BoxConfig(n_mesh=n, box_size=M.BOX_SIZE)
    time = config.TimeConfig(
        a_init=M.A_INIT, a_final=M.A_FINAL, n_steps=K, spacing="log", integrator="bullfrog"
    )
    quant = config.QuantConfig()

    t0 = _time.time()
    outs = integrate.simulate(
        box,
        time,
        quant,
        cosmo,
        seed=0,
        lpt_order=lpt_order,
        driver="perstep",
        paint="int",
        monitor=True,
    )
    x_f, v_f, max_w = outs
    x_f.block_until_ready()
    t_fwd = _time.time() - t0
    max_w = np.asarray(max_w)
    rec = dict(
        n=n,
        K=K,
        lpt_order=lpt_order,
        t_forward_s=round(t_fwd, 2),
        max_abs_w_per_step=[int(v) for v in max_w],
        w_headroom_frac=float(np.max(np.abs(max_w)) / 32767.0),
        x_finite=bool(jnp.all(jnp.isfinite(x_f))),
        backend=jax.default_backend(),
        device=str(jax.devices()[0]),
    )
    print(
        f"forward {n}^3 K={K} lpt{lpt_order}: {t_fwd:.1f} s, "
        f"max|w| {int(np.max(np.abs(max_w)))} / 32767 "
        f"({rec['w_headroom_frac']:.2f} of range), finite={rec['x_finite']}"
    )

    if do_roundtrip:
        from inexor.ic import gaussian_delta
        from inexor.lpt import lpt_ics

        key = jax.random.PRNGKey(0)
        delta0 = gaussian_delta(key, n, M.BOX_SIZE, cosmo)
        x0, v0 = lpt_ics(delta0, M.BOX_SIZE, M.A_INIT, cosmo, order=lpt_order)
        del delta0
        t0 = _time.time()
        ok, n_diff = integrate.replay_roundtrip(
            box, time, quant, cosmo, x0, v0, driver="perstep", paint="int"
        )
        t_rt = _time.time() - t0
        rec["roundtrip_exact"] = bool(ok)
        rec["roundtrip_n_diff"] = int(n_diff)
        rec["t_roundtrip_s"] = round(t_rt, 2)
        print(f"roundtrip {n}^3 K={K}: exact={ok} (n_diff={n_diff}), {t_rt:.1f} s")
        if not ok:
            raise SystemExit(f"ROUNDTRIP FAILED at {n}^3: {n_diff} differing words")
    return rec


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--n", type=int, default=512)
    ap.add_argument("--fallback", type=int, default=384)
    ap.add_argument("--steps", type=int, default=10)
    ap.add_argument("--lpt", type=int, default=2)
    ap.add_argument("--roundtrip", action="store_true")
    args = ap.parse_args()

    try:
        rec = attempt(args.n, args.steps, args.lpt, args.roundtrip)
        rec["fallback_used"] = False
    except Exception as e:
        msg = str(e)
        if "RESOURCE_EXHAUSTED" not in msg and "Out of memory" not in msg.lower():
            raise
        print(
            f"{args.n}^3 OOM on this device -- falling back to the pre-agreed "
            f"{args.fallback}^3 (documented, not silent):\n  {msg.splitlines()[0]}"
        )
        rec = attempt(args.fallback, args.steps, args.lpt, args.roundtrip)
        rec["fallback_used"] = True
        rec["oom_at_n"] = args.n

    rec["meta"] = M.make_meta("inexor", "m1-smoke", dict(requested_n=args.n), REPO)
    path = os.path.join(M.RUNS, f"smoke_n{rec['n']}.json")
    os.makedirs(M.RUNS, exist_ok=True)
    with open(path, "w") as f:
        json.dump(rec, f, indent=1)
    print(f"wrote {path}")


if __name__ == "__main__":
    main()
