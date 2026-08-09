"""M-v2-2 exit gate: promoted force == probe force, BITWISE, at the operating point.

D-v2-16 clause 7 gates promotion out of `scripts/v2_g5_core.py` on bitwise parity
against that probe kept UNMODIFIED, at three geometries. Two run on CPU in the
unit suite (`tests/test_two_level_force.py`): the tile identity, and a clustered
padding-dominated field. This is the third and the only one that needs a node --
**cgh64 at T=256 / b=32 / P=320, 64 tiles**, the configuration every D-v2-10,
D-v2-11 and D-v2-12 number was measured at.

WHY IT IS NOT A TOLERANCE. Those three ADRs are measurements OF the probe. If the
promoted engine is merely close, they quietly stop describing the shipped
artifact. So equality is exact and the probe is imported, not reimplemented.

## The determinism precondition, and why this job would otherwise be a lie

`tile_paint_f64` accumulates through `.at[].add` on an f64 mesh. On CUDA that is
an atomic scatter whose ORDER is not reproducible, so two runs of identical code
on identical input differ at roundoff (measured in M2 S4: `paint_f32` 156/4096
elements over 8 identical calls). A bitwise comparison of two arms under those
conditions is not a test of the promotion -- it is a coin flip that will
sometimes pass.

`XLA_FLAGS=--xla_gpu_deterministic_ops=true` removes it. But an env var is a
self-report, and a self-reported knob lies: this project has already recorded a
GPU/CPU pair whose log named no device and a run whose config field claimed a
setting it never applied. So leg 0 DEMONSTRATES determinism -- it paints the same
tile twice and requires bit equality -- and ABORTS if it does not hold. A green
parity result below is only readable because leg 0 passed above it.

Run:
    python scripts/v2_m2_parity_cgh64.py --config cdev8      # smoke
    python scripts/v2_m2_parity_cgh64.py --config cgh64 --tile 256 --buf 32
"""

import argparse
import json
import os
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

CONFIGS = {
    "smoke": dict(n_part=32, L=32.0, n_fine=64, n_coarse=16),
    "cdev8": dict(n_part=128, L=64.0, n_fine=256, n_coarse=64),
    "cdev": dict(n_part=256, L=128.0, n_fine=512, n_coarse=128),
    "cgh64": dict(n_part=512, L=256.0, n_fine=1024, n_coarse=256),
}
ALPHA = 1.0  # r_s / coarse_cell, the ratified kernel
A_INIT = 0.1
SEED = 0


def geometry(cfg):
    c = dict(CONFIGS[cfg])
    c["fine_cell"] = c["L"] / c["n_fine"]
    c["coarse_cell"] = c["L"] / c["n_coarse"]
    c["n_total"] = c["n_part"] ** 3
    return c


def make_positions(g, seed):
    """2LPT ICs at a_init -- the ratified trajectory's own starting field, so the
    parity is checked on a realistic clustered configuration rather than on
    uniform noise where every stencil sees the same thing."""
    import jax
    import jax.numpy as jnp

    from inexor.config import Cosmology
    from inexor.ic import linear_density
    from inexor.lpt import lpt_ics

    cosmo = Cosmology()
    delta0 = linear_density(
        jax.random.PRNGKey(seed), g["n_part"], g["L"], cosmo, fdtype=jnp.float64
    )
    # same call shape as v2_m1_migration.make_ics, so the field is the ratified
    # driver's own rather than a lookalike
    x, _ = lpt_ics(delta0, g["L"], A_INIT, cosmo, order=2, fdtype=jnp.float64)
    return np.asarray(np.mod(x, g["L"]), dtype=np.float64)


def assert_deterministic_paint(pos, g, tile, buf):
    """Leg 0. Paint one tile TWICE and require bit equality.

    If this fails on a GPU the deterministic-ops flag is not in force and every
    number below is uninterpretable, so it aborts rather than reporting a
    difference that would be read as a promotion defect.
    """
    import jax
    import jax.numpy as jnp

    from inexor.forces import padded_size, tile_origin_extent, tile_paint_f64

    P, b_real = padded_size(tile, buf, n_fine=g["n_fine"])
    cell = g["fine_cell"]
    origin, _ = tile_origin_extent((0, 0, 0), tile, b_real, cell)
    take = min(len(pos), 4_000_000)
    u = jnp.asarray(np.mod(pos[:take] - origin, g["L"]))
    live = jnp.asarray(np.ones((take,), dtype=bool))
    mean = g["n_total"] / float(g["n_fine"]) ** 3
    a = np.asarray(tile_paint_f64(u, live, (P,) * 3, cell, mean)[0])
    b = np.asarray(tile_paint_f64(u, live, (P,) * 3, cell, mean)[0])
    n_diff = int(np.count_nonzero(a != b))
    dev = jax.devices()[0].platform
    print(f"  leg 0: same tile painted twice on {dev} -> {n_diff} differing cells")
    if n_diff:
        raise SystemExit(
            f"FATAL: the f64 tile paint is NON-DETERMINISTIC here ({n_diff} cells differ "
            "between two identical calls). XLA_FLAGS=--xla_gpu_deterministic_ops=true is "
            "not in force, so a bitwise parity result would be a coin flip. Fix the flag; "
            "do NOT read the comparison below."
        )
    assert float(np.max(np.abs(a))) > 0.0, "leg 0 painted an empty mesh; it proved nothing"
    return dict(platform=dev, n_diff=n_diff, painted_rows=int(take))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="cdev8", choices=sorted(CONFIGS))
    ap.add_argument("--tile", type=int, default=None)
    ap.add_argument("--buf", type=int, default=32)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    import jax

    jax.config.update("jax_enable_x64", True)

    import v2_g5_core as probe

    from inexor import forces

    g = geometry(args.config)
    tile = args.tile if args.tile is not None else g["n_fine"] // 4
    buf = args.buf
    r_s = ALPHA * g["coarse_cell"]
    P, b_real = forces.padded_size(tile, buf, n_fine=g["n_fine"])
    n_side = g["n_fine"] // tile

    print(f"=== M-v2-2 parity [{args.config}] ===")
    print(f"  {g['n_total']:,} particles, n_fine {g['n_fine']}, T={tile} b={buf} "
          f"-> P={P} (b_realized {b_real}), {n_side**3} tiles")
    print(f"  XLA_FLAGS={os.environ.get('XLA_FLAGS', '<unset>')}")
    print(f"  jax {jax.__version__}  devices {jax.devices()}")

    t0 = time.perf_counter()
    pos = make_positions(g, SEED)
    print(f"  ICs: {time.perf_counter() - t0:.1f} s, "
          f"x in [{pos.min():.3f}, {pos.max():.3f}]")

    det = assert_deterministic_paint(pos, g, tile, buf)

    # ONE membership, used by both arms. The layout's tile_members returns the
    # same SET as the probe's but not necessarily the same ORDER, and order
    # changes the f64 scatter-add sequence hence the bits -- so driving both from
    # the probe's own bucketing is what makes this a test of the FORCE rather
    # than of two bucketings.
    cell = g["fine_cell"]
    n_brick = probe.choose_brick(tile, b_real, g["n_fine"])
    order, starts, nb = probe.brick_buckets(pos, g["n_fine"], n_brick, cell)
    tiles = [(i, j, k) for i in range(n_side) for j in range(n_side) for k in range(n_side)]
    cap, pad_frac = probe.tile_capacity(order, starts, nb, tiles, tile, b_real, n_brick)
    print(f"  brick {n_brick}, cap {cap:,}, pad_frac {pad_frac:.3f}")

    def member_fn(tijk):
        return probe.tile_members(order, starts, nb, tijk, tile, b_real, n_brick)

    t0 = time.perf_counter()
    theirs, pdiag = probe.force_short_tiled(
        pos, g["n_fine"], g["L"], g["n_total"], tile, buf, r_s=r_s, family="gauss"
    )
    t_probe = time.perf_counter() - t0
    print(f"  probe arm:    {t_probe:8.1f} s")

    t0 = time.perf_counter()
    mine, mdiag = forces.force_short_tiled(
        pos, g["n_fine"], g["L"], g["n_total"], tile, buf, member_fn, cap,
        r_s=r_s, max_accumulate_bytes=64 * 1024**3,
    )
    t_mine = time.perf_counter() - t0
    print(f"  promoted arm: {t_mine:8.1f} s")

    n_diff = int(np.count_nonzero(mine != theirs))
    peak = float(np.max(np.abs(theirs)))
    max_delta = float(np.max(np.abs(mine - theirs)))
    nonzero = int(np.count_nonzero(theirs))

    # Anti-vacuity, same discipline as the unit suite: an all-zero or degenerate
    # oracle makes equality meaningless. r_s=None would zero the short kernel
    # entirely, which is exactly how the first V4 parity check passed on nothing.
    vacuous = peak <= 1e-12 or nonzero < theirs.size // 2
    ok = (n_diff == 0) and not vacuous

    print("\n--- verdict ---")
    print(f"  oracle peak |g|      {peak:.6e}   nonzero {nonzero}/{theirs.size}")
    print(f"  differing elements   {n_diff} / {mine.size}")
    print(f"  max |delta|          {max_delta:.3e}")
    print(f"  partition_ok         mine={mdiag['partition_ok']} probe={pdiag['partition_ok']}")
    print(f"  overhang             mine={mdiag['n_overhang_total']} "
          f"probe={pdiag['n_overhang_total']}")
    print(f"  padded_P             mine={mdiag['padded_P']} probe={pdiag['padded_P']}")
    if vacuous:
        print("  !! VACUOUS: the oracle carries no signal; equality here proves nothing")
    print(f"  -> {'PASS' if ok else 'FAIL'}")

    card = dict(
        config=args.config, geometry=g, tile=tile, buf=buf, b_realized=b_real,
        padded_P=P, n_tiles=len(tiles), n_brick=n_brick, cap=int(cap),
        pad_frac=float(pad_frac), r_s=float(r_s),
        determinism=det,
        xla_flags=os.environ.get("XLA_FLAGS"),
        jax_version=jax.__version__, devices=[str(d) for d in jax.devices()],
        n_diff=n_diff, max_delta=max_delta, oracle_peak=peak,
        oracle_nonzero=nonzero, n_elements=int(mine.size),
        vacuous=bool(vacuous), all_ok=bool(ok),
        wall_probe_s=t_probe, wall_promoted_s=t_mine,
        diag_mine=mdiag, diag_probe=pdiag,
    )
    out = args.out or os.path.join(
        os.path.dirname(HERE), "runs", "v2", f"m2_parity_{args.config}.json"
    )
    os.makedirs(os.path.dirname(out), exist_ok=True)
    with open(out, "w") as fh:
        json.dump(card, fh, indent=2, default=str)
    print(f"\nwrote {out}")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
