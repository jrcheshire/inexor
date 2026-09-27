"""Coarse force meshes resident on the device, and a tile's sub-block gathered there.

A card holds a shard: a contiguous run of x-planes `x0 .. x0 + nx - 1` (global indices mod `n`)
covering the sub-blocks of the tiles it runs, halo included; `whole_mesh_shard` is the single-card
form (whole mesh plus `COARSE_HALO` planes wrapped onto each end). The tile program gathers its
sub-blocks from the shard, so the host never stages or uploads them.

The gather is one 3-D index gather (axis-by-axis slicing would materialize an (extent, n, n)
intermediate). x is shard-local and never wraps; y and z wrap mod `n` as the host staging does,
so the block is bitwise the host's. Containment is checked on the host (`check_covers`) because
an out-of-shard x index would be clamped by the gather, silently reading the wrong planes.
"""

from __future__ import annotations

import numpy as np


def shard_coarse_meshes(g_coarse, x0, nx, device=None):
    """Place x-planes `x0 .. x0 + nx - 1` (mod n) of each host coarse mesh on
    `device` (None: jax's default), as a shard dict: `meshes` (three (nx, n, n)
    device arrays), `x0`, `nx`, `n`. Copies; blocks until placed."""
    import jax
    import jax.numpy as jnp

    n = int(np.asarray(g_coarse[0]).shape[0])
    planes = (int(x0) + np.arange(int(nx), dtype=np.int64)) % n

    def place(g):
        a = np.take(np.asarray(g), planes, axis=0)  # a fresh buffer either way
        return jnp.array(a, copy=True) if device is None else jax.device_put(a, device)

    meshes = tuple(jax.block_until_ready(place(g)) for g in g_coarse)
    return dict(meshes=meshes, x0=int(x0), nx=int(nx), n=n)


def whole_mesh_shard(g_coarse, halo=None, device=None):
    """The whole mesh as one card's shard, `halo` planes wrapped onto each end."""
    from ..forces import COARSE_HALO

    h = COARSE_HALO if halo is None else int(halo)
    n = int(np.asarray(g_coarse[0]).shape[0])
    return shard_coarse_meshes(g_coarse, -h, n + 2 * h, device)


class CardShards:
    """Where the coarse solve writes its force meshes when they live on the cards.

    `ranges` is `(x0, nx, device)` per card, the planes that card holds (mod
    `n`), halo included. Passed as `out=` to `forces.coarse_force_meshes`, which
    then writes each component's planes straight onto the cards
    (`ooc_fft.inverse_to_card_shards`) and returns `assemble`'s shard dicts --
    the same dicts `shard_coarse_meshes` builds from host meshes, bitwise.
    """

    def __init__(self, ranges, n):
        self.ranges = [(int(x0), int(nx), dev) for x0, nx, dev in ranges]
        self.n = int(n)

    @classmethod
    def whole_mesh(cls, n, halo=None, device=None):
        """One card holding the whole mesh, `halo` planes wrapped onto each end:
        the card-side twin of `whole_mesh_shard`."""
        from ..forces import COARSE_HALO

        h = COARSE_HALO if halo is None else int(halo)
        return cls([(-h, int(n) + 2 * h, device)], n)

    def assemble(self, per_card):
        """Shard dicts from `per_card[k]` = card k's three meshes, in axis order."""
        return [dict(meshes=tuple(ms), x0=x0, nx=nx, n=self.n, device=dev)
                for (x0, nx, dev), ms in zip(self.ranges, per_card)]


def check_covers(shard, origin_cells, extent):
    """Refuse a sub-block whose x planes are not all in the shard."""
    lx = int(origin_cells[0]) - int(shard["x0"])
    if lx < 0 or lx + int(extent) > int(shard["nx"]):
        raise ValueError(
            f"sub-block x planes [{int(origin_cells[0])}, {int(origin_cells[0]) + int(extent)}) "
            f"are not inside the shard's [{shard['x0']}, {shard['x0'] + shard['nx']}); the gather "
            "would clamp and read the wrong planes silently")


def subblock_device(meshes, x0, n, origin_cells, extent):
    """The (extent,)*3 sub-block at `origin_cells` from each shard mesh. Pure jnp.

    `x0` and `origin_cells` may be traced; `n` and `extent` are static. The
    caller guarantees containment in x (`check_covers`).
    """
    import jax.numpy as jnp

    o = jnp.asarray(origin_cells, dtype=jnp.int64)
    r = jnp.arange(int(extent), dtype=jnp.int64)
    xi = o[0] - jnp.asarray(x0, dtype=jnp.int64) + r
    yi = (o[1] + r) % int(n)
    zi = (o[2] + r) % int(n)
    return tuple(m[xi[:, None, None], yi[None, :, None], zi[None, None, :]] for m in meshes)
