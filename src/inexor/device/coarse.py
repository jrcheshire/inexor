"""D2e: the coarse force meshes resident on the device, and a tile's sub-block gathered there.

WHAT THIS REPLACES. With the coarse meshes on the host, every tile stages its
three sub-blocks (`forces.stage_coarse_subblock`, ~23 ms per tile at P=576, flat
in mesh size: Vista 993849) and uploads them. With the meshes on the card, the
tile program gathers the blocks itself from a resident shard, and neither the
staging nor the upload exists.

THE SHARD. The design decomposes the coarse mesh along x across the cards, so a
card holds a contiguous run of x-planes, `x0 .. x0 + nx - 1` (global indices,
taken modulo `n`), wide enough to contain the sub-blocks of the tiles it runs,
halo included. `whole_mesh_shard` is the single-card development form: the
whole mesh plus `COARSE_HALO` planes wrapped onto each end.

THE GATHER IS ONE 3-D INDEX GATHER, for the reason `stage_coarse_subblock` is:
slicing axis by axis materializes an (extent, n, n) intermediate, 2.21 GB per
block at 4096^3, before the later axes narrow it. The x index is local to the
shard and never wraps -- the shard carries the planes a block needs -- and y and
z wrap modulo `n` exactly as the host staging does. Integer indexing of the same
values, so the block is bitwise the host's.

CONTAINMENT IS CHECKED ON THE HOST (`check_covers`): a tile's origin is a host
integer, and an x index past the shard would be clamped by the gather rather
than refused, silently reading the wrong planes.
"""

from __future__ import annotations

import numpy as np


def shard_coarse_meshes(g_coarse, x0, nx):
    """Place x-planes `x0 .. x0 + nx - 1` (mod n) of each host coarse mesh on the
    device, as a shard dict: `meshes` (three (nx, n, n) device arrays), `x0`,
    `nx`, `n`. Copies; blocks until placed."""
    import jax
    import jax.numpy as jnp

    n = int(np.asarray(g_coarse[0]).shape[0])
    planes = (int(x0) + np.arange(int(nx), dtype=np.int64)) % n
    meshes = tuple(jax.block_until_ready(jnp.array(np.take(np.asarray(g), planes, axis=0),
                                                   copy=True))
                   for g in g_coarse)
    return dict(meshes=meshes, x0=int(x0), nx=int(nx), n=n)


def whole_mesh_shard(g_coarse, halo=None):
    """The whole mesh as one card's shard, `halo` planes wrapped onto each end."""
    from ..forces import COARSE_HALO

    h = COARSE_HALO if halo is None else int(halo)
    n = int(np.asarray(g_coarse[0]).shape[0])
    return shard_coarse_meshes(g_coarse, -h, n + 2 * h)


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
