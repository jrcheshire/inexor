"""Who owns what across ranks and cards: the 1-D x decomposition by whole tile planes.

A rank (one process per node) owns a contiguous run of tile planes, and with them the brick
x-slabs and coarse x-planes underneath; its cards split that run again. The spectrum is split
by y-pencils, per rank and then per card. Every range is a `ooc_fft.partition_units` split, so
at one rank the card ranges are exactly the ones the single-node step has always used.

Node count is execution policy: nothing here enters a checkpoint fingerprint.
"""

from __future__ import annotations

from dataclasses import dataclass

from .ooc_fft import partition_units


@dataclass(frozen=True)
class Decomp:
    """The ownership map for rank `rank` of `n_ranks`, each rank driving `cards` cards.

    Build with `Decomp.build(cfg, ...)`. Plane, slab and pencil ranges are half-open global
    indices. `pad` is the buffer brick slabs a tile window reads past its core on each side
    (the ghost slabs a rank receives from each neighbour). Consumers: `engine.step` (tile
    planes per card, coarse shard ranges, per-rank membership, ghost slabs) and the resident
    coarse kernel (spectrum pencils per card).
    """

    n_ranks: int
    rank: int
    cards: int
    tiles_side: int
    bricks_per_tile: int
    coarse_per_tile: int
    n_coarse: int
    reach: int
    pencil_batch: int
    rank_planes: tuple
    rank_pencils: tuple
    pad: int = 0

    @classmethod
    def build(cls, cfg, n_ranks=1, rank=0, reach=1, pencil_batch=1):
        """From an `EngineConfig`'s geometry and `device_cards`.

        Refuses fewer tile planes than cards in total, and (across ranks) fewer than
        `2 * reach + 1` or `2 * pad` brick slabs per rank: the migrate hands particles to
        immediate neighbours only, and each neighbour's ghost slabs come from this rank alone.
        """
        from .layout import brick_span

        n_ranks, rank, cards = int(n_ranks), int(rank), int(cfg.device_cards)
        reach, pencil_batch = int(reach), int(pencil_batch)
        if n_ranks < 1:
            raise ValueError(f"n_ranks must be >= 1, got {n_ranks}")
        if not 0 <= rank < n_ranks:
            raise ValueError(f"rank {rank} is outside [0, {n_ranks})")
        s = int(cfg.tiles_side)
        if n_ranks * cards > s:
            raise ValueError(
                f"{n_ranks} rank(s) x {cards} card(s) = {n_ranks * cards} cards but only {s} "
                "tile planes: a card would run no tiles. Use fewer ranks or cards, or a "
                "geometry with more tile planes.")
        bpt = int(cfg.n_tile) // int(cfg.n_brick)
        pad = brick_span(int(cfg.n_tile), int(cfg._b_realized), int(cfg.n_brick),
                         int(cfg.n_fine) // int(cfg.n_brick))[0]
        rank_planes = tuple(partition_units(s, n_ranks, 1))
        if n_ranks > 1:
            thin = min(hi - lo for lo, hi in rank_planes) * bpt
            if thin < 2 * reach + 1:
                raise ValueError(
                    f"a rank would own {thin} brick slab(s), fewer than 2 * reach + 1 = "
                    f"{2 * reach + 1}: the migrate hands off to immediate neighbours only. "
                    "Use fewer ranks.")
            if thin < 2 * pad:
                raise ValueError(
                    f"a rank would own {thin} brick slab(s), fewer than 2 * {pad}: its "
                    f"neighbours' tile windows read {pad} slab(s) from each of its ends. "
                    "Use fewer ranks.")
        n_coarse = int(cfg.n_coarse)
        return cls(
            n_ranks=n_ranks, rank=rank, cards=cards, tiles_side=s, bricks_per_tile=bpt,
            coarse_per_tile=int(cfg.n_tile) // (int(cfg.n_fine) // n_coarse),
            n_coarse=n_coarse, reach=reach, pencil_batch=pencil_batch,
            rank_planes=rank_planes,
            rank_pencils=tuple(partition_units(n_coarse, n_ranks, pencil_batch)),
            pad=int(pad),
        )

    # ------------------------------------------------------------ this rank

    @property
    def planes(self):
        """This rank's tile planes [lo, hi)."""
        return self.rank_planes[self.rank]

    @property
    def slabs(self):
        """This rank's brick x-slabs [lo, hi)."""
        lo, hi = self.planes
        return lo * self.bricks_per_tile, hi * self.bricks_per_tile

    @property
    def coarse_planes(self):
        """This rank's coarse x-planes [lo, hi), halo excluded."""
        lo, hi = self.planes
        return lo * self.coarse_per_tile, hi * self.coarse_per_tile

    @property
    def pencils(self):
        """This rank's spectrum y-pencils [lo, hi)."""
        return self.rank_pencils[self.rank]

    @property
    def neighbours(self):
        """(left, right) ranks on the periodic x ring; both are this rank when alone."""
        return (self.rank - 1) % self.n_ranks, (self.rank + 1) % self.n_ranks

    def card_planes(self, rank=None):
        """Per card of `rank` (default this one), its tile planes [lo, hi)."""
        lo, hi = self.rank_planes[self.rank if rank is None else int(rank)]
        return [(lo + a, lo + b) for a, b in partition_units(hi - lo, self.cards, 1)]

    def coarse_shards(self, halo):
        """Per card, `(x0, nx)` of the coarse planes under its tile planes with `halo` planes
        each side: the ranges `device.coarse.CardShards` takes (x0 may be negative; mod n)."""
        p = self.coarse_per_tile
        return [(a * p - int(halo), (b - a) * p + 2 * int(halo)) for a, b in self.card_planes()]

    def card_pencils(self):
        """Per card, its spectrum y-pencils [lo, hi) within this rank's."""
        lo, hi = self.pencils
        return [(lo + a, lo + b)
                for a, b in partition_units(hi - lo, self.cards, self.pencil_batch)]

    def owner_of_plane(self, plane):
        """The rank owning tile plane `plane` (mod tiles_side)."""
        p = int(plane) % self.tiles_side
        for r, (lo, hi) in enumerate(self.rank_planes):
            if lo <= p < hi:
                return r
        raise AssertionError("rank_planes do not tile the tile planes")
