"""Frozen, hashable configuration dataclasses.

Hashability is load-bearing: Cosmology keys the lru_cache'd growth/amplitude
constants in cosmology.py, and BoxConfig keys per-box kernel caches in
forces.py. All classes are frozen dataclasses of plain floats/ints/strings.

House rules: no jax imports here; no arrays at import time.
"""

import math
from dataclasses import dataclass

U16_MOD = 2**16


@dataclass(frozen=True)
class Cosmology:
    """Flat-LCDM parameters (Planck-2018-flavoured defaults; mbody values)."""

    Omega_m: float = 0.31
    Omega_b: float = 0.049
    h: float = 0.677
    n_s: float = 0.965
    sigma8: float = 0.81
    T_cmb_K: float = 2.7255

    @property
    def Omega_Lambda(self):
        return 1.0 - self.Omega_m

    @property
    def Omega_cdm(self):
        return self.Omega_m - self.Omega_b

    @property
    def H0(self):
        """H0 in km/s/Mpc."""
        return 100.0 * self.h


PLANCK = Cosmology()


@dataclass(frozen=True)
class BoxConfig:
    """Periodic box + mesh + particle-count configuration.

    n_particles is the PER-DIMENSION count (mbody convention); the LPT/IC code
    path assumes one particle per Lagrangian cell (n_particles == n_mesh), which
    together with n_mesh | 2^16 makes every Lagrangian site an exact uint16
    lattice multiple.
    """

    n_mesh: int = 128
    box_size: float = 256.0  # Mpc/h
    n_particles: int | None = None  # None -> n_mesh (one particle per Lagrangian cell)

    def __post_init__(self):
        if self.n_particles is None:
            object.__setattr__(self, "n_particles", self.n_mesh)
        if self.n_mesh <= 0 or self.box_size <= 0.0:
            raise ValueError(f"n_mesh and box_size must be positive, got {self}")
        if U16_MOD % self.n_mesh != 0:
            raise ValueError(
                f"n_mesh = {self.n_mesh} must divide 2^16: the exact-Lagrangian-site "
                "identity (architecture.md Sec. 3) requires the mesh to be a sublattice "
                "of the uint16 position lattice."
            )

    @property
    def n_total(self):
        return self.n_particles**3

    @property
    def cell_size(self):
        return self.box_size / self.n_mesh

    @property
    def k_fundamental(self):
        return 2.0 * math.pi / self.box_size

    @property
    def k_nyquist(self):
        return math.pi * self.n_mesh / self.box_size

    @property
    def s_x(self):
        """Position lattice spacing: the uint16 lattice spans the box exactly."""
        return self.box_size / U16_MOD


_INTEGRATORS = ("bullfrog", "fastpm", "exact")
_SPACINGS = ("log", "linear")


