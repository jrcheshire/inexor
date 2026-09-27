"""config.py: frozen/hashable dataclasses, derived properties, loud validation."""

import dataclasses
import math

import pytest

from inexor.config import PLANCK, BoxConfig, Cosmology


def test_cosmology_hashable_and_frozen():
    c = Cosmology()
    assert hash(c) == hash(Cosmology())  # lru_cache key stability
    assert {c: 1}[Cosmology()] == 1
    with pytest.raises(dataclasses.FrozenInstanceError):
        c.Omega_m = 0.3


def test_cosmology_derived():
    c = PLANCK
    assert c.Omega_Lambda == pytest.approx(1.0 - c.Omega_m)
    assert c.Omega_cdm == pytest.approx(c.Omega_m - c.Omega_b)
    assert c.H0 == pytest.approx(67.7)


def test_box_defaults_and_derived():
    b = BoxConfig(n_mesh=128, box_size=256.0)
    assert b.n_particles == 128  # one particle per Lagrangian cell default
    assert b.n_total == 128**3
    assert b.cell_size == pytest.approx(2.0)
    assert b.k_fundamental == pytest.approx(2.0 * math.pi / 256.0)
    assert b.k_nyquist == pytest.approx(math.pi * 128 / 256.0)
    assert b.s_x == pytest.approx(256.0 / 2**16)
    assert hash(b) == hash(BoxConfig(n_mesh=128, box_size=256.0))


def test_box_rejects_non_sublattice_mesh():
    # 2^16 % 96 != 0: Lagrangian sites would not be exact uint16 lattice points.
    with pytest.raises(ValueError, match="divide 2\\^16"):
        BoxConfig(n_mesh=96, box_size=256.0)


def test_box_rejects_nonpositive():
    with pytest.raises(ValueError):
        BoxConfig(n_mesh=0, box_size=256.0)
    with pytest.raises(ValueError):
        BoxConfig(n_mesh=64, box_size=-1.0)


