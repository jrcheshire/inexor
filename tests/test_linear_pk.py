"""A tabulated linear P(k): the table's checks, the IC table it feeds, and how it travels.

`LinearPkTable.from_record` refuses a table that would silently seed another run (format,
z, cosmology, sigma8, k coverage, an altered embedded copy). An EH98 spectrum fed back as a
table reproduces EH98's P, the derived T(k) and the ICs to the input table's interpolation
error (bounds below are a few times the values measured 2026-10-07); a table with more power
moves the ICs by exactly that. The generators embed the table in the manifest, `epoch_record`
carries it into checkpoints only when there is one, and the card's oracle reads it.
"""

import dataclasses
import json
import os

import jax
import numpy as np
import pytest

from inexor import engine, icgen, summary
from inexor.config import Cosmology
from inexor.cosmology import (
    K_TABLE_MAX,
    K_TABLE_MIN,
    LINEAR_PK_FORMAT,
    LinearPkTable,
    growth_factor_a,
    ic_k_table,
    linear_power,
    load_linear_pk,
    transfer_eh98,
)

COSMO = Cosmology()
N, L, NB, A_INIT, SLAB = 32, 32.0, 4, 0.1, 5


@pytest.fixture(autouse=True)
def _x64():
    prev = jax.config.jax_enable_x64
    jax.config.update("jax_enable_x64", True)
    yield
    jax.config.update("jax_enable_x64", prev)


def _eh98_table(scale=1.0, n=4000, lo=K_TABLE_MIN / 1.05, hi=K_TABLE_MAX * 1.05):
    k = np.geomspace(lo, hi, n)
    return LinearPkTable(k, scale * linear_power(k, COSMO), "eh98-sampled",
                         dict(z=0.0, cosmology=dataclasses.asdict(COSMO)))


def test_a_table_round_trips_through_its_record_and_file(tmp_path):
    t = _eh98_table()
    path = tmp_path / "pk.json"
    path.write_text(json.dumps(t.record()))
    back = load_linear_pk(str(path), COSMO)
    assert back.sha256 == t.sha256
    assert back.record() == t.record()
    assert back.stamp() == dict(source="eh98-sampled", sha256=t.sha256)
    assert t.record()["format"] == LINEAR_PK_FORMAT


def _mutate(rec, what):
    rec = dict(rec)
    if what == "format":
        rec["format"] = "inexor-linear-pk-0"
    elif what == "z":
        rec["z"] = 0.5
    elif what == "cosmology":
        rec["cosmology"] = dict(rec["cosmology"], h=0.7)
    elif what == "missing cosmology":
        rec.pop("cosmology")
    elif what == "sigma8":
        # 0.4% more power: sigma8 0.2% high, twice the refusal threshold
        rec["P"] = [1.004 * p for p in rec["P"]]
        rec.pop("sha256")
    elif what == "k short":
        rec["k"], rec["P"] = rec["k"][:-200], rec["P"][:-200]
        rec.pop("sha256")
    elif what == "k unsorted":
        rec["k"] = rec["k"][:10][::-1] + rec["k"][10:]
        rec.pop("sha256")
    elif what == "negative P":
        rec["P"] = [-rec["P"][0]] + rec["P"][1:]
        rec.pop("sha256")
    elif what == "altered":
        rec["P"] = rec["P"][:5] + [rec["P"][5] * (1 + 1e-9)] + rec["P"][6:]
    return rec


@pytest.mark.parametrize("what,says", [
    ("format", "format"),
    ("z", "z = 0.5"),
    ("cosmology", "cosmology differs from the run's in h"),
    ("missing cosmology", "cosmology differs from the run's in"),
    ("sigma8", "sigma8"),
    ("k short", "must cover"),
    ("k unsorted", "strictly increasing"),
    ("negative P", "positive P"),
    ("altered", "sha256 does not match"),
])
def test_a_table_that_would_seed_another_run_is_refused(what, says):
    with pytest.raises(ValueError, match=says):
        LinearPkTable.from_record(_mutate(_eh98_table().record(), what), COSMO)


def test_a_table_at_another_cosmology_is_refused_against_the_run():
    t = _eh98_table()
    with pytest.raises(ValueError, match="n_s"):
        LinearPkTable.from_record(t.record(), dataclasses.replace(COSMO, n_s=0.96))


def test_an_eh98_table_gives_eh98s_spectrum_and_transfer():
    a = ic_k_table(COSMO, N, L)
    b = ic_k_table(COSMO, N, L, backend="table", table=LinearPkTable.from_record(
        _eh98_table().record(), COSMO))
    # the input table's own log-log interpolation error: 1.7e-5 measured
    assert np.max(np.abs(b.P / a.P - 1)) < 5e-5
    # T is derived from P, unity at K_TABLE_MIN; EH98's T there is 0.99991, so compare the
    # shape: 8.7e-6 measured
    assert np.max(np.abs(b.T / (a.T / a.T[0]) - 1)) < 3e-5
    assert b.T[0] == 1.0
    assert a.T[0] == pytest.approx(transfer_eh98(np.array([K_TABLE_MIN]), COSMO)[0])


def _staged(tmp_path, name, f_NL=0.0, **kw):
    d = str(tmp_path / name)
    man = icgen.generate_t9_slabs(d, jax.random.PRNGKey(11), N, L, COSMO, A_INIT, NB,
                                  slab=SLAB, f_NL=f_NL, keep_stage=True, **kw)
    return man, np.load(os.path.join(d, icgen.STAGE_DIR, "delta.npy"))


@pytest.mark.parametrize("f_NL", [0.0, 100.0])
def test_eh98_as_a_table_makes_eh98s_ics(tmp_path, f_NL):
    t = LinearPkTable.from_record(_eh98_table().record(), COSMO)
    man_e, d_e = _staged(tmp_path, "eh98", f_NL)
    man_t, d_t = _staged(tmp_path, "table", f_NL, backend="table", table=t)
    rms = np.sqrt(np.mean((d_t - d_e) ** 2) / np.mean(d_e ** 2))
    # 8e-7 (f_NL 0) and 1.1e-6 (f_NL 100) measured
    assert rms < 5e-6
    assert "linear_pk" not in man_e
    assert LinearPkTable.from_record(man_t["linear_pk"], COSMO).sha256 == t.sha256


def test_the_table_is_what_colours_the_ics(tmp_path):
    # 2% more power is 1% more amplitude, mode by mode (the white noise is the same)
    _, d_1 = _staged(tmp_path, "one", backend="table", table=_eh98_table())
    _, d_2 = _staged(tmp_path, "more", backend="table", table=_eh98_table(scale=1.02))
    assert np.std(d_2) / np.std(d_1) == pytest.approx(np.sqrt(1.02), rel=1e-6)


def test_a_generator_refuses_a_bare_table(tmp_path):
    k = np.geomspace(K_TABLE_MIN, K_TABLE_MAX, 400)
    with pytest.raises(TypeError, match="LinearPkTable"):
        icgen.generate_t9_slabs(str(tmp_path), jax.random.PRNGKey(0), N, L, COSMO, A_INIT,
                                NB, backend="table", table=(k, linear_power(k, COSMO)))


def test_a_checkpoint_carries_the_table_only_when_there_is_one():
    a_steps = np.linspace(0.1, 1.0, 5)
    rec = _eh98_table().record()
    assert set(engine.epoch_record((a_steps, COSMO), 2)) == {"a", "cosmology"}
    assert set(engine.epoch_record((a_steps, COSMO, None), 2)) == {"a", "cosmology"}
    assert engine.epoch_record((a_steps, COSMO, rec), 2)["linear_pk"] == rec


class _Cfg:
    n_coarse, box_size = 32, 32.0


def test_the_card_oracle_is_the_tables():
    t = _eh98_table(scale=1.02)
    k = np.geomspace(0.2, 3.0, 7)
    p_e, *_ = summary._card_model(_Cfg, COSMO, 0.5)
    p_t, *_ = summary._card_model(_Cfg, COSMO, 0.5, t)
    assert np.allclose(p_t(k), growth_factor_a(0.5, COSMO) ** 2
                       * linear_power(k, COSMO, backend="table", table=t), rtol=1e-12)
    assert np.allclose(p_t(k) / p_e(k), 1.02, rtol=5e-5)
