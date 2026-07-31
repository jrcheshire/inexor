"""Cross-implementation control: diagnostics.bispectrum vs DISCO-DJ.

SKIPPED unless the artifacts from scripts/_bispectrum_xcheck.py are on disk,
because discodj lives in the disco-mocks pixi env and cannot be imported here.
Regenerate with the three legs in that script's docstring.

This is a CONTROL, not an oracle. discodj ships no tests of its own, so what it
buys is independence, not authority: the deterministic tests in
test_bispectrum.py (plane-wave closed form, brute-force triangle count) are what
pin this side. Agreement between two independently written implementations of
the same convention is evidence neither one's internal tests can provide;
disagreement would be a question about both.

Measured 2026-07-30, N=32, L=256 Mpc/h, f64 both sides:
  own floor vs the plane-wave closed form: inexor 1.4e-15, discodj 1.2e-14
  agreement on B: 5.1e-15 (gaussian 5-4-3), 2.5e-14 (gaussian 8-8-2),
                  1.0e-14 (plane-wave 5-4-3)
The plane-wave 8-8-2 entry is excluded: the plane-wave field has no power in
those shells, so both sides return ~-1.8e-39 and the RATIO of two numerically
zero values carries no information (it read 1.5e-13, which is round-off
divided by round-off, not a disagreement).
"""

import json
import os

import numpy as np
import pytest

XCHECK_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "runs", "v2", "xcheck"
)
SPEC = os.path.join(XCHECK_DIR, "xcheck_spec.json")
DISCO = os.path.join(XCHECK_DIR, "xcheck_discodj.json")

pytestmark = pytest.mark.skipif(
    not (os.path.exists(SPEC) and os.path.exists(DISCO)),
    reason="run scripts/_bispectrum_xcheck.py (emit / discodj / compare) to generate artifacts",
)

# B below this magnitude is a shell with no power: the field simply has nothing
# in that configuration, so a relative comparison is round-off over round-off.
ZERO_B = 1e-30


def _load():
    return json.load(open(SPEC)), json.load(open(DISCO))


def test_reference_own_floor_is_measured_first():
    """discodj's own error against the closed form, before it is used as a control.

    A reference whose accuracy is unknown turns any comparison into a bound
    rather than a check, so this runs first and asserts the reference is
    actually good enough to be worth comparing against.
    """
    spec, dj = _load()
    ell = spec["box_size"]
    for i, t in enumerate(spec["tri_bins"]):
        if tuple(t) != (4, 3, 2):  # centres (5,4,3) k_f: the closed 3-4-5 triangle
            continue
        exact = ell**6 * 0.5**3 / (4.0 * spec["inexor"]["plane_wave"]["n_tri"][i])
        ours = abs(spec["inexor"]["plane_wave"]["B"][i] / exact - 1.0)
        theirs = abs(dj["plane_wave"]["B"][i] / exact - 1.0)
        assert ours < 1e-13, f"our own floor moved: {ours:.3e}"
        assert theirs < 1e-12, f"discodj floor {theirs:.3e} too poor to control against"
        return
    pytest.fail("the 3-4-5 triangle is missing from the artifacts")


def test_agrees_with_discodj():
    """Independent implementation, same convention, same field."""
    spec, dj = _load()
    compared = 0
    for name in ("plane_wave", "gaussian"):
        for i, t in enumerate(spec["tri_bins"]):
            ours = spec["inexor"][name]["B"][i]
            theirs = dj[name][i] if isinstance(dj[name], list) else dj[name]["B"][i]
            if abs(ours) < ZERO_B:
                continue  # no power in these shells; see the module docstring
            rel = abs(theirs / ours - 1.0)
            assert rel < 1e-12, f"{name} tri {t}: inexor {ours:.8e} vs discodj {theirs:.8e}"
            compared += 1
    # An empty loop would pass silently, which for a cross-check is the whole
    # failure mode: assert it actually compared something.
    assert compared >= 3, f"only {compared} triangles compared, expected >= 3"


def test_n_tri_agrees_with_discodj():
    """The triangle counts, which are field-independent geometry.

    Separated from B because a shared count error and a shared normalization
    error would otherwise be indistinguishable in the B comparison alone.
    """
    spec, dj = _load()
    ours = np.asarray(spec["inexor"]["gaussian"]["n_tri"], dtype=np.float64)
    theirs = np.asarray(dj["gaussian"]["n_tri"], dtype=np.float64)
    assert np.all(ours > 0)
    assert np.allclose(theirs, ours, rtol=1e-10), f"n_tri {ours} vs {theirs}"
