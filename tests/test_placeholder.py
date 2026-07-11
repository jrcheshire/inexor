"""Placeholder test so the suite runs green pre-implementation (pytest exits
nonzero on zero collected tests)."""

import inexor


def test_import():
    assert inexor.__version__.startswith("0.")
    assert inexor.__author__ == "James Cheshire"
