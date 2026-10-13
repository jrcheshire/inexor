"""Helpers shared by the validation scripts in this directory (not part of the package)."""

import json
import os
import resource
import sys

import numpy as np


def setup_jax(x64=True):
    """Default JAX to the CPU backend (unless JAX_PLATFORMS is set) and enable float64.

    Call before any JAX computation; the library never toggles `jax_enable_x64` itself.
    """
    import jax

    if not os.environ.get("JAX_PLATFORMS"):
        jax.config.update("jax_platforms", "cpu")
    if x64:
        jax.config.update("jax_enable_x64", True)
    return jax


def write_json(path, obj, indent=1):
    """Write `obj` as JSON to `path`, creating the parent directory."""
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, "w") as fh:
        json.dump(obj, fh, indent=indent)
    print(f"-> {path}", flush=True)


def maxrss(who="self"):
    """Peak resident set size in bytes of this process ("self") or its waited-for children."""
    w = resource.RUSAGE_SELF if who == "self" else resource.RUSAGE_CHILDREN
    rss = resource.getrusage(w).ru_maxrss
    return rss if sys.platform == "darwin" else rss * 1024   # bytes on macOS, KiB on Linux


def kgrid(n, box):
    """Angular wavenumbers of the rfftn half-grid of an n^3 mesh, as three broadcast arrays."""
    kx = 2 * np.pi * np.fft.fftfreq(n, d=box / n)
    kz = 2 * np.pi * np.fft.rfftfreq(n, d=box / n)
    return kx[:, None, None], kx[None, :, None], kz[None, None, :]


def tophat(x):
    """Fourier transform of the spherical top hat, 3 (sin x - x cos x) / x^3."""
    x = np.where(x == 0, 1e-12, x)
    return 3.0 * (np.sin(x) - x * np.cos(x)) / x ** 3


def jackknife_err(loo):
    """Leave-one-out jackknife standard error from the n leave-one-out estimates `loo`
    (first axis)."""
    loo = np.asarray(loo)
    n = loo.shape[0]
    return np.sqrt((n - 1.0) / n * np.sum((loo - loo.mean(0)) ** 2, axis=0))
