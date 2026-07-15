"""Differentiable jnp summary losses for the M2 adjoint gates and money plots.

diagnostics.py's estimators are host/numpy (they do not backprop). These are the
jnp twins used as `loss_field(x_f, v_f)` for the adjoint_grad_* wrappers and the
S3 fidelity gate. Painting uses the differentiable paint="f32" density (the VJP
path; D-006) -- never the deterministic int paint (which has no VJP rule).

Binning is host-static (|k| grid + bin assignment are numpy), so the loss is a
clean function of the traced field: one segment_sum over static segment ids.
"""

import jax
import jax.numpy as jnp
import numpy as np

from .painting import density_contrast


def density_f32(x_f, box):
    """Differentiable density contrast from particle positions (paint='f32').

    The int32-free f32 paint materializes an f32 mesh; cast back to the input
    dtype so the loss stays dtype-consistent with the caller (mirrors
    forces._cached_force_fn, which casts the force to fdtype). Under x64 this
    keeps cotangents f64 through the adjoint's reverse sweep."""
    d = density_contrast(x_f, box.n_mesh, box.box_size, box.n_total, paint="f32")
    return d.astype(x_f.dtype)


def _k_mag(n_mesh, box_size):
    """|k| on the rfftn half-grid (host float64), h/Mpc."""
    N, L = n_mesh, box_size
    kx = 2.0 * np.pi * np.fft.fftfreq(N, d=L / N)
    kz = 2.0 * np.pi * np.fft.rfftfreq(N, d=L / N)
    return np.sqrt(kx.reshape(N, 1, 1) ** 2 + kx.reshape(1, N, 1) ** 2 + kz.reshape(1, 1, -1) ** 2)


def fundamental_k_edges(n_mesh, box_size, n_bins=8, k_min=None, k_max=None):
    """Linear k-bin edges from ~1 fundamental to k_max (default 0.6 k_Nyquist)."""
    k_f = 2.0 * np.pi / box_size
    k_nyq = np.pi * n_mesh / box_size
    lo = 0.5 * k_f if k_min is None else k_min
    hi = 0.6 * k_nyq if k_max is None else k_max
    return np.linspace(lo, hi, n_bins + 1)


def power_spectrum(delta, box_size, k_edges):
    """Shell-averaged P(k) of a real field (differentiable in delta).

    Returns (n_bins,) jnp: mean |delta_k|^2 * (L^3 / N^6) per k shell, with
    empty shells returning 0. k_edges is host-static.
    """
    N = delta.shape[0]
    dk = jnp.fft.rfftn(delta)
    p3d = ((dk * jnp.conj(dk)).real * (box_size**3 / N**6)).ravel()
    kmag = _k_mag(N, box_size).ravel()
    n_bins = len(k_edges) - 1
    seg = np.clip(np.digitize(kmag, k_edges) - 1, 0, n_bins - 1)
    in_range = (kmag >= k_edges[0]) & (kmag < k_edges[-1])
    seg = np.where(in_range, seg, n_bins)  # dump out-of-range into a scratch bin
    seg_j = jnp.asarray(seg)
    counts = np.bincount(seg, minlength=n_bins + 1)[:n_bins]
    psum = jax.ops.segment_sum(p3d, seg_j, num_segments=n_bins + 1)[:n_bins]
    denom = jnp.asarray(np.maximum(counts, 1).astype(p3d.dtype))
    return psum / denom


def band_power_loss(x_f, box, k_edges, target_pk):
    """MSE of the field's binned P(k) against a target (differentiable in x_f)."""
    P = power_spectrum(density_f32(x_f, box), box.box_size, k_edges)
    return jnp.mean((P - jnp.asarray(target_pk, dtype=P.dtype)) ** 2)


def field_l2_loss(x_f, box, target_delta):
    """Mean-square density-field residual against a target field."""
    delta = density_f32(x_f, box)
    return jnp.mean((delta - jnp.asarray(target_delta, dtype=delta.dtype)) ** 2)
