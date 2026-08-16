"""Exponential map framework for generating interpretable neural spatial representations.

A transparent \"no-box\" alternative to deep learning models for constructing
spatial representations from first principles. Generator matrices define how
physical space maps to neural activity via the matrix exponential, with
algebraic properties (commutativity, skew-symmetry, roots-of-unity eigenvalues)
analytically guaranteeing path integration, translational invariance, and
metric preservation.

Quick start
-----------
>>> import expmap
>>> model, p0 = expmap.grid_cell_model(M=3, n_copies=50)
>>> import torch
>>> positions = torch.rand(100, 2) * 10 - 5
>>> p = model(positions, p0)
"""

import importlib.metadata

import numpy as np

from expmap.generators import (
    bessel_module_frequencies,
    block_diag_skew,
    build_generators,
    build_generators_with_context,
    gaussian_eigenvalues,
    multiscale_wavevectors,
    roots_of_unity_eigenvalues,
)
from expmap.linalg import random_orthogonal_matrix, starting_state
from expmap.model import ExponentialMap, RemappingModel
from expmap.navigation import bind, bundle, query

__version__ = importlib.metadata.version("expmap")

__all__ = [
    "ExponentialMap",
    "RemappingModel",
    "grid_cell_model",
    "place_cell_model",
    "remapping_model",
    "roots_of_unity_eigenvalues",
    "gaussian_eigenvalues",
    "bessel_module_frequencies",
    "multiscale_wavevectors",
    "block_diag_skew",
    "build_generators",
    "build_generators_with_context",
    "random_orthogonal_matrix",
    "starting_state",
    "bundle",
    "bind",
    "query",
]


def grid_cell_model(M=3, n_copies=50, seed=None, device="cpu"):
    """Construct a metric-preserving grid cell model with M-fold symmetry.

    Creates generator matrices whose eigenvalues form roots of unity,
    analytically guaranteeing hexagonal (M=3) or square (M=2) grid patterns
    with exact path integration and metric preservation (Sec. 2.4).

    Parameters
    ----------
    M : int, optional
        Symmetry order. M=3 gives hexagonal grids, M=2 gives square grids.
    n_copies : int, optional
        Number of repeated eigenvalue modules. Total neurons = 2 * M * n_copies.
    seed : int, optional
        Random seed for the orthogonal basis matrix R.
    device : str, optional
        Torch device.

    Returns
    -------
    tuple[ExponentialMap, torch.Tensor]
        Configured model and canonical initial state p0.
    """
    # roots_of_unity_eigenvalues returns angles; resolve them on the unit ring
    # into the x and y eigenvalues via lambda_x = cos(phi), lambda_y = sin(phi).
    phi = roots_of_unity_eigenvalues(M, n_copies)
    sigma_x = block_diag_skew(np.cos(phi))
    sigma_y = block_diag_skew(np.sin(phi))

    R = random_orthogonal_matrix(len(sigma_x), seed=seed)
    p0 = starting_state(R)
    Gx, Gy = build_generators(sigma_x, sigma_y, R)

    return ExponentialMap(Gx, Gy, device=device), p0


def place_cell_model(N=300, scale=None, seed=None, device="cpu"):
    """Construct a place cell model with Gaussian eigenvalue distribution.

    Creates generator matrices with random Gaussian-distributed eigenvalues,
    producing representations whose pairwise similarity approximates a
    Gaussian function of spatial displacement (Appendix H, Fig. 7; the
    sampling derivation is Appendix I). The resulting neural tuning curves are
    localized place-like fields.

    Appendix H uses N = 256 with eigenvalues from N(0, 2), which is
    `place_cell_model(N=256)` at the default scale.

    Parameters
    ----------
    N : int, optional
        Number of neurons (must be even).
    scale : float, optional
        Standard deviation of the eigenvalue distribution, not the width of
        the similarity function — see `gaussian_eigenvalues`. If None, uses
        sqrt(2), giving C(delta) ~= exp(-delta**2).
    seed : int, optional
        Random seed for reproducibility.
    device : str, optional
        Torch device.

    Returns
    -------
    tuple[ExponentialMap, torch.Tensor]
        Configured model and canonical initial state p0.
    """
    if scale is None:
        scale = np.sqrt(2)

    ev_x = gaussian_eigenvalues(N, scale=scale, seed=seed)
    ev_y = gaussian_eigenvalues(N, scale=scale, seed=seed + 1 if seed is not None else None)

    sigma_x = block_diag_skew(ev_x)
    sigma_y = block_diag_skew(ev_y)

    R = random_orthogonal_matrix(N, seed=seed)
    p0 = starting_state(R)
    Gx, Gy = build_generators(sigma_x, sigma_y, R)

    return ExponentialMap(Gx, Gy, device=device), p0


def remapping_model(M=3, n_copies=50, context_scale=None, seed=None, device="cpu"):
    """Construct a remapping model with spatial and context generators.

    Combines metric-preserving spatial generators (roots of unity) with a
    Gaussian-distributed context generator for modeling context-dependent
    remapping. Different context values produce orthogonal spatial maps
    within the same physical environment (Sec. 2.5).

    Parameters
    ----------
    M : int, optional
        Symmetry order for spatial generators.
    n_copies : int, optional
        Number of repeated eigenvalue modules.
    context_scale : float, optional
        Standard deviation for context generator eigenvalues.
        If None, uses sqrt(2).
    seed : int, optional
        Random seed for reproducibility.
    device : str, optional
        Torch device.

    Returns
    -------
    tuple[RemappingModel, torch.Tensor]
        Configured remapping model and canonical initial state p0.
    """
    if context_scale is None:
        context_scale = np.sqrt(2)

    N = 2 * M * n_copies

    # Spatial generators: roots of unity, angles resolved onto the unit ring
    phi = roots_of_unity_eigenvalues(M, n_copies)
    sigma_x = block_diag_skew(np.cos(phi))
    sigma_y = block_diag_skew(np.sin(phi))

    # Context generator: Gaussian eigenvalues
    ev_s = gaussian_eigenvalues(N, scale=context_scale, seed=seed)
    sigma_s = block_diag_skew(ev_s)

    R = random_orthogonal_matrix(N, seed=seed)
    p0 = starting_state(R)
    Gx, Gy, Gs = build_generators_with_context(sigma_x, sigma_y, sigma_s, R)

    return RemappingModel(Gx, Gy, Gs, device=device), p0
