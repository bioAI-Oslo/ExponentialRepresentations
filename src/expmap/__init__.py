import importlib.metadata

import numpy as np
import torch

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
    "bessel_module_frequencies",
    "bind",
    "block_diag_skew",
    "build_generators",
    "build_generators_with_context",
    "bundle",
    "gaussian_eigenvalues",
    "grid_cell_model",
    "multiscale_wavevectors",
    "place_cell_model",
    "query",
    "random_orthogonal_matrix",
    "remapping_model",
    "roots_of_unity_eigenvalues",
    "starting_state",
]


def grid_cell_model(
    M: int = 3, n_copies: int = 50, seed: int | None = None, device: str | torch.device = "cpu"
) -> tuple[ExponentialMap, torch.Tensor]:
    r"""Construct a metric-preserving grid cell model with :math:`M`-fold symmetry.

    Creates generator matrices whose eigenvalues form roots of unity, which
    gives hexagonal (:math:`M = 3`) or square (:math:`M = 2`) grid patterns with
    exact path integration and metric preservation (Sec. 2.4).

    Parameters
    ----------
    M : int, optional
        Symmetry order. :math:`M = 3` gives hexagonal grids, :math:`M = 2` square grids.
    n_copies : int, optional
        Number of repeated eigenvalue modules. The number of neurons is
        2 * M * n_copies.
    seed : int, optional
        Random seed for the orthogonal mixing matrix :math:`R`.
    device : str or torch.device, optional
        Device to place the generators on. The origin state is returned on
        the CPU; ``forward`` moves it to the device of the generators.

    Returns
    -------
    tuple[ExponentialMap, torch.Tensor]
        Configured model and canonical origin state :math:`\mathbf{p}_0`.
    """
    # roots_of_unity_eigenvalues returns angles, resolved here on the unit ring
    # into lambda_x = cos(phi) and lambda_y = sin(phi).
    phi = roots_of_unity_eigenvalues(M, n_copies)
    sigma_x = block_diag_skew(np.cos(phi))
    sigma_y = block_diag_skew(np.sin(phi))

    R = random_orthogonal_matrix(len(sigma_x), seed=seed)
    p0 = starting_state(R)
    Gx, Gy = build_generators(sigma_x, sigma_y, R)

    return ExponentialMap(Gx, Gy, device=device), p0


def place_cell_model(
    N: int = 300, scale: float | None = None, seed: int | None = None, device: str | torch.device = "cpu"
) -> tuple[ExponentialMap, torch.Tensor]:
    r"""Construct a model with place-like fields from Gaussian eigenvalues.

    Creates generator matrices with random Gaussian-distributed eigenvalues.
    The pairwise similarity of the representation then approximates a Gaussian
    function of spatial displacement (Appendix H, Fig. 7, with the sampling
    derivation in Appendix I), and the tuning curves are localized place-like
    fields.

    Appendix H uses :math:`N = 256` with eigenvalues from :math:`\mathcal{N}(0, 2)`,
    which is ``place_cell_model(N=256)`` at the default scale.

    Parameters
    ----------
    N : int, optional
        Number of neurons (must be even).
    scale : float, optional
        Standard deviation of the eigenvalue distribution, not the width of
        the similarity function, see ``gaussian_eigenvalues``. If None, uses
        :math:`\sqrt{2}`, giving :math:`C(\Delta) \approx e^{-\Delta^2}`.
    seed : int, optional
        Random seed for reproducibility.
    device : str or torch.device, optional
        Device to place the generators on. The origin state is returned on
        the CPU; ``forward`` moves it to the device of the generators.

    Returns
    -------
    tuple[ExponentialMap, torch.Tensor]
        Configured model and canonical origin state :math:`\mathbf{p}_0`.
    """
    if scale is None:
        scale = float(np.sqrt(2))

    ev_x = gaussian_eigenvalues(N, scale=scale, seed=seed)
    ev_y = gaussian_eigenvalues(N, scale=scale, seed=seed + 1 if seed is not None else None)

    sigma_x = block_diag_skew(ev_x)
    sigma_y = block_diag_skew(ev_y)

    R = random_orthogonal_matrix(N, seed=seed)
    p0 = starting_state(R)
    Gx, Gy = build_generators(sigma_x, sigma_y, R)

    return ExponentialMap(Gx, Gy, device=device), p0


def remapping_model(
    M: int = 3,
    n_copies: int = 50,
    context_scale: float | None = None,
    seed: int | None = None,
    device: str | torch.device = "cpu",
) -> tuple[RemappingModel, torch.Tensor]:
    r"""Construct a remapping model with spatial and context generators.

    Combines metric-preserving spatial generators (roots of unity) with a
    Gaussian-distributed context generator for modeling context-dependent
    remapping. Different context values produce orthogonal spatial maps
    within the same physical environment (Sec. 2.5).

    The context eigenvalues are drawn per subspace, so the copies of a
    wavevector get different ones and the pairwise phase relations between
    units are not preserved across contexts (App. M). The Fig. 3 construction,
    with a distinct orientation per hexagonal set and a block-diagonal :math:`R`, is
    built from the composable pieces in the ``remapping`` notebook.

    Parameters
    ----------
    M : int, optional
        Symmetry order for spatial generators.
    n_copies : int, optional
        Number of repeated eigenvalue modules.
    context_scale : float, optional
        Standard deviation for context generator eigenvalues.
        If None, uses :math:`\sqrt{2}`.
    seed : int, optional
        Random seed for reproducibility.
    device : str or torch.device, optional
        Device to place the generators on. The origin state is returned on
        the CPU; ``forward`` moves it to the device of the generators.

    Returns
    -------
    tuple[RemappingModel, torch.Tensor]
        Configured remapping model and canonical origin state :math:`\mathbf{p}_0`.
    """
    if context_scale is None:
        context_scale = float(np.sqrt(2))

    N = 2 * M * n_copies

    phi = roots_of_unity_eigenvalues(M, n_copies)
    sigma_x = block_diag_skew(np.cos(phi))
    sigma_y = block_diag_skew(np.sin(phi))

    ev_s = gaussian_eigenvalues(N, scale=context_scale, seed=seed)
    sigma_s = block_diag_skew(ev_s)

    R = random_orthogonal_matrix(N, seed=seed)
    p0 = starting_state(R)
    Gx, Gy, Gs = build_generators_with_context(sigma_x, sigma_y, sigma_s, R)

    return RemappingModel(Gx, Gy, Gs, device=device), p0
