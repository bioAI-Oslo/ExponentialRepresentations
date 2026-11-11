"""Linear algebra utilities for the exponential map framework.

Orthogonal mixing matrices, dense and block diagonal, the canonical origin
state, and the block rotations of Appendix C.
"""

import numpy as np
import torch
from scipy.linalg import block_diag
from scipy.stats import ortho_group


def random_orthogonal_matrix(n: int, seed: int | None = None) -> torch.Tensor:
    """Sample a random orthogonal matrix from the Haar measure.

    The orthogonal matrix :math:`R` mixes the plane waves of the canonical basis
    into the neural tuning curves. It is a free parameter: different :math:`R` give
    different single-neuron tuning curves and the same population similarity
    (Sec. 4).

    Parameters
    ----------
    n : int
        Matrix dimension. Produces an (n, n) orthogonal matrix.
    seed : int, optional
        Random seed for reproducibility.

    Returns
    -------
    torch.Tensor
        Random orthogonal matrix of shape (n, n), dtype float32.
    """
    rng = np.random.default_rng(seed)
    return torch.tensor(ortho_group.rvs(n, random_state=rng), dtype=torch.float32)


def block_diagonal_orthogonal(n_blocks: int, block_size: int, seed: int | None = None) -> torch.Tensor:
    r"""Sample an orthogonal matrix that mixes only within blocks.

    The other end of the :math:`R` continuum from ``random_orthogonal_matrix``. A
    dense :math:`R` mixes every subspace into every neuron, giving
    mixed-selectivity cells. A block-diagonal :math:`R` confines the mixing to
    one block per module, so each neuron draws on a single spatial frequency and
    reads as a module-pure grid cell.

    The population similarity is identical either way, because with the
    canonical :math:`\mathbf{p}_0 = R^\mathsf{T}\mathbf{1}/\sqrt{N}` the
    coefficients in Eq. 4 are uniform however :math:`R` mixes. :math:`R` therefore
    sets the single-cell organization without changing the population
    similarity.

    Parameters
    ----------
    n_blocks : int
        Number of blocks, one per module.
    block_size : int
        Dimension of each block. For a ring of M orientations this is 2 * M.
    seed : int, optional
        Block ``i`` is drawn with ``random_state=seed + i``, so the whole matrix is
        reproducible from one seed. If None, the blocks are unseeded.

    Returns
    -------
    torch.Tensor
        Orthogonal matrix of shape (n_blocks * block_size, n_blocks * block_size),
        dtype float32, block diagonal by construction.
    """
    blocks = [
        ortho_group.rvs(block_size, random_state=None if seed is None else seed + i).astype(np.float32)
        for i in range(n_blocks)
    ]
    return torch.tensor(block_diag(*blocks), dtype=torch.float32)


def starting_state(R: torch.Tensor) -> torch.Tensor:
    r"""Compute the canonical origin state :math:`\mathbf{p}_0`.

    The origin state :math:`\mathbf{p}_0 = R^\mathsf{T}\mathbf{1}/\sqrt{N}`
    (Appendix I) has unit norm because :math:`R` is orthogonal, and it makes the
    coefficients :math:`\boldsymbol{\alpha}_0 = R\mathbf{p}_0` of Eq. 4 uniform, so
    the similarity function does not depend on :math:`R` (Sec. 4, Appendix F).

    Parameters
    ----------
    R : torch.Tensor
        Orthogonal basis matrix of shape (N, N).

    Returns
    -------
    torch.Tensor
        Origin state of shape (N,), in the dtype of R.
    """
    N = len(R)
    return R.T @ torch.ones(N, dtype=R.dtype) / np.sqrt(N)


def orthogonal_basis(q0: np.ndarray, seed: int | None = None) -> torch.Tensor:
    r"""Complete a starting vector to a mixing matrix :math:`R` whose first row is :math:`\mathbf{q}_0`.

    Fills the remaining :math:`N - 1` directions with standard normal vectors and
    orthonormalizes them by QR decomposition, so that the first canonical
    direction of :math:`R` is the given :math:`\mathbf{q}_0`:
    :math:`R\,\mathbf{q}_0 / \lVert\mathbf{q}_0\rVert = \mathbf{e}_1`. Use it to pin
    one plane-wave direction to a chosen pattern across the population instead
    of sampling :math:`R` from the Haar measure with ``random_orthogonal_matrix``.

    Parameters
    ----------
    q0 : numpy.ndarray
        Starting vector of shape (N,). Must be nonzero. It is normalized to
        unit length as the first row.
    seed : int, optional
        Random seed for the vectors filling out the remaining :math:`N - 1` directions.

    Returns
    -------
    torch.Tensor
        Orthogonal matrix of shape (N, N), dtype float32, whose first row is
        :math:`\mathbf{q}_0` normalized.
    """
    rng = np.random.default_rng(seed)
    N = len(q0)
    Q, _ = np.linalg.qr(np.column_stack([q0, rng.standard_normal((N, N - 1))]))
    # QR fixes the sign of the first column arbitrarily, so align it with q0.
    if Q[:, 0] @ q0 < 0:
        Q = -Q
    return torch.tensor(Q.T, dtype=torch.float32)


def block_rotation(omegas: np.ndarray) -> np.ndarray:
    r"""Construct a block-diagonal rotation matrix from angles.

    Each angle :math:`\omega_i` produces a 2x2 rotation block,

    .. math::

        \begin{bmatrix} \cos\omega_i & -\sin\omega_i \\ \sin\omega_i & \cos\omega_i \end{bmatrix},

    and the blocks are assembled into a block-diagonal matrix, the direct
    product of planar rotations written out in Appendix C. This is the explicit
    form of :math:`e^{\Sigma}` for :math:`\Sigma` from
    ``generators.block_diag_skew``, so that
    ``block_rotation(omegas) == expm(block_diag_skew(omegas))``.

    Parameters
    ----------
    omegas : numpy.ndarray
        Rotation angles, shape (N // 2,).

    Returns
    -------
    numpy.ndarray
        Block-diagonal rotation matrix of shape (N, N).
    """
    cos, sin = np.cos(omegas), np.sin(omegas)
    N = 2 * len(omegas)
    first = np.arange(0, N, 2)
    Q = np.zeros((N, N), dtype=cos.dtype)
    Q[first, first] = cos
    Q[first + 1, first + 1] = cos
    Q[first, first + 1] = -sin
    Q[first + 1, first] = sin
    return Q
