"""Linear algebra utilities for the exponential map framework.

Provides orthogonal matrix sampling, canonical initial state construction,
and basis-building routines grounded in the framework's mathematical structure.
"""

import numpy as np
import torch
from scipy.linalg import block_diag
from scipy.stats import ortho_group


def random_orthogonal_matrix(n: int, seed: int | None = None) -> torch.Tensor:
    """Sample a random orthogonal matrix from the Haar measure.

    The orthogonal matrix R is a degree of freedom in the exponential map
    framework that mixes the underlying plane waves (canonical basis vectors)
    into the observed neural tuning curves. Different choices of R produce
    different spatial selectivity profiles for individual neurons while
    preserving the population-level similarity structure (Sec. 4).

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
    R = torch.tensor(ortho_group.rvs(n, random_state=rng), dtype=torch.float32)
    return R


def block_diagonal_orthogonal(n_blocks: int, block_size: int, seed: int | None = None) -> torch.Tensor:
    """Sample an orthogonal matrix that mixes only within blocks.

    The other end of the R continuum from `random_orthogonal_matrix`. A dense R
    mixes every subspace into every neuron, giving mixed-selectivity cells; a
    block-diagonal R confines the mixing to one block per module, so each neuron
    draws on a single spatial frequency and reads as a module-pure grid cell.

    The population similarity is identical either way, because with the
    canonical `p0 = R^T 1/sqrt(N)` the coefficients in Eq. 4 are uniform however
    R mixes. That invariance is the point of the exploration: R selects the
    single-cell organisation without touching the population metric.

    Parameters
    ----------
    n_blocks : int
        Number of blocks, i.e. modules.
    block_size : int
        Dimension of each block. For a ring of M orientations this is 2 * M.
    seed : int, optional
        Block `i` is drawn with `random_state=seed + i`, so the whole matrix is
        reproducible from one seed.

    Returns
    -------
    torch.Tensor
        Orthogonal matrix of shape (n_blocks * block_size, n_blocks * block_size),
        dtype float32, block diagonal by construction.
    """
    offset = 0 if seed is None else seed
    blocks = [ortho_group.rvs(block_size, random_state=offset + i).astype(np.float32) for i in range(n_blocks)]
    return torch.tensor(block_diag(*blocks), dtype=torch.float32)


def starting_state(R: torch.Tensor) -> torch.Tensor:
    """Compute the canonical initial state vector p0.

    The initial state is defined as p0 = R^T @ (1 / sqrt(N)) * ones(N),
    which ensures unit norm (||p0|| = 1) due to the orthogonality of R.
    Combined with the choice of R, the starting state determines the
    similarity function's independence from the basis (see Sec. 4,
    Appendix F).

    Parameters
    ----------
    R : torch.Tensor
        Orthogonal basis matrix of shape (N, N).

    Returns
    -------
    torch.Tensor
        Initial state vector of shape (N,), dtype float32.
    """
    N = len(R)
    return R.T @ torch.ones(N, dtype=torch.float32) / np.sqrt(N)


def orthogonal_basis(q0: np.ndarray, seed: int | None = None) -> np.ndarray:
    """Construct a full orthonormal basis from a starting vector via Gram-Schmidt.

    Starting from q0, constructs an orthonormal basis for R^N by iteratively
    sampling random vectors and orthogonalizing them against all previously
    accepted basis vectors.

    Parameters
    ----------
    q0 : numpy.ndarray
        Starting vector of shape (N,). Must be nonzero. Will be normalized
        to unit length as the first basis vector.
    seed : int, optional
        Random seed for the vectors filling out the remaining N-1 directions.

    Returns
    -------
    numpy.ndarray
        Orthonormal basis of shape (N, N), where the first row is q0
        normalized.
    """
    rng = np.random.default_rng(seed)
    basis = [q0 / np.sqrt(np.dot(q0, q0))]

    for _ in range(len(q0) - 1):
        q = rng.uniform(-1, 1, len(q0))
        for z in basis:
            q = q - np.dot(q, z) * z
        q = q / np.sqrt(np.dot(q, q))
        basis.append(q)

    return np.array(basis)


def block_rotation(omegas: np.ndarray) -> np.ndarray:
    """Construct a block-diagonal rotation matrix from angles.

    Each angle omega_i produces a 2x2 rotation block:
        [[cos(omega), -sin(omega)],
         [sin(omega),  cos(omega)]]
    These blocks are assembled into a block-diagonal matrix representing
    the direct product of planar rotations. This is the explicit form of
    exp(Sigma) for Sigma from `generators.block_diag_skew`, i.e.

        block_rotation(omegas) == expm(block_diag_skew(omegas))

    which `tests/test_linalg.py` asserts, and is the rotation written out in
    Appendix C.

    Parameters
    ----------
    omegas : numpy.ndarray
        Rotation angles, shape (N // 2,).

    Returns
    -------
    numpy.ndarray
        Block-diagonal rotation matrix of shape (N, N).
    """
    blocks = []
    for omega in omegas:
        block = np.array([[np.cos(omega), -np.sin(omega)], [np.sin(omega), np.cos(omega)]])
        blocks.append(block)
    return block_diag(*blocks)
