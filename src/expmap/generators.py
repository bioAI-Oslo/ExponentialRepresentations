"""Generator matrix construction for the exponential map framework.

This module implements the full pipeline from eigenvalue specification to
complete generator matrices. The eigenvalue structure directly determines
the type of spatial representation produced:

- **Roots of unity**: Metric-preserving representations (grid cells).
  Eigenvalue angles uniformly spaced as phi_j = pi * j / M yield M-fold
  symmetry — hexagonal grids for M=3, square grids for M=2 (Appendix E).

- **Gaussian**: Similarity-preserving representations (place cells).
  Random eigenvalues from a Gaussian distribution produce localized,
  approximately Gaussian spatial tuning (Appendix I).

- **Bessel-spaced rings**: Multi-scale, radially symmetric representations.
  Placing several rings at radii proportional to the zeros of J0 makes the
  similarity a Fourier-Bessel series (Appendix G), and choosing coprime
  symmetry and orientation orders pushes the angular dependence to a high
  harmonic. `bessel_module_frequencies` and `multiscale_wavevectors`.

The construction pipeline is:
    eigenvalues -> block_diag_skew -> build_generators(sigma_x, sigma_y, R)

with one step in between for the roots-of-unity case, because the two
eigenvalue functions return different quantities:

- `gaussian_eigenvalues` returns eigenvalues lambda directly, and feeds
  straight into `block_diag_skew`.
- `roots_of_unity_eigenvalues` returns eigenvalue *angles* phi. The caller
  resolves them into the two generators via the polar form
  lambda_x = k cos(phi), lambda_y = k sin(phi), so the pipeline reads
  phi -> cos/sin -> block_diag_skew. Passing phi to `block_diag_skew`
  directly silently builds the wrong matrix.
"""

import numpy as np
import torch
from scipy.special import jn_zeros


def roots_of_unity_eigenvalues(M: int, n_copies: int = 1) -> np.ndarray:
    """Compute eigenvalue angles for metric-preserving representations.

    Generates the angular positions of eigenvalue pairs that form M-fold
    roots of unity on a single ring in frequency space. The metric
    preservation condition requires that the phasor sum vanishes:
    sum_j exp(2i * phi_j) = 0, which is satisfied when the angles are
    uniformly distributed as phi_j = pi * j / M (Appendix E).

    For M=3 the resulting plane wave orientations are separated by 60 degrees,
    producing hexagonal grid patterns. For M=2 they are separated by 90
    degrees, producing square grids.

    Parameters
    ----------
    M : int
        Symmetry order. M=3 gives hexagonal grids, M=2 gives square grids.
    n_copies : int, optional
        Number of repeated eigenvalue modules. Total number of neurons
        in the representation will be 2 * M * n_copies.

    Returns
    -------
    numpy.ndarray
        Eigenvalue **angles** phi of shape (M * n_copies,), not eigenvalues.
        Resolve them with `k * np.cos(phi)` and `k * np.sin(phi)` for the x
        and y generators before calling `block_diag_skew`; the radius k sets
        the spatial frequency, and hence the grid scale. Contrast
        `gaussian_eigenvalues`, which returns eigenvalues directly.
    """
    phi = np.arange(0, M) * np.pi / M
    return np.tile(phi, n_copies)


def gaussian_eigenvalues(N: int, scale: float = np.sqrt(2), seed: int | None = None) -> np.ndarray:
    """Generate Gaussian-distributed eigenvalues for place-cell representations.

    Random eigenvalues drawn from a zero-mean Gaussian distribution produce
    representations whose pairwise similarity approximates a Gaussian function
    of spatial displacement. The resulting neural tuning curves are localized
    place-like fields rather than periodic grid patterns (Appendix H; the
    sampling derivation it refers back to is Appendix I).

    Sampling with standard deviation `scale` targets the similarity

        C(delta) ~= exp(-scale**2 * delta**2 / 2)

    since Appendix I draws from N(0, 2 sigma^2) for a target exp(-sigma^2 x^2),
    i.e. `scale**2 = 2 sigma**2`. The paper's sigma and this `scale` differ by
    that factor of two. The approximation is a Monte Carlo estimate over N // 2
    samples, so it holds to roughly 1 / sqrt(N // 2).

    Parameters
    ----------
    N : int
        Total number of neurons (must be even). The number of eigenvalue
        pairs is N // 2.
    scale : float, optional
        Standard deviation of the eigenvalue distribution — not the width of
        the resulting similarity function. The default sqrt(2) gives
        C(delta) ~= exp(-delta**2), the N(0, 2) of Appendix H.
    seed : int, optional
        Random seed for reproducibility.

    Returns
    -------
    numpy.ndarray
        Eigenvalues of shape (N // 2,).
    """
    rng = np.random.default_rng(seed)
    return scale * rng.standard_normal(N // 2)


def bessel_module_frequencies(K: int, arena_size: float, periods: float = 2.0) -> np.ndarray:
    """Ring radii proportional to the zeros of J0, for a multi-scale representation.

    Appendix G rewrites the similarity function in polar coordinates and applies
    the Jacobi-Anger expansion, which splits it into a radial J0 term and angular
    terms. Placing the rings at radii k_i proportional to the zeros j_{0,i} makes
    the radial part a Fourier-Bessel series, so a target radial kernel can be
    approximated by choosing the ring weights.

    This also gives the paper's normative prediction for grid module spacing: the
    ratio of successive J0 zeros, averaged over low-order zeros, is close to
    sqrt(2), which sits inside the experimentally observed spread. The absolute
    scale is set here by `arena_size` and `periods` — in the Fourier-Bessel
    picture it corresponds to the size of the region being encoded.

    Parameters
    ----------
    K : int
        Number of modules (rings).
    arena_size : float
        Half-width of the arena the representation should cover.
    periods : float, optional
        Approximate number of grid periods the *lowest* frequency module should
        show across the full 2 * arena_size extent. Sets the absolute scale.

    Returns
    -------
    numpy.ndarray
        Ring radii k of shape (K,), increasing. Ratios follow the J0 zeros, so
        `k[1:] / k[:-1]` reproduces the spacing prediction of Fig. 6.
    """
    zeros = jn_zeros(0, K)
    # Hexagonal grid period is 2 pi / (k sqrt(3)); solve for the k that puts
    # `periods` of the lowest module across the arena.
    k_lowest = 2 * np.pi * periods / (2 * arena_size * np.sqrt(3))
    return zeros * (k_lowest / zeros[0])


def multiscale_wavevectors(
    frequencies: np.ndarray,
    M: int,
    n_offsets: int = 1,
) -> tuple[np.ndarray, np.ndarray]:
    """Resolve ring radii into (lambda_x, lambda_y) eigenvalue pairs.

    Each ring carries `M` orientations evenly spaced by pi / M, repeated for
    `n_offsets` rotated copies offset by pi / (M * n_offsets). Writing L for
    `n_offsets`, Appendix G shows the angular terms survive only at harmonic
    orders that are multiples of both M and L, so when M and L are **coprime**
    the lowest angular dependence is pushed to cos(2 M L theta) and the
    similarity is radially symmetric over a wide range of displacements.

    With M = 3, L = 1 the lowest harmonic is cos(6 theta); with M = 3, L = 2 it
    is cos(12 theta), which is isotropic to numerical precision at small r.

    Parameters
    ----------
    frequencies : numpy.ndarray
        Ring radii of shape (K,), e.g. from `bessel_module_frequencies`.
    M : int
        Symmetry order per ring. 3 gives hexagonal grids.
    n_offsets : int, optional
        Number of rotated orientation sets, L above. Choose coprime to M to
        suppress low-order angular terms.

    Returns
    -------
    tuple[numpy.ndarray, numpy.ndarray]
        Eigenvalues (lambda_x, lambda_y), each of shape (K * M * n_offsets,).
        The representation has N = 2 * K * M * n_offsets units.
    """
    k = np.asarray(frequencies)[:, None, None]
    m = np.arange(M)[None, :, None]
    offset = np.arange(n_offsets)[None, None, :]

    phi = np.pi * m / M + np.pi * offset / (M * n_offsets)
    return (k * np.cos(phi)).ravel(), (k * np.sin(phi)).ravel()


def block_diag_skew(eigenvalues: np.ndarray | torch.Tensor) -> torch.Tensor:
    """Construct a block-diagonal skew-symmetric matrix from eigenvalues.

    Each eigenvalue lambda_i is placed into a 2x2 skew-symmetric block:
        [[0, -lambda_i], [lambda_i, 0]]
    and the blocks are assembled into a block-diagonal matrix. The resulting
    matrix is the canonical form of a real skew-symmetric matrix under
    orthogonal similarity transformation (spectral theorem).

    The sign follows Appendix C, where the subspace update is the
    counter-clockwise rotation

        alpha_i = [[cos(Omega_i), -sin(Omega_i)],
                   [sin(Omega_i),  cos(Omega_i)]] alpha_i0

    with Omega_i = x lambda_ix + y lambda_iy, so that the components come out as
    alpha_1 = A cos(x . lambda + omega) and alpha_2 = A sin(x . lambda + omega)
    as written in Appendix D. Equivalently
    `expm(block_diag_skew(w)) == linalg.block_rotation(w)`, which
    `tests/test_linalg.py` asserts.

    The opposite sign is equally skew-symmetric and leaves the similarity
    function (Eq. 4) unchanged, since it is even in lambda — it reflects each
    ratemap through the origin.

    Parameters
    ----------
    eigenvalues : numpy.ndarray or torch.Tensor
        Eigenvalue magnitudes, shape (N // 2,). The output matrix will
        have dimension (N, N) where N = 2 * len(eigenvalues).

    Returns
    -------
    torch.Tensor
        Block-diagonal skew-symmetric matrix of shape (N, N).
    """
    if isinstance(eigenvalues, np.ndarray):
        eigenvalues = torch.from_numpy(eigenvalues)
    eigenvalues = eigenvalues.to(torch.float32)

    blocks = []
    for lam in eigenvalues:
        block = torch.zeros((2, 2), dtype=torch.float32)
        block[0, 1] = -lam
        block[1, 0] = lam
        blocks.append(block)

    return torch.block_diag(*blocks)


def build_generators(
    sigma_x: torch.Tensor,
    sigma_y: torch.Tensor,
    R: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Construct full generator matrices from eigenvalue matrices and orthogonal basis.

    Computes Gx = R^T @ Sigma_x @ R and Gy = R^T @ Sigma_y @ R, transforming
    the canonical block-diagonal eigenvalue matrices into the chosen orthogonal
    basis. The orthogonal matrix R mixes the underlying plane waves into the
    final neural tuning curves; it is a degree of freedom that strongly
    influences the spatial selectivity profiles of individual neurons while
    leaving the population-level similarity structure unchanged (Sec. 4).

    Parameters
    ----------
    sigma_x : torch.Tensor
        Block-diagonal eigenvalue matrix for x-direction, shape (N, N).
    sigma_y : torch.Tensor
        Block-diagonal eigenvalue matrix for y-direction, shape (N, N).
    R : torch.Tensor
        Orthogonal basis matrix, shape (N, N). Must satisfy R @ R^T = I.

    Returns
    -------
    tuple[torch.Tensor, torch.Tensor]
        Generator matrices (Gx, Gy), each of shape (N, N).
    """
    Gx = R.T @ sigma_x @ R
    Gy = R.T @ sigma_y @ R
    return Gx, Gy


def build_generators_with_context(
    sigma_x: torch.Tensor,
    sigma_y: torch.Tensor,
    sigma_s: torch.Tensor,
    R: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Construct generator matrices including a context generator for remapping.

    Extends build_generators to also construct the context generator
    Gs = R^T @ Sigma_s @ R, which enables context-dependent remapping.
    The context generator should typically have Gaussian-distributed
    eigenvalues to produce orthogonal remapping across contexts (Sec. 2.5).

    Parameters
    ----------
    sigma_x : torch.Tensor
        Block-diagonal eigenvalue matrix for x-direction, shape (N, N).
    sigma_y : torch.Tensor
        Block-diagonal eigenvalue matrix for y-direction, shape (N, N).
    sigma_s : torch.Tensor
        Block-diagonal eigenvalue matrix for context, shape (N, N).
    R : torch.Tensor
        Orthogonal basis matrix, shape (N, N).

    Returns
    -------
    tuple[torch.Tensor, torch.Tensor, torch.Tensor]
        Generator matrices (Gx, Gy, Gs), each of shape (N, N).
    """
    Gx, Gy = build_generators(sigma_x, sigma_y, R)
    Gs = R.T @ sigma_s @ R
    return Gx, Gy, Gs
