r"""Generator matrix construction for the exponential map framework.

The pipeline runs from an eigenvalue specification to the generator matrices of
Eq. 1. Skew-symmetry and commutativity let both generators share one
block diagonalization (Sec. 2.3),

.. math::

    G_x = R^\mathsf{T} \Sigma_x R, \qquad G_y = R^\mathsf{T} \Sigma_y R,

with :math:`R` orthogonal and each :math:`\Sigma` block diagonal in the 2x2
skew-symmetric blocks

.. math::

    \begin{bmatrix} 0 & -\lambda_i \\ \lambda_i & 0 \end{bmatrix}.

The pairs :math:`(\lambda_{i,x}, \lambda_{i,y})` are the wavevectors of the plane
waves every unit is built from (Appendix C), so choosing the eigenvalues chooses
the representation:

- **Roots of unity** (Appendix E). Angles :math:`\varphi_j = \pi j / M` on one
  ring give metric preservation and grids, hexagonal for :math:`M = 3` and
  square for :math:`M = 2`.
- **Gaussian** (Appendix H, with the sampling derivation in Appendix I).
  Eigenvalues drawn from :math:`\mathcal{N}(0, 2\sigma^2)` give similarity
  preservation and localized, place-like fields.
- **Bessel-spaced rings** (Appendix G). Ring radii proportional to the zeros of
  :math:`J_0` make the similarity a Fourier-Bessel series. :math:`L` rotated
  copies of each ring push the angular dependence to harmonic order
  :math:`2ML`, so the similarity is radially symmetric. See
  ``bessel_module_frequencies`` and ``multiscale_wavevectors``.

For grids the construction reads

.. math::

    \varphi_j = \frac{\pi j}{M}
    \;\longrightarrow\;
    \lambda_{j,x} = k \cos\varphi_j,\quad \lambda_{j,y} = k \sin\varphi_j
    \;\longrightarrow\;
    \Sigma_x, \Sigma_y
    \;\longrightarrow\;
    G_x, G_y,

where ``roots_of_unity_eigenvalues`` returns the angles :math:`\varphi`, the
caller resolves them into the two wavevector components with the radius
:math:`k` that sets the grid scale, ``block_diag_skew`` builds each
:math:`\Sigma`, and ``build_generators`` applies :math:`R`. For place-like fields
``gaussian_eigenvalues`` returns the eigenvalues :math:`\lambda` themselves and
the cos/sin step is skipped. The two functions return same-shaped arrays of
different quantities: passing :math:`\varphi` to ``block_diag_skew`` directly
silently builds the wrong matrix.
"""

import numpy as np
import torch
from scipy.special import jn_zeros


def roots_of_unity_eigenvalues(M: int, n_copies: int = 1) -> np.ndarray:
    r"""Compute eigenvalue angles for metric-preserving representations.

    Generates the angular positions of eigenvalue pairs that form :math:`M`-fold
    roots of unity on a single ring in frequency space. Metric preservation
    requires the phasor sum to vanish, :math:`\sum_j e^{2i\varphi_j} = 0`, which
    holds when the angles are evenly spaced as :math:`\varphi_j = \pi j / M`
    (Appendix E).

    For :math:`M = 3` the plane wave orientations are 60 degrees apart, giving
    hexagonal grid patterns. For :math:`M = 2` they are 90 degrees apart, giving
    square grids.

    Parameters
    ----------
    M : int
        Symmetry order. :math:`M = 3` gives hexagonal grids, :math:`M = 2` square grids.
    n_copies : int, optional
        Number of repeated eigenvalue modules. The representation has
        2 * M * n_copies neurons.

    Returns
    -------
    numpy.ndarray
        Eigenvalue **angles** :math:`\varphi` of shape (M * n_copies,), not
        eigenvalues. Resolve them with ``k * np.cos(phi)`` and ``k * np.sin(phi)``
        for the x and y generators before calling ``block_diag_skew``. The radius
        :math:`k` sets the spatial frequency, and hence the grid scale. Contrast
        ``gaussian_eigenvalues``, which returns eigenvalues directly.
    """
    phi = np.arange(0, M) * np.pi / M
    return np.tile(phi, n_copies)


def gaussian_eigenvalues(N: int, scale: float = np.sqrt(2), seed: int | None = None) -> np.ndarray:
    r"""Generate Gaussian-distributed eigenvalues for place-cell representations.

    Random eigenvalues drawn from a zero-mean Gaussian distribution produce
    representations whose pairwise similarity approximates a Gaussian function
    of spatial displacement. The resulting neural tuning curves are localized
    place-like fields rather than periodic grid patterns (Appendix H, with the
    sampling derivation in Appendix I).

    Sampling with standard deviation ``scale`` targets the similarity

    .. math::

        C(\Delta) \approx e^{-\mathrm{scale}^2 \Delta^2 / 2},

    since Appendix I draws from :math:`\mathcal{N}(0, 2\sigma^2)` for a target
    :math:`e^{-\sigma^2 x^2}`, so :math:`\mathrm{scale}^2 = 2\sigma^2`. The paper's
    :math:`\sigma` and this ``scale`` differ by that factor of two. The
    approximation is a Monte Carlo estimate over :math:`N/2` samples, so it
    holds to roughly :math:`1/\sqrt{N/2}`.

    Parameters
    ----------
    N : int
        Total number of neurons (must be even). The number of eigenvalue
        pairs is N // 2.
    scale : float, optional
        Standard deviation of the eigenvalue distribution, not the width of
        the resulting similarity function. The default :math:`\sqrt{2}` gives
        :math:`C(\Delta) \approx e^{-\Delta^2}`, the :math:`\mathcal{N}(0, 2)` of
        Appendix H.
    seed : int, optional
        Random seed for reproducibility.

    Returns
    -------
    numpy.ndarray
        Eigenvalues of shape (N // 2,).

    Raises
    ------
    ValueError
        If N is odd. Each eigenvalue spans a 2x2 block, so an odd N would
        silently build a representation of N - 1 units.
    """
    if N % 2:
        raise ValueError(f"N must be even, got {N}")
    rng = np.random.default_rng(seed)
    return scale * rng.standard_normal(N // 2)


def bessel_module_frequencies(K: int, arena_size: float, periods: float = 1.0) -> np.ndarray:
    r"""Ring radii proportional to the zeros of :math:`J_0`, for a multi-scale representation.

    Appendix G rewrites the similarity function in polar coordinates and applies
    the Jacobi-Anger expansion, which splits it into a radial :math:`J_0` term and
    angular terms. Placing the rings at radii :math:`k_i` proportional to the zeros
    :math:`j_{0,i}` makes
    the radial part a Fourier-Bessel series, so a target radial kernel can be
    approximated by choosing the ring weights.

    This also gives the paper's prediction for grid module spacing. The ratios
    of successive :math:`J_0` zeros, 2.30, 1.57, 1.36, 1.27, ..., decrease slowly toward
    one, so the predicted spacing ratio grows with grid scale rather than
    staying constant, and beyond the first the average of the low-order ratios
    is close to the mean spacing ratio reported experimentally (Sec. 2.4,
    Appendix G). The absolute scale is set here by ``arena_size`` and ``periods``;
    in the Fourier-Bessel picture it is the size of the region being encoded.

    Parameters
    ----------
    K : int
        Number of modules (rings).
    arena_size : float
        Half-width of the arena the representation should cover.
    periods : float, optional
        Number of grid periods (peak-to-peak spacings of the hexagonal lattice)
        the *lowest* frequency module shows across the full 2 * arena_size
        extent. Sets the absolute scale.

    Returns
    -------
    numpy.ndarray
        Ring radii :math:`k` of shape (K,), increasing. Ratios follow the :math:`J_0` zeros, so
        ``k[1:] / k[:-1]`` reproduces the spacing prediction of Fig. 6.
    """
    zeros = jn_zeros(0, K)
    # Three plane waves 60 degrees apart peak every 4 pi / (k sqrt(3)).
    k_lowest = 4 * np.pi * periods / (2 * arena_size * np.sqrt(3))
    return zeros * (k_lowest / zeros[0])


def multiscale_wavevectors(
    frequencies: np.ndarray,
    M: int,
    n_offsets: int = 1,
) -> tuple[np.ndarray, np.ndarray]:
    r"""Resolve ring radii into :math:`(\lambda_x, \lambda_y)` eigenvalue pairs.

    Each ring carries :math:`M` orientations evenly spaced by :math:`\pi / M`,
    repeated for ``n_offsets`` rotated copies offset by :math:`\pi / (ML)`, writing
    :math:`L` for ``n_offsets``. The :math:`ML` orientations evenly subdivide
    :math:`[0, \pi)`, and Appendix G shows the angular terms of the similarity
    then survive only at harmonic orders that are multiples of :math:`ML`. The
    lowest is :math:`\cos(2ML\theta)`, so the similarity is radially symmetric
    over a wide range of displacements.

    With :math:`M = 3, L = 1` the lowest harmonic is :math:`\cos(6\theta)`, and
    with :math:`M = 3, L = 2` it is :math:`\cos(12\theta)`, which is isotropic to
    numerical precision at small :math:`r`.

    Parameters
    ----------
    frequencies : numpy.ndarray
        Ring radii of shape (K,), for example from ``bessel_module_frequencies``.
    M : int
        Symmetry order per ring. 3 gives hexagonal grids.
    n_offsets : int, optional
        Number of rotated orientation sets, :math:`L` above.

    Returns
    -------
    tuple[numpy.ndarray, numpy.ndarray]
        Eigenvalues :math:`(\lambda_x, \lambda_y)`, each of shape (K * M * n_offsets,).
        The representation has N = 2 * K * M * n_offsets units.
    """
    k = np.asarray(frequencies)[:, None, None]
    m = np.arange(M)[None, :, None]
    offset = np.arange(n_offsets)[None, None, :]

    phi = np.pi * m / M + np.pi * offset / (M * n_offsets)
    return (k * np.cos(phi)).ravel(), (k * np.sin(phi)).ravel()


def block_diag_skew(eigenvalues: np.ndarray | torch.Tensor) -> torch.Tensor:
    r"""Construct a block-diagonal skew-symmetric matrix from eigenvalues.

    Each eigenvalue :math:`\lambda_i` is placed into a 2x2 skew-symmetric block,

    .. math::

        \begin{bmatrix} 0 & -\lambda_i \\ \lambda_i & 0 \end{bmatrix},

    and the blocks are assembled into a block-diagonal matrix, the canonical
    form of a real skew-symmetric matrix under orthogonal similarity
    transformation (spectral theorem).

    The sign follows Appendix C, where the subspace update is the
    counter-clockwise rotation

    .. math::

        \boldsymbol{\alpha}_i = \begin{bmatrix} \cos\Omega_i & -\sin\Omega_i \\
        \sin\Omega_i & \cos\Omega_i \end{bmatrix} \boldsymbol{\alpha}_{i,0},
        \qquad \Omega_i = x\lambda_{i,x} + y\lambda_{i,y},

    so that the components come out as :math:`\alpha_1 = A\cos(\Omega_i + \omega)`
    and :math:`\alpha_2 = A\sin(\Omega_i + \omega)` as written in Appendix D.
    Equivalently ``expm(block_diag_skew(w)) == linalg.block_rotation(w)``.

    The opposite sign is equally skew-symmetric and leaves the similarity
    function (Eq. 4) unchanged, since it is even in :math:`\lambda`; it reflects
    each rate map through the origin.

    Parameters
    ----------
    eigenvalues : numpy.ndarray or torch.Tensor
        Eigenvalue magnitudes, shape (N // 2,). The output matrix will
        have dimension (N, N) where N = 2 * len(eigenvalues).

    Returns
    -------
    torch.Tensor
        Block-diagonal skew-symmetric matrix of shape (N, N), dtype float32 on
        the CPU whatever the input; the model moves it to its device.
    """
    if isinstance(eigenvalues, np.ndarray):
        eigenvalues = torch.from_numpy(eigenvalues)
    eigenvalues = eigenvalues.to(torch.float32)

    N = 2 * len(eigenvalues)
    first = torch.arange(0, N, 2)
    sigma = torch.zeros((N, N), dtype=torch.float32)
    sigma[first, first + 1] = -eigenvalues
    sigma[first + 1, first] = eigenvalues
    return sigma


def build_generators(
    sigma_x: torch.Tensor,
    sigma_y: torch.Tensor,
    R: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    r"""Construct full generator matrices from eigenvalue matrices and the mixing matrix.

    Computes :math:`G_x = R^\mathsf{T} \Sigma_x R` and :math:`G_y = R^\mathsf{T} \Sigma_y R`,
    transforming the canonical block-diagonal eigenvalue matrices into the chosen
    orthogonal basis. The orthogonal matrix :math:`R` mixes the underlying plane waves into the
    neural tuning curves. It is a free parameter that shapes the tuning of
    single neurons and leaves the population similarity unchanged (Sec. 4).

    Parameters
    ----------
    sigma_x : torch.Tensor
        Block-diagonal eigenvalue matrix for x-direction, shape (N, N).
    sigma_y : torch.Tensor
        Block-diagonal eigenvalue matrix for y-direction, shape (N, N).
    R : torch.Tensor
        Orthogonal basis matrix, shape (N, N). Must satisfy :math:`R R^\mathsf{T} = I`.

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
    r"""Construct generator matrices including a context generator for remapping.

    Extends ``build_generators`` to also construct the context generator
    :math:`G_s = R^\mathsf{T} \Sigma_s R` for context-dependent remapping. Gaussian context
    eigenvalues make the maps of dissimilar contexts nearly orthogonal
    (Sec. 2.5).

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
