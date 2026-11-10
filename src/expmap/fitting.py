r"""Fitting generators to recorded spatial representations (Sec. 3.2, Appendix K).

This is the one place in the framework where anything is learned by gradient
descent. Everywhere else the generators are built algebraically from an
eigenvalue specification. Here they are recovered from recorded rate maps, to
test whether grid and place cell populations are consistent with an exponential
map, and whether the fitted generators extrapolate beyond the fitted region.

Given rate maps sampled on a pixel grid, both fits minimize the one-step
prediction error of Eq. 1 over a batch of :math:`K` random displacements of up
to a few pixels,

.. math::

    \mathcal{L} = \frac{1}{K} \sum_{k=1}^{K}
    \left\lVert \mathbf{p}(\mathbf{r}_k + \delta\mathbf{r}_k)
    - e^{\delta x_k G_x + \delta y_k G_y}\,\mathbf{p}(\mathbf{r}_k) \right\rVert^2.

``CommutingGeneratorFit`` is the procedure of the paper (App. K). It parameterizes
the generators as the framework builds them, :math:`G = R^\mathsf{T} \Sigma R`, with one
orthogonal :math:`R` and block-diagonal :math:`\Sigma` holding the wavevectors. The
generators are then skew-symmetric and commute for every parameter value.
``GeneratorFit`` is the procedure of the first submitted version, kept as the record.
It learns free skew-symmetric generators :math:`G = A - A^\mathsf{T}` with a penalty
:math:`w \lVert [G_x, G_y] \rVert^2` on the commutator. Without the penalty the fitted
generators need not commute, so displacements need not compose.

Skew-symmetry matters for extrapolation. Only an orthogonal :math:`e^{\delta\mathbf{r} \cdot \mathbf{G}}`
keeps :math:`\lVert\mathbf{p}\rVert` constant once the displacements leave the range the fit saw. Pass
``enforce_skew=False`` to see the norm drift, which ``generator_diagnostics``
measures.
"""

import numpy as np
import torch
from tqdm import tqdm

from expmap.generators import block_diag_skew


def sample_displacements(
    n_bins: int,
    batch_size: int,
    max_step: int = 2,
    generator: torch.Generator | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Sample random start and end pixels a bounded step apart.

    Steps that would leave the arena are reflected rather than clipped, so the
    displacement keeps its magnitude and the sampled distribution stays
    symmetric. Clipping would bias short steps toward the walls.

    Parameters
    ----------
    n_bins : int
        Side length of the square rate map grid.
    batch_size : int
        Number of displacement pairs.
    max_step : int, optional
        Largest step in pixels along each axis; steps are drawn uniformly from
        ``-max_step ... +max_step``.
    generator : torch.Generator, optional
        Seed the sampling. Leave None to use the global RNG.

    Returns
    -------
    tuple[torch.Tensor, torch.Tensor]
        Start and end pixel indices, each of shape (batch_size, 2).
    """
    start = torch.randint(0, n_bins, size=(batch_size, 2), generator=generator)
    step = torch.randint(-max_step, max_step + 1, size=(batch_size, 2), generator=generator)

    outside = ((start + step) >= n_bins) | ((start + step) < 0)
    step = torch.where(outside, -step, step)
    return start, start + step


def pixels_to_arena(steps: torch.Tensor, n_bins: int, arena_size: float = 1.0) -> torch.Tensor:
    """Convert a pixel displacement to arena units.

    The experimental maps span ``[-arena_size, +arena_size]`` across ``n_bins``
    pixels, so one pixel is ``2 * arena_size / n_bins``.

    Parameters
    ----------
    steps : torch.Tensor
        Pixel displacements, shape (..., 2).
    n_bins : int
        Side length of the rate map grid.
    arena_size : float, optional
        Half-width of the arena in the units the generators are expressed in.

    Returns
    -------
    torch.Tensor
        Displacements in arena units, same shape as ``steps``.
    """
    return (2 * arena_size / n_bins) * steps.float()


class GeneratorFit:
    r"""Recover :math:`G_x` and :math:`G_y` from measured rate maps with a commutator penalty.

    The procedure of the submitted version. The paper's fit is
    ``CommutingGeneratorFit``, which shares this interface.

    Parameters
    ----------
    n_units : int
        Number of recorded units, which is the dimension of the population vector.
    commutator_weight : float, optional
        Weight on the :math:`\lVert [G_x, G_y] \rVert^2` penalty.
    lr : float, optional
        Adam learning rate.
    enforce_skew : bool, optional
        Parameterize :math:`G = A - A^\mathsf{T}` so the generators are exactly
        skew-symmetric and transport is exactly orthogonal. Set False to optimize
        the generators directly. That fits the training domain about as well,
        but the norm of :math:`\mathbf{p}` drifts when extrapolating.
    seed : int, optional
        Seeds both the initialization and the displacement sampling.

    Attributes
    ----------
    Gx, Gy : torch.Tensor
        The generators, shape (n_units, n_units). Derived from the learned
        parameters, so read them rather than assign to them.
    loss_history : list[float]
        Total loss per step, populated by ``fit``.
    """

    def __init__(
        self,
        n_units: int,
        commutator_weight: float = 1.0,
        lr: float = 1e-3,
        enforce_skew: bool = True,
        seed: int | None = None,
    ) -> None:
        rng = np.random.default_rng(seed)
        scale = 1 / np.sqrt(n_units)

        self._Ax = torch.tensor(scale * rng.uniform(-1, 1, (n_units, n_units)), dtype=torch.float32, requires_grad=True)
        self._Ay = torch.tensor(scale * rng.uniform(-1, 1, (n_units, n_units)), dtype=torch.float32, requires_grad=True)
        self.n_units = n_units
        self.commutator_weight = commutator_weight
        self.enforce_skew = enforce_skew
        self.optimizer = torch.optim.Adam([self._Ax, self._Ay], lr=lr)
        self.loss_history: list[float] = []

        self._generator = torch.Generator()
        if seed is not None:
            self._generator.manual_seed(seed)

    @property
    def Gx(self) -> torch.Tensor:
        """Generator for the x-direction, shape (n_units, n_units)."""
        return self._Ax - self._Ax.T if self.enforce_skew else self._Ax

    @property
    def Gy(self) -> torch.Tensor:
        """Generator for the y-direction, shape (n_units, n_units)."""
        return self._Ay - self._Ay.T if self.enforce_skew else self._Ay

    def transport(self, p0: torch.Tensor, dr: torch.Tensor) -> torch.Tensor:
        r"""Apply :math:`e^{\delta x G_x + \delta y G_y}` to a batch of population vectors (Eq. 1).

        Parameters
        ----------
        p0 : torch.Tensor
            Population vectors, shape (batch, n_units).
        dr : torch.Tensor
            Displacements in arena units, shape (batch, 2).

        Returns
        -------
        torch.Tensor
            Transported population vectors, shape (batch, n_units).
        """
        Q = torch.matrix_exp(dr[..., 0, None, None] * self.Gx[None] + dr[..., 1, None, None] * self.Gy[None])
        return (Q @ p0[..., None]).squeeze(-1)

    def loss(self, p0: torch.Tensor, pt: torch.Tensor, dr: torch.Tensor) -> torch.Tensor:
        """Prediction error plus the commutator penalty."""
        prediction_error = torch.mean((pt - self.transport(p0, dr)) ** 2)
        commutator = torch.mean((self.Gx @ self.Gy - self.Gy @ self.Gx) ** 2)
        return prediction_error + self.commutator_weight * commutator

    def fit(
        self,
        ratemaps: torch.Tensor,
        n_steps: int,
        batch_size: int = 32,
        max_step: int = 2,
        arena_size: float = 1.0,
        progress: bool = True,
    ) -> list[float]:
        """Fit the generators to rate maps.

        Parameters
        ----------
        ratemaps : torch.Tensor
            Rate maps indexed **(bins_x, bins_y, n_units)**, position first, so
            a pixel index returns a population vector directly.
        n_steps : int
            Number of Adam steps.
        batch_size : int, optional
            Displacement pairs per step.
        max_step : int, optional
            Largest pixel step per axis.
        arena_size : float, optional
            Half-width of the arena, setting the units of the generators.
        progress : bool, optional
            Show a progress bar.

        Returns
        -------
        list[float]
            Loss history, also stored on the instance.
        """
        n_bins = ratemaps.shape[0]

        for _ in tqdm(range(n_steps), disable=not progress):
            self.optimizer.zero_grad()

            start, end = sample_displacements(n_bins, batch_size, max_step, self._generator)
            dr = pixels_to_arena(end - start, n_bins, arena_size)

            loss = self.loss(ratemaps[start[:, 0], start[:, 1]], ratemaps[end[:, 0], end[:, 1]], dr)
            loss.backward()
            self.optimizer.step()
            self.loss_history.append(loss.item())

        return self.loss_history

    def generate(self, p0: torch.Tensor, n_bins: int, extent: float = 1.0, arena_size: float = 1.0) -> torch.Tensor:
        """Generate rate maps by transporting p0 across a grid of displacements.

        With ``extent=1`` this reconstructs the training domain; with ``extent=2``
        it generates an arena of twice the side length, as in Appendix K,
        where nothing in the fit saw those displacements.

        Parameters
        ----------
        p0 : torch.Tensor
            Population vector at the arena centre, shape (n_units,).
        n_bins : int
            Side length of the output grid.
        extent : float, optional
            Multiplier on the displacement range. 1 reproduces, 2 extrapolates.
        arena_size : float, optional
            Half-width used during fitting.

        Returns
        -------
        torch.Tensor
            Rate maps of shape (n_units, n_bins, n_bins), indexed
            (unit, bins_x, bins_y) to match ``analysis.compute_ratemaps``.
        """
        axis = torch.arange(n_bins) - n_bins // 2
        gx, gy = torch.meshgrid(axis, axis, indexing="ij")
        steps = torch.stack([gx.ravel(), gy.ravel()], dim=1)
        dr = extent * pixels_to_arena(steps, n_bins, arena_size)

        with torch.no_grad():
            p = self.transport(p0.expand(len(dr), -1), dr)
        return p.T.reshape(self.n_units, n_bins, n_bins)


class CommutingGeneratorFit(GeneratorFit):
    r"""Fit generators that commute by construction (Sec. 3.2, App. K).

    The generators are parameterized the way the framework builds them,

    .. math::

        G_x = R^\mathsf{T} \Sigma_x R, \qquad G_y = R^\mathsf{T} \Sigma_y R,

    with one orthogonal matrix :math:`R = e^{B - B^\mathsf{T}}` and block-diagonal
    skew matrices :math:`\Sigma_x`, :math:`\Sigma_y` holding the wavevector pairs
    :math:`(\lambda_{i,x}, \lambda_{i,y})` (Sec. 2.2, App. C). The generators
    share an eigenbasis, so :math:`[G_x, G_y] = 0` exactly and path integration
    is path independent without a penalty. The loss is the one-step prediction
    error alone. The fitted parameters are interpretable: the wavevectors are
    the spatial spectrum of the population, and :math:`R` is its mixing matrix.

    Transport is evaluated in closed form as the block rotations of App. C,

    .. math::

        \boldsymbol{\alpha}_i(\mathbf{r} + \delta\mathbf{r})
        = \mathrm{rot}(\delta\mathbf{r} \cdot \boldsymbol{\lambda}_i)\,\boldsymbol{\alpha}_i(\mathbf{r}),
        \qquad \boldsymbol{\alpha} = R\,\mathbf{p},

    rather than through ``matrix_exp``. This makes a step cheap.

    An odd ``n_units`` leaves one eigenvalue of each generator at zero, a
    direction of population space that is not transported. An even ``n_units``
    can learn the same by driving a wavevector pair to zero.

    Parameters
    ----------
    n_units : int
        Number of recorded units.
    lr : float, optional
        Adam learning rate for :math:`B`. The wavevectors use ``wavevector_lr``.
    wavevector_lr : float, optional
        Adam learning rate for the wavevectors. They are on the scale of the
        spectrum, several cycles per arena, not on the scale of :math:`R`.
    wavevector_scale : float, optional
        Standard deviation of the Gaussian initialization of the wavevectors,
        in cycles per arena half-width (``arena_size``).
    seed : int, optional
        Seeds initialization and displacement sampling.

    Attributes
    ----------
    Gx, Gy : torch.Tensor
        The generators, derived from :math:`R` and the wavevectors, both properties
        documented below.
    """

    def __init__(
        self,
        n_units: int,
        lr: float = 1e-3,
        wavevector_lr: float = 1e-2,
        wavevector_scale: float = 2.0,
        seed: int | None = None,
    ) -> None:
        rng = np.random.default_rng(seed)
        self.n_units = n_units
        self.n_pairs = n_units // 2
        self.commutator_weight = 0.0
        self.enforce_skew = True

        self._B = torch.tensor(rng.uniform(-1, 1, (n_units, n_units)) / np.sqrt(n_units), dtype=torch.float32)
        self._B.requires_grad_(True)
        self._lam = torch.tensor(wavevector_scale * rng.standard_normal((self.n_pairs, 2)), dtype=torch.float32)
        self._lam.requires_grad_(True)
        self.optimizer = torch.optim.Adam(
            [{"params": [self._B], "lr": lr}, {"params": [self._lam], "lr": wavevector_lr}]
        )
        self.loss_history: list[float] = []

        self._generator = torch.Generator()
        if seed is not None:
            self._generator.manual_seed(seed)

    @property
    def R(self) -> torch.Tensor:
        """Orthogonal mixing matrix, shape (n_units, n_units)."""
        return torch.matrix_exp(self._B - self._B.T)

    @property
    def wavevectors(self) -> torch.Tensor:
        r"""Wavevector pairs :math:`(\lambda_{i,x}, \lambda_{i,y})`, shape (n_units // 2, 2)."""
        return self._lam

    def _generator_matrix(self, axis: int) -> torch.Tensor:
        Sigma = block_diag_skew(self._lam[:, axis])
        if self.n_units % 2:
            Sigma = torch.block_diag(Sigma, torch.zeros(1, 1))
        R = self.R
        return R.T @ Sigma @ R

    @property
    def Gx(self) -> torch.Tensor:
        """Generator for the x-direction, shape (n_units, n_units)."""
        return self._generator_matrix(0)

    @property
    def Gy(self) -> torch.Tensor:
        """Generator for the y-direction, shape (n_units, n_units)."""
        return self._generator_matrix(1)

    def transport(self, p0: torch.Tensor, dr: torch.Tensor) -> torch.Tensor:
        r"""Apply :math:`e^{\delta x G_x + \delta y G_y}` as block rotations in the eigenbasis (App. C).

        Parameters
        ----------
        p0 : torch.Tensor
            Population vectors, shape (batch, n_units).
        dr : torch.Tensor
            Displacements in arena units, shape (batch, 2).

        Returns
        -------
        torch.Tensor
            Transported population vectors, shape (batch, n_units).
        """
        R = self.R
        alpha = p0 @ R.T
        n = 2 * self.n_pairs
        pairs = alpha[:, :n].reshape(-1, self.n_pairs, 2)
        Omega = dr @ self._lam.T
        c, s = torch.cos(Omega), torch.sin(Omega)
        rotated = torch.stack([c * pairs[..., 0] - s * pairs[..., 1], s * pairs[..., 0] + c * pairs[..., 1]], dim=-1)
        alpha = torch.cat([rotated.reshape(-1, n), alpha[:, n:]], dim=1)
        return alpha @ R

    def loss(self, p0: torch.Tensor, pt: torch.Tensor, dr: torch.Tensor) -> torch.Tensor:
        """One-step prediction error; commutativity needs no penalty here."""
        return torch.mean((pt - self.transport(p0, dr)) ** 2)


def generator_diagnostics(fit: GeneratorFit) -> dict[str, float]:
    r"""Measure how far the fitted generators are from the framework's ideal.

    With ``enforce_skew`` the symmetric fraction is 0 by construction and this is
    a consistency check. With ``enforce_skew=False`` it is a measurement, and the
    contrast shows why the generators are parameterized to be skew-symmetric
    rather than penalized toward it.

    Parameters
    ----------
    fit : GeneratorFit
        A fitted instance.

    Returns
    -------
    dict[str, float]
        ``commutator_rms``, and per generator ``*_rms`` alongside
        ``*_symmetric_fraction``, the RMS of :math:`G + G^\mathsf{T}` relative to that of :math:`G`,
        which is 0 for a perfectly skew-symmetric generator.
    """
    Gx, Gy = fit.Gx.detach(), fit.Gy.detach()

    def rms(x: torch.Tensor) -> float:
        return float(x.pow(2).mean().sqrt())

    diagnostics = {"commutator_rms": rms(Gx @ Gy - Gy @ Gx)}
    for name, G in [("Gx", Gx), ("Gy", Gy)]:
        diagnostics[f"{name}_rms"] = rms(G)
        diagnostics[f"{name}_symmetric_fraction"] = rms(G + G.T) / rms(G)
    return diagnostics
