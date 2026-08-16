"""Fitting generators to measured spatial representations (Appendix K).

This is the one place in the framework where anything is learned by gradient
descent. Everywhere else the generators are built algebraically from an
eigenvalue specification; here they are recovered from data, to ask whether real
grid cell ratemaps are consistent with an exponential map at all — and, if so,
whether the fitted generators extrapolate beyond the arena they were fitted in.

Given ratemaps sampled on a pixel grid, the fit minimises the one-step
prediction error of Eq. 1 over random displacements,

    L = mean || p(r + dr) - exp(dx Gx + dy Gy) p(r) ||^2 + w || [Gx, Gy] ||^2

The commutator penalty is what makes the result a *path-integrating* solution
rather than merely a good interpolator: without it nothing forces the two
generators to be simultaneously diagonalisable, and displacements would not
compose.

Skew-symmetry is imposed structurally rather than penalised, by learning a free
matrix A and taking `G = A - A^T`. Appendix K enforces it "to guarantee
orthogonal transport during spatial extrapolation": the point of the experiment
is to generate an arena twice the size of the one fitted, and only an orthogonal
`exp(dr . G)` keeps `||p||` from drifting once the displacements leave the
range the fit ever saw. Pass `enforce_skew=False` to optimise the generators
freely and watch that fail — `generator_diagnostics` measures it.

This module is torch, like `model` and `generators`.
"""

import numpy as np
import torch
from tqdm import tqdm


def sample_displacements(
    n_bins: int,
    batch_size: int,
    max_step: int = 2,
    generator: torch.Generator | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Sample random start and end pixels a bounded step apart.

    Steps that would leave the arena are reflected rather than clipped, so the
    displacement keeps its magnitude and the sampled distribution stays
    symmetric — clipping would bias short steps toward the walls.

    Parameters
    ----------
    n_bins : int
        Side length of the square ratemap grid.
    batch_size : int
        Number of displacement pairs.
    max_step : int, optional
        Largest step in pixels along each axis; steps are drawn uniformly from
        `-max_step ... +max_step`.
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

    The experimental maps span `[-arena_size, +arena_size]` across `n_bins`
    pixels, so one pixel is `2 * arena_size / n_bins`.

    Parameters
    ----------
    steps : torch.Tensor
        Pixel displacements, shape (..., 2).
    n_bins : int
        Side length of the ratemap grid.
    arena_size : float, optional
        Half-width of the arena in the units the generators are expressed in.

    Returns
    -------
    torch.Tensor
        Displacements in arena units, same shape as `steps`.
    """
    return (2 * arena_size / n_bins) * steps.float()


class GeneratorFit:
    """Recover Gx and Gy from measured ratemaps by gradient descent (App. K).

    Parameters
    ----------
    n_units : int
        Number of recorded units, i.e. the dimension of the population vector.
    commutator_weight : float, optional
        Weight on the `|| [Gx, Gy] ||^2` penalty, `gamma` in Appendix K.
    lr : float, optional
        Adam learning rate.
    enforce_skew : bool, optional
        Parameterise `G = A - A^T` so the generators are exactly skew-symmetric
        and transport is exactly orthogonal, as Appendix K does. Set False to
        optimise the generators directly, which fits the training domain about
        as well but does not preserve `||p||` when extrapolating.
    seed : int, optional
        Seeds both the initialisation and the displacement sampling.

    Attributes
    ----------
    Gx, Gy : torch.Tensor
        The generators, shape (n_units, n_units). Derived from the underlying
        parameters when `enforce_skew`, so read them rather than assigning.
    loss_history : list[float]
        Total loss per step, populated by `fit`.
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
        """Apply exp(dx Gx + dy Gy) to a batch of population vectors (Eq. 1).

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
        return torch.bmm(Q, p0[..., None]).squeeze(-1)

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
        """Fit the generators to ratemaps.

        Parameters
        ----------
        ratemaps : torch.Tensor
            Ratemaps indexed **(bins_x, bins_y, n_units)** — position first, so
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
        """Generate ratemaps by transporting p0 across a grid of displacements.

        With `extent=1` this reconstructs the training domain; with `extent=2`
        it extrapolates to an arena twice the size, which is the Appendix K
        result — nothing in the fit saw those displacements.

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
            Ratemaps of shape (n_units, n_bins, n_bins), indexed
            (unit, bins_x, bins_y) to match `analysis.compute_ratemaps`.
        """
        axis = torch.arange(n_bins) - n_bins // 2
        gx, gy = torch.meshgrid(axis, axis, indexing="ij")
        steps = torch.stack([gx.ravel(), gy.ravel()], dim=1)
        dr = extent * pixels_to_arena(steps, n_bins, arena_size)

        with torch.no_grad():
            p = self.transport(p0.expand(len(dr), -1), dr)
        return p.T.reshape(self.n_units, n_bins, n_bins)


def generator_diagnostics(fit: GeneratorFit) -> dict[str, float]:
    """Measure how far the fitted generators are from the framework's ideal.

    With `enforce_skew` the symmetric fraction is 0 by construction and this is
    a consistency check; with `enforce_skew=False` it is a measurement, and the
    contrast is what shows why Appendix K reparameterises rather than hoping.

    Parameters
    ----------
    fit : GeneratorFit
        A fitted instance.

    Returns
    -------
    dict[str, float]
        `commutator_rms`, and per generator `*_rms` alongside
        `*_symmetric_fraction` — the RMS of `G + G^T` relative to that of `G`,
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
