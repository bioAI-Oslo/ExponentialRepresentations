"""Core exponential map models for generating neural spatial representations.

This module implements the mathematical core of the exponential map framework:
the mapping from physical coordinates to neural population vectors via the
matrix exponential of generator matrices (Eq. 1 in the paper).

    p(x, y) = exp(x * Gx + y * Gy) * p0

The algebraic properties of the generator matrices directly determine the
navigational capabilities of the resulting representation.
"""

import torch
import torch.nn as nn


class ExponentialMap(nn.Module):
    """Map physical coordinates to neural population vectors via the matrix exponential.

    Implements the fundamental equation p(x, y) = exp(x * Gx + y * Gy) * p0,
    where Gx and Gy are generator matrices encoding how displacements in
    physical space translate into transformations in the high-dimensional
    neural state space. The matrix exponential composes these transformations
    to transport the initial state p0 to any target location (x, y).

    Path integration and translational invariance
    ----------------------------------------------
    When the generators commute ([Gx, Gy] = 0), path integration is exact
    and route-invariant (Sec. 2.2). When additionally the generators are
    skew-symmetric (G^T = -G), the matrix exponential produces orthogonal
    transformations that preserve the norm of the population vector everywhere,
    yielding translationally invariant similarity structures (Sec. 2.3).

    Parameters
    ----------
    Gx : torch.Tensor
        Generator matrix for the x-direction, shape (N, N).
    Gy : torch.Tensor
        Generator matrix for the y-direction, shape (N, N).
    device : str, optional
        Torch device to place the generators on. The generators are registered
        buffers, so `model.to(...)` afterwards works as usual — `forward`
        reads the device off `Gx` rather than caching it.
    """

    def __init__(self, Gx: torch.Tensor, Gy: torch.Tensor, device: str = "cpu") -> None:
        super().__init__()
        self.register_buffer("Gx", Gx.detach().clone().to(device))
        self.register_buffer("Gy", Gy.detach().clone().to(device))

    def forward(self, positions: torch.Tensor, p0: torch.Tensor) -> torch.Tensor:
        """Generate spatial representations at the given positions.

        Computes p(x, y) = exp(x * Gx + y * Gy) * p0 for each position
        in the batch. Uses torch.matrix_exp for numerically stable
        exponentiation of the generator linear combination.

        Parameters
        ----------
        positions : torch.Tensor
            Spatial coordinates, shape (batch_size, 2). Each row is (x, y).
        p0 : torch.Tensor
            Initial state vector, shape (N,).

        Returns
        -------
        torch.Tensor
            Population vectors at each position, shape (batch_size, N).
        """
        positions = positions.to(self.Gx.device)
        p0 = p0.to(self.Gx.device)

        # Linear combination of generators weighted by spatial coordinates
        Q = torch.matrix_exp(positions[:, 0, None, None] * self.Gx + positions[:, 1, None, None] * self.Gy)

        # Transport initial state to each target location
        p = torch.tensordot(Q, p0, dims=[(-1,), (-1,)])
        return p


class RemappingModel(ExponentialMap):
    """Exponential map model extended with a context generator for remapping.

    Adds a third generator Gs that maps a scalar context signal s into an
    orthogonal transformation, implementing context-dependent remapping via
    p(x, y, s) = exp(x * Gx + y * Gy + s * Gs) * p0. When the context shift
    is sufficiently large, the representations become nearly orthogonal across
    contexts while preserving spatial similarity structure within each context
    (Sec. 2.5). This orthogonalization functions as a hyperdimensional binding
    operation (Appendix L.2).

    Parameters
    ----------
    Gx : torch.Tensor
        Generator matrix for the x-direction, shape (N, N).
    Gy : torch.Tensor
        Generator matrix for the y-direction, shape (N, N).
    Gs : torch.Tensor
        Generator matrix for the context dimension, shape (N, N).
    device : str, optional
        Torch device for computation.
    """

    def __init__(self, Gx: torch.Tensor, Gy: torch.Tensor, Gs: torch.Tensor, device: str = "cpu") -> None:
        super().__init__(Gx, Gy, device=device)
        self.register_buffer("Gs", Gs.detach().clone().to(device))

    def forward(self, positions: torch.Tensor, p0: torch.Tensor, context: torch.Tensor | float = 0.0) -> torch.Tensor:
        """Generate context-dependent spatial representations.

        Computes p(x, y, s) = exp(x * Gx + y * Gy + s * Gs) * p0 for each
        position and context value. When queried with different context signals,
        the model produces orthogonal spatial maps for the same physical
        environment (global remapping).

        Parameters
        ----------
        positions : torch.Tensor
            Spatial coordinates, shape (batch_size, 2). Each row is (x, y).
        p0 : torch.Tensor
            Initial state vector, shape (N,).
        context : torch.Tensor or float
            Context signal. Either a scalar (applied uniformly) or a tensor
            of shape (batch_size,) for per-sample context values.

        Returns
        -------
        torch.Tensor
            Population vectors at each position and context, shape (batch_size, N).
        """
        device = self.Gx.device
        positions = positions.to(device)
        p0 = p0.to(device)

        # as_tensor covers python scalars, numpy scalars and tensors alike;
        # isinstance(context, float) missed numpy.float32.
        context = torch.as_tensor(context, dtype=positions.dtype, device=device)
        if context.dim() == 0:
            context = context.expand(positions.shape[0])

        Q = torch.matrix_exp(
            positions[:, 0, None, None] * self.Gx
            + positions[:, 1, None, None] * self.Gy
            + context[:, None, None] * self.Gs
        )

        p = torch.tensordot(Q, p0, dims=[(-1,), (-1,)])
        return p
