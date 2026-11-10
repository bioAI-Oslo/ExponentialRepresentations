r"""Core exponential map models for generating neural spatial representations.

This module implements the mathematical core of the exponential map framework:
the mapping from physical coordinates to neural population vectors via the
matrix exponential of generator matrices (Eq. 1 in the paper),

.. math::

    \mathbf{p}(x, y) = e^{x G_x + y G_y}\,\mathbf{p}_0.

The algebraic properties of the generators determine the navigational
properties of the representation.
"""

import torch
import torch.nn as nn


class ExponentialMap(nn.Module):
    r"""Map physical coordinates to neural population vectors via the matrix exponential.

    Implements Eq. 1, :math:`\mathbf{p}(x, y) = e^{x G_x + y G_y}\,\mathbf{p}_0`. The
    generators :math:`G_x` and :math:`G_y` encode how a displacement in physical
    space transforms the neural state. The matrix exponential composes these
    transformations and transports the origin state :math:`\mathbf{p}_0` to the
    location :math:`(x, y)`.

    .. rubric:: Path integration and translational invariance

    When the generators commute, :math:`[G_x, G_y] = 0`, path integration is exact
    and path independent (Sec. 2.2). When the generators are also
    skew-symmetric, :math:`G^\mathsf{T} = -G`, the matrix exponential is orthogonal and
    preserves the norm of the population vector everywhere. The similarity
    between two population vectors then depends only on their displacement
    (Sec. 2.3).

    Parameters
    ----------
    Gx : torch.Tensor
        Generator matrix for the x-direction, shape (N, N).
    Gy : torch.Tensor
        Generator matrix for the y-direction, shape (N, N).
    device : str or torch.device, optional
        Device to place the generators on. The generators are registered
        buffers, so ``model.to(...)`` works afterwards. ``forward`` reads the device
        from ``Gx`` and does not cache it.
    """

    Gx: torch.Tensor
    Gy: torch.Tensor

    def __init__(self, Gx: torch.Tensor, Gy: torch.Tensor, device: str | torch.device = "cpu") -> None:
        super().__init__()
        self.register_buffer("Gx", Gx.detach().clone().to(device))
        self.register_buffer("Gy", Gy.detach().clone().to(device))

    def forward(self, positions: torch.Tensor, p0: torch.Tensor) -> torch.Tensor:
        r"""Generate spatial representations at the given positions.

        Computes :math:`\mathbf{p}(x, y) = e^{x G_x + y G_y}\,\mathbf{p}_0` for each
        position in the batch.

        Parameters
        ----------
        positions : torch.Tensor
            Spatial coordinates, shape (batch_size, 2), in the dtype of the
            generators (float32 for every constructed model). Each row is (x, y).
        p0 : torch.Tensor
            Origin state, shape (N,).

        Returns
        -------
        torch.Tensor
            Population vectors at each position, shape (batch_size, N).
        """
        positions = positions.to(self.Gx.device)
        p0 = p0.to(self.Gx.device)

        Q = torch.matrix_exp(positions[:, 0, None, None] * self.Gx + positions[:, 1, None, None] * self.Gy)
        return Q @ p0


class RemappingModel(ExponentialMap):
    r"""Exponential map model extended with a context generator for remapping.

    Adds a third generator :math:`G_s` for a scalar context signal :math:`s`, so
    that :math:`\mathbf{p}(x, y, s) = e^{x G_x + y G_y + s G_s}\,\mathbf{p}_0`. A
    large context shift
    makes the representations of two contexts nearly orthogonal, while the
    spatial similarity within each context is unchanged (Sec. 2.5). This
    orthogonalization acts as a hyperdimensional binding operation
    (Appendix L.2).

    Parameters
    ----------
    Gx : torch.Tensor
        Generator matrix for the x-direction, shape (N, N).
    Gy : torch.Tensor
        Generator matrix for the y-direction, shape (N, N).
    Gs : torch.Tensor
        Generator matrix for the context dimension, shape (N, N).
    device : str or torch.device, optional
        Device to place the generators on.
    """

    Gs: torch.Tensor

    def __init__(
        self, Gx: torch.Tensor, Gy: torch.Tensor, Gs: torch.Tensor, device: str | torch.device = "cpu"
    ) -> None:
        super().__init__(Gx, Gy, device=device)
        self.register_buffer("Gs", Gs.detach().clone().to(device))

    def forward(self, positions: torch.Tensor, p0: torch.Tensor, context: torch.Tensor | float = 0.0) -> torch.Tensor:
        r"""Generate context-dependent spatial representations.

        Computes :math:`\mathbf{p}(x, y, s) = e^{x G_x + y G_y + s G_s}\,\mathbf{p}_0`
        for each position and context value. When queried with different context signals,
        the model produces orthogonal spatial maps for the same physical
        environment (global remapping).

        Parameters
        ----------
        positions : torch.Tensor
            Spatial coordinates, shape (batch_size, 2), in the dtype of the
            generators (float32 for every constructed model). Each row is (x, y).
        p0 : torch.Tensor
            Origin state, shape (N,).
        context : float or torch.Tensor, optional
            Context signal. Either a scalar (Python float or 0-d tensor) applied
            to the whole batch, or a tensor of shape (batch_size,) with one
            value per position. Cast to the dtype of ``positions``.

        Returns
        -------
        torch.Tensor
            Population vectors at each position and context, shape (batch_size, N).
        """
        device = self.Gx.device
        positions = positions.to(device)
        p0 = p0.to(device)

        context = torch.as_tensor(context, dtype=positions.dtype, device=device)
        if context.dim() == 0:
            context = context.expand(positions.shape[0])

        Q = torch.matrix_exp(
            positions[:, 0, None, None] * self.Gx
            + positions[:, 1, None, None] * self.Gy
            + context[:, None, None] * self.Gs
        )
        return Q @ p0
