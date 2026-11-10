r"""Hyperdimensional computing operations for exponential map representations.

Bundling, binding, querying and gradient-based navigation with the algebra of
the exponential map (Sec. 3.3, Appendix L). Bundling superposes the population
vectors of target locations :math:`\mathbf{r}_i` into one memory,

.. math::

    \mathbf{p}_R(s) = \frac{1}{N_R} \sum_{i=1}^{N_R} \mathbf{p}(\mathbf{r}_i, s),

and a query at :math:`\mathbf{r}` is the inner product
:math:`C_R(\mathbf{r}, s) = \mathbf{p}(\mathbf{r}, s)^\mathsf{T} \mathbf{p}_R(s)`,
a sum of similarity bumps centered on the targets. Binding is the operation that
makes vectors nearly orthogonal to their constituents, and here it is remapping:
the context shift :math:`e^{\Delta s\, G_s}` makes the maps of dissimilar
contexts nearly orthogonal, so the memories of several contexts can be bundled
into one composite and a query in one context retrieves its own targets while
the others cancel.
"""

import numpy as np
import torch


def bundle(representations: torch.Tensor) -> torch.Tensor:
    """Aggregate population vectors into a composite memory via superposition.

    Bundling computes the element-wise mean of a set of population vectors,
    forming a composite memory that remains similar to each of its
    constituents. When the pairwise similarity approximates a Gaussian,
    the resulting composite produces a smooth mixture-of-Gaussians reward
    landscape suitable for gradient-based navigation (Appendix L.1).

    Parameters
    ----------
    representations : torch.Tensor
        Population vectors to bundle, shape (n_items, N).

    Returns
    -------
    torch.Tensor
        Composite memory vector, shape (N,).
    """
    return representations.mean(dim=0)


def bind(memory_vectors: list[torch.Tensor]) -> torch.Tensor:
    """Superpose per-context memories into one multi-context composite.

    Sums memory vectors that were each bundled within a different context.
    This is *not* the binding operation itself: in this framework binding is
    remapping, the orthogonalizing transform exp(s * Gs) that ``RemappingModel``
    applies when the representations are generated (Appendix L.2, Eq. 9).
    Because that transform has already decorrelated the contexts, the summed
    composite holds several reward maps at once and each is recovered by
    querying with its own context value s.

    Parameters
    ----------
    memory_vectors : list[torch.Tensor]
        List of bundled memory vectors, one per context. Each has shape (N,).

    Returns
    -------
    torch.Tensor
        Composite multi-context memory vector, shape (N,).
    """
    return torch.stack(memory_vectors).sum(dim=0)


def query(current_state: torch.Tensor, memory: torch.Tensor | np.ndarray) -> torch.Tensor:
    r"""Compute similarity between a current state and a stored memory.

    Returns the inner product :math:`\mathbf{p}(\mathbf{r}, s)^\mathsf{T}\mathbf{p}_R(s)`, which measures how
    similar the current location's representation is to the composite
    memory. For similarity-preserving representations, this score
    approximates the sum of Gaussian bumps centered at stored locations
    (Appendix L.1).

    Parameters
    ----------
    current_state : torch.Tensor
        Population vector at the current location, shape (N,) or
        (batch, N) for multiple query locations.
    memory : torch.Tensor or numpy.ndarray
        Composite memory vector, shape (N,). A numpy memory, as bundled from
        ``analysis.batch_inference`` output, is accepted like in
        ``similarity_gradient``.

    Returns
    -------
    torch.Tensor
        Similarity score(s). Scalar if current_state is 1D, or
        shape (batch,) if current_state is 2D.
    """
    memory = torch.as_tensor(memory, dtype=current_state.dtype, device=current_state.device)
    return current_state @ memory


def similarity_gradient(
    model: torch.nn.Module,
    position: torch.Tensor,
    memory: torch.Tensor | np.ndarray,
    p0: torch.Tensor,
    context: torch.Tensor | float | None = None,
) -> torch.Tensor:
    r"""Compute the gradient of the similarity landscape in closed form.

    The similarity score is :math:`C_R(\mathbf{r}, s) = \mathbf{p}(\mathbf{r}, s)
    \cdot \mathbf{p}_R(s)`, and differentiating Eq. 1 gives the gradient exactly,

    .. math::

        \frac{\partial C_R}{\partial x} = (G_x\,\mathbf{p}(\mathbf{r}, s)) \cdot \mathbf{p}_R(s),
        \qquad
        \frac{\partial C_R}{\partial y} = (G_y\,\mathbf{p}(\mathbf{r}, s)) \cdot \mathbf{p}_R(s),

    because :math:`\partial_x e^{x G_x + y G_y + s G_s} = G_x e^{x G_x + y G_y + s G_s}`
    when all three generators commute, as they do by construction (shared
    :math:`R`). This gradient supports gradient-ascent navigation
    toward stored rewards, :math:`\mathbf{r}_{t+1} = \mathbf{r}_t + \eta\,\nabla C_R`
    (Appendix L.1).

    Evaluating the closed form costs one matrix exponential rather than the
    four a central difference would need, is exact, and removes the step-size
    parameter.

    Parameters
    ----------
    model : torch.nn.Module
        An ExponentialMap or RemappingModel instance.
    position : torch.Tensor
        Current position, shape (2,).
    memory : torch.Tensor or numpy.ndarray
        Composite memory vector, shape (N,).
    p0 : torch.Tensor
        Initial state vector, shape (N,).
    context : float or torch.Tensor, optional
        Context value for RemappingModel.

    Returns
    -------
    torch.Tensor
        Gradient vector (dCR/dx, dCR/dy), shape (2,), on the device of
        ``position`` so that the next step ``position + eta * grad`` needs no
        transfer.
    """
    positions = position.to(torch.float32).reshape(1, 2)

    with torch.no_grad():
        p = model(positions, p0, context=context) if context is not None else model(positions, p0)

    p = p[0]
    memory = torch.as_tensor(memory, dtype=p.dtype, device=p.device)
    return torch.stack([(model.Gx @ p) @ memory, (model.Gy @ p) @ memory]).to(position.device)
