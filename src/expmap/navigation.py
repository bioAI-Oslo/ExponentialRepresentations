"""Hyperdimensional computing operations for exponential map representations.

Implements bundling, binding, querying, and gradient-based navigation using
the algebraic properties of the exponential map framework (Sec. 3.2,
Appendix L). The high-dimensional structure of the population vectors
ensures that randomly sampled representations are nearly orthogonal,
enabling robust superposition (bundling) and context-dependent filtering
(binding) within a single neural population.
"""

import numpy as np
import torch


def bundle(representations):
    """Aggregate population vectors into a composite memory via superposition.

    Bundling computes the element-wise mean of a set of population vectors,
    forming a composite memory that remains similar to each of its
    constituents. When the pairwise similarity approximates a Gaussian,
    the resulting composite produces a smooth mixture-of-Gaussians reward
    landscape suitable for gradient-based navigation (Appendix L.1).

    Parameters
    ----------
    representations : torch.Tensor or numpy.ndarray
        Population vectors to bundle, shape (n_items, N).

    Returns
    -------
    torch.Tensor or numpy.ndarray
        Composite memory vector, shape (N,). Same type as input.
    """
    return representations.mean(axis=0)


def bind(memory_vectors):
    """Superpose per-context memories into one multi-context composite.

    Sums memory vectors that were each bundled within a different context.
    This is *not* the binding operation itself: in this framework binding is
    remapping, the orthogonalizing transform exp(s * Gs) that `RemappingModel`
    applies when the representations are generated (Appendix L.2, Eq. 9).
    Because that transform has already decorrelated the contexts, the summed
    composite holds several reward maps at once and each is recovered by
    querying with its own context value s.

    Parameters
    ----------
    memory_vectors : list[torch.Tensor] or list[numpy.ndarray]
        List of bundled memory vectors, one per context. Each has shape (N,).

    Returns
    -------
    torch.Tensor or numpy.ndarray
        Composite multi-context memory vector, shape (N,).
    """
    if isinstance(memory_vectors[0], torch.Tensor):
        return torch.stack(memory_vectors).sum(dim=0)
    return np.stack(memory_vectors).sum(axis=0)


def query(current_state, memory):
    """Compute similarity between a current state and a stored memory.

    Returns the inner product p(r, s)^T @ pR(s), which measures how
    similar the current location's representation is to the composite
    memory. For similarity-preserving representations, this score
    approximates the sum of Gaussian bumps centered at stored locations
    (Appendix L.1).

    Parameters
    ----------
    current_state : torch.Tensor or numpy.ndarray
        Population vector at the current location, shape (N,) or
        (batch, N) for multiple query locations.
    memory : torch.Tensor or numpy.ndarray
        Composite memory vector, shape (N,).

    Returns
    -------
    torch.Tensor or numpy.ndarray
        Similarity score(s). Scalar if current_state is 1D, or
        shape (batch,) if current_state is 2D.
    """
    if isinstance(current_state, torch.Tensor):
        if current_state.dim() == 1:
            return torch.dot(current_state, memory)
        return current_state @ memory
    if current_state.ndim == 1:
        return np.dot(current_state, memory)
    return current_state @ memory


def similarity_gradient(model, position, memory, p0, context=None):
    """Compute the gradient of the similarity landscape in closed form.

    The similarity score is CR(r, s) = p(r, s) . pR(s), and differentiating
    Eq. 1 under the integral gives the gradient exactly:

        dCR/dx = (Gx p(r, s)) . pR(s),    dCR/dy = (Gy p(r, s)) . pR(s)

    because d/dx exp(x Gx + y Gy) = Gx exp(x Gx + y Gy) for commuting
    generators. This gradient supports gradient-ascent navigation toward
    stored rewards: r_{t+1} = r_t + eta * grad (Appendix L.1).

    Evaluating the closed form costs one matrix exponential rather than the
    four a central difference would need, is exact, and removes the step-size
    parameter. `tests/test_navigation.py` checks it against a finite
    difference.

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
        Gradient vector (dCR/dx, dCR/dy), shape (2,).
    """
    positions = position.to(torch.float32).reshape(1, 2)

    with torch.no_grad():
        if context is not None:
            p = model(positions, p0, context=context)
        else:
            p = model(positions, p0)

    p = p[0]
    memory = torch.as_tensor(memory, dtype=p.dtype, device=p.device)
    return torch.stack([(model.Gx @ p) @ memory, (model.Gy @ p) @ memory]).cpu()
