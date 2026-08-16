"""Continuous attractor network dynamics underlying the exponential map.

The exponential map is not only an algebraic construction: it is the on-manifold
dynamics of a gain-modulated continuous attractor neural network (Sec. 3.1,
Appendix J.1). This module implements that network so the claim can be checked
rather than asserted.

The network state z in R^N evolves as

    tau * zdot = U(v) z - M sigma(z),      sigma(z) = (||z||^2 - 1) z     (Eq. 7)

with U(v) skew-symmetric and M positive definite. The Lyapunov energy

    E(z) = (||z||^2 - 1)^2 / 2                                           (Eq. 8)

is non-increasing, because skew-symmetry makes the transport term z^T U z vanish
identically and positive definiteness makes the remaining term negative:

    tau * Edot = -2 (||z||^2 - 1)^2 z^T M z <= 0

So the flow converges to the unit hypersphere without a topology being imposed.
Once on it sigma(z*) = 0, the attractor term drops out, and what is left is pure
transport, `tau zdot* = U(v) z*`, whose solution is the exponential map
`z*(t) = expm(t U(v) / tau) z*(0)`.

This module is numpy, like `analysis`; the generator construction in `model` and
`generators` is torch.
"""

import numpy as np
import scipy


def skew_symmetric_matrix(n: int, seed: int | None = None) -> np.ndarray:
    """Sample a random skew-symmetric transport matrix U.

    Built as A - A.T from a Gaussian A, which is skew-symmetric by construction.
    Skew-symmetry is what makes z^T U z vanish and hence the energy
    non-increasing; it is the same condition that makes the algebraic generators
    equinorm (Sec. 2.3).

    Parameters
    ----------
    n : int
        Matrix dimension.
    seed : int, optional
        Random seed.

    Returns
    -------
    numpy.ndarray
        Skew-symmetric matrix of shape (n, n).
    """
    rng = np.random.default_rng(seed)
    A = rng.standard_normal((n, n))
    return A - A.T


def positive_definite_matrix(n: int, seed: int | None = None) -> np.ndarray:
    """Sample a random positive definite attractor matrix M.

    Built as A.T @ A from a Gaussian A, which is positive definite almost
    surely. Positive definiteness is what makes the energy strictly decrease off
    the manifold.

    Parameters
    ----------
    n : int
        Matrix dimension.
    seed : int, optional
        Random seed.

    Returns
    -------
    numpy.ndarray
        Positive definite matrix of shape (n, n).
    """
    rng = np.random.default_rng(seed)
    A = rng.standard_normal((n, n))
    return A.T @ A


def lyapunov_energy(z: np.ndarray) -> np.ndarray:
    """Lyapunov energy E = (||z||^2 - 1)^2 / 2 (Eq. 8).

    Measures deviation from the unit hypersphere, and is zero exactly on it.

    Parameters
    ----------
    z : numpy.ndarray
        State vector of shape (N,) or a trajectory of shape (steps, N).

    Returns
    -------
    numpy.ndarray
        Scalar energy, or shape (steps,) for a trajectory.
    """
    return (np.sum(z**2, axis=-1) - 1) ** 2 / 2


def hypersphere_step(z: np.ndarray, U: np.ndarray, M: np.ndarray, tau: float = 1.0) -> np.ndarray:
    """Evaluate zdot for the hypersphere attractor network (Eq. 7).

    Parameters
    ----------
    z : numpy.ndarray
        Current state, shape (N,).
    U : numpy.ndarray
        Skew-symmetric transport matrix, shape (N, N).
    M : numpy.ndarray
        Positive definite attractor matrix, shape (N, N).
    tau : float, optional
        Effective time constant. Grid scale is predicted to scale with it
        (Appendix J.3).

    Returns
    -------
    numpy.ndarray
        Time derivative zdot, shape (N,).
    """
    sigma = (z @ z - 1) * z
    return (U @ z - M @ sigma) / tau


def simulate(
    z0: np.ndarray,
    U: np.ndarray,
    M: np.ndarray,
    dt: float,
    T: float,
    tau: float = 1.0,
) -> np.ndarray:
    """Integrate the attractor dynamics with forward Euler.

    Euler is what the figure used and is adequate here because the flow is
    strongly contracting onto the manifold; `dt` must nonetheless be small
    relative to the largest eigenvalue of U, which grows like sqrt(N) for
    `skew_symmetric_matrix`.

    The state settles just *outside* the manifold rather than on it. A single
    Euler step on the transport term inflates the norm exactly, since
    `||z + dt U z||^2 = ||z||^2 + dt^2 ||U z||^2` — the cross term vanishes
    because `z^T U z = 0` for skew U — and that balances the inward pull at
    `||z|| - 1 = O(dt)`. At N = 128 and dt = 5e-5 the offset is about 1e-5.
    It shrinks with dt, not with longer integration, and it sets the accuracy
    floor of anything built on this simulation.

    Parameters
    ----------
    z0 : numpy.ndarray
        Initial state, shape (N,).
    U, M : numpy.ndarray
        Transport and attractor matrices, shape (N, N).
    dt : float
        Integration step.
    T : float
        Total simulated time. The number of steps is int(T / dt).
    tau : float, optional
        Effective time constant.

    Returns
    -------
    numpy.ndarray
        Trajectory of shape (int(T / dt), N), starting at z0.
    """
    steps = int(T / dt)
    trajectory = np.zeros((steps, len(z0)))
    z = z0.copy()
    trajectory[0] = z
    for t in range(1, steps):
        z = z + dt * hypersphere_step(z, U, M, tau)
        trajectory[t] = z
    return trajectory


def exponential_map_trajectory(
    U: np.ndarray,
    z_star: np.ndarray,
    steps: int,
    dt: float,
    star_index: int,
    tau: float = 1.0,
) -> np.ndarray:
    """Predicted on-manifold trajectory expm((t - t*) U / tau) z*.

    This is the closed-form solution the network is compared against: once the
    state reaches the manifold at time t*, pure transport should carry it along
    the exponential map (Appendix J.1).

    Evaluated incrementally from a single matrix exponential. Because U is
    skew-symmetric, `E = expm(dt U / tau)` is orthogonal, so stepping forward is
    a multiplication by E and stepping backward is a multiplication by E.T. That
    replaces one `expm` per timestep with one in total — about 90x faster at
    N = 128 over 600 steps, and identical to machine precision.

    Parameters
    ----------
    U : numpy.ndarray
        Skew-symmetric transport matrix, shape (N, N).
    z_star : numpy.ndarray
        First state found on the manifold, shape (N,).
    steps : int
        Number of timesteps to predict.
    dt : float
        Integration step, matching the simulation.
    star_index : int
        Index of `z_star` within the trajectory, i.e. t* = star_index * dt.
    tau : float, optional
        Effective time constant.

    Returns
    -------
    numpy.ndarray
        Predicted trajectory of shape (steps, N).
    """
    E = scipy.linalg.expm(U * dt / tau)

    predicted = np.zeros((steps, len(z_star)))
    z = z_star.copy()
    for i in range(star_index, steps):
        predicted[i] = z
        z = E @ z

    z = E.T @ z_star  # E is orthogonal, so E.T runs the flow backwards
    for i in range(star_index - 1, -1, -1):
        predicted[i] = z
        z = E.T @ z

    return predicted
