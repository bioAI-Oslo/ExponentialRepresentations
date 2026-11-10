r"""Continuous attractor network dynamics underlying the exponential map.

The exponential map is also the on-manifold dynamics of a gain-modulated
continuous attractor neural network (Sec. 3.1, Appendix J.1). This module
implements that network, so the claim can be tested numerically.

The network state :math:`\mathbf{z} \in \mathbb{R}^N` evolves as (Eq. 7)

.. math::

    \tau\dot{\mathbf{z}} = U(\mathbf{v})\,\mathbf{z} - M\sigma(\mathbf{z}),
    \qquad \sigma(\mathbf{z}) = (\lVert\mathbf{z}\rVert^2 - 1)\,\mathbf{z},

with :math:`U(\mathbf{v})` skew-symmetric and :math:`M` positive definite. The
Lyapunov energy (Eq. 8)

.. math::

    E = \tfrac{1}{2}(\lVert\mathbf{z}\rVert^2 - 1)^2

is non-increasing. Skew-symmetry makes the transport term
:math:`\mathbf{z}^T U \mathbf{z}` vanish, and positive definiteness makes the
remaining term

.. math::

    \tau\dot{E} = -2(\lVert\mathbf{z}\rVert^2 - 1)^2\,\mathbf{z}^T M \mathbf{z} \le 0,

zero on the unit hypersphere and negative everywhere else except the origin.
The flow therefore converges to the hypersphere without a predefined toroidal
topology. On the sphere :math:`\sigma(\mathbf{z}^*) = 0`, so the attractor term vanishes and pure
transport remains, :math:`\tau\dot{\mathbf{z}}^* = U(\mathbf{v})\,\mathbf{z}^*`.
Its solution is the exponential map

.. math::

    \mathbf{z}^*(t) = e^{t U(\mathbf{v}) / \tau}\,\mathbf{z}^*(0).
"""

import numpy as np
from scipy.linalg import expm


def skew_symmetric_matrix(n: int, seed: int | None = None) -> np.ndarray:
    r"""Sample a random skew-symmetric transport matrix :math:`U`.

    Built as :math:`A - A^\mathsf{T}` from a Gaussian :math:`A`. Skew-symmetry
    makes :math:`\mathbf{z}^\mathsf{T} U \mathbf{z}` vanish, so
    transport leaves the Lyapunov energy unchanged. It is the same condition that
    gives the generators equinorm representations (Sec. 2.3).

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
    r"""Sample a random positive definite attractor matrix :math:`M`.

    Built as :math:`A^\mathsf{T} A` from a Gaussian :math:`A`, which is positive definite almost
    surely. Positive definiteness makes the Lyapunov energy strictly decrease
    off the manifold.

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
    r"""Lyapunov energy :math:`E = (\lVert\mathbf{z}\rVert^2 - 1)^2 / 2` (Eq. 8).

    Measures deviation from the unit hypersphere, and is zero exactly on it.

    Parameters
    ----------
    z : numpy.ndarray
        State vector of shape (N,) or a trajectory of shape (steps, N).

    Returns
    -------
    numpy.ndarray
        Scalar energy for one state, or shape (steps,) for a trajectory.
    """
    return (np.sum(z**2, axis=-1) - 1) ** 2 / 2


def hypersphere_step(z: np.ndarray, U: np.ndarray, M: np.ndarray, tau: float = 1.0) -> np.ndarray:
    r"""Evaluate :math:`\dot{\mathbf{z}}` for the hypersphere attractor network (Eq. 7).

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
        Time derivative :math:`\dot{\mathbf{z}}`, shape (N,).
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
    r"""Integrate the attractor dynamics with forward Euler.

    Forward Euler is adequate because the flow contracts strongly onto the
    manifold. ``dt`` must still be small against the largest eigenvalue of
    :math:`U`, which grows like :math:`\sqrt{N}` for ``skew_symmetric_matrix``.

    The state settles just *outside* the manifold rather than on it. A single
    Euler step on the transport term inflates the norm exactly, since

    .. math::

        \lVert\mathbf{z} + \delta t\, U\mathbf{z}\rVert^2
        = \lVert\mathbf{z}\rVert^2 + \delta t^2 \lVert U\mathbf{z}\rVert^2

    (the cross term vanishes because :math:`\mathbf{z}^\mathsf{T} U \mathbf{z} = 0`
    for skew :math:`U`), and that balances the inward pull at
    :math:`\lVert\mathbf{z}\rVert - 1 = O(\delta t)`. At :math:`N = 128` and
    ``dt = 5e-5`` the offset is about 1e-5. It shrinks with ``dt``, not with longer
    integration, and it sets the accuracy floor of anything built on this
    simulation.

    Parameters
    ----------
    z0 : numpy.ndarray
        Initial state, shape (N,).
    U, M : numpy.ndarray
        Transport and attractor matrices, shape (N, N).
    dt : float
        Integration step.
    T : float
        Total simulated time. The number of steps is round(T / dt).
    tau : float, optional
        Effective time constant.

    Returns
    -------
    numpy.ndarray
        Trajectory of shape (round(T / dt), N), starting at z0.
    """
    steps = round(T / dt)
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
    r"""Predicted on-manifold trajectory :math:`e^{(t - t^*) U / \tau}\,\mathbf{z}^*`.

    This is the closed-form solution the network is compared against: once the
    state reaches the manifold at time :math:`t^*`, pure transport should carry
    it along the exponential map (Appendix J.1).

    Evaluated from a single matrix exponential. Because :math:`U` is
    skew-symmetric, :math:`E = e^{\delta t\, U / \tau}` is orthogonal, so stepping
    forward is a multiplication by :math:`E` and stepping backward one by
    :math:`E^\mathsf{T}`. This replaces one ``expm`` per timestep with one in total
    and agrees with the per-step form to machine precision.

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
        Index of ``z_star`` within the trajectory, so :math:`t^*` is ``star_index * dt``.
    tau : float, optional
        Effective time constant.

    Returns
    -------
    numpy.ndarray
        Predicted trajectory of shape (steps, N).
    """
    E = expm(U * dt / tau)

    predicted = np.zeros((steps, len(z_star)))
    z = z_star.copy()
    for i in range(star_index, steps):
        predicted[i] = z
        z = E @ z

    z = E.T @ z_star
    for i in range(star_index - 1, -1, -1):
        predicted[i] = z
        z = E.T @ z

    return predicted
