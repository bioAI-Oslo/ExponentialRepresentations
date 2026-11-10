r"""Local learning rule for skew-symmetric connectivity (Appendix J.4).

The generators need not be hard-coded. Appendix J.4 proposes a local, biologically
plausible rule driven by time-lagged antisymmetric correlations,

.. math::

    \Delta W_{ij} = \eta \left( r_i(t)\, r_j(t + \tau_{\mathrm{lag}})
    - r_j(t)\, r_i(t + \tau_{\mathrm{lag}}) \right) - \alpha W_{ij},

and shows it drives :math:`W` toward the skew-symmetry that Sec. 2.3 requires.

Decomposing :math:`W` into :math:`S = W + W^T` and :math:`A = W - W^T` separates
the dynamics. The Hebbian term :math:`H` is antisymmetric by construction,
so it contributes nothing to :math:`S`, whose evolution is the decay alone,

.. math::

    \Delta S = -\alpha S \quad\Longrightarrow\quad S(t) = (1 - \alpha)^t S(0),

while :math:`A` is driven by :math:`H` and stabilized by the decay, giving the
steady state

.. math::

    A^* = \frac{2\eta}{\alpha} \langle H \rangle_t,

so that with :math:`S^* = 0` the weights themselves converge to
:math:`W^* = A^* / 2`.

What the rule learns is set by the time-lagged antisymmetric correlations of
the input. The decay of :math:`S` is algebra, not evidence: it holds for every
input, including white noise, where :math:`\langle H \rangle` vanishes and the
rule learns only fluctuation. The informative result is the steady state. If
the network is driven with activity transported by a generator, :math:`A`
recovers the correlations that the transport produced.

Appendix J.4 states the limit: the rule yields a skew-symmetric matrix, not the
commuting pair that a coherent spatial map needs.
"""

import numpy as np
from scipy.linalg import expm


def antisymmetric_correlation(rates: np.ndarray, lag: int = 1) -> np.ndarray:
    r"""Time-lagged antisymmetric correlation of the activity.

    Computes :math:`\langle H \rangle_t = \langle \mathbf{r}(t)\,\mathbf{r}(t+\tau)^\mathsf{T}
    - \mathbf{r}(t+\tau)\,\mathbf{r}(t)^\mathsf{T} \rangle_t` with :math:`\tau` the lag,
    the quantity the steady state is proportional to.

    Parameters
    ----------
    rates : numpy.ndarray
        Activity over time, shape (n_steps + lag, n_units).
    lag : int, optional
        Temporal lag in timesteps, at least 1.

    Returns
    -------
    numpy.ndarray
        Antisymmetric correlation matrix of shape (n_units, n_units).
    """
    early, late = rates[:-lag], rates[lag:]
    correlation = early.T @ late / len(early)
    return correlation - correlation.T


def steady_state_antisymmetric(rates: np.ndarray, eta: float, alpha: float, lag: int = 1) -> np.ndarray:
    r"""Predicted steady state :math:`A^* = (2\eta/\alpha)\langle H \rangle_t` (Appendix J.4).

    Parameters
    ----------
    rates : numpy.ndarray
        Activity over time, shape (n_steps + lag, n_units).
    eta : float
        Learning rate.
    alpha : float
        Weight decay.
    lag : int, optional
        Temporal lag in timesteps, at least 1.

    Returns
    -------
    numpy.ndarray
        Predicted antisymmetric part :math:`W - W^\mathsf{T}`, shape (n_units, n_units).
    """
    return (2 * eta / alpha) * antisymmetric_correlation(rates, lag)


def learn_connectivity(
    rates: np.ndarray,
    eta: float = 0.01,
    alpha: float = 0.001,
    lag: int = 1,
    W0: np.ndarray | None = None,
    seed: int | None = None,
) -> tuple[np.ndarray, dict[str, np.ndarray]]:
    """Run the antisymmetric Hebbian rule over an activity sequence.

    Parameters
    ----------
    rates : numpy.ndarray
        Activity over time, shape (n_steps + lag, n_units). The update is applied
        ``n_steps`` times.
    eta : float, optional
        Learning rate.
    alpha : float, optional
        Weight decay.
    lag : int, optional
        Temporal lag in timesteps, at least 1.
    W0 : numpy.ndarray, optional
        Initial weights, shape (n_units, n_units). Default: Gaussian with scale
        0.1, drawn from ``seed``.
    seed : int, optional
        Seed for the default initialization. Ignored when ``W0`` is given.

    Returns
    -------
    tuple[numpy.ndarray, dict[str, numpy.ndarray]]
        Final weights of shape (n_units, n_units), and the RMS of the symmetric
        and antisymmetric parts before every step, as ``symmetric_rms`` and
        ``antisymmetric_rms`` of shape (n_steps,).
    """
    n_units = rates.shape[1]
    W = np.random.default_rng(seed).standard_normal((n_units, n_units)) * 0.1 if W0 is None else W0

    n_steps = len(rates) - lag
    symmetric_rms = np.zeros(n_steps)
    antisymmetric_rms = np.zeros(n_steps)

    for t in range(n_steps):
        symmetric_rms[t] = np.sqrt(np.mean((W + W.T) ** 2))
        antisymmetric_rms[t] = np.sqrt(np.mean((W - W.T) ** 2))

        outer = np.outer(rates[t], rates[t + lag])
        W = W + eta * (outer - outer.T) - alpha * W

    return W, {"symmetric_rms": symmetric_rms, "antisymmetric_rms": antisymmetric_rms}


def transported_activity(
    generator: np.ndarray,
    n_steps: int,
    dt: float = 0.05,
    n_trajectories: int = 1,
    seed: int | None = None,
) -> np.ndarray:
    r"""Activity generated by transporting a state with a known skew generator.

    The input the rule is meant for: a population carried along by an
    exponential map, :math:`\mathbf{r}(t+1) = e^{\delta t\, G}\,\mathbf{r}(t)`. With this input the steady
    state is a statement about recovering the transport that produced the
    activity, not about white noise.

    Several trajectories from different starting states can be concatenated,
    since a single orbit explores only a low-dimensional slice of the state
    space and :math:`\langle H \rangle_t` from one trajectory is correspondingly
    rank-limited. The
    concatenation has one spurious transition per boundary, which is negligible
    while ``n_steps`` is much larger than the lag.

    Parameters
    ----------
    generator : numpy.ndarray
        Skew-symmetric generator, shape (n_units, n_units).
    n_steps : int
        Timesteps per trajectory.
    dt : float, optional
        Step size, so each step applies :math:`e^{\delta t\, G}`.
    n_trajectories : int, optional
        Number of trajectories to concatenate.
    seed : int, optional
        Seed for the starting states.

    Returns
    -------
    numpy.ndarray
        Activity of shape (n_trajectories * n_steps, n_units).
    """
    rng = np.random.default_rng(seed)
    step = expm(dt * generator)

    trajectories = []
    for _ in range(n_trajectories):
        r = rng.standard_normal(len(generator))
        r /= np.linalg.norm(r)

        activity = np.zeros((n_steps, len(generator)))
        for t in range(n_steps):
            activity[t] = r
            r = step @ r
        trajectories.append(activity)

    return np.concatenate(trajectories)
