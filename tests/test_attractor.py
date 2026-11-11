"""Tests for the CANN bridge (Sec. 3.1, Appendix J.1).

The claim these check is that the exponential map is not merely analogous to
attractor dynamics but is exactly its on-manifold solution.
"""

from itertools import pairwise

import numpy as np
import scipy

from expmap.attractor import (
    exponential_map_trajectory,
    hypersphere_step,
    lyapunov_energy,
    positive_definite_matrix,
    simulate,
    skew_symmetric_matrix,
)


class TestMatrices:
    def test_skew_symmetric(self):
        U = skew_symmetric_matrix(16, seed=0)
        np.testing.assert_allclose(U, -U.T, atol=1e-12)

    def test_skew_quadratic_form_vanishes(self):
        """z^T U z == 0 is why the transport term drops out of the energy."""
        U = skew_symmetric_matrix(16, seed=0)
        rng = np.random.default_rng(1)
        for _ in range(5):
            z = rng.standard_normal(16)
            assert abs(z @ U @ z) < 1e-10

    def test_positive_definite(self):
        M = positive_definite_matrix(16, seed=0)
        np.testing.assert_allclose(M, M.T, atol=1e-12)
        assert np.linalg.eigvalsh(M).min() > 0

    def test_seeds_reproduce(self):
        np.testing.assert_array_equal(skew_symmetric_matrix(8, seed=3), skew_symmetric_matrix(8, seed=3))
        assert not np.allclose(skew_symmetric_matrix(8, seed=3), skew_symmetric_matrix(8, seed=4))


class TestLyapunovEnergy:
    def test_zero_on_the_manifold(self):
        z = np.array([0.6, 0.8])  # unit norm
        np.testing.assert_allclose(lyapunov_energy(z), 0.0, atol=1e-12)

    def test_matches_equation_8(self):
        """E = (||z||^2 - 1)^2 / 2, with the factor of 1/2 of Eq. 8."""
        z = np.array([1.0, 1.0])  # ||z||^2 = 2
        np.testing.assert_allclose(lyapunov_energy(z), 0.5)

    def test_trajectory_shape(self):
        assert lyapunov_energy(np.zeros((7, 4))).shape == (7,)


class TestDynamics:
    def test_energy_is_non_increasing(self):
        """The core Lyapunov claim: Edot <= 0 everywhere off the manifold.

        Asserted while the state is genuinely off-manifold. The discrete
        trajectory does not stay monotone all the way down, for the reason
        `test_euler_settles_at_an_offset_linear_in_dt` pins: forward Euler
        parks the state slightly outside the sphere, so the energy dips to
        ~1e-15 and comes back up. That is the integrator, not the model.
        """
        n = 128
        U = skew_symmetric_matrix(n, seed=0)
        M = positive_definite_matrix(n, seed=1)
        z0 = np.random.default_rng(2).uniform(-1, 1, n) * 0.1

        energy = lyapunov_energy(simulate(z0, U, M, dt=5e-5, T=0.03))

        off_manifold = energy[:-1] > 1e-8
        assert off_manifold.sum() > 100, "too few off-manifold points to be a real test"
        assert np.all(np.diff(energy)[off_manifold] <= 0)

    def test_converges_to_unit_hypersphere(self):
        n = 128
        U = skew_symmetric_matrix(n, seed=0)
        M = positive_definite_matrix(n, seed=1)
        z0 = np.random.default_rng(2).uniform(-1, 1, n) * 0.1

        trajectory = simulate(z0, U, M, dt=5e-5, T=0.03)
        assert np.linalg.norm(trajectory[0]) < 0.7  # starts well inside
        np.testing.assert_allclose(np.linalg.norm(trajectory[-1]), 1.0, atol=1e-4)

    def test_euler_settles_at_an_offset_linear_in_dt(self):
        """Forward Euler parks the state just outside the manifold.

        A single Euler step on the transport term inflates the norm, since
        ||z + dt U z||^2 = ||z||^2 + dt^2 ||U z||^2 exactly (the cross term
        vanishes because z^T U z = 0 for skew U). The attractor pulls back at a
        rate proportional to dt, so the two balance at ||z|| - 1 = O(dt).

        Pinned because it sets the accuracy floor of every figure built on this
        simulation: the state is never exactly on the manifold, and refining dt,
        not running longer, is what tightens it.

        T must be long enough for the offset to *settle* before the scaling is
        meaningful. At T = 0.03 the finest dt has not equilibrated and the state
        is still approaching from inside, which gives a negative offset and a
        nonsense ratio; by T = 0.06 all three have settled.
        """
        n = 128
        U = skew_symmetric_matrix(n, seed=0)
        M = positive_definite_matrix(n, seed=1)
        z0 = np.random.default_rng(2).uniform(-1, 1, n) * 0.1

        offsets = [np.linalg.norm(simulate(z0, U, M, dt, T=0.06)[-1]) - 1 for dt in (1e-4, 5e-5, 2.5e-5)]

        assert all(offset > 0 for offset in offsets), "Euler should overshoot outward"
        for coarse, fine in pairwise(offsets):
            assert 1.8 < coarse / fine < 2.2, f"expected ~2x per halving of dt, got {coarse / fine:.2f}"

    def test_on_manifold_step_is_pure_transport(self):
        """sigma(z*) = 0 on the manifold, so zdot reduces to U z / tau."""
        n = 16
        U = skew_symmetric_matrix(n, seed=0)
        M = positive_definite_matrix(n, seed=1)

        z_star = np.random.default_rng(2).standard_normal(n)
        z_star /= np.linalg.norm(z_star)

        np.testing.assert_allclose(hypersphere_step(z_star, U, M), U @ z_star, atol=1e-12)

    def test_tau_scales_the_flow(self):
        n = 16
        U = skew_symmetric_matrix(n, seed=0)
        M = positive_definite_matrix(n, seed=1)
        z = np.random.default_rng(2).standard_normal(n)

        np.testing.assert_allclose(hypersphere_step(z, U, M, tau=4.0), hypersphere_step(z, U, M, tau=1.0) / 4.0)


class TestExponentialMapTrajectory:
    def test_matches_expm_per_step(self):
        """The incremental form must equal the naive one-expm-per-step version."""
        n, steps, dt, star = 24, 60, 1e-3, 17
        U = skew_symmetric_matrix(n, seed=0)
        z_star = np.random.default_rng(1).standard_normal(n)

        fast = exponential_map_trajectory(U, z_star, steps, dt, star)
        naive = np.array([scipy.linalg.expm(U * ((i - star) * dt)) @ z_star for i in range(steps)])

        np.testing.assert_allclose(fast, naive, atol=1e-10)

    def test_passes_through_z_star(self):
        n, steps, dt, star = 24, 60, 1e-3, 17
        U = skew_symmetric_matrix(n, seed=0)
        z_star = np.random.default_rng(1).standard_normal(n)

        predicted = exponential_map_trajectory(U, z_star, steps, dt, star)
        np.testing.assert_allclose(predicted[star], z_star, atol=1e-12)

    def test_tau_rescales_time(self):
        """expm(t U / tau): doubling tau halves the effective transport."""
        n, steps, dt, star = 16, 40, 1e-3, 0
        U = skew_symmetric_matrix(n, seed=0)
        z_star = np.random.default_rng(1).standard_normal(n)

        slow = exponential_map_trajectory(U, z_star, steps, dt, star, tau=2.0)
        fast = exponential_map_trajectory(U, z_star, steps, 2 * dt, star, tau=4.0)
        np.testing.assert_allclose(slow, fast, atol=1e-10)


class TestNetworkFollowsTheExponentialMap:
    def test_simulated_flow_matches_closed_form_once_on_manifold(self):
        """The headline claim of Sec. 3.1, end to end.

        Simulate the full nonlinear network, find where it first reaches the
        manifold, and check that from there the trajectory is the exponential
        map of that point.
        """
        n, dt, T = 128, 5e-5, 0.03
        U = skew_symmetric_matrix(n, seed=0)
        M = positive_definite_matrix(n, seed=1)
        z0 = np.random.default_rng(2).uniform(-1, 1, n) * 0.1

        trajectory = simulate(z0, U, M, dt, T)
        steps = len(trajectory)
        energy = lyapunov_energy(trajectory)

        on_manifold = np.flatnonzero(energy < 1e-8)
        assert on_manifold.size, "network did not reach the manifold"
        star = int(on_manifold[0])

        predicted = exponential_map_trajectory(U, trajectory[star], steps, dt, star)

        # Compare only from t* onward: before it the state is still off-manifold,
        # where the attractor term is active and the closed form does not apply.
        deviation = np.linalg.norm(trajectory[star:] - predicted[star:], axis=1)
        assert deviation.max() < 1e-3, f"max deviation {deviation.max():.2e}"
