"""Tests for the local learning rule (Appendix J.4).

The decay of the symmetric part is algebra, not evidence — it follows from the
Hebbian term being antisymmetric, whatever the input — so it is checked against
its closed form rather than merely observed to shrink. The claim with content is
the steady state, which is checked against structured input.
"""

import numpy as np

from expmap.plasticity import (
    antisymmetric_correlation,
    learn_connectivity,
    steady_state_antisymmetric,
    transported_activity,
)


def skew(n, seed=0):
    A = np.random.default_rng(seed).standard_normal((n, n))
    return (A - A.T) / np.sqrt(n)


def rms(x):
    return float(np.sqrt(np.mean(x**2)))


class TestAntisymmetricCorrelation:
    def test_is_antisymmetric(self):
        rates = np.random.default_rng(0).standard_normal((500, 8))
        H = antisymmetric_correlation(rates)
        np.testing.assert_allclose(H, -H.T, atol=1e-12)

    def test_vanishes_for_white_noise(self):
        """Uncorrelated input has no time-lagged structure, so <H> -> 0 as 1/sqrt(T)."""
        rng = np.random.default_rng(0)
        short = rms(antisymmetric_correlation(rng.standard_normal((1_000, 8))))
        long = rms(antisymmetric_correlation(rng.standard_normal((16_000, 8))))
        assert long < short / 2

    def test_converges_to_a_limit_for_transported_activity(self):
        """The distinction that matters is not magnitude but convergence.

        At any fixed T, white noise and transported activity give a comparable
        `<H>`; what separates them is that the transported one converges to a
        non-zero limit while the noise one keeps shrinking as 1/sqrt(T).
        """
        generator = skew(8)
        short = rms(antisymmetric_correlation(transported_activity(generator, 2000, seed=0)))
        long = rms(antisymmetric_correlation(transported_activity(generator, 8000, seed=0)))

        assert short > 1e-3
        assert abs(long - short) / short < 0.2, f"should converge, got {short:.4f} -> {long:.4f}"


class TestSymmetricDecay:
    def test_matches_the_closed_form_exactly(self):
        """dS = -alpha S, so S(t) = (1 - alpha)^t S(0) whatever the input.

        The Hebbian term is antisymmetric by construction and contributes nothing
        to S, so this is an identity — it should hold to machine precision, not
        approximately.
        """
        rng = np.random.default_rng(0)
        alpha, n_steps = 0.001, 400
        W0 = rng.standard_normal((10, 10)) * 0.1
        rates = rng.standard_normal((n_steps + 1, 10))

        _, history = learn_connectivity(rates, eta=0.01, alpha=alpha, W0=W0, record=True)

        expected = np.array([(1 - alpha) ** t * rms(W0 + W0.T) for t in range(n_steps)])
        np.testing.assert_allclose(history["symmetric_rms"], expected, rtol=1e-10)

    def test_holds_for_structured_input_too(self):
        alpha, n_steps = 0.002, 300
        W0 = np.random.default_rng(1).standard_normal((8, 8)) * 0.1
        rates = transported_activity(skew(8), n_steps=n_steps + 1, seed=0)

        _, history = learn_connectivity(rates, alpha=alpha, W0=W0, record=True)

        expected = np.array([(1 - alpha) ** t * rms(W0 + W0.T) for t in range(n_steps)])
        np.testing.assert_allclose(history["symmetric_rms"], expected, rtol=1e-10)

    def test_final_weights_are_near_skew_symmetric(self):
        """After many decay time constants the symmetric part is negligible.

        alpha = 0.001 gives a time constant of 1000 steps, so 6000 steps leaves
        (1 - alpha)^6000 ~ 2.5e-3 of the initial symmetric part.
        """
        rates = transported_activity(skew(8), n_steps=6000, seed=0)
        W, _ = learn_connectivity(rates, seed=0)
        assert rms(W + W.T) / rms(W - W.T) < 1e-2


class TestSteadyState:
    def test_converges_to_the_predicted_steady_state(self):
        """A* = (2 eta / alpha) <H>, checked where <H> is a real signal."""
        eta, alpha = 0.01, 0.001
        rates = transported_activity(skew(16), n_steps=5000, seed=0)

        W, _ = learn_connectivity(rates, eta=eta, alpha=alpha, seed=0)
        predicted = steady_state_antisymmetric(rates, eta=eta, alpha=alpha)

        correlation = np.corrcoef((W - W.T).ravel(), predicted.ravel())[0, 1]
        assert correlation > 0.95, f"correlation with the steady state: {correlation:.3f}"

    def test_scales_with_eta_over_alpha(self):
        rates = transported_activity(skew(8), n_steps=3000, seed=0)
        base = steady_state_antisymmetric(rates, eta=0.01, alpha=0.001)
        doubled = steady_state_antisymmetric(rates, eta=0.02, alpha=0.001)
        halved = steady_state_antisymmetric(rates, eta=0.01, alpha=0.002)

        np.testing.assert_allclose(doubled, 2 * base, rtol=1e-10)
        np.testing.assert_allclose(halved, base / 2, rtol=1e-10)

    def test_white_noise_learns_only_fluctuation(self):
        """With white noise there is no steady state to converge to.

        `A*` shrinks as 1/sqrt(T) because `<H>` does, while the learned `A` keeps
        a magnitude set by the noise. The gap between them therefore widens with
        T — which is why the original demonstration was weaker than it looked:
        the symmetric part decaying proves nothing about the input.
        """
        rng = np.random.default_rng(0)
        eta, alpha = 0.01, 0.001

        ratios = []
        for T in (2000, 32000):
            rates = rng.standard_normal((T, 12)) * 0.5
            W, _ = learn_connectivity(rates, eta=eta, alpha=alpha, seed=0)
            predicted = steady_state_antisymmetric(rates, eta=eta, alpha=alpha)
            ratios.append(rms(W - W.T) / rms(predicted))

        assert ratios[1] > 2 * ratios[0], f"gap should widen with T, got {ratios}"


class TestCaveat:
    def test_learned_generators_do_not_generally_commute(self):
        """App. J.4's own limitation: a skew matrix, not a *commuting pair*."""
        first, _ = learn_connectivity(transported_activity(skew(12, seed=0), 4000, seed=0), seed=0)
        second, _ = learn_connectivity(transported_activity(skew(12, seed=1), 4000, seed=1), seed=1)

        A, B = first - first.T, second - second.T
        assert rms(A @ B - B @ A) > 0.1 * rms(A) * rms(B)


class TestReproducibility:
    def test_seed_reproduces(self):
        rates = np.random.default_rng(0).standard_normal((500, 8))
        first, _ = learn_connectivity(rates, seed=7)
        second, _ = learn_connectivity(rates, seed=7)
        np.testing.assert_array_equal(first, second)

    def test_transported_activity_is_seeded_and_on_the_sphere(self):
        generator = skew(8)
        first = transported_activity(generator, 100, seed=3)
        np.testing.assert_array_equal(first, transported_activity(generator, 100, seed=3))
        # skew generator -> orthogonal transport -> unit norm throughout
        np.testing.assert_allclose(np.linalg.norm(first, axis=1), 1.0, atol=1e-10)

    def test_multiple_trajectories_concatenate(self):
        rates = transported_activity(skew(8), n_steps=50, n_trajectories=3, seed=0)
        assert rates.shape == (150, 8)
        assert not np.allclose(rates[0], rates[50])
