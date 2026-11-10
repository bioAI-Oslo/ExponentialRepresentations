import numpy as np
import pytest
import torch
from scipy.special import j0, jn_zeros

from expmap.generators import (
    bessel_module_frequencies,
    block_diag_skew,
    build_generators,
    build_generators_with_context,
    gaussian_eigenvalues,
    multiscale_wavevectors,
    roots_of_unity_eigenvalues,
)
from expmap.linalg import random_orthogonal_matrix, starting_state
from expmap.model import ExponentialMap


class TestRootsOfUnityEigenvalues:
    def test_shape(self):
        ev = roots_of_unity_eigenvalues(M=3, n_copies=2)
        assert ev.shape == (6,)

    def test_shape_single_copy(self):
        ev = roots_of_unity_eigenvalues(M=5, n_copies=1)
        assert ev.shape == (5,)

    def test_values_m3(self):
        ev = roots_of_unity_eigenvalues(M=3, n_copies=1)
        expected = np.array([0, np.pi / 3, 2 * np.pi / 3])
        np.testing.assert_allclose(ev, expected)

    def test_values_m2(self):
        ev = roots_of_unity_eigenvalues(M=2, n_copies=1)
        expected = np.array([0, np.pi / 2])
        np.testing.assert_allclose(ev, expected)

    def test_tiling(self):
        ev = roots_of_unity_eigenvalues(M=3, n_copies=3)
        single = roots_of_unity_eigenvalues(M=3, n_copies=1)
        np.testing.assert_allclose(ev, np.tile(single, 3))


class TestGaussianEigenvalues:
    def test_shape(self):
        ev = gaussian_eigenvalues(N=100, seed=42)
        assert ev.shape == (50,)

    def test_reproducibility(self):
        ev1 = gaussian_eigenvalues(N=100, seed=42)
        ev2 = gaussian_eigenvalues(N=100, seed=42)
        np.testing.assert_array_equal(ev1, ev2)

    def test_different_seeds(self):
        ev1 = gaussian_eigenvalues(N=100, seed=1)
        ev2 = gaussian_eigenvalues(N=100, seed=2)
        assert not np.allclose(ev1, ev2)

    def test_odd_N_raises(self):
        with pytest.raises(ValueError, match="even"):
            gaussian_eigenvalues(N=15, seed=0)


class TestBesselModuleFrequencies:
    def test_ratios_follow_the_bessel_zeros(self):
        """The scale is free; the *ratios* are the prediction (Fig. 6)."""
        zeros = jn_zeros(0, 6)
        k = bessel_module_frequencies(6, arena_size=6.0)
        np.testing.assert_allclose(k[1:] / k[:-1], zeros[1:] / zeros[:-1], rtol=1e-12)

    def test_spacing_ratios_decrease_toward_one(self):
        """The spacing prediction of Sec. 2.4 and App. G.

        The ratios of successive J0 zeros are 2.30, 1.57, 1.36, 1.27, ... and
        decrease toward one, so the predicted spacing ratio grows with grid
        scale. Beyond the first, the average of the low-order ratios is close
        to the mean module spacing ratio of about 1.4 reported experimentally.
        """
        k = bessel_module_frequencies(15, arena_size=6.0)
        ratios = k[1:] / k[:-1]

        np.testing.assert_allclose(ratios[:4], [2.30, 1.57, 1.36, 1.27], atol=0.005)
        assert np.all(np.diff(ratios) < 0)
        assert ratios[-1] > 1.0
        np.testing.assert_allclose(ratios[1:4].mean(), 1.40, atol=0.01)

    def test_periods_sets_the_absolute_scale(self):
        """The lowest module shows `periods` peak-to-peak spacings across the arena.

        Measured on the pattern itself: the peaks of three plane waves of
        wavenumber k at 60 degree spacing lie 4 pi / (k sqrt(3)) apart.
        """
        arena, periods = 6.0, 2.0
        k = bessel_module_frequencies(4, arena_size=arena, periods=periods)

        phi = np.arange(3) * np.pi / 3
        s = np.linspace(0, 4 * arena, 40001)
        x, y = s * np.cos(np.pi / 6), s * np.sin(np.pi / 6)  # a lattice direction
        pattern = np.cos(k[0] * (np.outer(x, np.cos(phi)) + np.outer(y, np.sin(phi)))).sum(axis=1)
        peaks = s[1:-1][(pattern[1:-1] > pattern[:-2]) & (pattern[1:-1] > pattern[2:]) & (pattern[1:-1] > 2.9)]
        grid_period = np.diff(peaks).mean()

        np.testing.assert_allclose(grid_period, 4 * np.pi / (k[0] * np.sqrt(3)), rtol=1e-6)
        np.testing.assert_allclose(2 * arena / grid_period, periods, rtol=1e-6)

    def test_increasing(self):
        k = bessel_module_frequencies(8, arena_size=3.0)
        assert np.all(np.diff(k) > 0)


class TestMultiscaleWavevectors:
    def test_shape_and_radii(self):
        k = bessel_module_frequencies(5, arena_size=6.0)
        lx, ly = multiscale_wavevectors(k, M=3, n_offsets=2)

        assert lx.shape == ly.shape == (5 * 3 * 2,)
        # every wavevector sits on its ring
        radii = np.sort(np.unique(np.round(np.hypot(lx, ly), 10)))
        np.testing.assert_allclose(radii, np.sort(k), rtol=1e-10)

    def test_orientations_are_evenly_spaced(self):
        lx, ly = multiscale_wavevectors(np.array([1.0]), M=3, n_offsets=1)
        angles = np.sort(np.arctan2(ly, lx))
        np.testing.assert_allclose(np.diff(angles), np.pi / 3, atol=1e-12)

    def test_offsets_interleave_the_orientations(self):
        """L rotated copies land halfway between, at pi / (M * L)."""
        lx, ly = multiscale_wavevectors(np.array([1.0]), M=3, n_offsets=2)
        angles = np.sort(np.arctan2(ly, lx) % np.pi)
        np.testing.assert_allclose(np.diff(angles), np.pi / 6, atol=1e-12)

    def test_metric_preservation_phasor_sum_vanishes(self):
        """Each ring must still satisfy sum_j exp(2 i phi_j) = 0 (App. E)."""
        k = bessel_module_frequencies(3, arena_size=6.0)
        lx, ly = multiscale_wavevectors(k, M=3, n_offsets=2)
        phi = np.arctan2(ly, lx)
        assert abs(np.sum(np.exp(2j * phi))) < 1e-10


class TestFourierBesselSimilarity:
    """App. G: Bessel-spaced rings make the similarity a Fourier-Bessel series."""

    @staticmethod
    def _model(K, M, n_offsets, arena=6.0, seed=0):
        k = bessel_module_frequencies(K, arena_size=arena)
        lx, ly = multiscale_wavevectors(k, M=M, n_offsets=n_offsets)
        N = 2 * len(lx)
        R = random_orthogonal_matrix(N, seed=seed)
        p0 = starting_state(R)
        Gx, Gy = build_generators(block_diag_skew(lx), block_diag_skew(ly), R)
        return ExponentialMap(Gx, Gy), p0, N, k

    def _angular_spread(self, model, p0, r, n_angles=180):
        theta = np.linspace(0, 2 * np.pi, n_angles, endpoint=False)
        positions = torch.tensor(np.stack([r * np.cos(theta), r * np.sin(theta)], axis=1), dtype=torch.float32)
        with torch.no_grad():
            origin = model(torch.zeros(1, 2), p0)[0]
            similarity = (model(positions, p0) @ origin).numpy()
        return similarity.max() - similarity.min(), similarity.mean()

    def test_offsets_suppress_angular_dependence(self):
        """M = 3 with L = 2 pushes the lowest harmonic from cos(6t) to cos(12t)."""
        one_offset, p0_one, _, _ = self._model(K=5, M=3, n_offsets=1)
        two_offsets, p0_two, _, _ = self._model(K=5, M=3, n_offsets=2)

        spread_one, _ = self._angular_spread(one_offset, p0_one, r=1.0)
        spread_two, _ = self._angular_spread(two_offsets, p0_two, r=1.0)

        assert spread_two < spread_one / 100, f"{spread_two:.2e} vs {spread_one:.2e}"
        assert spread_two < 1e-3

    def test_radial_part_matches_the_j0_series(self):
        """C(r) should be the Fourier-Bessel radial term sum_k alpha^2 J0(k_k r)."""
        K, M, L = 5, 3, 2
        model, p0, N, k = self._model(K=K, M=M, n_offsets=L)

        for r in (0.5, 1.0, 1.5, 2.0):
            _, mean = self._angular_spread(model, p0, r=r)
            # one subspace per wavevector, each contributing 2/N; M*L share each ring
            expected = np.sum((2 / N) * M * L * j0(k * r))
            np.testing.assert_allclose(mean, expected, atol=1e-5)


class TestBlockDiagSkew:
    def test_skew_symmetry(self):
        ev = np.array([1.0, 2.0, 3.0])
        sigma = block_diag_skew(ev)
        torch.testing.assert_close(sigma + sigma.T, torch.zeros_like(sigma), atol=1e-7, rtol=0)

    def test_shape(self):
        ev = np.array([1.0, 2.0, 3.0])
        sigma = block_diag_skew(ev)
        assert sigma.shape == (6, 6)

    def test_block_structure(self):
        """Sign follows Appendix C: expm of this block rotates counter-clockwise."""
        ev = np.array([2.5])
        sigma = block_diag_skew(ev)
        expected = torch.tensor([[0.0, -2.5], [2.5, 0.0]])
        torch.testing.assert_close(sigma, expected)

    def test_exponential_is_a_counter_clockwise_rotation(self):
        """Appendix C: alpha_1 = cos(Omega), alpha_2 = sin(Omega) from alpha_0 = e_1."""
        sigma = block_diag_skew(np.array([1.0]))
        omega = 0.7
        alpha = torch.matrix_exp(omega * sigma) @ torch.tensor([1.0, 0.0])
        torch.testing.assert_close(alpha, torch.tensor([np.cos(omega), np.sin(omega)]).float())

    def test_eigenvalues_preserved(self):
        ev = np.array([1.0, 3.0, 5.0])
        sigma = block_diag_skew(ev)
        actual_ev = torch.linalg.eigvals(sigma).imag
        actual_ev_sorted = torch.sort(actual_ev.abs())[0][::2]
        expected = torch.tensor([1.0, 3.0, 5.0])
        torch.testing.assert_close(actual_ev_sorted, expected, atol=1e-5, rtol=1e-5)


class TestBuildGenerators:
    def test_preserves_eigenvalues(self):
        ev = np.array([1.0, 2.0, 3.0])
        sigma_x = block_diag_skew(ev)
        sigma_y = block_diag_skew(ev * 0.5)

        from expmap.linalg import random_orthogonal_matrix

        R = random_orthogonal_matrix(6, seed=42)
        Gx, _Gy = build_generators(sigma_x, sigma_y, R)

        ev_sigma = torch.sort(torch.linalg.eigvals(sigma_x).imag.abs())[0]
        ev_G = torch.sort(torch.linalg.eigvals(Gx).imag.abs())[0]
        torch.testing.assert_close(ev_sigma, ev_G, atol=1e-4, rtol=1e-4)

    def test_skew_symmetry_preserved(self):
        ev = np.array([1.0, 2.0])
        sigma = block_diag_skew(ev)

        from expmap.linalg import random_orthogonal_matrix

        R = random_orthogonal_matrix(4, seed=42)
        Gx, _Gy = build_generators(sigma, sigma, R)
        torch.testing.assert_close(Gx + Gx.T, torch.zeros_like(Gx), atol=1e-6, rtol=0)

    def test_with_context(self):
        ev = np.array([1.0, 2.0])
        sigma = block_diag_skew(ev)

        from expmap.linalg import random_orthogonal_matrix

        R = random_orthogonal_matrix(4, seed=42)
        Gx, _Gy, Gs = build_generators_with_context(sigma, sigma, sigma, R)
        assert Gx.shape == (4, 4)
        assert Gs.shape == (4, 4)
