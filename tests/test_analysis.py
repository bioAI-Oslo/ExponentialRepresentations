import numpy as np
import torch

import expmap
from expmap.analysis import compute_ratemaps, compute_similarity, rectify


class TestComputeRatemaps:
    def test_shape(self):
        n_samples = 1000
        n_units = 10
        x = np.random.uniform(-1, 1, n_samples)
        y = np.random.uniform(-1, 1, n_samples)
        p = np.random.randn(n_samples, n_units)
        ratemaps = compute_ratemaps(x, y, p, bins=16)
        assert ratemaps.shape == (n_units, 16, 16)

    def test_different_bin_count(self):
        n_samples = 500
        n_units = 5
        x = np.random.uniform(-1, 1, n_samples)
        y = np.random.uniform(-1, 1, n_samples)
        p = np.random.randn(n_samples, n_units)
        ratemaps = compute_ratemaps(x, y, p, bins=8)
        assert ratemaps.shape == (n_units, 8, 8)


class TestComputeSimilarity:
    def test_shape(self):
        ratemaps = np.random.randn(10, 16, 16)
        sim = compute_similarity(ratemaps, ref_bin=(8, 8))
        assert sim.shape == (16, 16)

    def test_self_similarity_equals_squared_norm(self):
        ratemaps = np.random.randn(10, 16, 16)
        ref = (8, 8)
        sim = compute_similarity(ratemaps, ref_bin=ref)
        expected = np.sum(ratemaps[:, ref[0], ref[1]] ** 2)
        np.testing.assert_allclose(sim[ref[0], ref[1]], expected, rtol=1e-10)

    def test_unvisited_reference_bin_is_nan(self):
        """An empty reference bin must not read as an orthogonal population.

        nansum would treat the reference NaNs as zeros and return an
        identically-zero map, which is a plausible-looking wrong answer.
        """
        ratemaps = np.random.randn(10, 16, 16)
        ratemaps[:, 4, 4] = np.nan
        sim = compute_similarity(ratemaps, ref_bin=(4, 4))
        assert np.isnan(sim).all()


class TestSimilarityAgainstClosedForm:
    """compute_similarity against Eq. 4 (App. F).

    For commuting skew-symmetric generators and the canonical p0 = R^T 1/sqrt(N),
    the population similarity depends only on displacement and the coefficients
    are uniform at alpha_0,i^2 = 1/N:

        C(dx, dy) = sum_i (2/N) cos(lambda_ix dx + lambda_iy dy)

    summed over the N/2 rotational subspaces. This exercises the whole chain —
    generator construction, matrix_exp, binning and the dot product — against a
    number that is derivable rather than merely reproducible.
    """

    def test_matches_equation_4(self):
        M, n_copies, bins = 3, 20, 64
        N = 2 * M * n_copies
        model, p0 = expmap.grid_cell_model(M=M, n_copies=n_copies, seed=0)

        # Sample on exact bin centres over ~4 grid periods. Sampling on
        # linspace(-L, L, bins) instead lands two samples in some bins and none
        # in others, and the resulting binning error swamps the comparison.
        L = 4 * np.pi
        edges = np.linspace(-L, L, bins + 1)
        centres = (edges[:-1] + edges[1:]) / 2
        X, Y = np.meshgrid(centres, centres, indexing="ij")
        positions = torch.tensor(np.stack([X.ravel(), Y.ravel()], axis=1), dtype=torch.float32)

        with torch.no_grad():
            p = model(positions, p0).numpy()

        ratemaps = compute_ratemaps(positions[:, 0].numpy(), positions[:, 1].numpy(), p, bins=bins)
        assert not np.isnan(ratemaps).any(), "every bin should hold exactly one sample"

        ref = (bins // 2, bins // 2)
        empirical = compute_similarity(ratemaps, ref_bin=ref)

        phi = np.tile(np.arange(M) * np.pi / M, n_copies)
        dx = centres[:, None] - centres[ref[0]]
        dy = centres[None, :] - centres[ref[1]]
        closed_form = sum((2 / N) * np.cos(np.cos(phi[i]) * dx + np.sin(phi[i]) * dy) for i in range(len(phi)))

        np.testing.assert_allclose(empirical, closed_form, atol=1e-5)

    def test_gaussian_spectrum_targets_the_right_width(self):
        """Appendix I samples from N(0, 2 sigma^2) for a target exp(-sigma^2 x^2).

        `gaussian_eigenvalues`' `scale` is the eigenvalue standard deviation, so
        scale^2 = 2 sigma^2 and C(delta) ~= exp(-scale^2 delta^2 / 2). The factor
        of two is easy to lose; exp(-scale^2 delta^2) is the wrong target and this
        pins which one is right.
        """
        N, scale, bins, arena = 256, np.sqrt(2), 32, 2.0
        model, p0 = expmap.place_cell_model(N=N, scale=scale, seed=0)

        edges = np.linspace(-arena, arena, bins + 1)
        centres = (edges[:-1] + edges[1:]) / 2
        X, Y = np.meshgrid(centres, centres, indexing="ij")
        positions = torch.tensor(np.stack([X.ravel(), Y.ravel()], axis=1), dtype=torch.float32)
        with torch.no_grad():
            p = model(positions, p0).numpy()

        ref = (bins // 2, bins // 2)
        empirical = compute_similarity(p.T.reshape(N, bins, bins), ref_bin=ref)

        d2 = (centres[:, None] - centres[ref[0]]) ** 2 + (centres[None, :] - centres[ref[1]]) ** 2
        right = np.exp(-(scale**2) * d2 / 2)
        wrong = np.exp(-(scale**2) * d2)

        rms_right = np.sqrt(np.mean((empirical - right) ** 2))
        rms_wrong = np.sqrt(np.mean((empirical - wrong) ** 2))

        # Monte Carlo over N // 2 samples, so the right target fits only to
        # ~1/sqrt(N//2); the wrong one is a different function entirely.
        assert rms_right < 1 / np.sqrt(N // 2)
        assert rms_wrong > 2 * rms_right
        np.testing.assert_allclose(empirical[ref], 1.0, atol=1e-5)

    def test_translational_invariance(self):
        """C depends on displacement alone, so shifting the reference bin
        shifts the similarity map without changing it (Sec. 2.3)."""
        bins = 32
        model, p0 = expmap.grid_cell_model(M=3, n_copies=20, seed=0)
        L = 2 * np.pi
        edges = np.linspace(-L, L, bins + 1)
        centres = (edges[:-1] + edges[1:]) / 2
        X, Y = np.meshgrid(centres, centres, indexing="ij")
        positions = torch.tensor(np.stack([X.ravel(), Y.ravel()], axis=1), dtype=torch.float32)

        with torch.no_grad():
            p = model(positions, p0).numpy()
        ratemaps = compute_ratemaps(positions[:, 0].numpy(), positions[:, 1].numpy(), p, bins=bins)

        centre = compute_similarity(ratemaps, ref_bin=(16, 16))
        shifted = compute_similarity(ratemaps, ref_bin=(20, 16))

        # Same displacement, so the same similarity value.
        np.testing.assert_allclose(centre[10:22, 8:24], shifted[14:26, 8:24], atol=1e-5)


class TestRectify:
    def test_negative_zeroed(self):
        x = np.array([-1.0, -0.5, 0.0, 0.5, 1.0])
        result = rectify(x)
        np.testing.assert_array_equal(result, [0.0, 0.0, 0.0, 0.5, 1.0])

    def test_positive_preserved(self):
        x = np.array([1.0, 2.0, 3.0])
        result = rectify(x)
        np.testing.assert_array_equal(result, x)
