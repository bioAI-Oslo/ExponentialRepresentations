"""Tests for grid score and spatial autocorrelogram."""

import numpy as np

from expmap.analysis import annulus_radii, compute_grid_scores, grid_score, spatial_autocorrelogram


def hexagonal_ratemap(bins=64, k=3.0, extent=5.0):
    """Sum of three cosines 60 degrees apart, a clean hexagonal grid."""
    x = np.linspace(-extent, extent, bins)
    X, Y = np.meshgrid(x, x)
    ratemap = np.zeros((bins, bins))
    for n in range(3):
        phi = n * np.pi / 3
        ratemap += np.cos(k * (X * np.cos(phi) + Y * np.sin(phi)))
    return ratemap / 3.0


class TestSpatialAutocorrelogram:
    def test_shape(self):
        rm = np.random.default_rng(0).standard_normal((32, 32))
        acorr = spatial_autocorrelogram(rm)
        assert acorr.shape == (63, 63)

    def test_zero_lag_is_unity(self):
        """The zero-lag autocorrelation is normalized to exactly 1."""
        rm = np.random.default_rng(42).standard_normal((32, 32)) + 2.0  # positive mean
        acorr = spatial_autocorrelogram(rm)
        cy, cx = acorr.shape[0] // 2, acorr.shape[1] // 2
        np.testing.assert_allclose(acorr[cy, cx], 1.0, atol=0.01)

    def test_not_bounded_to_unit_interval(self):
        """Values are not Pearson correlations and can exceed 1 in magnitude.

        The numerator is divided by the per-lag overlap count but scaled by a
        global variance, so far lags, where few bin pairs contribute, are
        under-normalized, and the zero-lag peak is not the global maximum.
        Pinned because grid_score correlates this array and reports depend on
        the scale.
        """
        acorr = spatial_autocorrelogram(hexagonal_ratemap())
        cy, cx = acorr.shape[0] // 2, acorr.shape[1] // 2
        assert np.nanmax(acorr) > 1.0
        assert np.unravel_index(np.nanargmax(acorr), acorr.shape) != (cy, cx)

    def test_symmetry(self):
        """Autocorrelogram should be symmetric: acorr(dx, dy) = acorr(-dx, -dy)."""
        rm = np.random.default_rng(42).standard_normal((20, 20))
        acorr = spatial_autocorrelogram(rm)
        # Should be point-symmetric around center
        np.testing.assert_allclose(acorr, acorr[::-1, ::-1], atol=1e-10)

    def test_handles_nans(self):
        rm = np.random.default_rng(0).standard_normal((16, 16))
        rm[0, 0] = np.nan
        rm[5, 5] = np.nan
        acorr = spatial_autocorrelogram(rm)
        assert acorr.shape == (31, 31)
        cy, cx = 15, 15
        assert not np.isnan(acorr[cy, cx])


class TestOverlapCount:
    def test_single_pair_corner_lags_are_kept(self):
        """The FFT overlap count lands just below 1 at some sizes; those corners are real lags."""
        for n in (15, 31, 50, 64):
            acorr = spatial_autocorrelogram(np.random.default_rng(0).random((n, n)))
            assert not np.isnan(acorr).any(), n


class TestAnnulusRadii:
    def test_detected_radii_are_pinned(self):
        """Pin what the heuristic picks on the reference hexagonal ratemap.

        grid_score depends strongly on the annulus. On this same map the score
        ranges from -0.37 to +0.70 across plausible annuli, so a silent change
        to the detection would move every reported grid score without failing
        any other test.
        """
        acorr = spatial_autocorrelogram(hexagonal_ratemap())
        assert annulus_radii(acorr) == (7, 21)

    def test_detected_radii_reproduce_the_default_score(self):
        ratemap = hexagonal_ratemap()
        r_inner, r_outer = annulus_radii(spatial_autocorrelogram(ratemap))
        np.testing.assert_allclose(grid_score(ratemap, r_inner, r_outer), grid_score(ratemap))

    def test_explicit_radii_override_detection(self):
        ratemap = hexagonal_ratemap()
        assert grid_score(ratemap, r_inner=6, r_outer=24) != grid_score(ratemap)


class TestGridScore:
    def test_hexagonal_grid_has_positive_score(self):
        """A synthetic hexagonal grid should produce a positive grid score."""
        score = grid_score(hexagonal_ratemap())
        assert score > 0.3, f"Hexagonal grid should have high grid score, got {score}"

    def test_random_noise_has_low_score(self):
        """Random noise should have a grid score near zero."""
        ratemap = np.random.default_rng(42).standard_normal((32, 32))
        score = grid_score(ratemap)
        assert abs(score) < 0.5, f"Noise should have low grid score, got {score}"

    def test_square_grid_has_low_hexagonal_score(self):
        """A square grid (4-fold symmetry) should have low hexagonal grid score."""
        bins = 64
        x = np.linspace(-5, 5, bins)
        y = np.linspace(-5, 5, bins)
        X, Y = np.meshgrid(x, y)

        # Square grid: two cosines 90 degrees apart
        k = 3.0
        ratemap = np.cos(k * X) + np.cos(k * Y)

        score = grid_score(ratemap)
        # A square grid must not score as hexagonal
        assert score < 0.3, f"Square grid should have low hex score, got {score}"

    def test_score_depends_strongly_on_the_annulus(self):
        """Documents the sensitivity that motivates recording the radii.

        On this synthetic map the four annuli span roughly 0.53 to 1.38. On a
        ratemap from grid_cell_model the same sweep runs from -0.37 to +0.70,
        so the annulus can flip the sign. Pass explicit radii for any reported
        number.
        """
        ratemap = hexagonal_ratemap()
        scores = [grid_score(ratemap, lo, hi) for lo, hi in [(3, 12), (4, 14), (5, 20), (6, 24)]]
        assert max(scores) - min(scores) > 0.5


class TestComputeGridScores:
    def test_shape(self):
        ratemaps = np.random.default_rng(0).standard_normal((10, 32, 32))
        scores = compute_grid_scores(ratemaps)
        assert scores.shape == (10,)

    def test_consistency(self):
        """Batch computation should match individual computation."""
        ratemaps = np.random.default_rng(42).standard_normal((5, 32, 32))
        batch_scores = compute_grid_scores(ratemaps)
        for i in range(5):
            individual = grid_score(ratemaps[i])
            np.testing.assert_allclose(batch_scores[i], individual)
