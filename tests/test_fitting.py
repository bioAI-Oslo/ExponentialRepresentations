"""Tests for generator fitting (Appendix K).

The fit is checked on *predictions*, not on parameters. Given a set of ratemaps
there is no unique pair (Gx, Gy). The data only constrain the generators on the
subspace the population actually spans, and any change that leaves
`exp(dr . G) p` unchanged there is invisible. So these tests ask whether the fit
transports correctly, commutes, and extrapolates.
"""

import numpy as np
import torch

import expmap
from expmap.fitting import (
    CommutingGeneratorFit,
    GeneratorFit,
    generator_diagnostics,
    pixels_to_arena,
    sample_displacements,
)


def synthetic_ratemaps(n_bins=16, arena_size=1.0, M=3, n_copies=2, seed=0):
    """Ratemaps from a known exponential map, indexed (bins_x, bins_y, n_units)."""
    model, p0 = expmap.grid_cell_model(M=M, n_copies=n_copies, seed=seed)
    axis = (torch.arange(n_bins) - n_bins // 2) * (2 * arena_size / n_bins)
    gx, gy = torch.meshgrid(axis, axis, indexing="ij")
    positions = torch.stack([gx.ravel(), gy.ravel()], dim=1)
    with torch.no_grad():
        p = model(positions, p0)
    return p.reshape(n_bins, n_bins, -1), model, p0


class TestSampleDisplacements:
    def test_stays_in_bounds(self):
        start, end = sample_displacements(n_bins=32, batch_size=500, max_step=2)
        for index in (start, end):
            assert index.min() >= 0
            assert index.max() < 32

    def test_reflection_preserves_step_magnitude(self):
        """Out-of-bounds steps are reflected, not clipped.

        Clipping would shorten steps near the walls and bias the sampled
        displacement distribution; reflecting keeps |step| intact.
        """
        start, end = sample_displacements(n_bins=8, batch_size=2000, max_step=2)
        step = (end - start).abs()
        assert step.max() <= 2
        # Every magnitude in 0..2 should still occur on both axes.
        for axis in (0, 1):
            assert set(step[:, axis].tolist()) == {0, 1, 2}

    def test_seeded_sampling_is_reproducible(self):
        a = sample_displacements(16, 20, generator=torch.Generator().manual_seed(3))
        b = sample_displacements(16, 20, generator=torch.Generator().manual_seed(3))
        torch.testing.assert_close(a[0], b[0], atol=0, rtol=0)
        torch.testing.assert_close(a[1], b[1], atol=0, rtol=0)


class TestPixelsToArena:
    def test_scale(self):
        """One pixel of a 32-bin map spanning [-1, 1] is 2/32."""
        step = torch.tensor([[1, 0], [0, -2]])
        expected = torch.tensor([[2 / 32, 0.0], [0.0, -4 / 32]])
        torch.testing.assert_close(pixels_to_arena(step, n_bins=32), expected)

    def test_arena_size_scales_it(self):
        step = torch.tensor([[1, 1]])
        doubled = pixels_to_arena(step, n_bins=32, arena_size=2.0)
        torch.testing.assert_close(doubled, 2 * pixels_to_arena(step, n_bins=32))


def fit_with_generators(model, enforce_skew=True):
    """A GeneratorFit whose generators are exactly those of `model`.

    For the skew parameterization `G = A - A^T`, setting `A = G / 2` reproduces
    a skew `G` exactly.
    """
    fit = GeneratorFit(n_units=model.Gx.shape[0], enforce_skew=enforce_skew, seed=0)
    with torch.no_grad():
        fit._Ax.copy_(model.Gx / 2 if enforce_skew else model.Gx)
        fit._Ay.copy_(model.Gy / 2 if enforce_skew else model.Gy)
    return fit


class TestEnforceSkew:
    def test_generators_are_exactly_skew(self):
        fit = GeneratorFit(n_units=16, enforce_skew=True, seed=0)
        torch.testing.assert_close(fit.Gx, -fit.Gx.T, atol=1e-7, rtol=0)
        torch.testing.assert_close(fit.Gy, -fit.Gy.T, atol=1e-7, rtol=0)

    def test_free_generators_are_not_skew(self):
        fit = GeneratorFit(n_units=16, enforce_skew=False, seed=0)
        assert not torch.allclose(fit.Gx, -fit.Gx.T, atol=1e-3)

    def test_skew_parameterization_reproduces_a_skew_target(self):
        model, _ = expmap.grid_cell_model(M=3, n_copies=2, seed=0)
        torch.testing.assert_close(fit_with_generators(model).Gx, model.Gx, atol=1e-6, rtol=0)

    def test_skew_transport_preserves_norm_when_extrapolating(self):
        """Why App. K reparameterizes: orthogonal transport, so ||p|| cannot drift.

        Checked far outside any displacement a fit would train on, which is
        the regime of the doubled arena in App. K.
        """
        p0 = torch.randn(24)
        p0 = p0 / p0.norm()
        far = torch.tensor([[3.0, -2.5], [-4.0, 4.0]])

        skew = GeneratorFit(n_units=24, enforce_skew=True, seed=0)
        free = GeneratorFit(n_units=24, enforce_skew=False, seed=0)

        with torch.no_grad():
            skew_norms = skew.transport(p0.expand(len(far), -1), far).norm(dim=1)
            free_norms = free.transport(p0.expand(len(far), -1), far).norm(dim=1)

        torch.testing.assert_close(skew_norms, torch.ones(len(far)), atol=1e-4, rtol=0)
        assert (free_norms - 1).abs().max() > 1e-2, "free generators should drift"


class TestTransport:
    def test_matches_the_model(self):
        """GeneratorFit.transport is Eq. 1, so it must agree with ExponentialMap."""
        model, p0 = expmap.grid_cell_model(M=3, n_copies=2, seed=0)
        fit = fit_with_generators(model)

        dr = torch.tensor([[0.3, -0.2], [1.0, 0.5], [0.0, 0.0]])
        with torch.no_grad():
            transported = fit.transport(p0.expand(len(dr), -1), dr)
            expected = model(dr, p0)
        torch.testing.assert_close(transported, expected, atol=1e-5, rtol=1e-4)


class TestFit:
    def test_recovers_transport_from_synthetic_data(self):
        """Fitted generators should predict held-out displacements.

        Trained on steps of at most 2 pixels; checked on a 5-pixel step, which
        no training batch contained.
        """
        ratemaps, _model, _p0 = synthetic_ratemaps()
        fit = GeneratorFit(n_units=ratemaps.shape[-1], seed=0)
        history = fit.fit(ratemaps, n_steps=1500, arena_size=1.0, progress=False)

        assert np.mean(history[-50:]) < np.mean(history[:50]) / 20, "loss should fall by more than 20x"

        n_bins = ratemaps.shape[0]
        centre = n_bins // 2
        held_out = pixels_to_arena(torch.tensor([[5, -5]]), n_bins)
        with torch.no_grad():
            predicted = fit.transport(ratemaps[centre, centre][None], held_out)[0]
        truth = ratemaps[centre + 5, centre - 5]

        error = (predicted - truth).pow(2).mean().sqrt() / truth.pow(2).mean().sqrt()
        assert error < 0.2, f"relative error on a held-out step: {error:.3f}"

    def test_commutator_penalty_drives_generators_to_commute(self):
        ratemaps, _, _ = synthetic_ratemaps()
        fit = GeneratorFit(n_units=ratemaps.shape[-1], commutator_weight=1.0, seed=0)
        initial = generator_diagnostics(fit)["commutator_rms"]
        fit.fit(ratemaps, n_steps=1500, progress=False)
        assert generator_diagnostics(fit)["commutator_rms"] < initial / 10

    def test_unpenalized_generators_need_not_commute(self):
        """Without the penalty nothing forces path integration to be consistent."""
        ratemaps, _, _ = synthetic_ratemaps()
        penalized = GeneratorFit(n_units=ratemaps.shape[-1], commutator_weight=1.0, seed=0)
        free = GeneratorFit(n_units=ratemaps.shape[-1], commutator_weight=0.0, seed=0)
        penalized.fit(ratemaps, n_steps=800, progress=False)
        free.fit(ratemaps, n_steps=800, progress=False)

        assert generator_diagnostics(penalized)["commutator_rms"] < generator_diagnostics(free)["commutator_rms"]

    def test_seed_is_reproducible(self):
        ratemaps, _, _ = synthetic_ratemaps()
        histories = []
        for _ in range(2):
            fit = GeneratorFit(n_units=ratemaps.shape[-1], seed=5)
            histories.append(fit.fit(ratemaps, n_steps=50, progress=False))
        np.testing.assert_allclose(histories[0], histories[1])


class TestGenerate:
    def test_reproduces_the_training_domain(self):
        ratemaps, _, _ = synthetic_ratemaps()
        n_bins = ratemaps.shape[0]
        fit = GeneratorFit(n_units=ratemaps.shape[-1], seed=0)
        fit.fit(ratemaps, n_steps=1500, progress=False)

        centre = ratemaps[n_bins // 2, n_bins // 2]
        generated = fit.generate(centre, n_bins=n_bins, extent=1.0)

        truth = ratemaps.permute(2, 0, 1)
        error = (generated - truth).pow(2).mean().sqrt() / truth.pow(2).mean().sqrt()
        assert error < 0.25, f"relative reconstruction error: {error:.3f}"

    def test_shape_and_extent(self):
        ratemaps, _, _ = synthetic_ratemaps()
        fit = GeneratorFit(n_units=ratemaps.shape[-1], seed=0)
        centre = ratemaps[8, 8]

        same = fit.generate(centre, n_bins=16, extent=1.0)
        double = fit.generate(centre, n_bins=16, extent=2.0)
        assert same.shape == double.shape == (ratemaps.shape[-1], 16, 16)
        # extent only rescales displacement, so the centre pixel is unchanged
        torch.testing.assert_close(same[:, 8, 8], double[:, 8, 8])


class TestDiagnostics:
    def test_skew_symmetric_generators_report_zero_symmetric_fraction(self):
        model, _ = expmap.grid_cell_model(M=3, n_copies=2, seed=0)
        diagnostics = generator_diagnostics(fit_with_generators(model))
        assert diagnostics["Gx_symmetric_fraction"] < 1e-6
        assert diagnostics["commutator_rms"] < 1e-5

    def test_free_random_generators_report_a_large_symmetric_fraction(self):
        """A random matrix has ||G + G^T|| ~ sqrt(2) ||G||, so ~1.41 is the baseline."""
        diagnostics = generator_diagnostics(GeneratorFit(n_units=32, enforce_skew=False, seed=0))
        assert 1.2 < diagnostics["Gx_symmetric_fraction"] < 1.6

    def test_enforced_generators_report_zero_regardless_of_training(self):
        diagnostics = generator_diagnostics(GeneratorFit(n_units=32, enforce_skew=True, seed=0))
        assert diagnostics["Gx_symmetric_fraction"] < 1e-6


class TestCommutingGeneratorFit:
    def test_transport_is_the_matrix_exponential(self):
        """The closed-form block rotations must equal exp(dx Gx + dy Gy), for odd and even N."""
        for n_units in (24, 25):
            fit = CommutingGeneratorFit(n_units=n_units, seed=0)
            p0 = torch.randn(6, n_units)
            dr = torch.randn(6, 2)
            with torch.no_grad():
                closed_form = fit.transport(p0, dr)
                Q = torch.matrix_exp(dr[:, 0, None, None] * fit.Gx + dr[:, 1, None, None] * fit.Gy)
                expected = torch.bmm(Q, p0[..., None]).squeeze(-1)
            torch.testing.assert_close(closed_form, expected, atol=1e-5, rtol=1e-4)

    def test_generators_commute_and_are_skew_by_construction(self):
        fit = CommutingGeneratorFit(n_units=25, seed=1)
        d = generator_diagnostics(fit)
        assert d["commutator_rms"] < 1e-5
        assert d["Gx_symmetric_fraction"] < 1e-5 and d["Gy_symmetric_fraction"] < 1e-5
        R = fit.R.detach()
        torch.testing.assert_close(R @ R.T, torch.eye(25), atol=1e-5, rtol=0)

    def test_recovers_transport_from_synthetic_data(self):
        """Same held-out check as for GeneratorFit, without any penalty."""
        ratemaps, _, _ = synthetic_ratemaps()
        fit = CommutingGeneratorFit(n_units=ratemaps.shape[-1], seed=0)
        history = fit.fit(ratemaps, n_steps=1500, progress=False)
        assert np.mean(history[-50:]) < np.mean(history[:50]) / 20

        n_bins = ratemaps.shape[0]
        centre = n_bins // 2
        held_out = pixels_to_arena(torch.tensor([[5, -5]]), n_bins)
        with torch.no_grad():
            predicted = fit.transport(ratemaps[centre, centre][None], held_out)[0]
        truth = ratemaps[centre + 5, centre - 5]
        error = (predicted - truth).pow(2).mean().sqrt() / truth.pow(2).mean().sqrt()
        assert error < 0.2, f"relative error on a held-out step: {error:.3f}"

    def test_generate_matches_generator_fit_interface(self):
        ratemaps, _, _ = synthetic_ratemaps()
        fit = CommutingGeneratorFit(n_units=ratemaps.shape[-1], seed=0)
        maps = fit.generate(ratemaps[8, 8], n_bins=16, extent=2)
        assert maps.shape == (ratemaps.shape[-1], 16, 16)
        torch.testing.assert_close(maps.pow(2).sum(0), maps.pow(2).sum(0).mean().expand(16, 16), atol=1e-4, rtol=1e-4)
