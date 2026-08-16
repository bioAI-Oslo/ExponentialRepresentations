import numpy as np
import torch

import expmap
from expmap.model import ExponentialMap


class TestExponentialMap:
    def test_output_shape(self):
        model, p0 = expmap.grid_cell_model(M=3, n_copies=5, seed=42)
        positions = torch.rand(50, 2)
        p = model(positions, p0)
        assert p.shape == (50, 30)

    def test_norm_preservation(self):
        """Skew-symmetric generators produce orthogonal transformations
        that preserve population vector norm everywhere."""
        model, p0 = expmap.grid_cell_model(M=3, n_copies=5, seed=42)
        positions = torch.rand(100, 2) * 10 - 5
        p = model(positions, p0)
        expected_norm = p0.norm().expand(100)
        torch.testing.assert_close(p.norm(dim=1), expected_norm, atol=1e-5, rtol=1e-5)

    def test_path_independence(self):
        """Commuting generators guarantee path-independent representations.
        Going (0,0)->(1,0)->(1,1) must equal (0,0)->(1,1) directly."""
        model, p0 = expmap.grid_cell_model(M=3, n_copies=5, seed=42)

        # Direct path: (0,0) -> (1,1)
        p_direct = model(torch.tensor([[1.0, 1.0]]), p0)

        # Two-step path: (0,0) -> (1,0) -> (1,1)
        p_step1 = model(torch.tensor([[1.0, 0.0]]), p0)  # go to (1,0)
        # Relative displacement from (1,0) to (1,1) is (0,1)
        p_step2 = model(torch.tensor([[0.0, 1.0]]), p_step1[0])

        torch.testing.assert_close(p_direct, p_step2, atol=1e-4, rtol=1e-4)

    def test_origin_returns_p0(self):
        """Evaluating at the origin (0,0) should return p0."""
        model, p0 = expmap.grid_cell_model(M=3, n_copies=5, seed=42)
        p_origin = model(torch.tensor([[0.0, 0.0]]), p0)
        torch.testing.assert_close(p_origin[0], p0, atol=1e-6, rtol=1e-6)


class TestRemappingModel:
    def test_output_shape(self):
        model, p0 = expmap.remapping_model(M=3, n_copies=5, seed=42)
        positions = torch.rand(50, 2)
        p = model(positions, p0, context=0.0)
        assert p.shape == (50, 30)

    def test_zero_context_matches_base(self):
        """With context=0, RemappingModel should behave like ExponentialMap."""
        model_r, p0_r = expmap.remapping_model(M=3, n_copies=5, seed=42)
        positions = torch.rand(20, 2, dtype=torch.float32)
        p_remapped = model_r(positions, p0_r, context=0.0)

        # Build equivalent base model with same generators
        base_model = ExponentialMap(model_r.Gx, model_r.Gy)
        p_base = base_model(positions, p0_r)

        torch.testing.assert_close(p_remapped, p_base, atol=1e-5, rtol=1e-5)

    def test_context_orthogonality(self):
        """Large context shifts should produce nearly orthogonal representations."""
        model, p0 = expmap.remapping_model(M=3, n_copies=10, seed=42)
        positions = torch.rand(50, 2)

        p_c0 = model(positions, p0, context=0.0)
        p_c5 = model(positions, p0, context=5.0)

        # Average cosine similarity across positions
        cos_sim = torch.nn.functional.cosine_similarity(p_c0, p_c5, dim=1).mean()
        assert abs(cos_sim.item()) < 0.3, f"Expected near-orthogonal, got cosine similarity {cos_sim.item():.3f}"

    def test_scalar_and_tensor_context(self):
        """Scalar and tensor context inputs should produce identical results."""
        model, p0 = expmap.remapping_model(M=3, n_copies=5, seed=42)
        positions = torch.rand(10, 2)

        p_scalar = model(positions, p0, context=2.5)
        p_tensor = model(positions, p0, context=torch.tensor(2.5))

        torch.testing.assert_close(p_scalar, p_tensor, atol=1e-6, rtol=1e-6)

    def test_numpy_scalar_context(self):
        """numpy scalars must work too.

        isinstance(context, float) caught np.float64 (a float subclass) but
        not np.float32, which fell through to the tensor branch and raised
        AttributeError.
        """
        model, p0 = expmap.remapping_model(M=3, n_copies=5, seed=42)
        positions = torch.rand(10, 2)
        expected = model(positions, p0, context=2.5)

        for ctx in (np.float32(2.5), np.float64(2.5), np.int32(2)):
            result = model(positions, p0, context=ctx)
            if ctx != 2:
                torch.testing.assert_close(result, expected, atol=1e-6, rtol=1e-6)


class TestDevicePlacement:
    def test_forward_follows_module_to(self):
        """forward must read the device off the buffers, not a cached attribute.

        self.device could not be updated by nn.Module.to(), so moving the model
        left forward coercing inputs to the original device. Exercised here via
        dtype, which .to() propagates the same way.
        """
        model, p0 = expmap.grid_cell_model(M=3, n_copies=5, seed=42)
        model = model.to(torch.float64)
        assert model.Gx.dtype == torch.float64

        p = model(torch.rand(4, 2, dtype=torch.float64), p0.double())
        assert p.dtype == torch.float64
        assert not hasattr(model, "device"), "stale device attribute reintroduced"
