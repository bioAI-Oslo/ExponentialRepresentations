"""Tests for trajectory and position generation.

Mostly about seeding: these were the library's remaining sources of global RNG
state, and anything downstream of them feeds a figure.
"""

import numpy as np
import torch

from expmap.simulation import SimpleDatasetMaker, generate_uniform_positions


class TestGenerateUniformPositions:
    def test_shapes(self):
        batches = generate_uniform_positions(4, 25, arena_size=3.0, seed=0)
        assert len(batches) == 4
        assert all(b.shape == (25, 2) for b in batches)
        assert all(b.dtype == torch.float32 for b in batches)

    def test_within_arena(self):
        batches = generate_uniform_positions(4, 100, arena_size=3.0, seed=0)
        r = torch.cat(batches)
        assert r.abs().max() <= 3.0

    def test_seed_is_reproducible(self):
        a = torch.cat(generate_uniform_positions(3, 10, 1.0, seed=42))
        b = torch.cat(generate_uniform_positions(3, 10, 1.0, seed=42))
        torch.testing.assert_close(a, b, atol=0, rtol=0)

    def test_different_seeds_differ(self):
        a = torch.cat(generate_uniform_positions(3, 10, 1.0, seed=1))
        b = torch.cat(generate_uniform_positions(3, 10, 1.0, seed=2))
        assert not torch.allclose(a, b)

    def test_batches_are_not_identical(self):
        """One generator drives the whole list, so batches must not repeat."""
        batches = generate_uniform_positions(3, 10, 1.0, seed=0)
        assert not torch.allclose(batches[0], batches[1])

    def test_seed_does_not_disturb_global_rng(self):
        """A local torch.Generator, so seeded sampling must not shift global state."""
        torch.manual_seed(123)
        expected = torch.rand(5)

        torch.manual_seed(123)
        generate_uniform_positions(2, 10, 1.0, seed=999)
        torch.testing.assert_close(torch.rand(5), expected, atol=0, rtol=0)


class TestSimpleDatasetMaker:
    def test_shapes(self):
        maker = SimpleDatasetMaker(seed=0)
        r, v = maker.generate_dataset(samples=8, timesteps=20)
        assert r.shape == (8, 20, 2)
        assert v.shape == (8, 19, 2)

    def test_stays_in_chamber(self):
        maker = SimpleDatasetMaker(chamber_size=1, seed=0)
        r, _ = maker.generate_dataset(samples=16, timesteps=200)
        assert r.abs().max() <= 1.0

    def test_seed_is_reproducible(self):
        r1, v1 = SimpleDatasetMaker(seed=7).generate_dataset(4, 30)
        r2, v2 = SimpleDatasetMaker(seed=7).generate_dataset(4, 30)
        torch.testing.assert_close(r1, r2, atol=0, rtol=0)
        torch.testing.assert_close(v1, v2, atol=0, rtol=0)

    def test_successive_calls_advance_the_stream(self):
        maker = SimpleDatasetMaker(seed=7)
        first, _ = maker.generate_dataset(4, 30)
        second, _ = maker.generate_dataset(4, 30)
        assert not torch.allclose(first, second)

    def test_seed_does_not_disturb_global_rng(self):
        np.random.seed(123)
        expected = np.random.rand(5)

        np.random.seed(123)
        SimpleDatasetMaker(seed=999).generate_dataset(4, 30)
        np.testing.assert_array_equal(np.random.rand(5), expected)

    def test_velocities_are_position_differences(self):
        r, v = SimpleDatasetMaker(seed=0).generate_dataset(4, 30)
        torch.testing.assert_close(v, r[:, 1:] - r[:, :-1], atol=1e-6, rtol=1e-5)
