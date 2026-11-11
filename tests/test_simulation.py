"""Tests for position sampling.

Mostly about seeding: this was one of the library's sources of global RNG
state, and anything downstream of it feeds a figure.
"""

import torch

from expmap.simulation import generate_uniform_positions


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
