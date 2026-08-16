import numpy as np
import torch

from expmap.generators import block_diag_skew
from expmap.linalg import (
    block_diagonal_orthogonal,
    block_rotation,
    orthogonal_basis,
    random_orthogonal_matrix,
    starting_state,
)


class TestRandomOrthogonalMatrix:
    def test_orthogonality(self):
        R = random_orthogonal_matrix(10, seed=42)
        identity = torch.eye(10)
        torch.testing.assert_close(R @ R.T, identity, atol=1e-5, rtol=1e-5)

    def test_shape(self):
        R = random_orthogonal_matrix(8, seed=42)
        assert R.shape == (8, 8)

    def test_dtype(self):
        R = random_orthogonal_matrix(6, seed=42)
        assert R.dtype == torch.float32

    def test_reproducibility(self):
        R1 = random_orthogonal_matrix(10, seed=42)
        R2 = random_orthogonal_matrix(10, seed=42)
        torch.testing.assert_close(R1, R2)

    def test_different_seeds(self):
        R1 = random_orthogonal_matrix(10, seed=1)
        R2 = random_orthogonal_matrix(10, seed=2)
        assert not torch.allclose(R1, R2)


class TestStartingState:
    def test_unit_norm(self):
        R = random_orthogonal_matrix(20, seed=42)
        p0 = starting_state(R)
        assert abs(p0.norm().item() - 1.0) < 1e-5

    def test_shape(self):
        R = random_orthogonal_matrix(20, seed=42)
        p0 = starting_state(R)
        assert p0.shape == (20,)

    def test_dtype(self):
        R = random_orthogonal_matrix(20, seed=42)
        p0 = starting_state(R)
        assert p0.dtype == torch.float32


class TestBlockDiagonalOrthogonal:
    def test_is_orthogonal(self):
        R = block_diagonal_orthogonal(4, 6, seed=42)
        assert R.shape == (24, 24)
        torch.testing.assert_close(R @ R.T, torch.eye(24), atol=1e-5, rtol=0)

    def test_is_block_diagonal(self):
        """Off-block entries must be exactly zero — that is what makes cells module-pure."""
        n_blocks, block_size = 4, 6
        R = block_diagonal_orthogonal(n_blocks, block_size, seed=42)

        mask = torch.ones_like(R, dtype=torch.bool)
        for i in range(n_blocks):
            lo, hi = i * block_size, (i + 1) * block_size
            mask[lo:hi, lo:hi] = False
        assert R[mask].abs().max() == 0

    def test_seed_is_reproducible(self):
        torch.testing.assert_close(
            block_diagonal_orthogonal(3, 4, seed=7), block_diagonal_orthogonal(3, 4, seed=7), atol=0, rtol=0
        )
        assert not torch.allclose(block_diagonal_orthogonal(3, 4, seed=7), block_diagonal_orthogonal(3, 4, seed=8))

    def test_blocks_differ_from_each_other(self):
        """Each block gets its own random_state, so modules are mixed independently."""
        R = block_diagonal_orthogonal(2, 5, seed=0)
        assert not torch.allclose(R[:5, :5], R[5:, 5:])

    def test_gives_the_canonical_p0_unit_norm(self):
        """starting_state must work with either end of the R continuum."""
        p0 = starting_state(block_diagonal_orthogonal(5, 6, seed=1))
        torch.testing.assert_close(p0.norm(), torch.tensor(1.0), atol=1e-5, rtol=0)


class TestOrthogonalBasis:
    def test_full_rank(self):
        q0 = np.array([1.0, 0.0, 0.0, 0.0])
        basis = orthogonal_basis(q0)
        assert np.linalg.matrix_rank(basis) == 4

    def test_orthonormality(self):
        q0 = np.array([1.0, 2.0, 3.0])
        q0 = q0 / np.linalg.norm(q0)
        basis = orthogonal_basis(q0)
        identity = basis @ basis.T
        np.testing.assert_allclose(identity, np.eye(3), atol=1e-10)

    def test_first_vector_preserved(self):
        q0 = np.array([1.0, 2.0, 3.0])
        q0_normed = q0 / np.linalg.norm(q0)
        basis = orthogonal_basis(q0)
        np.testing.assert_allclose(basis[0], q0_normed, atol=1e-10)

    def test_seed_is_reproducible(self):
        q0 = np.array([1.0, 0.0, 0.0, 0.0])
        np.testing.assert_array_equal(orthogonal_basis(q0, seed=7), orthogonal_basis(q0, seed=7))
        assert not np.allclose(orthogonal_basis(q0, seed=7), orthogonal_basis(q0, seed=8))


class TestBlockRotation:
    def test_orthogonal(self):
        omegas = np.array([0.5, 1.0, 1.5])
        Q = block_rotation(omegas)
        identity = Q @ Q.T
        np.testing.assert_allclose(identity, np.eye(6), atol=1e-10)

    def test_shape(self):
        omegas = np.array([0.1, 0.2])
        Q = block_rotation(omegas)
        assert Q.shape == (4, 4)

    def test_zero_angle_is_identity(self):
        omegas = np.array([0.0])
        Q = block_rotation(omegas)
        np.testing.assert_allclose(Q, np.eye(2), atol=1e-10)

    def test_equals_expm_of_block_diag_skew(self):
        """Pin the sign convention against generators.block_diag_skew.

        Both follow Appendix C's counter-clockwise rotation, so block_rotation
        is the closed form of expm(block_diag_skew). The two used to differ by
        a transpose, which silently reflects every ratemap through the origin.
        """
        omegas = np.array([0.5, 1.0, 1.5])
        Q = block_rotation(omegas)
        expm_sigma = torch.matrix_exp(block_diag_skew(omegas)).numpy()

        np.testing.assert_allclose(Q, expm_sigma, atol=1e-6)
