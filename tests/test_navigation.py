import torch

import expmap
from expmap.navigation import bind, bundle, query, similarity_gradient


class TestBundle:
    def test_shape(self):
        representations = torch.randn(5, 60)
        memory = bundle(representations)
        assert memory.shape == (60,)

    def test_similarity_to_constituents(self):
        """Bundled memory should be similar to its constituent vectors."""
        model, p0 = expmap.grid_cell_model(M=3, n_copies=10, seed=42)
        positions = torch.tensor([[0.0, 0.0], [0.1, 0.0], [0.0, 0.1]])
        reps = model(positions, p0)
        memory = bundle(reps)

        for i in range(len(reps)):
            sim = torch.dot(reps[i], memory)
            assert sim.item() > 0, f"Expected positive similarity, got {sim.item()}"


class TestQuery:
    def test_at_stored_location(self):
        """Querying at a stored location should yield high similarity."""
        model, p0 = expmap.grid_cell_model(M=3, n_copies=10, seed=42)
        stored_pos = torch.tensor([[1.0, 1.0]])
        stored_rep = model(stored_pos, p0)
        memory = bundle(stored_rep)

        score_at_stored = query(stored_rep[0], memory)
        assert score_at_stored.item() > 0

    def test_batch_query(self):
        representations = torch.randn(10, 60)
        memory = torch.randn(60)
        scores = query(representations, memory)
        assert scores.shape == (10,)

    def test_single_query(self):
        state = torch.randn(60)
        memory = torch.randn(60)
        score = query(state, memory)
        assert score.dim() == 0  # scalar


class TestBind:
    def test_shape(self):
        memories = [torch.randn(60), torch.randn(60)]
        composite = bind(memories)
        assert composite.shape == (60,)

    def test_orthogonality_filtering(self):
        """Querying composite memory with the correct context should
        retrieve only the corresponding rewards, not cross-context items."""
        model, p0 = expmap.remapping_model(M=3, n_copies=10, seed=42)

        # Two reward locations in two different contexts
        pos1 = torch.tensor([[0.5, 0.5]])
        pos2 = torch.tensor([[0.5, 0.5]])

        rep_c1 = model(pos1, p0, context=0.0)
        rep_c2 = model(pos2, p0, context=5.0)

        memory_c1 = bundle(rep_c1)
        memory_c2 = bundle(rep_c2)
        composite = bind([memory_c1, memory_c2])

        # Query with context 1 at the stored position
        query_rep_c1 = model(pos1, p0, context=0.0)
        score_match = query(query_rep_c1[0], composite)

        # Cross-context query: context 1 representation against context 2's memory
        cross_score = query(query_rep_c1[0], memory_c2)

        # The matching context should produce higher similarity than cross-context
        assert score_match.item() > cross_score.item()


class TestSimilarityGradient:
    def test_shape(self):
        model, p0 = expmap.grid_cell_model(M=3, n_copies=5, seed=42)
        position = torch.tensor([1.0, 1.0])
        memory = model(torch.tensor([[0.0, 0.0]]), p0)[0].detach()

        grad = similarity_gradient(model, position, memory, p0)
        assert grad.shape == (2,)

    def test_gradient_points_toward_memory(self):
        """Gradient should generally point toward the stored location."""
        model, p0 = expmap.place_cell_model(N=60, seed=42)
        stored_pos = torch.tensor([[0.0, 0.0]])
        memory = model(stored_pos, p0)[0].detach()

        # Query from a displaced position
        query_pos = torch.tensor([0.5, 0.0])
        grad = similarity_gradient(model, query_pos, memory, p0)

        # For place-cell-like similarity (Gaussian), gradient should
        # point roughly back toward origin (negative x direction)
        assert grad[0].item() < 0, f"Expected negative x-gradient, got {grad[0].item()}"

    def test_matches_central_finite_difference(self):
        """The closed form (Gx p) . m must agree with a numerical derivative.

        similarity_gradient differentiates Eq. 1 analytically; the finite
        difference it replaced is kept here as the oracle.
        """
        model, p0 = expmap.grid_cell_model(M=3, n_copies=10, seed=42)
        memory = model(torch.tensor([[1.0, 0.5]]), p0).detach()[0]
        position = torch.tensor([0.3, -0.2])

        analytic = similarity_gradient(model, position, memory, p0)

        eps = 1e-3
        numeric = torch.zeros(2)
        for dim in range(2):
            plus, minus = position.clone(), position.clone()
            plus[dim] += eps
            minus[dim] -= eps
            with torch.no_grad():
                p = model(torch.stack([plus, minus]), p0)
            numeric[dim] = (query(p[0], memory) - query(p[1], memory)) / (2 * eps)

        torch.testing.assert_close(analytic, numeric, atol=1e-3, rtol=1e-2)

    def test_accepts_numpy_memory(self):
        """A numpy memory against a torch model used to raise TypeError."""
        model, p0 = expmap.grid_cell_model(M=3, n_copies=5, seed=42)
        memory = model(torch.tensor([[0.0, 0.0]]), p0)[0].detach().numpy()

        grad = similarity_gradient(model, torch.tensor([0.5, 0.5]), memory, p0)
        assert grad.shape == (2,)
