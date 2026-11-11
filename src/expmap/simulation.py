"""Position sampling for rate maps and similarity analyses.

Uniform samples over a square arena, seeded through a local generator so that
a figure can be regenerated without touching the global RNG.
"""

import torch


def generate_uniform_positions(
    n_batches: int, batch_size: int, arena_size: float, seed: int | None = None
) -> list[torch.Tensor]:
    """Generate batches of uniformly distributed spatial positions.

    Creates random (x, y) positions uniformly sampled within a square
    arena of half-width arena_size.

    Parameters
    ----------
    n_batches : int
        Number of position batches.
    batch_size : int
        Number of positions per batch.
    arena_size : float
        Half-width of the sampling region. Positions range from
        -arena_size to +arena_size in each dimension.
    seed : int, optional
        Seed for a local torch.Generator, leaving the global RNG untouched.
        Pass one for anything that produces a figure. If None, the global RNG
        is used and the output is not reproducible.

    Returns
    -------
    list[torch.Tensor]
        List of position tensors, each of shape (batch_size, 2).
    """
    generator = torch.Generator().manual_seed(seed) if seed is not None else None
    return [
        arena_size * (2 * torch.rand((batch_size, 2), dtype=torch.float32, generator=generator) - 1)
        for _ in range(n_batches)
    ]
