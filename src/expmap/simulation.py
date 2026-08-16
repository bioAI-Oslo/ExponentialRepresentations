"""Trajectory and position generation for evaluating spatial representations.

Provides utilities for generating simulated animal trajectories and uniform
spatial samples for model evaluation and ratemap computation.
"""

import numpy as np
import torch


class SimpleDatasetMaker:
    """Generate random walk trajectories in a square box with wall bouncing.

    Simulates an animal-like trajectory using a correlated random walk model:
    heading directions are drawn from a von Mises distribution (concentrated
    around the previous heading), and step sizes from a Rayleigh distribution.
    Trajectories reflect off the walls of a square enclosure.

    Parameters
    ----------
    chamber_size : float, optional
        Half-width of the square enclosure. Positions range from
        -chamber_size to +chamber_size in each dimension.
    stddev : float, optional
        Despite the name this is the *concentration* kappa of the von Mises
        heading distribution, not a standard deviation — higher values
        produce straighter trajectories. Kept for compatibility with the
        notebooks that construct this class by keyword.
    rayleigh_scale : float, optional
        Scale parameter of the Rayleigh speed distribution. Controls
        the typical step size.
    seed : int, optional
        Seed for this instance's generator. Successive `generate_dataset`
        calls draw from the same stream, so a seeded maker gives a
        reproducible *sequence* of datasets rather than the same one twice.
    """

    def __init__(self, chamber_size=1, stddev=4 * np.pi, rayleigh_scale=0.025, seed=None):
        self.chamber_size = chamber_size
        self.stddev = stddev
        self.rayleigh_scale = rayleigh_scale
        self.rng = np.random.default_rng(seed)

    def bounce(self, r, v):
        """Reflect velocity vectors at wall boundaries.

        Any component that would carry the walker past a wall is negated.
        Modifies `v` in place and returns it.

        Parameters
        ----------
        r : numpy.ndarray
            Current positions, shape (..., 2).
        v : numpy.ndarray
            Proposed velocities, shape (..., 2). Modified in place.

        Returns
        -------
        numpy.ndarray
            Corrected velocities with reflections applied.
        """
        outside = np.abs(r + v) >= self.chamber_size
        v[outside] = -v[outside]
        return v

    def generate_dataset(self, samples, timesteps, device="cpu"):
        """Generate random walk trajectories with positions and velocities.

        Parameters
        ----------
        samples : int
            Number of independent trajectories.
        timesteps : int
            Number of time steps per trajectory.
        device : str, optional
            Torch device for the output tensors.

        Returns
        -------
        tuple[torch.Tensor, torch.Tensor]
            - Positions of shape (samples, timesteps, 2).
            - Velocities of shape (samples, timesteps - 1, 2).
        """
        r = np.zeros((samples, timesteps, 2))
        s = self.rng.rayleigh(self.rayleigh_scale, (samples, timesteps))
        prev_hd = self.rng.uniform(0, 2 * np.pi, samples)
        r[:, 0] = self.rng.uniform(-self.chamber_size, self.chamber_size, (samples, 2))

        for i in range(timesteps - 1):
            hd = self.rng.vonmises(prev_hd, self.stddev, samples)
            prop_v = s[:, i, None] * np.stack((np.cos(hd), np.sin(hd)), axis=-1)
            v = self.bounce(r[:, i], prop_v)
            prev_hd = np.arctan2(v[:, 1], v[:, 0])
            r[:, i + 1] = r[:, i] + v

        v = np.diff(r, axis=1)
        return (
            torch.tensor(r.astype("float32"), device=device),
            torch.tensor(v.astype("float32"), device=device),
        )


def generate_uniform_positions(n_batches, batch_size, arena_size, seed=None):
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
