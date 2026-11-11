"""Visualization utilities for exponential map spatial representations."""

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.figure import Figure


def ratemap_collage(
    ratemaps: np.ndarray,
    rows: int = 5,
    cols: int = 5,
    figsize: tuple[float, float] = (5, 5),
    cmap: str = "viridis",
    vmin: float | None = None,
    vmax: float | None = None,
    **kwargs,
) -> tuple[Figure, np.ndarray]:
    """Plot a grid collage of spatial rate maps.

    Rate maps fill the grid in row-major order and any excess panels are left
    blank. All panels share one color scale when ``vmin`` and ``vmax`` are given,
    which makes them comparable.

    ``imshow`` draws the first array axis downward, so a ``(bins_x, bins_y)`` map
    from ``compute_ratemaps`` appears rotated relative to arena coordinates. That
    is harmless for grid patterns, whose orientation is arbitrary anyway, but
    wrong wherever position carries meaning, such as place fields or similarity
    maps with a marked reference. For those pass
    ``ratemaps.transpose(0, 2, 1)[:, ::-1]``, which puts x horizontal and y
    upward.

    Parameters
    ----------
    ratemaps : numpy.ndarray
        Rate maps of shape (n_units, bins, bins).
    rows, cols : int, optional
        Grid geometry.
    figsize : tuple[float, float], optional
        Figure size in inches, (width, height).
    cmap : str, optional
        Matplotlib colormap name.
    vmin, vmax : float, optional
        Color scale limits, passed to ``imshow``. Pass the same pair to every
        collage that is meant to be compared.
    **kwargs
        Passed to ``plt.subplots``.

    Returns
    -------
    tuple[matplotlib.figure.Figure, numpy.ndarray]
        Figure and axes array of shape (rows, cols).
    """
    # squeeze=False keeps axs 2-D when rows or cols is 1.
    fig, axs = plt.subplots(rows, cols, figsize=figsize, squeeze=False, **kwargs)

    for i, ax in enumerate(axs.flat):
        ax.axis("off")
        if i < len(ratemaps):
            ax.imshow(ratemaps[i], cmap=cmap, vmin=vmin, vmax=vmax)
    fig.subplots_adjust(wspace=0.1, hspace=0.1)
    return fig, axs
