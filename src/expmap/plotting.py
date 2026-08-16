"""Visualization utilities for exponential map spatial representations."""

import matplotlib.pyplot as plt


def ratemap_collage(ratemaps, rows=5, cols=5, figsize=(5, 5), cmap="viridis", vmin=None, vmax=None, **kwargs):
    """Plot a grid collage of spatial ratemaps.

    Arranges multiple ratemaps in a tiled grid layout with minimal spacing,
    suitable for visualizing ensemble spatial selectivity patterns. Ratemaps
    are filled in row-major order and any excess panels are left blank.

    Note that `imshow` renders `(bins_x, bins_y)` ratemaps from
    `compute_ratemaps` transposed and y-flipped relative to arena coordinates.
    That is invisible for symmetric grid patterns but wrong for place fields
    and similarity maps — see the `figures` skill.

    Parameters
    ----------
    ratemaps : numpy.ndarray
        Array of shape (n_units, bins, bins) containing the ratemaps.
    rows : int, optional
        Number of rows in the grid.
    cols : int, optional
        Number of columns in the grid.
    figsize : tuple, optional
        Figure size in inches.
    cmap : str, optional
        Matplotlib colormap name.
    vmin : float, optional
        Minimum value for color scaling.
    vmax : float, optional
        Maximum value for color scaling.
    **kwargs
        Additional keyword arguments passed to plt.subplots.

    Returns
    -------
    tuple[matplotlib.figure.Figure, numpy.ndarray]
        Figure and axes array of shape (rows, cols).
    """
    # squeeze=False keeps axs 2-D when rows or cols is 1.
    fig, axs = plt.subplots(rows, cols, figsize=figsize, squeeze=False, **kwargs)

    count = 0
    for row in range(rows):
        for col in range(cols):
            axs[row, col].axis("off")
            if count < len(ratemaps):
                axs[row, col].imshow(ratemaps[count], cmap=cmap, vmin=vmin, vmax=vmax)
            count += 1
    plt.subplots_adjust(wspace=0.1, hspace=0.1)
    return fig, axs
