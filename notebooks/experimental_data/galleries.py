"""Data-versus-model ratemap galleries shared by the fitting notebooks (12, 13, 15)."""

import matplotlib.pyplot as plt
import numpy as np


def gallery(data, model, scores, mask=None, box=None, n_show=32, n_cols=8, label="r"):
    """Rows alternating data and model for `n_show` units of `mask`, best to worst by `scores`.

    Parameters
    ----------
    data, model : numpy.ndarray
        Ratemaps indexed (unit, bins_x, bins_y).
    scores : numpy.ndarray
        One number per unit, shown in the title and used to order the units; the shown
        units are sampled evenly from best to worst.
    mask : numpy.ndarray of bool, optional
        Units eligible for showing. Default all.
    box : tuple[int, int], optional
        Pixel range `(lo, hi)` of a fitted square to outline in white.
    n_show, n_cols : int
        Units to show and units per row; each row of units takes two rows of panels.
    label : str
        Name of the score in the titles.

    Returns
    -------
    matplotlib.figure.Figure
    """
    n_bins = data.shape[1]
    idx = np.where(np.ones(len(data), bool) if mask is None else mask)[0]
    idx = idx[np.argsort(-scores[idx])]
    show = idx[np.linspace(0, len(idx) - 1, min(n_show, len(idx))).astype(int)]
    n_rows = 2 * int(np.ceil(len(show) / n_cols))
    fig, axs = plt.subplots(n_rows, n_cols, figsize=(n_cols * 1.0, n_rows * 1.05), layout="constrained")
    for ax in axs.ravel():
        ax.axis("off")
    for j, k in enumerate(show):
        row, col = 2 * (j // n_cols), j % n_cols
        vmax = max(np.percentile(data[k], 99), np.percentile(model[k], 99))
        axs[row, col].imshow(data[k].T, origin="lower", cmap="viridis", vmin=0, vmax=vmax)
        axs[row, col].set_title(f"{label} = {scores[k]:.2f}", fontsize=7)
        axs[row + 1, col].imshow(model[k].T, origin="lower", cmap="viridis", vmin=0, vmax=vmax)
        if box is not None:
            lo, hi = box
            for ax in (axs[row, col], axs[row + 1, col]):
                ax.add_patch(
                    plt.Rectangle((lo - 0.5, lo - 0.5), hi - lo, hi - lo, fill=False, edgecolor="w", linewidth=0.8)
                )
    for row in range(0, n_rows, 2):
        axs[row, 0].text(-4, n_bins / 2, "data", rotation=90, va="center", ha="right", fontsize=8)
        axs[row + 1, 0].text(-4, n_bins / 2, "model", rotation=90, va="center", ha="right", fontsize=8)
    return fig
