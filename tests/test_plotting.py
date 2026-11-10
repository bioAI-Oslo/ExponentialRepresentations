"""Tests for plotting layout.

Figure content is not asserted, only the grid geometry.
"""

import matplotlib as mpl
import numpy as np

mpl.use("Agg")

import matplotlib.pyplot as plt

from expmap.plotting import ratemap_collage


class TestRatemapCollage:
    def teardown_method(self):
        plt.close("all")

    def test_grid_shape_matches_rows_and_cols(self):
        """rows is the number of rows. plt.subplots(cols, rows) transposed it."""
        fig, axs = ratemap_collage(np.random.default_rng(0).random((12, 8, 8)), rows=6, cols=2)
        assert axs.shape == (6, 2)
        spec = fig.axes[0].get_subplotspec()
        assert spec is not None
        assert spec.get_gridspec().get_geometry() == (6, 2)

    def test_single_column(self):
        """squeeze=False keeps axs 2-D; this used to raise IndexError."""
        _, axs = ratemap_collage(np.random.default_rng(0).random((6, 8, 8)), rows=6, cols=1)
        assert axs.shape == (6, 1)

    def test_single_row(self):
        _, axs = ratemap_collage(np.random.default_rng(0).random((4, 8, 8)), rows=1, cols=4)
        assert axs.shape == (1, 4)

    def test_fills_row_major_and_leaves_excess_blank(self):
        _, axs = ratemap_collage(np.random.default_rng(0).random((3, 8, 8)), rows=2, cols=2)
        assert len(axs[0, 0].images) == 1
        assert len(axs[0, 1].images) == 1
        assert len(axs[1, 0].images) == 1
        assert len(axs[1, 1].images) == 0

    def test_more_ratemaps_than_panels(self):
        _, axs = ratemap_collage(np.random.default_rng(0).random((20, 8, 8)), rows=2, cols=2)
        assert all(len(ax.images) == 1 for ax in axs.ravel())
