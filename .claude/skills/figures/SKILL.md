---
name: figures
description: Conventions for plotting and publication figures in expmap - computation separate from plotting, the ax parameter pattern, rate map orientation and shared color scales, colormaps for rate and similarity, vector output to figures/, and sizing for the NeurIPS manuscript. Use when writing or refactoring any plotting function, building a panel or multi-panel figure, choosing a colormap, or preparing a figure for the paper.
---

# Figures

Figures are the output the paper is judged on, and the place where an implementation error
becomes visible. Two rules carry the rest. Computation and plotting are separate, and a figure
regenerates from the library plus a recorded seed.

## Computing and plotting

A `plot_*` function takes data that is already computed and draws it. A `compute_*` function
returns data and draws nothing. The package splits this way, `analysis` computes and `plotting`
draws, and the split is kept.

```python
ratemaps = compute_ratemaps(x, y, p, bins=32)
scores = compute_grid_scores(ratemaps)
plot_score_distribution(scores, ax=ax)
```

Never write a function that does both and signals it with a flag such as `return_data=True`.
It makes the computation untestable without matplotlib and hides the numbers from anyone who
wants them rather than the picture.

A plotting function accepts `ax=None`, creates the axes if absent and returns them, so that any
single-panel function composes into a multi-panel figure:

```python
def plot_similarity(similarity, *, ax=None, **kwargs):
    if ax is None:
        _, ax = plt.subplots(figsize=(3, 3), layout="constrained")
    ...
    return ax
```

It does not save. The caller decides where the figure goes. Styling parameters are keyword
arguments with defaults, not literals in the body. `ratemap_collage` is the one exception: it
builds its own grid of axes and returns `(fig, axs)`. A function that creates a panel layout may
own its figure. One that draws a single thing may not.

## Rate map orientation

`compute_ratemaps` returns `(n_units, bins_x, bins_y)`, with x as the first spatial axis. A raw
`imshow(ratemap)` puts that axis vertical and increasing downward, so the map appears transposed
and flipped in y relative to the arena. A symmetric grid hides this. A place field, a similarity
map with a marked reference location or anything drawn over a trajectory shows it.

Use `ax.imshow(ratemap.T, origin="lower", extent=[-a, a, -a, a])` when orientation carries
meaning, and say in the caption that the map is in arena coordinates. When a point is drawn on
top, a reference bin, a reward or a trajectory, check that it lands where it should before
trusting the panel.

Arena coordinates are dimensionless and centered on the origin. An axis label is `x` or
`Position (arena units)`, never a physical unit. The recorded rate maps in `data/` are the one
exception: 32 bins over the recorded arena, one pixel being `2/32` arena units.

## Color

- Rate and activity: `viridis`, which is perceptually uniform and safe for color blindness.
- Similarity: a diverging map with zero at the center. `C` runs on `[-1, 1]` and the sign is
  meaningful. A sequential map hides where the similarity crosses zero, which is the structure
  the figure exists to show. The exception is the similarity of recorded, non-negative rate
  maps, which never crosses zero. Use a sequential map over the occupied range, shared between
  data and model, and say so.
- Categorical: `sns.color_palette("colorblind", n)`, or matplotlib's default cycle for small n.
- Never `jet` or `rainbow`. They invent structure that is not in the data.
- Never encode a distinction by color alone. Pair it with a marker, a line style or a position,
  so that the figure survives grayscale printing.
- A condition keeps its color across every figure. Data is always one color and the model always
  another.

## Shared color scales

Panels meant to be compared share `vmin` and `vmax`: data and model, two contexts, two modules.
Independently normalized panels make a weak field look like a strong one. `ratemap_collage`
takes `vmin` and `vmax` for this reason.

```python
vmin, vmax = ratemaps.min(), ratemaps.max()
ratemap_collage(ratemaps_context_a, vmin=vmin, vmax=vmax)
ratemap_collage(ratemaps_context_b, vmin=vmin, vmax=vmax)
```

When panels are normalized per unit on purpose, to compare shape rather than amplitude as in the
data-over-model galleries of the fitting figures, say so in the caption and clip the scale at a
percentile of the data map rather than at its maximum.

## Layout and output

- `layout="constrained"`, never `tight_layout`.
- Anything going into the manuscript is saved as `.svg` with `bbox_inches="tight"`, and `.png`
  under the same stem for quick looks. Diagnostic figures are displayed in the notebook and not
  saved.
- Every panel is drawn at its printed size. The NeurIPS text block is 5.5 in wide, so a
  full-width panel is 5.5 in and panels that share a row divide that width. Font sizes are set
  explicitly, 8 pt labels and 7 pt ticks, in the paper's Times (`font.family = "serif"`,
  `font.serif = ["Times New Roman"]`, `mathtext.fontset = "stix"`), and the file is saved with
  `bbox_inches="tight", pad_inches=0.02`, so that LaTeX includes it at `scale=1`. Never draw
  large and scale down. That shrinks the text and makes sizes inconsistent across panels.
- One file per panel. Paper figures are assembled from single panels in a vector editor, where
  the panel letters `a)`, `b)` are added. Each panel is its own `.svg` with the text kept as text
  (`plt.rcParams["svg.fonttype"] = "none"`), never a stacked or lettered composite.
- Every axis has a label, except inside a rate map collage, where the axes are off and the arena
  extent belongs in the caption.
- A paper panel has no title. The caption carries it. Titles are for exploratory plots.

## Reproducibility

- A paper figure is produced by one notebook that calls into `expmap` and does no analysis of its
  own beyond drawing panels, and the notebook displays the figures it writes. The table in
  `notebooks/README.md` maps notebooks to figures. Keep it accurate when the mapping changes.
- Panels that depend on a long computation are drawn from saved results. `fitting_figures` draws
  the fitting panels from the saved fits and refits nothing, so a panel can be restyled in
  seconds.
- The seed is a named constant at the top of the notebook. `R` is sampled at random and
  determines every tuning curve in the figure, so an unseeded panel cannot be regenerated. The
  sampling functions take a `seed` that defaults to `None`. Pass it.
- Every panel regenerates from the library plus that seed. A figure that depends on a parameter
  typed once into a cell and lost is not reproducible.
- The notebooks set the font family and sizes with `rcParams` in their first cell. A shared
  `.mplstyle` file is the place for them if the styling is ever revised across notebooks.
