---
name: figures
description: Conventions for plotting and publication figures in expmap - separating computation from plotting, the ax parameter pattern, ratemap orientation and shared colour scales, colormaps for rate and similarity, vector output to figures/, and sizing for the NeurIPS manuscript. Use when writing or refactoring any plotting function, building a panel or multi-panel figure, choosing a colormap, or preparing a figure for the paper.
---

# Figures

Figures are the output the paper is judged on, and the place where an implementation error becomes
visible. Two rules dominate: **computation and plotting are separate**, and **a figure must be
regenerable from the library plus a recorded seed**.

## Separate computing from plotting

A `plot_*` function takes data that is already computed and draws it. A `compute_*` function
returns data and draws nothing. The package already splits this way — `analysis` computes,
`plotting` draws — and it is worth defending.

```python
ratemaps = compute_ratemaps(x, y, p, bins=32)
scores = compute_grid_scores(ratemaps)
plot_score_distribution(scores, ax=ax)
```

Never write a function that does both and signals it with a flag like `return_data=True`: it makes
the computation untestable without matplotlib, and invisible to anyone who wants the numbers rather
than the picture.

## Function signature pattern

```python
def plot_similarity(similarity, *, ax=None, **kwargs):
    if ax is None:
        _, ax = plt.subplots(figsize=(3, 3), layout="constrained")
    ...
    return ax
```

- **Accept `ax=None`, create if absent, return the axes.** It lets any single-panel function be
  composed into a multi-panel figure.
- **Do not save inside plotting functions.** Return the figure or axes; let the caller decide
  where it goes.
- Keep styling parameters as keyword arguments with sensible defaults rather than hardcoded values
  mid-body.
- `ratemap_collage` is the deliberate exception: it builds its own grid of axes and returns
  `(fig, axs)`. A function that *creates* a panel layout may own its figure; one that draws a
  single thing may not.

## Ratemap orientation

`compute_ratemaps` returns `(n_units, bins_x, bins_y)` — **the first spatial axis is x**. A raw
`imshow(ratemap)` puts that axis vertically and increasing downward, so the map is shown transposed
and y-flipped relative to the arena. For a symmetric grid pattern this is invisible; for a place
field, a similarity map with a marked reference location, or anything overlaid on a trajectory, it
is wrong.

Use `ax.imshow(ratemap.T, origin="lower", extent=[-a, a, -a, a])` when orientation carries meaning,
and say in the caption that the map is in arena coordinates. When you overlay a point — a reference
bin, a reward location, a trajectory — verify it lands where you expect before trusting the panel.

Arena coordinates are dimensionless and centred on the origin, so an axis label is `x` or
`Position (arena units)`, never a physical unit. The experimental ratemaps in `data/` are the one
exception: 32 bins spanning the recorded arena, one pixel being `2/32` model units.

## Colour

- **Rate and activity:** `viridis`. Perceptually uniform and colorblind-safe.
- **Similarity:** a diverging map with **zero pinned at the centre**. `C` runs on `[-1, 1]` and the
  sign is meaningful — a sequential map hides where similarity crosses zero, which is exactly the
  structure a similarity figure exists to show. The exception is the similarity of recorded,
  non-negative rate maps, which never crosses zero: use a sequential map over the occupied range,
  shared between data and model, and say so.
- **Categorical:** `sns.color_palette("colorblind", n)`, or matplotlib's default cycle for small n.
- **Never** `jet` or `rainbow`. They fabricate structure that is not in the data.
- Never encode a distinction by colour alone. Pair it with marker, linestyle, or position, so the
  figure survives greyscale printing.
- Keep a condition's colour constant across every figure — data always the same colour, model
  always the same colour.

## Shared colour scales

The single most common way a ratemap panel misleads. When panels are meant to be compared —
data and model, different contexts, different modules — **they must share `vmin`/`vmax`**.
Independently normalised panels make a weak field look identical to a strong one.

```python
vmin, vmax = ratemaps.min(), ratemaps.max()
ratemap_collage(ratemaps_context_a, vmin=vmin, vmax=vmax)
ratemap_collage(ratemaps_context_b, vmin=vmin, vmax=vmax)
```

`ratemap_collage` takes `vmin`/`vmax` for this reason. When panels are *deliberately* normalised
per-unit — to compare shape rather than amplitude, as in the data-over-model galleries of the
fitting figures — say so in the caption, and clip the scale at a percentile of the data map
rather than its single maximum.

## Layout and output

- Use `layout="constrained"`; never `tight_layout`.
- **Save vector `.svg`** for anything going into the manuscript, with `bbox_inches="tight"`. The
  existing convention in `figures/` is to write `.svg` and `.png` under the same stem — keep it;
  the PNG is for quick looks and README embedding.
- **Draw every panel at its final printed size.** The NeurIPS text block is 5.5 in wide, so a
  full-width panel is 5.5 in and panels that share a row divide that width. Set the font sizes
  explicitly — 8 pt labels, 7 pt ticks — in the paper's Times (`font.family = "serif"`,
  `font.serif = ["Times New Roman"]`, `mathtext.fontset = "stix"`), and save with
  `bbox_inches="tight", pad_inches=0.02`, so LaTeX includes the panel at `scale=1`. Never draw
  large and scale down; that shrinks the text to illegibility and makes sizes inconsistent across
  panels.
- **One file per panel.** Paper figures are assembled from single panels in a vector editor, and
  the panel letters are added there. Write each panel as its own `.svg` with the text kept as text
  (`plt.rcParams["svg.fonttype"] = "none"`), never as a stacked or lettered composite.
- Label every axis. An unlabeled axis is a bug — except inside a ratemap collage, where axes are
  deliberately off and the arena extent belongs in the caption.
- Don't put a title on a paper panel; the caption carries it. Titles are for exploratory plots.
- Panel letters are `a)`, `b)`, matching the manuscript's captions, and are not drawn by
  matplotlib (see above).

## Reproducibility

- A paper figure is produced by one numbered notebook that calls into `expmap` and does no
  analysis of its own beyond drawing panels — a notebook, not a script beside it, and one that
  displays the figures it writes (do not `plt.close` after saving). The README maps notebooks to
  manuscript figures; keep that mapping accurate when it changes.
- Panels that depend on a long computation are drawn from saved results. `notebooks/14` draws the
  fitting panels from the saved fits and refits nothing, so a panel can be restyled in seconds.
- **Record the seed in the notebook**, at the top, as a named constant. `R` is randomly sampled
  and it determines every tuning curve in the figure — an unseeded ratemap panel cannot be
  regenerated. The sampling functions take a `seed` that defaults to `None`; pass it.
- Every panel must be regenerable from the library plus that seed. If a figure depends on a
  parameter that was typed once into a cell and lost, it is not reproducible.
- Prefer a shared `.mplstyle` file over repeated `rcParams` assignments once the paper figures
  stabilise; there is no style file yet, and the notebooks currently set the font family and sizes
  inline, which is the natural thing to move first.
