# Reproducing the results

The notebooks reproduce the figures of the paper and are the record of the fits to recorded
cells. Each one calls into `expmap` and does no analysis of its own beyond drawing the panels,
and each names the figure it reproduces in its title. They are stored with their outputs, so the
figures can be read without running anything. To run one again:

```bash
uv run jupyter nbconvert --to notebook --execute --inplace notebooks/<name>.ipynb
```

Figures are written to `figures/` at the repository root, as `.svg` and `.png` side by side, one
file per panel. The figures of the paper are assembled from these panels.

## Figures

| Notebook | Description | Figure |
| --- | --- | --- |
| [`roots_of_unity`](roots_of_unity.ipynb) | Eigenvalue rings, plane waves and similarity for several symmetry orders, and a population of grid cells checked against the closed form | Fig. 2 |
| [`remapping`](remapping.ipynb) | Context-dependent remapping with rigid pairwise phase relations | Fig. 3 |
| [`reward_navigation`](reward_navigation.ipynb) | Multi-map navigation by bundling and binding | Fig. 5 |
| [`bessel_zeros`](bessel_zeros.ipynb) | Bessel function zeros and grid scale ratios | Fig. 6 (App. G) |
| [`place_like_fields`](place_like_fields.ipynb) | Place-like fields from Gaussian eigenvalues | Fig. 7 (App. H) |
| [`learning_rule`](learning_rule.ipynb) | Skew-symmetric connectivity from a local learning rule | Fig. 8 (App. J.4) |

Fig. 1 is a schematic. Fig. 4 and Figs. 9 to 12 come from the fits below.

## Fitting to recorded cells

The fits of Sec. 3.2 and App. K. All five run from a clone: the CA1 recordings are in
`data/ca1_events.npz`, see the [data page](../data/README.md). The two fits write the saved
files in `data/` that the two figure notebooks read, so the panels can be redrawn without
refitting.

| Notebook | Description | Figure |
| --- | --- | --- |
| [`ca1_population`](ca1_population.ipynb) | CA1 unit selection over all rats | — |
| [`fit_grid_cells`](fit_grid_cells.ipynb) | Commuting fit to the grid cells, writes `data/gc_fit_commuting.npz` | Fig. 4, App. K |
| [`fit_place_cells`](fit_place_cells.ipynb) | Commuting fit to the CA1 units of one rat, writes `data/ca1_fit_one_animal_si0.50.npz` | Fig. 4, App. K |
| [`fitting_figures`](fitting_figures.ipynb) | The figure panels, drawn from the saved fits | Fig. 4, Figs. 9 to 11 |
| [`fitted_similarity_structure`](fitted_similarity_structure.ipynb) | Similarity structure of the data and the fitted models | Fig. 12 (App. K) |

## Verification

| Notebook | Description | Reference |
| --- | --- | --- |
| [`attractor_network`](attractor_network.ipynb) | The attractor network whose on-manifold dynamics is the exponential map, checked against the closed form | Sec. 3.1, App. J.1 |

The helpers the fitting notebooks import live in [`experimental_data/`](https://github.com/bioAI-Oslo/ExponentialRepresentations/tree/main/notebooks/experimental_data):
`ca1.py` loads the recordings and computes rate maps and field metrics, `extract_ca1.py` turns
the raw release into `data/ca1_events.npz`, and `galleries.py` draws the rate map galleries.
