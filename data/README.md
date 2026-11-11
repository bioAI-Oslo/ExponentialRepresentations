# Data

The rate maps, the extracted CA1 recordings and the fitted generators behind the experimental
fitting of Sec. 3.2 and App. K. Everything else in this directory is generated locally and is
not tracked.

| File | Contents | Written by |
| --- | --- | --- |
| `gc_ratmaps.npy` | Rate maps of 102 grid cells, one module of one rat, shape `(102, 32, 32)`, one 32 × 32 map per unit over a square arena. Rate units are arbitrary. | — |
| `ca1_events.npz` | Tracked head position and spike times of the CA1 release, reduced to one file: 114 open-field sessions of five rats and 2729 cluster-cut units, with the recording system, duration and tracking scale of every session and the tetrode, cluster and mean rate of every unit. The arrays are documented in `notebooks/experimental_data/ca1.py`. 55 MB. | `notebooks/experimental_data/extract_ca1.py` |
| `gc_fit_commuting.npz` | The generators fitted to the grid-cell maps (`Gx`, `Gy`, `R`, `wavevectors`), the maps themselves (`experimental`), the regenerated maps and the extrapolation from the central three quarters of the arena. | `notebooks/fit_grid_cells.ipynb` |
| `ca1_fit_one_animal_si0.50.npz` | The same for 103 CA1 units of one rat. The rate maps the fit was made on are stored under `experimental`, shape `(103, 32, 32)`, with the recording date, tetrode and cluster of every unit. | `notebooks/fit_place_cells.ipynb` |
| `reward_navigation_spectra.npz` | The generator eigenvalues of Fig. 5, `lambda_x`, `lambda_y` and `lambda_s`, 150 each. They were drawn from a Gaussian without a seed when the figure was made, so they are kept as data. | — |

`notebooks/ca1_population.ipynb` and `notebooks/fit_place_cells.ipynb` read `ca1_events.npz`,
and `notebooks/fitting_figures.ipynb` and `notebooks/fitted_similarity_structure.ipynb` read the
two fit files. All four run from a clone as is.

## Sources and licences

**Grid cells.** Gardner, R. J., Hermansen, E., Pachitariu, M., Burak, Y., Baas, N. A.,
Dunn, B. A., Moser, M.-B. and Moser, E. I. (2022). Toroidal topology of population activity in
grid cells. *Nature* 602, 123–128. https://doi.org/10.1038/s41586-021-04268-7. The data are
available at https://doi.org/10.6084/m9.figshare.16764508 under CC0 1.0.

**CA1 units.** Blackstad, J. S., Moser, E. I. and Moser, M.-B. (2024). Electrophysiological
recordings of hippocampal CA1 units in rats during open field foraging (v1.0). EBRAINS.
https://doi.org/10.25493/36PM-H72. Released under
[CC BY-SA 4.0](https://creativecommons.org/licenses/by-sa/4.0/).

`ca1_events.npz` and `ca1_fit_one_animal_si0.50.npz` are adapted from that dataset and are
distributed under CC BY-SA 4.0, not under the MIT licence of the code. Changes made in
`ca1_events.npz`: the tracking and the cluster files were read from the raw release and
stored as plain arrays, head position was smoothed, the tracking scale of the Neuralynx
sessions was set to match the Axona sessions, and clusters nested in a larger cluster were
marked as duplicates. Changes made in the fit file: units of one rat were selected by mean
rate and spatial information, units recorded again on a later day were removed, rate maps
were computed from the spike times and tracked positions (speed filtered, 32 × 32 bins,
Gaussian smoothing) and scaled by one constant, and generators were fitted to them.

## Regenerating the extracted recordings

`ca1_events.npz` is tracked so that the CA1 notebooks run from a clone. To rebuild it from
the raw release (18 GB), download the dataset from EBRAINS at https://doi.org/10.25493/36PM-H72
into `ca1_data/`, so that `ca1_data/data/` holds one directory per animal with a directory per
session inside, and `ca1_data/scripts/` holds the loaders shipped with the release. Then run

```bash
uv run python notebooks/experimental_data/extract_ca1.py ca1_data data/ca1_events.npz
```

The script reads only the tracking files, the `.set` headers and the MClust `.t64` cluster
files, never the waveform or LFP files.
