# Data

The rate maps and fitted generators behind the experimental fitting of Sec. 3.2 and App. K.
Everything else in this directory is generated locally and is not tracked.

| File | Contents | Written by |
| --- | --- | --- |
| `gc_ratmaps.npy` | Rate maps of 102 grid cells, one module of one rat, shape `(102, 32, 32)`, one 32 × 32 map per unit over a square arena. Rate units are arbitrary. | — |
| `gc_fit_commuting.npz` | The generators fitted to the grid-cell maps (`Gx`, `Gy`, `R`, `wavevectors`), the maps themselves (`experimental`), the regenerated maps and the extrapolation from the central three quarters of the arena. | `notebooks/12_grid_fitting_commuting.ipynb` |
| `ca1_fit_one_animal_si0.50.npz` | The same for 103 CA1 units of one rat. The rate maps the fit was made on are stored under `experimental`, shape `(103, 32, 32)`, with the recording date, tetrode and cluster of every unit. | `notebooks/15_ca1_one_animal_fitting.ipynb` |

`notebooks/14_paper_figures.ipynb` and `notebooks/16_similarity_structure.ipynb` read the two
fit files and run from a clone as is.

## Sources and licences

**Grid cells.** Gardner, R. J., Hermansen, E., Pachitariu, M., Burak, Y., Baas, N. A.,
Dunn, B. A., Moser, M.-B. and Moser, E. I. (2022). Toroidal topology of population activity in
grid cells. *Nature* 602, 123–128. https://doi.org/10.1038/s41586-021-04268-7. The data are
available at https://doi.org/10.6084/m9.figshare.16764508 under CC0 1.0.

**CA1 units.** Blackstad, J. S., Moser, E. I. and Moser, M.-B. (2024). Electrophysiological
recordings of hippocampal CA1 units in rats during open field foraging (v1.0). EBRAINS.
https://doi.org/10.25493/36PM-H72. Released under
[CC BY-SA 4.0](https://creativecommons.org/licenses/by-sa/4.0/).

`ca1_fit_one_animal_si0.50.npz` is adapted from that dataset and is distributed under
CC BY-SA 4.0, not under the MIT licence of the code. Changes made: units of one rat were
selected by mean rate and spatial information, units recorded again on a later day were
removed, rate maps were computed from the spike times and tracked positions (speed filtered,
32 × 32 bins, Gaussian smoothing) and scaled by one constant, and generators were fitted to
them.

## Regenerating the CA1 fit

`notebooks/15_ca1_one_animal_fitting.ipynb` reads `data/ca1_events.npz`, the spike times and
positions extracted from the raw release, and so do the earlier CA1 notebooks 10, 11 and 13.
That file is 55 MB and is not tracked. To rebuild it, download the dataset from EBRAINS into
`ca1_data/` and run

```bash
uv run python notebooks/experimental_data/extract_ca1.py ca1_data data/ca1_events.npz
```
