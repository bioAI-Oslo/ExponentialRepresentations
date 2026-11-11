# expmap

Code supplementary to the paper **"Exponential Map Models as an Interpretable Framework for
Generating Neural Spatial Representations"** (Pettersen, Haug, Bergli, Surowiec and Lepperød).
Developed at bioAI Oslo.

The framework constructs neural spatial representations algebraically rather than by training a
network. Physical coordinates map to a population vector through the matrix exponential of
generator matrices,

    p(x, y) = exp(x·Gx + y·Gy) p0,

and the navigational properties follow from algebraic conditions on the generators. Commuting
generators, `[Gx, Gy] = 0`, give exact path integration. Skew-symmetric generators, `Gᵀ = −G`,
give equinorm representations with translation-invariant similarity. Eigenvalues that form sets
of roots of unity give metric preservation and hexagonal grids. Nothing here is fit by gradient
descent, with one exception: the fits to recorded grid and place cells of Sec. 3.2 and
Appendix K, which live in `expmap/fitting.py` and nowhere else.

## What is here

`src/expmap/` and the notebooks reproduce the figures of the paper and the fits to recorded cells
(Sec. 3.2, App. K). This code is the reference implementation of published results. A change to
it needs the verification described in `expmap-framework`.

## Layout

| Path | Contents |
| --- | --- |
| `src/expmap/` | The library |
| `tests/` | The pytest suite. Pure algebra and synthetic input, no data needed |
| `docs/`, `_config.yml`, `_toc.yml` | The Jupyter Book (v1): the demo notebook `basic_usage.ipynb`, the developer notes and the API reference pages in `docs/`. The book is rooted at the repository, so the notebooks hang off `notebooks/README.md`, which maps them to the figures. Build with `uv run jupyter-book build .` |
| `.github/workflows/` | `code_quality.yml` (ruff, pyright, pytest) and `deploy_docs.yml` (build the book on PRs, deploy to Pages from `main`) |
| `notebooks/` | One notebook per figure, named after its content with the figure in the title. See the table below |
| `notebooks/experimental_data/` | Helpers the fitting notebooks import: `ca1.py` (loader, rate maps, field metrics), `extract_ca1.py` (the raw release to `data/ca1_events.npz`) and `galleries.py` |
| `data/` | The rate maps and fits behind Sec. 3.2 and App. K, with sources and licenses in `data/README.md`. See Data below |
| `figures/` | Generated output, `.svg` and `.png` side by side |
| `ca1_data/` | The raw CA1 release, 18 GB. Only `extract_ca1.py` reads it |
| `.claude/skills/` | Conventions and context, loaded as needed |

Tracked are `src/`, `tests/`, `notebooks/` with the three helpers named above, the files listed
under Data, the book (`docs/`, `_config.yml`, `_toc.yml`), the workflows, this file and the
skills. `figures/` and `ca1_data/` are gitignored and exist only in a working checkout. Do not
assume them in a fresh clone, and keep references to them out of the package, the notebooks and
the READMEs.

The notebooks, in the order of `notebooks/README.md`:

| Notebook | What it does |
| --- | --- |
| `roots_of_unity` | Fig. 2: eigenvalue rings, plane waves and similarity per symmetry order, and a 300-unit grid population checked against Eq. 4 |
| `remapping` | Fig. 3: remapping with a distinct orientation per hexagonal set and a block-diagonal `R`, so that pairwise phase relations within a set are unchanged by a context shift (App. M) |
| `reward_navigation` | Fig. 5: bundling, binding and gradient ascent on the retrieved reward map |
| `bessel_zeros` | Fig. 6 (App. G): the `J0` zeros, grid scale ratios and the radial similarity |
| `place_like_fields` | Fig. 7 (App. H): place-like fields from Gaussian eigenvalues |
| `learning_rule` | Fig. 8 (App. J.4): skew-symmetric connectivity from the local learning rule |
| `ca1_population` | CA1 unit selection over all rats. Writes `data/ca1_ratemaps.npz` |
| `fit_grid_cells` | Commuting fit to the grid cells. Writes `data/gc_fit_commuting.npz` |
| `fit_place_cells` | Commuting fit to every spatially tuned CA1 unit of one rat, the place-cell fit of the paper. Writes `data/ca1_fit_one_animal_si0.50.npz` |
| `fitting_figures` | The figure panels of Sec. 3.2 and App. K (Fig. 4, Figs. 9 to 11), drawn from the saved fits without refitting |
| `fitted_similarity_structure` | Similarity structure of the data against the fitted models (App. K, Fig. 12) |
| `attractor_network` | Verification of Sec. 3.1 and App. J.1, with no figure in the paper: the on-manifold dynamics of the CANN against the closed form |

The package is organized along the mathematical structure, and it stays that way:

- `model.py`: `ExponentialMap` and `RemappingModel`, pure `nn.Module`s with `__init__` and
  `forward` only. They are stateless. `p0` is passed to `forward` and never stored.
- `generators.py`: the construction pipeline, from an eigenvalue specification through
  `block_diag_skew` to `build_generators(sigma_x, sigma_y, R)`.
- `linalg.py`: orthogonal mixing matrices `R`, dense (`random_orthogonal_matrix`) and
  block-diagonal (`block_diagonal_orthogonal`), the canonical starting state and basis
  construction.
- `analysis.py`: rate maps, batch inference, similarity, autocorrelograms, grid score.
- `navigation.py`: hyperdimensional bundling, binding, querying, similarity gradients.
- `attractor.py`: the CANN whose on-manifold limit is the exponential map (Sec. 3.1, App. J.1).
  Hypersphere dynamics, Lyapunov energy, and the closed-form transport they must agree with.
- `fitting.py`: the Appendix K exception. It recovers generators from measured rate maps by
  gradient descent and extrapolates beyond the fitted arena. `CommutingGeneratorFit` learns `R`
  and the wavevectors, so the generators commute by construction. `GeneratorFit` is the earlier
  version, which penalizes the commutator instead. This is the only module that learns.
- `plasticity.py`: the local learning rule of Appendix J.4, which grows skew-symmetric
  connectivity from time-lagged antisymmetric correlations.
- `simulation.py`: uniform position sampling.
- `plotting.py`: visualization.

There are two API tiers and both are public: the convenience constructors `grid_cell_model`,
`place_cell_model` and `remapping_model` in `__init__.py`, and the composable pieces underneath.
A change to a constructor's defaults changes published figures. Say so when you make one.

## Data

Tracked, and described for outside readers in `data/README.md`:

- `data/gc_ratmaps.npy`: experimental grid cell rate maps, `(102, 32, 32)` float64, one 32×32
  map per unit over a square arena, from one module of one rat (Gardner et al. 2022, CC0).
  Fitted in `fit_grid_cells`. Rate units are arbitrary, with values of about 0 to 0.2.
- `data/ca1_events.npz`: spike times and tracking of the whole CA1 release, 114 sessions and
  2729 units, extracted from `ca1_data/` by `extract_ca1.py`. 55 MB. `ca1_population` and
  `fit_place_cells` read it. Indexing the loaded `.npz` inside a loop decompresses the whole
  array each time, so load each array once. Derived from the same CC BY-SA 4.0 data as the fit
  below.
- `data/reward_navigation_spectra.npz`: the three Gaussian spectra of Fig. 5, `lambda_x`,
  `lambda_y` and `lambda_s`, 150 values each. They were drawn without a seed when the figure
  was made and are kept as data so that `reward_navigation` regenerates the printed landscapes.
- `data/gc_fit_commuting.npz` and `data/ca1_fit_one_animal_si0.50.npz`: the saved fits of
  `fit_grid_cells` and `fit_place_cells`. They hold `Gx`, `Gy`, `R` and `wavevectors`, the
  fitted maps (`experimental`), the regenerated maps (`generated`) and the extrapolation from
  the central three quarters of the arena (`extrapolated`, `crop_Gx`, `crop_Gy`, `inner`).
  `fitting_figures` and `fitted_similarity_structure` read them. The CA1 file holds 103 units of
  one rat and is derived from CC BY-SA 4.0 data (Blackstad, Moser and Moser 2024), so it carries
  that license rather than the code's MIT.

Arena coordinates are dimensionless and centered. Positions run from `-arena_size` to
`+arena_size` in each direction, and one pixel of the 32-bin experimental maps is `2/32` arena
units.

## Commands

```bash
uv sync --all-extras                 # set up environment
uv run pytest                        # tests
uv run ruff check . && uv run ruff format . && uv run pyright
uv run pre-commit run --all-files    # the same three, plus uv-lock and the whitespace hooks
uv run jupyter nbconvert --to notebook --execute --inplace notebooks/<name>.ipynb
uv run jupyter-book build .          # the documentation, to _build/html; notebooks are not executed
```

Always `uv run`. Never a bare `python`, never `pip install`. Add dependencies with `uv add <pkg>`
or `uv add --optional dev <pkg>`, never by editing `pyproject.toml` by hand. Python is pinned to
3.13. Pyright runs in standard mode over `src`, `tests` and the notebook helpers. Annotate public
signatures, and fix a wrong hint rather than silencing it.

Ruff is configured with `line-length = 120` and ignores `N802`, `N803`, `N806` and `N812`, which
is what lets the mathematical names `R`, `Gx`, `Q` and `Σ` match the paper. Keep them matching.
Notebooks are linted but excluded from `ruff format`, so `ruff check .` and `ruff format .` are
both safe on the whole tree. Keep a notebook you touch lint-clean.

## Rules that always apply

1. **Match the paper's notation.** A symbol in the code is the symbol in the paper, and a
   docstring says which equation, section or appendix it implements. When code and paper
   disagree, the code is authoritative. Say so in the docstring.
2. **Do not over-engineer.** Defensive type checks, redundant validation and narrating log calls
   earn nothing and bury the algebra that matters. Write robust code, not paranoid code. See
   `python-style` and `expmap-framework`.
3. **Verify against the closed form.** The advantage of this framework is that the right answer
   can be derived. An implementation that has not been checked against its analytic prediction
   is not yet a result. See `expmap-framework`.
4. **Reproducibility is the bar.** Seed every stochastic operation, keep analysis parameters as
   explicit defaults rather than literals in the body, and make sure a published figure can be
   regenerated from the library.
5. **Ask before restructuring.** The published notebooks depend on this package. Propose moves
   and deletions rather than making them.

## Skills

Conventions live in `.claude/skills/` and load when a task matches:

- **python-style** and **python-review**: the general Python standard and the review order.
- **expmap-framework**: the algebra of the framework, the symbol map between code and paper, the
  appendix map, what is specific to this repository on top of the general standard, and how to
  verify an implementation against the closed form.
- **figures**: plotting conventions and publication figures.
