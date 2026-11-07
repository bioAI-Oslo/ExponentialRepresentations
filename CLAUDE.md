# expmap

Code supplementary to the paper **"Exponential Map Models as an Interpretable Framework for
Generating Neural Spatial Representations"** (Pettersen, Haug, Bergli, Surowiec & Lepperød).
Developed at bioAI Oslo.

The framework constructs neural spatial representations algebraically rather than by training a
network. Physical coordinates map to a population vector through the matrix exponential of
generator matrices,

    p(x, y) = exp(x·Gx + y·Gy) p0,

and the navigational properties follow from algebraic conditions on the generators: commutativity
([Gx, Gy] = 0) gives exact path integration, skew-symmetry (Gᵀ = −G) gives equinorm
representations with translationally invariant similarity, and eigenvalues forming sets of roots of
unity give metric preservation and hexagonal grids. **Nothing here is fit by gradient descent**,
with one deliberate exception: the fits to recorded grid and place cells of Sec. 3.2 and
Appendix K, which live in `expmap/fitting.py` and nowhere else.

## What is here

`src/expmap/`, `notebooks/01`–`09` and `17` reproduce the manuscript figures, and `notebooks/10`–`16`
are the experimental fitting added for the camera-ready version (Sec. 3.2, App. K). This code is the
reference implementation of published results — changes to it need the verification bar in
`model-verification`.

## Layout

| Path | Contents |
| --- | --- |
| `src/expmap/` | The library |
| `tests/` | pytest suite; pure algebra and synthetic input, no data needed |
| `notebooks/` | `.ipynb` notebooks. `01`–`09` and `17` reproduce the manuscript figures; `10`–`16` are the experimental fitting — see below |
| `notebooks/experimental_data/` | Helpers the fitting notebooks import: `ca1.py` (loader, ratemaps, field metrics), `extract_ca1.py` (raw release → `data/ca1_events.npz`), `galleries.py` |
| `data/` | The rate maps and fits behind Sec. 3.2 / App. K, with sources and licences in `data/README.md` — see Data |
| `figures/` | Generated output, `.svg` and `.png` side by side |
| `paper/` | A mirror of the manuscript source, edited elsewhere: read it, do not edit it |
| `pdfs/` | The manuscript and reference PDFs; read them with `pdftotext` (see `reading-papers`) |
| `ca1_data/` | The raw CA1 release (18 GB); only `extract_ca1.py` reads it |
| `KNOWN_ISSUES.md` | Known breakage and inconsistencies; check before assuming something is your fault |
| `PROGRESS.md` | Working state: decisions, verification status, next steps. Consult at session start |
| `.claude/skills/` | Conventions and context, loaded as needed |

Only `src/`, `tests/`, `notebooks/` (with the three helpers named above), the files listed under
Data, this file and the skills are tracked. `figures/`, `paper/`, `pdfs/`, `ca1_data/`,
`PROGRESS.md`, `KNOWN_ISSUES.md` and any other working directories are gitignored and exist only
in a working checkout. Do not assume them in a fresh clone, and keep references to them out of
the package, the notebooks and the READMEs. Work that is not part of the paper stays out of
tracked files altogether, this file and the skills included.

The fitting notebooks, and the remapping notebook that replaced `03`:

| Notebook | What it does |
| --- | --- |
| `08` | The penalised grid-cell fit of the submitted version (`GeneratorFit`); kept as the record |
| `10` | CA1 unit selection over all rats, writes `data/ca1_ratemaps.npz` |
| `11`, `13` | The penalised and the commuting fit to the pooled single-field CA1 population; superseded by `15` |
| `12` | Commuting fit to the grid cells, writes `data/gc_fit_commuting.npz` |
| `15` | Commuting fit to every spatially tuned CA1 unit of one rat — the place-cell fit of the paper — writes `data/ca1_fit_one_animal_si*.npz` |
| `14` | The Sec. 3.2 / App. K figure panels, drawn from the saved fits without refitting |
| `16` | Similarity structure of the data against the fitted models; exploratory, not in the paper |
| `17` | The camera-ready Fig. 3: remapping with a distinct orientation per hexagonal set and a block-diagonal `R`, so that pairwise phase relations within a set are unchanged by a context shift (App. I.1). `03` is the submitted version's construction, kept as the record |

The package is organised along the mathematical structure, and it is worth keeping it that way:

- `model.py` — `ExponentialMap` and `RemappingModel`, pure `nn.Module`s with `__init__` and
  `forward` only. They are stateless; `p0` is passed to `forward`, never stored.
- `generators.py` — the construction pipeline: eigenvalue spec → `block_diag_skew` →
  `build_generators(sigma_x, sigma_y, R)`.
- `linalg.py` — orthogonal mixing matrices `R`, dense (`random_orthogonal_matrix`) and
  block-diagonal (`block_diagonal_orthogonal`), the canonical starting state, basis construction.
- `analysis.py` — ratemaps, batch inference, similarity, autocorrelograms, grid score.
- `navigation.py` — hyperdimensional bundling, binding, querying, similarity gradients.
- `attractor.py` — the CANN the exponential map is the on-manifold limit of (Sec. 3.1, App. J.1):
  hypersphere dynamics, Lyapunov energy, and the closed-form transport they should agree with.
- `fitting.py` — the Appendix K exception: recovering generators from measured ratemaps by
  gradient descent, and extrapolating beyond the fitted arena. `GeneratorFit` penalises the
  commutator (submitted version); `CommutingGeneratorFit` learns `R` and the wavevectors, so the
  generators commute by construction (camera-ready). The only module that learns.
- `plasticity.py` — the local learning rule of Appendix J.4, which grows skew-symmetric
  connectivity from time-lagged antisymmetric correlations.
- `simulation.py` — trajectories and uniform position sampling.
- `plotting.py` — visualisation.

There are two API tiers and both are public: the convenience constructors `grid_cell_model`,
`place_cell_model`, `remapping_model` in `__init__.py`, and the composable pieces underneath.
A change to a constructor's defaults changes published figures — say so.

## Data

Tracked, and described for outside readers in `data/README.md`:

- `data/gc_ratmaps.npy` — experimental grid cell ratemaps, `(102, 32, 32)` float64, one 32×32
  ratemap per unit over a square arena; one module of one rat (Gardner et al. 2022, CC0). Fitted
  in `notebooks/08` and `12`. Rate units are arbitrary (values ≈ 0–0.2).
- `data/gc_fit_commuting.npz`, `data/ca1_fit_one_animal_si0.50.npz` — the saved fits of
  `notebooks/12` and `15`: `Gx`, `Gy`, `R`, `wavevectors`, the fitted maps (`experimental`), the
  regenerated maps (`generated`) and the extrapolation from the central three quarters of the
  arena (`extrapolated`, `crop_Gx`, `crop_Gy`, `inner`). `notebooks/14` and `16` read these. The
  CA1 file holds 103 units of one rat and is derived from CC BY-SA 4.0 data (Blackstad, Moser &
  Moser 2024), so it carries that licence rather than the code's MIT.

Untracked:

- `data/ca1_events.npz` — spike times and tracking extracted from `ca1_data/` by
  `extract_ca1.py` (55 MB). `notebooks/10`, `11`, `13` and `15` need it. Indexing the loaded
  `.npz` inside a loop decompresses the whole array each time; load each array once.
- The remaining `data/*.npz` — outputs of superseded fits and one-off sweeps.
- `data/parameters/*.pt` — cached 300×300 block-diagonal Σ matrices from the first version of
  the figure notebooks; nothing reads them any more.

Arena coordinates are dimensionless and centred: positions run `[-arena_size, +arena_size]` in each
direction, and one ratemap pixel of the 32-bin experimental maps is `2/32` arena units.

## Commands

```bash
uv sync --all-extras                 # set up environment
uv run pytest                        # tests
uv run ruff check . && uv run ruff format src tests
uv run pre-commit run --all-files    # see KNOWN_ISSUES.md before trusting this
uv run jupyter nbconvert --to notebook --execute --inplace notebooks/<name>.ipynb
```

Always `uv run`; never a bare `python`, never `pip install`. Add dependencies with `uv add <pkg>`
(or `--optional dev`), never by hand-editing `pyproject.toml`. Python is pinned to 3.13. There is
no type checker configured — annotate public signatures because they are documentation, not because
a checker enforces them.

Ruff is configured with `line-length = 120` and ignores `N803`/`N806`/`N812`, which is what lets
the mathematical names (`R`, `Gx`, `Q`, `Σ`) match the paper. Keep them matching. `notebooks/` is
*intended* to be exempt from linting but the exclusion does not work — see `KNOWN_ISSUES.md`. The
whole tree passes `ruff check`, so keep a notebook you touch clean. `ruff format` would rewrite
several notebooks nobody is working on; format `src tests` and the notebook in hand, not `.`.

## Rules that always apply

1. **Match the paper's notation.** A symbol in the code should be the symbol in the manuscript, and
   a docstring should say which equation, section or appendix it implements. When code and paper
   disagree, the code is authoritative — record the divergence in `KNOWN_ISSUES.md`.
2. **Do not over-engineer.** Defensive type checks, redundant validation and narrating log calls
   earn nothing and bury the load-bearing algebra — the characteristic failure mode of AI-assisted
   code. Write robust code, not paranoid code. See `code-style`.
3. **Verify against the closed form.** This framework's advantage is that the right answer is
   derivable. An implementation that has not been checked against its analytic prediction is not
   yet a result. See `model-verification`.
4. **Reproducibility is the bar.** Seed every stochastic operation, keep analysis parameters as
   explicit defaults rather than buried literals, and ensure a published figure can be regenerated
   from the library.
5. **Ask before restructuring.** The published notebooks depend on this package;
   propose moves and deletions rather than making them.

## Skills

Conventions live in `.claude/skills/` and load automatically when a task matches:

- **theory-context** — the framework's algebra, the code↔paper symbol map, and the appendix map.
- **model-verification** — establishing that an implementation is correct before it becomes a result.
- **code-style** — Python conventions, numerical and array idioms, seeding, testing.
- **code-documentation** — docstrings and inline comments.
- **review-checklist** — what to look for when reviewing a diff, in priority order.
- **figures** — plotting conventions and publication figures.
- **manuscript** — writing style for manuscript text, captions and write-ups.
- **reading-papers** — reading the PDFs in `pdfs/` with `pdftotext`.
