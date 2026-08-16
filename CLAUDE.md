# expmap

Code supplementary to the paper **"Exponential Map Models as an Interpretable Framework for
Generating Neural Spatial Representations"** (Pettersen, Haug, Bergli, Surowiec & Lepperød), plus
ongoing exploration built on the same framework. Developed at bioAI Oslo.

The framework constructs neural spatial representations algebraically rather than by training a
network. Physical coordinates map to a population vector through the matrix exponential of
generator matrices,

    p(x, y) = exp(x·Gx + y·Gy) p0,

and the navigational properties follow from algebraic conditions on the generators: commutativity
([Gx, Gy] = 0) gives exact path integration, skew-symmetry (Gᵀ = −G) gives equinorm
representations with translationally invariant similarity, and eigenvalues forming sets of roots of
unity give metric preservation and hexagonal grids. **Nothing here is fit by gradient descent**,
with one deliberate exception: the experimental-data extrapolation of Appendix K, which lives in
`expmap/fitting.py` and nowhere else.

## Two strands of work

1. **The paper.** `src/expmap/` and `notebooks/01`–`09` reproduce the manuscript figures. This code
   is the reference implementation of published results — changes to it need the verification bar
   in `model-verification`.
2. **The R exploration.** `notebooks/11`–`12`, `reports/` and `tex/` pursue an open question from
   the paper's limitations: the orthogonal mixing matrix **R** is an unfixed degree of freedom, and
   the working claim is that it parametrises a continuum of single-neuron organisations — from
   module-pure grid cells (block-diagonal R) to mixed-selectivity cells (dense R) — while leaving
   the population-level similarity structure invariant. Treat this as exploratory: the results are
   not in the paper and the framing is still moving.

Assume a task concerns the paper strand unless told otherwise.

## Layout

| Path | Contents |
| --- | --- |
| `src/expmap/` | The library |
| `tests/` | pytest suite; pure algebra and synthetic input, no data needed |
| `notebooks/` | `01`–`09` are `.ipynb` figure notebooks; `11`–`12` are `# %%` cell scripts for the R exploration. `10` is superseded — see its header |
| `figures/` | Generated output, `.svg` and `.png` side by side |
| `reports/` | Markdown analyses and discussion — the record of the exploration strand |
| `tex/` | `continuum_of_R_writeup.tex`, a standalone write-up not yet folded into the paper |
| `pdfs/` | The manuscript and reference PDFs; read them with `pdftotext` (see `reading-papers`) |
| `data/` | `gc_ratmaps.npy` and cached generator parameters |
| `KNOWN_ISSUES.md` | Known breakage and inconsistencies; check before assuming something is your fault |
| `PROGRESS.md` | Working state: decisions, verification status, next steps. Consult at session start |
| `.claude/skills/` | Conventions and context, loaded as needed |

The package is organised along the mathematical structure, and it is worth keeping it that way:

- `model.py` — `ExponentialMap` and `RemappingModel`, pure `nn.Module`s with `__init__` and
  `forward` only. They are stateless; `p0` is passed to `forward`, never stored.
- `generators.py` — the construction pipeline: eigenvalue spec → `block_diag_skew` →
  `build_generators(sigma_x, sigma_y, R)`.
- `linalg.py` — orthogonal matrices at both ends of the R continuum (`random_orthogonal_matrix`,
  `block_diagonal_orthogonal`), the canonical starting state, basis construction.
- `analysis.py` — ratemaps, batch inference, similarity, autocorrelograms, grid score.
- `navigation.py` — hyperdimensional bundling, binding, querying, similarity gradients.
- `attractor.py` — the CANN the exponential map is the on-manifold limit of (Sec. 3.1, App. J.1):
  hypersphere dynamics, Lyapunov energy, and the closed-form transport they should agree with.
- `fitting.py` — the Appendix K exception: recovering generators from measured ratemaps by
  gradient descent, and extrapolating beyond the fitted arena. The only module that learns.
- `plasticity.py` — the local learning rule of Appendix J.4, which grows skew-symmetric
  connectivity from time-lagged antisymmetric correlations.
- `simulation.py` — trajectories and uniform position sampling.
- `plotting.py` — visualisation.

There are two API tiers and both are public: the convenience constructors `grid_cell_model`,
`place_cell_model`, `remapping_model` in `__init__.py`, and the composable pieces underneath.
A change to a constructor's defaults changes published figures — say so.

## Data

- `data/gc_ratmaps.npy` — experimental grid cell ratemaps, `(102, 32, 32)` float64, one 32×32
  ratemap per unit over a square arena. Used for the Appendix K extrapolation in
  `notebooks/08`. Rate units are arbitrary (values ≈ 0–0.2).
- `data/parameters/*.pt` — cached 300×300 block-diagonal Σ matrices. `sigma_A`/`sigma_B` are the
  x and y generators, `sigma_C` the context generator; `RoU` is the roots-of-unity (grid) spectrum
  and `pc` the Gaussian (place cell) spectrum.

Arena coordinates are dimensionless and centred: positions run `[-arena_size, +arena_size]` in each
direction, and one ratemap pixel of the 32-bin experimental maps is `2/32` arena units.

## Commands

```bash
uv sync --all-extras                 # set up environment
uv run pytest                        # tests
uv run ruff check . && uv run ruff format .
uv run pre-commit run --all-files    # see KNOWN_ISSUES.md before trusting this
```

Always `uv run`; never a bare `python`, never `pip install`. Add dependencies with `uv add <pkg>`
(or `--optional dev`), never by hand-editing `pyproject.toml`. Python is pinned to 3.13. There is
no type checker configured — annotate public signatures because they are documentation, not because
a checker enforces them.

Ruff is configured with `line-length = 120` and ignores `N803`/`N806`/`N812`, which is what lets
the mathematical names (`R`, `Gx`, `Q`, `Σ`) match the paper. Keep them matching. `notebooks/` is
*intended* to be exempt from linting but the exclusion does not work — see `KNOWN_ISSUES.md`, and
scope your ruff runs to `src/ tests/` unless you mean to touch the notebooks.

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
5. **Ask before restructuring.** The published notebooks and the exploration share this package;
   propose moves and deletions rather than making them.

## Skills

Conventions live in `.claude/skills/` and load automatically when a task matches:

- **theory-context** — the framework's algebra, the code↔paper symbol map, and the appendix map.
- **model-verification** — establishing that an implementation is correct before it becomes a result.
- **code-style** — Python conventions, numerical and array idioms, seeding, testing.
- **code-documentation** — docstrings and inline comments.
- **review-checklist** — what to look for when reviewing a diff, in priority order.
- **figures** — plotting conventions and publication figures.
- **manuscript** — writing style and LaTeX conventions for `tex/` and `reports/`.
- **reading-papers** — reading the PDFs in `pdfs/` with `pdftotext`.
