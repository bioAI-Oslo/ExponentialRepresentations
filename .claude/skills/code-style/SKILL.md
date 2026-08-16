---
name: code-style
description: Python conventions for the expmap repository - restraint over defensive boilerplate, where code lives, the numpy/torch boundary, float32 and matrix naming, seeding, logging, and testing algebra against its closed form. Use when writing, refactoring or reviewing Python in this repo, when porting an analysis out of a notebook, and when deciding whether validation or logging is warranted.
---

# Python code style

The standard here is not elegance but **reproducibility**: a reviewer, a collaborator, or you in
two years must be able to re-run a figure and get the same numbers, and must be able to read the
code against the paper's derivation. Aim well above typical research code, but get there through
clarity, not machinery.

## Restraint comes first

Defensive checks, redundant validation and narrating log calls bury the load-bearing algebra and
earn nothing — it is the characteristic failure mode of AI-assisted code, so hold generated code to
it especially. Do not write it, and remove it when you touch surrounding code.

**Delete or don't write:**

- `isinstance` checks on arguments that go straight into a NumPy, SciPy or torch call. Those
  libraries already type-check, and their errors are better than yours.
- Shape assertions immediately before an operation that would fail on the next line anyway.
- `try`/`except` that catches an exception only to re-raise it with reworded text.
- `logger.info` on function entry and exit, or reporting counts nobody will read. "Constructed 150
  eigenvalues" tells the reader nothing they cannot see in `len()`.
- `None` handling for parameters that are always passed, and defensive `.copy()` of arrays nothing
  mutates.
- Wrapper functions that only forward arguments.

**Validation is worth writing** when the failure would otherwise be silent or distant:

- A parameter that yields a plausible but wrong result rather than an error — a spatial frequency
  above the ratemap's Nyquist limit, an arena too small to contain a grid period, an odd `N` where
  2×2 blocks are assumed.
- A matrix that is supposed to be orthogonal or skew-symmetric and silently is not, at the point
  where it enters the construction.
- Preconditions at the public entry point of a construction, checked once, rather than repeated in
  every helper it calls.

The test: *would the failure be loud and immediate anyway?* If yes, let it fail. If it would
produce a ratemap that looks fine, check it.

## Where code lives

- `src/expmap/` — the library: importable, testable, deterministic. Organised along the
  mathematical structure (`model`, `generators`, `linalg`, `analysis`, `navigation`, `simulation`,
  `plotting`); keep that split.
- `tests/` — runs on pure algebra and synthetic input; never needs `data/`.
- `notebooks/` — exploration and figure generation. Analysis logic is allowed here while it is
  being worked out, but anything reusable, or that a paper figure depends on, belongs in the
  package. A notebook that has grown its own model class is a signal to move it.
- `reports/` — prose and results of the exploration strand, not code.

## Functions and classes

Prefer small, pure functions. Reach for a class only when there is real state to carry, as in
`ExponentialMap`; a class that is only a namespace should be a module.

- **Models stay stateless.** `ExponentialMap` and `RemappingModel` have `__init__` and `forward`
  and nothing else. `p0` is an argument to `forward`, never an attribute — this is deliberate, so
  one set of generators can be probed with different starting states.
- Generators are registered as buffers, so `.to(device)` and `state_dict` work. Keep it that way.
- No hidden state: no mutable module-level globals, no reading configuration mid-computation.
  Everything a function needs arrives through its parameters.
- Analysis parameters — bin counts, arena sizes, tolerances, annulus radii — are keyword arguments
  with explicit defaults, never literals buried in the body. The default is part of the method and
  will be asked about in review.
- Private helpers are `_`-prefixed and live beside their caller.
- Return plain data: arrays, tensors, dataclasses. Beyond three return values, use a `NamedTuple`.

## Typing

Annotate public signatures; leave obvious internal one-liners bare. There is **no type checker
configured** in this project, so annotations are documentation and nothing enforces them — which
makes it worse, not better, to write one that is wrong.

`np.ndarray` and `torch.Tensor` are the conventions in use; don't introduce
`npt.NDArray[np.float64]` piecemeal. Where a function genuinely accepts both, say
`np.ndarray | torch.Tensor` and mean it — `block_diag_skew` does, most functions should not.

## The numpy/torch boundary

This is the equivalent of a units bug in this codebase: it does not crash, it produces a wrong or
silently-detached number.

- **Be explicit about which side of the boundary a function lives on.** `model` and `navigation`
  are torch; `analysis` is numpy; `simulation` produces torch from numpy. A function that accepts
  either should be a deliberate exception with both paths tested, not an accident.
- Cross the boundary in one place, at the top of a computation, not inside a loop.
  `batch_inference` is the canonical crossing point: it detaches, moves to CPU and concatenates
  into a numpy array.
- **`.detach()` before leaving torch.** A tensor that still carries grad history keeps its graph
  alive and quietly changes memory behaviour in a long loop.
- **float32 is the working precision.** `block_diag_skew`, `random_orthogonal_matrix` and
  `starting_state` all cast to float32; a float64 array entering the chain gets silently
  downcast. Where numerical accuracy is the question rather than the model, promote deliberately
  to float64 and say why.
- Don't mix `torch.tensor(numpy_array)` and `torch.from_numpy` arbitrarily — the second shares
  memory with the array, which matters when the array is later mutated.

## Numerical work

- **Vectorise where it stays readable.** A one-liner that takes ten minutes to parse is worse than
  an explicit loop. Vectorise over positions and units; a Python loop over models or seeds is fine.
- **Never compare floats with `==`.** Use `np.isclose`/`np.allclose` or
  `torch.testing.assert_close` with an explicit tolerance, and pick the tolerance for a reason
  (see `model-verification`).
- **Be deliberate about NaN.** Ratemaps have empty bins. `np.nanmean` over `np.mean` is a choice —
  make it on purpose and say why when it is load-bearing.
- **Seed every stochastic operation.** `rng = np.random.default_rng(seed)`, threaded through as a
  parameter, and a `seed=` argument on anything that samples. Never the module-level
  `np.random.*` functions — `NPY002` is switched off in ruff so nothing will catch you.
  `random_orthogonal_matrix` is the pattern to copy; `SimpleDatasetMaker`,
  `generate_uniform_positions` and `orthogonal_basis` are not (see `KNOWN_ISSUES.md`).
- Handle genuinely degenerate input — an empty ratemap, a zero-variance autocorrelogram — by
  returning a documented sentinel (`np.nan`) rather than letting a divide-by-zero warning through.

## Naming

Match the paper. `Gx`, `Gy`, `Gs`, `R`, `Q`, `N`, `M`, `sigma_x`, `p0`, `lambda`/`lam`, `phi`, `k`
are the manuscript's symbols and ruff's `N803`/`N806`/`N812` are disabled precisely so they can
survive contact with PEP 8. Do not "fix" them to snake_case. Everything that is *not* a
mathematical symbol is ordinary `snake_case`. See `theory-context` for the full map.

## Errors and logging

Each module defines its logger once: `logger = logging.getLogger(__name__)`.

- `raise ValueError` for a caller mistake that cannot produce a meaningful result. Fail early; a
  crash is cheaper than a wrong figure.
- `logger.warning` when the code silently adjusts or drops something the caller did not ask for —
  a clamped radius, a returned NaN, a downcast. This is the log level that earns its place.
- `logger.info` sparingly, for progress through genuinely long batch operations. Not for narrating
  what a function just computed.
- Never `print` in library code. Never `except: pass`.
- `tqdm` is the convention for long loops (`batch_inference`, optimization loops). It is progress
  reporting, not logging, and does not need a log line beside it.

## Ruff

Configured in `pyproject.toml`: `line-length = 120`, `target-version = "py313"`, rule families
`E4`, `E7`, `E9`, `F`, `I`, `N`, `UP`, `C4`, ignoring `N803`, `N806`, `N812`, `NPY002`.

`lint.exclude = ["notebooks"]` is *meant* to exempt the notebooks and does not work (see
`KNOWN_ISSUES.md`), so a bare `uv run ruff check .` currently reports ~58 pre-existing notebook
errors and buries anything you introduced. **Run `uv run ruff check src/ tests/`** unless you are
deliberately working on a notebook. `src/` is clean; keep it that way.

- `I` — imports are sorted by ruff; don't hand-order them.
- `UP` — modern syntax: `X | None`, not `Optional[X]`.
- `C4` — comprehensions over `list(map(...))`.
- The `mccabe` config is present but `C90` is not selected, so complexity is not enforced. Don't
  flag it as a lint violation; judge it on readability.

Run `uv run ruff check .` and `uv run ruff format .` yourself before committing. Don't `# noqa`
without a trailing reason, and don't loosen the config to make code pass.

## Tests

The suite tests algebra, and algebra has right answers — so testing here is unusually strong.
Test what would be wrong silently:

- **The invariants**: commutativity, skew-symmetry, equinorm, path independence. These are the
  framework's guarantees; if they hold, most bugs are already excluded.
- **Against the closed form**: the empirical similarity against Eq. 4, a measured grid period
  against `2π/k`, plane wave orientations against `180°/M`.
- **Known-answer cases**: `exp(0) p0 = p0`, `C(0,0) = 1`, an ideal hexagonal pattern scoring well
  above the 0.37 grid-score threshold, an `M = 2` pattern scoring near zero.
- **Degenerate input**: odd `N`, a single eigenvalue, an all-NaN ratemap, a one-item bundle.

Don't test that a plotting call returns a figure, and don't chase coverage of glue code. Tests use
small models (`n_copies=5`) and fixed seeds so the suite stays fast; keep it that way.

## Reproducibility

- No machine-local absolute paths in library code; paths arrive as parameters.
- Output depends only on inputs — not the working directory, the wall clock, or cell execution
  order.
- When a choice is arbitrary but consequential — a bin count, an arena size, an annulus radius —
  make it a default argument and record the reason in a comment or docstring line.
- A change to a default in `grid_cell_model`, `place_cell_model` or `remapping_model` changes
  published figures. Flag it explicitly rather than folding it into a larger change.
