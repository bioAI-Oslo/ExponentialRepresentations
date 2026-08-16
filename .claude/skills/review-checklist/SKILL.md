---
name: review-checklist
description: What to look for when reviewing Python in expmap, in priority order - broken algebraic invariants, the numpy/torch and float32 boundary, sampling and aliasing, the defensive boilerplate typical of AI-assisted code, and what not to flag. Use when reviewing a diff, a pull request, a notebook being ported into the package, or code written with AI assistance, and before committing a change to the library.
---

# Review checklist

Standards live in `code-style` and `code-documentation`; whether a *result* is trustworthy is
`model-verification`; what the symbols mean is `theory-context`. This skill is about priority:
what actually goes wrong in this repository, in the order worth looking for it.

## Pass 0 — run the tooling

Before reading anything:

```bash
uv run ruff check src/ tests/
uv run ruff format src/ tests/
uv run pytest
```

Scope the ruff runs. A bare `ruff check .` reports ~58 pre-existing errors in `notebooks/` — the
`lint.exclude` setting meant to prevent that does not work (`KNOWN_ISSUES.md`) — and a real finding
in the diff will be lost in them. `src/` is clean today, so any error under `src/` came from the
change under review.

Fix what they report. There is no type checker configured, so nothing will catch a wrong
annotation for you — read the signatures.

Then review in the passes below, stopping to report as soon as you find something in pass 1. There
is no point polishing code that computes the wrong number.

## Pass 1 — broken algebra

The only category that can put a wrong figure in the paper. Everything else is cosmetic by
comparison. The framework's guarantees are cheap to check, so check them rather than reasoning
about whether they hold:

- **Commutativity.** `[Gx, Gy] = 0` is what makes path integration exact. It survives
  `G = Rᵀ Σ R` because both generators share `R` and both `Σ` are block diagonal on the same
  block structure. It does **not** survive independently permuted or differently ordered
  eigenvalue arrays, a per-generator `R`, or a `Σ` whose blocks are misaligned.
- **Skew-symmetry.** `Gᵀ = −G` is what makes the representation equinorm. It breaks if `R` is not
  orthogonal — check that `R` came from `random_orthogonal_matrix` or an equivalent, not from a
  normalised random matrix.
- **The canonical `p0`.** R-invariance of the similarity function holds only for
  `p0 = Rᵀ 1/√N`. Any diff that changes `p0` invalidates every claim about R that rests on it.
- **The direction of a transform.** `Rᵀ Σ R` and `R Σ Rᵀ` are both plausible-looking and only one
  matches the paper. Same for transposes in `compute_ratemaps` (`p.T` into
  `binned_statistic_2d`) and the `(n_units, bins, bins)` versus `(bins, bins, n_units)` ordering
  that moves between `analysis` and the notebooks.
- **Axis and shape.** `axis=0` versus `axis=1` on `(n_units, bins, bins)` is a silent wrong
  answer, not a crash. Same for broadcasting that succeeds by accident because `bins` and
  `n_units` happen to be equal in a test.

If any of these is in doubt, ask for the invariant check rather than reading harder.

## Pass 2 — the numpy/torch and precision boundary

The local equivalent of a units bug: it does not crash.

- **Which side is this function on?** `model` and `navigation` are torch, `analysis` is numpy. A
  new function that quietly accepts both, or returns whichever it was given, is a maintenance trap
  unless that is the explicit design.
- **`.detach()` and `.cpu()`** before a tensor leaves torch. A graph kept alive in a long loop is
  a real bug, not a style point.
- **float32 downcast.** `block_diag_skew`, `random_orthogonal_matrix` and `starting_state` all
  cast. A float64 input silently loses precision. If a diff claims a numerical result to more than
  ~6 significant figures, ask which precision it was computed in.
- **Tolerances.** `assert_close` with a tolerance that was tuned until the test passed is not a
  test. Ask what the two numbers are and why they should agree to that level.
- **`torch.tensor(arr)` versus `torch.from_numpy(arr)`** — the latter shares memory.

## Pass 3 — sampling, binning and figures

- **Aliasing.** A spectrum whose grid period is under a few ratemap bins produces a convincing
  wrong pattern. Check the bin count against the frequency before believing a ratemap.
- **Arena coverage.** A grid score computed on a map holding less than about two periods is
  meaningless; the autocorrelogram has no ring.
- **NaN propagation.** Empty occupancy bins become NaN. `nansum` and `nanmean` treat them as
  neutral, which is right sometimes and wrong sometimes — ask which.
- **Seeding.** Any figure-producing change must have a recorded seed. `generate_uniform_positions`
  and `SimpleDatasetMaker` use global RNG state, so a notebook that reproduces only when cells are
  run in order is not reproducible.
- **Changed defaults.** A modified default in `grid_cell_model`, `place_cell_model` or
  `remapping_model` silently changes published figures. It must be called out, not folded into a
  larger diff.

## Pass 4 — restraint

The characteristic failure mode of AI-assisted code in this repo. Flag and delete:

- `isinstance`/shape guards in front of an operation that would fail loudly anyway.
- `logger.info` narrating what a function just computed — "Generated 150 eigenvalues".
- `try`/`except` that only reworders an exception.
- A wrapper that forwards its arguments unchanged.
- Docstring bulk: restated types, `(default 32)`, three paragraphs of derivation where an
  `(Appendix E)` would do, underlined section headers on a six-line function.
- Commented-out code and unexplained TODOs.

Deleting these is part of review, not a follow-up ticket.

## Pass 5 — documentation and naming

- Does every public function cite the equation, section or appendix it implements, and is the
  citation the right one? H and I are routinely swapped.
- Are array shapes stated in `Parameters`/`Returns`?
- Do the symbols match the paper — `Gx`, `R`, `p0`, `M`, `phi`? Don't let a well-meaning rename to
  snake_case through; that is what `N803`/`N806` are disabled for.
- Is an ambiguous return described? An array of angles and an array of eigenvalues look identical.

## What not to flag

- **Mathematical names** — `R`, `Gx`, `Q`, `N`. They are deliberate and configured for.
- **Complexity.** `C90` is not selected. Judge a long function on readability, not a metric.
- **Missing type annotations on internal one-liners.** Public signatures only.
- **A notebook doing exploratory work.** Notebooks are allowed to be messy while a question is
  open. Flag it when the logic is load-bearing for a figure and still lives in a cell, or when a
  notebook has grown its own model class.
- **Test coverage of glue.** The suite is deliberately targeted at invariants and known answers.
- **Style ruff already fixed.** Don't re-review formatting.
