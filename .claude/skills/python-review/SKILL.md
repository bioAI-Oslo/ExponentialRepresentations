---
name: python-review
description: How to review Python code in priority order - run the tooling first, then hunt silent wrong results (units, axes, alignment, off-by-one, NaN, unseeded randomness, float equality), then strip defensive boilerplate, then structure and documentation, and how to report without padding. Use whenever asked to review, check, critique, or audit Python, whether a diff, a pull request, a file, a notebook being ported into a package, or code that was just written, and before committing a new analysis or module. Also use when the user asks "does this look right?" about Python code.
---

# Python review

The standards live in `python-style`. Read it, with `references/scientific.md` for numerical
code and `references/torch.md` for PyTorch code. This skill is about priority: what goes wrong,
in the order worth looking for it. Whether a result is scientifically sound is a separate
question from whether the code is. This skill covers the code.

## Pass 0: run the tooling

Before reading anything, run the project's checks, from its `CLAUDE.md` or by default:

```bash
uv run ruff check . && uv run ruff format . && uv run pyright && uv run pytest
```

Fix what they report. A review that starts with lint failures wastes a cycle and mixes
unreviewed automatic fixes into the change.

Then review in the passes below, and report as soon as pass 1 finds something. There is no point
polishing code that computes the wrong answer.

## Pass 1: silent wrong results

This is the only category that can put a wrong number in front of a reader without a crash.
Everything else is cosmetic by comparison.

- Units. Seconds against milliseconds, meters against pixels, radians against degrees, Hz against
  normalized frequency, bytes against megabytes. Check every boundary where a bare number meets a
  unit-carrying object, and every hand-rolled conversion.
- Axis and shape. `axis=0` against `axis=1`, a transpose, or broadcasting that succeeds by
  accident because two dimensions happen to be equal.
- Alignment. Streams at different sampling rates, an interval defined in one time base and
  applied in another, an offset applied twice or not at all, a merge or join that drops or
  duplicates rows.
- Off-by-one at boundaries. Inclusive against exclusive edges, the first and last bin, a window
  that runs past the end of the data, `range` against slice semantics.
- Missing values. Gaps propagating into a mean, or a `nan*` function hiding an input that is
  mostly missing.
- Unseeded randomness. Any `np.random.*`, `random.*` or framework RNG call without a threaded
  generator or seed.
- Float equality. `==` on floats, or a tolerance nobody stated.
- Mutation. A function that modifies an input the caller reuses, or a mutable default.
- Order dependence. A result that depends on dict or set iteration order, file listing order or
  the working directory.

For each finding, say what input produces the wrong answer and what the answer becomes.

## Pass 2: over-engineering

Check any new or revised code for defensive boilerplate and propose the deletion:

- `isinstance` and type checks in front of library calls that validate anyway.
- Assertions right before an operation that would fail loudly on its own.
- `try`/`except` that rewords an exception and re-raises, or catches `Exception` broadly.
- `logger.info` narrating entry, exit, or counts nobody reads.
- Defensive `.copy()`, `None` handling for arguments always passed, wrappers that only forward,
  classes that are only namespaces.
- Docstrings restating the signature, or documenting validation as if it were part of the
  contract.
- Abstractions with one implementation: a base class, a factory, a registry, a config object for
  two parameters.

The test: would the failure be loud and immediate anyway? If yes, the check earns nothing.

## Pass 3: structure

- A function doing two jobs: computing and plotting, computing and writing to disk. A flag such
  as `return_dataframe=True` is the sign.
- Tunable parameters as literals in the body rather than keyword arguments with defaults.
- Hidden state: module-level mutables, machine-local absolute paths, environment variables read
  during a computation, dependence on execution order.
- Logic that belongs in the package still sitting in a notebook or script.
- Concerns entangled in one function for no reason: IO and computation, two independent analyses
  sharing a loop.
- Legacy syntax the project's Python version has outgrown: `Optional[...]`, `os.path`,
  `%`-formatting outside logging, `typing.List`.

## Pass 4: documentation

- Missing units or array shapes in a docstring. This is the one omission always worth flagging.
- Comments that narrate rather than explain, comments that no longer match the code, and
  comments or docstrings that recount how a decision was reached or cite issue numbers and dated
  experiments.
- Commented-out code and unexplained TODOs.

## What not to flag

- Ruff findings listed by hand. Pass 0 fixed them. If ruff is clean, so is that category.
- Missing type hints on obvious internal one-liners.
- Style preferences with no correctness or clarity argument behind them.
- Absent tests for glue code, plotting or IO.
- Complexity as such. A long function that reads top to bottom is fine.
- Exploratory notebook code, unless it is being ported into the package.

## Reporting

Lead with the pass 1 findings and say concretely what breaks. For pass 2 findings, propose the
deletion rather than describing the problem. Separate what must change from what you would
prefer. Do not pad a review to look thorough. "Nothing in passes 1 to 3" is a valid and useful
result. If you fixed things during pass 0, list them in one line, not one bullet each.
