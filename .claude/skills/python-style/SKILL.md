---
name: python-style
description: Conventions for writing solid, reproducible, modern Python (3.11+) in any project - restraint over defensive boilerplate, small pure functions with explicit parameters, builtin generics and pyright-clean typing, uv/ruff/pytest tooling, logging instead of print, tests aimed at silent errors, and docstrings and comments scaled to their audience. Use this whenever writing, refactoring, or porting Python, adding a function, class, module, or script, writing a docstring or comment, or deciding whether validation, logging, or a test is warranted, even when the user only asks for "a quick script" or does not mention style. For NumPy, SciPy, pandas, xarray, or any numerical or data-analysis code, also read references/scientific.md. For PyTorch code, also read references/torch.md.
---

# Python style

The standard is reproducibility and clarity. A reviewer, a collaborator, or the author two years
from now must be able to run the code, understand it, and get the same result. Aim well above
typical research code, and get there through clarity rather than machinery.

Facts about one project, its layout, its tool commands, its shared types and how its tests fake
data, belong in that project's `CLAUDE.md` and not in this skill. Read it before writing code.
If it lacks the facts this skill expects, propose a block based on `references/project-config.md`.

## Restraint comes first

Defensive checks, redundant validation and narrating log calls bury the logic that matters and
earn nothing. Do not write them, and remove them when you touch the code around them.

Delete or do not write:

- `isinstance` checks on arguments that go straight into a library call that validates them.
  The library's error is better than yours.
- Assertions on shape, dtype or emptiness right before an operation that would fail on the next
  line anyway.
- `try`/`except` that catches an exception only to re-raise it with reworded text.
- `logger.info` on function entry and exit, or counts nobody will read.
- `None` handling for parameters that are always passed, and defensive `.copy()` of objects that
  nothing mutates.
- Wrapper functions that only forward arguments, and classes that only hold static methods.
- Docstrings that restate the signature, and comments that restate the code.

Validation earns its place when the failure would otherwise be silent or distant:

- A parameter that yields a plausible but wrong result rather than an error. A window longer
  than the signal, a threshold in the wrong unit, a rate above what the data can represent.
- Mismatched inputs where the bug surfaces several steps later, in a figure, a report or a file.
- Preconditions at the public entry point, checked once, rather than repeated in every helper.

The test is simple. Would the failure be loud and immediate anyway? If yes, let it fail. If it
would produce a number that looks fine, check it.

## Modern Python baseline

Target the version pinned by `requires-python` in `pyproject.toml`. Assume 3.11 or newer unless
the project says otherwise, and use what that buys.

- Builtin generics and union syntax: `list[int]`, `dict[str, float]`, `X | None`. Never
  `typing.List`, `Optional` or `Union`.
- `pathlib.Path` for every path. Never `os.path`, never string concatenation of paths.
- f-strings for formatting. Logging is the exception and takes lazy `%s` arguments.
- `@dataclass(frozen=True, slots=True)` or `NamedTuple` for records. `Enum` or `StrEnum` for a
  closed set of options instead of magic strings.
- `Protocol` for structural interfaces. Do not build abstract base classes for one
  implementation.
- `match` only when it reads better than `if`/`elif`, which is usually on enums or tagged unions.
- Context managers for anything that must be closed or restored.
- `functools.cache` over hand-rolled memoization. Comprehensions and `itertools` over manual
  accumulator loops, as long as they stay readable.
- Generators for streams consumed once. Lists when the whole collection is needed.
- No `import *`, no `global`, no mutable module-level state.

## Where code lives

The default is a src layout, and the project's `CLAUDE.md` names any deviation.

- `src/<package>/` is the library: importable, testable, deterministic. Every result anyone
  relies on is computed here.
- `tests/` runs without private data, network access or machine-local paths.
- `scripts/` and similar entry points hold argument parsing, IO and calls into the library. Each
  has a `main()` behind `if __name__ == "__main__":` and no logic worth testing.
- `notebooks/`, if present, is for exploration. Logic may be worked out there, but anything
  reusable, or anything a reported result depends on, moves into the package before it is relied
  on.

Configuration lives in `pyproject.toml`: dependencies, ruff, pyright, pytest. One file.

## Functions and data

Prefer small, pure functions. Reach for a class only when there is real state to carry across
calls. A class that is only a namespace should be a module.

- No hidden state. Everything a function needs arrives through its parameters. No reading of
  configuration or environment variables during a computation, no dependence on the working
  directory.
- Tunable parameters, thresholds, window lengths, sizes and tolerances, are keyword arguments
  with explicit defaults, never literals in the body. The default is part of the method and will
  be asked about in review.
- Keyword-only parameters, after `*`, for anything that would be ambiguous as a positional, which
  is most numeric parameters.
- One job per function. Computing and plotting, or computing and writing to disk, are two
  functions. A flag such as `return_dataframe=True` is the sign of a function doing two jobs.
- Private helpers are `_`-prefixed and live beside their caller.
- Return plain data: builtins, arrays, dataclasses or dicts with documented keys. Beyond three
  return values, use a `NamedTuple` or a dataclass.
- Never a mutable default argument. Use `None` and create the object inside, or a frozen type.

## Typing

Annotate every public signature. Leave obvious internal one-liners bare.

- Shared domain types live in one module, such as `types.py` or `type_aliases.py`, and are not
  respelled across files. Use `type X = ...` on 3.12 and later, or `TypeAlias` before that.
- Follow the array-typing convention already in use in the project, `np.ndarray` or
  `npt.NDArray[...]`. Do not mix them. Convert a whole module or leave it.
- Hints are enforced by `pyright` in standard mode. Fix a wrong hint rather than silencing the
  checker. When a third-party stub is at fault, use a targeted `# pyright: ignore[rule]` with a
  trailing reason, never a bare ignore.
- Typing-only imports go under `if TYPE_CHECKING:`.

## Errors and logging

Each module defines its logger once: `logger = logging.getLogger(__name__)`.

- `raise ValueError`, or a small project-specific exception, for a caller mistake that cannot
  produce a meaningful result. Fail early. A crash is cheaper than a wrong result.
- `logger.warning` when the code adjusts or drops something the caller did not ask for: a
  clipped parameter, a skipped record, an excluded item. Include the identifier that makes the
  message actionable. This is the log level that earns its place.
- `logger.info` sparingly, for progress through long operations.
- Never `print` in library code. Never `except: pass`. Never catch `Exception` unless the handler
  logs it and the program can continue without that unit of work.

## Tooling

The standard stack is `uv` for environments and running, `ruff` for lint and format, `pyright`
for types and `pytest` for tests, with pre-commit as a safeguard. Before handing over code, run:

```bash
uv run ruff check . && uv run ruff format . && uv run pyright && uv run pytest
```

Use the project's own commands if `CLAUDE.md` gives different ones. Pre-commit hooks that run
the same checks guard against forgetting. They are not the primary gate.

Do not loosen the configuration to make code pass. Do not add `# noqa` without a trailing
reason. Do not flag complexity as a lint violation unless the project enables a complexity rule.
Ruff findings in a change are fixed, not reported.

## Tests

Test what would be wrong silently, not what would crash anyway.

- Functions with a known answer: a hand-computed case, an analytic result, a round trip.
- Conversions and boundaries: units, coordinates, index conventions, inclusive against exclusive
  edges, wrapping.
- Degenerate input: empty, single element, zero length, all missing.
- A regression test for every bug fixed, reproducing the input that was wrong.

Do not test that a plotting call returns a figure, and do not chase coverage of glue code.

Tests must not need real data, the network or a machine-local path. Build small synthetic inputs
in the test. Fake external objects by stubbing only the attributes the code under test reads,
and keep the real contract of whatever is faked: if the real object is a `Mapping`, the fake
iterates over keys. Shared fakes go in `conftest.py` or a fixtures module once more than one
test file needs them. Use `pytest.mark.parametrize` over copied tests, and `pytest.approx` or
`np.testing` for floats.

## Documentation

Two audiences, two tools. Docstrings are for someone using the function: what it computes, in
what units and shapes, and what was decided when it was written. Inline comments are for someone
reading the implementation: why it is done this way, never what the line does.

- Public functions get a full NumPy-style docstring. Private helpers get one to three lines.
  Trivial helpers get one line or nothing.
- Units and array shapes are always stated. This is the most valuable sentence in any docstring.
- Comments and docstrings describe what the code is, with enough technical detail. They do not
  narrate how a decision was reached, and they do not cite issue numbers, dated experiments or
  known-issue lists. That history belongs in the issue tracker and the dev notes.
- No commented-out code. No decorative dashes in comments. Sentence case.

Read `references/documentation.md` before writing or reviewing docstrings.

## Reproducibility

- No machine-local absolute paths in library code. Paths arrive as parameters or from a project
  loader.
- Output depends only on inputs, not on the working directory, the wall clock, environment
  variables read during the computation, or execution order.
- Every stochastic operation takes a seed or a generator object as a parameter.
- When a choice is arbitrary but consequential, make it a default argument and record the reason
  in one docstring or comment line.
