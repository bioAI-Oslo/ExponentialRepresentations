# Project configuration

`python-style` is portable. The facts below are not. Each project states them in its `CLAUDE.md`
so that the skill can be applied without guessing. When a project is missing them, propose this
block filled in from what the repository shows, and ask about anything you cannot infer.

## CLAUDE.md block

```markdown
## Python conventions

Follow the `python-style` skill (and `python-review` for reviews). Project specifics:

- Python: 3.13 (`requires-python` in pyproject.toml)
- Layout: `src/<package>/` library, `tests/`, `scripts/` entry points, `notebooks/` exploration
- Run everything through uv: `uv run ruff check .`, `uv run ruff format .`, `uv run pyright`,
  `uv run pytest`
- Array typing: `np.ndarray` (do not introduce `npt.NDArray` piecemeal)
- Shared domain types: `src/<package>/type_aliases.py`
- Units: <bare floats with unit suffix in names | quantities | pint>; angles in radians on [0, 2π)
- Axis convention: arrays are (n_channels, n_samples). Positions are (n_samples, 2) in meters
- Test data: tests never read the real dataset. Fake <ExternalObject> by stubbing only
  `<attr>`, `<attr>`; it is a Mapping, so iteration yields keys. Shared fakes in `tests/conftest.py`.
- Gitignored scratch dirs (`dev/`, `scratch/`) are never imported from tests or the package.
- Exceptions to the skill: <none | list them, with reasons>
```

Delete the lines that do not apply and add the ones the skill would otherwise have to guess.

## Baseline pyproject.toml tool configuration

A starting point for a new project. Adjust the Python version and the rule list. Do not loosen
the configuration later to make code pass. Fix the code.

```toml
[project]
requires-python = ">=3.13"

[tool.ruff]
line-length = 120
target-version = "py313"
src = ["src", "tests", "scripts"]

[tool.ruff.lint]
select = [
    "E", "W",   # pycodestyle
    "F",        # pyflakes
    "I",        # isort
    "N",        # pep8-naming
    "UP",       # pyupgrade: modern syntax
    "B",        # bugbear: mutable defaults, bare except, and friends
    "C4",       # comprehensions
    "SIM",      # simplify
    "PTH",      # pathlib over os.path
    "TC",       # typing-only imports under TYPE_CHECKING
    "PT",       # pytest style
    "RET",      # return consistency
    "ARG",      # unused arguments
    "ERA",      # commented-out code
    "T20",      # print in library code
    "G", "LOG", # logging call style
    "PD",       # pandas idioms
    "NPY",      # numpy: legacy random API, deprecated aliases
    "PERF",     # obvious performance anti-patterns
    "RUF",      # ruff-specific
    "D",        # pydocstyle formatting (presence rules ignored below)
]
ignore = [
    "D1",       # missing-docstring rules: docstring presence is a judgment call, see python-style
    "N806",     # UPPER_CASE locals are fine for constants
    "E501",     # line length is handled by the formatter
]

[tool.ruff.lint.pydocstyle]
convention = "numpy"

[tool.ruff.lint.per-file-ignores]
"scripts/**" = ["T20"]      # entry points may print
"tests/**" = ["ARG", "D"]   # fixtures and test names need no docstrings

[tool.pyright]
typeCheckingMode = "standard"
pythonVersion = "3.13"
include = ["src", "tests", "scripts"]
venvPath = "."
venv = ".venv"

[tool.pytest.ini_options]
testpaths = ["tests"]
addopts = "-ra"
filterwarnings = ["error::RuntimeWarning"]  # divide-by-zero and invalid-value warnings fail tests
```

Pre-commit runs the same ruff and pyright commands. It guards against forgetting and is not the
primary gate. There is no complexity rule, `C90` or `PLR09xx`, in the baseline, and that is on
purpose.
