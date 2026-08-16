---
name: code-documentation
description: How to write docstrings, inline comments and module docstrings in the expmap repository - NumPy style scaled to the function's audience, array shapes and paper references always stated, comments that explain why rather than narrate what. Use when writing or editing any docstring or comment, when documenting a new construction or analysis function, or when reviewing whether existing documentation is too thin or too bloated.
---

# Code documentation

Two audiences, two tools:

- **Docstrings** are for someone using the function to reproduce or extend a result. They answer
  *what does this compute, in what shape, and which part of the paper is it*.
- **Inline comments** are for someone reading the implementation. They answer *why is it done this
  way* — never *what does this line do*.

This code is supplementary to a paper. Documentation should be enough to reproduce and trust the
results and to trace them back to the derivation; it does not need to be an exhaustive technical
reference. Over-documentation is a real cost — it goes stale, buries the load-bearing sentences,
and makes short functions unreadable. When in doubt, write less.

## Always cite the paper

This is the rule specific to this repository, and it applies to every public function. A docstring
that says what a function computes but not which equation it implements makes the code
uncheckable, because the paper is the only statement of the right answer.

```python
"""Compute eigenvalue angles for metric-preserving representations.

Metric preservation requires the phasor sum to vanish, which holds when the
eigenvalue angles are evenly spaced as phi_j = pi * j / M (Appendix E).
"""
```

Cite the *narrowest* correct location — `Eq. 4`, `Sec. 2.4`, `Appendix J.2` — and get the letter
right; `theory-context` carries the full section and appendix map. Appendices H and I are the pair
most often confused.

Cite the derivation; don't reproduce it. If you find yourself writing three paragraphs of algebra
in a docstring, the algebra belongs in `tex/` or `reports/` and the docstring belongs as a pointer
to it.

## Docstrings: scale to the function

**Public functions** — anything called from a notebook, a figure script, or another module — get a
full NumPy-style docstring:

```python
def build_generators(sigma_x, sigma_y, R):
    """Construct full generator matrices from canonical forms and an orthogonal basis.

    Computes Gx = R.T @ sigma_x @ R and Gy = R.T @ sigma_y @ R. R mixes the
    underlying plane waves into the neural tuning curves; it strongly shapes
    single-neuron selectivity while leaving the population similarity structure
    unchanged (Sec. 4).

    Parameters
    ----------
    sigma_x, sigma_y : torch.Tensor
        Block-diagonal skew-symmetric eigenvalue matrices, shape (N, N).
    R : torch.Tensor
        Orthogonal basis matrix, shape (N, N). Must satisfy R @ R.T = I.

    Returns
    -------
    tuple[torch.Tensor, torch.Tensor]
        Generator matrices (Gx, Gy), each of shape (N, N).
    """
```

**Private helpers** get one to three lines, with `Parameters`/`Returns` only where the signature
does not already make it obvious. **Trivial helpers** get one line, or nothing — `rectify(x)` needs
no more than its summary.

### What to include

- **A one-sentence summary**, imperative or third person, ending in a period. No "This function..."
  preamble.
- **Array shapes, always.** `(batch_size, 2)`, `(N, N)`, `(n_units, bins, bins)`, `(N // 2,)`. This
  is the highest-value thing you can write: control flow is recoverable from the code, but the
  reader cannot infer that an array holds `N // 2` eigenvalue *pairs* rather than `N` eigenvalues.
- **What a quantity actually is** when the name is ambiguous.
  `roots_of_unity_eigenvalues` returns *angles*, `gaussian_eigenvalues` returns *eigenvalues* — say
  so, because the name does not.
- **A short context paragraph** where the scientific rationale is non-obvious: why a spectrum
  produces grids rather than place fields, why an annulus is auto-detected. One or two paragraphs
  at most, ending in a citation.
- **`Raises`** when the function raises deliberately.
- **`Notes`** for assumptions a user could get wrong: NaN handling, an even-`N` requirement, a
  float32 downcast, a heuristic that can return NaN.
- **`References`** when implementing a published method — `grid_score` implements Sargolini et al.
  (2006). One citation beats three paragraphs.

### What to leave out

- Restating the type already given in the signature and the `Parameters` header.
- `(default 32)` — the signature shows it. State *why* that default, if anything.
- Documenting defensive plumbing: input coercion, device moves and logging are implementation
  details, not part of the contract.
- Long derivations. Cite the appendix.
- Underlined subsection headers inside a docstring, **unless** the object genuinely has several
  distinct things worth naming separately. The `ExponentialMap` class earns one because it carries
  the framework's two central guarantees; a six-line construction function does not.
- `Examples` blocks in internal code. Reserve them for the genuine entry points: the package
  `__init__` and the three convenience constructors.

## Inline comments

Comment the **why**, the surprising, and the convention. Not the what.

Good — the reader cannot recover this from the code:

```python
# X-direction takes cos(phi), Y-direction sin(phi): the polar form of the
# wavevector, so a single ring has unit radius (k = 1)
sigma_x = block_diag_skew(np.cos(phi_tiled))
```

```python
# Overlap count: how many valid bin-pairs contribute at each lag, so partially
# overlapping lags are not biased toward zero
overlap = np.real(np.fft.ifft2(np.abs(np.fft.fft2(valid, s=pad_shape)) ** 2))
```

Bad — pure narration the code already carries:

```python
# Loop over units
for i in range(n_units):
```

Write a comment when:

- a constant is not self-evident (where `sqrt(2)` came from, why the grid-score threshold is 0.37),
- there is a coordinate, index or convention conversion — pixel to arena units, angle to
  wavevector, `fftshift` putting zero lag at the centre,
- the obvious implementation was rejected for a reason — numerical stability, a float32 limit,
  a library quirk,
- a step is subtle enough that you had to think about it: an off-by-one in binning, a deliberate
  `nansum`, the eigenvalue tiling that adds neurons but not scales.

Don't write a comment when the fix is a better name.

### No banner comments

A comment is a line of prose, never a piece of ASCII decoration. Don't wrap one in dashes, equals
signs or box rules, and don't use them to divide a file into sections:

```python
# --- figure parameters ---
M = 3
ARENA_SIZE = 10
# -------------------------

# ============================================================
# PART 2: SCALING
# ============================================================
```

Write the plain form:

```python
# Figure parameters; the 20 x 20 domain is Appendix A
M = 3
ARENA_SIZE = 10
```

The rules are noise, they drift out of alignment the moment a line is edited, and they push the
actual sentence off to the side. If a script needs visible sections, use the structure the format
already gives you — a `# %%` cell boundary in `notebooks/10`–`12`, a markdown cell in an `.ipynb`,
or a function.

## Module docstrings

Every module in `src/expmap/` has one, and should: they say where the module sits in the
mathematical structure. Two to five lines, naming the pipeline stage and the relevant paper
section:

```python
"""Generator matrix construction for the exponential map framework.

The construction pipeline is:
    eigenvalues -> block_diag_skew -> build_generators(sigma, R)

The eigenvalue structure determines the representation: roots of unity give
metric-preserving grids (Appendix E), Gaussian samples give localized
place-like fields (Appendix H).
"""
```

Notebooks and `# %%` scripts get the same treatment at the top: what the script demonstrates, and
which figure it produces.

## Conventions

- **Sentence case** in docstrings and comments, except acronyms and proper nouns: "Grid score",
  not "Grid Score".
- **No decorative dashes**: `# Fast vectorized band score`, not `# --- Fast Vectorized Band Score ---`.
- **ASCII in code, symbols in prose.** Docstrings use `phi`, `lambda`, `sigma`, `G^T = -G`,
  `[Gx, Gy] = 0`; save `φ`, `Σ`, `λ` for `reports/` and LaTeX.
- **No commented-out code.** Delete it; git remembers. A stray `# it = torch.randint(...)` is noise
  the next reader has to evaluate.
- **`# TODO:` and `# FIXME:`** must say what and why: `# TODO: annulus auto-detection fails on
  low-frequency maps, returns NaN`. An unexplained TODO gets deleted at review.
- Comments sit on their own line above the code they explain, except very short annotations.
- Keep comments truthful. An out-of-date comment is worse than none — when you change code, reread
  the comments around it.
