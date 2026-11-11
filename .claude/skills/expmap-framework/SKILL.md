---
name: expmap-framework
description: The exponential map framework and how this repository implements it - the generators and the orthogonal mixing matrix R, what each eigenvalue spectrum produces, the similarity function, remapping, bundling and binding, the fits to recorded cells, the paper's section and appendix map, the numpy/torch and seed conventions, docstring rules, and how to verify an implementation against the closed form. Use when writing, reviewing or documenting any code in expmap, naming a variable, citing the paper, choosing a tolerance, interpreting a rate map or similarity plot, or deciding whether a result is trustworthy.
---

# The expmap framework

Every object in this package is a symbol in the paper, and the paper's derivations say what the
right answer is. Code that drifts from the paper's notation cannot be checked against them. The
general Python standard is `python-style` and `python-review`. This skill holds what those cannot
know about the framework and this repository. Plotting is in `figures`.

## The construction

A spatial representation assigns a population vector `p(x, y)` in `R^N`, the firing rates of N
neurons relative to a positive baseline, to each physical coordinate:

    p(x, y) = exp(x·Gx + y·Gy) p0                                    (Eq. 1)

`Gx` and `Gy` are the generator matrices and `p0` is the origin state. The matrix exponential
composes displacements, so the generators say how motion in physical space becomes a
transformation of neural state. Four algebraic conditions carry the framework.

| Condition | Guarantees | Where |
| --- | --- | --- |
| `[Gx, Gy] = 0` | Path integration that does not depend on the path taken | Sec. 2.2, App. B, J.2 |
| `Gᵀ = −G` | Orthogonal transport, constant norm `‖p‖ = ‖p0‖`, translation-invariant similarity | Sec. 2.3 |
| Eigenvalues form sets of roots of unity | Metric preservation and grids, hexagonal at `M = 3` | Sec. 2.4, App. E |
| Eigenvalues sampled from a target spectrum | Similarity preservation and place-like fields | App. H, I |

Skew-symmetric commuting generators share an eigenbasis, so one orthogonal matrix block
diagonalizes both:

    Gx = Rᵀ Σx R,    Gy = Rᵀ Σy R

`Σ` is block diagonal with 2×2 skew blocks `[[0, λ], [−λ, 0]]`. This is the pipeline in
`generators.py`: an eigenvalue specification goes through `block_diag_skew` and
`build_generators(sigma_x, sigma_y, R)`.

## Symbols and names

| Paper | Code | Meaning |
| --- | --- | --- |
| `N` | `N`, `len(R)` | Number of neurons. Even in every constructed model. A fit to an odd number of recorded cells leaves one direction that does not transport |
| `p(x, y)` | `p`, model output `(batch, N)` | Population vector |
| `p0` | `p0` | Origin state, unit norm |
| `Gx, Gy` | `Gx, Gy` | Spatial generators |
| `Gs` | `Gs` | Context generator of `RemappingModel` |
| `Σx, Σy, Σs` | `sigma_x, sigma_y, sigma_s` | Block-diagonal canonical forms |
| `R` | `R` | Orthogonal mixing matrix |
| `α0 = R p0 / ‖p0‖` | none | Coefficients of the subspaces in the similarity function |
| `λ_{i,x}, λ_{i,y}` | eigenvalue arrays | Imaginary parts of the generator eigenvalues, the wavevector of subspace i |
| `k_i, φ_i` | `k`, `phi` | Polar form, `λ_{i,x} = k_i cos φ_i` and `λ_{i,y} = k_i sin φ_i` |
| `M` | `M` | Symmetry order of a ring |
| `s` | `context` | Scalar context signal |
| `Q(Δx, Δy)` | `Q` in `forward` | Displacement operator `exp(ΔxGx + ΔyGy)` |

Identifiers match the paper: `Gx`, `Gy`, `Gs`, `R`, `Q`, `N`, `M`, `sigma_x`, `p0`, `lam`, `phi`,
`k`. Ruff's `N802`, `N803`, `N806` and `N812` are off for this. Do not rename them to snake_case
and do not flag them in review. Everything that is not a mathematical symbol is ordinary
`snake_case`.

Prose in docstrings and comments uses the paper's words: "path independent", "translation
invariant", "rate map" (functions keep `ratemap`), "origin state" for `p0` (the function keeps
`starting_state`), "orthogonal mixing matrix R", and "place-like fields" for model output.
Docstrings and comments are ASCII: `phi`, `sigma`, `G^T = -G`, `[Gx, Gy] = 0`. Unicode belongs in
notebook markdown and LaTeX.

Two functions return arrays that look alike and are not. `roots_of_unity_eigenvalues(M, n_copies)`
returns angles `phi`, which the caller pushes through `cos` and `sin` before `block_diag_skew`.
`gaussian_eigenvalues(N, scale)` returns eigenvalues `lam` directly.

## The similarity function

For commuting skew-symmetric generators the cosine similarity between two positions depends only
on their displacement:

    C(Δx, Δy) = Σ_i α0,i² cos(λ_{i,x} Δx + λ_{i,y} Δy)               (Eq. 4, App. F)

Three consequences follow, and each one can be tested.

- Similarity is translation invariant. It is a function of `(Δx, Δy)` and not of absolute
  position, which is what makes egocentric navigation possible.
- With the canonical `p0 = Rᵀ 1/√N` the coefficients are uniform, `α0,i² = 1/N`, and `C` does not
  depend on `R`. `R` changes every single-neuron tuning curve and nothing about the population
  similarity.
- Eq. 4 is a discrete inverse Fourier transform of a symmetric non-negative spectrum. Choosing the
  eigenvalues is choosing the similarity function. No optimization is involved.

`analysis.compute_similarity` computes the empirical version as a dot product against a reference
bin. It reproduces Eq. 4 to numerical precision. If it does not, something is wrong.

## What each spectrum produces

**Roots of unity give metric preservation and grids.** Metric preservation needs the phasor sum
`Σ_j ρ_j² e^{2iφ_j}` to vanish, which holds when the eigenvalue angles are evenly spaced on a
ring, `φ_i = φ_m + π i/M`. Each unit is then a superposition of plane waves (App. C) oriented
`180°/M` apart: 90° for `M = 2`, square grids, and 60° for `M = 3`, hexagonal grids. Hexagonal is
the optimum. It minimizes the higher-order distortion for large displacements.

**Gaussian eigenvalues give similarity preservation and place-like fields.** The eigenvalue density
is the Fourier transform of the target similarity. For a target `exp(−σ²Δ²)` App. I samples
eigenvalues from `N(0, 2σ²)`, and the units show heterogeneous, localized tuning. These are
place-like fields and not place cells. The paper says this is a proof of concept that does not
capture the sparsity, overdispersion or multi-field structure of real place cells (App. H).

The paper's `σ` is the width of the similarity. The `scale` of `gaussian_eigenvalues` is the
standard deviation of the eigenvalues. They differ by a factor of two:

    scale² = 2σ²        so        C(Δ) ≈ exp(−scale² Δ² / 2)

The default `scale = √2` gives `σ² = 1` and `C(Δ) ≈ exp(−Δ²)`, the `N(0, 2)` of App. H. This is
verified to the finite-N sampling noise, `1/√(N/2)`, in `tests/test_analysis.py` and in the
`place_like_fields` notebook.

**Rings, modules and copies.** A ring is a set of eigenvalues at one radius `k`, so one spatial
frequency and one grid scale. A module in the biological sense is a ring. `n_copies` in
`grid_cell_model` repeats the same `M` angles, which adds neurons at the same scale and not new
scales. Multi-scale representations need several radii, built by scaling the angle set by
different `k` before `block_diag_skew`. The convenience constructors do not do this. Module
spacings in the ratios of successive zeros of the Bessel function `J0` make the similarity
function close to radially symmetric, and the mean ratio lands inside the observed spread of
grid module spacing ratios (Sec. 2.4, App. G).

## Context and remapping

`p(x, y, s) = exp(x·Gx + y·Gy + s·Gs) p0`. `Gs` is built the same way as the spatial generators,
skew-symmetric and commuting with them, so similarity across contexts has the form of Eq. 4.
Increasing `Δs` decorrelates the representations and shifts the phase of the grids while leaving
their structure intact, which is what global remapping looks like (Sec. 2.5, Fig. 3). Context
eigenvalues are Gaussian. There is no metric to preserve on a non-spatial variable, only
similarity.

A context shift adds the phase `λ_{i,s} s` to the plane wave of subspace i, the same for every
unit. The spatial cross-correlogram of two units is therefore unchanged by the shift when
subspaces that share a wavevector also share a context eigenvalue, for any `R` (App. M).
`remapping_model` draws a context eigenvalue per subspace and does not have this property. The
Fig. 3 construction in the `remapping` notebook gives each hexagonal set its own orientation and
mixes `R` within a set, so it does.

## Bundling and binding

`navigation.bundle` averages the representations of target locations into one memory `pR(s)`.
Querying it with `CR(r, s) = p(r, s)ᵀ pR(s)` gives a landscape that is a sum of similarity
functions, and gradient ascent on it navigates to the rewards (App. L.1). This needs a
similarity-preserving representation. A pure grid code gives a periodic, ambiguous landscape.
Binding is remapping: a context shift makes maps orthogonal, so several reward maps coexist in
one population and the right `s` retrieves one of them (App. L.2). A bundled landscape depends on
the eigenvalues only, never on `R`.

## Attractor dynamics and oscillatory interference

The same generators read two ways (Sec. 3.1). At the population level, a gain-modulated attractor
network `τż = U(v)z + Mσ(z)` with a hypersphere attractor has on-manifold dynamics `τż = U(v)z`,
whose solution is the exponential map with `G = U/τ`. Skew-symmetry is the antisymmetric part of
the recurrent connectivity, and commutativity is what collapses the Magnus expansion so that
integration does not depend on the order of the velocities (App. J.1, J.2). The time constant sets
the grid scale, `λ_grid ∝ τ` (App. J.3), and the connectivity can grow from a local learning rule
on time-lagged antisymmetric correlations (App. J.4). At the single-cell level, block
diagonalization gives independent 2D rotations, phase-shifted sinusoids that integrate velocity
and reproduce the phase equations of oscillatory interference without temporal oscillators
(App. C, D).

## Fitting to recordings

Sec. 3.2 and App. K turn the construction around and recover the generators from recorded rate
maps, one model unit per recorded cell. `fitting.py` is the one module that learns.
`CommutingGeneratorFit` keeps the form `Gx = Rᵀ Σx R`, `Gy = Rᵀ Σy R` and learns the orthogonal
`R = exp(B − Bᵀ)` and the wavevector pairs. Skew-symmetry and commutativity hold by construction,
and the loss is the one-step prediction error between nearby bins. `GeneratorFit` is the earlier
version with free skew-symmetric generators and a commutator penalty. Maps are regenerated by
transporting the recorded population vector from the arena center, in closed form as the block
rotations of App. C. The fitted wavevectors are the population's spatial spectrum and `R` is its
mixing. The share of a pair in the similarity function is `α0,i²` summed over the pair, the same
at every position.

Keep apart what is by construction and what is a result. Constant norm, exact path independence
and translation-invariant similarity hold for any fit, good or bad, so reporting them checks the
code. The results are how well the regenerated maps match the data, how well a fit to the
central part of the arena predicts the rest, and whether the data's similarity structure is the
model's. Extrapolation continues the fitted waves beyond the arena: a grid lattice continues and
a place cell gains further fields. A field lying entirely outside the fitted region cannot be
recovered, since nothing in the fitted region says where it is.

## Section and appendix map

Cite the narrowest correct location in a docstring. Several appendices are close neighbors, and
H and I are the pair most often swapped: the eigenvalue-sampling derivation is in I, the
place-cell result in H.

| | |
| --- | --- |
| Sec. 2.1 | Constructing spatial representations (Eq. 1) |
| Sec. 2.2 | Path integration and path independence |
| Sec. 2.3 | Orthogonal transformations, translational invariance (Eqs. 2 to 4) |
| Sec. 2.4 | Metric preservation and grid scale ratios |
| Sec. 2.5 | Similarity preservation and context-dependent remapping |
| Sec. 3.1 | Relation to attractor dynamics and oscillatory interference |
| Sec. 3.2 | Fitting the exponential map to experimental recordings |
| Sec. 3.3 | Goal-oriented navigation via hyperdimensional bundling and binding |
| Sec. 3.4 | Accommodated phenomena and predictions |
| App. A | Methods |
| App. B | Commuting generators produce path-independent representations |
| App. C | From generators to plane waves |
| App. D | Connection to oscillatory interference |
| App. E | Metric preservation |
| App. F | Similarity function derivation |
| App. G | Designing spatial similarity functions |
| App. H | Similarity-preserving spatial representations, place-like fields, Fig. 7 |
| App. I | Similarity-preserving context representations, the sampling derivation |
| App. J | Attractor networks: J.1 CANN formulation, J.2 path integration and the Magnus expansion, J.3 membrane time constant and grid scale, J.4 local learning rule |
| App. K | Generative modeling of experimental data |
| App. L | Binding and bundling: L.1 reward maps, L.2 context-dependent retrieval |
| App. M | Pairwise phase relations under a context shift |

Figures: 1 framework overview, 2 roots of unity and grids, 3 remapping, 4 fits to recorded grid
and place cells, 5 multi-map navigation, 6 Bessel zeros, 7 place fields, 8 learning rule, 9 to
11 every fitted unit and the maps beyond the arena, 12 similarity structure of the data and the
fits.

The paper's scope is flat Euclidean environments, the idealized commuting case, and a scalar
context. `R` is a free parameter, sampled at random in the constructed models and learned in the
fits. Do not quietly exceed any of this.

## Code conventions

**The numpy/torch boundary.** Each module sits on one side. torch: `model`, `navigation`,
`fitting`. `generators` builds torch matrices from numpy eigenvalue arrays and `linalg` returns
torch matrices from numpy sampling. numpy: `analysis`, `attractor`, `plasticity`, `plotting` and
the notebook helpers in `notebooks/experimental_data/`. `simulation` samples in numpy and hands
out torch tensors. `analysis.batch_inference` is the one crossing from torch back to numpy. It
detaches, moves to CPU and concatenates. Do not add a second route, and do not write a function
that returns whichever array type it was given.

**float32 is the working precision.** `block_diag_skew`, `random_orthogonal_matrix`,
`block_diagonal_orthogonal` and `starting_state` cast to it, so a float64 array that enters the
construction is downcast without a message. When precision is the question rather than the
model, promote to float64 on purpose and say so in a comment.

**Seeds.** Every function that samples takes `seed: int | None = None` and builds its own
generator from it, `np.random.default_rng(seed)` in numpy code and
`torch.Generator().manual_seed(seed)` in torch code. This differs from `references/torch.md` of
`python-style`, which prefers a generator argument. Follow the convention here, and copy
`random_orthogonal_matrix`. A new sampling function without a `seed` parameter is a regression.
The default is `None`, so a call that omits the seed is unseeded and nothing flags it. Pass it
wherever a figure depends on the draw. `place_cell_model` derives the y seed as `seed + 1`, so
adjacent seeds share a spectrum across x and y. Do not treat a sweep over consecutive seeds as
independent draws.

**Docstrings.** Every public function cites the narrowest correct location in the paper: `Eq. 4`,
`Sec. 2.4`, `App. J.2`. Cite the derivation, do not reproduce it. Shapes are written in the
paper's terms, `(N, N)`, `(batch_size, 2)`, `(n_units, bins, bins)`, `(N // 2,)`, and a docstring
says what a quantity is when the name does not. Module docstrings name the pipeline stage and the
paper section. Math is typeset, raw docstrings with `.. math::` for displayed equations and
`:math:` inline, because the Jupyter Book is the documentation. Identifiers, code and literal
values use double backticks. A single-backtick span renders in italics. `Examples` blocks
appear only on the entry points, the package `__init__` and the three convenience constructors.

**Validation.** The package has no loggers and raises in one place, `gaussian_eigenvalues` on an
odd `N`. Library calls fail loudly on bad shapes, and that is enough for most inputs. A check
earns its place only when the failure would be a plausible wrong result. The cases here are a
grid period under two rate-map bins, which aliases into a convincing coarser grid, an odd `N`
where 2×2 blocks are assumed, and an `R` that is not orthogonal or a `Σ` built asymmetrically,
at the point it enters the construction. An analysis function given degenerate input, an all-NaN
rate map or a zero-variance autocorrelogram, returns a documented `np.nan` rather than a
divide-by-zero warning. Progress through a long loop is `tqdm` behind a `progress` flag.

**Defaults.** A changed default in `grid_cell_model`, `place_cell_model` or `remapping_model`
changes published figures. Call it out on its own and never fold it into a larger diff.

## Verification

The framework is closed-form, so an implementation can be checked against its own derivation. A
result verified only by "the rate maps look like grid cells" is not verified. Apply this to any
code heading for a figure, and to any result that surprises you. A surprising result is a reason
to check the code first.

**Start with the invariants.** They are cheap and they localize a bug. `tests/test_model.py`
checks the last three for the constructors. Check them on the generators you built.

```python
torch.testing.assert_close(Gx @ Gy, Gy @ Gx, atol=1e-5, rtol=1e-5)   # commutativity
torch.testing.assert_close(Gx.T, -Gx)                                 # skew-symmetry
p = model(positions, p0)
torch.testing.assert_close(p.norm(dim=1), p0.norm().expand(len(positions)))  # constant norm
# path independence: (1, 0) then (0, 1) equals (1, 1) directly
```

Which invariant breaks says where to look. Norm drift with commutativity intact points at
skew-symmetry, usually an `R` that is not orthogonal or a `Σ` built asymmetrically. Path
dependence with skew-symmetry intact points at generators that no longer share an eigenbasis,
which happens when `Σx` and `Σy` are built from independently shuffled eigenvalue arrays.
Commutativity survives `G = Rᵀ Σ R` only because both generators share `R` and both `Σ` have the
same block structure. `Rᵀ Σ R` and `R Σ Rᵀ` both look plausible. So do `p` and `p.T` into
`binned_statistic_2d`, and the `(n_units, bins, bins)` order that `compute_ratemaps` returns
against the `(bins, bins, n_units)` a notebook may hold. Broadcasting that works because `bins`
happens to equal `n_units` is a bug.

`p0` matters too. `starting_state(R)` returns the canonical `Rᵀ 1/√N`, and the R-invariance of
the similarity holds only for that choice. A diff that changes `p0` invalidates every claim about
`R` that rests on it. The cheapest end-to-end check is to regenerate the similarity under several
`R` with the canonical `p0`. It must come out identical.

**Then check against the closed form.** Compute the similarity both ways and compare.

```python
sim_empirical = compute_similarity(ratemaps, ref_bin)
sim_analytic = np.sum(alpha0**2 * np.cos(np.outer(dx, lam_x) + np.outer(dy, lam_y)), axis=-1)
```

Agreement to numerical precision says the whole chain is right: eigenvalues, `Σ`, `R`, `p0`,
`matrix_exp` and binning. The shape of a disagreement identifies the bug. A scale factor is a
normalization, a phase offset is a reference-bin or origin error, a decaying envelope is binning
or occupancy, and structure that varies with absolute position means translation invariance is
broken. Other quantities have a known answer too. Plane waves lie at 60° for `M = 3` and at 90°
for `M = 2`, read off the autocorrelogram peaks. The grid period along a wave direction is
`2π/k`. `exp(0) p0 = p0` and `C(0, 0) = 1`. The width of the Gaussian similarity matches `σ`.

**Tolerances.** `torch.matrix_exp` in float32 gives about 1e-6 to 1e-5 relative error near the
origin. Use `1e-5` as `atol` for invariants and `1e-4` for anything that composes two
exponentials. The error grows with `‖x·Gx + y·Gy‖`, so large arenas and high frequencies both
hurt. An invariant that passes near the origin and fails at the arena edge is conditioning, not
logic. Confirm by recomputing in float64. Never assert exact equality on floats, and never
tighten a tolerance until a test passes without knowing which of the two numbers moved. A small
but structured discrepancy is a bug. Random residual at the 1e-6 level is float32.

**Sampling, binning and aliasing.** Rate maps come from `binned_statistic_2d` over sampled
positions. A grid period under two bins aliases into a spurious coarser pattern, so with
`bins=32` over `[-a, a]` require `2π/k` of at least four bin widths for a pattern you intend to
read off the map. Before concluding that a spectrum produces no grid, check that it is not
aliased. Empty bins become NaN and propagate, and `compute_similarity` uses `nansum`, which
treats a missing unit as zero. Sample enough positions that no bin is empty. At least two or
three grid periods must fit in the arena before a grid score means anything, and the score
changes with the bin count, so state it wherever a number goes in a figure.

**Grid score.** `analysis.grid_score` follows Sargolini et al. (2006): autocorrelogram, annulus,
Pearson correlation against rotations, `min(60°, 120°) − max(30°, 90°, 150°)`, threshold about
0.37. The annulus radii are detected from the radial profile when not given. That fails quietly
on noisy or low-frequency maps, returning `np.nan` or picking the wrong ring, so count the NaNs in
any batch of scores and pass `r_inner` and `r_outer` on a result that matters. Check on
synthetic input first: an ideal hexagonal pattern scores well above 0.37, an `M = 2` pattern near
or below zero, white noise near zero. Report the distribution over a population, not the mean. A
control that fits the framework is the same spectrum with the phases or the angles randomized,
which destroys hexagonality and keeps the frequency content.

**Fitted models.** The checks split in two. By construction, and so a check of the code only:
for `CommutingGeneratorFit` the commutator is zero, the norm is constant over any arena, the
similarity is a function of displacement alone, the closed-form transport equals `matrix_exp` of
the generators (`tests/test_fitting.py`), and maps regenerated from the saved `R` and wavevectors
equal the saved reconstruction. None of this says the fit is good. The results need a reference.
Score the regenerated maps per unit against the data and give the spread, not only the median.
For extrapolation, fit the central part of the arena and score the part not trained on next to
chance, both with the bins of each recorded map shuffled and with each unit's prediction scored
against a random other unit. Population claims are tested on the data: the model's similarity is
translation invariant whatever it was fitted to, and the measurement is whether the recorded
population's similarity is, and whether it matches the model's, over all pairs of bins and from
each reference bin on its own. Fits are one seed. Save the result before computing summary
statistics, and state the seed, the number of iterations and the fitted fraction with every
number.

**What counts as verified.** The invariants hold for the generators used. The quantity has been
compared with its closed form or with a slow, obviously correct reimplementation. The sampling is
adequate and stated. The seed is recorded and the figure regenerates from the library. Parameters
that were chosen rather than derived are explicit defaults, and you can say why. Write the check
as a test in `tests/` when it is cheap and deterministic, which most of these are.

## Review and tests

`python-review` runs the tooling and then hunts silent wrong results. Here the silent wrong
results are algebraic, and the invariant checks above are cheaper than reasoning about them. If
any is in doubt, run the check. Do not flag the mathematical names, a notebook doing exploratory
work while a question is open, or missing tests for glue. Flag notebook code when a figure
depends on it and it still lives in a cell, or when a notebook has grown its own model class.

The test suite runs on pure algebra and synthetic input and never reads `data/`. Four kinds of
test earn their place.

- The four invariants.
- Agreement with the closed form: the empirical similarity against Eq. 4, a measured grid period
  against `2π/k`, plane-wave orientations against `180°/M`.
- Known answers: `exp(0) p0 = p0`, `C(0, 0) = 1`, an ideal hexagonal pattern scoring well above
  0.37 and an `M = 2` pattern near zero.
- Degenerate input: an odd `N`, a single eigenvalue, an all-NaN rate map, a one-item bundle.

Tests use small models and fixed seeds so the suite stays fast.
