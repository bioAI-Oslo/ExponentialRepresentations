---
name: model-verification
description: How to establish that an expmap implementation is actually correct before it becomes a result - checking algebraic invariants (commutativity, equinorm, path independence), comparing against the closed-form similarity function, numerical tolerances for matrix_exp in float32, sampling and aliasing in ratemaps, grid score sanity and controls, and seeding. Use when adding or changing anything in the package, interpreting a figure that looks surprising, choosing a tolerance, or deciding whether a result is trustworthy.
---

# Model verification

Most computational neuroscience has to argue from statistics because the ground truth is unknown.
Here it is known: the framework is closed-form, so **an implementation can be checked against its
own derivation**. Use that. A result verified only by "the ratemaps look like grid cells" is not
verified.

Apply this whenever code is heading for a figure, and whenever a result surprises you — **a
surprising result is a prior to check the code, not to write the discussion**.

## Start with the invariants

Four properties hold by construction for any correctly built canonical model. They are cheap,
they localise a bug immediately, and `tests/test_model.py` already checks the last three. Check
them on *your* generators, not just the constructors':

```python
# Commutativity — the path integration guarantee
torch.testing.assert_close(Gx @ Gy, Gy @ Gx, atol=1e-5, rtol=1e-5)

# Skew-symmetry — the equinorm guarantee
torch.testing.assert_close(Gx.T, -Gx)

# Equinorm — every representation has the norm of p0
p = model(positions, p0)
torch.testing.assert_close(p.norm(dim=1), p0.norm().expand(len(positions)))

# Path independence — via (1,0) then (0,1) must equal direct (1,1)
```

Which invariant breaks tells you where to look. Norm drift with commutativity intact points at
skew-symmetry — usually an `R` that is not orthogonal, or a `Σ` built asymmetrically. Path
dependence with skew-symmetry intact points at generators that no longer share an eigenbasis,
which is what happens when `Σx` and `Σy` are built from independently shuffled eigenvalue arrays.

`p0` matters too: `starting_state(R)` returns the canonical `Rᵀ1/√N`, and the R-invariance of the
similarity function holds **only** for that choice. If you swap `p0`, the coefficients `α0` stop
being uniform and results that were R-independent no longer are.

## Then check against the closed form

The population similarity has an exact expression (Eq. 4, and see `theory-context`). Compute it
both ways and compare:

```python
# Empirical: dot products of generated population vectors
sim_empirical = compute_similarity(ratemaps, ref_bin)

# Analytic: sum of cosines over the eigenvalue spectrum
sim_analytic = np.sum(alpha0**2 * np.cos(np.outer(dx, lam_x) + np.outer(dy, lam_y)), axis=-1)
```

Agreement to numerical precision is strong evidence the whole chain — eigenvalues, `Σ`, `R`, `p0`,
`matrix_exp`, binning — is right. Disagreement always finds a bug, and the shape of the
disagreement identifies it: a scale factor is a normalisation, a phase offset is a reference-bin
or coordinate-origin error, a decaying envelope is binning or occupancy, and structure that varies
with absolute position means translational invariance is broken.

Other quantities with a known answer:

- A single ring at `M = 3` gives plane waves at 60°, `M = 2` at 90°. Check the autocorrelogram
  peak angles, not just the visual impression.
- Grid period along a wave direction is `Λ = 2π/k`. Measure it off the autocorrelogram and compare
  with the eigenvalue radius you specified.
- `exp(0·Gx + 0·Gy) p0 = p0`, and `C(0, 0) = 1`.
- Gaussian eigenvalues at scale `σ` give `C(Δ) ≈ exp(−σ²Δ²)` — fit the empirical similarity and
  compare the width against `σ`.

## Numerical tolerances

The package is float32 throughout (`block_diag_skew`, `random_orthogonal_matrix` and
`starting_state` all cast). That fixes what tolerances are meaningful:

- `torch.matrix_exp` in float32 gives roughly 1e-6–1e-5 relative error near the origin. `1e-5` is
  a reasonable `atol` for invariants; `1e-4` for anything that composes two exponentials.
- **Error grows with `‖x·Gx + y·Gy‖`** — large arenas and high frequencies both hurt. If an
  invariant passes near the origin and fails at the arena edge, that is conditioning, not a logic
  bug. Confirm by recomputing the same check in float64.
- Never assert exact equality on floats, and never tighten a tolerance until a test passes without
  understanding which of the two numbers moved.

When a discrepancy is small but structured rather than noisy, it is a bug. Random-looking residual
at the 1e-6 level is float32.

## Sampling, binning and aliasing

Ratemaps come from `binned_statistic_2d` over sampled positions, which introduces its own
artefacts:

- **Spatial Nyquist.** A grid period smaller than two bins aliases into a spurious low-frequency
  pattern that looks like a real, coarser grid. With `bins=32` over `[-a, a]`, bin width is
  `2a/32`; require `Λ = 2π/k ≥ 4` bin widths for a pattern you intend to read off the map. Before
  concluding a spectrum "produces no grid", check that it is not simply aliased.
- **Occupancy.** Empty or near-empty bins become NaN and propagate. `compute_similarity` uses
  `nansum`, which silently treats a missing unit as contributing zero — sample enough positions
  that no bin is empty rather than relying on that.
- **Arena coverage.** Verify at least two to three full grid periods fit in the arena before
  computing a grid score; below that, the autocorrelogram has no ring to find.
- Changing `bins` changes the grid score. Where a number goes in a figure, state the bin count.

## Grid score

`analysis.grid_score` follows Sargolini et al. (2006): autocorrelogram, annular ring, Pearson
correlation against rotations, `min(60°, 120°) − max(30°, 90°, 150°)`. Threshold ≈ 0.37.

- The annulus radii are **auto-detected** from the radial profile when not given. This is a
  heuristic and it fails quietly on noisy or low-frequency maps — it returns `np.nan`, or worse
  picks a ring at the wrong radius. Count the NaNs in any batch of scores, and on a result that
  matters pass `r_inner`/`r_outer` explicitly so the choice is recorded.
- Sanity-check on synthetic input first: an ideal three-plane-wave hexagonal pattern should score
  well above 0.37, an `M = 2` square pattern near or below zero, and white noise near zero.
- For a population claim, report the distribution, not the mean. A mean grid score conflates "all
  cells weakly hexagonal" with "a few strong grid cells among mixed-selectivity units", and those
  are different claims.
- A control that fits this framework: compare against representations generated from the same
  spectrum with the phases or the eigenvalue angles randomised, so hexagonality is destroyed while
  the frequency content is kept.

## Seeding and reproducibility

Anything stochastic that reaches a figure must be seeded, threaded through as a parameter:

```python
rng = np.random.default_rng(seed)
R = random_orthogonal_matrix(N, seed=seed)
```

Two things to know about the current code:

- Every sampling function in the package takes a `seed` (`generate_uniform_positions`,
  `SimpleDatasetMaker`, `orthogonal_basis`, the eigenvalue samplers, the fits), and it defaults to
  `None`. Ruff's `NPY002` is switched off, so nothing flags a call that leaves it out. Pass the
  seed explicitly and say in the notebook which seed produced the figure.
- `place_cell_model` derives the y-eigenvalue seed as `seed + 1`. Adjacent seeds therefore share
  a spectrum across x and y — avoid running a sweep over consecutive seeds and treating the draws
  as independent.

An R-invariance check is the cheapest end-to-end reproducibility test available here: regenerate
the population similarity under several different `R` with the canonical `p0`. It must come out
identical. If it does not, either `p0` is not canonical or something in the chain is not
orthogonal.

## Fitted models

`fitting.py` is the one place where the right answer is not derivable, so the checks split in two.

- **By construction, so a check of the code only.** For `CommutingGeneratorFit` the commutator is
  zero, `‖p‖` is constant over any arena, and the similarity is a function of displacement alone.
  The closed-form transport must equal `matrix_exp` of the generators (`tests/test_fitting.py`),
  and maps regenerated from the saved `R` and wavevectors must equal the saved reconstruction.
  None of this says the fit is good.
- **Results, which need a reference.** Score the regenerated maps per unit against the data, and
  give the spread, not only the median. For extrapolation, fit the central part of the arena and
  score the part not trained on, next to chance (the same score with units shuffled) and a
  trivial control (a flat map at each unit's mean rate). A relative error near 100 % is the
  control's value for sparse maps, not a failure.
- **Population claims are tested on the data.** The model's similarity is translation invariant
  whatever it was fitted to. Whether the recorded population's is, and whether it matches the
  model's, is the measurement — compare against the data's own displacement-averaged similarity
  and against a surrogate with each unit's map shifted independently.
- Fits are one seed. Save the result before computing summary statistics, and state the seed,
  the number of iterations and the fitted fraction with every number.

## What counts as verified

Before a number or a figure is treated as a result:

1. The invariants hold for the generators actually used.
2. The quantity has been compared against its closed form, or against a slow, obviously-correct
   reimplementation.
3. The sampling is adequate — bins, arena size, position count — and stated.
4. The seed is recorded and the figure regenerates from the library.
5. Parameters that were chosen rather than derived are explicit defaults, and you can say why.

Write the check as a test in `tests/` when it is cheap and deterministic. Most of these are.
