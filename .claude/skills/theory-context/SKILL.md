---
name: theory-context
description: The algebra behind the exponential map framework and how it maps onto the code - generators, the orthogonal mixing matrix R, eigenvalue spectra and what each produces, the similarity function, remapping, hyperdimensional operations, and the paper's section and appendix map. Use when writing or reviewing any code in expmap, naming a variable, deciding what an eigenvalue or module means, citing the paper from a docstring, or interpreting a ratemap or similarity plot.
---

# Theory context

Every object in this package is a symbol in the paper. Code that drifts from the paper's notation
is code nobody can check against the derivation, and the derivation is the only thing that says
what the right answer is.

## The construction

A spatial representation assigns a population vector `p(x, y) ∈ R^N` — N neuron firing rates,
measured relative to a positive baseline — to each physical coordinate:

    p(x, y) = exp(x·Gx + y·Gy) p0                                    (Eq. 1)

`Gx, Gy ∈ R^{N×N}` are the generator matrices; `p0` is the origin state. The matrix exponential
composes displacements, so the generators say how motion in physical space becomes transformation
in neural state space. Four algebraic conditions carry the whole framework:

| Condition | Guarantees | Where |
| --- | --- | --- |
| `[Gx, Gy] = 0` (commuting) | Exact, trajectory-invariant path integration | Sec. 2.2, App. B, J.2 |
| `Gᵀ = −G` (skew-symmetric) | Orthogonal transport, equinorm `‖p‖ = ‖p0‖`, translationally invariant similarity | Sec. 2.3 |
| Eigenvalues form sets of roots of unity | Metric preservation; grids, hexagonal at `M = 3` | Sec. 2.4, App. E |
| Eigenvalues sampled from a target spectrum | Similarity preservation; localized place-like fields | App. H, I |

Skew-symmetry and commutativity together let the spectral theorem block-diagonalise both generators
simultaneously:

    Gx = Rᵀ Σx R,    Gy = Rᵀ Σy R

with `R` orthogonal and `Σ` block diagonal in 2×2 skew blocks `[[0, λ], [−λ, 0]]`. This is exactly
the pipeline in `generators.py`.

## Symbols

| Paper | Code | Meaning |
| --- | --- | --- |
| `N` | `N`, `len(R)` | Number of neurons; even in every constructed model (2×2 blocks). A fit to an odd number of recorded cells leaves one direction that does not transport |
| `p(x, y)` | `p`, model output `(batch, N)` | Population vector |
| `p0` | `p0` | Origin state, unit norm |
| `Gx, Gy` | `Gx, Gy` | Spatial generators |
| `Gs` | `Gs` | Context generator (`RemappingModel`) |
| `Σx, Σy, Σs` | `sigma_x, sigma_y, sigma_s` | Canonical block-diagonal forms |
| `R` | `R` | Orthogonal mixing matrix |
| `α0 = R p0 / ‖p0‖` | — | Per-subspace coefficients in the similarity function |
| `λ_{i,x}, λ_{i,y}` | eigenvalue arrays | Imaginary parts of the generator eigenvalues; the 2D wavevector |
| `k_i, φ_i` | `k`, `phi` | Polar form: `λ_{i,x} = k_i cos φ_i`, `λ_{i,y} = k_i sin φ_i` |
| `M` | `M` | Symmetry order of a ring |
| `s` | `context` | Scalar context signal |
| `Q(Δx, Δy)` | `Q` in `forward` | Displacement operator `exp(ΔxGx + ΔyGy)` |

`R`, `Gx`, `Q`, `N` are deliberately not snake_case; ruff's `N803`/`N806` are switched off for
precisely this. Keep matching the paper rather than PEP 8.

## The similarity function

The one equation most results are checked against. For commuting skew-symmetric generators the
cosine similarity depends only on displacement:

    C(Δx, Δy) = Σ_i α0,i² cos(λ_{i,x} Δx + λ_{i,y} Δy)               (Eq. 4, App. F)

Three things follow, and all three are testable:

- Similarity is **translationally invariant** — a function of `(Δx, Δy)` alone, not of absolute
  position. This is what makes egocentric navigation possible.
- With the canonical `p0 = Rᵀ 1/√N` the coefficients are uniform (`α0,i² = 1/N`) and **C does not
  depend on R at all**. R changes every single-neuron tuning curve and changes nothing about the
  population similarity structure.
- Eq. 4 is a discrete inverse Fourier transform of a symmetric non-negative spectrum. Choosing
  eigenvalues *is* choosing the similarity function — no optimization involved.

`analysis.compute_similarity` computes the empirical version as a dot product against a reference
bin. It should reproduce Eq. 4 to numerical precision; if it does not, something is wrong.

## What each spectrum produces

**Roots of unity → metric preservation → grids.** Metric preservation requires the phasor sum
`Σ_j ρ_j² e^{2iφ_j} = 0`, satisfied when eigenvalue angles are evenly spaced on a ring:
`φ_i = φ_m + π i/M`. The population vector entries are then superpositions of 2D plane waves
(App. C) oriented 180°/M apart — 90° for `M = 2` (square grids), 60° for `M = 3` (hexagonal).
Hexagonal is optimal: it minimises higher-order distortion for large displacements.

**Gaussian → similarity preservation → place-like fields.** The eigenvalue density is the Fourier
transform of the target similarity, so for a target `exp(−σ²Δ²)` App. I samples eigenvalues from
`N(0, 2σ²)`. Units then show heterogeneous, spatially localized tuning. These are *place-like*,
not place cells: the paper is explicit that this is a proof of concept that does not capture the
sparsity, overdispersion or multi-field structure of real place cells (App. H).

**Mind the factor of two.** The paper's `σ` is the width of the *similarity*; `gaussian_eigenvalues`'
`scale` is the standard deviation of the *eigenvalues*. They are not the same symbol:

    scale² = 2σ²        →        C(Δ) ≈ exp(−scale² Δ² / 2)

The default `scale = √2` therefore gives `σ² = 1` and `C(Δ) ≈ exp(−Δ²)`, which is the `N(0, 2)`
of App. H. Verified numerically to the finite-`N` sampling noise floor, `1/√(N/2)`, in
`tests/test_analysis.py` and `notebooks/02`.

**Watch the naming trap.** `roots_of_unity_eigenvalues(M, n_copies)` returns **angles** `φ`, which
the caller pushes through `cos`/`sin` before `block_diag_skew`. `gaussian_eigenvalues(N, scale)`
returns **eigenvalues** `λ` directly. Same-shaped arrays, different quantities.

## Rings, modules and copies

- A **ring** is a set of eigenvalues at one radius `k` — one spatial frequency, hence one grid
  scale. A **module** in the biological sense is a ring.
- `n_copies` in `grid_cell_model` tiles the *same* `M` angles repeatedly. It adds neurons at the
  same scale, not new scales. All copies sit on a single unit ring (`k = 1`).
- **Multi-scale representations need multiple radii**, built by scaling the angle set by different
  `k` before `block_diag_skew`. The convenience constructors do not do this.
- Relative module spacings proportional to zeros of the Bessel function `J0` make the similarity
  function approximate a Fourier–Bessel series and hence radially symmetric. The average ratio of
  successive low-order `J0` zeros lands inside the observed spread of grid module spacing ratios
  (Sec. 2.4, App. G).

## Context and remapping

`p(x, y, s) = exp(x·Gx + y·Gy + s·Gs) p0`. `Gs` is built the canonical way — skew-symmetric and
commuting with the spatial generators — so between-context similarity inherits the same form as
spatial similarity. Increasing `Δs` decorrelates representations while leaving the internal grid
structure intact and shifting its absolute phase, which is what biological global remapping looks
like (Sec. 2.5, Fig. 3). Context eigenvalues are Gaussian: there is no metric to preserve on a
non-spatial variable, only similarity.

## Hyperdimensional operations

- **Bundling** (`navigation.bundle`) averages representations of target locations into a composite
  memory `pR(s)`. Querying with `CR(r, s) = p(r, s)ᵀ pR(s)` yields a Gaussian-mixture similarity
  landscape; gradient ascent on it navigates to rewards (App. L.1). Bundling only produces a usable
  landscape for a **similarity-preserving** representation — a pure grid code gives a periodic,
  ambiguous one.
- **Binding** is remapping. A context shift orthogonalises maps, so several reward maps coexist in
  one population and are retrieved by querying with the right `s` (App. L.2).

## The CANN and OI bridge

The same generators read two ways (Sec. 3.1):

- **Population level.** A gain-modulated attractor network `τż = U(v)z + Mσ(z)` flowing to a
  hypersphere attractor has on-manifold dynamics `τż = U(v)z`, whose solution is the exponential
  map with `G = U/τ`. Skew-symmetry is the asymmetric part of recurrent connectivity;
  commutativity is what collapses the Magnus expansion so integration is order-independent
  (App. J.1, J.2). It also predicts `λ_grid ∝ τ`, a normative account of the dorsoventral grid
  scale gradient (App. J.3), and the connectivity can emerge from a local learning rule on
  time-lagged antisymmetric correlations (App. J.4).
- **Single-cell level.** Block-diagonalising gives independent 2D rotational subspaces —
  phase-shifted sinusoids integrating velocity, which reproduce the oscillatory-interference phase
  equations without temporal oscillators (App. C, D).

## Fitting to recordings

Sec. 3.2 and App. K turn the construction around: instead of choosing a spectrum, recover the
generators from recorded rate maps, one model unit per recorded cell. This is the one place
anything is learned (`fitting.py`).

- `CommutingGeneratorFit` keeps the form `Gx = Rᵀ Σx R`, `Gy = Rᵀ Σy R` and learns the orthogonal
  `R = exp(B − Bᵀ)` and the wavevector pairs `(λ_{i,x}, λ_{i,y})`. Skew-symmetry and
  commutativity therefore hold by construction, and the loss is the one-step prediction error
  between nearby pixels alone. `GeneratorFit` is the earlier version, with free skew-symmetric
  generators and a commutator penalty.
- Maps are regenerated by transporting the population vector from the arena centre, in closed form
  as the block rotations of App. C.
- The fitted wavevectors are the population's spatial spectrum and `R` its mixing. The weight of a
  pair in the similarity function is `α0,i²` summed over the pair (Eq. 4), and it is the same at
  every position.
- **What is by construction and what is a result.** Equinorm, exact path independence and
  translation-invariant similarity hold for any fit, good or bad; reporting them is a check of the
  code. The results are how well the regenerated maps match the data, how well a fit to the
  central part of the arena predicts the rest, and whether the *data's* similarity is the model's.
- Extrapolation continues the fitted waves beyond the arena: a grid lattice continues, and a place
  cell acquires further fields. A field lying entirely outside the fitted region cannot be
  recovered, since nothing in the fitted region says where it is.

## Section and appendix map

Cite these in docstrings; getting the letter right matters because several appendices are close
neighbours. Section 3 gained the fitting subsection in the camera-ready version, which moved the
navigation subsection from 3.2 to 3.3.

| | |
| --- | --- |
| Sec. 2.1 | Constructing spatial representations (Eq. 1) |
| Sec. 2.2 | Path integration and path independence |
| Sec. 2.3 | Orthogonal transformations, translational invariance (Eqs. 2–4) |
| Sec. 2.4 | Metric preservation and grid scale ratios |
| Sec. 2.5 | Similarity preservation and context-dependent remapping |
| Sec. 3.1 | Bridging attractor dynamics and oscillatory interference |
| Sec. 3.2 | Fitting the exponential map to experimental recordings |
| Sec. 3.3 | Goal-oriented navigation via hyperdimensional bundling |
| Sec. 3.4 | Accommodated phenomena and predictions |
| App. A | Methods |
| App. B | Commuting generators produce path-independent representations |
| App. C | From generators to plane waves |
| App. D | Mathematical connection to oscillatory interference |
| App. E | Metric preservation |
| App. F | Similarity function derivation |
| App. G | Designing spatial similarity functions |
| App. H | Similarity-preserving **spatial** representations — place-like fields, Fig. 7 |
| App. I | Similarity-preserving **context** representations — the sampling derivation |
| App. J | Attractor networks: J.1 CANN formulation, J.2 path integration and the Magnus expansion, J.3 membrane time constant and grid scale, J.4 local learning rule |
| App. K | Generative modeling of experimental data |
| App. L | Binding and bundling: L.1 reward maps, L.2 context-dependent retrieval |

H and I are the pair most often confused. The eigenvalue-sampling derivation is in I; the
place-cell result is in H.

Figures: 1 framework overview, 2 roots of unity and grids, 3 remapping, 4 attractor network,
5 multi-map navigation, 6 Bessel zeros, 7 place fields, 9 experimental extrapolation.

## Scope limits worth remembering

The paper states these; do not quietly exceed them.

- Flat Euclidean environments only. No obstacles, no curvature, no position-dependent generators.
- The idealized commuting case only. Real path-integration drift is proportional to how far from
  commuting the generators are, and is not modelled.
- Scalar context. Vector-valued `s·Gs` is future work.
- `R` is a free parameter, sampled at random in the constructed models and learned in the fits.
  Whether biology selects a particular R is open.
