# Numerical and data-analysis code

Read this when the code touches NumPy, SciPy, pandas, xarray, or any array, signal, image or
tabular data. Unit confusion, axis mix-ups and unseeded randomness are the bugs that put a wrong
number in a figure without an error. These rules make them impossible or loud.

## Arrays

- Vectorize where it stays readable. A NumPy one-liner that takes ten minutes to parse is worse
  than an explicit loop. Vectorize over samples. A Python loop over files, sessions or units is
  fine.
- State shapes. Every array parameter and return value has its shape in the docstring:
  `(n_samples, 2)`, `(n_channels, n_samples)`. Name the axis convention once per module and keep
  it. `axis=0` against `axis=1` on a 2-D array is a silent wrong answer, not a crash.
- Prefer explicit axes over broadcasting that happens to work because two dimensions are equal.
- Compare floats with `np.testing`, `np.isclose` or `np.allclose` and an explicit tolerance.
  Never `==`.
- Degenerate input, an empty array, a single sample or a zero-length window, returns a documented
  sentinel such as an empty `(0, 2)` array or `np.nan`, rather than letting a divide-by-zero
  warning through.
- Choose dtypes on purpose for large data, `float32` images, `int32` indices, and say so where it
  matters for precision.

## Units

Unit confusion is the most common bug in this domain.

- Where the project uses a units library, `pint`, `quantities` or `astropy.units`, keep values in
  it. Do not strip to bare floats to simplify.
- Where a value is a bare float, put the unit in the name, `min_duration_s`, `radius_m`,
  `rate_hz`, `pixel_size_um`, or in the docstring. Both when it is a public parameter.
- Convert at the boundary, once, in a named step. A hand-rolled conversion in the middle of a
  function is where seconds meet milliseconds.
- Angles: state the range, `[0, 2π)` or `[-π, π)`, as well as the unit.

## Missing data

- Be deliberate about NaN. Choose `np.nanmean` over `np.mean` on purpose, and say why when the
  choice matters.
- A `nan*` function can hide an input that is mostly missing. If a fraction of missing values
  above some threshold would invalidate the result, that check is worth writing.
- In pandas, prefer `.loc[mask]` and `.loc[:, cols]` over `df[mask]`. It is clearer and avoids
  stub and chained-assignment problems. Use `.to_numpy()` rather than `.values`.

## Randomness

- Every stochastic operation, a shuffle, a bootstrap, a permutation test, a random split, an
  augmentation, takes a `seed: int | None` or an `rng: np.random.Generator` parameter, threaded
  through from the entry point.
- Create generators with `np.random.default_rng(seed)`. Never call module-level `np.random.*`
  functions, and never set a global seed as a substitute for threading one through.
- The same applies to `random`, `torch`, `jax` and any library with its own RNG.

## Time series and signals

- Keep every stream's time base explicit: sampling rate or timestamps, and offset. An epoch
  defined in one time base and applied in another is a silent error.
- Window edges are inclusive or exclusive, stated in the docstring and tested at the boundary.
- Frequencies are in Hz. Filter parameters are checked once at the public entry point against
  Nyquist, since a band above Nyquist yields garbage and not an error.

## Tunable parameters

Thresholds, radii, window lengths, filter orders, bin sizes and smoothing widths are keyword
arguments with explicit defaults and units. When a default is arbitrary but consequential,
record where it came from in one docstring or comment line: "0.15 m, the box half-width plus the
body length".

## Tests for numerical code

- A synthetic input with a known answer: a sinusoid with a known spectral peak, a trajectory
  with a known occupancy, a constant field with a trivial mean.
- Unit and coordinate conversions, angle wrapping, and edge inclusion at window boundaries.
- Invariances. Shifting timestamps by an offset does not change durations. Permuting rows does
  not change a row-wise statistic.
- Degenerate input: empty, single sample, all NaN, zero-length window.
- Tests never read the real dataset. Build synthetic arrays in the test. For data containers,
  recordings, sessions or images with metadata, construct the objects directly with small
  synthetic contents rather than reading from disk, and fake only the attributes the code under
  test reads.
