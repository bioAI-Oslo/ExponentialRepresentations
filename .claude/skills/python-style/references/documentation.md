# Docstrings and comments

Two audiences, two tools. Docstrings are for someone using the function to reproduce or extend
the work. They answer what it computes, in what units and shapes, and what was decided when it
was written. Inline comments are for someone reading the implementation. They answer why it is
done this way, never what a line does.

Documentation should be enough to reproduce and trust the code. It need not be exhaustive
reference material unless the project is a library with outside users. Too much documentation
has a real cost: it goes stale, it buries the sentences that matter, and it makes short
functions unreadable. When in doubt, write less.

## Docstrings scale with the function

Public functions, anything called from outside its module, from a script or from a notebook,
get a full NumPy-style docstring:

```python
def identify_bouts(
    positions: np.ndarray,
    timestamps_s: np.ndarray,
    *,
    zone_radius_m: float = 0.15,
    min_duration_s: float = 2.0,
) -> np.ndarray:
    """
    Identify sustained bouts inside a circular zone from a position trace.

    Per-sample inside/outside states are smoothed with a grace period so brief
    exits do not fragment one sustained bout into several.

    Parameters
    ----------
    positions : np.ndarray
        (n_samples, 2) positions in meters.
    timestamps_s : np.ndarray
        (n_samples,) sample times in seconds, monotonically increasing.
    zone_radius_m : float
        Radius of the zone in meters, measured from the arena center.
    min_duration_s : float
        Bouts shorter than this are discarded.

    Returns
    -------
    np.ndarray
        (n_bouts, 2) array of (start, stop) times in seconds. Empty (0, 2)
        array when no bout qualifies.
    """
```

Private helpers get one to three lines. Add `Parameters` and `Returns` only where the signature
does not make them obvious:

```python
def _angle_within_cone(direction, target, half_width_rad) -> np.ndarray:
    """
    Check whether each direction points within a cone around the target.

    Handles angle wrapping so the shortest path around the circle is used.
    """
```

Trivial helpers get one line or nothing. `round_up_to_even(n)` needs no docstring.

### What to include

- A one-sentence summary, imperative or third person, ending in a period. No "This function"
  preamble.
- Units and array shapes, always: `(n_samples, 2) positions in meters`, `rate in Hz`,
  `(n_channels, n_freqs) power spectral density`, `angle in radians on [0, 2π)`. A reader can
  infer control flow from the code and cannot infer units or conventions.
- A short context paragraph only where the reason is not obvious: why a grace period exists, why
  a particular normalization was chosen. One or two paragraphs at most.
- `Raises` when the function raises on purpose.
- `Notes` for assumptions a user could get wrong: behavior with missing values, assumptions about
  sampling rate or ordering, known limitations.
- `References` when implementing a published method. One citation beats three paragraphs.

### What to leave out

- The type already given in the signature and the `Parameters` header.
- `(default 0.15)`. The signature shows it. State why that default, if anything.
- Defensive plumbing. Input validation, type coercion and logging are implementation details,
  not part of the contract.
- `Examples` blocks in internal code. Reserve them for a handful of real entry points.
- Underlined subsection headers inside a docstring, unless the function has several distinct
  stages worth naming. Usually it does not.
- Long derivations. Cite the paper, or put it in `docs/`.
- Narration of how the implementation was arrived at. "This used to be a problem, here is the
  fix", references to issue numbers, dated experiments or a known-issues list do not belong in
  code. That history lives in the issue tracker, the commit messages and the dev notes.

## Inline comments

Comment the why, the surprising, and the unit or convention. Not the what.

Good, because the reader cannot recover it from the code:

```python
# Map head direction from [0, 2pi) to [-pi, pi) to match arctan2
hd_angle = np.where(head_direction > np.pi, head_direction - 2 * np.pi, head_direction)
```

```python
# Capped so the zone cannot extend past the arena wall
effective_radius = min(requested_radius, wall_distance)
```

Bad, because the code already says it:

```python
# Filter intervals by minimum duration
valid = [(s, e) for s, e in intervals if (e - s) >= min_duration_s]
```

```python
# Loop over channels
for channel in channels:
```

Write a comment when:

- a constant is not self-evident, where 0.15 m came from, why filter order 5,
- there is a unit, coordinate-system or index-convention conversion,
- the obvious implementation was rejected for a reason: performance, a library bug, numerical
  stability,
- a step is subtle enough that you had to think about it: edge inclusion, an off-by-one in
  binning, a deliberate `nanmean`.

Do not write a comment when the fix is a better name. `t_stop` plus a comment is worse than
`recording_stop_time_s` with none.

## Module docstrings

A module gets one when its scope is not obvious from its filename, or when it needs a note about
where its scope ends. Two or three lines:

```python
"""
Detection of interaction bouts from position tracking.

Operates on one session at a time; aggregation across sessions lives in
``project_loader``.
"""
```

`utils.py` and `types.py` need none. A module implementing a specific published method does.

## Conventions

- Sentence case in docstrings and comments, except acronyms and proper nouns: "Grid score",
  "State machine", not "Grid Score", "State Machine".
- No decorative dashes or banners: `# Fast vectorized band score`, not
  `# --- Fast Vectorized Band Score ---`.
- No commented-out code. Delete it. Git remembers. A stray `# ax.legend(...)` is noise the next
  reader has to evaluate.
- `# TODO:` and `# FIXME:` say what and why: `# TODO: assumes a fixed sampling rate, breaks for
  variable-rate recordings`. An unexplained TODO is deleted at review.
- Comments sit on their own line above the code they explain, except very short unit
  annotations.
- Keep comments truthful. An out-of-date comment is worse than none. When you change code, reread
  the comments around it.
