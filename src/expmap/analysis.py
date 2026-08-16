"""Analysis tools for exponential map spatial representations.

Provides vectorized routines for computing spatial ratemaps, running batch
inference, and evaluating representational similarity structure.
"""

import numpy as np
from scipy.ndimage import rotate
from scipy.stats import binned_statistic_2d
from tqdm import tqdm

# Conventional significance threshold for the grid score: the 95th percentile of a
# shuffled distribution, as used by Sargolini et al. (2006). It is a convention from the
# experimental literature rather than anything the framework derives, and the shuffle it
# came from is not recomputed here — treat "fraction above threshold" as a descriptive
# summary, not a test.
GRID_SCORE_THRESHOLD = 0.37


def compute_ratemaps(x, y, p, bins=32):
    """Compute spatial ratemaps by binning neural activity by position.

    For each neuron, computes the mean firing rate in each spatial bin
    using scipy's 2D binned statistic.

    Parameters
    ----------
    x : numpy.ndarray
        X-coordinates, shape (n_samples,).
    y : numpy.ndarray
        Y-coordinates, shape (n_samples,).
    p : numpy.ndarray
        Neural activity, shape (n_samples, n_units).
    bins : int, optional
        Number of spatial bins per dimension.

    Returns
    -------
    numpy.ndarray
        Ratemaps of shape (n_units, bins, bins).
    """
    ratemaps = binned_statistic_2d(x, y, p.T, bins=bins)[0]
    return ratemaps


def batch_inference(model, position_batches, p0, context=None, progress=True):
    """Run model inference over batches, collecting results on CPU.

    Processes position batches sequentially to manage GPU memory,
    detaching and moving results to CPU after each batch.

    Parameters
    ----------
    model : torch.nn.Module
        An ExponentialMap or RemappingModel instance.
    position_batches : list[torch.Tensor]
        List of position tensors, each of shape (batch_size, 2).
    p0 : torch.Tensor
        Initial state vector, shape (N,).
    context : torch.Tensor, float, or None, optional
        Context value for RemappingModel. If None, calls model with
        positions only (ExponentialMap).
    progress : bool, optional
        Show a progress bar. Turn it off inside a loop, where one bar per
        iteration is noise rather than information.

    Returns
    -------
    numpy.ndarray
        Concatenated population vectors, shape (total_samples, N).
    """
    results = []
    for batch in tqdm(position_batches, disable=not progress):
        if context is not None:
            output = model(batch, p0, context=context)
        else:
            output = model(batch, p0)
        results.append(output.detach().cpu())
    return np.concatenate(results)


def compute_similarity(ratemaps, ref_bin):
    """Compute population vector similarity relative to a reference spatial bin.

    Computes the dot product between the population vector at the reference
    bin and the population vectors at all other bins. For skew-symmetric
    generators this corresponds to cosine similarity (Eq. 3) since all
    population vectors have equal norm.

    Parameters
    ----------
    ratemaps : numpy.ndarray
        Ratemaps of shape (n_units, bins_x, bins_y).
    ref_bin : tuple[int, int]
        Reference bin indices (x_bin, y_bin).

    Returns
    -------
    numpy.ndarray
        Similarity map of shape (bins_x, bins_y). All-NaN if the reference
        bin was never visited, since there is no reference vector to compare
        against.
    """
    ref_vector = ratemaps[:, ref_bin[0], ref_bin[1]]

    # An unoccupied reference bin is all-NaN. nansum would read those NaNs as
    # zeros and return an identically-zero map, which is indistinguishable
    # from a genuinely orthogonal representation.
    if np.isnan(ref_vector).any():
        return np.full(ratemaps.shape[1:], np.nan)

    similarity = np.nansum(ref_vector[:, None, None] * ratemaps, axis=0)
    return similarity


def rectify(x):
    """Apply ReLU nonlinearity (element-wise rectification).

    Parameters
    ----------
    x : numpy.ndarray
        Input array.

    Returns
    -------
    numpy.ndarray
        Rectified array with negative values set to zero.
    """
    return np.maximum(x, 0)


def spatial_autocorrelogram(ratemap):
    """Compute the 2D spatial autocorrelation of a ratemap.

    Uses FFT-based correlation and normalizes by the number of
    overlapping bins at each lag, handling NaN values by treating
    them as missing data.

    Parameters
    ----------
    ratemap : numpy.ndarray
        A 2D ratemap of shape (bins_x, bins_y).

    Returns
    -------
    numpy.ndarray
        Autocorrelogram of shape (2*bins_x - 1, 2*bins_y - 1), normalized to
        exactly 1 at zero lag. Values are **not** bounded to [-1, 1]: the
        numerator is divided by the per-lag overlap count but scaled by the
        global variance, so large lags — where few bin pairs contribute — can
        exceed 1 in magnitude and the zero-lag peak is not the global maximum.
        Lags with no overlap are NaN.
    """
    rm = ratemap.copy()
    nan_mask = np.isnan(rm)
    rm[nan_mask] = 0.0
    valid = (~nan_mask).astype(float)

    # Padded size for full correlation
    ny, nx = rm.shape
    pad_shape = (2 * ny - 1, 2 * nx - 1)

    # Overlap count (how many valid bin-pairs contribute at each lag)
    overlap = np.real(np.fft.ifft2(np.abs(np.fft.fft2(valid, s=pad_shape)) ** 2))

    # Mean-subtracted field (only over valid bins)
    n_valid = valid.sum()
    if n_valid < 2:
        return np.full(pad_shape, np.nan)

    mean_val = rm.sum() / n_valid
    rm_centered = (rm - mean_val) * valid

    # Cross-correlation via FFT
    ft = np.fft.fft2(rm_centered, s=pad_shape)
    autocorr = np.real(np.fft.ifft2(ft * np.conj(ft)))

    # Variance normalization
    var = np.sum(rm_centered**2)
    if var < 1e-12:
        return np.full(pad_shape, np.nan)

    # Normalize: divide by overlap count and global variance
    with np.errstate(divide="ignore", invalid="ignore"):
        autocorr = autocorr / np.where(overlap > 0, overlap, 1) * (n_valid / var)

    autocorr[overlap < 1] = np.nan

    # Shift so zero-lag is at center
    autocorr = np.fft.fftshift(autocorr)

    return autocorr


def lag_distances(shape):
    """Distance of each autocorrelogram bin from the zero-lag centre.

    Parameters
    ----------
    shape : tuple[int, int]
        Shape of the autocorrelogram, (2*bins_x - 1, 2*bins_y - 1).

    Returns
    -------
    numpy.ndarray
        Distance in bins from the centre, same shape as the input.
    """
    ny, nx = shape
    Y, X = np.meshgrid(np.arange(ny) - ny // 2, np.arange(nx) - nx // 2, indexing="ij")
    return np.sqrt(X**2 + Y**2)


def annulus_radii(autocorr):
    """Auto-detect the annular mask enclosing the first ring of grid peaks.

    Takes the radial mean of the autocorrelogram, walks out from the centre to
    the first trough (the dip between the central peak and the six inner
    peaks), and then on to the first peak beyond it. The annulus runs from just
    inside the trough to 1.5x the peak radius.

    This is a heuristic, and the grid score is sensitive to it: on a clean
    hexagonal ratemap the score varies by more than 1.0 across plausible
    annuli, and a poor choice can flip its sign. Prefer passing explicit radii
    to `grid_score` for any figure or reported number, and record them.

    Parameters
    ----------
    autocorr : numpy.ndarray
        Autocorrelogram from `spatial_autocorrelogram`, shape
        (2*bins_x - 1, 2*bins_y - 1).

    Returns
    -------
    tuple[int, int]
        Detected (r_inner, r_outer), in bins.
    """
    dist = lag_distances(autocorr.shape)
    max_r = min(autocorr.shape) // 2

    # Radial profile: mean autocorrelation at each distance
    radial_profile = np.zeros(max_r)
    for i in range(max_r):
        vals = autocorr[(dist >= i) & (dist < i + 1)]
        valid_vals = vals[~np.isnan(vals)]
        radial_profile[i] = np.mean(valid_vals) if len(valid_vals) > 0 else 0

    # First trough after the central peak, skipping the peak itself
    min_skip = max(2, len(radial_profile) // 10)
    trough_idx = min_skip
    for i in range(min_skip, len(radial_profile) - 1):
        if radial_profile[i] <= radial_profile[i - 1] and radial_profile[i] <= radial_profile[i + 1]:
            trough_idx = i
            break

    # First peak after the trough — the ring of six inner peaks
    peak_idx = trough_idx
    for i in range(trough_idx, len(radial_profile) - 1):
        if radial_profile[i] >= radial_profile[i - 1] and radial_profile[i] >= radial_profile[i + 1]:
            peak_idx = i
            break

    if peak_idx == trough_idx:
        # No clear peak found — use heuristic
        peak_idx = max_r // 3

    r_inner = max(trough_idx - 1, 1)
    # Outer radius: 1.5x the peak distance, but not more than half the map
    r_outer = min(int(peak_idx * 1.5), max_r - 1)
    return r_inner, r_outer


def grid_score(ratemap, r_inner=None, r_outer=None, return_radii=False):
    """Compute the hexagonal grid score of a ratemap.

    Follows the standard method (Sargolini et al. 2006): compute the
    2D spatial autocorrelogram, extract an annular ring containing the
    six inner peaks, then compute the Pearson correlation between the
    ring and rotated copies at 30°, 60°, 90°, 120°, 150°.

    Grid score = min(corr at 60°, 120°) − max(corr at 30°, 90°, 150°)

    A positive grid score indicates hexagonal symmetry; `GRID_SCORE_THRESHOLD`
    holds the conventional significance cutoff.

    The score depends strongly on the annulus. Pass `r_inner` and `r_outer`
    explicitly for reported numbers, or use `return_radii` to record what the
    `annulus_radii` heuristic chose.

    Parameters
    ----------
    ratemap : numpy.ndarray
        A 2D ratemap of shape (bins_x, bins_y).
    r_inner : float, optional
        Inner radius of the annular mask, in autocorrelogram bins.
        Default: auto-detected by `annulus_radii`.
    r_outer : float, optional
        Outer radius of the annular mask, in bins. Default: auto-detected.
    return_radii : bool, optional
        If True, also return the annulus actually used.

    Returns
    -------
    float or tuple[float, tuple[int, int]]
        The grid score, NaN if the annulus is empty or degenerate. With
        `return_radii=True`, the pair (score, (r_inner, r_outer)).
    """
    autocorr = spatial_autocorrelogram(ratemap)
    dist = lag_distances(autocorr.shape)

    if r_inner is None or r_outer is None:
        detected_inner, detected_outer = annulus_radii(autocorr)
        r_inner = detected_inner if r_inner is None else r_inner
        r_outer = detected_outer if r_outer is None else r_outer

    radii = (r_inner, r_outer)
    if r_outer <= r_inner:
        return (np.nan, radii) if return_radii else np.nan

    # Extract annular ring
    annulus_mask = (dist >= r_inner) & (dist <= r_outer) & ~np.isnan(autocorr)
    if annulus_mask.sum() < 10:
        return (np.nan, radii) if return_radii else np.nan

    # Compute correlations at rotation angles
    rotations = [30, 60, 90, 120, 150]
    corrs = {}

    for angle in rotations:
        rotated = rotate(autocorr, angle, reshape=False, order=1, cval=np.nan)
        # Use only bins valid in both original and rotated
        both_valid = annulus_mask & ~np.isnan(rotated)
        if both_valid.sum() < 10:
            corrs[angle] = np.nan
            continue

        a = autocorr[both_valid]
        b = rotated[both_valid]

        # Pearson correlation
        a_centered = a - np.mean(a)
        b_centered = b - np.mean(b)
        denom = np.sqrt(np.sum(a_centered**2) * np.sum(b_centered**2))
        if denom < 1e-12:
            corrs[angle] = 0.0
        else:
            corrs[angle] = np.sum(a_centered * b_centered) / denom

    if any(np.isnan(v) for v in corrs.values()):
        return (np.nan, radii) if return_radii else np.nan

    # Grid score: min of 60°/120° correlations minus max of 30°/90°/150°
    score = min(corrs[60], corrs[120]) - max(corrs[30], corrs[90], corrs[150])

    return (score, radii) if return_radii else score


def compute_grid_scores(ratemaps, r_inner=None, r_outer=None, return_radii=False):
    """Compute grid scores for all neurons in a set of ratemaps.

    Parameters
    ----------
    ratemaps : numpy.ndarray
        Ratemaps of shape (n_units, bins_x, bins_y).
    r_inner, r_outer : float, optional
        Annulus applied to every unit. Default: auto-detected per unit, which
        means different units may be scored on different annuli — pass explicit
        radii when comparing scores across a population.
    return_radii : bool, optional
        If True, also return the annulus used for each unit.

    Returns
    -------
    numpy.ndarray or tuple[numpy.ndarray, numpy.ndarray]
        Grid scores of shape (n_units,). With `return_radii=True`, also the
        radii actually used, shape (n_units, 2).
    """
    n_units = ratemaps.shape[0]
    scores = np.zeros(n_units)
    radii = np.zeros((n_units, 2))
    for i in range(n_units):
        scores[i], radii[i] = grid_score(ratemaps[i], r_inner, r_outer, return_radii=True)
    return (scores, radii) if return_radii else scores
