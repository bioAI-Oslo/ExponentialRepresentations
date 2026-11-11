"""Analysis tools for exponential map spatial representations.

Rate maps from sampled activity, batch inference, the population similarity
function, spatial autocorrelograms and the grid score.
"""

import numpy as np
import torch
from scipy.ndimage import rotate
from scipy.stats import binned_statistic_2d
from tqdm import tqdm

# The cutoff of Sargolini et al. (2006), the 95th percentile of their shuffled
# distribution. A convention from the experimental literature, not a derived quantity.
GRID_SCORE_THRESHOLD = 0.37


def compute_ratemaps(x: np.ndarray, y: np.ndarray, p: np.ndarray, bins: int = 32) -> np.ndarray:
    """Compute spatial rate maps by binning neural activity by position.

    For each neuron, computes the mean activity in each spatial bin. The bin
    edges span the sampled range of ``x`` and ``y``, so maps from different
    samples are comparable only when the samples cover the same arena. Bins
    with no samples are NaN.

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
        Rate maps of shape (n_units, bins, bins).
    """
    return binned_statistic_2d(x, y, p.T, bins=bins)[0]


def batch_inference(
    model: torch.nn.Module,
    position_batches: list[torch.Tensor],
    p0: torch.Tensor,
    context: torch.Tensor | float | None = None,
    progress: bool = True,
) -> np.ndarray:
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
        output = model(batch, p0, context=context) if context is not None else model(batch, p0)
        results.append(output.detach().cpu())
    return torch.cat(results).numpy()


def compute_similarity(ratemaps: np.ndarray, ref_bin: tuple[int, int]) -> np.ndarray:
    r"""Compute population vector similarity relative to a reference spatial bin.

    Computes the dot product between the population vector at the reference
    bin and the population vectors at all other bins. For skew-symmetric
    generators all population vectors have equal norm, so this is the cosine
    similarity of Eq. 3,

    .. math::

        C(\Delta x, \Delta y) = \frac{\mathbf{p}_0^T e^{\Delta x G_x + \Delta y G_y}\,\mathbf{p}_0}
                                      {\lVert\mathbf{p}_0\rVert^2},

    up to the factor :math:`\lVert\mathbf{p}_0\rVert^2`, which is 1 for the
    canonical origin state.

    Parameters
    ----------
    ratemaps : numpy.ndarray
        Rate maps of shape (n_units, bins_x, bins_y).
    ref_bin : tuple[int, int]
        Reference bin indices (x_bin, y_bin).

    Returns
    -------
    numpy.ndarray
        Similarity map of shape (bins_x, bins_y). NaN at every bin that was
        never visited, and all-NaN if the reference bin was, since there is
        no reference vector to compare against.
    """
    ref_vector = ratemaps[:, ref_bin[0], ref_bin[1]]
    # Not nansum: an unvisited bin must stay NaN rather than read as zero similarity.
    return (ref_vector[:, None, None] * ratemaps).sum(axis=0)


def rectify(x: np.ndarray) -> np.ndarray:
    """Set negative values to zero (element-wise rectification)."""
    return np.maximum(x, 0)


def spatial_autocorrelogram(ratemap: np.ndarray) -> np.ndarray:
    """Compute the 2D spatial autocorrelation of a rate map.

    Uses FFT-based correlation and normalizes by the number of
    overlapping bins at each lag, handling NaN values by treating
    them as missing data.

    Parameters
    ----------
    ratemap : numpy.ndarray
        A 2D rate map of shape (bins_x, bins_y).

    Returns
    -------
    numpy.ndarray
        Autocorrelogram of shape (2*bins_x - 1, 2*bins_y - 1), normalized to
        exactly 1 at zero lag. Values are **not** bounded to [-1, 1]: the
        numerator is divided by the per-lag overlap count but scaled by the
        global variance, so large lags, where few bin pairs contribute, can
        exceed 1 in magnitude and the zero-lag peak is not the global maximum.
        Lags with no overlap are NaN.
    """
    rm = ratemap.copy()
    nan_mask = np.isnan(rm)
    rm[nan_mask] = 0.0
    valid = (~nan_mask).astype(float)

    n_x, n_y = rm.shape
    pad_shape = (2 * n_x - 1, 2 * n_y - 1)

    # Rounded because FFT round-off can put a lag with one contributing pair below 1.
    overlap = np.round(np.real(np.fft.ifft2(np.abs(np.fft.fft2(valid, s=pad_shape)) ** 2)))

    n_valid = valid.sum()
    if n_valid < 2:
        return np.full(pad_shape, np.nan)

    mean_val = rm.sum() / n_valid
    rm_centered = (rm - mean_val) * valid

    ft = np.fft.fft2(rm_centered, s=pad_shape)
    autocorr = np.real(np.fft.ifft2(ft * np.conj(ft)))

    var = np.sum(rm_centered**2)
    if var < 1e-12:
        return np.full(pad_shape, np.nan)

    autocorr = autocorr / np.where(overlap > 0, overlap, 1) * (n_valid / var)

    autocorr[overlap < 1] = np.nan

    return np.fft.fftshift(autocorr)


def periodic_cross_correlograms(ratemaps: np.ndarray) -> np.ndarray:
    """Compute the normalized cross-correlograms of every pair of units on a periodic domain.

    The maps must be sampled on one period of the lattice, so that the circular
    correlation is the exact spatial cross-correlogram. This is the quantity of
    App. M: a context shift adds a phase per rotational subspace that is the
    same for every unit, so the cross-correlogram of two units is unchanged by
    remapping when subspaces that share a wavevector also share a context
    eigenvalue.

    Parameters
    ----------
    ratemaps : numpy.ndarray
        Maps of shape (n_units, bins_x, bins_y) on one period of the lattice,
        without NaNs.

    Returns
    -------
    numpy.ndarray
        Cross-correlograms of shape (n_units, n_units, bins_x, bins_y), with
        zero lag at the center as in ``spatial_autocorrelogram``. Entry
        ``[n, m]`` is the correlation coefficient between unit n and unit m
        displaced by the lag, so the diagonal is 1 at zero lag. Every entry
        involving a unit whose map has zero variance, such as a silent unit,
        is NaN. Memory grows with the square of the number of units.
    """
    centered = ratemaps - ratemaps.mean(axis=(1, 2), keepdims=True)
    ft = np.fft.fft2(centered, axes=(1, 2))
    cc = np.real(np.fft.ifft2(np.conj(ft)[:, None] * ft[None, :], axes=(2, 3)))
    sd = np.sqrt((centered**2).sum(axis=(1, 2)))
    with np.errstate(divide="ignore", invalid="ignore"):
        return np.fft.fftshift(cc / (sd[:, None] * sd[None, :])[:, :, None, None], axes=(2, 3))


def lag_distances(shape: tuple[int, int]) -> np.ndarray:
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
    n_x, n_y = shape
    X, Y = np.meshgrid(np.arange(n_x) - n_x // 2, np.arange(n_y) - n_y // 2, indexing="ij")
    return np.sqrt(X**2 + Y**2)


def annulus_radii(autocorr: np.ndarray) -> tuple[int, int]:
    """Auto-detect the annular mask enclosing the first ring of grid peaks.

    Takes the radial mean of the autocorrelogram, walks out from the centre to
    the first trough (the dip between the central peak and the six inner
    peaks), and then on to the first peak beyond it. The annulus runs from just
    inside the trough to 1.5x the peak radius.

    This is a heuristic, and the grid score is sensitive to it: on a clean
    hexagonal rate map the score varies by more than 1.0 across plausible
    annuli, and a poor choice can flip its sign. Prefer passing explicit radii
    to ``grid_score`` for any figure or reported number, and record them; when
    relying on the detection, record what it returned.

    Parameters
    ----------
    autocorr : numpy.ndarray
        Autocorrelogram from ``spatial_autocorrelogram``, shape
        (2*bins_x - 1, 2*bins_y - 1).

    Returns
    -------
    tuple[int, int]
        Detected (r_inner, r_outer), in bins.
    """
    dist = lag_distances(autocorr.shape)
    max_r = min(autocorr.shape) // 2

    radial_profile = np.zeros(max_r)
    for i in range(max_r):
        vals = autocorr[(dist >= i) & (dist < i + 1)]
        valid_vals = vals[~np.isnan(vals)]
        radial_profile[i] = np.mean(valid_vals) if len(valid_vals) > 0 else 0

    # Skip the central peak before looking for the trough.
    min_skip = max(2, len(radial_profile) // 10)
    trough_idx = min_skip
    for i in range(min_skip, len(radial_profile) - 1):
        if radial_profile[i] <= radial_profile[i - 1] and radial_profile[i] <= radial_profile[i + 1]:
            trough_idx = i
            break

    peak_idx = trough_idx
    for i in range(trough_idx, len(radial_profile) - 1):
        if radial_profile[i] >= radial_profile[i - 1] and radial_profile[i] >= radial_profile[i + 1]:
            peak_idx = i
            break

    if peak_idx == trough_idx:
        peak_idx = max_r // 3

    r_inner = max(trough_idx - 1, 1)
    r_outer = min(int(peak_idx * 1.5), max_r - 1)
    return r_inner, r_outer


def grid_score(ratemap: np.ndarray, r_inner: float | None = None, r_outer: float | None = None) -> float:
    """Compute the hexagonal grid score of a rate map.

    Follows the standard method (Sargolini et al. 2006): compute the
    2D spatial autocorrelogram, extract an annular ring containing the
    six inner peaks, then compute the Pearson correlation between the
    ring and rotated copies at 30, 60, 90, 120 and 150 degrees.

    Grid score = min(corr at 60, 120) - max(corr at 30, 90, 150)

    A positive grid score indicates hexagonal symmetry; ``GRID_SCORE_THRESHOLD``
    holds the conventional significance cutoff.

    The score depends strongly on the annulus. Pass ``r_inner`` and ``r_outer``
    explicitly for reported numbers; ``annulus_radii`` applied to
    ``spatial_autocorrelogram(rate map)`` gives the pair the detection would use.

    Parameters
    ----------
    ratemap : numpy.ndarray
        A 2D rate map of shape (bins_x, bins_y).
    r_inner : float, optional
        Inner radius of the annular mask, in autocorrelogram bins.
        Default: auto-detected by ``annulus_radii``.
    r_outer : float, optional
        Outer radius of the annular mask, in bins. Default: auto-detected.

    Returns
    -------
    float
        The grid score, NaN if the annulus is empty or degenerate.
    """
    autocorr = spatial_autocorrelogram(ratemap)
    dist = lag_distances(autocorr.shape)

    if r_inner is None or r_outer is None:
        detected_inner, detected_outer = annulus_radii(autocorr)
        r_inner = detected_inner if r_inner is None else r_inner
        r_outer = detected_outer if r_outer is None else r_outer

    if r_outer <= r_inner:
        return np.nan

    annulus_mask = (dist >= r_inner) & (dist <= r_outer) & ~np.isnan(autocorr)
    if annulus_mask.sum() < 10:
        return np.nan

    corrs = {}
    for angle in (30, 60, 90, 120, 150):
        rotated = rotate(autocorr, angle, reshape=False, order=1, cval=np.nan)
        both_valid = annulus_mask & ~np.isnan(rotated)
        if both_valid.sum() < 10:
            corrs[angle] = np.nan
            continue

        a = autocorr[both_valid]
        b = rotated[both_valid]
        a_centered = a - np.mean(a)
        b_centered = b - np.mean(b)
        denom = np.sqrt(np.sum(a_centered**2) * np.sum(b_centered**2))
        corrs[angle] = 0.0 if denom < 1e-12 else np.sum(a_centered * b_centered) / denom

    if any(np.isnan(v) for v in corrs.values()):
        return np.nan

    return min(corrs[60], corrs[120]) - max(corrs[30], corrs[90], corrs[150])


def compute_grid_scores(ratemaps: np.ndarray, r_inner: float | None = None, r_outer: float | None = None) -> np.ndarray:
    """Compute grid scores for all neurons in a set of rate maps.

    Parameters
    ----------
    ratemaps : numpy.ndarray
        Rate maps of shape (n_units, bins_x, bins_y).
    r_inner, r_outer : float, optional
        Annulus applied to every unit. Default: auto-detected per unit, which
        means different units may be scored on different annuli; pass explicit
        radii when comparing scores across a population.

    Returns
    -------
    numpy.ndarray
        Grid scores of shape (n_units,).
    """
    return np.array([grid_score(ratemap, r_inner, r_outer) for ratemap in ratemaps])
