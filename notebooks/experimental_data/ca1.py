"""CA1 place-cell recordings for the experimental fitting (Sec. 3.2, App. K).

The data are the open-field sessions published with Carpenter et al. (2024),
J. Neurosci. 44(38):e0040242024: five rats foraging in a 150 x 150 cm arena,
tetrode recordings from dorsal CA1, cluster-cut with MClust. The raw release
(`ca1_data/`, 18 GB) is reduced once by `extract_ca1.py` to a single
`.npz` of tracking and spike times; everything downstream works from that file.

Extracted format (`data/ca1_events.npz`)
-----------------------------------------
Sessions, indexed `s = 0 .. n_sessions - 1`:

    session_animal, session_date, session_system   str arrays
    session_duration                               recording length, s
    session_cm_per_px                              tracking scale used
    session_offsets                                (n_sessions + 1,) into pos_*
    pos_t                                          s from the first video frame
    pos_xy                                         head position, cm; NaN where lost

Units, indexed `u = 0 .. n_units - 1`, one per MClust cluster file:

    unit_session, unit_tetrode, unit_cluster
    unit_offsets                                   (n_units + 1,) into spike_t
    spike_t                                        s, on the session's clock
    unit_mean_rate                                 spikes / session_duration, Hz
    unit_duplicate_of                              index of the larger cluster this one is
                                                   nested in (> 50 % of spikes shared), else -1

Preprocessing follows the source paper: head position smoothed with a Gaussian of
sigma = 3 samples, running speed outside 6-60 cm/s discarded, mean rate between 1
and 10 Hz for a unit to count. The release holds 2729 cluster files but the paper
reports 1002 units, so some clusters are unfinished cuts; `unit_duplicate_of`
removes the nested ones and the rate criterion most of the rest. The scale of the
Neuralynx camera is not in its headers; `extract_ca1.py` sets it so that the
occupied extent matches the Axona animals (where 300 px/m is recorded), which only
affects the speed filter since rate maps are binned on the occupied extent.
"""

from dataclasses import dataclass

import numpy as np
from scipy.ndimage import gaussian_filter, gaussian_filter1d, label


@dataclass
class Session:
    animal: str
    date: str
    system: str
    duration: float
    cm_per_px: float
    t: np.ndarray
    xy: np.ndarray


@dataclass
class Unit:
    session: int
    tetrode: int
    cluster: int
    spike_t: np.ndarray
    mean_rate: float
    duplicate_of: int


def save_events(path, sessions: list[Session], units: list[Unit]) -> None:
    """Write the extracted format described in the module docstring."""
    session_offsets = np.cumsum([0] + [len(s.t) for s in sessions])
    unit_offsets = np.cumsum([0] + [len(u.spike_t) for u in units])
    np.savez_compressed(
        path,
        session_animal=np.array([s.animal for s in sessions]),
        session_date=np.array([s.date for s in sessions]),
        session_system=np.array([s.system for s in sessions]),
        session_duration=np.array([s.duration for s in sessions]),
        session_cm_per_px=np.array([s.cm_per_px for s in sessions]),
        session_offsets=session_offsets,
        pos_t=np.concatenate([s.t for s in sessions]).astype(np.float32),
        pos_xy=np.concatenate([s.xy for s in sessions]).astype(np.float32),
        unit_session=np.array([u.session for u in units]),
        unit_tetrode=np.array([u.tetrode for u in units]),
        unit_cluster=np.array([u.cluster for u in units]),
        unit_offsets=unit_offsets,
        spike_t=np.concatenate([u.spike_t for u in units]),
        unit_mean_rate=np.array([u.mean_rate for u in units]),
        unit_duplicate_of=np.array([u.duplicate_of for u in units]),
    )


def load_events(path) -> tuple[list[Session], list[Unit]]:
    """Read the extracted format back into sessions and units.

    Each array is decompressed once and sliced into views; indexing the archive
    inside the loops would decompress it afresh on every access.
    """
    with np.load(path) as archive:
        d = {key: archive[key] for key in archive.files}
    so, uo = d["session_offsets"], d["unit_offsets"]
    pos_t, pos_xy, spike_t = d["pos_t"].astype(np.float64), d["pos_xy"].astype(np.float64), d["spike_t"]
    sessions = [
        Session(
            animal=str(d["session_animal"][s]),
            date=str(d["session_date"][s]),
            system=str(d["session_system"][s]),
            duration=float(d["session_duration"][s]),
            cm_per_px=float(d["session_cm_per_px"][s]),
            t=pos_t[so[s] : so[s + 1]],
            xy=pos_xy[so[s] : so[s + 1]],
        )
        for s in range(len(so) - 1)
    ]
    units = [
        Unit(
            session=int(d["unit_session"][u]),
            tetrode=int(d["unit_tetrode"][u]),
            cluster=int(d["unit_cluster"][u]),
            spike_t=spike_t[uo[u] : uo[u + 1]],
            mean_rate=float(d["unit_mean_rate"][u]),
            duplicate_of=int(d["unit_duplicate_of"][u]),
        )
        for u in range(len(uo) - 1)
    ]
    return sessions, units


def select_units(units: list[Unit], rate_range: tuple[float, float] = (1.0, 10.0)) -> np.ndarray:
    """Indices of units passing the source paper's mean-rate criterion, duplicates removed."""
    lo, hi = rate_range
    return np.array([i for i, u in enumerate(units) if lo < u.mean_rate < hi and u.duplicate_of < 0], dtype=int)


def preprocess_position(
    session: Session, sigma: float = 3.0, speed_range: tuple[float, float] = (6.0, 60.0)
) -> tuple[np.ndarray, np.ndarray]:
    """Interpolate lost frames, smooth, and flag frames by running speed.

    Returns
    -------
    xy : np.ndarray
        Smoothed head position, cm, shape (n_frames, 2).
    valid : np.ndarray
        Boolean mask of frames whose speed lies inside `speed_range`.
    """
    xy = session.xy.copy()
    for c in range(2):
        lost = np.isnan(xy[:, c])
        xy[lost, c] = np.interp(session.t[lost], session.t[~lost], xy[~lost, c])
    xy = gaussian_filter1d(xy, sigma, axis=0)
    dt = np.median(np.diff(session.t))
    speed = np.linalg.norm(np.gradient(xy, dt, axis=0), axis=1)
    valid = (speed >= speed_range[0]) & (speed <= speed_range[1])
    return xy, valid


def arena_box(xy: np.ndarray, valid: np.ndarray, quantile: float = 0.005) -> tuple[np.ndarray, float]:
    """Square box covering the occupied extent: (centre, side), cm.

    The quantile trims tracking outliers; the side is the larger of the two
    axes' extents so the box stays square for a square rate map grid.
    """
    lo = np.quantile(xy[valid], quantile, axis=0)
    hi = np.quantile(xy[valid], 1 - quantile, axis=0)
    return (lo + hi) / 2, float((hi - lo).max())


def ratemap(
    spike_t: np.ndarray,
    t: np.ndarray,
    xy: np.ndarray,
    valid: np.ndarray,
    box: tuple[np.ndarray, float],
    n_bins: int = 32,
    sigma: float = 1.0,
) -> tuple[np.ndarray, np.ndarray]:
    """Occupancy-normalized rate map on the box, indexed (bins_x, bins_y).

    Spike counts and occupancy are smoothed separately before dividing, so
    sparsely visited bins borrow from their neighbors rather than exploding.

    Returns
    -------
    rate : np.ndarray
        Firing rate, Hz, shape (n_bins, n_bins); NaN where smoothed occupancy is zero.
    occupancy : np.ndarray
        Unsmoothed time spent per bin, s, same shape.
    """
    centre, side = box
    edges = [centre[i] - side / 2 + side * np.arange(n_bins + 1) / n_bins for i in range(2)]
    dt = np.median(np.diff(t))
    occupancy = np.histogram2d(xy[valid, 0], xy[valid, 1], bins=edges)[0] * dt

    frame = np.clip(np.searchsorted(t, spike_t), 0, len(t) - 1)
    frame = frame[valid[frame]]
    counts = np.histogram2d(xy[frame, 0], xy[frame, 1], bins=edges)[0]

    smoothed_occupancy = gaussian_filter(occupancy, sigma)
    with np.errstate(invalid="ignore", divide="ignore"):
        rate = gaussian_filter(counts, sigma) / smoothed_occupancy
    rate[smoothed_occupancy == 0] = np.nan
    return rate, occupancy


def spatial_information(rate: np.ndarray, occupancy: np.ndarray) -> float:
    """Skaggs spatial information, bits per spike, over visited bins."""
    visited = (occupancy > 0) & np.isfinite(rate)
    p = occupancy[visited] / occupancy[visited].sum()
    r = rate[visited]
    mean_rate = (p * r).sum()
    if mean_rate == 0:
        return 0.0
    nonzero = r > 0
    return float((p[nonzero] * r[nonzero] / mean_rate * np.log2(r[nonzero] / mean_rate)).sum())


def population_ratemaps(
    sessions: list[Session],
    units: list[Unit],
    unit_indices: np.ndarray,
    n_bins: int = 32,
    sigma: float = 1.0,
    position_sigma: float = 3.0,
    speed_range: tuple[float, float] = (6.0, 60.0),
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Rate maps for a set of units, each on its own session's occupied box.

    Returns
    -------
    rates : np.ndarray
        Shape (n_selected, n_bins, n_bins), indexed (unit, bins_x, bins_y) like
        `analysis.compute_ratemaps` and `data/gc_ratmaps.npy`.
    occupancy : np.ndarray
        The session occupancy behind each map, same shape.
    information : np.ndarray
        Spatial information per unit, bits per spike.
    """
    cache = {}
    rates, occupancies, information = [], [], []
    for i in unit_indices:
        u = units[i]
        if u.session not in cache:
            xy, valid = preprocess_position(sessions[u.session], position_sigma, speed_range)
            cache[u.session] = (xy, valid, arena_box(xy, valid))
        xy, valid, box = cache[u.session]
        rate, occupancy = ratemap(u.spike_t, sessions[u.session].t, xy, valid, box, n_bins, sigma)
        rates.append(rate)
        occupancies.append(occupancy)
        information.append(spatial_information(rate, occupancy))
    return np.stack(rates), np.stack(occupancies), np.array(information)


def place_fields(rate: np.ndarray, threshold: float = 0.2, min_bins: int = 9) -> np.ndarray:
    """Label contiguous regions above `threshold` x peak with at least `min_bins` bins.

    Returns an integer map, 0 outside fields and 1..n_fields inside, in
    descending order of peak rate.
    """
    filled = np.nan_to_num(rate)
    labels = np.zeros(rate.shape, dtype=np.int32)
    # scipy's stub types the return as a union of the two call forms; with `output` it is the count.
    n: int = label(filled >= threshold * filled.max(), output=labels)  # pyright: ignore[reportAssignmentType]
    peaks = [(filled[labels == k].max(), k) for k in range(1, n + 1) if (labels == k).sum() >= min_bins]
    fields = np.zeros_like(labels)
    for new, (_, k) in enumerate(sorted(peaks, reverse=True), start=1):
        fields[labels == k] = new
    return fields


def border_score(rate: np.ndarray, fields: np.ndarray) -> float:
    """Solstad et al. (2008) border score, b = (cM - dM) / (cM + dM), over all fields.

    cM is the largest fraction of any wall covered by a field; dM is the
    rate-weighted mean distance of field bins from the nearest wall, in units
    of half the arena side. b is 1 for a field hugging a whole wall and -1
    for one in the centre; NaN if there is no field.
    """
    if fields.max() == 0:
        return float("nan")
    n = rate.shape[0]
    inside = fields > 0
    coverage = max(inside[0].mean(), inside[-1].mean(), inside[:, 0].mean(), inside[:, -1].mean())
    index = np.indices(rate.shape)
    distance = np.minimum.reduce([index[0], n - 1 - index[0], index[1], n - 1 - index[1]]) / (n / 2)
    weights = np.nan_to_num(rate)[inside]
    mean_distance = (distance[inside] * weights).sum() / weights.sum()
    return float((coverage - mean_distance) / (coverage + mean_distance))


def split_half_correlation(
    sessions: list[Session],
    units: list[Unit],
    unit_indices: np.ndarray,
    n_bins: int = 32,
    sigma: float = 1.0,
    position_sigma: float = 3.0,
    speed_range: tuple[float, float] = (6.0, 60.0),
) -> np.ndarray:
    """Pearson correlation between rate maps from the two halves of the session.

    A stability measure: a unit whose field moves or is only present in one
    half correlates poorly. Computed over bins visited in both halves.
    """
    cache = {}
    result = []
    for i in unit_indices:
        u = units[i]
        s = sessions[u.session]
        if u.session not in cache:
            xy, valid = preprocess_position(s, position_sigma, speed_range)
            cache[u.session] = (xy, valid, arena_box(xy, valid))
        xy, valid, box = cache[u.session]
        first = s.t < s.t[-1] / 2
        maps = []
        for half in (first, ~first):
            rate, occupancy = ratemap(u.spike_t, s.t, xy, valid & half, box, n_bins, sigma)
            maps.append((rate, occupancy > 0))
        both = maps[0][1] & maps[1][1]
        with np.errstate(invalid="ignore"):  # NaN for a unit silent in one half
            result.append(float(np.corrcoef(maps[0][0][both], maps[1][0][both])[0, 1]))
    return np.array(result)
