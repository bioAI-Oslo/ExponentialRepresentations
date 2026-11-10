"""Reduce the raw CA1 release to `data/ca1_events.npz`. See `ca1.py` for the format.

    uv run python notebooks/experimental_data/extract_ca1.py ca1_data data/ca1_events.npz

Reads only the tracking files, the `.set` headers and the MClust `.t64` cluster
files; the raw waveform and LFP files are never touched. Axona sessions go
through the loaders shipped with the release (`ca1_data/scripts`), which need a
NumPy 2 shim; the Neuralynx video file is read directly because the shipped
reader's bit-field decoding fails under NumPy 2 and the extracted position is
stored in plain fields anyway.
"""

import re
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
from ca1 import Session, Unit, save_events

AXONA_PX_PER_M = 300  # recorded in every .pos header
AXONA_COVERAGE_CM = 137.0  # median occupied extent of the Axona animals at that scale
NVT_HEADER_BYTES = 16384
NVT_RECORD = np.dtype(
    [
        ("swstx", np.uint16),
        ("swid", np.uint16),
        ("swdata_size", np.uint16),
        ("qwTimeStamp", np.uint64),
        ("dwPoints", (np.uint32, 400)),
        ("sncrc", np.int16),
        ("dnextracted_x", np.int32),
        ("dnextracted_y", np.int32),
        ("dnextracted_angle", np.int32),
        ("dntargets", (np.int32, 50)),
    ]
)
CLUSTER_FILE = re.compile(r"(\d+)_(\d+)\.t64$")


def occupied_extent(xy: np.ndarray) -> float:
    return float((np.nanquantile(xy, 0.995, axis=0) - np.nanquantile(xy, 0.005, axis=0)).mean())


def axona_session(folder: Path, axona) -> Session:
    base = folder / folder.name
    settings = axona.load_set_file(str(base) + ".set")
    _, t, coords, _ = axona.load_pos(str(base) + ".pos")
    # Head = midpoint of the two LEDs where both are tracked, else the first.
    xy = coords[:, :2].copy()
    both = ~np.isnan(coords).any(axis=1)
    xy[both] = (coords[both, :2] + coords[both, 2:4]) / 2
    cm_per_px = 100 / AXONA_PX_PER_M
    return Session(folder.parent.name, folder.name, "axona", float(settings["duration"]), cm_per_px, t, xy * cm_per_px)


def neuralynx_session(folder: Path) -> tuple[Session, float]:
    (nvt,) = folder.glob("*.nvt")
    with nvt.open("rb") as f:
        f.seek(NVT_HEADER_BYTES)
        records = np.fromfile(f, dtype=NVT_RECORD)
    t_us = records["qwTimeStamp"].astype(np.float64)
    xy = np.stack([records["dnextracted_x"], records["dnextracted_y"]], axis=1).astype(np.float64)
    xy[(xy == 0).all(axis=1)] = np.nan  # (0, 0) marks a lost frame
    cm_per_px = AXONA_COVERAGE_CM / occupied_extent(xy)
    t = (t_us - t_us[0]) / 1e6
    return Session(folder.parent.name, folder.name, "neuralynx", float(t[-1]), cm_per_px, t, xy * cm_per_px), t_us[0]


def cluster_units(folder: Path, session_index: int, duration: float, mclust, clock_offset_s: float = 0.0) -> list[Unit]:
    units = []
    for f in sorted(folder.glob("*.t64")):
        match = CLUSTER_FILE.search(f.name)
        if match is None:
            raise ValueError(f"unexpected cluster file name {f.name}")
        tetrode, cluster = (int(g) for g in match.groups())
        _, spike_t = mclust.load_tfile(str(f))  # seconds; Neuralynx on the Cheetah clock
        spike_t = spike_t - clock_offset_s
        units.append(Unit(session_index, tetrode, cluster, spike_t, len(spike_t) / duration, -1))
    mark_nested_duplicates(units)
    return units


def mark_nested_duplicates(units: list[Unit], threshold: float = 0.5) -> None:
    """A cluster sharing more than `threshold` of its spikes with a larger one is a duplicate of it."""
    order = sorted(range(len(units)), key=lambda i: -len(units[i].spike_t))
    for a, i in enumerate(order):
        for j in order[:a]:
            if len(units[i].spike_t) and np.isin(units[i].spike_t, units[j].spike_t).mean() > threshold:
                units[i].duplicate_of = j
                break


def main(raw: Path, out: Path) -> None:
    sys.path.insert(0, str(raw))
    # The shipped loaders predate NumPy 2 and still use the removed alias.
    np.string_ = np.bytes_  # noqa: NPY201  # pyright: ignore[reportAttributeAccessIssue]
    from scripts import axona, mclust  # pyright: ignore[reportMissingImports]  # loaders live in the raw release

    sessions, units = [], []
    for animal in sorted((raw / "data").iterdir()):
        if not animal.is_dir():
            continue
        for folder in sorted(p for p in animal.iterdir() if p.is_dir()):
            index = len(sessions)
            if (folder / f"{folder.name}.set").exists():
                session = axona_session(folder, axona)
                offset = 0.0
            else:
                session, t0_us = neuralynx_session(folder)
                offset = t0_us / 1e6
            new_units = cluster_units(folder, index, session.duration, mclust, offset)
            for u in new_units:
                if u.duplicate_of >= 0:
                    u.duplicate_of += len(units)
            sessions.append(session)
            units.extend(new_units)
            print(
                f"{session.animal} {session.date} {session.system:9s} {session.duration:6.0f} s  {len(new_units):3d} clusters"
            )
    save_events(out, sessions, units)
    print(f"\n{len(sessions)} sessions, {len(units)} clusters, {sum(len(u.spike_t) for u in units)} spikes -> {out}")


if __name__ == "__main__":
    main(Path(sys.argv[1]), Path(sys.argv[2]))
