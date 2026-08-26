"""TPCpp-10M event loading, coordinate conversion, and serialization."""

from __future__ import annotations

import json
from importlib.resources import files
from pathlib import Path
from typing import Literal

import numpy as np


InputFormat = Literal["cartesian", "preprocessed"]
Ordering = Literal["voxel", "radial"]


def preprocessing_statistics() -> dict[str, object]:
    resource = files("fm4npp_inference").joinpath("data/tpcpp10m_test_bins.json")
    return json.loads(resource.read_text(encoding="utf-8"))


def _validate_points(points: np.ndarray, label: str) -> np.ndarray:
    points = np.asarray(points)
    if points.ndim != 2 or points.shape[1] != 4:
        raise ValueError(f"{label} must have shape [num_points, 4], got {points.shape}")
    if len(points) == 0:
        raise ValueError(f"{label} contains no points")
    if not np.isfinite(points).all():
        raise ValueError(f"{label} contains NaN or infinite values")
    return points.astype(np.float32, copy=False)


def load_npz_event(path: str | Path, event_index: int = 0) -> tuple[np.ndarray, dict[str, object]]:
    """Load one event from the public flat ``data``/``size`` layout or a simple array."""

    input_path = Path(path).expanduser().resolve()
    if not input_path.is_file():
        raise FileNotFoundError(f"Input does not exist: {input_path}")

    if input_path.suffix == ".npy":
        array = np.load(input_path, allow_pickle=False)
        if array.ndim == 3:
            if not 0 <= event_index < len(array):
                raise IndexError(f"event_index {event_index} is outside [0, {len(array)})")
            array = array[event_index]
        elif event_index != 0:
            raise IndexError("event_index must be zero for a single 2D .npy event")
        return _validate_points(array, "Numpy input"), {
            "source": str(input_path),
            "event_index": event_index,
            "layout": "npy",
        }

    with np.load(input_path, allow_pickle=False) as archive:
        keys = list(archive.files)
        if "data" in archive and "size" in archive:
            sizes = np.asarray(archive["size"], dtype=np.int64)
            if not 0 <= event_index < len(sizes):
                raise IndexError(f"event_index {event_index} is outside [0, {len(sizes)})")
            start = int(sizes[:event_index].sum(dtype=np.int64))
            stop = start + int(sizes[event_index])
            event = archive["data"][start:stop]
            layout = "tpcpp10m-flat"
        else:
            candidate = next(
                (name for name in ("points", "spacepoints", "preprocessed", "arr_0") if name in archive),
                None,
            )
            if candidate is None:
                raise ValueError(
                    f"No supported point array in {input_path}; available keys: {keys}"
                )
            array = archive[candidate]
            if array.ndim == 3:
                if not 0 <= event_index < len(array):
                    raise IndexError(f"event_index {event_index} is outside [0, {len(array)})")
                event = array[event_index]
            else:
                if event_index != 0:
                    raise IndexError("event_index must be zero for a single 2D event")
                event = array
            layout = f"npz:{candidate}"

    return _validate_points(event, "NPZ event"), {
        "source": str(input_path),
        "event_index": event_index,
        "layout": layout,
    }


def cartesian_to_preprocessed(
    points: np.ndarray, ordering: Ordering = "voxel"
) -> tuple[np.ndarray, np.ndarray, dict[str, object]]:
    """Convert ``[E,x,y,z]`` into normalized, serialized ``[E,eta,phi,r]``."""

    points = _validate_points(points, "Cartesian points").astype(np.float64)
    stats = preprocessing_statistics()
    energy, x, y, z = points.T
    radius = np.sqrt(x * x + y * y)
    distance = np.sqrt(x * x + y * y + z * z)
    ratio = np.clip(z / np.maximum(distance, np.finfo(np.float64).tiny), -1 + 1e-12, 1 - 1e-12)
    eta = np.arctanh(ratio)
    phi = np.arctan2(y, x)

    normalized = np.stack(
        (
            (energy - float(stats["energy_mean"])) / float(stats["energy_std"]),
            (eta - float(stats["eta_min"]))
            / (float(stats["eta_max"]) - float(stats["eta_min"])),
            (phi - float(stats["phi_min"]))
            / (float(stats["phi_max"]) - float(stats["phi_min"])),
            (radius - float(stats["radius_min"]))
            / (float(stats["radius_max"]) - float(stats["radius_min"])),
        ),
        axis=-1,
    )

    # The publication loader first sorts by radius. Its subsequent voxel sort is
    # stable, so radial order remains the within-voxel ordering.
    original_indices = np.arange(len(normalized), dtype=np.int64)
    radial_order = np.argsort(normalized[:, 3], kind="stable")
    normalized = normalized[radial_order]
    original_indices = original_indices[radial_order]

    if ordering == "voxel":
        # ``side='left'`` matches torch.bucketize(..., right=False), including
        # the otherwise easy-to-miss behavior for values exactly on an edge.
        eta_bin = np.searchsorted(
            np.asarray(stats["eta_edges"]), normalized[:, 1], side="left"
        )
        phi_bin = np.searchsorted(
            np.asarray(stats["phi_edges"]), normalized[:, 2], side="left"
        )
        radius_bin = np.searchsorted(
            np.asarray(stats["radius_edges"]), normalized[:, 3], side="left"
        )
        eta_bin = np.clip(eta_bin, 1, 8)
        phi_bin = np.clip(phi_bin, 1, 8)
        radius_bin = np.clip(radius_bin, 1, 6)
        # EPR sweep in the upstream config: eta varies fastest, radius slowest.
        group_id = ((radius_bin - 1) * 8 + (phi_bin - 1)) * 8 + (eta_bin - 1)
        voxel_order = np.argsort(group_id, kind="stable")
        normalized = normalized[voxel_order]
        original_indices = original_indices[voxel_order]
    elif ordering != "radial":
        raise ValueError(f"Unknown ordering {ordering!r}")

    metadata = {
        "input_columns": ["E", "x", "y", "z"],
        "model_columns": ["normalized_E", "normalized_eta", "normalized_phi", "normalized_r"],
        "ordering": ordering,
        "statistics": stats["name"],
        "preprocessing_warning": stats["warning"],
    }
    return normalized.astype(np.float32), original_indices, metadata


def denormalize_neighbor_coordinates(predictions: np.ndarray) -> np.ndarray:
    """Convert predicted normalized ``[eta,phi,r]`` triples to physical coordinates."""

    stats = preprocessing_statistics()
    result = np.asarray(predictions, dtype=np.float32).copy()
    result[..., 0] = result[..., 0] * (
        float(stats["eta_max"]) - float(stats["eta_min"])
    ) + float(stats["eta_min"])
    result[..., 1] = result[..., 1] * (
        float(stats["phi_max"]) - float(stats["phi_min"])
    ) + float(stats["phi_min"])
    result[..., 2] = result[..., 2] * (
        float(stats["radius_max"]) - float(stats["radius_min"])
    ) + float(stats["radius_min"])
    return result


def synthetic_preprocessed(sequence_length: int, seed: int = 42) -> np.ndarray:
    if sequence_length < 1:
        raise ValueError("sequence_length must be positive")
    generator = np.random.default_rng(seed)
    energy = generator.normal(0.0, 1.0, size=(sequence_length, 1))
    coordinates = generator.uniform(0.0, 1.0, size=(sequence_length, 3))
    points = np.concatenate((energy, coordinates), axis=-1).astype(np.float32)
    # Give synthetic inputs a stable outward ordering too.
    return points[np.argsort(points[:, 3], kind="stable")]
