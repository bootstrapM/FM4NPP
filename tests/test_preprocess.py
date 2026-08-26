from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import numpy as np

from fm4npp_inference.preprocess import (
    cartesian_to_preprocessed,
    denormalize_neighbor_coordinates,
    load_npz_event,
    preprocessing_statistics,
)


class PreprocessTests(unittest.TestCase):
    def test_cartesian_conversion_and_radial_order(self) -> None:
        stats = preprocessing_statistics()
        points = np.asarray(
            [
                [stats["energy_mean"], 50.0, 0.0, 0.0],
                [stats["energy_mean"], 40.0, 0.0, 0.0],
            ],
            dtype=np.float32,
        )
        normalized, original_indices, metadata = cartesian_to_preprocessed(
            points, ordering="radial"
        )

        self.assertEqual(original_indices.tolist(), [1, 0])
        self.assertTrue(np.allclose(normalized[:, 0], 0.0, atol=1e-6))
        self.assertTrue(np.allclose(normalized[:, 1], 0.5, atol=1e-6))
        self.assertTrue(np.allclose(normalized[:, 2], 0.5, atol=1e-6))
        self.assertLess(normalized[0, 3], normalized[1, 3])
        self.assertEqual(metadata["ordering"], "radial")

    def test_flat_tpcpp_layout_event_selection(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "events.npz"
            first = np.arange(8, dtype=np.float32).reshape(2, 4)
            second = np.arange(12, dtype=np.float32).reshape(3, 4) + 100
            np.savez(path, data=np.concatenate((first, second)), size=np.asarray([2, 3]))

            event, metadata = load_npz_event(path, event_index=1)

        self.assertTrue(np.array_equal(event, second))
        self.assertEqual(metadata["layout"], "tpcpp10m-flat")

    def test_denormalization(self) -> None:
        stats = preprocessing_statistics()
        normalized = np.asarray(
            [[[0.5, 0.5, 0.0], [1.0, 1.0, 1.0]]], dtype=np.float32
        )
        physical = denormalize_neighbor_coordinates(normalized)

        self.assertTrue(
            np.allclose(physical[0, 0], [0.0, 0.0, stats["radius_min"]])
        )
        self.assertTrue(
            np.allclose(
                physical[0, 1],
                [stats["eta_max"], stats["phi_max"], stats["radius_max"]],
            )
        )
