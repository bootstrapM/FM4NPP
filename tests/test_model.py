from __future__ import annotations

import torch
import unittest

from fm4npp_inference.config import ModelConfig
from fm4npp_inference.model import FM4NPP


class ModelTests(unittest.TestCase):
    def test_reference_forward_shape_and_features(self) -> None:
        config = ModelConfig(
            embed_dim=64,
            num_layers=2,
            d_state=8,
            d_conv=3,
            expand=2,
            headdim=32,
            ngroups=1,
            k_neighbors=5,
        )
        model = FM4NPP(config, backend="reference").eval()
        points = torch.randn(2, 7, 4)

        with torch.inference_mode():
            predictions, features, positions = model(points, return_features=True)

        self.assertEqual(predictions.shape, (2, 7, 15))
        self.assertEqual(positions.shape, (2, 7, 64))
        self.assertEqual(len(features), 2)
        self.assertTrue(all(feature.shape == (2, 7, 64) for feature in features))
        self.assertTrue(torch.isfinite(predictions).all())
        self.assertEqual(model.active_backend("cpu"), "reference")

    def test_mask_sentinel_matches_zero(self) -> None:
        config = ModelConfig(
            embed_dim=64,
            num_layers=1,
            d_state=8,
            d_conv=2,
            expand=2,
            headdim=32,
            ngroups=1,
            k_neighbors=2,
        )
        model = FM4NPP(config, backend="reference").eval()
        masked = torch.full((1, 3, 4), -100.0)
        zeros = torch.zeros_like(masked)
        with torch.inference_mode():
            self.assertTrue(torch.equal(model(masked), model(zeros)))
