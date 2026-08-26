from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import torch

from fm4npp_inference.checkpoint import infer_model_config, load_checkpoint_state
from fm4npp_inference.config import ModelConfig
from fm4npp_inference.model import FM4NPP


class CheckpointTests(unittest.TestCase):
    def test_checkpoint_round_trip_and_config_inference(self) -> None:
        expected = ModelConfig(
            embed_dim=64,
            num_layers=2,
            d_state=8,
            d_conv=3,
            expand=2,
            headdim=32,
            ngroups=1,
            k_neighbors=5,
        )
        model = FM4NPP(expected, backend="reference")
        parallel_state = {f"module.{key}": value for key, value in model.state_dict().items()}
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "test.ckpt"
            torch.save({"iters": 12, "epoch": 3, "model_state": parallel_state}, path)

            state, metadata = load_checkpoint_state(path)
            actual = infer_model_config(state)

        self.assertEqual(actual, expected)
        self.assertEqual(metadata["iters"], 12)
        self.assertEqual(metadata["epoch"], 3)
        self.assertTrue(all(not key.startswith("module.") for key in state))
