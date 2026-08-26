"""Minimal FM4NPP Python API example."""

from __future__ import annotations

import torch

from fm4npp_inference import load_model


def main() -> None:
    device = "cuda" if torch.cuda.is_available() else "cpu"
    backend = "fused" if device == "cuda" else "reference"
    dtype = torch.float16 if device == "cuda" else torch.float32

    loaded = load_model("m1", device=device, dtype=dtype, backend=backend)
    points = torch.randn(1, 64, 4, device=device, dtype=dtype)
    with torch.inference_mode():
        output = loaded.model(points)
    neighbors = output.reshape(1, 64, loaded.config.k_neighbors, 3)

    print(f"checkpoint: {loaded.checkpoint_path}")
    print(f"config: {loaded.config}")
    print(f"predicted neighbors: {tuple(neighbors.shape)}")


if __name__ == "__main__":
    main()

