"""Safe loading and structural inspection of the published checkpoints."""

from __future__ import annotations

import re
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import torch

from .config import ModelConfig


class CheckpointError(RuntimeError):
    """Raised when a checkpoint is not compatible with this inference harness."""


class _FallbackScalarFloat(float):
    """Safe stand-in used only when ruamel.yaml is not installed."""


def _safe_checkpoint_globals() -> list[Any]:
    try:
        from ruamel.yaml.scalarfloat import ScalarFloat

        return [ScalarFloat]
    except ImportError:
        # Newer PyTorch versions accept (callable, fully-qualified-name). The
        # normal project install includes ruamel.yaml; this fallback keeps
        # source-tree diagnostics usable in minimal environments.
        return [(_FallbackScalarFloat, "ruamel.yaml.scalarfloat.ScalarFloat")]


def load_checkpoint_state(path: str | Path) -> tuple[dict[str, torch.Tensor], dict[str, Any]]:
    """Load only allowlisted checkpoint objects and return the model tensors.

    The Hugging Face artifacts are training checkpoints containing optimizer
    and scheduler state. ``weights_only=True`` prevents arbitrary pickle code
    execution, while ``mmap=True`` avoids eagerly reading every optimizer
    tensor from the larger artifacts.
    """

    checkpoint_path = Path(path).expanduser().resolve()
    if not checkpoint_path.is_file():
        raise FileNotFoundError(f"Checkpoint does not exist: {checkpoint_path}")

    try:
        with torch.serialization.safe_globals(_safe_checkpoint_globals()):
            payload = torch.load(
                checkpoint_path,
                map_location="cpu",
                mmap=True,
                weights_only=True,
            )
    except Exception as exc:  # Give a useful error without silently using unsafe pickle.
        raise CheckpointError(
            f"Could not safely load {checkpoint_path}. The loader will not retry "
            "with weights_only=False."
        ) from exc

    if not isinstance(payload, Mapping):
        raise CheckpointError(f"Expected a mapping in {checkpoint_path}, got {type(payload)!r}")

    state_candidate = payload.get("model_state", payload.get("state_dict", payload))
    if not isinstance(state_candidate, Mapping):
        raise CheckpointError("Checkpoint does not contain model_state or state_dict")

    state: dict[str, torch.Tensor] = {}
    for raw_key, value in state_candidate.items():
        if not isinstance(raw_key, str) or not torch.is_tensor(value):
            raise CheckpointError("Model state must contain only string keys and tensors")
        key = raw_key.removeprefix("module.")
        if key in state:
            raise CheckpointError(f"Duplicate state key after prefix cleanup: {key}")
        state[key] = value

    metadata = {
        "path": str(checkpoint_path),
        "iters": payload.get("iters"),
        "epoch": payload.get("epoch"),
        "checkpoint_keys": list(payload.keys()),
    }
    return state, metadata


def infer_model_config(state: Mapping[str, torch.Tensor]) -> ModelConfig:
    """Infer architecture dimensions from tensor shapes, not artifact names."""

    required = {
        "embedder.proj",
        "embedder.embed.projection",
        "mamba_layers.0.1.dt_bias",
        "mamba_layers.0.1.lin_B.weight",
        "mamba_layers.0.1.conv1d.weight",
        "mamba_layers.0.1.out_proj.weight",
        "output_layer.weight",
    }
    missing = sorted(required.difference(state))
    if missing:
        raise CheckpointError(
            "Checkpoint is not the published FM4NPP Mamba2 format; missing: "
            + ", ".join(missing)
        )

    layer_indices = []
    for key in state:
        match = re.match(r"mamba_layers\.(\d+)\.", key)
        if match:
            layer_indices.append(int(match.group(1)))
    if not layer_indices:
        raise CheckpointError("No mamba_layers were found")
    num_layers = max(layer_indices) + 1
    if set(layer_indices) != set(range(num_layers)):
        raise CheckpointError("Mamba layer indices are not contiguous")

    embed_dim = int(state["embedder.proj"].shape[1])
    out_proj = state["mamba_layers.0.1.out_proj.weight"]
    d_inner = int(out_proj.shape[1])
    if d_inner % embed_dim:
        raise CheckpointError("Cannot infer an integer Mamba expansion factor")
    expand = d_inner // embed_dim

    nheads = int(state["mamba_layers.0.1.dt_bias"].numel())
    if not nheads or d_inner % nheads:
        raise CheckpointError("Cannot infer a valid Mamba head dimension")
    headdim = d_inner // nheads

    # Published checkpoints use ngroups=1. Only the product ngroups*d_state is
    # identifiable from tensor shapes, so fail loudly if future artifacts need
    # a different registry entry rather than guessing silently.
    ngroups = 1
    d_state = int(state["mamba_layers.0.1.lin_B.weight"].shape[0])
    conv_weight = state["mamba_layers.0.1.conv1d.weight"]
    d_conv = int(conv_weight.shape[-1])
    expected_conv_channels = d_inner + 2 * ngroups * d_state
    if int(conv_weight.shape[0]) != expected_conv_channels:
        raise CheckpointError(
            "Checkpoint does not match the published ngroups=1 Mamba2 layout"
        )

    output_dim = int(state["output_layer.weight"].shape[0])
    if output_dim % 3:
        raise CheckpointError("Output dimension is not divisible into 3D neighbors")

    embed_method = "concat" if "embedder.proj2" in state else "add"
    return ModelConfig(
        embed_dim=embed_dim,
        num_layers=num_layers,
        d_state=d_state,
        d_conv=d_conv,
        expand=expand,
        headdim=headdim,
        ngroups=ngroups,
        k_neighbors=output_dim // 3,
        embed_method=embed_method,
        pe_method="nerf",
    )

