"""Hugging Face download and checkpoint-to-model loading."""

from __future__ import annotations

import gc
from dataclasses import dataclass
from pathlib import Path

import torch

from .checkpoint import infer_model_config, load_checkpoint_state
from .config import HF_REPO_ID, HF_REVISION, MODEL_FILES, ModelConfig
from .model import Backend, FM4NPP


@dataclass
class LoadedModel:
    model: FM4NPP
    config: ModelConfig
    checkpoint_path: Path
    checkpoint_metadata: dict[str, object]
    parameter_count: int


def download_checkpoint(
    model_name: str = "m1",
    *,
    revision: str = HF_REVISION,
    cache_dir: str | Path | None = None,
    local_dir: str | Path | None = None,
    local_files_only: bool = False,
) -> Path:
    if model_name not in MODEL_FILES:
        raise ValueError(f"Unknown model {model_name!r}; choose from {sorted(MODEL_FILES)}")
    try:
        from huggingface_hub import hf_hub_download
    except ImportError as exc:
        raise RuntimeError(
            "huggingface-hub is required for downloads. Install the project first."
        ) from exc

    path = hf_hub_download(
        repo_id=HF_REPO_ID,
        filename=MODEL_FILES[model_name],
        revision=revision,
        cache_dir=str(cache_dir) if cache_dir is not None else None,
        local_dir=str(local_dir) if local_dir is not None else None,
        local_files_only=local_files_only,
    )
    return Path(path).resolve()


def load_model(
    model_name: str = "m1",
    *,
    checkpoint: str | Path | None = None,
    revision: str = HF_REVISION,
    cache_dir: str | Path | None = None,
    local_files_only: bool = False,
    device: str | torch.device = "cpu",
    dtype: torch.dtype = torch.float32,
    backend: Backend = "auto",
) -> LoadedModel:
    """Download if needed, infer the architecture, and load strict weights."""

    checkpoint_path = (
        Path(checkpoint).expanduser().resolve()
        if checkpoint is not None
        else download_checkpoint(
            model_name,
            revision=revision,
            cache_dir=cache_dir,
            local_files_only=local_files_only,
        )
    )
    state, metadata = load_checkpoint_state(checkpoint_path)
    config = infer_model_config(state)
    model = FM4NPP(config, backend=backend, dropout=0.0)
    incompatible = model.load_state_dict(state, strict=True)
    if incompatible.missing_keys or incompatible.unexpected_keys:
        raise RuntimeError(f"Strict state load failed: {incompatible}")
    parameter_count = sum(parameter.numel() for parameter in model.parameters())

    # The checkpoint contains large optimizer states. Drop our references before
    # moving the actual model so mmap-backed pages can be reclaimed promptly.
    del state
    gc.collect()

    model = model.to(device=device, dtype=dtype)
    model.eval()
    return LoadedModel(
        model=model,
        config=config,
        checkpoint_path=checkpoint_path,
        checkpoint_metadata=metadata,
        parameter_count=parameter_count,
    )

