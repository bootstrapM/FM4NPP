"""Inference utilities for the FM4NPP proton-proton collision model."""

from .config import HF_REPO_ID, HF_REVISION, MODEL_FILES, ModelConfig
from .loader import load_model

__all__ = [
    "HF_REPO_ID",
    "HF_REVISION",
    "MODEL_FILES",
    "ModelConfig",
    "load_model",
]

__version__ = "0.1.0"

