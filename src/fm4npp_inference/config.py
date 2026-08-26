"""Published artifact names and checkpoint-derived model configuration."""

from __future__ import annotations

from dataclasses import asdict, dataclass


HF_REPO_ID = "FM4NPP/PP_collision"
HF_REVISION = "8aeb7f86a3eb4ed07876e8527d55fc98f5003254"

# These are artifact names, not a claim that they map one-to-one to the m1--m6
# labels in Table 1 of the paper. The loader derives the architecture from each
# checkpoint instead of relying on ambiguous filename labels.
MODEL_FILES = {
    "m1": "pp_nerf_m1_k30.ckpt",
    "m3": "pp_nerf_m3_k30.ckpt",
    "m4": "pp_nerf_m4_k30.ckpt",
    "m5": "pp_nerf_m5_k30.ckpt",
}


@dataclass(frozen=True)
class ModelConfig:
    embed_dim: int
    num_layers: int
    d_state: int
    d_conv: int
    expand: int
    headdim: int
    ngroups: int
    k_neighbors: int
    embed_method: str = "add"
    pe_method: str = "nerf"

    @property
    def output_dim(self) -> int:
        return self.k_neighbors * 3

    def to_dict(self) -> dict[str, int | str]:
        return asdict(self)

