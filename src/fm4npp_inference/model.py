"""Checkpoint-compatible FM4NPP Mamba2 backbone.

The CUDA path calls the official Mamba fused SSD kernel. The reference path is
an exact recurrent form of the same selective state-space equations; it is
portable but intentionally slower for long sequences.
"""

from __future__ import annotations

import math
from typing import Literal

import torch
from torch import Tensor, nn
from torch.nn import functional as F

from .config import ModelConfig


Backend = Literal["auto", "fused", "reference"]

try:
    from mamba_ssm.ops.triton.ssd_combined import (
        mamba_split_conv1d_scan_combined as _fused_mamba,
    )

    _FUSED_IMPORT_ERROR: str | None = None
except Exception as exc:  # Native packages can fail with ImportError or OSError.
    _fused_mamba = None
    _FUSED_IMPORT_ERROR = f"{type(exc).__name__}: {exc}"


def fused_backend_status() -> tuple[bool, str | None]:
    return _fused_mamba is not None, _FUSED_IMPORT_ERROR


class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-5) -> None:
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x: Tensor) -> Tensor:
        variance = x.float().square().mean(dim=-1, keepdim=True)
        normalized = x * torch.rsqrt(variance + self.eps).to(x.dtype)
        return normalized * self.weight


class GatedRMSNorm(nn.Module):
    def __init__(
        self,
        dim: int,
        eps: float = 1e-5,
        group_size: int | None = None,
        norm_before_gate: bool = False,
    ) -> None:
        super().__init__()
        self.eps = eps
        self.group_size = group_size
        self.norm_before_gate = norm_before_gate
        self.weight = nn.Parameter(torch.ones(dim))

    def _normalize(self, x: Tensor) -> Tensor:
        if self.group_size is None:
            variance = x.square().mean(dim=-1, keepdim=True)
            return x * torch.rsqrt(variance + self.eps)
        groups = x.shape[-1] // self.group_size
        grouped = x.reshape(*x.shape[:-1], groups, self.group_size)
        variance = grouped.square().mean(dim=-1, keepdim=True)
        normalized = grouped * torch.rsqrt(variance + self.eps)
        return normalized.reshape_as(x)

    def forward(self, x: Tensor, gate: Tensor | None = None) -> Tensor:
        output_dtype = x.dtype
        x = x.float()
        gate = gate.float() if gate is not None else None
        if gate is not None and not self.norm_before_gate:
            x = x * F.silu(gate)
        output = self._normalize(x) * self.weight.float()
        if gate is not None and self.norm_before_gate:
            output = output * F.silu(gate)
        return output.to(output_dtype)


class NerfCoordinateEmbedder(nn.Module):
    def __init__(self, target_dim: int, num_frequencies: int = 10) -> None:
        super().__init__()
        encoded_dim = 3 + 2 * 3 * num_frequencies
        self.projection = nn.Parameter(
            torch.randn(encoded_dim, target_dim), requires_grad=False
        )
        frequencies = 2.0 ** torch.linspace(0.0, float(num_frequencies - 1), num_frequencies)
        self.register_buffer("frequencies", frequencies, persistent=False)

    def forward(self, coordinates: Tensor) -> Tensor:
        frequencies = self.frequencies.to(device=coordinates.device, dtype=coordinates.dtype)
        scaled = coordinates.unsqueeze(-2) * frequencies.view(
            *((1,) * (coordinates.ndim - 1)), -1, 1
        )
        periodic = torch.stack((torch.sin(scaled), torch.cos(scaled)), dim=-2)
        encoded = torch.cat((coordinates, periodic.flatten(start_dim=-3)), dim=-1)
        return encoded @ self.projection


class AdditiveEmbedder(nn.Module):
    def __init__(self, embed_dim: int) -> None:
        super().__init__()
        self.proj = nn.Parameter(torch.randn(1, embed_dim), requires_grad=False)
        self.embed = NerfCoordinateEmbedder(embed_dim)

    def forward(self, points: Tensor) -> tuple[Tensor, Tensor]:
        energy = points[..., :1] @ self.proj
        position = self.embed(points[..., 1:4])
        return energy + position, position


class ConcatEmbedder(nn.Module):
    def __init__(self, embed_dim: int) -> None:
        super().__init__()
        energy_dim = embed_dim // 4
        self.proj = nn.Parameter(torch.randn(1, energy_dim), requires_grad=False)
        self.embed = NerfCoordinateEmbedder(embed_dim)
        self.proj2 = nn.Parameter(
            torch.randn(energy_dim + embed_dim, embed_dim), requires_grad=False
        )

    def forward(self, points: Tensor) -> tuple[Tensor, Tensor]:
        energy = points[..., :1] @ self.proj
        position = self.embed(points[..., 1:4])
        return torch.cat((energy, position), dim=-1) @ self.proj2, position


class FM4NPPMamba2(nn.Module):
    """Mamba2 block matching the custom layout in the published checkpoints."""

    def __init__(
        self,
        d_model: int,
        d_state: int,
        d_conv: int,
        expand: int,
        headdim: int,
        ngroups: int,
        backend: Backend,
    ) -> None:
        super().__init__()
        if backend not in {"auto", "fused", "reference"}:
            raise ValueError(f"Unknown backend: {backend}")
        self.backend = backend
        self.d_model = d_model
        self.d_state = d_state
        self.d_conv = d_conv
        self.expand = expand
        self.d_inner = expand * d_model
        self.headdim = headdim
        self.ngroups = ngroups
        if self.d_inner % headdim:
            raise ValueError("Expanded model dimension must be divisible by headdim")
        self.nheads = self.d_inner // headdim
        if self.nheads % ngroups:
            raise ValueError("Number of heads must be divisible by ngroups")

        self.in_proj = nn.Linear(d_model, 2 * self.d_inner + self.nheads, bias=False)
        self.lin_B = nn.Linear(d_model, ngroups * d_state, bias=False)
        self.lin_C = nn.Linear(d_model, ngroups * d_state, bias=False)

        conv_channels = self.d_inner + 2 * ngroups * d_state
        self.conv1d = nn.Conv1d(
            conv_channels,
            conv_channels,
            kernel_size=d_conv,
            groups=conv_channels,
            padding=d_conv - 1,
            bias=True,
        )
        self.norm = GatedRMSNorm(
            self.d_inner,
            eps=1e-5,
            group_size=self.d_inner // ngroups,
            norm_before_gate=False,
        )
        self.out_proj = nn.Linear(self.d_inner, d_model, bias=False)

        dt = torch.exp(
            torch.rand(self.nheads) * (math.log(0.1) - math.log(0.001))
            + math.log(0.001)
        )
        self.dt_bias = nn.Parameter(dt + torch.log(-torch.expm1(-dt)))
        self.A_log = nn.Parameter(torch.log(torch.empty(self.nheads).uniform_(1, 16)))
        self.D = nn.Parameter(torch.ones(self.nheads))

    def backend_for(self, x: Tensor) -> str:
        if self.backend == "reference":
            return "reference"
        if self.backend == "fused":
            if _fused_mamba is None:
                raise RuntimeError(
                    "The fused backend was requested but mamba-ssm/Triton could not be imported: "
                    f"{_FUSED_IMPORT_ERROR}"
                )
            if not x.is_cuda:
                raise RuntimeError("The fused Mamba backend requires a CUDA tensor")
            return "fused"
        return "fused" if _fused_mamba is not None and x.is_cuda else "reference"

    def _project(self, u: Tensor) -> tuple[Tensor, Tensor, Tensor, Tensor, Tensor]:
        projected = self.in_proj(u)
        gate, x, dt = torch.split(
            projected, [self.d_inner, self.d_inner, self.nheads], dim=-1
        )
        return gate, x, self.lin_B(u), self.lin_C(u), dt

    def _forward_fused(
        self, gate: Tensor, x: Tensor, B: Tensor, C: Tensor, dt: Tensor
    ) -> Tensor:
        assert _fused_mamba is not None
        packed = torch.cat((gate, x, B, C, dt), dim=-1)
        return _fused_mamba(
            packed,
            self.conv1d.weight.squeeze(1),
            self.conv1d.bias,
            self.dt_bias,
            -torch.exp(self.A_log.float()),
            D=self.D,
            chunk_size=256,
            activation="silu",
            rmsnorm_weight=self.norm.weight,
            rmsnorm_eps=self.norm.eps,
            outproj_weight=self.out_proj.weight,
            outproj_bias=self.out_proj.bias,
            headdim=self.headdim,
            ngroups=self.ngroups,
            norm_before_gate=self.norm.norm_before_gate,
        )

    def _forward_reference(
        self, gate: Tensor, x: Tensor, B: Tensor, C: Tensor, dt: Tensor
    ) -> Tensor:
        batch, sequence_length, _ = x.shape
        conv_input = torch.cat((x, B, C), dim=-1).transpose(1, 2)
        convolved = self.conv1d(conv_input)[..., :sequence_length].transpose(1, 2)
        convolved = F.silu(convolved)
        x, B, C = torch.split(
            convolved,
            [self.d_inner, self.ngroups * self.d_state, self.ngroups * self.d_state],
            dim=-1,
        )

        x_heads = x.reshape(batch, sequence_length, self.nheads, self.headdim)
        B_groups = B.reshape(batch, sequence_length, self.ngroups, self.d_state)
        C_groups = C.reshape(batch, sequence_length, self.ngroups, self.d_state)
        heads_per_group = self.nheads // self.ngroups
        B_heads = B_groups.repeat_interleave(heads_per_group, dim=2)
        C_heads = C_groups.repeat_interleave(heads_per_group, dim=2)

        steps = F.softplus(dt.float() + self.dt_bias.float())
        rates = -torch.exp(self.A_log.float())
        state = torch.zeros(
            batch,
            self.nheads,
            self.headdim,
            self.d_state,
            device=x.device,
            dtype=torch.float32,
        )
        outputs = []
        for index in range(sequence_length):
            step = steps[:, index]
            decay = torch.exp(step * rates).unsqueeze(-1).unsqueeze(-1)
            x_t = x_heads[:, index].float()
            B_t = B_heads[:, index].float()
            state = decay * state + torch.einsum("bh,bhn,bhp->bhpn", step, B_t, x_t)
            C_t = C_heads[:, index].float()
            y_t = torch.einsum("bhpn,bhn->bhp", state, C_t)
            y_t = y_t + self.D.float().view(1, self.nheads, 1) * x_t
            outputs.append(y_t)

        y = torch.stack(outputs, dim=1).to(x.dtype).reshape(
            batch, sequence_length, self.d_inner
        )
        return self.out_proj(self.norm(y, gate))

    def forward(self, u: Tensor) -> Tensor:
        gate, x, B, C, dt = self._project(u)
        if self.backend_for(u) == "fused":
            return self._forward_fused(gate, x, B, C, dt)
        return self._forward_reference(gate, x, B, C, dt)


class FM4NPP(nn.Module):
    """FM4NPP pretraining backbone and k-next-nearest-neighbor head."""

    def __init__(self, config: ModelConfig, backend: Backend = "auto", dropout: float = 0.0):
        super().__init__()
        self.config = config
        self.backend = backend
        embedder = AdditiveEmbedder if config.embed_method == "add" else ConcatEmbedder
        self.embedder = embedder(config.embed_dim)
        self.mamba_layers = nn.ModuleList(
            [
                nn.Sequential(
                    RMSNorm(config.embed_dim),
                    FM4NPPMamba2(
                        d_model=config.embed_dim,
                        d_state=config.d_state,
                        d_conv=config.d_conv,
                        expand=config.expand,
                        headdim=config.headdim,
                        ngroups=config.ngroups,
                        backend=backend,
                    ),
                    nn.Dropout(dropout),
                )
                for _ in range(config.num_layers)
            ]
        )
        self.output_layer = nn.Linear(config.embed_dim, config.output_dim)
        self.norm = RMSNorm(config.embed_dim)

    def active_backend(self, device: torch.device | str) -> str:
        probe = torch.empty(0, device=device)
        return self.mamba_layers[0][1].backend_for(probe)

    def forward(
        self, points: Tensor, return_features: bool = False
    ) -> Tensor | tuple[Tensor, tuple[Tensor, ...], Tensor]:
        if points.ndim != 3 or points.shape[-1] != 4:
            raise ValueError(
                f"Expected [batch, points, 4] (E, eta, phi, r), got {tuple(points.shape)}"
            )
        points = torch.where(points == -100, torch.zeros_like(points), points)
        hidden, positional = self.embedder(points)
        layer_outputs = []
        for layer in self.mamba_layers:
            residual_branch = layer(hidden)
            if return_features:
                layer_outputs.append(residual_branch)
            hidden = hidden + residual_branch
        predictions = self.output_layer(self.norm(hidden))
        if return_features:
            return predictions, tuple(layer_outputs), positional
        return predictions
