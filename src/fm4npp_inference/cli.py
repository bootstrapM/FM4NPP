"""Command-line interface for download, inference, diagnostics, and timing."""

from __future__ import annotations

import argparse
import json
import platform
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch

from .config import HF_REVISION, MODEL_FILES
from .loader import download_checkpoint, load_model
from .model import fused_backend_status
from .preprocess import (
    cartesian_to_preprocessed,
    denormalize_neighbor_coordinates,
    load_npz_event,
    synthetic_preprocessed,
)


def _json(data: dict[str, Any]) -> str:
    return json.dumps(data, indent=2, sort_keys=True)


def _resolve_device(requested: str) -> torch.device:
    if requested != "auto":
        device = torch.device(requested)
    elif torch.cuda.is_available():
        device = torch.device("cuda")
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but torch.cuda.is_available() is false")
    return device


def _resolve_dtype(name: str) -> torch.dtype:
    return {
        "fp32": torch.float32,
        "fp16": torch.float16,
        "bf16": torch.bfloat16,
    }[name]


def _synchronize(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def _model_kwargs(args: argparse.Namespace, device: torch.device) -> dict[str, Any]:
    return {
        "model_name": args.model,
        "checkpoint": args.checkpoint,
        "revision": args.revision,
        "cache_dir": args.cache_dir,
        "local_files_only": args.local_files_only,
        "device": device,
        "dtype": _resolve_dtype(args.dtype),
        "backend": args.backend,
    }


def command_diagnose(args: argparse.Namespace) -> int:
    fused, fused_error = fused_backend_status()
    cuda_devices = []
    if torch.cuda.is_available():
        for index in range(torch.cuda.device_count()):
            props = torch.cuda.get_device_properties(index)
            cuda_devices.append(
                {
                    "index": index,
                    "name": props.name,
                    "total_memory_gib": round(props.total_memory / 2**30, 2),
                    "compute_capability": f"{props.major}.{props.minor}",
                }
            )
    result = {
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "torch": torch.__version__,
        "cuda_available": torch.cuda.is_available(),
        "torch_cuda": torch.version.cuda,
        "cuda_devices": cuda_devices,
        "fused_mamba_importable": fused,
        "fused_mamba_import_error": fused_error,
        "reference_backend_available": True,
    }
    print(_json(result))
    if args.require_fused and (not fused or not torch.cuda.is_available()):
        return 1
    return 0


def command_download(args: argparse.Namespace) -> int:
    path = download_checkpoint(
        args.model,
        revision=args.revision,
        cache_dir=args.cache_dir,
        local_dir=args.local_dir,
        local_files_only=args.local_files_only,
    )
    print(path)
    return 0


def command_inspect(args: argparse.Namespace) -> int:
    loaded = load_model(**_model_kwargs(args, torch.device("cpu")))
    result = {
        "checkpoint": str(loaded.checkpoint_path),
        "checkpoint_metadata": loaded.checkpoint_metadata,
        "config": loaded.config.to_dict(),
        "parameter_count": loaded.parameter_count,
    }
    print(_json(result))
    return 0


def _prepare_points(args: argparse.Namespace) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    if args.input is None:
        points = synthetic_preprocessed(args.sequence_length, seed=args.seed)
        return points, np.arange(len(points), dtype=np.int64), {
            "source": "synthetic",
            "seed": args.seed,
            "input_format": "preprocessed",
        }

    raw_points, source_metadata = load_npz_event(args.input, args.event_index)
    if args.input_format == "cartesian":
        points, original_indices, preprocess_metadata = cartesian_to_preprocessed(
            raw_points, ordering=args.ordering
        )
        metadata = {**source_metadata, **preprocess_metadata, "input_format": "cartesian"}
    else:
        points = raw_points
        original_indices = np.arange(len(points), dtype=np.int64)
        metadata = {
            **source_metadata,
            "input_format": "preprocessed",
            "model_columns": ["normalized_E", "normalized_eta", "normalized_phi", "normalized_r"],
        }

    if args.max_points is not None:
        points = points[: args.max_points]
        original_indices = original_indices[: args.max_points]
        metadata["truncated_to_points"] = args.max_points
    return points, original_indices, metadata


def command_infer(args: argparse.Namespace) -> int:
    device = _resolve_device(args.device)
    loaded = load_model(**_model_kwargs(args, device))
    points, original_indices, input_metadata = _prepare_points(args)
    tensor = torch.from_numpy(points).unsqueeze(0).to(
        device=device, dtype=_resolve_dtype(args.dtype)
    )

    _synchronize(device)
    start = time.perf_counter()
    with torch.inference_mode():
        raw = loaded.model(tensor)
    _synchronize(device)
    elapsed = time.perf_counter() - start

    assert isinstance(raw, torch.Tensor)
    normalized = (
        raw.reshape(1, len(points), loaded.config.k_neighbors, 3)[0]
        .float()
        .cpu()
        .numpy()
    )
    physical = denormalize_neighbor_coordinates(normalized)
    metadata = {
        "model_artifact": args.model,
        "checkpoint": str(loaded.checkpoint_path),
        "checkpoint_metadata": loaded.checkpoint_metadata,
        "config": loaded.config.to_dict(),
        "parameter_count": loaded.parameter_count,
        "device": str(device),
        "dtype": args.dtype,
        "backend": loaded.model.active_backend(device),
        "elapsed_seconds": elapsed,
        "num_points": len(points),
        "prediction_semantics": "k next-nearest-neighbor coordinates; not track IDs",
        "input": input_metadata,
    }

    output_path = Path(args.output).expanduser().resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        output_path,
        predictions_normalized=normalized,
        predictions_eta_phi_r=physical,
        input_preprocessed=points,
        original_indices=original_indices,
        metadata=np.asarray(json.dumps(metadata, sort_keys=True)),
    )
    print(_json({"output": str(output_path), **metadata}))
    return 0


def command_benchmark(args: argparse.Namespace) -> int:
    device = _resolve_device(args.device)
    loaded = load_model(**_model_kwargs(args, device))
    points = synthetic_preprocessed(args.sequence_length, seed=args.seed)
    tensor = torch.from_numpy(points).unsqueeze(0).repeat(args.batch_size, 1, 1)
    tensor = tensor.to(device=device, dtype=_resolve_dtype(args.dtype))
    model: Any = loaded.model
    if args.compile:
        model = torch.compile(model)

    with torch.inference_mode():
        for _ in range(args.warmup):
            model(tensor)
        _synchronize(device)
        timings = []
        if device.type == "cuda":
            torch.cuda.reset_peak_memory_stats(device)
        for _ in range(args.steps):
            start = time.perf_counter()
            model(tensor)
            _synchronize(device)
            timings.append(time.perf_counter() - start)

    timings_ms = np.asarray(timings) * 1000.0
    mean_seconds = float(np.mean(timings))
    record = {
        "model_artifact": args.model,
        "checkpoint": str(loaded.checkpoint_path),
        "config": loaded.config.to_dict(),
        "parameter_count": loaded.parameter_count,
        "device": str(device),
        "dtype": args.dtype,
        "backend": loaded.model.active_backend(device),
        "compiled": args.compile,
        "batch_size": args.batch_size,
        "sequence_length": args.sequence_length,
        "warmup_steps": args.warmup,
        "measured_steps": args.steps,
        "latency_ms_mean": float(timings_ms.mean()),
        "latency_ms_p50": float(np.percentile(timings_ms, 50)),
        "latency_ms_p95": float(np.percentile(timings_ms, 95)),
        "events_per_second": args.batch_size / mean_seconds,
        "points_per_second": args.batch_size * args.sequence_length / mean_seconds,
        "peak_cuda_memory_mib": (
            torch.cuda.max_memory_allocated(device) / 2**20 if device.type == "cuda" else None
        ),
    }
    print(_json(record))
    if args.output is not None:
        output_path = Path(args.output).expanduser().resolve()
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with output_path.open("a", encoding="utf-8") as stream:
            stream.write(json.dumps(record, sort_keys=True) + "\n")
    return 0


def _add_model_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--model", choices=sorted(MODEL_FILES), default="m1")
    parser.add_argument("--checkpoint", help="Use a local checkpoint instead of downloading")
    parser.add_argument("--revision", default=HF_REVISION, help="Hugging Face revision")
    parser.add_argument("--cache-dir", help="Hugging Face cache directory")
    parser.add_argument("--local-files-only", action="store_true")
    parser.add_argument("--dtype", choices=("fp32", "fp16", "bf16"), default="fp32")
    parser.add_argument("--backend", choices=("auto", "fused", "reference"), default="auto")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="fm4npp", description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    diagnose = subparsers.add_parser("diagnose", help="Show PyTorch, CUDA, and Mamba status")
    diagnose.add_argument("--require-fused", action="store_true")
    diagnose.set_defaults(function=command_diagnose)

    download = subparsers.add_parser("download", help="Download a pinned Hugging Face artifact")
    download.add_argument("--model", choices=sorted(MODEL_FILES), default="m1")
    download.add_argument("--revision", default=HF_REVISION)
    download.add_argument("--cache-dir")
    download.add_argument("--local-dir")
    download.add_argument("--local-files-only", action="store_true")
    download.set_defaults(function=command_download)

    inspect = subparsers.add_parser("inspect", help="Inspect checkpoint-derived architecture")
    _add_model_arguments(inspect)
    inspect.set_defaults(function=command_inspect)

    infer = subparsers.add_parser("infer", help="Run one event through the pretrained backbone")
    _add_model_arguments(infer)
    infer.add_argument("--input", help=".npz/.npy event input; omit for a synthetic smoke test")
    infer.add_argument("--input-format", choices=("cartesian", "preprocessed"), default="cartesian")
    infer.add_argument("--event-index", type=int, default=0)
    infer.add_argument("--ordering", choices=("voxel", "radial"), default="voxel")
    infer.add_argument("--max-points", type=int)
    infer.add_argument("--sequence-length", type=int, default=128)
    infer.add_argument("--seed", type=int, default=42)
    infer.add_argument("--device", default="auto")
    infer.add_argument("--output", default="outputs/predictions.npz")
    infer.set_defaults(function=command_infer)

    benchmark = subparsers.add_parser("benchmark", help="Measure synthetic inference throughput")
    _add_model_arguments(benchmark)
    benchmark.add_argument("--device", default="auto")
    benchmark.add_argument("--batch-size", type=int, default=1)
    benchmark.add_argument("--sequence-length", type=int, default=512)
    benchmark.add_argument("--warmup", type=int, default=5)
    benchmark.add_argument("--steps", type=int, default=20)
    benchmark.add_argument("--seed", type=int, default=42)
    benchmark.add_argument("--compile", action="store_true")
    benchmark.add_argument("--output", help="Append a JSON record to this JSONL file")
    benchmark.set_defaults(function=command_benchmark)
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        return int(args.function(args))
    except (FileNotFoundError, IndexError, RuntimeError, ValueError) as exc:
        parser.exit(2, f"error: {exc}\n")
    return 2


if __name__ == "__main__":
    raise SystemExit(main())

