# FM4NPP inference starter

A small, reproducible inference repository for the proton-proton collision
foundation model from **FM4NPP: A Scaling Foundation Model for Nuclear and
Particle Physics**.

This project loads the real checkpoints from
[FM4NPP/PP_collision on Hugging Face](https://huggingface.co/FM4NPP/PP_collision),
derives the architecture from their tensor shapes, and runs the published
Mamba2 backbone on local machines, GPU servers, or Slurm clusters. Model
weights and datasets stay outside Git.

> Important: this is a point-sequence model, not a language model. One token is
> one TPC spacepoint. Input is `[batch, points, 4]` in normalized
> `(E, eta, phi, radius)` form. The pretrained head predicts the normalized 3D
> coordinates of 30 next-nearest neighbors for every point. It does **not**
> directly emit track IDs, PID labels, or text.

## What is ready

- Pinned downloads from Hugging Face revision
  `8aeb7f86a3eb4ed07876e8527d55fc98f5003254`.
- Safe loading of the published pickle checkpoints with
  `torch.load(weights_only=True)`; unsafe pickle fallback is never used.
- Strict checkpoint-compatible model construction, including automatic width,
  depth, state-size, head-size, and output-size inference.
- Exact fused Mamba2 inference on NVIDIA GPUs through the official
  `mamba-ssm` SSD kernel.
- A slower exact PyTorch recurrence for CPU/MPS smoke tests and debugging.
- TPCpp-10M `.npz` event loading, Cartesian-to-polar normalization, and
  hierarchical voxel serialization.
- Local, Docker, and Slurm entry points.
- A basic latency/throughput benchmark that writes scaling-friendly JSONL.

## Fastest start on an NVIDIA GPU

Requirements: Linux, Python 3.10 or 3.11, an NVIDIA GPU, a working compiler,
and CUDA 12.1-compatible drivers.

```bash
git clone <your-new-repository-url>
cd <your-new-repository>

bash scripts/setup_gpu_env.sh
source .venv/bin/activate

# Downloads the 59 MB artifact on first use and runs synthetic inference.
fm4npp infer \
  --model m1 \
  --device cuda \
  --backend fused \
  --sequence-length 128 \
  --output outputs/smoke.npz
```

The setup script ends with `fm4npp diagnose --require-fused`, so a CUDA or
native-extension problem is caught before the first experiment.

The script automatically searches for `python3.12`, `python3.11`,
`python3.10`, and then compatible `python3`/`python` commands. On module-based
clusters, load Python first or specify its path explicitly:

```bash
module load python/3.11                # module name varies by site
PYTHON_BIN="$(command -v python3)" bash scripts/setup_gpu_env.sh
```

To download explicitly—useful before moving to an offline cluster:

```bash
fm4npp download --model m1 --local-dir checkpoints
fm4npp inspect --checkpoint checkpoints/pp_nerf_m1_k30.ckpt
```

## CPU or Apple Silicon smoke test

The reference backend uses the same selective state-space recurrence but loops
over sequence positions. It is intended for correctness checks and short
events, not throughput measurements.

```bash
python3 -m venv .venv  # use any installed Python 3.10--3.12
source .venv/bin/activate
python -m pip install -e ".[dev]"

fm4npp infer \
  --model m1 \
  --backend reference \
  --sequence-length 32 \
  --output outputs/cpu-smoke.npz
```

## Run a real TPCpp-10M event

The public 7,000-event labeled test subset is about 78 MB compressed:

```bash
mkdir -p data
curl -L --fail \
  -o data/TPCpp-10M_labeled_test.zip \
  https://zenodo.org/api/records/16970029/files/TPCpp-10M_labeled_test.zip/content
unzip data/TPCpp-10M_labeled_test.zip -d data/TPCpp-10M_labeled_test

fm4npp infer \
  --model m1 \
  --input data/TPCpp-10M_labeled_test/labeled/test/spacepoints.npz \
  --input-format cartesian \
  --event-index 0 \
  --device cuda \
  --backend fused \
  --output outputs/event-0000.npz
```

For a quick debug run, add `--max-points 128`. Truncation is recorded in the
output metadata and should not be used for scientific evaluation.

The result archive contains:

| Key | Shape | Meaning |
| --- | --- | --- |
| `predictions_normalized` | `[N, 30, 3]` | Predicted normalized `(eta, phi, r)` neighbors |
| `predictions_eta_phi_r` | `[N, 30, 3]` | Predictions converted back to physical coordinates |
| `input_preprocessed` | `[N, 4]` | Serialized model input |
| `original_indices` | `[N]` | Map from sequence positions back to the raw event |
| `metadata` | scalar JSON string | Checkpoint, model, device, timing, and preprocessing provenance |

If you already have normalized and serialized `(E, eta, phi, r)` inputs, use
`--input-format preprocessed`; no coordinate conversion or reordering will be
applied.

### Preprocessing caveat

The upstream publication code refers to
`bin_edges_v3_nbins_8_8_6.pkl`, but that file is not in the GitHub or Hugging
Face repositories. This project includes clearly labeled approximate 8×8×6
bin edges computed from all 6,013,910 points in the public labeled test split.
Energy and coordinate normalization constants come directly from the upstream
loader. This is suitable for getting the model running, but exact paper-metric
reproduction requires the authors' original training-time bin file. Every raw
event output carries this warning in its metadata.

## Model artifacts

The Hugging Face repository currently exposes these full training checkpoints:

| CLI name | File | Published size |
| --- | --- | ---: |
| `m1` | `pp_nerf_m1_k30.ckpt` | 59.1 MB |
| `m3` | `pp_nerf_m3_k30.ckpt` | 234 MB |
| `m4` | `pp_nerf_m4_k30.ckpt` | 933 MB |
| `m5` | `pp_nerf_m5_k30.ckpt` | 2.1 GB |

Artifact names do not line up cleanly with the paper's m1–m6 scale labels, so
the code never guesses architecture from a filename. For the smallest artifact
we verified 12 layers, width 256, state size 16, 4.92M model tensors, 30
neighbors, and a 100,000-step checkpoint. Run `fm4npp inspect` after downloading
any other artifact to see its derived configuration.

These files also contain AdamW and scheduler state. `mmap=True` prevents eager
optimizer reads, but the first load of a large artifact still needs enough CPU
address space for the checkpoint and enough accelerator memory for the model.

## Python API

```python
import torch

from fm4npp_inference import load_model

loaded = load_model(
    "m1",
    device="cuda",
    dtype=torch.float16,
    backend="fused",
)

# Already normalized and serialized: [batch, points, (E, eta, phi, r)]
points = torch.randn(1, 512, 4, device="cuda", dtype=torch.float16)
with torch.inference_mode():
    output = loaded.model(points)

neighbors = output.reshape(1, 512, loaded.config.k_neighbors, 3)
print(loaded.config)
print(neighbors.shape)
```

See [examples/python_api.py](examples/python_api.py) for a runnable version.

## Docker server

```bash
docker build -t fm4npp-inference .

docker run --rm --gpus all \
  -v fm4npp-hf-cache:/cache \
  -v "$(pwd)/outputs:/workspace/fm4npp/outputs" \
  -e HF_HOME=/cache \
  fm4npp-inference \
  fm4npp infer --model m1 --device cuda --backend fused \
    --output outputs/docker-smoke.npz
```

Mount a data directory and pass `--input` for real events.

## Slurm cluster

Create the environment once on a login or build node, adjusting modules and the
CUDA wheel index for the cluster if needed:

```bash
bash scripts/setup_gpu_env.sh
source .venv/bin/activate
fm4npp download --model m1 --local-dir checkpoints
```

Submit inference arguments directly after the script name:

```bash
sbatch scripts/slurm_infer.sh \
  --model m1 \
  --checkpoint checkpoints/pp_nerf_m1_k30.ckpt \
  --input /shared/TPCpp-10M/labeled/test/spacepoints.npz \
  --event-index 0 \
  --output outputs/slurm-event-0000.npz
```

Set `PROJECT_DIR` or `ENV_DIR` with `sbatch --export=...` if the repository or
virtual environment is not under the submission directory. Add your site's
account, partition, and module commands to `scripts/slurm_infer.sh`.

## First inference-scaling measurements

The benchmark deliberately records raw measurements rather than hiding them in
a dashboard:

```bash
fm4npp benchmark \
  --model m1 \
  --device cuda \
  --backend fused \
  --dtype fp16 \
  --batch-size 1 \
  --sequence-length 512 \
  --warmup 10 \
  --steps 50 \
  --output benchmarks/runs.jsonl
```

Each JSONL record includes checkpoint/config identity, parameter count, batch
and sequence sizes, mean/p50/p95 latency, events/s, points/s, and peak CUDA
memory. This gives a clean base for later sweeps over model size, sequence
length, batch size, precision, compilation, and GPU type.

## Validation and troubleshooting

```bash
python -m pytest
fm4npp diagnose
```

- If fused imports fail, ensure `torch` is installed before `mamba-ssm` and use
  `--no-build-isolation`, as done by `scripts/setup_gpu_env.sh`.
- A CUDA driver can support a newer runtime than the locally installed toolkit;
  check both `nvidia-smi` and the `torch_cuda` field from `fm4npp diagnose`.
- `--backend auto` selects fused Mamba only when it is importable and the input
  is on CUDA. Use `--backend fused` in performance jobs so a missing extension
  fails instead of silently timing the reference path.
- Keep Hugging Face caches and `.npz` outputs on local scratch or a high-throughput
  shared filesystem for cluster jobs.

## Scope and provenance

The upstream [GitHub repository](https://github.com/FM4NPP/PP_collision) is a
publication-oriented training/downstream code release. This repository is an
independent, inference-first harness; it does not vendor their training stack,
weights, or dataset. The architecture was checked against upstream commit
`2cd116b493aba86faed0164bbb6c140b35bd512d` and the official Mamba2 equations.
See [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md) for licensing provenance.

The Hugging Face release contains pretrained backbone checkpoints but no
published task-adapter checkpoint. Track finding, particle identification, and
noise tagging therefore require training or obtaining the corresponding
adapter; backbone inference alone cannot reproduce the downstream metrics.

## References

- [FM4NPP paper on OpenReview](https://openreview.net/forum?id=qaI3cLFsiX)
- [FM4NPP paper on arXiv](https://arxiv.org/abs/2508.14087)
- [FM4NPP publication code](https://github.com/FM4NPP/PP_collision)
- [FM4NPP model checkpoints](https://huggingface.co/FM4NPP/PP_collision)
- [TPCpp-10M dataset](https://zenodo.org/records/16970029)

Please cite both the FM4NPP and TPCpp-10M papers when using the model and data.
