#!/usr/bin/env bash
set -euo pipefail

ENV_DIR="${ENV_DIR:-.venv}"
MAX_JOBS="${MAX_JOBS:-4}"
CAUSAL_CONV1D_VERSION="${CAUSAL_CONV1D_VERSION:-1.7.0}"
MAMBA_SSM_VERSION="${MAMBA_SSM_VERSION:-2.3.2.post1}"
PYTHON="${ENV_DIR}/bin/python"

if [[ ! -x "${PYTHON}" ]]; then
    echo "ERROR: ${PYTHON} does not exist. Run scripts/setup_gpu_env.sh first." >&2
    exit 1
fi

if ! command -v nvcc >/dev/null 2>&1; then
    cat >&2 <<'EOF'
ERROR: nvcc was not found. The fused GPU dependencies must be compiled against
the active PyTorch installation. Load your cluster's matching CUDA toolkit
module (or use a CUDA devel container), then rerun this script.
EOF
    exit 1
fi

"${PYTHON}" - <<'PY'
import re
import subprocess

import torch

if torch.version.cuda is None:
    raise SystemExit("ERROR: The active PyTorch build has no CUDA runtime.")

nvcc_output = subprocess.check_output(
    ["nvcc", "--version"], text=True, stderr=subprocess.STDOUT
)
match = re.search(r"release\s+(\d+)\.(\d+)", nvcc_output)
if match is None:
    raise SystemExit("ERROR: Could not determine the CUDA version from nvcc --version.")

nvcc_cuda = tuple(map(int, match.groups()))
torch_cuda = tuple(map(int, torch.version.cuda.split(".")[:2]))
if nvcc_cuda[0] != torch_cuda[0]:
    raise SystemExit(
        "ERROR: CUDA toolkit/PyTorch major-version mismatch:\n"
        f"  nvcc CUDA:  {nvcc_cuda[0]}.{nvcc_cuda[1]}\n"
        f"  torch CUDA: {torch.version.cuda}\n\n"
        "For a Blackwell GPU, install a CUDA 13 PyTorch wheel, for example:\n"
        "  python -m pip install --upgrade --force-reinstall torch==2.13.0 "
        "--index-url https://download.pytorch.org/whl/cu132\n"
        "Then rerun this script."
    )

devices = []
if torch.cuda.is_available():
    for index in range(torch.cuda.device_count()):
        props = torch.cuda.get_device_properties(index)
        devices.append(f"{props.name} (sm_{props.major}{props.minor})")
        if props.major >= 12 and torch_cuda < (12, 8):
            raise SystemExit(
                "ERROR: This Blackwell GPU requires a newer PyTorch/CUDA build; "
                f"the active PyTorch was compiled for CUDA {torch.version.cuda}. "
                "Install the official PyTorch CUDA 13.2 wheel."
            )

print(
    f"Building extensions for torch={torch.__version__}, "
    f"torch_cuda={torch.version.cuda}, nvcc_cuda={nvcc_cuda[0]}.{nvcc_cuda[1]}"
)
if devices:
    print("Visible GPUs: " + ", ".join(devices))
if nvcc_cuda != torch_cuda:
    print("NOTE: The CUDA minor versions differ; PyTorch permits this with a warning.")
PY
nvcc --version

# Native PyTorch extensions are ABI-coupled to the exact PyTorch installation.
# Remove any cached/stale binaries before compiling against the active stack.
"${PYTHON}" -m pip uninstall -y causal-conv1d mamba-ssm

echo "Building causal-conv1d ${CAUSAL_CONV1D_VERSION} (this can take several minutes)"
CAUSAL_CONV1D_FORCE_BUILD=TRUE MAX_JOBS="${MAX_JOBS}" \
    "${PYTHON}" -m pip install --verbose --no-cache-dir --no-build-isolation \
    "causal-conv1d==${CAUSAL_CONV1D_VERSION}"

echo "Building mamba-ssm ${MAMBA_SSM_VERSION} (this can take several minutes)"
MAMBA_FORCE_BUILD=TRUE MAX_JOBS="${MAX_JOBS}" \
    "${PYTHON}" -m pip install --verbose --no-cache-dir --no-build-isolation \
    "mamba-ssm==${MAMBA_SSM_VERSION}"

"${ENV_DIR}/bin/fm4npp" diagnose --require-fused
