#!/usr/bin/env bash
set -euo pipefail

PYTHON_BIN="${PYTHON_BIN:-python3.10}"
ENV_DIR="${ENV_DIR:-.venv}"
TORCH_INDEX_URL="${TORCH_INDEX_URL:-https://download.pytorch.org/whl/cu121}"

"${PYTHON_BIN}" -m venv "${ENV_DIR}"
"${ENV_DIR}/bin/python" -m pip install --upgrade pip setuptools wheel packaging ninja
"${ENV_DIR}/bin/python" -m pip install torch==2.4.1 --index-url "${TORCH_INDEX_URL}"
"${ENV_DIR}/bin/python" -m pip install -e ".[dev]"
"${ENV_DIR}/bin/python" -m pip install "causal-conv1d>=1.4,<2" "mamba-ssm>=2.2.4,<3" --no-build-isolation

"${ENV_DIR}/bin/fm4npp" diagnose --require-fused

