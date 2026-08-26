#!/usr/bin/env bash
set -euo pipefail

ENV_DIR="${ENV_DIR:-.venv}"
TORCH_INDEX_URL="${TORCH_INDEX_URL:-https://download.pytorch.org/whl/cu121}"

select_python() {
    local candidates=()
    local candidate

    if [[ -n "${PYTHON_BIN:-}" ]]; then
        candidates+=("${PYTHON_BIN}")
    else
        candidates+=(python3.12 python3.11 python3.10 python3 python)
    fi

    for candidate in "${candidates[@]}"; do
        if command -v "${candidate}" >/dev/null 2>&1 \
            && "${candidate}" -c 'import sys; raise SystemExit(0 if (3, 10) <= sys.version_info[:2] < (3, 13) else 1)' \
                >/dev/null 2>&1; then
            command -v "${candidate}"
            return 0
        fi
    done
    return 1
}

if ! PYTHON_BIN="$(select_python)"; then
    cat >&2 <<'EOF'
ERROR: No compatible Python interpreter was found.

FM4NPP GPU setup requires Python 3.10, 3.11, or 3.12. On a cluster, try:

  module avail python
  module load python/3.11
  bash scripts/setup_gpu_env.sh

Or point to an installed interpreter explicitly:

  PYTHON_BIN=/path/to/python3.11 bash scripts/setup_gpu_env.sh

With Conda:

  conda create -n fm4npp python=3.11
  conda activate fm4npp
  PYTHON_BIN="$(command -v python)" bash scripts/setup_gpu_env.sh
EOF
    exit 1
fi

echo "Using ${PYTHON_BIN} ($("${PYTHON_BIN}" --version 2>&1))"
"${PYTHON_BIN}" -m venv "${ENV_DIR}"
"${ENV_DIR}/bin/python" -m pip install --upgrade pip setuptools wheel packaging ninja
"${ENV_DIR}/bin/python" -m pip install torch==2.4.1 --index-url "${TORCH_INDEX_URL}"
"${ENV_DIR}/bin/python" -m pip install -e ".[dev]"
"${ENV_DIR}/bin/python" -m pip install "causal-conv1d>=1.4,<2" "mamba-ssm>=2.2.4,<3" --no-build-isolation

"${ENV_DIR}/bin/fm4npp" diagnose --require-fused
