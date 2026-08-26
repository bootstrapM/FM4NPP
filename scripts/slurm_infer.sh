#!/usr/bin/env bash
#SBATCH --job-name=fm4npp-infer
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --time=00:30:00
#SBATCH --output=slurm-%j.out
#SBATCH --error=slurm-%j.err

set -euo pipefail

PROJECT_DIR="${PROJECT_DIR:-${SLURM_SUBMIT_DIR:-$(pwd)}}"
ENV_DIR="${ENV_DIR:-${PROJECT_DIR}/.venv}"

cd "${PROJECT_DIR}"
"${ENV_DIR}/bin/fm4npp" diagnose --require-fused
srun "${ENV_DIR}/bin/fm4npp" infer --device cuda --backend fused "$@"

