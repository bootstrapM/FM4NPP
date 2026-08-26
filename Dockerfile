FROM pytorch/pytorch:2.4.1-cuda12.1-cudnn9-devel

ENV PIP_NO_CACHE_DIR=1 \
    PYTHONUNBUFFERED=1

RUN apt-get update \
    && apt-get install -y --no-install-recommends build-essential ninja-build \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /workspace/fm4npp
COPY . .

RUN python -m pip install --upgrade pip setuptools wheel packaging ninja \
    && python -m pip install -e . \
    && python -m pip install "causal-conv1d>=1.4,<2" "mamba-ssm>=2.2.4,<3" --no-build-isolation

CMD ["fm4npp", "diagnose"]

