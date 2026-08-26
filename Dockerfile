FROM pytorch/pytorch:2.13.0-cuda13.2-cudnn9-devel

ENV PIP_NO_CACHE_DIR=1 \
    PYTHONUNBUFFERED=1

RUN apt-get update \
    && apt-get install -y --no-install-recommends build-essential ninja-build \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /workspace/fm4npp
COPY . .

RUN python -m pip install --upgrade pip setuptools wheel packaging ninja \
    && python -m pip install -e . \
    && CAUSAL_CONV1D_FORCE_BUILD=TRUE MAX_JOBS=4 python -m pip install \
        --no-cache-dir --no-build-isolation "causal-conv1d==1.7.0" \
    && MAMBA_FORCE_BUILD=TRUE MAX_JOBS=4 python -m pip install \
        --no-cache-dir --no-build-isolation "mamba-ssm==2.3.2.post1"

CMD ["fm4npp", "diagnose"]
