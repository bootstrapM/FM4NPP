.PHONY: install install-gpu test diagnose

install:
	python -m pip install -e ".[dev]"

install-gpu:
	python -m pip install -e ".[dev,gpu]" --no-build-isolation

test:
	python -m pytest

diagnose:
	fm4npp diagnose

