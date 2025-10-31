.PHONY: help setup setup-gpu setup-cpu dev test clean validate flow lint format

help:
	@echo "Telemetry Processing Pipeline - Make targets:"
	@echo "  setup-gpu    - Install dependencies with GPU support (NVIDIA CUDA required)"
	@echo "  setup-cpu    - Install dependencies for CPU-only mode"
	@echo "  setup        - Alias for setup-gpu"
	@echo "  dev          - Install development dependencies"
	@echo "  test         - Run test suite"
	@echo "  validate     - Run Great Expectations validation"
	@echo "  flow         - Run end-to-end Prefect pipeline"
	@echo "  lint         - Run ruff linter"
	@echo "  format       - Format code with black"
	@echo "  clean        - Remove generated files and caches"

setup: setup-gpu

setup-gpu:
	pip install -e ".[gpu]"
	@echo "GPU-accelerated setup complete. Ensure CUDA 11+ is installed."

setup-cpu:
	pip install -e .
	@echo "CPU-only setup complete."

dev:
	pip install -e ".[dev]"

test:
	pytest tests/ -v

validate:
	python -m src.validation.run_validation

flow:
	python flows/e2e_pipeline.py

lint:
	ruff check src/ tests/ flows/

format:
	black src/ tests/ flows/

clean:
	find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name "*.egg-info" -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name ".pytest_cache" -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name ".ruff_cache" -exec rm -rf {} + 2>/dev/null || true
	rm -rf htmlcov/ .coverage
	@echo "Cleaned up generated files and caches."
