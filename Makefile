.PHONY: setup clean test lint format dev-deps lock

# Default target
.DEFAULT_GOAL := help

help:
	@echo "Available targets:"
	@echo "  setup     - Create new virtualenv and sync dependencies"
	@echo "  lock      - Update uv.lock file"
	@echo "  clean     - Remove build artifacts and virtual environment"
	@echo "  test      - Run tests"
	@echo "  lint      - Run linter"
	@echo "  format    - Format code"
	@echo "  dev-deps  - Set up pre-commit hooks"

setup: clean
	@echo "Creating fresh virtual environment..."
	uv venv
	uv sync
	@echo "Setup complete! Activate the virtual environment with: source .venv/bin/activate"

lock:
	@echo "Updating uv.lock file..."
	. .venv/bin/activate && uv lock

clean:
	@echo "Cleaning build artifacts..."
	rm -rf build/
	rm -rf dist/
	rm -rf *.egg-info/
	rm -rf .pytest_cache/
	rm -rf __pycache__/
	rm -rf src/**/__pycache__/
	@echo "Removing virtual environment..."
	rm -rf .venv/
	@echo "Clean complete!"

test:
	@echo "Running tests..."
	. .venv/bin/activate && python -m pytest tests/ -v

lint:
	@echo "Running linter..."
	. .venv/bin/activate && ruff check .

format:
	@echo "Formatting code..."
	. .venv/bin/activate && ruff format .

# Add dev dependencies - now this just installs pre-commit hooks
dev-deps:
	. .venv/bin/activate && \
	git config --unset-all core.hooksPath || true && \
	pre-commit install
