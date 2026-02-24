help:
	@echo ""
	@echo "Anonymizer Makefile"
	@echo "==================="
	@echo ""
	@echo "  install                - Install project dependencies with uv"
	@echo "  install-dev            - Install project with dev dependencies"
	@echo "  install-dev-notebooks  - Install dev + notebook dependencies"
	@echo "  install-pre-commit     - Install pre-commit hooks (run once after cloning)"
	@echo ""
	@echo "  test                   - Run all unit tests"
	@echo "  coverage               - Run tests with coverage report"
	@echo ""
	@echo "  format                 - Format code with ruff"
	@echo "  format-check           - Check code formatting without changes"
	@echo "  lint                   - Lint with ruff"
	@echo "  lint-fix               - Lint with ruff and apply fixes"
	@echo "  check-all              - Run all checks (format-check + lint)"
	@echo "  check-all-fix          - Run all checks with autofix (format + lint-fix)"
	@echo ""
	@echo "  clean                  - Remove coverage reports and cache files"
	@echo "  clean-merged-branches  - Checkout main, fetch --prune, delete local branches merged into main"
	@echo ""

install:
	@echo "Installing project dependencies..."
	uv sync
	@echo "Done!"

install-dev:
	@echo "Installing project with dev dependencies..."
	uv sync --group dev
	@echo "Done!"

install-dev-notebooks:
	@echo "Installing project with dev + notebook dependencies..."
	uv sync --group dev --group notebooks
	@echo "Done!"

install-pre-commit:
	@echo "Installing pre-commit hooks..."
	uv run pre-commit install
	@echo "Done! Hooks will run on git commit."

test:
	@echo "Running unit tests..."
	uv run --group dev pytest

coverage:
	@echo "Running tests with coverage analysis..."
	uv run --group dev pytest --cov=anonymizer --cov-report=term-missing --cov-report=html
	@echo "Coverage report generated in htmlcov/index.html"

format:
	@echo "Formatting code with ruff..."
	uv run ruff format .

format-check:
	@echo "Checking code formatting with ruff..."
	uv run ruff format --check .

lint:
	@echo "Linting code with ruff..."
	uv run ruff check .

lint-fix:
	@echo "Fixing linting issues with ruff..."
	uv run ruff check --fix .

check-all: format-check lint

check-all-fix: format lint-fix

clean-pycache:
	@echo "Cleaning Python cache files..."
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
	find . -type f -name "*.pyc" -delete 2>/dev/null || true

clean: clean-pycache
	@echo "Cleaning coverage reports and test cache..."
	rm -rf htmlcov .coverage .coverage.* .pytest_cache

clean-merged-branches:
	@echo "Cleaning merged local branches..."
	git checkout main && git fetch --prune && git branch --merged | grep -v '^\*\|main' | xargs -n 1 git branch -d || true
	@echo "Done!"

.PHONY: help install install-dev install-dev-notebooks install-pre-commit test coverage format format-check lint lint-fix check-all check-all-fix clean clean-pycache clean-merged-branches
