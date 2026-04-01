help:
	@echo ""
	@echo "Anonymizer Makefile"
	@echo "==================="
	@echo ""
	@echo "  bootstrap              - Install Python dependencies (dev group)"
	@echo "  install                - Install project dependencies with uv"
	@echo "  install-dev            - Install project with dev dependencies"
	@echo "  install-dev-notebooks  - Install dev + notebook dependencies"
	@echo "  install-pre-commit     - Install pre-commit hooks (run once after cloning)"
	@echo ""
	@echo "  format                 - Format and fix code"
	@echo "  format-check           - Check format and lint (read-only)"
	@echo "  typecheck              - Run type checks (advisory, non-blocking)"
	@echo "  copyright              - Add missing SPDX headers to source files"
	@echo "  copyright-check        - Check all source files have SPDX headers (read-only)"
	@echo "  check                  - Run all read-only checks"
	@echo "  lock-check             - Check uv.lock is up to date"
	@echo ""
	@echo "  test                   - Run all unit tests"
	@echo "  test-e2e               - Run end-to-end tests"
	@echo "  coverage               - Run tests with coverage report"
	@echo ""
	@echo "  build-wheel            - Build wheel (version from git tag)"
	@echo "  publish-pypi           - Publish wheel to PyPI"
	@echo ""
	@echo "  install-dev-docs       - Install dev + docs dependencies"
	@echo "  docs-serve             - Start docs dev server (live-reload)"
	@echo "  docs-build             - Build docs site (strict mode)"
	@echo ""
	@echo "  install-dev-docs       - Install dev + docs dependencies"
	@echo "  docs-serve             - Start docs dev server (live-reload)"
	@echo "  docs-build             - Build docs site (strict mode)"
	@echo ""
	@echo "  clean                  - Remove coverage reports and cache files"
	@echo "  clean-merged-branches  - Checkout main, fetch --prune, delete local branches merged into main"
	@echo ""

bootstrap:
	@echo "Installing Python dependencies (dev group)..."
	uv sync --group dev

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

format:
	@echo "Formatting and fixing code..."
	uv run tools/codestyle/format.sh

format-check:
	@echo "Checking format and lint (read-only)..."
	uv run tools/codestyle/format.sh --check
	uv run tools/codestyle/ruff_check.sh

typecheck:
	@echo "Running type checks (advisory -- see issue tracking full compliance)..."
	-uv run tools/codestyle/typecheck.sh

copyright:
	@echo "Adding missing SPDX headers..."
	uv run tools/codestyle/copyright_fixer.py .

copyright-check:
	@echo "Checking SPDX headers (read-only)..."
	uv run tools/codestyle/copyright_fixer.py --check .

check:
	@echo "Running all read-only checks..."
	$(MAKE) format-check typecheck lock-check copyright-check

lock-check:
	@echo "Checking uv.lock is up to date..."
	uv lock --check

test:
	@echo "Running unit tests..."
	uv run --group dev pytest

test-e2e:
	@echo "Running end-to-end tests..."
	uv run --group dev pytest -m e2e

coverage:
	@echo "Running tests with coverage analysis..."
	uv run --group dev pytest --cov=anonymizer --cov-report=term-missing --cov-report=html
	@echo "Coverage report generated in htmlcov/index.html"

build-wheel:
	@echo "Building wheel (version from git tag via uv-dynamic-versioning)..."
	rm -rf dist/
	uv build --wheel

publish-pypi: build-wheel
	@echo "Publishing wheel to PyPI..."
	uvx twine upload \
	  --username "$$TWINE_USERNAME" \
	  --password "$$TWINE_PASSWORD" \
	  --non-interactive \
	  dist/*.whl
	@echo "published: $$(ls dist/*.whl)"

install-dev-docs:
	@echo "Installing dev + docs dependencies..."
	uv sync --group dev --group docs

docs-serve:
	@echo "Starting docs dev server (live-reload)..."
	uv run --group docs mkdocs serve

docs-build:
	@echo "Building docs site..."
	uv run --group docs mkdocs build --strict

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

.PHONY: help bootstrap install install-dev install-dev-notebooks install-pre-commit format format-check typecheck copyright copyright-check check lock-check test test-e2e coverage build-wheel publish-pypi install-dev-docs docs-serve docs-build clean clean-pycache clean-merged-branches
