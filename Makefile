.PHONY: help install build test format lint check fix clean version patch minor major release-version

help: ## Show available commands
	@awk 'BEGIN {FS = ":.*?## "} /^[a-zA-Z_-]+:.*?## / {printf "  \033[36m%-16s\033[0m %s\n", $$1, $$2}' $(MAKEFILE_LIST)

install: ## Install package in development mode
	uv sync --all-extras --dev

build: ## Build the package
	uv run python -m build

test: ## Run tests
	uv run pytest

format: ## Format code
	uv run ruff format dataprism/

lint: ## Run linters (ruff + mypy + bandit)
	uv run ruff check dataprism/
	uv run mypy dataprism/
	uv run bandit -r dataprism/ -c pyproject.toml

check: lint test ## Run all checks (lint + test)

fix: ## Auto-fix all fixable issues
	uv run ruff format dataprism/
	uv run ruff check --fix dataprism/

clean: ## Clean build artifacts and caches
	rm -rf build/ dist/ *.egg-info/
	rm -rf .pytest_cache/ .mypy_cache/ .ruff_cache/
	rm -rf htmlcov/ .coverage coverage.xml
	find . -type d -name __pycache__ -exec rm -rf {} +
	find . -type f -name "*.pyc" -delete

# Version management — read from pyproject.toml
version: ## Show current version
	@uv run python -c "import tomllib; print(tomllib.load(open('pyproject.toml', 'rb'))['project']['version'])"

release-version:
	@if [ -z "$(VERSION)" ]; then echo "VERSION is required"; exit 1; fi
	@echo "Creating release $(VERSION)..."
	@sed -i.bak 's/^version = "[^"]*"/version = "$(VERSION)"/' pyproject.toml && rm pyproject.toml.bak
	@sed -i.bak 's/__version__ = "[^"]*"/__version__ = "$(VERSION)"/' dataprism/__init__.py && rm dataprism/__init__.py.bak
	@uv lock
	@git add pyproject.toml dataprism/__init__.py uv.lock
	@git commit -m "Bump version to $(VERSION)"
	@git tag v$(VERSION)
	@echo "Run 'git push --tags' to publish"

patch: ## Bump patch version and create tag
	@current=$$(uv run python -c "import tomllib; print(tomllib.load(open('pyproject.toml', 'rb'))['project']['version'])"); \
	new=$$(uv run python -c "v='$$current'.split('.'); v[2]=str(int(v[2])+1); print('.'.join(v))"); \
	echo "Current version: $$current"; \
	$(MAKE) release-version VERSION=$$new

minor: ## Bump minor version and create tag
	@current=$$(uv run python -c "import tomllib; print(tomllib.load(open('pyproject.toml', 'rb'))['project']['version'])"); \
	new=$$(uv run python -c "v='$$current'.split('.'); v[1]=str(int(v[1])+1); v[2]='0'; print('.'.join(v))"); \
	echo "Current version: $$current"; \
	$(MAKE) release-version VERSION=$$new

major: ## Bump major version and create tag
	@current=$$(uv run python -c "import tomllib; print(tomllib.load(open('pyproject.toml', 'rb'))['project']['version'])"); \
	new=$$(uv run python -c "v='$$current'.split('.'); v[0]=str(int(v[0])+1); v[1]='0'; v[2]='0'; print('.'.join(v))"); \
	echo "Current version: $$current"; \
	$(MAKE) release-version VERSION=$$new
