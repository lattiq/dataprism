.PHONY: help build test clean install release release-patch release-minor release-major

# Default target
help:
	@echo "Available targets:"
	@echo "  build          - Build the package"
	@echo "  test           - Run tests"
	@echo "  clean          - Clean build artifacts"
	@echo "  install        - Install package in development mode"
	@echo "  release        - Create release (prompts for version)"
	@echo "  release-patch  - Create patch release (0.0.1 -> 0.0.2)"
	@echo "  release-minor  - Create minor release (0.0.1 -> 0.1.0)"
	@echo "  release-major  - Create major release (0.0.1 -> 1.0.0)"

# Build the package
build:
	python -m build

# Run tests
test:
	python -c "import dataprism; print(dataprism.hello())"

# Clean build artifacts
clean:
	rm -rf build/
	rm -rf dist/
	rm -rf *.egg-info/
	find . -type d -name __pycache__ -exec rm -rf {} +
	find . -type f -name "*.pyc" -delete

# Install in development mode
install:
	pip install -e .

# Get current version from pyproject.toml
get-version:
	@python -c "import tomllib; print(tomllib.load(open('pyproject.toml', 'rb'))['project']['version'])"

# Create a release (interactive)
release:
	@echo "Current version: $$(make get-version)"
	@read -p "Enter new version: " version; \
	if [ -n "$$version" ]; then \
		make release-version VERSION=$$version; \
	else \
		echo "Version cannot be empty"; exit 1; \
	fi

# Create release with specific version
release-version:
	@if [ -z "$(VERSION)" ]; then echo "VERSION is required"; exit 1; fi
	@echo "Creating release $(VERSION)..."
	@sed -i.bak 's/^version = "[^"]*"/version = "$(VERSION)"/' pyproject.toml && rm pyproject.toml.bak
	@sed -i.bak 's/__version__ = "[^"]*"/__version__ = "$(VERSION)"/' dataprism/__init__.py && rm dataprism/__init__.py.bak
	@git add pyproject.toml dataprism/__init__.py
	@git commit -m "Bump version to $(VERSION)"
	@git tag v$(VERSION)
	@echo "Release $(VERSION) created locally."
	@echo "Run 'git push origin main && git push origin v$(VERSION)' to publish"
	@echo "Or run 'gh release create v$(VERSION) --generate-notes' to create GitHub release"

# Patch release (0.0.1 -> 0.0.2)
release-patch:
	@current=$$(make get-version); \
	new=$$(python -c "v='$$current'.split('.'); v[2]=str(int(v[2])+1); print('.'.join(v))"); \
	make release-version VERSION=$$new

# Minor release (0.0.1 -> 0.1.0)
release-minor:
	@current=$$(make get-version); \
	new=$$(python -c "v='$$current'.split('.'); v[1]=str(int(v[1])+1); v[2]='0'; print('.'.join(v))"); \
	make release-version VERSION=$$new

# Major release (0.0.1 -> 1.0.0)
release-major:
	@current=$$(make get-version); \
	new=$$(python -c "v='$$current'.split('.'); v[0]=str(int(v[0])+1); v[1]='0'; v[2]='0'; print('.'.join(v))"); \
	make release-version VERSION=$$new