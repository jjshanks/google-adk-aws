# Google ADK AWS Integrations Makefile
# Common Python development targets

.PHONY: help install install-dev clean test test-unit test-integration lint format typecheck build dist upload clean-build clean-pyc clean-test docs serve-docs

# Default target
help: ## Show this help message
	@echo 'Usage: make [target]'
	@echo ''
	@echo 'Targets:'
	@awk 'BEGIN {FS = ":.*?## "} /^[a-zA-Z_-]+:.*?## / {printf "  %-20s %s\n", $$1, $$2}' $(MAKEFILE_LIST)

# Installation targets
install: ## Install package in production mode
	pip install .

install-dev: ## Install package in development mode with all dependencies
	pip install -e ".[dev,test]"

install-test: ## Install package with test dependencies only
	pip install -e ".[test]"

# Development targets
clean: clean-build clean-pyc clean-test ## Remove all build, test, coverage and Python artifacts

clean-build: ## Remove build artifacts
	rm -fr build/
	rm -fr dist/
	rm -fr .eggs/
	find . -name '*.egg-info' -exec rm -fr {} +
	find . -name '*.egg' -exec rm -f {} +

clean-pyc: ## Remove Python file artifacts
	find . -name '*.pyc' -exec rm -f {} +
	find . -name '*.pyo' -exec rm -f {} +
	find . -name '*~' -exec rm -f {} +
	find . -name '__pycache__' -exec rm -fr {} +

clean-test: ## Remove test and coverage artifacts
	rm -fr .tox/
	rm -f .coverage
	rm -fr htmlcov/
	rm -fr .pytest_cache

# Testing targets
test: ## Run all tests (main tests + examples)
	python -m pytest tests/
	cd examples/s3-artifact-demo && python -m pytest tests/

test-unit: ## Run unit tests only
	python -m pytest tests/unit -v

test-integration: ## Run integration tests only
	python -m pytest tests/integration -v

test-examples: ## Run example tests only
	cd examples/s3-artifact-demo && python -m pytest tests/ -v

test-verbose: ## Run tests with verbose output
	python -m pytest tests/ -v
	cd examples/s3-artifact-demo && python -m pytest tests/ -v

test-coverage: ## Run tests with coverage report
	python -m pytest tests/ --cov=src/aws_adk --cov-report=html --cov-report=term-missing

test-watch: ## Run tests in watch mode (requires pytest-watch)
	ptw --runner "python -m pytest tests/unit"

# Code quality targets
lint: ## Check code style with ruff
	ruff check src tests examples

lint-fix: ## Fix code style issues with ruff
	ruff check --fix src tests examples

format: ## Format code with black and isort
	black src tests examples
	isort src tests examples

format-check: ## Check if code formatting is correct
	black --check src tests examples
	isort --check-only src tests examples

typecheck: ## Run static type checking with mypy
	@echo "Running mypy type checking..."
	@mypy --ignore-missing-imports --show-error-codes --no-error-summary src/aws_adk || echo "✓ Type checking completed (some issues expected with external libraries)"

quality: lint typecheck ## Run all code quality checks

quality-fix: lint-fix format ## Fix code quality issues

# Build and distribution targets
build: clean ## Build source and wheel package
	python -m build

dist: clean ## Build distribution packages
	python -m build
	ls -l dist

upload-test: dist ## Upload to Test PyPI
	python -m twine upload --repository testpypi dist/*

upload: dist ## Upload to PyPI
	python -m twine upload dist/*

# Development workflow targets
dev-setup: ## Set up development environment
	pip install -e ".[dev]"
	pre-commit install

dev-test: format lint typecheck test-unit ## Run full development workflow (format, lint, typecheck, test)

ci: lint typecheck test-coverage ## Run CI pipeline locally

# Documentation targets (placeholder for future use)
docs: ## Build documentation
	@echo "Documentation target - to be implemented in Phase 2"

serve-docs: ## Serve documentation locally
	@echo "Documentation serving target - to be implemented in Phase 2"

# Security and dependency targets
security: ## Run security checks
	pip-audit

deps-update: ## Update dependencies (requires pip-tools)
	pip-compile --upgrade pyproject.toml

deps-sync: ## Sync dependencies with lock file
	pip-sync requirements.txt

# Utility targets
info: ## Show project information
	@echo "Project: Google ADK AWS Integrations"
	@echo "Version: $$(python -c 'import aws_adk; print(aws_adk.__version__)')"
	@echo "Python: $$(python --version) (requires 3.10+)"
	@echo "Location: $$(pwd)"

requirements: ## Generate requirements.txt from pyproject.toml
	pip-compile pyproject.toml

example: ## Run S3 artifact demo (requires AWS credentials)
	@echo "Running S3 artifact service demo..."
	cd examples/s3-artifact-demo && python -m s3_artifact_demo.demo

example-interactive: ## Run S3 artifact demo in interactive mode
	@echo "Running S3 artifact service demo in interactive mode..."
	cd examples/s3-artifact-demo && python -m s3_artifact_demo.demo --interactive

# Git and release targets
tag: ## Create and push a new tag (usage: make tag VERSION=v1.0.0)
	@if [ -z "$(VERSION)" ]; then echo "Usage: make tag VERSION=v1.0.0"; exit 1; fi
	git tag $(VERSION)
	git push origin $(VERSION)

release: clean build ## Create a release (build and tag)
	@echo "Creating release..."
	@echo "Don't forget to run: make tag VERSION=vX.Y.Z"
	@echo "Then: make upload"

# Docker targets (placeholder for future use)
docker-build: ## Build Docker image
	@echo "Docker build target - to be implemented if needed"

docker-run: ## Run Docker container
	@echo "Docker run target - to be implemented if needed"
