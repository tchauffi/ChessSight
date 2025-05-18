.PHONY: help install format lint test pre-commit clean

help:
	@echo "Available commands:"
	@echo "install    - Install project dependencies and pre-commit hooks"
	@echo "format     - Format code using black and ruff"
	@echo "lint       - Run all linters (ruff, mypy)"
	@echo "test       - Run tests"
	@echo "pre-commit - Run pre-commit hooks on all files"
	@echo "clean      - Remove cache files"

install:
	poetry install
	poetry run pre-commit install

format:
	poetry run black .
	poetry run ruff check --fix .

lint:
	poetry run ruff check .
	poetry run mypy .

test:
	poetry run pytest

pre-commit:
	poetry run pre-commit run --all-files

clean:
	find . -type d -name "__pycache__" -exec rm -rf {} +
	find . -type f -name "*.pyc" -delete
	find . -type f -name "*.pyo" -delete
	find . -type f -name "*.pyd" -delete
	find . -type f -name ".coverage" -delete
	find . -type d -name "*.egg-info" -exec rm -rf {} +
	find . -type d -name "*.egg" -exec rm -rf {} +
	find . -type d -name ".mypy_cache" -exec rm -rf {} +
	find . -type d -name ".pytest_cache" -exec rm -rf {} +
	find . -type d -name ".ruff_cache" -exec rm -rf {} +