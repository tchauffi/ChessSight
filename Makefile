.PHONY: help install format lint test test-integration pre-commit clean \
        doctor synth-smoke synth-sheet site site-refresh

# Where generated datasets go. Override with `make synth-smoke RUN=/path/to/run`.
RUN ?= $(HOME)/datasets/chesssight/smoke
N ?= 16

help:
	@echo "Available commands:"
	@echo "install          - Install project dependencies and pre-commit hooks"
	@echo "format           - Format code using black and ruff"
	@echo "lint             - Run all linters (ruff, mypy)"
	@echo "test             - Run unit tests (no Blender required)"
	@echo "test-integration - Run tests that launch Blender"
	@echo "doctor           - Check Blender, GPU and output directory"
	@echo "synth-smoke      - Render a small dataset and verify it"
	@echo "synth-sheet      - Render a dataset and write an annotated contact sheet"
	@echo "site             - Rebuild docs/index.html from the committed data"
	@echo "site-refresh     - Re-measure the pipeline and redraw the site figures"
	@echo "pre-commit       - Run pre-commit hooks on all files"
	@echo "clean            - Remove cache files"

doctor:
	uv run chesssight doctor

synth-smoke:
	uv run chesssight synth run -c configs/smoke.yaml -n $(N) -o $(RUN)
	uv run chesssight synth verify $(RUN)
	uv run chesssight qa stats $(RUN)

synth-sheet:
	uv run chesssight synth run -c configs/smoke.yaml -n $(N) -o $(RUN)
	uv run chesssight qa overlay $(RUN) -n 8 --sheet $(RUN)/sheet.png
	@echo "wrote $(RUN)/sheet.png"

test-integration:
	uv run pytest tests/integration -v

# The published page. `site` needs nothing but the repository; `site-refresh`
# re-runs both models over a real split, so it needs the checkpoints.
DETECTOR ?= $(HOME)/runs/rtdetr_v4/best
CORNERS  ?= $(HOME)/runs/corner_swin_v2/best
CHESSRED ?= $(HOME)/datasets/chesssight/chessred

site:
	uv run python scripts/build_site.py

site-refresh:
	uv run python scripts/build_site.py --refresh \
	  --detector $(DETECTOR) --corners $(CORNERS) --data $(CHESSRED)

install:
	uv sync
	uv run pre-commit install

format:
	uv run black .
	uv run ruff check --fix .

lint:
	uv run ruff check .
	uv run mypy .

test:
	uv run pytest

pre-commit:
	uv run pre-commit run --all-files

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
