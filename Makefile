# Persian Voice Cloning Studio - common tasks
.DEFAULT_GOAL := help
PY := .venv/bin/python
PIP := .venv/bin/pip

.PHONY: help setup setup-full models app cli test lint format doctor clean clean-cache

help:  ## Show this help
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) \
		| awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-14s\033[0m %s\n", $$1, $$2}'

setup:  ## Create .venv and install core dependencies
	./scripts/setup_env.sh

setup-full:  ## Install core + all optional extras
	./scripts/setup_env.sh --full

models:  ## Download model weights for the configured engine
	$(PY) scripts/download_models.py

app:  ## Launch the Gradio interface
	$(PY) app.py

cli:  ## Show CLI help
	$(PY) -m voiceclone.cli --help

doctor:  ## Report installed engines and backends
	$(PY) -m voiceclone.cli doctor

test:  ## Run the test suite
	$(PY) -m pytest tests/ -q

lint:  ## Lint with ruff
	$(PY) -m ruff check src tests scripts app.py

format:  ## Auto-fix lint issues and format
	$(PY) -m ruff check --fix src tests scripts app.py
	$(PY) -m ruff format src tests scripts app.py

clean:  ## Remove caches and build artefacts
	find . -type d -name __pycache__ -prune -exec rm -rf {} +
	rm -rf .pytest_cache .ruff_cache .mypy_cache build dist *.egg-info

clean-cache:  ## Delete downloaded model weights (frees several GB)
	rm -rf ~/.cache/persian-voice-cloning
	@echo "Model cache removed. Re-run 'make models' to fetch it again."
