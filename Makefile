# tcda_uq - common developer tasks.
# Uses uv if available; falls back to the active Python otherwise.
PY ?= python

.PHONY: help install install-dev test test-slow test-all reproduce reproduce-quick notebooks clean

help:
	@echo "Targets:"
	@echo "  install        editable install (core)"
	@echo "  install-dev    editable install with [dev] extra"
	@echo "  test           fast test suite (smoke + unit + API)"
	@echo "  test-slow      statistical coverage-property tests (pytest -m slow)"
	@echo "  test-all       fast + slow"
	@echo "  reproduce      regenerate the coverage/width table (scripts/reproduce_coverage.py)"
	@echo "  reproduce-quick  fast smoke of the reproduction script"
	@echo "  notebooks      execute the tutorial notebooks in place"
	@echo "  clean          remove caches and build artifacts"

install:
	uv pip install -e . 2>/dev/null || $(PY) -m pip install -e .

install-dev:
	uv pip install -e ".[dev]" 2>/dev/null || $(PY) -m pip install -e ".[dev]"

test:
	$(PY) -m pytest

test-slow:
	$(PY) -m pytest -m slow

test-all:
	$(PY) -m pytest -m "slow or not slow"

reproduce:
	$(PY) scripts/reproduce_coverage.py

reproduce-quick:
	$(PY) scripts/reproduce_coverage.py --quick

notebooks:
	$(PY) -m jupyter nbconvert --to notebook --execute --inplace notebooks/*.ipynb

clean:
	rm -rf .pytest_cache dist build *.egg-info
	find . -type d -name __pycache__ -prune -exec rm -rf {} +
