PY ?= python3
DATA ?= ../kairos-data/data

.PHONY: help bootstrap test lint format clean demo equity multi-asset execution research
help:
	@echo "targets: bootstrap test lint format clean demo equity multi-asset execution research"

bootstrap:
	$(PY) -m venv .venv
	. .venv/bin/activate && $(PY) -m pip install -U pip && pip install -e ".[dev]"

test:
	$(PY) -m pytest -q

lint:
	@if command -v ruff >/dev/null 2>&1; then ruff check .; else echo "pip install ruff 后再 lint"; fi

format:
	@if command -v ruff >/dev/null 2>&1; then ruff format .; else echo "pip install ruff 后再 format"; fi

# 三条端到端流水线（默认读 ../kairos-data/data 下的真实行情）
equity:
	$(PY) examples/run_e2e_equity.py --data-dir $(DATA)/ashare

multi-asset:
	$(PY) examples/run_e2e_multi_asset.py --data-dir $(DATA)/etf

execution:
	$(PY) examples/run_execution_demo.py --data-dir $(DATA)/ashare

demo: equity multi-asset execution

research: demo
	@echo "产物见 research/*/{REPORT.md,metrics.json,equity.csv}"

clean:
	find . -type d -name __pycache__ -prune -exec rm -rf {} +
	find . -type d -name '*.egg-info' -prune -exec rm -rf {} +
	rm -rf .pytest_cache build dist
