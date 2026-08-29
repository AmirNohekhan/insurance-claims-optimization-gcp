PYTHON ?= python
.PHONY: setup data train evaluate optimize backtest serve demo test lint typecheck
setup:
	$(PYTHON) -m pip install -e ".[dev]"
data:
	$(PYTHON) scripts/generate_data.py
train evaluate optimize backtest demo:
	$(PYTHON) scripts/run_demo.py
serve:
	$(PYTHON) -m uvicorn insurance_claims_platform.serving.app:app --reload
test:
	$(PYTHON) -m pytest -q
lint:
	$(PYTHON) -m ruff check src tests scripts
typecheck:
	$(PYTHON) -m mypy src

