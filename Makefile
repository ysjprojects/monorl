PY ?= .venv/bin/python
UV ?= uv
PNPM ?= pnpm

.PHONY: help install test test-fast web serve envs lint vendor-list

help:
	@echo "install    create .venv and install monorl with dev/trl/distributed/rewards/openenv/hermes extras"
	@echo "test       run the full test suite"
	@echo "test-fast  run tests except 'slow' and 'network'"
	@echo "web        build the web UI (web/dist, served by 'monorl serve')"
	@echo "serve      start the web UI + API on http://127.0.0.1:8000"
	@echo "envs       build the Go/C++ example MEP/1 environments into envs/bin"
	@echo "lint       ruff check + format check"

install:
	$(UV) venv .venv --python 3.12
	$(UV) pip install --python $(PY) -e ".[dev,trl,distributed,rewards,openenv,hermes]"

test:
	$(PY) -m pytest tests -q

test-fast:
	$(PY) -m pytest tests -q -m 'not slow and not network'

web:
	cd web && $(PNPM) install --frozen-lockfile && $(PNPM) build

serve:
	$(PY) -m monorl.cli serve

envs:
	$(MAKE) -C envs all

lint:
	$(PY) -m ruff check src tests scripts
	$(PY) -m ruff format --check src tests scripts

vendor-list:
	$(PY) scripts/vendor.py --list
