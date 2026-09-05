PYTHON ?= .venv/bin/python
RUFF ?= .venv/bin/ruff
CLANG_FORMAT ?= clang-format
CPP_SOURCES := $(wildcard kernels/*.cpp kernels/*.hpp kernels/*.cu kernels/*.cuh)

.PHONY: format lint test lean figures paper

format:
	$(RUFF) format cmk tests scripts
	$(RUFF) check --fix cmk tests scripts
	$(RUFF) format cmk tests scripts
	$(CLANG_FORMAT) -i $(CPP_SOURCES)

lint:
	$(RUFF) check cmk tests scripts
	$(RUFF) format --check cmk tests scripts
	$(CLANG_FORMAT) --dry-run --Werror $(CPP_SOURCES)

test:
	OPENBLAS_NUM_THREADS=1 $(PYTHON) -m pytest -q

lean:
	bash scripts/check_lean.sh

figures:
	MPLBACKEND=Agg $(PYTHON) scripts/figures.py

paper:
	bash scripts/build_paper.sh
