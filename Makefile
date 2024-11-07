MANAGER=conda
LIBRARY_NAME=lnb
PYTHON_VERSION=3.9
PYTHON_FILES=src/lnb experiments

RUN_CMD=$(MANAGER) run -n $(LIBRARY_NAME)

.PHONY: create-env
create-env:
	$(MANAGER) create -y -n $(LIBRARY_NAME) python=$(PYTHON_VERSION)

.PHONY: install-qbs
install-qbs:
	git clone -b main https://github.com/computationalprivacy/querysnout.git
	cd querysnout/src/optimized_qbs && $(RUN_CMD) pip install .
	rm -rf querysnout

.PHONY: install-pkg-dev
install-pkg-dev:
	$(RUN_CMD) pip install -e .[dev]
	$(MAKE) install-qbs

.PHONY: install-pkg
install-pkg:
	$(RUN_CMD) pip install .
	$(MAKE) install-qbs

.PHONY: install-kernel
install-kernel:
	$(RUN_CMD) pip install ipykernel
	$(RUN_CMD) python -m ipykernel install --user --name $(LIBRARY_NAME) --display-name "$(LIBRARY_NAME): Python($(LIBRARY_NAME))"

.PHONY: py-format
py-format:
	ruff format $(PYTHON_FILES)

.PHONY: py-lint
py-lint:
	ruff check --fix $(PYTHON_FILES)