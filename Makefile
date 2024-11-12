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
	cd querysnout/src/optimized_qbs && $(RUN_CMD) pip install .

.PHONY: install-pkg-dev
install-pkg-dev:
	$(RUN_CMD) pip install -e .[dev]

.PHONY: install-pkg
install-pkg:
	$(RUN_CMD) pip install .

.PHONY: install-old-mbi
install-old-mbi:
	$(RUN_CMD) pip install git+https://github.com/ryan112358/private-pgm.git@4152cc591f30560ce994870318ee4801b016bf94

.PHONY: install-kernel
install-kernel:
	$(RUN_CMD) pip install ipykernel
	$(RUN_CMD) python -m ipykernel install --user --name $(LIBRARY_NAME) --display-name "$(LIBRARY_NAME): Python($(LIBRARY_NAME))"

.PHONY: setup-env
setup-env:
	$(MAKE) create-env
	$(MAKE) install-pkg
	$(MAKE) install-qbs
	$(MAKE) install-old-mbi
	$(MAKE) install-kernel

.PHONY: py-format
py-format:
	ruff format $(PYTHON_FILES)

.PHONY: py-lint
py-lint:
	ruff check --fix $(PYTHON_FILES)