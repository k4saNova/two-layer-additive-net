PYTHON = python3.8
VENV_DIR = venv
VPYTHON = $(VENV_DIR)/bin/python
PIP = $(VENV_DIR)/bin/pip
INSTALL_PATH = $(VENV_DIR)/lib/$(PYTHON)/site-packages/

.PHONY: venv tests
venv:
	@ $(PYTHON) -m venv $(VENV_DIR)

requirements:
	@ $(PIP) install -r examples/requirements.txt

tests:
	@ $(VPYTHON) tests/test.py

clean:
	rm $(PWD)/build/ $(PWD)/dist/ -rf

install:
	@ $(VPYTHON) setup.py install
