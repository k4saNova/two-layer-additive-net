ENV_PATH = $(PWD)/env
PYTHON_VERSION = `which python3.8`
PYTHON = $(ENV_PATH)/bin/python

.PHONY: env tests
env:
	@ virtualenv $(ENV_PATH) --python=$(PYTHON_VERSION)

tests:
	@ $(PYTHON) tests/test.py

install:
	@ $(PYTHON) setup.py install

clean:
	@ rm $(PWD)/build/ $(PWD)/dist/ $(ENV_PATH) -rf
	@ find . -type d -name "__pycache__" -delete
