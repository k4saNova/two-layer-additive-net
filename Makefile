ENV_PATH = $(PWD)/env
PYTHON = `which python3.8`

.PHONY: env test
env:
	@ virtualenv $(ENV_PATH) --python=$(PYTHON)

test:
	@ echo "test"

install:
	@ $(PYTHON) setup.py install

clean:
	@ rm $(PWD)/build/ $(PWD)/dist/ $(ENV_PATH) -rf
	@ find . -type d -name "__pycache__" -delete
