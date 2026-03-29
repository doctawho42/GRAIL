PYTHON ?= python

.PHONY: install test smoke presets build release predict

install:
	$(PYTHON) -m pip install --upgrade pip
	$(PYTHON) -m pip install -r requirements.txt
	$(PYTHON) -m pip install -e .[tuning]

test:
	$(PYTHON) -m pytest grail_metabolism/tests -q

smoke:
	$(PYTHON) -m grail_metabolism predict CCO --rules grail_metabolism/resources/example_rules.txt
	$(PYTHON) -m grail_metabolism presets --export-dir /tmp/grail_presets

presets:
	$(PYTHON) -m grail_metabolism presets --export-dir configs/generated

build:
	$(PYTHON) -m build

release:
	bash scripts/build_release.sh

predict:
	$(PYTHON) -m grail_metabolism predict CCO --rules grail_metabolism/resources/example_rules.txt
