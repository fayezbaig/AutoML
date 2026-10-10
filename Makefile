.PHONY: install check help

help:
	@echo "Makefile help:"
	@echo "* install	to install the package and dev dependencies"
	@echo "* check		to run the code style checker"

install:
	pip install -e ".[dev]"

check:
	pycodestyle --max-line-length=120 --ignore=E402 src
	@echo "All good!"

.DEFAULT_GOAL := help
