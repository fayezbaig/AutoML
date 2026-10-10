help:
	@echo "Makefile help:"
	@echo "* install	to install the package and dependencies"
	@echo "* check		to run the code style checker"

install:
	pip install -e ".[dev]"

check:
	pycodestyle --max-line-length=120 src
	@echo "All good!"

.PHONY: install check help
.DEFAULT_GOAL := help