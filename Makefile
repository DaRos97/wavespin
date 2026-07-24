.PHONY: docs docs-serve docs-build

docs:
	pip install -e ".[docs]"

docs-serve:
	mkdocs serve

docs-build:
	mkdocs build
