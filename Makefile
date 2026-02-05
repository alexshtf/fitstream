
.PHONY: setup build test lint pre-commit doc doc-open

setup:
	uv sync --all-groups

build:
	uv build

test:
	uv run pytest tests

lint:
	uv run ruff check .
	uv run pyright src

pre-commit:
	uv run pre-commit install

doc:
	uv run mkdocs build

doc-open:
	uv run mkdocs serve --open
