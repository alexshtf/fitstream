
.PHONY: sync test doc

setup:
	uv sync --all-groups

test:
	uv run pytest tests

lint:
	uv run ruff check .
	uv run pyright src

pre-commit:
	uv run pre-commit install

doc:
	uv run mkdocs build
