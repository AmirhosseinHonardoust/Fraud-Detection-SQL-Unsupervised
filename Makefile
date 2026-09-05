.PHONY: lint format-check typecheck test gate

lint:
	ruff check src tests

format-check:
	black --check src tests

typecheck:
	mypy src

test:
	pytest

gate: lint format-check typecheck test
