.PHONY: lint format-check typecheck test coverage gate

lint:
	ruff check src tests

format-check:
	black --check src tests

typecheck:
	mypy src tests

test:
	pytest

coverage:
	pytest --cov=src --cov-report=term-missing

gate: lint format-check typecheck test coverage
