.PHONY: install dev test lint type-check format clean docs bench

install:
	pip install -e .

dev:
	pip install -e ".[dev,qiskit]"
	pre-commit install

test:
	pytest tests/unit -v --timeout=30

test-all:
	pytest tests/ -v --timeout=120

test-cov:
	pytest tests/unit -v --cov=src/qb_compiler --cov-report=term-missing --cov-report=html

lint:
	ruff check src/ tests/

format:
	ruff format src/ tests/
	ruff check --fix src/ tests/

type-check:
	mypy src/qb_compiler/

clean:
	rm -rf dist/ build/ *.egg-info .mypy_cache .ruff_cache .pytest_cache htmlcov/
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true

docs:
	cd docs && make html

bench:
	pytest tests/benchmarks/ -v --benchmark-only
