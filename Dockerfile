FROM python:3.11-slim

WORKDIR /app

COPY pyproject.toml LICENSE README.md ./
COPY src/ src/
COPY tests/ tests/

RUN pip install --no-cache-dir -e ".[dev]"

CMD ["pytest", "tests/", "-v", "--timeout=60"]
