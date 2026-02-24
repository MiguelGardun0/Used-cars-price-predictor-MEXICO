FROM python:3.13-slim-bookworm

COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /bin/

WORKDIR /code

ENV PATH="/code/.venv/bin:$PATH"

ENV PYTHONPATH=/code

COPY "pyproject.toml" "uv.lock" ".python-version" ./

RUN uv sync --locked

COPY  "models/" ./models/

COPY "app/" ./app/

COPY "tests/" ./tests/

EXPOSE 9696


CMD uvicorn app.predict:app --host 0.0.0.0 --port ${PORT:-9696}
