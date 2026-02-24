FROM python:3.13-slim-bookworm

COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /bin/

WORKDIR /code

ENV PATH="/code/.venv/bin:$PATH"

COPY "pyproject.toml" "uv.lock" ".python-version" ./

RUN uv sync --locked

COPY  "models/model.pkl" ./models

COPY "app/predict.py" ./

EXPOSE 9696


CMD uvicorn predict:app --host 0.0.0.0 --port ${PORT:-9696}
