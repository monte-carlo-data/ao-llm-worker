FROM python:3.14-slim

COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /bin/

WORKDIR /app

COPY pyproject.toml uv.lock ./
RUN uv sync --frozen --no-dev --no-install-project
COPY src/ src/
RUN uv sync --frozen --no-dev

# Licensing artifacts: this project's license plus the NOTICE and verbatim
# third-party license texts for everything bundled in the image.
COPY LICENSE NOTICE ./
COPY LICENSES/ LICENSES/

CMD [".venv/bin/llm-worker"]
