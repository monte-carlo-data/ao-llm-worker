# A rebuild on the current slim tag ships openssl/libssl3t64 >= 3.5.4-1~deb13u2
# (OLYM-7625, OLYM-7626). Left as a floating tag so future base-image security
# updates arrive automatically on each rebuild.
FROM python:3.14-slim

# Pinned to a uv release bundling astral-tokio-tar >= 0.6.1 (OLYM-7622) and
# quinn-proto >= 0.11.15 (OLYM-7642); uv 0.11.26+ carries both fixes. A version
# tag (over :latest) keeps builds reproducible.
COPY --from=ghcr.io/astral-sh/uv:0.11.29 /uv /uvx /bin/

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
