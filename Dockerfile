FROM python:3.11
WORKDIR /earning-calls-rag

# Install the application dependencies
COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /bin/
COPY pyproject.toml uv.lock ./
RUN --mount=type=cache,target=/root/.cache/uv uv sync --frozen --no-install-project

# Pre-download the embedding model so the image is self-contained
ENV HF_HOME=/earning-calls-rag/.cache
RUN /earning-calls-rag/.venv/bin/python -c "from sentence_transformers import SentenceTransformer; SentenceTransformer('all-MiniLM-L6-v2')"
RUN chmod -R 777 /earning-calls-rag/.cache

# Code to run the application
COPY api ./api
COPY src ./src
EXPOSE 8080

# Set venv in PATH to make uvicorn directly available 
ENV PATH="/earning-calls-rag/.venv/bin:$PATH"

# Setup an app user so the container doesn't run as the root user
RUN useradd app
ENV USER=app
USER app

CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8080"]
