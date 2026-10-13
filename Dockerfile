# Runs the Streamlit app on port 7860, which is what Hugging Face Spaces expects.
# The API can run from the same image:
#   docker run -p 8000:8000 IMAGE uvicorn policy_assistant.api:app --host 0.0.0.0 --port 8000
FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    HOME=/home/user \
    FASTEMBED_CACHE_PATH=/home/user/.cache/fastembed

# Hugging Face Spaces runs containers as user 1000.
RUN useradd -m -u 1000 user
WORKDIR /home/user/app

COPY --chown=user pyproject.toml README.md ./
COPY --chown=user src ./src
RUN pip install --no-cache-dir .

COPY --chown=user app.py ./
COPY --chown=user .streamlit ./.streamlit
COPY --chown=user data/policies ./data/policies

USER user
# Download the embedding model at build time so the first visitor doesn't wait for it.
RUN python -c "from fastembed import TextEmbedding; TextEmbedding('BAAI/bge-small-en-v1.5')"

EXPOSE 7860
CMD ["streamlit", "run", "app.py", "--server.port=7860", "--server.address=0.0.0.0"]
