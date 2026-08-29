# A known-good environment for the dashboard.
#
#   docker build -t chemzero-ai .
#   docker run --rm -p 8501:8501 chemzero-ai
#
# The image generates the dataset and trains the model at build time, so the
# container starts with a working dashboard rather than an empty one.

FROM python:3.12-slim

# Fail fast and log immediately: a container that buffers its output is a
# container you cannot debug.
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONIOENCODING=utf-8 \
    PIP_NO_CACHE_DIR=1

WORKDIR /app

# Dependencies first, so a source edit does not invalidate the install layer.
COPY requirements.txt constraints.txt ./
RUN python -m pip install --upgrade pip \
    && python -m pip install -r requirements.txt -c constraints.txt

COPY pyproject.toml README.md ./
COPY src/ ./src/
COPY app/ ./app/
COPY tests/ ./tests/

# Build the artifacts the dashboard expects.
RUN python src/generate_data.py && python src/train_model.py

# Run as a non-root user.
RUN useradd --create-home --uid 10001 chemzero \
    && chown -R chemzero:chemzero /app
USER chemzero

EXPOSE 8501

HEALTHCHECK --interval=30s --timeout=5s --start-period=20s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8501/_stcore/health')"

ENTRYPOINT ["streamlit", "run", "app/streamlit_app.py", \
            "--server.port=8501", \
            "--server.address=0.0.0.0", \
            "--server.headless=true", \
            "--browser.gatherUsageStats=false"]
