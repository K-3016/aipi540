FROM python:3.11-slim

WORKDIR /app
COPY pyproject.toml setup.py README.md ./
COPY src ./src
RUN pip install --no-cache-dir .

COPY models ./models
RUN test -f models/deep_model.pt && test -f models/metadata.json
ENV BRAIN_TUMOR_ARTIFACT_DIR=/app/models
EXPOSE 8000
CMD ["sh", "-c", "uvicorn brain_tumor_ml.api:app --host 0.0.0.0 --port ${PORT:-8000}"]
