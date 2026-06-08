FROM python:3.11-slim

WORKDIR /app
COPY pyproject.toml setup.py README.md ./
COPY src ./src
RUN pip install --no-cache-dir .

COPY models ./models
ENV BRAIN_TUMOR_ARTIFACT_DIR=/app/models
EXPOSE 8000
CMD ["uvicorn", "brain_tumor_ml.api:app", "--host", "0.0.0.0", "--port", "8000"]
