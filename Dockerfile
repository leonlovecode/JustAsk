FROM python:3.12-slim

WORKDIR /app

RUN pip install --no-cache-dir \
    openai \
    python-dotenv \
    numpy \
    anthropic

COPY src/ /app/src/
COPY config/ /app/config/

# Fresh data directory per container (no cross-contamination)
RUN mkdir -p /app/data /app/logs

ENV PYTHONPATH=/app/src
ENV PYTHONUNBUFFERED=1

ENTRYPOINT ["python"]
