FROM python:3.11-slim

WORKDIR /app

# copy project
COPY . /app

RUN pip install --no-cache-dir --upgrade pip \
    && pip install --no-cache-dir fastapi uvicorn[standard] requests pandas numpy

# Render provides the port in the $PORT env var. Default to 8000 if not provided.
ENV PORT=8000

EXPOSE ${PORT}

# Use shell form so environment variables are expanded
CMD uvicorn app:app --host 0.0.0.0 --port ${PORT}
