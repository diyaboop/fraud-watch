FROM python:3.11-slim

WORKDIR /app

# Install system deps for matplotlib/sklearn
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy source (ML pipeline untouched, only add API layer)
COPY train.py audit.py monitor.py api.py ./
COPY static/ static/
COPY outputs/ outputs/

EXPOSE $PORT

CMD uvicorn api:app --host 0.0.0.0 --port ${PORT:-8000}
