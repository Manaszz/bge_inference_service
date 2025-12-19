# torch>=2.6 required by transformers safe torch.load checks (CVE-2025-32434)
FROM pytorch/pytorch:2.6.0-cuda12.4-cudnn9-runtime

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

WORKDIR /app

# System deps (git is sometimes needed for HF downloads)
RUN apt-get update && apt-get install -y --no-install-recommends \
    git \
    && rm -rf /var/lib/apt/lists/*

# Standalone repo layout: build context is this directory
COPY requirements.txt /app/requirements.txt
RUN pip install --no-cache-dir -r /app/requirements.txt

COPY . /app/bge_inference_service

EXPOSE 8011

CMD ["python", "-m", "uvicorn", "bge_inference_service.main:app", "--host", "0.0.0.0", "--port", "8011"]
