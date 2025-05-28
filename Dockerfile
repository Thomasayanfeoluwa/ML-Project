FROM public.ecr.aws/docker/library/python:3.13-slim

WORKDIR /app

RUN apt-get update -y && \
    apt-get install -y --no-install-recommends \
        gcc \
        build-essential \
        awscli \
        git && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

COPY requirements.txt .

# Force no cache, no hash checking
RUN pip install --upgrade pip setuptools wheel && \
    pip install --default-timeout=300 --retries=10 --no-cache-dir --no-deps -r requirements.txt || true && \
    pip install --default-timeout=300 --retries=10 --no-cache-dir -r requirements.txt

COPY . .

CMD ["python", "app.py"]
