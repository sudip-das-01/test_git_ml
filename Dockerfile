FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

WORKDIR /app

RUN apt-get update && \
    apt-get install -y --no-install-recommends build-essential && \
    rm -rf /var/lib/apt/lists/*

COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

COPY . .
RUN mkdir -p output

CMD ["sh", "-c", "if [ -f scripts/setup.sh ]; then chmod +x scripts/setup.sh && scripts/setup.sh; fi; if [ -f inference.py ]; then python inference.py; elif [ -f scripts/inference.py ]; then python scripts/inference.py; elif [ -f test.py ]; then python test.py; elif [ -f scripts/test.py ]; then python scripts/test.py; else echo \"No inference entrypoint found\" && exit 2; fi"]

