FROM python:3.11-slim

ENV PYTHONUNBUFFERED=1

RUN apt-get update && apt-get install -y \
    git curl wget gcc g++ \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /workspace
COPY requirements.txt .
RUN pip install --upgrade pip && pip install -r requirements.txt

COPY . .

EXPOSE 9090
ENTRYPOINT ["python", "main.py"]
CMD ["--mode", "simulate", "--nodes", "10", "--steps", "200"]
