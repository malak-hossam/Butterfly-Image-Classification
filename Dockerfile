FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

WORKDIR /app

COPY requirements.txt /app/requirements.txt
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r /app/requirements.txt

COPY . /app

# Example default command (override in `docker run` as needed).
CMD ["python", "-m", "src.data.sanity_check", "--config", "src/config/default.yaml"]
