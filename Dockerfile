FROM python:3.11-slim

WORKDIR /app

# Install only the health-check dependency
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Copy dependency specifications
COPY requirements.txt pyproject.toml /app/

# Install python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy project files
COPY . /app/

EXPOSE 8000

ENV PYTHONUNBUFFERED=1

CMD ["python", "application.py"]
