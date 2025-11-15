FROM python:3.10-slim

# Install OS dependencies
RUN apt-get update && apt-get install -y git && apt-get clean

WORKDIR /app

ENV PYTHONPATH="/app:${PYTHONPATH:-}"

# Copy requirements first
COPY requirements.txt .

# Install dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of the application
COPY . .