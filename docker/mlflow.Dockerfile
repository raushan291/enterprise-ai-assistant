FROM ghcr.io/mlflow/mlflow:v2.14.1

# Install psycopg2
USER root
RUN pip install psycopg2-binary

# Create a user named 'mlflow' with home directory and proper permissions
RUN useradd -m -s /bin/bash mlflow

# Set ownership of /app to mlflow
RUN mkdir -p /app && chown -R mlflow:mlflow /app

WORKDIR /app

# Switch to mlflow user
USER mlflow