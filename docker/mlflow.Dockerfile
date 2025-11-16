FROM ghcr.io/mlflow/mlflow:v2.14.1

# Install psycopg2
USER root
RUN pip install psycopg2-binary

# Create a user named 'mlflow' with home directory and proper permissions
RUN useradd -m -s /bin/bash mlflow

# Set ownership of /mlflow to mlflow
RUN mkdir -p /mlflow && chown -R mlflow:mlflow /mlflow

WORKDIR /mlflow

# Switch to mlflow user
USER mlflow