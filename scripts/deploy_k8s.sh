#!/bin/bash

echo "Starting Minikube..."
minikube start --driver=docker --memory=4096 --cpus=2

echo "Enabling Ingress..."
minikube addons enable ingress

echo "Applying Kubernetes manifests..."
kubectl apply -R -f k8s/

echo "Waiting for pods to be ready..."
kubectl wait --for=condition=available --timeout=90s deployment/eka-api
kubectl wait --for=condition=available --timeout=90s deployment/eka-ui
kubectl wait --for=condition=available --timeout=90s deployment/chroma

echo "Setting up port-forwarding..."
# Run all port forwards in background
kubectl port-forward deployment/eka-api 8001:8001 >/tmp/eka-api.log 2>&1 &
kubectl port-forward deployment/eka-ui 8501:8501 >/tmp/eka-ui.log 2>&1 &
kubectl port-forward deployment/chroma 8000:8000 >/tmp/chroma.log 2>&1 &

echo "All set!"
echo "API:      http://localhost:8001"
echo "UI:       http://localhost:8501"
echo "ChromaDB: http://localhost:8000"
