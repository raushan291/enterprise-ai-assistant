#!/bin/bash

echo "Stopping port-forward processes..."
pkill -f "kubectl port-forward" || true

echo "Stopping Minikube..."
minikube stop

# echo "Deleting Minikube cluster (optional)..."
# Uncomment if you want to delete the cluster completely
# minikube delete

echo "Shutdown complete"
