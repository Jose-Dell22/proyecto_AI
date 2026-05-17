#!/bin/sh
set -e

MODEL_PATH="/app/models/DenseNet121.pth"

if [ ! -f "$MODEL_PATH" ]; then
  echo "ERROR: Model weights not found at $MODEL_PATH"
  echo "Mount the file into /app/models/DenseNet121.pth before starting the container."
  exit 1
fi

echo "Starting Alzheimer MRI backend..."
exec "$@"
