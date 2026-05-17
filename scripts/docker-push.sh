#!/usr/bin/env sh
set -e

USER_NAME="${1:?Usage: ./scripts/docker-push.sh <dockerhub_user> [tag]}"
TAG="${2:-latest}"

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

export DOCKERHUB_USER="$USER_NAME"
export IMAGE_TAG="$TAG"

echo "Building images..."
docker compose build

echo "Pushing backend..."
docker push "${USER_NAME}/alzheimer-backend:${TAG}"

echo "Pushing frontend..."
docker push "${USER_NAME}/alzheimer-frontend:${TAG}"

echo "Done."
