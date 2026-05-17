# Build and push images to Docker Hub
# Usage: .\scripts\docker-push.ps1 -User your_dockerhub_username -Tag latest

param(
  [Parameter(Mandatory = $true)]
  [string]$User,
  [string]$Tag = "latest"
)

$ErrorActionPreference = "Stop"
$Root = Split-Path -Parent (Split-Path -Parent $MyInvocation.MyCommand.Path)

Set-Location $Root

$env:DOCKERHUB_USER = $User
$env:IMAGE_TAG = $Tag

Write-Host "Building images..."
docker compose build

Write-Host "Pushing backend..."
docker push "${User}/alzheimer-backend:${Tag}"

Write-Host "Pushing frontend..."
docker push "${User}/alzheimer-frontend:${Tag}"

Write-Host "Done. Deploy with:"
Write-Host "  docker compose pull"
Write-Host "  docker compose up -d"
