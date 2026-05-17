# Build and push FRONTEND image only (standalone mode)
# Usage: .\scripts\docker-push-frontend.ps1 -User youruser -ApiUrl http://YOUR_SERVER:5000

param(
  [Parameter(Mandatory = $true)]
  [string]$User,
  [string]$Tag = "latest",
  [Parameter(Mandatory = $true)]
  [string]$ApiUrl
)

$ErrorActionPreference = "Stop"
$AppRoot = Split-Path -Parent (Split-Path -Parent $MyInvocation.MyCommand.Path)
Set-Location $AppRoot

$image = "${User}/alzheimer-frontend:${Tag}"

Write-Host "Building frontend (API -> $ApiUrl)..."
docker build `
  --build-arg REACT_APP_API_URL="$ApiUrl" `
  --build-arg NGINX_CONF=nginx.standalone.conf `
  -t $image .

Write-Host "Pushing $image ..."
docker push $image

Write-Host "Done. Run on server:"
Write-Host "  docker run -d -p 80:80 --name alzheimer-frontend $image"
