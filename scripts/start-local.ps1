# Start full stack locally with Docker
$Root = Split-Path -Parent (Split-Path -Parent $MyInvocation.MyCommand.Path)
Set-Location $Root

if (-not (Test-Path ".env")) {
  Copy-Item ".env.example" ".env"
  Write-Host "Created .env from .env.example"
}

if (-not (Test-Path "Backend\models\DenseNet121.pth")) {
  Write-Host "WARNING: Place DenseNet121.pth in Backend\models\ before predictions will work."
}

Write-Host "Building and starting (UI http://localhost:8080 , API http://localhost:5000) ..."
docker compose up -d --build

Write-Host ""
Write-Host "Ready:"
Write-Host "  App:     http://localhost:8080"
Write-Host "  API:     http://localhost:5000/health"
Write-Host "  Logs:    docker compose logs -f"
Write-Host "  Stop:    docker compose down"
