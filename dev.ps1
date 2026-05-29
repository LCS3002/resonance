Write-Host "Starting Resonance frontend — DEV" -ForegroundColor Cyan
Write-Host "  Vite dev server with HMR at http://localhost:5173" -ForegroundColor Gray
Write-Host "  API proxied to http://localhost:8000 (run start.bat separately)" -ForegroundColor Gray
Write-Host ""
docker compose -f docker-compose.dev.yml up --build
