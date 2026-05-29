Write-Host "Building Resonance frontend — PROD" -ForegroundColor Cyan
Write-Host "  Vite build + nginx at http://localhost:3000" -ForegroundColor Gray
Write-Host "  API proxied to http://localhost:8000 (run start.bat separately)" -ForegroundColor Gray
Write-Host ""
docker compose -f docker-compose.prod.yml up --build -d
Write-Host ""
Write-Host "Frontend live at http://localhost:3000" -ForegroundColor Green
