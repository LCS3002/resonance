$root = $PSScriptRoot

Write-Host ""
Write-Host "  Resonance — PROD" -ForegroundColor Cyan
Write-Host "  API:      http://localhost:8000" -ForegroundColor Gray
Write-Host "  Frontend: http://localhost:3000  <- open this" -ForegroundColor Gray
Write-Host ""

# Kill anything already on port 8000
foreach ($line in (netstat -ano 2>$null | Select-String ":8000 ")) {
    $p = ($line -split '\s+')[-1]
    if ($p -match '^\d+$') { Stop-Process -Id ([int]$p) -Force -ErrorAction SilentlyContinue }
}

# Start API in background, log to file
$logPath = Join-Path $root "api.log"
$api = Start-Process python `
    -ArgumentList "-m", "uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8000" `
    -WorkingDirectory $root `
    -NoNewWindow -PassThru `
    -RedirectStandardOutput $logPath `
    -RedirectStandardError $logPath

Write-Host "  API started (PID $($api.Id)) — logs: api.log" -ForegroundColor Green

# Build and start frontend container (detached)
docker compose -f docker-compose.prod.yml up --build -d

Write-Host ""
Write-Host "  Resonance is live at http://localhost:3000" -ForegroundColor Green
Write-Host ""
Write-Host "  To stop:" -ForegroundColor Gray
Write-Host "    docker compose -f docker-compose.prod.yml down" -ForegroundColor Gray
Write-Host "    Stop-Process -Id $($api.Id) -Force" -ForegroundColor Gray
Write-Host ""
