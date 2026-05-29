$root = $PSScriptRoot

Write-Host ""
Write-Host "  Resonance — DEV" -ForegroundColor Cyan
Write-Host "  API:      http://localhost:8000" -ForegroundColor Gray
Write-Host "  Frontend: http://localhost:5173  <- open this" -ForegroundColor Gray
Write-Host ""

# Kill anything already on port 8000
foreach ($line in (netstat -ano 2>$null | Select-String ":8000 ")) {
    $p = ($line -split '\s+')[-1]
    if ($p -match '^\d+$') { Stop-Process -Id ([int]$p) -Force -ErrorAction SilentlyContinue }
}

# Start API in a new window so its logs stay visible separately
Start-Process powershell -ArgumentList @(
    "-NoExit", "-Command",
    "cd '$root'; Write-Host 'Resonance API — http://localhost:8000' -ForegroundColor Cyan; python -m uvicorn api.main:app --host 0.0.0.0 --port 8000 --reload"
)

Write-Host "  API starting in a new window..." -ForegroundColor Green
Write-Host "  Ctrl+C here stops the frontend. Close the API window separately." -ForegroundColor Yellow
Write-Host ""

Start-Sleep -Seconds 2
docker compose -f docker-compose.dev.yml up --build
