@echo off
echo.
echo  Resonance -- starting...
echo.

cd /d "%~dp0"

:: Kill anything already on port 8000
for /f "tokens=5" %%a in ('netstat -ano 2^>nul ^| findstr ":8000 "') do (
    taskkill /F /PID %%a >nul 2>&1
)

:: Open browser after 8 seconds (fire and forget)
start "" /B cmd /c "timeout /t 8 /nobreak >nul && start http://localhost:8000"

echo  http://localhost:8000  (opening in browser in 8s)
echo  http://localhost:8000/docs   API docs
echo  Ctrl+C to stop
echo.

:: Start server (foreground — Ctrl+C stops it)
python -m uvicorn api.main:app --host 0.0.0.0 --port 8000
