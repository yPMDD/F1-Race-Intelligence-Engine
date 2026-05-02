# ============================================================
# F1 Race Intelligence Engine - Full Stack Startup
# Run this script once to start everything:
#   ollama serve   -> AI Strategist (port 11434)
#   uvicorn        -> FastAPI Backend (port 8000)
#   npm run dev    -> React Dashboard (port 5173)
# Usage: .\start.ps1
# ============================================================

Write-Host ""
Write-Host "==========================================" -ForegroundColor Cyan
Write-Host " F1 RACE INTELLIGENCE ENGINE - 2026" -ForegroundColor Red
Write-Host "==========================================" -ForegroundColor Cyan
Write-Host ""

# --- 1. [Removed Local Ollama] ---
Write-Host "[1/2] Connecting to Groq Cloud API for AI Strategist..." -ForegroundColor Green

# --- 2. FastAPI Backend ---
Write-Host ""
Write-Host "[1/2] Starting FastAPI Backend (port 8000)..." -ForegroundColor Yellow
$backendJob = Start-Job -ScriptBlock {
    Set-Location "c:\wamp64\www\MyProjects\F1-Race-Intelligence-Engine"
    uv run uvicorn apps.api.main:app --reload --host 127.0.0.1 --port 8000
}
Start-Sleep -Seconds 4

# Verify backend
try {
    $health = Invoke-WebRequest -Uri "http://localhost:8000/health" -TimeoutSec 5 -ErrorAction Stop
    Write-Host "      Backend is healthy!" -ForegroundColor Green
} catch {
    Write-Host "      WARNING: Backend may still be starting up. Check http://localhost:8000/docs" -ForegroundColor Yellow
}

# --- 3. React Dashboard ---
Write-Host ""
Write-Host "[2/2] Starting React Dashboard (port 5173)..." -ForegroundColor Yellow
$frontendJob = Start-Job -ScriptBlock {
    Set-Location "c:\wamp64\www\MyProjects\F1-Race-Intelligence-Engine\apps\dashboard"
    npm run dev
}
Start-Sleep -Seconds 3
Write-Host "      Dashboard starting..." -ForegroundColor Green

# --- Summary ---
Write-Host ""
Write-Host "==========================================" -ForegroundColor Cyan
Write-Host " ALL SERVICES STARTED" -ForegroundColor Green
Write-Host "==========================================" -ForegroundColor Cyan
Write-Host ""
Write-Host "  Dashboard  -> http://localhost:5173" -ForegroundColor White
Write-Host "  API Docs   -> http://localhost:8000/docs" -ForegroundColor White
Write-Host "  LLM Engine -> Groq Cloud API (Active)" -ForegroundColor White
Write-Host ""
Write-Host "Press CTRL+C to stop all services." -ForegroundColor DarkGray
Write-Host ""

# Keep the script alive and show backend logs
try {
    while ($true) {
        $backendOutput = Receive-Job -Job $backendJob
        if ($backendOutput) { Write-Host "[API] $backendOutput" -ForegroundColor DarkGray }
        Start-Sleep -Seconds 2
    }
} finally {
    Write-Host "Stopping all services..." -ForegroundColor Yellow
    Stop-Job -Job $backendJob, $frontendJob -ErrorAction SilentlyContinue
    Remove-Job -Job $backendJob, $frontendJob -ErrorAction SilentlyContinue
    Write-Host "All services stopped." -ForegroundColor Red
}
