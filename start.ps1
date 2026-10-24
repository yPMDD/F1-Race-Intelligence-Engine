# ============================================================
# F1 Race Intelligence Engine - Full Stack Startup
# Usage:  .\start.ps1
#
# Boots:
#   uvicorn      -> FastAPI backend (port 8000)
#   npm run dev  -> React dashboard (port 5173)
#
# The LLM is Groq Cloud; no local model server is required.
# ============================================================

$ErrorActionPreference = "Continue"

# Resolve the project root from this script's own location so the script works
# regardless of where the repo is checked out. The previous version hardcoded
# "c:\wamp64\www\MyProjects\F1-Race-Intelligence-Engine", which failed on every
# other machine.
$ProjectRoot = Split-Path -Parent $MyInvocation.MyCommand.Path
$DashboardDir = Join-Path $ProjectRoot "apps\dashboard"

Write-Host ""
Write-Host "==========================================" -ForegroundColor Cyan
Write-Host " F1 RACE INTELLIGENCE ENGINE - 2026" -ForegroundColor Red
Write-Host "==========================================" -ForegroundColor Cyan
Write-Host " Root: $ProjectRoot" -ForegroundColor DarkGray
Write-Host ""

# --- Locate the Python toolchain -------------------------------------------
$VenvPython = Join-Path $ProjectRoot ".venv\Scripts\python.exe"
if (Test-Path $VenvPython) {
    $Py = $VenvPython
    Write-Host "[env] Using project venv: .venv" -ForegroundColor Green
} elseif (Get-Command uv -ErrorAction SilentlyContinue) {
    $Py = "uv"
    Write-Host "[env] Using uv" -ForegroundColor Green
} else {
    $Py = "python"
    Write-Host "[env] uv not found and .venv missing - using system python." -ForegroundColor Yellow
    Write-Host "      Run: python -m pip install uv && uv sync" -ForegroundColor Yellow
}

# --- Backend ---------------------------------------------------------------
Write-Host "[1/2] Starting FastAPI backend (port 8000)..." -ForegroundColor Yellow
$backendJob = Start-Job -ScriptBlock {
    param($Root, $PyExe)
    Set-Location $Root
    if ($PyExe -eq "uv") { uv run uvicorn apps.api.main:app --reload --host 127.0.0.1 --port 8000 }
    elseif ($PyExe -like "*.venv*") { & $PyExe -m uvicorn apps.api.main:app --reload --host 127.0.0.1 --port 8000 }
    else { python -m uvicorn apps.api.main:app --reload --host 127.0.0.1 --port 8000 }
} -ArgumentList $ProjectRoot, $Py

Start-Sleep -Seconds 5

try {
    $health = Invoke-WebRequest -Uri "http://localhost:8000/health" -TimeoutSec 5 -ErrorAction Stop
    Write-Host "      Backend healthy (HTTP $($health.StatusCode))" -ForegroundColor Green
} catch {
    Write-Host "      WARNING: backend not answering yet. Check http://localhost:8000/docs" -ForegroundColor Yellow
}

# --- Frontend --------------------------------------------------------------
Write-Host "[2/2] Starting React dashboard (port 5173)..." -ForegroundColor Yellow
$frontendJob = Start-Job -ScriptBlock {
    param($Dir)
    Set-Location $Dir
    npm run dev
} -ArgumentList $DashboardDir

Start-Sleep -Seconds 3
Write-Host "      Dashboard starting..." -ForegroundColor Green

# --- Summary ---------------------------------------------------------------
Write-Host ""
Write-Host "==========================================" -ForegroundColor Cyan
Write-Host " ALL SERVICES STARTED" -ForegroundColor Green
Write-Host "==========================================" -ForegroundColor Cyan
Write-Host ""
Write-Host "  Dashboard  -> http://localhost:5173" -ForegroundColor White
Write-Host "  API Docs   -> http://localhost:8000/docs" -ForegroundColor White
Write-Host "  LLM Engine -> Groq Cloud API" -ForegroundColor White
Write-Host ""
Write-Host "Press CTRL+C to stop all services." -ForegroundColor DarkGray
Write-Host ""

try {
    while ($true) {
        $backendOutput = Receive-Job -Job $backendJob
        if ($backendOutput) { Write-Host "[API] $backendOutput" -ForegroundColor DarkGray }
        $frontendOutput = Receive-Job -Job $frontendJob
        if ($frontendOutput) { Write-Host "[UI ] $frontendOutput" -ForegroundColor DarkGray }
        Start-Sleep -Seconds 2
    }
} finally {
    Write-Host ""
    Write-Host "Stopping all services..." -ForegroundColor Yellow
    Stop-Job -Job $backendJob, $frontendJob -ErrorAction SilentlyContinue
    Remove-Job -Job $backendJob, $frontendJob -ErrorAction SilentlyContinue
    Write-Host "All services stopped." -ForegroundColor Red
}
