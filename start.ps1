﻿param (
    [string]$StockfishPath = "assets\stockfish.exe"
)

$ErrorActionPreference = "Stop"

Set-Location -Path $PSScriptRoot

Write-Host "==================================================" -ForegroundColor Magenta
Write-Host "   PAWN CHESS AI - WINDOWS LAUNCHER               " -ForegroundColor Magenta
Write-Host "==================================================" -ForegroundColor Magenta
Write-Host ""

Write-Host "[INFO] Checking Python installation..." -ForegroundColor Cyan
try {
    $pythonCmd = Get-Command python -ErrorAction SilentlyContinue
    if ($null -eq $pythonCmd) {
        throw "Python executable not found in PATH."
    }

    $pyVersionInfo = (python --version 2>&1 | Out-String).Trim()
    if ($pyVersionInfo -match "Python 3") {
        Write-Host "[OK] Found: $pyVersionInfo" -ForegroundColor Green
    } else {
        throw "Python 3.x is required. Found: $pyVersionInfo"
    }
} catch {
    Write-Host "[ERR] Python detection failed: $_" -ForegroundColor Red
    Write-Host "[ERR] Please install Python 3.8+ from python.org and ensure it's in PATH." -ForegroundColor Red
    exit 1
}

$VenvPath = Join-Path $PSScriptRoot "venv"
if (-not (Test-Path $VenvPath)) {
    Write-Host "[INFO] Creating virtual environment at $VenvPath ..." -ForegroundColor Cyan
    try {
        python -m venv $VenvPath
        if ($LASTEXITCODE -ne 0) { throw "python -m venv failed with exit code $LASTEXITCODE" }
        Write-Host "[OK] Virtual environment created." -ForegroundColor Green
    } catch {
        Write-Host "[ERR] Failed to create venv: $_" -ForegroundColor Red
        exit 1
    }
} else {
    Write-Host "[INFO] Virtual environment already exists." -ForegroundColor Cyan
}

$VenvScripts = Join-Path $VenvPath "Scripts"
$VenvPython = Join-Path $VenvScripts "python.exe"
$VenvPip = Join-Path $VenvScripts "pip.exe"
$VenvStreamlit = Join-Path $VenvScripts "streamlit.exe"

if (-not (Test-Path $VenvPython)) {
    Write-Host "[ERR] Venv Python not found at: $VenvPython" -ForegroundColor Red
    Write-Host "[ERR] Delete the 'venv' folder and re-run start.ps1." -ForegroundColor Red
    exit 1
}

Write-Host "[OK] Using: $VenvPython" -ForegroundColor Green

Write-Host "[INFO] Installing/updating dependencies..." -ForegroundColor Cyan
try {
    if (Test-Path (Join-Path $PSScriptRoot "requirements.txt")) {
        & $VenvPython -m pip install --upgrade pip setuptools wheel | Out-Null
        & $VenvPip install -r (Join-Path $PSScriptRoot "requirements.txt")
        if ($LASTEXITCODE -ne 0) { throw "pip install failed with exit code $LASTEXITCODE" }
        Write-Host "[OK] Dependencies installed." -ForegroundColor Green
    } else {
        Write-Host "[WARN] requirements.txt not found. Skipping dependency install." -ForegroundColor Yellow
    }
} catch {
    Write-Host "[ERR] Failed to install dependencies: $_" -ForegroundColor Red
    exit 1
}

Write-Host "[INFO] Verifying directory structure..." -ForegroundColor Cyan
$Dirs = @("logs", "checkpoints", "data", "assets")
foreach ($d in $Dirs) {
    $p = Join-Path $PSScriptRoot $d
    if (-not (Test-Path $p)) {
        New-Item -ItemType Directory -Path $p | Out-Null
        Write-Host "[OK] Created directory: $d" -ForegroundColor Green
    }
}

$AbsStockfishPath = $StockfishPath
if (-not (Test-Path $AbsStockfishPath)) {
    $AbsStockfishPath = Join-Path $PSScriptRoot $StockfishPath
}

if (-not (Test-Path $AbsStockfishPath)) {
    Write-Host "[WARN] Stockfish executable not found at: $StockfishPath" -ForegroundColor Yellow
} else {
    Write-Host "[OK] Stockfish found at: $AbsStockfishPath" -ForegroundColor Green
}

Write-Host ""
Write-Host "==================================================" -ForegroundColor Magenta
Write-Host "   SYSTEM READY - STARTING SERVICES               " -ForegroundColor Magenta
Write-Host "==================================================" -ForegroundColor Magenta
Write-Host ""

Write-Host "[INFO] Starting Dashboard..." -ForegroundColor Cyan

$DashboardPath = Join-Path $PSScriptRoot "dashboard.py"
if (-not (Test-Path $DashboardPath)) {
    Write-Host "[ERR] dashboard.py not found at: $DashboardPath" -ForegroundColor Red
    exit 1
}

if (-not (Test-Path $VenvStreamlit)) {
    Write-Host "[ERR] streamlit.exe not found at: $VenvStreamlit" -ForegroundColor Red
    Write-Host "[ERR] Try re-running dependency installation or check requirements.txt." -ForegroundColor Red
    exit 1
}

try {
    Start-Process -FilePath $VenvStreamlit -ArgumentList @("run", $DashboardPath) -NoNewWindow
    Write-Host "[OK] Dashboard launched." -ForegroundColor Green
} catch {
    Write-Host "[ERR] Failed to launch dashboard: $_" -ForegroundColor Red
}

Write-Host ""
Write-Host "[OK] Setup complete." -ForegroundColor Green
Write-Host "[INFO] Training can be started manually with:" -ForegroundColor Cyan
Write-Host "  & `"$VenvPython`" `"$($PSScriptRoot)\train_end_to_end.py`"" -ForegroundColor Gray