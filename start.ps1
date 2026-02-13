<#
.SYNOPSIS
    Pawn Chess AI - Windows Startup Script
#>

param (
    [string]$StockfishPath = "assets\stockfish.exe"
)

$ErrorActionPreference = "Stop"

Write-Host "==================================================" -ForegroundColor Magenta
Write-Host "   PAWN CHESS AI - WINDOWS LAUNCHER               " -ForegroundColor Magenta
Write-Host "==================================================" -ForegroundColor Magenta
Write-Host ""

# 1. Python Detection
Write-Host "ℹ Checking Python installation..." -ForegroundColor Cyan
try {
    if (-not (Get-Command python -ErrorAction SilentlyContinue)) {
        throw "Python executable not found in PATH."
    }

    $pyVersionInfo = python --version 2>&1 | Out-String
    
    if ($pyVersionInfo -match "Python 3") {
        Write-Host "✔ Found Python: $pyVersionInfo" -ForegroundColor Green
    } else {
        throw "Python 3.x is required. Found: $pyVersionInfo"
    }
} catch {
    Write-Host "✖ Python detection failed: $_" -ForegroundColor Red
    Write-Host "  Please install Python 3.8+ from https://www.python.org/" -ForegroundColor Red
    exit 1
}

# 2. Virtual Environment Setup
$VenvPath = Join-Path $PSScriptRoot "venv"
if (-not (Test-Path $VenvPath)) {
    Write-Host "ℹ Creating virtual environment at $VenvPath..." -ForegroundColor Cyan
    try {
        python -m venv $VenvPath
        if ($LASTEXITCODE -ne 0) { throw "Return code $LASTEXITCODE" }
        Write-Host "✔ Virtual environment created." -ForegroundColor Green
    } catch {
        Write-Host "✖ Failed to create venv: $_" -ForegroundColor Red
        exit 1
    }
} else {
    Write-Host "ℹ Virtual environment already exists." -ForegroundColor Cyan
}

# Define Venv Paths
$VenvScripts = Join-Path $VenvPath "Scripts"
$VenvPython = Join-Path $VenvScripts "python.exe"
$VenvPip = Join-Path $VenvScripts "pip.exe"
$VenvStreamlit = Join-Path $VenvScripts "streamlit.exe"

if (-not (Test-Path $VenvPython)) {
    Write-Host "✖ Venv Python not found at $VenvPython. The venv might be corrupted." -ForegroundColor Red
    exit 1
}

Write-Host "✔ Using Python at: $VenvPython" -ForegroundColor Green

# 3. Dependencies
Write-Host "ℹ Installing/Updating dependencies..." -ForegroundColor Cyan
try {
    & $VenvPython -m pip install --upgrade pip setuptools wheel | Out-Null
    
    if (Test-Path "requirements.txt") {
        & $VenvPip install -r requirements.txt
        if ($LASTEXITCODE -ne 0) { throw "Pip install failed." }
        Write-Host "✔ Dependencies installed." -ForegroundColor Green
    } else {
        Write-Host "⚠ requirements.txt not found. Skipping dependency install." -ForegroundColor Yellow
    }
} catch {
    Write-Host "✖ Failed to install dependencies: $_" -ForegroundColor Red
    exit 1
}

# 4. Directory Structure
Write-Host "ℹ Verifying directory structure..." -ForegroundColor Cyan
$Dirs = @("logs", "checkpoints", "data", "assets")
foreach ($d in $Dirs) {
    $p = Join-Path $PSScriptRoot $d
    if (-not (Test-Path $p)) {
        New-Item -ItemType Directory -Path $p | Out-Null
        Write-Host "✔ Created directory: $d" -ForegroundColor Green
    }
}

# 5. Stockfish Validation
$AbsStockfishPath = $StockfishPath
if (-not (Test-Path $AbsStockfishPath)) {
    $AbsStockfishPath = Join-Path $PSScriptRoot $StockfishPath
}

if (-not (Test-Path $AbsStockfishPath)) {
    Write-Host "⚠ Stockfish executable not found at: $StockfishPath" -ForegroundColor Yellow
} else {
    Write-Host "✔ Stockfish found at: $AbsStockfishPath" -ForegroundColor Green
}

# 6. Launch
Write-Host ""
Write-Host "==================================================" -ForegroundColor Magenta
Write-Host "   SYSTEM READY - STARTING SERVICES               " -ForegroundColor Magenta
Write-Host "==================================================" -ForegroundColor Magenta
Write-Host ""

Write-Host "ℹ Starting Dashboard..." -ForegroundColor Cyan

if (-not (Test-Path "dashboard.py")) {
    Write-Host "✖ dashboard.py not found!" -ForegroundColor Red
    exit 1
}

try {
    Start-Process -FilePath $VenvStreamlit -ArgumentList "run dashboard.py" -NoNewWindow
    Write-Host "✔ Dashboard launched." -ForegroundColor Green
} catch {
    Write-Host "✖ Failed to launch dashboard: $_" -ForegroundColor Red
}

Write-Host ""
Write-Host "✔ Setup Complete." -ForegroundColor Green
Write-Host "  You can now run training scripts manually using:" -ForegroundColor Cyan
Write-Host "  & '$VenvPython' train_end_to_end.py" -ForegroundColor Gray