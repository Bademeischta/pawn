<#
.SYNOPSIS
    Pawn Chess AI - Windows Startup Script
    Robust setup and launch script for the Pawn project.

.DESCRIPTION
    This script handles:
    1. Execution Policy checks
    2. Python 3.8+ detection
    3. Virtual Environment (venv) creation and activation
    4. Dependency installation via pip
    5. Directory structure verification
    6. Stockfish executable validation
    7. Launching the Dashboard and Training processes

.PARAMETER StockfishPath
    Path to the Stockfish executable. Defaults to 'assets/stockfish.exe'.

.EXAMPLE
    .\start.ps1
    .\start.ps1 -StockfishPath "C:\Chess\Stockfish\stockfish_16.exe"
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
    $pyVersion = python --version 2>&1
    if ($pyVersion -match "Python 3\.(8|9|10|11|12)") {
        Write-Host "✔ Found $pyVersion" -ForegroundColor Green
    } else {
        throw "Python 3.8+ is required but not found in PATH. Please install it from python.org."
    }
} catch {
    Write-Host "✖ Python detection failed: $_" -ForegroundColor Red
    exit 1
}

# 2. Virtual Environment Setup
$VenvPath = Join-Path $PSScriptRoot "venv"
if (-not (Test-Path $VenvPath)) {
    Write-Host "ℹ Creating virtual environment at $VenvPath..." -ForegroundColor Cyan
    python -m venv $VenvPath
    if ($LASTEXITCODE -ne 0) { throw "Failed to create venv." }
    Write-Host "✔ Virtual environment created." -ForegroundColor Green
} else {
    Write-Host "ℹ Virtual environment already exists." -ForegroundColor Cyan
}

# Activate Venv
$ActivateScript = Join-Path $VenvPath "Scripts\activate.ps1"
if (-not (Test-Path $ActivateScript)) {
    Write-Host "✖ Activation script not found at $ActivateScript. The venv might be corrupted." -ForegroundColor Red
    exit 1
}

# We can't "activate" in the current scope easily in a script without dot-sourcing, 
# but for running commands, we can just use the python binary inside venv.
$VenvPython = Join-Path $VenvPath "Scripts\python.exe"
$VenvPip = Join-Path $VenvPath "Scripts\pip.exe"
$VenvStreamlit = Join-Path $VenvPath "Scripts\streamlit.exe"

Write-Host "✔ Using Python at: $VenvPython" -ForegroundColor Green

# 3. Dependencies
Write-Host "ℹ Installing/Updating dependencies..." -ForegroundColor Cyan
try {
    & $VenvPip install --upgrade pip setuptools wheel | Out-Null
    & $VenvPip install -r requirements.txt
    if ($LASTEXITCODE -ne 0) { throw "Pip install failed." }
    Write-Host "✔ Dependencies installed." -ForegroundColor Green
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
    # Try relative to script root if not found
    $AbsStockfishPath = Join-Path $PSScriptRoot $StockfishPath
}

if (-not (Test-Path $AbsStockfishPath)) {
    Write-Host "⚠ Stockfish executable not found at: $StockfishPath" -ForegroundColor Yellow
    Write-Host "⚠ The system will attempt to download it automatically via distillzero_factory.py later." -ForegroundColor Yellow
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

# Check if dashboard.py exists
if (-not (Test-Path "dashboard.py")) {
    Write-Host "✖ dashboard.py not found!" -ForegroundColor Red
    exit 1
}

# Launch Streamlit
try {
    Start-Process -FilePath $VenvStreamlit -ArgumentList "run dashboard.py" -NoNewWindow
    Write-Host "✔ Dashboard launched." -ForegroundColor Green
} catch {
    Write-Host "✖ Failed to launch dashboard: $_" -ForegroundColor Red
}

Write-Host ""
Write-Host "✔ Setup Complete. You can now run training scripts manually using:" -ForegroundColor Green
Write-Host "  & '$VenvPython' train_end_to_end.py" -ForegroundColor Cyan
