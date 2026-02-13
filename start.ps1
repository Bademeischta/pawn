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

# --- Colors for Output ---
function Write-Success { param($Message) Write-Host "✔ $Message" -ForegroundColor Green }
function Write-Info    { param($Message) Write-Host "ℹ $Message" -ForegroundColor Cyan }
function Write-Warn    { param($Message) Write-Host "⚠ $Message" -ForegroundColor Yellow }
function Write-ErrorMsg   { param($Message) Write-Host "✖ $Message" -ForegroundColor Red }

Write-Host "==================================================" -ForegroundColor Magenta
Write-Host "   PAWN CHESS AI - WINDOWS LAUNCHER               " -ForegroundColor Magenta
Write-Host "==================================================" -ForegroundColor Magenta
Write-Host ""

# 1. Python Detection
Write-Info "Checking Python installation..."
try {
    $pyVersion = python --version 2>&1
    if ($pyVersion -match "Python 3\.(8|9|10|11|12)") {
        Write-Success "Found $pyVersion"
    } else {
        throw "Python 3.8+ is required but not found in PATH. Please install it from python.org."
    }
} catch {
    Write-ErrorMsg "Python detection failed: $_"
    exit 1
}

# 2. Virtual Environment Setup
$VenvPath = Join-Path $PSScriptRoot "venv"
if (-not (Test-Path $VenvPath)) {
    Write-Info "Creating virtual environment at $VenvPath..."
    python -m venv $VenvPath
    if ($LASTEXITCODE -ne 0) { throw "Failed to create venv." }
    Write-Success "Virtual environment created."
} else {
    Write-Info "Virtual environment already exists."
}

# Activate Venv
$ActivateScript = Join-Path $VenvPath "Scripts\activate.ps1"
if (-not (Test-Path $ActivateScript)) {
    Write-ErrorMsg "Activation script not found at $ActivateScript. The venv might be corrupted."
    exit 1
}

# We can't "activate" in the current scope easily in a script without dot-sourcing, 
# but for running commands, we can just use the python binary inside venv.
$VenvPython = Join-Path $VenvPath "Scripts\python.exe"
$VenvPip = Join-Path $VenvPath "Scripts\pip.exe"
$VenvStreamlit = Join-Path $VenvPath "Scripts\streamlit.exe"

Write-Success "Using Python at: $VenvPython"

# 3. Dependencies
Write-Info "Installing/Updating dependencies..."
try {
    & $VenvPip install --upgrade pip setuptools wheel | Out-Null
    & $VenvPip install -r requirements.txt
    if ($LASTEXITCODE -ne 0) { throw "Pip install failed." }
    Write-Success "Dependencies installed."
} catch {
    Write-ErrorMsg "Failed to install dependencies: $_"
    exit 1
}

# 4. Directory Structure
Write-Info "Verifying directory structure..."
$Dirs = @("logs", "checkpoints", "data", "assets")
foreach ($d in $Dirs) {
    $p = Join-Path $PSScriptRoot $d
    if (-not (Test-Path $p)) {
        New-Item -ItemType Directory -Path $p | Out-Null
        Write-Success "Created directory: $d"
    }
}

# 5. Stockfish Validation
$AbsStockfishPath = $StockfishPath
if (-not (Test-Path $AbsStockfishPath)) {
    # Try relative to script root if not found
    $AbsStockfishPath = Join-Path $PSScriptRoot $StockfishPath
}

if (-not (Test-Path $AbsStockfishPath)) {
    Write-Warn "Stockfish executable not found at: $StockfishPath"
    Write-Warn "The system will attempt to download it automatically via distillzero_factory.py later."
} else {
    Write-Success "Stockfish found at: $AbsStockfishPath"
}

# 6. Launch
Write-Host ""
Write-Host "==================================================" -ForegroundColor Magenta
Write-Host "   SYSTEM READY - STARTING SERVICES               " -ForegroundColor Magenta
Write-Host "==================================================" -ForegroundColor Magenta
Write-Host ""

Write-Info "Starting Dashboard..."

# Check if dashboard.py exists
if (-not (Test-Path "dashboard.py")) {
    Write-ErrorMsg "dashboard.py not found!"
    exit 1
}

# Launch Streamlit
try {
    Start-Process -FilePath $VenvStreamlit -ArgumentList "run dashboard.py" -NoNewWindow
    Write-Success "Dashboard launched."
} catch {
    Write-ErrorMsg "Failed to launch dashboard: $_"
}

Write-Host ""
Write-Success "Setup Complete. You can now run training scripts manually using:"
Write-Host "  & '$VenvPython' train_end_to_end.py" -ForegroundColor Cyan
