#!/bin/bash
set -e  # Exit immediately if a command exits with a non-zero status.

# Colors
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m' # No Color

echo -e "${GREEN}==================================================${NC}"
echo -e "${GREEN}   PAWN CHESS AI - LINUX/MACOS LAUNCHER           ${NC}"
echo -e "${GREEN}==================================================${NC}"
echo ""

# 1. Check for Python 3.8+
if ! command -v python3 &> /dev/null; then
    echo -e "${RED}Error: Python 3 is not installed or not in PATH.${NC}"
    exit 1
fi

PYTHON_VERSION=$(python3 -c 'import sys; print(f"{sys.version_info.major}.{sys.version_info.minor}")')
MAJOR=$(echo "$PYTHON_VERSION" | cut -d. -f1)
MINOR=$(echo "$PYTHON_VERSION" | cut -d. -f2)

if [ "$MAJOR" -lt 3 ] || ([ "$MAJOR" -eq 3 ] && [ "$MINOR" -lt 8 ]); then
    echo -e "${RED}Error: Python 3.8+ is required. Found $PYTHON_VERSION.${NC}"
    exit 1
fi
echo -e "${GREEN}✔ Found Python $PYTHON_VERSION${NC}"

# 2. Virtual Environment
if [ ! -d "venv" ]; then
    echo -e "${YELLOW}Creating virtual environment...${NC}"
    python3 -m venv venv
    echo -e "${GREEN}✔ Virtual environment created.${NC}"
else
    echo -e "${GREEN}✔ Virtual environment already exists.${NC}"
fi

# Activate venv
source venv/bin/activate

# 3. Dependencies
echo -e "${YELLOW}Installing/Updating dependencies...${NC}"
pip install --upgrade pip setuptools wheel > /dev/null
if [ -f "requirements.txt" ]; then
    pip install -r requirements.txt
    echo -e "${GREEN}✔ Dependencies installed.${NC}"
else
    echo -e "${RED}Error: requirements.txt not found!${NC}"
    exit 1
fi

# 4. Directories
echo -e "${YELLOW}Verifying directory structure...${NC}"
mkdir -p logs checkpoints data assets
echo -e "${GREEN}✔ Directory structure verified.${NC}"

# 5. Launch
echo ""
echo -e "${GREEN}==================================================${NC}"
echo -e "${GREEN}   SYSTEM READY - STARTING SERVICES               ${NC}"
echo -e "${GREEN}==================================================${NC}"
echo ""

if [ -f "dashboard.py" ]; then
    echo -e "${GREEN}Starting Dashboard...${NC}"
    streamlit run dashboard.py &
    DASHBOARD_PID=$!
    echo -e "${GREEN}✔ Dashboard started (PID: $DASHBOARD_PID)${NC}"
else
    echo -e "${RED}Error: dashboard.py not found.${NC}"
fi

echo ""
echo -e "${GREEN}Setup Complete.${NC}"
echo -e "To start training, run: ${YELLOW}python train_end_to_end.py${NC}"
