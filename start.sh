#!/bin/bash

# Archimedes Chess AI - Startup Script
# Installs dependencies and runs training + dashboard in parallel

set -e  # Exit on error

echo "=========================================="
echo "  ARCHIMEDES CHESS AI - STARTUP"
echo "=========================================="
echo ""

# Colors for output
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m' # No Color

# Check Python version
echo -e "${YELLOW}[1/6] Checking Python version...${NC}"
python_version=$(python3 --version 2>&1 | awk '{print $2}')
echo "Python version: $python_version"

required_version="3.8"
if [ "$(printf '%s\n' "$required_version" "$python_version" | sort -V | head -n1)" != "$required_version" ]; then
    echo -e "${RED}Error: Python 3.8+ required${NC}"
    exit 1
fi
echo -e "${GREEN}✓ Python version OK${NC}"
echo ""

# Check for GPU
echo -e "${YELLOW}[2/6] Checking for GPU...${NC}"
if command -v nvidia-smi &> /dev/null; then
    echo "GPU detected:"
    nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
    echo -e "${GREEN}✓ GPU available${NC}"
else
    echo -e "${YELLOW}⚠ No GPU detected - training will use CPU (slower)${NC}"
fi
echo ""

# Create virtual environment (optional but recommended)
echo -e "${YELLOW}[3/6] Setting up virtual environment...${NC}"
if [ ! -d "venv" ]; then
    python3 -m venv venv
    echo -e "${GREEN}✓ Virtual environment created${NC}"
else
    echo "Virtual environment already exists"
fi

# Activate virtual environment
source venv/bin/activate
echo -e "${GREEN}✓ Virtual environment activated${NC}"
echo ""

# Install dependencies
echo -e "${YELLOW}[4/6] Installing dependencies...${NC}"
echo "This may take a few minutes..."

# Upgrade pip
pip install --upgrade pip setuptools wheel > /dev/null 2>&1

# Install PyTorch (with CUDA if available)
if command -v nvidia-smi &> /dev/null; then
    echo "Installing PyTorch with CUDA support..."
    pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
else
    echo "Installing PyTorch (CPU only)..."
    pip install torch torchvision torchaudio
fi

# Install other requirements
echo "Installing other dependencies..."
pip install -r requirements.txt

echo -e "${GREEN}✓ All dependencies installed${NC}"
echo ""

# Create necessary directories
echo -e "${YELLOW}[5/6] Creating directories...${NC}"
mkdir -p checkpoints
mkdir -p logs
mkdir -p data
echo -e "${GREEN}✓ Directories created${NC}"
echo ""

# Setup ngrok (optional)
echo -e "${YELLOW}[6/6] Ngrok setup...${NC}"
echo "To access the dashboard remotely, you need an ngrok auth token."
echo "Get one from: https://dashboard.ngrok.com/get-started/your-authtoken"
echo ""
read -p "Do you have an ngrok auth token? (y/n): " has_token

if [ "$has_token" = "y" ] || [ "$has_token" = "Y" ]; then
    read -p "Enter your ngrok auth token: " ngrok_token
    ngrok config add-authtoken "$ngrok_token"
    echo -e "${GREEN}✓ Ngrok configured${NC}"
else
    echo -e "${YELLOW}⚠ Skipping ngrok setup - dashboard will only be accessible locally${NC}"
fi
echo ""

# Start training and dashboard
echo "=========================================="
echo "  STARTING DISTILLZERO"
echo "=========================================="
echo ""

# Parse command line arguments
EPOCHS=100
GAMES_PER_EPOCH=10
BATCH_SIZE=64
LEARNING_RATE=0.001
H5_PATH="distillzero_dataset.h5"

while [[ $# -gt 0 ]]; do
    case $1 in
        --epochs)
            EPOCHS="$2"
            shift 2
            ;;
        --games-per-epoch)
            GAMES_PER_EPOCH="$2"
            shift 2
            ;;
        --batch-size)
            BATCH_SIZE="$2"
            shift 2
            ;;
        --lr)
            LEARNING_RATE="$2"
            shift 2
            ;;
        --h5)
            H5_PATH="$2"
            shift 2
            ;;
        --dashboard-only)
            DASHBOARD_ONLY=true
            shift
            ;;
        --training-only)
            TRAINING_ONLY=true
            shift
            ;;
        *)
            echo "Unknown option: $1"
            exit 1
            ;;
    esac
done

echo "Configuration:"
echo "  Epochs: $EPOCHS"
echo "  Games per epoch: $GAMES_PER_EPOCH"
echo "  Batch size: $BATCH_SIZE"
echo "  Learning rate: $LEARNING_RATE"
echo ""

# Function to cleanup on exit
cleanup() {
    echo ""
    echo -e "${YELLOW}Shutting down...${NC}"
    kill $TRAINING_PID 2>/dev/null || true
    kill $DASHBOARD_PID 2>/dev/null || true
    echo -e "${GREEN}✓ Cleanup complete${NC}"
    exit 0
}

trap cleanup SIGINT SIGTERM

# Start training in background
if [ "$DASHBOARD_ONLY" != true ]; then
    echo -e "${GREEN}[*] Starting training...${NC}"
    python train_end_to_end.py \
        --epochs $EPOCHS \
        --games-per-epoch $GAMES_PER_EPOCH \
        --batch-size $BATCH_SIZE \
        --lr $LEARNING_RATE \
        --h5 $H5_PATH \
        > logs/training.log 2>&1 &
    TRAINING_PID=$!
    echo "Training PID: $TRAINING_PID"
    echo "Training logs: logs/training.log"
    echo ""
fi

# Wait a bit for training to initialize
sleep 5

# Start dashboard in background
if [ "$TRAINING_ONLY" != true ]; then
    echo -e "${GREEN}[*] Starting dashboard...${NC}"
    streamlit run dashboard.py --server.port 8501 --server.headless true > logs/dashboard.log 2>&1 &
    DASHBOARD_PID=$!
    echo "Dashboard PID: $DASHBOARD_PID"
    echo "Dashboard logs: logs/dashboard.log"
    echo ""
    
    # Wait for dashboard to start
    sleep 5
    
    echo "=========================================="
    echo -e "${GREEN}  ✓ DISTILLZERO IS RUNNING!${NC}"
    echo "=========================================="
    echo ""
    echo "📊 Dashboard: http://localhost:8501"
    echo ""
    
    if command -v ngrok &> /dev/null; then
        echo "🌐 Creating public URL with ngrok..."
        ngrok http 8501 > /dev/null &
        NGROK_PID=$!
        sleep 3
        
        # Get ngrok URL
        NGROK_URL=$(curl -s http://localhost:4040/api/tunnels | grep -o '"public_url":"[^"]*' | grep -o 'https://[^"]*' | head -1)
        if [ ! -z "$NGROK_URL" ]; then
            echo "🔗 Public URL: $NGROK_URL"
        fi
    fi
    
    echo ""
    echo "📝 Logs:"
    echo "  Training: logs/training.log"
    echo "  Dashboard: logs/dashboard.log"
    echo ""
    echo "Press Ctrl+C to stop"
    echo ""
fi

# Keep script running
if [ "$TRAINING_ONLY" != true ]; then
    wait $DASHBOARD_PID
else
    wait $TRAINING_PID
fi
