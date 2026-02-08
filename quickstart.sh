#!/bin/bash
# DistillZero Quick Start Script
# Runs all tests and generates a small test dataset

set -e

echo "╔════════════════════════════════════════════════════════════════╗"
echo "║                                                                ║"
echo "║              DistillZero - Quick Start Script                 ║"
echo "║                                                                ║"
echo "║  This script will:                                            ║"
echo "║  1. Check dependencies                                        ║"
echo "║  2. Run unit tests                                            ║"
echo "║  3. Generate a small test dataset (1000 positions)            ║"
echo "║  4. Analyze the dataset                                       ║"
echo "║                                                                ║"
echo "╚════════════════════════════════════════════════════════════════╝"
echo ""

# Check Python version
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "Step 1: Checking Python version..."
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"

PYTHON_VERSION=$(python3 --version 2>&1 | awk '{print $2}')
echo "✓ Python version: $PYTHON_VERSION"

# Check if dependencies are installed
echo ""
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "Step 2: Checking dependencies..."
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"

if ! python3 -c "import chess" 2>/dev/null; then
    echo "⚠ python-chess not found. Installing dependencies..."
    pip install -r requirements_dataset.txt
else
    echo "✓ Dependencies already installed"
fi

# Check for Stockfish
echo ""
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "Step 3: Checking for Stockfish..."
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"

if command -v stockfish &> /dev/null; then
    STOCKFISH_PATH=$(which stockfish)
    echo "✓ Stockfish found: $STOCKFISH_PATH"
else
    echo "⚠ Stockfish not found!"
    echo ""
    echo "Please install Stockfish:"
    echo "  • Ubuntu/Debian: sudo apt-get install stockfish"
    echo "  • Mac: brew install stockfish"
    echo "  • Windows: Download from https://stockfishchess.org/download/"
    echo ""
    read -p "Continue anyway? (y/N): " -n 1 -r
    echo
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        exit 1
    fi
    STOCKFISH_PATH="stockfish"
fi

# Run unit tests
echo ""
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "Step 4: Running unit tests..."
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"

python3 test_dataset_generator.py || {
    echo "❌ Tests failed!"
    exit 1
}

# Generate test dataset
echo ""
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "Step 5: Generating test dataset (1000 positions)..."
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo ""
echo "This will take 1-2 minutes..."
echo ""

python3 dataset_generator.py \
    --output quickstart_test.h5 \
    --positions 1000 \
    --workers 4 \
    --stockfish "$STOCKFISH_PATH" || {
    echo "❌ Dataset generation failed!"
    exit 1
}

# Analyze dataset
echo ""
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "Step 6: Analyzing dataset..."
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"

python3 visualize_dataset.py quickstart_test.h5 || {
    echo "⚠ Analysis failed, but dataset was created"
}

# Success!
echo ""
echo "╔════════════════════════════════════════════════════════════════╗"
echo "║                                                                ║"
echo "║                    ✅ QUICK START COMPLETE!                    ║"
echo "║                                                                ║"
echo "╚════════════════════════════════════════════════════════════════╝"
echo ""
echo "Test dataset created: quickstart_test.h5"
echo ""
echo "Next steps:"
echo ""
echo "1. Generate a larger dataset for training:"
echo "   python3 dataset_generator.py --output train.h5 --positions 1000000"
echo ""
echo "2. Download real Lichess data (optional but recommended):"
echo "   ./download_lichess_data.sh 2024 01"
echo ""
echo "3. Read the documentation:"
echo "   • README.md - Project overview"
echo "   • DATASET_README.md - Detailed dataset documentation"
echo "   • PHASE1_COMPLETE.md - What we've built so far"
echo ""
echo "4. Start Phase 2 (Neural Network Training):"
echo "   Coming next: chess_net.py and train.py"
echo ""
echo "════════════════════════════════════════════════════════════════"
