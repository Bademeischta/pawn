#!/bin/bash
# Download Lichess databases for high-quality training data
# Usage: ./download_lichess_data.sh [year] [month]

set -e

YEAR=${1:-2024}
MONTH=${2:-01}
DATA_DIR="lichess_data"

echo "=================================================="
echo "Lichess Data Downloader for DistillZero"
echo "=================================================="
echo "Year: $YEAR"
echo "Month: $MONTH"
echo "Output directory: $DATA_DIR"
echo "=================================================="

# Create data directory
mkdir -p "$DATA_DIR"
cd "$DATA_DIR"

# Download standard rated games
echo ""
echo "[1/3] Downloading standard rated games..."
GAMES_FILE="lichess_db_standard_rated_${YEAR}-${MONTH}.pgn.bz2"
GAMES_URL="https://database.lichess.org/standard/$GAMES_FILE"

if [ -f "$GAMES_FILE" ]; then
    echo "✓ File already exists: $GAMES_FILE"
else
    echo "Downloading from: $GAMES_URL"
    wget -c "$GAMES_URL" || {
        echo "❌ Failed to download games database"
        echo "Check if the URL is valid: $GAMES_URL"
        exit 1
    }
    echo "✓ Downloaded: $GAMES_FILE"
fi

# Download puzzle database
echo ""
echo "[2/3] Downloading puzzle database..."
PUZZLE_FILE="lichess_db_puzzle.csv.bz2"
PUZZLE_URL="https://database.lichess.org/$PUZZLE_FILE"

if [ -f "$PUZZLE_FILE" ]; then
    echo "✓ File already exists: $PUZZLE_FILE"
else
    echo "Downloading from: $PUZZLE_URL"
    wget -c "$PUZZLE_URL" || {
        echo "❌ Failed to download puzzle database"
        exit 1
    }
    echo "✓ Downloaded: $PUZZLE_FILE"
fi

# Decompress files (optional - can work with compressed files directly)
echo ""
echo "[3/3] Decompressing files (optional, can be skipped)..."
read -p "Decompress files? This will use ~10-20GB disk space. (y/N): " -n 1 -r
echo
if [[ $REPLY =~ ^[Yy]$ ]]; then
    if [ -f "$GAMES_FILE" ] && [ ! -f "${GAMES_FILE%.bz2}" ]; then
        echo "Decompressing games..."
        bunzip2 -k "$GAMES_FILE"
        echo "✓ Decompressed: ${GAMES_FILE%.bz2}"
    fi
    
    if [ -f "$PUZZLE_FILE" ] && [ ! -f "${PUZZLE_FILE%.bz2}" ]; then
        echo "Decompressing puzzles..."
        bunzip2 -k "$PUZZLE_FILE"
        echo "✓ Decompressed: ${PUZZLE_FILE%.bz2}"
    fi
else
    echo "Skipped decompression. Python can read .bz2 files directly."
fi

# Show statistics
echo ""
echo "=================================================="
echo "Download Complete!"
echo "=================================================="
echo "Files in $DATA_DIR:"
ls -lh
echo ""
echo "Estimated positions:"
echo "  - Games database: ~50-100M positions"
echo "  - Puzzle database: ~3-4M positions"
echo ""
echo "Next steps:"
echo "1. Update dataset_generator.py to use these files"
echo "2. Run: python dataset_generator.py --positions 10000000"
echo "=================================================="
