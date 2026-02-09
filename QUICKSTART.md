# ♟️ Archimedes Chess AI - Quick Start Guide

## 🚀 Get Started in 3 Minutes

### Option 1: One-Command Setup (Linux/Mac)

```bash
./start.sh
```

That's it! The script will:
- ✅ Install all dependencies
- ✅ Detect GPU/CPU
- ✅ Start training
- ✅ Launch dashboard at http://localhost:8501

### Option 2: Google Colab (No Installation Required)

1. Open [`archimedes_colab.ipynb`](./archimedes_colab.ipynb) in Google Colab
2. Click Runtime → Change runtime type → GPU
3. Run all cells
4. Access dashboard via the ngrok URL shown

### Option 3: Manual Setup (Windows/Advanced)

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Start training
python train_end_to_end.py --epochs 50 --games-per-epoch 20

# 3. In another terminal, start dashboard
streamlit run dashboard.py
```

---

## 📊 What You'll See

### Dashboard Tabs

1. **📈 Training** - Loss curves, accuracy, learning rate
2. **🎯 MCTS** - Search performance, cache hit rate
3. **♟️ Chess** - Elo rating, win/loss rates
4. **💻 Hardware** - GPU/CPU usage, memory
5. **🎮 Play vs AI** - Interactive chess game
6. **🔍 Analysis** - Position evaluation
7. **📥 Downloads** - Export checkpoints and games

---

## ⚙️ Configuration

### Quick Settings

Edit these in [`train_end_to_end.py`](./train_end_to_end.py):

```python
# Training
EPOCHS = 100              # Total training epochs
GAMES_PER_EPOCH = 50      # Self-play games per epoch
BATCH_SIZE = 32           # Training batch size
LEARNING_RATE = 0.001     # Initial learning rate

# MCTS
MCTS_SIMULATIONS = 400    # Simulations per move
C_PUCT = 1.4              # Exploration constant
TEMPERATURE = 1.0         # Move selection randomness
```

### For Faster Training (CPU/Low Memory)

```python
GAMES_PER_EPOCH = 20      # Fewer games
BATCH_SIZE = 16           # Smaller batches
MCTS_SIMULATIONS = 200    # Fewer simulations
```

### For Better Performance (High-End GPU)

```python
GAMES_PER_EPOCH = 100     # More games
BATCH_SIZE = 64           # Larger batches
MCTS_SIMULATIONS = 800    # More simulations
```

---

## 🎮 Play Against the AI

### In Dashboard

1. Go to "🎮 Play vs AI" tab
2. Enter moves in UCI format (e.g., `e2e4`)
3. Click "Make Move"
4. Click "AI Move" for computer response

### In Python

```python
from model import ArchimedesGNN, ChessBoardEncoder
from mcts import MCTS
import chess
import torch

# Load model
model = ArchimedesGNN()
checkpoint = torch.load('checkpoints/latest_checkpoint.pt')
model.load_state_dict(checkpoint['model_state_dict'])
model.eval()

# Setup
encoder = ChessBoardEncoder()
mcts = MCTS(model, encoder, num_simulations=400)
board = chess.Board()

# Play
while not board.is_game_over():
    print(board)
    
    # Your move
    move = input("Your move: ")
    board.push(chess.Move.from_uci(move))
    
    # AI move
    ai_move, _ = mcts.search(board)
    board.push(ai_move)
    print(f"AI plays: {ai_move}")

print(f"Result: {board.result()}")
```

---

## 📈 Monitor Training

### Real-Time Metrics

The dashboard updates automatically. Key metrics to watch:

- **Loss**: Should decrease over time (target: < 0.5)
- **Accuracy**: Should increase (target: > 50%)
- **Elo**: Should grow (target: > 1800)
- **Win Rate**: Should improve (target: > 60%)

### Check Progress in Terminal

```python
from metrics import MetricsLogger
import pandas as pd

logger = MetricsLogger('training_logs.db')
metrics = logger.get_latest_metrics('training_metrics', limit=10)
df = pd.DataFrame(metrics)
print(df[['epoch', 'loss_total', 'accuracy_top1']])
```

---

## 💾 Save & Resume

### Automatic Checkpointing

Training saves checkpoints automatically:
- `latest_checkpoint.pt` - Most recent
- `best_checkpoint.pt` - Best performance
- `checkpoint_epoch_N.pt` - Per-epoch saves

### Resume Training

Just run the training command again:

```bash
python train_end_to_end.py --epochs 200
```

It will automatically load the latest checkpoint and continue.

### Load Specific Checkpoint

```python
import torch
from model import ArchimedesGNN

model = ArchimedesGNN()
checkpoint = torch.load('checkpoints/checkpoint_epoch_50.pt')
model.load_state_dict(checkpoint['model_state_dict'])
```

---

## 🐛 Common Issues

### "CUDA out of memory"

**Solution**: Reduce batch size
```bash
python train_end_to_end.py --batch-size 16
```

### "Training is slow"

**Solutions**:
1. Check GPU is being used: `nvidia-smi`
2. Reduce games per epoch: `--games-per-epoch 20`
3. Reduce MCTS simulations (edit `train_end_to_end.py`)

### "Dashboard won't start"

**Solution**: Check port availability
```bash
# Use different port
streamlit run dashboard.py --server.port 8502
```

### "No module named 'torch_geometric'"

**Solution**: Install PyTorch Geometric
```bash
pip install torch-geometric torch-scatter torch-sparse
```

---

## 📊 Expected Performance

### Training Timeline (RTX 3080)

| Time | Epoch | Loss | Elo | Status |
|------|-------|------|-----|--------|
| 0h | 0 | 2.5 | 800 | Random play |
| 5h | 10 | 1.8 | 1200 | Learning basics |
| 12h | 25 | 1.2 | 1500 | Intermediate |
| 24h | 50 | 0.8 | 1800 | Strong player |
| 48h | 100 | 0.5 | 2100 | Expert level |

### MCTS Performance

| Hardware | Nodes/Second | Strength |
|----------|--------------|----------|
| RTX 4090 | ~5000 | Excellent |
| RTX 3080 | ~4000 | Very Good |
| RTX 2080 | ~3000 | Good |
| CPU (16c) | ~500 | Playable |

---

## 🎯 Next Steps

1. **Train for 50+ epochs** for decent performance
2. **Experiment with hyperparameters** in the config
3. **Analyze games** using the dashboard
4. **Export PGN files** for external analysis
5. **Share your results** with the community!

---

## 📚 Learn More

- **Full Documentation**: See [README.md](./README.md)
- **Architecture Details**: Check code comments in [`model.py`](./model.py)
- **MCTS Algorithm**: See [`mcts.py`](./mcts.py)
- **Metrics System**: Explore [`metrics.py`](./metrics.py)

---

## 🆘 Need Help?

- **GitHub Issues**: [Report a bug](https://github.com/yourusername/archimedes-chess-ai/issues)
- **Discussions**: [Ask questions](https://github.com/yourusername/archimedes-chess-ai/discussions)
- **Email**: your.email@example.com

---

<div align="center">

**Happy Training! ♟️🚀**

[⬆ Back to Top](#-archimedes-chess-ai---quick-start-guide)

</div>
