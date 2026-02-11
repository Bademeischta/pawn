# ♟️ DistillZero Quick Start Guide

## 🚀 3 Steps to 2200 ELO

### 1. Build the Dataset
DistillZero needs data from Stockfish.
```bash
python distillzero_factory.py --max-games 5000
```
This downloads Stockfish, streams Lichess PGNs, and evaluates positions.

### 2. Train the Network
Phase 1: Supervised (Stockfish Labels) → Phase 2: Self-Play RL.
```bash
python train_end_to_end.py --h5 distillzero_dataset.h5
```
Watch for **Phase 1 ending** when Accuracy hits >55%.

### 3. Play & Monitor
```bash
streamlit run dashboard.py
```
Use the "Play vs AI" tab to test the model.

---

## ⚙️ Key Configuration (model.py)
- **128 Channels**: Optimized for speed.
- **10 Blocks**: Depth for tactical depth.
- **1,968 Outputs**: Accurate move prediction.

## 📊 Monitoring
- **Loss**: Total loss should trend toward 0.3-0.5.
- **ELO**: Estimated Elo is logged every 5 epochs.
- **NPS**: Nodes Per Second (should be >10k on GPU).
