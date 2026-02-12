# ♟️ DistillZero Chess AI (formerly Archimedes)

<div align="center">

**A high-performance chess AI using Knowledge Distillation from Stockfish into a Deep ResNet.**

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-red.svg)](https://pytorch.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

[Features](#-features) • [Installation](#-installation) • [Quick Start](#-quick-start) • [Architecture](#-architecture) • [Dashboard](#-dashboard)

</div>

---

## 🎯 Features

### 🧠 DistillZero Architecture
- **Knowledge Distillation**: Learns from Stockfish (3500+ ELO) instead of training from zero.
- **Deep ResNet**: 10-block Residual Neural Network optimized for single-GPU convergence.
- **Batched MCTS**: Optimized Monte Carlo Tree Search with virtual loss and GPU batching (5x+ speedup).
- **Two-Phase Training**: Supervised Pretraining on 5M+ Stockfish positions → Self-Play RL Finetuning.

### 📊 Advanced Components
- **Prioritized Experience Replay**: Focuses learning on high TD-error positions.
- **Curriculum Learning**: Progresses through Stockfish levels (1350 → 1800 → 2200 ELO).
- **AlphaZero Encoding**: Standard 119-plane input representation with history support.
- **Reduced Action Space**: 1,968-output policy head for faster inference and lower VRAM.

### 🚀 Production Ready
- **AMP (Mixed Precision)**: Optimized for RTX 4090/3090 series.
- **TorchScript Support**: Traced models for low-latency inference.
- **Live Dashboard**: Real-time monitoring of Elo, training loss, and search depth.

---

## 🏗️ Architecture: ChessResNet

```
Input: 8x8x119 Tensor (AlphaZero planes)
    ↓
Conv Input (3x3, 128 channels) + BN + ReLU
    ↓
10x Residual Blocks (2x Conv 3x3 + BN + Skip)
    ↓
    ├─→ Policy Head
    │   ├─ Conv (1x1, 2 channels) + BN + ReLU
    │   └─ Linear(128, 1968) → Move Probabilities
    │
    └─→ Value Head
        ├─ Conv (1x1, 1 channel) + BN + ReLU
        ├─ Linear(64, 128) + ReLU
        └─ Linear(128, 1) → Tanh [-1, 1]
```

---

## 🚀 Quick Start

### 1. Generate Data (Factory)
```bash
python distillzero_factory.py --max-games 10000 --output distillzero_dataset.h5
```

### 2. Start Training
```bash
python train_end_to_end.py --h5 distillzero_dataset.h5 --epochs 100 --batch-size 64
```

### 3. Open Dashboard
```bash
streamlit run dashboard.py
```

---

## 📈 Expected Performance

| Metric | DistillZero | Archimedes (Old) | Improvement |
|--------|-------------|------------------|-------------|
| **ELO (100 Ep)** | ~2200 | ~1400 | +800 |
| **Data Efficiency**| 5M SF Positions | 5k Self-Play | 1000x |
| **Nodes/Second** | 15,000 | 3,000 | 5x |
| **Model Size** | 3.3M Params | 5.5M Params | 40% Smaller |

---

## 📄 License
MIT License. Made with ♟️ and 🧠.
