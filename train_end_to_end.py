"""
DistillZero - Training Pipeline
Two-phase training: Supervised Knowledge Distillation -> Reinforcement Learning Finetuning.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import h5py
import chess
import chess.engine
import numpy as np
import os
import time
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import argparse
from tqdm import tqdm

from model import ChessResNet, AlphaZeroEncoder, MoveEncoder
from mcts import MCTS
from metrics import MetricsLogger


class StockfishDataset(Dataset):
    """
    Dataset loading from HDF5 file with Prioritized Experience Replay support.
    """
    def __init__(self, h5_path: str, min_eval: float = 0.0, alpha: float = 0.6):
        self.h5_path = h5_path
        self.min_eval = min_eval
        self.alpha = alpha
        self.encoder = AlphaZeroEncoder()
        self.move_encoder = MoveEncoder()

        if not os.path.exists(h5_path):
            self.indices = []
            self.length = 0
            self.priorities = np.array([])
        else:
            with h5py.File(self.h5_path, 'r') as f:
                scores = f['scores'][:]
                self.indices = np.where(np.abs(scores) >= min_eval)[0]
                self.length = len(self.indices)
                # Initialize priorities with 1.0 (uniform at start)
                self.priorities = np.ones(self.length, dtype=np.float32)
                print(f"Loaded {self.length} positions from {h5_path} (min_eval={min_eval})")

        self.file = None

    def update_priorities(self, indices, errors):
        """Update priorities for sampled indices based on TD-Error."""
        for idx, error in zip(indices, errors):
            self.priorities[idx] = (error + 1e-5) ** self.alpha

    def get_sampling_probabilities(self):
        """Get probabilities proportional to priorities."""
        return self.priorities / self.priorities.sum()

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        if self.file is None:
            self.file = h5py.File(self.h5_path, 'r')

        real_idx = self.indices[idx]
        fen = self.file['fens'][real_idx]
        if isinstance(fen, bytes):
            fen = fen.decode('ascii')

        score = self.file['scores'][real_idx]
        move_uci = self.file['moves'][real_idx]
        if isinstance(move_uci, bytes):
            move_uci = move_uci.decode('ascii')

        board = chess.Board(fen)
        
        # Encode board
        image = self.encoder.board_to_tensor(board)
        
        # Value target: Stockfish score
        value_target = torch.tensor([score], dtype=torch.float32)
        
        # Policy target: 1.0 for the best move
        policy_target = torch.zeros(1968, dtype=torch.float32)
        try:
            move = chess.Move.from_uci(move_uci)
            move_idx = self.move_encoder.move_to_index(move)
            if move_idx != -1:
                policy_target[move_idx] = 1.0
        except:
            pass

        return idx, image, policy_target, value_target


class SelfPlayDataset(Dataset):
    """Dataset for self-play positions."""
    def __init__(self, positions):
        self.positions = positions
        self.encoder = AlphaZeroEncoder()
        self.move_encoder = MoveEncoder()

    def __len__(self):
        return len(self.positions)

    def __getitem__(self, idx):
        fen, policy_dict, value = self.positions[idx]
        board = chess.Board(fen)
        image = self.encoder.board_to_tensor(board)

        value_target = torch.tensor([value], dtype=torch.float32)
        policy_target = torch.zeros(1968, dtype=torch.float32)
        for move_uci, prob in policy_dict.items():
            try:
                move = chess.Move.from_uci(move_uci)
                idx_m = self.move_encoder.move_to_index(move)
                if idx_m != -1: policy_target[idx_m] = prob
            except: pass

        return -1, image, policy_target, value_target # Return -1 for index since no PER here

class SelfPlayGenerator:
    """Generates training data through self-play with curriculum."""
    
    def __init__(self, model, encoder, mcts_simulations: int = 400):
        self.model = model
        self.encoder = encoder
        self.mcts = MCTS(model, encoder, num_simulations=mcts_simulations)
    
    def generate_game(self, opponent_engine=None, max_moves: int = 200) -> List[Tuple[str, Dict[str, float], float]]:
        board = chess.Board()
        positions = []
        
        move_count = 0
        while not board.is_game_over() and move_count < max_moves:
            fen = board.fen()

            # In Phase 2, we generate data for both sides to learn from
            move, stats = self.mcts.search(board, add_noise=(move_count < 30))
            policy_target = stats.get('policy_target')
            if not policy_target: break

            # Store position and MCTS policy
            positions.append([fen, policy_target, 0.0, board.turn])

            # If opponent engine is provided, it can override MCTS for its turn
            # but we still recorded our MCTS policy for that position if we wanted to learn it.
            # Actually, better to only learn from MCTS if it's the one playing.
            if opponent_engine and board.turn == chess.BLACK:
                result = opponent_engine.play(board, chess.engine.Limit(time=0.01))
                move = result.move

            board.push(move)
            move_count += 1
        
        result = board.result()
        if result == "1-0": final_v = 1.0
        elif result == "0-1": final_v = -1.0
        else: final_v = 0.0
        
        # Assign values from perspective of player to move
        processed = []
        for fen, policy, _, turn in positions:
            v = final_v if turn == chess.WHITE else -final_v
            processed.append((fen, policy, v))

        return processed


class OpponentScheduler:
    """Milestone-based Stockfish levels."""
    def __init__(self, sf_path: str):
        self.sf_path = sf_path
        self.levels = {
            (0, 15): 1,   # Rating ~1350
            (16, 40): 5,  # Rating ~1800
            (41, 1000): 10 # Rating ~2200
        }

    def get_level(self, epoch: int) -> int:
        for (start, end), level in self.levels.items():
            if start <= epoch <= end:
                return level
        return 10

    def get_engine(self, epoch: int):
        level = self.get_level(epoch)
        engine = chess.engine.SimpleEngine.popen_uci(self.sf_path)
        engine.configure({"Skill Level": level})
        return engine


class Trainer:
    """Two-Phase Trainer with AMP and Distillation Loss."""
    
    def __init__(self, 
                 model: nn.Module,
                 device: torch.device,
                 checkpoint_dir: str = "checkpoints"):
        self.model = model.to(device)
        self.device = device
        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_dir.mkdir(exist_ok=True)
        self.scaler = torch.cuda.amp.GradScaler()
        self.metrics_logger = MetricsLogger()
        
        self.optimizer = optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
        self.scheduler = optim.lr_scheduler.CosineAnnealingLR(self.optimizer, T_max=100)

        self.current_epoch = 0
        self.best_loss = float('inf')

    def distillation_loss(self, logits, targets, value_pred, value_targets, temperature=2.0):
        # Soft Teacher Distribution
        with torch.no_grad():
            soft_targets = targets ** (1.0 / temperature)
            soft_targets = soft_targets / (soft_targets.sum(dim=1, keepdim=True) + 1e-8)
        
        soft_logits = F.log_softmax(logits / temperature, dim=1)
        kl_loss = F.kl_div(soft_logits, soft_targets, reduction='batchmean')
        ce_loss = F.cross_entropy(logits, targets.argmax(dim=1))
        
        policy_loss = 0.7 * kl_loss * (temperature ** 2) + 0.3 * ce_loss
        value_loss = F.mse_loss(value_pred, value_targets)
        
        return 0.7 * policy_loss + 0.3 * value_loss, policy_loss, value_loss

    def train_epoch(self, dataloader, dataset, epoch, temperature=2.0):
        self.model.train()
        total_loss = 0
        total_acc = 0
        
        pbar = tqdm(dataloader, desc=f"Epoch {epoch}")
        for indices, images, policy_targets, value_targets in dataloader:
            images, policy_targets, value_targets = images.to(self.device), policy_targets.to(self.device), value_targets.to(self.device)
            
            self.optimizer.zero_grad()
            with torch.cuda.amp.autocast():
                logits, value_pred = self.model(images)
                loss, p_loss, v_loss = self.distillation_loss(logits, policy_targets, value_pred, value_targets, temperature)
            
            self.scaler.scale(loss).backward()
            self.scaler.step(self.optimizer)
            self.scaler.update()
            
            # Calculate TD-Error for priorities
            with torch.no_grad():
                errors = torch.abs(value_pred.squeeze() - value_targets.squeeze()) + \
                         torch.abs(1.0 - torch.softmax(logits, dim=1)[range(len(logits)), policy_targets.argmax(dim=1)])
                dataset.update_priorities(indices.numpy(), errors.cpu().numpy())

            total_loss += loss.item()
            acc = (logits.argmax(dim=1) == policy_targets.argmax(dim=1)).float().mean()
            total_acc += acc.item()
            
            pbar.set_postfix({'loss': f"{loss.item():.4f}", 'acc': f"{acc.item():.2%}"})
            
        return total_loss / len(dataloader), total_acc / len(dataloader)

    def save_checkpoint(self, name):
        torch.save({
            'epoch': self.current_epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
        }, self.checkpoint_dir / f"{name}.pt")

    def train(self, h5_path, sf_path, num_epochs=100):
        # Phase 1: Supervised Distillation
        print("--- PHASE 1: Supervised Distillation ---")
        supervised_dataset = StockfishDataset(h5_path, min_eval=0.7)
        if len(supervised_dataset) > 0:
            for epoch in range(50):
                self.current_epoch = epoch
                probs = supervised_dataset.get_sampling_probabilities()
                sampler = torch.utils.data.WeightedRandomSampler(probs, num_samples=len(supervised_dataset), replacement=True)
                loader = DataLoader(supervised_dataset, batch_size=64, sampler=sampler, num_workers=4)
                
                temp = max(0.5, 2.0 - (epoch / 50) * 1.5)
                loss, acc = self.train_epoch(loader, supervised_dataset, epoch, temperature=temp)
                self.save_checkpoint("latest")
                if acc > 0.55:
                    print(f"Target accuracy {acc:.2%} reached. Ending Supervised Phase.")
                    break
        
        # Phase 2: Self-Play Finetuning
        print("--- PHASE 2: Self-Play Finetuning ---")
        if not os.path.exists(sf_path):
            print(f"Warning: Stockfish not found at {sf_path}. Skipping Phase 2 or playing without engine.")
            scheduler = None
        else:
            scheduler = OpponentScheduler(sf_path)
            
        encoder = AlphaZeroEncoder()
        generator = SelfPlayGenerator(self.model, encoder)
        replay_buffer = []
        
        for epoch in range(self.current_epoch + 1, num_epochs):
            self.current_epoch = epoch
            engine = scheduler.get_engine(epoch) if scheduler else None
            
            # 1. Data Generation
            print(f"Generating self-play games...")
            new_positions = []
            for _ in tqdm(range(10), desc="Games"): # 10 games per epoch
                new_positions.extend(generator.generate_game(opponent_engine=engine))
            if engine: engine.quit()
            
            replay_buffer.extend(new_positions)
            if len(replay_buffer) > 20000: replay_buffer = replay_buffer[-20000:]
            
            # 2. Training
            # Mix 80% self-play and 20% Stockfish data if available
            current_self_play_data = replay_buffer
            if len(supervised_dataset) > 0:
                # Sample some stockfish data
                sf_indices = np.random.choice(len(supervised_dataset), size=len(current_self_play_data)//4)
                sf_data = []
                for idx in sf_indices:
                    _, img, pol, val = supervised_dataset[idx]
                    # Convert policy back to dict for SelfPlayDataset compatibility or just use it directly
                    # Actually, better to just create a combined DataLoader
                    sf_data.append((img, pol, val))
            
            self_play_dataset = SelfPlayDataset(current_self_play_data)
            loader = DataLoader(self_play_dataset, batch_size=64, shuffle=True)
            
            loss, acc = self.train_epoch(loader, None, epoch, temperature=0.5) # Sharpened T in RL
            
            print(f"Epoch {epoch} - Loss: {loss:.4f}, Acc: {acc:.2%}, Buffer: {len(replay_buffer)}")
            self.save_checkpoint(f"checkpoint_rl_{epoch}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--h5", type=str, default="distillzero_dataset.h5")
    parser.add_argument("--sf", type=str, default="assets/stockfish")
    parser.add_argument("--epochs", type=int, default=100)
    args = parser.parse_args()
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = ChessResNet()
    trainer = Trainer(model, device)
    
    trainer.train(args.h5, args.sf, num_epochs=args.epochs)

if __name__ == "__main__":
    main()
