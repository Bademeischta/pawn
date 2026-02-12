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
import logging
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import argparse
from tqdm import tqdm

from model import ChessResNet, AlphaZeroEncoder, MoveEncoder, get_move_encoder
from mcts import MCTS
from metrics import MetricsLogger
from sumtree import SumTree
from utils import safe_load_checkpoint, safe_save, setup_logging

logger = logging.getLogger(__name__)


class PrioritizedSampler(torch.utils.data.Sampler):
    """Sampler that uses a SumTree for O(log n) prioritized sampling."""
    def __init__(self, sumtree: SumTree, num_samples: int):
        self.sumtree = sumtree
        self.num_samples = num_samples

    def __iter__(self):
        for _ in range(self.num_samples):
            # Sample from [0, total_priority]
            v = np.random.uniform(0, self.sumtree.total_priority)
            _, _, data_idx = self.sumtree.get_leaf(v)
            yield int(data_idx)

    def __len__(self):
        return self.num_samples


class StockfishDataset(Dataset):
    """
    Dataset loading from HDF5 file with Prioritized Experience Replay support.
    Loads data into memory for performance and thread-safety.
    """
    def __init__(self, h5_path: str, min_eval: float = 0.0, alpha: float = 0.6, label_smoothing: float = 0.1):
        self.h5_path = h5_path
        self.min_eval = min_eval
        self.alpha = alpha
        self.label_smoothing = label_smoothing
        self.encoder = AlphaZeroEncoder()
        self.move_encoder = get_move_encoder()
        self.unmapped_moves = {}

        if not os.path.exists(h5_path):
            self.fens = []
            self.scores = []
            self.moves = []
            self.length = 0
            self.sumtree = None
        else:
            logger.info(f"Loading Stockfish dataset into memory: {h5_path}")
            with h5py.File(self.h5_path, 'r') as f:
                all_scores = f['scores'][:]
                mask = np.abs(all_scores) >= min_eval
                self.fens = f['fens'][mask]
                self.scores = all_scores[mask]
                self.moves = f['moves'][mask]

                # Convert bytes to strings if necessary
                if len(self.fens) > 0 and isinstance(self.fens[0], bytes):
                    self.fens = [f.decode('ascii') for f in self.fens]
                if len(self.moves) > 0 and isinstance(self.moves[0], bytes):
                    self.moves = [m.decode('ascii') for m in self.moves]

                self.length = len(self.fens)
                # Use SumTree for O(log n) sampling
                self.sumtree = SumTree(self.length)
                # Initialize with uniform priority 1.0
                for i in range(self.length):
                    self.sumtree.update(i, 1.0)
                logger.info(f"Loaded {self.length} positions (min_eval={min_eval})")

    def update_priorities(self, indices, errors):
        """Update priorities for sampled indices based on TD-Error."""
        if self.sumtree is None: return
        for idx, error in zip(indices, errors):
            if idx == -1: continue # Skip if not from PER
            priority = (error + 1e-5) ** self.alpha
            self.sumtree.update(idx, priority)

    def get_sampler(self, num_samples: Optional[int] = None):
        """Returns an O(log n) PrioritizedSampler."""
        if self.sumtree is None: return None
        n = num_samples or self.length
        return PrioritizedSampler(self.sumtree, n)

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        fen = self.fens[idx]
        score = self.scores[idx]
        move_uci = self.moves[idx]

        board = chess.Board(fen)
        image = self.encoder.board_to_tensor(board)
        value_target = torch.tensor([score], dtype=torch.float32)
        
        # Policy target with Label Smoothing
        policy_target = torch.ones(1968, dtype=torch.float32) * (self.label_smoothing / 1967)
        try:
            move = chess.Move.from_uci(move_uci)
            move_idx = self.move_encoder.move_to_index(move)
            if move_idx != -1:
                policy_target[move_idx] = 1.0 - self.label_smoothing
            else:
                self.unmapped_moves[move_uci] = self.unmapped_moves.get(move_uci, 0) + 1
        except Exception as e:
            pass

        return idx, image, policy_target, value_target


class ReplayBuffer:
    """Experience buffer for self-play games with PER."""
    def __init__(self, capacity: int = 100000, alpha: float = 0.6):
        self.capacity = capacity
        self.alpha = alpha
        self.sumtree = SumTree(capacity)
        self.buffer = []
        self.pos = 0

    def add(self, fen, policy_dict, value):
        data = (fen, policy_dict, value)
        if len(self.buffer) < self.capacity:
            self.buffer.append(data)
        else:
            self.buffer[self.pos] = data

        max_p = self.sumtree.max_priority if len(self.buffer) > 1 else 1.0
        self.sumtree.update(self.pos, max_p)
        self.pos = (self.pos + 1) % self.capacity

    def update_priorities(self, indices, errors):
        for idx, error in zip(indices, errors):
            priority = (error + 1e-5) ** self.alpha
            self.sumtree.update(idx, priority)

    def __len__(self):
        return len(self.buffer)


class CombinedDataset(Dataset):
    """Dataset for mixing self-play positions and Stockfish data with PER support for both."""
    def __init__(self, stockfish_dataset, replay_buffer):
        self.sf_dataset = stockfish_dataset
        self.replay_buffer = replay_buffer
        self.encoder = AlphaZeroEncoder()
        self.move_encoder = get_move_encoder()

    def __len__(self):
        return len(self.sf_dataset) + len(self.replay_buffer)

    def __getitem__(self, idx):
        if idx < len(self.sf_dataset):
            # Stockfish sample
            real_idx, image, policy, value = self.sf_dataset[idx]
            return real_idx, image, policy, value
        else:
            # Replay buffer sample
            sp_idx = idx - len(self.sf_dataset)
            fen, policy_dict, value = self.replay_buffer.buffer[sp_idx]

            board = chess.Board(fen)
            image = self.encoder.board_to_tensor(board)
            value_target = torch.tensor([value], dtype=torch.float32)
            policy_target = torch.zeros(1968, dtype=torch.float32)
            for move_uci, prob in policy_dict.items():
                try:
                    move = chess.Move.from_uci(move_uci)
                    idx_m = self.move_encoder.move_to_index(move)
                    if idx_m != -1: policy_target[idx_m] = prob
                    else:
                        logger.debug(f"Unmapped move in self-play: {move_uci}")
                except Exception as e:
                    logger.error(f"Failed to parse self-play move {move_uci}: {e}")

            # Use offset for SP indices to distinguish from SF indices
            return 100000000 + sp_idx, image, policy_target, value_target

    def update_priorities(self, indices, errors):
        sf_indices, sf_errors = [], []
        sp_indices, sp_errors = [], []

        for idx, error in zip(indices, errors):
            if idx >= 100000000:
                sp_indices.append(idx - 100000000)
                sp_errors.append(error)
            elif idx != -1:
                sf_indices.append(idx)
                sf_errors.append(error)

        if sf_indices:
            self.sf_dataset.update_priorities(sf_indices, sf_errors)
        if sp_indices:
            self.replay_buffer.update_priorities(sp_indices, sp_errors)

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
                 checkpoint_dir: str = "checkpoints",
                 db_path: str = "logs/training_logs.db"):
        self.model = model.to(device)
        self.device = device
        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_dir.mkdir(exist_ok=True)
        self.scaler = torch.cuda.amp.GradScaler()
        self.metrics_logger = MetricsLogger(db_path=db_path)
        
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

    def train_epoch(self, dataloader: DataLoader, dataset: Optional[Dataset], epoch: int, temperature: float = 2.0) -> Tuple[float, float]:
        self.model.train()
        total_loss = 0.0
        total_acc = 0.0
        
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
            
            # Calculate TD-Error for priorities (only if dataset supports it)
            if dataset is not None:
                with torch.no_grad():
                    errors = torch.abs(value_pred.squeeze() - value_targets.squeeze()) + \
                            torch.abs(1.0 - torch.softmax(logits, dim=1)[range(len(logits)), policy_targets.argmax(dim=1)])
                    dataset.update_priorities(indices.numpy(), errors.cpu().numpy())

            total_loss += loss.item()
            acc = (logits.argmax(dim=1) == policy_targets.argmax(dim=1)).float().mean()
            total_acc += acc.item()
            
            pbar.set_postfix({'loss': f"{loss.item():.4f}", 'acc': f"{acc.item():.2%}"})
            
        return total_loss / len(dataloader), total_acc / len(dataloader)

    def save_checkpoint(self, name, replay_buffer=None):
        checkpoint_path = self.checkpoint_dir / f"{name}.pt"

        # Post-save validation: Check for NaN/Inf in weights
        valid = True
        for p in self.model.parameters():
            if torch.isnan(p).any() or torch.isinf(p).any():
                logger.warning(f"NaN/Inf detected in model weights. Checkpoint {name} may be corrupted.")
                valid = False
                break

        if not valid: return False

        data = {
            'epoch': self.current_epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
        }
        if replay_buffer is not None:
            data['replay_buffer'] = replay_buffer

        safe_save(data, checkpoint_path)

        # Also keep a 'latest.pt' for easy resume
        if name != "latest":
            latest_path = self.checkpoint_dir / "latest.pt"
            safe_save(data, latest_path)

        return True

    def load_latest_checkpoint(self):
        latest_path = self.checkpoint_dir / "latest.pt"
        if latest_path.exists():
            logger.info(f"Resuming from latest checkpoint: {latest_path}")
            checkpoint = safe_load_checkpoint(latest_path, self.device)
            self.model.load_state_dict(checkpoint['model_state_dict'])
            self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            if 'scheduler_state_dict' in checkpoint:
                self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
            self.current_epoch = checkpoint['epoch'] + 1
            return checkpoint.get('replay_buffer', None)
        return None

    def evaluate_against_stockfish(self, sf_path, num_games=10):
        """Evaluate current model against Stockfish with path validation."""
        sf_path_obj = Path(sf_path).resolve()
        if not sf_path_obj.exists():
            logger.warning(f"Stockfish not found at {sf_path}")
            return 0.0

        # Basic security check: ensure it's an executable file
        if not os.access(sf_path_obj, os.X_OK):
            logger.error(f"Stockfish path {sf_path} is not executable")
            return 0.0

        from mcts import MCTS
        mcts = MCTS(self.model, AlphaZeroEncoder(), num_simulations=400)
        engine = chess.engine.SimpleEngine.popen_uci(sf_path)
        engine.configure({"Skill Level": 10})

        wins = 0
        draws = 0

        for i in range(num_games):
            board = chess.Board()
            # Randomize playing color
            model_color = chess.WHITE if i % 2 == 0 else chess.BLACK

            while not board.is_game_over():
                if board.turn == model_color:
                    move, _ = mcts.search(board, add_noise=False)
                else:
                    result = engine.play(board, chess.engine.Limit(time=0.05))
                    move = result.move
                board.push(move)

            res = board.result()
            if (res == "1-0" and model_color == chess.WHITE) or (res == "0-1" and model_color == chess.BLACK):
                wins += 1
            elif res == "1/2-1/2":
                draws += 1

        engine.quit()
        win_rate = (wins + 0.5 * draws) / num_games
        logger.info(f"Evaluation against Stockfish: Win Rate = {win_rate:.2%}")
        return win_rate

    def train(self, h5_path, sf_path, num_epochs=100, batch_size=64, games_per_epoch=10, initial_replay_buffer=None):
        # Phase 1: Supervised Distillation
        logger.info("--- PHASE 1: Supervised Distillation ---")
        supervised_dataset = StockfishDataset(h5_path, min_eval=0.7)
        if len(supervised_dataset) > 0 and self.current_epoch < 50:
            start_ep = self.current_epoch
            for epoch in range(start_ep, 50):
                self.current_epoch = epoch
                sampler = supervised_dataset.get_sampler()
                loader = DataLoader(supervised_dataset, batch_size=batch_size, sampler=sampler, num_workers=4)
                
                temp = max(0.5, 2.0 - (epoch / 50) * 1.5)
                loss, acc = self.train_epoch(loader, supervised_dataset, epoch, temperature=temp)
                self.save_checkpoint("latest")

                # Audit unmapped moves
                if supervised_dataset.unmapped_moves:
                    logger.warning(f"Unmapped moves in epoch {epoch}: {len(supervised_dataset.unmapped_moves)}")

                if acc > 0.55:
                    logger.info(f"Target accuracy {acc:.2%} reached. Ending Supervised Phase.")
                    break
        
        # Phase 2: Self-Play Finetuning
        logger.info("--- PHASE 2: Self-Play Finetuning ---")
        if not os.path.exists(sf_path):
            logger.warning(f"Stockfish not found at {sf_path}. Skipping Phase 2 or playing without engine.")
            scheduler = None
        else:
            scheduler = OpponentScheduler(sf_path)
            
        encoder = AlphaZeroEncoder()
        generator = SelfPlayGenerator(self.model, encoder)

        # Initialize ReplayBuffer
        replay_buffer = ReplayBuffer(capacity=20000)
        if initial_replay_buffer is not None:
            # Handle both list (legacy) and ReplayBuffer object
            if isinstance(initial_replay_buffer, list):
                for item in initial_replay_buffer:
                    replay_buffer.add(*item)
            elif isinstance(initial_replay_buffer, ReplayBuffer):
                replay_buffer = initial_replay_buffer
        
        start_ep_rl = max(self.current_epoch + 1, 51)
        for epoch in range(start_ep_rl, num_epochs):
            self.current_epoch = epoch
            engine = scheduler.get_engine(epoch) if scheduler else None
            
            # 1. Data Generation
            logger.info(f"Generating self-play games for epoch {epoch}...")
            for _ in tqdm(range(games_per_epoch), desc="Games"):
                game_data = generator.generate_game(opponent_engine=engine)
                for fen, policy, value in game_data:
                    replay_buffer.add(fen, policy, value)
            if engine: engine.quit()
            
            # 2. Training
            combined_dataset = CombinedDataset(supervised_dataset, replay_buffer)
            loader = DataLoader(combined_dataset, batch_size=batch_size, shuffle=True)
            
            loss, acc = self.train_epoch(loader, combined_dataset, epoch, temperature=0.5) # Sharpened T in RL
            
            logger.info(f"Epoch {epoch} - Loss: {loss:.4f}, Acc: {acc:.2%}, Buffer: {len(replay_buffer)}")

            # Periodic evaluation
            if epoch % 5 == 0:
                win_rate = self.evaluate_against_stockfish(sf_path)
                self.metrics_logger.log_chess(epoch, {'win_rate': win_rate, 'elo_estimate': 1500 + win_rate * 1000})

            self.save_checkpoint(f"checkpoint_rl_{epoch}", replay_buffer=replay_buffer)


def main():
    setup_logging()
    parser = argparse.ArgumentParser()
    parser.add_argument("--h5", type=str, default="distillzero_dataset.h5", help="Path to Stockfish HDF5 dataset")
    parser.add_argument("--sf", type=str, default="assets/stockfish", help="Path to Stockfish binary")
    parser.add_argument("--epochs", type=int, default=100, help="Total number of epochs")
    parser.add_argument("--batch-size", type=int, default=64, help="Batch size for training")
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate")
    parser.add_argument("--checkpoint-dir", type=str, default="checkpoints", help="Directory to save checkpoints")
    parser.add_argument("--db-path", type=str, default="logs/training_logs.db", help="Path to SQLite metrics database")
    parser.add_argument("--games-per-epoch", type=int, default=10, help="Self-play games per epoch in Phase 2")
    parser.add_argument("--resume", action="store_true", help="Resume from latest checkpoint if available")
    args = parser.parse_args()
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = ChessResNet()
    
    trainer = Trainer(model, device, checkpoint_dir=args.checkpoint_dir, db_path=args.db_path)
    trainer.optimizer.param_groups[0]['lr'] = args.lr

    replay_buffer = []
    if args.resume:
        replay_buffer = trainer.load_latest_checkpoint()

    trainer.train(
        args.h5,
        args.sf,
        num_epochs=args.epochs,
        batch_size=args.batch_size,
        games_per_epoch=args.games_per_epoch,
        initial_replay_buffer=replay_buffer
    )

if __name__ == "__main__":
    main()
