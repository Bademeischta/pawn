"""
Archimedes Chess AI - End-to-End Training Script
Robust, resumable training with automatic checkpointing and comprehensive metrics.
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import chess
import chess.pgn
import numpy as np
import time
import os
import sys
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import argparse
from tqdm import tqdm
import io

from model import ArchimedesGNN, ChessBoardEncoder, MoveEncoder
from mcts import MCTS
from metrics import MetricsLogger


class ChessDataset(Dataset):
    """Dataset for chess positions with policy and value targets."""
    
    def __init__(self, positions: List[Tuple[str, Dict[str, float], float]]):
        """
        Args:
            positions: List of (fen, policy_dict, value) tuples
        """
        self.positions = positions
        self.encoder = ChessBoardEncoder()
        self.move_encoder = MoveEncoder()
    
    def __len__(self):
        return len(self.positions)
    
    def __getitem__(self, idx):
        fen, policy_dict, value = self.positions[idx]
        
        # Create board
        board = chess.Board(fen)
        
        # Encode as graph
        graph_data = self.encoder.board_to_graph(board)
        
        # Create policy target (sparse)
        policy_target = torch.zeros(1968)
        for move_uci, prob in policy_dict.items():
            try:
                move = chess.Move.from_uci(move_uci)
                move_idx = self.move_encoder.move_to_index(move)
                policy_target[move_idx] = prob
            except:
                pass
        
        # Normalize policy target
        if policy_target.sum() > 0:
            policy_target = policy_target / policy_target.sum()
        
        value_target = torch.tensor([value], dtype=torch.float32)
        
        return graph_data, policy_target, value_target


def collate_fn(batch):
    """Custom collate function for graph data."""
    from torch_geometric.data import Batch as GeometricBatch
    
    graphs, policies, values = zip(*batch)
    
    # Batch graphs
    batched_graphs = GeometricBatch.from_data_list(graphs)
    
    # Stack policies and values
    policies = torch.stack(policies)
    values = torch.stack(values)
    
    return batched_graphs, policies, values


class SelfPlayGenerator:
    """Generates training data through self-play."""
    
    def __init__(self, model, encoder, mcts_simulations: int = 400):
        self.model = model
        self.encoder = encoder
        self.mcts = MCTS(model, encoder, num_simulations=mcts_simulations)
    
    def generate_game(self, max_moves: int = 200) -> List[Tuple[str, Dict[str, float], float]]:
        """
        Play one game and return training positions.
        
        Returns:
            List of (fen, policy_target, value) tuples
        """
        board = chess.Board()
        positions = []
        
        move_count = 0
        while not board.is_game_over() and move_count < max_moves:
            # Store position
            fen = board.fen()
            
            # Run MCTS
            move, stats = self.mcts.search(board, add_noise=(move_count < 30))
            
            # Get policy target from visit counts
            policy_target = self.mcts.get_policy_target(stats.get('root_node'))
            
            # Store position (value will be filled at end)
            positions.append((fen, policy_target, 0.0))
            
            # Make move
            board.push(move)
            move_count += 1
        
        # Determine game outcome
        result = board.result()
        if result == "1-0":
            final_value = 1.0
        elif result == "0-1":
            final_value = -1.0
        else:
            final_value = 0.0
        
        # Assign values from perspective of player to move
        for i, (fen, policy, _) in enumerate(positions):
            # Alternate perspective
            value = final_value if i % 2 == 0 else -final_value
            positions[i] = (fen, policy, value)
        
        return positions, board
    
    def generate_batch(self, num_games: int = 10) -> List[Tuple[str, Dict[str, float], float]]:
        """Generate multiple games."""
        all_positions = []
        
        for _ in tqdm(range(num_games), desc="Self-play"):
            positions, _ = self.generate_game()
            all_positions.extend(positions)
        
        return all_positions


class Trainer:
    """Main training orchestrator with checkpointing and metrics."""
    
    def __init__(self, 
                 model: nn.Module,
                 device: torch.device,
                 learning_rate: float = 0.001,
                 weight_decay: float = 1e-4,
                 checkpoint_dir: str = "checkpoints"):
        
        self.model = model.to(device)
        self.device = device
        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_dir.mkdir(exist_ok=True)
        
        # Optimizer and scheduler
        self.optimizer = optim.AdamW(
            model.parameters(),
            lr=learning_rate,
            weight_decay=weight_decay
        )
        
        self.scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(
            self.optimizer,
            T_0=10,
            T_mult=2
        )
        
        # Metrics logger
        self.metrics_logger = MetricsLogger()
        
        # Training state
        self.current_epoch = 0
        self.best_loss = float('inf')
        self.global_step = 0
        
        # Load checkpoint if exists
        self.load_checkpoint()
    
    def save_checkpoint(self, is_best: bool = False):
        """Save training checkpoint."""
        checkpoint = {
            'epoch': self.current_epoch,
            'global_step': self.global_step,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'best_loss': self.best_loss,
        }
        
        # Save latest checkpoint
        latest_path = self.checkpoint_dir / "latest_checkpoint.pt"
        torch.save(checkpoint, latest_path)
        print(f"[Checkpoint] Saved to {latest_path}")
        
        # Save best checkpoint
        if is_best:
            best_path = self.checkpoint_dir / "best_checkpoint.pt"
            torch.save(checkpoint, best_path)
            print(f"[Checkpoint] New best model saved to {best_path}")
        
        # Save epoch checkpoint
        epoch_path = self.checkpoint_dir / f"checkpoint_epoch_{self.current_epoch}.pt"
        torch.save(checkpoint, epoch_path)
    
    def load_checkpoint(self):
        """Load checkpoint if exists."""
        latest_path = self.checkpoint_dir / "latest_checkpoint.pt"
        
        if not latest_path.exists():
            print("[Checkpoint] No checkpoint found, starting from scratch")
            return
        
        print(f"[Checkpoint] Loading from {latest_path}")
        checkpoint = torch.load(latest_path, map_location=self.device)
        
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        self.current_epoch = checkpoint['epoch']
        self.global_step = checkpoint['global_step']
        self.best_loss = checkpoint['best_loss']
        
        print(f"[Checkpoint] Resumed from epoch {self.current_epoch}")
    
    def train_epoch(self, dataloader: DataLoader, epoch: int) -> Dict[str, float]:
        """Train for one epoch."""
        self.model.train()
        
        total_loss = 0.0
        total_policy_loss = 0.0
        total_value_loss = 0.0
        total_correct_top1 = 0
        total_correct_top5 = 0
        total_samples = 0
        
        epoch_start = time.time()
        
        pbar = tqdm(dataloader, desc=f"Epoch {epoch}")
        for batch_idx, (graphs, policy_targets, value_targets) in enumerate(pbar):
            # Move to device
            graphs = graphs.to(self.device)
            policy_targets = policy_targets.to(self.device)
            value_targets = value_targets.to(self.device)
            
            # Forward pass
            policy_logits, value_pred, aux = self.model(graphs)
            
            # Compute losses
            policy_loss = F.cross_entropy(policy_logits, policy_targets)
            value_loss = F.mse_loss(value_pred, value_targets)
            
            # Combined loss
            loss = policy_loss + value_loss
            
            # Backward pass
            self.optimizer.zero_grad()
            loss.backward()
            
            # Gradient clipping
            grad_norm = torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            
            self.optimizer.step()
            
            # Metrics
            batch_size = policy_targets.size(0)
            total_samples += batch_size
            total_loss += loss.item() * batch_size
            total_policy_loss += policy_loss.item() * batch_size
            total_value_loss += value_loss.item() * batch_size
            
            # Top-k accuracy
            _, top5_pred = policy_logits.topk(5, dim=1)
            _, top1_pred = policy_logits.topk(1, dim=1)
            _, targets = policy_targets.topk(1, dim=1)
            
            total_correct_top1 += (top1_pred == targets).sum().item()
            total_correct_top5 += (top5_pred == targets.expand_as(top5_pred)).any(dim=1).sum().item()
            
            # Log batch metrics
            if batch_idx % 10 == 0:
                self.metrics_logger.log_training(
                    epoch=epoch,
                    batch=batch_idx,
                    metrics={
                        'loss_total': loss.item(),
                        'loss_policy': policy_loss.item(),
                        'loss_value': value_loss.item(),
                        'learning_rate': self.optimizer.param_groups[0]['lr'],
                        'gradient_norm': grad_norm.item(),
                    }
                )
                
                # Log hardware metrics
                if batch_idx % 50 == 0:
                    self.metrics_logger.log_hardware(epoch)
            
            # Update progress bar
            pbar.set_postfix({
                'loss': f"{loss.item():.4f}",
                'policy': f"{policy_loss.item():.4f}",
                'value': f"{value_loss.item():.4f}",
            })
            
            self.global_step += 1
        
        epoch_duration = time.time() - epoch_start
        
        # Compute epoch metrics
        avg_loss = total_loss / total_samples
        avg_policy_loss = total_policy_loss / total_samples
        avg_value_loss = total_value_loss / total_samples
        acc_top1 = total_correct_top1 / total_samples
        acc_top5 = total_correct_top5 / total_samples
        
        # Get model weight statistics
        weights = []
        for param in self.model.parameters():
            weights.append(param.data.cpu().numpy().flatten())
        weights = np.concatenate(weights)
        
        # Log epoch metrics
        self.metrics_logger.log_training(
            epoch=epoch,
            batch=-1,  # Epoch summary
            metrics={
                'loss_total': avg_loss,
                'loss_policy': avg_policy_loss,
                'loss_value': avg_value_loss,
                'learning_rate': self.optimizer.param_groups[0]['lr'],
                'accuracy_top1': acc_top1,
                'accuracy_top5': acc_top5,
                'epoch_duration': epoch_duration,
                'samples_trained': total_samples,
                'weight_mean': float(weights.mean()),
                'weight_std': float(weights.std()),
            }
        )
        
        return {
            'loss': avg_loss,
            'policy_loss': avg_policy_loss,
            'value_loss': avg_value_loss,
            'acc_top1': acc_top1,
            'acc_top5': acc_top5,
        }
    
    def evaluate(self, num_games: int = 10) -> Dict[str, float]:
        """Evaluate model through self-play."""
        self.model.eval()
        
        encoder = ChessBoardEncoder()
        mcts = MCTS(self.model, encoder, num_simulations=200)
        
        results = {'wins': 0, 'losses': 0, 'draws': 0}
        game_lengths = []
        
        for _ in tqdm(range(num_games), desc="Evaluation"):
            board = chess.Board()
            move_count = 0
            
            while not board.is_game_over() and move_count < 200:
                move, _ = mcts.search(board, add_noise=False)
                board.push(move)
                move_count += 1
            
            result = board.result()
            if result == "1-0":
                results['wins'] += 1
            elif result == "0-1":
                results['losses'] += 1
            else:
                results['draws'] += 1
            
            game_lengths.append(move_count)
        
        # Calculate metrics
        total_games = num_games
        win_rate = results['wins'] / total_games
        loss_rate = results['losses'] / total_games
        draw_rate = results['draws'] / total_games
        avg_game_length = np.mean(game_lengths)
        
        # Estimate Elo (simplified)
        elo_estimate = 1500 + 400 * np.log10(max(win_rate / max(loss_rate, 0.01), 0.01))
        
        return {
            'win_rate': win_rate,
            'loss_rate': loss_rate,
            'draw_rate': draw_rate,
            'avg_game_length': avg_game_length,
            'elo_estimate': elo_estimate,
        }
    
    def train(self, num_epochs: int, games_per_epoch: int = 100, batch_size: int = 32):
        """Main training loop."""
        print(f"\n{'='*60}")
        print(f"  ARCHIMEDES TRAINING - Starting from Epoch {self.current_epoch}")
        print(f"{'='*60}\n")
        
        encoder = ChessBoardEncoder()
        
        for epoch in range(self.current_epoch, num_epochs):
            self.current_epoch = epoch
            
            print(f"\n[Epoch {epoch+1}/{num_epochs}]")
            
            # Generate self-play data
            print("Generating self-play data...")
            generator = SelfPlayGenerator(self.model, encoder, mcts_simulations=400)
            positions = generator.generate_batch(num_games=games_per_epoch)
            
            print(f"Generated {len(positions)} training positions")
            
            # Create dataset and dataloader
            dataset = ChessDataset(positions)
            dataloader = DataLoader(
                dataset,
                batch_size=batch_size,
                shuffle=True,
                collate_fn=collate_fn,
                num_workers=0  # Set to 0 for compatibility
            )
            
            # Train epoch
            train_metrics = self.train_epoch(dataloader, epoch)
            
            # Update learning rate
            self.scheduler.step()
            
            # Evaluate
            if (epoch + 1) % 5 == 0:
                print("\nEvaluating model...")
                eval_metrics = self.evaluate(num_games=10)
                
                # Log chess metrics
                self.metrics_logger.log_chess(epoch, eval_metrics)
                
                print(f"Win Rate: {eval_metrics['win_rate']:.2%}")
                print(f"Elo Estimate: {eval_metrics['elo_estimate']:.0f}")
            
            # Save checkpoint
            is_best = train_metrics['loss'] < self.best_loss
            if is_best:
                self.best_loss = train_metrics['loss']
            
            self.save_checkpoint(is_best=is_best)
            
            print(f"\nEpoch {epoch+1} Summary:")
            print(f"  Loss: {train_metrics['loss']:.4f}")
            print(f"  Policy Loss: {train_metrics['policy_loss']:.4f}")
            print(f"  Value Loss: {train_metrics['value_loss']:.4f}")
            print(f"  Top-1 Accuracy: {train_metrics['acc_top1']:.2%}")
            print(f"  Top-5 Accuracy: {train_metrics['acc_top5']:.2%}")
        
        print(f"\n{'='*60}")
        print(f"  TRAINING COMPLETED")
        print(f"{'='*60}\n")
        
        self.metrics_logger.close()


def main():
    parser = argparse.ArgumentParser(description="Archimedes Chess AI Training")
    parser.add_argument("--epochs", type=int, default=100, help="Number of epochs")
    parser.add_argument("--games-per-epoch", type=int, default=50, help="Self-play games per epoch")
    parser.add_argument("--batch-size", type=int, default=32, help="Batch size")
    parser.add_argument("--lr", type=float, default=0.001, help="Learning rate")
    parser.add_argument("--device", type=str, default="auto", help="Device (cuda/cpu/auto)")
    parser.add_argument("--checkpoint-dir", type=str, default="checkpoints", help="Checkpoint directory")
    
    args = parser.parse_args()
    
    # Determine device
    if args.device == "auto":
        if torch.cuda.is_available():
            device = torch.device("cuda")
            print(f"[Device] Using GPU: {torch.cuda.get_device_name(0)}")
        else:
            device = torch.device("cpu")
            print("[Device] GPU not available, using CPU (training will be slower)")
    else:
        device = torch.device(args.device)
    
    # Create model
    print("[Model] Initializing Archimedes GNN...")
    model = ArchimedesGNN(
        hidden_dim=256,
        num_layers=4,
        num_heads=8,
        dropout=0.1
    )
    
    print(f"[Model] Parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # Create trainer
    trainer = Trainer(
        model=model,
        device=device,
        learning_rate=args.lr,
        checkpoint_dir=args.checkpoint_dir
    )
    
    # Train
    trainer.train(
        num_epochs=args.epochs,
        games_per_epoch=args.games_per_epoch,
        batch_size=args.batch_size
    )


if __name__ == "__main__":
    import torch.nn.functional as F
    main()
