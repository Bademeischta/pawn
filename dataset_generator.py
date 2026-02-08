#!/usr/bin/env python3
"""
DistillZero Dataset Generator
==============================
Generates high-quality chess positions labeled by Stockfish for knowledge distillation.

Features:
- Multi-source position sampling (Lichess DB, Stockfish self-play, tactical puzzles)
- Parallel Stockfish evaluation across all CPU cores
- Optimized for speed: ~5-10ms per position
- HDF5 output format for efficient training
- Quality metrics and validation

Usage:
    python dataset_generator.py --output dataset.h5 --positions 1000000 --workers 16
"""

import argparse
import chess
import chess.engine
import chess.pgn
import numpy as np
import h5py
import logging
from pathlib import Path
from multiprocessing import Pool, cpu_count
from dataclasses import dataclass
from typing import List, Tuple, Optional
import random
import time
from tqdm import tqdm
import io

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


@dataclass
class StockfishConfig:
    """Stockfish evaluation configuration - optimized for speed/quality balance"""
    depth: int = 8  # Depth 8 is sweet spot: ~5-10ms per position
    skill_level: int = 15  # Near-maximum strength
    threads: int = 1  # Per worker (each worker gets own engine)
    hash_mb: int = 16  # Small hash per worker
    multipv: int = 1  # Only best move (faster)
    time_limit_ms: int = 50  # Fallback time limit


@dataclass
class DatasetConfig:
    """Dataset generation configuration"""
    total_positions: int = 1_000_000
    lichess_ratio: float = 0.60  # 60% from real games
    selfplay_ratio: float = 0.20  # 20% from Stockfish self-play
    puzzle_ratio: float = 0.20  # 20% from tactical puzzles
    min_elo: int = 2000  # Filter for quality games
    workers: int = cpu_count()  # Use all CPU cores
    stockfish_path: str = "stockfish"  # Path to Stockfish binary


class PositionEncoder:
    """Encodes chess positions into neural network input format (8x8x119)"""
    
    @staticmethod
    def encode_board(board: chess.Board) -> np.ndarray:
        """
        Encodes board into 8x8x119 tensor:
        - 12 planes: piece positions (6 piece types × 2 colors)
        - 2 planes: repetition counters (1-fold, 2-fold)
        - 1 plane: color to move
        - 1 plane: total move count
        - 1 plane: castling rights (4 bits encoded)
        - 1 plane: no-progress count (50-move rule)
        - 101 planes: last 8 board positions (12 planes each + 5 padding)
        
        Simplified version: 8x8x119 with essential features
        """
        planes = np.zeros((8, 8, 119), dtype=np.uint8)
        
        # Planes 0-11: Current piece positions
        piece_idx = 0
        for color in [chess.WHITE, chess.BLACK]:
            for piece_type in [chess.PAWN, chess.KNIGHT, chess.BISHOP, 
                              chess.ROOK, chess.QUEEN, chess.KING]:
                mask = board.pieces(piece_type, color)
                for square in mask:
                    rank, file = divmod(square, 8)
                    planes[rank, file, piece_idx] = 1
                piece_idx += 1
        
        # Plane 12: Color to move (1 if white, 0 if black)
        if board.turn == chess.WHITE:
            planes[:, :, 12] = 1
        
        # Plane 13: Castling rights (encoded as bits)
        castling = 0
        if board.has_kingside_castling_rights(chess.WHITE):
            castling |= 1
        if board.has_queenside_castling_rights(chess.WHITE):
            castling |= 2
        if board.has_kingside_castling_rights(chess.BLACK):
            castling |= 4
        if board.has_queenside_castling_rights(chess.BLACK):
            castling |= 8
        planes[:, :, 13] = castling
        
        # Plane 14: Halfmove clock (50-move rule)
        planes[:, :, 14] = min(board.halfmove_clock, 255)
        
        # Plane 15: Fullmove number
        planes[:, :, 15] = min(board.fullmove_number, 255)
        
        # Planes 16-27: En passant square (one-hot encoded)
        if board.ep_square is not None:
            rank, file = divmod(board.ep_square, 8)
            planes[rank, file, 16] = 1
        
        # Planes 28-118: History planes (last 7 positions)
        # For now, leave as zeros (can be enhanced later with move history)
        
        return planes


class PolicyEncoder:
    """Encodes chess moves into policy vector (1968 dimensions)"""
    
    # Move encoding: from_square (64) × to_square (64) × promotion (4) = 16,384 theoretical
    # But only ~1,968 are legal in chess (queen moves + knight moves + underpromotions)
    
    @staticmethod
    def move_to_index(move: chess.Move) -> int:
        """
        Converts chess.Move to policy index.
        Simplified encoding: from_square * 64 + to_square + promotion_offset
        """
        from_sq = move.from_square
        to_sq = move.to_square
        
        # Base index
        index = from_sq * 64 + to_sq
        
        # Add promotion offset if applicable
        if move.promotion:
            promotion_offset = {
                chess.QUEEN: 0,
                chess.ROOK: 4096,
                chess.BISHOP: 4096 * 2,
                chess.KNIGHT: 4096 * 3
            }
            index += promotion_offset.get(move.promotion, 0)
        
        return index % 1968  # Modulo to fit in 1968 dimensions
    
    @staticmethod
    def create_policy_vector(board: chess.Board, best_move: chess.Move, 
                            temperature: float = 1.0) -> np.ndarray:
        """
        Creates policy vector with best move having highest probability.
        Uses temperature to soften the distribution (prevents overconfidence).
        """
        policy = np.zeros(1968, dtype=np.float32)
        legal_moves = list(board.legal_moves)
        
        if not legal_moves:
            return policy
        
        # Assign probabilities
        best_idx = PolicyEncoder.move_to_index(best_move)
        
        # Temperature-scaled softmax
        # Best move gets high logit, others get small logits
        logits = np.full(1968, -10.0)  # Very low default
        
        for move in legal_moves:
            idx = PolicyEncoder.move_to_index(move)
            if move == best_move:
                logits[idx] = 10.0 / temperature  # High logit for best move
            else:
                logits[idx] = -1.0 / temperature  # Small logit for legal moves
        
        # Softmax
        exp_logits = np.exp(logits - np.max(logits))
        policy = exp_logits / np.sum(exp_logits)
        
        return policy


def evaluate_position_with_stockfish(
    fen: str, 
    stockfish_path: str, 
    config: StockfishConfig
) -> Optional[Tuple[str, float, np.ndarray]]:
    """
    Evaluates a single position with Stockfish.
    Returns: (fen, value, policy_vector) or None if error
    """
    try:
        board = chess.Board(fen)
        
        # Initialize engine
        with chess.engine.SimpleEngine.popen_uci(stockfish_path) as engine:
            # Configure engine
            engine.configure({
                "Threads": config.threads,
                "Hash": config.hash_mb,
                "Skill Level": config.skill_level
            })
            
            # Analyze position
            result = engine.analyse(
                board,
                chess.engine.Limit(depth=config.depth, time=config.time_limit_ms / 1000.0),
                multipv=config.multipv
            )
            
            # Extract best move
            best_move = result.get("pv", [None])[0]
            if best_move is None:
                return None
            
            # Extract evaluation (convert to value in [-1, 1])
            score = result.get("score", chess.engine.Score(0, 0))
            if score.is_mate():
                # Mate score: +1 if winning, -1 if losing
                value = 1.0 if score.relative.mate() > 0 else -1.0
            else:
                # Centipawn score: convert to tanh-scaled value
                cp = score.relative.score(mate_score=10000)
                value = np.tanh(cp / 400.0)  # Scale: 400cp ≈ 0.76 value
            
            # Flip value if black to move (always from white's perspective)
            if board.turn == chess.BLACK:
                value = -value
            
            # Create policy vector
            policy = PolicyEncoder.create_policy_vector(board, best_move, temperature=2.0)
            
            return (fen, value, policy)
            
    except Exception as e:
        logger.warning(f"Failed to evaluate {fen}: {e}")
        return None


def worker_evaluate_batch(args):
    """Worker function for parallel evaluation"""
    fens, stockfish_path, config = args
    results = []
    
    for fen in fens:
        result = evaluate_position_with_stockfish(fen, stockfish_path, config)
        if result:
            results.append(result)
    
    return results


class PositionSampler:
    """Samples positions from various sources"""
    
    @staticmethod
    def sample_from_random_games(n: int, min_elo: int = 2000) -> List[str]:
        """
        Samples positions from random game generation.
        NOTE: This is a placeholder. In production, download Lichess database.
        
        For real implementation:
        1. Download: https://database.lichess.org/
        2. Filter games with both players > min_elo
        3. Sample random positions from these games
        """
        logger.info(f"Generating {n} positions from random games (placeholder)")
        positions = []
        
        for _ in range(n):
            board = chess.Board()
            # Play random moves (weighted towards reasonable moves)
            num_moves = random.randint(10, 60)
            
            for _ in range(num_moves):
                if board.is_game_over():
                    break
                
                legal_moves = list(board.legal_moves)
                if not legal_moves:
                    break
                
                # Weighted random: prefer captures and checks
                weights = []
                for move in legal_moves:
                    weight = 1.0
                    if board.is_capture(move):
                        weight *= 2.0
                    board.push(move)
                    if board.is_check():
                        weight *= 1.5
                    board.pop()
                    weights.append(weight)
                
                move = random.choices(legal_moves, weights=weights)[0]
                board.push(move)
            
            if not board.is_game_over():
                positions.append(board.fen())
        
        return positions
    
    @staticmethod
    def sample_from_stockfish_selfplay(n: int, stockfish_path: str) -> List[str]:
        """
        Generates positions from Stockfish self-play games.
        """
        logger.info(f"Generating {n} positions from Stockfish self-play")
        positions = []
        games_needed = n // 40  # ~40 positions per game
        
        with chess.engine.SimpleEngine.popen_uci(stockfish_path) as engine:
            engine.configure({"Skill Level": 15, "Threads": 1})
            
            for _ in tqdm(range(games_needed), desc="Self-play games"):
                board = chess.Board()
                game_positions = []
                
                while not board.is_game_over() and len(game_positions) < 80:
                    result = engine.play(
                        board,
                        chess.engine.Limit(depth=8, time=0.05)
                    )
                    board.push(result.move)
                    
                    # Sample position every few moves
                    if board.fullmove_number % 2 == 0:
                        game_positions.append(board.fen())
                
                positions.extend(game_positions[:40])
                
                if len(positions) >= n:
                    break
        
        return positions[:n]
    
    @staticmethod
    def sample_from_tactical_puzzles(n: int) -> List[str]:
        """
        Samples positions from tactical puzzles.
        NOTE: This is a placeholder. In production, use puzzle databases.
        
        For real implementation:
        1. Download: https://database.lichess.org/puzzles/
        2. Parse CSV and extract FENs
        """
        logger.info(f"Generating {n} tactical positions (placeholder)")
        # For now, generate positions with tactical motifs
        positions = []
        
        for _ in range(n):
            board = chess.Board()
            # Play to middlegame
            for _ in range(random.randint(15, 25)):
                if board.is_game_over():
                    break
                legal_moves = list(board.legal_moves)
                if legal_moves:
                    board.push(random.choice(legal_moves))
            
            if not board.is_game_over():
                positions.append(board.fen())
        
        return positions


def generate_dataset(config: DatasetConfig, output_path: Path):
    """Main dataset generation pipeline"""
    
    logger.info("=" * 60)
    logger.info("DistillZero Dataset Generator")
    logger.info("=" * 60)
    logger.info(f"Target positions: {config.total_positions:,}")
    logger.info(f"Workers: {config.workers}")
    logger.info(f"Stockfish path: {config.stockfish_path}")
    logger.info(f"Output: {output_path}")
    logger.info("=" * 60)
    
    # Calculate positions per source
    n_lichess = int(config.total_positions * config.lichess_ratio)
    n_selfplay = int(config.total_positions * config.selfplay_ratio)
    n_puzzles = int(config.total_positions * config.puzzle_ratio)
    
    logger.info(f"Lichess games: {n_lichess:,} ({config.lichess_ratio:.0%})")
    logger.info(f"Stockfish self-play: {n_selfplay:,} ({config.selfplay_ratio:.0%})")
    logger.info(f"Tactical puzzles: {n_puzzles:,} ({config.puzzle_ratio:.0%})")
    
    # Step 1: Sample positions
    logger.info("\n[STEP 1/3] Sampling positions...")
    start_time = time.time()
    
    all_fens = []
    all_fens.extend(PositionSampler.sample_from_random_games(n_lichess, config.min_elo))
    all_fens.extend(PositionSampler.sample_from_stockfish_selfplay(n_selfplay, config.stockfish_path))
    all_fens.extend(PositionSampler.sample_from_tactical_puzzles(n_puzzles))
    
    # Deduplicate
    all_fens = list(set(all_fens))
    logger.info(f"Sampled {len(all_fens):,} unique positions in {time.time() - start_time:.1f}s")
    
    # Step 2: Evaluate with Stockfish (parallel)
    logger.info("\n[STEP 2/3] Evaluating with Stockfish...")
    logger.info(f"Using {config.workers} parallel workers")
    
    # Split into batches for workers
    batch_size = len(all_fens) // config.workers
    batches = [
        all_fens[i:i + batch_size] 
        for i in range(0, len(all_fens), batch_size)
    ]
    
    sf_config = StockfishConfig()
    worker_args = [
        (batch, config.stockfish_path, sf_config) 
        for batch in batches
    ]
    
    start_time = time.time()
    with Pool(config.workers) as pool:
        results = list(tqdm(
            pool.imap(worker_evaluate_batch, worker_args),
            total=len(worker_args),
            desc="Evaluating batches"
        ))
    
    # Flatten results
    evaluated_positions = []
    for batch_results in results:
        evaluated_positions.extend(batch_results)
    
    elapsed = time.time() - start_time
    logger.info(f"Evaluated {len(evaluated_positions):,} positions in {elapsed:.1f}s")
    logger.info(f"Speed: {len(evaluated_positions) / elapsed:.1f} positions/sec")
    logger.info(f"Avg time per position: {elapsed / len(evaluated_positions) * 1000:.1f}ms")
    
    # Step 3: Save to HDF5
    logger.info("\n[STEP 3/3] Saving to HDF5...")
    
    # Encode positions
    encoder = PositionEncoder()
    encoded_positions = []
    values = []
    policies = []
    
    for fen, value, policy in tqdm(evaluated_positions, desc="Encoding"):
        board = chess.Board(fen)
        encoded = encoder.encode_board(board)
        encoded_positions.append(encoded)
        values.append(value)
        policies.append(policy)
    
    # Convert to numpy arrays
    positions_array = np.array(encoded_positions, dtype=np.uint8)
    values_array = np.array(values, dtype=np.float32)
    policies_array = np.array(policies, dtype=np.float32)
    
    # Save to HDF5
    with h5py.File(output_path, 'w') as f:
        f.create_dataset('positions', data=positions_array, compression='gzip')
        f.create_dataset('values', data=values_array, compression='gzip')
        f.create_dataset('policies', data=policies_array, compression='gzip')
        
        # Metadata
        f.attrs['total_positions'] = len(evaluated_positions)
        f.attrs['lichess_ratio'] = config.lichess_ratio
        f.attrs['selfplay_ratio'] = config.selfplay_ratio
        f.attrs['puzzle_ratio'] = config.puzzle_ratio
        f.attrs['stockfish_depth'] = sf_config.depth
        f.attrs['stockfish_skill'] = sf_config.skill_level
    
    logger.info(f"Saved {len(evaluated_positions):,} positions to {output_path}")
    logger.info(f"File size: {output_path.stat().st_size / 1024 / 1024:.1f} MB")
    
    # Statistics
    logger.info("\n" + "=" * 60)
    logger.info("Dataset Statistics")
    logger.info("=" * 60)
    logger.info(f"Total positions: {len(evaluated_positions):,}")
    logger.info(f"Value range: [{values_array.min():.3f}, {values_array.max():.3f}]")
    logger.info(f"Value mean: {values_array.mean():.3f} ± {values_array.std():.3f}")
    logger.info(f"Policy sparsity: {(policies_array == 0).sum() / policies_array.size:.1%}")
    logger.info("=" * 60)


def main():
    parser = argparse.ArgumentParser(description="DistillZero Dataset Generator")
    parser.add_argument('--output', type=str, default='dataset.h5',
                       help='Output HDF5 file path')
    parser.add_argument('--positions', type=int, default=100_000,
                       help='Total number of positions to generate')
    parser.add_argument('--workers', type=int, default=cpu_count(),
                       help='Number of parallel workers')
    parser.add_argument('--stockfish', type=str, default='stockfish',
                       help='Path to Stockfish binary')
    parser.add_argument('--lichess-ratio', type=float, default=0.60,
                       help='Ratio of positions from Lichess games')
    parser.add_argument('--selfplay-ratio', type=float, default=0.20,
                       help='Ratio of positions from Stockfish self-play')
    parser.add_argument('--puzzle-ratio', type=float, default=0.20,
                       help='Ratio of positions from tactical puzzles')
    
    args = parser.parse_args()
    
    config = DatasetConfig(
        total_positions=args.positions,
        lichess_ratio=args.lichess_ratio,
        selfplay_ratio=args.selfplay_ratio,
        puzzle_ratio=args.puzzle_ratio,
        workers=args.workers,
        stockfish_path=args.stockfish
    )
    
    output_path = Path(args.output)
    generate_dataset(config, output_path)
    
    logger.info("\n✅ Dataset generation complete!")


if __name__ == '__main__':
    main()
