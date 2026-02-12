"""
Archimedes Chess AI - Monte Carlo Tree Search
MCTS implementation with transposition tables, PUCT algorithm, and comprehensive metrics.
"""

import chess
import chess.polyglot
import numpy as np
import math
import time
import torch
import threading
from typing import Dict, List, Tuple, Optional, Union
from dataclasses import dataclass, field


@dataclass
class MCTSNode:
    """Node in the MCTS tree."""
    board_hash: Union[str, int]
    parent: Optional['MCTSNode'] = None
    move: Optional[chess.Move] = None
    children: Dict[chess.Move, 'MCTSNode'] = field(default_factory=dict)
    
    # Statistics
    visit_count: int = 0
    total_value: float = 0.0
    prior_prob: float = 0.0
    virtual_loss: int = 0
    
    # Cached values
    is_terminal: bool = False
    terminal_value: Optional[float] = None
    legal_moves: List[chess.Move] = field(default_factory=list)

    # Thread safety
    _lock: threading.RLock = field(default_factory=threading.RLock, init=False, repr=False)
    
    def update_stats(self, delta_visit: int, delta_value: float, delta_vloss: int):
        """Atomic update of node statistics."""
        with self._lock:
            self.visit_count += delta_visit
            self.total_value += delta_value
            self.virtual_loss += delta_vloss

    def q_value(self, use_virtual_loss: bool = False) -> float:
        """Average value of this node, optionally including virtual loss."""
        with self._lock:
            visits = self.visit_count + (self.virtual_loss if use_virtual_loss else 0)
            if visits == 0:
                return 0.0
            return self.total_value / visits
    
    def uct_value(self, parent_visits: int, c_puct: float = 1.4) -> float:
        """
        Upper Confidence Bound for Trees (UCT) with PUCT.
        Balances exploitation (Q) and exploration (U).
        """
        # Note: This method is now primarily for documentation as the logic
        # is inlined in MCTS._select_child for performance.
        if self.visit_count == 0:
            u = c_puct * self.prior_prob * math.sqrt(parent_visits)
            return u
        
        q = self.q_value()
        u = c_puct * self.prior_prob * math.sqrt(parent_visits) / (1 + self.visit_count)
        return q + u


class TranspositionTable:
    """
    Hash table for storing previously evaluated positions.
    Dramatically speeds up MCTS by avoiding re-evaluation.
    """
    
    def __init__(self, max_size: int = 1000000):
        self.table: Dict[Union[str, int], MCTSNode] = {}
        self.max_size = max_size
        self.hits = 0
        self.misses = 0
    
    def get_hash(self, board: chess.Board) -> int:
        """Generate hash for board position."""
        # Optimization: Use Zobrist hashing which is much faster than FEN+MD5
        return chess.polyglot.zobrist_hash(board)
    
    def get(self, board_hash: Union[str, int]) -> Optional[MCTSNode]:
        """Retrieve node from table."""
        node = self.table.get(board_hash)
        if node:
            self.hits += 1
        else:
            self.misses += 1
        return node
    
    def put(self, board_hash: Union[str, int], node: MCTSNode):
        """Store node in table with LRU eviction."""
        if board_hash in self.table:
            # Move to end to mark as recently used
            del self.table[board_hash]
        elif len(self.table) >= self.max_size:
            # Remove oldest entry (first in dict)
            it = iter(self.table)
            try:
                first = next(it)
                del self.table[first]
            except StopIteration:
                pass
        self.table[board_hash] = node
    
    def get_hit_rate(self) -> float:
        """Calculate cache hit rate."""
        total = self.hits + self.misses
        return self.hits / total if total > 0 else 0.0
    
    def clear(self):
        """Clear the table."""
        self.table.clear()
        self.hits = 0
        self.misses = 0


class MCTSMetrics:
    """Tracks MCTS performance metrics."""
    
    def __init__(self):
        self.reset()
    
    def reset(self):
        self.total_nodes = 0
        self.max_depth = 0
        self.depth_sum = 0
        self.depth_count = 0
        # Optimization: Use running sums for mean and std to avoid large list allocations
        self.q_sum = 0.0
        self.q_sq_sum = 0.0
        self.visit_sum = 0
        self.visit_sq_sum = 0
        self.branch_sum = 0
        self.branch_count = 0
        self.cutoffs = 0
        self.total_branches = 0
        self.search_time = 0.0
        self.positions_evaluated = 0
    
    def record_node(self, depth: int, q_value: float, visit_count: int, num_children: int):
        """Record metrics for a node."""
        self.total_nodes += 1
        self.max_depth = max(self.max_depth, depth)
        self.depth_sum += depth
        self.depth_count += 1

        # Accumulate sums for mean and variance
        self.q_sum += q_value
        self.q_sq_sum += q_value * q_value
        self.visit_sum += visit_count
        self.visit_sq_sum += visit_count * visit_count

        if num_children > 0:
            self.branch_sum += num_children
            self.branch_count += 1
    
    def record_cutoff(self):
        """Record a branch cutoff."""
        self.cutoffs += 1
    
    def get_summary(self) -> Dict[str, float]:
        """Get summary statistics."""
        def calc_stats(count, total_sum, sq_sum):
            if count == 0:
                return 0.0, 0.0
            mean = total_sum / count
            variance = max(0, (sq_sum / count) - (mean * mean))
            return mean, math.sqrt(variance)

        q_mean, q_std = calc_stats(self.depth_count, self.q_sum, self.q_sq_sum)
        visit_mean, visit_std = calc_stats(self.depth_count, self.visit_sum, self.visit_sq_sum)

        return {
            'total_nodes': self.total_nodes,
            'max_search_depth': self.max_depth,
            'avg_search_depth': self.depth_sum / max(self.depth_count, 1),
            'nodes_per_second': self.total_nodes / max(self.search_time, 0.001),
            'avg_branching_factor': self.branch_sum / max(self.branch_count, 1),
            'cutoff_rate': self.cutoffs / max(self.total_nodes, 1),
            'q_value_mean': q_mean,
            'q_value_std': q_std,
            'visit_count_mean': visit_mean,
            'visit_count_std': visit_std,
        }


class MCTS:
    """
    Monte Carlo Tree Search for chess.
    Uses neural network for position evaluation and move priors.
    """
    
    def __init__(self,
                 model,
                 encoder,
                 num_simulations: int = 800,
                 c_puct: float = 1.4,
                 temperature: float = 1.0,
                 dirichlet_alpha: float = 0.3,
                 dirichlet_epsilon: float = 0.25,
                 use_transposition_table: bool = True):
        """
        Args:
            model: Neural network model for evaluation
            encoder: Board encoder
            num_simulations: Number of MCTS simulations per move
            c_puct: Exploration constant for PUCT
            temperature: Temperature for move selection
            dirichlet_alpha: Dirichlet noise alpha for exploration
            dirichlet_epsilon: Weight of Dirichlet noise
            use_transposition_table: Whether to use transposition table
        """
        self.model = model
        self.encoder = encoder
        self.num_simulations = num_simulations
        self.c_puct = c_puct
        self.temperature = temperature
        self.dirichlet_alpha = dirichlet_alpha
        self.dirichlet_epsilon = dirichlet_epsilon
        
        self.transposition_table = TranspositionTable() if use_transposition_table else None
        self.metrics = MCTSMetrics()

        # Optimization: Cache model device to avoid repeated discovery
        self.device = next(model.parameters()).device

        # Optionally use TorchScript for faster inference
        self.use_traced_model = False
        if hasattr(model, 'traced_model') and model.traced_model is not None:
            self.traced_model = model.traced_model
            self.use_traced_model = True
    
    def search(self, board: chess.Board, add_noise: bool = True, batch_size: int = 8) -> Tuple[chess.Move, Dict]:
        """
        Perform MCTS search and return best move with statistics.
        """
        self.metrics.reset()
        start_time = time.time()
        
        # Ensure deterministic hashing for transposition table
        root_hash = self.transposition_table.get_hash(board) if self.transposition_table else chess.polyglot.zobrist_hash(board)
        root = self._get_or_create_node(board, root_hash, None, None)
        
        if add_noise and root.legal_moves:
            noise = np.random.dirichlet([self.dirichlet_alpha] * len(root.legal_moves))
            for move, n in zip(root.legal_moves, noise):
                if move in root.children:
                    child = root.children[move]
                    child.prior_prob = (1 - self.dirichlet_epsilon) * child.prior_prob + \
                                      self.dirichlet_epsilon * n
        
        # Run simulations in batches
        num_batches = max(1, self.num_simulations // batch_size)
        for _ in range(num_batches):
            self._simulate_batch(board, root, batch_size)
        
        self.metrics.search_time = time.time() - start_time
        
        # Select best move
        best_move = self._select_move(root, board)
        
        # Gather statistics
        stats = self._gather_stats(root)
        stats['policy_target'] = self.get_policy_target(root)
        stats.update(self.metrics.get_summary())
        
        if self.transposition_table:
            stats['cache_hit_rate'] = self.transposition_table.get_hit_rate()
            stats['transposition_hits'] = self.transposition_table.hits
            stats['transposition_misses'] = self.transposition_table.misses
        
        return best_move, stats
    
    def _simulate_batch(self, board: chess.Board, root: MCTSNode, batch_size: int):
        """Perform multiple simulations in parallel and evaluate as a batch."""
        paths = []
        leaf_boards = []
        leaf_nodes = []

        # 1. Selection Phase for the whole batch
        for _ in range(batch_size):
            path = []
            current_board = board.copy()
            current_node = root
            depth = 0

            # Navigate until we hit a leaf or terminal
            while current_node.visit_count > 0 and not current_node.is_terminal:
                move, next_node = self._select_child(current_node, use_virtual_loss=True)
                if next_node is None: break

                # Apply virtual loss to encourage exploration within the batch
                current_node.update_stats(delta_visit=0, delta_value=0, delta_vloss=1)
                path.append(current_node)

                current_board.push(move)
                current_node = next_node
                depth += 1

            paths.append(path)
            leaf_boards.append(current_board)
            leaf_nodes.append(current_node)
            self.metrics.record_node(depth, current_node.q_value(), current_node.visit_count, len(current_node.children))

        # 2. Batch Evaluation
        # Filter out terminal nodes (they don't need evaluation)
        eval_indices = [i for i, node in enumerate(leaf_nodes) if not node.is_terminal]
        
        if eval_indices:
            eval_boards = [leaf_boards[i] for i in eval_indices]
            eval_tensors = [self.encoder.board_to_tensor(b) for b in eval_boards]
            batch_tensor = torch.stack(eval_tensors).to(self.device)

            self.model.eval()
            with torch.no_grad():
                if self.use_traced_model:
                    batch_logits, batch_values = self.traced_model(batch_tensor)
                else:
                    batch_logits, batch_values = self.model(batch_tensor)
                batch_probs = torch.softmax(batch_logits, dim=1).cpu().numpy()
                batch_values = batch_values.squeeze(-1).cpu().numpy()

            from model import get_move_encoder
            move_encoder = get_move_encoder()

            # Process each evaluated leaf
            for i, idx in enumerate(eval_indices):
                node = leaf_nodes[idx]
                board_at_leaf = leaf_boards[idx]
                value = batch_values[i]
                probs = batch_probs[i]

                # Expand leaf
                with node._lock:
                    if node.visit_count > 0 and node.children:
                        # Already expanded by another thread/simulation
                        continue

                    legal_moves = list(board_at_leaf.legal_moves)
                    node.legal_moves = legal_moves

                    if not legal_moves:
                        node.is_terminal = True
                        node.terminal_value = 0.0
                    else:
                        for move in legal_moves:
                            move_idx = move_encoder.move_to_index(move)
                            prior = probs[move_idx] if move_idx != -1 else 0.0

                            board_at_leaf.push(move)
                            child_hash = self.transposition_table.get_hash(board_at_leaf) if self.transposition_table else chess.polyglot.zobrist_hash(board_at_leaf)
                            child = self._get_or_create_node(board_at_leaf, child_hash, node, move)
                            child.prior_prob = prior
                            node.children[move] = child
                            board_at_leaf.pop()

                        # Normalize priors
                        total_prior = sum(child.prior_prob for child in node.children.values())
                        if total_prior > 0:
                            for child in node.children.values():
                                child.prior_prob /= total_prior

                        node.visit_count = 1
                        node.total_value = value

                self.metrics.positions_evaluated += 1

        # 3. Backpropagation
        for i, path in enumerate(paths):
            leaf_node = leaf_nodes[i]
            # Use terminal value or predicted value
            with leaf_node._lock:
                value = leaf_node.terminal_value if leaf_node.is_terminal else leaf_node.total_value / max(1, leaf_node.visit_count)

            # Backpropagate through the path
            # Values alternate perspective
            current_value = -value
            for node in reversed(path):
                node.update_stats(delta_visit=1, delta_value=current_value, delta_vloss=-1)
                current_value = -current_value
    
    def _expand_and_evaluate(self, board: chess.Board, node: MCTSNode) -> float:
        """Expand node and evaluate position with neural network."""
        with node._lock:
            if node.visit_count > 0 and node.children:
                return node.q_value()

            # Check terminal
            if board.is_game_over():
                node.is_terminal = True
                result = board.result()
                if result == "1-0":
                    value = 1.0 if board.turn == chess.WHITE else -1.0
                elif result == "0-1":
                    value = -1.0 if board.turn == chess.WHITE else 1.0
                else:
                    value = 0.0
                node.terminal_value = value
                node.visit_count = 1
                node.total_value = value
                return value
            
            # Get legal moves
            legal_moves = list(board.legal_moves)
            node.legal_moves = legal_moves
            
            if not legal_moves:
                node.is_terminal = True
                node.terminal_value = 0.0
                return 0.0

            # Evaluate with neural network
            self.model.eval()
            with torch.no_grad():
                tensor = self.encoder.board_to_tensor(board).unsqueeze(0).to(self.device)

                policy_logits, value = self.model(tensor)
                value = value.item()

                # Get move probabilities
                policy_probs = torch.softmax(policy_logits[0], dim=0).cpu().numpy()

            self.metrics.positions_evaluated += 1

            # Create child nodes with priors
            from model import get_move_encoder
            move_encoder = get_move_encoder()

            for move in legal_moves:
                move_idx = move_encoder.move_to_index(move)
                prior = policy_probs[move_idx] if move_idx != -1 else 0.0

                board.push(move)
                child_hash = self.transposition_table.get_hash(board) if self.transposition_table else chess.polyglot.zobrist_hash(board)
                child = self._get_or_create_node(board, child_hash, node, move)
                child.prior_prob = prior
                node.children[move] = child
                board.pop()
            
            # Normalize priors
            total_prior = sum(child.prior_prob for child in node.children.values())
            if total_prior > 0:
                for child in node.children.values():
                    child.prior_prob /= total_prior

            # Initialize node
            node.visit_count = 1
            node.total_value = value

            return value
    
    def _select_child(self, node: MCTSNode, use_virtual_loss: bool = False) -> Tuple[chess.Move, MCTSNode]:
        """Select child with highest UCT value."""
        with node._lock:
            best_move = None
            best_child = None
            best_value = -float('inf')

            visits = node.visit_count + (node.virtual_loss if use_virtual_loss else 0)
            sqrt_parent_visits = math.sqrt(max(1, visits))
            c_puct_sqrt = self.c_puct * sqrt_parent_visits

            items = list(node.children.items())
            np.random.shuffle(items)

            for move, child in items:
                with child._lock:
                    child_visits = child.visit_count + (child.virtual_loss if use_virtual_loss else 0)
                    if child_visits == 0:
                        uct = c_puct_sqrt * (child.prior_prob + 1e-8)
                    else:
                        q = child.total_value / child_visits
                        u = c_puct_sqrt * child.prior_prob / (1 + child_visits)
                        uct = q + u

                if uct > best_value:
                    best_value = uct
                    best_move = move
                    best_child = child
            
            return best_move, best_child
    
    def _select_move(self, root: MCTSNode, board: chess.Board) -> chess.Move:
        """Select move based on visit counts and temperature."""
        if not root.children:
            # Fallback to random legal move
            return np.random.choice(list(board.legal_moves))
        
        moves = list(root.children.keys())
        visit_counts = np.array([root.children[m].visit_count for m in moves])
        
        if self.temperature == 0:
            # Greedy selection
            best_idx = np.argmax(visit_counts)
            return moves[best_idx]
        else:
            # Stochastic selection with temperature
            visit_counts = visit_counts ** (1.0 / self.temperature)
            probs = visit_counts / visit_counts.sum()
            return np.random.choice(moves, p=probs)
    
    def _get_or_create_node(self, board: chess.Board, board_hash: str, 
                           parent: Optional[MCTSNode], move: Optional[chess.Move]) -> MCTSNode:
        """Get node from transposition table or create new one."""
        if self.transposition_table:
            node = self.transposition_table.get(board_hash)
            if node:
                return node
        
        node = MCTSNode(board_hash=board_hash, parent=parent, move=move)
        
        if self.transposition_table:
            self.transposition_table.put(board_hash, node)
        
        return node
    
    def _gather_stats(self, root: MCTSNode) -> Dict:
        """Gather statistics about the search tree."""
        stats = {
            'root_visits': root.visit_count,
            'root_value': root.q_value(),
            'num_children': len(root.children),
        }
        
        # Top moves
        if root.children:
            moves_data = []
            for move, child in root.children.items():
                moves_data.append({
                    'move': move.uci(),
                    'visits': child.visit_count,
                    'q_value': child.q_value(),
                    'prior': child.prior_prob,
                })
            
            # Sort by visits
            moves_data.sort(key=lambda x: x['visits'], reverse=True)
            stats['top_moves'] = moves_data[:10]
        
        return stats
    
    def get_policy_target(self, root: MCTSNode) -> Dict[str, float]:
        """
        Get policy target for training (visit count distribution).
        """
        if not root.children:
            return {}
        
        moves = list(root.children.keys())
        visit_counts = np.array([root.children[m].visit_count for m in moves])
        
        # Normalize to probabilities
        total = visit_counts.sum()
        if total == 0:
            return {}
        
        probs = visit_counts / total
        
        return {move.uci(): prob for move, prob in zip(moves, probs)}


if __name__ == "__main__":
    # Test MCTS
    print("Testing MCTS implementation...")
    
    from model import ChessResNet, AlphaZeroEncoder
    
    # Create model and encoder
    encoder = AlphaZeroEncoder()
    model = ChessResNet()
    model.eval()
    
    # Create MCTS
    mcts = MCTS(model, encoder, num_simulations=100)
    
    # Test on starting position
    board = chess.Board()
    print(f"Position: {board.fen()}")
    
    start = time.time()
    best_move, stats = mcts.search(board)
    elapsed = time.time() - start
    
    print(f"\nBest move: {best_move}")
    print(f"Search time: {elapsed:.2f}s")
    print(f"Nodes per second: {stats['nodes_per_second']:.0f}")
    print(f"Max depth: {stats['max_search_depth']}")
    print(f"Avg depth: {stats['avg_search_depth']:.1f}")
    print(f"Cache hit rate: {stats.get('cache_hit_rate', 0):.2%}")
    
    print("\nTop moves:")
    for move_data in stats['top_moves'][:5]:
        print(f"  {move_data['move']}: visits={move_data['visits']}, "
              f"Q={move_data['q_value']:.3f}, prior={move_data['prior']:.3f}")
    
    print("\nMCTS test completed successfully!")
