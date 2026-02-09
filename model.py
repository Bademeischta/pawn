"""
Archimedes Chess AI - Graph Neural Network Model
PyTorch Geometric implementation with attention mechanisms for chess position evaluation.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATConv, global_mean_pool, global_max_pool
from torch_geometric.data import Data, Batch
import chess
import numpy as np
from typing import Tuple, List, Dict


class ChessBoardEncoder:
    """
    Encodes chess board positions as graphs for GNN processing.
    Nodes = pieces, Edges = attacks/defends relationships.
    """
    
    # Piece type encoding
    PIECE_TO_IDX = {
        chess.PAWN: 0, chess.KNIGHT: 1, chess.BISHOP: 2,
        chess.ROOK: 3, chess.QUEEN: 4, chess.KING: 5
    }
    
    def __init__(self, feature_dim: int = 32):
        self.feature_dim = feature_dim
    
    def board_to_graph(self, board: chess.Board) -> Data:
        """
        Convert chess board to PyTorch Geometric Data object.
        
        Node features (per piece):
        - Piece type (one-hot, 6 dims)
        - Color (1 dim: 1=white, -1=black)
        - Position (x, y normalized to [0,1], 2 dims)
        - Mobility (number of legal moves, 1 dim)
        - Is attacked (1 dim)
        - Is defended (1 dim)
        - Material value (1 dim)
        - Distance to enemy king (1 dim)
        - Distance to own king (1 dim)
        Total: 15 base features
        
        Edge features:
        - Attack relationship (1 dim)
        - Defense relationship (1 dim)
        - Distance (1 dim)
        """
        
        # Collect pieces
        pieces = []
        piece_squares = {}
        
        for square in chess.SQUARES:
            piece = board.piece_at(square)
            if piece:
                pieces.append((square, piece))
                piece_squares[square] = len(pieces) - 1
        
        if len(pieces) == 0:
            # Empty board edge case
            x = torch.zeros((1, 15), dtype=torch.float32)
            edge_index = torch.zeros((2, 0), dtype=torch.long)
            edge_attr = torch.zeros((0, 3), dtype=torch.float32)
            return Data(x=x, edge_index=edge_index, edge_attr=edge_attr)
        
        # Build node features
        node_features = []
        white_king_sq = board.king(chess.WHITE)
        black_king_sq = board.king(chess.BLACK)
        
        for square, piece in pieces:
            # Piece type (one-hot)
            piece_type = [0.0] * 6
            piece_type[self.PIECE_TO_IDX[piece.piece_type]] = 1.0
            
            # Color
            color = 1.0 if piece.color == chess.WHITE else -1.0
            
            # Position (normalized)
            rank = chess.square_rank(square) / 7.0
            file = chess.square_file(square) / 7.0
            
            # Mobility (pseudo-legal moves from this square)
            mobility = len([m for m in board.legal_moves if m.from_square == square]) / 27.0  # Normalize
            
            # Is attacked / defended
            is_attacked = 1.0 if board.is_attacked_by(not piece.color, square) else 0.0
            is_defended = 1.0 if board.is_attacked_by(piece.color, square) else 0.0
            
            # Material value (normalized)
            material_values = {chess.PAWN: 1, chess.KNIGHT: 3, chess.BISHOP: 3,
                             chess.ROOK: 5, chess.QUEEN: 9, chess.KING: 0}
            material = material_values[piece.piece_type] / 9.0
            
            # Distance to kings (Chebyshev distance, normalized)
            enemy_king = black_king_sq if piece.color == chess.WHITE else white_king_sq
            own_king = white_king_sq if piece.color == chess.WHITE else black_king_sq
            
            dist_enemy_king = chess.square_distance(square, enemy_king) / 7.0 if enemy_king else 1.0
            dist_own_king = chess.square_distance(square, own_king) / 7.0 if own_king else 1.0
            
            # Combine features
            features = piece_type + [color, rank, file, mobility, is_attacked, 
                                    is_defended, material, dist_enemy_king, dist_own_king]
            node_features.append(features)
        
        x = torch.tensor(node_features, dtype=torch.float32)
        
        # Build edges (attack/defense relationships)
        edge_list = []
        edge_features = []
        
        for i, (sq1, piece1) in enumerate(pieces):
            for j, (sq2, piece2) in enumerate(pieces):
                if i == j:
                    continue
                
                # Check if piece1 attacks square of piece2
                attacks = board.is_attacked_by(piece1.color, sq2)
                
                if attacks or chess.square_distance(sq1, sq2) <= 2:  # Connect nearby pieces
                    edge_list.append([i, j])
                    
                    # Edge features
                    is_attack = 1.0 if (attacks and piece1.color != piece2.color) else 0.0
                    is_defense = 1.0 if (attacks and piece1.color == piece2.color) else 0.0
                    distance = chess.square_distance(sq1, sq2) / 7.0
                    
                    edge_features.append([is_attack, is_defense, distance])
        
        if edge_list:
            edge_index = torch.tensor(edge_list, dtype=torch.long).t().contiguous()
            edge_attr = torch.tensor(edge_features, dtype=torch.float32)
        else:
            edge_index = torch.zeros((2, 0), dtype=torch.long)
            edge_attr = torch.zeros((0, 3), dtype=torch.float32)
        
        # Global features (board state)
        turn = 1.0 if board.turn == chess.WHITE else -1.0
        castling_rights = [
            1.0 if board.has_kingside_castling_rights(chess.WHITE) else 0.0,
            1.0 if board.has_queenside_castling_rights(chess.WHITE) else 0.0,
            1.0 if board.has_kingside_castling_rights(chess.BLACK) else 0.0,
            1.0 if board.has_queenside_castling_rights(chess.BLACK) else 0.0,
        ]
        halfmove_clock = board.halfmove_clock / 100.0
        fullmove_number = min(board.fullmove_number / 100.0, 1.0)
        
        global_features = torch.tensor(
            [turn] + castling_rights + [halfmove_clock, fullmove_number],
            dtype=torch.float32
        )
        
        return Data(x=x, edge_index=edge_index, edge_attr=edge_attr, 
                   global_features=global_features)


class ArchimedesGNN(nn.Module):
    """
    Graph Neural Network for chess position evaluation.
    Uses Graph Attention Networks (GAT) for learning piece relationships.
    """
    
    def __init__(self, 
                 node_features: int = 15,
                 edge_features: int = 3,
                 global_features: int = 7,
                 hidden_dim: int = 256,
                 num_layers: int = 4,
                 num_heads: int = 8,
                 dropout: float = 0.1,
                 num_policy_outputs: int = 1968):  # All possible moves in chess
        super().__init__()
        
        self.node_features = node_features
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        
        # Input projection
        self.node_encoder = nn.Linear(node_features, hidden_dim)
        self.edge_encoder = nn.Linear(edge_features, hidden_dim)
        self.global_encoder = nn.Linear(global_features, hidden_dim)
        
        # GAT layers with residual connections
        self.gat_layers = nn.ModuleList()
        self.layer_norms = nn.ModuleList()
        
        for i in range(num_layers):
            self.gat_layers.append(
                GATConv(
                    hidden_dim,
                    hidden_dim // num_heads,
                    heads=num_heads,
                    dropout=dropout,
                    edge_dim=hidden_dim,
                    concat=True
                )
            )
            self.layer_norms.append(nn.LayerNorm(hidden_dim))
        
        # Pooling
        self.pool_attention = nn.Linear(hidden_dim, 1)
        
        # Policy head (move prediction)
        self.policy_head = nn.Sequential(
            nn.Linear(hidden_dim * 2 + hidden_dim, hidden_dim * 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_policy_outputs)
        )
        
        # Value head (position evaluation)
        self.value_head = nn.Sequential(
            nn.Linear(hidden_dim * 2 + hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, 1),
            nn.Tanh()
        )
        
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, data: Data) -> Tuple[torch.Tensor, torch.Tensor, Dict[str, torch.Tensor]]:
        """
        Forward pass.
        
        Returns:
            policy: (batch_size, num_policy_outputs) - move probabilities
            value: (batch_size, 1) - position evaluation [-1, 1]
            aux: Dictionary with auxiliary outputs for visualization
        """
        x, edge_index, edge_attr = data.x, data.edge_index, data.edge_attr
        batch = data.batch if hasattr(data, 'batch') else torch.zeros(x.size(0), dtype=torch.long)
        global_feat = data.global_features if hasattr(data, 'global_features') else None
        
        # Encode inputs
        x = self.node_encoder(x)
        x = F.relu(x)
        
        if edge_attr.size(0) > 0:
            edge_attr = self.edge_encoder(edge_attr)
            edge_attr = F.relu(edge_attr)
        else:
            edge_attr = torch.zeros((0, self.hidden_dim), device=x.device)
        
        # Store attention weights for visualization
        attention_weights = []
        
        # GAT layers with residual connections
        for i, (gat, norm) in enumerate(zip(self.gat_layers, self.layer_norms)):
            x_residual = x
            
            if edge_index.size(1) > 0:
                x = gat(x, edge_index, edge_attr=edge_attr)
            
            x = self.dropout(x)
            x = x + x_residual  # Residual connection
            x = norm(x)
            x = F.relu(x)
        
        # Global pooling (attention-based + mean + max)
        attention_scores = torch.sigmoid(self.pool_attention(x))
        attention_weights.append(attention_scores)
        
        # Weighted pooling
        weighted_x = x * attention_scores
        pooled_weighted = global_mean_pool(weighted_x, batch)
        
        # Standard pooling
        pooled_mean = global_mean_pool(x, batch)
        pooled_max = global_max_pool(x, batch)
        
        # Combine pooled representations
        graph_repr = torch.cat([pooled_weighted, pooled_mean, pooled_max], dim=-1)
        
        # Add global features if available
        if global_feat is not None:
            if global_feat.dim() == 1:
                global_feat = global_feat.unsqueeze(0)
            global_encoded = self.global_encoder(global_feat)
            global_encoded = F.relu(global_encoded)
            graph_repr = torch.cat([graph_repr, global_encoded], dim=-1)
        
        # Policy and value heads
        policy_logits = self.policy_head(graph_repr)
        value = self.value_head(graph_repr)
        
        # Auxiliary outputs for visualization
        aux = {
            'attention_weights': attention_weights,
            'node_embeddings': x,
            'graph_representation': graph_repr,
        }
        
        return policy_logits, value, aux
    
    def get_move_probabilities(self, board: chess.Board, encoder: ChessBoardEncoder, 
                               temperature: float = 1.0) -> Dict[str, float]:
        """
        Get move probabilities for a given board position.
        
        Returns:
            Dictionary mapping UCI moves to probabilities
        """
        self.eval()
        
        with torch.no_grad():
            # Encode board
            data = encoder.board_to_graph(board)
            data = data.to(next(self.parameters()).device)
            
            # Forward pass
            policy_logits, value, aux = self(data)
            
            # Apply temperature
            policy_logits = policy_logits / temperature
            
            # Get legal moves
            legal_moves = list(board.legal_moves)
            
            # Map moves to indices (simplified - in practice use proper move encoding)
            move_probs = {}
            policy_probs = F.softmax(policy_logits[0], dim=0)
            
            for i, move in enumerate(legal_moves):
                # Simple hash-based indexing (replace with proper move encoding)
                move_idx = hash(move.uci()) % policy_probs.size(0)
                move_probs[move.uci()] = policy_probs[move_idx].item()
            
            # Normalize
            total = sum(move_probs.values())
            if total > 0:
                move_probs = {k: v/total for k, v in move_probs.items()}
            
            return move_probs, value.item(), aux


class MoveEncoder:
    """
    Encodes chess moves to indices for policy head.
    Uses a simplified encoding scheme.
    """
    
    @staticmethod
    def move_to_index(move: chess.Move) -> int:
        """
        Convert move to index.
        Encoding: from_square (6 bits) + to_square (6 bits) + promotion (3 bits) = 15 bits
        """
        from_sq = move.from_square
        to_sq = move.to_square
        promotion = 0
        
        if move.promotion:
            promotion_map = {
                chess.KNIGHT: 1, chess.BISHOP: 2, 
                chess.ROOK: 3, chess.QUEEN: 4
            }
            promotion = promotion_map.get(move.promotion, 0)
        
        index = (from_sq << 9) | (to_sq << 3) | promotion
        return index % 1968  # Modulo to fit in policy output size
    
    @staticmethod
    def index_to_move(index: int, board: chess.Board) -> chess.Move:
        """
        Convert index back to move (approximate - needs legal move validation).
        """
        promotion = index & 0x7
        to_sq = (index >> 3) & 0x3F
        from_sq = (index >> 9) & 0x3F
        
        promotion_map = {
            0: None, 1: chess.KNIGHT, 2: chess.BISHOP,
            3: chess.ROOK, 4: chess.QUEEN
        }
        
        try:
            move = chess.Move(from_sq, to_sq, promotion=promotion_map[promotion])
            if move in board.legal_moves:
                return move
        except:
            pass
        
        return None


if __name__ == "__main__":
    # Test the model
    print("Testing Archimedes GNN Model...")
    
    # Create encoder and model
    encoder = ChessBoardEncoder()
    model = ArchimedesGNN()
    
    # Test with starting position
    board = chess.Board()
    data = encoder.board_to_graph(board)
    
    print(f"Graph nodes: {data.x.size(0)}")
    print(f"Graph edges: {data.edge_index.size(1)}")
    print(f"Node features: {data.x.size(1)}")
    
    # Forward pass
    policy, value, aux = model(data)
    
    print(f"Policy shape: {policy.shape}")
    print(f"Value: {value.item():.3f}")
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # Test move probabilities
    move_probs, val, _ = model.get_move_probabilities(board, encoder)
    print(f"\nTop 5 moves:")
    for move, prob in sorted(move_probs.items(), key=lambda x: x[1], reverse=True)[:5]:
        print(f"  {move}: {prob:.4f}")
    
    print("\nModel test completed successfully!")
