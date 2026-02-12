"""
DistillZero - Chess ResNet Architecture
AlphaZero-inspired ResNet for high-performance chess evaluation.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import chess
import numpy as np
from typing import Tuple, List, Dict, Optional


class MoveEncoder:
    """
    Encodes chess moves to indices for policy head.
    1,968 possible legal moves mapping.
    """
    def __init__(self):
        # Generate all possible move triplets (from, to, promotion)
        # that could ever be legal for any piece.
        self.all_moves = []
        for square in chess.SQUARES:
            # Queen-like moves
            r, c = chess.square_rank(square), chess.square_file(square)
            for dr, dc in [(1,0), (-1,0), (0,1), (0,-1), (1,1), (1,-1), (-1,1), (-1,-1)]:
                for dist in range(1, 8):
                    nr, nc = r + dist*dr, c + dist*dc
                    if 0 <= nr < 8 and 0 <= nc < 8:
                        self.all_moves.append((square, chess.square(nc, nr), None))
                    else:
                        break
            # Knight moves
            for dr, dc in [(2,1), (2,-1), (-2,1), (-2,-1), (1,2), (1,-2), (-1,2), (-1,-2)]:
                nr, nc = r + dr, c + dc
                if 0 <= nr < 8 and 0 <= nc < 8:
                    self.all_moves.append((square, chess.square(nc, nr), None))
            
            # Pawn moves and promotions
            # White pawn
            if r < 7:
                # We include all possible pawn moves from any square (except rank 0/7)
                # to keep the mapping consistent.
                if r > 0:
                    # Forward
                    target = chess.square(c, r+1)
                    if r == 6: # Promotion
                        for prom in [chess.QUEEN, chess.ROOK, chess.BISHOP, chess.KNIGHT]:
                            self.all_moves.append((square, target, prom))
                    else:
                        self.all_moves.append((square, target, None))
                        if r == 1: # Double push
                            self.all_moves.append((square, chess.square(c, r+2), None))

                    # Captures
                    for dc in [-1, 1]:
                        nc = c + dc
                        if 0 <= nc < 8:
                            target = chess.square(nc, r+1)
                            if r == 6:
                                for prom in [chess.QUEEN, chess.ROOK, chess.BISHOP, chess.KNIGHT]:
                                    self.all_moves.append((square, target, prom))
                            else:
                                self.all_moves.append((square, target, None))
            
            # Black pawn
            if r > 0:
                if r < 7:
                    # Forward
                    target = chess.square(c, r-1)
                    if r == 1: # Promotion
                        for prom in [chess.QUEEN, chess.ROOK, chess.BISHOP, chess.KNIGHT]:
                            self.all_moves.append((square, target, prom))
                    else:
                        self.all_moves.append((square, target, None))
                        if r == 6: # Double push
                            self.all_moves.append((square, chess.square(c, r-2), None))

                    # Captures
                    for dc in [-1, 1]:
                        nc = c + dc
                        if 0 <= nc < 8:
                            target = chess.square(nc, r-1)
                            if r == 1:
                                for prom in [chess.QUEEN, chess.ROOK, chess.BISHOP, chess.KNIGHT]:
                                    self.all_moves.append((square, target, prom))
                            else:
                                self.all_moves.append((square, target, None))

        # Castling
        self.all_moves.append((chess.E1, chess.G1, None))
        self.all_moves.append((chess.E1, chess.C1, None))
        self.all_moves.append((chess.E8, chess.G8, None))
        self.all_moves.append((chess.E8, chess.C8, None))

        # Deduplicate and sort
        # Using a custom sort key to handle None values in promotion
        self.all_moves = sorted(list(set(self.all_moves)),
                              key=lambda x: (x[0], x[1], x[2] if x[2] is not None else 0))
        self.move_to_idx = {m: i for i, m in enumerate(self.all_moves)}
        self.idx_to_move_tuple = {i: m for i, m in enumerate(self.all_moves)}

        # Verify size
        assert len(self.all_moves) == 1968, f"Expected 1968 moves, got {len(self.all_moves)}"

    def move_to_index(self, move: chess.Move) -> int:
        m = (move.from_square, move.to_square, move.promotion)
        return self.move_to_idx.get(m, -1)

    def index_to_move(self, index: int) -> Optional[chess.Move]:
        m = self.idx_to_move_tuple.get(index)
        if m:
            return chess.Move(m[0], m[1], m[2])
        return None


class AlphaZeroEncoder:
    """
    Encodes chess board into 119 planes (8x8x119).
    """
    def __init__(self, history_len: int = 8):
        self.history_len = history_len

    def board_to_tensor(self, board: chess.Board) -> torch.Tensor:
        """
        Convert board to (119, 8, 8) tensor.
        """
        planes = np.zeros((119, 8, 8), dtype=np.float32)
        
        # Fill history planes (14 planes per step)
        # For simplicity in this implementation, if history is not available (e.g. FEN input),
        # we only fill the current state and zero out the rest.
        # If board.move_stack is available, we could reconstruct history.
        
        current_board = board.copy()
        for i in range(self.history_len):
            offset = i * 14
            self._fill_14_planes(current_board, planes, offset)

            if len(current_board.move_stack) > 0:
                current_board.pop()
            else:
                # No more history available
                break
        
        # Fill constant planes (last 7 planes)
        offset = 112
        # Plane 112: Color (1 if white, 0 if black)
        planes[offset] = 1.0 if board.turn == chess.WHITE else 0.0
        # Plane 113: Total move count (normalized)
        planes[offset+1] = board.fullmove_number / 100.0
        # Planes 114-117: Castling rights
        planes[offset+2] = 1.0 if board.has_kingside_castling_rights(chess.WHITE) else 0.0
        planes[offset+3] = 1.0 if board.has_queenside_castling_rights(chess.WHITE) else 0.0
        planes[offset+4] = 1.0 if board.has_kingside_castling_rights(chess.BLACK) else 0.0
        planes[offset+5] = 1.0 if board.has_queenside_castling_rights(chess.BLACK) else 0.0
        # Plane 118: No-progress count (50-move rule)
        planes[offset+6] = board.halfmove_clock / 100.0
        
        return torch.from_numpy(planes)

    def _fill_14_planes(self, board: chess.Board, planes: np.ndarray, offset: int):
        """Fills 12 piece planes and 2 repetition planes for a given board state."""
        # 6 piece types for current player, then 6 for opponent
        # We always encode from the perspective of the current player?
        # Actually, AlphaZero encodes white pieces then black pieces?
        # Standard: 6 planes for White, 6 for Black.
        
        piece_map = {
            chess.PAWN: 0, chess.KNIGHT: 1, chess.BISHOP: 2,
            chess.ROOK: 3, chess.QUEEN: 4, chess.KING: 5
        }
        
        for square in chess.SQUARES:
            piece = board.piece_at(square)
            if piece:
                rank = chess.square_rank(square)
                file = chess.square_file(square)
                p_idx = piece_map[piece.piece_type]
                if piece.color == chess.WHITE:
                    planes[offset + p_idx, rank, file] = 1.0
                else:
                    planes[offset + 6 + p_idx, rank, file] = 1.0
        
        # Repetition planes (13 and 14 in the 14-block)
        # Note: True repetition counting requires full game history.
        # Since we often train on isolated FENs (e.g. from the factory),
        # these planes are left as 0. For full strength, the input should
        # include the previous 7 board states via the board.move_stack.


class SEBlock(nn.Module):
    """
    Squeeze-and-Excitation Block for channel-wise attention.
    Standard in high-performance engines like Leela Chess Zero.
    """
    def __init__(self, channels: int, reduction: int = 16):
        super().__init__()
        self.squeeze = nn.AdaptiveAvgPool2d(1)
        self.excitation = nn.Sequential(
            nn.Linear(channels, channels // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channels // reduction, channels, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.squeeze(x).view(b, c)
        y = self.excitation(y).view(b, c, 1, 1)
        return x * y.expand_as(x)


class ResBlock(nn.Module):
    def __init__(self, channels: int, use_se: bool = True):
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(channels)
        self.conv2 = nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(channels)
        self.se = SEBlock(channels, reduction=16) if use_se else None

    def forward(self, x):
        residual = x
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        if self.se:
            out = self.se(out)
        out += residual
        return F.relu(out)


class ChessResNet(nn.Module):
    """
    AlphaZero-style ResNet for chess.
    10 ResNet blocks, 256 channels.
    """
    def __init__(self, input_channels: int = 119, num_blocks: int = 10, channels: int = 128, num_policy_outputs: int = 1968):
        super().__init__()
        
        # Initial convolution
        self.conv_input = nn.Conv2d(input_channels, channels, kernel_size=3, padding=1, bias=False)
        self.bn_input = nn.BatchNorm2d(channels)
        
        # Backbone (Residual Blocks)
        self.res_blocks = nn.ModuleList([ResBlock(channels) for _ in range(num_blocks)])
        
        # Policy Head
        # AlphaZero uses a reduction to 2 filters before the linear layer to save parameters
        self.policy_head = nn.Sequential(
            nn.Conv2d(channels, 2, kernel_size=1, bias=False),
            nn.BatchNorm2d(2),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(2 * 8 * 8, num_policy_outputs)
        )
        
        # Value Head
        # AlphaZero uses a reduction to 1 filter then a hidden linear layer
        self.value_head = nn.Sequential(
            nn.Conv2d(channels, 1, kernel_size=1, bias=False),
            nn.BatchNorm2d(1),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(1 * 8 * 8, channels),
            nn.ReLU(),
            nn.Linear(channels, 1),
            nn.Tanh()
        )

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        x = F.relu(self.bn_input(self.conv_input(x)))
        
        for block in self.res_blocks:
            x = block(x)
            
        policy_logits = self.policy_head(x)
        value = self.value_head(x)
        
        return policy_logits, value

    def export_torchscript(self, output_path: str = "model_traced.pt"):
        """Export model to TorchScript for faster inference."""
        self.eval()
        dummy_input = torch.randn(1, 119, 8, 8)
        traced = torch.jit.trace(self, dummy_input)
        traced.save(output_path)
        print(f"Model exported to TorchScript: {output_path}")

    def export_onnx(self, output_path: str = "model.onnx"):
        """Export model to ONNX."""
        self.eval()
        dummy_input = torch.randn(1, 119, 8, 8)
        torch.onnx.export(
            self, dummy_input, output_path,
            input_names=['board'], output_names=['policy', 'value'],
            dynamic_axes={'board': {0: 'batch'}}
        )
        print(f"Model exported to ONNX: {output_path}")


if __name__ == "__main__":
    # Test
    print("Initializing components...")
    encoder = AlphaZeroEncoder()
    move_encoder = MoveEncoder()
    model = ChessResNet()
    
    board = chess.Board()
    tensor = encoder.board_to_tensor(board).unsqueeze(0)
    
    print(f"Input shape: {tensor.shape}")
    
    policy, value = model(tensor)
    
    print(f"Policy shape: {policy.shape}")
    print(f"Value shape: {value.shape}")
    print(f"Model Parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # Test move encoding
    move = chess.Move.from_uci("e2e4")
    idx = move_encoder.move_to_index(move)
    print(f"Move e2e4 index: {idx}")
    back_move = move_encoder.index_to_move(idx)
    print(f"Back to move: {back_move.uci() if back_move else 'None'}")
