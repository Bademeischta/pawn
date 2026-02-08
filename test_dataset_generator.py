#!/usr/bin/env python3
"""
Quick test script for dataset generator
Tests all components without requiring Stockfish installation
"""

import numpy as np
import chess
from dataset_generator import PositionEncoder, PolicyEncoder

def test_position_encoder():
    """Test position encoding"""
    print("Testing PositionEncoder...")
    
    # Test starting position
    board = chess.Board()
    encoder = PositionEncoder()
    encoded = encoder.encode_board(board)
    
    assert encoded.shape == (8, 8, 119), f"Wrong shape: {encoded.shape}"
    assert encoded.dtype == np.uint8, f"Wrong dtype: {encoded.dtype}"
    
    # Check piece planes (should have 32 pieces)
    piece_count = np.sum(encoded[:, :, 0:12])
    assert piece_count == 32, f"Wrong piece count: {piece_count}"
    
    # Check color to move (white = 1)
    assert np.all(encoded[:, :, 12] == 1), "Color to move should be 1 for white"
    
    # Test after some moves
    board.push_san("e4")
    board.push_san("e5")
    encoded2 = encoder.encode_board(board)
    
    # Should still have 32 pieces
    piece_count2 = np.sum(encoded2[:, :, 0:12])
    assert piece_count2 == 32, f"Wrong piece count after moves: {piece_count2}"
    
    print("✅ PositionEncoder tests passed!")


def test_policy_encoder():
    """Test policy encoding"""
    print("\nTesting PolicyEncoder...")
    
    board = chess.Board()
    encoder = PolicyEncoder()
    
    # Test move to index conversion
    move = chess.Move.from_uci("e2e4")
    idx = encoder.move_to_index(move)
    assert 0 <= idx < 1968, f"Invalid index: {idx}"
    
    # Test policy vector creation
    policy = encoder.create_policy_vector(board, move, temperature=2.0)
    
    assert policy.shape == (1968,), f"Wrong policy shape: {policy.shape}"
    assert policy.dtype == np.float32, f"Wrong policy dtype: {policy.dtype}"
    assert np.isclose(np.sum(policy), 1.0, atol=1e-5), f"Policy doesn't sum to 1: {np.sum(policy)}"
    assert np.all(policy >= 0), "Policy has negative values"
    
    # Best move should have highest probability
    best_idx = encoder.move_to_index(move)
    assert policy[best_idx] == np.max(policy), "Best move doesn't have highest probability"
    
    print("✅ PolicyEncoder tests passed!")


def test_encoding_consistency():
    """Test that encoding is consistent"""
    print("\nTesting encoding consistency...")
    
    board = chess.Board()
    encoder = PositionEncoder()
    
    # Encode same position twice
    encoded1 = encoder.encode_board(board)
    encoded2 = encoder.encode_board(board)
    
    assert np.array_equal(encoded1, encoded2), "Encoding is not deterministic"
    
    # Test different positions produce different encodings
    board.push_san("e4")
    encoded3 = encoder.encode_board(board)
    
    assert not np.array_equal(encoded1, encoded3), "Different positions have same encoding"
    
    print("✅ Encoding consistency tests passed!")


def test_edge_cases():
    """Test edge cases"""
    print("\nTesting edge cases...")
    
    encoder_pos = PositionEncoder()
    encoder_pol = PolicyEncoder()
    
    # Test position with few pieces (endgame)
    board = chess.Board("8/8/8/4k3/8/8/4K3/8 w - - 0 1")  # King vs King
    encoded = encoder_pos.encode_board(board)
    piece_count = np.sum(encoded[:, :, 0:12])
    assert piece_count == 2, f"Wrong piece count in endgame: {piece_count}"
    
    # Test position with promotion
    board = chess.Board("8/P7/8/8/8/8/8/K6k w - - 0 1")
    if board.is_valid():
        encoded = encoder_pos.encode_board(board)
        assert encoded.shape == (8, 8, 119), "Failed to encode promotion position"
    
    # Test checkmate position
    board = chess.Board("r1bqkb1r/pppp1Qpp/2n2n2/4p3/2B1P3/8/PPPP1PPP/RNB1K1NR b KQkq - 0 4")
    if board.is_valid():
        encoded = encoder_pos.encode_board(board)
        assert encoded.shape == (8, 8, 119), "Failed to encode checkmate position"
    
    print("✅ Edge case tests passed!")


def benchmark_encoding_speed():
    """Benchmark encoding speed"""
    print("\nBenchmarking encoding speed...")
    
    import time
    
    encoder_pos = PositionEncoder()
    encoder_pol = PolicyEncoder()
    
    # Generate random positions
    boards = []
    for _ in range(100):
        board = chess.Board()
        for _ in range(np.random.randint(10, 40)):
            if board.is_game_over():
                break
            legal_moves = list(board.legal_moves)
            if legal_moves:
                board.push(np.random.choice(legal_moves))
        boards.append(board)
    
    # Benchmark position encoding
    start = time.time()
    for board in boards:
        encoded = encoder_pos.encode_board(board)
    pos_time = time.time() - start
    
    print(f"  Position encoding: {pos_time*1000/len(boards):.2f}ms per position")
    print(f"  Throughput: {len(boards)/pos_time:.0f} positions/sec")
    
    # Benchmark policy encoding
    start = time.time()
    for board in boards:
        if not board.is_game_over():
            legal_moves = list(board.legal_moves)
            if legal_moves:
                policy = encoder_pol.create_policy_vector(board, legal_moves[0])
    pol_time = time.time() - start
    
    print(f"  Policy encoding: {pol_time*1000/len(boards):.2f}ms per position")
    print(f"  Throughput: {len(boards)/pol_time:.0f} positions/sec")
    
    print("✅ Benchmark complete!")


def main():
    print("=" * 60)
    print("Dataset Generator Component Tests")
    print("=" * 60)
    
    try:
        test_position_encoder()
        test_policy_encoder()
        test_encoding_consistency()
        test_edge_cases()
        benchmark_encoding_speed()
        
        print("\n" + "=" * 60)
        print("✅ ALL TESTS PASSED!")
        print("=" * 60)
        print("\nNext steps:")
        print("1. Install Stockfish: sudo apt-get install stockfish")
        print("2. Run small test: python dataset_generator.py --positions 1000")
        print("3. Check output: python -c 'import h5py; f=h5py.File(\"dataset.h5\"); print(f.keys())'")
        
    except AssertionError as e:
        print(f"\n❌ TEST FAILED: {e}")
        return 1
    except Exception as e:
        print(f"\n❌ ERROR: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    return 0


if __name__ == '__main__':
    exit(main())
