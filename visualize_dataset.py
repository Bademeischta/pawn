#!/usr/bin/env python3
"""
Dataset Visualization Tool
Analyzes and visualizes the generated chess dataset
"""

import h5py
import numpy as np
import argparse
from pathlib import Path

def print_header(text):
    """Print formatted header"""
    print("\n" + "=" * 70)
    print(f"  {text}")
    print("=" * 70)

def analyze_dataset(dataset_path: Path):
    """Analyze and display dataset statistics"""
    
    print_header("DistillZero Dataset Analysis")
    print(f"File: {dataset_path}")
    print(f"Size: {dataset_path.stat().st_size / 1024 / 1024:.2f} MB")
    
    with h5py.File(dataset_path, 'r') as f:
        # Basic info
        print_header("Dataset Structure")
        print(f"Datasets:")
        for key in f.keys():
            dataset = f[key]
            print(f"  • {key:12s}: {dataset.shape} {dataset.dtype}")
        
        # Metadata
        print_header("Metadata")
        if f.attrs:
            for key, value in f.attrs.items():
                print(f"  • {key:20s}: {value}")
        else:
            print("  No metadata found")
        
        # Load data
        positions = f['positions'][:]
        values = f['values'][:]
        policies = f['policies'][:]
        
        n_positions = len(positions)
        
        # Position statistics
        print_header("Position Statistics")
        print(f"Total positions: {n_positions:,}")
        
        # Analyze piece counts
        piece_counts = []
        for i in range(min(1000, n_positions)):  # Sample first 1000
            piece_count = np.sum(positions[i, :, :, 0:12])
            piece_counts.append(piece_count)
        
        print(f"\nPiece count distribution (sample of {len(piece_counts)}):")
        print(f"  Mean:   {np.mean(piece_counts):.1f} pieces")
        print(f"  Std:    {np.std(piece_counts):.1f}")
        print(f"  Range:  [{np.min(piece_counts)}, {np.max(piece_counts)}]")
        
        # Analyze color to move
        white_to_move = np.sum(positions[:, 0, 0, 12])
        print(f"\nColor to move:")
        print(f"  White: {white_to_move:,} ({white_to_move/n_positions*100:.1f}%)")
        print(f"  Black: {n_positions - white_to_move:,} ({(n_positions - white_to_move)/n_positions*100:.1f}%)")
        
        # Value statistics
        print_header("Value Statistics")
        print(f"Range:      [{values.min():.3f}, {values.max():.3f}]")
        print(f"Mean:       {values.mean():.3f}")
        print(f"Std:        {values.std():.3f}")
        print(f"Median:     {np.median(values):.3f}")
        
        # Value distribution
        print(f"\nValue distribution:")
        bins = [(-1.0, -0.5), (-0.5, -0.1), (-0.1, 0.1), (0.1, 0.5), (0.5, 1.0)]
        for low, high in bins:
            count = np.sum((values >= low) & (values < high))
            pct = count / n_positions * 100
            bar = "█" * int(pct / 2)
            print(f"  [{low:5.1f}, {high:5.1f}): {count:7,} ({pct:5.1f}%) {bar}")
        
        # Policy statistics
        print_header("Policy Statistics")
        
        # Sparsity
        sparsity = (policies == 0).sum() / policies.size
        print(f"Sparsity:   {sparsity*100:.1f}% (zeros)")
        
        # Entropy
        from scipy.stats import entropy
        sample_size = min(1000, n_positions)
        entropies = []
        for i in range(sample_size):
            ent = entropy(policies[i] + 1e-10)
            entropies.append(ent)
        
        print(f"Entropy:    {np.mean(entropies):.2f} ± {np.std(entropies):.2f}")
        print(f"            (higher = more diverse move distributions)")
        
        # Top move probability
        top_probs = []
        for i in range(sample_size):
            top_prob = np.max(policies[i])
            top_probs.append(top_prob)
        
        print(f"\nTop move probability (sample of {sample_size}):")
        print(f"  Mean:   {np.mean(top_probs):.3f}")
        print(f"  Median: {np.median(top_probs):.3f}")
        print(f"  Range:  [{np.min(top_probs):.3f}, {np.max(top_probs):.3f}]")
        
        # Quality assessment
        print_header("Quality Assessment")
        
        # Check for balanced dataset
        value_balance = abs(values.mean())
        if value_balance < 0.1:
            print("✅ Value distribution is well-balanced (mean ≈ 0)")
        else:
            print(f"⚠️  Value distribution is imbalanced (mean = {values.mean():.3f})")
        
        # Check for diversity
        if np.mean(entropies) > 2.0:
            print("✅ Policy entropy is good (diverse move distributions)")
        else:
            print("⚠️  Policy entropy is low (may be too deterministic)")
        
        # Check for reasonable piece counts
        if 20 <= np.mean(piece_counts) <= 32:
            print("✅ Piece counts are reasonable (varied game phases)")
        else:
            print(f"⚠️  Unusual piece counts (mean = {np.mean(piece_counts):.1f})")
        
        # Check for color balance
        if 0.45 <= white_to_move/n_positions <= 0.55:
            print("✅ Color distribution is balanced")
        else:
            print(f"⚠️  Color distribution is imbalanced ({white_to_move/n_positions*100:.1f}% white)")
        
        # Storage efficiency
        print_header("Storage Efficiency")
        
        uncompressed_size = (
            positions.nbytes + values.nbytes + policies.nbytes
        ) / 1024 / 1024
        compressed_size = dataset_path.stat().st_size / 1024 / 1024
        compression_ratio = uncompressed_size / compressed_size
        
        print(f"Uncompressed: {uncompressed_size:.2f} MB")
        print(f"Compressed:   {compressed_size:.2f} MB")
        print(f"Ratio:        {compression_ratio:.2f}x")
        print(f"Per position: {compressed_size / n_positions * 1024:.2f} KB")
        
        # Recommendations
        print_header("Recommendations")
        
        if n_positions < 100_000:
            print("📊 Dataset is small - good for testing")
            print("   → For production, aim for 1M-10M positions")
        elif n_positions < 1_000_000:
            print("📊 Dataset is medium - good for initial training")
            print("   → For best results, scale to 10M+ positions")
        else:
            print("📊 Dataset is large - excellent for training")
            print("   → This should produce strong results")
        
        if value_balance > 0.2:
            print("\n⚠️  Consider rebalancing dataset:")
            print("   → Sample more positions from losing/winning sides")
        
        if np.mean(entropies) < 2.0:
            print("\n⚠️  Consider increasing temperature in policy generation:")
            print("   → Edit PolicyEncoder.create_policy_vector(temperature=3.0)")
        
        print("\n✅ Dataset analysis complete!")


def main():
    parser = argparse.ArgumentParser(description="Analyze DistillZero dataset")
    parser.add_argument('dataset', type=str, help='Path to HDF5 dataset file')
    
    args = parser.parse_args()
    dataset_path = Path(args.dataset)
    
    if not dataset_path.exists():
        print(f"❌ Error: Dataset not found: {dataset_path}")
        return 1
    
    try:
        analyze_dataset(dataset_path)
        return 0
    except Exception as e:
        print(f"\n❌ Error analyzing dataset: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == '__main__':
    exit(main())
