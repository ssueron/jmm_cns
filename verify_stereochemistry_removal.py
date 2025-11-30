#!/usr/bin/env python3
"""
Verification script to confirm stereochemistry has been removed.
"""

import pandas as pd
from pathlib import Path


def count_stereo_markers(smiles):
    """Count stereochemistry markers in a SMILES string."""
    stereo_count = 0
    stereo_count += smiles.count('@')  # Chiral centers
    stereo_count += smiles.count('/')  # E/Z double bonds
    stereo_count += smiles.count('\\')  # E/Z double bonds
    return stereo_count


def analyze_file(filepath):
    """Analyze a CSV file for stereochemistry markers."""
    df = pd.read_csv(filepath)

    total_molecules = len(df)
    total_stereo_markers = 0
    molecules_with_stereo = 0

    for smiles in df['smiles']:
        count = count_stereo_markers(smiles)
        total_stereo_markers += count
        if count > 0:
            molecules_with_stereo += 1

    return {
        'total_molecules': total_molecules,
        'molecules_with_stereo': molecules_with_stereo,
        'total_stereo_markers': total_stereo_markers,
        'percent_with_stereo': (molecules_with_stereo / total_molecules * 100) if total_molecules > 0 else 0
    }


def main():
    """Main verification function."""
    data_dir = Path("data")

    print("="*70)
    print("STEREOCHEMISTRY REMOVAL VERIFICATION")
    print("="*70)

    for filename in ['train_90.csv', 'test_10.csv']:
        filepath = data_dir / filename
        print(f"\n{filename}:")
        print("-" * 70)

        stats = analyze_file(filepath)

        print(f"  Total molecules: {stats['total_molecules']:,}")
        print(f"  Molecules with stereochemistry: {stats['molecules_with_stereo']:,}")
        print(f"  Total stereochemistry markers (@, /, \\): {stats['total_stereo_markers']:,}")
        print(f"  Percentage with stereochemistry: {stats['percent_with_stereo']:.2f}%")

        if stats['total_stereo_markers'] == 0:
            print("  ✓ SUCCESS: No stereochemistry markers found!")
        else:
            print("  ✗ WARNING: Stereochemistry markers still present!")

    print("\n" + "="*70)
    print("VERIFICATION COMPLETE")
    print("="*70)


if __name__ == "__main__":
    main()
