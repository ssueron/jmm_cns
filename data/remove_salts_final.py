#!/usr/bin/env python3
"""
Remove all SMILES containing the '.' (dot) character.

The dot character is used in SMILES to separate disconnected molecular fragments,
typically representing salts, counterions, or co-crystals. Since the S4 pretrained
vocabulary does not include the '.' token, these SMILES must be removed.
"""

import csv
from pathlib import Path
from typing import List, Dict


def has_dot(smiles: str) -> bool:
    """
    Check if SMILES contains a dot (disconnected fragment separator).

    Args:
        smiles: SMILES string

    Returns:
        True if contains '.', False otherwise
    """
    return '.' in smiles


def filter_csv_file(input_path: Path, output_path: Path) -> dict:
    """
    Filter CSV file to remove SMILES containing dots.

    Args:
        input_path: Path to input CSV file
        output_path: Path to output CSV file

    Returns:
        Dictionary with filtering statistics
    """
    stats = {
        'total_rows': 0,
        'removed_rows': 0,
        'kept_rows': 0,
        'removed_smiles': []
    }

    # Read the CSV file
    with open(input_path, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        rows = list(reader)
        fieldnames = reader.fieldnames

    stats['total_rows'] = len(rows)

    # Filter rows
    kept_rows = []
    for idx, row in enumerate(rows, start=2):  # Start at 2 to account for header
        smiles = row['smiles']

        if has_dot(smiles):
            stats['removed_rows'] += 1
            # Only store first 20 examples to avoid memory issues
            if len(stats['removed_smiles']) < 20:
                stats['removed_smiles'].append({
                    'row': idx,
                    'smiles': smiles,
                    'source': row.get('source_dataset', 'unknown')
                })
        else:
            kept_rows.append(row)
            stats['kept_rows'] += 1

    # Write filtered CSV file
    with open(output_path, 'w', encoding='utf-8', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(kept_rows)

    return stats


def print_report(filename: str, stats: dict):
    """Print filtering report."""
    print(f"\n{'='*60}")
    print(f"Filtering Report: {filename}")
    print(f"{'='*60}")
    print(f"Total SMILES: {stats['total_rows']}")
    print(f"Kept: {stats['kept_rows']} ({stats['kept_rows']/stats['total_rows']*100:.2f}%)")
    print(f"Removed (with dots): {stats['removed_rows']} ({stats['removed_rows']/stats['total_rows']*100:.2f}%)")

    if stats['removed_smiles']:
        print(f"\n{'-'*60}")
        print(f"Sample removed SMILES (first {len(stats['removed_smiles'])}):")
        print(f"{'-'*60}")
        for removed in stats['removed_smiles'][:10]:
            print(f"\nRow {removed['row']}:")
            # Truncate long SMILES for display
            smiles_display = removed['smiles']
            if len(smiles_display) > 80:
                smiles_display = smiles_display[:77] + "..."
            print(f"  SMILES: {smiles_display}")
            print(f"  Source: {removed['source']}")
            # Show what comes after the dot
            parts = removed['smiles'].split('.')
            if len(parts) > 1:
                print(f"  Main compound: {parts[0][:50]}{'...' if len(parts[0]) > 50 else ''}")
                print(f"  Salt/fragment: {'.'.join(parts[1:])}")


def main():
    """Main execution function."""
    data_dir = Path(__file__).parent

    # Files to process
    files_to_process = [
        ('test_10.csv', 'test_10.csv'),
        ('train_90.csv', 'train_90.csv')
    ]

    print("="*60)
    print("Removing SMILES with Dot (Salt) Character")
    print("="*60)
    print(f"\nThe '.' character separates disconnected fragments (salts).")
    print(f"Since '.' is not in the S4 vocabulary, these must be removed.")

    all_stats = {}

    for input_file, output_file in files_to_process:
        input_path = data_dir / input_file
        output_path = data_dir / output_file

        if not input_path.exists():
            print(f"WARNING: {input_path} does not exist. Skipping.")
            continue

        print(f"\nProcessing {input_file}...")
        stats = filter_csv_file(input_path, output_path)
        all_stats[input_file] = stats
        print_report(input_file, stats)

    # Overall summary
    print(f"\n{'='*60}")
    print("OVERALL SUMMARY")
    print(f"{'='*60}")
    total_removed = sum(s['removed_rows'] for s in all_stats.values())
    total_kept = sum(s['kept_rows'] for s in all_stats.values())
    total_rows = sum(s['total_rows'] for s in all_stats.values())

    print(f"Total SMILES processed: {total_rows}")
    print(f"Total SMILES kept: {total_kept} ({total_kept/total_rows*100:.2f}%)")
    print(f"Total SMILES removed (with dots): {total_removed} ({total_removed/total_rows*100:.2f}%)")

    print("\nSalt removal complete! CSV files have been updated.")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
