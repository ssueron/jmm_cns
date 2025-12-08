#!/usr/bin/env python3
"""
Desalt SMILES in CSV files by removing salt fragments.
Keeps the largest fragment when multiple components are present (separated by '.').
"""

import csv
from pathlib import Path
from typing import Tuple, List
from rdkit import Chem
from rdkit import RDLogger

# Suppress RDKit warnings
RDLogger.DisableLog('rdApp.*')


def desalt_smiles(smiles: str) -> Tuple[str, bool]:
    """
    Remove salts from SMILES by selecting the largest fragment.

    Args:
        smiles: Input SMILES string (may contain '.' separators)

    Returns:
        Tuple of (desalted_smiles, was_modified)
    """
    # Check if there are multiple components
    if '.' not in smiles:
        return smiles, False

    # Split by '.' and select largest fragment by string length
    fragments = smiles.split('.')
    largest_fragment = max(fragments, key=len)

    return largest_fragment, True


def validate_smiles(smiles: str) -> bool:
    """
    Validate SMILES using RDKit.

    Args:
        smiles: SMILES string to validate

    Returns:
        True if valid, False otherwise
    """
    try:
        mol = Chem.MolFromSmiles(smiles)
        return mol is not None
    except:
        return False


def process_csv_file(input_path: Path, output_path: Path) -> dict:
    """
    Process a CSV file to desalt SMILES.

    Args:
        input_path: Path to input CSV file
        output_path: Path to output CSV file

    Returns:
        Dictionary with processing statistics
    """
    stats = {
        'total_rows': 0,
        'modified_rows': 0,
        'invalid_smiles': [],
        'modifications': []
    }

    # Read the CSV file
    with open(input_path, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        rows = list(reader)
        fieldnames = reader.fieldnames

    stats['total_rows'] = len(rows)

    # Process each row
    for idx, row in enumerate(rows, start=2):  # Start at 2 to account for header
        original_smiles = row['smiles']
        desalted_smiles, was_modified = desalt_smiles(original_smiles)

        if was_modified:
            # Validate the desalted SMILES
            if validate_smiles(desalted_smiles):
                row['smiles'] = desalted_smiles
                stats['modified_rows'] += 1
                stats['modifications'].append({
                    'row': idx,
                    'original': original_smiles,
                    'desalted': desalted_smiles
                })
            else:
                stats['invalid_smiles'].append({
                    'row': idx,
                    'original': original_smiles,
                    'desalted': desalted_smiles
                })
                # Keep original if desalted version is invalid
                print(f"WARNING: Row {idx} produced invalid SMILES after desalting. Keeping original.")

    # Write the updated CSV file
    with open(output_path, 'w', encoding='utf-8', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    return stats


def print_report(filename: str, stats: dict):
    """Print processing report."""
    print(f"\n{'='*60}")
    print(f"Processing Report: {filename}")
    print(f"{'='*60}")
    print(f"Total rows processed: {stats['total_rows']}")
    print(f"Rows modified (desalted): {stats['modified_rows']}")
    print(f"Invalid SMILES after desalting: {len(stats['invalid_smiles'])}")

    if stats['modifications']:
        print(f"\n{'-'*60}")
        print("Sample modifications (first 5):")
        print(f"{'-'*60}")
        for mod in stats['modifications'][:5]:
            print(f"\nRow {mod['row']}:")
            print(f"  Original: {mod['original']}")
            print(f"  Desalted: {mod['desalted']}")

    if stats['invalid_smiles']:
        print(f"\n{'-'*60}")
        print("WARNING: Invalid SMILES found:")
        print(f"{'-'*60}")
        for invalid in stats['invalid_smiles']:
            print(f"\nRow {invalid['row']}:")
            print(f"  Original: {invalid['original']}")
            print(f"  Desalted (INVALID): {invalid['desalted']}")


def main():
    """Main execution function."""
    data_dir = Path(__file__).parent

    # Files to process
    files_to_process = [
        ('test_10.csv', 'test_10.csv'),
        ('train_90.csv', 'train_90.csv')
    ]

    print("Starting SMILES desalting process...")
    print(f"Using RDKit for validation")

    all_stats = {}

    for input_file, output_file in files_to_process:
        input_path = data_dir / input_file
        output_path = data_dir / output_file

        if not input_path.exists():
            print(f"WARNING: {input_path} does not exist. Skipping.")
            continue

        print(f"\nProcessing {input_file}...")
        stats = process_csv_file(input_path, output_path)
        all_stats[input_file] = stats
        print_report(input_file, stats)

    # Overall summary
    print(f"\n{'='*60}")
    print("OVERALL SUMMARY")
    print(f"{'='*60}")
    total_modified = sum(s['modified_rows'] for s in all_stats.values())
    total_invalid = sum(len(s['invalid_smiles']) for s in all_stats.values())
    total_rows = sum(s['total_rows'] for s in all_stats.values())

    print(f"Total SMILES processed: {total_rows}")
    print(f"Total SMILES desalted: {total_modified}")
    print(f"Total invalid SMILES: {total_invalid}")
    print(f"Success rate: {(total_modified - total_invalid) / total_modified * 100:.2f}%")
    print("\nDesalting complete! CSV files have been updated.")


if __name__ == "__main__":
    main()
