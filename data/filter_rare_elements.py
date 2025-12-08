#!/usr/bin/env python3
"""
Filter SMILES containing rare/unsupported elements (Li, Ne, Ar, etc.)
while preserving SMILES with [nH], [N+], [O-] which are supported by the S4 vocabulary.

The S4 model vocabulary (37 tokens) supports ONLY these 10 elements:
- Atoms: C, N, O, S, F, H, P, I, Br, Cl (+ aromatic variants: c, n, o, s)
- Bracket expressions like [nH], [N+], [O-] work because they decompose into supported tokens
  Example: [nH] = '[', 'n', 'H', ']' (all in vocabulary)
- NOTE: B (Boron) is NOT supported despite being common in some drug molecules

This script ONLY removes SMILES with elements NOT in the supported list.
"""

import csv
from pathlib import Path
from typing import Tuple, List, Set
from rdkit import Chem
from rdkit import RDLogger

# Suppress RDKit warnings
RDLogger.DisableLog('rdApp.*')


# Elements supported by S4 vocabulary (can appear as individual tokens)
# Based on chembl_pretrained/init_arguments.json - only these 10 elements are in the vocabulary
SUPPORTED_ELEMENTS = {
    'C', 'N', 'O', 'S', 'F', 'H', 'P', 'I', 'Br', 'Cl'
}


def get_elements_in_smiles(smiles: str) -> Set[str]:
    """
    Get all element symbols present in a SMILES string.

    Args:
        smiles: SMILES string

    Returns:
        Set of element symbols (e.g., {'C', 'N', 'O', 'Li'})
    """
    try:
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return set()

        elements = set()
        for atom in mol.GetAtoms():
            elements.add(atom.GetSymbol())

        return elements
    except:
        return set()


def has_unsupported_elements(smiles: str) -> Tuple[bool, List[str]]:
    """
    Check if SMILES contains elements not supported by S4 vocabulary.

    Args:
        smiles: SMILES string

    Returns:
        Tuple of (has_unsupported, list_of_unsupported_elements)
    """
    elements = get_elements_in_smiles(smiles)
    unsupported = sorted(elements - SUPPORTED_ELEMENTS)

    return len(unsupported) > 0, unsupported


def filter_csv_file(input_path: Path, output_path: Path) -> dict:
    """
    Filter CSV file to remove SMILES with unsupported elements.

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
        has_unsupported, unsupported_elements = has_unsupported_elements(smiles)

        if has_unsupported:
            stats['removed_rows'] += 1
            stats['removed_smiles'].append({
                'row': idx,
                'smiles': smiles,
                'unsupported_elements': unsupported_elements,
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
    print(f"Removed: {stats['removed_rows']} ({stats['removed_rows']/stats['total_rows']*100:.2f}%)")

    if stats['removed_smiles']:
        print(f"\n{'-'*60}")
        print("Removed SMILES (containing unsupported elements):")
        print(f"{'-'*60}")
        for removed in stats['removed_smiles']:
            print(f"\nRow {removed['row']}:")
            print(f"  SMILES: {removed['smiles']}")
            print(f"  Unsupported elements: {', '.join(removed['unsupported_elements'])}")
            print(f"  Source: {removed['source']}")


def main():
    """Main execution function."""
    data_dir = Path(__file__).parent

    # Files to process
    files_to_process = [
        ('test_10.csv', 'test_10.csv'),
        ('train_90.csv', 'train_90.csv')
    ]

    print("="*60)
    print("Filtering SMILES with Unsupported Elements")
    print("="*60)
    print(f"\nSupported elements (S4 vocabulary): {', '.join(sorted(SUPPORTED_ELEMENTS))}")
    print(f"\nAny other elements (Li, Na, K, Ca, Ne, Ar, metals, etc.) will be filtered out.")
    print(f"Note: [nH], [N+], [O-] are KEPT (they work with the vocabulary)")

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
    print(f"Total SMILES removed: {total_removed} ({total_removed/total_rows*100:.2f}%)")

    # List all unique unsupported elements found
    all_unsupported = set()
    for stats in all_stats.values():
        for removed in stats['removed_smiles']:
            all_unsupported.update(removed['unsupported_elements'])

    if all_unsupported:
        print(f"\nUnsupported elements found: {', '.join(sorted(all_unsupported))}")

    print("\nFiltering complete! CSV files have been updated.")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
