#!/usr/bin/env python3
"""
Update txt files from desalted CSV files.
Extracts only the SMILES column from CSV files.
"""

import csv
from pathlib import Path


def csv_to_txt(csv_path: Path, txt_path: Path) -> int:
    """
    Extract SMILES from CSV and write to txt file.

    Args:
        csv_path: Path to input CSV file
        txt_path: Path to output txt file

    Returns:
        Number of SMILES written
    """
    smiles_list = []

    # Read CSV and extract SMILES column
    with open(csv_path, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            smiles_list.append(row['smiles'])

    # Write SMILES to txt file (one per line)
    with open(txt_path, 'w', encoding='utf-8') as f:
        for smiles in smiles_list:
            f.write(f"{smiles}\n")

    return len(smiles_list)


def main():
    """Main execution function."""
    data_dir = Path(__file__).parent

    # Mapping: CSV file -> txt file
    conversions = [
        ('train_90.csv', 'train.txt'),
        ('test_10.csv', 'valid.txt')
    ]

    print("Updating txt files from desalted CSV files...")
    print(f"{'='*60}")

    for csv_file, txt_file in conversions:
        csv_path = data_dir / csv_file
        txt_path = data_dir / txt_file

        if not csv_path.exists():
            print(f"ERROR: {csv_path} does not exist. Skipping.")
            continue

        count = csv_to_txt(csv_path, txt_path)
        print(f"✓ {csv_file} -> {txt_file}: {count} SMILES written")

    print(f"{'='*60}")
    print("txt files updated successfully!")


if __name__ == "__main__":
    main()
