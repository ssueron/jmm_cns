#!/usr/bin/env python3
"""
Script to remove stereochemistry from SMILES in molecular datasets.
Updates CSV files and corresponding text files.
"""

import pandas as pd
from rdkit import Chem
from pathlib import Path


def remove_stereochemistry(smiles):
    """
    Remove stereochemistry from a SMILES string.

    Args:
        smiles: SMILES string with potential stereochemistry

    Returns:
        SMILES string without stereochemistry, or None if parsing fails
    """
    try:
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            print(f"Warning: Could not parse SMILES: {smiles}")
            return None
        # Generate canonical SMILES without stereochemistry
        non_stereo_smiles = Chem.MolToSmiles(mol, isomericSmiles=False)
        return non_stereo_smiles
    except Exception as e:
        print(f"Error processing SMILES {smiles}: {e}")
        return None


def process_csv_file(input_path, output_path):
    """
    Process a CSV file to remove stereochemistry from SMILES column.

    Args:
        input_path: Path to input CSV file
        output_path: Path to output CSV file
    """
    print(f"Processing {input_path}...")

    # Read CSV
    df = pd.read_csv(input_path)

    print(f"  Original rows: {len(df)}")
    print(f"  Columns: {list(df.columns)}")

    # Remove stereochemistry from SMILES
    df['smiles'] = df['smiles'].apply(remove_stereochemistry)

    # Remove any rows where SMILES parsing failed
    failed_count = df['smiles'].isna().sum()
    if failed_count > 0:
        print(f"  Warning: {failed_count} SMILES failed to parse and will be removed")
        df = df.dropna(subset=['smiles'])

    print(f"  Final rows: {len(df)}")

    # Save to CSV
    df.to_csv(output_path, index=False)
    print(f"  Saved to {output_path}")

    return df


def update_txt_file(df, output_path):
    """
    Create/update a .txt file with one SMILES per line.

    Args:
        df: DataFrame with 'smiles' column
        output_path: Path to output .txt file
    """
    print(f"Writing SMILES to {output_path}...")

    with open(output_path, 'w') as f:
        for smiles in df['smiles']:
            f.write(f"{smiles}\n")

    print(f"  Wrote {len(df)} SMILES to {output_path}")


def main():
    """Main execution function."""
    data_dir = Path("data")

    # Process train_90.csv
    print("\n" + "="*60)
    print("PROCESSING TRAINING DATA")
    print("="*60)
    train_df = process_csv_file(
        data_dir / "train_90.csv",
        data_dir / "train_90.csv"
    )

    # Update train.txt
    update_txt_file(train_df, data_dir / "train.txt")

    # Process test_10.csv
    print("\n" + "="*60)
    print("PROCESSING TEST/VALIDATION DATA")
    print("="*60)
    test_df = process_csv_file(
        data_dir / "test_10.csv",
        data_dir / "test_10.csv"
    )

    # Update valid.txt (corresponds to test_10.csv)
    update_txt_file(test_df, data_dir / "valid.txt")

    print("\n" + "="*60)
    print("SUMMARY")
    print("="*60)
    print(f"Training samples: {len(train_df)}")
    print(f"Validation/Test samples: {len(test_df)}")
    print("\nFiles updated:")
    print("  - data/train_90.csv")
    print("  - data/test_10.csv")
    print("  - data/train.txt")
    print("  - data/valid.txt")
    print("\nNote: train.zip and valid.zip need to be regenerated separately")
    print("="*60)


if __name__ == "__main__":
    main()
