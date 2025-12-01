#!/usr/bin/env python3
"""
Verify that all SMILES in the dataset are compatible with the S4 vocabulary.
Checks that all tokens can be mapped to the 37-token vocabulary.
"""

import csv
import zipfile
from pathlib import Path
from collections import Counter
from rdkit import Chem
from rdkit import RDLogger

# Suppress RDKit warnings
RDLogger.DisableLog('rdApp.*')

# S4 vocabulary - 37 tokens
S4_VOCABULARY = {
    "[PAD]", "[BEG]", "[END]",
    "C", "N", "O", "S", "F", "H", "P", "I", "Cl", "Br", "B",
    "c", "n", "o", "s",
    "[", "]", "(", ")",
    "=", "#", "-", "+",
    "1", "2", "3", "4", "5", "6", "7", "8", "9",
    "%10", "%11", "%12"
}

# Supported elements (atoms that can appear in molecules)
# Based on chembl_pretrained/init_arguments.json - only these 10 elements
SUPPORTED_ELEMENTS = {'C', 'N', 'O', 'S', 'F', 'H', 'P', 'I', 'Br', 'Cl'}


def get_elements_in_smiles(smiles: str) -> set:
    """Get all element symbols in a SMILES string."""
    try:
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return set()
        return {atom.GetSymbol() for atom in mol.GetAtoms()}
    except:
        return set()


def read_csv_smiles(csv_path: Path) -> list:
    """Read SMILES from CSV file."""
    with open(csv_path, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        return [row['smiles'] for row in reader]


def read_txt_smiles(txt_path: Path) -> list:
    """Read SMILES from TXT file."""
    with open(txt_path, 'r', encoding='utf-8') as f:
        return [line.strip() for line in f if line.strip()]


def read_zip_smiles(zip_path: Path) -> list:
    """Read SMILES from ZIP file."""
    with zipfile.ZipFile(zip_path, 'r') as zf:
        fname = zf.namelist()[0]
        with zf.open(fname) as f:
            content = f.read().decode('utf-8')
            return [line.strip() for line in content.strip().splitlines() if line.strip()]


def check_vocabulary_compatibility(smiles_list: list) -> dict:
    """Check if all SMILES are compatible with S4 vocabulary."""
    stats = {
        'total': len(smiles_list),
        'compatible': 0,
        'incompatible': 0,
        'incompatible_smiles': [],
        'all_elements': Counter(),
        'unsupported_elements': Counter()
    }

    for smiles in smiles_list:
        elements = get_elements_in_smiles(smiles)
        stats['all_elements'].update(elements)

        unsupported = elements - SUPPORTED_ELEMENTS
        if unsupported:
            stats['incompatible'] += 1
            stats['unsupported_elements'].update(unsupported)
            if len(stats['incompatible_smiles']) < 10:  # Store first 10 examples
                stats['incompatible_smiles'].append({
                    'smiles': smiles,
                    'unsupported': sorted(unsupported)
                })
        else:
            stats['compatible'] += 1

    return stats


def main():
    """Main verification function."""
    data_dir = Path(__file__).parent

    print("="*60)
    print("S4 VOCABULARY COMPATIBILITY VERIFICATION")
    print("="*60)
    print(f"\nS4 Vocabulary: {len(S4_VOCABULARY)} tokens")
    print(f"Supported elements: {', '.join(sorted(SUPPORTED_ELEMENTS))}")

    datasets = [
        {
            'name': 'Training Set',
            'csv': data_dir / 'train_90.csv',
            'txt': data_dir / 'train.txt',
            'zip': data_dir / 'train.zip',
        },
        {
            'name': 'Validation Set',
            'csv': data_dir / 'test_10.csv',
            'txt': data_dir / 'valid.txt',
            'zip': data_dir / 'valid.zip',
        }
    ]

    all_passed = True
    grand_totals = {
        'total': 0,
        'compatible': 0,
        'incompatible': 0,
        'all_elements': Counter(),
        'unsupported_elements': Counter()
    }

    for dataset in datasets:
        print(f"\n{'='*60}")
        print(f"{dataset['name']}")
        print(f"{'='*60}")

        # Read CSV file
        csv_smiles = read_csv_smiles(dataset['csv'])
        print(f"\nTotal SMILES: {len(csv_smiles)}")

        # Check compatibility
        stats = check_vocabulary_compatibility(csv_smiles)

        print(f"\nCompatibility Check:")
        print(f"  Compatible: {stats['compatible']} ({stats['compatible']/stats['total']*100:.2f}%)")
        print(f"  Incompatible: {stats['incompatible']} ({stats['incompatible']/stats['total']*100:.2f}%)")

        if stats['incompatible'] > 0:
            print(f"\n  ✗ FAIL - Incompatible SMILES found!")
            print(f"\n  Unsupported elements:")
            for element, count in stats['unsupported_elements'].most_common():
                print(f"    {element}: {count} occurrences")

            print(f"\n  Sample incompatible SMILES:")
            for item in stats['incompatible_smiles'][:5]:
                print(f"    {item['smiles']}")
                print(f"      Unsupported: {', '.join(item['unsupported'])}")

            all_passed = False
        else:
            print(f"  ✓ PASS - All SMILES compatible with S4 vocabulary")

        # Show element distribution
        print(f"\nElement Distribution:")
        for element, count in sorted(stats['all_elements'].items()):
            pct = count / stats['total'] * 100
            supported = '✓' if element in SUPPORTED_ELEMENTS else '✗'
            print(f"  {supported} {element}: {count} ({pct:.1f}%)")

        # Update grand totals
        grand_totals['total'] += stats['total']
        grand_totals['compatible'] += stats['compatible']
        grand_totals['incompatible'] += stats['incompatible']
        grand_totals['all_elements'].update(stats['all_elements'])
        grand_totals['unsupported_elements'].update(stats['unsupported_elements'])

        # Verify file synchronization
        txt_smiles = read_txt_smiles(dataset['txt'])
        zip_smiles = read_zip_smiles(dataset['zip'])

        print(f"\nFile Synchronization:")
        print(f"  CSV: {len(csv_smiles)} SMILES")
        print(f"  TXT: {len(txt_smiles)} SMILES")
        print(f"  ZIP: {len(zip_smiles)} SMILES")

        if len(csv_smiles) == len(txt_smiles) == len(zip_smiles) and csv_smiles == txt_smiles == zip_smiles:
            print(f"  ✓ All files synchronized")
        else:
            print(f"  ✗ Files not synchronized!")
            all_passed = False

    # Grand summary
    print(f"\n{'='*60}")
    print("GRAND SUMMARY")
    print(f"{'='*60}")
    print(f"Total SMILES: {grand_totals['total']}")
    print(f"Compatible: {grand_totals['compatible']} ({grand_totals['compatible']/grand_totals['total']*100:.2f}%)")
    print(f"Incompatible: {grand_totals['incompatible']} ({grand_totals['incompatible']/grand_totals['total']*100:.2f}%)")

    if grand_totals['unsupported_elements']:
        print(f"\nUnsupported elements found:")
        for element, count in grand_totals['unsupported_elements'].most_common():
            print(f"  {element}: {count} occurrences")

    print(f"\n{'='*60}")
    if all_passed:
        print("✓ ALL CHECKS PASSED!")
        print("\nYour dataset is fully compatible with the S4 vocabulary.")
        print("You can now train without tokenization errors.")
    else:
        print("✗ SOME CHECKS FAILED")
        print("\nPlease review the issues above.")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
