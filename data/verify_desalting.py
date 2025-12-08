#!/usr/bin/env python3
"""
Verify integrity of desalted files.
Checks:
1. Line counts match expected values
2. No salts remain (no '.' in SMILES)
3. No stereochemistry markers
4. CSV and TXT files are synchronized
"""

import csv
import zipfile
from pathlib import Path


def read_csv_smiles(csv_path: Path) -> list:
    """Read SMILES from CSV file."""
    smiles_list = []
    with open(csv_path, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            smiles_list.append(row['smiles'])
    return smiles_list


def read_txt_smiles(txt_path: Path) -> list:
    """Read SMILES from TXT file."""
    with open(txt_path, 'r', encoding='utf-8') as f:
        return [line.strip() for line in f]


def read_zip_smiles(zip_path: Path) -> list:
    """Read SMILES from ZIP file."""
    with zipfile.ZipFile(zip_path, 'r') as zf:
        fname = zf.namelist()[0]
        with zf.open(fname) as f:
            content = f.read().decode('utf-8')
            return [line.strip() for line in content.strip().splitlines()]


def check_for_salts(smiles_list: list) -> dict:
    """Check if any SMILES contain salts (indicated by '.')."""
    salted = [s for s in smiles_list if '.' in s]
    return {
        'total': len(smiles_list),
        'salted': len(salted),
        'samples': salted[:5] if salted else []
    }


def check_stereochemistry(smiles_list: list) -> dict:
    """Check if any SMILES contain stereochemistry markers."""
    markers = {'@', '/', '\\'}
    smiles_with_markers = []

    for smiles in smiles_list:
        has_markers = []
        if '@' in smiles:
            has_markers.append('@')
        if '/' in smiles:
            has_markers.append('/')
        if '\\' in smiles:
            has_markers.append('\\')
        if has_markers:
            smiles_with_markers.append((smiles, has_markers))

    return {
        'total': len(smiles_list),
        'with_stereo': len(smiles_with_markers),
        'samples': smiles_with_markers[:5]
    }


def verify_synchronization(csv_smiles: list, txt_smiles: list, zip_smiles: list, name: str) -> bool:
    """Verify that CSV, TXT, and ZIP contain the same SMILES."""
    print(f"\n{'-'*60}")
    print(f"Verifying synchronization: {name}")
    print(f"{'-'*60}")

    all_match = True

    # Check counts
    print(f"Line counts:")
    print(f"  CSV: {len(csv_smiles)}")
    print(f"  TXT: {len(txt_smiles)}")
    print(f"  ZIP: {len(zip_smiles)}")

    if len(csv_smiles) != len(txt_smiles) or len(csv_smiles) != len(zip_smiles):
        print(f"  ✗ MISMATCH: Line counts don't match!")
        all_match = False
    else:
        print(f"  ✓ All line counts match")

    # Check content
    if csv_smiles == txt_smiles == zip_smiles:
        print(f"  ✓ All SMILES content matches exactly")
    else:
        print(f"  ✗ MISMATCH: SMILES content differs!")
        all_match = False

        # Find first difference
        for i, (csv_s, txt_s, zip_s) in enumerate(zip(csv_smiles, txt_smiles, zip_smiles)):
            if csv_s != txt_s or csv_s != zip_s:
                print(f"  First difference at line {i+1}:")
                print(f"    CSV: {csv_s}")
                print(f"    TXT: {txt_s}")
                print(f"    ZIP: {zip_s}")
                break

    return all_match


def main():
    """Main verification function."""
    data_dir = Path(__file__).parent

    print("="*60)
    print("DESALTING VERIFICATION REPORT")
    print("="*60)

    # Files to verify
    datasets = [
        {
            'name': 'Training Set',
            'csv': data_dir / 'train_90.csv',
            'txt': data_dir / 'train.txt',
            'zip': data_dir / 'train.zip',
            'expected_count': 11790
        },
        {
            'name': 'Validation Set',
            'csv': data_dir / 'test_10.csv',
            'txt': data_dir / 'valid.txt',
            'zip': data_dir / 'valid.zip',
            'expected_count': 1310
        }
    ]

    all_passed = True

    for dataset in datasets:
        print(f"\n{'='*60}")
        print(f"{dataset['name']}")
        print(f"{'='*60}")

        # Read files
        csv_smiles = read_csv_smiles(dataset['csv'])
        txt_smiles = read_txt_smiles(dataset['txt'])
        zip_smiles = read_zip_smiles(dataset['zip'])

        # Check line counts
        print(f"\n1. Line Count Verification:")
        print(f"   Expected: {dataset['expected_count']}")
        print(f"   Actual:   {len(csv_smiles)}")
        if len(csv_smiles) == dataset['expected_count']:
            print(f"   ✓ PASS")
        else:
            print(f"   ✗ FAIL")
            all_passed = False

        # Check for salts
        print(f"\n2. Salt Check (looking for '.' in SMILES):")
        salt_check = check_for_salts(csv_smiles)
        print(f"   Total SMILES: {salt_check['total']}")
        print(f"   SMILES with salts: {salt_check['salted']}")
        if salt_check['salted'] == 0:
            print(f"   ✓ PASS - No salts found")
        else:
            print(f"   ✗ FAIL - Salts still present!")
            print(f"   Samples: {salt_check['samples']}")
            all_passed = False

        # Check for stereochemistry
        print(f"\n3. Stereochemistry Check:")
        stereo_check = check_stereochemistry(csv_smiles)
        print(f"   Total SMILES: {stereo_check['total']}")
        print(f"   SMILES with stereochemistry: {stereo_check['with_stereo']}")
        if stereo_check['with_stereo'] == 0:
            print(f"   ✓ PASS - No stereochemistry markers found")
        else:
            print(f"   ✗ FAIL - Stereochemistry markers present!")
            for smiles, markers in stereo_check['samples']:
                print(f"     {smiles} (markers: {markers})")
            all_passed = False

        # Check synchronization
        sync_pass = verify_synchronization(csv_smiles, txt_smiles, zip_smiles, dataset['name'])
        if not sync_pass:
            all_passed = False

    # Final summary
    print(f"\n{'='*60}")
    print("FINAL SUMMARY")
    print(f"{'='*60}")

    if all_passed:
        print("✓ ALL CHECKS PASSED!")
        print("\nYour data is clean and ready for training:")
        print("  • All salts removed")
        print("  • No stereochemistry markers")
        print("  • CSV, TXT, and ZIP files synchronized")
        print("  • Correct line counts")
    else:
        print("✗ SOME CHECKS FAILED")
        print("\nPlease review the issues above.")

    print(f"{'='*60}")


if __name__ == "__main__":
    main()
