#!/usr/bin/env python3
"""
Create cleaned zip files from stereochemistry-free txt files.

This script:
1. Backs up existing train.zip and valid.zip
2. Creates new zip files from the cleaned train.txt and valid.txt
3. Verifies no stereochemistry markers (@, @@) remain in the data
"""

import zipfile
import os
import shutil
from pathlib import Path

def backup_file(filepath):
    """Backup a file by renaming it with _old suffix."""
    if os.path.exists(filepath):
        backup_path = filepath.replace('.zip', '_old.zip')
        shutil.move(filepath, backup_path)
        print(f"✓ Backed up {filepath} → {backup_path}")
        return backup_path
    else:
        print(f"  Note: {filepath} doesn't exist, no backup needed")
        return None

def create_zip_from_txt(txt_file, zip_file):
    """Create a zip file containing the txt file."""
    with zipfile.ZipFile(zip_file, 'w', zipfile.ZIP_DEFLATED) as zf:
        # Add the txt file to the zip with just its basename
        arcname = os.path.basename(txt_file)
        zf.write(txt_file, arcname=arcname)
    print(f"✓ Created {zip_file} from {txt_file}")

def count_smiles_and_check_stereo(filepath):
    """Count SMILES and check for stereochemistry markers."""
    if filepath.endswith('.zip'):
        with zipfile.ZipFile(filepath, 'r') as zf:
            fname = zf.namelist()[0]
            with zf.open(fname) as f:
                content = f.read().decode('utf-8')
    else:
        with open(filepath, 'r') as f:
            content = f.read()

    lines = content.strip().splitlines()
    num_smiles = len(lines)

    # Check for stereochemistry markers
    stereo_markers = {'@', '//', '\\\\'}  # @, @@, /, \
    has_at = '@' in content
    has_slash = '/' in content
    has_backslash = '\\' in content

    stereo_found = []
    if has_at:
        stereo_found.append('@')
    if has_slash:
        stereo_found.append('/')
    if has_backslash:
        stereo_found.append('\\')

    return num_smiles, stereo_found

def main():
    # Get script directory
    script_dir = Path(__file__).parent

    # File paths
    train_txt = script_dir / 'train.txt'
    valid_txt = script_dir / 'valid.txt'
    train_zip = script_dir / 'train.zip'
    valid_zip = script_dir / 'valid.zip'

    print("=" * 60)
    print("Creating Cleaned Zip Files from Stereochemistry-Free TXT")
    print("=" * 60)

    # Check input files exist
    if not train_txt.exists():
        print(f"ERROR: {train_txt} not found!")
        return
    if not valid_txt.exists():
        print(f"ERROR: {valid_txt} not found!")
        return

    print("\nStep 1: Analyzing input TXT files...")
    train_count, train_stereo = count_smiles_and_check_stereo(str(train_txt))
    valid_count, valid_stereo = count_smiles_and_check_stereo(str(valid_txt))

    print(f"  train.txt: {train_count:,} SMILES")
    if train_stereo:
        print(f"    WARNING: Found stereochemistry markers: {train_stereo}")
    else:
        print(f"    ✓ No stereochemistry markers found")

    print(f"  valid.txt: {valid_count:,} SMILES")
    if valid_stereo:
        print(f"    WARNING: Found stereochemistry markers: {valid_stereo}")
    else:
        print(f"    ✓ No stereochemistry markers found")

    # Backup existing zip files
    print("\nStep 2: Backing up existing zip files...")
    backup_file(str(train_zip))
    backup_file(str(valid_zip))

    # Create new zip files
    print("\nStep 3: Creating new zip files...")
    create_zip_from_txt(str(train_txt), str(train_zip))
    create_zip_from_txt(str(valid_txt), str(valid_zip))

    # Verify new zip files
    print("\nStep 4: Verifying new zip files...")
    train_zip_count, train_zip_stereo = count_smiles_and_check_stereo(str(train_zip))
    valid_zip_count, valid_zip_stereo = count_smiles_and_check_stereo(str(valid_zip))

    print(f"  train.zip: {train_zip_count:,} SMILES")
    if train_zip_stereo:
        print(f"    WARNING: Found stereochemistry markers: {train_zip_stereo}")
    else:
        print(f"    ✓ No stereochemistry markers found")

    print(f"  valid.zip: {valid_zip_count:,} SMILES")
    if valid_zip_stereo:
        print(f"    WARNING: Found stereochemistry markers: {valid_zip_stereo}")
    else:
        print(f"    ✓ No stereochemistry markers found")

    # Report file sizes
    print("\nStep 5: File sizes...")
    print(f"  train.txt: {train_txt.stat().st_size / 1024:.1f} KB")
    print(f"  train.zip: {train_zip.stat().st_size / 1024:.1f} KB")
    print(f"  valid.txt: {valid_txt.stat().st_size / 1024:.1f} KB")
    print(f"  valid.zip: {valid_zip.stat().st_size / 1024:.1f} KB")

    print("\n" + "=" * 60)
    print("SUCCESS! Cleaned zip files created.")
    print("=" * 60)
    print("\nYou can now run your finetune.py script using:")
    print("  training_molecules_path='data/train.zip'")
    print("  val_molecules_path='data/valid.zip'")
    print("\nThese zip files contain SMILES without stereochemistry markers.")
    print("=" * 60)

if __name__ == '__main__':
    main()
