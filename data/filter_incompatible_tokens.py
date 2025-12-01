#!/usr/bin/env python3
"""
Filter SMILES to match pretrained S4 vocabulary.

The pretrained S4 model supports only these 37 tokens:
- Special: [PAD], [BEG], [END]
- Atoms: C, N, O, S, F, H, P, I, Cl, Br (and aromatic: c, n, o, s)
- Bonds: =, #, -, +
- Brackets: [, ]
- Parentheses: (, )
- Digits: 1-9, %10, %11, %12

This script removes SMILES containing unsupported tokens like:
- [nH], [N+], [O-], [S+] (charged/modified atoms)
- Li, Si, Na (unsupported elements)
- Ring numbers > %12
"""

import re
import zipfile
from pathlib import Path
from collections import Counter

# Pretrained S4 vocabulary (from chembl_pretrained/init_arguments.json)
SUPPORTED_TOKENS = {
    '[PAD]', '[BEG]', '[END]',
    'O', '=', 'C', '(', 'c', '1', '2', ')', 'N', '3', 'n', '#', 'S', 'F',
    'Cl', '[', 'H', ']', '-', '4', '5', '6', 's', 'o', 'P', 'I', '+',
    'Br', '7', '8', '9', '%10', '%11', '%12'
}

# Unsupported patterns to filter out
UNSUPPORTED_PATTERNS = [
    # Charged atoms (most common issues)
    r'\[nH\]',      # Pyrrole nitrogen - VERY COMMON (~21% of data)
    r'\[N\+\]',     # Charged nitrogen
    r'\[O-\]',      # Negatively charged oxygen
    r'\[S\+\]',     # Charged sulfur
    r'\[n\+\]',     # Charged aromatic nitrogen
    r'\[N-\]',      # Negatively charged nitrogen
    r'\[C\+\]',     # Charged carbon
    r'\[C-\]',      # Negatively charged carbon
    r'\[O\+\]',     # Charged oxygen
    r'\[P\+\]',     # Charged phosphorus
    r'\[o\+\]',     # Charged aromatic oxygen

    # Bracketed atoms
    r'\[NH\]',      # Bracketed NH
    r'\[NH-\]',     # Charged NH
    r'\[SH\]',      # Bracketed SH
    r'\[N\]',       # Standalone bracketed N
    r'\[S\]',       # Standalone bracketed S
    r'\[Cl-\]',     # Charged chlorine

    # Unsupported elements (Li, Si, Na, etc.)
    r'\[Li\+?\]',   # Lithium (charged or neutral)
    r'\[Si',        # Silicon (any form)
    r'\[Na\+?\]',   # Sodium
    r'\[K\+?\]',    # Potassium
    r'\[Ca',        # Calcium
    r'\[Mg',        # Magnesium
    r'\[Al',        # Aluminum
    r'\[Fe',        # Iron
    r'\[Cu',        # Copper
    r'\[Zn',        # Zinc
    r'\[Ag',        # Silver
    r'\[Au',        # Gold
    r'\[Pt',        # Platinum
    r'\[Pd',        # Palladium
    r'\[Ni',        # Nickel
    r'\[Co',        # Cobalt
    r'\[Mn',        # Manganese
    r'\[Cr',        # Chromium
    r'\[Ti',        # Titanium
    r'\[B\]',       # Boron
    r'\[Se',        # Selenium
    r'\[Te',        # Tellurium
    r'\[As',        # Arsenic
    r'\[Sn',        # Tin
    r'\[Pb',        # Lead
    r'\[Hg',        # Mercury
    r'\[Ne\]',      # Neon
    r'\[Ar\]',      # Argon
    r'\[Kr\]',      # Krypton
    r'\[Xe\]',      # Xenon

    # Ring numbers beyond %12
    r'%1[3-9]',     # %13-%19
    r'%[2-9]\d',    # %20-%99

    # Standalone unsupported elements (not in brackets)
    r'(?<![A-Z])Li(?![a-z])',  # Standalone Li
    r'(?<![A-Z])Si(?![a-z])',  # Standalone Si
    r'(?<![A-Z])Na(?![a-z])',  # Standalone Na
    r'(?<![A-Z])Mg(?![a-z])',  # Standalone Mg
    r'(?<![A-Z])Al(?![a-z])',  # Standalone Al
    r'(?<![A-Z])Ca(?![a-z])',  # Standalone Ca
    r'(?<![A-Z])Fe(?![a-z])',  # Standalone Fe
    r'(?<![A-Z])Zn(?![a-z])',  # Standalone Zn
]

# Compile patterns for efficiency
COMPILED_PATTERNS = [re.compile(pattern) for pattern in UNSUPPORTED_PATTERNS]

def contains_unsupported_token(smiles):
    """Check if SMILES contains any unsupported tokens."""
    for pattern in COMPILED_PATTERNS:
        if pattern.search(smiles):
            return True, pattern.pattern
    return False, None

def filter_smiles_file(input_file, output_file):
    """Filter SMILES from input file and write compatible ones to output."""
    print(f"Reading from: {input_file}")

    # Read input SMILES
    if str(input_file).endswith('.zip'):
        with zipfile.ZipFile(input_file, 'r') as zf:
            fname = zf.namelist()[0]
            with zf.open(fname) as f:
                all_smiles = f.read().decode('utf-8').splitlines()
    else:
        with open(input_file, 'r') as f:
            all_smiles = f.read().splitlines()

    print(f"  Total SMILES: {len(all_smiles):,}")

    # Filter SMILES
    compatible_smiles = []
    incompatible_reasons = Counter()

    for smiles in all_smiles:
        is_incompatible, reason = contains_unsupported_token(smiles)
        if is_incompatible:
            incompatible_reasons[reason] += 1
        else:
            compatible_smiles.append(smiles)

    print(f"  Compatible SMILES: {len(compatible_smiles):,}")
    print(f"  Removed: {len(all_smiles) - len(compatible_smiles):,} ({100*(len(all_smiles)-len(compatible_smiles))/len(all_smiles):.1f}%)")

    # Show top reasons for removal
    if incompatible_reasons:
        print(f"\n  Top reasons for removal:")
        for reason, count in incompatible_reasons.most_common(10):
            print(f"    {reason}: {count:,} SMILES")

    # Write output
    with open(output_file, 'w') as f:
        f.write('\n'.join(compatible_smiles))

    print(f"  Wrote to: {output_file}")

    return len(all_smiles), len(compatible_smiles), incompatible_reasons

def main():
    script_dir = Path(__file__).parent

    # Input files
    train_txt = script_dir / 'train.txt'
    valid_txt = script_dir / 'valid.txt'

    # Output files
    train_filtered = script_dir / 'train_filtered.txt'
    valid_filtered = script_dir / 'valid_filtered.txt'

    print("=" * 70)
    print("Filtering SMILES for S4 Pretrained Model Compatibility")
    print("=" * 70)
    print("\nSupported vocabulary (37 tokens):")
    print("  Atoms: C, N, O, S, F, H, P, I, Cl, Br (+ aromatic: c, n, o, s)")
    print("  Other: [, ], (, ), =, #, -, +, 1-9, %10, %11, %12")
    print("=" * 70)

    # Check input files
    if not train_txt.exists():
        print(f"ERROR: {train_txt} not found!")
        return
    if not valid_txt.exists():
        print(f"ERROR: {valid_txt} not found!")
        return

    # Filter training data
    print("\n[1/2] Filtering training data...")
    train_total, train_kept, train_reasons = filter_smiles_file(train_txt, train_filtered)

    # Filter validation data
    print("\n[2/2] Filtering validation data...")
    valid_total, valid_kept, valid_reasons = filter_smiles_file(valid_txt, valid_filtered)

    # Summary
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print(f"Training:   {train_total:,} → {train_kept:,} ({100*train_kept/train_total:.1f}% kept)")
    print(f"Validation: {valid_total:,} → {valid_kept:,} ({100*valid_kept/valid_total:.1f}% kept)")
    print(f"\nTotal dataset: {train_total+valid_total:,} → {train_kept+valid_kept:,} SMILES")
    print(f"Removed: {(train_total+valid_total)-(train_kept+valid_kept):,} SMILES")

    # All reasons combined
    all_reasons = train_reasons + valid_reasons
    print(f"\nMost common incompatibility patterns:")
    for reason, count in all_reasons.most_common(15):
        print(f"  {reason}: {count:,}")

    print("\n" + "=" * 70)
    print("Next steps:")
    print("  1. Run: python data/create_cleaned_zips.py")
    print("     (to create train_filtered.zip and valid_filtered.zip)")
    print("  2. Update finetune.py to use:")
    print("     training_molecules_path='data/train_filtered.zip'")
    print("     val_molecules_path='data/valid_filtered.zip'")
    print("=" * 70)

if __name__ == '__main__':
    main()
