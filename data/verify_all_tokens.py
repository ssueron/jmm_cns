#!/usr/bin/env python3
"""
Comprehensive token verification script.
Tokenizes all SMILES using the actual S4 tokenization method and verifies
that every token exists in the pretrained vocabulary.
"""

import csv
import json
import re
from pathlib import Path
from collections import Counter
from typing import List, Dict

# S4 tokenization regex (copied from s4dd/smiles_utils.py to avoid torch dependency)
_ELEMENTS_STR = r"(?<=\[)Cs(?=\])|Si|Xe|Ba|Rb|Ra|Sr|Dy|Li|Kr|Bi|Mn|He|Am|Pu|Cm|Pm|Ne|Th|Ni|Pr|Fe|Lu|Pa|Fm|Tm|Tb|Er|Be|Al|Gd|Eu|te|As|Pt|Lr|Sm|Ca|La|Ti|Te|Ac|Cf|Rf|Na|Cu|Au|Nd|Ag|Se|se|Zn|Mg|Br|Cl|Pb|U|V|K|C|B|H|N|O|S|P|F|I|b|c|n|o|s|p"
_TOKENIZATION_REGEX = re.compile(
    rf"(\[|\]|{_ELEMENTS_STR}|" + r"\(|\)|\.|=|#|-|\+|\\\\|\/|:|~|@|\?|>|\*|\$|\%\d{2}|\d)"
)


def load_pretrained_vocabulary(vocab_path: Path) -> Dict[str, int]:
    """Load the pretrained token2label vocabulary."""
    with open(vocab_path, 'r') as f:
        init_args = json.load(f)
    return init_args['token2label']


def read_csv_smiles(csv_path: Path) -> List[str]:
    """Read SMILES from CSV file."""
    with open(csv_path, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        return [row['smiles'] for row in reader]


def tokenize_smiles(smiles: str) -> List[str]:
    """Tokenize SMILES using the S4 tokenization method."""
    return _TOKENIZATION_REGEX.findall(smiles)


def verify_tokens(smiles_list: List[str], vocab: Dict[str, int]) -> Dict:
    """
    Verify that all tokens in SMILES list exist in vocabulary.

    Returns statistics about token compatibility.
    """
    stats = {
        'total_smiles': len(smiles_list),
        'compatible_smiles': 0,
        'incompatible_smiles': 0,
        'all_tokens': Counter(),
        'unknown_tokens': Counter(),
        'unknown_token_examples': {},
        'incompatible_smiles_examples': []
    }

    vocab_tokens = set(vocab.keys())

    for idx, smiles in enumerate(smiles_list):
        try:
            tokens = tokenize_smiles(smiles)
            stats['all_tokens'].update(tokens)

            # Find unknown tokens
            unknown = [t for t in tokens if t not in vocab_tokens]

            if unknown:
                stats['incompatible_smiles'] += 1
                stats['unknown_tokens'].update(unknown)

                # Store examples
                for token in unknown:
                    if token not in stats['unknown_token_examples']:
                        stats['unknown_token_examples'][token] = []
                    if len(stats['unknown_token_examples'][token]) < 5:
                        stats['unknown_token_examples'][token].append(smiles)

                # Store incompatible SMILES examples
                if len(stats['incompatible_smiles_examples']) < 10:
                    stats['incompatible_smiles_examples'].append({
                        'smiles': smiles,
                        'unknown_tokens': unknown
                    })
            else:
                stats['compatible_smiles'] += 1

        except Exception as e:
            print(f"Error tokenizing SMILES: {smiles[:50]}... - {e}")
            stats['incompatible_smiles'] += 1

    return stats


def main():
    """Main verification function."""
    data_dir = Path(__file__).parent
    s4_dir = data_dir.parent / 's4-for-de-novo-drug-design'
    vocab_path = s4_dir / 'chembl_pretrained' / 'init_arguments.json'

    print("="*70)
    print("COMPREHENSIVE TOKEN VERIFICATION")
    print("="*70)

    # Load vocabulary
    print(f"\nLoading pretrained vocabulary from:")
    print(f"  {vocab_path}")
    vocab = load_pretrained_vocabulary(vocab_path)
    print(f"\nVocabulary size: {len(vocab)} tokens")
    print(f"Tokens: {', '.join(sorted(vocab.keys()))}")

    # Datasets to verify
    datasets = [
        {
            'name': 'Training Set',
            'csv': data_dir / 'train_90.csv',
        },
        {
            'name': 'Validation Set',
            'csv': data_dir / 'test_10.csv',
        }
    ]

    all_passed = True
    grand_stats = {
        'total_smiles': 0,
        'compatible_smiles': 0,
        'incompatible_smiles': 0,
        'all_tokens': Counter(),
        'unknown_tokens': Counter(),
        'unknown_token_examples': {}
    }

    for dataset in datasets:
        print(f"\n{'='*70}")
        print(f"{dataset['name']}")
        print(f"{'='*70}")

        # Read SMILES
        smiles_list = read_csv_smiles(dataset['csv'])
        print(f"\nTotal SMILES: {len(smiles_list)}")

        # Verify tokens
        print(f"Tokenizing and verifying...")
        stats = verify_tokens(smiles_list, vocab)

        # Update grand stats
        grand_stats['total_smiles'] += stats['total_smiles']
        grand_stats['compatible_smiles'] += stats['compatible_smiles']
        grand_stats['incompatible_smiles'] += stats['incompatible_smiles']
        grand_stats['all_tokens'].update(stats['all_tokens'])
        grand_stats['unknown_tokens'].update(stats['unknown_tokens'])
        for token, examples in stats['unknown_token_examples'].items():
            if token not in grand_stats['unknown_token_examples']:
                grand_stats['unknown_token_examples'][token] = []
            grand_stats['unknown_token_examples'][token].extend(examples[:5])

        # Report results
        print(f"\nResults:")
        print(f"  Compatible SMILES: {stats['compatible_smiles']} ({stats['compatible_smiles']/stats['total_smiles']*100:.2f}%)")
        print(f"  Incompatible SMILES: {stats['incompatible_smiles']} ({stats['incompatible_smiles']/stats['total_smiles']*100:.2f}%)")

        if stats['unknown_tokens']:
            print(f"\n  ✗ FAIL - Unknown tokens found:")
            for token, count in stats['unknown_tokens'].most_common():
                print(f"    '{token}': {count} occurrences")

            print(f"\n  Sample incompatible SMILES:")
            for item in stats['incompatible_smiles_examples'][:5]:
                smiles_display = item['smiles']
                if len(smiles_display) > 70:
                    smiles_display = smiles_display[:67] + "..."
                print(f"    {smiles_display}")
                print(f"      Unknown tokens: {item['unknown_tokens']}")

            all_passed = False
        else:
            print(f"  ✓ PASS - All tokens in vocabulary")

        # Show token distribution
        print(f"\n  Unique tokens used: {len(stats['all_tokens'])}")
        print(f"  Top 10 most frequent tokens:")
        for token, count in stats['all_tokens'].most_common(10):
            pct = count / sum(stats['all_tokens'].values()) * 100
            in_vocab = '✓' if token in vocab else '✗'
            print(f"    {in_vocab} '{token}': {count} ({pct:.1f}%)")

    # Grand summary
    print(f"\n{'='*70}")
    print("GRAND SUMMARY")
    print(f"{'='*70}")
    print(f"Total SMILES: {grand_stats['total_smiles']}")
    print(f"Compatible: {grand_stats['compatible_smiles']} ({grand_stats['compatible_smiles']/grand_stats['total_smiles']*100:.2f}%)")
    print(f"Incompatible: {grand_stats['incompatible_smiles']} ({grand_stats['incompatible_smiles']/grand_stats['total_smiles']*100:.2f}%)")

    print(f"\nTotal unique tokens found: {len(grand_stats['all_tokens'])}")
    print(f"Vocabulary size: {len(vocab)}")

    if grand_stats['unknown_tokens']:
        print(f"\n✗ UNKNOWN TOKENS FOUND:")
        print(f"{'-'*70}")
        for token, count in grand_stats['unknown_tokens'].most_common():
            print(f"\nToken: '{token}'")
            print(f"  Occurrences: {count}")
            print(f"  Sample SMILES:")
            for smiles in grand_stats['unknown_token_examples'][token][:3]:
                smiles_display = smiles if len(smiles) <= 70 else smiles[:67] + "..."
                print(f"    {smiles_display}")

    print(f"\n{'='*70}")
    if all_passed:
        print("✓ ALL CHECKS PASSED!")
        print("\nYour dataset is fully compatible with the S4 vocabulary.")
        print("All tokens can be mapped to the pretrained model.")
        print("Training should work without KeyError exceptions.")
    else:
        print("✗ VERIFICATION FAILED")
        print("\nSome SMILES contain tokens not in the vocabulary.")
        print("These must be removed before training.")
    print(f"{'='*70}")


if __name__ == "__main__":
    main()
