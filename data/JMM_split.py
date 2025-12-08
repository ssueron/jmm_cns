#!/usr/bin/env python3
"""
Scaffold-based clustering split for molecular datasets using spectral clustering.

This script performs dataset splitting with the following methodology:
1. Extracts Bemis-Murcko scaffolds from SMILES
2. Computes ECFP6 fingerprints on scaffolds (default: radius=3)
3. Calculates Tanimoto similarity matrix on scaffolds
4. Uses spectral clustering with automatic cluster detection (eigenvalue/kneedle)
5. Assigns whole clusters to train/val/test splits (70%/15%/15%)

Usage:
    # Basic usage
    python data/S4_split.py data/unique_smiles_CHEMBL_no_shared.csv

    # With t-SNE visualization
    python data/S4_split.py data/unique_smiles_CHEMBL_no_shared.csv --visualize

    # Using cyclic_skeleton scaffolds (like original 2.0_split_data.py)
    python data/S4_split.py data/unique_smiles_B3DB_no_shared.csv --scaffold-type cyclic_skeleton

    Note: If you have .txt files, first convert them to CSV using:
    python data/convert_txt_to_csv.py data/unique_smiles_CHEMBL_no_shared.txt

Output:
    Creates 3 CSV files: *_train.csv, *_val.csv, *_test.csv
    Optional: *_tsne.png (with --visualize flag)
"""

import os
import sys
import argparse
import numpy as np
import pandas as pd
from pathlib import Path
from collections import Counter
from rdkit import Chem
from scipy.sparse import csgraph
from scipy.linalg import eigh
from kneed import KneeLocator
from sklearn.cluster import SpectralClustering
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import seaborn as sns

# Add parent directory to path to import from jointmolecularmodel
SCRIPT_DIR = Path(__file__).parent.absolute()
PROJECT_ROOT = SCRIPT_DIR.parent
sys.path.insert(0, str(PROJECT_ROOT / 'jointmolecularmodel'))

from cheminformatics.multiprocessing import tanimoto_matrix
from cheminformatics.splitting import map_scaffolds
from cheminformatics.descriptors import mols_to_ecfp


def load_smiles_from_csv(filepath: str) -> tuple[list[str], list[int], pd.DataFrame]:
    """
    Load SMILES from CSV file with columns: original_index, smiles
    Also returns the full dataframe to preserve additional columns.

    Args:
        filepath: Path to CSV file containing SMILES

    Returns:
        Tuple of (smiles_list, original_indices, full_dataframe)
    """
    df = pd.read_csv(filepath)

    # Check required columns
    if 'smiles' not in df.columns:
        raise ValueError(f"CSV file must contain a 'smiles' column. Found columns: {list(df.columns)}")

    # Use original_index if available, otherwise use DataFrame index
    if 'original_index' in df.columns:
        original_indices = df['original_index'].tolist()
    else:
        original_indices = list(range(len(df)))
        print(f"Warning: No 'original_index' column found, using sequential indices")

    smiles_list = df['smiles'].tolist()

    print(f"Loaded {len(smiles_list)} SMILES from {filepath}")
    return smiles_list, original_indices, df


def organize_dataframe(uniques: dict, smiles: list[str]) -> pd.DataFrame:
    """
    Puts all scaffolds next to their original smiles in a dataframe.
    Lists of original SMILES are joined by ';'

    Args:
        uniques: dict with scaffold smiles as keys and list of indices as values
        smiles: list of original SMILES strings

    Returns:
        DataFrame with columns: ['scaffolds', 'original_smiles', 'n']
    """
    smiles_belonging_to_scaffs = []
    n_mols_with_scaff = []
    unique_scaffold_smiles = []

    for scaf, idx_list in uniques.items():
        smiles_belonging_to_scaffs.append(';'.join([smiles[i] for i in idx_list]))
        unique_scaffold_smiles.append(scaf)
        n_mols_with_scaff.append(len(idx_list))

    df = pd.DataFrame({
        'scaffolds': unique_scaffold_smiles,
        'original_smiles': smiles_belonging_to_scaffs,
        'n': n_mols_with_scaff
    })

    return df


def eigenvalue_cluster_approx(x: np.ndarray) -> int:
    """
    Estimate the number of clusters for spectral clustering using Eigenvalues
    of the Laplacian. Uses the kneedle algorithm to find the elbow in the curve.

    Args:
        x: Similarity/affinity matrix (Tanimoto similarity matrix)

    Returns:
        Number of clusters
    """
    # Compute the symmetrically normalized Laplacian
    laplacian = csgraph.laplacian(x, normed=True)

    # Perform Eigen-decomposition
    eigenvalues, eigenvectors = eigh(laplacian)

    # Estimate the 'elbow'/'knee' of the curve using the kneedle algorithm
    kn = KneeLocator(
        range(len(eigenvalues)),
        eigenvalues,
        curve='concave',
        direction='increasing',
        interp_method='interp1d'
    )

    n_clusters = kn.knee

    if n_clusters is None or n_clusters == 0:
        # Fallback if knee detection fails or returns invalid value
        n_clusters = max(2, int(np.sqrt(len(eigenvalues))))
        print(f"Warning: Knee detection failed, using sqrt heuristic: {n_clusters} clusters")
    else:
        print(f"Detected {n_clusters} clusters via eigenvalue analysis")

    return n_clusters


def cluster_similarity(X: np.ndarray, clusters: np.ndarray) -> np.ndarray:
    """
    Find the pairwise similarity of every cluster.

    Args:
        X: Tanimoto similarity matrix (N, N)
        clusters: cluster membership vector (N), where every item is cluster ID

    Returns:
        Cluster similarity matrix (n_clusters, n_clusters)
    """
    n_clusters = len(set(clusters))

    # Which molecules belong to which cluster?
    clust_molidx = {c: np.where(clusters == c)[0] for c in set(clusters)}

    # Empty matrix (n_clust x n_clust)
    clust_sims = np.zeros((n_clusters, n_clusters))

    for i in range(n_clusters):
        for j in range(i, n_clusters):
            # Get the indices of the scaffolds in this cluster
            row_idx, col_idx = clust_molidx[i], clust_molidx[j]

            # Find the mean inter-cluster similarity
            clust_sim_matrix = X[row_idx][:, col_idx]
            clust_sims[i, j] = np.mean(clust_sim_matrix)

    # Mirror out the lower triangle of the matrix
    clust_sims = clust_sims + clust_sims.T - np.diag(np.diag(clust_sims))

    return clust_sims


def mean_cluster_sim_to_all_clusters(x: np.ndarray) -> np.ndarray:
    """
    Calculate the mean similarity of each cluster to all other clusters.
    Masks the diagonal so self-similarity is not taken into account.

    Args:
        x: Cluster similarity matrix (n_clusters, n_clusters)

    Returns:
        Mean similarity vector (n_clusters,)
    """
    n_clusters = len(x)

    mean_clust_sim = []
    for i in range(n_clusters):
        mask = np.array([j for j in range(n_clusters) if j != i])
        mean_clust_sim.append(np.mean(x[i][mask]))

    return np.array(mean_clust_sim)


def group_and_sort(clusters: np.ndarray, similarity_matrix: np.ndarray,
                   n_smiles_for_each_scaffold: list[int]) -> pd.DataFrame:
    """
    Group clusters and sort them by mean similarity to all other clusters.

    Args:
        clusters: Cluster membership vector
        similarity_matrix: Tanimoto similarity matrix
        n_smiles_for_each_scaffold: Number of SMILES per scaffold

    Returns:
        DataFrame with cluster info sorted by mean similarity
    """
    # Compute the pairwise similarity between whole clusters
    clust_sims = cluster_similarity(similarity_matrix, clusters)
    mean_clust_sims = mean_cluster_sim_to_all_clusters(clust_sims)

    # Get the original size of each cluster (in terms of molecules, not scaffolds)
    cluster_size = [
        sum(np.array(n_smiles_for_each_scaffold)[np.argwhere(clusters == c).flatten()])
        for c in set(clusters)
    ]

    # Put everything together
    df = pd.DataFrame({
        'cluster': list(range(len(set(clusters)))),
        'size (scaffolds)': [i[1] for i in sorted(Counter(clusters).items())],
        'size (molecules)': cluster_size,
        'mean_sim': mean_clust_sims
    })

    # Sort by mean similarity (ascending = most dissimilar first)
    df.sort_values(by=['mean_sim'], inplace=True)

    return df


def assign_clusters_to_splits(df_clusters: pd.DataFrame,
                              test_frac: float = 0.15,
                              val_frac: float = 0.15,
                              input_df: pd.DataFrame = None,
                              smiles_list: list[str] = None,
                              df_scaffs: pd.DataFrame = None,
                              clusters: np.ndarray = None) -> tuple[list[int], list[int], list[int]]:
    """
    Assign whole clusters to test/val/train splits with balanced source dataset distribution.
    Uses iterative optimization to balance B3DB/CHEMBL ratios across splits when source info available.

    Args:
        df_clusters: DataFrame with cluster info (sorted by mean_sim)
        test_frac: Fraction for test set (default 0.15)
        val_frac: Fraction for validation set (default 0.15)
        input_df: Original input DataFrame (to extract source_dataset if available)
        smiles_list: List of SMILES strings
        df_scaffs: DataFrame with scaffold info
        clusters: Array of cluster assignments

    Returns:
        Tuple of (test_clusters, val_clusters, train_clusters)
    """
    total_molecules = df_clusters['size (molecules)'].sum()
    test_target = total_molecules * test_frac
    val_target = total_molecules * val_frac

    # Check if we have source dataset information
    has_source_info = (input_df is not None and
                      'source_dataset' in input_df.columns and
                      smiles_list is not None and
                      df_scaffs is not None and
                      clusters is not None)

    if has_source_info:
        print("\nUsing balanced assignment with source dataset optimization...")
        return _assign_clusters_balanced(df_clusters, test_target, val_target, total_molecules,
                                         input_df, smiles_list, df_scaffs, clusters)
    else:
        print("\nUsing simple sequential assignment...")
        return _assign_clusters_simple(df_clusters, test_target, val_target, total_molecules)


def _assign_clusters_simple(df_clusters: pd.DataFrame, test_target: float, val_target: float,
                            total_molecules: int) -> tuple[list[int], list[int], list[int]]:
    """Simple sequential cluster assignment (original method)."""
    test_clusters = []
    val_clusters = []
    train_clusters = []

    test_size = 0
    val_size = 0

    for _, row in df_clusters.iterrows():
        cluster_id = row['cluster']
        cluster_size = row['size (molecules)']

        if test_size < test_target:
            test_clusters.append(cluster_id)
            test_size += cluster_size
        elif val_size < val_target:
            val_clusters.append(cluster_id)
            val_size += cluster_size
        else:
            train_clusters.append(cluster_id)

    train_size = total_molecules - test_size - val_size

    print(f"\nSplit assignment:")
    print(f"  Test:  {len(test_clusters)} clusters, {int(test_size)} molecules ({test_size/total_molecules*100:.1f}%)")
    print(f"  Val:   {len(val_clusters)} clusters, {int(val_size)} molecules ({val_size/total_molecules*100:.1f}%)")
    print(f"  Train: {len(train_clusters)} clusters, {int(train_size)} molecules ({train_size/total_molecules*100:.1f}%)")

    return test_clusters, val_clusters, train_clusters


def _assign_clusters_balanced(df_clusters: pd.DataFrame, test_target: float, val_target: float,
                              total_molecules: int, input_df: pd.DataFrame, smiles_list: list[str],
                              df_scaffs: pd.DataFrame, clusters: np.ndarray) -> tuple[list[int], list[int], list[int]]:
    """Balanced cluster assignment considering source dataset distribution."""

    # Map smiles to source datasets
    smiles_to_source = dict(zip(input_df['smiles'], input_df['source_dataset']))

    # Get source for each cluster
    cluster_sources = {}
    for cluster_id in df_clusters['cluster']:
        # Get all smiles in this cluster
        cluster_indices = np.where(clusters == cluster_id)[0]
        cluster_smiles = [df_scaffs.iloc[idx]['original_smiles'].split(';') for idx in cluster_indices]
        cluster_smiles_flat = [smi for sublist in cluster_smiles for smi in sublist]

        # Count sources
        source_counts = {}
        for smi in cluster_smiles_flat:
            source = smiles_to_source.get(smi, 'unknown')
            source_counts[source] = source_counts.get(source, 0) + 1

        cluster_sources[cluster_id] = source_counts

    # Calculate overall source distribution
    total_sources = {}
    for source_counts in cluster_sources.values():
        for source, count in source_counts.items():
            total_sources[source] = total_sources.get(source, 0) + count

    # Target ratios for each split (should match overall distribution)
    target_ratios = {source: count / total_molecules for source, count in total_sources.items()}

    print(f"  Overall source distribution:")
    for source, ratio in sorted(target_ratios.items()):
        print(f"    {source}: {ratio*100:.1f}%")

    # Iterative optimization approach
    # Start with initial assignment (most dissimilar clusters first)
    test_clusters = []
    val_clusters = []
    train_clusters = []

    test_size = 0
    val_size = 0
    test_sources = {s: 0 for s in total_sources}
    val_sources = {s: 0 for s in total_sources}
    train_sources = {s: 0 for s in total_sources}

    # Sort clusters by dissimilarity, then try to balance
    for _, row in df_clusters.iterrows():
        cluster_id = row['cluster']
        cluster_size = row['size (molecules)']
        cluster_src_counts = cluster_sources[cluster_id]

        # Calculate composite score (balance + size target)
        def calc_score(split_sources, split_size, split_target):
            if split_size == 0:
                # Prefer filling empty splits
                return 0.0

            # Component 1: Source ratio imbalance (weighted heavily)
            ratio_imbalance = 0
            for source in total_sources:
                current_ratio = split_sources.get(source, 0) / split_size
                target_ratio = target_ratios[source]
                ratio_imbalance += abs(current_ratio - target_ratio) ** 2  # Squared penalty

            # Component 2: Size deviation from target
            size_deviation = abs(split_size - split_target) / total_molecules

            # Combined score (lower is better)
            # Heavily weight ratio balance (10x) over size target
            score = (10.0 * ratio_imbalance) + size_deviation
            return score

        # Try adding to each split and calculate resulting score
        test_score = float('inf')
        val_score = float('inf')
        train_score = float('inf')

        # Only consider test if we haven't exceeded target significantly
        if test_size < test_target * 1.3:  # Allow 30% overflow
            test_sources_new = {s: test_sources.get(s, 0) + cluster_src_counts.get(s, 0) for s in total_sources}
            test_score = calc_score(test_sources_new, test_size + cluster_size, test_target)

        # Only consider val if we haven't exceeded target significantly
        if val_size < val_target * 1.3:
            val_sources_new = {s: val_sources.get(s, 0) + cluster_src_counts.get(s, 0) for s in total_sources}
            val_score = calc_score(val_sources_new, val_size + cluster_size, val_target)

        # Train can always take more (it's the largest split)
        train_sources_new = {s: train_sources.get(s, 0) + cluster_src_counts.get(s, 0) for s in total_sources}
        train_size_new = total_molecules - test_size - val_size + cluster_size
        train_score = calc_score(train_sources_new, train_size_new, total_molecules - test_target - val_target)

        # Assign to split with lowest score (best balance)
        min_score = min(test_score, val_score, train_score)

        if min_score == test_score and test_size < test_target * 1.3:
            test_clusters.append(cluster_id)
            test_size += cluster_size
            for s in total_sources:
                test_sources[s] = test_sources.get(s, 0) + cluster_src_counts.get(s, 0)
        elif min_score == val_score and val_size < val_target * 1.3:
            val_clusters.append(cluster_id)
            val_size += cluster_size
            for s in total_sources:
                val_sources[s] = val_sources.get(s, 0) + cluster_src_counts.get(s, 0)
        else:
            train_clusters.append(cluster_id)
            for s in total_sources:
                train_sources[s] = train_sources.get(s, 0) + cluster_src_counts.get(s, 0)

    train_size = total_molecules - test_size - val_size

    # Print detailed statistics
    print(f"\nSplit assignment:")
    print(f"  Test:  {len(test_clusters)} clusters, {int(test_size)} molecules ({test_size/total_molecules*100:.1f}%)")
    for source in sorted(total_sources):
        count = test_sources.get(source, 0)
        pct = (count / test_size * 100) if test_size > 0 else 0
        print(f"    {source}: {int(count)} ({pct:.1f}%)")

    print(f"  Val:   {len(val_clusters)} clusters, {int(val_size)} molecules ({val_size/total_molecules*100:.1f}%)")
    for source in sorted(total_sources):
        count = val_sources.get(source, 0)
        pct = (count / val_size * 100) if val_size > 0 else 0
        print(f"    {source}: {int(count)} ({pct:.1f}%)")

    print(f"  Train: {len(train_clusters)} clusters, {int(train_size)} molecules ({train_size/total_molecules*100:.1f}%)")
    for source in sorted(total_sources):
        count = train_sources.get(source, 0)
        pct = (count / train_size * 100) if train_size > 0 else 0
        print(f"    {source}: {int(count)} ({pct:.1f}%)")

    return test_clusters, val_clusters, train_clusters


def map_clusters_to_molecules(smiles: list[str], df_scaffs: pd.DataFrame,
                              clusters: np.ndarray) -> list[int]:
    """
    Map cluster assignments from scaffolds back to original molecules.

    Args:
        smiles: List of original SMILES
        df_scaffs: DataFrame with scaffold info
        clusters: Cluster assignments for scaffolds

    Returns:
        List of cluster IDs for each molecule
    """
    clusters_per_molecule = [-1] * len(smiles)

    for originals, c in zip(df_scaffs['original_smiles'], clusters):
        for smi in originals.split(';'):
            idx = smiles.index(smi)
            clusters_per_molecule[idx] = c

    return clusters_per_molecule


def create_output_dataframes(smiles: list[str], original_indices: list[int],
                            scaffold_smiles: list[str], clusters_per_molecule: list[int],
                            test_clusters: list[int], val_clusters: list[int],
                            train_clusters: list[int], input_df: pd.DataFrame = None) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Create separate DataFrames for train, val, and test splits.
    Preserves extra columns from the input dataframe if provided.

    Args:
        smiles: List of SMILES
        original_indices: Original line numbers from input file
        scaffold_smiles: Scaffold SMILES for each molecule
        clusters_per_molecule: Cluster ID for each molecule
        test_clusters: List of cluster IDs for test set
        val_clusters: List of cluster IDs for val set
        train_clusters: List of cluster IDs for train set
        input_df: Original input dataframe (optional, to preserve extra columns)

    Returns:
        Tuple of (train_df, val_df, test_df)
    """
    # Create base dataframe with split info
    df = pd.DataFrame({
        'smiles': smiles,
        'original_index': original_indices,
        'scaffold': scaffold_smiles,
        'cluster': clusters_per_molecule
    })

    # If input_df provided, merge to preserve extra columns
    if input_df is not None:
        # Get extra columns (exclude 'smiles' and 'original_index' as they're already in df)
        extra_cols = [col for col in input_df.columns
                     if col not in ['smiles', 'original_index']]

        if extra_cols:
            # Merge on smiles to preserve extra columns
            # Use left join to keep all rows from df
            input_df_subset = input_df[['smiles'] + extra_cols].copy()
            df = df.merge(input_df_subset, on='smiles', how='left')

    # Split into separate dataframes
    test_df = df[df['cluster'].isin(test_clusters)].copy()
    val_df = df[df['cluster'].isin(val_clusters)].copy()
    train_df = df[df['cluster'].isin(train_clusters)].copy()

    # Reset indices
    test_df.reset_index(drop=True, inplace=True)
    val_df.reset_index(drop=True, inplace=True)
    train_df.reset_index(drop=True, inplace=True)

    return train_df, val_df, test_df


def visualize_tsne(fingerprints: np.ndarray, clusters_per_molecule: list[int],
                   test_clusters: list[int], val_clusters: list[int],
                   train_clusters: list[int], output_path: Path,
                   perplexity: int = 30, random_state: int = 42):
    """
    Create t-SNE visualization of molecular fingerprints colored by split assignment.

    Args:
        fingerprints: ECFP fingerprints array (n_molecules, n_bits)
        clusters_per_molecule: Cluster ID for each molecule
        test_clusters: List of cluster IDs for test set
        val_clusters: List of cluster IDs for val set
        train_clusters: List of cluster IDs for train set
        output_path: Path to save the plot
        perplexity: t-SNE perplexity parameter (default: 30)
        random_state: Random seed for reproducibility (default: 42)
    """
    print("\nGenerating t-SNE visualization...")

    # Assign split labels to each molecule
    split_labels = []
    for cluster_id in clusters_per_molecule:
        if cluster_id in test_clusters:
            split_labels.append('Test')
        elif cluster_id in val_clusters:
            split_labels.append('Val')
        elif cluster_id in train_clusters:
            split_labels.append('Train')
        else:
            split_labels.append('Unknown')

    # Perform t-SNE dimensionality reduction
    print(f"  Running t-SNE (perplexity={perplexity})...")
    tsne = TSNE(
        n_components=2,
        perplexity=perplexity,
        random_state=random_state,
        max_iter=1000,
        verbose=0
    )
    embeddings = tsne.fit_transform(fingerprints)

    # Create DataFrame for plotting
    df_plot = pd.DataFrame({
        'x': embeddings[:, 0],
        'y': embeddings[:, 1],
        'split': split_labels
    })

    # Set up the plot style
    sns.set_style("whitegrid")
    plt.figure(figsize=(12, 10))

    # Define colors for each split
    colors = {'Train': '#1f77b4', 'Val': '#ff7f0e', 'Test': '#2ca02c'}
    split_order = ['Train', 'Val', 'Test']

    # Plot each split with different colors
    for split_name in split_order:
        subset = df_plot[df_plot['split'] == split_name]
        plt.scatter(
            subset['x'],
            subset['y'],
            c=colors[split_name],
            label=f"{split_name} ({len(subset)} molecules)",
            alpha=0.6,
            s=20,
            edgecolors='none'
        )

    plt.xlabel('t-SNE Dimension 1', fontsize=12)
    plt.ylabel('t-SNE Dimension 2', fontsize=12)
    plt.title('t-SNE Visualization of Molecular Scaffolds by Split', fontsize=14, fontweight='bold')
    plt.legend(title='Split', fontsize=10, title_fontsize=11, loc='best')
    plt.tight_layout()

    # Save plot
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"  Saved t-SNE plot to: {output_path}")
    plt.close()


def main():
    parser = argparse.ArgumentParser(
        description='Scaffold-based clustering split for molecular datasets',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )
    parser.add_argument(
        'input_file',
        type=str,
        help='Path to input CSV file with SMILES (columns: original_index, smiles)'
    )
    parser.add_argument(
        '--test-frac',
        type=float,
        default=0.15,
        help='Fraction for test set (default: 0.15)'
    )
    parser.add_argument(
        '--val-frac',
        type=float,
        default=0.15,
        help='Fraction for validation set (default: 0.15)'
    )
    parser.add_argument(
        '--radius',
        type=int,
        default=3,
        help='ECFP radius (3 for ECFP6, default: 3)'
    )
    parser.add_argument(
        '--scaffold-type',
        type=str,
        default='bemis_murcko',
        choices=['bemis_murcko', 'bemis_murcko_bajorath', 'generic', 'cyclic_skeleton'],
        help='Scaffold type to use (default: bemis_murcko)'
    )
    parser.add_argument(
        '--nbits',
        type=int,
        default=2048,
        help='Number of bits for ECFP (default: 2048)'
    )
    parser.add_argument(
        '--visualize',
        action='store_true',
        help='Generate t-SNE visualization of splits'
    )
    parser.add_argument(
        '--tsne-perplexity',
        type=int,
        default=30,
        help='t-SNE perplexity parameter (default: 30)'
    )

    args = parser.parse_args()

    # Validate input file exists
    input_path = Path(args.input_file)
    if not input_path.exists():
        print(f"Error: Input file not found: {args.input_file}")
        sys.exit(1)

    print(f"\n{'='*60}")
    print(f"Scaffold-based Clustering Split")
    print(f"{'='*60}")
    print(f"Input file: {args.input_file}")
    print(f"Split ratio: {1-args.test_frac-args.val_frac:.0%} train / {args.val_frac:.0%} val / {args.test_frac:.0%} test")
    print(f"Scaffold type: {args.scaffold_type}")
    print(f"ECFP settings: radius={args.radius} (ECFP{args.radius*2}), nbits={args.nbits}")
    print(f"{'='*60}\n")

    # Step 1: Load SMILES
    print("Step 1/7: Loading SMILES from CSV file...")
    smiles, original_indices, input_df = load_smiles_from_csv(args.input_file)

    # Step 2: Extract scaffolds
    print(f"\nStep 2/7: Extracting {args.scaffold_type} scaffolds...")
    scaffold_smiles, uniques = map_scaffolds(smiles, scaffold_type=args.scaffold_type)
    print(f"Found {len(uniques)} unique scaffolds from {len(smiles)} molecules")

    # Organize into dataframe
    df_scaffs = organize_dataframe(uniques, smiles)

    # Step 3: Compute ECFP6 on scaffolds
    print(f"\nStep 3/7: Computing ECFP{args.radius*2} fingerprints on scaffolds...")
    scaffold_mols = [Chem.MolFromSmiles(smi) for smi in df_scaffs['scaffolds']]
    ecfps = mols_to_ecfp(scaffold_mols, radius=args.radius, nbits=args.nbits)

    # Step 4: Calculate Tanimoto similarity matrix
    print(f"\nStep 4/7: Computing Tanimoto similarity matrix...")
    S = tanimoto_matrix(ecfps, dtype=float)
    print(f"Similarity matrix shape: {S.shape}")
    print(f"Similarity range: [{S.min():.3f}, {S.max():.3f}]")

    # Step 5: Spectral clustering
    print("\nStep 5/7: Performing spectral clustering...")
    n_clusters = eigenvalue_cluster_approx(S)

    spectral = SpectralClustering(
        n_clusters=n_clusters,
        affinity='precomputed',
        assign_labels='kmeans',
        random_state=42
    )
    clusters = spectral.fit_predict(S)
    df_scaffs['cluster'] = clusters

    # Step 6: Sort clusters and assign to splits
    print("\nStep 6/7: Sorting clusters by dissimilarity and assigning to splits...")
    df_clusters = group_and_sort(clusters, S, df_scaffs['n'].tolist())

    test_clusters, val_clusters, train_clusters = assign_clusters_to_splits(
        df_clusters,
        test_frac=args.test_frac,
        val_frac=args.val_frac,
        input_df=input_df,
        smiles_list=smiles,
        df_scaffs=df_scaffs,
        clusters=clusters
    )

    # Map clusters back to molecules
    clusters_per_molecule = map_clusters_to_molecules(smiles, df_scaffs, clusters)

    # Step 7: Create output files
    print("\nStep 7/7: Creating output CSV files...")
    train_df, val_df, test_df = create_output_dataframes(
        smiles, original_indices, scaffold_smiles, clusters_per_molecule,
        test_clusters, val_clusters, train_clusters, input_df
    )

    # Generate output filenames
    output_base = input_path.stem  # filename without extension
    output_dir = input_path.parent

    train_path = output_dir / f"{output_base}_train.csv"
    val_path = output_dir / f"{output_base}_val.csv"
    test_path = output_dir / f"{output_base}_test.csv"

    # Write to CSV
    train_df.to_csv(train_path, index=False)
    val_df.to_csv(val_path, index=False)
    test_df.to_csv(test_path, index=False)

    print(f"\nOutput files created:")
    print(f"  Train: {train_path} ({len(train_df)} molecules)")
    print(f"  Val:   {val_path} ({len(val_df)} molecules)")
    print(f"  Test:  {test_path} ({len(test_df)} molecules)")

    # Optional: Generate t-SNE visualization
    if args.visualize:
        print("\n" + "="*60)
        print("Generating t-SNE Visualization")
        print("="*60)

        # Compute fingerprints for all molecules (not just scaffolds)
        print("Computing ECFP fingerprints for all molecules...")
        all_mols = [Chem.MolFromSmiles(smi) for smi in smiles]
        all_fps = mols_to_ecfp(all_mols, radius=args.radius, nbits=args.nbits)

        # Convert to numpy array if it's a list
        if isinstance(all_fps, list):
            all_fps = np.array(all_fps)

        # Generate visualization
        plot_path = output_dir / f"{output_base}_tsne.png"
        visualize_tsne(
            all_fps,
            clusters_per_molecule,
            test_clusters,
            val_clusters,
            train_clusters,
            plot_path,
            perplexity=args.tsne_perplexity
        )

    print(f"\n{'='*60}")
    print("Split completed successfully!")
    print(f"{'='*60}\n")


if __name__ == '__main__':
    main()
