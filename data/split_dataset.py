#!/usr/bin/env python3
"""
Simple 90/10 random split of merged_dataset.csv for S4 CLM training.
"""

import pandas as pd
import numpy as np

# Set random seed for reproducibility
np.random.seed(42)

# Load the merged dataset
df = pd.read_csv('merged_dataset.csv')

print(f"Total molecules: {len(df)}")
print(f"Columns: {df.columns.tolist()}")

# Keep only smiles and source_dataset columns
df_subset = df[['smiles', 'source_dataset']].copy()

# Shuffle the dataframe
df_shuffled = df_subset.sample(frac=1, random_state=42).reset_index(drop=True)

# Calculate split index (90% for train, 10% for test)
split_idx = int(len(df_shuffled) * 0.9)

# Split the data
train_df = df_shuffled.iloc[:split_idx]
test_df = df_shuffled.iloc[split_idx:]

# Save the splits
train_df.to_csv('train_90.csv', index=False)
test_df.to_csv('test_10.csv', index=False)

print(f"\nSplit complete:")
print(f"  train_90.csv: {len(train_df)} molecules ({len(train_df)/len(df)*100:.1f}%)")
print(f"  test_10.csv: {len(test_df)} molecules ({len(test_df)/len(df)*100:.1f}%)")

# Verify source dataset distribution
print(f"\nSource dataset distribution:")
print(f"Original:\n{df['source_dataset'].value_counts(normalize=True)}")
print(f"\nTrain:\n{train_df['source_dataset'].value_counts(normalize=True)}")
print(f"\nTest:\n{test_df['source_dataset'].value_counts(normalize=True)}")
